"""Immutable runtime authorities compiled directly from :class:`CompiledSpecIR`.

This module is the F1 boundary between compilation and country runtime
adapters.  It never imports a country runtime, constants module, or the legacy
payload adapter.  Every projection is derived from a sealed IR surface and is
frozen before it can be handed to orchestration code.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .battery_semantics import (
    project_battery_authority_components,
    project_battery_legacy_contract,
)
from .canonical import sha256_json
from .compiler_ir import (
    CompiledNode,
    CompiledSpecIR,
    CompilerIRABI,
    SeedStreamMap,
)
from .errors import SpecValidationError
from .executor import RunProvenanceIdentity, build_run_provenance_identity
from .imputation_semantics import project_imputation_legacy_payloads
from .model import (
    FrozenMap,
    FrozenValue,
    GrammarReceipt,
    SpecBinding,
    freeze_json,
    thaw_json,
)
from .publication_semantics import (
    project_publication_legacy_release,
    project_spine_legacy_sampling,
)
from .stacked_authority_semantics import (
    project_stacked_authority_receipt,
    project_stacked_checkpoint_static_components,
)
from .take_up_semantics import project_legacy_take_up_contract


class RuntimeAuthorityError(ValueError):
    """A compiled plan cannot issue a closed runtime authority."""


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeAuthorityError(f"{location}: object required")
    return value


def _frozen(value: object, *, location: str) -> FrozenMap:
    result = freeze_json(value)
    if not isinstance(result, FrozenMap):
        raise RuntimeAuthorityError(f"{location}: object required")
    return result


def _wire(value: FrozenValue) -> object:
    return thaw_json(value)


def _surface_domains(
    compiled: CompiledSpecIR,
    surface: str,
) -> dict[str, object]:
    authority = _mapping(
        _wire(compiled.runtime_authorities), location="runtime_authorities"
    )
    surfaces = _mapping(
        authority.get("surfaces"), location="runtime_authorities/surfaces"
    )
    return dict(
        _mapping(
            surfaces.get(surface),
            location=f"runtime_authorities/surfaces/{surface}",
        )
    )


def _behavior_domains(compiled: CompiledSpecIR) -> dict[str, object]:
    return _surface_domains(compiled, "behavior")


def _generated_engine_lock(compiled: CompiledSpecIR) -> Mapping[str, object] | None:
    generated = _mapping(
        _wire(compiled.generated_authorities),
        location="authorities/generated",
    )
    value = generated.get("engine_abi_lock")
    if value is None:
        return None
    return _mapping(value, location="authorities/generated/engine_abi_lock")


def _compile_projections(compiled: CompiledSpecIR) -> FrozenMap:
    behavior = _behavior_domains(compiled)
    # A few generation-0 checkpoint identities historically include values
    # that the five-surface model now classifies as operational or
    # documentation.  Compile only the narrow compatibility projections from
    # the sealed normalized IR here; the capability never exposes the source
    # documents themselves.
    compatibility_fence = compiled.resources_wire()
    projections: dict[str, object] = {}

    publication = behavior.get("publication")
    spine = behavior.get("spine")
    if isinstance(publication, Mapping):
        projections["publication"] = project_publication_legacy_release(publication)
    if isinstance(spine, Mapping) and isinstance(publication, Mapping):
        projections["sampling"] = project_spine_legacy_sampling(
            spine,
            publication=publication,
        )

    imputation = compatibility_fence.get("imputation")
    sources = compatibility_fence.get("sources")
    compatibility_spine = compatibility_fence.get("spine")
    bundle = compatibility_fence.get("bundle")
    if all(
        isinstance(value, Mapping)
        for value in (imputation, sources, compatibility_spine, bundle)
    ):
        assert isinstance(imputation, Mapping)
        assert isinstance(sources, Mapping)
        assert isinstance(compatibility_spine, Mapping)
        assert isinstance(bundle, Mapping)
        projections["imputation"] = project_imputation_legacy_payloads(
            imputation,
            sources_document=sources,
            spine_document=compatibility_spine,
            bundle_document=bundle,
        )

    take_up = compatibility_fence.get("take_up")
    engine_lock = _generated_engine_lock(compiled)
    if (
        isinstance(take_up, Mapping)
        and isinstance(sources, Mapping)
        and engine_lock is not None
    ):
        projections["take_up"] = project_legacy_take_up_contract(
            take_up,
            sources_document=sources,
            engine_abi_lock=engine_lock,
        )

    battery = behavior.get("battery")
    if isinstance(battery, Mapping):
        projections["battery_components"] = project_battery_authority_components(
            battery
        )

    pipeline = spine.get("pipeline_contract") if isinstance(spine, Mapping) else None
    if isinstance(pipeline, Mapping):
        stacked_authority = project_stacked_authority_receipt(compiled)
        projections["stacked_authority"] = stacked_authority
        projections["stacked_checkpoint_static_components"] = (
            project_stacked_checkpoint_static_components(compiled)
        )
        if isinstance(battery, Mapping):
            projections["battery"] = project_battery_legacy_contract(
                battery,
                authority_receipt=stacked_authority,
            )

    return _frozen(projections, location="runtime projection")


def _compile_declared_sources(compiled: CompiledSpecIR) -> FrozenMap:
    """Select the closed source registry needed by file/source brokers.

    Source acquisition locators are operational and verification dates are
    documentation, so neither belongs in the normative spec identity.  They
    still have to reach runtime without exposing the full normalized source
    document.  This projection retains only content pins, typed loader/vintage
    bindings, and the optional acquisition receipt for each declared source.
    """

    resources = compiled.resources_wire()
    source_document = _mapping(resources.get("sources"), location="sources")
    raw_rows = source_document.get("sources")
    if not isinstance(raw_rows, list):
        raise RuntimeAuthorityError("sources/sources: array required")
    selected_rows: list[dict[str, object]] = []
    seen: set[str] = set()
    allowed = (
        "id",
        "role",
        "sha256",
        "byte_size",
        "loader",
        "vintages",
        "acquisition",
    )
    for index, value in enumerate(raw_rows):
        row = _mapping(value, location=f"sources/sources/{index}")
        source_id = row.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeAuthorityError(
                f"sources/sources/{index}/id: identifier required"
            )
        if source_id in seen:
            raise RuntimeAuthorityError(
                f"sources/sources/{index}/id: duplicate {source_id!r}"
            )
        seen.add(source_id)
        selected_rows.append({key: row[key] for key in allowed if key in row})
    unsigned = {"schema_version": 1, "sources": selected_rows}
    return _frozen(
        {**unsigned, "sha256": sha256_json(unsigned)},
        location="declared source authority",
    )


def _runtime_identity_wire(
    *,
    spec_binding: SpecBinding,
    compiler_ir_abi: CompilerIRABI,
    identity_generation: int,
    normative: FrozenMap,
    execution_profile: FrozenMap,
    run_request_contracts: FrozenMap,
    declared_sources: FrozenMap,
    generated_authorities: FrozenMap,
    vintage_authorities: FrozenMap,
    execution_abi: FrozenMap,
    seed_stream_map: SeedStreamMap,
    nodes: tuple[CompiledNode, ...],
    projections: FrozenMap,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "spec_binding": spec_binding.to_wire(),
        "compiler_ir_abi": compiler_ir_abi.to_wire(),
        "identity_generation": identity_generation,
        "normative_sha256": sha256_json(_wire(normative)),
        "execution_profile_sha256": sha256_json(_wire(execution_profile)),
        "run_request_contracts_sha256": sha256_json(_wire(run_request_contracts)),
        "declared_sources_sha256": declared_sources["sha256"],
        "generated_authorities_sha256": sha256_json(_wire(generated_authorities)),
        "vintage_authorities_sha256": sha256_json(_wire(vintage_authorities)),
        "execution_abi_sha256": execution_abi["sha256"],
        "projection_sha256": sha256_json(_wire(projections)),
        "seed_stream_map_sha256": sha256_json(seed_stream_map.to_wire()),
        "node_keys": [node.node_key for node in nodes],
    }


@dataclass(frozen=True, slots=True)
class RuntimeAuthorities:
    """One compiler-issued, immutable authority capability for a run."""

    grammar_receipt: GrammarReceipt
    spec_binding: SpecBinding
    compiler_ir_abi: CompilerIRABI
    identity_generation: int
    normative: FrozenMap
    execution_profile: FrozenMap
    run_request_contracts: FrozenMap
    declared_sources: FrozenMap
    generated_authorities: FrozenMap
    vintage_authorities: FrozenMap
    execution_abi: FrozenMap
    seed_stream_map: SeedStreamMap
    nodes: tuple[CompiledNode, ...]
    projections: FrozenMap
    authority_sha256: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.identity_generation, bool)
            or not isinstance(self.identity_generation, int)
            or self.identity_generation != 1
        ):
            raise RuntimeAuthorityError(
                "bundle runtime authority requires identity_generation 1"
            )
        for name in (
            "normative",
            "execution_profile",
            "run_request_contracts",
            "declared_sources",
            "generated_authorities",
            "vintage_authorities",
            "execution_abi",
            "projections",
        ):
            if not isinstance(getattr(self, name), FrozenMap):
                raise RuntimeAuthorityError(f"{name} authority must be frozen")
        if not isinstance(self.nodes, tuple) or not all(
            isinstance(node, CompiledNode) for node in self.nodes
        ):
            raise RuntimeAuthorityError("compiled nodes must be an immutable tuple")
        expected = sha256_json(self.identity_wire())
        if self.authority_sha256 != expected:
            raise RuntimeAuthorityError(
                "runtime authority digest differs from its compiled identity"
            )

    def resource(self, kind: str, *, surface: str = "normative") -> FrozenMap:
        surfaces = {
            "normative": self.normative,
            "execution_profile": self.execution_profile,
            "run_request": self.run_request_contracts,
        }
        try:
            selected = surfaces[surface]
        except KeyError as error:
            raise RuntimeAuthorityError(
                f"unknown runtime authority surface {surface!r}"
            ) from error
        value = selected.get(kind)
        if not isinstance(value, FrozenMap):
            raise RuntimeAuthorityError(
                f"runtime authority has no object resource {kind!r} on {surface!r}"
            )
        return value

    def projection(self, name: str) -> FrozenMap:
        value = self.projections.get(name)
        if not isinstance(value, FrozenMap):
            raise RuntimeAuthorityError(f"runtime projection {name!r} is absent")
        return value

    def identity_wire(self) -> dict[str, object]:
        return _runtime_identity_wire(
            spec_binding=self.spec_binding,
            compiler_ir_abi=self.compiler_ir_abi,
            identity_generation=self.identity_generation,
            normative=self.normative,
            execution_profile=self.execution_profile,
            run_request_contracts=self.run_request_contracts,
            declared_sources=self.declared_sources,
            generated_authorities=self.generated_authorities,
            vintage_authorities=self.vintage_authorities,
            execution_abi=self.execution_abi,
            seed_stream_map=self.seed_stream_map,
            nodes=self.nodes,
            projections=self.projections,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            **self.identity_wire(),
            "authority_sha256": self.authority_sha256,
        }

    def run_provenance_identity(
        self,
        *,
        run_request: Mapping[str, object],
        execution_receipt: Mapping[str, object],
    ) -> RunProvenanceIdentity:
        binding = self.spec_binding.to_wire()
        binding["attestation"] = "bundle-authoritative"
        execution_abi = _mapping(_wire(self.execution_abi), location="execution_abi")
        return build_run_provenance_identity(
            identity_generation=self.identity_generation,
            source_grammar_receipt=self.grammar_receipt.to_wire(),
            spec_binding=binding,
            authority_versions={
                "runtime_authority": self.authority_sha256,
                "execution_abi": execution_abi["sha256"],
            },
            code_inventory_digest=self.compiler_ir_abi.sha256,
            artifact_protocol_inventory={
                "pipeline": execution_abi.get("pipeline"),
                "normative_artifact_vector": execution_abi.get(
                    "normative_artifact_vector"
                ),
            },
            run_request=run_request,
            execution_receipt=execution_receipt,
        )


def compile_runtime_authorities(compiled: CompiledSpecIR) -> RuntimeAuthorities:
    """Issue the sole runtime capability for a compiled plan."""

    if not isinstance(compiled, CompiledSpecIR):
        raise TypeError("compile_runtime_authorities requires CompiledSpecIR")
    normative = _frozen(
        _surface_domains(compiled, "normative"), location="normative authority"
    )
    execution_profile = _frozen(
        _surface_domains(compiled, "execution_profile"),
        location="execution-profile authority",
    )
    run_request_contracts = _frozen(
        _surface_domains(compiled, "run_request"),
        location="run-request authority",
    )
    declared_sources = _compile_declared_sources(compiled)
    bundle = normative.get("bundle")
    if not isinstance(bundle, FrozenMap):
        raise RuntimeAuthorityError("normative bundle authority is absent")
    identity_generation = bundle.get("identity_generation")
    if (
        isinstance(identity_generation, bool)
        or not isinstance(identity_generation, int)
        or identity_generation != 1
    ):
        raise RuntimeAuthorityError(
            "normative bundle must select identity_generation 1"
        )
    try:
        projections = _compile_projections(compiled)
    except SpecValidationError as error:
        raise RuntimeAuthorityError(
            f"compiled runtime projection is invalid: {error}"
        ) from error
    identity = _runtime_identity_wire(
        spec_binding=compiled.spec_binding,
        compiler_ir_abi=compiled.compiler_ir_abi,
        identity_generation=identity_generation,
        normative=normative,
        execution_profile=execution_profile,
        run_request_contracts=run_request_contracts,
        declared_sources=declared_sources,
        generated_authorities=compiled.generated_authorities,
        vintage_authorities=compiled.vintage_authorities,
        execution_abi=compiled.execution_abi,
        seed_stream_map=compiled.seed_stream_map,
        nodes=compiled.nodes,
        projections=projections,
    )
    return RuntimeAuthorities(
        grammar_receipt=compiled.grammar_receipt,
        spec_binding=compiled.spec_binding,
        compiler_ir_abi=compiled.compiler_ir_abi,
        identity_generation=identity_generation,
        normative=normative,
        execution_profile=execution_profile,
        run_request_contracts=run_request_contracts,
        declared_sources=declared_sources,
        generated_authorities=compiled.generated_authorities,
        vintage_authorities=compiled.vintage_authorities,
        execution_abi=compiled.execution_abi,
        seed_stream_map=compiled.seed_stream_map,
        nodes=compiled.nodes,
        projections=projections,
        authority_sha256=sha256_json(identity),
    )


__all__ = [
    "RuntimeAuthorities",
    "RuntimeAuthorityError",
    "compile_runtime_authorities",
]
