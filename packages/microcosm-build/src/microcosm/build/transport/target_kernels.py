"""Target compilation, the ordered calibration problem, and hold-out comparison.

Three country-neutral graph kernels share one path from published facts to
measured aggregates:

- ``targets.compile@1`` resolves a node's target references against one
  Chronicle consumer artifact (a declared source) with the shared
  :func:`~microcosm.build.ledger_targets.compile_ledger_target_references`,
  and emits the compiled registry as a ``microcosm.targets.surface`` artifact.
  The surface also carries a trace from every compiled target to the one
  reference that produced it and the fact it resolved to.
- ``targets.problem@1`` turns a surface into the portable
  ``microcosm.calibrate.ordered-problem`` artifact: it rebuilds the registry,
  takes :meth:`~microcosm.calibrate.TargetRegistry.to_target_set`, compiles
  the constraint matrix on the node's population with
  :func:`~microcosm.calibrate.matrix.build_constraint_matrix`, and encodes it
  with :func:`~microcosm.calibrate.artifacts.encode_problem`. The neutral
  solve kernel ``calibrate.ordered_adam@1`` consumes it.
- ``takeup.compare@1`` compiles hold-out references the same way and measures
  them on the population with its current weights, so a comparator such as a
  published recipient count is read through exactly the semantics a
  calibration target would have been. It reports; it does not gate.

References arrive as a node parameter: a canonical-JSON document with the
shape of a country's ``target_references.json`` (``country``,
``schema_version``, ``hierarchy`` and the node's ``target_references``),
parsed by the same validator :func:`~microcosm.build.country_spec.load_country_spec`
uses, plus the SHA-256 of the spec resource the rows came from; a document
key outside that shape is refused rather than ignored. Every
reference must name prepared columns: a measure or filter column the node's
slices do not carry refuses the node rather than skipping the target.

Compilation is strict. ``compile_ledger_target_references`` raises on a
placeholder reference (any ``activation_status`` other than empty or
``"active"``, ``ledger_targets.py`` ``_require_executable_reference``), on a
reference no fact matches and on an ambiguous match, and this module adds no
skip path: a compiled surface therefore contains one target per reference,
each traced to one executable reference.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import microcosm.build.chronicle_epoch as chronicle_epoch_module
import microcosm.build.country_spec as country_spec_module
import microcosm.build.ledger_artifact as ledger_artifact_module
import microcosm.build.ledger_targets as ledger_targets_module
import microcosm.calibrate.artifacts as calibrate_artifacts_module
import microcosm.calibrate.hierarchy as hierarchy_module
import microcosm.calibrate.matrix as matrix_module
import microcosm.calibrate.monetary_binding as monetary_binding_module
import microcosm.calibrate.registry as registry_module
import microcosm.calibrate.target as target_module
import microcosm.frame.bundle as frame_bundle_module
import microcosm.frame.schema as frame_schema_module
import microcosm.frame.weights as frame_weights_module
from microcosm.build.country_spec import _validate_target_references
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.calibrate.artifacts import encode_problem
from microcosm.calibrate.matrix import CalibrationProblem, build_constraint_matrix
from microcosm.frame import Frame
from microcosm.graph import (
    Capabilities,
    Determinism,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Numeric,
    SeedSource,
    source_hash,
)
from microcosm.graph.canonical import canonical_json

from . import artifact_types, graph_inputs
from .artifact_types import COMPARISON_TYPE, TARGET_SURFACE_TYPE
from .graph_inputs import (
    canonical_document_param,
    context_frame,
    entities_param,
    require_outputs,
    require_params,
    sha256_param,
    sha256_text,
    single_source,
    string_param,
)

__all__ = [
    "COMPARISON_TYPE",
    "TARGETS_COMPILE",
    "TARGETS_PROBLEM",
    "TAKEUP_COMPARE",
    "TARGET_SURFACE_TYPE",
    "TakeupCompareKernel",
    "TargetSurface",
    "TargetsCompileKernel",
    "TargetsProblemKernel",
    "compile_target_surface",
    "decode_target_surface",
    "encode_target_surface",
    "parse_reference_document",
    "register_target_kernels",
]

_SURFACE_KIND = "transport_target_surface"
_COMPARISON_KIND = "transport_target_comparison"
_PROBLEM_BINDINGS_SCHEMA = "microcosm.transport.problem-bindings.v1"

#: The keys of a node's target-reference document: the subset of a country's
#: ``target_references.json`` its references need.
_REFERENCE_DOCUMENT_KEYS = frozenset(
    {"country", "schema_version", "hierarchy", "target_references"}
)

#: Statuses ``compile_ledger_target_references`` treats as executable
#: (``ledger_targets.py`` ``_require_executable_reference``): an absent or
#: empty ``activation_status``, or ``"active"``.
_EXECUTABLE_STATUSES = frozenset({"", "active"})


# ---------------------------------------------------------------------------
# References and the surface artifact
# ---------------------------------------------------------------------------


def parse_reference_document(
    document: Mapping[str, object], *, country: str
) -> tuple[LedgerTargetReference, ...]:
    """Parse a target-references document exactly as the spec loader does.

    The validator is the country-spec loader's own
    (``country_spec._validate_target_references``), so a reference a node
    receives parses to the object ``load_country_spec`` would have built from
    the same rows. Reference names must be unique: the surface traces each
    compiled target to its reference by name.
    """

    unknown = sorted(set(document) - _REFERENCE_DOCUMENT_KEYS)
    if unknown:
        raise ValueError(
            f"A node's target-reference document carries only "
            f"{sorted(_REFERENCE_DOCUMENT_KEYS)}; got also {unknown}. Keys the "
            "kernel would not read must not ride in its parameters."
        )
    references = _validate_target_references(document, country=country)
    names = [reference.name for reference in references]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(
            f"Target references repeat the name(s) {duplicated}; each compiled "
            "target must trace to exactly one reference."
        )
    return references


@dataclass(frozen=True)
class TargetSurface:
    """A decoded ``microcosm.targets.surface`` artifact.

    Attributes:
        country: The registry's country label.
        registry: The compiled registry, rebuilt from its specs.
        trace: One row per spec, in registry order: the reference that
            produced the target and the fact identity it resolved to.
        references_sha256: The spec resource the references came from.
        facts: Content identity of the consumer artifact compiled against.
        sha256: SHA-256 of the artifact bytes.
    """

    country: str
    registry: TargetRegistry
    trace: tuple[Mapping[str, object], ...]
    references_sha256: str
    facts: Mapping[str, object]
    sha256: str


def _trace_row(spec: TargetSpec, reference: LedgerTargetReference) -> dict[str, object]:
    return {
        "target": spec.name,
        "period": spec.period,
        "reference": reference.name,
        "activation_status": reference.metadata.get("activation_status", ""),
        "ledger_fact_key": spec.metadata.get("ledger_fact_key", ""),
        "ledger_source_record_id": spec.metadata.get("ledger_source_record_id", ""),
    }


def encode_target_surface(
    registry: TargetRegistry,
    references: Sequence[LedgerTargetReference],
    *,
    references_sha256: str,
    facts: Mapping[str, object],
) -> bytes:
    """Encode a compiled registry with its one-to-one reference trace."""

    specs = registry.specs
    if len(specs) != len(references) or any(
        spec.name != reference.name
        for spec, reference in zip(specs, references, strict=True)
    ):
        raise ValueError(
            "A target surface needs exactly one compiled target per reference, "
            "in reference order."
        )
    payload = canonical_json(
        {
            "schema_version": 1,
            "kind": _SURFACE_KIND,
            "country": registry.country,
            "registry": {
                "country": registry.country,
                "version": registry.version,
                "specs": [spec.to_dict() for spec in specs],
            },
            "trace": [
                _trace_row(spec, reference)
                for spec, reference in zip(specs, references, strict=True)
            ],
            "references_sha256": references_sha256,
            "facts": dict(facts),
        }
    )
    decode_target_surface(payload)
    return payload


def decode_target_surface(payload: bytes) -> TargetSurface:
    """Validate a surface's shape, its registry identity and its trace."""

    if type(payload) is not bytes:
        raise TypeError("A target surface is immutable bytes.")
    document = json.loads(payload)
    if not isinstance(document, dict) or canonical_json(document) != payload:
        raise ValueError("A target surface must be canonical JSON.")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != _SURFACE_KIND
        or set(document)
        != {
            "schema_version",
            "kind",
            "country",
            "registry",
            "trace",
            "references_sha256",
            "facts",
        }
    ):
        raise ValueError("Unsupported target surface artifact.")
    country = document["country"]
    raw_registry = document["registry"]
    if not isinstance(raw_registry, dict) or raw_registry.get("country") != country:
        raise ValueError("Target surface registry country differs from the surface.")
    registry = TargetRegistry(
        [TargetSpec.from_dict(spec) for spec in raw_registry["specs"]],
        country=country,
    )
    if registry.version != raw_registry.get("version"):
        raise ValueError("Target surface registry version differs from its specs.")
    trace = document["trace"]
    if not isinstance(trace, list) or len(trace) != len(registry.specs):
        raise ValueError("Target surface trace must cover every compiled target.")
    references = [row.get("reference") for row in trace]
    if len(set(references)) != len(references):
        raise ValueError("Target surface traces two targets to one reference.")
    for spec, row in zip(registry.specs, trace, strict=True):
        if (
            row.get("target") != spec.name
            or row.get("period") != spec.period
            or row.get("reference") != spec.name
            or row.get("activation_status") not in _EXECUTABLE_STATUSES
        ):
            raise ValueError(
                f"Target surface trace for {spec.name!r} does not name one "
                "executable reference."
            )
    return TargetSurface(
        country=country,
        registry=registry,
        trace=tuple(trace),
        references_sha256=document["references_sha256"],
        facts=document["facts"],
        sha256=sha256_text(payload),
    )


def compile_target_surface(
    facts_path: Path,
    document: Mapping[str, object],
    *,
    country: str,
    references_sha256: str,
    expected_facts_sha256: str | None = None,
) -> tuple[bytes, TargetRegistry]:
    """Compile a reference document against one consumer artifact.

    Returns the surface bytes and the registry they encode. The artifact
    payload records content identities only, never the source path, so moving
    a source file changes neither a node key nor an artifact byte.
    """

    references = parse_reference_document(document, country=country)
    artifact = load_ledger_consumer_artifact(
        facts_path, expected_facts_sha256=expected_facts_sha256
    )
    registry = compile_ledger_target_references(
        artifact.facts, references, country=country
    )
    payload = encode_target_surface(
        registry,
        references,
        references_sha256=references_sha256,
        facts={
            "facts_sha256": artifact.facts_sha256,
            "manifest_sha256": artifact.manifest_sha256,
            "n_facts": len(artifact.facts),
        },
    )
    return payload, registry


def _registry_from_context(
    context: KernelContext, ref: str
) -> tuple[TargetRegistry, bytes]:
    """Compile the node's declared references against its one source."""

    country = string_param(context, ref, "country")
    document = canonical_document_param(context, ref, "references")
    references_sha256 = sha256_param(context, ref, "references_sha256")
    expected = (
        sha256_param(context, ref, "facts_sha256")
        if "facts_sha256" in context.params
        else None
    )
    source = single_source(context, ref)
    payload, registry = compile_target_surface(
        context.sources[source],
        document,
        country=country,
        references_sha256=references_sha256,
        expected_facts_sha256=expected,
    )
    return registry, payload


# ---------------------------------------------------------------------------
# Matrix compilation shared by the problem and the comparison
# ---------------------------------------------------------------------------


def _compiled_problem(
    frame: Frame, registry: TargetRegistry, weight_entity: str, ref: str
) -> CalibrationProblem:
    problem = build_constraint_matrix(frame, registry.to_target_set(), weight_entity)
    if problem.skipped:
        reasons = "; ".join(
            f"{item.target.name}: {item.reason}" for item in problem.skipped[:5]
        )
        raise ValueError(
            f"{ref} refuses {len(problem.skipped)} uncompilable target(s); every "
            f"reference must name columns the node slices ({reasons})."
        )
    expected = tuple(f"{spec.name}@{spec.period}" for spec in registry.specs)
    if problem.names != expected:
        raise ValueError(f"{ref} compiled rows differ from the registry order.")
    return problem


def _entity_ids(frame: Frame, entity: str) -> list[int | str]:
    column = frame.schema.entity_id_column(entity)
    return frame.table(entity)[column].tolist()


def _optional_float(value: float) -> float | None:
    value = float(value)
    return value if math.isfinite(value) else None


_COMPILE_MODULES = (
    graph_inputs,
    artifact_types,
    country_spec_module,
    ledger_targets_module,
    ledger_artifact_module,
    chronicle_epoch_module,
    registry_module,
    target_module,
    hierarchy_module,
)
_MATRIX_MODULES = (
    matrix_module,
    monetary_binding_module,
    target_module,
    frame_bundle_module,
    frame_schema_module,
    frame_weights_module,
)


class TargetsCompileKernel(KernelBase):
    """``targets.compile@1``: references plus facts become a target surface."""

    ref = "targets.compile@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
    )
    _required = frozenset({"country", "references", "references_sha256"})
    _optional = frozenset({"facts_sha256"})

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            *_COMPILE_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context, self.ref, required=self._required, optional=self._optional
        )
        require_outputs(context, self.ref)
        registry, payload = _registry_from_context(context, self.ref)
        return KernelResult(
            artifacts={"surface": payload},
            receipt={
                "country": registry.country,
                "registry_version": registry.version,
                "n_targets": len(registry.specs),
                "surface_sha256": sha256_text(payload),
            },
        )


class TargetsProblemKernel(KernelBase):
    """``targets.problem@1``: a surface measured on the node's population."""

    ref = "targets.problem@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas", "scipy"),
    )
    _required = frozenset({"entities", "weight_entity"})

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            graph_inputs,
            artifact_types,
            registry_module,
            hierarchy_module,
            calibrate_artifacts_module,
            *_MATRIX_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(context, self.ref, required=self._required)
        require_outputs(context, self.ref)
        entities = entities_param(context, self.ref, "entities")
        weight_entity = string_param(context, self.ref, "weight_entity")
        value = context.artifacts.get("surface")
        if value is None or value.type != TARGET_SURFACE_TYPE:
            raise ValueError(
                f"{self.ref} reads a {TARGET_SURFACE_TYPE.name} artifact "
                "under the alias 'surface'."
            )
        surface = decode_target_surface(value.payload)
        frame = context_frame(
            context, self.ref, entities=entities, weight_entity=weight_entity
        )
        problem = _compiled_problem(frame, surface.registry, weight_entity, self.ref)
        target_metadata = [
            {
                **spec.metadata,
                "reference": spec.name,
                "period": str(spec.period),
                "entity": spec.entity,
                "measure": spec.measure,
                "filter": spec.filter or "",
                "family": spec.family,
                "source": spec.source,
            }
            for spec in surface.registry.specs
        ]
        payload = encode_problem(
            problem,
            entity_ids=_entity_ids(frame, weight_entity),
            target_metadata=target_metadata,
            bindings={
                "schema": _PROBLEM_BINDINGS_SCHEMA,
                "country": surface.country,
                "registry_version": surface.registry.version,
                "surface_sha256": surface.sha256,
                "entities": list(entities),
                "weight_entity": weight_entity,
            },
        )
        return KernelResult(
            artifacts={"problem": payload},
            receipt={
                "problem_sha256": sha256_text(payload),
                "surface_sha256": surface.sha256,
                "registry_version": surface.registry.version,
                "n_targets": int(problem.n_targets),
                "n_records": int(problem.n_weights),
                "weight_kind": problem.initial_weights.kind.value,
            },
        )


class TakeupCompareKernel(KernelBase):
    """``takeup.compare@1``: hold-out comparators measured on the population."""

    ref = "takeup.compare@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        seed_source=SeedSource.NONE,
        dependencies=("numpy", "pandas", "scipy"),
    )
    _required = frozenset(
        {"country", "references", "references_sha256", "entities", "weight_entity"}
    )
    _optional = frozenset({"facts_sha256"})

    def implementation_hash(self) -> str:
        return source_hash(
            type(self),
            *_COMPILE_MODULES,
            *_MATRIX_MODULES,
            dependencies=self.capabilities.dependencies,
        )

    def run(self, context: KernelContext) -> KernelResult:
        require_params(
            context, self.ref, required=self._required, optional=self._optional
        )
        require_outputs(context, self.ref)
        entities = entities_param(context, self.ref, "entities")
        weight_entity = string_param(context, self.ref, "weight_entity")
        registry, surface_payload = _registry_from_context(context, self.ref)
        surface = decode_target_surface(surface_payload)
        frame = context_frame(
            context, self.ref, entities=entities, weight_entity=weight_entity
        )
        problem = _compiled_problem(frame, registry, weight_entity, self.ref)
        weights = frame.weights_for(weight_entity)
        model = problem.estimates(weights.values)
        rows = []
        for spec, trace, estimate in zip(
            registry.specs, surface.trace, model, strict=True
        ):
            comparator = float(spec.value)
            estimate = float(estimate)
            rows.append(
                {
                    "name": spec.name,
                    "period": spec.period,
                    "entity": spec.entity,
                    "measure": spec.measure,
                    "filter": spec.filter,
                    "family": spec.family,
                    "source": spec.source,
                    "reference": trace["reference"],
                    "ledger_fact_key": trace["ledger_fact_key"],
                    "comparator": comparator,
                    "model": _optional_float(estimate),
                    "difference": _optional_float(estimate - comparator),
                    "ratio": (
                        None
                        if comparator == 0
                        else _optional_float(estimate / comparator)
                    ),
                }
            )
        payload = canonical_json(
            {
                "schema_version": 1,
                "kind": _COMPARISON_KIND,
                "country": registry.country,
                "registry_version": registry.version,
                "references_sha256": surface.references_sha256,
                "facts": dict(surface.facts),
                "weight_entity": weight_entity,
                "weight_kind": weights.kind.value,
                "weight_total": _optional_float(np.sum(weights.values)),
                "comparisons": rows,
            }
        )
        return KernelResult(
            artifacts={"comparison": payload},
            receipt={
                "n_comparisons": len(rows),
                "registry_version": registry.version,
                "comparison_sha256": sha256_text(payload),
            },
        )


TARGETS_COMPILE = TargetsCompileKernel()
TARGETS_PROBLEM = TargetsProblemKernel()
TAKEUP_COMPARE = TakeupCompareKernel()


def register_target_kernels(registry: KernelRegistry) -> None:
    """Register the three kernels; repeating the call is a no-op."""

    for kernel in (TARGETS_COMPILE, TARGETS_PROBLEM, TAKEUP_COMPARE):
        registry.register(kernel)
