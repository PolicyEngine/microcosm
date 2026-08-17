"""Pure generation-0 stacked-authority and checkpoint identity projections.

The constants-era stacked builder combines several independently owned
contracts into one cache identity.  F0 must reproduce that identity without
constructing the executor authority (an F1 responsibility), so this module
reads only a normalized :class:`ResolvedSpec`, its generated authorities, and
the run-request values that the historical identity accepted.

``canonical_identity`` in the compatibility receipt is consequently a
mirror assertion: it means that the normalized bundle supplied every closed
component of the generation-0 semantic object.  It is not a bundle-mode
runtime authority construction, and the enclosing ``SpecBinding`` remains
``mirror-attested`` until F1.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy

from .battery_semantics import project_battery_authority_components
from .canonical import canonical_json_bytes, sha256_json
from .errors import SpecValidationError
from .imputation_semantics import project_imputation_legacy_payloads
from .model import ResolvedSpec, thaw_json
from .take_up_semantics import project_legacy_take_up_contract

_CANONICAL_AUTHORITY_FORM = "CANONICAL"
_RECIPIENT_SELECTION = "target_specific_complement_of_declared_producer_rows"
_LATE_BATCH_SUFFIX = re.compile(r"__batch_[0-9]+$")
_ENGINE_VERSION_REF = {
    "kind": "engine_abi_lock",
    "pointer": "/engine/version",
}
_REMAINING_STAGE_VERSIONED_CONTRACTS = (
    "ssi_dependency_contract",
    "engine_input_projection_contract",
)


def _generation_zero_sha256(value: object) -> str:
    """Hash one legacy component with its original ASCII-escaped JSON codec."""

    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SpecValidationError(f"{location}: object required")
    return value


def _array(value: object, *, location: str) -> Sequence[object]:
    if not isinstance(value, (list, tuple)):
        raise SpecValidationError(f"{location}: array required")
    return value


def _nonempty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecValidationError(f"{location}: non-empty string required")
    return value


def _integer(value: object, *, location: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecValidationError(f"{location}: integer required")
    if minimum is not None and value < minimum:
        raise SpecValidationError(f"{location}: must be at least {minimum}")
    return value


def _finite_float(value: object, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SpecValidationError(f"{location}: finite number required")
    result = float(value)
    if not math.isfinite(result):
        raise SpecValidationError(f"{location}: finite number required")
    return result


def _domain(spec: ResolvedSpec, kind: str) -> dict[str, object]:
    value = spec.domain(kind).to_wire()
    if not isinstance(value, dict):  # pragma: no cover - domain schemas guarantee
        raise SpecValidationError(f"{kind}: normalized domain must be an object")
    return value


def _us_domains(spec: ResolvedSpec) -> dict[str, dict[str, object]]:
    if spec.country != "us":
        raise SpecValidationError(
            "stacked checkpoint semantics require the normalized US bundle"
        )
    return {
        kind: _domain(spec, kind)
        for kind in (
            "battery",
            "bundle",
            "imputation",
            "publication",
            "sources",
            "spine",
            "take_up",
        )
    }


def _generated_engine_lock(spec: ResolvedSpec) -> dict[str, object]:
    generated = thaw_json(spec.generated_authorities)
    if not isinstance(generated, dict):  # pragma: no cover - model invariant
        raise SpecValidationError("generated_authorities: object required")
    lock = _mapping(
        generated.get("engine_abi_lock"),
        location="generated_authorities/engine_abi_lock",
    )
    return deepcopy(dict(lock))


def _imputation_projection(
    domains: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return project_imputation_legacy_payloads(
        domains["imputation"],
        sources_document=domains["sources"],
        spine_document=domains["spine"],
        bundle_document=domains["bundle"],
    )


def _late_families(
    imputation: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        family
        for index, value in enumerate(
            _array(imputation.get("families"), location="imputation/families")
        )
        for family in [
            _mapping(value, location=f"imputation/families/{index}")
        ]
        if family.get("stage") == "late_producer_dag"
    )


def _late_transfer_surface(
    imputation: Mapping[str, object],
) -> tuple[dict[str, dict[str, list[str]]], str, int, str]:
    """Compile the 70-target post-PUF surface and its shared donor contract."""

    surface: dict[str, dict[str, list[str]]] = {}
    donor_channels: set[str] = set()
    donor_clone_indices: set[int] = set()
    recipient_selections: set[str] = set()
    seen_targets: set[tuple[str, str, str]] = set()
    for index, family in enumerate(_late_families(imputation)):
        family_id = _nonempty_string(
            family.get("id"), location=f"imputation/late_families/{index}/id"
        )
        parts = family_id.split("/", 2)
        if len(parts) != 3 or parts[0] != "late":
            raise SpecValidationError(
                f"imputation/late_families/{index}/id: malformed late family"
            )
        entity = parts[1]
        family_name = _LATE_BATCH_SUFFIX.sub("", parts[2])
        targets = surface.setdefault(entity, {}).setdefault(family_name, [])
        for target_index, value in enumerate(
            _array(
                family.get("targets"),
                location=f"imputation/late_families/{index}/targets",
            )
        ):
            target = _mapping(
                value,
                location=(
                    f"imputation/late_families/{index}/targets/{target_index}"
                ),
            )
            name = _nonempty_string(
                target.get("name"),
                location=(
                    f"imputation/late_families/{index}/targets/{target_index}/name"
                ),
            )
            key = (entity, family_name, name)
            if key in seen_targets:
                raise SpecValidationError(
                    "imputation late transfer surface repeats "
                    f"{entity}/{family_name}/{name}"
                )
            seen_targets.add(key)
            targets.append(name)

        donor = _mapping(
            family.get("donor_contract"),
            location=f"imputation/late_families/{index}/donor_contract",
        )
        projection = _mapping(
            donor.get("projection"),
            location=(
                f"imputation/late_families/{index}/donor_contract/projection"
            ),
        )
        donor_channels.add(
            _nonempty_string(
                projection.get("support_channel"),
                location=(
                    "imputation/late_families/"
                    f"{index}/donor_contract/projection/support_channel"
                ),
            )
        )
        donor_clone_indices.add(
            _integer(
                projection.get("support_clone_index"),
                location=(
                    "imputation/late_families/"
                    f"{index}/donor_contract/projection/support_clone_index"
                ),
                minimum=0,
            )
        )
        recipient = _mapping(
            family.get("recipient_contract"),
            location=f"imputation/late_families/{index}/recipient_contract",
        )
        recipient_selections.add(
            _nonempty_string(
                recipient.get("selection"),
                location=(
                    "imputation/late_families/"
                    f"{index}/recipient_contract/selection"
                ),
            )
        )

    if len(donor_channels) != 1 or len(donor_clone_indices) != 1:
        raise SpecValidationError(
            "imputation late families must share one donor channel and clone index"
        )
    if recipient_selections != {_RECIPIENT_SELECTION}:
        raise SpecValidationError(
            "imputation late families have no reviewed recipient selection"
        )
    # Transfer-chain order is normally the late-family order.  A family whose
    # complete surface comes from one of the post-clone source producers keeps
    # that producer's declared output order instead.  Immigration is the
    # generation-0 case where the joint transfer order (SSN then status) differs
    # from the authority surface order (status then SSN).
    source_order = _source_output_order(imputation)
    for entity, families in surface.items():
        for targets in families.values():
            keys = [(entity, target) for target in targets]
            if all(key in source_order for key in keys):
                targets.sort(key=lambda target: source_order[(entity, target)])
    return (
        surface,
        donor_channels.pop(),
        donor_clone_indices.pop(),
        recipient_selections.pop(),
    )


def _primary_puf_output_keys(
    imputation: Mapping[str, object],
) -> set[tuple[str, str]]:
    primary = [
        family
        for value in _array(
            imputation.get("families"), location="imputation/families"
        )
        for family in [_mapping(value, location="imputation/families row")]
        if family.get("stage") == "primary_puf_qrf"
    ]
    if len(primary) != 1:
        raise SpecValidationError(
            "imputation/families: exactly one primary PUF family required"
        )
    keys: set[tuple[str, str]] = set()
    for index, value in enumerate(
        _array(primary[0].get("targets"), location="imputation/primary/targets")
    ):
        target = _mapping(value, location=f"imputation/primary/targets/{index}")
        if target.get("output_coverage_scope") != "puf_clone":
            continue
        keys.add(
            (
                _nonempty_string(
                    target.get("entity"),
                    location=f"imputation/primary/targets/{index}/entity",
                ),
                _nonempty_string(
                    target.get("name"),
                    location=f"imputation/primary/targets/{index}/name",
                ),
            )
        )
    return keys


def _source_output_order(
    imputation: Mapping[str, object],
) -> dict[tuple[str, str], int]:
    graph = _mapping(
        imputation.get("producer_graph"), location="imputation/producer_graph"
    )
    result: dict[tuple[str, str], int] = {}
    for node_index, node_value in enumerate(
        _array(graph.get("nodes"), location="imputation/producer_graph/nodes")
    ):
        node = _mapping(
            node_value, location=f"imputation/producer_graph/nodes/{node_index}"
        )
        if node.get("kind") != "post_clone_source":
            continue
        for output_index, output_value in enumerate(
            _array(
                node.get("outputs"),
                location=f"imputation/producer_graph/nodes/{node_index}/outputs",
            )
        ):
            output = _mapping(
                output_value,
                location=(
                    "imputation/producer_graph/nodes/"
                    f"{node_index}/outputs/{output_index}"
                ),
            )
            column = _nonempty_string(
                output.get("column"),
                location=(
                    "imputation/producer_graph/nodes/"
                    f"{node_index}/outputs/{output_index}/column"
                ),
            )
            if column.startswith("@"):
                continue
            key = (
                _nonempty_string(
                    output.get("entity"),
                    location=(
                        "imputation/producer_graph/nodes/"
                        f"{node_index}/outputs/{output_index}/entity"
                    ),
                ),
                column,
            )
            result.setdefault(key, len(result))
    return result


def _source_output_keys(
    imputation: Mapping[str, object],
) -> set[tuple[str, str]]:
    return set(_source_output_order(imputation))


def _filter_surface(
    surface: Mapping[str, Mapping[str, Sequence[str]]],
    output_keys: set[tuple[str, str]],
) -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for entity, families in surface.items():
        for family, targets in families.items():
            selected = [target for target in targets if (entity, target) in output_keys]
            if selected:
                result.setdefault(entity, {})[family] = selected
    return result


def _surface_target_count(
    surface: Mapping[str, Mapping[str, Sequence[object]]],
) -> int:
    return sum(
        len(targets) for families in surface.values() for targets in families.values()
    )


def _gap_fill_target_count(plan: Sequence[object]) -> int:
    count = 0
    for index, value in enumerate(plan):
        direction = _mapping(value, location=f"gap_fill_plan/{index}")
        surface = _mapping(
            direction.get("target_families"),
            location=f"gap_fill_plan/{index}/target_families",
        )
        for entity, families_value in surface.items():
            families = _mapping(
                families_value,
                location=f"gap_fill_plan/{index}/target_families/{entity}",
            )
            count += sum(
                len(_array(targets, location="gap-fill target array"))
                for targets in families.values()
            )
    return count


def _authority_component_payloads(
    domains: Mapping[str, Mapping[str, object]],
    imputation_payloads: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    battery = project_battery_authority_components(domains["battery"])
    transfer_surface, donor_channel, donor_clone_index, recipient_selection = (
        _late_transfer_surface(domains["imputation"])
    )
    puf_surface = _filter_surface(
        transfer_surface, _primary_puf_output_keys(domains["imputation"])
    )
    source_surface = _filter_surface(
        transfer_surface, _source_output_keys(domains["imputation"])
    )
    late_schedule = deepcopy(
        dict(
            _mapping(
                imputation_payloads.get("late_producer_schedule_receipt"),
                location="imputation projection/late_producer_schedule_receipt",
            )
        )
    )
    gap_fill_plan = deepcopy(
        list(
            _array(
                imputation_payloads.get("gap_fill_plan"),
                location="imputation projection/gap_fill_plan",
            )
        )
    )
    tail_support = _tail_support_contract(domains["spine"])[0]
    payloads = {
        "gap_fill_plan": gap_fill_plan,
        "post_puf_transfer_surface": {
            "donor_channel": donor_channel,
            "donor_clone_index": donor_clone_index,
            "recipient_selection": recipient_selection,
            "producer_surfaces": {
                "puf_clone": puf_surface,
                "post_clone_source": source_surface,
            },
            "target_families": transfer_surface,
        },
        "declared_surface": deepcopy(battery["declared_surface"]),
        "metric_registry": deepcopy(battery["metric_registry"]),
        "joint_metric_registry": deepcopy(battery["joint_metric_registry"]),
        "support_profile": deepcopy(battery["support_profile"]),
        "puf_capital_gains_tail_support_contract": tail_support,
        "late_producer_schedule": late_schedule,
    }
    counts = {
        "gap_fill_target_count": _gap_fill_target_count(gap_fill_plan),
        "gap_fill_direction_count": len(gap_fill_plan),
        "post_puf_target_count": _surface_target_count(transfer_surface),
        "puf_producer_target_count": _surface_target_count(puf_surface),
        "source_producer_target_count": _surface_target_count(source_surface),
        "declared_target_count": _surface_target_count(
            _mapping(
                battery["declared_surface"],
                location="battery projection/declared_surface",
            )
        ),
        "declared_entity_count": len(
            _mapping(
                battery["declared_surface"],
                location="battery projection/declared_surface",
            )
        ),
        "metric_target_count": len(
            _array(
                battery["metric_registry"],
                location="battery projection/metric_registry",
            )
        ),
        "joint_metric_target_count": len(
            _array(
                battery["joint_metric_registry"],
                location="battery projection/joint_metric_registry",
            )
        ),
    }
    return payloads, counts


def project_stacked_authority_receipt(spec: ResolvedSpec) -> dict[str, object]:
    """Project the exact constants-era stacked authority receipt.

    Component and bundle digests are compiled from their normalized owners.
    The duplicated ``declared_*`` values are compatibility aliases, not a
    second authored authority.
    """

    domains = _us_domains(spec)
    imputation_payloads = _imputation_projection(domains)
    payloads, counts = _authority_component_payloads(domains, imputation_payloads)
    binding = _mapping(
        domains["battery"].get("authority_binding"),
        location="battery/authority_binding",
    )
    authority_id = _nonempty_string(
        binding.get("authority_id"),
        location="battery/authority_binding/authority_id",
    )
    version = _integer(
        binding.get("version"),
        location="battery/authority_binding/version",
        minimum=1,
    )
    digests = {
        name: _generation_zero_sha256(payload) for name, payload in payloads.items()
    }
    bundle_sha256 = _generation_zero_sha256(
        {
            "authority_id": authority_id,
            "version": version,
            "components": payloads,
        }
    )
    support_profile = _mapping(
        payloads["support_profile"], location="authority/support_profile"
    )
    late_schedule = _mapping(
        payloads["late_producer_schedule"],
        location="authority/late_producer_schedule",
    )
    components: dict[str, dict[str, object]] = {
        "gap_fill_plan": {
            "sha256": digests["gap_fill_plan"],
            "declared_sha256": digests["gap_fill_plan"],
            "target_count": counts["gap_fill_target_count"],
            "direction_count": counts["gap_fill_direction_count"],
            "digest_matches_declared": True,
        },
        "post_puf_transfer_surface": {
            "sha256": digests["post_puf_transfer_surface"],
            "declared_sha256": digests["post_puf_transfer_surface"],
            "target_count": counts["post_puf_target_count"],
            "puf_producer_target_count": counts["puf_producer_target_count"],
            "source_producer_target_count": counts[
                "source_producer_target_count"
            ],
            "donor_channel": _mapping(
                payloads["post_puf_transfer_surface"],
                location="authority/post_puf_transfer_surface",
            )["donor_channel"],
            "donor_clone_index": _mapping(
                payloads["post_puf_transfer_surface"],
                location="authority/post_puf_transfer_surface",
            )["donor_clone_index"],
            "recipient_selection": _mapping(
                payloads["post_puf_transfer_surface"],
                location="authority/post_puf_transfer_surface",
            )["recipient_selection"],
            "digest_matches_declared": True,
        },
        "declared_surface": {
            "sha256": digests["declared_surface"],
            "declared_sha256": digests["declared_surface"],
            "target_count": counts["declared_target_count"],
            "entity_count": counts["declared_entity_count"],
            "digest_matches_declared": True,
        },
        "metric_registry": {
            "sha256": digests["metric_registry"],
            "declared_sha256": digests["metric_registry"],
            "target_count": counts["metric_target_count"],
            "digest_matches_declared": True,
        },
        "joint_metric_registry": {
            "sha256": digests["joint_metric_registry"],
            "declared_sha256": digests["joint_metric_registry"],
            "target_count": counts["joint_metric_target_count"],
            "digest_matches_declared": True,
        },
        "support_profile": {
            **deepcopy(dict(support_profile)),
            "sha256": digests["support_profile"],
            "declared_sha256": digests["support_profile"],
            "digest_matches_declared": True,
        },
        "puf_capital_gains_tail_support_contract": {
            "identity": deepcopy(
                payloads["puf_capital_gains_tail_support_contract"]
            ),
            "sha256": digests["puf_capital_gains_tail_support_contract"],
            "declared_sha256": digests[
                "puf_capital_gains_tail_support_contract"
            ],
            "digest_matches_declared": True,
        },
        "late_producer_schedule": {
            "identity": deepcopy(dict(late_schedule)),
            "sha256": digests["late_producer_schedule"],
            "declared_sha256": digests["late_producer_schedule"],
            "schedule_sha256": late_schedule.get("schedule_sha256"),
            "producer_count": late_schedule.get("producer_count"),
            "digest_matches_declared": True,
        },
    }
    return {
        "authority_id": authority_id,
        "version": version,
        "authority_form": _CANONICAL_AUTHORITY_FORM,
        "declared_authority_form": _CANONICAL_AUTHORITY_FORM,
        "canonical": True,
        "production_manifest_permitted": True,
        "canonical_identity": True,
        "canonical_content": True,
        "integrity_valid": True,
        "sha256": bundle_sha256,
        "declared_sha256": bundle_sha256,
        "digest_matches_declared": True,
        "components": components,
    }


def _build_model_seed(spec: ResolvedSpec) -> int:
    defaults = {
        site.default
        for site in spec.seed_protocol.sites
        if site.value_source == "run_request.build_model_seed"
        and site.default is not None
    }
    if len(defaults) != 1:
        raise SpecValidationError(
            "seed_protocol: build_model_seed sites must share one default"
        )
    return _integer(defaults.pop(), location="seed_protocol/build_model_seed")


def _tail_support_contract(
    spine: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    matches = [
        role
        for index, value in enumerate(
            _array(spine.get("support_roles"), location="spine/support_roles")
        )
        for role in [_mapping(value, location=f"spine/support_roles/{index}")]
        if role.get("id") == "puf_tax_detail"
    ]
    if len(matches) != 1:
        raise SpecValidationError(
            "spine/support_roles: exactly one puf_tax_detail role required"
        )
    tail = _mapping(
        matches[0].get("tail_support"),
        location="spine/support_roles/puf_tax_detail/tail_support",
    )
    contract = _mapping(
        tail.get("legacy_contract"),
        location=(
            "spine/support_roles/puf_tax_detail/tail_support/legacy_contract"
        ),
    )
    schema_version = _integer(
        tail.get("manifest_schema_version"),
        location=(
            "spine/support_roles/puf_tax_detail/tail_support/"
            "manifest_schema_version"
        ),
        minimum=1,
    )
    return deepcopy(dict(contract)), schema_version


def _acs_earnings_contract(imputation: Mapping[str, object]) -> dict[str, object]:
    graph = _mapping(
        imputation.get("producer_graph"), location="imputation/producer_graph"
    )
    matches: list[Mapping[str, object]] = []
    for node_index, node_value in enumerate(
        _array(graph.get("nodes"), location="imputation/producer_graph/nodes")
    ):
        node = _mapping(
            node_value, location=f"imputation/producer_graph/nodes/{node_index}"
        )
        if node.get("kind") != "acs_earnings_universe":
            continue
        for resource_index, resource_value in enumerate(
            _array(
                node.get("virtual_resources"),
                location=(
                    f"imputation/producer_graph/nodes/{node_index}/virtual_resources"
                ),
            )
        ):
            resource = _mapping(
                resource_value,
                location=(
                    "imputation/producer_graph/nodes/"
                    f"{node_index}/virtual_resources/{resource_index}"
                ),
            )
            binding = _mapping(
                resource.get("binding"),
                location=(
                    "imputation/producer_graph/nodes/"
                    f"{node_index}/virtual_resources/{resource_index}/binding"
                ),
            )
            if (
                binding.get("resource_kind")
                == "acs_pums_earnings_universe_execution_config"
            ):
                matches.append(binding)
    if len(matches) != 1:
        raise SpecValidationError(
            "imputation producer graph must own one ACS earnings contract"
        )
    identity = deepcopy(
        dict(
            _mapping(
                matches[0].get("contract_identity"),
                location="imputation/ACS earnings contract_identity",
            )
        )
    )
    if "sha256" in identity:
        raise SpecValidationError(
            "imputation/ACS earnings contract_identity: sha256 must be derived"
        )
    identity["sha256"] = _generation_zero_sha256(identity)
    return identity


def _take_up_identity(
    take_up: Mapping[str, object],
    *,
    sources: Mapping[str, object],
    engine_lock: Mapping[str, object],
) -> dict[str, object]:
    legacy = project_legacy_take_up_contract(
        take_up,
        sources_document=sources,
        engine_abi_lock=engine_lock,
    )
    asserted = _mapping(
        legacy.get("asserted_engine"), location="projected take_up/asserted_engine"
    )
    programs = _array(
        legacy.get("programs"), location="projected take_up/programs"
    )
    return {
        "version": legacy["version"],
        "country": legacy["country"],
        "resource_sha256": sha256_json(legacy),
        "asserted_constraint": asserted.get("constraint", ""),
        "inventory_built_against": asserted.get("inventory_built_against", ""),
        "programs": deepcopy(list(programs)),
    }


def _remaining_stage_receipt(engine_lock: Mapping[str, object]) -> dict[str, object]:
    engine = _mapping(engine_lock.get("engine"), location="engine_abi.lock/engine")
    version = _nonempty_string(
        engine.get("version"), location="engine_abi.lock/engine/version"
    )
    manifest = _mapping(
        engine_lock.get("remaining_stage_input_manifest"),
        location="engine_abi.lock/remaining_stage_input_manifest",
    )
    receipt = deepcopy(
        dict(
            _mapping(
                manifest.get("receipt"),
                location="engine_abi.lock/remaining_stage_input_manifest/receipt",
            )
        )
    )
    for contract_name in _REMAINING_STAGE_VERSIONED_CONTRACTS:
        contract = deepcopy(
            dict(
                _mapping(
                    receipt.get(contract_name),
                    location=(
                        "engine_abi.lock/remaining_stage_input_manifest/receipt/"
                        f"{contract_name}"
                    ),
                )
            )
        )
        if contract.pop("engine_version_ref", None) != _ENGINE_VERSION_REF:
            raise SpecValidationError(
                "engine_abi.lock remaining-stage contract has a stale version ref: "
                f"{contract_name}"
            )
        contract["engine_version"] = version
        receipt[contract_name] = contract
    return receipt


def _rung_token(publication: Mapping[str, object], fraction: float) -> str:
    release = _mapping(publication.get("release"), location="publication/release")
    matches: list[str] = []
    for index, value in enumerate(
        _array(
            release.get("rung_fractions"),
            location="publication/release/rung_fractions",
        )
    ):
        row = _mapping(
            value, location=f"publication/release/rung_fractions/{index}"
        )
        if _finite_float(
            row.get("fraction"),
            location=f"publication/release/rung_fractions/{index}/fraction",
        ) == fraction:
            matches.append(
                _nonempty_string(
                    row.get("token"),
                    location=f"publication/release/rung_fractions/{index}/token",
                )
            )
    if len(matches) != 1:
        raise SpecValidationError(
            "sample_fraction must name exactly one declared publication rung"
        )
    return matches[0]


def _input_pins_payload(
    input_pins: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for role in sorted(input_pins):
        if not isinstance(role, str) or not role:
            raise SpecValidationError("input_pins: roles must be non-empty strings")
        pin = _mapping(input_pins[role], location=f"input_pins/{role}")
        sha256 = _nonempty_string(
            pin.get("sha256"), location=f"input_pins/{role}/sha256"
        )
        if len(sha256) != 64 or any(
            character not in "0123456789abcdef" for character in sha256
        ):
            raise SpecValidationError(
                f"input_pins/{role}/sha256: lowercase SHA-256 required"
            )
        size_bytes = _integer(
            pin.get("size_bytes"),
            location=f"input_pins/{role}/size_bytes",
            minimum=0,
        )
        result[role] = {"sha256": sha256, "size_bytes": size_bytes}
    return result


def _resolved_late_resource_semantics(
    projected: Mapping[str, object],
    *,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
) -> dict[str, object]:
    """Resolve the two run-bound fields in the normalized resource receipt."""

    result = deepcopy(dict(projected))
    producers = _array(
        result.get("producers"), location="late resource semantics/producers"
    )
    matches = [
        _mapping(value, location="late resource semantics producer")
        for value in producers
        if _mapping(
            value, location="late resource semantics producer"
        ).get("producer")
        == "primary_puf_qrf"
    ]
    if len(matches) != 1:
        raise SpecValidationError(
            "late resource semantics must contain one primary_puf_qrf producer"
        )
    resources = _mapping(
        matches[0].get("resources"),
        location="late resource semantics/primary_puf_qrf/resources",
    )
    resource = _mapping(
        resources.get("tax_unit.@primary_puf_execution_config"),
        location=(
            "late resource semantics/primary_puf_qrf/resources/"
            "tax_unit.@primary_puf_execution_config"
        ),
    )
    binding = _mapping(
        resource.get("binding"),
        location="late resource semantics/primary PUF binding",
    )
    attachment = _mapping(
        binding.get("clone_attachment"),
        location="late resource semantics/primary PUF clone_attachment",
    )
    attachment["fraction"] = clone_attachment_fraction  # type: ignore[index]
    attachment["seed"] = clone_attachment_seed  # type: ignore[index]
    result.pop("sha256", None)
    result["sha256"] = _generation_zero_sha256(result)
    return result


def _model_parameters(imputation: Mapping[str, object]) -> tuple[int, int]:
    models = _mapping(imputation.get("models"), location="imputation/models")
    model = _mapping(
        models.get("regime_gated_qrf"),
        location="imputation/models/regime_gated_qrf",
    )
    params = _mapping(
        model.get("params"), location="imputation/models/regime_gated_qrf/params"
    )
    n_estimators = _integer(
        params.get("n_estimators"),
        location="imputation/models/regime_gated_qrf/params/n_estimators",
        minimum=1,
    )
    target_limits = {
        _integer(
            family.get("max_targets_per_fit"),
            location="imputation/families/max_targets_per_fit",
            minimum=1,
        )
        for value in _array(
            imputation.get("families"), location="imputation/families"
        )
        for family in [_mapping(value, location="imputation/families row")]
        if family.get("max_targets_per_fit") is not None
    }
    if len(target_limits) != 1:
        raise SpecValidationError(
            "imputation families must share one non-null max_targets_per_fit"
        )
    return n_estimators, target_limits.pop()


def project_stacked_checkpoint_base_identity(
    spec: ResolvedSpec,
    *,
    input_pins: Mapping[str, Mapping[str, object]],
    stack_receipt: Mapping[str, object],
    sample_fraction: float,
    sample_seed: int,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
) -> dict[str, object]:
    """Project ``_stacked_checkpoint_base_identity`` without runtime imports."""

    domains = _us_domains(spec)
    engine_lock = _generated_engine_lock(spec)
    fraction = _finite_float(sample_fraction, location="sample_fraction")
    attachment_fraction = _finite_float(
        clone_attachment_fraction, location="clone_attachment_fraction"
    )
    sample_seed_value = _integer(sample_seed, location="sample_seed", minimum=0)
    attachment_seed_value = _integer(
        clone_attachment_seed, location="clone_attachment_seed", minimum=0
    )
    stack = deepcopy(dict(_mapping(stack_receipt, location="stack_receipt")))
    if stack.get("sample_fraction") != fraction:
        raise SpecValidationError(
            "stack_receipt/sample_fraction: differs from the run request"
        )
    if stack.get("sample_seed") != sample_seed_value:
        raise SpecValidationError(
            "stack_receipt/sample_seed: differs from the run request"
        )

    pipeline = _mapping(
        domains["spine"].get("pipeline_contract"),
        location="spine/pipeline_contract",
    )
    artifact_protocol = deepcopy(
        dict(
            _mapping(
                pipeline.get("artifact_protocol"),
                location="spine/pipeline_contract/artifact_protocol",
            )
        )
    )
    bundle_run = _mapping(
        domains["bundle"].get("dataset_run"), location="bundle/dataset_run"
    )
    period = _integer(
        bundle_run.get("target_period"),
        location="bundle/dataset_run/target_period",
    )
    model_seed = _build_model_seed(spec)
    imputation_payloads = _imputation_projection(domains)
    primary_qrf = _mapping(
        imputation_payloads.get("primary_qrf"),
        location="imputation projection/primary_qrf",
    )
    n_estimators, max_targets_per_fit = _model_parameters(domains["imputation"])
    tail_support, tail_schema_version = _tail_support_contract(domains["spine"])
    engine = _mapping(engine_lock.get("engine"), location="engine_abi.lock/engine")
    simulation_batch = _mapping(
        pipeline.get("simulation_household_batch_size"),
        location="spine/pipeline_contract/simulation_household_batch_size",
    )
    resource_semantics = _resolved_late_resource_semantics(
        _mapping(
            imputation_payloads.get("late_producer_resource_semantics"),
            location="imputation projection/late_producer_resource_semantics",
        ),
        clone_attachment_fraction=attachment_fraction,
        clone_attachment_seed=attachment_seed_value,
    )

    return {
        **artifact_protocol,
        "period": period,
        "model_seed": model_seed,
        "policyengine_us_version": _nonempty_string(
            engine.get("version"), location="engine_abi.lock/engine/version"
        ),
        "inputs": _input_pins_payload(input_pins),
        "sampling": {
            "sample_fraction": fraction,
            "fraction_token": _rung_token(domains["publication"], fraction),
            "sample_seed": sample_seed_value,
            "stack_manifest_sha256": sha256_json(stack),
            "stack_manifest": stack,
        },
        "clone_attachment": {
            "fraction": attachment_fraction,
            "seed": attachment_seed_value,
        },
        "stacked_authority": project_stacked_authority_receipt(spec),
        "pool_code": {
            "operator_order": deepcopy(
                list(
                    _array(
                        pipeline.get("stacked_operator_order"),
                        location="spine/pipeline_contract/stacked_operator_order",
                    )
                )
            ),
            "pre_clone_source_operator_order": deepcopy(
                list(
                    _array(
                        pipeline.get("pre_clone_source_operator_order"),
                        location=(
                            "spine/pipeline_contract/"
                            "pre_clone_source_operator_order"
                        ),
                    )
                )
            ),
            "gap_fill_producer_schedule": deepcopy(
                imputation_payloads["gap_fill_producer_schedule_receipt"]
            ),
            "post_clone_source_operator_order": deepcopy(
                list(
                    _array(
                        pipeline.get("post_clone_source_operator_order"),
                        location=(
                            "spine/pipeline_contract/"
                            "post_clone_source_operator_order"
                        ),
                    )
                )
            ),
            "late_producer_schedule": deepcopy(
                imputation_payloads["late_producer_schedule_receipt"]
            ),
            "late_producer_resource_semantics": resource_semantics,
            "derive_operator_order": deepcopy(
                list(
                    _array(
                        pipeline.get("derive_operator_order"),
                        location="spine/pipeline_contract/derive_operator_order",
                    )
                )
            ),
            "remaining_stage_input_manifest": _remaining_stage_receipt(engine_lock),
            "primary_qrf_target_order": deepcopy(
                list(
                    _array(
                        primary_qrf.get("target_order"),
                        location="imputation projection/primary_qrf/target_order",
                    )
                )
            ),
            "primary_qrf_checkpoint_schema_version": _integer(
                primary_qrf.get("checkpoint_schema_version"),
                location=(
                    "imputation projection/primary_qrf/checkpoint_schema_version"
                ),
                minimum=1,
            ),
            "acs_pums_earnings_universe_contract": _acs_earnings_contract(
                domains["imputation"]
            ),
            "us_qbi_reconciliation_contract": deepcopy(
                dict(
                    _mapping(
                        pipeline.get("qbi_reconciliation"),
                        location="spine/pipeline_contract/qbi_reconciliation",
                    )
                )
            ),
            "take_up_contract": _take_up_identity(
                domains["take_up"],
                sources=domains["sources"],
                engine_lock=engine_lock,
            ),
            "puf_capital_gains_tail_manifest_schema_version": tail_schema_version,
            "puf_capital_gains_tail_support_contract": tail_support,
            "primary_qrf_n_estimators": n_estimators,
            "acs_transfer_n_estimators": n_estimators,
            "acs_transfer_max_targets_per_fit": max_targets_per_fit,
            "simulation_household_batch_size": _integer(
                simulation_batch.get("value"),
                location=(
                    "spine/pipeline_contract/"
                    "simulation_household_batch_size/value"
                ),
                minimum=1,
            ),
        },
    }


def project_stacked_checkpoint_static_components(
    spec: ResolvedSpec,
) -> dict[str, object]:
    """Return the exact non-run subset of the stacked checkpoint identity.

    The constants adapter uses this projection during compilation, before an
    input stack or run request exists.  Dynamic input pins, sampling evidence,
    and clone-attachment request fields remain outside it.  Resource semantics
    are resolved with the bundle-declared attachment defaults, matching the
    generation-0 identity for its default run configuration.
    """

    domains = _us_domains(spec)
    spine = domains["spine"]
    sampling = _mapping(spine.get("sampling"), location="spine/sampling")
    fraction_contract = _mapping(
        sampling.get("fraction"), location="spine/sampling/fraction"
    )
    seed_contract = _mapping(sampling.get("seed"), location="spine/sampling/seed")
    sample_fraction = _finite_float(
        fraction_contract.get("default"),
        location="spine/sampling/fraction/default",
    )
    sample_seed = _integer(
        seed_contract.get("default"),
        location="spine/sampling/seed/default",
        minimum=0,
    )
    support_roles = [
        role
        for index, value in enumerate(
            _array(spine.get("support_roles"), location="spine/support_roles")
        )
        for role in [_mapping(value, location=f"spine/support_roles/{index}")]
        if role.get("id") == "puf_tax_detail"
    ]
    if len(support_roles) != 1:
        raise SpecValidationError(
            "spine/support_roles: exactly one puf_tax_detail role required"
        )
    attachment = _mapping(
        support_roles[0].get("attachment"),
        location="spine/support_roles/puf_tax_detail/attachment",
    )
    attachment_fraction = _finite_float(
        _mapping(
            attachment.get("fraction"),
            location="spine/support_roles/puf_tax_detail/attachment/fraction",
        ).get("default"),
        location=(
            "spine/support_roles/puf_tax_detail/attachment/fraction/default"
        ),
    )
    attachment_seed = _integer(
        _mapping(
            attachment.get("seed"),
            location="spine/support_roles/puf_tax_detail/attachment/seed",
        ).get("default"),
        location="spine/support_roles/puf_tax_detail/attachment/seed/default",
        minimum=0,
    )
    full = project_stacked_checkpoint_base_identity(
        spec,
        input_pins={},
        stack_receipt={
            "sample_fraction": sample_fraction,
            "sample_seed": sample_seed,
        },
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        clone_attachment_fraction=attachment_fraction,
        clone_attachment_seed=attachment_seed,
    )
    pipeline = _mapping(
        spine.get("pipeline_contract"), location="spine/pipeline_contract"
    )
    artifact_protocol = _mapping(
        pipeline.get("artifact_protocol"),
        location="spine/pipeline_contract/artifact_protocol",
    )
    return {
        **{key: deepcopy(full[key]) for key in artifact_protocol},
        "period": full["period"],
        "model_seed": full["model_seed"],
        "policyengine_us_version": full["policyengine_us_version"],
        "stacked_authority": deepcopy(full["stacked_authority"]),
        "pool_code": deepcopy(full["pool_code"]),
    }


def stacked_identity_bytes(identity: Mapping[str, object]) -> bytes:
    """Return the exact canonical bytes used by generation-0 cache hashing."""

    return canonical_json_bytes(identity)


__all__ = [
    "project_stacked_authority_receipt",
    "project_stacked_checkpoint_base_identity",
    "project_stacked_checkpoint_static_components",
    "stacked_identity_bytes",
]
