"""Resolve declarative reference profiles before activation or compilation.

This layer changes selectors, never Ledger observations. Selection dimensions
are supplied by the consumer; subsequent transformation kinds can share the
same effective-document and preparation identities.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from microcosm.build.country_spec import (
    _carried_value_keys,
    _validate_target_references,
)
from microcosm.build.ledger_targets import ALLOCATED_RESIDUAL_BY_RATIO
from microcosm.graph.canonical import canonical_json

REFERENCE_PROFILE_RESOLVER_VERSION = "reference-profiles-v1"
FACTOR_REFERENCE_PROFILE_RESOLVER_VERSION = "reference-profiles-v2"
ALLOWED_REFERENCE_PROFILE_RESOLVER_VERSIONS = frozenset(
    {REFERENCE_PROFILE_RESOLVER_VERSION, FACTOR_REFERENCE_PROFILE_RESOLVER_VERSION}
)
SCALE_MONEY_AND_INTERVALS = "scale_money_and_intervals"
REQUIRES_REFERENCE_PROFILE_RESOLUTION = "requires_reference_profile_resolution"
REQUIRES_PREREQUISITES = "requires_prerequisites"
REFERENCE_DOCUMENT_KEYS = frozenset(
    {"country", "schema_version", "hierarchy", "target_references"}
)


def reference_document_sha256(document: object) -> str:
    """Hash the actual canonical document, rather than its resource filename."""
    return hashlib.sha256(canonical_json(document)).hexdigest()


def _copy(document: object):
    return json.loads(canonical_json(document))


def _mapping(value: object, context: str) -> dict:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object.")
    return dict(value)


def _sequence(value: object, context: str) -> list:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{context} must be a list.")
    return list(value)


def _selection_identity(receipt: Mapping) -> dict:
    identity = {
        key: receipt[key]
        for key in (
            "resolver_version",
            "base_references_sha256",
            "base_intervals_sha256",
            "selected_declarations_sha256",
            "selection",
        )
    }
    if receipt.get("resolver_version") == FACTOR_REFERENCE_PROFILE_RESOLVER_VERSION:
        identity["selected_factor_identities"] = receipt["selected_factor_identities"]
    return identity


def validate_reference_factor_evidence(receipt: Mapping) -> dict:
    """Bind selected declarations and receipts without claiming feed authenticity."""
    from microcosm.build.reference_factors import validate_factor_context

    if receipt.get("resolver_version") == REFERENCE_PROFILE_RESOLVER_VERSION:
        if "selected_factors" in receipt or "selected_factor_identities" in receipt:
            raise ValueError("Legacy resolution cannot carry factor evidence.")
        return {}
    factors = _mapping(receipt.get("selected_factors"), "selected factors")
    if not factors:
        raise ValueError("Factor resolution needs selected factor evidence.")
    validated = validate_factor_context(
        {"schema_version": 1, "factors": list(factors.values())}, list(factors)
    )
    if validated != factors:
        raise ValueError("Selected factor identities do not match their declarations.")
    identities = {
        factor_id: {
            "declaration_sha256": reference_document_sha256(entry["declaration"]),
            "receipt_sha256": entry["receipt"]["receipt_sha256"],
        }
        for factor_id, entry in sorted(factors.items())
    }
    if identities != receipt.get("selected_factor_identities"):
        raise ValueError("Selected factor evidence identity mismatch.")
    return validated


def _profile_metadata(transform: Mapping) -> dict:
    metadata = _mapping(transform.get("metadata", {}), "profile metadata")
    if any(
        not isinstance(key, str) or not isinstance(value, str) or not value
        for key, value in metadata.items()
    ):
        raise ValueError("Profile metadata requires nonempty strings.")
    if set(metadata) & {
        "activation_status",
        "activation_reason",
        "parking_reason",
        "required_reference_profile",
        "required_measurement_binding",
        "reference_value_kind",
        "reference_prerequisites",
        "reference_prerequisites_remaining",
        "reference_selection_sha256",
        "effective_intervals_sha256",
        "reference_profiles_resolved",
    }:
        raise ValueError("A profile cannot override resolution identities.")
    return metadata


def _explicit_names(transform: Mapping, field: str) -> list[str]:
    names = _sequence(transform.get(field), field)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("A transformation needs unique explicit reference names.")
    return names


def _companion_key(row: Mapping, *, scope: bool = False) -> bytes:
    selector = _mapping(row.get("ledger_selector"), "companion selector")
    excluded = {"aggregate_fact_key", "source_record_id"}
    if not scope:
        excluded.update({"source_measure_id", "source_concept"})
    else:
        excluded.add("dimension_values")
    payload = {
        "selector": {
            key: value for key, value in selector.items() if key not in excluded
        },
        "entity": row.get("entity"),
        "period": row.get("period"),
        "reference_set": row.get("metadata", {}).get("reference_set"),
    }
    if not scope:
        payload["filter"] = row.get("filter")
    return canonical_json(payload)


def _validate_companions(transform: Mapping, by_name: Mapping) -> tuple[list, list]:
    money = _explicit_names(transform, "reference_names")
    counts = _explicit_names(transform, "count_reference_names")
    if set(money) & set(counts) or len(money) != len(counts):
        raise ValueError("Money and count rosters need disjoint complete companions.")
    for kind, names in (("money", money), ("count", counts)):
        if set(names) - set(by_name):
            raise ValueError("A profile names missing companion references.")
        if any(
            by_name[name].get("metadata", {}).get("reference_value_kind") != kind
            for name in names
        ):
            raise ValueError(
                "Money and count rosters require matching declared value kinds."
            )
        scopes = {_companion_key(by_name[name], scope=True) for name in names}
        complete = {
            name
            for name, row in by_name.items()
            if isinstance(row.get("ledger_selector"), Mapping)
            and _companion_key(row, scope=True) in scopes
        }
        if complete != set(names):
            raise ValueError("Money and count rosters are incomplete companions.")
    money_keys = [_companion_key(by_name[name]) for name in money]
    count_keys = [_companion_key(by_name[name]) for name in counts]
    if len(set(money_keys)) != len(money) or set(money_keys) != set(count_keys):
        raise ValueError("Money and count rosters must match one-to-one companions.")
    if {_companion_key(by_name[name], scope=True) for name in money} & {
        _companion_key(by_name[name], scope=True) for name in counts
    }:
        raise ValueError("Money and count companions must select distinct measures.")
    return money, counts


def _satisfy_factor_prerequisite(row: dict, dimension: str, binding: str) -> list:
    metadata = row.setdefault("metadata", {})
    if metadata.get("activation_status") != REQUIRES_PREREQUISITES:
        raise ValueError(
            "Money scaling requires the explicit prerequisite parking status."
        )
    if (
        metadata.get("required_reference_profile") != dimension
        or metadata.get("required_measurement_binding") != binding
    ):
        raise ValueError("Money scaling has incompatible reference prerequisites.")
    try:
        prerequisites = json.loads(metadata.get("reference_prerequisites", ""))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Reference prerequisites must be an explicit JSON list."
        ) from error
    required = {f"reference_profile:{dimension}", f"measurement:{binding}"}
    if (
        not isinstance(prerequisites, list)
        or any(not isinstance(item, str) or not item for item in prerequisites)
        or len(set(prerequisites)) != len(prerequisites)
        or set(prerequisites) != required
    ):
        raise ValueError(
            "Reference prerequisites must name the profile and measurement."
        )
    remaining = [
        item for item in prerequisites if item != f"reference_profile:{dimension}"
    ]
    metadata["reference_prerequisites_remaining"] = canonical_json(remaining).decode()
    return remaining


def resolve_reference_profiles(
    base_reference_document: Mapping,
    profile_declarations: Mapping,
    selection: Mapping,
    interval_document: Mapping | None = None,
    *,
    factor_context: Mapping | None = None,
) -> tuple[dict, dict | None, dict]:
    """Return effective references, effective intervals and a bound receipt.

    Only selected declarations enter selection identity. An unavailable profile
    refuses; an explicitly deferred dimension is reported without claiming its
    transformations ran. Money scaling consumes declared factor evidence;
    compilation authenticates it against the feed. Interval scaling is reserved
    for the subsequent interval implementation.
    Resource headers are bound into the base identity; effective references use
    the compiler's closed document shape.
    """
    base = _copy(_mapping(base_reference_document, "base references"))
    declarations = _copy(_mapping(profile_declarations, "profile declarations"))
    if _carried_value_keys(declarations):
        raise ValueError("Reference profile declarations carry observed values.")
    selected = _mapping(selection, "profile selection")
    if (
        type(declarations.get("schema_version")) is not int
        or declarations["schema_version"] != 1
    ):
        raise ValueError("Profile declarations require schema_version 1.")
    dimensions = _mapping(declarations.get("dimensions"), "profile dimensions")
    if set(selected) - set(dimensions):
        raise ValueError("Unknown profile selection dimension.")
    rows = _sequence(base.get("target_references"), "target references")
    by_name = {}
    for row in rows:
        row = _mapping(row, "reference")
        name = row.get("name")
        if not isinstance(name, str) or not name or name in by_name:
            raise ValueError("Reference names must be nonempty and unique.")
        metadata = _mapping(row.get("metadata", {}), "reference metadata")
        if metadata.get("reference_profiles_resolved"):
            raise ValueError("Reference profiles have already been resolved.")
        by_name[name] = row
    intervals = None if interval_document is None else _copy(interval_document)
    chosen, resolved_selection, deferred = {}, {}, []
    for dimension, raw_dimension in sorted(dimensions.items()):
        dimension_row = _mapping(raw_dimension, "profile dimension")
        profiles = _mapping(dimension_row.get("profiles"), "named profiles")
        name = selected.get(dimension, dimension_row.get("default"))
        if not isinstance(name, str) or name not in profiles:
            raise ValueError(f"Unknown profile {name!r} for {dimension!r}.")
        profile = _mapping(profiles[name], "selected profile")
        if type(profile.get("deferred", False)) is not bool:
            raise ValueError("A profile's deferred status must be boolean.")
        if profile.get("available", True) is not True:
            raise ValueError(f"Profile {dimension!r}/{name!r} is unavailable.")
        transformations = _sequence(
            profile.get("transformations", []), "profile transformations"
        )
        if profile.get("deferred", False):
            if transformations:
                raise ValueError("A deferred profile cannot apply transformations.")
            deferred.append(dimension)
        chosen[dimension] = {"name": name, "declaration": profile}
        resolved_selection[dimension] = name
    factor_ids = set()
    for choice in chosen.values():
        for raw_transform in choice["declaration"].get("transformations", []):
            transform = _mapping(raw_transform, "reference transformation")
            if transform.get("operation") == SCALE_MONEY_AND_INTERVALS:
                if _sequence(transform.get("interval_table_ids"), "interval table ids"):
                    raise ValueError("interval scaling lands in r2b")
                factor_id = transform.get("factor_id")
                if not isinstance(factor_id, str) or not factor_id:
                    raise ValueError("Money scaling needs a declared factor id.")
                factor_ids.add(factor_id)
    selected_factors = {}
    if factor_ids:
        from microcosm.build.reference_factors import validate_factor_context

        selected_factors = validate_factor_context(factor_context, sorted(factor_ids))
    receipt = {
        "resolver_version": REFERENCE_PROFILE_RESOLVER_VERSION,
        "base_references_sha256": reference_document_sha256(base),
        "base_intervals_sha256": (
            None if intervals is None else reference_document_sha256(intervals)
        ),
        "selected_declarations_sha256": reference_document_sha256(chosen),
        "selection": resolved_selection,
    }
    if selected_factors:
        receipt.update(
            resolver_version=FACTOR_REFERENCE_PROFILE_RESOLVER_VERSION,
            selected_declarations=_copy(chosen),
            base_reference_document=_copy(base),
            selected_factors=_copy(selected_factors),
            selected_factor_identities={
                factor_id: {
                    "declaration_sha256": reference_document_sha256(
                        entry["declaration"]
                    ),
                    "receipt_sha256": entry["receipt"]["receipt_sha256"],
                }
                for factor_id, entry in sorted(selected_factors.items())
            },
        )
    selection_hash = reference_document_sha256(_selection_identity(receipt))
    interval_hash = None if intervals is None else reference_document_sha256(intervals)
    transformed, policies = {}, []
    for dimension, choice in chosen.items():
        for raw_transform in choice["declaration"].get("transformations", []):
            transform = _mapping(raw_transform, "reference transformation")
            operation = transform.get("operation")
            allowed = {
                "reference_names",
                "operation",
                "value_operands",
                "metadata",
                "notes",
                "target_label",
            }
            if operation == SCALE_MONEY_AND_INTERVALS:
                allowed = {
                    "reference_names",
                    "count_reference_names",
                    "operation",
                    "factor_id",
                    "interval_table_ids",
                    "required_measurement_binding",
                    "metadata",
                }
            if set(transform) - allowed:
                raise ValueError("Unknown reference transformation fields.")
            if operation not in {
                ALLOCATED_RESIDUAL_BY_RATIO,
                SCALE_MONEY_AND_INTERVALS,
            }:
                raise ValueError("Unsupported reference profile transformation.")
            metadata = _profile_metadata(transform)
            if _carried_value_keys(transform):
                raise ValueError("Reference transformations carry observed values.")
            if operation == SCALE_MONEY_AND_INTERVALS:
                tables = _sequence(
                    transform.get("interval_table_ids"), "interval table ids"
                )
                if tables:
                    raise ValueError("interval scaling lands in r2b")
                binding = transform.get("required_measurement_binding")
                if not isinstance(binding, str) or not binding:
                    raise ValueError(
                        "Money scaling needs a required measurement binding."
                    )
                money, counts = _validate_companions(transform, by_name)
                factor_id = transform["factor_id"]
                factor = selected_factors[factor_id]
                declaration = factor["declaration"]
                if declaration["purpose"] != "profile_input":
                    raise ValueError("A diagnostic factor cannot be a profile input.")
                rational = factor["operational_factor"]
                weight = rational["numerator"] / rational["denominator"]
                for name in [*money, *counts]:
                    if name in transformed:
                        raise ValueError(
                            f"Duplicate or incompatible transformation: {name!r}."
                        )
                    row = by_name[name]
                    if (
                        row.get("ledger_selector", {}).get("period_value")
                        != declaration["source_period"]
                        or row.get("period") != declaration["target_period"]
                    ):
                        raise ValueError(
                            "Factor and reference source/target periods are incompatible."
                        )
                    remaining = _satisfy_factor_prerequisite(row, dimension, binding)
                    if name in money:
                        if row.get(
                            "value_operation", "identity"
                        ) != "identity" or row.get("value_operands", []):
                            raise ValueError(
                                "Money scaling requires an untransformed source reference."
                            )
                        selector = _mapping(
                            row.get("ledger_selector"), "money selector"
                        )
                        key = selector.get("aggregate_fact_key")
                        if not isinstance(key, str) or not key:
                            raise ValueError(
                                "Money scaling needs a singleton source fact key."
                            )
                        row["value_operation"] = "linear_combination"
                        row["value_operands"] = [
                            {
                                "aggregate_fact_key": key,
                                "expected_member_count": 1,
                                "weight": weight,
                            }
                        ]
                    row_metadata = row.setdefault("metadata", {})
                    row_metadata.update(metadata)
                    row_metadata["reference_selection_sha256"] = selection_hash
                    row_metadata["reference_profiles_resolved"] = "1"
                    if interval_hash is not None:
                        row_metadata["effective_intervals_sha256"] = interval_hash
                    transformed[name] = {
                        "dimension": dimension,
                        "profile": choice["name"],
                        "operation": "linear_combination"
                        if name in money
                        else "identity",
                        "factor_id": factor_id,
                        "operational_factor": _copy(rational),
                        "metadata": metadata,
                        "operand_selectors": _copy(row.get("value_operands", [])),
                        "source_period": row.get("ledger_selector", {}).get(
                            "period_value"
                        ),
                        "target_period": row.get("period"),
                        "unresolved_prerequisites": remaining,
                    }
                policies.append(
                    {"dimension": dimension, "profile": choice["name"], **metadata}
                )
                continue
            names = _sequence(transform.get("reference_names"), "transformed names")
            if not names or any(not isinstance(name, str) for name in names):
                raise ValueError("A transformation needs explicit reference names.")
            operands = _sequence(transform.get("value_operands"), "value operands")
            for name in names:
                if name not in by_name:
                    raise ValueError(f"Profile names missing reference {name!r}.")
                if name in transformed:
                    raise ValueError(
                        f"Duplicate or incompatible transformation: {name!r}."
                    )
                row = by_name[name]
                row["value_operation"] = transform["operation"]
                row["value_operands"] = _copy(operands)
                row_metadata = row.setdefault("metadata", {})
                prerequisite = row_metadata.get("required_reference_profile")
                if prerequisite and prerequisite != dimension:
                    raise ValueError(
                        "Transformation does not satisfy the reference profile prerequisite."
                    )
                row_metadata.update(metadata)
                row_metadata["reference_selection_sha256"] = selection_hash
                row_metadata["reference_profiles_resolved"] = "1"
                if interval_hash is not None:
                    row_metadata["effective_intervals_sha256"] = interval_hash
                if (
                    row_metadata.get("activation_status")
                    == REQUIRES_REFERENCE_PROFILE_RESOLUTION
                ):
                    row_metadata["activation_status"] = "active"
                    row_metadata.pop("activation_reason", None)
                    row_metadata.pop("parking_reason", None)
                if "notes" in transform:
                    if not isinstance(transform["notes"], str):
                        raise ValueError("Transformation notes must be text.")
                    row["notes"] = transform["notes"]
                if "target_label" in transform:
                    hierarchy = _mapping(row.get("hierarchy"), "reference hierarchy")
                    hierarchy["target_label"] = transform["target_label"]
                    row["hierarchy"] = hierarchy
                transformed[name] = {
                    "dimension": dimension,
                    "profile": choice["name"],
                    "operation": transform["operation"],
                    "metadata": metadata,
                    "operand_selectors": _copy(operands),
                    "source_period": row.get("ledger_selector", {}).get("period_value"),
                    "target_period": row.get("period"),
                }
            policies.append(
                {"dimension": dimension, "profile": choice["name"], **metadata}
            )
    base["target_references"] = [by_name[row["name"]] for row in rows]
    effective = {
        key: value for key, value in base.items() if key in REFERENCE_DOCUMENT_KEYS
    }
    _validate_target_references(effective, country=effective.get("country"))
    receipt.update(
        reference_selection_sha256=selection_hash,
        effective_references_sha256=reference_document_sha256(effective),
        effective_intervals_sha256=interval_hash,
        transformed_names=sorted(transformed),
        transformations={name: transformed[name] for name in sorted(transformed)},
        policies=policies,
        deferred_dimensions=deferred,
    )
    receipt["receipt_sha256"] = reference_document_sha256(receipt)
    return effective, intervals, receipt


def _factor_reference_bindings(resolution: Mapping) -> tuple[dict, set]:
    """Recover each factor row's role and source behavior from bound documents."""
    if resolution["resolver_version"] == REFERENCE_PROFILE_RESOLVER_VERSION:
        return {}, set()
    chosen = _mapping(resolution.get("selected_declarations"), "selected declarations")
    base = _mapping(resolution.get("base_reference_document"), "base references")
    if reference_document_sha256(chosen) != resolution.get(
        "selected_declarations_sha256"
    ) or reference_document_sha256(base) != resolution.get("base_references_sha256"):
        raise ValueError("Factor companion source document identity mismatch.")
    if {
        dimension: choice.get("name") for dimension, choice in chosen.items()
    } != resolution.get("selection"):
        raise ValueError("Selected profile declaration identity mismatch.")
    source_rows = _sequence(base.get("target_references"), "base target references")
    by_name = {row["name"]: row for row in source_rows}
    if len(by_name) != len(source_rows):
        raise ValueError("Base reference names must be unique.")
    bindings, other_names = {}, set()
    for dimension, choice in chosen.items():
        for transform in choice["declaration"].get("transformations", []):
            if transform.get("operation") != SCALE_MONEY_AND_INTERVALS:
                other_names.update(_explicit_names(transform, "reference_names"))
                continue
            money, counts = _validate_companions(transform, by_name)
            for role, names in (("money", money), ("count", counts)):
                for name in names:
                    if name in bindings:
                        raise ValueError("Duplicate factor companion roster entry.")
                    bindings[name] = {
                        "dimension": dimension,
                        "profile": choice["name"],
                        "factor_id": transform["factor_id"],
                        "role": role,
                        "source": by_name[name],
                    }
    return bindings, other_names


def validate_reference_resolution(
    document: Mapping,
    receipt: Mapping,
    *,
    references_sha256: str | None = None,
    interval_document: Mapping | None = None,
) -> dict:
    """Refuse stale effective documents, receipts or interval companions."""
    resolution = _copy(_mapping(receipt, "reference resolution receipt"))
    if (
        resolution.get("resolver_version")
        not in ALLOWED_REFERENCE_PROFILE_RESOLVER_VERSIONS
    ):
        raise ValueError("Unknown reference resolver version.")
    claimed_receipt_hash = resolution.pop("receipt_sha256", None)
    if claimed_receipt_hash != reference_document_sha256(resolution):
        raise ValueError("Reference resolution receipt hash mismatch.")
    resolution["receipt_sha256"] = claimed_receipt_hash
    factors = validate_reference_factor_evidence(resolution)
    actual = reference_document_sha256(document)
    if actual != resolution.get("effective_references_sha256") or (
        references_sha256 is not None and actual != references_sha256
    ):
        raise ValueError("Effective reference document hash mismatch.")
    try:
        selection_hash = reference_document_sha256(_selection_identity(resolution))
    except KeyError as error:
        raise ValueError("Reference resolution identity is incomplete.") from error
    if selection_hash != resolution.get("reference_selection_sha256"):
        raise ValueError("Reference selection identity mismatch.")
    if interval_document is not None and reference_document_sha256(
        interval_document
    ) != resolution.get("effective_intervals_sha256"):
        raise ValueError("Effective interval document hash mismatch.")
    factor_bindings, other_names = _factor_reference_bindings(resolution)
    names, used_factors = [], set()
    for row in document.get("target_references", ()):
        metadata = row.get("metadata", {})
        if metadata.get("reference_profiles_resolved"):
            names.append(row["name"])
            if metadata.get("reference_selection_sha256") != selection_hash or (
                metadata.get("effective_intervals_sha256")
                != resolution.get("effective_intervals_sha256")
            ):
                raise ValueError("Effective reference profile identity mismatch.")
            action = resolution.get("transformations", {}).get(row["name"], {})
            binding = factor_bindings.get(row["name"])
            if factors and binding is None and row["name"] not in other_names:
                raise ValueError(
                    "Factor action names a row outside its companion roster."
                )
            if binding is not None or "factor_id" in action:
                if binding is None:
                    raise ValueError(
                        "Factor action names a row outside its companion roster."
                    )
                role = binding["role"]
                operation = "linear_combination" if role == "money" else "identity"
                if action.get("operation") != operation or any(
                    action.get(key) != binding[key]
                    for key in ("dimension", "profile", "factor_id")
                ):
                    raise ValueError(
                        "Factor action disagrees with its companion roster role."
                    )
                source = binding["source"]
                if (
                    metadata.get("reference_value_kind") != role
                    or row.get("ledger_selector") != source.get("ledger_selector")
                    or row.get("period") != source.get("period")
                ):
                    raise ValueError(
                        "Effective factor companion source semantics changed."
                    )
                factor_id = binding["factor_id"]
                if factor_id not in factors:
                    raise ValueError(
                        "A transformed reference has missing factor evidence."
                    )
                used_factors.add(factor_id)
                entry = factors[factor_id]
                declaration = entry["declaration"]
                rational = entry["operational_factor"]
                if (
                    action.get("operational_factor") != rational
                    or action.get("source_period") != declaration["source_period"]
                    or action.get("target_period") != declaration["target_period"]
                ):
                    raise ValueError("Effective reference factor identity mismatch.")
                if role == "money":
                    expected = [
                        {
                            "aggregate_fact_key": source.get("ledger_selector", {}).get(
                                "aggregate_fact_key"
                            ),
                            "expected_member_count": 1,
                            "weight": rational["numerator"] / rational["denominator"],
                        }
                    ]
                    if (
                        row.get("value_operation") != "linear_combination"
                        or row.get("value_operands") != expected
                        or action.get("operand_selectors") != expected
                    ):
                        raise ValueError(
                            "Effective money factor must be applied exactly once."
                        )
                elif (
                    row.get("value_operation", "identity")
                    != source.get("value_operation", "identity")
                    or row.get("value_operands", []) != source.get("value_operands", [])
                    or action.get("operand_selectors")
                    != source.get("value_operands", [])
                ):
                    raise ValueError(
                        "Effective count value semantics must remain unchanged."
                    )
    if sorted(names) != resolution.get("transformed_names"):
        raise ValueError("Resolution receipt transformed names mismatch.")
    if used_factors != set(factors):
        raise ValueError("Resolution receipt has unused selected factor evidence.")
    if set(factor_bindings) - set(names):
        raise ValueError("Resolution receipt is missing factor companion rows.")
    return resolution
