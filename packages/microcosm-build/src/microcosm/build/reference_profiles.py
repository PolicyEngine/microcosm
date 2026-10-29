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
REQUIRES_REFERENCE_PROFILE_RESOLUTION = "requires_reference_profile_resolution"
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
    return {
        key: receipt[key]
        for key in (
            "resolver_version",
            "base_references_sha256",
            "base_intervals_sha256",
            "selected_declarations_sha256",
            "selection",
        )
    }


def resolve_reference_profiles(
    base_reference_document: Mapping,
    profile_declarations: Mapping,
    selection: Mapping,
    interval_document: Mapping | None = None,
) -> tuple[dict, dict | None, dict]:
    """Return effective references, effective intervals and a bound receipt.

    Only selected declarations enter selection identity. An unavailable profile
    refuses; an explicitly deferred dimension is reported without claiming its
    transformations ran. Interval transformations are intentionally unsupported
    until a consumer declares their complete transformation contract.
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
    receipt = {
        "resolver_version": REFERENCE_PROFILE_RESOLVER_VERSION,
        "base_references_sha256": reference_document_sha256(base),
        "base_intervals_sha256": (
            None if intervals is None else reference_document_sha256(intervals)
        ),
        "selected_declarations_sha256": reference_document_sha256(chosen),
        "selection": resolved_selection,
    }
    selection_hash = reference_document_sha256(_selection_identity(receipt))
    interval_hash = None if intervals is None else reference_document_sha256(intervals)
    transformed, policies = {}, []
    for dimension, choice in chosen.items():
        for raw_transform in choice["declaration"].get("transformations", []):
            transform = _mapping(raw_transform, "reference transformation")
            allowed = {
                "reference_names",
                "operation",
                "value_operands",
                "metadata",
                "notes",
                "target_label",
            }
            if set(transform) - allowed:
                raise ValueError("Unknown reference transformation fields.")
            if transform.get("operation") != ALLOCATED_RESIDUAL_BY_RATIO:
                raise ValueError("Unsupported reference profile transformation.")
            names = _sequence(transform.get("reference_names"), "transformed names")
            if not names or any(not isinstance(name, str) for name in names):
                raise ValueError("A transformation needs explicit reference names.")
            operands = _sequence(transform.get("value_operands"), "value operands")
            metadata = _mapping(transform.get("metadata", {}), "profile metadata")
            if any(
                not isinstance(key, str) or not isinstance(value, str) or not value
                for key, value in metadata.items()
            ):
                raise ValueError("Profile metadata requires nonempty strings.")
            if set(metadata) & {
                "activation_status",
                "required_reference_profile",
                "reference_selection_sha256",
                "effective_intervals_sha256",
                "reference_profiles_resolved",
            }:
                raise ValueError("A profile cannot override resolution identities.")
            if _carried_value_keys(transform):
                raise ValueError("Reference transformations carry observed values.")
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


def validate_reference_resolution(
    document: Mapping,
    receipt: Mapping,
    *,
    references_sha256: str | None = None,
    interval_document: Mapping | None = None,
) -> dict:
    """Refuse stale effective documents, receipts or interval companions."""
    resolution = _copy(_mapping(receipt, "reference resolution receipt"))
    if resolution.get("resolver_version") != REFERENCE_PROFILE_RESOLVER_VERSION:
        raise ValueError("Unknown reference resolver version.")
    claimed_receipt_hash = resolution.pop("receipt_sha256", None)
    if claimed_receipt_hash != reference_document_sha256(resolution):
        raise ValueError("Reference resolution receipt hash mismatch.")
    resolution["receipt_sha256"] = claimed_receipt_hash
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
    names = []
    for row in document.get("target_references", ()):
        metadata = row.get("metadata", {})
        if metadata.get("reference_profiles_resolved"):
            names.append(row["name"])
            if metadata.get("reference_selection_sha256") != selection_hash or (
                metadata.get("effective_intervals_sha256")
                != resolution.get("effective_intervals_sha256")
            ):
                raise ValueError("Effective reference profile identity mismatch.")
    if sorted(names) != resolution.get("transformed_names"):
        raise ValueError("Resolution receipt transformed names mismatch.")
    return resolution
