"""Pure export comparison against a supplied common parent; no source issuer.

The host must authenticate the actual enriched Population and calibration
ancestry, including targets, ordered weights and geographic scope. A supplied
reference, specification or returned digest cannot establish that ancestry.
The host must also seal those inputs across writer/readback I/O and finish its
owner checks before accepting an export. This module performs no I/O or solve.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from microcosm.frame import Frame, MassChange, WeightKind, Weights
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import storage_equal

PROTOCOL = "microcosm.us.supplied-parent-export-comparison.v1"
MAX_SPECIFICATION_BYTES = 1024 * 1024


class RetainedFrameExportError(ValueError):
    """An export differs from its supplied parent or declared weight/scope inputs."""


def _require(condition, code):
    if not condition:
        raise RetainedFrameExportError(code)


def _vector(value, dtype, code):
    _require(
        type(value) is np.ndarray and value.ndim == 1 and value.dtype == dtype,
        code,
    )
    return value


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _array_binding(value):
    _require(value.dtype.kind in ("i", "u", "f"), "NUMERICAL_BINDING_DTYPE")
    return {
        "dtype": value.dtype.str,
        "rows": len(value),
        "sha256": _sha(np.ascontiguousarray(value).tobytes()),
    }


def _same_series(expected, actual, *, readback):
    """Allow only masked-cell codec canonicalization on logical readback."""
    if expected.dtype != actual.dtype or len(expected) != len(actual):
        return False
    expected_null = expected.isna().to_numpy(dtype=np.bool_)
    actual_null = actual.isna().to_numpy(dtype=np.bool_)
    if not np.array_equal(expected_null, actual_null):
        return False
    if readback and expected.dtype == object:
        # Checkpoint supports object-string strata. Raw object pointers cannot
        # identify values across a serialization boundary. Arbitrary object
        # equality is outside this small numerical/string contract.
        left = expected.to_numpy()[~expected_null].tolist()
        right = actual.to_numpy()[~actual_null].tolist()
        return all(type(v) is str for v in (*left, *right)) and left == right
    return storage_equal(expected, actual, ~expected_null if readback else None)


def verify_retained_frame_export(
    parent: Frame,
    candidate: Frame,
    *,
    parent_reference: str,
    ordered_household_ids: np.ndarray,
    calibrated_weights: np.ndarray,
    calibration_specification: bytes,
    scope_household_ids: np.ndarray | None = None,
    prune_zero_weight: bool = True,
    comparison: str = "storage",
    expected_binding: bytes | None = None,
) -> bytes:
    """Compare complete retained inputs and bind supplied numerical export inputs.

    IDs retain the parent's exact signed/unsigned integer dtype; weights are
    exact float64 arrays in the *whole parent*
    household order. Scope IDs must be a unique subset in that same order;
    None explicitly denotes all households. Pruning removes exactly zero
    weights, with no tolerance or default origin-quality requirement. Household
    selection carries all its persons and precisely their referenced groups.

    Call before writing, then on the serializer's logical readback with
    comparison="frame-checkpoint-readback" and the prior expected_binding.
    That closed mode allows only existing
    codec normalization under null masks; known values, dtype and masks remain
    exact. Extra or missing columns and record reordering refuse. Pandas row
    indices may be reset: entity IDs, not incidental indices, define alignment.

    No Population ownership, design anchor, mass ledger, target interpretation,
    geographic predicate meaning or source authenticity is established here.
    The host must bind parent_reference to actual graph ancestry and verify the
    specification/scope against that ancestry; opaque bytes are not authority.
    Empty support, scopes with no positive weight, and experimental separate
    link tables refuse explicitly, matching the Frame weight contract.
    """
    _require(isinstance(parent, Frame) and isinstance(candidate, Frame), "FRAME_TYPE")
    _require(
        type(parent_reference) is str and bool(parent_reference), "PARENT_REFERENCE"
    )
    _require(
        type(calibration_specification) is bytes
        and 0 < len(calibration_specification) <= MAX_SPECIFICATION_BYTES,
        "CALIBRATION_SPECIFICATION",
    )
    _require(type(prune_zero_weight) is bool, "OPTIONS")
    _require(
        type(comparison) is str
        and comparison in ("storage", "frame-checkpoint-readback"),
        "COMPARISON_MODE",
    )
    readback = comparison == "frame-checkpoint-readback"
    _require(
        expected_binding is None or type(expected_binding) is bytes, "BINDING_TYPE"
    )
    _require(parent.schema == candidate.schema, "SCHEMA")
    _require(
        not parent.schema.links and parent.links == candidate.links == (),
        "SEPARATE_LINK_TABLES_UNSUPPORTED",
    )
    _require("household" in parent.schema.group_entities, "HOUSEHOLD_SCHEMA")
    _require(
        "household" in parent.weighted_entities
        and parent.weighted_entities == candidate.weighted_entities,
        "WEIGHT_TOPOLOGY",
    )
    _require(parent.metadata == candidate.metadata, "FRAME_METADATA")
    household_id = parent.schema.entity_id_column("household")
    parent_ids = parent.table("household")[household_id].to_numpy()
    _require(parent_ids.dtype.kind in ("i", "u"), "INTEGER_HOUSEHOLD_IDS")
    ids = _vector(ordered_household_ids, parent_ids.dtype, "HOUSEHOLD_ID_VECTOR")
    weights = _vector(calibrated_weights, np.dtype("float64"), "WEIGHT_VECTOR")
    _require(
        parent_ids.dtype == ids.dtype
        and np.array_equal(parent_ids, ids)
        and len(np.unique(ids)) == len(ids),
        "COMPLETE_ORDERED_HOUSEHOLD_IDS",
    )
    _require(
        weights.shape == ids.shape
        and np.isfinite(weights).all()
        and (weights >= 0).all(),
        "FINITE_ALIGNED_WEIGHTS",
    )
    if scope_household_ids is None:
        scope = ids
        scope_mask = np.ones(len(ids), dtype=np.bool_)
    else:
        scope = _vector(scope_household_ids, ids.dtype, "SCOPE_ID_VECTOR")
        scope_mask = np.isin(ids, scope)
        _require(np.array_equal(ids[scope_mask], scope), "ORDERED_SCOPE_SUBSET")
    keep = scope_mask & (weights > 0) if prune_zero_weight else scope_mask
    _require(keep.any() and (weights[keep] > 0).any(), "EMPTY_EXPORT_SUPPORT")
    membership = parent.schema.membership_column("household")
    comparison_parent = parent.with_weights(
        "household",
        Weights(weights, WeightKind.CALIBRATED),
        mass=MassChange(
            factor=None,
            reason="Temporary supplied-weight view for retained export comparison",
        ),
    )
    selected = comparison_parent.select(
        np.isin(parent.person[membership].to_numpy(), ids[keep])
    )
    _require(selected.entities == candidate.entities, "ENTITY_ROSTER")
    entity_ids = {}
    for entity in selected.entities:
        expected, actual = selected.table(entity), candidate.table(entity)
        _require(
            not actual.columns.has_duplicates
            and tuple(expected.columns) == tuple(actual.columns),
            "COLUMN_ROSTER:" + entity,
        )
        id_column = parent.schema.entity_id_column(entity)
        _require(
            _same_series(expected[id_column], actual[id_column], readback=False),
            "RETAINED_ENTITY_IDS:" + entity,
        )
        for name in expected.columns:
            _require(
                _same_series(expected[name], actual[name], readback=readback),
                "RETAINED_INPUT:" + entity + "." + str(name),
            )
        entity_ids[entity] = _array_binding(expected[id_column].to_numpy())
    _require(
        selected.strata.name == candidate.strata.name
        and _same_series(selected.strata, candidate.strata, readback=readback),
        "RETAINED_STRATA",
    )
    for entity in selected.weighted_entities:
        actual = candidate.weights_for(entity)
        if entity == "household":
            expected_values, expected_kind = weights[keep], WeightKind.CALIBRATED
        else:
            expected_weight = selected.weights_for(entity)
            expected_values, expected_kind = (
                expected_weight.values,
                expected_weight.kind,
            )
        _require(
            actual.kind is expected_kind
            and storage_equal(pd.Series(expected_values), pd.Series(actual.values)),
            "RETAINED_WEIGHTS:" + entity,
        )
    binding = canonical_json(
        {
            "protocol": PROTOCOL,
            "scope": "comparison_against_supplied_parent_only",
            "parent_reference": parent_reference,
            "actual_graph_calibration_ancestry_verified": False,
            "release_eligible": False,
            "calibration_specification_sha256": _sha(calibration_specification),
            "complete_ordered_household_ids": _array_binding(ids),
            "complete_calibrated_weights": _array_binding(weights),
            "scope_household_ids": _array_binding(scope),
            "prune_zero_weight": prune_zero_weight,
            "retained_entity_ids": entity_ids,
        }
    )
    _require(expected_binding is None or binding == expected_binding, "EXPORT_BINDING")
    return binding
