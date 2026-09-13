"""Directional ContentStore replay checks, never source or population authority.

The caller must separately retain the original source/population seals. These
comparisons permit the store's missing-buffer normalization and scalar object
encoding only at a named materialization boundary. They are not mutation seals.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from microcosm.frame import Frame, MassChangeRecord, WeightKind
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import MassRecord, Population
from microcosm.graph.store import (
    _axis_name_payload,
    _encode_frame_metadata,
    _encode_object_scalar,
)


def _require(condition, reason):
    if not condition:
        raise ValueError("SURVEY_POPULATION_REPLAY_" + reason)


def _array_bytes_equal(left, right):
    return (
        type(left) is np.ndarray
        and type(right) is np.ndarray
        and left.dtype == right.dtype
        and not left.dtype.hasobject
        and left.shape == right.shape
        and left.tobytes() == right.tobytes()
    )


def _object_bytes(value):
    try:
        return _encode_object_scalar(value)
    except (TypeError, ValueError, OverflowError, UnicodeError):
        raise ValueError("SURVEY_POPULATION_REPLAY_UNSUPPORTED_OBJECT") from None


def _series(expected, actual):
    _require(
        type(expected) is pd.Series
        and type(actual) is pd.Series
        and type(expected.dtype) is type(actual.dtype)
        and expected.dtype == actual.dtype
        and len(expected) == len(actual),
        "SERIES_DTYPE_OR_LENGTH",
    )
    dtype = expected.dtype
    left, right = expected.array, actual.array
    if isinstance(dtype, pd.BooleanDtype) or (
        isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and pd.api.types.is_integer_dtype(dtype)
    ):
        # The v1 country profile supports pandas' actual masked integer/bool
        # storage. Other extension-array implementations must be reviewed first.
        ld, rd = getattr(left, "_data", None), getattr(right, "_data", None)
        lm, rm = getattr(left, "_mask", None), getattr(right, "_mask", None)
        _require(
            _array_bytes_equal(lm, rm)
            and lm.dtype == np.dtype(np.bool_)
            and lm.shape == (len(expected),)
            and type(ld) is np.ndarray
            and type(rd) is np.ndarray
            and ld.dtype == rd.dtype
            and not ld.dtype.hasobject
            and ld.shape == rd.shape == lm.shape,
            "MASKED_STORAGE",
        )
        _require(_array_bytes_equal(ld[~lm], rd[~rm]), "PRESENT_BITS")
        _require(
            _array_bytes_equal(ld, rd) or bool(np.all(rd[rm] == 0)),
            "NONCANONICAL_NULL_BACKING",
        )
        return
    if isinstance(dtype, pd.StringDtype):
        _require(
            dtype.storage == actual.dtype.storage
            and (dtype.na_value is pd.NA) == (actual.dtype.na_value is pd.NA),
            "STRING_POLICY",
        )
        lm = expected.isna().to_numpy(dtype=np.bool_, copy=False)
        rm = actual.isna().to_numpy(dtype=np.bool_, copy=False)
        _require(_array_bytes_equal(lm, rm), "STRING_MASK")
        for missing, a, b in zip(lm, expected, actual, strict=True):
            if not missing:
                _require(type(a) is str and type(b) is str and a == b, "STRING_VALUE")
        return
    if pd.api.types.is_object_dtype(dtype):
        # Reuse the store's typed scalar bytes. None/pd.NA/NaT, bool/int,
        # float signed zero and NaN payloads remain distinguished by that codec.
        # Object pointers and np scalar wrappers are not portable identities.
        for a, b in zip(expected, actual, strict=True):
            _require(_object_bytes(a) == _object_bytes(b), "OBJECT_VALUE")
        return
    _require(
        not isinstance(dtype, pd.api.extensions.ExtensionDtype),
        "UNSUPPORTED_EXTENSION_DTYPE",
    )
    _require(
        _array_bytes_equal(expected.to_numpy(copy=False), actual.to_numpy(copy=False)),
        "NATIVE_BITS",
    )


def _name_bytes(value):
    try:
        return canonical_json(_axis_name_payload(value))
    except (TypeError, ValueError, OverflowError, UnicodeError):
        raise ValueError("SURVEY_POPULATION_REPLAY_UNSUPPORTED_AXIS_NAME") from None


def _axis(expected, actual):
    _require(
        type(expected) is type(actual)
        and expected.identical(actual)
        and not isinstance(expected, pd.MultiIndex),
        "AXIS",
    )
    _require(_name_bytes(expected.name) == _name_bytes(actual.name), "AXIS_NAME")
    _series(pd.Series(expected.array, copy=False), pd.Series(actual.array, copy=False))


def same_replayed_frame(expected: Frame, actual: Frame) -> None:
    """Compare actual Frames, allowing only the named store representation rule.

    No I/O or allocation of another Frame. Temporary buffers are proportional
    to individual columns and context metadata, without a total input RAM cap.
    """
    _require(isinstance(expected, Frame) and isinstance(actual, Frame), "FRAME_TYPE")
    _require(
        expected.schema == actual.schema
        and expected.entities == actual.entities
        and expected.links == actual.links == ()
        and canonical_json(_encode_frame_metadata(expected.metadata))
        == canonical_json(_encode_frame_metadata(actual.metadata))
        and all(
            type(r) is MassChangeRecord for r in (*expected.mass_log, *actual.mass_log)
        )
        and canonical_json([asdict(r) for r in expected.mass_log])
        == canonical_json([asdict(r) for r in actual.mass_log])
        and expected.weighted_entities == actual.weighted_entities,
        "FRAME_CONTEXT",
    )
    for entity in expected.entities:
        left, right = expected.table(entity), actual.table(entity)
        _require(
            type(left) is pd.DataFrame
            and type(right) is pd.DataFrame
            and left.flags == right.flags,
            "TABLE_TYPE_OR_FLAGS",
        )
        _axis(left.index, right.index)
        _axis(left.columns, right.columns)
        for column in left:
            _series(left[column], right[column])
    _axis(expected.strata.index, actual.strata.index)
    _require(
        _name_bytes(expected.strata.name) == _name_bytes(actual.strata.name),
        "STRATA_NAME",
    )
    _series(expected.strata, actual.strata)
    for entity in expected.weighted_entities:
        left, right = expected.weights_for(entity), actual.weights_for(entity)
        _require(
            left.kind is right.kind and _array_bytes_equal(left.values, right.values),
            "WEIGHT_BYTES",
        )


def same_replayed_population(expected: Population, actual: Population) -> None:
    """Require complete receiving state as well as the directional Frame check."""
    _require(
        type(expected) is Population and type(actual) is Population, "POPULATION_TYPE"
    )
    same_replayed_frame(expected.frame, actual.frame)
    _require(
        type(expected.version) is type(actual.version) is str
        and expected.version == actual.version
        and tuple(sorted(expected.owners.items()))
        == tuple(sorted(actual.owners.items()))
        and all(type(v) is str for v in actual.owners.values())
        and tuple(expected.weight_kind.items()) == tuple(actual.weight_kind.items())
        and all(type(v) is WeightKind for v in actual.weight_kind.values())
        and type(expected.mass_ledger) is type(actual.mass_ledger) is tuple
        and all(
            type(r) is MassRecord for r in (*expected.mass_ledger, *actual.mass_ledger)
        )
        and canonical_json([asdict(r) for r in expected.mass_ledger])
        == canonical_json([asdict(r) for r in actual.mass_ledger])
        and tuple(expected.design_weights) == tuple(actual.design_weights),
        "POPULATION_CONTEXT",
    )
    for entity in expected.design_weights:
        _require(
            _array_bytes_equal(
                expected.design_weights[entity], actual.design_weights[entity]
            ),
            "DESIGN_BYTES",
        )
