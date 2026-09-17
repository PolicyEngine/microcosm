"""Directional ContentStore replay checks, never source or population authority.

The caller must separately retain the original source/population seals. These
comparisons permit the store's missing-buffer normalization and scalar object
encoding only at a named materialization boundary. They are not mutation seals.
"""

from __future__ import annotations

import hashlib
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


# --- Content seals for the same comparisons, without retaining the objects ---
#
# ``same_replayed_population`` needs both Populations alive at once. A runner
# that observes nineteen node populations before it can build its own replay
# must therefore keep nineteen detached copies (about 290 GiB at full US
# source). The seal below carries, per node, what those comparisons actually
# compare, in space proportional to the number of columns rather than rows.
#
# Two existing seals were measured and are not this one, which is why this is
# a third: ``survey_atomic_geography._population_stamp`` folds masked ``_data``
# beneath nulls, so it refuses the store round trip ``NONCANONICAL_NULL_BACKING``
# deliberately accepts; and ``survey_population_preparation._frame_identity``
# spells every NaN ``null`` and folds no ``DataFrame.flags``, so it misses
# ``NATIVE_BITS`` and ``TABLE_TYPE_OR_FLAGS`` defects. See
# ``docs/us-native-retention-seal.md`` section 0.
#
# The rule the record follows, and the only substitution it makes:
#
#   * every comparison of BYTE STRINGS becomes a comparison of their sha256;
#   * every other predicate is preserved verbatim, by retaining the small
#     object it is applied to (a dtype, a class, an axis name, a WeightKind,
#     a flags value) and applying the identical ``is``/``==`` to it;
#   * every predicate about ONE side is asserted when the seal is built, with
#     the same refusal code, so it fires on arrival instead of at comparison.
#
# The record is nested tuples of immutable values. It is never persisted and
# never compared across processes: the retained dtype and class objects are
# compared by identity, which is exactly what the object comparison does.

SEAL_PROTOCOL = "microcosm.us.survey-population-replay-seal.v1"


def _seal_digest_bytes(*parts: bytes) -> bytes:
    """sha256 over length-prefixed parts, so concatenation is unambiguous."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.digest()


def _array_seal(values):
    """The retained half of ``_array_bytes_equal``: its type/dtype/shape + bytes."""
    _require(type(values) is np.ndarray and not values.dtype.hasobject, "ARRAY_STORAGE")
    return (values.dtype, values.shape, _seal_digest_bytes(values.tobytes()))


def _same_array_seal(expected, actual, reason):
    _require(
        expected[0] == actual[0]
        and expected[1] == actual[1]
        and expected[2] == actual[2],
        reason,
    )


def _name_seal(value):
    return _seal_digest_bytes(_name_bytes(value))


def _series_seal(series):
    """Seal one Series, asserting every one-sided predicate ``_series`` makes.

    The branches, their order and their refusal codes mirror ``_series`` line
    for line; only the byte comparisons become digests.
    """
    _require(type(series) is pd.Series, "SERIES_DTYPE_OR_LENGTH")
    dtype = series.dtype
    length = len(series)
    head = (type(dtype), dtype, length)
    array = series.array
    if isinstance(dtype, pd.BooleanDtype) or (
        isinstance(dtype, pd.api.extensions.ExtensionDtype)
        and pd.api.types.is_integer_dtype(dtype)
    ):
        data, mask = getattr(array, "_data", None), getattr(array, "_mask", None)
        _require(
            type(mask) is np.ndarray
            and mask.dtype == np.dtype(np.bool_)
            and mask.shape == (length,)
            and type(data) is np.ndarray
            and not data.dtype.hasobject
            and data.shape == mask.shape,
            "MASKED_STORAGE",
        )
        # ``np.all(rd[rm] == 0)`` for this side. An unmasked column is
        # canonical by construction, and the gather is skipped for it.
        canonical = not mask.any() or not data[mask].any()
        return (
            "masked",
            head,
            data.dtype,
            _seal_digest_bytes(np.ascontiguousarray(mask).tobytes()),
            _seal_digest_bytes(np.ascontiguousarray(data[~mask]).tobytes()),
            _seal_digest_bytes(np.ascontiguousarray(data).tobytes()),
            bool(canonical),
        )
    if isinstance(dtype, pd.StringDtype):
        nulls = series.isna().to_numpy(dtype=np.bool_, copy=False)
        payload = bytearray()
        for missing, value in zip(nulls, series, strict=True):
            if missing:
                continue
            _require(type(value) is str, "STRING_VALUE")
            encoded = value.encode("utf-8", "surrogatepass")
            payload += len(encoded).to_bytes(8, "little") + encoded
        return (
            "string",
            head,
            dtype.storage,
            dtype.na_value is pd.NA,
            _seal_digest_bytes(np.ascontiguousarray(nulls).tobytes()),
            _seal_digest_bytes(bytes(payload)),
        )
    if pd.api.types.is_object_dtype(dtype):
        payload = bytearray()
        for value in series:
            encoded = _object_bytes(value)
            payload += len(encoded).to_bytes(8, "little") + encoded
        return ("object", head, _seal_digest_bytes(bytes(payload)))
    _require(
        not isinstance(dtype, pd.api.extensions.ExtensionDtype),
        "UNSUPPORTED_EXTENSION_DTYPE",
    )
    return ("native", head, _array_seal(series.to_numpy(copy=False)))


def _same_series_seal(expected, actual):
    """Refuse exactly what ``_series`` refuses, with the same codes."""
    _require(expected[0] == actual[0], "SERIES_DTYPE_OR_LENGTH")
    left, right = expected[1], actual[1]
    _require(
        left[0] is right[0] and left[1] == right[1] and left[2] == right[2],
        "SERIES_DTYPE_OR_LENGTH",
    )
    kind = expected[0]
    if kind == "masked":
        _require(
            expected[2] == actual[2] and expected[3] == actual[3], "MASKED_STORAGE"
        )
        _require(expected[4] == actual[4], "PRESENT_BITS")
        # Directional, and per series: the ACTUAL side may carry canonically
        # zeroed backing beneath its nulls, which is what a store round trip
        # produces. One non-canonical column must not force byte equality on
        # the others, so this disjunction stays inside the per-series walk.
        _require(actual[6] or expected[5] == actual[5], "NONCANONICAL_NULL_BACKING")
        return
    if kind == "string":
        _require(expected[2] == actual[2] and expected[3] == actual[3], "STRING_POLICY")
        _require(expected[4] == actual[4], "STRING_MASK")
        _require(expected[5] == actual[5], "STRING_VALUE")
        return
    if kind == "object":
        _require(expected[2] == actual[2], "OBJECT_VALUE")
        return
    _same_array_seal(expected[2], actual[2], "NATIVE_BITS")


def _axis_seal(index):
    """Seal one axis: ``_axis``'s class, ``identical`` parts, name and values."""
    _require(not isinstance(index, pd.MultiIndex), "AXIS")
    # ``Index.identical`` is ``equals`` and every name in ``_comparables`` and
    # the exact class and the dtype. ``_comparables`` is ``['name']`` for
    # ``Index`` and ``RangeIndex`` and ``['name', 'freq']`` for
    # ``DatetimeIndex``, so folding it generically closes a gap that folding
    # only (class, dtype, name, values) would leave open.
    comparables = tuple(type(index)._comparables)
    return (
        type(index),
        comparables,
        tuple(getattr(index, name, None) for name in comparables),
        _name_seal(index.name),
        _series_seal(pd.Series(index.array, copy=False)),
    )


def _same_axis_seal(expected, actual):
    _require(
        expected[0] is actual[0]
        and expected[1] == actual[1]
        and expected[2] == actual[2],
        "AXIS",
    )
    _require(expected[3] == actual[3], "AXIS_NAME")
    _same_series_seal(expected[4], actual[4])


def _flags_seal(table):
    """Every documented flag, read off ``type(flags)._keys`` rather than named."""
    flags = table.flags
    keys = getattr(type(flags), "_keys", None)
    _require(
        type(keys) is set and keys and all(type(k) is str for k in keys),
        "TABLE_TYPE_OR_FLAGS",
    )
    return tuple((key, getattr(flags, key)) for key in sorted(keys))


def replayed_frame_seal(frame: Frame) -> tuple:
    """Seal everything ``same_replayed_frame`` compares, in O(columns) space."""
    _require(isinstance(frame, Frame), "FRAME_TYPE")
    _require(frame.links == (), "FRAME_CONTEXT")
    _require(all(type(r) is MassChangeRecord for r in frame.mass_log), "FRAME_CONTEXT")
    entities = []
    for entity in frame.entities:
        table = frame.table(entity)
        _require(type(table) is pd.DataFrame, "TABLE_TYPE_OR_FLAGS")
        entities.append(
            (
                entity,
                _flags_seal(table),
                _axis_seal(table.index),
                _axis_seal(table.columns),
                tuple((column, _series_seal(table[column])) for column in table),
            )
        )
    weights = []
    for entity in frame.weighted_entities:
        held = frame.weights_for(entity)
        weights.append((entity, held.kind, _array_seal(held.values)))
    return (
        SEAL_PROTOCOL,
        frame.schema,
        tuple(frame.entities),
        _seal_digest_bytes(canonical_json(_encode_frame_metadata(frame.metadata))),
        _seal_digest_bytes(
            canonical_json([asdict(record) for record in frame.mass_log])
        ),
        tuple(frame.weighted_entities),
        tuple(entities),
        _axis_seal(frame.strata.index),
        _name_seal(frame.strata.name),
        _series_seal(frame.strata),
        tuple(weights),
    )


def same_replayed_frame_seals(expected: tuple, actual: tuple) -> None:
    """Refuse exactly what ``same_replayed_frame`` refuses, with the same codes."""
    _require(
        type(expected) is tuple
        and type(actual) is tuple
        and len(expected) == len(actual) == 11
        and expected[0] == actual[0] == SEAL_PROTOCOL,
        "FRAME_SEAL_PROTOCOL",
    )
    _require(
        expected[1] == actual[1]
        and expected[2] == actual[2]
        and expected[3] == actual[3]
        and expected[4] == actual[4]
        and expected[5] == actual[5],
        "FRAME_CONTEXT",
    )
    _require(
        tuple(entity[0] for entity in expected[6])
        == tuple(entity[0] for entity in actual[6]),
        "FRAME_CONTEXT",
    )
    for left, right in zip(expected[6], actual[6], strict=True):
        _require(left[1] == right[1], "TABLE_TYPE_OR_FLAGS")
        _same_axis_seal(left[2], right[2])
        _same_axis_seal(left[3], right[3])
        _require(
            tuple(column for column, _ in left[4])
            == tuple(column for column, _ in right[4]),
            "AXIS",
        )
        for (_, one), (_, other) in zip(left[4], right[4], strict=True):
            _same_series_seal(one, other)
    _same_axis_seal(expected[7], actual[7])
    _require(expected[8] == actual[8], "STRATA_NAME")
    _same_series_seal(expected[9], actual[9])
    _require(
        tuple(entity for entity, _, _ in expected[10])
        == tuple(entity for entity, _, _ in actual[10]),
        "FRAME_CONTEXT",
    )
    for (_, kind, values), (_, other_kind, other_values) in zip(
        expected[10], actual[10], strict=True
    ):
        _require(kind is other_kind, "WEIGHT_BYTES")
        _same_array_seal(values, other_values, "WEIGHT_BYTES")


def replayed_population_seal(population: Population) -> tuple:
    """Seal everything ``same_replayed_population`` compares.

    Every one-sided predicate that comparison makes is asserted here, on this
    population, with the same code -- including the four it writes against the
    ``actual`` operand alone, which sealing both operands makes fire on both.
    """
    _require(type(population) is Population, "POPULATION_TYPE")
    _require(type(population.version) is str, "POPULATION_CONTEXT")
    _require(
        all(type(v) is str for v in population.owners.values()), "POPULATION_CONTEXT"
    )
    _require(
        all(type(v) is WeightKind for v in population.weight_kind.values()),
        "POPULATION_CONTEXT",
    )
    _require(type(population.mass_ledger) is tuple, "POPULATION_CONTEXT")
    _require(
        all(type(r) is MassRecord for r in population.mass_ledger), "POPULATION_CONTEXT"
    )
    return (
        SEAL_PROTOCOL,
        replayed_frame_seal(population.frame),
        population.version,
        tuple(sorted(population.owners.items())),
        tuple(population.weight_kind.items()),
        _seal_digest_bytes(
            canonical_json([asdict(record) for record in population.mass_ledger])
        ),
        tuple(population.design_weights),
        tuple(
            (name, _array_seal(values))
            for name, values in population.design_weights.items()
        ),
    )


def same_replayed_population_seals(expected: tuple, actual: tuple) -> None:
    """Refuse exactly what ``same_replayed_population`` refuses, same codes."""
    _require(
        type(expected) is tuple
        and type(actual) is tuple
        and len(expected) == len(actual) == 8
        and expected[0] == actual[0] == SEAL_PROTOCOL,
        "POPULATION_SEAL_PROTOCOL",
    )
    same_replayed_frame_seals(expected[1], actual[1])
    _require(
        expected[2] == actual[2]
        and expected[3] == actual[3]
        and expected[4] == actual[4]
        and expected[5] == actual[5]
        and expected[6] == actual[6],
        "POPULATION_CONTEXT",
    )
    for (_, values), (_, other) in zip(expected[7], actual[7], strict=True):
        _same_array_seal(values, other, "DESIGN_BYTES")


def seal_identity(seal: tuple) -> str:
    """A stable hex digest of a seal record, for retaining it under a stamp.

    ``repr`` is the encoding because a seal holds live dtype, class and
    ``WeightKind`` objects whose canonical JSON does not exist; every one of
    them has an unambiguous ``repr`` and the digest is only ever compared with
    another digest taken in the same process.
    """
    _require(type(seal) is tuple, "SEAL_TYPE")
    return hashlib.sha256(repr(seal).encode("utf-8", "surrogatepass")).hexdigest()
