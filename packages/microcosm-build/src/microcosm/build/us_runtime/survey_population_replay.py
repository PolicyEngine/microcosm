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


def _array_seal(values, reason):
    """The retained half of ``_array_bytes_equal``: its type/dtype/shape + bytes.

    ``_array_bytes_equal`` returns False rather than raising, so its caller's
    own code is what a defect surfaces as. The seal takes that code as an
    argument so a structured dtype carrying an object field, or a buffer that
    is not an ``ndarray`` at all, refuses under ``NATIVE_BITS``,
    ``WEIGHT_BYTES`` or ``DESIGN_BYTES`` exactly as it does today, instead of
    under a code this comparison does not have.
    """
    _require(type(values) is np.ndarray and not values.dtype.hasobject, reason)
    return (values.dtype, values.shape, _seal_digest_bytes(values.tobytes()))


def _same_array_seal(expected, actual, reason):
    # Unpacked rather than indexed. `test_us_spine_blindness.py`'s
    # source-spine analyser refuses a subscript it cannot resolve statically on
    # a name it has inferred to be a column container, and every seal record in
    # this module is a positional tuple. Naming the fields satisfies that gate
    # without a single behavioural change -- the records, their order and their
    # `repr` are untouched, so no seal identity moves.
    expected_dtype, expected_shape, expected_digest = expected
    actual_dtype, actual_shape, actual_digest = actual
    _require(
        expected_dtype == actual_dtype
        and expected_shape == actual_shape
        and expected_digest == actual_digest,
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
        # ``bool(np.all(rd[rm] == 0))`` for this side, spelled exactly as the
        # comparison spells it: a truthiness test instead would call an empty
        # string canonical while ``== 0`` does not. An unmasked column is
        # canonical by construction and skips the gather entirely.
        canonical = not mask.any() or bool(np.all(data[mask] == 0))
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
    return ("native", head, _array_seal(series.to_numpy(copy=False), "NATIVE_BITS"))


def _same_series_seal(expected, actual):
    """Refuse exactly what ``_series`` refuses, with the same codes."""
    expected_kind, expected_head, *expected_rest = expected
    actual_kind, actual_head, *actual_rest = actual
    _require(expected_kind == actual_kind, "SERIES_DTYPE_OR_LENGTH")
    expected_class, expected_dtype, expected_length = expected_head
    actual_class, actual_dtype, actual_length = actual_head
    _require(
        expected_class is actual_class
        and expected_dtype == actual_dtype
        and expected_length == actual_length,
        "SERIES_DTYPE_OR_LENGTH",
    )
    if expected_kind == "masked":
        (
            expected_data_dtype,
            expected_mask,
            expected_present,
            expected_backing,
            _expected_canonical,
        ) = expected_rest
        (
            actual_data_dtype,
            actual_mask,
            actual_present,
            actual_backing,
            actual_canonical,
        ) = actual_rest
        _require(
            expected_data_dtype == actual_data_dtype and expected_mask == actual_mask,
            "MASKED_STORAGE",
        )
        _require(expected_present == actual_present, "PRESENT_BITS")
        # Directional, and per series: the ACTUAL side may carry canonically
        # zeroed backing beneath its nulls, which is what a store round trip
        # produces. One non-canonical column must not force byte equality on
        # the others, so this disjunction stays inside the per-series walk.
        _require(
            actual_canonical or expected_backing == actual_backing,
            "NONCANONICAL_NULL_BACKING",
        )
        return
    if expected_kind == "string":
        expected_storage, expected_na, expected_nulls, expected_values = expected_rest
        actual_storage, actual_na, actual_nulls, actual_values = actual_rest
        _require(
            expected_storage == actual_storage and expected_na == actual_na,
            "STRING_POLICY",
        )
        _require(expected_nulls == actual_nulls, "STRING_MASK")
        _require(expected_values == actual_values, "STRING_VALUE")
        return
    if expected_kind == "object":
        (expected_values,) = expected_rest
        (actual_values,) = actual_rest
        _require(expected_values == actual_values, "OBJECT_VALUE")
        return
    (expected_array,) = expected_rest
    (actual_array,) = actual_rest
    _same_array_seal(expected_array, actual_array, "NATIVE_BITS")


# The equivalence classes ``Index.equals`` puts an OBJECT-axis value in, as
# measured on this pandas pin rather than read off its source: ``{None}`` and
# every NaN are ONE class, ``pd.NA`` and ``pd.NaT`` are each their own, bool,
# int and float share one numeric class by EXACT value (``True == 1 == 1.0``,
# ``0 == False == -0.0``, and ``2**53 == float(2**53)`` while ``2**53 + 1`` is
# distinct from both), and ``str`` and ``bytes`` are each their own. These tags
# are a fold-local canonical form, never persisted and never compared across
# processes -- see ``seal_identity`` -- so they are not a stored encoding and
# they move no digest that leaves this process.
_AXIS_NULL = b"\x00"  # None and every NaN
_AXIS_PD_NA = b"\x01"
_AXIS_PD_NAT = b"\x02"
_AXIS_NUMBER = b"\x03"
_AXIS_POSITIVE_INFINITY = b"\x04"
_AXIS_NEGATIVE_INFINITY = b"\x05"
_AXIS_TEXT = b"\x06"
_AXIS_BYTES = b"\x07"


def _plain_object(value):
    """The builtin-typed value the store codec encodes ``value`` as."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, str):
        return str(value)
    return bytes(value)


def _object_axis_equivalence(value):
    """The class token for one object-axis value, under the ``AXIS`` code.

    ``Index.equals`` over an object axis is an element-wise ``==`` between
    arbitrary Python objects, so a fold that reproduces each refusal's own code
    has to put two values in one class exactly when ``==`` holds them equal --
    no more and no less. A digest of the store codec's bytes does neither: it
    is WEAKER, because two values with equal bytes need not be ``==`` when one
    carries its own ``__eq__``; and STRICTER, because the codec spells apart
    ``None`` from NaN, ``True`` from ``1`` and ``-0.0`` from ``0.0``, all three
    of which ``==`` holds equal. Both directions were reproduced before this
    was written.

    So the codec is the gatekeeper and not the fold: ``_object_bytes`` runs
    first, which keeps ``UNSUPPORTED_OBJECT`` firing on a leaf no column may
    carry, and the class token is then the measured equivalence class above.
    The one thing a digest cannot reproduce is an ``__eq__`` of the value's
    own, so a value that is not ``==`` to the plain builtin the codec encodes
    it as refuses ``AXIS`` when its seal is built, on both operands, rather
    than being folded into a class it does not belong to.

    The byte-exact discriminations the codec makes and ``==`` does not are not
    lost: ``_axis_seal``'s last element is ``_series_seal`` over the same
    materialised array, which is the arm ``_axis`` itself falls through to, and
    it reports them under ``OBJECT_VALUE`` exactly as the comparison does.
    """
    _object_bytes(value)
    if value is pd.NA:
        return _AXIS_PD_NA
    if value is pd.NaT:
        return _AXIS_PD_NAT
    if value is None:
        return _AXIS_NULL
    if isinstance(value, (float, np.floating)) and value != value:
        return _AXIS_NULL
    plain = _plain_object(value)
    try:
        faithful = bool(value == plain) and bool(plain == value)
    except (TypeError, ValueError):
        faithful = False
    _require(faithful, "AXIS")
    if isinstance(plain, float):
        if plain == float("inf"):
            return _AXIS_POSITIVE_INFINITY
        if plain == float("-inf"):
            return _AXIS_NEGATIVE_INFINITY
    if isinstance(plain, (bool, int, float)):
        numerator, denominator = plain.as_integer_ratio()
        return _AXIS_NUMBER + f"{numerator}/{denominator}".encode("ascii")
    if isinstance(plain, str):
        return _AXIS_TEXT + plain.encode("utf-8", "surrogatepass")
    return _AXIS_BYTES + plain


def _axis_values_fold(index):
    """A fold over the index's OWN values, under the ``AXIS`` code.

    ``_axis`` refuses ``AXIS`` when ``identical`` fails and only then falls
    through to the byte-exact ``_series`` codes, so an axis needs a
    value-equality fold as well as a byte fold if each refusal is to keep its
    own code. ``Index.equals`` runs ``array_equivalent`` over the index's own
    values -- NOT over ``pd.Series(index.array)``, which pandas 3 re-infers:
    an object axis of ``["r", None]`` and one of ``["r", pd.NA]`` both
    materialise as a ``str`` Series of ``["r", nan]``, so a fold taken through
    a Series would accept a pair ``identical`` refuses. This one is taken
    through ``np.asarray(index.array)``.

    ``array_equivalent`` is tolerant of NaN payloads, NaN sign and signed zeros
    for float and complex (``dtype.kind in "fc"``) and of bool bytes outside
    ``{0, 1}``, so those are collapsed before hashing and the byte difference
    reaches the series code that reports it today. For an object axis the fold
    is ``_object_axis_equivalence``, which reproduces ``==``'s own classes;
    read its docstring for why the store codec's bytes are not that.

    An EXTENSION-dtype axis is taken through its exact object view rather than
    ``np.asarray(index.array)``, because a masked integer array materialises
    as float64 with NA as NaN: every distinction above ``2**53`` would be lost
    and the fold of an ``Int64`` axis would silently be a float fold. The
    object view keeps ``pd.NA`` and the full integer width, and was measured to
    do so for ``Int64``, ``UInt64``, ``boolean`` and ``string``.
    """
    values = np.asarray(index.array)
    if values.dtype != object and isinstance(
        index.dtype, pd.api.extensions.ExtensionDtype
    ):
        values = np.asarray(index.array, dtype=object)
    if values.dtype == object:
        payload = bytearray()
        for value in values:
            encoded = _object_axis_equivalence(value)
            payload += len(encoded).to_bytes(8, "little") + encoded
        return ("object", values.shape, _seal_digest_bytes(bytes(payload)))
    _require(not values.dtype.hasobject, "AXIS")
    original = values.dtype
    if original.kind in "fc":
        values = np.array(values, dtype=original)
        values[np.isnan(values)] = np.nan  # one spelling for every payload
        values = values + original.type(0)  # -0.0 + 0.0 is +0.0, part by part
    elif original.kind == "b":
        values = np.ascontiguousarray(values).view(np.uint8) != 0
    return (
        "raw",
        original,
        values.shape,
        _seal_digest_bytes(np.ascontiguousarray(values).tobytes()),
    )


# ``Index._comparables`` is ``['name']`` for ``Index`` and ``RangeIndex`` and
# ``['name', 'freq']`` for ``DatetimeIndex``, measured. Reading them through a
# literal reader per name rather than ``getattr(index, name, None)`` does two
# things: it satisfies the source-spine analyser, which fails closed on a
# dynamic attribute, and it makes an index class that declares a comparable
# this module has never seen REFUSE rather than fold ``None`` for it silently.
_AXIS_COMPARABLE_READERS = {
    "name": lambda index: index.name,
    "freq": lambda index: getattr(index, "freq", None),
}


def _axis_comparable_values(index, comparables):
    """Read each declared comparable through its own literal reader."""
    values = []
    for comparable in comparables:
        reader = _AXIS_COMPARABLE_READERS.get(comparable)
        _require(reader is not None, "AXIS")
        values.append(reader(index))
    return tuple(values)


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
        _axis_comparable_values(index, comparables),
        index.dtype,
        _axis_values_fold(index),
        _name_seal(index.name),
        # ``_axis`` compares the materialised arrays, re-inference included, so
        # the byte arm is taken through the same construction it uses. The
        # dtype CLASS check lives here rather than above because ``identical``
        # compares dtypes with ``==`` only and leaves the class to ``_series``.
        _series_seal(pd.Series(index.array, copy=False)),
    )


def _same_axis_seal(expected, actual):
    (
        expected_class,
        expected_comparables,
        expected_comparable_values,
        expected_dtype,
        expected_fold,
        expected_name,
        expected_series,
    ) = expected
    (
        actual_class,
        actual_comparables,
        actual_comparable_values,
        actual_dtype,
        actual_fold,
        actual_name,
        actual_series,
    ) = actual
    _require(
        expected_class is actual_class
        and expected_comparables == actual_comparables
        and len(expected_comparable_values) == len(actual_comparable_values)
        # ``identical`` applies ``==`` to each comparable; a tuple comparison
        # would short-circuit on identity and accept a value whose own
        # ``__eq__`` refuses itself.
        and all(
            bool(one == other)
            for one, other in zip(
                expected_comparable_values, actual_comparable_values, strict=True
            )
        )
        and expected_dtype == actual_dtype
        and expected_fold == actual_fold,
        "AXIS",
    )
    _require(expected_name == actual_name, "AXIS_NAME")
    _same_series_seal(expected_series, actual_series)


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
        weights.append((entity, held.kind, _array_seal(held.values, "WEIGHT_BYTES")))
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
        and len(expected) == len(actual) == 11,
        "FRAME_SEAL_PROTOCOL",
    )
    (
        expected_protocol,
        expected_schema,
        expected_entities,
        expected_metadata,
        expected_mass_log,
        expected_weighted,
        expected_tables,
        expected_strata_index,
        expected_strata_name,
        expected_strata,
        expected_weights,
    ) = expected
    (
        actual_protocol,
        actual_schema,
        actual_entities,
        actual_metadata,
        actual_mass_log,
        actual_weighted,
        actual_tables,
        actual_strata_index,
        actual_strata_name,
        actual_strata,
        actual_weights,
    ) = actual
    _require(
        expected_protocol == actual_protocol == SEAL_PROTOCOL,
        "FRAME_SEAL_PROTOCOL",
    )
    _require(
        expected_schema == actual_schema
        and expected_entities == actual_entities
        and expected_metadata == actual_metadata
        and expected_mass_log == actual_mass_log
        and expected_weighted == actual_weighted,
        "FRAME_CONTEXT",
    )
    _require(
        tuple(entity for entity, _, _, _, _ in expected_tables)
        == tuple(entity for entity, _, _, _, _ in actual_tables),
        "FRAME_CONTEXT",
    )
    for left, right in zip(expected_tables, actual_tables, strict=True):
        _, left_flags, left_index, left_columns, left_series = left
        _, right_flags, right_index, right_columns, right_series = right
        _require(left_flags == right_flags, "TABLE_TYPE_OR_FLAGS")
        _same_axis_seal(left_index, right_index)
        _same_axis_seal(left_columns, right_columns)
        _require(
            tuple(column for column, _ in left_series)
            == tuple(column for column, _ in right_series),
            "AXIS",
        )
        for (_, one), (_, other) in zip(left_series, right_series, strict=True):
            _same_series_seal(one, other)
    _same_axis_seal(expected_strata_index, actual_strata_index)
    _require(expected_strata_name == actual_strata_name, "STRATA_NAME")
    _same_series_seal(expected_strata, actual_strata)
    _require(
        tuple(entity for entity, _, _ in expected_weights)
        == tuple(entity for entity, _, _ in actual_weights),
        "FRAME_CONTEXT",
    )
    for (_, kind, values), (_, other_kind, other_values) in zip(
        expected_weights, actual_weights, strict=True
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
            (name, _array_seal(values, "DESIGN_BYTES"))
            for name, values in population.design_weights.items()
        ),
    )


def same_replayed_population_seals(expected: tuple, actual: tuple) -> None:
    """Refuse exactly what ``same_replayed_population`` refuses, same codes."""
    _require(
        type(expected) is tuple
        and type(actual) is tuple
        and len(expected) == len(actual) == 8,
        "POPULATION_SEAL_PROTOCOL",
    )
    (
        expected_protocol,
        expected_frame,
        expected_version,
        expected_owners,
        expected_weight_kind,
        expected_mass_ledger,
        expected_design_names,
        expected_design_weights,
    ) = expected
    (
        actual_protocol,
        actual_frame,
        actual_version,
        actual_owners,
        actual_weight_kind,
        actual_mass_ledger,
        actual_design_names,
        actual_design_weights,
    ) = actual
    _require(
        expected_protocol == actual_protocol == SEAL_PROTOCOL,
        "POPULATION_SEAL_PROTOCOL",
    )
    same_replayed_frame_seals(expected_frame, actual_frame)
    _require(
        expected_version == actual_version
        and expected_owners == actual_owners
        and expected_weight_kind == actual_weight_kind
        and expected_mass_ledger == actual_mass_ledger
        and expected_design_names == actual_design_names,
        "POPULATION_CONTEXT",
    )
    for (_, values), (_, other) in zip(
        expected_design_weights, actual_design_weights, strict=True
    ):
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
