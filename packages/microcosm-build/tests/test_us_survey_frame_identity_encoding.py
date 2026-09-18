"""Frame identity byte parity, refusal behavior, and mutation detection.

Fixtures are invented and never acquire survey files. The legacy oracle retains
an independent implementation of the previous frame traversal.
"""

from __future__ import annotations

import datetime
import hashlib
import sys
import weakref
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import survey_population_preparation as owner
from microcosm.frame import US_SCHEMA, Frame, MassChangeRecord, WeightKind, Weights


def _legacy_cell(value, maximum=owner.MAX_PAYLOAD_BYTES):
    return owner._encode(owner._cell(value), maximum)


def _outcome(call):
    """Compare bytes or the exact exception type/message, including precedence."""
    try:
        return ("value", call())
    except Exception as error:
        return ("error", type(error), str(error))


@pytest.mark.parametrize(
    "value",
    [
        None,
        pd.NA,
        False,
        True,
        -(2**63) - 1,
        -(2**63),
        -1,
        0,
        1,
        2**63 - 1,
        2**63,
        2**64 - 1,
        2**64,
        10**100,
        0.0,
        -0.0,
        1.0,
        -1.25,
        float.fromhex("0x0.0000000000001p-1022"),
        float.fromhex("0x1.fffffffffffffp+1023"),
        float("nan"),
        -float("nan"),
        "",
        'quote" backslash\\ slash/',
        "\x00\b\t\n\f\r\x1f\x7f",
        "é e\u0301 中文 🧪 \u2028\u2029",
        "null true false 17 -0.0",
        "x" * 4095,
        "x" * 4096,
        "x" * 4097,
        "\x00" * 4096,
        "🧪" * 4096,
    ],
)
def test_scalar_exact_bytes_and_payload_boundaries(value):
    expected = _legacy_cell(value)
    assert owner._frame_cell_encode(value) == expected
    for maximum in (-1, 0, 1, len(expected) - 1, len(expected), len(expected) + 1):
        assert _outcome(
            lambda maximum=maximum: owner._frame_cell_encode(value, maximum)
        ) == _outcome(lambda maximum=maximum: _legacy_cell(value, maximum))


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_native_signed_integer_extremes(dtype):
    bounds = np.iinfo(dtype)
    for raw in (bounds.min, -1, 0, 1, bounds.max):
        value = dtype(raw)
        assert owner._frame_cell_encode(value) == _legacy_cell(value)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
def test_native_unsigned_integer_extremes(dtype):
    for raw in (0, 1, np.iinfo(dtype).max):
        value = dtype(raw)
        assert owner._frame_cell_encode(value) == _legacy_cell(value)


@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64, np.longdouble])
def test_numpy_floating_conversion_and_platform_wide_refusals(dtype):
    # longdouble.item() can remain a NumPy scalar. The existing _cell decides
    # admission; neither the candidate nor this test coerces it to Python float.
    for raw in ("0", "-0", "1.25", "nan", "inf", "-inf"):
        value = dtype(raw)
        assert _outcome(
            lambda value=value: owner._frame_cell_encode(value)
        ) == _outcome(lambda value=value: _legacy_cell(value))
    for raw in (np.finfo(dtype).smallest_subnormal, np.finfo(dtype).max):
        value = dtype(raw)
        assert _outcome(
            lambda value=value: owner._frame_cell_encode(value)
        ) == _outcome(lambda value=value: _legacy_cell(value))


def test_signed_zero_and_missing_spellings_are_pinned():
    assert owner._frame_cell_encode(0.0) == b'["float","0x0.0p+0"]'
    assert owner._frame_cell_encode(-0.0) == b'["float","-0x0.0p+0"]'
    assert owner._frame_cell_encode(0.0) != owner._frame_cell_encode(-0.0)
    nan_payloads = np.array(
        [0x7FF8000000000001, 0x7FF8000000000002, 0xFFF8000000000001],
        dtype=np.uint64,
    ).view(np.float64)
    for value in [None, pd.NA, float("nan"), *nan_payloads]:
        assert owner._frame_cell_encode(value) == b"null"
    assert owner._frame_cell_encode(False) == b"false"
    assert owner._frame_cell_encode(0) == b"0"
    assert owner._frame_cell_encode("0") == b'"0"'


@pytest.mark.parametrize(
    "value",
    [
        np.bool_(True),
        np.bool_(False),
        np.str_("é\n🧪"),
        np.bytes_(b"invented"),
        np.complex64(1 + 2j),
        np.complex128(1 + 2j),
        np.datetime64("2001-02-03", "D"),
        np.datetime64("2001-02-03", "ns"),
        np.datetime64("NaT"),
        np.timedelta64(2, "D"),
        np.timedelta64(2, "ns"),
        np.void(b"invented"),
        np.array(1),
        np.array([1]),
        pd.NaT,
        pd.Timestamp("2001-02-03"),
        datetime.date(2001, 2, 3),
        b"invented",
        bytearray(b"invented"),
        Decimal("1.25"),
        Fraction(1, 2),
        1 + 2j,
        ["float", "0x0.0p+0"],
        (1,),
        {"value": 1},
        {1},
        object(),
    ],
)
def test_numpy_and_unsupported_scalar_admission_matches_legacy(value):
    assert _outcome(lambda value=value: owner._frame_cell_encode(value)) == _outcome(
        lambda value=value: _legacy_cell(value)
    )


def test_scalar_subclasses_remain_refused():
    class Text(str):
        pass

    class Integer(int):
        pass

    class Floating(float):
        pass

    for value in (Text("x"), Integer(1), Floating(1.25)):
        for encode in (_legacy_cell, owner._frame_cell_encode):
            with pytest.raises(
                owner.SurveyPopulationPreparationError, match="^FRAME_CELL_TYPE$"
            ):
                encode(value)
    # The predecessor's early missing check also accepts a float-subclass NaN.
    assert owner._frame_cell_encode(Floating("nan")) == _legacy_cell(Floating("nan"))


@pytest.mark.parametrize(
    "value, maximum, error_type, code",
    [
        (object(), -1, owner.SurveyPopulationPreparationError, "FRAME_CELL_TYPE"),
        (float("inf"), -1, owner.SurveyPopulationPreparationError, "FRAME_NONFINITE"),
        (-float("inf"), -1, owner.SurveyPopulationPreparationError, "FRAME_NONFINITE"),
        ("\ud800", 0, UnicodeEncodeError, None),
        ("prefix\udfff", 0, UnicodeEncodeError, None),
        ("x" * 4097 + "\ud800", 0, UnicodeEncodeError, None),
        (
            "x" * (owner._MAX_SCALAR_BYTES + 1),
            0,
            owner.SurveyPopulationPreparationError,
            "SCALAR_LIMIT",
        ),
        (
            "\ud800" + "x" * owner._MAX_SCALAR_BYTES,
            0,
            owner.SurveyPopulationPreparationError,
            "SCALAR_LIMIT",
        ),
        ("x", 0, owner.SurveyPopulationPreparationError, "PAYLOAD_LIMIT"),
    ],
)
def test_refusal_type_code_and_precedence(value, maximum, error_type, code):
    expected = _outcome(lambda maximum=maximum: _legacy_cell(value, maximum))
    assert expected[0:2] == ("error", error_type)
    if code is not None:
        assert expected[2] == code
    assert (
        _outcome(lambda maximum=maximum: owner._frame_cell_encode(value, maximum))
        == expected
    )


def test_scalar_check_retained_before_fast_encoding(monkeypatch):
    monkeypatch.setattr(owner, "_MAX_SCALAR_BYTES", 4)
    for value in ("12345", 1.25, -0.0):
        expected = _outcome(lambda value=value: _legacy_cell(value, 0))
        assert expected == (
            "error",
            owner.SurveyPopulationPreparationError,
            "SCALAR_LIMIT",
        )
        assert (
            _outcome(lambda value=value: owner._frame_cell_encode(value, 0)) == expected
        )


def test_large_integer_decimal_policy_is_legacy_fallback():
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("Interpreter decimal conversion limit is disabled")
    if limit > 10_000:
        pytest.skip("Decimal refusal case exceeds the invented 10000-digit cap")
    value = 10**limit
    expected = _outcome(lambda value=value: _legacy_cell(value, 0))
    assert expected[0:2] == ("error", ValueError)
    assert _outcome(lambda value=value: owner._frame_cell_encode(value, 0)) == expected


def test_large_scalar_fallback_remains_bounded_and_called(monkeypatch):
    original = owner._encode
    calls = []

    def record(value, maximum=owner.MAX_PAYLOAD_BYTES):
        calls.append((value, maximum))
        return original(value, maximum)

    monkeypatch.setattr(owner, "_encode", record)
    for value in ("x" * 4097, -(2**63) - 1, 2**64):
        assert owner._frame_cell_encode(value) == original(owner._cell(value))
        assert calls[-1] == (value, owner.MAX_PAYLOAD_BYTES)
    assert len(calls) == 3
    for value in (None, True, -1, 2**64 - 1, "x" * 4096, -0.0):
        assert owner._frame_cell_encode(value) == original(owner._cell(value))
    assert len(calls) == 3


@pytest.mark.parametrize("length_delta", [-1, 0, 1])
def test_maximum_scalar_length_fallback_equality(length_delta):
    value = "x" * (owner._MAX_SCALAR_BYTES + length_delta)
    expected = _outcome(lambda value=value: _legacy_cell(value))
    if length_delta <= 0:
        assert expected[0] == "value"
    else:
        assert expected == (
            "error",
            owner.SurveyPopulationPreparationError,
            "SCALAR_LIMIT",
        )
    assert _outcome(lambda value=value: owner._frame_cell_encode(value)) == expected


@pytest.mark.parametrize(
    "change", ["helper_code", "helper_defaults", "bound", "string_provider"]
)
def test_live_producer_seal_binds_fast_encoder(change, monkeypatch):
    # Restrict the owner scan to in-memory source; the refusing producer must
    # short-circuit before its source-file hashing callback can run.
    monkeypatch.setattr(owner, "_modules", lambda: (owner,))
    reads = []

    def refuse_source_reads():
        reads.append(True)
        pytest.fail("Changed live code must be refused before source hashing")

    monkeypatch.setattr(owner, "_code_bytes", refuse_source_reads)
    original = owner._live()
    monkeypatch.setattr(owner, "_LIVE", original)
    if change == "helper_code":

        def changed(value, maximum=owner.MAX_PAYLOAD_BYTES):
            return b"changed"

        monkeypatch.setattr(owner._frame_cell_encode, "__code__", changed.__code__)
    elif change == "helper_defaults":
        monkeypatch.setattr(
            owner._frame_cell_encode,
            "__defaults__",
            (owner.MAX_PAYLOAD_BYTES - 1,),
        )
    elif change == "bound":
        monkeypatch.setattr(
            owner, "_FRAME_FAST_STRING_CHARS", owner._FRAME_FAST_STRING_CHARS - 1
        )
    else:
        monkeypatch.setattr(owner.json.encoder, "encode_basestring", object())
    assert owner._live() != original
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="^PRODUCER_CHANGED$"
    ):
        owner._producer()
    assert reads == []


def _frame():
    index = pd.Index([101, 103, 107], dtype="int64", name="invented_row")
    person = pd.DataFrame(
        {
            "person_id": np.array([1, 2, 3], dtype=np.int64),
            "native_integer": np.array([-9, 0, 11], dtype=np.int32),
            "native_unsigned": np.array([0, 2**63, 2**64 - 1], dtype=np.uint64),
            "native_boolean": np.array([True, False, True], dtype=np.bool_),
            "native_float": np.array([-0.0, np.nan, 1.25], dtype=np.float64),
            "nullable_integer": pd.array([11, pd.NA, -7], dtype="Int64"),
            "nullable_boolean": pd.array([True, pd.NA, False], dtype="boolean"),
            "nullable_float": pd.array([-0.0, pd.NA, 1.25], dtype="Float64"),
            "text": pd.array(
                ["", "é\x00🧪", pd.NA], dtype=pd.StringDtype(storage="python")
            ),
            "object_value": np.array([None, True, "invented\nlabel"], dtype=object),
        },
        index=index,
    )
    tables = {US_SCHEMA.person_entity: person}
    for ordinal, group in enumerate(US_SCHEMA.group_entities, 1):
        first, second = 10 * ordinal, 10 * ordinal + 1
        person[US_SCHEMA.membership_column(group)] = np.array(
            [first, first, second], dtype=np.int64
        )
        tables[group] = pd.DataFrame(
            {US_SCHEMA.entity_id_column(group): [first, second]}
        )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([-0.0, 2.5]), WeightKind.DESIGN)},
        pd.Series(
            ["invented_a", "invented_a", "invented_b"],
            index=index,
            name="stratum",
            dtype=object,
        ),
        mass_log=(MassChangeRecord("household", 1.25, 2.5, 2.0, "invented"),),
        metadata={"us_spine_assembly_manifest": {"invented_marker": "initial"}},
    )


def _rebuild(frame, **changes):
    arguments = {
        "tables": {entity: frame.table(entity) for entity in frame.entities},
        "schema": frame.schema,
        "weights": {
            entity: frame.weights_for(entity) for entity in frame.weighted_entities
        },
        "strata": frame.strata,
        "mass_log": frame.mass_log,
        "metadata": frame.metadata,
    }
    arguments.update(changes)
    return Frame(**arguments)


def _legacy_frame_preimage(frame):
    """Explicit predecessor traversal, including every newline and raw weight byte."""
    owner._require(
        isinstance(frame, Frame) and frame.schema == US_SCHEMA and not frame.links,
        "FRAME_TYPE",
    )
    chunks = []

    def append(value):
        chunks.extend((owner._encode(value), b"\n"))

    append([list(frame.entities), list(frame.weighted_entities)])
    for entity in frame.entities:
        table = frame.table(entity)
        owner._require(
            type(table) is pd.DataFrame and not table.columns.has_duplicates,
            "FRAME_TABLE",
        )
        append(
            [
                entity,
                list(table.columns),
                type(table.columns).__name__,
                str(table.columns.dtype),
                list(table.columns.names),
                type(table.index).__name__,
                str(table.index.dtype),
                list(table.index.names),
            ]
        )
        for value in table.index:
            append(owner._cell(value))
        for column in table:
            series = table[column]
            dtype = series.dtype
            append(
                [
                    column,
                    str(dtype),
                    getattr(dtype, "storage", None),
                    str(getattr(dtype, "na_value", "")),
                ]
            )
            for value in series:
                append(owner._cell(value))
    append(
        [
            str(frame.strata.dtype),
            type(frame.strata.index).__name__,
            str(frame.strata.index.dtype),
            list(frame.strata.index.names),
            getattr(frame.strata.dtype, "storage", None),
            frame.strata.name,
        ]
    )
    for value in frame.strata.index:
        append(owner._cell(value))
    for value in frame.strata:
        append(owner._cell(value))
    for entity in frame.weighted_entities:
        weights = frame.weights_for(entity)
        append(
            [
                entity,
                weights.kind.value,
                str(weights.values.dtype),
                list(weights.values.shape),
            ]
        )
        chunks.append(weights.values.tobytes())
    chunks.append(owner.graph_context.encode_us_frame_context(frame))
    return b"".join(chunks)


def _assert_exact_frame_preimage(frame, monkeypatch):
    expected = _legacy_frame_preimage(frame)
    original_sha256 = hashlib.sha256
    expected_digest = original_sha256(expected).hexdigest()
    captures = []

    class RecordingSHA:
        def __init__(self, *args, **kwargs):
            self.digest = original_sha256(*args, **kwargs)
            self.parts = [args[0]] if args else []
            captures.append(self)

        def update(self, data):
            self.parts.append(bytes(data))
            self.digest.update(data)

        def hexdigest(self):
            return self.digest.hexdigest()

    # graph_context also calls hashlib; its separate seeded hashers are recorded
    # independently. The frame hasher is always the first, unseeded instance.
    with monkeypatch.context() as local:
        local.setattr(owner.hashlib, "sha256", RecordingSHA)
        actual = owner._frame_identity(frame)
    assert actual == expected_digest
    assert b"".join(captures[0].parts) == expected


def test_complete_frame_exact_preimage(monkeypatch):
    _assert_exact_frame_preimage(_frame(), monkeypatch)


def test_empty_table_and_strata_preimage_after_mutation(monkeypatch):
    frame = _frame()
    original = owner._frame_identity(frame)
    # Frame construction requires nonempty weights. This deliberately corrupts
    # an invented borrow to test the identity traversal's empty-loop semantics;
    # it makes no claim that the mutated object passes Frame.revalidate().
    for entity in frame.entities:
        frame._tables[entity] = frame.table(entity).iloc[:0].copy()
    frame._strata = frame.strata.iloc[:0].copy()
    assert owner._frame_identity(frame) != original
    _assert_exact_frame_preimage(frame, monkeypatch)


def test_multiple_weighted_entities_and_weight_order(monkeypatch):
    frame = _frame()
    weights = {
        "person": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.DESIGN),
        "household": frame.weights_for("household"),
    }
    multiple = _rebuild(frame, weights=weights)
    _assert_exact_frame_preimage(multiple, monkeypatch)
    assert owner._frame_identity(multiple) != owner._frame_identity(frame)
    reversed_weights = _rebuild(frame, weights=dict(reversed(list(weights.items()))))
    _assert_exact_frame_preimage(reversed_weights, monkeypatch)
    assert multiple.weighted_entities == reversed_weights.weighted_entities
    assert owner._frame_identity(reversed_weights) == owner._frame_identity(multiple)


@pytest.mark.parametrize(
    "dtype, values",
    [
        ("int8", [-128, 0, 127]),
        ("int16", [-32768, 0, 32767]),
        ("int32", [-(2**31), 0, 2**31 - 1]),
        ("int64", [-(2**63), 0, 2**63 - 1]),
        ("uint8", [0, 1, 255]),
        ("uint16", [0, 1, 65535]),
        ("uint32", [0, 1, 2**32 - 1]),
        ("uint64", [0, 2**63, 2**64 - 1]),
        ("float16", [-0.0, np.nan, 1.25]),
        ("float32", [-0.0, np.nan, 1.25]),
        ("float64", [-0.0, np.nan, 1.25]),
        ("Int8", [-128, pd.NA, 127]),
        ("Int16", [-32768, pd.NA, 32767]),
        ("Int32", [-(2**31), pd.NA, 2**31 - 1]),
        ("Int64", [-(2**63), pd.NA, 2**63 - 1]),
        ("UInt8", [0, pd.NA, 255]),
        ("UInt16", [0, pd.NA, 65535]),
        ("UInt32", [0, pd.NA, 2**32 - 1]),
        ("UInt64", [0, pd.NA, 2**64 - 1]),
        ("Float32", [-0.0, pd.NA, 1.25]),
        ("Float64", [-0.0, pd.NA, 1.25]),
        ("boolean", [True, pd.NA, False]),
        ("category", ["invented", None, "é"]),
        ("object", [2**80, "x" * 4097, -0.0]),
    ],
)
def test_representative_column_dtypes_exact_preimage(dtype, values, monkeypatch):
    frame = _frame()
    frame.person["invented_variant"] = pd.array(values, dtype=dtype)
    _assert_exact_frame_preimage(frame, monkeypatch)


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_string_storage_descriptor_and_values_are_preserved(storage, monkeypatch):
    if storage == "pyarrow":
        pytest.importorskip("pyarrow")
    frame = _frame()
    frame.person["text"] = pd.array(
        ["a", pd.NA, "é\x00🧪"], dtype=pd.StringDtype(storage=storage)
    )
    _assert_exact_frame_preimage(frame, monkeypatch)


@pytest.mark.parametrize(
    "index",
    [
        pd.RangeIndex(3, name="range"),
        pd.Index([9, 7, 5], dtype="int32", name="integer"),
        pd.Index([0, 2**63, 2**64 - 1], dtype="uint64", name="unsigned"),
        pd.Index([-0.0, np.nan, 1.25], dtype="float64", name="float"),
        pd.Index(["", "é", "\x00🧪"], dtype=object, name="text"),
        pd.Index([11, pd.NA, 13], dtype="Int64", name="nullable"),
        pd.CategoricalIndex(["a", "b", "a"], name="category"),
    ],
)
def test_person_and_strata_index_preimages(index, monkeypatch):
    frame = _frame()
    # Mutate both borrowed objects deliberately; the identity owner does not
    # normalize index order, index type, missing labels, or duplicate labels.
    frame.person.index = index.copy()
    frame.strata.index = index.copy()
    _assert_exact_frame_preimage(frame, monkeypatch)


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "dtype",
        "index_value",
        "index_name",
        "column_order",
        "column_name",
        "column_index_dtype",
        "strata_value",
        "strata_name",
        "strata_index",
        "strata_dtype",
    ],
)
def test_each_repeated_call_rechecks_mutable_frame(change):
    frame = _frame()
    original = owner._frame_identity(frame)
    assert owner._frame_identity(frame) == original
    if change == "value":
        frame.person.loc[101, "native_integer"] = 17
    elif change == "dtype":
        frame.person["native_integer"] = frame.person["native_integer"].astype("int64")
    elif change == "index_value":
        frame.person.index = pd.Index([109, 103, 107], name="invented_row")
    elif change == "index_name":
        frame.person.index.name = "changed"
    elif change == "column_order":
        frame.person.insert(0, "moved", frame.person.pop("native_integer"))
        frame.person.rename(columns={"moved": "native_integer"}, inplace=True)
    elif change == "column_name":
        frame.person.columns.name = "changed"
    elif change == "column_index_dtype":
        frame.person.columns = pd.Index(
            frame.person.columns, dtype=pd.StringDtype(storage="python")
        )
    elif change == "strata_value":
        frame.strata.iloc[0] = "changed"
    elif change == "strata_name":
        frame.strata.name = "changed"
    elif change == "strata_index":
        frame.strata.index = pd.Index([109, 103, 107], name="invented_row")
    else:
        frame._strata = frame.strata.astype(pd.StringDtype(storage="python"))
    changed = owner._frame_identity(frame)
    assert changed != original
    assert changed == hashlib.sha256(_legacy_frame_preimage(frame)).hexdigest()
    assert owner._frame_identity(frame) == changed


def test_same_object_mutate_restore_mutate_has_no_identity_cache():
    frame = _frame()
    original = owner._frame_identity(frame)
    for value in (17, 19, 23):
        frame.person.loc[101, "native_integer"] = value
        assert owner._frame_identity(frame) != original
        frame.person.loc[101, "native_integer"] = -9
        assert owner._frame_identity(frame) == original


def test_raw_weight_signed_zero_kind_metadata_and_mass_context(monkeypatch):
    frame = _frame()
    original = owner._frame_identity(frame)
    variants = [
        _rebuild(
            frame,
            weights={"household": Weights(np.array([0.0, 2.5]), WeightKind.DESIGN)},
        ),
        _rebuild(
            frame,
            weights={"household": Weights(np.array([-0.0, 2.75]), WeightKind.DESIGN)},
        ),
        _rebuild(
            frame,
            weights={
                "household": Weights(np.array([-0.0, 2.5]), WeightKind.IMPORTANCE)
            },
        ),
        _rebuild(
            frame,
            metadata={"us_spine_assembly_manifest": {"invented_marker": "changed"}},
        ),
        _rebuild(
            frame, mass_log=(MassChangeRecord("household", 1.25, 2.5, 2.0, "changed"),)
        ),
    ]
    for changed in variants:
        assert owner._frame_identity(changed) != original
        _assert_exact_frame_preimage(changed, monkeypatch)
    values = frame.weights_for("household").values
    values.setflags(write=True)
    try:
        values[1] = 3.0
    finally:
        values.setflags(write=False)
    assert owner._frame_identity(frame) != original


@pytest.mark.parametrize("place", ["index", "column", "strata_index", "strata"])
def test_repeated_identity_refuses_new_unsupported_cell(place):
    frame = _frame()
    owner._frame_identity(frame)
    if place == "index":
        frame.person.index = pd.Index([b"bad", 103, 107], dtype=object)
    elif place == "column":
        frame.person["object_value"] = [b"bad", "fine", None]
    elif place == "strata_index":
        frame.strata.index = pd.Index([b"bad", 103, 107], dtype=object)
    else:
        frame.strata.iloc[0] = b"bad"
    expected = _outcome(lambda: _legacy_frame_preimage(frame))
    assert expected == (
        "error",
        owner.SurveyPopulationPreparationError,
        "FRAME_CELL_TYPE",
    )
    assert _outcome(lambda: owner._frame_identity(frame)) == expected


def test_table_and_traversal_error_precedence():
    frame = _frame()
    frame.person["bad_later"] = [float("inf"), 1.0, 2.0]
    frame.person.index = pd.Index([b"bad_first", 103, 107], dtype=object)
    expected = _outcome(lambda: _legacy_frame_preimage(frame))
    assert expected == (
        "error",
        owner.SurveyPopulationPreparationError,
        "FRAME_CELL_TYPE",
    )
    assert _outcome(lambda: owner._frame_identity(frame)) == expected
    frame.person.columns = ["duplicate"] * len(frame.person.columns)
    expected = _outcome(lambda: _legacy_frame_preimage(frame))
    assert expected == ("error", owner.SurveyPopulationPreparationError, "FRAME_TABLE")
    assert _outcome(lambda: owner._frame_identity(frame)) == expected
    with pytest.raises(owner.SurveyPopulationPreparationError, match="^FRAME_TYPE$"):
        owner._frame_identity(object())


@pytest.mark.parametrize("early", ["surrogate", "scalar_bound"])
@pytest.mark.parametrize("later", ["unsupported", "nonfinite"])
def test_earlier_series_encoding_error_precedes_later_cell_error(early, later):
    frame = _frame()
    first = "\ud800" if early == "surrogate" else "x" * (owner._MAX_SCALAR_BYTES + 1)
    second = b"unsupported" if later == "unsupported" else float("inf")
    frame.person["object_value"] = pd.Series(
        [first, second, None], index=frame.person.index, dtype=object
    )
    expected = _outcome(lambda: _legacy_frame_preimage(frame))
    if early == "surrogate":
        assert expected[0:2] == ("error", UnicodeEncodeError)
    else:
        assert expected == (
            "error",
            owner.SurveyPopulationPreparationError,
            "SCALAR_LIMIT",
        )
    assert _outcome(lambda: owner._frame_identity(frame)) == expected


@pytest.mark.parametrize(
    "change, code",
    [
        ("schema", "FRAME_TYPE"),
        ("links", "FRAME_TYPE"),
        ("table_subclass", "FRAME_TABLE"),
    ],
)
def test_frame_structure_refusals_still_precede_cells(change, code):
    frame = _frame()
    frame.person["bad_later"] = [float("inf"), 1.0, 2.0]
    if change == "schema":
        frame._schema = object()
    elif change == "links":

        class LinkedFrame(Frame):
            @property
            def links(self):
                return ("invented_link",)

        frame = LinkedFrame(frame._tables, frame.schema, frame._weights, frame.strata)
    else:

        class Table(pd.DataFrame):
            pass

        frame._tables[US_SCHEMA.person_entity] = Table(frame.person)
    expected = ("error", owner.SurveyPopulationPreparationError, code)
    assert _outcome(lambda: _legacy_frame_preimage(frame)) == expected
    assert _outcome(lambda: owner._frame_identity(frame)) == expected


def _invented_preparation(frame, monkeypatch):
    """Install an in-memory synthetic issuance solely for boundary unit tests."""
    preparation = owner.AuthenticatedSurveyPopulationPreparation(
        b"{}", _token=owner._TOKEN
    )
    state = SimpleNamespace(frame=frame, identity=owner._frame_identity(frame))
    entry = (weakref.ref(preparation), preparation.payload, state)
    monkeypatch.setitem(owner._ISSUED, id(preparation), entry)
    return preparation, state, entry


def test_checked_boundary_revalidates_each_borrow(monkeypatch):
    frame = _frame()
    preparation, state, _entry = _invented_preparation(frame, monkeypatch)
    calls = []

    def validate(candidate):
        assert candidate is state
        calls.append(True)
        owner._require(
            owner._frame_identity(frame) == state.identity, "PREPARED_FRAME_CHANGED"
        )

    monkeypatch.setattr(owner, "_validate", validate)
    assert preparation.frame is frame
    assert preparation.frame is frame
    assert calls == [True, True]
    frame.person.loc[101, "native_integer"] = 17
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="^PREPARED_FRAME_CHANGED$"
    ):
        _ = preparation.frame
    assert calls == [True, True, True]


@pytest.mark.parametrize("change", ["entry", "payload"])
def test_checked_boundary_post_validation_issuance_guard(change, monkeypatch):
    preparation, _state, entry = _invented_preparation(_frame(), monkeypatch)

    def validate(_candidate):
        if change == "entry":
            # Equal tuple contents do not satisfy the required same-entry guard.
            monkeypatch.setitem(owner._ISSUED, id(preparation), tuple(list(entry)))
        else:
            object.__setattr__(preparation, "payload", b'{"changed":true}')

    monkeypatch.setattr(owner, "_validate", validate)
    with pytest.raises(
        owner.SurveyPopulationPreparationError, match="^UNISSUED_OR_CHANGED$"
    ):
        preparation._checked()


@pytest.mark.parametrize("mutation", ["none", "frame", "payload", "frame_and_payload"])
def test_materialized_frame_two_pass_recheck_and_toctou(mutation, monkeypatch):
    frame = _frame()
    preparation, state, entry = _invented_preparation(frame, monkeypatch)
    calls = []
    original_identity = owner._frame_identity

    def checked(candidate):
        assert candidate is preparation
        calls.append("checked")
        return entry

    def identity(candidate):
        assert candidate is frame
        calls.append("identity")
        return original_identity(candidate)

    def validate(candidate):
        assert candidate is state
        calls.append("validate")
        if mutation in ("frame", "frame_and_payload"):
            frame.person.loc[101, "native_integer"] = 17
        if mutation in ("payload", "frame_and_payload"):
            object.__setattr__(preparation, "payload", b'{"changed":true}')

    # Isolate this API's ordered rechecks from acquisition/ancestry validation.
    monkeypatch.setattr(
        owner.AuthenticatedSurveyPopulationPreparation, "_checked", checked
    )
    monkeypatch.setattr(owner, "_frame_identity", identity)
    monkeypatch.setattr(owner, "_validate", validate)
    if mutation != "none":
        code = (
            "UNISSUED_OR_CHANGED"
            if mutation == "payload"
            else "MATERIALIZED_FRAME_CHANGED"
        )
        with pytest.raises(owner.SurveyPopulationPreparationError, match=f"^{code}$"):
            owner.verify_materialized_survey_population(preparation, frame)
    else:
        owner.verify_materialized_survey_population(preparation, frame)
    assert calls == ["checked", "identity", "validate", "identity"]
