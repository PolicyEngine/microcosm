"""Exact frame memo encodings on invented values; no source archives are read."""

import json
import struct

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import source_memo as memo


def _float(bits):
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def test_object_columns_preserve_scalar_types_missing_markers_and_float_bits():
    values = [
        "00123",
        "é\U0001f642",
        None,
        pd.NA,
        True,
        1,
        2**80,
        1.0,
        -0.0,
        _float("7ff8000000000001"),
        _float("fff8000000000002"),
        b"\x00\xff",
    ]
    original = pd.DataFrame({"mixed": pd.Series(values, dtype=object)})
    original.columns = pd.Index(["mixed"], dtype=object)
    original.index = pd.RangeIndex(3, 3 + 2 * len(values), 2, name="row")
    decoded = memo.decode_frame(memo.encode_frame(original))
    pd.testing.assert_frame_equal(original, decoded)
    assert decoded.columns.dtype == object
    assert decoded["mixed"].dtype == object
    for expected, actual in zip(values, decoded["mixed"], strict=True):
        assert type(actual) is type(expected)
        if type(expected) is float:
            assert struct.pack(">d", actual) == struct.pack(">d", expected)
        elif expected is pd.NA:
            assert actual is pd.NA
        else:
            assert actual == expected
    assert memo.frames_identical(original, decoded)


@pytest.mark.parametrize("dtype", ["bool", "int8", "uint64", "float32", "float64"])
def test_numpy_column_round_trip_preserves_raw_bits(dtype):
    if dtype.startswith("float"):
        integer_dtype = "uint32" if dtype == "float32" else "uint64"
        patterns = (
            [0x80000000, 0x7FC00001, 0xFFC00002]
            if dtype == "float32"
            else [0x8000000000000000, 0x7FF8000000000001, 0xFFF8000000000002]
        )
        values = np.array(patterns, dtype=integer_dtype).view(dtype)
    else:
        values = np.array([0, 1, 1], dtype=dtype)
    original = pd.DataFrame({"value": values})
    decoded = memo.decode_frame(memo.encode_frame(original))
    assert decoded["value"].dtype == original["value"].dtype
    assert decoded["value"].to_numpy().tobytes() == values.tobytes()
    assert memo.frames_identical(original, decoded)


@pytest.mark.parametrize("storage", ["python", "pyarrow"])
@pytest.mark.parametrize("missing", [pd.NA, np.nan])
def test_string_column_and_label_storage_survive(storage, missing):
    dtype = pd.StringDtype(storage=storage, na_value=missing)
    original = pd.DataFrame({"text": pd.Series(["001", None, "é"], dtype=dtype)})
    original.columns = pd.Index(["text"], dtype=dtype)
    decoded = memo.decode_frame(memo.encode_frame(original))
    pd.testing.assert_frame_equal(original, decoded)
    assert decoded["text"].dtype.storage == storage
    assert decoded.columns.dtype.storage == storage
    assert memo.frames_identical(original, decoded)


@pytest.mark.parametrize("replacement", [None, pd.NA, True, 1, 0.0, -0.0])
def test_object_identity_distinguishes_scalars(replacement):
    original = pd.DataFrame({"value": pd.Series([1.0], dtype=object)})
    changed = pd.DataFrame({"value": pd.Series([replacement], dtype=object)})
    assert not memo.frames_identical(original, changed)


def test_object_identity_distinguishes_nan_payloads():
    left = pd.DataFrame(
        {"value": pd.Series([_float("7ff8000000000001")], dtype=object)}
    )
    right = pd.DataFrame(
        {"value": pd.Series([_float("7ff8000000000002")], dtype=object)}
    )
    assert not memo.frames_identical(left, right)


def test_object_identity_distinguishes_bool_from_int():
    left = pd.DataFrame({"value": pd.Series([True], dtype=object)})
    right = pd.DataFrame({"value": pd.Series([1], dtype=object)})
    assert not memo.frames_identical(left, right)


@pytest.mark.parametrize("value", [object(), ["nested"], {"a": 1}, np.int64(2)])
def test_arbitrary_object_cells_are_refused(value):
    frame = pd.DataFrame({"value": pd.Series([value], dtype=object)})
    with pytest.raises(memo.SourceMemoError, match="^FRAME_OBJECT$"):
        memo.encode_frame(frame)
    assert not memo.frames_identical(frame, frame)


@pytest.mark.parametrize(
    "frame",
    [
        pd.DataFrame({"value": pd.Series([1, None], dtype="Int64")}),
        pd.DataFrame({"value": pd.Categorical(["a"])}),
        pd.DataFrame({"value": pd.to_datetime(["2026-01-01"])}),
        pd.DataFrame([[1, 2]], columns=["a", "a"]),
        pd.DataFrame({"value": [1]}, index=pd.Index(["a"])),
    ],
)
def test_unsupported_frame_shapes_are_refused(frame):
    with pytest.raises(memo.SourceMemoError):
        memo.encode_frame(frame)


def test_numpy_dtype_metadata_is_refused_instead_of_silently_lost():
    dtype = np.dtype("float64", metadata={"unit": "invented"})
    frame = pd.DataFrame({"value": np.array([1.0], dtype=dtype)})
    assert frame["value"].dtype.metadata == {"unit": "invented"}
    with pytest.raises(memo.SourceMemoError, match="^FRAME_DTYPE$"):
        memo.encode_frame(frame)
    assert not memo.frames_identical(frame, frame)


def test_empty_frames_and_consecutive_frames_preserve_indexes():
    frames = [
        pd.DataFrame(
            index=pd.RangeIndex(4, 4, name="empty"),
            columns=pd.Index([], dtype=object),
        ),
        pd.DataFrame({"text": pd.Series([], dtype=object)}),
        pd.DataFrame({"value": [1, 2]}),
    ]
    blobs = [blob for frame in frames for blob in memo.encode_frame(frame)]
    decoded = memo.decode_frames(blobs)
    assert len(decoded) == len(frames)
    for original, recalled in zip(frames, decoded, strict=True):
        pd.testing.assert_frame_equal(original, recalled)
        assert memo.frames_identical(original, recalled)


def test_object_decoder_refuses_containers_and_incorrect_row_counts():
    blobs = memo.encode_frame(pd.DataFrame({"value": pd.Series(["a"], dtype=object)}))
    with pytest.raises(memo.SourceMemoError, match="^FRAME_OBJECT$"):
        memo.decode_frame([blobs[0], b'[["L","nested"]]'])
    with pytest.raises(memo.SourceMemoError, match="^FRAME_ROWS$"):
        memo.decode_frame([blobs[0], b"[]"])
    header = json.loads(blobs[0])
    header["columns"][0][2] = "unexpected"
    with pytest.raises(memo.SourceMemoError, match="^FRAME_DTYPE$"):
        memo.decode_frame([memo.ordered(header), blobs[1]])


def test_typed_metadata_preserves_container_order_and_ieee_float_bits():
    original = {
        "z": ("001", None, True, 2**80, b"\xff"),
        "a": [-0.0, _float("7ff8000000000001"), _float("fff8000000000002")],
    }
    decoded = memo.decode_typed(memo.encode_typed(original))
    assert list(decoded) == ["z", "a"]
    assert type(decoded["z"]) is tuple
    assert decoded["z"] == original["z"]
    assert type(decoded["a"]) is list
    for expected, actual in zip(original["a"], decoded["a"], strict=True):
        assert struct.pack(">d", actual) == struct.pack(">d", expected)
