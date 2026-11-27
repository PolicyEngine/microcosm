"""Text columns in frame checkpoints, against today's per-value loop.

A checkpoint stores text with the ``object_scalars_v1`` encoding: for each
value, one tag byte (7 for a string) followed by the value's UTF-8 bytes, and
one int64 offset per value. ``_encode_object_values`` and
``_decode_object_values`` convert one value at a time. They stay as the
fallback for a column with a missing value or a value that is not a string,
and these tests call them to produce the expected bytes and values.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import microcosm.build.frame_checkpoint as frame_checkpoint_module
from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

h5py = pytest.importorskip("h5py")

PYARROW_STR = pd.StringDtype(storage="pyarrow", na_value=np.nan)
PYARROW_STRING = pd.StringDtype(storage="pyarrow", na_value=pd.NA)
PYTHON_STR = pd.StringDtype(storage="python", na_value=np.nan)
PYTHON_STRING = pd.StringDtype(storage="python", na_value=pd.NA)
TEXT_DTYPES: dict[str, pd.StringDtype | str] = {
    "pyarrow_str": PYARROW_STR,
    "pyarrow_string": PYARROW_STRING,
    "python_str": PYTHON_STR,
    "python_string": PYTHON_STRING,
    "object": "object",
}

# Today's per-value loop, held before any test replaces the module attributes.
PER_VALUE_ENCODE = frame_checkpoint_module._encode_object_values
PER_VALUE_DECODE = frame_checkpoint_module._decode_object_values

_PERSON_TEXT = ["alpha", "", "é中😀", "alpha", "x" * 70_000]


def _text_frame(
    *, person_index_dtype: pd.StringDtype | str, strata_dtype: pd.StringDtype | str
) -> Frame:
    """A frame with text in every place a checkpoint stores it, none missing.

    Persons are keyed by string index values of ``person_index_dtype``, and
    households by an object index of strings. The person table holds one
    text column per string dtype plus an object column of strings, and the
    strata are strings of ``strata_dtype``.
    """

    person_index = pd.Index(
        ["p-0", "p-1", "p-2", "p-3", "p-4"],
        dtype=person_index_dtype,
        name="person_key",
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(5, dtype=np.int64),
            "person_household_id": np.asarray([10, 10, 20, 20, 20], dtype=np.int64),
            **{
                f"label_{dtype_name}": pd.Series(
                    _PERSON_TEXT, index=person_index, dtype=dtype
                )
                for dtype_name, dtype in TEXT_DTYPES.items()
            },
        },
        index=person_index,
    )
    household_index = pd.Index(["h-10", "h-20"], dtype=object, name="household_key")
    household = pd.DataFrame(
        {
            "household_id": np.asarray([10, 20], dtype=np.int64),
            "tenure_type": pd.Series(
                ["OWNED", "RENTED"], index=household_index, dtype=PYARROW_STR
            ),
        },
        index=household_index,
    )
    strata = pd.Series(
        ["asec", "asec", "puf", "puf", "acs"],
        index=person_index,
        name="stratum",
        dtype=strata_dtype,
    )
    return Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray([1.0, 2.0]), WeightKind.DESIGN)},
        strata,
    )


_FRAME_CASES = {
    "pyarrow str index, object strata": {
        "person_index_dtype": PYARROW_STR,
        "strata_dtype": "object",
    },
    "python string index, pyarrow str strata": {
        "person_index_dtype": PYTHON_STRING,
        "strata_dtype": PYARROW_STR,
    },
    "object index, python str strata": {
        "person_index_dtype": "object",
        "strata_dtype": PYTHON_STR,
    },
}


@pytest.fixture(params=list(_FRAME_CASES), ids=list(_FRAME_CASES))
def text_frame(request: pytest.FixtureRequest) -> Frame:
    return _text_frame(**_FRAME_CASES[request.param])


def _refuse_per_value_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_arguments: object, **_keywords: object) -> None:
        raise AssertionError("A text column went through the per-value loop.")

    monkeypatch.setattr(frame_checkpoint_module, "_encode_object_values", refuse)
    monkeypatch.setattr(frame_checkpoint_module, "_decode_object_values", refuse)


def _use_only_per_value_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send every text column through today's loop, as before the change."""

    def no_conversion(*_arguments: object, **_keywords: object) -> None:
        return None

    monkeypatch.setattr(frame_checkpoint_module, "_encode_string_values", no_conversion)
    monkeypatch.setattr(frame_checkpoint_module, "_decode_string_values", no_conversion)


class _LoopCalls:
    """Counts the values each per-value loop converts, then runs that loop."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.encoded: list[int] = []
        self.decoded: list[int] = []

        def counting_encode(values: np.ndarray) -> tuple[np.ndarray, bytes]:
            self.encoded.append(len(values))
            return PER_VALUE_ENCODE(values)

        def counting_decode(
            offsets: np.ndarray, payload: bytes, *, path: Path, label: str
        ) -> np.ndarray:
            self.decoded.append(len(offsets) - 1)
            return PER_VALUE_DECODE(offsets, payload, path=path, label=label)

        monkeypatch.setattr(
            frame_checkpoint_module, "_encode_object_values", counting_encode
        )
        monkeypatch.setattr(
            frame_checkpoint_module, "_decode_object_values", counting_decode
        )


def _metadata(path: Path) -> dict[str, object]:
    with h5py.File(path, mode="r") as h5:
        raw = np.asarray(h5["_populace_frame_checkpoint/metadata_json"]).tobytes()
    return json.loads(raw.decode("utf-8"))


def _object_encoded_series(
    frame: Frame, metadata: dict[str, object]
) -> Iterator[tuple[str, pd.Series]]:
    """Each object-encoded series of ``frame`` with its HDF5 group path."""

    tables = dict(frame_checkpoint_module._frame_tables(frame))
    for table_position, table_spec in enumerate(metadata["tables"]):
        table = tables[table_spec["name"]]
        table_path = f"_populace_frame_checkpoint/tables/t{table_position:05d}"
        index_spec = table_spec["index"]
        if index_spec.get("encoding") == "object_scalars_v1":
            yield f"{table_path}/index", pd.Series(table.index.array, copy=False)
        for column_position, column_spec in enumerate(table_spec["columns"]):
            if column_spec["encoding"] == "object_scalars_v1":
                yield (
                    f"{table_path}/columns/c{column_position:05d}",
                    table[column_spec["name"]],
                )
    if metadata["strata"]["encoding"] == "object_scalars_v1":
        yield "_populace_frame_checkpoint/strata", frame.strata


def _assert_same_text_storage(
    actual: pd.Series | pd.Index, expected: pd.Series | pd.Index, label: str
) -> None:
    """Same dtype, string storage, NA marker, Arrow layout and value types."""

    assert actual.dtype == expected.dtype, label
    if isinstance(expected.dtype, pd.StringDtype):
        assert actual.dtype.storage == expected.dtype.storage, label
        assert (actual.dtype.na_value is pd.NA) == (expected.dtype.na_value is pd.NA), (
            label
        )
    if isinstance(expected.dtype, pd.StringDtype) and (
        expected.dtype.storage == "pyarrow"
    ):
        actual_chunks = actual.array.__arrow_array__()
        expected_chunks = expected.array.__arrow_array__()
        assert actual_chunks.type == expected_chunks.type, label
        assert actual_chunks.num_chunks == expected_chunks.num_chunks, label
    elif expected.dtype == object or isinstance(expected.dtype, pd.StringDtype):
        actual_types = [type(value) for value in actual.to_numpy(dtype=object)]
        expected_types = [type(value) for value in expected.to_numpy(dtype=object)]
        assert actual_types == expected_types, label


def _assert_same_frame(actual: Frame, expected: Frame) -> None:
    actual_tables = dict(frame_checkpoint_module._frame_tables(actual))
    expected_tables = dict(frame_checkpoint_module._frame_tables(expected))
    assert list(actual_tables) == list(expected_tables)
    for name, expected_table in expected_tables.items():
        actual_table = actual_tables[name]
        pd.testing.assert_frame_equal(
            actual_table,
            expected_table,
            check_dtype=True,
            check_index_type=True,
            check_column_type=True,
            check_exact=True,
        )
        _assert_same_text_storage(
            actual_table.index, expected_table.index, f"{name}.index"
        )
        for column in expected_table.columns:
            _assert_same_text_storage(
                actual_table[column], expected_table[column], f"{name}.{column}"
            )
    pd.testing.assert_series_equal(
        actual.strata,
        expected.strata,
        check_dtype=True,
        check_index_type=True,
        check_exact=True,
    )
    _assert_same_text_storage(actual.strata, expected.strata, "strata")


def test_text_columns_are_written_and_read_without_the_per_value_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text_frame: Frame
) -> None:
    path = tmp_path / "text.h5"
    _refuse_per_value_loop(monkeypatch)

    write_frame_checkpoint(path, text_frame)
    load_frame_checkpoint(path)


def test_text_columns_write_the_bytes_of_the_per_value_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text_frame: Frame
) -> None:
    converted_path = tmp_path / "converted.h5"
    loop_path = tmp_path / "loop.h5"

    write_frame_checkpoint(converted_path, text_frame)
    with monkeypatch.context() as patch:
        _use_only_per_value_loop(patch)
        write_frame_checkpoint(loop_path, text_frame)

    assert converted_path.read_bytes() == loop_path.read_bytes()
    metadata = _metadata(converted_path)
    assert metadata["schema_version"] == 2
    compared: list[str] = []
    with h5py.File(converted_path, mode="r") as h5:
        for group_path, series in _object_encoded_series(text_frame, metadata):
            offsets, payload = PER_VALUE_ENCODE(
                series.to_numpy(dtype=object, copy=False)
            )
            stored_offsets = np.asarray(h5[f"{group_path}/offsets"])
            stored_payload = np.asarray(h5[f"{group_path}/payload"])
            assert stored_offsets.dtype == np.dtype(np.int64), group_path
            assert stored_payload.dtype == np.dtype(np.uint8), group_path
            np.testing.assert_array_equal(stored_offsets, offsets, err_msg=group_path)
            assert stored_payload.tobytes() == payload, group_path
            compared.append(group_path)
    # Five person text columns, the household text column, both indexes
    # and the strata.
    assert len(compared) == 9


def test_text_columns_load_the_frame_of_the_per_value_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, text_frame: Frame
) -> None:
    path = tmp_path / "text.h5"
    rewrite_path = tmp_path / "rewrite.h5"
    write_frame_checkpoint(path, text_frame)

    converted = load_frame_checkpoint(path).frame
    with monkeypatch.context() as patch:
        _use_only_per_value_loop(patch)
        by_loop = load_frame_checkpoint(path).frame

    _assert_same_frame(converted, by_loop)
    write_frame_checkpoint(rewrite_path, converted)
    assert rewrite_path.read_bytes() == path.read_bytes()


def _round_trip_series(
    path: Path, series: pd.Series
) -> tuple[np.ndarray, np.ndarray, pd.Series, dict[str, object]]:
    """Write ``series`` with the checkpoint writer, then read it back."""

    spec = frame_checkpoint_module._series_spec(series, label="column")
    with h5py.File(path, mode="w") as h5:
        group = h5.create_group("column", track_times=False)
        frame_checkpoint_module._write_series(group, series, spec)
    with h5py.File(path, mode="r") as h5:
        offsets = np.asarray(h5["column/offsets"])
        payload = np.asarray(h5["column/payload"])
        restored = frame_checkpoint_module._read_series(
            h5["column"], spec, path, label="column"
        )
    return offsets, payload, restored, spec


def _expected_by_loop(
    path: Path, series: pd.Series, spec: dict[str, object]
) -> tuple[np.ndarray, bytes, pd.Series]:
    """Today's offsets, payload and restored series for ``series``."""

    offsets, payload = PER_VALUE_ENCODE(series.to_numpy(dtype=object, copy=False))
    restore_dtype = frame_checkpoint_module._declared_dtype(
        spec, path=path, label="column"
    )
    values = PER_VALUE_DECODE(offsets, payload, path=path, label="column")
    return offsets, payload, pd.Series(values, dtype=restore_dtype, copy=False)


def _assert_matches_loop(path: Path, series: pd.Series) -> pd.Series:
    offsets, payload, restored, spec = _round_trip_series(path, series)
    expected_offsets, expected_payload, expected = _expected_by_loop(path, series, spec)
    assert offsets.dtype == np.dtype(np.int64)
    assert payload.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal(offsets, expected_offsets)
    assert payload.tobytes() == expected_payload
    pd.testing.assert_series_equal(restored, expected, check_exact=True)
    _assert_same_text_storage(restored, expected, "column")
    return restored


_EDGE_CASE_VALUES: dict[str, list[str]] = {
    "empty column": [],
    "empty strings": ["", "", ""],
    "non-ASCII UTF-8": ["é", "中文", "😀", "a\x00b", "ÿĀ߿ࠀ￿"],
    "one value throughout": ["same"] * 1_000,
    "very long string": ["short", "é" * 3_000_000, "tail"],
}


@pytest.mark.parametrize("dtype_name", list(TEXT_DTYPES))
@pytest.mark.parametrize("case", list(_EDGE_CASE_VALUES))
def test_text_edge_cases_keep_the_per_value_loop_bytes_and_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, dtype_name: str
) -> None:
    series = pd.Series(_EDGE_CASE_VALUES[case], dtype=TEXT_DTYPES[dtype_name])
    calls = _LoopCalls(monkeypatch)

    restored = _assert_matches_loop(tmp_path / "column.h5", series)

    assert calls.encoded == [] and calls.decoded == []
    assert restored.tolist() == _EDGE_CASE_VALUES[case]


def _concatenated_pyarrow_series() -> pd.Series:
    return pd.concat(
        [
            pd.Series(["first", "é"], dtype=PYARROW_STR),
            pd.Series(["", "third"], dtype=PYARROW_STR),
        ],
        ignore_index=True,
    )


def _sliced_chunks_pyarrow_series() -> pd.Series:
    return _concatenated_pyarrow_series().iloc[1:].reset_index(drop=True)


def _sliced_chunk_pyarrow_series() -> pd.Series:
    series = pd.Series(["dropped", "é", "", "kept"], dtype=PYARROW_STR)
    return series.iloc[1:].reset_index(drop=True)


def _narrow_offset_pyarrow_series() -> pd.Series:
    """Chunks given with 32-bit offsets; pandas stores them widened to 64-bit."""

    chunks = pa.chunked_array(
        [pa.array(["thirty", "two"], pa.string()), pa.array(["bit"], pa.string())]
    )
    return pd.Series(pd.arrays.ArrowStringArray(chunks), copy=False)


def _padded_offsets_pyarrow_series() -> pd.Series:
    """One chunk whose offsets buffer is not a whole number of int64 values.

    Arrow accepts an offsets buffer larger than its values need; here it has
    three spare bytes after the three offsets.
    """

    offsets = np.asarray([0, 1, 3], dtype=np.int64).tobytes() + b"\0\0\0"
    chunk = pa.LargeStringArray.from_buffers(
        2, pa.py_buffer(offsets), pa.py_buffer(b"abc")
    )
    chunk.validate(full=True)
    return pd.Series(pd.arrays.ArrowStringArray(pa.chunked_array([chunk])), copy=False)


# Each layout's builder, chunk count and the first chunk's starting offset.
_PYARROW_LAYOUTS: dict[str, tuple[Callable[[], pd.Series], int, int]] = {
    "two chunks": (_concatenated_pyarrow_series, 2, 0),
    "two chunks, the first starting at an offset": (
        _sliced_chunks_pyarrow_series,
        2,
        1,
    ),
    "one chunk starting at an offset": (_sliced_chunk_pyarrow_series, 1, 1),
    "chunks built with 32-bit offsets": (_narrow_offset_pyarrow_series, 2, 0),
    "offsets buffer with spare bytes": (_padded_offsets_pyarrow_series, 1, 0),
}


@pytest.mark.parametrize("layout", list(_PYARROW_LAYOUTS))
def test_pyarrow_text_layouts_keep_the_per_value_loop_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
) -> None:
    build_series, chunk_count, first_chunk_offset = _PYARROW_LAYOUTS[layout]
    series = build_series()
    chunks = series.array.__arrow_array__()
    assert chunks.num_chunks == chunk_count
    assert chunks.chunk(0).offset == first_chunk_offset
    calls = _LoopCalls(monkeypatch)

    _assert_matches_loop(tmp_path / "column.h5", series)

    assert calls.encoded == [] and calls.decoded == []


def test_numpy_strings_in_an_object_column_keep_the_per_value_loop_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    series = pd.Series(
        np.asarray([np.str_("numpy"), "python", np.str_("é")], dtype=object),
        dtype=object,
    )
    calls = _LoopCalls(monkeypatch)

    restored = _assert_matches_loop(tmp_path / "column.h5", series)

    assert calls.encoded == [] and calls.decoded == []
    assert [type(value) for value in restored] == [str, str, str]


class _EncodeOverridingStr(str):
    """A string whose ``encode`` returns other bytes than its UTF-8."""

    def encode(self, *_arguments: object, **_keywords: object) -> bytes:
        return b"OVERRIDDEN"


class _EqualToStrMeta(type):
    """Makes a class hash like ``str`` and compare equal to it."""

    def __eq__(cls, other: object) -> bool:
        return other is str or other is cls

    def __hash__(cls) -> int:
        return hash(str)


class _StrLookalike(str, metaclass=_EqualToStrMeta):
    def encode(self, *_arguments: object, **_keywords: object) -> bytes:
        return b"OVERRIDDEN"


class _UnhashableTypeMeta(type):
    """Makes a class that cannot be hashed, so it cannot go into a set."""

    __hash__ = None  # type: ignore[assignment]


class _UnhashableTypeStr(str, metaclass=_UnhashableTypeMeta):
    pass


class _UnhashableTypeFloat(float, metaclass=_UnhashableTypeMeta):
    pass


_STRING_SUBCLASSES = {
    "overrides encode": _EncodeOverridingStr,
    "type compares equal to str": _StrLookalike,
    "type cannot be hashed": _UnhashableTypeStr,
}


@pytest.mark.parametrize("dtype_name", ["python_str", "python_string", "object"])
@pytest.mark.parametrize("subclass", list(_STRING_SUBCLASSES))
def test_string_subclasses_go_through_the_per_value_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subclass: str, dtype_name: str
) -> None:
    """The per-value loop calls each value's own ``encode``; keep its bytes."""

    string_type = _STRING_SUBCLASSES[subclass]
    values = np.asarray(["plain", string_type("odd")], dtype=object)
    series = pd.Series(values, dtype=TEXT_DTYPES[dtype_name])
    assert type(series.to_numpy(dtype=object)[1]) is string_type
    calls = _LoopCalls(monkeypatch)

    _assert_matches_loop(tmp_path / "column.h5", series)

    assert calls.encoded == [2]


@pytest.mark.parametrize("dtype_name", ["python_str", "python_string", "object"])
def test_writing_python_strings_needs_no_more_memory_than_the_per_value_loop(
    dtype_name: str,
) -> None:
    """Peak traced memory of encoding short strings, against the loop's.

    The loop holds the offsets, a growing ``bytearray`` and its final
    ``bytes`` copy. Encoding every value at once would also hold one
    ``bytes`` object per value and ``bytes.join``'s buffer table, about six
    times the loop's peak for short strings.
    """

    values = np.asarray(
        ["".join(["v", str(position % 1_000).zfill(6)]) for position in range(300_000)],
        dtype=object,
    )
    series = pd.Series(values, dtype=TEXT_DTYPES[dtype_name], copy=False)
    stored_values = series.to_numpy(dtype=object, copy=False)

    def traced_peak(encode: Callable[[], object]) -> int:
        tracemalloc.start()
        try:
            encode()
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    loop_peak = traced_peak(lambda: PER_VALUE_ENCODE(stored_values))
    converted_peak = traced_peak(
        lambda: frame_checkpoint_module._encode_string_values(series)
    )

    assert converted_peak <= loop_peak


@pytest.mark.parametrize("dtype_name", ["python_str", "python_string", "object"])
def test_writing_python_strings_leaves_their_memory_size_unchanged(
    tmp_path: Path, dtype_name: str
) -> None:
    """Writing must not attach a cached UTF-8 copy to non-ASCII strings.

    CPython's ``PyUnicode_AsUTF8AndSize``, which Arrow uses to read a
    ``str``, keeps a UTF-8 copy inside every non-ASCII string it reads, for
    the rest of that string's life. ``str.encode``, which the per-value loop
    calls, keeps nothing.
    """

    values = np.asarray(
        ["".join(["caf", "é-", str(position)]) for position in range(3)]
        + ["".join(["中文-", str(position)]) for position in range(3)],
        dtype=object,
    )
    series = pd.Series(values, dtype=TEXT_DTYPES[dtype_name], copy=False)
    stored_values = series.to_numpy(dtype=object, copy=False)
    sizes_before = [sys.getsizeof(value) for value in stored_values]

    _round_trip_series(tmp_path / "column.h5", series)

    assert [sys.getsizeof(value) for value in stored_values] == sizes_before


_FALLBACK_COLUMNS: dict[str, Callable[[], pd.Series]] = {
    "object strings with None": lambda: pd.Series(["a", None, "c"], dtype=object),
    "object strings with NaN": lambda: pd.Series(["a", np.nan], dtype=object),
    "object strings with pd.NA": lambda: pd.Series(["a", pd.NA], dtype=object),
    "object None only": lambda: pd.Series([None, None], dtype=object),
    "object integers": lambda: pd.Series([1, 2], dtype=object),
    "object floats": lambda: pd.Series([1.5, np.nan], dtype=object),
    "object bytes": lambda: pd.Series([b"x", b"y"], dtype=object),
    "object booleans": lambda: pd.Series([True, False], dtype=object),
    "object floats whose type cannot be hashed": lambda: pd.Series(
        np.asarray(
            [_UnhashableTypeFloat(1.5), _UnhashableTypeFloat(2.0)], dtype=object
        ),
        dtype=object,
    ),
    **{
        f"{dtype_name} with a missing value": (
            lambda dtype=dtype: pd.Series(["a", None, "c"], dtype=dtype)
        )
        for dtype_name, dtype in TEXT_DTYPES.items()
        if dtype != "object"
    },
}


@pytest.mark.parametrize("case", list(_FALLBACK_COLUMNS))
def test_missing_and_non_string_values_go_through_the_per_value_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    series = _FALLBACK_COLUMNS[case]()
    calls = _LoopCalls(monkeypatch)

    _assert_matches_loop(tmp_path / "column.h5", series)

    assert calls.encoded == [len(series)]
    assert calls.decoded == [len(series)]


@pytest.mark.parametrize("dtype_name", ["python_str", "python_string", "object"])
def test_unencodable_strings_raise_the_per_value_loop_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype_name: str
) -> None:
    series = pd.Series(["fine", "\ud800"], dtype=TEXT_DTYPES[dtype_name])
    calls = _LoopCalls(monkeypatch)

    with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
        _round_trip_series(tmp_path / "column.h5", series)

    assert calls.encoded == [2]


def _read_stored(
    path: Path,
    dtype_name: str,
    offsets: np.ndarray,
    payload: np.ndarray,
) -> pd.Series:
    spec = frame_checkpoint_module._series_spec(
        pd.Series(["a"], dtype=TEXT_DTYPES[dtype_name]), label="column"
    )
    with h5py.File(path, mode="w") as h5:
        group = h5.create_group("column", track_times=False)
        group.create_dataset("offsets", data=offsets, track_times=False)
        group.create_dataset("payload", data=payload, track_times=False)
    with h5py.File(path, mode="r") as h5:
        return frame_checkpoint_module._read_series(
            h5["column"], spec, path, label="column"
        )


def _bytes(*parts: bytes) -> np.ndarray:
    return np.frombuffer(b"".join(parts), dtype=np.uint8)


@pytest.mark.parametrize("dtype_name", list(TEXT_DTYPES))
def test_invalid_utf8_raises_the_per_value_loop_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype_name: str
) -> None:
    calls = _LoopCalls(monkeypatch)

    with pytest.raises(ValueError, match="invalid object value at row 1"):
        _read_stored(
            tmp_path / "column.h5",
            dtype_name,
            np.asarray([0, 3, 5], dtype=np.int64),
            _bytes(b"\x07ok", b"\x07\xff"),
        )

    assert calls.decoded == [2]


_MALFORMED_OFFSETS = {
    "int32 offsets": (
        np.asarray([0, 2], dtype=np.int32),
        _bytes(b"\x07a"),
        "malformed object offsets",
    ),
    "no offsets": (
        np.asarray([], dtype=np.int64),
        _bytes(b""),
        "malformed object offsets",
    ),
    "first offset not zero": (
        np.asarray([1, 2], dtype=np.int64),
        _bytes(b"\x07a"),
        "invalid object boundaries",
    ),
    "last offset short of the payload": (
        np.asarray([0, 2], dtype=np.int64),
        _bytes(b"\x07ab"),
        "invalid object boundaries",
    ),
    "value without its tag byte": (
        np.asarray([0, 2, 2], dtype=np.int64),
        _bytes(b"\x07a"),
        "invalid object boundaries",
    ),
}


@pytest.mark.parametrize("dtype_name", list(TEXT_DTYPES))
@pytest.mark.parametrize("case", list(_MALFORMED_OFFSETS))
def test_malformed_offsets_raise_the_per_value_loop_error(
    tmp_path: Path, case: str, dtype_name: str
) -> None:
    offsets, payload, message = _MALFORMED_OFFSETS[case]

    with pytest.raises(ValueError, match=message):
        _read_stored(tmp_path / "column.h5", dtype_name, offsets, payload)


def _read_error(
    path: Path, dtype_name: str, offsets: np.ndarray, payload: np.ndarray
) -> tuple[type[BaseException], str]:
    with pytest.raises(Exception) as raised:
        _read_stored(path, dtype_name, offsets, payload)
    return raised.type, str(raised.value)


# The differences of these offsets overflow int64 and wrap around, so they
# pass the boundary check although two of them lie outside the payload.
_WRAPPING_OFFSETS = np.asarray([0, 2**63 - 1, -3, 5], dtype=np.int64)
_WRAPPING_PAYLOADS = {
    "first value invalid": _bytes(b"\x00abcd"),
    "first value valid": _bytes(b"\x07ab\x07c"),
}


@pytest.mark.parametrize("dtype_name", list(TEXT_DTYPES))
@pytest.mark.parametrize("case", list(_WRAPPING_PAYLOADS))
def test_wrapping_offsets_raise_the_per_value_loop_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, dtype_name: str
) -> None:
    payload = _WRAPPING_PAYLOADS[case]
    path = tmp_path / "column.h5"  # error messages name the file
    with monkeypatch.context() as patch:
        _use_only_per_value_loop(patch)
        expected = _read_error(path, dtype_name, _WRAPPING_OFFSETS, payload)

    actual = _read_error(path, dtype_name, _WRAPPING_OFFSETS, payload)

    assert actual == expected


@pytest.mark.parametrize("dtype_name", ["pyarrow_str", "pyarrow_string"])
def test_reading_falls_back_when_pandas_refuses_the_arrow_array(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dtype_name: str
) -> None:
    """A pandas that casts one Arrow array to 32-bit offsets refuses 2 GiB.

    pandas 2.2's ``ArrowStringArray._from_sequence`` casts an Arrow array to
    ``pa.string()``, which refuses 2**31 or more bytes of text, while the
    per-value loop's object values are split into chunks. The reader then
    uses the loop.
    """

    series = pd.Series(["a", "é", ""], dtype=TEXT_DTYPES[dtype_name])
    original_array = pd.array

    def refusing_array(data: object, *arguments: object, **keywords: object):
        if isinstance(data, pa.Array):
            raise pa.ArrowInvalid("input array too large")
        return original_array(data, *arguments, **keywords)

    monkeypatch.setattr(pd, "array", refusing_array)
    calls = _LoopCalls(monkeypatch)

    _assert_matches_loop(tmp_path / "column.h5", series)

    assert calls.decoded == [3]


def test_object_tags_other_than_string_go_through_the_per_value_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _LoopCalls(monkeypatch)

    restored = _read_stored(
        tmp_path / "column.h5",
        "object",
        np.asarray([0, 2, 3], dtype=np.int64),
        _bytes(b"\x07a", b"\x00"),
    )

    assert restored.tolist() == ["a", None]
    assert calls.decoded == [2]


_LARGE_TEXT_VARIABLE = "MICROCOSM_FRAME_CHECKPOINT_LARGE_TEXT_TEST"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get(_LARGE_TEXT_VARIABLE) != "1",
    reason=f"set {_LARGE_TEXT_VARIABLE}=1 to run; it needs about 6 GB of memory",
)
@pytest.mark.parametrize("dtype_name", ["pyarrow_str", "object"])
def test_text_payload_over_two_gibibytes_keeps_the_per_value_loop_bytes(
    dtype_name: str,
) -> None:
    """Offsets past 2**31, where 32-bit Arrow offsets would overflow."""

    long_length = 2**30 + 7
    values = ["a" * long_length, "b" * long_length, "after-é", "", "z"]
    series = pd.Series(values, dtype=TEXT_DTYPES[dtype_name])
    del values
    expected_offsets = [0]
    expected_digest = hashlib.sha256()
    for position in range(len(series)):
        offsets, payload = PER_VALUE_ENCODE(
            np.asarray([series.iloc[position]], dtype=object)
        )
        expected_offsets.append(expected_offsets[-1] + int(offsets[-1]))
        expected_digest.update(payload)
        del offsets, payload

    encoded = frame_checkpoint_module._encode_string_values(series)
    assert encoded is not None
    offsets, payload = encoded
    assert offsets.dtype == np.dtype(np.int64)
    assert int(offsets[-2]) > 2**31
    np.testing.assert_array_equal(offsets, np.asarray(expected_offsets))
    assert hashlib.sha256(payload).hexdigest() == expected_digest.hexdigest()

    restore_dtype = frame_checkpoint_module._declared_dtype(
        frame_checkpoint_module._series_spec(series, label="column"),
        path=Path("large.h5"),
        label="column",
    )
    restored = frame_checkpoint_module._decode_string_values(
        offsets, payload, restore_dtype
    )
    del payload
    assert restored is not None
    assert restored.dtype == series.dtype
    assert restored.equals(series)
