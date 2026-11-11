"""Byte-exact parity for the executor's vectorised column digest stream.

Invented columns only: no data file, no store, no engine.  ``_update_series``
now builds the framed object-projection bytes for plain float, integer and
boolean columns with numpy instead of boxing every value into a Python object
and framing it one at a time.  The emitted bytes -- the ``object`` dtype
header, the shape header, every per-value length prefix and every payload --
must be identical to what the boxing loop wrote.  Every test compares the live
helper against ``_reference_update_series`` below, a verbatim copy of the
pre-change body, at the byte level rather than only at the digest.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import pytest

from microcosm.graph.executor import _object_stream, _update_series


def _reference_update_scalar(digest, value):
    """The pre-change scalar body, kept verbatim as the parity reference."""

    if value is pd.NA:
        payload = b"pd.NA"
    elif value is pd.NaT:
        payload = b"pd.NaT"
    elif value is None:
        payload = b"None"
    elif isinstance(value, (float, np.floating)):
        payload = b"f" + np.asarray([value], dtype=np.float64).tobytes()
    elif isinstance(value, (bool, np.bool_)):
        payload = b"b1" if bool(value) else b"b0"
    elif isinstance(value, (int, np.integer)):
        payload = b"i" + str(int(value)).encode("ascii")
    elif isinstance(value, str):
        payload = b"s" + value.encode("utf-8")
    elif isinstance(value, (bytes, np.bytes_)):
        payload = b"y" + bytes(value)
    else:
        payload = b"r" + repr(value).encode("utf-8")
    digest.update(len(payload).to_bytes(8, "little"))
    digest.update(payload)


def _reference_update_array(digest, values):
    """The pre-change array body, kept verbatim as the parity reference."""

    array = np.asarray(values)
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(array.shape).encode("ascii"))
    digest.update(b"\0")
    if array.dtype.hasobject:
        for value in array.ravel(order="C"):
            _reference_update_scalar(digest, value)
    else:
        digest.update(np.ascontiguousarray(array).tobytes())


def _reference_update_series(digest, series):
    """The pre-change series body, kept verbatim as the parity reference."""

    digest.update(str(series.dtype).encode("utf-8"))
    digest.update(b"\0")
    extension = series.array
    data = getattr(extension, "_data", None)
    mask = getattr(extension, "_mask", None)
    if isinstance(data, np.ndarray) and isinstance(mask, np.ndarray):
        _reference_update_array(digest, data)
        _reference_update_array(digest, mask)
    else:
        _reference_update_array(digest, series.to_numpy(dtype=object, copy=False))
    _reference_update_array(digest, series.index.to_numpy(copy=False))


class _Recorder:
    """A digest-shaped sink that keeps every byte the helper writes."""

    def __init__(self):
        self.chunks = []

    def update(self, payload):
        self.chunks.append(bytes(payload))

    def bytes(self):
        return b"".join(self.chunks)


def _written(helper, series):
    recorder = _Recorder()
    helper(recorder, series)
    return recorder.bytes()


def _digest(helper, series):
    digest = hashlib.sha256()
    helper(digest, series)
    return digest.digest()


def _assert_identical(series):
    assert _written(_update_series, series) == _written(
        _reference_update_series, series
    )
    assert _digest(_update_series, series) == _digest(_reference_update_series, series)


def _invented_columns():
    """One column of every kind a projected kernel context can carry."""

    generator = np.random.default_rng(20260915)
    columns = {
        "float64": pd.Series(generator.normal(size=97)),
        "float32": pd.Series(generator.normal(size=97).astype("float32")),
        "float16": pd.Series(generator.normal(size=97).astype("float16")),
        "longdouble": pd.Series(np.array([1.5, 1 / 3, np.nan], dtype=np.longdouble)),
        "int64": pd.Series(generator.integers(-(10**15), 10**15, size=97)),
        "int32": pd.Series(
            generator.integers(-(2**30), 2**30, size=97).astype("int32")
        ),
        "int8": pd.Series(generator.integers(-128, 127, size=97).astype("int8")),
        "uint64": pd.Series(np.array([0, 1, np.iinfo(np.uint64).max], dtype="uint64")),
        "uint8": pd.Series(np.arange(256, dtype="uint8")),
        "bool": pd.Series(generator.integers(0, 2, size=97).astype(bool)),
        "complex": pd.Series(np.array([1 + 2j, 3 - 4j])),
        "Int64": pd.Series(pd.array([1, None, -3], dtype="Int64")),
        "Float64": pd.Series(pd.array([1.0, None, -0.0], dtype="Float64")),
        "boolean": pd.Series(pd.array([True, None, False], dtype="boolean")),
        "string": pd.Series(["a", "bb", None, "é\U0001f600"], dtype="str"),
        "object": pd.Series(
            ["x", 1, None, 2.5, True, b"z", pd.NA, pd.NaT, (1, 2)], dtype="object"
        ),
        "category": pd.Series(pd.Categorical(generator.choice(list("abcd"), size=97))),
        "datetime": pd.Series(pd.to_datetime([0, 1_600_000_000], unit="s")),
        "timedelta": pd.Series(pd.to_timedelta([0, 90_061], unit="s")),
    }
    for name in ("float64", "int64", "bool", "object", "str"):
        columns[f"empty-{name}"] = pd.Series([], dtype=name)
    columns["sliced-float"] = pd.Series(generator.normal(size=500))[13:400:7]
    columns["strided-int"] = pd.Series(generator.integers(0, 10**6, size=500))[::11]
    columns["labelled-index"] = pd.Series(
        generator.normal(size=17), index=[f"k{i}" for i in range(17)]
    )
    return columns


@pytest.mark.parametrize("name", sorted(_invented_columns()))
def test_every_column_kind_streams_identical_bytes(name):
    _assert_identical(_invented_columns()[name])


def test_float_specials_stream_identical_bytes():
    """Signed zero, both infinities, NaN and the subnormal extremes."""

    _assert_identical(
        pd.Series(
            [
                0.0,
                -0.0,
                np.inf,
                -np.inf,
                np.nan,
                1.7976931348623157e308,
                -1.7976931348623157e308,
                5e-324,
                -5e-324,
                -1.5,
            ]
        )
    )


def test_nan_payload_bits_survive_the_vectorised_path():
    """A non-canonical NaN keeps its exact payload, sign bit included."""

    raw = np.array(
        [0x7FF8000000ABCDEF, 0xFFF8000000ABCDEF, 0x7FF0000000000001], dtype="uint64"
    )
    series = pd.Series(raw.view("float64"))
    assert series.isna().all()
    _assert_identical(series)


def test_integer_extremes_stream_identical_bytes():
    """Variable-width decimal payloads, including both int64 endpoints."""

    _assert_identical(
        pd.Series(
            np.array(
                [0, -1, 1, 9, 10, -10, np.iinfo(np.int64).min, np.iinfo(np.int64).max],
                dtype="int64",
            )
        )
    )


def test_categorical_is_not_mistaken_for_its_integer_codes():
    """The fast path must refuse any column whose values are a coded view."""

    series = pd.Series(pd.Categorical(list("abca")))
    assert series.array.dtype.kind not in "fiub"
    assert _object_stream(series) is None
    _assert_identical(series)


def test_datetime_and_string_columns_take_the_reference_loop():
    for series in (
        pd.Series(pd.to_datetime([0, 1], unit="s")),
        pd.Series(pd.to_timedelta([0, 1], unit="s")),
        pd.Series(["a", None], dtype="str"),
        pd.Series([object()], dtype="object"),
    ):
        assert _object_stream(series) is None
        _assert_identical(series)


def test_masked_columns_keep_their_existing_data_and_mask_path():
    """Nullable columns already streamed their raw buffers; nothing moves."""

    for dtype in ("Int64", "Float64", "boolean"):
        series = pd.Series(pd.array([1, None, 0], dtype=dtype))
        assert isinstance(series.array._data, np.ndarray)
        _assert_identical(series)


def test_random_frames_of_every_kind_stream_identical_bytes():
    """The property sweep: random widths and contents, every column kind."""

    generator = np.random.default_rng(20260916)
    kinds = (
        "float64",
        "float32",
        "int64",
        "int32",
        "uint32",
        "bool",
        "Int64",
        "str",
        "category",
        "object",
    )
    for trial in range(200):
        rows = int(generator.integers(0, 60))
        kind = kinds[trial % len(kinds)]
        if kind.startswith("float"):
            values = generator.normal(size=rows).astype(kind)
            series = pd.Series(values)
            if rows:
                series.iloc[int(generator.integers(0, rows))] = np.nan
        elif kind in ("int64", "int32", "uint32"):
            series = pd.Series(generator.integers(0, 10**9, size=rows).astype(kind))
        elif kind == "bool":
            series = pd.Series(generator.integers(0, 2, size=rows).astype(bool))
        elif kind == "Int64":
            series = pd.Series(
                pd.array(
                    [
                        None if generator.random() < 0.3 else int(value)
                        for value in generator.integers(-99, 99, size=rows)
                    ],
                    dtype="Int64",
                )
            )
        elif kind == "str":
            series = pd.Series(
                [
                    None
                    if generator.random() < 0.2
                    else "s" * int(generator.integers(0, 5))
                    for _ in range(rows)
                ],
                dtype="str",
            )
        elif kind == "category":
            series = pd.Series(pd.Categorical(generator.choice(list("xyz"), size=rows)))
        else:
            series = pd.Series(
                [
                    [1, "a", None, 2.5, True][int(generator.integers(0, 5))]
                    for _ in range(rows)
                ],
                dtype="object",
            )
        _assert_identical(series)


# --------------------------------------------------------------------------
# The length prefix, built without numpy on either side.
# --------------------------------------------------------------------------


def _expected_stream(payloads):
    """The framed stream, assembled in pure Python.

    Every parity assertion above compares one numpy construction against
    another, so a byte order both sides take from the host cancels out. This
    reference takes its byte order from `int.to_bytes(8, "little")` and its
    payloads from Python `bytes`, which is what `_update_scalar` writes and
    what neither branch of `_object_stream` may drift from.
    """

    return b"".join(
        len(payload).to_bytes(8, "little") + payload for payload in payloads
    )


@pytest.mark.parametrize(
    "dtype", ["int64", "int32", "int16", "int8", "uint32", "uint8"]
)
def test_the_integer_prefix_is_little_endian_whatever_the_host_is(dtype):
    """The vectorised prefix is cast after the arithmetic, not before it.

    `filled.sum(axis=1)` is a ufunc result and a ufunc's output carries the
    host's byte order, so casting it to `<u8` and then adding one for the type
    byte hands `tobytes()` a native-order array again. Casting the finished
    length is what makes these eight bytes little-endian on every host. The
    two agree on a little-endian host, which is every host this suite runs on
    today, so what this case pins is the construction rather than a difference
    visible here.
    """

    info = np.iinfo(dtype)
    values = [0, 1, 9, 10, info.min, info.max]
    series = pd.Series(np.array(values, dtype=dtype))
    expected = _expected_stream(
        [b"i" + str(int(value)).encode("ascii") for value in values]
    )

    assert _object_stream(series) == expected
    # The first prefix is the shortest payload's, so a byte-swapped eight-byte
    # word would be caught by this whole-stream comparison and not only by the
    # widest one.
    assert expected[:8] == (2).to_bytes(8, "little")


def test_the_boolean_prefix_is_little_endian_too():
    """`_framed`'s `<u8` prefix, against the same pure-Python reference."""

    series = pd.Series(np.array([True, False, True], dtype=bool))
    assert _object_stream(series) == _expected_stream([b"b1", b"b0", b"b1"])


def test_the_float_prefix_is_little_endian_while_its_payload_is_native():
    """Nine bytes of payload, prefixed little-endian, packed in host order.

    The payload deliberately follows the host: the loop packs
    `np.asarray([value], dtype=np.float64).tobytes()`. The prefix deliberately
    does not.
    """

    values = [1.5, -0.0, np.inf]
    series = pd.Series(np.array(values, dtype="float64"))
    expected = _expected_stream(
        [b"f" + np.asarray([value], dtype=np.float64).tobytes() for value in values]
    )

    assert _object_stream(series) == expected
    assert expected[:8] == (9).to_bytes(8, "little")
