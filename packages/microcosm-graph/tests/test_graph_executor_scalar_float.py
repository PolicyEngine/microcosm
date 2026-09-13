"""Exact-float packing parity for the executor's scalar digest framing.

Invented scalars only: no data file, no store, no engine.  ``_update_scalar``
now packs an exact Python float with ``struct.pack("=d", ...)`` instead of
building a one-element float64 array; the emitted payload, its length prefix
and its framing must be identical, sign bit, NaN payload and all.  Every test
compares the live helper against ``_reference_update_scalar`` below, a verbatim
copy of the pre-change body, at the byte level rather than only at the digest.
"""

from __future__ import annotations

import decimal
import struct
import sys

import numpy as np
import pandas as pd
import pytest

from microcosm.graph.executor import _update_array, _update_scalar


def _reference_update_scalar(digest, value):
    """The exact pre-change body, kept verbatim as the parity reference."""

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
    """The pre-change ``_update_array``: only its scalar helper differs."""

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


class _Recorder:
    """Stands in for a hash object: keeps every update chunk, in order."""

    def __init__(self):
        self.chunks = []

    def update(self, data):
        self.chunks.append(bytes(data))


def _emitted(update, value):
    recorder = _Recorder()
    update(recorder, value)
    return tuple(recorder.chunks)


def _nan(payload):
    return struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000000 | payload))[0]


def _signed_nan(payload):
    return struct.unpack("<d", struct.pack("<Q", 0xFFF8000000000000 | payload))[0]


_NAN_ONE = _nan(0x1)
_NAN_TWO = _nan(0x2)
_NAN_SIGNED = _signed_nan(0x1)
_NAN_QUIET = float("nan")


class _Money(float):
    """A float subclass: it must keep taking the numpy path, unchanged."""


class _Opaque:
    def __repr__(self):
        return "<opaque>"


_FLOATS = {
    "zero": 0.0,
    "negative-zero": -0.0,
    "one": 1.0,
    "negative": -1.5,
    "infinity": float("inf"),
    "negative-infinity": float("-inf"),
    "quiet-nan": _NAN_QUIET,
    "nan-payload-one": _NAN_ONE,
    "nan-payload-two": _NAN_TWO,
    "nan-signed": _NAN_SIGNED,
    "subnormal": 5e-324,
    "negative-subnormal": -5e-324,
    "max": sys.float_info.max,
    "min-normal": sys.float_info.min,
    "epsilon": sys.float_info.epsilon,
    "integral": 2.0**53,
    "tenth": 0.1,
}

_OTHER_SCALARS = {
    "none": None,
    "pd-na": pd.NA,
    "pd-nat": pd.NaT,
    "true": True,
    "false": False,
    "np-bool": np.bool_(True),
    "int": 5,
    "negative-int": -7,
    "huge-int": -(2**70),
    "np-int64": np.int64(5),
    "np-uint8": np.uint8(255),
    "str": "text",
    "empty-str": "",
    "unicode": "ä🙂",
    "bytes": b"raw",
    "np-bytes": np.bytes_(b"raw"),
    "opaque-repr": _Opaque(),
    "decimal": decimal.Decimal("1.5"),
}


@pytest.mark.parametrize("name", sorted(_FLOATS))
def test_exact_float_packing_matches_the_array_reference(name):
    value = _FLOATS[name]
    assert _emitted(_update_scalar, value) == _emitted(_reference_update_scalar, value)


@pytest.mark.parametrize("name", sorted(_FLOATS))
def test_struct_packing_is_the_array_byte_payload(name):
    value = _FLOATS[name]
    assert struct.pack("=d", value) == np.asarray([value], dtype=np.float64).tobytes()


@pytest.mark.parametrize("name", sorted(_FLOATS))
def test_numpy_floats_still_take_the_array_path_and_agree(name):
    value = _FLOATS[name]
    for numpy_value in (np.float64(value), _Money(value)):
        assert _emitted(_update_scalar, numpy_value) == _emitted(
            _reference_update_scalar, numpy_value
        )
    # The fast path must not diverge from the array path for the same number.
    assert _emitted(_update_scalar, value) == _emitted(
        _update_scalar, np.float64(value)
    )


@pytest.mark.parametrize("name", sorted(_OTHER_SCALARS))
def test_every_other_scalar_is_unchanged(name):
    value = _OTHER_SCALARS[name]
    assert _emitted(_update_scalar, value) == _emitted(_reference_update_scalar, value)


@pytest.mark.parametrize(
    "value", [np.float32(1.5), np.float16(1.5), np.float32(-0.0), np.float32(np.inf)]
)
def test_narrow_numpy_floats_keep_their_widened_bytes(value):
    assert _emitted(_update_scalar, value) == _emitted(_reference_update_scalar, value)


def _distinct_pairs():
    yield "signed-zero", 0.0, -0.0
    yield "nan-payload", _NAN_ONE, _NAN_TWO
    yield "nan-sign", _NAN_ONE, _NAN_SIGNED
    yield "infinity-sign", float("inf"), float("-inf")
    yield "subnormal", 5e-324, 1e-323
    yield "float-vs-int", 5.0, 5
    yield "float-vs-str", 1.0, "1.0"
    yield "true-vs-one", True, 1


@pytest.mark.parametrize("label,left,right", list(_distinct_pairs()), ids=lambda v: v)
def test_distinguishable_scalars_stay_distinguishable(label, left, right):
    assert _emitted(_update_scalar, left) != _emitted(_update_scalar, right)
    assert _emitted(_reference_update_scalar, left) != _emitted(
        _reference_update_scalar, right
    )


def test_framing_is_one_length_prefix_and_one_payload():
    chunks = _emitted(_update_scalar, 1.5)
    assert len(chunks) == 2
    assert chunks[0] == (9).to_bytes(8, "little")
    assert chunks[1] == b"f" + struct.pack("=d", 1.5)


def _object_arrays():
    array = np.empty(6, dtype=object)
    array[0] = 1.5
    array[1] = -0.0
    array[2] = _NAN_ONE
    array[3] = np.float64(2.5)
    array[4] = _Money(3.5)
    array[5] = None
    yield "mixed-floats", array

    nested = np.empty((2, 2), dtype=object)
    nested[0, 0] = _NAN_TWO
    nested[0, 1] = "text"
    nested[1, 0] = 7
    nested[1, 1] = float("-inf")
    yield "two-dimensional", nested

    yield "empty", np.empty(0, dtype=object)


@pytest.mark.parametrize("label,values", list(_object_arrays()), ids=lambda v: v)
def test_object_arrays_emit_identical_bytes(label, values):
    live, reference = _Recorder(), _Recorder()
    _update_array(live, values)
    _reference_update_array(reference, values)
    assert live.chunks == reference.chunks


def test_dense_float_arrays_are_untouched_by_the_scalar_fast_path():
    values = np.array([-0.0, _NAN_ONE, float("inf"), 5e-324], dtype=np.float64)
    live, reference = _Recorder(), _Recorder()
    _update_array(live, values)
    _reference_update_array(reference, values)
    assert live.chunks == reference.chunks
    assert b"".join(live.chunks).endswith(values.tobytes())
