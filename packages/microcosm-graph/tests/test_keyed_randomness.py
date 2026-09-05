"""Random coordinates retain draws through ordering and chunking changes."""

import numpy as np
import pytest

from microcosm.graph import SeedSource, keyed_uniform

STREAM = ("sha256-u53-v1", "comparison", 0, 42)
KEYS = [
    (10, "mortality", 2027, 0),
    (11, "mortality", 2027, 0),
    (12, "mortality", 2027, 0),
]


def test_stability():
    whole = keyed_uniform(stream=STREAM, keys=KEYS)
    assert whole.dtype == np.float64 and not whole.flags.writeable
    assert ((whole >= 0) & (whole < 1)).all()
    np.testing.assert_array_equal(
        whole[::-1], keyed_uniform(stream=STREAM, keys=KEYS[::-1])
    )
    np.testing.assert_array_equal(
        whole,
        np.concatenate(
            [
                keyed_uniform(stream=STREAM, keys=KEYS[:1]),
                keyed_uniform(stream=STREAM, keys=KEYS[1:]),
            ]
        ),
    )
    np.testing.assert_array_equal(
        whole, keyed_uniform(stream=STREAM, keys=[("extra",), *KEYS])[1:]
    )
    assert SeedSource.KEYED.value == "keyed"
    assert keyed_uniform(stream=STREAM, keys=[]).shape == (0,)
    with pytest.raises(ValueError):
        whole.setflags(write=True)


def test_coordinate_boundaries_and_streams():
    keys = [(1,), ("1",), ("ab", "c"), ("a", "bc")]
    values = keyed_uniform(stream=STREAM, keys=keys)
    assert len(set(values)) == 4
    assert not np.array_equal(
        values, keyed_uniform(stream=(STREAM[0], STREAM[1], 1, 42), keys=keys)
    )
    np.testing.assert_array_equal(
        keyed_uniform(stream=STREAM, keys=[(np.int64(1),)]), values[:1]
    )


@pytest.mark.parametrize(
    "keys",
    [
        [(None,)],
        [(float("nan"),)],
        [(float("inf"),)],
        [(object(),)],
        [()],
        ["not-a-tuple"],
    ],
)
def test_bad_coordinates(keys):
    with pytest.raises((TypeError, ValueError)):
        keyed_uniform(stream=STREAM, keys=keys)


@pytest.mark.parametrize(
    "stream",
    [
        ("unknown", "x", 0, 1),
        ("sha256-u53-v1", "", 0, 1),
        ("sha256-u53-v1", "x", True, 1),
        ("sha256-u53-v1", "x", -1, 1),
    ],
)
def test_bad_stream(stream):
    with pytest.raises((TypeError, ValueError)):
        keyed_uniform(stream=stream, keys=KEYS)


def test_version_one_fixed_vectors():
    # SHA-256 domain/coordinate encoding + high 53 bits, not an RNG-library pin.
    assert [value.hex() for value in keyed_uniform(stream=STREAM, keys=KEYS)] == [
        "0x1.1128e9d291d62p-1",
        "0x1.854814a32b494p-3",
        "0x1.780f9dd36f1e0p-3",
    ]
