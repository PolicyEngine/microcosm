"""Keyed draw streams (amendment 20): ``microcosm.graph.randomness``.

A ``SeedSource.KEYED`` kernel does not consume an RNG in row order; it asks for
the uniform belonging to a coordinate. These tests hold ``keyed_uniform`` to the
three properties that make it worth having — stable coordinates give the same
draw, different coordinates give a different one, and nothing about packing,
ordering, or unrelated identities can reach a draw — plus the documented
``sha256-u53-v1`` formula, recomputed here from the specification rather than
read back from the implementation.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from microcosm.graph import keyed_uniform
from microcosm.graph.canonical import canonical_json

STREAM = ("sha256-u53-v1", "amendment-20", 0, 0)


def _spec_uniform(stream: tuple, key: tuple) -> float:
    """The documented draw, derived here from the specification's own words.

    ``keyed_uniform``'s docstring defines a draw as the top 53 bits of the
    SHA-256 digest over a domain prefix, the canonical stream, and the canonical
    tagged coordinates, divided by ``2**53``. Recomputing it from that sentence
    is the difference between testing the algorithm and testing the code.
    """
    tags = {bool: "bool", int: "int", str: "str", float: "float"}
    encoded = canonical_json([[tags[type(value)], value] for value in key])
    digest = hashlib.sha256(
        b"microcosm-graph/keyed-uniform/1\0" + canonical_json(stream) + b"\0" + encoded
    ).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / 2**53


def test_stable_coordinates_give_one_draw_and_different_ones_give_another() -> None:
    keys = [(1, "wages", 2026, 0), (2, "wages", 2026, 0)]
    first = keyed_uniform(stream=STREAM, keys=keys)
    again = keyed_uniform(stream=STREAM, keys=keys)

    assert first.dtype == np.dtype("float64")
    assert first.shape == (2,)
    assert first.tobytes() == again.tobytes()
    assert first[0] != first[1]
    assert ((first >= 0.0) & (first < 1.0)).all()

    # Every coordinate position is live: moving any one of the four moves the
    # draw, so a process, a period, or a draw index cannot collide with another.
    base = keyed_uniform(stream=STREAM, keys=[(1, "wages", 2026, 0)])[0]
    for moved in (
        (2, "wages", 2026, 0),
        (1, "dividends", 2026, 0),
        (1, "wages", 2027, 0),
        (1, "wages", 2026, 1),
    ):
        assert keyed_uniform(stream=STREAM, keys=[moved])[0] != base


def test_repeated_coordinates_intentionally_repeat_the_draw() -> None:
    """The docstring says so out loud; a row asked for twice gets one answer."""
    values = keyed_uniform(stream=STREAM, keys=[(7, "wages"), (7, "wages")])
    assert values[0] == values[1]


def test_a_draw_is_blind_to_order_batching_and_unrelated_identities() -> None:
    """Packing cannot reach a draw: this is C1 and C2 for randomness."""
    keys = [(1, "wages"), (2, "wages"), (3, "wages")]
    straight = keyed_uniform(stream=STREAM, keys=keys)
    reversed_ = keyed_uniform(stream=STREAM, keys=list(reversed(keys)))
    assert straight.tobytes() == reversed_[::-1].copy().tobytes()

    chunked = np.concatenate(
        [
            keyed_uniform(stream=STREAM, keys=keys[:1]),
            keyed_uniform(stream=STREAM, keys=keys[1:]),
        ]
    )
    assert straight.tobytes() == chunked.tobytes()

    # Inserting an unrelated identity ahead of a row leaves that row's draw
    # alone — the failure mode a positionally consumed RNG cannot avoid.
    with_intruder = keyed_uniform(stream=STREAM, keys=[(99, "wages"), *keys])
    assert with_intruder[1:].tobytes() == straight.tobytes()


def test_the_draw_is_the_documented_sha256_u53_formula() -> None:
    keys = [(1, "wages", 2026, 0), (2, "wages", 2026, 0), ("h-3", True, -1.5)]
    expected = np.asarray(
        [_spec_uniform(STREAM, key) for key in keys], dtype=np.float64
    )
    assert keyed_uniform(stream=STREAM, keys=keys).tobytes() == expected.tobytes()


def test_streams_are_separated_by_experiment_replicate_and_base_seed() -> None:
    key = [(1, "wages")]
    base = keyed_uniform(stream=STREAM, keys=key)[0]
    for stream in (
        ("sha256-u53-v1", "amendment-20-b", 0, 0),
        ("sha256-u53-v1", "amendment-20", 1, 0),
        ("sha256-u53-v1", "amendment-20", 0, 1),
    ):
        assert keyed_uniform(stream=stream, keys=key)[0] != base


def test_coordinate_types_are_tagged_so_look_alikes_do_not_collide() -> None:
    """``True`` is not ``1`` and ``1`` is not ``"1"``: each carries its tag."""
    drawn = {
        label: keyed_uniform(stream=STREAM, keys=[(value,)])[0]
        for label, value in (
            ("bool", True),
            ("int", 1),
            ("str", "1"),
            ("float", 1.0),
        )
    }
    assert len(set(drawn.values())) == 4


def test_numpy_scalars_are_the_python_scalars_they_hold() -> None:
    """A coordinate read out of a column must not draw differently."""
    assert (
        keyed_uniform(stream=STREAM, keys=[(np.int64(5), "wages")])[0]
        == keyed_uniform(stream=STREAM, keys=[(5, "wages")])[0]
    )
    assert (
        keyed_uniform(stream=STREAM, keys=[(np.bool_(True),)])[0]
        == keyed_uniform(stream=STREAM, keys=[(True,)])[0]
    )


def test_negative_zero_is_the_same_float_coordinate_as_positive_zero() -> None:
    """``-0.0 == 0.0`` and they hash alike, so they must draw alike.

    A negation or a CSV literal can hand a column ``-0.0`` where another run
    holds ``0.0``; without normalisation ``canonical_json`` would spell them
    differently and silently re-randomise that row.
    """
    positive = keyed_uniform(stream=STREAM, keys=[(7, 0.0)])[0]
    assert keyed_uniform(stream=STREAM, keys=[(7, -0.0)])[0] == positive
    assert keyed_uniform(stream=STREAM, keys=[(7, np.float64(-0.0))])[0] == positive
    # The normalisation is to positive zero: the documented formula over
    # ``0.0`` is the draw both spellings produce.
    assert positive == _spec_uniform(STREAM, (7, 0.0))


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
@pytest.mark.parametrize("unit", ["D", "s", "ns", "ps"])
def test_temporal_coordinates_refuse_before_losing_type_or_units(scalar, unit):
    # Fine temporal units become bare integers under .item(); they cannot be
    # admitted as the same random identity as an integer or another unit.
    with pytest.raises(TypeError, match="scalar identities"):
        keyed_uniform(stream=STREAM, keys=[(scalar(1, unit),)])


def test_draws_are_bytes_backed_and_read_only() -> None:
    values = keyed_uniform(stream=STREAM, keys=[(1,), (2,)])
    assert not values.flags.writeable
    with pytest.raises(ValueError):
        values[0] = 0.5


def test_an_empty_key_list_draws_nothing() -> None:
    values = keyed_uniform(stream=STREAM, keys=[])
    assert values.dtype == np.dtype("float64")
    assert values.shape == (0,)


def test_a_malformed_stream_is_refused() -> None:
    with pytest.raises(TypeError, match="algorithm, experiment_id"):
        keyed_uniform(stream=("sha256-u53-v1", "e", 0), keys=[(1,)])
    with pytest.raises(TypeError, match="algorithm, experiment_id"):
        keyed_uniform(stream=["sha256-u53-v1", "e", 0, 0], keys=[(1,)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported random stream algorithm"):
        keyed_uniform(stream=("sha256-u53-v2", "e", 0, 0), keys=[(1,)])
    with pytest.raises(ValueError, match="experiment_id must be non-empty"):
        keyed_uniform(stream=("sha256-u53-v1", "", 0, 0), keys=[(1,)])
    with pytest.raises(ValueError, match="non-negative integers"):
        keyed_uniform(stream=("sha256-u53-v1", "e", -1, 0), keys=[(1,)])
    with pytest.raises(ValueError, match="non-negative integers"):
        keyed_uniform(stream=("sha256-u53-v1", "e", 0, -1), keys=[(1,)])
    # A bool is not a replicate index; it spells an integer without being one.
    with pytest.raises(ValueError, match="non-negative integers"):
        keyed_uniform(stream=("sha256-u53-v1", "e", True, 0), keys=[(1,)])


def test_a_malformed_coordinate_is_refused() -> None:
    with pytest.raises(TypeError, match="nonempty coordinate tuple"):
        keyed_uniform(stream=STREAM, keys=[()])
    with pytest.raises(TypeError, match="nonempty coordinate tuple"):
        keyed_uniform(stream=STREAM, keys=[[1, "wages"]])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="non-null finite scalar identities"):
        keyed_uniform(stream=STREAM, keys=[(None,)])
    with pytest.raises(TypeError, match="non-null finite scalar identities"):
        keyed_uniform(stream=STREAM, keys=[(float("nan"),)])
    with pytest.raises(TypeError, match="non-null finite scalar identities"):
        keyed_uniform(stream=STREAM, keys=[(float("inf"),)])
    with pytest.raises(TypeError, match="non-null finite scalar identities"):
        keyed_uniform(stream=STREAM, keys=[({"person": 1},)])  # type: ignore[dict-item]


def test_keyed_draws_read_and_mutate_no_numpy_rng_state() -> None:
    """The point of the amendment: a draw is not a position in a stream.

    Neither a generator nor the process-global legacy state moves, so a keyed
    kernel cannot perturb an unrelated stream and cannot be perturbed by one.
    """
    rng = np.random.default_rng(20)
    generator_before = rng.bit_generator.state
    global_before = np.random.get_state()

    values = keyed_uniform(stream=STREAM, keys=[(i, "wages") for i in range(64)])

    assert rng.bit_generator.state == generator_before
    after = np.random.get_state()
    assert after[0] == global_before[0]
    assert np.array_equal(after[1], global_before[1])
    assert after[2:] == global_before[2:]
    # The same draws again, with a generator drained in between: consumption
    # order elsewhere is invisible to a keyed draw.
    rng.random(1_000)
    assert (
        keyed_uniform(stream=STREAM, keys=[(i, "wages") for i in range(64)]).tobytes()
        == values.tobytes()
    )
