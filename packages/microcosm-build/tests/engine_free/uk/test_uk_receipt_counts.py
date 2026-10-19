"""Receipt counts of licensed records are banded at source (microcosm#1164)."""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.receipt_counts import (
    distinct_carriers,
    extreme_count,
    floored_share,
    rounded_count,
    small_cell_share,
    support_band,
)


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (1, "<10"),
        (9, "<10"),
        (10, 10),
        (14, 10),
        (25, 30),
        (994, 990),
        (995, 1000),
        (4321, 4300),
        (4350, 4400),
    ],
)
def test_rounded_count(count, expected) -> None:
    assert rounded_count(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (3, "<10"),
        (10, "10-49"),
        (49, "10-49"),
        (50, "50-99"),
        (99, "50-99"),
        (100, 100),
        (230, 200),
        (250, 300),
    ],
)
def test_extreme_count(count, expected) -> None:
    assert extreme_count(count) == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, 0), (1, "<10"), (9, "<10"), (10, "10-29"), (29, "10-29"), (30, "30+")],
)
def test_support_band(count, expected) -> None:
    # An empty cell reads 0, as the other counts here, never "<10" beside a
    # zero weight.
    assert support_band(count) == expected


def test_counts_refuse_a_negative_count() -> None:
    with pytest.raises(ValueError, match="negative"):
        rounded_count(-1)
    with pytest.raises(ValueError, match="negative"):
        extreme_count(-1)
    with pytest.raises(ValueError, match="negative"):
        support_band(-1)


@pytest.mark.parametrize(
    ("share", "expected"),
    [(0.0, 0.0), (0.987, 0.98), (0.98, 0.98), (0.999, 0.99), (1.0, 1.0)],
)
def test_floored_share(share, expected) -> None:
    assert floored_share(share) == expected


def test_floored_share_refuses_a_share_outside_zero_one() -> None:
    with pytest.raises(ValueError, match="share"):
        floored_share(1.2)


def test_small_cell_share() -> None:
    assert small_cell_share(0.0, 0) == 0.0
    assert small_cell_share(0.01, 9) == "<10"
    assert small_cell_share(0.25, 40) == pytest.approx(0.25)
    with pytest.raises(ValueError, match="negative"):
        small_cell_share(0.1, -1)


def test_distinct_carriers_count_keyed_rows_of_positive_weight() -> None:
    """The receipts' one carrier rule (the third review of #1305)."""

    keys = ["a", "a", "b", "", None, "c"]
    assert distinct_carriers(keys) == 3
    assert distinct_carriers(keys, [1.0, 0.0, 2.0, 1.0, 1.0, 0.0]) == 2
    with pytest.raises(ValueError, match="align"):
        distinct_carriers(keys, [1.0])
