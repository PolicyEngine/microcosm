"""The months since each person's last birthday, stored once in the spine (microcosm#1095)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime.birth_month import (
    MONTHS_IN_YEAR,
    stratified_positions,
    uk_months_since_last_birthday,
)


def test_positions_sit_at_the_midpoints_of_each_stratum_s_cumulative_weight() -> None:
    strata = np.asarray([0, 0, 0, 1, 1])
    draws = np.asarray([0.9, 0.1, 0.5, 0.3, 0.2])
    weights = np.asarray([1.0, 1.0, 2.0, 3.0, 1.0])

    positions = stratified_positions(strata, draws, weights)

    # Stratum 0 in draw order: 0.1 (w 1), 0.5 (w 2), 0.9 (w 1) of total 4.
    np.testing.assert_allclose(positions[[1, 2, 0]], [0.125, 0.5, 0.875])
    # Stratum 1 in draw order: 0.2 (w 1), 0.3 (w 3) of total 4.
    np.testing.assert_allclose(positions[[4, 3]], [0.125, 0.625])


def test_a_stratum_without_weight_keeps_its_draws() -> None:
    positions = stratified_positions(
        np.asarray([0, 0]), np.asarray([0.25, 0.75]), np.asarray([0.0, 0.0])
    )

    np.testing.assert_allclose(positions, [0.25, 0.75])


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.integers(0, 3),
            st.floats(0.0, 0.999999, allow_nan=False),
            st.floats(0.01, 100.0, allow_nan=False),
        ),
        min_size=1,
        max_size=40,
        # Identity-keyed draws do not tie; with a tie the order would decide.
        unique_by=lambda row: row[1],
    ),
    st.randoms(use_true_random=False),
)
def test_positions_do_not_depend_on_record_order(rows, rng) -> None:
    strata, draws, weights = (np.asarray(column) for column in zip(*rows, strict=True))
    order = list(range(len(rows)))
    rng.shuffle(order)

    positions = stratified_positions(strata, draws, weights)
    shuffled = stratified_positions(strata[order], draws[order], weights[order])

    np.testing.assert_allclose(shuffled, positions[order])
    assert ((positions >= 0.0) & (positions < 1.0)).all()


def _people(n: int = 200):
    rng = np.random.default_rng(7)
    return {
        "age": rng.integers(0, 90, n),
        "is_male": rng.integers(0, 2, n).astype(bool),
        "person_ids": np.arange(1, n + 1) * 100 + 1,
        "weights": rng.uniform(500.0, 3_000.0, n),
    }


def test_months_spread_each_age_and_sex_evenly_over_the_year() -> None:
    people = _people(4_000)
    people["age"] = np.full(4_000, 40)
    people["is_male"] = np.zeros(4_000, dtype=bool)

    months = uk_months_since_last_birthday(**people)

    assert ((months >= 0.0) & (months < MONTHS_IN_YEAR)).all()
    # Within one age and sex, the weight in each month is a twelfth of the
    # whole, to within the largest single weight.
    share = (
        np.bincount(
            np.floor(months).astype(int), weights=people["weights"], minlength=12
        )
        / people["weights"].sum()
    )
    assert (
        np.abs(share - 1.0 / 12.0).max()
        < people["weights"].max() / (people["weights"].sum()) + 1e-12
    )


def test_months_are_keyed_on_the_person_not_on_the_row() -> None:
    people = _people()
    months = uk_months_since_last_birthday(**people)
    order = np.random.default_rng(3).permutation(len(months))

    again = uk_months_since_last_birthday(
        **{name: values[order] for name, values in people.items()}
    )

    np.testing.assert_array_equal(again, months[order])
    # Other identifiers deal the months otherwise.
    renamed = dict(people, person_ids=people["person_ids"] + 7)
    assert not np.array_equal(uk_months_since_last_birthday(**renamed), months)


def test_fractional_ages_refuse() -> None:
    people = _people(3)
    people["age"] = np.asarray([30.5, 40.0, 50.0])

    with pytest.raises(ValueError, match="whole ages"):
        uk_months_since_last_birthday(**people)
