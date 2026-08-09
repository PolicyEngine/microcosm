"""Shared fixtures for the microcosm-calibrate suite.

The builders construct small, seeded frames sized for CI speed (n<=300, a few
hundred optimization epochs). They cover the scenarios the behavioral contracts
need: a frame with feasible sum targets, and a "landmine" frame carrying a rare
high-value near-zero-weight donor whose weight an unbounded calibration wants to
detonate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

#: A person-per-household schema: one person per household, so household weights
#: are unambiguous and household-level measures align to the calibrated vector.
HOUSEHOLD_SCHEMA = EntitySchema(group_entities=("household",))


def household_frame(
    *,
    household_columns: dict[str, np.ndarray],
    weights: np.ndarray,
    kind: WeightKind = WeightKind.DESIGN,
    person_columns: dict[str, np.ndarray] | None = None,
    persons_per_household: np.ndarray | None = None,
) -> Frame:
    """Assemble a household-weighted frame.

    By default one person per household. ``persons_per_household`` (a per-household
    integer count) instead builds a multi-person frame, with ``person_columns``
    aligned to the expanded person rows.

    Args:
        household_columns: Columns on the household table (each length
            ``n_households``); a household measure target reads these.
        weights: Household weight vector (length ``n_households``).
        kind: Weight kind to tag the household weights with.
        person_columns: Optional person-level columns (aligned to the person
            rows the build produces).
        persons_per_household: Optional per-household person count; defaults to
            one person per household.

    Returns:
        A validated :class:`~microcosm.frame.Frame` with household weights.
    """
    n_households = len(weights)
    if persons_per_household is None:
        persons_per_household = np.ones(n_households, dtype=int)
    household_ids = np.arange(n_households, dtype="int64")
    person_household = np.repeat(household_ids, persons_per_household)
    n_persons = len(person_household)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n_persons, dtype="int64"),
            "person_household_id": person_household,
            **(person_columns or {}),
        }
    )
    household = pd.DataFrame({"household_id": household_ids, **household_columns})
    return Frame(
        {"person": person, "household": household},
        HOUSEHOLD_SCHEMA,
        {"household": Weights(values=weights, kind=kind)},
    )


@pytest.fixture
def feasible_frame():
    """Factory: a frame with sum targets it can hit exactly.

    A household frame of ``n`` one-person households, each with a positive
    ``income`` and a binary ``is_renter`` flag, at uniform design weight. Returns
    the frame plus the *true* weighted aggregates under those weights, so a test
    can ask for a shifted target and check the calibration reaches it.

    Returns:
        A callable ``make(seed=0, n=200, weight=1000.0)`` returning
        ``(frame, truths)`` where ``truths`` is a dict with ``population`` (the
        weighted household count), ``income`` (the weighted income sum), and
        ``renters`` (the weighted renter count).
    """

    def make(
        seed: int = 0, n: int = 200, weight: float = 1000.0
    ) -> tuple[Frame, dict[str, float]]:
        rng = np.random.default_rng(seed)
        income = rng.lognormal(10.5, 0.7, n)
        is_renter = (rng.random(n) < 0.35).astype(float)
        weights = np.full(n, weight)
        frame = household_frame(
            household_columns={
                "household_count": np.ones(n),
                "income": income,
                "is_renter": is_renter,
            },
            weights=weights,
        )
        truths = {
            "population": float(weights.sum()),
            "income": float((income * weights).sum()),
            "renters": float((is_renter * weights).sum()),
        }
        return frame, truths

    return make


@pytest.fixture
def landmine_frame():
    """Factory: a frame with one rare high-value near-zero-weight donor.

    Exactly one household carries a large ``capital_gains`` value at a near-zero
    weight; every other household has zero capital gains at a normal weight, plus
    a benign ``income`` everyone has. Because only the donor has capital gains, a
    capital-gains *total* target far above the current value can be reached
    **only** by inflating that one record's weight — the landmine. An unbounded
    calibration detonates it; a bounded one cannot.

    Returns:
        A callable ``make(seed=0, n=200, donor_value=1e6, normal_weight=1000.0)``
        returning ``(frame, donor_index, donor_value)``.
    """

    def make(
        seed: int = 0,
        n: int = 200,
        donor_value: float = 1_000_000.0,
        normal_weight: float = 1000.0,
    ) -> tuple[Frame, int, float]:
        rng = np.random.default_rng(seed)
        donor_index = 0
        capital_gains = np.zeros(n)
        capital_gains[donor_index] = donor_value
        income = rng.lognormal(10.0, 0.5, n)
        weights = np.full(n, normal_weight)
        weights[donor_index] = 1.0  # the near-zero weight
        frame = household_frame(
            household_columns={"capital_gains": capital_gains, "income": income},
            weights=weights,
        )
        return frame, donor_index, donor_value

    return make


@pytest.fixture
def multiperson_frame():
    """Factory: a household frame with multiple persons per household.

    Households hold several persons each (so person- and household-length
    vectors differ), every person carries an ``age``, and household weights are
    uniform. This is the frame a person-entity ``sum`` target must compile
    against while calibrating *household* weights — the cross-entity collapse
    path. Returns the per-person ``age`` and household membership so a test can
    compute the true weighted person aggregates by hand.

    Returns:
        A callable ``make(seed=0, n_households=60, max_persons=4, weight=1000.0)``
        returning ``(frame, age, person_household, weights)`` where ``age`` and
        ``person_household`` are person-aligned and ``weights`` is the household
        weight vector.
    """

    def make(
        seed: int = 0,
        n_households: int = 60,
        max_persons: int = 4,
        weight: float = 1000.0,
    ) -> tuple[Frame, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        persons_per_household = rng.integers(1, max_persons + 1, n_households)
        household_ids = np.arange(n_households, dtype="int64")
        person_household = np.repeat(household_ids, persons_per_household)
        n_persons = len(person_household)
        age = rng.integers(1, 90, n_persons).astype(float)
        weights = np.full(n_households, weight)
        frame = household_frame(
            household_columns={},
            weights=weights,
            person_columns={"age": age},
            persons_per_household=persons_per_household,
        )
        return frame, age, person_household, weights

    return make


@pytest.fixture
def multiperiod_frame():
    """Factory: a frame carrying two periods' income columns over one weight set.

    Each household has ``income_2026`` and ``income_2030`` (the same households,
    different period snapshots) at a uniform weight. A multi-period target set
    constrains the *same* household weight vector to both years' income totals —
    the "one weight per trajectory" rule. Returns the per-period true sums so a
    test can shift both and check both improve.

    Returns:
        A callable ``make(seed=0, n=200, weight=1000.0)`` returning
        ``(frame, truth_2026, truth_2030)``.
    """

    def make(
        seed: int = 0, n: int = 200, weight: float = 1000.0
    ) -> tuple[Frame, float, float]:
        rng = np.random.default_rng(seed)
        income_2026 = rng.lognormal(10.5, 0.6, n)
        # 2030 income tracks 2026 with growth and noise — same households.
        income_2030 = income_2026 * rng.normal(1.15, 0.05, n)
        income_2030 = np.clip(income_2030, 1.0, None)
        weights = np.full(n, weight)
        frame = household_frame(
            household_columns={
                "income_2026": income_2026,
                "income_2030": income_2030,
            },
            weights=weights,
        )
        return (
            frame,
            float((income_2026 * weights).sum()),
            float((income_2030 * weights).sum()),
        )

    return make
