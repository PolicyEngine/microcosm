"""Shared fixtures for the populace-fit suite.

The builders here construct small, seeded frames whose *weighted* conditional
distribution differs sharply from the unweighted one — the microimpute#196
shape — so the behavioral contracts can assert that weighting actually moves the
draws. Everything is sized for CI speed (n=5000, small forests).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.frame import EntitySchema, Frame, WeightKind, Weights

#: Person-only schema (person + a one-person-per household, so person weights
#: are unambiguous and every predictor/target lives on the person entity).
PERSON_SCHEMA = EntitySchema(group_entities=("household",))

#: Small but statistically stable sample size for the weighted-fit contracts.
CONTRACT_N = 5000


def _person_household_frame(
    columns: dict[str, np.ndarray],
    weights: np.ndarray,
    *,
    kind: WeightKind = WeightKind.DESIGN,
) -> Frame:
    """Assemble a one-person-per-household frame with person-level weights.

    Args:
        columns: Person-level columns (each length ``n``).
        weights: Person weight vector (length ``n``).
        kind: Weight kind to tag the vector with.

    Returns:
        A validated :class:`~populace.frame.Frame`.
    """
    n = len(weights)
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype="int64"),
            "person_household_id": np.arange(n, dtype="int64"),
            **columns,
        }
    )
    household = pd.DataFrame({"household_id": np.arange(n, dtype="int64")})
    return Frame(
        {"person": person, "household": household},
        PERSON_SCHEMA,
        {"person": Weights(values=weights, kind=kind)},
    )


@pytest.fixture
def make_person_frame():
    """Factory: assemble a one-person-per-household frame from raw columns.

    A thin wrapper over the module builder, for tests that need a bespoke target
    distribution (e.g. a degenerate-zero target).

    Returns:
        A callable ``make(columns, weights=None)`` returning a
        :class:`~populace.frame.Frame`; ``weights`` defaults to a constant
        vector sized to the columns.
    """

    def make(columns: dict[str, np.ndarray], weights: np.ndarray | None = None):
        n = len(next(iter(columns.values())))
        if weights is None:
            weights = np.full(n, 5.0)
        return _person_household_frame(columns, weights)

    return make


@pytest.fixture
def weight_correlated_frame():
    """Factory: a frame whose target is large exactly where weight is small.

    The microimpute#196 repro. About 20% of rows are a "low-weight, huge-value"
    regime: their target is ~15x the bulk and they carry a small design weight,
    so the *unweighted* mean is dominated by them while the *weighted* mean is
    not. A fit that honors weights must reproduce the weighted mean; one that
    ignores them reproduces the unweighted mean. The two truths differ by well
    over 3x, so the contract can separate them cleanly.

    Crucially, the predictors (``age``, ``is_male``) are drawn *independently*
    of the regime, so they do not reveal which rows are the huge-value ones.
    That is exactly the #196 condition: the weight correlates with the target in
    the part of its variance the predictors do **not** explain, so honoring the
    weight is the only way to recover the weighted conditional. (Were the regime
    identifiable from the predictors, each leaf would be pure and reweighting it
    would not move its value — there would be no weighting effect to test.)

    Returns:
        A callable ``make(seed=0, n=CONTRACT_N, low_fraction=0.2)`` returning
        ``(frame, target_values, weight_values)``.
    """

    def make(
        seed: int = 0,
        n: int = CONTRACT_N,
        low_fraction: float = 0.2,
    ) -> tuple[Frame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        # Predictors are independent of the regime: they carry no signal about
        # which rows are the huge-value, low-weight ones.
        age = rng.integers(18, 75, n).astype(float)
        is_male = rng.integers(0, 2, n).astype(float)
        low_weight = rng.random(n) < low_fraction
        # Huge values (~15x the bulk) for the low-weight rows; the wide gap
        # keeps the unweighted/weighted separation above 3x even after the mild
        # shrinkage a forest applies at the top of the support.
        target = np.where(
            low_weight,
            rng.normal(600_000.0, 40_000.0, n),
            rng.normal(40_000.0, 7_000.0, n),
        )
        target = np.clip(target, 0.0, None)
        weights = np.where(low_weight, 1.0, 50.0)
        frame = _person_household_frame(
            {"age": age, "is_male": is_male, "target": target}, weights
        )
        return frame, target, weights

    return make


@pytest.fixture
def rare_positive_frame():
    """Factory: a target positive on a few low-weight rows, zero elsewhere.

    The reviewer's gate-bug repro. About ``n_positive`` rows carry a positive
    value at weight 1; every other row is zero at weight 50, so the *weighted*
    positive share is tiny (~4e-5 at the defaults). An n-of-n weighted bootstrap
    of the gate draws each positive row with probability ~4e-5, so the resampled
    labels almost always contain only the zero class — the gate collapses to one
    class and can never draw the positive sign. Weighting the gate directly by
    ``sample_weight`` keeps every row, so both classes survive.

    The predictors carry a weak signal about which rows are positive (so the
    forest can place them), but the positive rows remain rare enough under the
    weights that the bootstrap would drop them.

    Returns:
        A callable ``make(seed=0, n=CONTRACT_N, n_positive=10)`` returning
        ``(frame, target_values, weight_values)``.
    """

    def make(
        seed: int = 0,
        n: int = CONTRACT_N,
        n_positive: int = 10,
    ) -> tuple[Frame, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        positive_idx = rng.choice(n, size=n_positive, replace=False)
        is_positive = np.zeros(n, dtype=bool)
        is_positive[positive_idx] = True
        # A predictor weakly correlated with being positive, so the gate has a
        # signal to learn (without it, even a correct gate could not separate
        # the classes and the test would not isolate the bootstrap bug).
        signal = rng.normal(0.0, 1.0, n)
        signal[is_positive] += 2.5
        noise = rng.normal(0.0, 1.0, n)
        target = np.where(is_positive, np.abs(rng.normal(100_000.0, 10_000.0, n)), 0.0)
        weights = np.where(is_positive, 1.0, 50.0)
        frame = _person_household_frame(
            {"signal": signal, "noise": noise, "target": target}, weights
        )
        return frame, target, weights

    return make


@pytest.fixture
def zero_inflated_frame():
    """Factory: a frame with a zero-inflated, sign-mixed target.

    The target is zero for a controlled share of rows, positive for most of the
    rest, and negative for a small tail — the three-sign regime. Used to assert
    the gate preserves the zero mass and both signs.

    Returns:
        A callable ``make(seed=0, n=CONTRACT_N, zero_fraction=0.4,
        neg_fraction=0.1)`` returning ``(frame, target_values)``.
    """

    def make(
        seed: int = 0,
        n: int = CONTRACT_N,
        zero_fraction: float = 0.4,
        neg_fraction: float = 0.1,
    ) -> tuple[Frame, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, n)
        u = rng.random(n)
        target = np.zeros(n, dtype=float)
        pos = u >= zero_fraction + neg_fraction
        neg = (u >= zero_fraction) & (u < zero_fraction + neg_fraction)
        target[pos] = np.abs(rng.normal(10_000.0, 2_000.0, n))[pos]
        target[neg] = -np.abs(rng.normal(3_000.0, 800.0, n))[neg]
        frame = _person_household_frame({"x": x, "target": target}, np.full(n, 10.0))
        return frame, target

    return make


@pytest.fixture
def correlated_targets_frame():
    """Factory: two targets with a strong built-in correlation.

    ``second`` is a noisy linear function of ``first`` that the predictors do
    not explain on their own, so only a model that *chains* (conditions
    ``second`` on the imputed ``first``) can reproduce their correlation.

    Returns:
        A callable ``make(seed=0, n=CONTRACT_N)`` returning ``(frame, rho)``
        where ``rho`` is the true Pearson correlation of the two targets.
    """

    def make(seed: int = 0, n: int = CONTRACT_N) -> tuple[Frame, float]:
        rng = np.random.default_rng(seed)
        x = rng.normal(0.0, 1.0, n)
        first = 100.0 + 5.0 * x + rng.normal(0.0, 20.0, n)
        # second tracks first, with noise the predictor x cannot account for.
        second = 3.0 * first + rng.normal(0.0, 30.0, n)
        rho = float(np.corrcoef(first, second)[0, 1])
        frame = _person_household_frame(
            {"x": x, "first": first, "second": second}, np.full(n, 10.0)
        )
        return frame, rho

    return make
