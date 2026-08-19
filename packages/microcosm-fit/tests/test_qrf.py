"""Mechanics of the canonical model: regimes, gates, chaining, determinism.

These tests pin the structural guarantees of the regime-gated, chained,
weighted-bootstrap QRF that are not the weighted-mean contract itself: that the
gate preserves a zero-inflated target's zero mass and sign structure, that
chaining reproduces a cross-target correlation, that the draw count matches the
input, and that a fixed seed makes a freshly fitted model reproducible.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import QRFChainState, Regime, RegimeGatedQRF, fit
from microcosm.fit.qrf import (
    _DISCRETE_Y_LEAF_BOUND,
    _fit_n_jobs,
    _rng_from_state_json,
    _rng_state_json,
    detect_regime,
)


def test_rng_state_restore_uses_no_ambient_default_rng(monkeypatch) -> None:
    source = np.random.default_rng(1_947)
    source.random(11)
    state = _rng_state_json(source)
    expected = source.random(16)

    def refuse_ambient_entropy(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ambient default_rng access")

    monkeypatch.setattr(np.random, "default_rng", refuse_ambient_entropy)
    restored = _rng_from_state_json(state, stream="test")

    np.testing.assert_array_equal(restored.random(16), expected)


# ----------------------------------------------------------------------------
# Regime detection: structural, unweighted support classification
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([1.0, 2.0, 3.0], Regime.POSITIVE_ONLY),
        ([-1.0, -2.0, -3.0], Regime.NEGATIVE_ONLY),
        ([0.0, 1.0, 0.0, 2.0], Regime.ZERO_INFLATED_POSITIVE),
        ([0.0, -1.0, 0.0, -2.0], Regime.ZERO_INFLATED_NEGATIVE),
        ([-1.0, 1.0, 2.0], Regime.SIGN_ONLY),
        ([-1.0, 0.0, 1.0], Regime.THREE_SIGN),
        ([0.0, 0.0, 0.0], Regime.DEGENERATE_ZERO),
    ],
)
def test_detect_regime_classifies_support(values, expected) -> None:
    """The regime is read from which signs appear in the (unweighted) support."""
    assert detect_regime(np.array(values), zero_atol=1e-6) == expected


def test_regime_detection_ignores_weights(weight_correlated_frame) -> None:
    """Regime detection is structural: the same regime weighted or not.

    Which signs *exist* is a property of the variable, not the weighting, so a
    weighted and an unweighted fit of the same target detect the same regime.
    """
    frame, _, _ = weight_correlated_frame(seed=0)
    weighted = fit(frame, ["age", "is_male"], ["target"], n_estimators=20, seed=0)
    unweighted = fit(
        frame, ["age", "is_male"], ["target"], weights="none", n_estimators=20, seed=0
    )
    assert weighted.regimes() == unweighted.regimes()
    assert weighted.regimes()["target"] == Regime.POSITIVE_ONLY


# ----------------------------------------------------------------------------
# Gates: a zero-inflated target's draws preserve zero mass and sign structure
# ----------------------------------------------------------------------------


def test_zero_inflated_draws_preserve_zero_mass_and_signs(
    zero_inflated_frame,
) -> None:
    """The gate reproduces the zero share and both signs, with no zero-crossing.

    A single ungated regressor would smear the zeros into small nonzero values
    and interpolate across the zero gap. The gate must instead (1) put roughly
    the training share of draws at exactly zero, (2) keep some draws on each
    sign, and (3) never produce a value strictly between the negative and
    positive clusters that no training row occupies.
    """
    frame, target = zero_inflated_frame(seed=0)
    fitted = fit(frame, ["x"], ["target"], n_estimators=80, seed=0)
    assert fitted.regimes()["target"] == Regime.THREE_SIGN

    draws = fitted.predict(frame)["target"].to_numpy()

    true_zero_share = float((np.abs(target) <= 1e-6).mean())
    draw_zero_share = float((draws == 0.0).mean())
    # Zero mass is preserved (not smeared away), within a sampling tolerance.
    assert draw_zero_share == pytest.approx(true_zero_share, abs=0.07)

    # Both signs survive — the negative tail is not dropped.
    assert (draws > 0).any()
    assert (draws < 0).any()

    # No zero-crossing interpolation: drawn magnitudes lie in the support of
    # their sign, never in the empty gap between the clusters.
    min_train_pos = float(target[target > 1e-6].min())
    max_train_neg = float(target[target < -1e-6].max())
    nonzero = draws[draws != 0.0]
    assert (nonzero[nonzero > 0] >= 0.5 * min_train_pos).all()
    assert (nonzero[nonzero < 0] <= 0.5 * max_train_neg).all()


def test_gate_retains_a_rare_low_weight_class_and_draws_it(
    rare_positive_frame,
) -> None:
    """The gate keeps a vanishingly-rare sign class and can draw it.

    The reviewer's repro: ~10 positive rows at weight 1 among ~4990 zeros at
    weight 50 (weighted positive share ~4e-5). The old gate was fit on an n-of-n
    weighted bootstrap, which drew the positive rows with probability ~4e-5 and
    so produced a single-class gate that drew the positive sign with probability
    zero (0 positive draws across millions). Fitting the gate directly with
    ``sample_weight`` keeps every row, so:

    (a) the fitted gate retains both classes (0 and 1); and
    (b) across seeds the model produces a positive draw at least once — the
        positive regime is reachable, not collapsed to zero.
    """
    frame, target, _ = rare_positive_frame(seed=0)
    # The target is zero-inflated positive: zeros plus a positive tail.
    fitted = fit(frame, ["signal", "noise"], ["target"], n_estimators=60, seed=0)
    assert fitted.regimes()["target"] == Regime.ZERO_INFLATED_POSITIVE

    # (a) The gate retained both sign classes (0 and 1); the bootstrap dropped 1.
    gate = fitted._target_models["target"].gate
    assert set(np.asarray(gate.classes_).tolist()) == {0, 1}

    # (b) Positive draws appear. One predict is ~n*4e-5 ~ 0.2 expected positives,
    # so aggregate several seeds to make the assertion robust: the share is small
    # but emphatically not zero (the bug drew exactly zero, forever).
    total_positive = 0
    for _ in range(8):
        draws = fitted.predict(frame)["target"].to_numpy()
        total_positive += int((draws > 0).sum())
    assert total_positive > 0, (
        "the rare positive regime was never drawn; the gate collapsed to the "
        "zero class (the reproduced bootstrap bug)"
    )


def test_gate_consistency_guard_rejects_a_dropped_class(make_person_frame) -> None:
    """The internal-consistency guard fires if the gate drops a training class.

    Regime detection found a sign class, but the fitted gate's ``classes_`` lack
    it — drawing it at probability zero would silently lose that sign. The fit
    must raise rather than ship an inconsistent gate. We force the inconsistency
    by monkeypatching the gate factory to return a classifier that always fits a
    single class.
    """
    import microcosm.fit.qrf as qrf_module

    n = 400
    rng = np.random.default_rng(0)
    # A genuine zero-inflated-positive target (so a gate is built).
    x = rng.normal(size=n)
    target = np.where(rng.random(n) < 0.3, np.abs(rng.normal(1000.0, 100.0, n)), 0.0)
    frame = make_person_frame({"x": x, "target": target})

    class _SingleClassGate:
        """A stand-in classifier that always fits only the zero class."""

        def __init__(self, *args, **kwargs) -> None:
            self.classes_ = np.array([0])

        def fit(self, x, y, sample_weight=None):  # noqa: A002 - mirror sklearn
            return self

    original = qrf_module._make_gate
    qrf_module._make_gate = lambda seed: _SingleClassGate()
    try:
        with pytest.raises(ValueError, match="Sign gate dropped class"):
            fit(frame, ["x"], ["target"], n_estimators=10, seed=0)
    finally:
        qrf_module._make_gate = original


def test_fit_rng_orders_gate_then_positive_then_negative_forests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin every fit-child event around the estimator-private RNGs."""

    import microcosm.fit.qrf as qrf_module

    events: list[tuple[object, ...]] = []

    class SpyRng:
        def __init__(self) -> None:
            self.seeds = iter((101, 202, 303))

        def integers(self, low: int, high: int) -> int:
            value = next(self.seeds)
            events.append(("integer", low, high, value))
            return value

        def choice(
            self,
            population: int,
            *,
            size: int,
            replace: bool,
            p: np.ndarray,
        ) -> np.ndarray:
            events.append(("choice", population, size, replace, tuple(np.asarray(p))))
            return np.arange(size) % population

    class FakeGate:
        def __init__(self, seed: int) -> None:
            events.append(("gate_init", seed))
            self.classes_ = np.array([-1, 0, 1])

        def fit(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            *,
            sample_weight: np.ndarray,
        ) -> FakeGate:
            events.append(
                (
                    "gate_fit",
                    tuple(labels),
                    tuple(np.asarray(sample_weight)),
                    features.shape,
                )
            )
            return self

    class FakeForest:
        def __init__(self, **kwargs: object) -> None:
            events.append(("forest_init", kwargs["random_state"]))

        def fit(self, features: np.ndarray, target: np.ndarray) -> FakeForest:
            events.append(("forest_fit", tuple(target), features.shape))
            return self

    monkeypatch.setattr(qrf_module, "_make_gate", FakeGate)
    monkeypatch.setattr(qrf_module, "RandomForestQuantileRegressor", FakeForest)
    features = np.arange(10, dtype=np.float64).reshape(5, 2)
    target = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    RegimeGatedQRF(max_samples_leaf=7)._fit_target(
        features=features,
        y=target,
        columns=("x", "z"),
        weights=weights,
        rng=SpyRng(),  # type: ignore[arg-type]
    )

    assert events == [
        ("integer", 0, 2**31 - 1, 101),
        ("gate_init", 101),
        ("gate_fit", (-1, -1, 0, 1, 1), tuple(weights), (5, 2)),
        ("integer", 0, 2**31 - 1, 202),
        ("choice", 2, 2, True, (4 / 9, 5 / 9)),
        ("forest_init", 202),
        ("forest_fit", (1.0, 2.0), (2, 2)),
        ("integer", 0, 2**31 - 1, 303),
        ("choice", 2, 2, True, (1 / 3, 2 / 3)),
        ("forest_init", 303),
        ("forest_fit", (-2.0, -1.0), (2, 2)),
    ]


def test_degenerate_zero_target_draws_all_zero(make_person_frame) -> None:
    """A target that is constant zero in training draws exactly zero."""
    n = 200
    rng = np.random.default_rng(0)
    frame = make_person_frame({"x": rng.normal(size=n), "target": np.zeros(n)})
    fitted = fit(frame, ["x"], ["target"], n_estimators=10, seed=0)
    assert fitted.regimes()["target"] == Regime.DEGENERATE_ZERO
    draws = fitted.predict(frame)["target"].to_numpy()
    assert (draws == 0.0).all()


def test_nan_target_raises_naming_the_column_and_count(make_person_frame) -> None:
    """A target with NaNs is refused at fit, naming the column and NaN count.

    Without the guard, a NaN target is silently relabeled to the zero class —
    the sign labels (``y > atol`` / ``y < -atol``) are both False for NaN — so
    missing values masquerade as structural zeros, NaN-blind. Fit must instead
    raise, naming the offending column and how many values are non-finite.
    """
    n = 200
    rng = np.random.default_rng(0)
    target = np.abs(rng.normal(1000.0, 100.0, n))
    target[5] = np.nan
    target[17] = np.nan
    target[42] = np.nan
    frame = make_person_frame({"x": rng.normal(size=n), "target": target})

    with pytest.raises(ValueError, match=r"Target column 'target' contains 3 "):
        fit(frame, ["x"], ["target"], n_estimators=10, seed=0)


def test_inf_target_is_also_refused(make_person_frame) -> None:
    """Infinity in a target is non-finite too, and refused the same way."""
    n = 120
    rng = np.random.default_rng(1)
    target = np.abs(rng.normal(500.0, 50.0, n))
    target[3] = np.inf
    frame = make_person_frame({"x": rng.normal(size=n), "target": target})
    with pytest.raises(ValueError, match="non-finite"):
        fit(frame, ["x"], ["target"], n_estimators=10, seed=0)


# ----------------------------------------------------------------------------
# Chaining: a target conditioned on a prior imputed target tracks correlation
# ----------------------------------------------------------------------------


def test_chaining_reproduces_cross_target_correlation(
    correlated_targets_frame,
) -> None:
    """Chained draws of two correlated targets keep their correlation.

    ``second`` is a noisy linear function of ``first`` that the predictor ``x``
    alone cannot reconstruct. Because the model chains — conditioning
    ``second`` on the drawn ``first`` — the drawn pair stays strongly
    correlated. Without chaining, the two would be independent given ``x`` and
    their drawn correlation would collapse.
    """
    frame, true_rho = correlated_targets_frame(seed=0)
    assert true_rho > 0.9  # the fixture builds a strong correlation

    fitted = fit(frame, ["x"], ["first", "second"], n_estimators=80, seed=0)
    draws = fitted.predict(frame)
    drawn_rho = float(np.corrcoef(draws["first"], draws["second"])[0, 1])

    # The chained draws retain most of the correlation.
    assert drawn_rho > 0.8


def test_chaining_columns_grow_along_the_chain(correlated_targets_frame) -> None:
    """Each target is fit on the predictors plus the targets chained before it."""
    frame, _ = correlated_targets_frame(seed=0)
    fitted = fit(frame, ["x"], ["first", "second"], n_estimators=10, seed=0)
    models = fitted._target_models
    # first conditions on x only; second on x and the drawn first.
    assert models["first"].columns == ("x",)
    assert models["second"].columns == ("x", "first")


# ----------------------------------------------------------------------------
# Tail-draw fidelity: full leaf samples, interpolation, drawable extremes
# ----------------------------------------------------------------------------

#: The heavy-tail threshold for the weight_correlated fixture: the bulk sits at
#: ~40k and the rare regime at ~600k, so 300k cleanly separates them.
_TAIL_THRESHOLD = 300_000.0

#: Independent predict() rounds averaged per tail-share estimate, and
#: independent fit seeds averaged over: one round's share noise (~0.0011)
#: matches the ~0.0015 nearest-snap bias the contract must detect, and a
#: single fitted forest can sit ~0.0013 low for every draw round (forest-level
#: wobble — observed on x86 CI). Averaging across fits and draws puts the
#: bias gap near 4 sigma of the comparison.
_TAIL_DRAW_ROUNDS = 3
_TAIL_FIT_SEEDS = (1, 2, 3, 4)


def test_tail_share_tracks_weighted_truth_and_beats_nearest_snap(
    weight_correlated_frame,
) -> None:
    """Weighted-draw tail share matches truth and beats the pre-fix nearest-snap.

    The fixture's weighted-population share above 300k is ~0.0050. The pre-fix
    forest kept one sample per leaf (``max_samples_leaf=1``) and snapped each
    draw to the nearest of 201 interior grid points, which thinned the
    conditional and undershot the tail by roughly a third (share ~0.0035). The
    fix — ``max_samples_leaf=None`` (all leaf samples) plus linear
    interpolation at the exact per-row quantile — recovers the tail. The
    contract:

    (a) the fix's weighted tail share is within ~2x of the weighted truth; and
    (b) averaged over draws, it is closer to the truth than the nearest-snap
        baseline.

    Both shares are averaged over ``_TAIL_DRAW_ROUNDS`` independent
    ``predict()`` rounds (successive calls advance the fitted model's RNG). A
    single round's share has sd ~ ``sqrt(p(1-p)/n_eff)`` ~ 0.0011 at the
    fixture's n=5000 — the same order as the ~0.0015 nearest-snap bias under
    test — so a one-round comparison is a platform coin flip (it failed
    exactly that way on x86 CI while passing on arm64). Averaging K rounds
    shrinks the comparison noise by ~sqrt(K), putting the bias gap near 3
    sigma.
    """
    frame, target, weights = weight_correlated_frame(seed=0)
    truth_share = float(
        np.average((target > _TAIL_THRESHOLD).astype(float), weights=weights)
    )
    # The fixture must actually have a tail to test (guard against a fixture
    # change collapsing it).
    assert truth_share > 0.0

    def averaged_tail_share(model) -> float:
        shares = [
            float(
                np.average(
                    (
                        model.predict(frame)["target"].to_numpy() > _TAIL_THRESHOLD
                    ).astype(float),
                    weights=weights,
                )
            )
            for _ in range(_TAIL_DRAW_ROUNDS)
        ]
        return float(np.mean(shares))

    fix_share = float(
        np.mean(
            [
                averaged_tail_share(
                    fit(frame, ["age", "is_male"], ["target"], n_estimators=100, seed=s)
                )
                for s in _TAIL_FIT_SEEDS
            ]
        )
    )

    # The nearest-snap, one-sample-per-leaf baseline (the pre-fix behavior),
    # reconstructed by monkeypatching the draw read-out and forcing
    # max_samples_leaf=1.
    import microcosm.fit.qrf as qrf_module

    interior_grid = np.linspace(1.0 / 202.0, 1.0 - 1.0 / 202.0, 201)

    def nearest_snap_draw(self, frame_in, quantiles):
        feats = frame_in.loc[:, list(self.columns)].to_numpy(dtype=np.float64)
        preds = np.asarray(
            self.model.predict(feats, quantiles=list(interior_grid))
        ).reshape(len(feats), len(interior_grid))
        idx = np.clip(
            np.rint(quantiles * (len(interior_grid) - 1)).astype(int),
            0,
            len(interior_grid) - 1,
        )
        return preds[np.arange(len(feats)), idx]

    original_draw = qrf_module._Forest.draw
    qrf_module._Forest.draw = nearest_snap_draw
    try:
        baseline_share = float(
            np.mean(
                [
                    averaged_tail_share(
                        fit(
                            frame,
                            ["age", "is_male"],
                            ["target"],
                            n_estimators=100,
                            max_samples_leaf=1,
                            seed=s,
                        )
                    )
                    for s in _TAIL_FIT_SEEDS
                ]
            )
        )
    finally:
        qrf_module._Forest.draw = original_draw

    # (a) within ~2x of the weighted truth, both directions.
    assert 0.5 * truth_share <= fix_share <= 2.0 * truth_share, (
        f"fix tail share {fix_share:.5f} not within 2x of truth {truth_share:.5f}"
    )
    # (b) materially closer to truth than the nearest-snap baseline.
    assert abs(fix_share - truth_share) < abs(baseline_share - truth_share), (
        f"fix share {fix_share:.5f} (err "
        f"{abs(fix_share - truth_share):.5f}) is not closer to truth "
        f"{truth_share:.5f} than the nearest-snap baseline {baseline_share:.5f} "
        f"(err {abs(baseline_share - truth_share):.5f})"
    )


def test_high_quantile_draw_reaches_the_observed_max() -> None:
    """A draw at q->1 reaches the observed conditional max via the grid endpoint.

    The pre-fix grid stopped at q=0.995 and snapped to it, so a lone extreme —
    the observed maximum, which sits in the conditional's top atom above
    q=0.995 — was unreachable: q=1 is the observed max, not extrapolation, and
    the grid excluded it. The fix prepends/appends grid points adjacent to 0 and
    1, so the observed max is drawable. This is a direct probe of the forest's
    draw at q=1 against the interior-only (winsorized) grid.
    """
    from quantile_forest import RandomForestQuantileRegressor

    from microcosm.fit.qrf import _QUANTILE_GRID, _Forest

    rng = np.random.default_rng(0)
    n = 2000
    # All rows share one feature value, so every tree has a single leaf holding
    # the whole sample — the conditional is the marginal, with a clean max.
    x = np.zeros((n, 1))
    y = rng.normal(100.0, 10.0, n)
    observed_max = 1000.0
    y[0] = observed_max  # a lone extreme: the observed conditional maximum
    model = RandomForestQuantileRegressor(
        n_estimators=50, max_samples_leaf=None, random_state=0
    )
    model.fit(x, y)
    forest = _Forest(model=model, columns=("x",))

    import pandas as pd

    probe = pd.DataFrame({"x": np.zeros(1)})
    draw_at_one = forest.draw(probe, np.array([1.0]))[0]
    # The endpoint grid reaches the observed max (within sampling tolerance).
    assert draw_at_one == pytest.approx(observed_max, rel=0.02)

    # The interior-only grid (the pre-fix winsorized grid) cannot: its top
    # quantile q=0.995 reads far below the lone extreme.
    interior_grid = np.linspace(1.0 / 202.0, 1.0 - 1.0 / 202.0, 201)
    interior_top = float(
        np.asarray(model.predict(np.zeros((1, 1)), quantiles=[interior_grid[-1]]))[0]
    )
    assert interior_top < 0.5 * observed_max
    # And the endpoint of the live grid is the one that closes the gap.
    assert _QUANTILE_GRID[-1] > interior_grid[-1]


# ----------------------------------------------------------------------------
# Shape and determinism
# ----------------------------------------------------------------------------


def test_predict_row_count_matches_input(weight_correlated_frame) -> None:
    """One drawn row per input row, with the input index preserved."""
    frame, _, _ = weight_correlated_frame(seed=0, n=300)
    fitted = fit(frame, ["age", "is_male"], ["target"], n_estimators=10, seed=0)

    # From a Frame.
    drawn_frame = fitted.predict(frame)
    assert len(drawn_frame) == frame.n("person")
    assert list(drawn_frame.columns) == ["target"]

    # From a bare DataFrame subset of arbitrary length and a non-range index.
    subset = frame.table("person").iloc[:50].set_axis(range(100, 150))
    drawn_df = fitted.predict(subset)
    assert len(drawn_df) == 50
    assert drawn_df.index.tolist() == list(range(100, 150))


def test_fixed_seed_is_deterministic(weight_correlated_frame) -> None:
    """Two models fit with the same seed draw identically on the first predict."""
    frame, _, _ = weight_correlated_frame(seed=0, n=400)
    a = fit(frame, ["age", "is_male"], ["target"], n_estimators=25, seed=7)
    b = fit(frame, ["age", "is_male"], ["target"], n_estimators=25, seed=7)
    np.testing.assert_array_equal(
        a.predict(frame)["target"].to_numpy(),
        b.predict(frame)["target"].to_numpy(),
    )


def test_successive_predicts_draw_independently(weight_correlated_frame) -> None:
    """Repeated predicts on one fitted model give independent draws.

    The draw RNG advances across calls, so a second predict is not a carbon
    copy of the first (draws are samples, not a point estimate).
    """
    frame, _, _ = weight_correlated_frame(seed=0, n=400)
    fitted = fit(frame, ["age", "is_male"], ["target"], n_estimators=25, seed=7)
    first = fitted.predict(frame)["target"].to_numpy()
    second = fitted.predict(frame)["target"].to_numpy()
    assert not np.array_equal(first, second)


def test_different_seeds_give_different_draws(weight_correlated_frame) -> None:
    """Changing the seed changes the draws (the seed actually seeds)."""
    frame, _, _ = weight_correlated_frame(seed=0, n=400)
    a = fit(frame, ["age", "is_male"], ["target"], n_estimators=25, seed=1)
    b = fit(frame, ["age", "is_male"], ["target"], n_estimators=25, seed=2)
    assert not np.array_equal(
        a.predict(frame)["target"].to_numpy(),
        b.predict(frame)["target"].to_numpy(),
    )


def test_fit_and_draw_use_independent_rng_streams(weight_correlated_frame) -> None:
    """The draw RNG is an independent SeedSequence child of the model seed.

    Before the fix the fitted model was seeded with the raw model seed, so its
    draw uniforms were bit-identical to the fit's bootstrap-selection uniforms
    (max |diff| = 0). The draw stream must instead come from the second spawned
    child, independent of the fit's resampling, yet still reproducible.
    """
    frame, _, _ = weight_correlated_frame(seed=0, n=400)
    seed = 7
    fitted = fit(frame, ["age", "is_male"], ["target"], n_estimators=20, seed=seed)
    _, draw_child = np.random.SeedSequence(seed).spawn(2)
    expected = np.random.default_rng(draw_child).random(5)
    actual = fitted._rng.random(5)  # initial draw-stream state, before any predict
    np.testing.assert_array_equal(actual, expected)
    # And it must NOT be the raw-seed stream that the fit also used.
    assert not np.array_equal(actual, np.random.default_rng(seed).random(5))


def test_weighted_gate_reproduces_population_zero_share_not_sample(
    make_person_frame,
) -> None:
    """The zero gate reproduces the *weighted* zero-share, not the sample's.

    The sample is half zeros, but zeros carry low weight and positives high
    weight, so the population (weighted) zero-share is 0.1 while the sample's is
    0.5. The predictor is independent of zero-ness, so a correctly weighted gate
    reproduces the marginal weighted zero rate. A gate fit unweighted (or on an
    unweighted resample) would reproduce 0.5.
    """
    rng = np.random.default_rng(0)
    n = 6000
    is_zero = np.arange(n) < n // 2
    target = np.where(is_zero, 0.0, 1.0 + rng.random(n))
    weights = np.where(is_zero, 1.0, 9.0)  # weighted zero-share = 3000/30000 = 0.1
    x = rng.normal(size=n)  # predictor independent of zero-ness
    frame = make_person_frame({"x": x, "target": target}, weights=weights)
    draws = fit(frame, ["x"], ["target"], n_estimators=80, seed=0).predict(frame)
    draw_zero_share = float((draws["target"].to_numpy() == 0).mean())
    assert abs(draw_zero_share - 0.1) < 0.05  # tracks the weighted population
    assert abs(draw_zero_share - 0.5) > 0.2  # decisively not the sample share


def test_forests_keep_all_leaf_samples(make_person_frame) -> None:
    """max_samples_leaf=None is pinned independently of the readout fix.

    quantile-forest defaults to one sample per leaf, which thins each row's
    conditional to ~n_estimators atoms and undershoots tail mass. The tail
    fidelity fix sets max_samples_leaf=None; without this explicit pin a revert
    to the default passes the rest of the suite (the tail test's baseline also
    uses leaf=1, so interpolation alone clears it). This pins the leaf component.
    """
    rng = np.random.default_rng(0)
    n = 400
    frame = make_person_frame(
        {"x": rng.normal(size=n), "target": np.abs(rng.normal(size=n)) + 0.1}
    )
    fitted = fit(frame, ["x"], ["target"], n_estimators=8, seed=0)
    forest = fitted._target_models["target"].positive.model
    assert forest.max_samples_leaf is None


def test_near_discrete_targets_get_bounded_leaf_samples(make_person_frame) -> None:
    """A few-atom target trades full leaf retention for a bounded subsample.

    With ``max_samples_leaf=None`` a near-discrete ``y`` (here six atoms, one
    dominant) stops splitting once nodes are y-pure, so a dominant atom piles
    hundreds of thousands of identical samples into single leaves and the
    forest's dense per-leaf store (``trees x leaves x largest_leaf``) reaches
    tens of GB — the QBI ``*_would_be_qualified`` targets jetsammed the Build M
    base at 57-67GB per worker this way. The guard caps such targets' leaf
    storage at _DISCRETE_Y_LEAF_BOUND; their leaf conditionals hold at most a
    few dozen atoms, so the capped subsample reproduces every leaf quantile to
    multinomial noise.
    """
    rng = np.random.default_rng(0)
    n = 600
    atoms = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    target = atoms[rng.choice(len(atoms), size=n, p=[0.05, 0.05, 0.7, 0.1, 0.05, 0.05])]
    frame = make_person_frame({"x": rng.normal(size=n), "target": target})
    fitted = fit(frame, ["x"], ["target"], n_estimators=8, seed=0)
    forest = fitted._target_models["target"].positive.model
    assert forest.max_samples_leaf == _DISCRETE_Y_LEAF_BOUND


def test_near_discrete_bound_never_overrides_explicit_config(
    make_person_frame,
) -> None:
    """An explicit ``max_samples_leaf`` wins over the near-discrete guard.

    The guard exists to rescue the ``None`` (keep-everything) default from
    pathological targets; a caller who pinned a leaf budget said what they
    meant, and silently rewriting it would make the recorded chain config lie.
    """
    rng = np.random.default_rng(0)
    n = 600
    atoms = np.array([0.0, 0.5, 1.0])
    target = atoms[rng.choice(len(atoms), size=n, p=[0.2, 0.6, 0.2])]
    frame = make_person_frame({"x": rng.normal(size=n), "target": target})
    fitted = RegimeGatedQRF(n_estimators=8, seed=0, max_samples_leaf=7).fit(
        frame, ["x"], ["target"]
    )
    forest = fitted._target_models["target"].positive.model
    assert forest.max_samples_leaf == 7


def test_near_discrete_bound_draws_are_seed_deterministic(make_person_frame) -> None:
    """The bounded-leaf path draws identically across identical fits.

    The guard is a pure function of the pre-bootstrap ``y``, consumes no RNG,
    and the capped forest is seeded — so two fits from the same seed must agree
    bit for bit, exactly like the uncapped path. This is what lets a resumed
    per-target chain refit a capped target without perturbing the stream.
    """
    rng = np.random.default_rng(1)
    n = 500
    atoms = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    data = {
        "x": rng.normal(size=n),
        "target": atoms[rng.choice(len(atoms), size=n)],
    }
    frame = make_person_frame(data)
    first = fit(frame, ["x"], ["target"], n_estimators=8, seed=7).predict(
        frame.table("person").loc[:, ["x"]]
    )
    second = fit(frame, ["x"], ["target"], n_estimators=8, seed=7).predict(
        frame.table("person").loc[:, ["x"]]
    )
    np.testing.assert_array_equal(
        first["target"].to_numpy(), second["target"].to_numpy()
    )


# ----------------------------------------------------------------------------
# Checkpointed targetwise chain: exact monolith equivalence and safe resume
# ----------------------------------------------------------------------------


def _json_roundtrip_chain_state(state: QRFChainState) -> QRFChainState:
    """Exercise the public JSON checkpoint contract exactly as a worker does."""
    return QRFChainState.from_dict(json.loads(json.dumps(state.to_dict())))


def test_targetwise_chain_is_bit_identical_to_monolith_after_json_resumes(
    correlated_targets_frame,
    monkeypatch,
) -> None:
    """A per-target subprocess chain reproduces monolithic float bits.

    The state is JSON-roundtripped before the first target and after every
    target, which proves that the separately restored fit/draw RNG states and
    external raw recipient priors are sufficient to resume without changing a
    single output bit.
    """
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    frame, _ = correlated_targets_frame(seed=3, n=240)
    recipient = frame.table("person").iloc[::2].loc[:, ["x"]].copy()
    model = RegimeGatedQRF(n_estimators=8, seed=19)

    monolith = model.fit(frame, ["x"], ["first", "second"]).predict(recipient)
    state = _json_roundtrip_chain_state(
        model.start_chain(frame, ["x"], ["first", "second"])
    )
    assert state.next_target == "first"
    assert not state.is_complete
    raw_priors = pd.DataFrame(index=recipient.index)

    for expected_target in ("first", "second"):
        step = model.fit_draw_next(
            frame,
            recipient,
            raw_priors,
            state=state,
        )
        assert step.target == expected_target
        assert step.raw_draw.dtype == np.dtype(np.float64)
        assert not step.raw_draw.flags.writeable
        np.testing.assert_array_equal(
            step.raw_draw,
            monolith[expected_target].to_numpy(),
        )
        raw_priors[step.target] = step.raw_draw
        state = _json_roundtrip_chain_state(step.state)

    assert state.is_complete
    assert state.next_target is None
    assert state.completed_targets == state.targets == ("first", "second")
    np.testing.assert_array_equal(raw_priors.to_numpy(), monolith.to_numpy())


def test_targetwise_chain_refuses_config_prefix_order_and_index_drift(
    correlated_targets_frame,
    monkeypatch,
) -> None:
    """Resume checks fail closed before fitting on drifted chain inputs."""
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    frame, _ = correlated_targets_frame(seed=4, n=120)
    recipient = frame.table("person").iloc[:50].loc[:, ["x"]].copy()
    model = RegimeGatedQRF(n_estimators=4, seed=23)
    state = model.start_chain(frame, ["x"], ["first", "second"])
    empty_priors = pd.DataFrame(index=recipient.index)

    changed_model = RegimeGatedQRF(n_estimators=5, seed=23)
    with pytest.raises(ValueError, match="model configuration changed"):
        changed_model.fit_draw_next(frame, recipient, empty_priors, state=state)
    with pytest.raises(ValueError, match="exact completed target prefix"):
        model.fit_draw_next(
            frame,
            recipient,
            pd.DataFrame({"second": np.zeros(len(recipient))}, index=recipient.index),
            state=state,
        )

    first = model.fit_draw_next(frame, recipient, empty_priors, state=state)
    priors = pd.DataFrame({"first": first.raw_draw}, index=recipient.index)

    reordered_recipient = recipient.iloc[::-1]
    reordered_priors = priors.iloc[::-1]
    with pytest.raises(ValueError, match="recipient index/order changed"):
        model.fit_draw_next(
            frame,
            reordered_recipient,
            reordered_priors,
            state=first.state,
        )

    float32_priors = priors.astype(np.float32)
    with pytest.raises(ValueError, match="must retain float64 raw QRF draws"):
        model.fit_draw_next(
            frame,
            recipient,
            float32_priors,
            state=first.state,
        )

    drifted_order = first.state.to_dict()
    drifted_order["targets"] = ["second", "first"]
    with pytest.raises(ValueError, match="exact target-order prefix"):
        QRFChainState.from_dict(drifted_order)


def test_targetwise_chain_locks_resolved_fit_threading(
    correlated_targets_frame,
    monkeypatch,
) -> None:
    """A subprocess cannot silently resume under a different fit width."""
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    frame, _ = correlated_targets_frame(seed=5, n=80)
    recipient = frame.table("person").iloc[:20].loc[:, ["x"]].copy()
    model = RegimeGatedQRF(n_estimators=3, seed=29)
    state = model.start_chain(frame, ["x"], ["first", "second"])
    assert state.fit_n_jobs == 1

    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "2")
    with pytest.raises(ValueError, match="model configuration changed"):
        model.fit_draw_next(
            frame,
            recipient,
            pd.DataFrame(index=recipient.index),
            state=state,
        )


def test_fit_n_jobs_unset_preserves_historical_default(monkeypatch) -> None:
    """Unset fit threading keeps quantile-forest's existing all-core default."""
    monkeypatch.delenv("POPULACE_FIT_N_JOBS", raising=False)
    assert _fit_n_jobs() == -1


@pytest.mark.parametrize(("configured", "expected"), [("1", 1), ("4", 4)])
def test_fit_n_jobs_accepts_positive_integers(
    configured,
    expected,
    monkeypatch,
) -> None:
    """A positive integer environment override is resolved exactly."""
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", configured)
    assert _fit_n_jobs() == expected


@pytest.mark.parametrize("configured", ["", "0", "-1", "1.5", " 1", "01", "+1"])
def test_fit_n_jobs_rejects_non_strict_positive_integers(
    configured,
    monkeypatch,
) -> None:
    """Configured fit threading is strict: only canonical positive ints pass."""
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", configured)
    with pytest.raises(ValueError, match="must be a positive integer"):
        _fit_n_jobs()
