"""Mechanics of the canonical model: regimes, gates, chaining, determinism.

These tests pin the structural guarantees of the regime-gated, chained,
weighted-bootstrap QRF that are not the weighted-mean contract itself: that the
gate preserves a zero-inflated target's zero mass and sign structure, that
chaining reproduces a cross-target correlation, that the draw count matches the
input, and that a fixed seed makes a freshly fitted model reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest

from populace.fit import Regime, fit
from populace.fit.qrf import detect_regime

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
    import populace.fit.qrf as qrf_module

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


def test_degenerate_zero_target_draws_all_zero(make_person_frame) -> None:
    """A target that is constant zero in training draws exactly zero."""
    n = 200
    rng = np.random.default_rng(0)
    frame = make_person_frame({"x": rng.normal(size=n), "target": np.zeros(n)})
    fitted = fit(frame, ["x"], ["target"], n_estimators=10, seed=0)
    assert fitted.regimes()["target"] == Regime.DEGENERATE_ZERO
    draws = fitted.predict(frame)["target"].to_numpy()
    assert (draws == 0.0).all()


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
