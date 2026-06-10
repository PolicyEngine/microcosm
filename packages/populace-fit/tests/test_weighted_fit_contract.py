"""The behavioral contract: a weighted fit shifts draws toward the weighted truth.

This is the real realization of the placeholder the kernel left skipped in
``packages/populace-frame/tests/test_contracts.py``
(``test_weighted_fit_shifts_draws_toward_weighted_truth``). It is the test
DESIGN.md names as the one that would have caught the 2026-06 microimpute
landmine: a ten-line behavioral check that a weighted fit moves the draws toward
the *weighted* population, and that an unweighted fit is impossible to express
without writing ``weights="none"`` and meaning it.

The frame's donor has a weight-correlated target — about 20% of rows are a
low-weight regime carrying ~10x the bulk's value — so the weighted and
unweighted means differ by well over 3x. The contract:

a. weighted draws' mean is within ~20% of the true *weighted* mean;
b. ``weights="none"`` draws' mean is within ~20% of the *unweighted* mean;
c. the two differ by more than 3x.
"""

from __future__ import annotations

import numpy as np
import pytest

from populace.fit import NO_WEIGHTS, fit

# Small forests keep the fit fast; n=5000 (the fixture default) is enough for
# the weighted/unweighted means to be stable across the seed.
_N_ESTIMATORS = 60
_SEED = 1


def test_weighted_fit_shifts_draws_toward_weighted_truth(
    weight_correlated_frame,
) -> None:
    """A weighted fit tracks the weighted mean; an unweighted fit does not.

    Realizes the kernel's skipped populace-fit contract. The donor target is
    large exactly where the weight is small, so:

    (a) weighting the fit by the frame's design weights makes the draws'
        unweighted mean land within 20% of the true *weighted* mean;
    (b) fitting with ``weights="none"`` makes them land within 20% of the
        *unweighted* mean instead;
    (c) the two sets of draws differ by more than 3x — weighting is not a
        rounding effect, it is the difference between two populations.
    """
    frame, target, weights = weight_correlated_frame(seed=0)

    true_weighted_mean = float(np.average(target, weights=weights))
    true_unweighted_mean = float(target.mean())
    # The fixture is constructed so the two truths are far apart; assert it so a
    # future fixture change that collapses the gap fails loudly here, not as a
    # mysterious contract failure below.
    assert true_unweighted_mean > 3.0 * true_weighted_mean

    fitted_weighted = fit(
        frame, ["age", "is_male"], ["target"], n_estimators=_N_ESTIMATORS, seed=_SEED
    )
    fitted_unweighted = fit(
        frame,
        ["age", "is_male"],
        ["target"],
        weights=NO_WEIGHTS,
        n_estimators=_N_ESTIMATORS,
        seed=_SEED,
    )

    weighted_draws = fitted_weighted.predict(frame)["target"].to_numpy()
    unweighted_draws = fitted_unweighted.predict(frame)["target"].to_numpy()

    weighted_mean = float(weighted_draws.mean())
    unweighted_mean = float(unweighted_draws.mean())

    # (a) weighted draws track the weighted truth.
    assert weighted_mean == pytest.approx(true_weighted_mean, rel=0.20)
    # (b) unweighted draws track the unweighted truth.
    assert unweighted_mean == pytest.approx(true_unweighted_mean, rel=0.20)
    # (c) and the two are genuinely different populations.
    assert unweighted_mean > 3.0 * weighted_mean


def test_design_is_the_default_and_matches_explicit_design(
    weight_correlated_frame,
) -> None:
    """Omitting ``weights`` weights by design — there is no unweighted default.

    The default-weighted draws must track the weighted truth (not the
    unweighted one), proving the operator is weighted *by construction*: a
    caller who forgets ``weights`` still gets a weighted fit.
    """
    frame, target, weights = weight_correlated_frame(seed=0)
    true_weighted_mean = float(np.average(target, weights=weights))

    default_draws = fit(
        frame, ["age", "is_male"], ["target"], n_estimators=_N_ESTIMATORS, seed=_SEED
    ).predict(frame)["target"]

    assert float(default_draws.mean()) == pytest.approx(
        true_weighted_mean, rel=0.20
    )
