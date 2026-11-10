"""Class-balanced cross-fitted sign gates: a very rare class is ranked, and drawn at its rate.

A gate fitted on the rows' own weights gives a class that holds a hundredth of
a percent of the weight almost none of the loss, so it cannot say which rows
carry it. :attr:`GateCalibration.CROSS_FITTED_BALANCED` fits the fold gates on
class-balanced weights and keeps the calibration on the rows' own weights
(microcosm#1276). These tests pin that the fold gates see balanced weights, the
calibrated probabilities still average each class's weighted rate, the ranking
on rows the gates were not fitted to improves, and the two other calibrations
do not move.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.fit import GateCalibration, GateOutcome, RegimeGatedQRF
from microcosm.fit.qrf import _class_balanced_weights, _cross_fit_gate


def _very_rare_carriers(
    seed: int = 11, n: int = 12_000
) -> tuple[pd.DataFrame, list[str]]:
    """Carriers are frequent rows with tiny weights, as a certainty stratum's are."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6))
    carrier = np.zeros(n, dtype=bool)
    carrier[rng.choice(n, size=240, replace=False)] = True
    x[carrier, 0] += 2.5
    columns = [f"x{i}" for i in range(6)]
    table = pd.DataFrame(x, columns=columns)
    table["target"] = np.where(carrier, rng.lognormal(3.0, 0.5, n), 0.0)
    table["weight"] = np.where(carrier, 1.0, 400.0)
    return table, columns


def _oof_auc(scores: np.ndarray, carrier: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = int(carrier.sum())
    n0 = len(scores) - n1
    return float((ranks[carrier].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def test_balanced_weights_give_every_class_the_same_total() -> None:
    labels = np.array([0, 0, 0, 1, 1, -1])
    weights = np.array([10.0, 30.0, 60.0, 1.0, 3.0, 0.5])
    balanced = _class_balanced_weights(labels, weights)
    totals = [balanced[labels == value].sum() for value in (-1, 0, 1)]
    np.testing.assert_allclose(totals, totals[0])
    np.testing.assert_allclose(balanced.mean(), 1.0)
    # Within a class the rows keep their relative weights.
    np.testing.assert_allclose(balanced[1] / balanced[0], 3.0)
    np.testing.assert_allclose(
        _class_balanced_weights(labels, None)[labels == 0].sum(),
        _class_balanced_weights(labels, None)[labels == 1].sum(),
    )


def test_the_balanced_gate_ranks_a_very_rare_class_and_keeps_its_rate() -> None:
    table, columns = _very_rare_carriers()
    features = table[columns].to_numpy(dtype=np.float64)
    labels = (table["target"].to_numpy() != 0.0).astype(np.int64)
    weights = table["weight"].to_numpy(dtype=np.float64)
    rate = weights[labels == 1].sum() / weights.sum()
    assert rate < 0.001

    plain, _ = _cross_fit_gate(features, labels, weights, seed=7)
    balanced, outcome = _cross_fit_gate(
        features, labels, weights, seed=7, balanced=True
    )
    assert outcome == GateOutcome.CROSS_FITTED

    held_out = _very_rare_carriers(seed=12)[0]
    held_features = held_out[columns].to_numpy(dtype=np.float64)
    held_carrier = held_out["target"].to_numpy() != 0.0
    plain_auc = _oof_auc(plain.predict_proba(held_features)[:, 1], held_carrier)
    balanced_auc = _oof_auc(balanced.predict_proba(held_features)[:, 1], held_carrier)
    assert balanced_auc > 0.9
    assert balanced_auc > plain_auc

    # The level is the rows' own: the mean probability over fresh rows of the
    # same population is the weighted rate, not the balanced half.
    held_weights = held_out["weight"].to_numpy(dtype=np.float64)
    drawn = (held_weights * balanced.predict_proba(held_features)[:, 1]).sum()
    assert 0.5 < drawn / (rate * held_weights.sum()) < 2.0


def test_the_model_takes_the_balanced_calibration_and_names_its_outcome() -> None:
    table, columns = _very_rare_carriers(n=6_000)
    fitted = RegimeGatedQRF(
        n_estimators=4,
        seed=5,
        gate_calibration=GateCalibration.CROSS_FITTED_BALANCED,
    ).fit(table, columns, ["target"], weights="weight")
    assert fitted.gate_outcomes() == {"target": GateOutcome.CROSS_FITTED}
    draws = fitted.predict(table)
    assert (draws["target"] >= 0.0).all()


def test_the_plain_cross_fitted_gate_does_not_move() -> None:
    table, columns = _very_rare_carriers(n=3_000)
    features = table[columns].to_numpy(dtype=np.float64)
    labels = (table["target"].to_numpy() != 0.0).astype(np.int64)
    weights = table["weight"].to_numpy(dtype=np.float64)
    default, _ = _cross_fit_gate(features, labels, weights, seed=3)
    named, _ = _cross_fit_gate(features, labels, weights, seed=3, balanced=False)
    np.testing.assert_array_equal(
        default.predict_proba(features), named.predict_proba(features)
    )
