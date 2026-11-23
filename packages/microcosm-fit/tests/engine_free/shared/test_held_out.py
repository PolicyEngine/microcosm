"""Held-out evidence for categorical models: folds, out-of-fold probabilities, scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import (
    BoostedClassifierSettings,
    CategoricalClassifier,
    can_deal_folds,
    categorical_scores,
    out_of_fold_proba,
    stratified_folds,
)


def test_folds_deal_whole_groups_and_every_class() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 3)
    labels = np.where(groups % 4 == 0, "rare", "common")
    folds = stratified_folds(labels, 5, rng, groups=groups)
    by_group = pd.Series(folds).groupby(groups).nunique()
    assert (by_group == 1).all()
    for fold in range(5):
        assert set(labels[folds == fold]) == {"rare", "common"}


def test_can_deal_folds_counts_groups_not_rows() -> None:
    labels = np.array(["a"] * 10 + ["b"] * 10)
    one_household_of_b = np.r_[np.arange(10), np.full(10, 99)]
    assert can_deal_folds(labels, 5)
    assert not can_deal_folds(labels, 5, groups=one_household_of_b)


def _data(n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, n)
    label = np.where(x + rng.normal(0, 0.7, n) > 0, "up", "down")
    return pd.DataFrame({"x": x, "label": label, "weight": rng.uniform(0.5, 2, n)})


def test_out_of_fold_scores_every_row_once_with_each_learner() -> None:
    data = _data()
    model = CategoricalClassifier(boosted=BoostedClassifierSettings(max_iter=40))
    folds = stratified_folds(data["label"].to_numpy(), 5, np.random.default_rng(2))
    out = out_of_fold_proba(
        model, data, ["x"], "label", weights=data["weight"].to_numpy(), folds=folds
    )
    assert list(out["classes"]) == ["down", "up"]
    for name in ("model", "boosted", "spline_logit", "marginal"):
        assert out[name].shape == (len(data), 2)
        assert np.allclose(out[name].sum(axis=1), 1.0)
    assert np.allclose(out["model"], 0.5 * (out["boosted"] + out["spline_logit"]))


def test_scores_reward_the_model_over_the_constant() -> None:
    data = _data()
    model = CategoricalClassifier(boosted=BoostedClassifierSettings(max_iter=40))
    folds = stratified_folds(data["label"].to_numpy(), 5, np.random.default_rng(2))
    w = data["weight"].to_numpy()
    out = out_of_fold_proba(model, data, ["x"], "label", weights=w, folds=folds)
    labels = data["label"].to_numpy()
    fitted = categorical_scores(out["model"], labels, out["classes"], w)
    constant = categorical_scores(out["marginal"], labels, out["classes"], w)
    assert fitted["log_loss"] < constant["log_loss"] - 0.2
    assert fitted["calibration_outcome"] == "cross_fitted"
    assert 0.8 < fitted["calibration_slope"] < 1.2


def _calibrated(n: int = 20000, seed: int = 4):
    """Probabilities and outcomes drawn from them, so the slope is about 1."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, (n, 3))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    labels = (np.cumsum(proba, axis=1) > rng.random(n)[:, None]).argmax(axis=1)
    return proba, labels, np.ones(n)


def _tempered(proba: np.ndarray, power: float) -> np.ndarray:
    scaled = proba**power
    return scaled / scaled.sum(axis=1, keepdims=True)


def test_slope_reads_the_spread_of_the_probabilities() -> None:
    proba, labels, w = _calibrated()
    classes = np.arange(3)
    right = categorical_scores(proba, labels, classes, w)["calibration_slope"]
    flat = categorical_scores(_tempered(proba, 0.5), labels, classes, w)[
        "calibration_slope"
    ]
    sharp = categorical_scores(_tempered(proba, 2.0), labels, classes, w)[
        "calibration_slope"
    ]
    assert abs(right - 1.0) < 0.05
    assert flat > 1.5
    assert sharp < 0.7


def test_share_gaps_within_strata() -> None:
    labels = np.array([0, 0, 1, 1])
    proba = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    scores = categorical_scores(
        proba,
        labels,
        np.array([0, 1]),
        np.ones(4),
        strata={"half": ["x", "x", "y", "y"]},
    )
    assert scores["share_gap"] == pytest.approx(0.25)
    assert scores["strata"]["half"]["x"] == {"rows": 2, "weight": 2.0, "share_gap": 0.0}
    assert scores["strata"]["half"]["y"]["share_gap"] == pytest.approx(0.5)


def test_labels_outside_the_classes_are_refused() -> None:
    with pytest.raises(ValueError, match="one of classes"):
        categorical_scores(
            np.array([[1.0]]), np.array(["b"]), np.array(["a"]), np.ones(1)
        )
