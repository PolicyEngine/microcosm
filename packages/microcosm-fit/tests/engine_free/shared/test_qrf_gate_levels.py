"""Level offsets in a cross-fitted gate's calibration: the rate holds within each level.

One slope and one offset per sign class make a cross-fitted gate's calibrated
probabilities average each class's weighted rate over the whole frame, and over
no part of it. Where a level of one predictor holds many of a class's rows but
little weight, as the SPI tape's composite records do for some rare outputs, a
gate that cannot separate that level fully has its shortfall there spread onto
every other row (microcosm#1276). ``gate_level_columns`` gives each supported
level of the named columns offsets of its own. These tests pin that the fitted
calibration then holds each class's rate within each offset level, that thin
and unseen levels take the reference's offsets, that the draw path applies the
offsets, that a fitted model keeps the rates by level on fresh rows, and that
without level columns nothing moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import GateCalibration, GateOutcome, RegimeGatedQRF
from microcosm.fit.qrf import (
    _GATE_LEVEL_MIN_ROWS,
    _cross_fit_gate,
    _CrossFittedGate,
    _fit_gate_calibration,
    _fit_gate_level_calibration,
    _GateLevels,
    _select_gate_levels,
)


def _level_rates(proba, labels_onehot, weights, member):
    w = weights[member]
    return w @ proba[member] / w.sum(), w @ labels_onehot[member] / w.sum()


def test_the_level_calibration_holds_each_level_s_rate() -> None:
    rng = np.random.default_rng(3)
    n = 6_000
    level = rng.random(n) < 0.05
    carrier = np.where(level, rng.random(n) < 0.5, rng.random(n) < 0.02)
    labels = carrier.astype(np.int64)
    # Scores that rank within a level but carry nothing about the level itself.
    score = np.clip(0.1 + 0.05 * carrier + 0.02 * rng.normal(size=n), 0.01, 0.99)
    log_proba = np.log(np.column_stack([1.0 - score, score]))
    weights = np.where(level, 1.0, 40.0) * rng.uniform(0.5, 1.5, n)
    design = level[:, None].astype(np.float64)
    onehot = np.column_stack([1 - labels, labels]).astype(np.float64)

    slope, offsets, level_offsets, ok = _fit_gate_level_calibration(
        log_proba, labels, weights, design
    )
    assert ok
    assert slope >= 0.0
    assert offsets[0] == 0.0
    assert level_offsets.shape == (1, 2)
    assert level_offsets[0, 0] == 0.0
    logits = slope * log_proba + offsets + design @ level_offsets
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba /= proba.sum(axis=1, keepdims=True)
    for member in (level, ~level, np.ones(n, dtype=bool)):
        calibrated, rate = _level_rates(proba, onehot, weights, member)
        np.testing.assert_allclose(calibrated, rate, atol=1e-5)

    # One offset per class holds the frame's rate and misses the level's.
    slope0, offsets0, _ = _fit_gate_calibration(log_proba, labels, weights)
    logits0 = slope0 * log_proba + offsets0
    proba0 = np.exp(logits0 - logits0.max(axis=1, keepdims=True))
    proba0 /= proba0.sum(axis=1, keepdims=True)
    calibrated0, rate0 = _level_rates(proba0, onehot, weights, level)
    assert calibrated0[1] < 0.5 * rate0[1]


def test_three_sign_classes_hold_their_rates_within_a_level() -> None:
    rng = np.random.default_rng(5)
    n = 9_000
    level = rng.random(n) < 0.1
    u = rng.random(n)
    labels = np.where(
        level,
        np.where(u < 0.2, -1, np.where(u < 0.6, 1, 0)),
        np.where(u < 0.01, -1, np.where(u < 0.05, 1, 0)),
    )
    index = np.searchsorted([-1, 0, 1], labels)
    proba_raw = np.full((n, 3), [0.02, 0.93, 0.05]) + 0.01 * rng.random((n, 3))
    proba_raw /= proba_raw.sum(axis=1, keepdims=True)
    weights = rng.uniform(0.5, 2.0, n)
    design = level[:, None].astype(np.float64)
    onehot = np.eye(3)[index]

    slope, offsets, level_offsets, ok = _fit_gate_level_calibration(
        np.log(proba_raw), index, weights, design
    )
    assert ok
    logits = slope * np.log(proba_raw) + offsets + design @ level_offsets
    proba = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba /= proba.sum(axis=1, keepdims=True)
    for member in (level, ~level):
        calibrated, rate = _level_rates(proba, onehot, weights, member)
        np.testing.assert_allclose(calibrated, rate, atol=1e-5)


def test_levels_skip_the_reference_and_thin_levels() -> None:
    n = 2_000
    column = np.zeros(n)
    labels = np.zeros(n, dtype=np.int64)
    # Level 1: enough rows of both classes. Level 2: too few carriers.
    column[:200] = 1.0
    labels[: _GATE_LEVEL_MIN_ROWS + 5] = 1
    column[200:300] = 2.0
    labels[200:205] = 1
    labels[300:400] = 1  # carriers in the reference level
    constant = np.ones(n)
    features = np.column_stack([column, constant])

    levels = _select_gate_levels(features, labels, None, (0, 1))
    assert levels.positions == (0,)
    assert levels.values == (1.0,)

    # A level whose carriers weigh nothing gets no offset either.
    weights = np.ones(n)
    weights[: _GATE_LEVEL_MIN_ROWS + 5] = 0.0
    assert _select_gate_levels(features, labels, weights, (0,)).positions == ()


class _ConstantGate:
    classes_ = np.array([0, 1])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.tile([0.9, 0.1], (len(x), 1))


def test_the_draw_path_applies_level_offsets_and_unseen_levels_take_the_reference() -> (
    None
):
    gate = _CrossFittedGate(
        [_ConstantGate()],
        slope=1.0,
        offsets=np.zeros(2),
        levels=_GateLevels(positions=(0,), values=(1.0,)),
        level_offsets=np.array([[0.0, 2.0]]),
    )
    x = np.array([[0.0], [1.0], [7.0]])
    proba = gate.predict_proba(x)
    base = 0.1
    shifted = 1.0 / (1.0 + np.exp(-(np.log(base / (1 - base)) + 2.0)))
    np.testing.assert_allclose(proba[:, 1], [base, shifted, base])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    with pytest.raises(ValueError, match="level offsets must match"):
        _CrossFittedGate(
            [_ConstantGate()],
            slope=1.0,
            offsets=np.zeros(2),
            levels=_GateLevels(positions=(0,), values=(1.0,)),
            level_offsets=np.zeros((2, 2)),
        )


def _composite_like(seed: int, n: int = 12_000) -> pd.DataFrame:
    """A light, carrier-dense level beside a heavy bulk, as the tape's composites."""
    rng = np.random.default_rng(seed)
    level = (rng.random(n) < 0.04).astype(np.float64)
    x = rng.normal(size=(n, 4))
    carrier = np.where(level == 1.0, rng.random(n) < 0.5, rng.random(n) < 0.03)
    x[carrier, 0] += 1.0
    table = pd.DataFrame(x, columns=[f"x{i}" for i in range(4)])
    table["level"] = level
    table["target"] = np.where(carrier, rng.lognormal(3.0, 0.5, n), 0.0)
    table["weight"] = np.where(level == 1.0, 1.0, 30.0)
    return table


def test_a_fitted_model_keeps_each_level_s_rate_on_fresh_rows() -> None:
    columns = ["x0", "x1", "x2", "x3", "level"]
    table = _composite_like(seed=21)
    fitted = RegimeGatedQRF(
        n_estimators=4,
        seed=9,
        gate_calibration=GateCalibration.CROSS_FITTED_BALANCED,
        gate_level_columns=("level",),
    ).fit(table, columns, ["target"], weights="weight")
    assert fitted.gate_outcomes() == {"target": GateOutcome.CROSS_FITTED}
    gate = fitted._target_models["target"].gate
    assert isinstance(gate, _CrossFittedGate)
    assert gate.levels is not None and gate.levels.values == (1.0,)
    assert fitted.gate_levels() == {"target": ("level=1",)}

    fresh = _composite_like(seed=22)
    proba = gate.predict_proba(fresh[columns].to_numpy(dtype=np.float64))[:, 1]
    carrier = fresh["target"].to_numpy() != 0.0
    for value in (0.0, 1.0):
        member = fresh["level"].to_numpy() == value
        ratio = proba[member].mean() / carrier[member].mean()
        assert 0.8 < ratio < 1.25, (value, ratio)
    draws = fitted.predict(fresh[columns], release_models=True)
    assert (draws["target"] >= 0.0).all()
    # Read at fit time, so a draw that releases the models keeps the receipt.
    assert fitted.gate_levels() == {"target": ("level=1",)}


def test_grouped_folds_keep_each_level_s_rate() -> None:
    # The FRS reference forests fit with each household's members in one fold.
    columns = ["x0", "x1", "x2", "x3", "level"]
    table = _composite_like(seed=25)
    groups = np.arange(len(table)) // 2
    fitted = RegimeGatedQRF(
        n_estimators=4,
        seed=9,
        gate_calibration=GateCalibration.CROSS_FITTED_BALANCED,
        gate_level_columns=("level",),
    ).fit(table, columns, ["target"], weights="weight", groups=groups)
    assert fitted.gate_outcomes() == {"target": GateOutcome.CROSS_FITTED}
    assert fitted.gate_levels() == {"target": ("level=1",)}

    gate = fitted._target_models["target"].gate
    fresh = _composite_like(seed=26)
    proba = gate.predict_proba(fresh[columns].to_numpy(dtype=np.float64))[:, 1]
    carrier = fresh["target"].to_numpy() != 0.0
    for value in (0.0, 1.0):
        member = fresh["level"].to_numpy() == value
        ratio = proba[member].mean() / carrier[member].mean()
        assert 0.8 < ratio < 1.25, (value, ratio)


def test_gate_levels_are_empty_without_level_columns() -> None:
    columns = ["x0", "x1", "x2", "x3", "level"]
    table = _composite_like(seed=23, n=2_000)
    fitted = RegimeGatedQRF(
        n_estimators=2, seed=9, gate_calibration=GateCalibration.CROSS_FITTED_BALANCED
    ).fit(table, columns, ["target"], weights="weight")
    assert fitted.gate_levels() == {"target": ()}


def test_without_levels_or_with_only_thin_levels_nothing_moves() -> None:
    table = _composite_like(seed=31, n=4_000)
    features = table[["x0", "x1", "x2", "x3", "level"]].to_numpy(dtype=np.float64)
    labels = (table["target"].to_numpy() != 0.0).astype(np.int64)
    weights = table["weight"].to_numpy(dtype=np.float64)
    default, outcome = _cross_fit_gate(features, labels, weights, seed=4, balanced=True)
    empty, _ = _cross_fit_gate(
        features, labels, weights, seed=4, balanced=True, level_positions=()
    )
    # Column x1 is continuous: every value is its own level and none is supported.
    thin, thin_outcome = _cross_fit_gate(
        features, labels, weights, seed=4, balanced=True, level_positions=(1,)
    )
    assert thin.levels is None
    assert thin_outcome == outcome
    for other in (empty, thin):
        np.testing.assert_array_equal(
            default.predict_proba(features), other.predict_proba(features)
        )


def test_level_columns_are_checked() -> None:
    with pytest.raises(ValueError, match="needs a cross-fitted"):
        RegimeGatedQRF(gate_level_columns=("level",))
    with pytest.raises(ValueError, match="repeats a column"):
        RegimeGatedQRF(
            gate_calibration=GateCalibration.CROSS_FITTED,
            gate_level_columns=("level", "level"),
        )
    model = RegimeGatedQRF(
        n_estimators=2,
        gate_calibration=GateCalibration.CROSS_FITTED,
        gate_level_columns=("missing",),
    )
    table = _composite_like(seed=41, n=500)
    with pytest.raises(ValueError, match="not among the predictors"):
        model.fit(table, ["x0", "level"], ["target"], weights="weight")
    chained = RegimeGatedQRF(
        n_estimators=2,
        gate_calibration=GateCalibration.CROSS_FITTED,
        gate_level_columns=("level",),
    )
    with pytest.raises(ValueError, match="checkpointed QRF chain"):
        chained.start_chain(table, ["x0", "level"], ["target"], weights="weight")
