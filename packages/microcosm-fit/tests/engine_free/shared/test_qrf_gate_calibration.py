"""Cross-fitted sign gates: a rare class drawn at its rate off the training rows.

The in-sample gate is one boosted classifier calibrated only on its own rows.
On rows it was not fitted to, its mean probability of a rare class falls short
of the class's rate, so recipients draw the class too rarely. These tests pin
the opt-in :attr:`GateCalibration.CROSS_FITTED`: it restores the rate off
sample, leaves the forests and every draw uniform where they were, keeps a
class too thin to cross-fit on the in-sample gate, calibrates three-sign gates,
and checkpoints its choice only when it is made.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from microcosm.fit import GateCalibration, QRFChainState, Regime, RegimeGatedQRF, fit
from microcosm.fit.qrf import _fit_gate_calibration

IN_SAMPLE = GateCalibration.IN_SAMPLE
CROSS_FITTED = GateCalibration.CROSS_FITTED


def _rare_positive_table(
    seed: int = 3, n: int = 4000
) -> tuple[pd.DataFrame, list[str]]:
    """A positive class on about 7% of the weight: a weak signal among noise."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 12))
    logit = -3.0 + 0.9 * x[:, 0] + 0.5 * x[:, 1]
    positive = rng.random(n) < 1.0 / (1.0 + np.exp(-logit))
    columns = [f"x{i}" for i in range(12)]
    table = pd.DataFrame(x, columns=columns)
    table["target"] = np.where(positive, rng.lognormal(3.0, 0.5, n), 0.0)
    table["second"] = np.where(positive, 2.0 * table["target"], 0.0) + np.where(
        rng.random(n) < 0.3, rng.lognormal(1.0, 0.5, n), 0.0
    )
    table["weight"] = rng.uniform(1.0, 9.0, n)
    return table, columns


def _fit(table, columns, calibration, targets=("target",)):
    model = RegimeGatedQRF(n_estimators=4, seed=5, gate_calibration=calibration)
    return model.fit(table, columns, list(targets), weights="weight")


def _positive_probability(fitted, table, columns) -> np.ndarray:
    gate = fitted._target_models["target"].gate
    proba = gate.predict_proba(table.loc[:, columns].to_numpy(dtype=np.float64))
    return proba[:, list(gate.classes_).index(1)]


def _softmax(logits: np.ndarray) -> np.ndarray:
    scaled = np.exp(logits - logits.max(axis=1, keepdims=True))
    return scaled / scaled.sum(axis=1, keepdims=True)


def test_in_sample_is_the_default_gate() -> None:
    table, columns = _rare_positive_table(n=600)
    default = RegimeGatedQRF(n_estimators=4, seed=5).fit(
        table, columns, ["target"], weights="weight"
    )
    named = _fit(table, columns, IN_SAMPLE)
    assert isinstance(
        default._target_models["target"].gate, HistGradientBoostingClassifier
    )
    uniforms = {
        "quantiles": {"target": np.linspace(0.0, 0.99, len(table))},
        "sign_uniforms": {"target": np.random.default_rng(1).random(len(table))},
    }
    pd.testing.assert_frame_equal(
        default.predict_from_uniforms(table, **uniforms),
        named.predict_from_uniforms(table, **uniforms),
    )


def test_an_unknown_gate_calibration_is_refused() -> None:
    with pytest.raises(ValueError, match="gate_calibration 'platt' is not one of"):
        RegimeGatedQRF(gate_calibration="platt")


def test_cross_fitted_gate_draws_a_rare_class_at_its_held_out_rate() -> None:
    """Off the training rows the in-sample gate under-predicts; cross-fitting does not."""
    table, columns = _rare_positive_table()
    train, held_out = table.iloc[:2000], table.iloc[2000:]
    weights = held_out["weight"].to_numpy()
    rate = float(weights @ (held_out["target"] > 0) / weights.sum())
    means = {
        calibration: float(
            weights
            @ _positive_probability(
                _fit(train, columns, calibration), held_out, columns
            )
            / weights.sum()
        )
        for calibration in (IN_SAMPLE, CROSS_FITTED)
    }
    assert means[IN_SAMPLE] < 0.6 * rate
    assert means[CROSS_FITTED] == pytest.approx(rate, rel=0.25)


def test_cross_fitting_moves_only_the_gate() -> None:
    """The forests and the uniforms are the same; only who is positive moves."""
    table, columns = _rare_positive_table(n=1200)
    fits = {
        calibration: _fit(table, columns, calibration)
        for calibration in (IN_SAMPLE, CROSS_FITTED)
    }
    quantiles = {"target": np.linspace(0.0, 0.99, len(table))}
    pd.testing.assert_frame_equal(
        fits[IN_SAMPLE].predict_positive_from_uniforms(table, quantiles=quantiles),
        fits[CROSS_FITTED].predict_positive_from_uniforms(table, quantiles=quantiles),
    )
    signs = {"target": np.random.default_rng(2).random(len(table))}
    drawn = {
        calibration: fitted.predict_from_uniforms(
            table, quantiles=quantiles, sign_uniforms=signs
        )["target"].to_numpy()
        for calibration, fitted in fits.items()
    }
    both = (drawn[IN_SAMPLE] > 0) & (drawn[CROSS_FITTED] > 0)
    assert both.any()
    np.testing.assert_array_equal(drawn[IN_SAMPLE][both], drawn[CROSS_FITTED][both])


def test_calibration_matches_each_class_weighted_rate() -> None:
    """The fitted map's weighted mean is each class's rate; two classes are Platt."""
    rng = np.random.default_rng(0)
    n = 3000
    logits = np.column_stack(
        [
            np.zeros(n),
            0.8 * rng.normal(size=n) - 1.5,
            0.5 * rng.normal(size=n) + 0.3,
        ]
    )
    labels = (rng.random(n)[:, None] > np.cumsum(_softmax(logits), axis=1)).sum(axis=1)
    weights = rng.uniform(500.0, 9000.0, n)
    overconfident = np.log(_softmax(2.0 * logits))
    slope, offsets = _fit_gate_calibration(overconfident, labels, weights)
    calibrated = _softmax(slope * overconfident + offsets)
    onehot = np.eye(3)[labels]
    np.testing.assert_allclose(
        weights @ calibrated / weights.sum(),
        weights @ onehot / weights.sum(),
        rtol=1e-9,
    )
    assert offsets[0] == 0.0
    assert 0.3 < slope < 0.7

    # Two classes: a logistic regression of the label on the gate's logit, so
    # at the optimum both of Platt's score equations hold.
    positive = (labels == 2).astype(int)
    two = np.log(_softmax(np.column_stack([np.zeros(n), 2.0 * logits[:, 2]])))
    slope, offsets = _fit_gate_calibration(two, positive, weights)
    score = two[:, 1] - two[:, 0]
    residual = 1.0 / (1.0 + np.exp(-(slope * score + offsets[1]))) - positive
    assert abs(weights @ residual) / weights.sum() < 1e-9
    assert abs(weights @ (residual * score)) / weights.sum() < 1e-9
    assert 0.3 < slope < 0.7


def test_a_class_too_thin_to_cross_fit_keeps_the_in_sample_gate(
    rare_positive_frame,
) -> None:
    frame, _, _ = rare_positive_frame(seed=0, n=1000, n_positive=4)
    gates = {}
    for calibration in (IN_SAMPLE, CROSS_FITTED):
        fitted = fit(
            frame,
            ["signal", "noise"],
            ["target"],
            n_estimators=4,
            seed=0,
            gate_calibration=calibration,
        )
        gates[calibration] = fitted._target_models["target"].gate
    assert isinstance(gates[CROSS_FITTED], HistGradientBoostingClassifier)
    x = frame.table("person").loc[:, ["signal", "noise"]].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(
        gates[IN_SAMPLE].predict_proba(x), gates[CROSS_FITTED].predict_proba(x)
    )


def test_three_sign_gate_cross_fits_every_class(zero_inflated_frame) -> None:
    frame, _ = zero_inflated_frame(seed=0, n=1500)
    fitted = fit(
        frame, ["x"], ["target"], n_estimators=4, seed=0, gate_calibration=CROSS_FITTED
    )
    assert fitted.regimes()["target"] == Regime.THREE_SIGN
    assert fitted.sign_classes("target") == (-1, 0, 1)
    gate = fitted._target_models["target"].gate
    assert not isinstance(gate, HistGradientBoostingClassifier)
    proba = gate.predict_proba(frame.table("person").loc[:, ["x"]].to_numpy())
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    draws = fitted.predict(frame)["target"].to_numpy()
    assert (draws > 0).any() and (draws < 0).any() and (draws == 0).any()


def test_conditional_uniforms_round_trip_under_a_cross_fitted_gate() -> None:
    """Locating observed targets and drawing at those uniforms gives them back."""
    x = np.tile(np.arange(30, dtype=float), 6)
    observed = pd.DataFrame(
        {
            "x": x,
            "mixed": np.tile([-2.0, 0.0, 3.0], len(x) // 3),
            "inflated": np.tile([0.0, 4.0], len(x) // 2),
        }
    )
    fitted = fit(
        observed,
        ["x"],
        ["mixed", "inflated"],
        weights="none",
        n_estimators=4,
        seed=7,
        gate_calibration=CROSS_FITTED,
    )
    rng = np.random.default_rng(1)
    before = copy.deepcopy(fitted._rng.bit_generator.state)
    located = fitted.conditional_uniforms(
        observed,
        sign_placement={t: rng.random(len(x)) for t in fitted.targets},
        quantile_placement={t: rng.random(len(x)) for t in fitted.targets},
    )
    assert fitted._rng.bit_generator.state == before
    drawn = fitted.predict_from_uniforms(
        observed[["x"]],
        quantiles=located.quantiles,
        sign_uniforms=located.sign_uniforms,
    )
    pd.testing.assert_frame_equal(
        drawn, observed[fitted.targets], check_exact=False, rtol=1e-9
    )


def test_targetwise_cross_fitted_chain_is_bit_identical_to_the_monolith(
    monkeypatch,
) -> None:
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    table, columns = _rare_positive_table(n=800)
    recipient = table.iloc[::2].loc[:, columns].copy()
    model = RegimeGatedQRF(n_estimators=4, seed=5, gate_calibration=CROSS_FITTED)
    targets = ["target", "second"]
    uniforms = {
        "quantiles": {t: np.linspace(0.0, 0.99, len(recipient)) for t in targets},
        "sign_uniforms": {
            t: np.random.default_rng(i).random(len(recipient))
            for i, t in enumerate(targets)
        },
    }
    monolith = model.fit(
        table, columns, targets, weights="weight"
    ).predict_from_uniforms(recipient, **uniforms)
    state = model.start_chain(table, columns, targets, weights="weight")
    raw = pd.DataFrame(index=recipient.index)
    for target in targets:
        state = QRFChainState.from_dict(json.loads(json.dumps(state.to_dict())))
        step = model.fit_draw_next(
            table,
            recipient,
            raw,
            state=state,
            weights="weight",
            quantiles=uniforms["quantiles"][target],
            sign_uniforms=uniforms["sign_uniforms"][target],
        )
        np.testing.assert_array_equal(step.raw_draw, monolith[target].to_numpy())
        raw[target] = step.raw_draw
        state = step.state


def test_chain_state_names_the_calibration_only_when_cross_fitted(
    correlated_targets_frame,
) -> None:
    frame, _ = correlated_targets_frame(seed=4, n=120)
    states = {
        calibration: RegimeGatedQRF(
            n_estimators=4, seed=23, gate_calibration=calibration
        ).start_chain(frame, ["x"], ["first", "second"])
        for calibration in (IN_SAMPLE, CROSS_FITTED)
    }
    assert "gate_calibration" not in states[IN_SAMPLE].to_dict()["model_config"]
    assert states[CROSS_FITTED].to_dict()["model_config"]["gate_calibration"] == (
        CROSS_FITTED
    )
    for state in states.values():
        assert QRFChainState.from_dict(json.loads(json.dumps(state.to_dict()))) == state

    recipient = frame.table("person").iloc[:50].loc[:, ["x"]].copy()
    with pytest.raises(ValueError, match="model configuration changed"):
        RegimeGatedQRF(n_estimators=4, seed=23).fit_draw_next(
            frame,
            recipient,
            pd.DataFrame(index=recipient.index),
            state=states[CROSS_FITTED],
        )
    payload = states[CROSS_FITTED].to_dict()
    payload["model_config"]["gate_calibration"] = "platt"
    with pytest.raises(ValueError, match="gate_calibration 'platt'"):
        QRFChainState.from_dict(payload)
