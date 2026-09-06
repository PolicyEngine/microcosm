"""Real fitted target reuse without changing the legacy QRF estimator/draws."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.fit import RegimeGatedQRF
from microcosm.fit import qrf as legacy
from microcosm.fit import qrf_target as split
from microcosm.graph import Numeric, platform_fingerprint


@pytest.fixture(autouse=True)
def serial_workers(monkeypatch):
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")


@pytest.fixture
def donors():
    rng = np.random.default_rng(7)
    n = 90
    positive = np.exp(rng.normal(2, 0.4, n))
    sign = np.resize(np.array([-1.0, 0.0, 1.0]), n)
    return pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "zero": np.zeros(n),
            "constant": np.full(n, 2.5),
            "positive": positive,
            "negative": -positive,
            "zero_positive": np.where(sign > 0, positive, 0.0),
            "zero_negative": np.where(sign < 0, -positive, 0.0),
            "signed": np.where(sign > 0, positive, -positive),
            "mixed": sign * positive,
            "last": positive * positive + rng.uniform(0, 0.2, n),
            "weight": np.where(sign > 0, 0.03, 12.0),
        },
        index=pd.Index(np.arange(n) + 100, name="donor"),
    )


def _model():
    return RegimeGatedQRF(n_estimators=3, seed=29)


def _fit_chain(model, donors, initial, *, weights="weight"):
    state = split.LegacyQRFTrainingState.from_chain(initial)
    artifacts = []
    while not state.is_complete:
        artifact = split.fit_target(model, donors, state=state, weights=weights)
        artifacts.append(artifact)
        # Every fit completes before the first recipient is seen; checkpoints
        # have no draw stream or recipient identity, even between targets.
        assert "recipient_index" not in state.to_dict()
        assert "draw_rng_state" not in state.to_dict()
        state = split.LegacyQRFTrainingState.from_dict(
            json.loads(json.dumps(artifact.next_training_state.to_dict()))
        )
    return artifacts


def _apply_chain(artifacts, recipients, initial):
    state = initial
    raw = pd.DataFrame(index=recipients.index)
    for artifact in artifacts:
        step = split.apply_target(artifact, recipients, raw, state=state)
        assert not step.raw_draw.flags.writeable
        raw[step.target] = step.raw_draw
        state = step.state
    return raw, state


@pytest.mark.parametrize("weights", ["weight", "none"])
def test_all_regimes_and_raw_chain_are_bit_identical(donors, weights):
    """Fit all targets first, then roundtrip models and match both old APIs."""
    model = _model()
    targets = [column for column in donors if column not in {"x", "weight"}]
    initial = model.start_chain(donors, ["x"], targets, weights=weights)
    recipients = donors.iloc[::3][["x"]].copy()
    artifacts = _fit_chain(model, donors, initial, weights=weights)
    restored = []
    for artifact in artifacts:
        payload = artifact.to_bytes()
        restored.append(
            split.LegacyQRFTargetArtifact.from_trusted_bytes(
                payload, expected_sha256=hashlib.sha256(payload).hexdigest()
            )
        )
        assert artifact.numeric is Numeric.PLATFORM_BITWISE
        assert artifact.platform == platform_fingerprint()
    actual, final = _apply_chain(restored, recipients, initial)
    expected = model.fit(donors, ["x"], targets, weights=weights).predict(recipients)
    np.testing.assert_array_equal(actual.to_numpy(), expected.to_numpy())
    old_state = initial
    old_raw = pd.DataFrame(index=recipients.index)
    for artifact in artifacts:
        old = model.fit_draw_next(
            donors, recipients, old_raw, state=old_state, weights=weights
        )
        np.testing.assert_array_equal(actual[old.target], old.raw_draw)
        old_raw[old.target] = old.raw_draw
        old_state = old.state
        if artifact.target == "zero":
            assert old.state.fit_rng_state_json == initial.fit_rng_state_json
            assert old.state.draw_rng_state_json == initial.draw_rng_state_json
    assert final == old_state


def test_model_reuse_excludes_recipient_and_draw_identity(donors, monkeypatch):
    model = _model()
    initial = model.start_chain(donors, ["x"], ["positive", "last"], weights="weight")
    calls = []
    real_fit = RegimeGatedQRF._fit_target

    def count_fit(self, **kwargs):
        calls.append(kwargs["columns"])
        return real_fit(self, **kwargs)

    monkeypatch.setattr(RegimeGatedQRF, "_fit_target", count_fit)
    artifacts = _fit_chain(model, donors, initial)
    assert calls == [("x",), ("x", "positive")]
    identity = artifacts[0].training_id
    alternate = replace(
        initial,
        draw_rng_state_json=legacy._rng_state_json(np.random.default_rng(802)),
    )
    assert split.LegacyQRFTrainingState.from_chain(initial) == (
        split.LegacyQRFTrainingState.from_chain(alternate)
    )
    for state, recipients in (
        (initial, donors.iloc[:21][["x"]]),
        (initial, donors.iloc[20:60][["x"]].set_axis(range(500, 540))),
        (alternate, donors.iloc[::2][["x"]]),
    ):
        raw, _ = _apply_chain(artifacts, recipients, state)
        # Independent legacy reference can use a different draw checkpoint.
        expected = pd.DataFrame(index=recipients.index)
        for target in state.targets:
            step = model.fit_draw_next(
                donors, recipients, expected, state=state, weights="weight"
            )
            state = step.state
            expected[target] = step.raw_draw
        np.testing.assert_array_equal(raw.to_numpy(), expected.to_numpy())
    assert artifacts[0].training_id == identity
    assert len(calls) == 2 + 3 * 2  # all six extra fits are legacy references


def test_training_identity_binds_consumed_donor_values_and_weights(donors):
    model = _model()
    initial = model.start_chain(donors, ["x"], ["positive"], weights="weight")
    state = split.LegacyQRFTrainingState.from_chain(initial)
    first = split.fit_target(model, donors, state=state, weights="weight")
    changed = donors.copy()
    changed.loc[changed.index[0], "positive"] += 1
    second = split.fit_target(model, changed, state=state, weights="weight")
    assert first.training_id != second.training_id
    ignored = donors.assign(unused=np.arange(len(donors)))
    same = split.fit_target(model, ignored, state=state, weights="weight")
    assert first.training_id == same.training_id
    assert first.to_bytes() == same.to_bytes()
    changed.loc[changed.index[0], "weight"] += 1
    with pytest.raises(ValueError, match="weight values/order changed"):
        split.fit_target(model, changed, state=state, weights="weight")
    with pytest.raises(ValueError, match="donor index/order changed"):
        split.fit_target(model, donors.iloc[::-1], state=state, weights="weight")


def test_state_and_raw_prefix_drift_are_refused(donors):
    model = _model()
    initial = model.start_chain(donors, ["x"], ["positive", "last"], weights="weight")
    artifacts = _fit_chain(model, donors, initial)
    recipients = donors.iloc[:20][["x"]]
    empty = pd.DataFrame(index=recipients.index)
    with pytest.raises(ValueError, match="training state"):
        split.apply_target(artifacts[1], recipients, empty, state=initial)
    changed = replace(
        initial, fit_rng_state_json=legacy._rng_state_json(np.random.default_rng(99))
    )
    with pytest.raises(ValueError, match="training state"):
        split.apply_target(artifacts[0], recipients, empty, state=changed)
    first = split.apply_target(artifacts[0], recipients, empty, state=initial)
    raw = pd.DataFrame({"positive": first.raw_draw}, index=recipients.index)
    with pytest.raises(ValueError, match="recipient index/order changed"):
        split.apply_target(
            artifacts[1], recipients.iloc[::-1], raw.iloc[::-1], state=first.state
        )
    with pytest.raises(ValueError, match="float64"):
        split.apply_target(
            artifacts[1], recipients, raw.astype("float32"), state=first.state
        )
    with pytest.raises(ValueError, match="exact completed target prefix"):
        split.apply_target(artifacts[1], recipients, empty, state=first.state)
    last = split.apply_target(artifacts[1], recipients, raw, state=first.state)
    with pytest.raises(ValueError, match="complete"):
        split.apply_target(artifacts[1], recipients, raw, state=last.state)
    with pytest.raises(ValueError, match="complete"):
        split.fit_target(
            model, donors, state=artifacts[1].next_training_state, weights="weight"
        )


def test_frame_front_door_uses_real_typed_weights(make_person_frame):
    x = np.arange(60, dtype=np.float64)
    frame = make_person_frame(
        {"x": x, "target": x + 1}, weights=np.where(x < 30, 0.02, 12)
    )
    model = _model()
    initial = model.start_chain(frame, ["x"], ["target"])
    artifact = split.fit_target(
        model, frame, state=split.LegacyQRFTrainingState.from_chain(initial)
    )
    raw = pd.DataFrame(index=frame.table("person").index)
    actual = split.apply_target(artifact, frame, raw, state=initial)
    expected = model.fit_draw_next(frame, frame, raw, state=initial)
    np.testing.assert_array_equal(actual.raw_draw, expected.raw_draw)
    assert actual.weight_kind == "design"
    assert actual.state == expected.state


def test_fit_uses_observed_priors_apply_uses_raw_draws(donors, monkeypatch):
    model = _model()
    initial = model.start_chain(donors, ["x"], ["positive", "last"], weights="weight")
    fit_features = []
    draw_features = []
    real_fit = RegimeGatedQRF._fit_target
    real_draw = legacy._draw_target_with_rng

    def inspect_fit(self, **kwargs):
        fit_features.append(kwargs["features"].copy())
        return real_fit(self, **kwargs)

    def inspect_draw(features, target_model, rng):
        draw_features.append(features.copy())
        return real_draw(features, target_model, rng)

    monkeypatch.setattr(RegimeGatedQRF, "_fit_target", inspect_fit)
    monkeypatch.setattr(legacy, "_draw_target_with_rng", inspect_draw)
    artifacts = _fit_chain(model, donors, initial)
    recipients = donors.iloc[:23][["x"]]
    raw, _ = _apply_chain(artifacts, recipients, initial)
    np.testing.assert_array_equal(fit_features[1][:, 1], donors["positive"].to_numpy())
    np.testing.assert_array_equal(draw_features[1]["positive"], raw["positive"])
    assert (raw["positive"] != np.round(raw["positive"])).any()


@pytest.mark.parametrize("drift", ["config", "fit_workers"])
def test_changed_fit_configuration_is_rejected(donors, monkeypatch, drift):
    model = _model()
    initial = model.start_chain(donors, ["x"], ["positive"], weights="weight")
    state = split.LegacyQRFTrainingState.from_chain(initial)
    if drift == "config":
        model = RegimeGatedQRF(n_estimators=4, seed=29)
    else:
        monkeypatch.setenv("POPULACE_FIT_N_JOBS", "2")
    with pytest.raises(ValueError, match="model configuration changed"):
        split.fit_target(model, donors, state=state, weights="weight")


@pytest.mark.parametrize("field", ["schema", "next_state", "draw_state", "versions"])
def test_malformed_envelope_is_rejected_before_unpickle(donors, monkeypatch, field):
    model = _model()
    initial = model.start_chain(donors, ["x"], ["positive"], weights="weight")
    artifact = split.fit_target(
        model,
        donors,
        state=split.LegacyQRFTrainingState.from_chain(initial),
        weights="weight",
    )
    payload = artifact.to_bytes()
    offset = len(split._MAGIC)
    size = int.from_bytes(payload[offset : offset + 8], "big")
    offset += 8
    metadata = json.loads(payload[offset : offset + size])
    if field == "schema":
        metadata["schema_version"] = 2
    elif field == "next_state":
        metadata["next_training_state"]["completed_targets"] = []
    elif field == "draw_state":
        metadata["training_state"]["draw_rng_state"] = np.random.PCG64(0).state
    else:
        metadata["implementation"]["versions"]["numpy"] = "different-version"
    header = json.dumps(metadata).encode()
    changed = (
        split._MAGIC
        + len(header).to_bytes(8, "big")
        + header
        + payload[offset + size :]
    )
    calls = []
    monkeypatch.setattr(split.pickle, "loads", lambda value: calls.append(value))
    with pytest.raises(ValueError):
        split.LegacyQRFTargetArtifact.from_trusted_bytes(
            changed, expected_sha256=hashlib.sha256(changed).hexdigest()
        )
    assert not calls


def test_integrity_platform_and_implementation_checked_before_unpickle(
    donors, monkeypatch
):
    model = _model()
    initial = model.start_chain(donors, ["x"], ["positive"], weights="weight")
    artifact = split.fit_target(
        model,
        donors,
        state=split.LegacyQRFTrainingState.from_chain(initial),
        weights="weight",
    )
    payload = artifact.to_bytes()
    expected = hashlib.sha256(payload).hexdigest()
    called = []
    monkeypatch.setattr(split.pickle, "loads", lambda value: called.append(value))
    with pytest.raises(ValueError, match="digest"):
        split.LegacyQRFTargetArtifact.from_trusted_bytes(
            payload[:-1] + b"!", expected_sha256=expected
        )
    with pytest.raises(ValueError, match="digest"):
        split.LegacyQRFTargetArtifact.from_trusted_bytes(
            payload, expected_sha256="invalid"
        )
    original = split._implementation_binding
    for key, value in (
        ("platform", "other-architecture/linux/py3.13"),
        ("qrf_sha256", "0" * 64),
    ):
        monkeypatch.setattr(
            split,
            "_implementation_binding",
            lambda key=key, value=value: {**original(), key: value},
        )
        with pytest.raises(ValueError, match="implementation"):
            split.LegacyQRFTargetArtifact.from_trusted_bytes(
                payload, expected_sha256=expected
            )
        with pytest.raises(ValueError, match="implementation"):
            split.apply_target(
                artifact, donors[["x"]], pd.DataFrame(index=donors.index), state=initial
            )
    assert called == []


def test_apply_retains_prediction_chunk_ceiling(donors, monkeypatch):
    model = _model()
    initial = model.start_chain(donors, ["x"], ["positive"], weights="weight")
    artifact = split.fit_target(
        model,
        donors,
        state=split.LegacyQRFTrainingState.from_chain(initial),
        weights="weight",
    )
    sizes = []
    original = legacy.RandomForestQuantileRegressor.predict

    def observe(self, x, *args, **kwargs):
        sizes.append(len(x))
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(legacy.RandomForestQuantileRegressor, "predict", observe)
    recipients = pd.DataFrame({"x": np.linspace(-2, 2, 40_001)})
    step = split.apply_target(
        artifact, recipients, pd.DataFrame(index=recipients.index), state=initial
    )
    assert len(step.raw_draw) == 40_001
    assert sum(sizes) == 40_001
    assert max(sizes) == legacy._PREDICT_CHUNK_ROWS == 10_000
