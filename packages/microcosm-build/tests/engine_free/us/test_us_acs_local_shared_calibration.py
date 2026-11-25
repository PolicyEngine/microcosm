"""The ACS local release reads the shared calibration machinery (#1729).

Two kinds of test:

- **Defaults give the same solve.** With the default declarations the tool
  calls ``calibrate`` with exactly the arguments it used before, and the
  calibrated weights are the bytes a direct call with those arguments returns.
  The loss weights are held to a verbatim copy of the formula as it stood
  before it became a shared rule.
- **Each option is off until a build turns it on**, and then does what its
  declaration says.

A small checkpoint goes through the real ``do_calibrate``; the fixture is the
one ``test_us_acs_local_target_loss_weights.py`` builds.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import microcosm.calibrate as calibrate_package
from microcosm.build.us_runtime import target_loss_weights as lw
from microcosm.calibrate.target_loss_rules import TargetWeightRule
from test_support.microcosm_build import us_target_loss_rows_reference as before
from test_support.microcosm_build.us_batched_target_materialization import (
    _nested_frame,
)


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        f"shared_calibration_{name}", Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_FIXTURE = _sibling("test_us_acs_local_target_loss_weights")

#: Every keyword the calibrate stage passed to ``calibrate`` before #1729.
HISTORICAL_CALIBRATE_KEYWORDS = {
    "weight_entity",
    "method",
    "epochs",
    "learning_rate",
    "mass",
    "max_weight_ratio",
    "target_loss_cap",
    "l2_lambda",
    "l2_basis",
    "mass_parametrization",
    "seed",
    "target_loss_weights",
    "warm_start_weights",
}


@pytest.fixture
def calibrate_run(monkeypatch, tmp_path):
    """Run ``do_calibrate`` on the fixture; keep every ``calibrate`` call."""

    module = _FIXTURE._load_tool_module()
    registry, records, matrix, design = _FIXTURE._surface(module)
    frame = _nested_frame()
    identity = {"staging_sha256": "s", "target_roles_sha256": "r"}
    monkeypatch.setattr(module, "_verify_run_identity", lambda args: identity)
    monkeypatch.setattr(
        module,
        "load_checkpoint_surface",
        lambda *_a, **_k: (frame, design, registry, records, matrix),
    )
    calls: list[dict] = []
    real_calibrate = calibrate_package.calibrate

    def spy(frame, targets, **kwargs):
        result = real_calibrate(frame, targets, **kwargs)
        calls.append({"frame": frame, "targets": targets, "kwargs": dict(kwargs)})
        return result

    monkeypatch.setattr(calibrate_package, "calibrate", spy)
    release_tool_tests = _FIXTURE._load_release_tool_tests()

    def run(**overrides):
        calls.clear()
        args = _FIXTURE._args(tmp_path, **overrides)
        release_tool_tests._record_current_materialization(module, args, identity)
        module.do_calibrate(args)
        return calls

    held = {record["name"] for record in records if record["role"] == "holdout"}
    return SimpleNamespace(
        module=module,
        run=run,
        path=tmp_path,
        real_calibrate=real_calibrate,
        train=[spec for spec in registry.specs if spec.name not in held],
    )


def _summary(run) -> dict:
    return json.loads((run.path / "calibration_summary.json").read_text())


def _saved_weights(run) -> np.ndarray:
    return np.load(run.path / "weights_latest.npz")["weights"]


def _reference_weights(train) -> np.ndarray:
    """The loss weights the formula gave before it became a shared rule."""

    mapping = lw.US_ACS_LOCAL_TARGET_LOSS_ROW_MAPPING
    return before.target_loss_weights_from_rows(
        [spec.value for spec in train],
        [mapping.value_basis(spec) for spec in train],
        [mapping.concept_budget_key(spec) for spec in train],
        [spec.family for spec in train],
    )


# ---------------------------------------------------------------------------
# Defaults give the same solve
# ---------------------------------------------------------------------------


def test_the_default_run_calls_calibrate_exactly_as_before(calibrate_run) -> None:
    calls = calibrate_run.run()
    reference = _reference_weights(calibrate_run.train)

    assert len(calls) == 2  # 6 epochs in batches of 3
    for index, call in enumerate(calls):
        kwargs = call["kwargs"]
        # No new keyword reaches the solver unless a build asks for it.
        assert set(kwargs) == HISTORICAL_CALIBRATE_KEYWORDS
        assert {
            key: kwargs[key]
            for key in HISTORICAL_CALIBRATE_KEYWORDS
            - {"target_loss_weights", "warm_start_weights"}
        } == {
            "weight_entity": "household",
            "method": "adam",
            "epochs": 3,
            "learning_rate": 0.02,
            "mass": "conserve",
            "max_weight_ratio": 5.0,
            "target_loss_cap": 1.0,
            "l2_lambda": 0.0,
            "l2_basis": "record",
            "mass_parametrization": "projection",
            "seed": 0,
        }
        assert isinstance(kwargs["target_loss_cap"], float)
        # The loss weights are the pre-change formula's bytes.
        assert kwargs["target_loss_weights"].tobytes() == reference.tobytes()
        assert (kwargs["warm_start_weights"] is None) == (index == 0)


def test_the_default_run_returns_the_weights_of_the_historical_call(
    calibrate_run,
) -> None:
    calls = calibrate_run.run()
    produced = _saved_weights(calibrate_run)

    # The same two batches, called directly with the historical arguments and
    # the pre-change loss weights: nothing of the tool or the shared rule.
    warm = None
    for call in calls:
        result = calibrate_run.real_calibrate(
            call["frame"],
            call["targets"],
            weight_entity="household",
            method="adam",
            epochs=3,
            learning_rate=0.02,
            mass="conserve",
            max_weight_ratio=5.0,
            target_loss_cap=1.0,
            l2_lambda=0.0,
            l2_basis="record",
            mass_parametrization="projection",
            seed=0,
            target_loss_weights=_reference_weights(calibrate_run.train),
            warm_start_weights=warm,
        )
        warm = result.weights.copy()

    assert produced.tobytes() == warm.tobytes()
    assert not np.array_equal(produced, [1.0, 2.0, 3.0, 4.0, 5.0])


def test_the_default_stamp_and_summary_record_nothing_new(calibrate_run) -> None:
    calibrate_run.run()
    summary = _summary(calibrate_run)

    assert set(summary["solver_settings"]) == {
        "method",
        "learning_rate",
        "mass",
        "max_weight_ratio",
        "target_loss_cap",
        "l2_lambda",
        "l2_basis",
        "mass_parametrization",
        "seed",
        "epoch_batch",
        "target_loss",
    }
    assert set(summary["solver_settings"]["target_loss"]) == {
        "weighting",
        "row_mapping",
        "family_multipliers",
        "n_targets",
        "weights_sha256",
    }
    assert summary["solver_settings"]["target_loss"]["weighting"] == (
        lw.US_FISCAL_TARGET_LOSS_WEIGHTING
    )


_BASES = st.sampled_from(["count", "amount"])
_ROWS = st.lists(
    st.tuples(
        st.floats(-1e12, 1e12, allow_nan=False),
        _BASES,
        st.integers(0, 6),
        st.sampled_from(["irs_soi", "usda_snap", "census_population", None]),
    ),
    min_size=0,
    max_size=40,
)


@settings(max_examples=200, deadline=None)
@given(rows=_ROWS, multiplier=st.sampled_from([None, 0.25, 3.0]))
def test_the_shared_rule_is_the_formula_bit_for_bit(rows, multiplier) -> None:
    values = [row[0] for row in rows]
    bases = [row[1] for row in rows]
    keys = [row[2] for row in rows]
    families = [row[3] for row in rows]
    multipliers = (
        {"irs_soi": multiplier}
        if multiplier is not None and "irs_soi" in families
        else None
    )

    moved = lw.target_loss_weights_from_rows(values, bases, keys, families, multipliers)
    reference = before.target_loss_weights_from_rows(
        values, bases, keys, families, multipliers
    )

    assert moved.dtype == reference.dtype
    assert moved.tobytes() == reference.tobytes()


def test_an_unmatched_multiplier_keeps_the_release_wording() -> None:
    arguments = ([1.0, 2.0], ["count", "amount"], [0, 1], ["a", "b"])
    for function in (
        lw.target_loss_weights_from_rows,
        before.target_loss_weights_from_rows,
    ):
        with pytest.raises(
            ValueError,
            match="--target-family-loss-multiplier family 'c' matches no compiled",
        ):
            function(*arguments, {"c": 2.0, "a": 3.0})


# ---------------------------------------------------------------------------
# Item 2: the weight rule is a declaration
# ---------------------------------------------------------------------------


def test_the_release_declares_its_weight_rules_as_data() -> None:
    assert lw.US_ACS_LOCAL_TARGET_WEIGHT_RULE == "concept_budget"
    assert lw.US_ACS_LOCAL_TARGET_WEIGHT_RULES["concept_budget"] == TargetWeightRule(
        "sqrt_value_concept_budget", {"value_power": 0.5}
    )
    assert lw.acs_local_target_weight_rule("level_family_equal_sqrt_count") == (
        TargetWeightRule("level_family_equal", {"count_power": 0.5})
    )
    with pytest.raises(ValueError, match="must be one of"):
        lw.acs_local_target_weight_rule("grain_equal")


def test_the_cli_defaults_to_the_formula_every_build_has_used(tmp_path) -> None:
    module = _FIXTURE._load_tool_module()
    base = [
        "--stage", "calibrate",
        "--staging-h5", str(tmp_path / "s.h5"),
        "--checkpoint-dir", str(tmp_path),
        "--out-h5", str(tmp_path / "out.h5"),
    ]  # fmt: skip
    assert module._parse_args(base).target_weight_rule == "concept_budget"
    chosen = module._parse_args(
        [*base, "--target-weight-rule", "level_family_equal_sqrt_count"]
    )
    assert chosen.target_weight_rule == "level_family_equal_sqrt_count"
    with pytest.raises(SystemExit):
        module._parse_args([*base, "--target-weight-rule", "uniform"])


def test_the_level_rule_gives_each_geography_level_one_share(calibrate_run) -> None:
    calls = calibrate_run.run(target_weight_rule="level_family_equal_sqrt_count")
    train = calibrate_run.train
    weights = calls[0]["kwargs"]["target_loss_weights"]
    levels = np.asarray([lw.target_geography_level(spec) for spec in train])

    assert set(levels) == {"state", "congressional_district"}
    for level in set(levels):
        assert weights[levels == level].sum() / weights.sum() == pytest.approx(0.5)
    # Under the default formula the two levels do not share equally.
    default = _reference_weights(train)
    state_share = default[levels == "state"].sum() / default.sum()
    assert abs(state_share - 0.5) > 0.05

    stamp = _summary(calibrate_run)["solver_settings"]["target_loss"]
    assert stamp["weighting"] == "level_family_equal_sqrt_count"
    assert stamp["rule"] == {
        "rule": "level_family_equal",
        "parameters": {"count_power": 0.5},
    }


def test_resume_refuses_weights_calibrated_under_another_rule(calibrate_run) -> None:
    calibrate_run.run()
    with pytest.raises(SystemExit, match="different solver settings"):
        calibrate_run.run(
            resume=True, target_weight_rule="level_family_equal_sqrt_count"
        )
