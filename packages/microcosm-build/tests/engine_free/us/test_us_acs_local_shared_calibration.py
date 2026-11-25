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
        registry=registry,
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


# ---------------------------------------------------------------------------
# Item 3: a loss cap for the county rows only
# ---------------------------------------------------------------------------


def _cli(module, *options):
    return module._parse_args(
        [
            "--stage",
            "calibrate",
            "--staging-h5",
            "staging.h5",
            "--checkpoint-dir",
            "checkpoint",
            "--out-h5",
            "calibrated.h5",
            *options,
        ]  # fmt: skip
    )


def test_the_county_cap_applies_to_the_run_by_default() -> None:
    module = _FIXTURE._load_tool_module()
    args = _cli(module, "--geographies", "state,cd,county")
    assert args.county_target_loss_cap_scope == "run"
    assert args.target_loss_cap == 10.0
    assert args.county_rows_target_loss_cap is None
    assert args.county_solver_profile == {
        "scope": "run",
        "target_loss_cap": 10.0,
        "max_weight_ratio": 5.0,
    }


def test_the_county_rows_scope_keeps_the_cap_of_the_other_rows() -> None:
    module = _FIXTURE._load_tool_module()
    args = _cli(
        module,
        "--geographies", "state,cd,county",
        "--county-target-loss-cap-scope", "county_rows",
        "--target-loss-cap", "2",
    )  # fmt: skip
    assert args.target_loss_cap == 2.0
    assert args.county_rows_target_loss_cap == 10.0
    settings = module._solver_settings(args, {})
    assert settings["target_loss_cap"] == 2.0
    # The scope is in the stamp, so a resume refuses weights of the other one.
    assert settings["county_solver_profile"] == {
        "scope": "county_rows",
        "target_loss_cap": 10.0,
        "max_weight_ratio": 5.0,
    }


def test_the_scope_changes_nothing_without_county_rows() -> None:
    module = _FIXTURE._load_tool_module()
    plain = _cli(module)
    scoped = _cli(module, "--county-target-loss-cap-scope", "county_rows")
    assert scoped.county_rows_target_loss_cap is None
    assert module._solver_settings(scoped, {}) == module._solver_settings(plain, {})


def _joint_problem(module):
    """Three households; a county row and a state row, each a 200% miss."""

    import pandas as pd
    from scipy import sparse

    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    design = np.asarray([300.0, 300.0, 300.0])
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": [1, 2, 3], "person_household_id": [1, 2, 3]}
            ),
            "household": pd.DataFrame(
                {
                    "household_id": [1, 2, 3],
                    "state_fips": [1, 1, 1],
                    "county_fips": [1001, 1003, 1005],
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(design, WeightKind.DESIGN)},
    )
    county = module.county_surface.population_spec("pop_county_01001", 100.0)
    (state,) = module.population_target_specs(["pop_state_01"], [100.0])
    records = [
        {**module.county_surface.target_record(county), "role": "train"},
        {**module.cd_surface.target_record(state), "role": "train"},
    ]
    targets = module.cd_surface.calibration_target_set(
        records,
        sparse.csr_array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        3,
        specs=[county, state],
    )
    return frame, targets, [county, state], design


def _solve_joint(module, *options):
    args = _cli(module, "--geographies", "state,county", *options)
    frame, targets, specs, design = _joint_problem(module)
    loss = module.release_target_loss_caps(args, targets, specs, np.ones(2))
    result, _done = module.calibrate_surface(
        frame,
        targets,
        epochs=80,
        epoch_batch=40,
        max_weight_ratio=args.max_weight_ratio,
        target_loss_cap=loss["cap"],
        l2_lambda=0.0,
        seed=0,
        target_loss_weights=loss["weights"],
        target_loss_scales=loss["scales"],
    )
    return loss, result, design


def test_the_run_scope_passes_one_cap_and_no_scales() -> None:
    module = _FIXTURE._load_tool_module()
    loss, result, design = _solve_joint(module)
    assert loss == {
        "cap": 10.0,
        "weights": loss["weights"],
        "scales": None,
        "receipt": None,
    }
    np.testing.assert_array_equal(loss["weights"], np.ones(2))
    # Cap 10 gives both rows a gradient, so both far misses are corrected.
    assert result.weights[0] < design[0] and result.weights[1] < design[1]


def test_the_county_rows_scope_corrects_the_county_row_only() -> None:
    module = _FIXTURE._load_tool_module()
    loss, result, design = _solve_joint(
        module, "--county-target-loss-cap-scope", "county_rows"
    )
    # Declared caps: county 10, state 1 (the default --target-loss-cap).
    assert loss["receipt"] == {
        "default": 1.0,
        "by_level": {"county": 10.0},
        "method": "single_cap_equivalent",
        "equivalent_cap": 5.5,
        "rows_by_cap": {"1.0": 1, "10.0": 1},
    }
    # The solver's one-cap loss gives each row its own cap in its own units:
    # scale * cap0 / (target scale) is the miss at which the row goes flat.
    flat_at = loss["scales"] * loss["cap"] / 100.0
    np.testing.assert_allclose(flat_at, [10.0, 1.0])
    assert result.weights.sum() == pytest.approx(design.sum())
    # The county row (200% miss, cap 10) has a gradient and moves to its target.
    assert abs(result.weights[0] - 100.0) < abs(design[0] - 100.0)
    # The state row (200% miss, cap 1) is past its cap, as in a run without
    # county rows: it has no gradient, so its household is not pulled down.
    assert result.weights[1] >= design[1]


def test_a_county_rows_cap_without_a_county_training_row_is_refused() -> None:
    module = _FIXTURE._load_tool_module()
    args = _cli(
        module,
        "--geographies", "state,county",
        "--county-target-loss-cap-scope", "county_rows",
    )  # fmt: skip
    frame, targets, specs, _design = _joint_problem(module)
    with pytest.raises(SystemExit, match="level 'county' matches no target"):
        module.release_target_loss_caps(args, targets[1:], specs[1:], np.ones(1))


def test_diagnostics_of_a_row_cap_solve_match_the_published_schema(
    calibrate_run, tmp_path
) -> None:
    from microcosm.calibrate import default_target_loss_scales
    from microcosm.calibrate.target_loss_rules import (
        single_cap_equivalent,
        target_loss_caps,
    )

    call = calibrate_run.run()[0]
    train = calibrate_run.train
    caps = target_loss_caps(
        1.0,
        levels=[lw.target_geography_level(spec) for spec in train],
        by_level={"congressional_district": 10.0},
    )
    equivalent = single_cap_equivalent(
        caps,
        weights=call["kwargs"]["target_loss_weights"],
        scales=default_target_loss_scales(
            np.asarray([target.value for target in call["targets"]])
        ),
    )
    result = calibrate_run.real_calibrate(
        call["frame"],
        call["targets"],
        **{
            **call["kwargs"],
            "target_loss_cap": equivalent.cap,
            "target_loss_weights": equivalent.weights,
            "target_loss_scales": equivalent.scales,
        },
    )
    written = tmp_path / "row_cap_diagnostics.json"
    outcome = calibrate_package.write_calibration_diagnostics(
        result, written, target_registry=calibrate_run.registry
    )

    assert outcome.status == "available"
    assert (
        _FIXTURE.published_json_schema_errors(
            _FIXTURE.load_strict_json(written.read_bytes())
        )
        == []
    )
    assert result.options["target_loss_scales"]["kind"] == "provided"
    assert result.target_loss_cap == pytest.approx(equivalent.cap)


# ---------------------------------------------------------------------------
# Item 4: the district gate reads the shared area support gate
# ---------------------------------------------------------------------------


_ESS = st.floats(0.0, 500.0, allow_nan=False) | st.sampled_from([0.0, 15.0, 60.0])
_DISTRICT_ESS = st.dictionaries(
    st.sampled_from(["0601", "0602", "2401", "3601", "3602", "5000"]),
    st.tuples(_ESS, _ESS),
    min_size=1,
)
_GATE_FLOORS = st.tuples(
    st.sampled_from([0.0, 0.25, 0.5, 1.0]) | st.floats(0.0, 1.0),
    st.sampled_from([0.0, 15.0]) | st.floats(0.0, 100.0),
    st.booleans(),
)


@settings(max_examples=300, deadline=None)
@given(districts=_DISTRICT_ESS, floors=_GATE_FLOORS)
def test_the_district_gate_entry_is_the_entry_of_before(districts, floors) -> None:
    from test_support.microcosm_build import us_district_ess_gate_reference as old

    module = _FIXTURE._load_tool_module()
    origin = {
        "design": {
            "effective_sample_size_by_district": {
                district: design for district, (design, _c) in districts.items()
            }
        },
        "calibrated": {
            "effective_sample_size_by_district": {
                district: calibrated for district, (_d, calibrated) in districts.items()
            }
        },
    }
    relative_floor, absolute_floor, blocking = floors
    arguments = dict(
        relative_floor=relative_floor, absolute_floor=absolute_floor, blocking=blocking
    )

    entry = module.district_ess_collapse_gate(origin, **arguments)

    assert entry == old.district_ess_collapse_gate(origin, **arguments)
    assert entry["report_only"] is (not blocking)
    if not blocking:
        # Report-only: the entry passes whatever it measured.
        assert entry["passed"] is True


def test_the_district_gate_is_report_only_by_default(tmp_path) -> None:
    module = _FIXTURE._load_tool_module()
    args = _cli(module)
    assert args.district_ess_gate_blocking is False
    collapsed = {
        "design": {"effective_sample_size_by_district": {"0601": 400.0}},
        "calibrated": {"effective_sample_size_by_district": {"0601": 2.0}},
    }
    entry = module.district_ess_collapse_gate(collapsed)
    assert entry["criteria_met"] is False
    assert entry["passed"] is True and entry["blocking"] is False
    assert module.district_ess_collapse_failures(entry) == []


def test_the_tools_hold_no_copy_of_the_ess_or_gate_code() -> None:
    module = _FIXTURE._load_tool_module()
    assert not hasattr(module.cd_surface, "ess_by_group")
    repository = _FIXTURE._TEST_PATHS.repository
    for tool in ("build_us_acs_local_release.py", "us_acs_local_cd_surface.py"):
        source = (repository / "tools" / tool).read_text()
        # No Kish arithmetic of its own: sum(w)^2 over sum(w^2).
        assert "** 2 / (" not in source and '["w"] ** 2' not in source
    gate = (repository / "tools" / "build_us_acs_local_release.py").read_text()
    body = gate[
        gate.index("def district_ess_collapse_gate(") : gate.index(
            "def district_ess_collapse_failures("
        )
    ]
    # The floors are compared in the shared gate, not here.
    assert "area_support_gate(" in body and "area_support_floor_table(" in body
    assert "relative_floor *" not in body and "< absolute_floor" not in body


# ---------------------------------------------------------------------------
# Item 1: the L2 anchor
# ---------------------------------------------------------------------------


def test_the_cli_exposes_the_l2_anchor_with_the_default_of_every_solve(
    tmp_path,
) -> None:
    module = _FIXTURE._load_tool_module()
    default = _cli(module)
    assert (default.l2_anchor, default.l2_anchor_weights) == ("initial", None)
    assert _cli(module, "--l2-anchor", "uniform").l2_anchor == "uniform"
    anchor = tmp_path / "anchor.npy"
    named = _cli(
        module,
        "--l2-anchor",
        "household_proportional",
        "--l2-anchor-weights",
        str(anchor),
    )
    assert (named.l2_anchor, named.l2_anchor_weights) == (
        "household_proportional",
        anchor,
    )
    for bad in (
        ("--l2-anchor", "household_proportional"),  # a name with no vector
        ("--l2-anchor-weights", str(anchor)),  # a vector with no name
        ("--l2-anchor", "uniform", "--l2-anchor-weights", str(anchor)),
    ):
        with pytest.raises(SystemExit):
            _cli(module, *bad)


def test_the_default_anchor_adds_no_keyword_and_no_stamp() -> None:
    module = _FIXTURE._load_tool_module()
    args = _cli(module)
    assert module.release_l2_anchor(args, np.arange(5)) == ({}, None)
    assert "l2_anchor" not in module._solver_settings(args, {}, None)


def test_the_uniform_anchor_reaches_calibrate_and_the_stamp(calibrate_run) -> None:
    calls = calibrate_run.run(
        l2_anchor="uniform", l2_lambda=0.05, l2_basis="chi_square"
    )
    for call in calls:
        assert set(call["kwargs"]) == HISTORICAL_CALIBRATE_KEYWORDS | {"l2_anchor"}
        assert call["kwargs"]["l2_anchor"] == "uniform"
    summary = _summary(calibrate_run)
    assert summary["l2_anchor"] == "uniform"
    assert summary["solver_settings"]["l2_anchor"] == "uniform"
    assert "l2_anchor_weights_sha256" not in summary["solver_settings"]


def test_anchor_weights_pull_toward_a_declared_design(calibrate_run) -> None:
    from microcosm.calibrate import chi_square_distance

    module = calibrate_run.module
    design = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    # A declared design that is not the starting weights: equal households.
    declared = np.full(5, 3.0)
    anchor = calibrate_run.path / "anchor.npz"
    np.savez(anchor, weights=declared, household_id=np.arange(1, 6))
    penalty = dict(l2_lambda=5.0, l2_basis="chi_square", epochs=60, epoch_batch=30)

    calibrate_run.run(**penalty)
    toward_start = _saved_weights(calibrate_run)
    calls = calibrate_run.run(
        **penalty, l2_anchor="equal_households", l2_anchor_weights=anchor
    )
    toward_declared = _saved_weights(calibrate_run)

    for call in calls:
        assert call["kwargs"]["l2_anchor"] == "equal_households"
        np.testing.assert_array_equal(call["kwargs"]["l2_anchor_weights"], declared)
    # The starting weights, and so the bound and the mass, did not change.
    saved = np.load(calibrate_run.path / "weights_latest.npz")
    np.testing.assert_array_equal(saved["initial_weights"], design)
    assert toward_declared.sum() == pytest.approx(design.sum())
    assert (toward_declared <= 5.0 * design * (1 + 1e-12)).all()
    # The penalty pulled toward the declared design instead of the start.
    assert chi_square_distance(toward_declared, declared) < chi_square_distance(
        toward_start, declared
    )
    stamp = _summary(calibrate_run)["solver_settings"]
    assert stamp["l2_anchor"] == "equal_households"
    assert stamp["l2_anchor_weights_sha256"] == module._weights_digest(declared)
    # Another anchor is another solve: a resume refuses these weights.
    with pytest.raises(SystemExit, match="different solver settings"):
        calibrate_run.run(**penalty, resume=True)


def test_anchor_weights_must_fit_the_checkpoint(calibrate_run, tmp_path) -> None:
    module = calibrate_run.module
    households = np.arange(1, 6)

    def anchor(name, **arrays):
        path = tmp_path / name
        if name.endswith(".npy"):
            np.save(path, arrays["weights"])
        else:
            np.savez(path, **arrays)
        return SimpleNamespace(l2_anchor="declared", l2_anchor_weights=path)

    keywords, stamp = module.release_l2_anchor(
        anchor("plain.npy", weights=np.ones(5)), households
    )
    assert keywords["l2_anchor"] == "declared" and stamp["l2_anchor"] == "declared"
    np.testing.assert_array_equal(keywords["l2_anchor_weights"], np.ones(5))
    for name, arrays, match in (
        ("short.npy", {"weights": np.ones(4)}, "has shape"),
        ("zero.npy", {"weights": np.asarray([1.0, 1, 1, 1, 0])}, "strictly positive"),
        ("nokey.npz", {"anchor": np.ones(5)}, "has no 'weights' array"),
        (
            "order.npz",
            {"weights": np.ones(5), "household_id": households[::-1]},
            "other households",
        ),
    ):
        with pytest.raises(SystemExit, match=match):
            module.release_l2_anchor(anchor(name, **arrays), households)
