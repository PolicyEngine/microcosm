"""The UK size-experiment harness, scorecard and tool (microcosm#1124)."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys

import numpy as np
import pytest

from microcosm.build.uk_runtime import dataset_size
from microcosm.build.uk_runtime.local_doctrine import UK_LOCAL_TARGET_WEIGHT_RULES
from microcosm.build.uk_runtime.size_experiment import (
    UKSizeExperiment,
    build_uk_size_experiment_cache,
    load_uk_size_experiment_baseline,
    load_uk_size_scoring_inputs,
    parse_uk_size_experiments,
    run_uk_size_census,
    run_uk_size_control,
    run_uk_size_experiment,
    score_uk_size_experiments,
    uk_size_step1b_plan,
    uk_size_weights_of,
)
from microcosm.build.uk_runtime.size_experiment_scorecard import (
    build_uk_pool_profile,
    disclosure_controlled,
    load_uk_pool_profile,
    uk_size_acceptance,
    write_uk_pool_profile,
)
from microcosm.calibrate.artifacts import encode_solution
from test_support.microcosm_build.uk_size_experiment import (
    synthetic_uk_pool,
    synthetic_uk_size_run,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("uk-size-experiment")
    run = synthetic_uk_size_run(root / "run")
    build_uk_size_experiment_cache(root / "run", root / "cache")
    return root, run


@pytest.fixture
def baseline(built):
    root, _ = built
    return load_uk_size_experiment_baseline(root / "run", root / "cache")


def test_cache_and_baseline_decode_the_run_and_find_its_rule(built, baseline) -> None:
    root, run = built
    manifest = json.loads((root / "cache/size_experiment_cache.json").read_text())
    assert manifest["households"] == run["frame"].n("household")
    assert baseline.run_rule == "grain_equal"
    assert baseline.settings["households"] == run["k"]
    assert baseline.settings["pi_hi"] == run["settings"]["pi_hi"]
    np.testing.assert_array_equal(baseline.draw.support, run["draw"].support)
    # the skeleton keeps ids and memberships only
    assert list(baseline.frame.table("household").columns) == ["household_id"]
    assert set(baseline.frame.table("person").columns) == {
        "person_id",
        "person_household_id",
        "person_benunit_id",
    }
    with pytest.raises(ValueError, match="outside the run directory"):
        build_uk_size_experiment_cache(root / "run", root / "run" / "cache")


def test_control_reproduces_the_stored_refit_exactly(baseline, built) -> None:
    _, run = built
    control = run_uk_size_control(baseline)
    assert control["passed"] and control["exact"]
    assert control["receipt_differences"] == []
    np.testing.assert_array_equal(
        control["sized"].result.weights, run["sized"].result.weights
    )


def test_control_and_variants_carry_the_stored_stages_l2(tmp_path) -> None:
    synthetic_uk_size_run(
        tmp_path / "run", selection_l2_lambda=1e-3, refit_l2_lambda=0.02
    )
    build_uk_size_experiment_cache(tmp_path / "run", tmp_path / "cache")
    baseline = load_uk_size_experiment_baseline(tmp_path / "run", tmp_path / "cache")
    assert baseline.settings["selection_l2"]["lambda"] == 1e-3
    assert baseline.settings["refit_l2"] == {
        "lambda": 0.02,
        "anchor": "initial",
        "basis": "chi_square",
    }
    control = run_uk_size_control(baseline)
    assert control["passed"] and control["exact"]
    # A variant on the stored search names the search's own L2 for reuse.
    outcome = run_uk_size_experiment(baseline, UKSizeExperiment(name="r"))
    assert outcome["results"]["refit"].receipt["selection_l2"]["lambda"] == 1e-3
    assert "refit_l2" not in outcome["results"]["refit"].receipt
    holdout = run_uk_size_experiment(
        baseline, UKSizeExperiment(name="h", mode="refit_holdout")
    )
    assert len(holdout["receipt"]["holdout"]["folds"]) == 5


def test_control_fails_when_the_stored_refit_differs(built, tmp_path) -> None:
    root, run = built
    clone = tmp_path / "run"
    import shutil

    shutil.copytree(root / "run", clone)
    evidence = json.loads((clone / "evidence-index.json").read_text())
    entry = evidence["uk.full.size_refit/solution"]
    sized = run["sized"]
    payload = encode_solution(
        np.asarray(sized.result.weights) * (1.0 + 1e-3),
        entity_ids=sized.result.frame.table("household")["household_id"].tolist(),
        problem_sha256=run["problem"].sha256,
    )
    (clone / entry["filename"]).write_bytes(payload)
    entry.update(sha256=hashlib.sha256(payload).hexdigest(), size_bytes=len(payload))
    (clone / "evidence-index.json").write_text(json.dumps(evidence))
    manifest = json.loads((root / "cache/size_experiment_cache.json").read_text())
    shutil.copytree(root / "cache", tmp_path / "cache")
    manifest["run_dir"] = str(clone.resolve())
    (tmp_path / "cache/size_experiment_cache.json").write_text(json.dumps(manifest))
    control = run_uk_size_control(
        load_uk_size_experiment_baseline(clone, tmp_path / "cache")
    )
    assert not control["passed"] and not control["exact"]
    assert control["max_relative_weight_difference"] == pytest.approx(1e-3, rel=1e-2)


def test_a_tampered_artifact_is_refused(built, tmp_path) -> None:
    root, _ = built
    import shutil

    clone = tmp_path / "run"
    shutil.copytree(root / "run", clone)
    path = clone / "uk.full.size_draw.draw.json"
    path.write_bytes(path.read_bytes().replace(b"exact_count", b"exact_count "))
    shutil.copytree(root / "cache", tmp_path / "cache")
    manifest = json.loads((tmp_path / "cache/size_experiment_cache.json").read_text())
    manifest["run_dir"] = str(clone.resolve())
    (tmp_path / "cache/size_experiment_cache.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="evidence-index digest"):
        load_uk_size_experiment_baseline(clone, tmp_path / "cache")


def test_refit_experiments_vary_rule_l2_and_floor_on_the_stored_support(
    baseline,
) -> None:
    rule = run_uk_size_experiment(
        baseline, UKSizeExperiment(name="a", refit_rule="nation_grain_family_equal")
    )
    penalized = run_uk_size_experiment(
        baseline,
        UKSizeExperiment(
            name="e",
            refit_l2={"lambda": 0.03, "anchor": "initial"},
            baseline_pi_floor=0.5,
        ),
    )
    for outcome in (rule, penalized):
        sized = outcome["results"]["refit"]
        np.testing.assert_array_equal(sized.support, baseline.draw.support)
        receipt = outcome["receipt"]
        assert {"training", "grain_equal_yardstick", "family_equal_yardstick"} <= set(
            receipt["losses"]
        )
        assert receipt["peak_rss_bytes"] > 0 and receipt["wall_seconds"] >= 0.0
        assert "pool_row_indices" not in receipt["size_receipt"]
    assert rule["receipt"]["size_receipt"]["refit_target_weighting"]["rule"] == (
        "nation_grain_family_equal"
    )
    block = penalized["receipt"]["size_receipt"]["refit_l2"]
    assert block["anchor_reference"] == "normalized_horvitz_thompson_w_over_q_floored"
    assert penalized["receipt"]["settings"]["baseline_pi_floor"] == 0.5
    perturbed = run_uk_size_experiment(
        baseline, UKSizeExperiment(name="lr", learning_rate=0.049)
    )
    assert perturbed["receipt"]["refit_learning_rate_override"] == 0.049


def test_refit_holdout_trains_without_each_local_fold(baseline) -> None:
    outcome = run_uk_size_experiment(
        baseline, UKSizeExperiment(name="h", mode="refit_holdout")
    )
    holdout = outcome["receipt"]["holdout"]
    folds = holdout["folds"]
    assert len(folds) == 5
    assert sum(fold["n_held"] for fold in folds) == int(np.sum(baseline.rows.local))
    for fold in outcome["results"].values():
        weights = fold.result.target_loss_weights
        assert (weights == 0).sum() >= 1
        national = ~np.asarray(baseline.rows.local)
        assert (weights[national] > 0).all()
    assert 0.0 <= holdout["mean"]["held_within_25pct"] <= 1.0


def test_selection_experiment_searches_again_from_the_stored_dense(baseline) -> None:
    cold = run_uk_size_experiment(
        baseline,
        UKSizeExperiment(
            name="s",
            mode="selection",
            selection_rule="grain_family_equal",
            initial_lambda=None,
        ),
    )
    assert cold["receipt"]["search"]["l0_lambda"] > 0
    assert cold["results"]["refit"].receipt["selection_target_weighting"]["rule"] == (
        "grain_family_equal"
    )
    warm = UKSizeExperiment(name="w", mode="selection", initial_lambda="scaled")
    if (
        "initial_lambda"
        in inspect.signature(dataset_size.select_uk_dataset_size).parameters
    ):
        assert run_uk_size_experiment(baseline, warm)["receipt"]["initial_lambda"] > 0
    else:
        with pytest.raises(RuntimeError, match="microcosm#1115"):
            run_uk_size_experiment(baseline, warm)


def test_experiment_declarations_refuse_unknown_or_inconsistent_settings() -> None:
    parsed = parse_uk_size_experiments(
        [
            {"name": "a", "refit_rule": "grain_family_equal"},
            {"name": "b", "mode": "refit_holdout"},
        ]
    )
    assert [experiment.name for experiment in parsed] == ["a", "b"]
    with pytest.raises(ValueError, match="unknown keys"):
        parse_uk_size_experiments([{"name": "a", "lambda": 0.1}])
    with pytest.raises(ValueError, match="unique"):
        parse_uk_size_experiments([{"name": "a"}, {"name": "a"}])
    with pytest.raises(ValueError, match="mode must be"):
        UKSizeExperiment(name="a", mode="dense")
    with pytest.raises(ValueError, match="need mode 'selection'"):
        UKSizeExperiment(name="a", selection_l2={"lambda": 0.1})
    with pytest.raises(ValueError, match="unknown rule"):
        UKSizeExperiment(name="a", refit_rule="family_equal")
    with pytest.raises(ValueError, match="'record' is refused"):
        UKSizeExperiment(name="a", refit_l2={"lambda": 0.1, "basis": "record"})
    with pytest.raises(ValueError, match="plain directory name"):
        UKSizeExperiment(name="../a")
    for reserved in ("control", "census"):
        with pytest.raises(ValueError, match="reserved"):
            UKSizeExperiment(name=reserved)
    with pytest.raises(ValueError, match="refit-only override"):
        UKSizeExperiment(name="a", mode="selection", epochs=2)
    with pytest.raises(ValueError, match="refit-only override"):
        UKSizeExperiment(name="a", epochs=0)


def test_an_epoch_override_shortens_only_the_refit(baseline) -> None:
    outcome = run_uk_size_experiment(baseline, UKSizeExperiment(name="s", epochs=2))
    sized = outcome["results"]["refit"]
    assert outcome["receipt"]["refit_epochs_override"] == 2
    assert sized.receipt["refit_epochs"] == 2
    assert len(sized.result.loss_trajectory) == 2
    # the stored search is reused as stored: no new search ran
    assert sized.receipt["selection_reused"] is True


def test_census_reads_the_stored_support_and_checks_itself(baseline) -> None:
    census = run_uk_size_census(baseline)
    assert census["checks"] == {
        "s0_above_cap_rows": 0,
        "start_matches_receipt": True,
        "passed": True,
    }
    assert set(census["floors"]) == {"0", "0.1", "0.5", "1"}
    # the floor-0 start is the refit's own: its per-nation capacity matches the
    # cap on the control refit's starting weights
    control = run_uk_size_control(baseline)["sized"]
    support = np.asarray(control.support)
    start = np.asarray(control.result.initial_weights)
    ratio = baseline.settings["max_weight_ratio"]
    dense = np.asarray(baseline.dense.weights)
    nation = baseline.profile.nation
    for name, value in census["floors"]["0"]["capacity_to_dense"]["by_nation"].items():
        expected = ratio * start[nation[support] == name].sum()
        assert value == pytest.approx(expected / dense[nation == name].sum(), rel=1e-12)
    # at floor 1 the start is the kept design rescaled to the pool mass
    design = np.asarray(baseline.problem.problem.initial_weights.values)
    kept = design[support] * design.sum() / design[support].sum()
    one = census["floors"]["1"]["start_to_dense"]["by_household_type"]
    for name, value in one.items():
        types = baseline.profile.household_type
        expected = kept[types[support] == name].sum() / dense[types == name].sum()
        assert value == pytest.approx(expected, rel=1e-12)
    assert census["floors"]["1"]["floored_rows"] == control.receipt["boundary_draws"]
    k_min = census["search"]["k_min"]
    assert 1 <= k_min["total"] <= len(design)
    assert set(k_min["by_nation"]) == set(nation)
    assert set(census["rules"]) == set(UK_LOCAL_TARGET_WEIGHT_RULES)
    assert census["rules"][baseline.run_rule]["loss_s0_to_run_rule"] == 1.0
    uniform = census["penalty"]["uniform"]
    assert set(uniform["pressure"]) == {"0.001", "0.003", "0.01", "0.03", "0.1"}
    placed = uniform["shifted_grid"][0] * uniform["scale"]
    placed /= census["penalty"]["loss_s0_run_rule"]
    assert 0.05 / 10**0.25 <= placed <= 0.05 * 10**0.25
    assert set(census["areas"]) == {"constituency", "la"}
    assert census["areas"]["la"]["areas"] == len(
        set(baseline.profile.local_authority_code)
    )
    assert set(census["early_size_triggers"]) == {
        "k_min_exceeds_households",
        "nation_k_min_exceeds_households",
        "areas_support_ceiling",
        "any",
    }


def test_scorecard_blocks_and_acceptance(built, baseline) -> None:
    root, _ = built
    control = run_uk_size_control(baseline)["sized"]
    candidate = run_uk_size_experiment(
        baseline, UKSizeExperiment(name="e", refit_l2={"lambda": 0.03})
    )["results"]["refit"]
    inputs = load_uk_size_scoring_inputs(root / "run", root / "cache")
    ratio = inputs.max_weight_ratio
    scorecard = score_uk_size_experiments(
        inputs,
        uk_size_weights_of("S0", control, max_weight_ratio=ratio),
        [uk_size_weights_of("e", candidate, max_weight_ratio=ratio)],
    )
    assert list(scorecard["sets"]) == ["D", "S0", "e"]
    for block in scorecard["sets"].values():
        assert sum(block["nation_share"].values()) == pytest.approx(1.0)
        assert sum(block["household_size_share"].values()) == pytest.approx(1.0)
        assert set(block["areas"]) == {"constituency", "la"}
        assert block["lone_person_share"] is not None
    assert "below_relative_collapse" not in scorecard["sets"]["D"]["areas"]["la"]
    assert "below_relative_collapse" in scorecard["sets"]["S0"]["areas"]["la"]
    capacity = scorecard["sets"]["S0"]["nation_capacity_to_dense"]
    assert set(capacity) == set(scorecard["sets"]["D"]["nation_share"])
    assert all(value > 0 for value in capacity.values())
    assert {"household_type", "type_x_tenure", "type_x_income_decile"} <= set(
        scorecard["representativeness"]["e"]["constituency"]
    )
    assert set(scorecard["acceptance"]) == {"S0", "e"}
    assert set(scorecard["acceptance"]["e"]["criteria"]) == {
        "household_total",
        "nation_shares",
        "lone_person_share",
        "local_family_within_10",
        "national_past_25",
        "relative_collapse",
    }
    tenure = scorecard["representativeness"]["e"]["la"]["tenure"]
    assert 0.0 <= tenure["tvd_median"] <= 1.0
    # the dense set scored against itself has no representativeness distance
    dense = inputs.dense
    self_card = score_uk_size_experiments(inputs, dense, [dense])
    assert (
        self_card["representativeness"]["D"]["constituency"]["tenure"]["tvd_max"] == 0.0
    )


def test_acceptance_thresholds_are_the_pre_registered_ones() -> None:
    dense = {
        "households": 100.0,
        "nation_share": {"England": 0.9, "Northern Ireland": 0.1},
        "lone_person_share": 0.30,
        "national_past_25": [{}],
        "fit_by_grain_family": {"constituency/age": {"share_within_10pct": 1.0}},
        "areas": {"constituency": {"below_ess_floor": 0, "ess_min": 90.0}},
    }
    control = {
        **dense,
        "fit_by_grain_family": {"constituency/age": {"share_within_10pct": 0.90}},
    }
    good = {
        **dense,
        "households": 99.5,
        "nation_share": {"England": 0.904, "Northern Ireland": 0.096},
        "lone_person_share": 0.295,
        "national_past_25": [{}, {}, {}],
        "fit_by_grain_family": {"constituency/age": {"share_within_10pct": 0.895}},
        "areas": {
            "constituency": {
                "below_ess_floor": 3,
                "ess_min": 40.0,
                "below_relative_collapse": 0,
            }
        },
    }
    assert uk_size_acceptance(good, dense=dense, control=control)["all_pass"]
    bad = {**good, "nation_share": {"England": 0.92, "Northern Ireland": 0.08}}
    verdict = uk_size_acceptance(bad, dense=dense, control=control)
    assert not verdict["criteria"]["nation_shares"]["pass"]
    collapsed = {
        **good,
        "areas": {
            "constituency": {
                **good["areas"]["constituency"],
                "below_relative_collapse": 1,
            }
        },
    }
    assert not uk_size_acceptance(collapsed, dense=dense, control=control)["criteria"][
        "relative_collapse"
    ]["pass"]


def test_pool_profile_round_trips_and_binds_its_axis(tmp_path) -> None:
    pool = synthetic_uk_pool(8)
    profile = build_uk_pool_profile(pool, problem_sha256="abc")
    assert profile.household_size.tolist() == [1, 2, 3, 1, 2, 3, 1, 2]
    assert set(profile.nation) == {"England", "Wales", "Northern Ireland"}
    write_uk_pool_profile(profile, tmp_path)
    loaded = load_uk_pool_profile(
        tmp_path, problem_sha256="abc", household_ids=profile.household_ids
    )
    np.testing.assert_array_equal(loaded.gross_income, profile.gross_income)
    assert list(loaded.tenure) == list(profile.tenure)
    assert list(loaded.household_type) == list(profile.household_type)
    assert profile.binding["household_type_column"] == "ons_household_type"
    with pytest.raises(ValueError, match="another problem"):
        load_uk_pool_profile(
            tmp_path, problem_sha256="xyz", household_ids=profile.household_ids
        )
    with pytest.raises(ValueError, match="another household axis"):
        load_uk_pool_profile(
            tmp_path, problem_sha256="abc", household_ids=profile.household_ids[::-1]
        )


def test_disclosure_control_suppresses_small_unit_counts() -> None:
    value = {
        "rows_min": 4,
        "sources_median": 3.5,
        "areas": 4,
        "nested": [{"nonzero_households": 2}],
        "ess_min": 7.2,
        "ess_to_dense_median": 0.3,
        "floored_rows": 0,
        "boundary_draws": 3,
        "certainties_share_at_cap": 0.46,
        "relative_collapse": {"ess_min": {"constituency": 4.0, "la": 55.0}},
    }
    assert disclosure_controlled(value) == {
        "rows_min": "<10",
        "sources_median": "<10",
        "areas": 4,
        "nested": [{"nonzero_households": "<10"}],
        "ess_min": "<10",
        "ess_to_dense_median": 0.3,
        "floored_rows": 0,
        "boundary_draws": "<10",
        "certainties_share_at_cap": 0.46,
        "relative_collapse": {"ess_min": {"constituency": "<10", "la": 55.0}},
    }


def test_tool_records_a_refused_configuration_and_moves_on(
    built, tmp_path, monkeypatch
) -> None:
    root, _ = built
    tool = _load_tool()
    monkeypatch.setattr(tool, "_LOCK", tmp_path / "lock")
    common = ["--run-dir", str(root / "run"), "--cache-dir", str(root / "cache")]
    out = tmp_path / "out"
    exclusive = ["--out", str(out), "--confirm-exclusive"]
    assert tool.main(["control", *common, *exclusive]) == 0
    assert tool.main(["census", *common, *exclusive]) == 0
    real = tool.run_uk_size_experiment

    def refusing(baseline, experiment):
        if experiment.name == "bad":
            raise RuntimeError(
                "compact refit lost targets or positive household support."
            )
        return real(baseline, experiment)

    monkeypatch.setattr(tool, "run_uk_size_experiment", refusing)
    experiments = tmp_path / "experiments.json"
    experiments.write_text(json.dumps([{"name": "bad"}, {"name": "good", "epochs": 2}]))
    run = ["run", *common, *exclusive, "--experiments", str(experiments)]
    assert tool.main(run) == 0
    bad = json.loads((out / "bad" / "receipt.json").read_text())
    assert bad["status"] == "failed"
    assert bad["error"] == {
        "type": "RuntimeError",
        "message": "compact refit lost targets or positive household support.",
    }
    assert not (out / "bad" / "weights.npz").exists()
    good = json.loads((out / "good" / "receipt.json").read_text())
    assert good["status"] == "finished" and good["refit_epochs_override"] == 2
    # a re-run skips both receipts, the failed one included
    assert tool.main(run) == 0
    assert json.loads((out / "bad" / "receipt.json").read_text()) == bad
    assert tool.main(["score", *common, "--out", str(out)]) == 0
    scorecard = json.loads((out / "scorecard.json").read_text())
    assert set(scorecard["acceptance"]) == {"S0", "good"}
    # step 1b's planner on a scored out directory (no ladders here, so the
    # holdout stage plans C0 alone)
    for stage in ("ae", "holdout"):
        to = tmp_path / f"step1b_{stage}.json"
        assert (
            tool.main(
                ["plan-step1b", "--out", str(out), "--stage", stage, "--to", str(to)]
            )
            == 0
        )
        assert (out / f"step1b_{stage}_plan.json").is_file()
    assert json.loads((tmp_path / "step1b_ae.json").read_text()) == []
    assert [
        e["name"] for e in json.loads((tmp_path / "step1b_holdout.json").read_text())
    ] == ["H_C0"]
    published = tmp_path / "published"
    assert tool.main(["publish", "--out", str(out), "--to", str(published)]) == 0
    results = json.loads((published / "results.json").read_text())
    assert set(results) == {"scorecard", "receipts", "census", "step1b"}
    assert set(results["step1b"]) == {"ae", "holdout"}
    assert results["receipts"]["bad"]["status"] == "failed"
    assert "traceback" not in results["receipts"]["bad"]
    assert results["census"]["checks"]["passed"] is True


_CRITERIA = (
    "household_total",
    "nation_shares",
    "lone_person_share",
    "local_family_within_10",
    "national_past_25",
)


def _step1a(configs):
    """A scored step 1a: name -> (declaration, floor, passes 1-5, passed, collapse, loss)."""

    scorecard = {"acceptance": {}, "sets": {}}
    receipts = {}
    for name, (declared, floor, ok, passed, collapse, loss) in configs.items():
        criteria = {key: {"pass": ok} for key in _CRITERIA}
        criteria["relative_collapse"] = {"value": collapse, "pass": collapse == 0}
        scorecard["acceptance"][name] = {
            "criteria": criteria,
            "passes": [f"criterion_{index}" for index in range(passed)],
        }
        scorecard["sets"][name] = {"loss_grain_equal_yardstick": loss}
        receipts[name] = {
            "status": "finished",
            "experiment": {"name": name, "mode": "refit", **declared},
            "settings": {"baseline_pi_floor": floor},
        }
    return scorecard, receipts


def _l2(value, anchor):
    return {"refit_l2": {"lambda": value, "anchor": anchor}, "baseline_pi_floor": 0.5}


def test_step1b_reads_its_picks_off_step_1a_by_the_pre_registered_rules() -> None:
    configs = {
        "C2_f0.5": ({"baseline_pi_floor": 0.5}, 0.5, False, 2, 200, 0.05),
        # the 1e-9 point is the iterate-switch control, not a ladder point
        "C3_f0.5_l2_1e-9": (
            {"refit_l2": {"lambda": 1e-9}, "baseline_pi_floor": 0.5},
            0.5,
            True,
            5,
            10,
            0.01,
        ),
        "E_init_f0.5_1e-3": (_l2(1e-3, "initial"), 0.5, True, 5, 150, 0.03),
        "E_init_f0.5_3e-3": (_l2(3e-3, "initial"), 0.5, True, 5, 120, 0.03),
        "E_init_f0.5_1e-2": (_l2(1e-2, "initial"), 0.5, True, 5, 100, 0.04),
        "E_init_f0.5_3e-2": (_l2(3e-2, "initial"), 0.5, False, 4, 95, 0.05),
        "E_init_f0.5_1e-1": (_l2(1e-1, "initial"), 0.5, False, 4, 92, 0.06),
        "E_unif_f0.5_1e-3": (_l2(1e-3, "uniform"), 0.5, False, 3, 180, 0.03),
        "E_unif_f0.5_1e-2": (_l2(1e-2, "uniform"), 0.5, False, 3, 60, 0.04),
        "E_unif_f0.5_1e-1": (_l2(1e-1, "uniform"), 0.5, False, 3, 58, 0.07),
        # the floor-0 E points and the learning-rate control stay out of the ladders
        "E_init_f0_1e-2": (
            {"refit_l2": {"lambda": 1e-2, "anchor": "initial"}},
            0.0,
            True,
            5,
            1,
            0.01,
        ),
        "C5_lr0.1485": ({"learning_rate": 0.1485}, 0.0, True, 5, 1, 0.01),
        "A_gfe_f0.5": (
            {"refit_rule": "grain_family_equal", "baseline_pi_floor": 0.5},
            0.5,
            False,
            3,
            150,
            0.06,
        ),
        "A_ngfes_f0.5": (
            {
                "refit_rule": "nation_grain_family_equal_sqrt_count",
                "baseline_pi_floor": 0.5,
            },
            0.5,
            False,
            4,
            140,
            0.07,
        ),
        "A_ngfe_f0": (
            {"refit_rule": "nation_grain_family_equal"},
            0.0,
            False,
            4,
            160,
            0.065,
        ),
    }
    scorecard, receipts = _step1a(configs)
    receipts["broken"] = {"status": "failed", "experiment": {"name": "broken"}}
    census = {"rules": {"nation_grain_family_equal": {"loss_s0_to_run_rule": 2.0}}}
    plan = uk_size_step1b_plan(scorecard, receipts, census, stage="ae")
    picks = plan["picks"]
    # best E: the largest ladder λ meeting criteria 1-5; none on the uniform ladder
    assert picks["best_e"] == {"initial": "E_init_f0.5_1e-2", "uniform": None}
    # knee: the smallest ladder λ within 10 % of the ladder's best collapse count
    assert picks["knee"] == {"initial": 1e-2, "uniform": 1e-2}
    # best A: most criteria passed, the tie broken by the grain_equal yardstick
    assert picks["best_a"] == "A_ngfe_f0"
    assert [len(picks["ladders"][anchor]) for anchor in ("initial", "uniform")] == [
        5,
        3,
    ]
    experiments = plan["experiments"]
    assert len(experiments) == 6
    assert {e["refit_rule"] for e in experiments} == {"nation_grain_family_equal"}
    assert {e["baseline_pi_floor"] for e in experiments} == {0.5}
    for anchor in ("initial", "uniform"):
        values = sorted(
            e["refit_l2"]["lambda"]
            for e in experiments
            if e["refit_l2"]["anchor"] == anchor
        )
        expected = [1e-2 * 10**step * 2.0 for step in (-0.5, 0.0, 0.5)]
        assert values == pytest.approx(expected)
    # the holdout stage, once the A×E points are scored: the winner rule picks
    # the A×E point that meets criteria 1-5 over one with fewer collapses
    scored = {
        **configs,
        "AE_ngfe_init_f0.5_0.02": (
            {
                "refit_rule": "nation_grain_family_equal",
                **_l2(0.02, "initial"),
            },
            0.5,
            True,
            5,
            90,
            0.08,
        ),
        "AE_ngfe_unif_f0.5_0.02": (
            {
                "refit_rule": "nation_grain_family_equal",
                **_l2(0.02, "uniform"),
            },
            0.5,
            False,
            3,
            50,
            0.04,
        ),
    }
    scorecard, receipts = _step1a(scored)
    holdout = uk_size_step1b_plan(scorecard, receipts, census, stage="holdout")
    assert holdout["picks"]["best_ae"] == "AE_ngfe_init_f0.5_0.02"
    assert [e["name"] for e in holdout["experiments"]] == [
        "H_C0",
        "H_C2_f0.5",
        "H_E_init_f0.5_1e-2",
        "H_A_ngfe_f0",
        "H_AE_ngfe_init_f0.5_0.02",
    ]
    assert {e["mode"] for e in holdout["experiments"]} == {"refit_holdout"}
    by_name = {e["name"]: e for e in holdout["experiments"]}
    assert by_name["H_A_ngfe_f0"] == {
        "name": "H_A_ngfe_f0",
        "mode": "refit_holdout",
        "refit_rule": "nation_grain_family_equal",
        "notes": "step 1b holdout: best A",
    }
    assert by_name["H_E_init_f0.5_1e-2"]["refit_l2"] == {
        "lambda": 1e-2,
        "anchor": "initial",
    }
    with pytest.raises(ValueError, match="stage must be"):
        uk_size_step1b_plan(scorecard, receipts, census, stage="1c")


def _load_tool():
    path = _TEST_PATHS.repository / "tools/run_uk_size_experiment.py"
    spec = importlib.util.spec_from_file_location("run_uk_size_experiment", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["run_uk_size_experiment"] = module
    spec.loader.exec_module(module)
    return module


def test_tool_runs_control_experiments_score_and_publish(
    built, tmp_path, monkeypatch
) -> None:
    root, _ = built
    tool = _load_tool()
    monkeypatch.setattr(tool, "_LOCK", tmp_path / "lock")
    common = ["--run-dir", str(root / "run"), "--cache-dir", str(root / "cache")]
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="--confirm-exclusive"):
        tool.main(["control", *common, "--out", str(out)])
    experiments = tmp_path / "experiments.json"
    experiments.write_text(
        json.dumps(
            [
                {"name": "e", "refit_l2": {"lambda": 0.03}},
                {"name": "h", "mode": "refit_holdout"},
            ]
        )
    )
    with pytest.raises(SystemExit, match="run `control` first"):
        tool.main(
            [
                "run",
                *common,
                "--out",
                str(out),
                "--experiments",
                str(experiments),
                "--confirm-exclusive",
            ]
        )
    assert (
        tool.main(["control", *common, "--out", str(out), "--confirm-exclusive"]) == 0
    )
    assert (
        tool.main(
            [
                "run",
                *common,
                "--out",
                str(out),
                "--experiments",
                str(experiments),
                "--confirm-exclusive",
            ]
        )
        == 0
    )
    assert (out / "e" / "weights.npz").is_file() and (
        out / "h" / "receipt.json"
    ).is_file()
    assert not (out / "h" / "weights.npz").exists()
    assert tool.main(["score", *common, "--out", str(out)]) == 0
    scorecard = json.loads((out / "scorecard.json").read_text())
    assert set(scorecard["acceptance"]) == {"S0", "e"}
    published = tmp_path / "published"
    assert tool.main(["publish", "--out", str(out), "--to", str(published)]) == 0
    results = json.loads((published / "results.json").read_text())
    assert set(results) == {"scorecard", "receipts"}
    assert "household_ids" not in results["receipts"]["e"].get("size_receipt", {})
    with pytest.raises(SystemExit, match="inside the repository"):
        tool.main(["score", *common, "--out", str(_TEST_PATHS.repository / "tmp-out")])
