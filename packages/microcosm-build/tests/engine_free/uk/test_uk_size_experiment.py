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
from microcosm.build.uk_runtime.size_experiment import (
    UKSizeExperiment,
    build_uk_size_experiment_cache,
    load_uk_size_experiment_baseline,
    load_uk_size_scoring_inputs,
    parse_uk_size_experiments,
    run_uk_size_control,
    run_uk_size_experiment,
    score_uk_size_experiments,
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
    }
    assert disclosure_controlled(value) == {
        "rows_min": "<10",
        "sources_median": 3.5,
        "areas": 4,
        "nested": [{"nonzero_households": "<10"}],
    }


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
