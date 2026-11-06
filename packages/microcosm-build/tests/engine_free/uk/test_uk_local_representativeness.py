"""Per-area representativeness of a sized UK local run (microcosm#1124 item G)."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from microcosm.build.uk_runtime.representativeness import (
    build_uk_pool_profile,
    read_uk_local_run,
    uk_area_representativeness,
    uk_area_tvd,
    uk_representativeness_summary,
)
from test_support.microcosm_build.uk_local_representativeness import (
    SYNTHETIC_AREAS,
    synthetic_uk_local_pool,
    write_synthetic_sized_run,
)
from test_support.paths import paths_for

_TOOL_PATH = (
    paths_for("microcosm-build").repository
    / "tools"
    / "score_uk_local_representativeness.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "score_uk_local_representativeness", _TOOL_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pool():
    frame = synthetic_uk_local_pool()
    profile = build_uk_pool_profile(frame, problem_sha256="0" * 64)
    dense = np.asarray(frame.weights_for("household").values, dtype=np.float64)
    return frame, profile, dense


def test_s_equal_to_d_is_at_zero_distance_everywhere() -> None:
    _, profile, dense = _pool()
    table = uk_area_representativeness(
        profile, dense_weights=dense, support=np.arange(len(profile)), weights=dense
    )
    assert table["valid"].all()
    np.testing.assert_allclose(table["tvd"], 0.0, atol=1e-12)
    assert set(table["grain"]) == {"constituency", "local_authority"}
    assert table["variable"].nunique() == 11


def test_a_tenure_shift_in_one_area_shows_there_only() -> None:
    _, profile, dense = _pool()
    weights = dense.copy()
    first = profile.constituency_code == SYNTHETIC_AREAS[0][0]
    weights[first & (profile.tenure == "RENT_PRIVATELY")] *= 3.0
    table = uk_area_representativeness(
        profile, dense_weights=dense, support=np.arange(len(profile)), weights=weights
    )
    tenure = table[
        (table["variable"] == "tenure") & (table["grain"] == "constituency")
    ].set_index("area_code")
    assert tenure.loc[SYNTHETIC_AREAS[0][0], "tvd"] > 0.05
    for constituency, _ in SYNTHETIC_AREAS[1:]:
        assert tenure.loc[constituency, "tvd"] == pytest.approx(0.0, abs=1e-12)


def test_the_noise_null_shrinks_as_the_effective_sample_grows() -> None:
    category = np.array(["a", "b"] * 50, dtype=object)
    area = np.zeros(100, dtype=np.int64)
    weights = np.ones(100)
    support = np.arange(100)
    small = uk_area_tvd(category, area, 1, weights, support, weights, np.array([10.0]))
    large = uk_area_tvd(
        category, area, 1, weights, support, weights, np.array([1_000.0])
    )
    assert large["noise"][0] < small["noise"][0]
    assert small["tvd"][0] == 0.0 and small["ratio"][0] == 0.0
    # An area S leaves empty is not valid, so it carries no distance.
    empty = uk_area_tvd(
        category, area, 1, weights, support[:0], weights[:0], np.array([0.0])
    )
    assert not empty["valid"][0] and np.isnan(empty["tvd"][0])


def test_the_summary_scores_supported_areas_and_names_none() -> None:
    _, profile, dense = _pool()
    # S keeps every row of three areas and five of the fourth, which then
    # rests on fewer than ten distinct source households.
    fourth = profile.constituency_code == SYNTHETIC_AREAS[3][0]
    support = np.sort(
        np.concatenate([np.flatnonzero(~fourth), np.flatnonzero(fourth)[:5]])
    )
    table = uk_area_representativeness(
        profile, dense_weights=dense, support=support, weights=dense[support]
    )
    summary = uk_representativeness_summary(table, minimum_areas=3)
    cell = summary["grains"]["constituency"]["tenure"]
    assert (cell["areas"], cell["areas_scored"], cell["areas_below_source_floor"]) == (
        4,
        3,
        1,
    )
    assert cell["tvd_median"] is not None
    # Fewer scored areas than the default minimum: the quantiles are withheld.
    withheld = uk_representativeness_summary(table)["grains"]["constituency"]
    assert withheld["tenure"]["tvd_median"] is None
    assert withheld["tenure"]["areas_above_ratio_threshold"] is None
    text = json.dumps(summary)
    for constituency, authority in SYNTHETIC_AREAS:
        assert constituency not in text and authority not in text
    for grain in summary["grains"].values():
        for variable in grain.values():
            assert not any("max" in key or key.endswith("min") for key in variable)


def test_d_and_s_are_read_from_a_finished_run(tmp_path) -> None:
    frame, _, dense = _pool()
    support = np.arange(0, len(dense), 2)
    weights = dense[support] * 1.5
    write_synthetic_sized_run(
        tmp_path, frame, dense_weights=dense * 2.0, support=support, weights=weights
    )

    run = read_uk_local_run(tmp_path)

    assert run is not None
    np.testing.assert_array_equal(run.support, support)
    np.testing.assert_allclose(run.dense_weights, dense * 2.0)
    np.testing.assert_allclose(run.weights, weights)
    assert len(run.profile) == len(dense)
    assert set(run.artifacts) == {
        "uk.full.problem/problem",
        "uk.full.dense/solution",
        "uk.full.size_refit/solution",
        "uk.full.size_refit/size",
    }


def test_a_dense_only_run_has_nothing_to_compare(tmp_path) -> None:
    frame, _, dense = _pool()
    write_synthetic_sized_run(
        tmp_path, frame, dense_weights=dense, support=None, weights=None
    )
    assert read_uk_local_run(tmp_path) is None


def test_a_changed_artifact_is_refused(tmp_path) -> None:
    frame, _, dense = _pool()
    support = np.arange(10)
    write_synthetic_sized_run(
        tmp_path, frame, dense_weights=dense, support=support, weights=dense[support]
    )
    receipt = tmp_path / "uk.full.size_refit.size.json"
    receipt.write_bytes(receipt.read_bytes() + b" ")
    with pytest.raises(ValueError, match="evidence-index digest"):
        read_uk_local_run(tmp_path)


def test_the_tool_writes_a_private_table_and_a_summary(tmp_path) -> None:
    frame, _, dense = _pool()
    run_dir = tmp_path / "run"
    support = np.arange(0, len(dense), 2)
    write_synthetic_sized_run(
        run_dir, frame, dense_weights=dense, support=support, weights=dense[support]
    )
    out = tmp_path / "private"
    tool = _load_tool()

    assert tool.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0

    summary = json.loads((out / tool.SUMMARY_FILE).read_text())
    assert summary["status"] == "scored"
    assert set(summary["grains"]) == {"constituency", "local_authority"}
    assert (out / tool.TABLE_FILE).is_file()
    # The per-area table may not land inside the repository.
    with pytest.raises(SystemExit):
        tool.main(
            [
                "--run-dir",
                str(run_dir),
                "--out",
                str(paths_for("microcosm-build").repository / "scratch"),
            ]
        )


def test_the_tool_reports_a_dense_only_run_as_not_applicable(tmp_path) -> None:
    frame, _, dense = _pool()
    run_dir = tmp_path / "run"
    write_synthetic_sized_run(
        run_dir, frame, dense_weights=dense, support=None, weights=None
    )
    out = tmp_path / "private"
    tool = _load_tool()
    assert tool.main(["--run-dir", str(run_dir), "--out", str(out)]) == 0
    summary = json.loads((out / tool.SUMMARY_FILE).read_text())
    assert summary["status"] == "not_applicable"
    assert not (out / tool.TABLE_FILE).exists()
