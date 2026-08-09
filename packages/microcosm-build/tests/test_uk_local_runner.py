from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.uk_runtime.local_runner as local_runner
from microcosm.build.uk_runtime import (
    build_local_candidate,
    build_local_candidate_from_dataset,
    build_metric_tables_from_dataset,
    load_metric_tables,
    prepare_area_frame,
    prepare_household_frame,
    read_local_table,
    set_simulation_area_group,
    write_local_candidate_outputs,
)


class Result:
    def __init__(self, values):
        self.values = np.asarray(values)


class FakeSimulation:
    def __init__(self, _dataset):
        self.inputs = {}

    def calculate(self, variable, **_kwargs):
        assert variable == "household_id"
        return Result([101, 102])

    def set_input(self, variable, period, values):
        self.inputs[(variable, period)] = list(values)


class SingleHouseholdSimulation(FakeSimulation):
    def calculate(self, variable, **_kwargs):
        assert variable == "household_id"
        return Result([101])


def test_prepare_area_frame_sorts_and_validates_codes() -> None:
    areas = pd.DataFrame({"code": ["S001", "E001"], "country": ["Scotland", "England"]})

    prepared = prepare_area_frame(areas)

    assert prepared["code"].tolist() == ["E001", "S001"]
    assert prepared["country"].tolist() == ["England", "Scotland"]


def test_prepare_area_frame_rejects_duplicate_codes() -> None:
    areas = pd.DataFrame({"code": ["E001", "E001"]})

    with pytest.raises(ValueError, match="unique"):
        prepare_area_frame(areas)


def test_prepare_household_frame_sorts_weights_and_fills_lineage() -> None:
    households = pd.DataFrame(
        {"household_id": [102, 101], "household_weight": [2.0, 1.0]}
    )

    prepared = prepare_household_frame(households, source_year=2023)

    assert prepared["household_id"].tolist() == [101, 102]
    assert prepared["household_weight"].tolist() == [1.0, 2.0]
    assert prepared["source_household_id"].tolist() == [101, 102]
    assert prepared["source_year"].tolist() == [2023, 2023]
    assert prepared["clone_index"].tolist() == [0, 0]
    assert prepared["source_household_key"].tolist() == ["2023:101", "2023:102"]


def test_read_local_table_and_load_metric_tables(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    pd.DataFrame(
        {
            "household_id": [101, 102],
            "population": [1.0, 2.0],
        }
    ).to_csv(path, index=False)

    table = read_local_table(path)
    metrics = load_metric_tables({"England": path})

    assert table["population"].tolist() == [1.0, 2.0]
    assert metrics["England"].index.tolist() == [101, 102]
    assert metrics["England"]["population"].tolist() == [1.0, 2.0]


def test_set_simulation_area_group_sets_country_region() -> None:
    sim = FakeSimulation(None)

    set_simulation_area_group(sim, "Scotland", period=2023)

    assert sim.inputs[("region", 2023)] == ["SCOTLAND", "SCOTLAND"]


def test_build_metric_tables_from_dataset_sets_each_country(monkeypatch) -> None:
    calls = []

    def factory(dataset):
        sim = FakeSimulation(dataset)
        calls.append(sim)
        return sim

    def fake_compute(sim, area_type, *, period=None, household_ids=None):
        assert household_ids is None
        return pd.DataFrame(
            {"population": [1.0, 2.0]},
            index=pd.Index([101, 102]),
        )

    monkeypatch.setattr(local_runner, "compute_household_metrics", fake_compute)

    tables = build_metric_tables_from_dataset(
        dataset=type("Dataset", (), {"time_period": 2023})(),
        area_groups={"E001": "England", "S001": "Scotland"},
        area_type="constituency",
        household_ids=[101, 102],
        simulation_factory=factory,
    )

    assert set(tables) == {"England", "Scotland"}
    assert tables["England"].index.tolist() == [101, 102]
    regions = [sim.inputs[("region", 2023)][0] for sim in calls]
    assert regions == ["SOUTH_EAST", "SCOTLAND"]


def test_build_metric_tables_from_dataset_reindexes_simulation_household_order(
    monkeypatch,
) -> None:
    class ReversedSimulation(FakeSimulation):
        def calculate(self, variable, **_kwargs):
            assert variable == "household_id"
            return Result([102, 101])

    def fake_compute(sim, area_type, *, period=None, household_ids=None):
        assert household_ids is None
        return pd.DataFrame(
            {"population": [20.0, 10.0]},
            index=pd.Index([102, 101]),
        )

    monkeypatch.setattr(local_runner, "compute_household_metrics", fake_compute)

    tables = build_metric_tables_from_dataset(
        dataset=type("Dataset", (), {"time_period": 2023})(),
        area_groups={"E001": "England"},
        area_type="constituency",
        household_ids=[101, 102],
        simulation_factory=ReversedSimulation,
    )

    assert tables["England"].index.tolist() == [101, 102]
    assert tables["England"]["population"].tolist() == [10.0, 20.0]


def test_build_local_candidate_solves_and_exports_long_weights() -> None:
    areas = pd.DataFrame({"code": ["S001", "E001"], "country": ["Scotland", "England"]})
    targets = pd.DataFrame(
        {
            "code": ["E001", "S001"],
            "population": [2.0, 2.0],
        }
    )
    metrics = {
        "England": pd.DataFrame({"population": [1.0, 0.0]}, index=[101, 102]),
        "Scotland": pd.DataFrame({"population": [0.0, 1.0]}, index=[101, 102]),
    }
    households = pd.DataFrame(
        {
            "household_id": [102, 101],
            "household_weight": [2.0, 1.0],
            "source_year": [2022, 2023],
            "source_household_id": ["b", "a"],
            "clone_index": [1, 0],
        }
    )

    result = build_local_candidate(
        area_type="constituency",
        area_frame=areas,
        targets=targets,
        metrics=metrics,
        household_frame=households,
        solver_options={"epochs": 60, "learning_rate": 0.2, "seed": 1},
    )

    assert result.problem.area_codes == ("E001", "S001")
    assert result.solve_result.final_loss < result.solve_result.initial_loss
    assert set(result.long_weights["area_code"]) == {"E001", "S001"}
    assert set(result.long_weights["source_household_key"]) == {
        "2023:a",
        "2022:b",
    }
    assert result.support_summary["area_code"].tolist() == ["E001", "S001"]
    assert "nonzero_source_households" in result.support_summary.columns
    assert "effective_sample_size" in result.support_summary.columns


def test_build_local_candidate_can_limit_pilot_areas() -> None:
    areas = pd.DataFrame(
        {
            "code": ["S001", "E001"],
            "country": ["Scotland", "England"],
        }
    )
    targets = pd.DataFrame(
        {
            "code": ["E001", "S001"],
            "population": [1.0, 1.0],
        }
    )
    metrics = {
        "England": pd.DataFrame({"population": [1.0]}, index=[101]),
        "Scotland": pd.DataFrame({"population": [1.0]}, index=[101]),
    }
    households = pd.DataFrame({"household_id": [101], "household_weight": [1.0]})

    result = build_local_candidate(
        area_type="constituency",
        area_frame=areas,
        targets=targets,
        metrics=metrics,
        household_frame=households,
        max_areas=1,
        solver_options={"epochs": 2},
    )

    assert result.problem.area_codes == ("E001",)
    assert result.long_weights["area_code"].unique().tolist() == ["E001"]


def test_build_local_candidate_from_dataset_computes_metrics(monkeypatch) -> None:
    areas = pd.DataFrame({"code": ["E001"], "country": ["England"]})
    targets = pd.DataFrame({"code": ["E001"], "population": [1.0]})
    households = pd.DataFrame({"household_id": [101], "household_weight": [1.0]})
    target_profile = {
        "targets": [
            {
                "geography_levels": ["constituency"],
                "bindings": {"policyengine": {"metric_name": "population"}},
            }
        ]
    }

    def fake_compute(
        sim,
        area_type,
        *,
        period=None,
        household_ids=None,
        target_profile=None,
    ):
        assert area_type == "constituency"
        assert period == 2023
        assert sim.inputs[("region", 2023)] == ["SOUTH_EAST"]
        assert household_ids is None
        assert target_profile is not None
        return pd.DataFrame({"population": [1.0]}, index=pd.Index([101]))

    monkeypatch.setattr(local_runner, "compute_household_metrics", fake_compute)

    result = build_local_candidate_from_dataset(
        dataset=type("Dataset", (), {"time_period": 2023})(),
        area_type="constituency",
        area_frame=areas,
        targets=targets,
        household_frame=households,
        simulation_factory=SingleHouseholdSimulation,
        target_profile=target_profile,
        solver_options={"epochs": 2},
    )

    assert result.problem.area_codes == ("E001",)
    assert result.support_summary["nonzero_households"].tolist() == [1]


def test_write_local_candidate_outputs(tmp_path: Path) -> None:
    areas = pd.DataFrame({"code": ["E001"], "country": ["England"]})
    targets = pd.DataFrame({"code": ["E001"], "population": [1.0]})
    metrics = pd.DataFrame({"population": [1.0]}, index=[101])
    households = pd.DataFrame({"household_id": [101], "household_weight": [1.0]})
    result = build_local_candidate(
        area_type="la",
        area_frame=areas,
        targets=targets,
        metrics=metrics,
        household_frame=households,
        solver_options={"epochs": 2},
    )

    summary = write_local_candidate_outputs(result, tmp_path)

    assert (tmp_path / "local_geography_weights.csv.gz").exists()
    assert (tmp_path / "solve_diagnostics.csv").exists()
    assert (tmp_path / "area_support_summary.csv").exists()
    saved = json.loads((tmp_path / "solve_summary.json").read_text())
    assert saved == summary
    assert summary["area_type"] == "la"
    assert summary["n_areas"] == 1
