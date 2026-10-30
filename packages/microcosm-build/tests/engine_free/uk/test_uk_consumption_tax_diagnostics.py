"""The consumption-tax diagnose tool's engine-free helpers (microcosm#1113).

The tool keeps the rows the measure-exclusion register holds out of the
objective measured; these tests pin its aggregation, its comparison against
the run's declared values, and that every row it reports is a committed
reference name, so a register entry that names the tool measures a real row.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from importlib.resources import files
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.lcfs_consumption import (
    CONSUMPTION_VARIABLE_RENAMES,
    UK_LCFS_COICOP_DIVISION_COLUMNS,
)
from test_support.paths import paths_for

_TOOL = (
    paths_for("microcosm-build").repository / "tools/diagnose_uk_consumption_taxes.py"
)
_SPEC = importlib.util.spec_from_file_location("diagnose_uk_consumption_taxes", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
_tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_tool)


def _household() -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "household_weight": [1.0, 2.0, 1.0],
            "petrol_spending": [1_000.0, 0.0, 0.0],
            "diesel_spending": [0.0, 0.0, 500.0],
            "electricity_consumption": [900.0, 800.0, 700.0],
            "gas_consumption": [600.0, 0.0, 500.0],
            "has_fuel_consumption": [True, True, False],
        }
    )
    for column in UK_LCFS_COICOP_DIVISION_COLUMNS:
        table[column] = 100.0
    return table


def test_division_columns_are_the_twelve_lcfs_division_totals() -> None:
    assert len(UK_LCFS_COICOP_DIVISION_COLUMNS) == 12
    assert UK_LCFS_COICOP_DIVISION_COLUMNS[0] == CONSUMPTION_VARIABLE_RENAMES["p601"]
    assert UK_LCFS_COICOP_DIVISION_COLUMNS[-1] == CONSUMPTION_VARIABLE_RENAMES["p612"]
    assert "petrol_spending" not in UK_LCFS_COICOP_DIVISION_COLUMNS


def test_composition_weights_the_household_table() -> None:
    summary = _tool.composition(_household())
    assert summary["households"] == 4.0
    # Households one and three are on gas: weight 2 of 4.
    assert summary["gas_connected_share"] == pytest.approx(0.5)
    assert summary["petrol_positive_share"] == pytest.approx(0.25)
    # Two fuel-car households carry weight 3; the weight-2 one buys no fuel.
    assert summary["fuel_car_household_share"] == pytest.approx(0.75)
    assert summary["fuel_car_zero_fuel_share"] == pytest.approx(2.0 / 3.0)
    assert summary["consumption_stored"] == pytest.approx(4.0 * 12 * 100.0)
    assert summary["electricity_consumption_stored"] == pytest.approx(
        900.0 + 1_600.0 + 700.0
    )


def test_composition_refuses_zero_weight() -> None:
    table = _household()
    table["household_weight"] = 0.0
    with pytest.raises(ValueError, match="sum to zero"):
        _tool.composition(table)


def test_compare_reports_relative_error_and_missing_targets() -> None:
    rows = _tool.compare(
        {"obr.vat": 220.0, "ons.household_gas_expenditure": 18.0},
        {"obr.vat": 176.0},
    )
    assert rows["obr.vat"]["relative_error"] == pytest.approx(0.25)
    assert rows["ons.household_gas_expenditure"] == {
        "estimate": 18.0,
        "target": None,
        "relative_error": None,
    }


def test_registry_values_reads_only_the_tool_rows(tmp_path) -> None:
    path = tmp_path / "national_contract_registry.json"
    path.write_text(
        json.dumps(
            {
                "specs": [
                    {"name": "obr.vat", "value": 177.9e9},
                    {"name": "obr.income_tax", "value": 1.0},
                    {"name": "ons.household_gas_expenditure", "value": None},
                ]
            }
        )
    )
    assert _tool.registry_values(path) == {"obr.vat": 177.9e9}


def test_tool_rows_are_committed_reference_names() -> None:
    payload = json.loads(
        files("microcosm.build.uk").joinpath("target_references.json").read_text()
    )
    committed = {reference["name"] for reference in payload["target_references"]}
    assert set(_tool.ROWS) <= committed, sorted(set(_tool.ROWS) - committed)
    for name, variables in _tool.ROWS.items():
        assert variables and all(isinstance(v, str) for v in variables), name


def test_excise_rows_are_measured_at_their_own_period() -> None:
    """The OBR excise rows read the FY2024-25 quantities as stored (#1359)."""

    payload = json.loads(
        files("microcosm.build.uk").joinpath("uk_population_targets.json").read_text()
    )
    periods = {
        target["target_id"]: target["bindings"]["policyengine"].get(
            "measurement_period"
        )
        for target in payload["targets"]
    }
    assert set(_tool.ROW_YEARS) <= set(_tool.ROWS)
    for name, year in _tool.ROW_YEARS.items():
        assert periods[name] == year, name
    # Every measured row whose binding declares its own period is measured
    # there: electricity and road fuel at calendar 2024 on the frame's own
    # columns, not through the engine's projection to --year (the re-review
    # of microcosm#1420, A).
    assert _tool.ROW_YEARS == {
        name: int(period)
        for name, period in periods.items()
        if name in _tool.ROWS and period is not None
    }
    assert _tool.ROW_YEARS["ons.household_electricity_expenditure"] == 2024
    assert _tool.ROW_YEARS["ons.household_road_fuel_expenditure"] == 2024


def test_rows_at_an_earlier_period_are_weighted_at_the_calibration_year(
    monkeypatch,
) -> None:
    # The re-reviews of microcosm#1420, 5 and 4: rows are weighted with the
    # solved weights, read by household id, never with the engine's weights
    # at the row's own period, and each duty is summed to households first.
    calls = []

    class _Simulation:
        def __init__(self, dataset):
            pass

        def calculate(self, variable, period, map_to=None):
            calls.append((variable, period, map_to))
            if variable == "household_id":
                values = [12, 11]
            else:
                values = [10.0, 20.0] if period == 2024 else [100.0, 200.0]
            return SimpleNamespace(values=np.asarray(values))

    monkeypatch.setitem(
        sys.modules, "policyengine_uk", SimpleNamespace(Microsimulation=_Simulation)
    )
    solved = pd.Series([2.0, 3.0], index=[11, 12])
    estimates = _tool.engine_estimates(
        "dataset.h5",
        2025,
        {"ons.household_electricity_expenditure": 2024},
        weights=solved,
    )
    # The engine's households come in the order 12, 11.
    assert estimates["ons.household_electricity_expenditure"] == 3 * 10 + 2 * 20
    # Two variables at --year.
    assert estimates["obr.vat"] == 2 * (3 * 100 + 2 * 200)
    assert not any(variable == "household_weight" for variable, _, _ in calls)
    assert {map_to for variable, _, map_to in calls if variable != "household_id"} == {
        "household"
    }
    with pytest.raises(ValueError, match="households and the frame's"):
        _tool.engine_estimates(
            "dataset.h5", 2025, {}, weights=pd.Series([1.0], index=[11])
        )


def test_main_reads_the_registry_and_the_release_year(tmp_path, monkeypatch) -> None:
    # The fourth re-review of microcosm#1420, 4: --registry supplies both the
    # declared values and the periods the run measured, and --year defaults to
    # the FRS release's calibration year.
    from microcosm.build.uk_runtime.frs_release import load_uk_frs_release

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "specs": [
                    {
                        "name": "ons.household_road_fuel_expenditure",
                        "value": 4.0,
                        "metadata": {"measurement_period": "2024"},
                    }
                ]
            }
        )
    )
    seen = {}

    def estimates(dataset, year, row_years):
        seen.update(dataset=dataset, year=year, row_years=dict(row_years))
        return {"ons.household_road_fuel_expenditure": 5.0}

    monkeypatch.setattr(_tool, "engine_estimates", estimates)
    monkeypatch.setattr(_tool, "composition", lambda household: {})
    monkeypatch.setattr(_tool.pd, "read_hdf", lambda path, key: pd.DataFrame())
    out = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["tool", "--dataset", "x.h5", "--registry", str(registry), "--out", str(out)],
    )
    _tool.main()
    assert seen == {
        "dataset": "x.h5",
        "year": load_uk_frs_release().calibration_year,
        "row_years": {"ons.household_road_fuel_expenditure": 2024},
    }
    report = json.loads(out.read_text())
    assert report["rows"]["ons.household_road_fuel_expenditure"][
        "relative_error"
    ] == pytest.approx(0.25)


def test_a_registry_supplies_the_periods_its_run_measured(tmp_path) -> None:
    # The re-review of microcosm#1420, 5: a run built before electricity moved
    # to 2024 holds the 2025 fact, so its own registry's periods win.
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "specs": [
                    {
                        "name": "ons.household_road_fuel_expenditure",
                        "value": 1.0,
                        "metadata": {"measurement_period": "2024"},
                    },
                    {"name": "ons.household_electricity_expenditure", "value": 2.0},
                    {
                        "name": "not.a.tool.row",
                        "metadata": {"measurement_period": "2024"},
                    },
                ]
            }
        )
    )
    assert _tool.registry_row_years(path) == {
        "ons.household_road_fuel_expenditure": 2024
    }


def test_the_tool_reads_the_solved_weights_of_a_release_and_a_solved_file(
    tmp_path, monkeypatch
) -> None:
    # The fifth re-review of microcosm#1420, 2: the default weight path, from
    # the H5 through load_uk_national_frame and uk_solve_year_frame to the
    # engine's households by id, on a file divided for release and on one
    # written as solved.
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        write_uk_national_frame,
    )
    from microcosm.build.uk_runtime.weight_year import uk_dataset_year_frame
    from microcosm.frame import WeightKind

    ids = np.asarray([11, 12, 13], dtype="int64")
    solved = np.asarray([2.0, 3.0, 5.0])

    def frame():
        return uk_national_frame(
            person=pd.DataFrame(
                {"person_id": ids, "person_benunit_id": ids, "person_household_id": ids}
            ),
            benunit=pd.DataFrame({"benunit_id": ids}),
            household=pd.DataFrame({"household_id": ids, "household_weight": solved}),
            time_period="2024",
            weight_kind=WeightKind.CALIBRATED,
        )

    as_solved = tmp_path / "solved.h5"
    write_uk_national_frame(frame(), as_solved)
    released = tmp_path / "released.h5"
    write_uk_national_frame(uk_dataset_year_frame(frame(), solve_year=2025), released)
    for path in (as_solved, released):
        weights = _tool.solved_household_weights(str(path), 2025)
        np.testing.assert_allclose(weights.loc[ids].to_numpy(), solved)

    households = [13, 11, 12]

    class _Simulation:
        def __init__(self, dataset):
            pass

        def calculate(self, variable, period, map_to=None):
            if variable == "household_id":
                return SimpleNamespace(values=np.asarray(households))
            return SimpleNamespace(values=np.asarray([1.0, 10.0, 100.0]))

    monkeypatch.setitem(
        sys.modules, "policyengine_uk", SimpleNamespace(Microsimulation=_Simulation)
    )
    estimates = _tool.engine_estimates(str(released), 2025, {})
    # Engine order 13, 11, 12 takes the weights 5, 2, 3.
    assert estimates["obr.tobacco_duties"] == pytest.approx(5 * 1 + 2 * 10 + 3 * 100)
    # A household on either side alone refuses.
    households = [11, 12]
    with pytest.raises(ValueError, match="households and the frame's"):
        _tool.engine_estimates(str(released), 2025, {})


def test_main_takes_the_year_from_the_runs_registry(tmp_path, monkeypatch) -> None:
    # The fifth re-review of microcosm#1420, 3: with --registry and no --year,
    # the run's own calibration period wins over the package's release.
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"specs": [{"name": "obr.vat", "value": 1.0, "period": 2026}]})
    )
    seen = {}

    def estimates(dataset, year, row_years):
        seen["year"] = year
        return {"obr.vat": 1.0}

    monkeypatch.setattr(_tool, "engine_estimates", estimates)
    monkeypatch.setattr(_tool, "composition", lambda household: {})
    monkeypatch.setattr(_tool.pd, "read_hdf", lambda path, key: pd.DataFrame())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tool",
            "--dataset",
            "x.h5",
            "--registry",
            str(registry),
            "--out",
            str(tmp_path / "out.json"),
        ],
    )
    _tool.main()
    assert seen == {"year": 2026}
