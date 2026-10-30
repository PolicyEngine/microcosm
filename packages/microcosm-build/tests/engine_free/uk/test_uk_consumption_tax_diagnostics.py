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
    # The re-review of microcosm#1420, 5: a release H5 stores the solved
    # weights divided by the engine's projection, so the engine's own 2024
    # weights would read a 2024 row 0.3% low. Rows are weighted at --year.
    calls = []

    class _Simulation:
        def __init__(self, dataset):
            pass

        def calculate(self, variable, period, map_to=None):
            calls.append((variable, period))
            if variable == "household_weight":
                values = [2.0, 3.0] if period == 2025 else [1.0, 1.0]
            else:
                values = [10.0, 20.0] if period == 2024 else [100.0, 200.0]
            return SimpleNamespace(values=np.asarray(values))

    monkeypatch.setitem(
        sys.modules, "policyengine_uk", SimpleNamespace(Microsimulation=_Simulation)
    )
    estimates = _tool.engine_estimates(
        "dataset.h5", 2025, {"ons.household_electricity_expenditure": 2024}
    )
    assert estimates["ons.household_electricity_expenditure"] == 2 * 10 + 3 * 20
    # Two variables at --year.
    assert estimates["obr.vat"] == 2 * (2 * 100 + 3 * 200)
    assert ("household_weight", 2024) not in calls


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
