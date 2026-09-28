"""Synthetic childcare contracts for the engine/us environment."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_childcare_attendance import *


def test_engine_accepts_child_inputs_and_derives_weekly_hours():
    from policyengine_us import Simulation

    result = _impute(_people(), weights=[0, 0, 1])
    year = 2026
    child = {"age": {year: 3}}
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        value = result.loc[0, column]
        child[column] = {year: int(value) if column == MONTH else float(value)}
    simulation = Simulation(
        situation={
            "people": {"parent": {"age": {year: 30}}, "child": child},
            "households": {"household": {"members": ["parent", "child"]}},
        }
    )
    assert simulation.calculate("childcare_hours_per_week", year).tolist() == [0, 40]
    assert simulation.calculate(MONTH, year).tolist() == [0, 22]
    # No mutation of the engine defaults by the dataset preparation function.
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        assert simulation.tax_benefit_system.variables[column].default_value == 0
