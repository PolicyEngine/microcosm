"""The SPM measurement release gate against the engine that actually refuses.

``check_release_spm_measurement`` grades a release against what its certified
engine measures. These tests build one population, derive a release frame and
a policyengine-us situation from it, and require the gate's verdict to equal
the installed engine's own: the gate fails exactly when the engine's SPM
threshold raises ``SPM_COMPOSITION_REQUIRED``. They also pin that the
installed engine measures every unit, which is why the gate grades every unit
of a release certified against it.
"""

from __future__ import annotations

import pytest

from microcosm.build.us_runtime.spm_measurement_gate import (
    check_release_spm_measurement,
    engine_scopes_spm_universe,
)
from test_support.microcosm_build.us_spm_measurement_gate import (
    ASEC,
    Household,
    release_frame,
)

YEAR = 2024
_NATIONAL = {"geography_kind": "national"}

_POPULATIONS = {
    "measurable": [
        Household([(41, True), (39, True), (9, False)]),
        Household([(67, True)], kind=3),
        Household([(16, True)], kind=2),
        Household([(16, True)], spine=ASEC),
    ],
    "group_quarters_child": [
        Household([(41, True), (9, False)]),
        Household([(11, True)], kind=2),
    ],
    "group_quarters_teen_without_role": [
        Household([(41, True)]),
        Household([(16, False)], kind=3),
    ],
    "zero_adult_housing_unit": [
        Household([(41, True)]),
        Household([(16, False), (12, False)]),
    ],
}


def _situation(households: list[Household]) -> dict:
    people: dict[str, dict] = {}
    units: dict[str, dict] = {}
    for index, household in enumerate(households):
        for position, (age, role) in enumerate(household.members):
            name = f"person{index}_{position}"
            people[name] = {
                "age": {YEAR: int(age)},
                "is_spm_independent_minor_role": bool(role),
            }
            units.setdefault(f"unit{index}", {"members": []})["members"].append(name)
    return {
        "people": people,
        "households": {"household": {"members": list(people)}},
        "spm_units": units,
    }


def _engine_refuses(households: list[Household]) -> bool:
    from policyengine_us import Simulation
    from spm_calculator.errors import SPMInputError

    simulation = Simulation(situation=_situation(households), spm=_NATIONAL)
    try:
        simulation.calculate("spm_unit_spm_threshold", YEAR)
    except SPMInputError as error:
        assert error.to_dict()["code"] == "SPM_COMPOSITION_REQUIRED"
        return True
    return False


def test_the_installed_engine_measures_every_unit():
    from policyengine_us.system import system

    assert not engine_scopes_spm_universe(system.variables)


@pytest.mark.parametrize("population", sorted(_POPULATIONS))
def test_the_gate_fails_exactly_when_the_installed_engine_refuses(population):
    households = _POPULATIONS[population]
    from policyengine_us.system import system

    gate = check_release_spm_measurement(
        release_frame(households),
        engine_scopes_universe=engine_scopes_spm_universe(system.variables),
    )

    assert (not gate.passed) is _engine_refuses(households)
    assert gate.passed is (population == "measurable")
