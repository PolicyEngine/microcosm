"""The SPM composition check against the engine that actually refuses.

``check_spm_composition`` is a *reimplementation* of a rule that lives in
``spm-calculator`` and ``policyengine-us``. A reimplementation can drift silently
in the one direction that matters: the check keeps passing while the engine has
started refusing, which is exactly the failure the check exists to prevent. The
unit tests beside the other preflight checks pin the check's own behaviour on
synthetic frames; these pin it against the installed engine.

Every test builds ONE population spec and derives both artifacts from it — a
microcosm ``Frame`` for the check and a policyengine-us situation for the engine
— so the two cannot be given different populations by accident.

Marked ``requires_us``: the root collection hook skips these when the
policyengine-us extra is absent, so they skip in the engine-free ``fast`` lane
and run in the US engine lane.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.release_gate_preflight import check_spm_composition
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

pytestmark = pytest.mark.requires_us

YEAR = 2024

#: The engine measures SPM nationally here: geography is a separate contract
#: (``SPM_GEOGRAPHY_REQUIRED``) and supplying a county would test that instead.
_NATIONAL = {"geography_kind": "national"}

#: Person columns a dataset may carry that change the classification.
_ROLE_COLUMNS = (
    "is_spm_independent_minor_role",
    "is_household_head",
    "is_household_spouse",
)


def _frame(people: list[dict]) -> Frame:
    """A US-schema frame from ``{name, spm, age, **role columns}`` specs.

    One household, one tax unit and one family hold everyone; ``spm`` names the
    SPM unit. Only the role columns a spec actually mentions are materialized,
    because the engine's fallback turns on column *presence*, not on its values.
    """
    rows = []
    for index, person in enumerate(people, start=1):
        row = {
            "person_id": index,
            "person_household_id": 1,
            "person_tax_unit_id": 1,
            "person_spm_unit_id": int(person["spm"]),
            "person_family_id": 1,
            "person_marital_unit_id": index,
            "age": float(person["age"]),
        }
        for column in _ROLE_COLUMNS:
            if any(column in candidate for candidate in people):
                row[column] = bool(person.get(column, False))
        rows.append(row)
    person_table = pd.DataFrame(rows)
    spm_ids = sorted({int(person["spm"]) for person in people})
    return Frame(
        {
            "person": person_table,
            "household": pd.DataFrame({"household_id": [1]}),
            "tax_unit": pd.DataFrame({"tax_unit_id": [1]}),
            "spm_unit": pd.DataFrame({"spm_unit_id": spm_ids}),
            "family": pd.DataFrame({"family_id": [1]}),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": person_table["person_marital_unit_id"].tolist()}
            ),
        },
        US_SCHEMA,
        {"household": Weights(np.array([100.0]), WeightKind.CALIBRATED)},
    )


def _situation(people: list[dict]) -> dict:
    """The same population as a policyengine-us situation."""
    members: dict[str, dict] = {}
    units: dict[str, dict] = {}
    for index, person in enumerate(people, start=1):
        name = f"person{index}"
        entry: dict = {"age": {YEAR: int(person["age"])}}
        for column in _ROLE_COLUMNS:
            if any(column in candidate for candidate in people):
                entry[column] = bool(person.get(column, False))
        members[name] = entry
        units.setdefault(f"unit{person['spm']}", {"members": []})["members"].append(
            name
        )
    return {
        "people": members,
        "households": {"household": {"members": list(members)}},
        "spm_units": units,
    }


def _engine_refuses(people: list[dict]) -> bool:
    """Does the installed engine refuse this population's SPM measurement?"""
    from policyengine_us import Simulation
    from spm_calculator.errors import SPMInputError

    simulation = Simulation(situation=_situation(people), spm=_NATIONAL)
    try:
        simulation.calculate("spm_unit_spm_threshold", YEAR)
    except SPMInputError as error:
        assert error.to_dict()["code"] == "SPM_COMPOSITION_REQUIRED"
        return True
    return False


def _engine_adults(people: list[dict]) -> list[int]:
    from policyengine_us import Simulation

    simulation = Simulation(situation=_situation(people), spm=_NATIONAL)
    return [
        int(value) for value in simulation.calculate("spm_measurement_adults", YEAR)
    ]


# The lane's population: one ordinary unit, and one whose only members are a
# 16-year-old and an 8-year-old.
_MINOR_ONLY = [
    {"spm": 1, "age": 40},
    {"spm": 2, "age": 16},
    {"spm": 2, "age": 8},
]


def test__minor_only_unit__engine_refuses_and_check_names_it() -> None:
    assert _engine_refuses(_MINOR_ONLY)

    result = check_spm_composition(_frame(_MINOR_ONLY))

    assert result.status == "FAIL"
    assert result.details["n_units_without_classified_adult"] == 1
    assert result.details["role_source"] == "unclassified"
    assert [row["spm_unit_id"] for row in result.rows] == [2]
    assert result.rows[0]["member_age_bands"] == ["15_to_17", "under_15"]
    # The remedy travels with the refusal; the engine's own message carries none.
    assert any("Remedy:" in failure for failure in result.failures)
    assert any("SPM_COMPOSITION_REQUIRED" in line for line in [result.summary])


def test__source_role_on_the_minor__engine_accepts_and_check_passes() -> None:
    people = [
        {"spm": 1, "age": 40, "is_spm_independent_minor_role": False},
        {"spm": 2, "age": 16, "is_spm_independent_minor_role": True},
        {"spm": 2, "age": 8, "is_spm_independent_minor_role": False},
    ]

    assert not _engine_refuses(people)

    result = check_spm_composition(_frame(people))

    assert result.status == "PASS"
    assert result.details["role_source"] == "source_column"


def test__household_head_on_the_minor__engine_accepts_and_check_passes() -> None:
    """The fallback formula's own path: no role column, head/spouse structure."""
    people = [
        {"spm": 1, "age": 40, "is_household_head": True},
        {"spm": 2, "age": 16, "is_household_head": True},
        {"spm": 2, "age": 8, "is_household_head": False},
    ]

    assert not _engine_refuses(people)

    result = check_spm_composition(_frame(people))

    assert result.status == "PASS"
    assert result.details["role_source"] == "household_structure_fallback"


def test__household_spouse_on_the_minor__engine_accepts_and_check_passes() -> None:
    people = [
        {"spm": 1, "age": 40, "is_household_spouse": False},
        {"spm": 2, "age": 16, "is_household_spouse": True},
        {"spm": 2, "age": 8, "is_household_spouse": False},
    ]

    assert not _engine_refuses(people)

    result = check_spm_composition(_frame(people))

    assert result.status == "PASS"
    assert result.details["role_source"] == "household_structure_fallback"


def test__source_role_false_overrides_a_true_head__both_refuse() -> None:
    """A supplied column wins over the fallback formula, in the engine and here.

    ``policyengine_us.spm.DATASET_SOURCE_INPUTS`` declares exactly this input as
    source-deliverable, so an observed False must not be overwritten by a
    household-structure guess. If that ever inverted, a check that read the
    fallback would pass while the engine refused.
    """
    people = [
        {
            "spm": 1,
            "age": 40,
            "is_household_head": True,
            "is_spm_independent_minor_role": True,
        },
        {
            "spm": 2,
            "age": 16,
            "is_household_head": True,
            "is_spm_independent_minor_role": False,
        },
        {
            "spm": 2,
            "age": 8,
            "is_household_head": False,
            "is_spm_independent_minor_role": False,
        },
    ]

    assert _engine_refuses(people)

    result = check_spm_composition(_frame(people))

    assert result.status == "FAIL"
    assert result.details["role_source"] == "source_column"
    assert [row["spm_unit_id"] for row in result.rows] == [2]


@pytest.mark.parametrize("age", [0, 13, 14, 15, 16, 17, 18, 19, 64])
@pytest.mark.parametrize("role", [False, True])
def test__classification_matches_the_engine_across_the_age_boundary(
    age: int, role: bool
) -> None:
    """The check's adult count equals ``spm_measurement_adults``, element-wise.

    This is the drift guard proper: it compares the reimplementation against the
    engine's own answer at every point where the rule bends — the 15 floor, the
    18 floor, and the role's effect between them — rather than asserting a
    remembered rule. The single-person unit also makes the check's verdict and
    the engine's refusal the same fact.
    """
    people = [{"spm": 1, "age": age, "is_spm_independent_minor_role": role}]

    engine_adults = _engine_adults(people)[0]
    result = check_spm_composition(_frame(people))

    assert engine_adults == (1 if age >= 18 or (age >= 15 and role) else 0)
    assert (result.status == "FAIL") is (engine_adults < 1)
    assert _engine_refuses(people) is (result.status == "FAIL")
