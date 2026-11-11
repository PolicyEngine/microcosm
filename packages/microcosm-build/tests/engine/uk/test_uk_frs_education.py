"""policyengine-uk reads frs_education's full-time education flag (microcosm#1095).

The stage's own derivations are tested engine-free in
tests/engine_free/uk/test_uk_frs_education.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.frs_education import derive_frs_education

YEAR = 2025
#: The stage's outputs that policyengine-uk 2.174.1 reads as person inputs on
#: the way to the Universal Credit education condition.
ENGINE_COLUMNS = (
    "current_education",
    "is_in_full_time_education",
    "is_in_non_advanced_education",
    "is_in_approved_training",
    "age_started_or_accepted_current_education_or_training",
    "is_before_universal_credit_qualifying_young_person_terminal_date",
)


def test_the_flag_applies_the_universal_credit_education_condition() -> None:
    """A full-time student on an advanced course claims only with an exception.

    On policyengine-uk 2.174.1, a childless full-time university student with
    no disability benefit receives education (UC Regs 2013 reg 12(2)(a)) and is
    not eligible for Universal Credit; one with PIP and limited capability for
    work is excepted (reg 14(1)(b)) and is eligible. A full-time
    further-education student aged 25 is on no advanced course, has no
    maintenance funding (the signed default) and is no qualifying young person,
    so receives no education and is eligible.
    """

    from policyengine_uk import Simulation

    names = ["university", "excepted", "further_education"]
    person = pd.DataFrame({"person_id": [1, 2, 3], "age": [20, 20, 25]})
    raw = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            # EDUCFT 1 = yes; TYPEED2 9 = university or higher education,
            # 7 = non-advanced further education.
            "educft": [1, 1, 1],
            "typeed2": [9, 9, 7],
            "educqual": [17, 17, 17],
            "trainee": [2, 2, 2],
            "_frs_table": ["adult", "adult", "adult"],
            "slos": [2, 2, 2],
            "educleft": [np.nan, np.nan, np.nan],
        }
    )
    derived = derive_frs_education(person, raw)
    assert derived["is_in_full_time_education"].tolist() == [True, True, True]
    assert derived["current_education"].tolist() == [
        "TERTIARY",
        "TERTIARY",
        "UPPER_SECONDARY",
    ]

    people = {
        name: {
            "age": {YEAR: int(person["age"].iloc[index])},
            **{
                column: {YEAR: _python(derived[column].iloc[index])}
                for column in ENGINE_COLUMNS
            },
        }
        for index, name in enumerate(names)
    }
    # Reg 14(1)(b): PIP, with the limited capability for work the engine
    # derives from is_disabled_for_benefits (frs_disability's outputs).
    people["excepted"].update(
        pip_dl_category={YEAR: "STANDARD"},
        is_disabled_for_benefits={YEAR: True},
    )
    sim = Simulation(
        situation={
            "people": people,
            "benunits": {
                f"b_{name}": {"members": [name], "uc_reported_capital": {YEAR: 0}}
                for name in names
            },
            "households": {f"h_{name}": {"members": [name]} for name in names},
        }
    )

    def calculate(variable: str) -> list[bool]:
        return np.asarray(sim.calculate(variable, YEAR), dtype=bool).tolist()

    assert calculate("uc_receiving_education") == [True, True, False]
    assert calculate("uc_is_excepted_from_education_condition") == [
        False,
        True,
        False,
    ]
    assert calculate("is_uc_eligible") == [False, True, True]


def _python(value: object) -> object:
    """A numpy scalar as the plain value a situation takes."""

    return value.item() if isinstance(value, np.generic) else value
