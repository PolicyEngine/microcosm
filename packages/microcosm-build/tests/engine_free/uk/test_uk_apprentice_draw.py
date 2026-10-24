"""is_apprentice on the UK spine: employees aged 21 to 24 at DfE's rate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_person_draws import (
    APPRENTICE_OUTPUT,
    APPRENTICE_RATE_KEY,
    UKFRSPersonDrawsStageTransform,
    derive_is_apprentice,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract
from test_support.microcosm_build.uk_frs_stochastic import _Contract

_STATUSES = ("FT_EMPLOYED", "PT_EMPLOYED", "FT_SELF_EMPLOYED", "STUDENT")


def _people(n_per_cell: int = 2000) -> pd.DataFrame:
    """Every age from 18 to 30 crossed with four statuses."""

    ages = np.arange(18, 31)
    rows = [
        (age, status) for age in ages for status in _STATUSES for _ in range(n_per_cell)
    ]
    frame = pd.DataFrame(rows, columns=["age", "employment_status"])
    frame.insert(0, "person_id", np.arange(1, len(frame) + 1))
    return frame


def test_only_employees_aged_21_to_24_are_flagged() -> None:
    person = _people()

    flags = derive_is_apprentice(person, rate=0.06)

    age = person["age"].to_numpy()
    employee = person["employment_status"].isin(["FT_EMPLOYED", "PT_EMPLOYED"])
    eligible = (age >= 21) & (age <= 24) & employee.to_numpy()
    assert flags.dtype == bool
    assert not flags[~eligible].any()
    assert flags[eligible].any()


def test_flags_match_the_per_head_rate_over_ages_21_to_24() -> None:
    person = _people()
    rng = np.random.default_rng(0)
    weights = rng.uniform(100.0, 3000.0, len(person))
    rate = 0.06

    flags = derive_is_apprentice(person, rate=rate, person_weights=weights)

    age = person["age"].to_numpy()
    in_ages = (age >= 21) & (age <= 24)
    # Half the people aged 21 to 24 are employees, so the rate among them is
    # about 0.12, and the weighted flags are about 6% of the age group.
    share = weights[flags].sum() / weights[in_ages].sum()
    assert share == pytest.approx(rate, rel=0.05)


def test_draw_is_identity_keyed_and_uses_the_conditional_rate() -> None:
    person = pd.DataFrame(
        {
            "person_id": [11, 12, 13, 14],
            "age": [22, 22, 23, 40],
            "employment_status": [
                "FT_EMPLOYED",
                "STUDENT",
                "PT_EMPLOYED",
                "FT_EMPLOYED",
            ],
        }
    )
    weights = np.array([1.0, 1.0, 2.0, 5.0])
    rate = 0.2

    flags = derive_is_apprentice(person, rate=rate, person_weights=weights)

    # W(21 to 24) = 4 and W(eligible) = 3, so p = 0.2 * 4 / 3.
    draws = stable_identity_uniforms(
        person["person_id"].to_numpy(), seed=0, salt=APPRENTICE_OUTPUT
    )
    expected = np.array([True, False, True, False]) & (draws < 0.2 * 4 / 3)
    np.testing.assert_array_equal(flags, expected)
    reordered = derive_is_apprentice(
        person.iloc[::-1].reset_index(drop=True),
        rate=rate,
        person_weights=weights[::-1],
    )
    np.testing.assert_array_equal(reordered, expected[::-1])


def test_conditional_rate_is_capped_at_one() -> None:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "age": [21, 22, 23],
            "employment_status": ["FT_EMPLOYED", "UNEMPLOYED", "UNEMPLOYED"],
        }
    )

    flags = derive_is_apprentice(person, rate=0.9)

    np.testing.assert_array_equal(flags, [True, False, False])


def test_no_eligible_employee_flags_nobody() -> None:
    person = pd.DataFrame(
        {
            "person_id": [1, 2],
            "age": [22, 30],
            "employment_status": ["STUDENT", "FT_EMPLOYED"],
        }
    )

    assert not derive_is_apprentice(person, rate=0.06).any()


def test_refuses_a_frame_without_employment_status() -> None:
    person = pd.DataFrame({"person_id": [1], "age": [22]})

    with pytest.raises(ValueError, match="employment_status"):
        derive_is_apprentice(person, rate=0.06)


def test_refuses_a_rate_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        derive_is_apprentice(_people(1), rate=1.5)


def _two_household_frame(**extra_person_columns):
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 4001),
            "person_benunit_id": np.arange(1, 4001),
            "person_household_id": np.repeat([1, 2], 2000),
            "age": np.tile([22, 23], 2000),
            "employment_status": np.repeat(["FT_EMPLOYED", "STUDENT"], 2000),
            **extra_person_columns,
        }
    )
    return uk_national_frame(
        person=person,
        benunit=pd.DataFrame({"benunit_id": np.arange(1, 4001), "is_married": False}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "region": ["LONDON", "WALES"],
                "household_weight": [1.0, 3.0],
            }
        ),
        time_period="2024",
    )


def test_stage_flags_use_the_frame_household_weights() -> None:
    stored = UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)(
        _two_household_frame()
    ).table("person")

    # Household 1 holds the employees and weighs 1, household 2 the students
    # and weighs 3, so p = 0.06 * 4 / 1 = 0.24 among the employees.
    flags = stored[APPRENTICE_OUTPUT].to_numpy()
    assert not flags[2000:].any()
    assert flags[:2000].mean() == pytest.approx(0.24, abs=0.03)


def test_stage_refuses_a_stored_flag() -> None:
    frame = _two_household_frame(is_apprentice=False)

    with pytest.raises(ValueError, match="already stored"):
        UKFRSPersonDrawsStageTransform(contract=_Contract(), stage=None)(frame)


def test_contract_carries_the_dfe_rate_for_the_build_year() -> None:
    contract = load_uk_take_up_contract()

    entry = contract.entry(APPRENTICE_RATE_KEY)

    assert entry.raw["output"] == APPRENTICE_OUTPUT
    assert entry.raw["entity"] == "person"
    # DfE 2024/25: 252,960 participants aged 19 to 24 over 4,221,504 people.
    assert contract.rate(APPRENTICE_RATE_KEY, contract.build_year) == 0.05992
