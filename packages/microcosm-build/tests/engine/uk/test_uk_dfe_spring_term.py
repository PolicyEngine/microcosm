"""The DfE spring-term measures on hand-built households (policyengine-uk 2.147.2)."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.dfe_spring_term import (
    DFE_AGE_AT_31_DECEMBER,
    DFE_SPRING_EARLY_LEARNING_2,
    DFE_SPRING_TERM_ENGINE_INPUTS,
    DFE_SPRING_UNIVERSAL,
    DFE_SPRING_WORKING_PARENT,
    dfe_spring_term_status,
)

YEAR = 2025


def _age_on_6_october(date_of_birth: int) -> int:
    birth_year, month_day = divmod(date_of_birth, 10_000)
    return YEAR - birth_year - (1 if month_day > 1006 else 0)


def _child_status(
    date_of_birth: int,
    *,
    earnings: tuple[float, ...] = (40_000.0, 30_000.0),
    deferred: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    from policyengine_uk import Simulation

    people = {
        f"parent{index}": {"age": {YEAR: 35}, "employment_income": {YEAR: pay}}
        for index, pay in enumerate(earnings)
    }
    child = {
        "age": {YEAR: _age_on_6_october(date_of_birth)},
        "date_of_birth": {YEAR: date_of_birth},
    }
    if deferred:
        child["early_years_school_start_deferred"] = {YEAR: True}
    people["child"] = child
    simulation = Simulation(
        situation={
            "people": people,
            "benunits": {"family": {"members": list(people)}},
            "households": {
                "home": {"members": list(people), "country": {YEAR: "ENGLAND"}}
            },
        }
    )

    def calculate(name: str) -> np.ndarray:
        result = simulation.calculate(name, YEAR)
        values = (
            np.asarray(result.decode_to_str())
            if hasattr(result, "decode_to_str")
            else np.asarray(getattr(result, "values", result))
        )
        # One family and one household, so a group value is every member's.
        return np.repeat(values, len(people)) if len(values) == 1 else values

    status = dfe_spring_term_status(calculate, YEAR)
    child_row = list(people).index("child")
    engine = {
        name: float(np.asarray(simulation.calculate(name, YEAR))[child_row])
        for name in (
            "is_child_receiving_universal_childcare",
            "is_child_receiving_extended_childcare",
            "is_child_receiving_targeted_childcare",
            "early_years_term_share_before_school",
        )
    }
    return {name: float(values[child_row]) for name, values in status.items()}, engine


def test_the_engine_year_starts_its_spring_term_on_1_january() -> None:
    from policyengine_uk.utils.dates import term_starts

    assert term_starts(YEAR) == (20250401, 20250901, 20260101)


def test_the_measures_read_existing_person_or_group_variables() -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for name in DFE_SPRING_TERM_ENGINE_INPUTS:
        assert name in variables, name


@pytest.mark.parametrize(
    ("date_of_birth", "expected"),
    [
        # Born in 2021 before September: reception from September 2025, so
        # not funded in the spring term although the engine funds its summer.
        (20210615, {DFE_AGE_AT_31_DECEMBER: 4, DFE_SPRING_UNIVERSAL: 0}),
        # Born after 1 September 2021: nursery until September 2026.
        (20211015, {DFE_AGE_AT_31_DECEMBER: 4, DFE_SPRING_UNIVERSAL: 1}),
        # Three on 31 December 2025 but two on 6 October: funded from January
        # under the regulations, and the engine pays no universal hours.
        (20221115, {DFE_AGE_AT_31_DECEMBER: 3, DFE_SPRING_UNIVERSAL: 1}),
        # Five on 31 December: outside DfE's 3 and 4.
        (20200915, {DFE_AGE_AT_31_DECEMBER: 5, DFE_SPRING_UNIVERSAL: 0}),
    ],
)
def test_universal_hours_in_the_spring_term(date_of_birth, expected) -> None:
    status, engine = _child_status(date_of_birth)
    for name, value in expected.items():
        assert status[name] == value, name
    if date_of_birth == 20221115:
        assert engine["is_child_receiving_universal_childcare"] == 0
    if date_of_birth == 20210615:
        assert engine["is_child_receiving_universal_childcare"] == 1
        assert engine["early_years_term_share_before_school"] == pytest.approx(1 / 3)


@pytest.mark.parametrize(
    ("date_of_birth", "funded"),
    [
        # Nine months on 15 November 2025: funded from 1 January 2026.
        (20250215, 1),
        # Nine months on 15 January 2026: not yet funded in that term.
        (20250415, 0),
        # In reception from September 2025.
        (20210615, 0),
        (20221115, 1),
    ],
)
def test_working_parent_hours_in_the_spring_term(date_of_birth, funded) -> None:
    status, _ = _child_status(date_of_birth)
    assert status[DFE_SPRING_WORKING_PARENT] == funded


def test_early_learning_for_two_year_olds_follows_dfe_age() -> None:
    # A lone parent with no earnings: the family qualifies for the targeted
    # offer. Born November 2023, the child is two on 31 December 2025 and
    # funded in the spring term, while the engine (age one on 6 October)
    # pays it nothing.
    status, engine = _child_status(20231115, earnings=(0.0,))
    assert status[DFE_AGE_AT_31_DECEMBER] == 2
    assert status[DFE_SPRING_EARLY_LEARNING_2] == 1
    assert engine["is_child_receiving_targeted_childcare"] == 0
    # A working couple does not qualify; its child takes working-parent hours.
    status, _ = _child_status(20231115)
    assert status[DFE_SPRING_EARLY_LEARNING_2] == 0
    assert status[DFE_SPRING_WORKING_PARENT] == 1


def test_a_deferred_school_start_keeps_the_spring_term() -> None:
    status, engine = _child_status(20210615, deferred=True)
    assert engine["early_years_term_share_before_school"] == 1
    assert status[DFE_SPRING_UNIVERSAL] == 1
    assert status[DFE_SPRING_WORKING_PARENT] == 1


@pytest.mark.parametrize("date_of_birth", [20210915, 20211015, 20220315, 20220915])
def test_at_engine_age_three_or_four_universal_is_the_engines_flag(
    date_of_birth,
) -> None:
    # The measure departs from the engine only for children the engine still
    # gates out on their age at 6 October.
    status, engine = _child_status(date_of_birth)
    in_nursery = engine["early_years_term_share_before_school"] > 0.5
    assert status[DFE_SPRING_UNIVERSAL] == float(
        bool(engine["is_child_receiving_universal_childcare"]) and in_nursery
    )
