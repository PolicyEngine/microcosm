"""The DfE spring-term measures on hand-built households (policyengine-uk 2.162.12)."""

from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.dfe_spring_term import (
    DFE_AGE_AT_31_DECEMBER,
    DFE_SPRING_EARLY_LEARNING_2,
    DFE_SPRING_TERM_ENGINE_INPUTS,
    DFE_SPRING_UNIVERSAL,
    DFE_SPRING_WORKING_PARENT,
    UNIVERSAL_WEEKLY_HOURS,
    dfe_spring_term_status,
)

YEAR = 2025
WORKING_COUPLE = (40_000.0, 30_000.0)
#: A lone parent on Universal Credit earning enough for working-parent hours
#: and little enough for early learning for 2-year-olds.
LOW_PAID_LONE_PARENT = (12_000.0,)
_SPRING = (
    DFE_AGE_AT_31_DECEMBER,
    DFE_SPRING_WORKING_PARENT,
    DFE_SPRING_UNIVERSAL,
    DFE_SPRING_EARLY_LEARNING_2,
)


def _age_on_6_october(date_of_birth: int) -> int:
    birth_year, month_day = divmod(date_of_birth, 10_000)
    return YEAR - birth_year - (1 if month_day > 1006 else 0)


def _children(
    *dates_of_birth: int,
    earnings: tuple[float, ...] = WORKING_COUPLE,
    claims: dict[str, bool] | None = None,
    hours: float | None = None,
    country: str = "ENGLAND",
    deferred: bool = False,
    give_date_of_birth: bool = True,
) -> list[dict[str, float]]:
    """Each child's spring-term status, with the engine's own amounts."""

    from policyengine_uk import Simulation

    people = {
        f"parent{index}": {"age": {YEAR: 35}, "employment_income": {YEAR: pay}}
        for index, pay in enumerate(earnings)
    }
    for index, date_of_birth in enumerate(dates_of_birth):
        child = {"age": {YEAR: _age_on_6_october(date_of_birth)}}
        if give_date_of_birth:
            child["date_of_birth"] = {YEAR: date_of_birth}
        if hours is not None:
            child["max_free_entitlement_hours_used"] = {YEAR: hours}
        if deferred:
            child["early_years_school_start_deferred"] = {YEAR: True}
        people[f"child{index}"] = child
    family = {"members": list(people)}
    for name, value in (claims or {}).items():
        family[name] = {YEAR: value}
    simulation = Simulation(
        situation={
            "people": people,
            "benunits": {"family": family},
            "households": {
                "home": {"members": list(people), "country": {YEAR: country}}
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
    rows = [list(people).index(f"child{i}") for i in range(len(dates_of_birth))]
    engine = {
        name: np.asarray(simulation.calculate(name, YEAR), dtype=float)
        for name in (
            "universal_childcare_entitlement",
            "targeted_childcare_entitlement",
            "extended_childcare_entitlement_per_child",
            "early_years_term_share_before_school",
        )
    }
    return [
        {
            **{name: float(status[name][row]) for name in _SPRING},
            **{name: float(values[row]) for name, values in engine.items()},
        }
        for row in rows
    ]


def test_the_engine_year_starts_its_spring_term_on_1_january() -> None:
    from policyengine_uk.utils.dates import term_starts

    assert term_starts(YEAR) == (20250401, 20250901, 20260101)


def test_the_measures_read_existing_variables() -> None:
    from policyengine_uk import CountryTaxBenefitSystem

    system = CountryTaxBenefitSystem()
    for name in DFE_SPRING_TERM_ENGINE_INPUTS:
        assert name in system.variables, name
    universal = system.parameters(f"{YEAR}-04-30").gov.dfe
    weekly = universal.universal_childcare_entitlement.hours / universal.weeks_per_year
    assert weekly == UNIVERSAL_WEEKLY_HOURS


def test_universal_hours_follow_dfe_age_and_school_start() -> None:
    # Born on 31 August 2021 the child starts reception in September 2025;
    # born on 1 September it stays in nursery until September 2026.
    in_reception, in_nursery = _children(20210831, 20210901)
    assert (
        in_reception[DFE_AGE_AT_31_DECEMBER] == in_nursery[DFE_AGE_AT_31_DECEMBER] == 4
    )
    assert in_reception[DFE_SPRING_UNIVERSAL] == 0
    assert in_reception["early_years_term_share_before_school"] == pytest.approx(1 / 3)
    assert in_nursery[DFE_SPRING_UNIVERSAL] == 1
    # Three on 31 December whether born on 6 or 7 October 2022. Since
    # policyengine-uk 2.156.1 the engine pays both from the term after the
    # third birthday, the spring term, where it used to pay the second nothing
    # (two on 6 October).
    sixth, seventh = _children(20221006, 20221007)
    assert sixth[DFE_SPRING_UNIVERSAL] == seventh[DFE_SPRING_UNIVERSAL] == 1
    assert sixth["universal_childcare_entitlement"] > 0
    assert seventh["universal_childcare_entitlement"] == pytest.approx(
        sixth["universal_childcare_entitlement"]
    )
    # 31 December 2022 is three on 31 December 2025; 1 January 2023 is two.
    older, younger = _children(20221231, 20230101)
    assert (older[DFE_AGE_AT_31_DECEMBER], younger[DFE_AGE_AT_31_DECEMBER]) == (3, 2)
    assert (older[DFE_SPRING_UNIVERSAL], younger[DFE_SPRING_UNIVERSAL]) == (1, 0)


def test_working_parent_hours_start_the_term_after_nine_months() -> None:
    # Nine months on 31 December 2025: funded from 1 January 2026. Nine
    # months on 1 January 2026: not yet funded in that term.
    funded, not_yet = _children(20250331, 20250401)
    assert funded[DFE_AGE_AT_31_DECEMBER] == not_yet[DFE_AGE_AT_31_DECEMBER] == 0
    assert (funded[DFE_SPRING_WORKING_PARENT], not_yet[DFE_SPRING_WORKING_PARENT]) == (
        1,
        0,
    )


def test_a_family_that_claims_only_working_parent_hours_is_counted() -> None:
    # The engine allocates the universal and targeted hours from the gross
    # working-parent hours to a family that does not claim them on their own.
    (universal_child,) = _children(
        20220515, claims={"would_claim_universal_childcare": False}, hours=12
    )
    assert universal_child["universal_childcare_entitlement"] > 0
    assert universal_child[DFE_SPRING_UNIVERSAL] == 1
    # Twelve hours is within the universal 15, so not a working-parent child.
    assert universal_child[DFE_SPRING_WORKING_PARENT] == 0
    (two_year_old,) = _children(
        20230515,
        earnings=LOW_PAID_LONE_PARENT,
        claims={"would_claim_targeted_childcare": False},
    )
    assert two_year_old["targeted_childcare_entitlement"] > 0
    assert two_year_old[DFE_SPRING_EARLY_LEARNING_2] == 1
    assert two_year_old[DFE_SPRING_WORKING_PARENT] == 0


def test_dfe_counts_a_two_year_old_under_one_entitlement() -> None:
    # Two on 31 December but one on 6 October: the engine gives working-parent
    # hours and no targeted hours, but DfE records the child under early
    # learning for 2-year-olds only.
    (child,) = _children(20231115, earnings=LOW_PAID_LONE_PARENT)
    assert child["extended_childcare_entitlement_per_child"] > 0
    assert child[DFE_SPRING_EARLY_LEARNING_2] == 1
    assert child[DFE_SPRING_WORKING_PARENT] == 0
    # Three on 31 December but two on 6 October: the engine nets the
    # targeted hours, but in the spring term the child takes universal and
    # working-parent hours.
    (child,) = _children(20221115, earnings=LOW_PAID_LONE_PARENT)
    assert child["extended_childcare_entitlement_per_child"] == 0
    assert child[DFE_SPRING_WORKING_PARENT] == 1
    assert child[DFE_SPRING_UNIVERSAL] == 1
    assert child[DFE_SPRING_EARLY_LEARNING_2] == 0


@pytest.mark.parametrize(
    "dates_of_birth",
    [(20231115,), (20230515,), (20221115, 20231115), (20220101, 20230701, 20240301)],
)
@pytest.mark.parametrize("earnings", [WORKING_COUPLE, LOW_PAID_LONE_PARENT])
def test_no_child_is_counted_under_two_dfe_entitlements_at_age_two(
    dates_of_birth, earnings
) -> None:
    for child in _children(*dates_of_birth, earnings=earnings):
        assert not (
            child[DFE_SPRING_WORKING_PARENT] and child[DFE_SPRING_EARLY_LEARNING_2]
        )


def test_a_deferred_school_start_keeps_the_spring_term() -> None:
    (child,) = _children(20210615, deferred=True)
    assert child["early_years_term_share_before_school"] == 1
    assert child[DFE_SPRING_UNIVERSAL] == 1
    assert child[DFE_SPRING_WORKING_PARENT] == 1


def test_outside_england_no_dfe_entitlement_is_counted() -> None:
    (child,) = _children(20211015, country="WALES")
    assert child[DFE_SPRING_UNIVERSAL] == 0
    assert child[DFE_SPRING_WORKING_PARENT] == 0
    assert child[DFE_SPRING_EARLY_LEARNING_2] == 0


def test_a_child_placed_by_age_alone_is_classified() -> None:
    # Without a date of birth the engine places a whole age of 3 at the
    # middle of the year (born in April 2022), so it is three on 31 December.
    (child,) = _children(20220101, give_date_of_birth=False)
    assert child[DFE_AGE_AT_31_DECEMBER] == 3
    assert child[DFE_SPRING_UNIVERSAL] == 1
    assert child[DFE_SPRING_WORKING_PARENT] == 1


def test_a_child_under_two_that_does_not_use_its_hours_gets_none() -> None:
    """The under-2 usage draw writes max_free_entitlement_hours_used 0 for a
    child that does not use the working-parent hours (frs_take_up)."""

    from policyengine_uk import CountryTaxBenefitSystem

    from microcosm.build.uk_runtime.frs_take_up import (
        UK_ENGINE_FREE_HOURS_DEFAULT,
        UK_UNDER_TWO_USAGE_OUTPUT,
    )

    variable = CountryTaxBenefitSystem().variables[UK_UNDER_TWO_USAGE_OUTPUT]
    assert variable.entity.key == "person"
    assert not variable.formulas
    assert variable.default_value == UK_ENGINE_FREE_HOURS_DEFAULT
    (uses,) = _children(20240515)
    (does_not,) = _children(20240515, hours=0.0)
    assert uses["extended_childcare_entitlement_per_child"] > 0
    assert uses[DFE_SPRING_WORKING_PARENT] == 1
    assert does_not["extended_childcare_entitlement_per_child"] == 0
    assert does_not[DFE_SPRING_WORKING_PARENT] == 0


def test_the_build_stages_dfe_age_is_the_engines() -> None:
    """The under-2 usage draw runs without the engine, so it restates the
    engine's date of birth; it must give the engine's Y - birth_year."""

    from policyengine_uk.utils.dates import (
        birth_instant_from_age,
        grid_months_to_yyyymmdd,
    )

    from microcosm.build.uk_runtime.dfe_spring_term import (
        dfe_age_at_31_december,
        engine_birth_year,
    )

    rng = np.random.default_rng(7)
    age = rng.integers(0, 6, 20_000).astype(float)
    months = rng.uniform(0, 12, age.size)
    # Around the 31 December boundary (a birthday on 7 October, about 9.19
    # months before 6 October).
    months[:2_000] = np.linspace(9.18, 9.21, 2_000)
    for year in (2024, 2025, 2026):
        engine = (
            grid_months_to_yyyymmdd(
                birth_instant_from_age(
                    year, age.astype(np.float32), months.astype(np.float32)
                )
            )
            // 10_000
        )
        np.testing.assert_array_equal(engine_birth_year(age, months, year), engine)
        np.testing.assert_array_equal(
            dfe_age_at_31_december(age, months, year),
            dfe_age_at_31_december(age, months, 2025),
        )
