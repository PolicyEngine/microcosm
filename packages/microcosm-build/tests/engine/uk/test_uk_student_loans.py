"""Tests split from packages/microcosm-build/tests/test_uk_student_loans.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.uk_student_loans import *


def test_values_match_policyengine_enum_when_available() -> None:
    module = pytest.importorskip(
        "policyengine_uk.variables.gov.hmrc.student_loans.student_loan_plan"
    )
    engine_names = set(module.StudentLoanPlan.__members__)

    assert set(STUDENT_LOAN_ENUM_DOMAIN) <= engine_names
    assert "PLAN_4" in engine_names
    assert "PLAN_4" not in STUDENT_LOAN_ENUM_DOMAIN


def test_the_engine_relabels_course_dated_plans_by_age_at_2025() -> None:
    """policyengine-uk projects the 2024 dataset to the 2025 calibration year and
    relabels every holder by a course started at 18 (uprate_student_loan_plans,
    step 1), while the stage dates a course from the age its borrower left
    full-time education (microcosm#868, #1299). Where the two dates fall on
    either side of a plan boundary, or the age rule writes the loan off, the
    engine's plan is the one the calibration measures at 2025. This pins which,
    so an engine that starts to keep the stage's dating fails it and the SLC
    rows are re-measured (microcosm#1095)."""

    from policyengine_uk.data import UKSingleYearDataset
    from policyengine_uk.data.economic_assumptions import uprate_student_loan_plans
    from policyengine_uk.system import system

    from microcosm.build.uk_runtime.student_loans import course_start_year

    year = 2025
    # (age, age left full-time education, the stage's plan)
    people = [
        (33, 25, "PLAN_2"),  # left at 25: a 2014 course, Plan 2
        (25, 21, "PLAN_2"),  # left at 21: a 2018 course, both rules agree
        (45, 21, "PLAN_1"),  # a 1998 course, both rules agree
        (50, 21, "PLAN_1"),  # a 1993 course: the age rule writes it off
    ]
    starts = course_start_year(
        year=year,
        age=np.asarray([age for age, _, _ in people]),
        age_left=np.asarray([left for _, left, _ in people]),
    )
    assert starts.tolist() == [2014, 2018, 1998, 1993]
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, len(people) + 1),
            "person_benunit_id": np.arange(1, len(people) + 1),
            "person_household_id": np.arange(1, len(people) + 1),
            "age": [age for age, _, _ in people],
            "student_loan_plan": [plan for _, _, plan in people],
            "student_loan_repayments": [500.0] * len(people),
            "highest_education": ["TERTIARY"] * len(people),
        }
    )
    benunit = pd.DataFrame({"benunit_id": person["person_benunit_id"]})
    household = pd.DataFrame(
        {
            "household_id": person["person_household_id"],
            "region": ["LONDON"] * len(people),
            "household_weight": [1.0] * len(people),
        }
    )
    base = UKSingleYearDataset(
        person=person.copy(),
        benunit=benunit.copy(),
        household=household.copy(),
        fiscal_year=2024,
    )
    projected = UKSingleYearDataset(
        person=person.copy(),
        benunit=benunit.copy(),
        household=household.copy(),
        fiscal_year=year,
    )

    out = uprate_student_loan_plans(projected, base, system.parameters, base_year=2024)

    # The 2014 course becomes a 2010 one, Plan 1, and the 1993 course is
    # written off; the two the rules agree on keep their plans.
    assert out.person["student_loan_plan"].astype(str).tolist() == [
        "PLAN_1",
        "PLAN_2",
        "PLAN_1",
        "NONE",
    ]
