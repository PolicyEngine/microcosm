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
