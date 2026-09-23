"""Tests split from packages/microcosm-build/tests/test_us_fiscal_targets.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_fiscal_targets import *


def test_obbba_no_tax_channels_are_absent_from_2024_law_deduction_lists() -> None:
    # The jct.obbba_title_vii parity fence and the inert-compile test above
    # rest on one engine premise: at the 2024 target period the
    # qualified-overtime and qualified-tips deductions do not exist in law
    # (tyba 12/31/24, in force TY2025-TY2028 only), so a neutralize-variable
    # income-tax delta is structurally zero. Pin that premise to the locked
    # policyengine-us so a version bump that moves the provision window
    # breaks this test instead of silently invalidating the fence.
    __import__("policyengine_us")
    from policyengine_us import CountryTaxBenefitSystem

    parameters = CountryTaxBenefitSystem().parameters
    for list_name in ("deductions_if_itemizing", "deductions_if_not_itemizing"):
        node = getattr(parameters.gov.irs.deductions, list_name)
        for deduction in ("overtime_income_deduction", "tip_income_deduction"):
            assert deduction not in node("2024-01-01"), (list_name, deduction)
            for instant in ("2025-01-01", "2026-01-01", "2028-01-01"):
                assert deduction in node(instant), (list_name, deduction, instant)
            assert deduction not in node("2029-01-01"), (list_name, deduction)
