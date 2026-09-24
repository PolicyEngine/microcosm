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


def test_m_chip_state_set_matches_engine_separate_chip_child_limits() -> None:
    """Two-way engine consistency for the #321 state set.

    PolicyEngine-US marks a state that runs no separate CHIP for children
    with a -inf ``gov.hhs.chip.child.income_limit``; its children are covered
    through Medicaid (M-CHIP), so ``chip_enrolled`` there can come only from
    the pregnant-person paths. That -inf set must equal ``_M_CHIP_STATE_FIPS``
    plus one reviewed carve-out: Rhode Island ("44") has a -inf child limit
    but sits outside the #321 set, so its CHIP row stays on the surface (the
    pinned feed's 2024-12 RI CHIP count is 0, the month RI did not report
    Medicaid enrollment either per the #386 substitution register; 2025-12
    is 33,661).
    An engine bump that adds or removes a -inf state fails here, so the set
    is re-reviewed instead of silently drifting from the engine.
    """
    from policyengine_us import CountryTaxBenefitSystem

    limits = (
        CountryTaxBenefitSystem()
        .parameters("2024-01-01")
        .gov.hhs.chip.child.income_limit
    )
    no_separate_child_chip = {
        state_fips
        for state_fips, postal in US_STATE_FIPS_TO_POSTAL.items()
        if limits[postal.upper()] == -math.inf
    }
    reviewed_carve_outs = {"44"}
    assert no_separate_child_chip == _M_CHIP_STATE_FIPS | reviewed_carve_outs
    assert not reviewed_carve_outs & _M_CHIP_STATE_FIPS
