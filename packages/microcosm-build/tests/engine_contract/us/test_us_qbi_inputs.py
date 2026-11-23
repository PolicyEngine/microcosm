"""Tests split from packages/microcosm-build/tests/test_us_qbi_inputs.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_qbi_inputs import *


def test_all_qbi_outputs_are_live_policyengine_us_input_leaves() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for column in US_QBI_OUTPUT_COLUMNS:
        variable = variables[column]
        assert variable.entity.key == "person"
        assert not variable.formulas
        assert getattr(variable, "adds", None) is None
        assert getattr(variable, "subtracts", None) is None


def test_asec_qualification_defaults_are_the_installed_engine_defaults() -> None:
    # The base ASEC channel keeps PolicyEngine-US's own defaults (estate income
    # became non-qualified by default after 2.2.1), so a default flip fails here.
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    assert {
        column: variables[column].default_value
        for column in US_QBI_ASEC_QUALIFICATION_DEFAULTS
    } == US_QBI_ASEC_QUALIFICATION_DEFAULTS
