"""Tests split from packages/microcosm-build/tests/test_us_weeks_unemployed.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_weeks_unemployed import *


def test_live_engine_graph_has_exactly_the_reviewed_consumers() -> None:
    import policyengine_us
    system = policyengine_us.CountryTaxBenefitSystem()

    consumers = {
        name
        for name, variable in system.variables.items()
        for formula in variable.formulas.values()
        if _OUTPUT in formula.__code__.co_consts
    }
    # policyengine-us 2.60 converts weeks into months of unemployment
    # compensation for the SNAP work registration exemption (7 CFR
    # 273.7(b)(1)(v)). That consumer is live: it reads the compensation the
    # release stores. Zero weeks with positive compensation counts as all 12
    # months, the engine's behavior before months were modeled.
    assert consumers == {
        "al_ui",
        "ny_ui",
        "ok_ui",
        "pa_uc",
        "unemployment_compensation_months",
    }
    months = system.variables["unemployment_compensation_months"]
    assert months.entity.key == "person"
    assert "total_unemployment_compensation" in (
        months.formulas[min(months.formulas)].__code__.co_consts
    )
    weeks = system.variables[_OUTPUT]
    assert weeks.default_value == 0
    assert not weeks.formulas

    # These UI outputs cannot support an honest monetary neutralization probe
    # until their independent claim-wage inputs are restored. Pin the complete
    # formula-less, default-zero input surface for every direct consumer.
    blocked_inputs = {
        "al_ui": {
            "al_ui_base_period_wages",
            "al_ui_high_quarter_wages",
            "al_ui_quarters_with_wages",
            "al_ui_second_high_quarter_wages",
            "al_ui_weekly_earnings",
        },
        "ny_ui": {
            "ny_ui_base_period_wages",
            "ny_ui_gross_weekly_earnings",
            "ny_ui_high_quarter_wages",
            "ny_ui_quarters_with_wages",
            "ny_ui_second_high_quarter_wages",
            "ny_ui_weekly_hours_worked",
        },
        "ok_ui": {
            "ok_ui_base_period_taxable_wages",
            "ok_ui_base_period_total_wages",
            "ok_ui_gross_weekly_earnings",
            "ok_ui_high_quarter_taxable_wages",
            "ok_ui_high_quarter_total_wages",
        },
        "pa_uc": {
            "pa_uc_base_year_wages",
            "pa_uc_credit_weeks",
            "pa_uc_gross_weekly_earnings",
            "pa_uc_highest_quarter_wages",
        },
    }
    for consumer, names in blocked_inputs.items():
        observed = {
            name
            for name, variable in system.variables.items()
            if name.startswith(f"{consumer}_") and not variable.formulas
        }
        assert observed == names
        for name in names:
            assert system.variables[name].default_value == 0
