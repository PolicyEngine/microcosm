"""Tests split from packages/microcosm-build/tests/test_us_disability_benefits.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_disability_benefits import *


def test_policyengine_us_2_2_1_contract_and_positive_annual_behavior() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                # SNAP 1.769.0+ applies this person's countable-income share
                # to unearned income; make the graph fixture work-eligible.
                "weekly_hours_worked_before_lsr": {"2024": 40},
                _OUTPUT: {"2024": 6_000.0},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    simulation = Simulation(situation=situation)

    assert simulation.calculate(_OUTPUT, 2024)[0] == pytest.approx(6_000.0)
    assert simulation.calculate(_OUTPUT, "2024-01")[0] == pytest.approx(500.0)
    assert simulation.calculate("snap_unearned_income", "2024-01")[0] == pytest.approx(
        500.0
    )


def test_shipped_snap_exclusion_probe_binds_with_positive_sign() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "disability_benefits_snap_exclusion"
    )
    reform = Reform.from_dict(dict(probe.parameter_changes), country_id="us")
    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 12_000.0},
                "weekly_hours_worked_before_lsr": {"2024": 40},
                _OUTPUT: {"2024": 6_000.0},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    baseline = Simulation(situation=situation)
    reformed = Simulation(
        tax_benefit_system=CountryTaxBenefitSystem(reform=(reform,)),
        situation=situation,
    )

    effect = reformed.calculate("snap", 2024)[0] - baseline.calculate("snap", 2024)[0]
    assert effect > 1_000.0
