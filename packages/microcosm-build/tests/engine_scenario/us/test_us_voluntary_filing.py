"""Tests split from packages/microcosm-build/tests/test_us_voluntary_filing.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_voluntary_filing import *


def test_policyengine_2_2_1_contract_and_aca_ptc_neutralization() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OUTPUT]
    assert variable.is_input_variable()
    assert variable.entity.key == "tax_unit"
    assert variable.value_type is bool
    assert variable.default_value is False

    situation = {
        "people": {
            "adult": {
                "age": {2024: 45},
                # Isolate the filing gate from unrelated ACA eligibility facts.
                "is_aca_ptc_eligible": {2024: True},
            }
        },
        "tax_units": {
            "unit": {
                "members": ["adult"],
                "filing_status": {2024: "SINGLE"},
                _OUTPUT: {2024: True},
                "tax_unit_is_required_to_file": {2024: False},
                "eligible_for_refundable_credits": {2024: False},
                "would_file_if_eligible_for_refundable_credit": {2024: False},
                "slcsp": {"2024-01": 11_600},
                "aca_magi": {2024: 0},
                "aca_required_contribution_percentage": {2024: 0},
            }
        },
        "families": {"family": {"members": ["adult"]}},
        "spm_units": {"spm": {"members": ["adult"]}},
        "households": {
            "household": {
                "members": ["adult"],
                "state_code_str": {2024: "CA"},
            }
        },
        "marital_units": {"marital": {"members": ["adult"]}},
    }
    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "voluntary_filing_aca_ptc_neutralization"
    )
    assert probe.neutralized_variable == _OUTPUT
    assert probe.binding_inputs == (_OUTPUT,)
    assert probe.budget_measure == "aca_ptc"
    assert probe.min_abs_effect == 100_000_000.0
    baseline = Simulation(situation=situation)
    neutralized = Simulation(situation=situation, reform=_build_reform(probe))
    assert baseline.calculate("tax_unit_is_filer", 2024)[0]
    assert not neutralized.calculate("tax_unit_is_filer", 2024)[0]
    assert baseline.calculate("aca_ptc", 2024)[0] == pytest.approx(11_600.0)
    assert neutralized.calculate("aca_ptc", 2024)[0] == 0.0
