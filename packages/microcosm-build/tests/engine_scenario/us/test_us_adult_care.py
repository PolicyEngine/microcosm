"""Tests split from packages/microcosm-build/tests/test_us_adult_care.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_adult_care import *


def test_policyengine_2_2_1_cdcc_adult_care_contract_and_live_binding() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variables = CountryTaxBenefitSystem().variables
    flag = variables[_FLAG]
    assert flag.is_input_variable()
    assert flag.entity.key == "person"
    assert flag.value_type is bool
    expense = variables[_EXPENSE]
    assert expense.is_input_variable()
    assert expense.entity.key == "person"
    assert expense.value_type is float

    situation = {
        "people": {
            "head": {
                "age": {"2024": 45},
                "employment_income": {"2024": 80_000},
            },
            "spouse": {
                "age": {"2024": 44},
                _FLAG: {"2024": True},
                _EXPENSE: {"2024": 3_000},
            },
        },
        "tax_units": {
            "tax_unit": {
                "members": ["head", "spouse"],
                "filing_status": {"2024": "JOINT"},
            }
        },
        "spm_units": {"spm_unit": {"members": ["head", "spouse"]}},
        "households": {
            "household": {
                "members": ["head", "spouse"],
                "state_code": {"2024": "TX"},
            }
        },
    }

    baseline = Simulation(situation=situation)
    # 21(b)(1)(C) + 21(d)(2): the incapacitated spouse is the qualifying
    # individual and is deemed the earnings floor, so the credit binds.
    assert baseline.calculate("count_cdcc_eligible", 2024)[0] == 1
    assert baseline.calculate("cdcc_relevant_expenses", 2024)[0] == pytest.approx(
        3_000.0
    )
    assert baseline.calculate("cdcc", 2024)[0] > 0.0

    class NeutralizeAdultCare(Reform):
        def apply(self) -> None:
            self.neutralize_variable(_EXPENSE)

    neutralized = Simulation(situation=situation, reform=NeutralizeAdultCare)
    assert neutralized.calculate("cdcc_relevant_expenses", 2024)[0] == 0.0
    assert neutralized.calculate("cdcc", 2024)[0] == 0.0
    assert (
        neutralized.calculate("income_tax", 2024)[0]
        > baseline.calculate("income_tax", 2024)[0]
    )
