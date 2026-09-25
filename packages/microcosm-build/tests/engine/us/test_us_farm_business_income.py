"""Tests split from packages/microcosm-build/tests/test_us_farm_business_income.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_farm_business_income import *


def test_policyengine_us_2_2_1_qbi_graph_reads_each_farm_leaf() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    system = CountryTaxBenefitSystem()
    for output in US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS:
        variable = system.variables[output]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert str(variable.definition_period).lower() == "year"
        assert variable.default_value == 0

    reform = Reform.from_dict(
        {
            "gov.irs.deductions.qbi.income_definition": {
                "2026-01-01.2026-12-31": [
                    "self_employment_income",
                    "partnership_s_corp_income",
                    "rental_income",
                    "estate_income",
                ]
            }
        },
        country_id="us",
    )

    def situation(output: str) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2026": 40},
                    output: {"2026": 10_000.0},
                }
            },
            "tax_units": {
                "tax_unit": {
                    "members": ["adult"],
                    "filing_status": {"2026": "SINGLE"},
                }
            },
            "spm_units": {"spm_unit": {"members": ["adult"]}},
            "households": {
                "household": {
                    "members": ["adult"],
                    "state_code": {"2026": "CA"},
                }
            },
        }

    reformed_system = CountryTaxBenefitSystem(reform=(reform,))
    for output in US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS:
        baseline = Simulation(tax_benefit_system=system, situation=situation(output))
        reformed = Simulation(
            tax_benefit_system=reformed_system,
            situation=situation(output),
        )
        assert baseline.calculate("qualified_business_income", 2026)[0] > 9_000.0
        assert baseline.calculate("qualified_business_income_deduction", 2026)[
            0
        ] == pytest.approx(400.0)
        assert reformed.calculate("qualified_business_income", 2026)[0] == 0.0
        assert reformed.calculate("qualified_business_income_deduction", 2026)[0] == 0.0
