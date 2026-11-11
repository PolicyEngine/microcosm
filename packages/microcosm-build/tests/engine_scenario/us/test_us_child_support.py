"""Tests split from packages/microcosm-build/tests/test_us_child_support.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_child_support import *


def test_policyengine_us_contract_is_two_person_year_input_leaves() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variables = CountryTaxBenefitSystem().variables
    for name in US_CHILD_SUPPORT_OUTPUT_COLUMNS:
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert str(variable.definition_period).lower() == "year"
        assert variable.default_value == 0


def test_policyengine_us_graph_uses_positive_annual_received_and_expense() -> None:
    from policyengine_us import Simulation

    def situation(received: float, expense: float) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2024": 35},
                    # PE-US 1.769.0 began applying the payer's SNAP income
                    # counted share to deductible child-support expense, and
                    # 1.794.3 changed missing pre-LSR work hours from 40 to 0.
                    # Keep this graph test on the positive-expense path by
                    # making the adult satisfy the work requirement explicitly.
                    "weekly_hours_worked_before_lsr": {"2024": 40},
                    _RECEIVED: {"2024": received},
                    _EXPENSE: {"2024": expense},
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

    baseline = Simulation(situation=situation(0.0, 0.0))
    active = Simulation(situation=situation(3_600.0, 2_400.0))

    assert (
        active.calculate("spm_unit_benefits", 2024)[0]
        - baseline.calculate("spm_unit_benefits", 2024)[0]
    ) == pytest.approx(3_600.0)
    assert (
        active.calculate("spm_unit_spm_expenses", 2024)[0]
        - baseline.calculate("spm_unit_spm_expenses", 2024)[0]
    ) == pytest.approx(2_400.0)
    assert active.calculate("snap_child_support_deduction", "2024-01")[
        0
    ] == pytest.approx(200.0)
