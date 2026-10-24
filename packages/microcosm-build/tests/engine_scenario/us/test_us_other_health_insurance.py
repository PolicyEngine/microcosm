"""Tests split from packages/microcosm-build/tests/test_us_other_health_insurance.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_other_health_insurance import *


def test_policyengine_us_graph_uses_other_health_insurance_premiums() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variable = CountryTaxBenefitSystem().variables[_OTHER]
    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0

    def situation(premiums: float) -> dict[str, object]:
        return {
            "people": {
                "adult": {
                    "age": {"2024": 35},
                    _OTHER: {"2024": premiums},
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

    baseline = Simulation(situation=situation(0.0))
    active = Simulation(situation=situation(1_200.0))
    expected_deltas = {
        "spm_unit_health_insurance_premiums": 1_200.0,
        "spm_unit_non_premium_medical_out_of_pocket_expenses": 0.0,
        "spm_unit_medical_out_of_pocket_expenses": 1_200.0,
        "spm_unit_spm_expenses": 1_200.0,
        "spm_unit_net_income": -1_200.0,
    }
    for name, expected in expected_deltas.items():
        delta = active.calculate(name, 2024)[0] - baseline.calculate(name, 2024)[0]
        assert delta == pytest.approx(expected)


def test_policyengine_2_2_1_se_health_ald_contract_and_live_binding() -> None:
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    assert version("policyengine-us") == "2.2.1"
    variables = CountryTaxBenefitSystem().variables
    for name in US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS:
        variable = variables[name]
        assert variable.is_input_variable()
        assert variable.entity.key == "person"
        assert str(variable.definition_period).lower() == "year"
    assert variables[_PREMIUMS_OUTPUT].value_type is float
    assert variables[_FLAG_OUTPUT].value_type is bool

    def situation(
        *,
        premiums: float,
        self_employed: bool,
        self_employment_income: float = 0.0,
        sstb_income: float = 0.0,
        reported: float = 0.0,
    ) -> dict[str, object]:
        person: dict[str, object] = {
            "age": {"2024": 40},
            _PREMIUMS_OUTPUT: {"2024": premiums},
            _FLAG_OUTPUT: {"2024": self_employed},
            "self_employment_income": {"2024": self_employment_income},
            "sstb_self_employment_income": {"2024": sstb_income},
            _REPORTED: {"2024": reported},
        }
        return {
            "people": {"adult": person},
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

    # min(total self-employment income, premiums) is the section 162(l) chain.
    bound = Simulation(
        situation=situation(
            premiums=8_000.0, self_employed=True, self_employment_income=30_000.0
        )
    )
    assert bound.calculate("self_employed_health_insurance_premiums", 2024)[
        0
    ] == pytest.approx(8_000.0)
    assert bound.calculate("self_employed_health_insurance_ald", 2024)[
        0
    ] == pytest.approx(8_000.0)

    earnings_capped = Simulation(
        situation=situation(
            premiums=50_000.0, self_employed=True, self_employment_income=30_000.0
        )
    )
    assert earnings_capped.calculate("self_employed_health_insurance_ald", 2024)[
        0
    ] == pytest.approx(30_000.0)

    sstb_bound = Simulation(
        situation=situation(premiums=8_000.0, self_employed=True, sstb_income=10_000.0)
    )
    assert sstb_bound.calculate("self_employed_health_insurance_ald", 2024)[
        0
    ] == pytest.approx(8_000.0)

    # The flag gates the adds-chain: without it the person-level premium
    # input never reaches the deduction.
    unflagged = Simulation(
        situation=situation(
            premiums=8_000.0, self_employed=False, self_employment_income=30_000.0
        )
    )
    assert unflagged.calculate("self_employed_health_insurance_ald", 2024)[0] == 0.0

    # Medical-expense invariance receipt: for a non-Medicare person the
    # direct premium input equals the decomposed reported premium, so the
    # statutory medical-expense concept is unchanged by attribution.
    attributed = Simulation(
        situation=situation(
            premiums=8_000.0,
            self_employed=True,
            self_employment_income=30_000.0,
            reported=8_000.0,
        )
    )
    unattributed = Simulation(
        situation=situation(
            premiums=0.0,
            self_employed=False,
            self_employment_income=30_000.0,
            reported=8_000.0,
        )
    )
    assert attributed.calculate("medical_expense_health_insurance_premiums", 2024)[
        0
    ] == pytest.approx(
        unattributed.calculate("medical_expense_health_insurance_premiums", 2024)[0]
    )

    # Neutralizing the shipped premium leaf removes the deduction, which is
    # exactly what the release coverage probe measures on income_tax.
    from policyengine_core.reforms import Reform

    class NeutralizePremiums(Reform):
        def apply(self) -> None:
            self.neutralize_variable(_PREMIUMS_OUTPUT)

    neutralized = Simulation(
        situation=situation(
            premiums=8_000.0, self_employed=True, self_employment_income=30_000.0
        ),
        reform=NeutralizePremiums,
    )
    assert neutralized.calculate("self_employed_health_insurance_ald", 2024)[0] == 0.0
    assert (
        neutralized.calculate("income_tax", 2024)[0]
        > bound.calculate("income_tax", 2024)[0]
    )
