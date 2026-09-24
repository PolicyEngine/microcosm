"""Tests split from packages/microcosm-build/tests/test_us_educator_expenses.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.us_educator_expenses import *


def test_processed_puf_person_array_is_available_to_tax_unit_donor() -> None:
    donor = puf_tax_unit_donor_from_arrays(
        {
            "tax_unit_id": [10, 20],
            "household_weight": [100.0, 200.0],
            "filing_status": [b"SINGLE", b"JOINT"],
            "person_tax_unit_id": [10, 20, 20],
            "educator_expense": [300.0, 150.0, 150.0],
        },
        person_outputs=("educator_expense",),
        tax_unit_outputs=(),
    )

    assert donor["educator_expense"].tolist() == [300.0, 300.0]
    assert donor["weight"].tolist() == [100.0, 200.0]
    assert donor["tax_unit_person_count"].tolist() == [1, 2]


def test_weighted_qrf_writes_only_puf_channel_and_allocates_by_employment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeQRF:
        def __init__(self, *, n_estimators: int, seed: int) -> None:
            assert n_estimators == 4
            assert seed == 9

        def fit(
            self,
            frame,
            predictors,
            outputs,
            *,
            weights,
        ) -> "FakeQRF":
            assert predictors == [
                "puf_predictor_filing_status_code",
                "puf_predictor_tax_unit_person_count",
            ]
            assert outputs == ["educator_expense"]
            assert weights == "design"
            return self

        def predict(
            self, features: pd.DataFrame, *, release_models: bool = False
        ) -> pd.DataFrame:
            assert len(features) == 2
            assert release_models is True
            # Sparse snapping maps these to observed donor values 600 and 0.
            return pd.DataFrame(
                {"educator_expense": [500.0, 100.0]},
                index=features.index,
            )

    monkeypatch.setattr(puf_support_module, "QRF", FakeQRF)
    expanded = clone_us_frame_for_puf_support(_minimal_us_frame())
    donor = pd.DataFrame(
        {
            "puf_predictor_filing_status_code": [1.0, 2.0],
            "puf_predictor_tax_unit_person_count": [1.0, 2.0],
            "educator_expense": [0.0, 600.0],
            "weight": [1.0, 1.0],
        }
    )

    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(
            "puf_predictor_filing_status_code",
            "puf_predictor_tax_unit_person_count",
        ),
        person_outputs=("educator_expense",),
        tax_unit_outputs=(),
        n_estimators=4,
        seed=9,
    )

    person = imputed.table("person")
    channel = support_channel_column("person")
    asec = person[person[channel] == BASE_ASEC_SUPPORT_CHANNEL]
    puf = person[person[channel] == PUF_TAX_DETAIL_SUPPORT_CHANNEL]
    assert asec["educator_expense"].tolist() == [0.0, 0.0, 0.0]
    assert puf["educator_expense"].tolist() == [450.0, 150.0, 0.0]
    assert puf.groupby("person_tax_unit_id")["educator_expense"].sum().tolist() == [
        600.0,
        0.0,
    ]
    assert "educator_expense" not in expanded.table("person")


def test_policyengine_us_17646_contract_is_person_year_input_in_ald_graph() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    assert version("policyengine-us") == "2.2.1"
    system = CountryTaxBenefitSystem()
    variable = system.variables["educator_expense"]

    assert variable.is_input_variable()
    assert variable.entity.key == "person"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0
    assert (
        variable.uprating
        == "calibration.gov.cbo.income_by_source.adjusted_gross_income"
    )
    assert system.variables["above_the_line_deductions"].adds == (
        "gov.irs.ald.deductions"
    )
    assert "educator_expense" in system.parameters.gov.irs.ald.deductions("2024-01-01")


def test_shipped_abolition_probe_has_positive_sign_and_binds_live() -> None:
    from policyengine_core.reforms import Reform
    from policyengine_us import CountryTaxBenefitSystem, Simulation

    probe = next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "educator_expense_ald_abolition"
    )
    assert probe.period == 2024
    assert probe.expected_sign == "positive"
    assert probe.effect_direction == "reform_minus_baseline"
    assert probe.budget_measure == "income_tax"
    assert probe.binding_inputs == ("educator_expense",)
    assert probe.min_abs_effect == 1_000_000.0
    deductions = probe.parameter_changes["gov.irs.ald.deductions"]
    assert set(deductions) == {"2024-01-01.2024-12-31"}
    assert "educator_expense" not in deductions["2024-01-01.2024-12-31"]

    situation = {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": 100_000},
                "educator_expense": {"2024": 300},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
            }
        },
        "households": {
            "household": {
                "members": ["adult"],
                "state_code": {"2024": "CA"},
            }
        },
    }
    reform = Reform.from_dict(dict(probe.parameter_changes), country_id="us")
    baseline = Simulation(situation=situation)
    reformed = Simulation(
        tax_benefit_system=CountryTaxBenefitSystem(reform=(reform,)),
        situation=situation,
    )

    assert baseline.calculate("above_the_line_deductions", 2024)[0] == 300.0
    assert reformed.calculate("above_the_line_deductions", 2024)[0] == 0.0
    assert (
        reformed.calculate("income_tax", 2024)[0]
        - baseline.calculate("income_tax", 2024)[0]
        > 0.0
    )
