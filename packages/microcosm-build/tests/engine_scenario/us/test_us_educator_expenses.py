"""Tests split from packages/microcosm-build/tests/test_us_educator_expenses.py."""

# ruff: noqa: F403, F405
from hypothesis import given, settings
from hypothesis import strategies as st

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


_ALD_DEDUCTIONS = "gov.irs.ald.deductions"


def _abolition_probe():
    return next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "educator_expense_ald_abolition"
    )


def _educator_household(educator_expense: float, employment_income: float) -> dict:
    return {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": employment_income},
                "educator_expense": {"2024": educator_expense},
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


@pytest.fixture(scope="module")
def abolition():
    """The smoke's own reform and the reform system the release scorer builds."""
    from policyengine_us import CountryTaxBenefitSystem

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    reform = _build_reform(_abolition_probe())
    return reform, CountryTaxBenefitSystem(reform=(reform,))


def test_shipped_abolition_probe_has_positive_sign_and_binds_live(abolition) -> None:
    from datetime import date, timedelta

    from policyengine_us import Simulation

    from microcosm.build.us_runtime.release_input_coverage import (
        ListParameterEdit,
        resolve_probe_parameter_changes,
    )

    probe = _abolition_probe()
    assert probe.period == 2024
    assert probe.expected_sign == "positive"
    assert probe.effect_direction == "reform_minus_baseline"
    assert probe.budget_measure == "income_tax"
    assert probe.binding_inputs == ("educator_expense",)
    assert probe.min_abs_effect == 1_000_000.0
    assert probe.parameter_changes == {}
    edit = ListParameterEdit(
        period="2024-01-01.2024-12-31", remove=("educator_expense",)
    )
    assert probe.list_edits == {_ALD_DEDUCTIONS: edit}

    reform, reformed_system = abolition
    situation = _educator_household(300.0, 100_000.0)
    baseline = Simulation(situation=situation)
    reformed = Simulation(tax_benefit_system=reformed_system, situation=situation)

    # Not a no-op: the installed engine resolves the declared removal to its
    # own deduction list minus the educator expense, and the reformed engine
    # holds exactly that list over the edit period and the baseline outside it.
    deductions = baseline.tax_benefit_system.parameters.get_child(_ALD_DEDUCTIONS)
    reformed_deductions = reformed_system.parameters.get_child(_ALD_DEDUCTIONS)
    assert "educator_expense" in deductions(edit.start)
    expected = [item for item in deductions(edit.start) if item != "educator_expense"]
    assert resolve_probe_parameter_changes(probe) == {
        _ALD_DEDUCTIONS: {edit.period: expected}
    }
    assert reform.resolved_list_edits == {_ALD_DEDUCTIONS: expected}
    for instant in (edit.start, edit.stop):
        assert list(reformed_deductions(instant)) == expected
    for instant in (
        date.fromisoformat(edit.start) - timedelta(days=1),
        date.fromisoformat(edit.stop) + timedelta(days=1),
    ):
        assert list(reformed_deductions(str(instant))) == list(deductions(str(instant)))

    assert baseline.calculate("above_the_line_deductions", 2024)[0] == 300.0
    assert reformed.calculate("above_the_line_deductions", 2024)[0] == 0.0
    assert (
        reformed.calculate("income_tax", 2024)[0]
        - baseline.calculate("income_tax", 2024)[0]
        > 0.0
    )


@settings(max_examples=15, deadline=None)
@given(
    educator_expense=st.integers(min_value=0, max_value=5_000),
    employment_income=st.integers(min_value=0, max_value=300_000),
)
def test_abolition_probe_removes_exactly_the_educator_deduction(
    abolition, educator_expense: int, employment_income: int
) -> None:
    # Invariant: for any educator expense and wage, the reform lowers
    # above-the-line deductions by exactly the expense and never lowers income
    # tax, so the probe's positive reform-minus-baseline sign holds per unit.
    from policyengine_us import Simulation

    _, reformed_system = abolition
    situation = _educator_household(float(educator_expense), float(employment_income))
    baseline = Simulation(situation=situation)
    reformed = Simulation(tax_benefit_system=reformed_system, situation=situation)

    deduction_drop = (
        baseline.calculate("above_the_line_deductions", 2024)[0]
        - reformed.calculate("above_the_line_deductions", 2024)[0]
    )
    assert deduction_drop == pytest.approx(float(educator_expense), abs=0.01)
    assert (
        reformed.calculate("income_tax", 2024)[0]
        >= baseline.calculate("income_tax", 2024)[0] - 0.01
    )
