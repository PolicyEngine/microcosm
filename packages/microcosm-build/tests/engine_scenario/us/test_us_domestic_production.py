"""Tests split from packages/microcosm-build/tests/test_us_domestic_production.py."""

# ruff: noqa: F403, F405
from hypothesis import given, settings
from hypothesis import strategies as st

from test_support.microcosm_build.us_domestic_production import *


def test_policyengine_us_contract_is_a_tax_unit_year_input_leaf() -> None:
    from policyengine_us import CountryTaxBenefitSystem

    variable = CountryTaxBenefitSystem().variables["domestic_production_ald"]

    assert variable.is_input_variable()
    assert variable.entity.key == "tax_unit"
    assert str(variable.definition_period).lower() == "year"
    assert variable.default_value == 0


_ALD_DEDUCTIONS = "gov.irs.ald.deductions"


def _reactivation_probe():
    return next(
        probe
        for probe in us_release_reform_coverage_probes()
        if probe.id == "domestic_production_ald_reactivation"
    )


def _domestic_production_household(
    domestic_production_ald: float, employment_income: float
) -> dict:
    return {
        "people": {
            "adult": {
                "age": {"2024": 40},
                "employment_income": {"2024": employment_income},
            }
        },
        "tax_units": {
            "tax_unit": {
                "members": ["adult"],
                "filing_status": {"2024": "SINGLE"},
                "domestic_production_ald": {"2024": domestic_production_ald},
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
def reactivation():
    """The smoke's own reform and the reform system the release scorer builds."""
    from policyengine_us import CountryTaxBenefitSystem

    from microcosm.build.us_runtime.reform_coverage_smoke import _build_reform

    reform = _build_reform(_reactivation_probe())
    return reform, CountryTaxBenefitSystem(reform=(reform,))


def test_2024_reactivation_probe_binds_only_when_the_input_is_populated(
    reactivation,
) -> None:
    from datetime import date, timedelta

    from policyengine_us import Simulation

    probe = _reactivation_probe()
    assert probe.parameter_changes == {}
    assert set(probe.list_edits) == {_ALD_DEDUCTIONS}
    edit = probe.list_edits[_ALD_DEDUCTIONS]
    assert (edit.remove, edit.add) == ((), ("domestic_production_ald",))

    reform, reformed_system = reactivation
    situation = _domestic_production_household(10_000.0, 100_000.0)
    baseline = Simulation(situation=situation)
    reformed = Simulation(tax_benefit_system=reformed_system, situation=situation)

    # Not a no-op: over the edit period the reformed engine holds the installed
    # baseline's deduction list with the repealed deduction appended, and
    # outside it the baseline list, which no longer names it.
    deductions = baseline.tax_benefit_system.parameters.get_child(_ALD_DEDUCTIONS)
    reformed_deductions = reformed_system.parameters.get_child(_ALD_DEDUCTIONS)
    assert "domestic_production_ald" not in deductions(edit.start)
    expected = [*deductions(edit.start), "domestic_production_ald"]
    assert reform.resolved_list_edits == {_ALD_DEDUCTIONS: expected}
    for instant in (edit.start, edit.stop):
        assert list(reformed_deductions(instant)) == expected
    for instant in (
        date.fromisoformat(edit.start) - timedelta(days=1),
        date.fromisoformat(edit.stop) + timedelta(days=1),
    ):
        assert list(reformed_deductions(str(instant))) == list(deductions(str(instant)))

    assert baseline.calculate("above_the_line_deductions", 2024)[0] == 0.0
    assert reformed.calculate("above_the_line_deductions", 2024)[0] == 10_000.0
    effect = (
        baseline.calculate("income_tax", 2024)[0]
        - reformed.calculate("income_tax", 2024)[0]
    )
    assert effect > 1_000.0


@settings(max_examples=15, deadline=None)
@given(
    domestic_production_ald=st.integers(min_value=0, max_value=50_000),
    employment_income=st.integers(min_value=0, max_value=300_000),
)
def test_reactivation_probe_adds_exactly_the_domestic_production_deduction(
    reactivation, domestic_production_ald: int, employment_income: int
) -> None:
    # Invariant: for any domestic-production ALD and wage, the reform raises
    # above-the-line deductions by exactly the input and never raises income
    # tax, so the probe's positive baseline-minus-reform sign holds per unit.
    from policyengine_us import Simulation

    _, reformed_system = reactivation
    situation = _domestic_production_household(
        float(domestic_production_ald), float(employment_income)
    )
    baseline = Simulation(situation=situation)
    reformed = Simulation(tax_benefit_system=reformed_system, situation=situation)

    deduction_rise = (
        reformed.calculate("above_the_line_deductions", 2024)[0]
        - baseline.calculate("above_the_line_deductions", 2024)[0]
    )
    assert deduction_rise == pytest.approx(float(domestic_production_ald), abs=0.01)
    assert (
        reformed.calculate("income_tax", 2024)[0]
        <= baseline.calculate("income_tax", 2024)[0] + 0.01
    )
