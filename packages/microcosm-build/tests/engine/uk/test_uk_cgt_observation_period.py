"""Tests split from packages/microcosm-build/tests/test_uk_cgt_observation_period.py."""

# ruff: noqa: F403, F405
from microcosm.build.uk_runtime.measure_simulation import (
    cgt_taxable_income_from_components,
)
from test_support.microcosm_build.uk_cgt_observation_period import *


def test_actual_engine_resolves_observed_cgt_without_relabelling_source(tmp_path):
    frame = _frame()
    resolver = UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=2025
    )
    gains, route = resolver.compute("person", "cgt_2024_gains")
    tax, _ = resolver.compute("person", "cgt_2024_tax")
    np.testing.assert_array_equal(gains, frame.table("person").capital_gains)
    np.testing.assert_array_equal(
        tax, np.asarray(resolver.simulation.calculate("capital_gains_tax", 2024))
    )
    assert route == "engine_period:2024:capital_gains:native"
    assert (gains > 3000).tolist() == [False, True, True]
    assert tax[0] == 0 and np.all(tax[1:] > 0)
    # A separate forward calculation crosses the fixed AEA; it is not the fit.
    future, _ = resolver.compute("person", "cgt_2025_gains")
    assert future[0] > 3000
    assert frame.metadata["time_period"] == "2024"
    np.testing.assert_array_equal(
        frame.table("person").capital_gains, [3000, 5000, 20000]
    )


def test_actual_engine_resolves_the_dated_cgt_taxable_income(tmp_path):
    """Table 3's banding measure comes from engine arrays at the disposal year."""
    frame = _frame()
    frame.table("person")["employment_income"] = [0.0, 30_000.0, 110_000.0]
    resolver = UKMeasureResolver(
        simulation_source=None, frame=frame, scratch_dir=tmp_path, year=2025
    )
    assert resolver.knows("person", "cgt_2024_taxable_income")
    assert not resolver.knows("household", "cgt_2024_taxable_income")
    assert resolver.entity_for("cgt_2024_taxable_income") == "person"
    income, route = resolver.compute("person", "cgt_2024_taxable_income")
    assert route == (
        "engine_period:2024:cgt_taxable_income:"
        "engine_components:capital_gains_tax_taxable_income"
    )
    # Hand-computed at the 2024-25 GBP 12,570 allowance: the third person's is
    # tapered by half the GBP 10,000 of income above GBP 100,000.
    np.testing.assert_allclose(income, [0.0, 17_430.0, 102_430.0])
    parts = {
        name: resolver.simulation.calculate(name, 2024)
        for name in measure_simulation._CGT_TAXABLE_INCOME_COMPONENTS
    }
    np.testing.assert_array_equal(income, cgt_taxable_income_from_components(parts))


#: One persona per term the engine's CGT formula nets off adjusted net income
#: (policyengine-uk capital_gains_tax): the personal allowance and its taper,
#: pension relief above and below the personal contributions that extend the
#: band, gift aid, Blind Person's Allowance, charitable investment gifts, no
#: income, a self-employment loss and a Scottish resident (UK rate bands).
_TAXABLE_INCOME_PERSONAS = {
    "zero_income": {},
    "basic_earner": {"employment_income": 30_000},
    "pa_taper": {"employment_income": 110_000},
    "gift_aid": {"employment_income": 60_000, "gift_aid": 8_000},
    "pension_above_relief": {
        "employment_income": 20_000,
        "personal_pension_contributions": 30_000,
    },
    "pension_below_relief": {
        "employment_income": 80_000,
        "personal_pension_contributions": 5_000,
        "employee_pension_contributions": 10_000,
    },
    "blind": {"employment_income": 40_000, "blind_persons_allowance": 3_130},
    "charitable_investment_gifts": {
        "employment_income": 50_000,
        "charitable_investment_gifts": 5_000,
    },
    "self_employment_loss": {
        "employment_income": 25_000,
        "self_employment_income": -5_000,
    },
    "scottish": {"employment_income": 45_000},
}


def test_derived_cgt_taxable_income_is_the_income_the_engine_stacks_gains_on():
    """Invert the engine's CGT rather than re-implement it (microcosm#1014).

    With main-schedule gains above the whole basic band the engine charges
    r_b * RB + r_h * (g - AEA - RB), so RB = (r_h * (g - AEA) - CGT) / (r_h -
    r_b) is the remaining basic band it stacked gains on. The reform lifts the
    basic rate limit above every persona's taxable income without touching
    adjusted net income or allowances, so RB = BRL - TI pins the taxable
    income exactly and the derived measure must reproduce it.
    """
    from policyengine_uk import Simulation

    from microcosm.build.uk_runtime.cgt_imputation import uk_cgt_policy_parameters

    year, gains = 2024, 3_000_000.0
    people, benunits, households = {}, {}, {}
    for name, inputs in _TAXABLE_INCOME_PERSONAS.items():
        people[name] = {
            "age": {year: 45},
            "capital_gains_before_response": {year: gains},
            **{variable: {year: value} for variable, value in inputs.items()},
        }
        benunits[f"b_{name}"] = {"members": [name]}
        households[f"h_{name}"] = {
            "members": [name],
            "region": {year: "SCOTLAND" if name == "scottish" else "LONDON"},
        }
    simulation = Simulation(
        situation={"people": people, "benunits": benunits, "households": households},
        reform={
            "gov.hmrc.income_tax.rates.uk[1].threshold": {
                "2020-01-01.2030-12-31": 1_000_000
            },
            "gov.hmrc.income_tax.rates.uk[2].threshold": {
                "2020-01-01.2030-12-31": 2_000_000
            },
        },
    )
    parameters = simulation.tax_benefit_system.parameters(f"{year}-01-01").gov.hmrc
    cgt, rates = parameters.cgt, parameters.income_tax.rates.uk
    # In law the additional rate equals the higher rate, so gains above the
    # higher rate limit do not change the inversion.
    assert cgt.additional_rate == cgt.higher_rate
    tax = np.asarray(simulation.calculate("capital_gains_tax", year), dtype=float)
    band = (cgt.higher_rate * (gains - cgt.annual_exempt_amount) - tax) / (
        cgt.higher_rate - cgt.basic_rate
    )
    parts = {
        name: simulation.calculate(name, year)
        for name in measure_simulation._CGT_TAXABLE_INCOME_COMPONENTS
    }
    taxable_income = cgt_taxable_income_from_components(parts)
    band_extension = np.asarray(
        simulation.calculate("gift_aid_grossed_up", year), dtype=float
    ) + np.minimum(
        parts["personal_pension_contributions"], parts["pension_contributions_relief"]
    )
    basic_rate_limit = rates.thresholds[1] + band_extension
    assert np.all(taxable_income < basic_rate_limit)
    # The engine computes in float32: pounds of rounding on a GBP 3m gain.
    np.testing.assert_allclose(band, basic_rate_limit - taxable_income, atol=2.0)
    assert len(set(taxable_income.round().tolist())) >= 8
    # The stage's 1 June read and the engine's own 2024 parameters agree.
    stage = uk_cgt_policy_parameters(year)
    allowance = parameters.income_tax.allowances.personal_allowance
    assert stage.annual_exempt_amount == cgt.annual_exempt_amount
    assert stage.personal_allowance == allowance.amount
    assert stage.personal_allowance_taper_threshold == allowance.maximum_ANI
    assert stage.personal_allowance_taper_rate == allowance.reduction_rate
