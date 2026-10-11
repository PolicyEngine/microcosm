"""The live counterfactual route against the real engine (microcosm#1069 c11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.income_tax_rate_classes import (
    UK_RATE_CLASS_ALLOCATION,
    UK_RATE_CLASSES,
)
from microcosm.build.uk_runtime.measure_simulation import UKMeasureResolver
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind


def _frame():
    ids = np.arange(3, dtype=np.int64)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [40, 40, 40],
                "employment_income": [40_000.0, 30_000.0, 0.0],
                "pension_contributions_via_salary_sacrifice": [3_000.0, 0.0, 0.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [1.0, 1.0, 1.0],
                "region": "LONDON",
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


def _relief_binding(output: str) -> dict:
    return {
        "metric_name": f"hmrc/salary_sacrifice_{output}_relief",
        "kind": "input_substitution_counterfactual",
        "zeroed_input": "pension_contributions_via_salary_sacrifice",
        "folded_into": "employment_income",
        "output_delta": "counterfactual_minus_baseline",
        "output_variable": output,
        "from_entity": "person",
    }


def test_returned_sacrifice_is_taxed_at_the_basic_rate_and_main_nics_rates(tmp_path):
    resolver = UKMeasureResolver(
        simulation_source=None, frame=_frame(), scratch_dir=tmp_path, year=2025
    )
    sacrificed = np.asarray(
        resolver.simulation.calculate(
            "pension_contributions_via_salary_sacrifice", 2025
        ),
        dtype=float,
    )
    pay = np.asarray(resolver.simulation.calculate("employment_income", 2025))
    # The returned pay stays inside the 2025-26 basic-rate band and below the
    # NICs upper earnings limit, so the relief is linear in the sacrifice.
    assert sacrificed[0] > 0.0 and pay[0] + sacrificed[0] < 50_270.0

    income_tax, route = resolver.counterfactual_delta(
        _relief_binding("income_tax"), 2025
    )
    employee, _ = resolver.counterfactual_delta(_relief_binding("ni_employee"), 2025)
    employer, _ = resolver.counterfactual_delta(_relief_binding("ni_employer"), 2025)

    # Hand-computed at the 2025-26 rates: 20% income tax, 8% employee and 15%
    # employer NICs on the sacrificed pay; nobody else moves.
    expected = np.array([sacrificed[0], 0.0, 0.0])
    np.testing.assert_allclose(income_tax, 0.20 * expected, atol=0.01)
    np.testing.assert_allclose(employee, 0.08 * expected, atol=0.01)
    np.testing.assert_allclose(employer, 0.15 * expected, atol=0.01)
    assert "folded into employment_income" in route
    measures = resolver.receipt()["counterfactual_measures"]
    assert set(measures) == {
        "hmrc/salary_sacrifice_income_tax_relief",
        "hmrc/salary_sacrifice_ni_employee_relief",
        "hmrc/salary_sacrifice_ni_employer_relief",
    }


def _band_frame():
    """uk-data#533's three examples at 2025 (pay after the sacrifice), plus the
    first with GBP 10,000 of dividends."""

    ids = np.arange(4, dtype=np.int64)
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [40, 40, 40, 40],
                "employment_income": [48_000.0, 42_000.0, 100_000.0, 48_000.0],
                "pension_contributions_via_salary_sacrifice": [
                    4_000.0,
                    4_000.0,
                    10_000.0,
                    4_000.0,
                ],
                "dividend_income": [0.0, 0.0, 0.0, 10_000.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [1.0, 1.0, 1.0, 1.0],
                "region": ["LONDON", "SCOTLAND", "LONDON", "LONDON"],
                "council_tax": 0.0,
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        time_period="2025",
        weight_kind=WeightKind.DESIGN,
    )


def _band_binding(rate_class: str) -> dict:
    return {
        "metric_name": f"hmrc/salary_sacrifice_it_relief_{rate_class}_rate",
        "kind": "input_substitution_counterfactual",
        "zeroed_input": "pension_contributions_via_salary_sacrifice",
        "folded_into": "employment_income",
        "output_delta": "counterfactual_minus_baseline",
        "output_variable": "earned_taxable_income",
        "allocation": {**UK_RATE_CLASS_ALLOCATION, "rate_class": rate_class},
        "from_entity": "person",
    }


def test_relief_bands_relieve_each_slice_at_its_rate(tmp_path):
    """uk-data#533's examples (microcosm#1095): rest of the UK, GBP 4,000 on
    GBP 48,000 is 454 basic and 692 higher; Scotland, GBP 4,000 on GBP 42,000
    is 349.02 basic and 981.96 higher; GBP 10,000 on GBP 100,000 withdraws
    GBP 5,000 of personal allowance, so 6,000, all higher."""

    resolver = UKMeasureResolver(
        simulation_source=None, frame=_band_frame(), scratch_dir=tmp_path, year=2025
    )

    relief = {
        rate_class: resolver.counterfactual_delta(_band_binding(rate_class), 2025)[0]
        for rate_class in UK_RATE_CLASSES
    }

    expected = {
        "basic": [454.0, 349.02, 0.0, 454.0],
        "higher": [692.0, 981.96, 6_000.0, 692.0],
        "additional": [0.0, 0.0, 0.0, 0.0],
    }
    for rate_class in UK_RATE_CLASSES:
        np.testing.assert_allclose(relief[rate_class], expected[rate_class], atol=0.01)
    # With pay the only income the bands add up to the rise in income tax. The
    # returned pay also pushes the fourth person's dividends into the higher
    # dividend band: that tax is not salary-sacrifice relief, and the receipt
    # records the gap.
    income_tax, _ = resolver.counterfactual_delta(_relief_binding("income_tax"), 2025)
    total = sum(relief.values())
    np.testing.assert_allclose(total[:3], income_tax[:3], atol=0.01)
    assert income_tax[3] > total[3] + 100.0
    (receipt,) = resolver.receipt()["rate_class_reliefs"].values()
    assert receipt["persons_with_gap"] == 1
    assert receipt["gap_to_income_tax_change_unweighted"] == pytest.approx(
        float(np.sum(income_tax) - np.sum(total)), abs=0.01
    )
    # One counterfactual simulation serves the three bands and the total.
    assert len(resolver._counterfactuals) == 1
