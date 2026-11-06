"""The microcosm#1326 tool against policyengine-uk: the back-cast inverts the
engine's own forward step, and the Scottish property-rate reform charges
exactly ``delta`` on the property slice."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from policyengine_uk import CountryTaxBenefitSystem, Microsimulation
from policyengine_uk.data import UKSingleYearDataset
from policyengine_uk.data.economic_assumptions import extend_single_year_dataset

from tools.uk_scotland_property_income.backcast import (
    NOT_INVERTED,
    backcast_one_year,
    held_at_file_year,
    refuse_earlier_period,
    with_household_weights,
)
from tools.uk_scotland_property_income.reform import scottish_property_rate_reform

REGIONS = ("SCOTLAND", "SCOTLAND", "LONDON")


def _dataset(year: int) -> UKSingleYearDataset:
    ids = np.arange(len(REGIONS), dtype=np.int64)
    return UKSingleYearDataset(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": [45, 52, 45],
                "employment_income": [30_000.0, 48_000.0, 30_000.0],
                "property_income": [10_000.0, 14_000.0, 10_000.0],
                "property_finance_costs": [0.0, 3_000.0, 0.0],
                "savings_interest_income": [200.0, 0.0, 200.0],
                # Unmapped: the forward step leaves it flat, and so does the back-cast.
                "gift_aid": [100.0, 0.0, 0.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {
                "household_id": ids,
                "household_weight": [1_000.0, 800.0, 1_000.0],
                "region": list(REGIONS),
                "council_tax": [1_500.0, 1_800.0, 1_500.0],
                "rent": 0.0,
                "tenure_type": "OWNED_OUTRIGHT",
            }
        ),
        fiscal_year=year,
    )


def test_the_backcast_inverts_the_engines_forward_step():
    parameters = CountryTaxBenefitSystem().parameters
    released = _dataset(2024)
    earlier, divided = backcast_one_year(released, parameters)
    assert earlier.time_period == "2023"
    assert {
        "employment_income",
        "property_income",
        "property_finance_costs",
        "household_weight",
    } <= set(divided)
    assert (
        earlier.person["employment_income"].iloc[0]
        < released.person["employment_income"].iloc[0]
    )

    assert held_at_file_year(released, divided)["person"] == ["gift_aid"]

    # Every column the forward step touches comes back, not only the divided ones.
    forward = extend_single_year_dataset(earlier, parameters, end_year=2024).datasets[
        2024
    ]
    for table in ("person", "benunit", "household"):
        for column in getattr(released, table).columns:
            if column in NOT_INVERTED:
                continue
            after, before = (
                getattr(forward, table)[column],
                getattr(released, table)[column],
            )
            if pd.api.types.is_numeric_dtype(before.dtype):
                np.testing.assert_allclose(
                    after.to_numpy(dtype=float),
                    before.to_numpy(dtype=float),
                    rtol=1e-12,
                    err_msg=column,
                )
            else:
                assert after.tolist() == before.tolist(), column


def test_an_earlier_period_is_refused():
    with pytest.raises(ValueError, match="Back-cast it first"):
        refuse_earlier_period(_dataset(2024), 2023)
    refuse_earlier_period(_dataset(2024), 2024)


def _income_tax(dataset: UKSingleYearDataset, reform=None) -> np.ndarray:
    simulation = Microsimulation(
        dataset=with_household_weights(dataset, dataset.household["household_weight"]),
        reform=reform,
    )
    return np.asarray(simulation.calculate("income_tax", 2023), dtype=float), simulation


def test_the_reform_adds_delta_on_the_scottish_property_slice_only():
    dataset = _dataset(2023)
    baseline, simulation = _income_tax(dataset)
    property_slice = np.asarray(
        simulation.calculate("taxable_property_income_after_allowances", 2023),
        dtype=float,
    )
    reformed, _ = _income_tax(dataset, scottish_property_rate_reform(0.01))
    scottish = np.array([region == "SCOTLAND" for region in REGIONS])
    np.testing.assert_allclose(
        reformed - baseline, np.where(scottish, 0.01 * property_slice, 0.0), atol=1e-6
    )
    assert (reformed - baseline)[scottish].min() > 0


def test_the_relief_variant_matches_the_engine_at_zero_and_follows_the_rate():
    dataset = _dataset(2023)
    baseline, simulation = _income_tax(dataset)
    unchanged, _ = _income_tax(
        dataset, scottish_property_rate_reform(0.0, relief_follows=True)
    )
    np.testing.assert_allclose(unchanged, baseline, atol=1e-6)

    property_slice = np.asarray(
        simulation.calculate("taxable_property_income_after_allowances", 2023),
        dtype=float,
    )
    relieved = np.asarray(
        simulation.calculate("property_finance_costs_relieved", 2023), dtype=float
    )
    assert relieved[1] > 0
    following, _ = _income_tax(
        dataset, scottish_property_rate_reform(0.01, relief_follows=True)
    )
    expected = np.array(
        [0.01 * property_slice[0], 0.01 * (property_slice[1] - relieved[1]), 0.0]
    )
    np.testing.assert_allclose(following - baseline, expected, atol=1e-6)
