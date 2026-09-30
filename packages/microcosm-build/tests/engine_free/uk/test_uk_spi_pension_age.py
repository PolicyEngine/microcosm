"""State Pension age coherence of the SPI income stage (microcosm#1069, WP2 c5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import spi_income
from microcosm.build.uk_runtime.spi_income import (
    SPIDonorAgeModel,
    UKSPIStatePensionAgeGuard,
)


def _flat_model(state_pension_age: int = 66) -> SPIDonorAgeModel:
    populations = {
        (sex, age): 100.0 for sex in ("MALE", "FEMALE") for age in range(0, 91)
    }
    return SPIDonorAgeModel(
        populations=populations,
        state_pension_age=state_pension_age,
        source="fixture",
    )


def test_spi_age_bands_follow_the_published_agerange_edges() -> None:
    # Code 6 is "65 to 74" and code 7 "75 and over": a 74-year-old belongs to
    # band 6, which the old [65, 74) edge excluded.
    assert spi_income._SPI_AGE_RANGES[6] == (65, 75)
    assert spi_income._SPI_AGE_RANGES[7][0] == 75
    codes = spi_income._spi_agerange_codes(np.asarray([16.0, 64.9, 65.0, 74.5, 75.0]))
    assert codes.tolist() == [1, 5, 6, 6, 7]


def test_recipients_in_the_band_straddling_state_pension_age_draw_at_or_above_it() -> (
    None
):
    count = 4_000
    rng = np.random.default_rng(3)
    codes = np.full(count, 6)
    sex = np.asarray(["MALE", "FEMALE"] * (count // 2), dtype=object)
    receives = rng.random(count) < 0.8
    fact = np.ones(count)

    ages, receipt = spi_income._draw_spi_donor_ages(
        codes, sex, receives, fact, np.random.default_rng(1), model=_flat_model()
    )

    assert ((ages >= 65.0) & (ages < 75.0)).all()
    assert (ages[receives] >= 66.0).all()
    # A flat population puts one tenth of the band at 65; non-recipients supply
    # all of it.
    below = ages < 66.0
    assert below.sum() / count == pytest.approx(0.1, abs=0.02)
    assert not (below & receives).any()
    cells = {(row["agerange"], row["sex"]): row for row in receipt}
    assert cells[(6, "MALE")]["ons_below_state_pension_age_share"] == pytest.approx(0.1)
    assert cells[(6, "MALE")]["recipients_below_state_pension_age"] == 0


def test_an_infeasible_below_age_share_is_receipted_not_forced() -> None:
    codes = np.full(10, 6)
    sex = np.asarray(["MALE"] * 10, dtype=object)
    receives = np.asarray([True] * 10)
    ages, receipt = spi_income._draw_spi_donor_ages(
        codes,
        sex,
        receives,
        np.ones(10),
        np.random.default_rng(0),
        model=_flat_model(),
    )

    assert (ages >= 66.0).all()
    (cell,) = receipt
    assert cell["non_recipient_below_probability"] == 0.0
    assert cell["unfilled_below_state_pension_age_mass"] == pytest.approx(1.0)


def test_single_year_shares_follow_the_population_within_a_band() -> None:
    populations = {
        (sex, age): (900.0 if age == 80 else 1.0)
        for sex in ("MALE", "FEMALE")
        for age in range(0, 91)
    }
    model = SPIDonorAgeModel(
        populations=populations, state_pension_age=66, source="fixture"
    )
    ages, _ = spi_income._draw_spi_donor_ages(
        np.full(2_000, 7),
        np.asarray(["FEMALE"] * 2_000, dtype=object),
        np.ones(2_000, dtype=bool),
        np.ones(2_000),
        np.random.default_rng(2),
        model=model,
    )

    assert (np.floor(ages) == 80).mean() > 0.9


def test_unknown_sex_draws_on_both_sexes() -> None:
    model = _flat_model()
    both = model.weights("UNKNOWN", range(65, 70))
    male = model.weights("MALE", range(65, 70))
    np.testing.assert_array_equal(both, 2 * male)


def test_age_model_refuses_invalid_keys_and_values() -> None:
    with pytest.raises(ValueError, match="invalid population key"):
        SPIDonorAgeModel(
            populations={("OTHER", 1): 1.0}, state_pension_age=66, source="x"
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        SPIDonorAgeModel(
            populations={("MALE", 1): -1.0}, state_pension_age=66, source="x"
        )
    with pytest.raises(ValueError, match="integer State Pension age"):
        SPIDonorAgeModel(populations={}, state_pension_age=0, source="x")


def test_zeroing_below_state_pension_age_receipts_what_it_removes() -> None:
    person = pd.DataFrame(
        {
            "age": [65, 66, 40, 70],
            "state_pension_reported": [9_000.0, 11_000.0, 500.0, 12_000.0],
            "pension_credit_reported": [300.0, 0.0, 0.0, 400.0],
        },
        index=[1, 2, 3, 4],
    )
    rows = pd.Series([True, True, False, True], index=person.index)
    weights = pd.Series([10.0, 20.0, 30.0, 40.0], index=person.index)

    result, receipt = spi_income._zero_below_state_pension_age(
        person.copy(),
        rows=rows,
        columns=("state_pension_reported", "pension_credit_reported"),
        state_pension_age=66,
        person_weights=weights,
        step="fixture",
    )

    assert result["state_pension_reported"].tolist() == [0.0, 11_000.0, 500.0, 12_000.0]
    assert result["pension_credit_reported"].tolist() == [0.0, 0.0, 0.0, 400.0]
    removed = receipt["removed"]["state_pension_reported"]
    assert removed == {
        "rows": 1,
        "weighted_people": 10.0,
        "unweighted_amount": 9_000.0,
        "weighted_amount": 90_000.0,
    }
    assert receipt["rows_below_state_pension_age"] == 1


def test_zeroing_refuses_a_column_absent_from_the_frame() -> None:
    person = pd.DataFrame({"age": [65]})
    with pytest.raises(ValueError, match="absent from the frame"):
        spi_income._zero_below_state_pension_age(
            person,
            rows=pd.Series([True]),
            columns=("state_pension_reported",),
            state_pension_age=66,
            person_weights=pd.Series([1.0]),
            step="fixture",
        )


def test_guard_refuses_duplicate_columns() -> None:
    with pytest.raises(ValueError, match="distinct names"):
        UKSPIStatePensionAgeGuard(
            state_pension_age=66,
            spi_channel_columns=("a", "a"),
            base_channel_columns=(),
        )


def _band_resample_inputs(carrier_age: float):
    columns = list(spi_income.SPI_INCOME_QRF_OUTPUT_COLUMNS)
    rows = 60
    donor = pd.DataFrame(0.0, index=range(rows), columns=columns)
    # Every donor sits in the GBP 200,000 band; the first 30 are aged 40 to
    # 44 (code 3), the rest 75 and over (code 7) with State Pension.
    donor["employment_income"] = 250_000.0
    donor["total_income"] = 250_000.0
    donor["is_composite"] = False
    donor["region"] = "LONDON"
    donor["FACT"] = 1.0
    donor["agerange"] = [3] * 30 + [7] * 30
    donor.loc[30:, spi_income.SPI_HMRC_STATE_PENSION_INCOME_COLUMN] = 10_000.0
    person = pd.DataFrame(
        {
            "person_household_id": [1],
            "age": [carrier_age],
            spi_income.SPI_INCOME_BAND_CARRIER_COLUMN: [True],
        }
    )
    for column in columns:
        person[column] = 0.0
    household = pd.DataFrame(
        {
            "household_id": [1],
            "region": ["LONDON"],
            spi_income.SPI_INCOME_BAND_LOWER_BOUND_COLUMN: [200_000.0],
        }
    )
    return person, donor, household


@pytest.mark.parametrize("seed", range(5))
def test_band_carriers_draw_from_their_own_age_band_when_the_pool_allows(
    seed: int,
) -> None:
    person, donor, household = _band_resample_inputs(carrier_age=42.0)
    factors = dict.fromkeys(spi_income.SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0)

    result, receipt = spi_income._resample_band_donor_leaves(
        person,
        donor,
        household=household,
        spi_people=pd.Series([True]),
        uprating_factors=factors,
        lower_bounds=(200_000,),
        regional_pool_minimum=5,
        seed=seed,
        age_pool_minimum=10,
    )

    # The carrier is 42, so it never inherits a 75-and-over record's State
    # Pension leaf.
    assert result.loc[0, spi_income.SPI_HMRC_STATE_PENSION_INCOME_COLUMN] == 0.0
    assert receipt["bands"][0]["age_matched_carriers"] == 1
    assert receipt["age_pool_minimum"] == 10


def test_band_carriers_fall_back_to_the_band_pool_when_the_age_pool_is_thin() -> None:
    person, donor, household = _band_resample_inputs(carrier_age=42.0)
    factors = dict.fromkeys(spi_income.SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0)

    _, receipt = spi_income._resample_band_donor_leaves(
        person,
        donor,
        household=household,
        spi_people=pd.Series([True]),
        uprating_factors=factors,
        lower_bounds=(200_000,),
        regional_pool_minimum=5,
        seed=0,
        age_pool_minimum=31,
    )

    assert receipt["bands"][0]["age_matched_carriers"] == 0


def test_band_resample_without_age_pooling_keeps_the_legacy_pools() -> None:
    person, donor, household = _band_resample_inputs(carrier_age=42.0)
    factors = dict.fromkeys(spi_income.SPI_INCOME_QRF_OUTPUT_COLUMNS, 1.0)

    _, receipt = spi_income._resample_band_donor_leaves(
        person,
        donor.drop(columns=["agerange"]),
        household=household,
        spi_people=pd.Series([True]),
        uprating_factors=factors,
        lower_bounds=(200_000,),
        regional_pool_minimum=5,
        seed=0,
    )

    assert receipt["age_pool_minimum"] is None
    assert receipt["bands"][0]["age_matched_carriers"] == 0
