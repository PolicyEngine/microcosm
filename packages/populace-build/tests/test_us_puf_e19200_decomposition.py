from __future__ import annotations

import numpy as np
import pytest

from populace.build.us_runtime.puf_interest_components import (
    US_PUF_E19200_AGI_BANDS,
    US_PUF_E19200_ALL_RETURNS_COMPONENTS,
    split_us_puf_e19200_by_agi_band,
)
from populace.build.us_runtime.puf_source_agi import (
    source_year_puf_adjusted_gross_income,
)
from populace.build.us_runtime.puf_support import puf_tax_unit_donor_from_arrays


def _representative_agi(lower: float | None, upper: float | None) -> float:
    if lower is None:
        return -10_000.0
    if upper is None:
        return lower + 1_000_000.0
    return (lower + upper) / 2


def _one_record_per_band_arrays() -> tuple[dict[str, list[object]], np.ndarray]:
    tax_unit_ids = np.arange(1, len(US_PUF_E19200_AGI_BANDS) + 1, dtype=np.int64)
    total_interest = np.full(len(tax_unit_ids), 1_000.0, dtype=np.float64)
    adjusted_gross_income = np.asarray(
        [
            _representative_agi(band.lower_bound, band.upper_bound)
            for band in US_PUF_E19200_AGI_BANDS
        ],
        dtype=np.float64,
    )
    arrays: dict[str, list[object]] = {
        "tax_unit_id": tax_unit_ids.tolist(),
        "household_weight": np.ones(len(tax_unit_ids)).tolist(),
        "filing_status": [b"SINGLE"] * len(tax_unit_ids),
        "person_tax_unit_id": tax_unit_ids.tolist(),
        "home_mortgage_interest": total_interest.tolist(),
        # The processed PUF leaf is all-zero before this decomposition. The
        # split must replace it from E19200, not preserve or add this sentinel.
        "investment_interest_expense": np.zeros(len(tax_unit_ids)).tolist(),
    }
    return arrays, adjusted_gross_income


def _raw_puf_source_fixture() -> tuple[dict[str, list[float]], np.ndarray, np.ndarray]:
    regular_ids = np.arange(1, 31, dtype=np.int64)
    regular_agi = np.asarray(
        [
            4_000.0,
            *np.linspace(-500_000.0, -10_000.0, 9),
            *np.linspace(10_000.0, 9_000_000.0, 10),
            *np.linspace(10_000_000.0, 200_000_000.0, 10),
        ],
        dtype=np.float64,
    )
    aggregate_ids = np.asarray([999_996, 999_997, 999_998, 999_999])
    aggregate_agi = np.asarray(
        [-100_000.0, 2_000_000.0, 20_000_000.0, 200_000_000.0]
    )
    recid = np.concatenate((regular_ids, aggregate_ids))
    agi = np.concatenate((regular_agi, aggregate_agi))
    source: dict[str, list[float]] = {
        "RECID": recid.tolist(),
        "MARS": np.concatenate((np.ones(len(regular_ids)), np.zeros(4))).tolist(),
        "S006": np.concatenate(
            (
                np.full(len(regular_ids), 100.0),
                np.asarray([14_000.0, 23_000.0, 39_000.0, 10_000.0]),
            )
        ).tolist(),
        "E00100": agi.tolist(),
    }
    screened_fields = (
        "E00200",
        "P23250",
        "P22250",
        "E00650",
        "E00300",
        "E26270",
        "E00900",
        "E02100",
        "E00400",
        "E00600",
    )
    for offset, field in enumerate(screened_fields, start=1):
        source[field] = (np.abs(agi) / offset + offset).tolist()

    synthetic_ids = np.arange(1_000_000, 1_000_080, dtype=np.int64)
    processed_ids = np.concatenate((regular_ids, synthetic_ids))
    processed_weights = np.concatenate(
        (
            np.ones(len(regular_ids)),
            np.full(20, 7.0),
            np.full(23, 10.0),
            np.full(39, 10.0),
            np.full(18, 100.0 / 18.0),
        )
    )
    return source, processed_ids, processed_weights


def test_ty2015_e19200_component_rows_are_complete_cited_and_conservative() -> None:
    bands = US_PUF_E19200_AGI_BANDS

    assert len(bands) == 22
    assert [band.source_row for band in bands] == list(range(11, 33))
    assert bands[0].label == "Under $5,000"
    assert bands[0].lower_bound is None
    assert bands[0].upper_bound == 5_000
    assert bands[-1].lower_bound == 10_000_000
    assert bands[-1].upper_bound is None
    for previous, following in zip(bands[:-1], bands[1:], strict=True):
        assert previous.upper_bound == following.lower_bound
    for band in bands:
        assert band.source_cells == {
            "total_interest_paid_amount": f"CF{band.source_row}",
            "home_mortgage_interest_amount": f"CH{band.source_row}",
            "deductible_points_amount": f"CN{band.source_row}",
            "qualified_mortgage_insurance_premiums_amount": (f"CP{band.source_row}"),
            "investment_interest_amount": f"CR{band.source_row}",
        }
        published_components = (
            band.home_mortgage_interest_amount
            + band.deductible_points_amount
            + band.qualified_mortgage_insurance_premiums_amount
            + band.investment_interest_amount
        )
        assert abs(band.total_interest_paid_amount - published_components) <= 1

    assert sum(band.total_interest_paid_amount for band in bands) == 304_461_163
    assert sum(band.home_mortgage_interest_amount for band in bands) == 283_004_467
    assert sum(band.non_mortgage_interest_amount for band in bands) == 21_456_696
    assert sum(band.investment_interest_amount for band in bands) == 13_895_493

    all_returns = US_PUF_E19200_ALL_RETURNS_COMPONENTS
    assert all_returns.source_row == 10
    assert all_returns.total_interest_paid_amount == 304_461_163
    assert all_returns.home_mortgage_interest_amount == 283_004_465
    assert all_returns.deductible_points_amount == 1_273_716
    assert all_returns.qualified_mortgage_insurance_premiums_amount == 6_287_486
    assert all_returns.investment_interest_amount == 13_895_495


def test_e19200_first_band_includes_negative_zero_and_below_5000_agi() -> None:
    values = np.ones(4, dtype=np.float64)
    mortgage, non_mortgage = split_us_puf_e19200_by_agi_band(
        values,
        np.asarray([-100_000.0, 0.0, 4_999.99, 5_000.0]),
    )
    first_share = (
        US_PUF_E19200_AGI_BANDS[0].home_mortgage_interest_amount
        / US_PUF_E19200_AGI_BANDS[0].total_interest_paid_amount
    )
    second_share = (
        US_PUF_E19200_AGI_BANDS[1].home_mortgage_interest_amount
        / US_PUF_E19200_AGI_BANDS[1].total_interest_paid_amount
    )

    np.testing.assert_array_equal(
        mortgage[:3].view(np.uint64),
        np.full(3, first_share, dtype=np.float64).view(np.uint64),
    )
    assert mortgage[3] == second_share
    np.testing.assert_array_equal(mortgage + non_mortgage, values)


def test_e19200_donor_split_preserves_each_band_and_published_shares() -> None:
    arrays, adjusted_gross_income = _one_record_per_band_arrays()

    donor = puf_tax_unit_donor_from_arrays(
        arrays,
        adjusted_gross_income=adjusted_gross_income,
        person_outputs=(
            "home_mortgage_interest",
            "investment_interest_expense",
        ),
        tax_unit_outputs=(),
    )

    total = np.asarray(
        [band.total_interest_paid_amount for band in US_PUF_E19200_AGI_BANDS],
        dtype=np.float64,
    )
    expected_mortgage = np.asarray(
        [1_000.0 * band.home_mortgage_share for band in US_PUF_E19200_AGI_BANDS],
        dtype=np.float64,
    )
    expected_non_mortgage = 1_000.0 - expected_mortgage
    np.testing.assert_allclose(donor["home_mortgage_interest"], expected_mortgage)
    np.testing.assert_allclose(
        donor["investment_interest_expense"],
        expected_non_mortgage,
    )
    np.testing.assert_array_equal(
        donor["home_mortgage_interest"].to_numpy()
        + donor["investment_interest_expense"].to_numpy(),
        np.full(len(total), 1_000.0),
    )
    assert (donor["investment_interest_expense"] > 0).all()

    # The same proportional rule applied to the literal source rows recovers
    # the published mortgage amounts and the full conserving residual mass.
    source_mortgage, source_non_mortgage = split_us_puf_e19200_by_agi_band(
        total,
        adjusted_gross_income,
    )
    np.testing.assert_allclose(
        source_mortgage,
        np.asarray(
            [band.home_mortgage_interest_amount for band in US_PUF_E19200_AGI_BANDS],
            dtype=np.float64,
        ),
    )
    assert source_non_mortgage.sum() == 21_456_696


def test_e19200_donor_split_requires_explicit_adjusted_gross_income() -> None:
    arrays = {
        "tax_unit_id": [1],
        "household_weight": [1.0],
        "filing_status": [b"SINGLE"],
        "person_tax_unit_id": [1],
        "home_mortgage_interest": [100.0],
        "investment_interest_expense": [0.0],
        # This predictor is deliberately tempting but is not AGI and cannot
        # silently select a source-table band.
        "employment_income": [250_000.0],
    }

    with pytest.raises(ValueError, match="adjusted_gross_income"):
        puf_tax_unit_donor_from_arrays(
            arrays,
            person_outputs=(
                "home_mortgage_interest",
                "investment_interest_expense",
            ),
            tax_unit_outputs=(),
        )


def test_e19200_donor_split_is_bit_deterministic_and_order_invariant() -> None:
    arrays, adjusted_gross_income = _one_record_per_band_arrays()
    first = puf_tax_unit_donor_from_arrays(
        arrays,
        adjusted_gross_income=adjusted_gross_income,
        person_outputs=(
            "home_mortgage_interest",
            "investment_interest_expense",
        ),
        tax_unit_outputs=(),
    )
    second = puf_tax_unit_donor_from_arrays(
        arrays,
        adjusted_gross_income=adjusted_gross_income,
        person_outputs=(
            "home_mortgage_interest",
            "investment_interest_expense",
        ),
        tax_unit_outputs=(),
    )

    permutation = np.arange(len(adjusted_gross_income) - 1, -1, -1)
    shuffled_arrays = {
        name: np.asarray(values)[permutation].tolist()
        for name, values in arrays.items()
    }
    shuffled = puf_tax_unit_donor_from_arrays(
        shuffled_arrays,
        adjusted_gross_income=adjusted_gross_income[permutation],
        person_outputs=(
            "home_mortgage_interest",
            "investment_interest_expense",
        ),
        tax_unit_outputs=(),
    ).sort_values("tax_unit_id")

    columns = ("home_mortgage_interest", "investment_interest_expense")
    for column in columns:
        np.testing.assert_array_equal(
            first[column].to_numpy().view(np.uint64),
            second[column].to_numpy().view(np.uint64),
        )
        np.testing.assert_array_equal(
            first[column].to_numpy().view(np.uint64),
            shuffled[column].to_numpy().view(np.uint64),
        )


def test_source_year_puf_agi_aligns_regular_and_synthetic_records() -> None:
    source, processed_ids, processed_weights = _raw_puf_source_fixture()

    adjusted_gross_income = source_year_puf_adjusted_gross_income(
        source,
        processed_tax_unit_ids=processed_ids,
        processed_tax_unit_weights=processed_weights,
    )

    # The source record remains in its literal TY2015 band. A target-year
    # formula or nominal uprating could move this $4,000 return into a later
    # band, which is precisely what this source-aligned seam prevents.
    assert adjusted_gross_income[0] == 4_000.0
    np.testing.assert_array_equal(
        adjusted_gross_income[:30],
        np.asarray(source["E00100"][:30]),
    )
    assert (adjusted_gross_income[30:50] < 5_000.0).all()
    assert (
        (adjusted_gross_income[50:73] >= 0.0)
        & (adjusted_gross_income[50:73] < 10_000_000.0)
    ).all()
    assert (adjusted_gross_income[73:] >= 10_000_000.0).all()


def test_source_year_puf_agi_is_bit_deterministic() -> None:
    source, processed_ids, processed_weights = _raw_puf_source_fixture()

    first = source_year_puf_adjusted_gross_income(
        source,
        processed_tax_unit_ids=processed_ids,
        processed_tax_unit_weights=processed_weights,
    )
    second = source_year_puf_adjusted_gross_income(
        source,
        processed_tax_unit_ids=processed_ids,
        processed_tax_unit_weights=processed_weights,
    )

    np.testing.assert_array_equal(first.view(np.uint64), second.view(np.uint64))


def test_source_year_puf_agi_fails_closed_on_processed_id_drift() -> None:
    source, processed_ids, processed_weights = _raw_puf_source_fixture()
    processed_ids = processed_ids.copy()
    processed_ids[3] = 123_456

    with pytest.raises(ValueError, match="regular RECID order"):
        source_year_puf_adjusted_gross_income(
            source,
            processed_tax_unit_ids=processed_ids,
            processed_tax_unit_weights=processed_weights,
        )
