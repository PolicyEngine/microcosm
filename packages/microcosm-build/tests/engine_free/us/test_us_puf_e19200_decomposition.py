from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime.puf_interest_components import (
    US_PUF_E19200_AGI_BANDS,
    US_PUF_E19200_ALL_RETURNS_COMPONENTS,
    split_us_puf_e19200_by_agi_band,
)
from microcosm.build.us_runtime.puf_source_agi import (
    source_year_puf_adjusted_gross_income,
)
from microcosm.build.us_runtime.puf_support import puf_tax_unit_donor_from_arrays

_REAL_PUF_STORAGE = (
    Path.home()
    / "PolicyEngine"
    / ("policyengine" + "-us-data")
    / ("policyengine" + "_us_data")
    / "storage"
)
_REAL_PROCESSED_PUF = _REAL_PUF_STORAGE / "puf_2024.h5"
_REAL_SOURCE_PUF = _REAL_PUF_STORAGE / "puf_2015.csv"
_REAL_PROCESSED_PUF_SHA256 = (
    "7669f5b5281f20080e77204f9bd4aabfad0aa101fa283e22caf9ba8d61d4d6df"
)
_REAL_SOURCE_PUF_SHA256 = (
    "0a7fd643edb1acc55c507db795914b41d232922be78c149b58d111f4672499df"
)
_REAL_SOURCE_AGI_VECTOR_SHA256 = (
    "8b2b6c206e8e9f7b80dcfd86554962b0966b25ad8c47eab7d9db4f27f69ebc0d"
)


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
    aggregate_agi = np.asarray([-100_000.0, 2_000_000.0, 20_000_000.0, 200_000_000.0])
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

    synthetic_ids = np.arange(1_000_000, 1_000_102, dtype=np.int64)
    processed_ids = np.concatenate((regular_ids, synthetic_ids))
    processed_weights = np.concatenate(
        (
            np.ones(len(regular_ids)),
            np.full(20, 7.0),
            np.full(23, 10.0),
            np.full(39, 10.0),
            np.full(20, 5.0),
        )
    )
    return source, processed_ids, processed_weights


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def test_source_year_puf_agi_matches_pinned_flat_array_artifact() -> None:
    if not _REAL_PROCESSED_PUF.exists() or not _REAL_SOURCE_PUF.exists():
        pytest.skip("Pinned restricted PUF artifacts are not available.")
    assert _file_sha256(_REAL_PROCESSED_PUF) == _REAL_PROCESSED_PUF_SHA256
    assert _file_sha256(_REAL_SOURCE_PUF) == _REAL_SOURCE_PUF_SHA256

    import h5py

    with h5py.File(_REAL_PROCESSED_PUF, "r") as h5:
        # This is deliberately the production artifact's flat root-array
        # layout, not a USSingleYearDataset entity-table fixture.
        assert "person" not in h5
        adjusted_gross_income = source_year_puf_adjusted_gross_income(
            _REAL_SOURCE_PUF,
            processed_tax_unit_ids=np.asarray(h5["tax_unit_id"]),
            processed_tax_unit_weights=np.asarray(h5["household_weight"]),
        )

    assert len(adjusted_gross_income) == 211_677
    assert (
        sha256(adjusted_gross_income.astype("<f8").tobytes()).hexdigest()
        == _REAL_SOURCE_AGI_VECTOR_SHA256
    )
    assert (adjusted_gross_income[-3_900:] == 100_000_000.0).all()


def test_split_conserves_bit_exactly_on_adversarial_floats() -> None:
    """PR #561 review finding 1: the naive (total*share, total-mortgage)
    pair fails bit-exact conservation (E19200=1.53 in the top band summed to
    1.5300000000000002). The reconciled split must reproduce the total
    bit-for-bit on the reviewer's reproducer and on an adversarial sweep."""
    import numpy as np

    from microcosm.build.us_runtime.puf_interest_components import (
        split_us_puf_e19200_by_agi_band,
    )

    total = np.asarray([1.53], dtype=np.float64)
    agi = np.asarray([10_000_000.0], dtype=np.float64)
    mortgage, non_mortgage = split_us_puf_e19200_by_agi_band(total, agi)
    assert float(mortgage[0] + non_mortgage[0]) == 1.53

    rng = np.random.default_rng(561)
    totals = np.round(rng.uniform(0.01, 5_000_000.0, size=20_000), 2)
    agis = rng.uniform(-100_000.0, 20_000_000.0, size=20_000)
    mortgage, non_mortgage = split_us_puf_e19200_by_agi_band(totals, agis)
    recon = mortgage + non_mortgage
    exact = recon == totals
    assert exact.all(), (
        f"{(~exact).sum()} of 20,000 adversarial records failed bit-exact "
        "conservation after reconciliation"
    )
    assert (mortgage >= 0.0).all() and (non_mortgage >= 0.0).all()


def test_donor_api_rejects_non_real_agi_through_a_valid_fixture() -> None:
    """PR #561 review findings 2 (rounds 1-2): the donor API must fail
    closed on AGI that is not a finite real number. Round 1's
    ``_numeric_array`` coerced nonnumeric/NaN to 0.0; round 2's
    ``errors="raise"`` was parse-strict but not real-number-strict —
    datetime/timedelta arrays (including NaT) convert to finite epoch
    sentinels, complex drops its imaginary part, and booleans pass, all
    silently routing records to the wrong AGI band. The fixture below is
    proven valid by a positive control, so each rejection can only come
    from the AGI validation itself."""
    import numpy as np
    import pandas as pd
    import pytest as _pytest

    from microcosm.build.us_runtime.puf_support import puf_tax_unit_donor_from_arrays

    arrays = {
        "tax_unit_id": np.asarray([1], dtype="int64"),
        "person_tax_unit_id": np.asarray([1], dtype="int64"),
        "household_weight": np.asarray([1.0]),
        "filing_status": np.asarray(["SINGLE"]),
    }

    def call(agi):
        return puf_tax_unit_donor_from_arrays(
            arrays,
            adjusted_gross_income=agi,
            person_outputs=(),
            tax_unit_outputs=(),
        )

    # Positive control: the fixture succeeds with valid AGI (the band
    # column is reserved for internal processing and is consumed before
    # return), so the rejections below cannot be masked by an unrelated
    # fixture defect.
    donor = call(np.asarray([50_000.0]))
    assert len(donor) == 1
    assert donor["tax_unit_id"].tolist() == [1]
    # Faithful float conversions are accepted (review round 3): Decimal
    # and numeric strings parse strictly to the same real value.
    from decimal import Decimal

    assert len(call([Decimal("50000")])) == 1
    assert len(call(["50000"])) == 1

    with _pytest.raises(ValueError, match="finite"):
        call(np.asarray([np.nan]))
    with _pytest.raises(ValueError, match="finite"):
        call(np.asarray([np.inf]))
    with _pytest.raises(ValueError, match="Unable to parse"):
        call(["not-a-number"])
    # Typed non-real arrays parse to finite numbers under pd.to_numeric and
    # must be rejected by dtype, not by parsing.
    with _pytest.raises(TypeError, match="real-valued"):
        call(pd.Series([pd.NaT]))
    with _pytest.raises(TypeError, match="real-valued"):
        call(np.asarray(["NaT"], dtype="datetime64[ns]"))
    with _pytest.raises(TypeError, match="real-valued"):
        call(np.asarray([np.timedelta64(1, "D")]))
    with _pytest.raises(TypeError, match="real-valued"):
        call(np.asarray([1.0 + 2.0j]))
    with _pytest.raises(TypeError, match="real-valued"):
        call(np.asarray([True]))
    # Object-wrapped and categorical payloads dodge the dtype-kind gate
    # (review round 3): True parses to 1.0 and complex keeps its real
    # part, so the element screen must reject them too.
    with _pytest.raises(TypeError, match="real-valued"):
        call(np.asarray([True], dtype=object))
    with _pytest.raises(TypeError, match="real-valued"):
        call(np.asarray([1.0 + 2.0j], dtype=object))
    with _pytest.raises(TypeError, match="real-valued"):
        call(pd.Categorical([True]))
