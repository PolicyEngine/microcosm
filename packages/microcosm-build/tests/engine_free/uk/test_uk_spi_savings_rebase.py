"""The SPI donors' savings rebase (microcosm#1418 savings follow-up).

The 2022-23 tape's interest and other income move to HMRC's published
2023-24 Table 3.7 amounts band by band, then on the OBR's savings income
growth to the frame's year, in place of each column's engine index.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.uk_runtime import spi_income
from microcosm.build.uk_runtime.hmrc_uprating import uk_obr_savings_income_level
from microcosm.build.uk_runtime.spi_income import (
    SPI_HISTORY_INDEX_DIFFERS_FROM_PROJECTION,
    SPI_SAVINGS_REBASE_CONCEPTS,
    SPI_SAVINGS_REBASE_YEAR,
    rebase_spi_donor_savings,
    spi_savings_rebase_published,
)

PUBLISHED = {
    "savings_interest_income": {12_570: 1_500.0, 50_000: 2_000.0, 1_000_000: 3_000.0},
    "other_investment_income": {12_570: 400.0, 50_000: 400.0, 1_000_000: 18_000.0},
}


def _rows(amounts):
    return [
        {
            "concept": SPI_SAVINGS_REBASE_CONCEPTS[column],
            "period": {"type": "tax_year", "value": SPI_SAVINGS_REBASE_YEAR},
            "dimensions": {"total_income_lower_bound": bound},
            "value": value,
        }
        for column, by_bound in amounts.items()
        for bound, value in by_bound.items()
    ]


def _donor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "total_income": [5_000.0, 15_000.0, 15_000.0, 50_000.0, 2_000_000.0],
            "FACT": [10.0, 5.0, 5.0, 2.0, 1.0],
            "savings_interest_income": [100.0, 40.0, 60.0, 500.0, 1_000.0],
            "other_investment_income": [0.0, 10.0, 30.0, 100.0, 9_000.0],
            "dividend_income": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_the_rebased_tape_reproduces_the_published_bands() -> None:
    donor = _donor()
    rebased, receipt = rebase_spi_donor_savings(donor, PUBLISHED)

    # A donor at a band's lower bound sits in that band.
    rows_by_band = {12_570: [1, 2], 50_000: [3], 1_000_000: [4]}
    for column, by_bound in PUBLISHED.items():
        for bound, rows in rows_by_band.items():
            weighted = rebased.loc[rows, column] * rebased.loc[rows, "FACT"]
            assert weighted.sum() == pytest.approx(by_bound[bound])
    # Interest: the tape's 500, 1,000 and 1,000 against 1,500, 2,000 and
    # 3,000; a donor below the first band takes all bands' 6,500 over 2,500.
    assert receipt["columns"]["savings_interest_income"]["band_factors"] == {
        "12570": 3.0,
        "50000": 2.0,
        "1000000": 3.0,
    }
    assert rebased.loc[0, "savings_interest_income"] == pytest.approx(100.0 * 2.6)
    assert receipt["columns"]["savings_interest_income"][
        "below_first_band_factor"
    ] == pytest.approx(2.6)
    # Other columns, and the tape passed in, are left as they were.
    pd.testing.assert_series_equal(rebased["dividend_income"], donor["dividend_income"])
    assert donor.loc[0, "savings_interest_income"] == 100.0
    assert receipt["from_period"] == 2022 and receipt["to_period"] == 2023
    # The receipt carries factors, never a count of tape records.
    assert set(receipt["columns"]["other_investment_income"]) == {
        "concept",
        "band_factors",
        "below_first_band_factor",
        "bands_without_tape_amounts",
    }
    assert (
        receipt["columns"]["other_investment_income"]["bands_without_tape_amounts"]
        == []
    )


def test_a_band_the_tape_does_not_carry_is_left_as_it_is() -> None:
    # A synthetic tape need not carry every band; there is nothing to scale
    # there, and the receipt names the band.
    donor = _donor()
    donor.loc[4, "other_investment_income"] = 0.0
    rebased, receipt = rebase_spi_donor_savings(donor, PUBLISHED)
    other = receipt["columns"]["other_investment_income"]
    assert other["bands_without_tape_amounts"] == [1_000_000]
    assert "1000000" not in other["band_factors"]
    assert rebased.loc[4, "other_investment_income"] == 0.0
    # The other bands, and below the first, still rebase: 800 over 400.
    assert other["below_first_band_factor"] == pytest.approx(2.0)
    assert rebased.loc[3, "other_investment_income"] == pytest.approx(200.0)


def test_the_published_rows_are_checked() -> None:
    rows = _rows(PUBLISHED)
    assert spi_savings_rebase_published(rows) == PUBLISHED
    stale = [dict(rows[0], period={"type": "tax_year", "value": 2022}), *rows[1:]]
    with pytest.raises(ValueError, match="describes 2022"):
        spi_savings_rebase_published(stale)
    with pytest.raises(ValueError, match="two rows from"):
        spi_savings_rebase_published([*rows, rows[0]])
    uneven = [
        row
        for row in rows
        if not (
            row["concept"] == SPI_SAVINGS_REBASE_CONCEPTS["other_investment_income"]
            and row["dimensions"]["total_income_lower_bound"] == 50_000
        )
    ]
    with pytest.raises(ValueError, match="same total-income bands"):
        spi_savings_rebase_published(uneven)


def test_the_committed_rows_give_both_columns_table_3_7s_thirteen_bands() -> None:
    published = spi_savings_rebase_published()
    assert {column: len(bands) for column, bands in published.items()} == {
        "savings_interest_income": 13,
        "other_investment_income": 13,
    }
    assert min(published["savings_interest_income"]) == 12_570
    # SPI 2023-24 Table 3.7, all bands: GBP 18.34bn of interest and GBP
    # 5.85bn of other income.
    assert sum(published["savings_interest_income"].values()) == pytest.approx(
        18.343e9, rel=1e-3
    )
    assert sum(published["other_investment_income"].values()) == pytest.approx(
        5.846e9, rel=1e-3
    )


def test_the_savings_columns_grow_on_the_obr_path_from_2023_24(monkeypatch) -> None:
    variables = {
        name: SimpleNamespace(uprating="index")
        for name in spi_income.SPI_INCOME_UPRATING_VARIABLES.values()
        if name is not None
    }

    def parameters(period):
        return SimpleNamespace(index=100.0 if period.startswith("2022") else 120.0)

    system = SimpleNamespace(variables=variables, parameters=parameters)
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(CountryTaxBenefitSystem=lambda: system),
    )
    factors, receipt = spi_income._spi_income_uprating_factors.__wrapped__(2024)

    level, used = uk_obr_savings_income_level(2023, 2024)
    # The OBR's +35.0% for 2024-25 (October 2026 EFO, Table 4.5).
    assert level == pytest.approx(1.3496108, abs=1e-7)
    assert sorted(used) == [2024]
    for column in SPI_SAVINGS_REBASE_CONCEPTS:
        row = receipt["columns"][column]
        assert row["basis"] == "spi_published_bands_then_obr_savings_income"
        assert row["from_period"] == SPI_SAVINGS_REBASE_YEAR
        assert factors[column] == pytest.approx(level)
        assert (
            row["dataset_projection_index"]
            == (SPI_HISTORY_INDEX_DIFFERS_FROM_PROJECTION[column])
        )
    # Every other indexed column keeps its engine index from the tape's year.
    assert factors["dividend_income"] == pytest.approx(1.2)
    # Before 2023-24 there is nothing to rebase to: the source year is the
    # identity, and the path refuses to run backwards.
    unchanged, _ = spi_income._spi_income_uprating_factors.__wrapped__(2022)
    assert unchanged["other_investment_income"] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="runs forward only"):
        uk_obr_savings_income_level(2024, 2023)


def test_the_factors_read_the_tapes_taxpayers() -> None:
    # Table 3.7 counts taxpayers only, so a band's factor is its published
    # amount over the tape's taxpayers' amount there; a donor who pays no tax
    # takes the factor of all bands together (option F of the microcosm#1428
    # other-income investigation).
    donor = _donor()
    donor[spi_income.SPI_DONOR_TAXPAYER_COLUMN] = [False, True, False, True, True]
    rebased, receipt = rebase_spi_donor_savings(donor, PUBLISHED)
    interest = receipt["columns"]["savings_interest_income"]
    assert receipt["factors_over"] == "taxpayers"
    # The £12,570 band's taxpayers hold 5 x 40 = 200 of the tape's interest
    # there: 1,500 over 200. All bands: 6,500 over 2,200.
    assert interest["band_factors"]["12570"] == pytest.approx(7.5)
    assert rebased.loc[1, "savings_interest_income"] == pytest.approx(40.0 * 7.5)
    all_bands = 6_500.0 / 2_200.0
    assert interest["below_first_band_factor"] == pytest.approx(all_bands)
    assert rebased.loc[2, "savings_interest_income"] == pytest.approx(60.0 * all_bands)
    # The band's taxpayers reproduce its published amount.
    weighted = rebased.loc[[1], "savings_interest_income"] * rebased.loc[[1], "FACT"]
    assert weighted.sum() == pytest.approx(1_500.0)
    # Without the flag every record counts, as before.
    _, plain = rebase_spi_donor_savings(_donor(), PUBLISHED)
    assert plain["factors_over"] == "all_records"
