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
        "banding_factor",
        "band_factors",
        "below_first_band_factor",
        "bands_without_tape_amounts",
        "bands_too_thin_to_gross",
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
    assert receipt["factors_over"].startswith("taxpayers")
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
    assert plain["factors_over"] == "all_records_at_or_above_the_first_band"


def test_a_tape_that_does_not_gross_to_the_population_is_left_as_it_is() -> None:
    # A synthetic tape's weights do not gross to the published amounts, and its
    # factors would run to millions; the rebase leaves such a column as it is.
    donor = _donor()
    donor["FACT"] = donor["FACT"] / 1e6
    rebased, receipt = rebase_spi_donor_savings(donor, PUBLISHED)
    for column in PUBLISHED:
        assert "skipped" in receipt["columns"][column]
        pd.testing.assert_series_equal(rebased[column], donor[column])


def test_gift_leaves_are_anchored_to_the_tapes_weighted_totals() -> None:
    # Two copies drawn, one FRS row and one undrawn copy; the tape's
    # FACT-weighted totals are 3,000 of Gift Aid and 600 of investment gifts.
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_household_id": [10, 20, 30, 40],
            "gift_aid": [100.0, 300.0, 50.0, 70.0],
            "charitable_investment_gifts": [0.0, 400.0, 0.0, 9.0],
        }
    )
    household = pd.DataFrame(
        {"household_id": [10, 20, 30, 40], "household_weight": [2.0, 3.0, 5.0, 7.0]}
    )
    donor = pd.DataFrame(
        {
            "FACT": [10.0, 20.0],
            "gift_aid": [100.0, 100.0],
            "charitable_investment_gifts": [0.0, 30.0],
        }
    )
    drawn = pd.Series([True, True, False, False])
    anchored, receipt = spi_income._anchor_spi_gifts_to_tape(
        person.copy(), household, donor, spi_people=drawn
    )
    weights = pd.Series([2.0, 3.0])
    # The drawn copies carried 1,100 of Gift Aid and 1,200 of investment gifts.
    assert (weights * anchored.loc[:1, "gift_aid"]).sum() == pytest.approx(3_000.0)
    assert (weights * anchored.loc[:1, "charitable_investment_gifts"]).sum() == (
        pytest.approx(600.0)
    )
    assert receipt["factors"] == pytest.approx(
        {"gift_aid": 3_000.0 / 1_100.0, "charitable_investment_gifts": 0.5}
    )
    # Rows the stage did not draw keep their values.
    assert anchored.loc[2:, "gift_aid"].tolist() == [50.0, 70.0]
    assert anchored.loc[2:, "charitable_investment_gifts"].tolist() == [0.0, 9.0]
    # A column the tape carries none of is left as drawn.
    _, receipt = spi_income._anchor_spi_gifts_to_tape(
        person.copy(),
        household,
        donor.assign(charitable_investment_gifts=0.0),
        spi_people=drawn,
    )
    assert receipt["factors"]["charitable_investment_gifts"] == 1.0


def test_drawn_rows_drop_their_twins_annuity_and_trust_descriptor() -> None:
    # A drawn row's private pension is the tape's, which books annuity and
    # trust payments as other income; an undrawn copy keeps its twin's part
    # (microcosm#1428).
    person = pd.DataFrame({"annuity_and_trust_income": [500.0, 700.0, 0.0]})
    drawn = pd.Series([True, False, False])
    cleared = spi_income._clear_annuity_and_trust_descriptor(
        person.copy(), spi_people=drawn
    )
    assert cleared["annuity_and_trust_income"].tolist() == [0.0, 700.0, 0.0]
    # A frame without the descriptor is left as it is.
    bare = pd.DataFrame({"private_pension_income": [1.0]})
    assert spi_income._clear_annuity_and_trust_descriptor(
        bare.copy(), spi_people=pd.Series([True])
    ).equals(bare)


INTEREST_ONLY = {"savings_interest_income": {12_570: 100.0, 50_000: 400.0}}


def _edge_donor() -> pd.DataFrame:
    # A just below the £50,000 edge in 2022-23, B well above it.
    return pd.DataFrame(
        {
            "total_income": [49_950.0, 60_000.0],
            "FACT": [100.0, 100.0],
            "savings_interest_income": [1.0, 1.0],
            "dividend_income": [5_000.0, 0.0],
        }
    )


def test_the_tape_is_banded_on_its_income_moved_to_2023_24() -> None:
    # Banded on its 2022-23 income, A sits in the £12,570 band. Moved to
    # 2023-24 (dividends x1.1, interest at all bands' 500 over 200), it crosses
    # the £50,000 edge, the band the consumers will find it in after uprating,
    # so its migration is counted once (microcosm#1428 review, finding 6).
    held, held_receipt = rebase_spi_donor_savings(_edge_donor(), INTEREST_ONLY)
    assert held["savings_interest_income"].tolist() == pytest.approx([1.0, 4.0])
    moved, receipt = rebase_spi_donor_savings(
        _edge_donor(), INTEREST_ONLY, income_factors={"dividend_income": 1.1}
    )
    assert moved["savings_interest_income"].tolist() == pytest.approx([2.0, 2.0])
    assert receipt["columns"]["savings_interest_income"]["band_factors"] == {
        "50000": pytest.approx(2.0)
    }
    assert receipt["band_by"] == "total_income_moved_to_2023_24"
    assert receipt["columns"]["savings_interest_income"][
        "banding_factor"
    ] == pytest.approx(2.5)


def test_savings_move_by_their_factor_over_all_bands_for_the_banding() -> None:
    # A's interest is a large part of its income. All bands' factor (3,000
    # over the tape's 600) moves it from 500 to 2,500 for the banding, which
    # takes A across the £50,000 edge; there the band's factor is 2,000 over
    # 600 for both records, and A's rebased income stays above the edge. One
    # pass: a record's own band's factor would set the band it is banded in
    # (microcosm#1428 review, finding 6).
    donor = pd.DataFrame(
        {
            "total_income": [49_000.0, 60_000.0],
            "FACT": [1.0, 1.0],
            "savings_interest_income": [500.0, 100.0],
        }
    )
    rebased, receipt = rebase_spi_donor_savings(
        donor, {"savings_interest_income": {12_570: 1_000.0, 50_000: 2_000.0}}
    )
    interest = receipt["columns"]["savings_interest_income"]
    assert interest["banding_factor"] == pytest.approx(5.0)
    assert interest["band_factors"] == {"50000": pytest.approx(2_000.0 / 600.0)}
    assert interest["bands_without_tape_amounts"] == [12_570]
    assert rebased["savings_interest_income"].tolist() == pytest.approx(
        [500.0 * 2_000.0 / 600.0, 100.0 * 2_000.0 / 600.0]
    )
    assert 48_500.0 + rebased.loc[0, "savings_interest_income"] >= 50_000.0


def test_a_saver_who_becomes_a_taxpayer_counts_once() -> None:
    # C pays no tax in 2022-23 and crosses the Personal Allowance by 2023-24;
    # D is a taxpayer throughout. Both are 2023-24 taxpayers in the £12,570
    # band, so their 2,000 meets its 4,000 at a factor of 2. Counting only D
    # (factor 4) and giving C the all-bands factor would count C twice
    # (microcosm#1428 review, finding 10).
    donor = pd.DataFrame(
        {
            "total_income": [12_000.0, 20_000.0],
            "FACT": [10.0, 10.0],
            "savings_interest_income": [100.0, 100.0],
            "dividend_income": [1_000.0, 0.0],
            spi_income.SPI_DONOR_TAXPAYER_COLUMN: [False, True],
        }
    )
    rebased, receipt = rebase_spi_donor_savings(
        donor,
        {"savings_interest_income": {12_570: 4_000.0, 50_000: 1.0}},
        income_factors={"dividend_income": 1.6},
    )
    assert receipt["columns"]["savings_interest_income"]["band_factors"][
        "12570"
    ] == pytest.approx(2.0)
    weighted = rebased["savings_interest_income"] * rebased["FACT"]
    assert weighted.sum() == pytest.approx(4_000.0)


def test_a_flagged_taxpayer_below_the_first_band_sits_in_it() -> None:
    # Table 3.7's lowest band holds taxpayers below the Personal Allowance
    # (note 8): a flagged taxpayer there shares the band's factor.
    donor = pd.DataFrame(
        {
            "total_income": [8_000.0, 20_000.0],
            "FACT": [10.0, 10.0],
            "savings_interest_income": [100.0, 100.0],
            spi_income.SPI_DONOR_TAXPAYER_COLUMN: [True, True],
        }
    )
    rebased, _ = rebase_spi_donor_savings(
        donor, {"savings_interest_income": {12_570: 6_000.0, 50_000: 1.0}}
    )
    assert rebased["savings_interest_income"].tolist() == pytest.approx([300.0, 300.0])


def test_a_band_too_thin_to_gross_takes_the_all_bands_factor() -> None:
    # The £50,000 band's tape holds 10 against a published 1,000, under a
    # tenth: its factor would be 100, so it takes the all-bands factor and is
    # named (microcosm#1428 review, finding 10).
    donor = pd.DataFrame(
        {
            "total_income": [20_000.0, 60_000.0],
            "FACT": [10.0, 1.0],
            "savings_interest_income": [100.0, 10.0],
        }
    )
    rebased, receipt = rebase_spi_donor_savings(
        donor, {"savings_interest_income": {12_570: 2_000.0, 50_000: 1_000.0}}
    )
    interest = receipt["columns"]["savings_interest_income"]
    assert interest["bands_too_thin_to_gross"] == [50_000]
    assert "50000" not in interest["band_factors"]
    assert rebased.loc[1, "savings_interest_income"] == pytest.approx(
        10.0 * interest["below_first_band_factor"]
    )


def test_a_skipped_column_keeps_its_engine_index_from_2022_23(monkeypatch) -> None:
    variables = {
        name: SimpleNamespace(uprating="index")
        for name in spi_income.SPI_INCOME_UPRATING_VARIABLES.values()
        if name is not None
    }

    def parameters(period):
        return SimpleNamespace(index=100.0 if period.startswith("2022") else 120.0)

    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(
            CountryTaxBenefitSystem=lambda: SimpleNamespace(
                variables=variables, parameters=parameters
            )
        ),
    )
    factors, receipt = spi_income._spi_income_uprating_factors.__wrapped__(
        2024, rebased_columns=frozenset({"other_investment_income"})
    )
    # Interest the rebase left as it was grows on its engine index from
    # 2022-23, not on the OBR's path from 2023-24 (microcosm#1428 review).
    assert receipt["columns"]["savings_interest_income"]["basis"] == "model_index"
    assert factors["savings_interest_income"] == pytest.approx(1.2)
    assert receipt["columns"]["other_investment_income"]["basis"] == (
        "spi_published_bands_then_obr_savings_income"
    )


def test_the_tape_taxpayer_flag_is_validated(tmp_path) -> None:
    from test_support.microcosm_build.uk_spi_income import _write_donor

    path = tmp_path / "donor.tab"
    _write_donor(path)
    raw = pd.read_csv(path, delimiter="\t")
    # Without the flag the prepared tape carries none.
    assert spi_income.SPI_DONOR_TAXPAYER_COLUMN not in (
        spi_income.prepare_spi_donor_table(raw, seed=0)
    )
    flagged = raw.assign(TAXPAYER=[1, 2, 1, 2])
    prepared = spi_income.prepare_spi_donor_table(flagged, seed=0)
    assert prepared[spi_income.SPI_DONOR_TAXPAYER_COLUMN].tolist() == [
        True,
        False,
        True,
        False,
    ]
    with pytest.raises(ValueError, match="TAXPAYER"):
        spi_income.prepare_spi_donor_table(raw.assign(TAXPAYER=[1, 0, 1, 2]), seed=0)
