"""UK firm taxable profit: HMRC liability bands through the CT schedule."""

from __future__ import annotations

import hashlib
import json
from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_profit import (
    UK_FIRM_PROFIT_CT_SCHEDULE,
    UK_FIRM_PROFIT_FILE,
    UK_FIRM_PROFIT_SEED,
    UK_FIRM_PROFIT_SOURCE_COLUMNS,
    UK_FIRM_PROFIT_STATUS,
    UKFirmProfitBand,
    UKFirmProfitSource,
    assign_uk_firm_taxable_profit,
    read_uk_firm_profit_source,
    uk_corporation_tax_liability,
    uk_corporation_tax_profit,
    uk_firm_profit_bands,
    uk_firm_profit_diagnostics,
    uk_firm_profit_section_liability,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURE = PATHS.tests / "fixtures" / "uk" / "firm_ct_2024"
SECTIONS_FILE = "ct_sic_sections.csv"
ODS_SHA256 = "d1a1ea0e9ae5c8e3045c99d5b1aad242b659d955d9569df4d202feac9b24aa21"


@cache
def _source() -> UKFirmProfitSource:
    return read_uk_firm_profit_source(FIXTURE)


def _small_source() -> UKFirmProfitSource:
    """Three made-up bands, small enough for a unit-weight frame."""

    bands = (
        UKFirmProfitBand("low", 0.0, 5_000.0, 40.0, 40 * 2_000.0),
        UKFirmProfitBand("mid", 5_000.0, 80_000.0, 25.0, 25 * 30_000.0),
        UKFirmProfitBand("top", 80_000.0, np.inf, 5.0, 5 * 400_000.0),
    )
    return UKFirmProfitSource(
        bands=bands,
        zero_companies=100.0,
        positive_companies=70.0,
        positive_liability=sum(band.liability for band in bands),
        all_returns=170.0,
        file="synthetic",
        sha256="",
    )


def _frame(n: int = 160, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    status = np.where(np.arange(n) % 4 == 3, 2, 1)
    status[5] = 3
    status[9] = 7
    return pd.DataFrame(
        {
            "firm_id": np.arange(1, n + 1),
            "firm_weight": np.ones(n),
            "annual_turnover_k": np.round(rng.lognormal(5.0, 1.5, n), 0),
            "sic_code": rng.choice([1, 25, 47, 62, 64, 86], n),
            "employment": rng.integers(1, 50, n),
            "ons_legal_status": status,
        }
    )


def test_fixture_is_well_formed() -> None:
    source = _source()
    frame = pd.read_csv(FIXTURE / UK_FIRM_PROFIT_FILE, dtype=str, keep_default_na=False)
    assert tuple(frame.columns) == UK_FIRM_PROFIT_SOURCE_COLUMNS
    assert len(source.bands) == 13
    assert source.bands[0].lower == 0.0
    assert np.isinf(source.bands[-1].upper)
    for band, following in zip(source.bands, source.bands[1:], strict=False):
        assert band.upper == following.lower
    # Read off Tables 9A and 9B, column 2024-25.
    assert source.zero_companies == 1_676_855
    assert source.positive_companies == 1_637_230
    assert source.all_returns == 3_314_085
    assert source.positive_liability == 87_945e6
    assert (source.bands[-1].companies, source.bands[-1].liability) == (155, 19_248e6)
    # Counts are rounded to 5 and amounts to GBP 1 million, band by band.
    counts = sum(band.companies for band in source.bands)
    assert abs(counts - source.positive_companies) <= 2.5 * len(source.bands)
    assert abs(counts + source.zero_companies - source.all_returns) <= 2.5 * 15
    liability = sum(band.liability for band in source.bands)
    assert abs(liability - source.positive_liability) <= 0.5e6 * len(source.bands)


def test_fixture_provenance() -> None:
    for name in (UK_FIRM_PROFIT_FILE, SECTIONS_FILE):
        record = json.loads(
            (FIXTURE / f"{name}.provenance.json").read_text(encoding="utf-8")
        )
        payload = (FIXTURE / name).read_bytes()
        assert record["file"]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert record["sha256"] == ODS_SHA256
        assert record["accessed"] == "2026-11-22"
        assert record["url"].startswith("https://assets.publishing.service.gov.uk/")
        assert record["landing_page"].startswith("https://www.gov.uk/")
        for key in ("edition", "method", "period"):
            assert record[key]


def test_held_out_sector_table_is_well_formed() -> None:
    frame = pd.read_csv(FIXTURE / SECTIONS_FILE)
    assert list(frame["hmrc_section"]) == [*"ABCDEFGHIJKLMNOPQRSTU", "ALL"]
    sections, total = frame.iloc[:-1], frame.iloc[-1]
    assert total["liability_gbp_million"] == _source().positive_liability / 1e6
    assert total["companies_with_liability"] == _source().positive_companies
    assert abs(sections["liability_gbp_million"].sum() - 87_945) <= 0.5 * 21
    assert abs(sections["companies_with_liability"].sum() - 1_637_230) <= 2.5 * 21


def test_inversion_round_trips_on_the_band_edges() -> None:
    schedule = UK_FIRM_PROFIT_CT_SCHEDULE
    bands = uk_firm_profit_bands(_source())
    edges = np.unique(
        np.concatenate(
            [
                bands["liability_lower_gbp"],
                bands["liability_upper_gbp"][:-1],
                bands["mean_liability_gbp"],
                # The schedule's own kinks, as liabilities.
                [
                    schedule.small_profits_rate * schedule.lower_limit,
                    schedule.main_rate * schedule.upper_limit,
                ],
            ]
        )
    )
    profit = uk_corporation_tax_profit(edges)
    np.testing.assert_allclose(
        uk_corporation_tax_liability(profit), edges, rtol=1e-12, atol=1e-6
    )
    assert np.all(np.diff(profit) > 0)
    assert np.isinf(bands["profit_upper_gbp"].iloc[-1])
    assert (bands["profit_lower_gbp"] < bands["mean_profit_gbp"]).all()
    assert (bands["mean_profit_gbp"] < bands["profit_upper_gbp"]).all()
    # And from the profit side, through the limits.
    profits = np.array([0.0, 1.0, 50_000.0, 50_000.01, 150_000.0, 250_000.0, 1e9])
    np.testing.assert_allclose(
        uk_corporation_tax_profit(uk_corporation_tax_liability(profits)),
        profits,
        rtol=1e-12,
        atol=1e-6,
    )


def test_ct_formula_matches_the_statute() -> None:
    # Small profits rate, marginal relief of 3/200 x (250,000 - profit), main rate.
    np.testing.assert_allclose(
        uk_corporation_tax_liability(np.array([40_000.0, 100_000.0, 300_000.0])),
        [0.19 * 40_000, 0.25 * 100_000 - 0.015 * 150_000, 0.25 * 300_000],
    )
    assert uk_corporation_tax_liability(-5.0) == 0.0


def test_assignment_matches_band_counts_and_totals() -> None:
    firms, source = _frame(), _small_source()
    profit = assign_uk_firm_taxable_profit(firms, source)
    table = uk_firm_profit_diagnostics(firms, source)
    companies = int((firms["ons_legal_status"] == 1).sum())
    assert list(table["band"]) == ["0", "low", "mid", "top"]
    np.testing.assert_allclose(table["weighted_companies"][1:], [40, 25, 5])
    assert table["weighted_companies"][0] == companies - 70
    np.testing.assert_allclose(
        table["weighted_liability_gbp"][1:], table["table_liability_gbp"][1:]
    )
    assert not table["table_mean_used"].any()
    np.testing.assert_allclose(
        table["weighted_profit_gbp"].sum(), (profit * firms["firm_weight"]).sum()
    )
    # Each liability sits inside its band.
    tax = uk_corporation_tax_liability(profit)
    for band, n in zip(source.bands, (40, 25, 5), strict=True):
        inside = (tax > band.lower) & (tax <= band.upper)
        assert inside.sum() == n


def test_profit_rises_with_turnover_within_a_section() -> None:
    firms, source = _frame(), _small_source()
    firms = firms.assign(profit=assign_uk_firm_taxable_profit(firms, source))
    companies = firms[firms["ons_legal_status"] == 1]
    for _, group in companies.groupby("sic_code"):
        ordered = group.sort_values("annual_turnover_k")
        strict = ordered.drop_duplicates("annual_turnover_k", keep=False)
        assert strict["profit"].is_monotonic_increasing


def test_non_companies_get_zero() -> None:
    firms = _frame()
    profit = assign_uk_firm_taxable_profit(firms, _small_source())
    others = firms["ons_legal_status"].to_numpy() != 1
    assert others.any()
    assert np.all(profit[others] == 0.0)
    assert np.all(profit >= 0.0)
    assert (profit[~others] > 0).sum() == 70


def test_deterministic_and_row_order_free() -> None:
    firms, source = _frame(), _small_source()
    first = assign_uk_firm_taxable_profit(firms, source)
    np.testing.assert_array_equal(first, assign_uk_firm_taxable_profit(firms, source))
    shuffled = firms.sample(frac=1.0, random_state=11)
    again = pd.Series(
        assign_uk_firm_taxable_profit(shuffled, source), index=shuffled.index
    )
    np.testing.assert_allclose(again.sort_index().to_numpy(), first, rtol=1e-12)
    # The stream is the module's own: the global state is left alone.
    state = np.random.get_state()[1].copy()
    assign_uk_firm_taxable_profit(firms, source, seed=UK_FIRM_PROFIT_SEED)
    np.testing.assert_array_equal(np.random.get_state()[1], state)


def test_firm_table_columns_give_the_same_profit() -> None:
    firms, source = _frame(), _small_source()
    sections = {1: "A", 25: "C", 47: "G", 62: "J", 64: "K", 86: "Q"}
    forms = {1: "COMPANY", 2: "SOLE_TRADER", 3: "PARTNERSHIP", 7: "OTHER"}
    table = pd.DataFrame(
        {
            "firm_id": firms["firm_id"],
            "firm_weight": firms["firm_weight"],
            "firm_turnover": firms["annual_turnover_k"] * 1_000.0,
            "firm_sic_section": firms["sic_code"].map(sections),
            "firm_legal_form": firms["ons_legal_status"].map(forms),
        }
    )
    np.testing.assert_allclose(
        assign_uk_firm_taxable_profit(table, source),
        assign_uk_firm_taxable_profit(firms, source),
    )
    by_section = uk_firm_profit_section_liability(table, source)
    assert set(by_section["sic_section"]) <= set(sections.values())
    np.testing.assert_allclose(
        by_section["weighted_liability_gbp"].sum(), source.positive_liability
    )
    assert by_section["weighted_companies_with_liability"].sum() == 70


def test_weights_and_shortfall() -> None:
    # Too few companies for the bands: the top bands fill first, the lowest
    # band is short and no company is left without a liability.
    firms = _frame(40).assign(ons_legal_status=1, firm_weight=1.5)
    table = uk_firm_profit_diagnostics(firms, _small_source())
    np.testing.assert_allclose(table["weighted_companies"], [0.0, 30.0, 25.5, 4.5])
    assert table["weighted_companies"].sum() == 60.0


def test_real_table_on_a_weighted_frame() -> None:
    source = _source()
    n = 40_000
    rng = np.random.default_rng(5)
    firms = pd.DataFrame(
        {
            "firm_id": np.arange(n),
            "firm_weight": np.full(n, 2_000_000 / n),
            "annual_turnover_k": rng.lognormal(5.0, 2.0, n),
            "sic_code": rng.choice([10, 41, 47, 62, 64, 70], n),
            "ons_legal_status": np.ones(n, dtype=np.int64),
        }
    )
    table = uk_firm_profit_diagnostics(firms, source)
    positive = table.iloc[1:]
    # A row is 50 companies: each band is within a row of the table.
    assert (
        (positive["weighted_companies"] - positive["table_companies"]).abs() <= 50
    ).all()
    assert table["weighted_companies"].sum() == pytest.approx(2_000_000)
    matched = ~positive["table_mean_used"]
    assert matched.all()
    np.testing.assert_allclose(
        positive.loc[matched, "weighted_liability_gbp"],
        positive.loc[matched, "table_liability_gbp"],
        rtol=1e-9,
    )


def test_refusals() -> None:
    firms, source = _frame(), _small_source()
    with pytest.raises(ValueError, match="one row per firm_id"):
        assign_uk_firm_taxable_profit(firms.assign(firm_id=1), source)
    with pytest.raises(ValueError, match="ons_legal_status or firm_legal_form"):
        assign_uk_firm_taxable_profit(firms.drop(columns="ons_legal_status"), source)
    with pytest.raises(ValueError, match="annual_turnover_k or firm_turnover"):
        assign_uk_firm_taxable_profit(firms.drop(columns="annual_turnover_k"), source)


def test_source_refusals(tmp_path) -> None:
    text = (FIXTURE / UK_FIRM_PROFIT_FILE).read_text(encoding="utf-8")
    path = tmp_path / UK_FIRM_PROFIT_FILE
    path.write_text(text.replace("band,lower_gbp", "bands,lower_gbp"), "utf-8")
    with pytest.raises(ValueError, match="columns"):
        read_uk_firm_profit_source(tmp_path)
    path.write_text(
        text.replace(">50m,50000000,,", ">50m,50000000,900000000,"), "utf-8"
    )
    with pytest.raises(ValueError, match="open band"):
        read_uk_firm_profit_source(path)
    path.write_text(text.replace(">5m-10m,5000000,", ">5m-10m,6000000,"), "utf-8")
    with pytest.raises(ValueError, match="does not follow on"):
        read_uk_firm_profit_source(path)


def test_status_is_assumed() -> None:
    assert UK_FIRM_PROFIT_STATUS == "assumed"
