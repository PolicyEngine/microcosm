"""UK firm consumption links from the ONS Supply and Use Tables 2024."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_consumption import (
    UK_COICOP_DIVISIONS,
    UK_FIRM_CONSUMPTION_COLUMNS,
    UK_FIRM_CONSUMPTION_STATUS,
    UK_FIRM_SUT_FILES,
    UK_FIRM_TABLE3_GATE_TOLERANCE_GBP_MILLION,
    UK_FIRM_TRADE_DIVISIONS,
    UKFirmSupplyUseSource,
    check_uk_table3_consumer_trends,
    read_uk_firm_supply_use_source,
    uk_firm_consumption_sales,
    uk_firm_division_coicop_shares,
    uk_firm_division_use_shares,
    uk_firm_household_sales_by_coicop,
    uk_household_purchases_supply_decomposition,
    uk_table3_consumer_trends_comparison,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURE = PATHS.tests / "fixtures" / "uk" / "firm_supply_use_2024"
PROVENANCE = FIXTURE / "firm_supply_use_2024.provenance.json"
TOOL = PATHS.repository / "tools" / "extract_uk_firm_supply_use_tables.py"
SHARES = ("household_share", "intermediate_share", "other_final_share")


@cache
def _source() -> UKFirmSupplyUseSource:
    return read_uk_firm_supply_use_source(FIXTURE)


@cache
def _uses() -> pd.DataFrame:
    return uk_firm_division_use_shares(_source())


@cache
def _provenance() -> dict:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


@cache
def _load_tool():
    name = "extract_uk_firm_supply_use_tables"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_fixture_hashes_match_provenance():
    record = _provenance()
    assert record["source"]["sha256"] == _load_tool().XLSX_SHA256
    assert "October 2026" in record["source"]["edition"]
    assert record["accessed"] == "2026-11-22"
    for name in UK_FIRM_SUT_FILES:
        payload = (FIXTURE / name).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert record["files"][name]["sha256"] == digest
        assert _source().sha256[name] == digest


def test_fixture_shape_and_identities():
    uses = _source().uses
    assert len(uses) == 103
    # Supply side adds up to total supply, which equals total demand.
    supply = uses[
        ["domestic_output", "imports", "distributors_margins", "taxes_less_subsidies"]
    ].sum(axis=1)
    assert np.array_equal(supply, uses["total_supply"])
    assert np.array_equal(uses["total_supply"], uses["total_demand"])
    assert uses["distributors_margins"].sum() == 0
    # Table 2 households is the national concept: 1,708,930 million pounds.
    assert uses["households"].sum() == 1_708_930
    # Table 3 classes (domestic concept) total 1,692,413 million pounds.
    assert _source().hhfce["value_gbp_million"].sum() == 1_692_413


def test_trade_margins_are_negative_only_on_trade_products():
    uses = _source().uses
    negative = uses.index[uses["distributors_margins"] < 0].tolist()
    assert negative == ["CPA_G45", "CPA_G46", "CPA_G47"]
    assert uses.loc["CPA_G47", "total_demand"] == 0


def test_product_division_parse():
    divisions = _source().divisions
    of = divisions.groupby("product_code")["sic_division"].apply(list)
    assert of["CPA_C102_3"] == [10]
    assert of["CPA_B05 & B06"] == [5, 6]
    assert of["CPA_C11.01-6 & C12"] == [11, 12]
    assert of["CPA_F41, F42 & F43"] == [41, 42, 43]
    assert of["CPA_K65.1-2 & K65.3"] == [65]
    assert of["CPA_L68A"] == [68]
    assert 98 not in set(divisions["sic_division"])


def test_shares_in_unit_interval_and_sum_to_one():
    shares = _uses()[list(SHARES)]
    assert ((shares >= 0) & (shares <= 1)).all().all()
    assert np.allclose(shares.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_aggregation_sums_numerators_and_denominators():
    uses = _source().uses
    products = [f"CPA_C10{k}" for k in (1, 4, 5, 6, 7, 8, 9)] + ["CPA_C102_3"]
    rows = uses.loc[products]
    floored = rows.drop(columns=["product_label", "total_demand"]).iloc[:, :9]
    denominator = floored.clip(lower=0).sum().sum()
    expected = rows["households"].sum() / denominator
    assert _uses().loc[10, "household_share"] == pytest.approx(expected, rel=1e-12)
    mean_of_shares = (rows["households"] / rows["total_demand"]).mean()
    assert _uses().loc[10, "household_share"] != pytest.approx(mean_of_shares)


def test_imputed_rent_left_out_of_division_68():
    uses = _source().uses
    kept = uses.loc[["CPA_L68BXL683", "CPA_L683"]]
    assert _uses().loc[68, "total_demand_gbp_million"] == kept["total_demand"].sum()


def test_trade_divisions_are_margin_weighted():
    uses = _source().uses
    goods = uses[uses["distributors_margins"] > 0]
    other = goods[
        [
            "npish",
            "central_government",
            "local_government",
            "gfcf",
            "valuables",
            "exports",
        ]
    ].sum(axis=1) + goods["changes_in_inventories"].clip(lower=0)
    total = goods["households"] + goods["intermediate_consumption"] + other
    margin = goods["distributors_margins"]
    expected = (margin * goods["households"] / total).sum() / margin.sum()
    for division in UK_FIRM_TRADE_DIVISIONS:
        row = _uses().loc[division]
        assert row["basis"] == "trade_margin"
        assert row["household_share"] == pytest.approx(expected, rel=1e-12)
        assert row["total_demand_gbp_million"] == pytest.approx(margin.sum())


def test_firm_columns_sum_to_turnover():
    rng = np.random.default_rng(7)
    divisions = rng.choice(np.arange(1, 100), size=5000)
    turnover = rng.normal(2e6, 5e6, size=5000)
    turnover[:10] = 0.0
    result = uk_firm_consumption_sales(divisions, turnover, _source())
    columns = result.columns
    assert tuple(columns.columns) == UK_FIRM_CONSUMPTION_COLUMNS
    assert result.status == UK_FIRM_CONSUMPTION_STATUS == "assumed"
    total = columns.sum(axis=1).to_numpy()
    assert np.allclose(total, turnover, rtol=1e-15, atol=1e-6)
    assert (columns.iloc[:10] == 0).all().all()
    # Positive turnover gives non-negative columns, negative gives non-positive.
    assert (columns[turnover > 0] >= -1e-6).all().all()
    assert (columns[turnover < 0] <= 1e-6).all().all()


def test_firm_columns_use_division_shares():
    result = uk_firm_consumption_sales([56, 47], [100.0, 1000.0], _source())
    assert result.columns.loc[0, "firm_household_sales"] == pytest.approx(
        100.0 * _uses().loc[56, "household_share"]
    )
    assert result.columns.loc[1, "firm_intermediate_sales"] == pytest.approx(
        1000.0 * _uses().loc[47, "intermediate_share"]
    )


def test_fallbacks_are_counted():
    divisions = [10, "47", 98, 99, 4, 0, None, "x"]
    result = uk_firm_consumption_sales(divisions, np.ones(8), _source())
    # 98 takes section T (division 97); 99 (section U, no product), 4 (not a
    # SIC division), 0, missing and unreadable take the economy.
    assert dict(result.fallback_counts) == {"division": 2, "section": 1, "economy": 5}
    assert result.columns.loc[2, "firm_household_sales"] == pytest.approx(
        _uses().loc[97, "household_share"]
    )


def test_coicop_shares_sum_to_one_per_division():
    shares = uk_firm_division_coicop_shares(_source())
    values = shares[list(UK_COICOP_DIVISIONS)]
    assert ((values >= 0) & (values <= 1)).all().all()
    assert np.allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    assert set(shares.index) == set(_uses().index)
    assert shares.loc[56, "11"] == pytest.approx(1.0)
    assert shares.loc[64, "12"] == pytest.approx(1.0)
    for division in UK_FIRM_TRADE_DIVISIONS:
        assert shares.loc[division, "basis"] == "trade_margin"
    # Divisions without Table 3 household purchases fall back to the section.
    assert shares.loc[63, "basis"] == "section"


def test_table3_matches_consumer_trends():
    frame = check_uk_table3_consumer_trends(_source())
    assert list(frame.index) == list(UK_COICOP_DIVISIONS)
    assert frame["difference"].abs().max() <= UK_FIRM_TABLE3_GATE_TOLERANCE_GBP_MILLION
    assert frame.loc["02", "excluded"] == "02.3"
    assert frame.loc["04", "excluded"] == "04.2"
    assert frame["consumer_trends"].sum() == pytest.approx(1_692_413 - 3_044 - 291_626)


def test_household_sales_diagnostic_runs():
    result = uk_firm_consumption_sales([56, 64], [1e9, 2e9], _source())
    frame = uk_firm_household_sales_by_coicop(
        [56, 64], result.columns["firm_household_sales"], [1.0, 2.0], _source()
    )
    assert frame.loc["11", "firm_household_sales"] == pytest.approx(
        1e3 * _uses().loc[56, "household_share"]
    )
    assert frame.loc["12", "firm_household_sales"] == pytest.approx(
        4e3 * _uses().loc[64, "household_share"]
    )
    assert frame.loc["01", "firm_household_sales"] == 0


def test_deterministic():
    first = read_uk_firm_supply_use_source(FIXTURE)
    pd.testing.assert_frame_equal(
        uk_firm_division_use_shares(first), uk_firm_division_use_shares(_source())
    )
    pd.testing.assert_frame_equal(
        uk_firm_division_coicop_shares(first),
        uk_firm_division_coicop_shares(_source()),
    )
    pd.testing.assert_frame_equal(
        uk_table3_consumer_trends_comparison(first),
        uk_table3_consumer_trends_comparison(_source()),
    )


def test_wrong_columns_refused(tmp_path):
    for name in UK_FIRM_SUT_FILES:
        (tmp_path / name).write_bytes((FIXTURE / name).read_bytes())
    uses = pd.read_csv(tmp_path / UK_FIRM_SUT_FILES[0])
    uses.drop(columns="exports").to_csv(tmp_path / UK_FIRM_SUT_FILES[0], index=False)
    with pytest.raises(ValueError, match="columns"):
        read_uk_firm_supply_use_source(tmp_path)


def test_tool_parses_product_codes():
    tool = _load_tool()
    assert tool.product_divisions("CPA_C11.01-6 & C12") == [("C", "11"), ("C", "12")]
    assert tool.product_divisions("CPA_L68BXL683") == [("L", "68")]
    assert tool.table3_products("CPA_C27 & C28", ["CPA_C27", "CPA_C28"]) == [
        "CPA_C27",
        "CPA_C28",
    ]


def test_household_purchases_split_by_supply_sums_to_table_2() -> None:
    """Domestic, imports, margins and taxes add up to each non-trade
    product's household purchases; imputed rent is left out and trade rows
    keep their own demand."""

    parts = uk_household_purchases_supply_decomposition(_source())
    uses = _source().uses
    assert "CPA_L68A" not in parts.index
    split = parts[
        ["domestic_basic", "imports", "distributors_margins", "taxes_less_subsidies"]
    ].sum(axis=1)
    trade = parts["trade_own_demand"] > 0
    np.testing.assert_allclose(
        split[~trade], parts.loc[~trade, "households"], rtol=1e-9, atol=1e-9
    )
    assert not split[trade].any()
    expected = uses["households"].drop("CPA_L68A").sum()
    assert parts["households"].sum() == pytest.approx(expected)
    assert split.sum() + parts["trade_own_demand"].sum() == pytest.approx(expected)
    assert set(parts.index[trade]) <= {c for c in uses.index if c.startswith("CPA_G4")}
