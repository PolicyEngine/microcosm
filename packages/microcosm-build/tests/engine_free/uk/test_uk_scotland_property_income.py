"""The microcosm#1326 tool's engine-free pieces: bands, published values, the
reweight, the record audit and the disclosure statuses."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from tools.uk_scotland_property_income import measures, rake, registry, release
from tools.uk_scotland_property_income.bands import (
    INTENTIONAL_EDGES,
    REQUESTED_EDGES,
    SCOTTISH_RATE_STARTS_2023,
    assign_bands,
    band_labels,
    coarse_index_of_fine,
)

AMOUNTS = ("taxable_property_income",)


def test_bands_are_closed_below_and_open_above():
    values = [12_569.99, 12_570, 14_999.99, 15_000, 89_999, 90_000, 2e6]
    assert assign_bands(values, REQUESTED_EDGES).tolist() == [-1, 0, 0, 1, 8, 9, 9]
    assert band_labels(REQUESTED_EDGES)[0] == "PA–£14,999"
    assert band_labels(REQUESTED_EDGES)[-1] == "£90,000+"
    assert band_labels(INTENTIONAL_EDGES) == [
        "PA–£24,999",
        "£25,000–£39,999",
        "£40,000–£89,999",
        "£90,000+",
    ]


def test_release_2_bins_are_unions_of_the_requested_bands():
    assert coarse_index_of_fine(REQUESTED_EDGES, INTENTIONAL_EDGES).tolist() == [
        0,
        0,
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
    ]
    with pytest.raises(ValueError, match="not unions"):
        coarse_index_of_fine(REQUESTED_EDGES, (12_570, 26_000))


def test_each_bin_edge_is_the_requested_edge_nearest_a_rate_start():
    starts = [SCOTTISH_RATE_STARTS_2023[name] for name in ("intermediate", "higher")]
    for edge, start in zip(INTENTIONAL_EDGES[1:3], starts, strict=True):
        assert edge == min(
            REQUESTED_EDGES, key=lambda candidate: abs(candidate - start)
        )


def _spec(name, value, **metadata):
    return {
        "name": name,
        "value": value,
        "metadata": {"uprating_from_period": "2023", **metadata},
    }


def test_published_value_divides_out_the_uprating_and_the_cross_grain_control():
    plain = _spec(
        "hmrc/property_income_income_band_20_000_to_30_000",
        3_831.0,
        uprating_factor="1.0025",
    )
    assert registry.published_value(plain) == pytest.approx(3_831.0 / 1.0025)
    controlled = _spec(
        "hmrc.spi_region.taxpayers_by_region_20000_30000@S92000003",
        849_390.0,
        uprating_factor="1.00245",
        cross_grain_value_before="843061.0",
    )
    assert registry.published_value(controlled) == pytest.approx(843_061.0 / 1.00245)
    with pytest.raises(ValueError, match="not a 2023-24 fact"):
        registry.published_value(
            {**plain, "metadata": {"uprating_from_period": "2024"}}
        )


def test_spi_facts_read_bands_measures_and_regions_from_the_registry():
    predicate = json.dumps(
        {"variable": "region", "operator": "==", "value": "SCOTLAND"}
    )
    specs = [
        _spec(
            "hmrc/property_income_income_band_1_000_000_to_inf",
            4.0,
            uprating_factor="2",
        ),
        _spec(
            "hmrc/property_income_count_income_band_12_570_to_15_000",
            9.0,
            uprating_factor="3",
        ),
        _spec(
            "hmrc.spi_region.income_tax_by_region_200000_plus@S92000003",
            10.0,
            uprating_factor="2",
            geography_predicate=predicate,
        ),
        _spec("obr.income_tax", 1.0, uprating_factor="1"),
    ]
    facts = registry.spi_2023_facts({"specs": specs})
    assert [(f.measure, f.lower, f.upper, f.region, f.value) for f in facts] == [
        ("property_amount", 1e6, np.inf, None, 2.0),
        ("property_count", 12_570, 15_000, None, 3.0),
        ("income_tax", 200_000, np.inf, "SCOTLAND", 5.0),
    ]


def _persons():
    return pd.DataFrame(
        {
            "household_row": [0, 0, 1, 2],
            "region": ["SCOTLAND", "SCOTLAND", "SCOTLAND", "LONDON"],
            "income_tax": [1_000.0, 0.0, 2_000.0, 3_000.0],
            "property_income": [5_000.0, 7_000.0, 0.0, 9_000.0],
            "total_income_after_loss_relief": [25_000.0, 22_000.0, 28_000.0, 26_000.0],
        }
    )


def _band_fact(measure, region=None):
    return registry.SpiFact("x", measure, 20_000, 30_000, region, 1.0)


def _contribution(persons, measure, region=None):
    return measures.contribution(persons, _band_fact(measure, region)).tolist()


def test_contributions_follow_the_build_bindings():
    persons = _persons()
    fact = _band_fact
    # Property amounts and counts are taxpayers' only.
    assert _contribution(persons, "property_amount") == [5_000, 0, 0, 9_000]
    assert _contribution(persons, "property_count") == [1, 0, 0, 1]
    # Regional total income has no taxpayer filter; the region predicate applies.
    assert _contribution(persons, "total_income", "SCOTLAND") == [
        25_000,
        22_000,
        28_000,
        0,
    ]
    assert _contribution(persons, "taxpayers", "SCOTLAND") == [1, 0, 1, 0]
    matrix = measures.household_matrix(persons, [fact("property_amount")], 3)
    assert matrix.toarray().ravel().tolist() == [5_000, 0, 9_000]


def test_total_income_reads_trading_profit_after_losses_brought_forward():
    after = measures.total_income_after_loss_relief(
        [30_000, 30_000], [10_000, 10_000], [4_000, 15_000]
    )
    assert after.tolist() == [26_000, 20_000]


def test_rake_meets_reachable_facts_within_its_bounds():
    rng = np.random.default_rng(1)
    matrix = sparse.csr_matrix(
        rng.uniform(0, 1, size=(400, 5)) * (rng.uniform(size=(400, 5)) < 0.4)
    )
    initial = rng.uniform(50, 150, size=400)
    true_ratio = rng.uniform(0.7, 1.4, size=400)
    targets = matrix.T @ (initial * true_ratio)
    result = rake.rake(matrix, initial, targets, lower=0.5, upper=2.0)
    assert np.abs(result.relative_error_after).max() < 1e-5
    assert result.ratio.min() >= 0.5 and result.ratio.max() <= 2.0


def test_rake_stops_at_its_bounds_when_a_fact_is_out_of_reach():
    matrix = sparse.csr_matrix(np.ones((10, 1)))
    result = rake.rake(matrix, np.ones(10), [100.0], lower=0.5, upper=2.0)
    assert np.allclose(result.ratio, 2.0)
    assert result.after[0] == pytest.approx(20.0)


def test_source_keys_count_clones_and_copies_once():
    table = pd.DataFrame(
        {
            "person_support_channel": [b"frs", b"frs", b"spi", b"spi", b"spi"],
            "person_source_id": [1001, 1001, 2001, 2001, 2001],
            **{name: [0.0] * 5 for name in release.SOURCE_INCOME_COLUMNS},
        }
    )
    table.loc[[2, 3], "property_income"] = 12_000.0
    table.loc[4, "property_income"] = 13_000.0
    keys = release.source_keys(table)
    assert keys[0] == keys[1] == "frs:1001"
    # SPI rows share their FRS host's id, so their incomes tell them apart.
    assert keys[2] == keys[3] != keys[4]


def _cells(sources_per_band, rows_per_source=3, band_values=(13_000, 16_000, 21_000)):
    rows = []
    for band_value, n_sources in zip(band_values, sources_per_band, strict=True):
        for source in range(n_sources):
            for _clone in range(rows_per_source):
                rows.append(
                    {
                        "nsnd": band_value,
                        "source": f"{band_value}:{source}",
                        "w": 10.0,
                        "taxable_property_income": 1_000.0 + source,
                    }
                )
    return pd.DataFrame(rows)


def test_audit_counts_sources_not_rows():
    edges = (12_570, 15_000, 20_000)
    table = release.audit(_cells((4, 12, 20)), edges, AMOUNTS)
    assert table["rows"].tolist() == [12, 36, 60]
    assert table["sources"].tolist() == [4, 12, 20]
    # Equal weight per source: the effective sample size is the source count.
    assert table["source_ess"].tolist() == pytest.approx([4, 12, 20])


def test_bootstrap_moves_clones_together():
    cells = _cells((1, 0, 0), rows_per_source=5)
    intervals = release.bootstrap(
        cells, (12_570, 15_000, 20_000), AMOUNTS, replicates=50
    )
    # A single source is drawn once in every replicate, so its band never varies.
    assert intervals.loc[0, "count_low"] == intervals.loc[0, "count_high"] == 50.0


FEW, SECOND, PUBLISHED = release.FEWER_THAN_FLOOR, release.SECONDARY, release.PUBLISHED
FINE_EDGES = (12_570, 15_000, 20_000)


def test_a_lone_withheld_band_is_protected_inside_its_bin():
    statuses = release.disclosure_statuses(
        _cells((4, 12, 20)), FINE_EDGES, (12_570,), AMOUNTS
    )
    assert statuses == ([FEW, SECOND, PUBLISHED], [PUBLISHED])


def test_a_published_total_protects_a_lone_withheld_bin():
    cells = _cells((4, 12, 20))
    # The first bin is the thin band alone. With the total published, it
    # equals the total less the other bin, so that bin is withheld too, and
    # one of its bands, so it can't be rebuilt from them.
    assert release.disclosure_statuses(
        cells, FINE_EDGES, (12_570, 15_000), AMOUNTS
    ) == ([FEW, SECOND, PUBLISHED], [FEW, SECOND])
    assert release.disclosure_statuses(
        cells, FINE_EDGES, (12_570, 15_000), AMOUNTS, total_published=False
    ) == ([FEW, PUBLISHED, PUBLISHED], [FEW, PUBLISHED])


def test_one_dominant_record_withholds_its_band():
    cells = _cells((0, 12, 0))
    cells.loc[cells["source"] == "16000:0", "w"] = 1_000.0
    fine_status, _ = release.disclosure_statuses(cells, FINE_EDGES, (12_570,), AMOUNTS)
    assert fine_status[1] == release.DOMINATED


def test_public_tables_blank_withheld_bands_and_floor_record_counts():
    edges = (12_570, 15_000, 20_000)
    cells = _cells((4, 12, 20))
    table = release.audit(cells, edges, AMOUNTS)
    intervals = release.bootstrap(cells, edges, AMOUNTS, replicates=20)
    statuses = [release.FEWER_THAN_FLOOR, release.SECONDARY, release.PUBLISHED]
    public = release.render(table, intervals, statuses, AMOUNTS, mode="public")
    assert public["distinct_records"].tolist() == ["<10", "12", "20"]
    assert pd.isna(public.loc[0, "count"]) and pd.isna(
        public.loc[1, "taxable_property_income"]
    )
    assert public.loc[2, "count"] == 600.0
    restricted = release.render(table, intervals, statuses, AMOUNTS, mode="restricted")
    assert restricted["distinct_records"].tolist() == ["4", "12", "20"]
    assert restricted.loc[0, "count"] == 100.0
