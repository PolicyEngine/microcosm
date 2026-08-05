"""Tests for import-entry fact emission and the CBP entry-anchor parse."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from populace.build.ledger_artifact import load_ledger_consumer_artifact
from populace.build.ledger_targets import (
    LedgerTargetReference,
    compile_ledger_target_references,
)
from populace.build.us_trade.cbp_entry_stats import parse_cbp_trade_stats
from populace.build.us_trade.import_entry_facts import (
    IMPORT_ENTRY_FACT_GRAINS,
    build_cbp_entry_fact_rows,
    build_import_entry_fact_rows,
    default_generator_block,
    write_consumer_artifact,
)

GOLDEN = Path(__file__).parent / "golden" / "us_trade"
CBP_FIXTURE = GOLDEN / "cbp_trade_stats_fragment.html"


def _margins() -> pd.DataFrame:
    rows = [
        ("2026-01", "0101210010", "01", "1220", "CA", 1_000, 1_100, 10, 500),
        ("2026-01", "0101210010", "01", "2010", "MX", 2_000, 2_000, 40, 900),
        ("2026-01", "8471300100", "84", "5700", "CN", 30_000, 31_000, 9_000, 30_000),
        ("2026-02", "8471300100", "84", "5700", "CN", 40_000, 40_000, 12_000, 40_000),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "period",
            "hts10",
            "chapter",
            "cty_code",
            "iso2",
            "con_val_mo",
            "gen_val_mo",
            "cal_dut_mo",
            "dut_val_mo",
        ],
    )
    frame["con_qy1_mo"] = pd.array([1, 2, 3, 4], dtype="Int64")
    frame["gen_qy1_mo"] = pd.array([1, 2, 3, 4], dtype="Int64")
    frame["unit_qy1"] = "NO"
    frame["country_name"] = ""
    return frame


def _manifest_entries() -> list[dict]:
    return [
        {
            "month": "2026-01",
            "chapter": "01",
            "sha256": "aa" * 32,
            "filename": "imports_hs10_2026-01_ch01.json",
        },
        {
            "month": "2026-01",
            "chapter": "84",
            "sha256": "bb" * 32,
            "filename": "imports_hs10_2026-01_ch84.json",
        },
        {
            "month": "2026-02",
            "chapter": "84",
            "sha256": "cc" * 32,
            "filename": "imports_hs10_2026-02_ch84.json",
        },
    ]


def test_fact_rows_sum_margins_exactly_per_grain():
    margins = _margins()
    rows = build_import_entry_fact_rows(
        margins,
        retrieval_manifest=_manifest_entries(),
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    by_id = {row["lineage"]["source_record_id"]: row for row in rows}
    assert len(by_id) == len(rows)

    national = by_id[
        "census_intltrade.imports_hs10.month_2026_01.national.all.con_val_mo"
    ]
    assert national["value"] == 33_000
    assert national["period"] == {"type": "month", "value": "2026-01"}
    assert national["entity"] == {"name": "import_entry", "role": "customs_entry"}

    chapter = by_id[
        "census_intltrade.imports_hs10.month_2026_01.chapter.ch01.cal_dut_mo"
    ]
    assert chapter["value"] == 50
    assert chapter["lineage"]["source_file_sha256s"] == ["aa" * 32]

    country = by_id[
        "census_intltrade.imports_hs10.month_2026_01.country.5700.con_val_mo"
    ]
    assert country["value"] == 30_000
    assert country["dimensions"]["country_of_origin"] == "CN"
    assert country["dimensions"]["census_country_code"] == "5700"

    cross = by_id[
        "census_intltrade.imports_hs10.month_2026_02.chapter_country"
        ".ch84.5700.con_val_mo"
    ]
    assert cross["value"] == 40_000
    with pytest.raises(KeyError):
        by_id[
            "census_intltrade.imports_hs10.month_2026_02.chapter_country"
            ".ch84.5700.gen_val_mo"
        ]


def test_fact_rows_are_deterministic_and_carry_producer_identity():
    margins = _margins()
    first = build_import_entry_fact_rows(
        margins,
        retrieval_manifest=_manifest_entries(),
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    second = build_import_entry_fact_rows(
        margins.sample(frac=1.0, random_state=7),
        retrieval_manifest=_manifest_entries(),
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    assert first == second
    for row in first:
        assert row["aggregate_fact_key"].startswith(
            "populace_us_trade.aggregate_fact.v1:"
        )
        assert "populace-minted" in row["source"]["extraction_method"]
        assert row["assertion"] == "observation"
        assert row["aggregation"] == {"method": "sum"}


def test_customs_value_facts_align_to_the_composition_input():
    rows = build_import_entry_fact_rows(
        _margins(),
        retrieval_manifest=_manifest_entries(),
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    customs = [row for row in rows if row["layout"]["measure_id"] == "con_val_mo"]
    assert customs
    for row in customs:
        alignment = row["concept_alignment"]
        assert alignment["canonical_concept"] == (
            "us:policies/cbp/us-tariff-duty/composition#input.customs_value"
        )
        assert alignment["relation"] == "exact"
    general = [row for row in rows if row["layout"]["measure_id"] == "gen_val_mo"]
    assert general
    assert all("concept_alignment" not in row for row in general)


def test_unknown_grain_and_empty_margins_are_refused():
    with pytest.raises(ValueError, match="Unknown import-entry fact grain"):
        build_import_entry_fact_rows(
            _margins(),
            retrieval_manifest=(),
            extracted_at="2026-08-05T00:00:00+00:00",
            grains=("national", "bogus"),
        )
    with pytest.raises(ValueError, match="empty margins"):
        build_import_entry_fact_rows(
            _margins().iloc[:0],
            retrieval_manifest=(),
            extracted_at="2026-08-05T00:00:00+00:00",
        )


def test_artifact_round_trips_through_the_populace_loader(tmp_path):
    rows = build_import_entry_fact_rows(
        _margins(),
        retrieval_manifest=_manifest_entries(),
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    manifest = write_consumer_artifact(
        tmp_path / "artifact",
        rows,
        retrieval_manifest=_manifest_entries(),
        generator=default_generator_block(months=("2026-01", "2026-02")),
    )
    artifact = load_ledger_consumer_artifact(
        tmp_path / "artifact",
        expected_facts_sha256=manifest["facts_sha256"],
    )
    assert artifact.fact_row_count == len(rows)
    provenance = artifact.provenance()
    assert provenance["schema_version"] == ("policyengine_ledger.consumer_artifact.v1")
    assert artifact.manifest["generator"]["producer"] == "populace.build.us_trade"
    assert artifact.manifest["source_manifest"]["set_digest"]

    with pytest.raises(ValueError, match="empty consumer artifact"):
        write_consumer_artifact(
            tmp_path / "empty",
            [],
            retrieval_manifest=(),
            generator={},
        )


def test_facts_compile_into_populace_ledger_targets(tmp_path):
    rows = build_import_entry_fact_rows(
        _margins(),
        retrieval_manifest=_manifest_entries(),
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    reference = LedgerTargetReference(
        name="us_imports_customs_value_2026_01",
        ledger_selector={
            "source_record_id": (
                "census_intltrade.imports_hs10.month_2026_01.national.all.con_val_mo"
            ),
        },
        entity="import_entry",
        measure="customs_value",
        period="2026-01",
        family="trade_imports",
    )
    registry = compile_ledger_target_references(rows, [reference], country="us")
    (spec,) = registry.specs
    assert spec.value == 33_000.0
    assert spec.measure == "customs_value"
    assert spec.period == "2026-01"
    assert spec.metadata["ledger_measure_unit"] == "usd"
    assert spec.metadata["ledger_fact_period"] == "2026-01"
    assert spec.metadata["ledger_domain"] == "imports_for_consumption"


def test_cbp_fixture_parses_exact_cells_only():
    stats = parse_cbp_trade_stats(CBP_FIXTURE.read_bytes())
    exact = {
        (cell.measure_id, cell.fiscal_year): cell.exact_value
        for cell in stats.exact_cells()
    }
    assert exact == {
        ("total_import_value", 2026): 2_736_460_052_598,
        ("total_entry_summaries", 2026): 83_133_856,
        ("informal_entry_summaries", 2026): 56_722_729,
        ("duty_taxes_fees_collected", 2026): 255_739_474_306,
    }
    rounded = [cell for cell in stats.cells if not cell.is_exact]
    assert rounded
    assert all(cell.fiscal_year != 2026 for cell in rounded)
    assert "updated as of July 27, 2026" in stats.as_of_note


def test_cbp_fact_rows_from_fixture():
    stats = parse_cbp_trade_stats(CBP_FIXTURE.read_bytes())
    rows = build_cbp_entry_fact_rows(
        stats,
        page_sha256="dd" * 32,
        retrieved_at="2026-08-05T00:00:00+00:00",
    )
    assert len(rows) == 4
    by_measure = {row["layout"]["measure_id"]: row for row in rows}
    entries = by_measure["total_entry_summaries"]
    assert entries["value"] == 83_133_856
    assert entries["period"] == {"type": "fiscal_year", "value": 2026}
    assert entries["observed_measure"]["unit"] == "count"
    assert entries["source"]["source_sha256"] == "dd" * 32
    assert "publisher_as_of_note" in entries["dimensions"]


def test_cbp_parse_fails_closed_on_layout_change():
    with pytest.raises(ValueError, match="no table containing"):
        parse_cbp_trade_stats(b"<html><table><tr><td>x</td></tr></table></html>")


def test_grain_catalog_is_the_documented_set():
    assert IMPORT_ENTRY_FACT_GRAINS == (
        "national",
        "chapter",
        "country",
        "chapter_country",
    )
