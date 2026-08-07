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
from populace.build.us_runtime.us_trade.cbp_entry_stats import (
    CbpEntryStats,
    CbpFiscalYearCell,
    parse_cbp_trade_stats,
)
from populace.build.us_runtime.us_trade.import_entry_facts import (
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
            "retrieved_at": "2026-08-05T00:00:00+00:00",
        },
        {
            "month": "2026-01",
            "chapter": "84",
            "sha256": "bb" * 32,
            "filename": "imports_hs10_2026-01_ch84.json",
            "retrieved_at": "2026-08-05T00:00:00+00:00",
        },
        {
            "month": "2026-02",
            "chapter": "84",
            "sha256": "cc" * 32,
            "filename": "imports_hs10_2026-02_ch84.json",
            "retrieved_at": "2026-08-05T00:00:00+00:00",
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


def test_split_chapter_facts_carry_the_full_file_set():
    """A chapter served as several prefix files hashes the whole set."""
    manifest = _manifest_entries() + [
        {
            "month": "2026-01",
            "chapter": "84",
            "prefix": "847",
            "sha256": "ee" * 32,
            "filename": "imports_hs10_2026-01_p847.json",
        },
        {
            "month": "2026-01",
            "chapter": "84",
            "prefix": "84",
            "sha256": None,
            "superseded_by_split": True,
        },
    ]
    rows = build_import_entry_fact_rows(
        _margins(),
        retrieval_manifest=manifest,
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    by_id = {row["lineage"]["source_record_id"]: row for row in rows}
    split = by_id["census_intltrade.imports_hs10.month_2026_01.chapter.ch84.con_val_mo"]
    assert split["lineage"]["source_file_sha256s"] == sorted(["bb" * 32, "ee" * 32])
    assert split["source"]["source_file"] == "2 prefix files (set digest)"
    assert split["source"]["source_sha256"] not in ("bb" * 32, "ee" * 32)
    single = by_id[
        "census_intltrade.imports_hs10.month_2026_01.chapter.ch01.con_val_mo"
    ]
    assert single["lineage"]["source_file_sha256s"] == ["aa" * 32]
    assert single["source"]["source_sha256"] == "aa" * 32


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
    assert (
        artifact.manifest["generator"]["producer"]
        == "populace.build.us_runtime.us_trade"
    )
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


def test_cbp_fact_rows_from_fixture_are_fytd_snapshots():
    """FY2026 was in progress at the page's as-of endpoint: every exact cell
    must publish as a fiscal-year-to-date observation, never as a completed
    annual total (the archived count covers October 2025 – July 27, 2026)."""

    stats = parse_cbp_trade_stats(CBP_FIXTURE.read_bytes())
    assert stats.as_of_date == "2026-07-27"
    rows = build_cbp_entry_fact_rows(
        stats,
        page_sha256="dd" * 32,
        retrieved_at="2026-08-05T00:00:00+00:00",
    )
    assert len(rows) == 4
    by_measure = {row["layout"]["measure_id"]: row for row in rows}
    entries = by_measure["total_entry_summaries"]
    assert entries["value"] == 83_133_856
    assert entries["period"] == {
        "type": "fiscal_year_to_date",
        "value": 2026,
        "start": "2025-10-01",
        "as_of": "2026-07-27",
        "as_of_basis": "publisher_as_of_note",
    }
    assert entries["source"]["vintage"] == ("fiscal_year_to_date_2026_as_of_2026_07_27")
    assert entries["layout"]["record_set_id"] == (
        "cbp_trade_stats.imports_revenue_collection.fytd.fy2026_as_of_2026_07_27"
    )
    assert "FYTD 2026 (through 2026-07-27)" in entries["label"]
    assert entries["observed_measure"]["unit"] == "count"
    assert entries["source"]["source_sha256"] == "dd" * 32
    assert entries["source"]["source_file"] == "cbp_newsroom_stats_trade.html"
    assert entries["dimensions"] == {}
    assert "updated as of" in entries["source"]["publisher_as_of_note"]


def _fy2025_complete_stats(as_of_date: str = "2026-07-27") -> CbpEntryStats:
    return CbpEntryStats(
        cells=(
            CbpFiscalYearCell(
                measure_id="total_entry_summaries",
                unit="count",
                fiscal_year=2025,
                text="99,000,000",
                exact_value=99_000_000,
            ),
        ),
        as_of_note=f"FY 2026 and FY 2025 are updated as of {as_of_date}.",
        as_of_date=as_of_date,
    )


def test_cbp_fiscal_year_end_day_itself_is_still_fiscal_year_to_date():
    """An endpoint ON September 30 covers a day still in progress — the
    year completes only once its end day has elapsed. The inclusive
    comparison this replaces marked the year complete a day early, on
    both the as-of and the retrieval-fallback bases."""

    (on_the_end,) = build_cbp_entry_fact_rows(
        _fy2025_complete_stats(as_of_date="2025-09-30"),
        page_sha256="dd" * 32,
        retrieved_at="2025-10-15T00:00:00+00:00",
    )
    assert on_the_end["period"]["type"] == "fiscal_year_to_date"
    assert on_the_end["period"]["as_of"] == "2025-09-30"

    noteless = CbpEntryStats(
        cells=_fy2025_complete_stats().cells, as_of_note="", as_of_date=""
    )
    (via_retrieval,) = build_cbp_entry_fact_rows(
        noteless,
        page_sha256="dd" * 32,
        retrieved_at="2025-09-30T00:00:00+00:00",
    )
    assert via_retrieval["period"]["type"] == "fiscal_year_to_date"
    assert via_retrieval["period"]["as_of"] == "2025-09-30"

    (day_after,) = build_cbp_entry_fact_rows(
        _fy2025_complete_stats(as_of_date="2025-10-01"),
        page_sha256="dd" * 32,
        retrieved_at="2025-10-15T00:00:00+00:00",
    )
    assert day_after["period"] == {"type": "fiscal_year", "value": 2025}


def test_cbp_fact_builder_refuses_future_and_malformed_retrievals():
    """The retrieval timestamp is parsed whole — truncating to the date
    prefix would bless ``2026-09-30T99:99:99+00:00`` — must carry a
    timezone, and a publisher as-of endpoint later than the retrieval is
    refused: a page cannot vouch for coverage beyond the moment it was
    read."""

    with pytest.raises(ValueError, match="not a full ISO timestamp"):
        build_cbp_entry_fact_rows(
            _fy2025_complete_stats(),
            page_sha256="dd" * 32,
            retrieved_at="2026-09-30T99:99:99+00:00",
        )
    with pytest.raises(ValueError, match="carries no timezone"):
        build_cbp_entry_fact_rows(
            _fy2025_complete_stats(),
            page_sha256="dd" * 32,
            retrieved_at="2026-08-05T00:00:00",
        )
    with pytest.raises(ValueError, match="postdates the retrieval"):
        build_cbp_entry_fact_rows(
            _fy2025_complete_stats(as_of_date="2026-08-24"),
            page_sha256="dd" * 32,
            retrieved_at="2026-08-05T00:00:00+00:00",
        )


def test_cbp_completed_fiscal_year_still_publishes_as_fiscal_year():
    """A fiscal year whose September 30 end predates the as-of endpoint is a
    completed annual observation and keeps the plain fiscal-year encoding."""

    rows = build_cbp_entry_fact_rows(
        _fy2025_complete_stats(),
        page_sha256="dd" * 32,
        retrieved_at="2026-08-05T00:00:00+00:00",
    )
    (row,) = rows
    assert row["period"] == {"type": "fiscal_year", "value": 2025}
    assert row["source"]["vintage"] == "fiscal_year_2025"
    assert row["layout"]["record_set_id"] == (
        "cbp_trade_stats.imports_revenue_collection.fy2025"
    )
    assert "FY2025" in row["label"]


def test_cbp_fytd_snapshots_never_collide_across_as_of_endpoints():
    """Two snapshots of the same in-progress year are distinct observations:
    their aggregate keys must differ, and the retrieval date is the honest
    fallback endpoint when the page carries no as-of note."""

    stats = parse_cbp_trade_stats(CBP_FIXTURE.read_bytes())
    early = build_cbp_entry_fact_rows(
        stats,
        page_sha256="dd" * 32,
        retrieved_at="2026-08-05T00:00:00+00:00",
    )
    later_stats = CbpEntryStats(
        cells=stats.cells, as_of_note=stats.as_of_note, as_of_date="2026-08-24"
    )
    later = build_cbp_entry_fact_rows(
        later_stats,
        page_sha256="ee" * 32,
        retrieved_at="2026-09-01T00:00:00+00:00",
    )
    early_keys = {row["aggregate_fact_key"] for row in early}
    later_keys = {row["aggregate_fact_key"] for row in later}
    assert early_keys.isdisjoint(later_keys)

    noteless = CbpEntryStats(cells=stats.cells, as_of_note="", as_of_date="")
    fallback = build_cbp_entry_fact_rows(
        noteless,
        page_sha256="ff" * 32,
        retrieved_at="2026-08-05T00:00:00+00:00",
    )
    entry = next(
        row
        for row in fallback
        if row["layout"]["measure_id"] == "total_entry_summaries"
    )
    assert entry["period"]["as_of"] == "2026-08-05"
    assert entry["period"]["as_of_basis"] == "retrieval_date"


def test_cbp_parse_fails_closed_on_layout_change():
    with pytest.raises(ValueError, match="no table containing"):
        parse_cbp_trade_stats(b"<html><table><tr><td>x</td></tr></table></html>")


_FIXTURE_NOTE = "FY 2026 and FY 2025 is updated as of July 27, 2026."


def _fixture_with_note(replacement: str) -> bytes:
    html = CBP_FIXTURE.read_text()
    assert _FIXTURE_NOTE in html
    return html.replace(_FIXTURE_NOTE, replacement).encode("utf-8")


def test_cbp_single_fiscal_year_note_parses_and_keeps_fytd_labeling():
    """The r2 probe: with the note reworded to name a single fiscal year
    and retrieval after September 30, the old parser treated the note as
    absent and emitted plain completed-year facts. The single-year wording
    must parse, and the late retrieval must still label FY2026 as FYTD
    through the note's endpoint."""

    stats = parse_cbp_trade_stats(
        _fixture_with_note("FY 2026 is updated as of July 27, 2026.")
    )
    assert stats.as_of_date == "2026-07-27"
    rows = build_cbp_entry_fact_rows(
        stats,
        page_sha256="dd" * 32,
        retrieved_at="2026-10-01T00:00:00+00:00",
    )
    assert len(rows) == 4
    for row in rows:
        assert row["period"]["type"] == "fiscal_year_to_date"
        assert row["period"]["as_of"] == "2026-07-27"
        assert row["period"]["as_of_basis"] == "publisher_as_of_note"
        assert ".fytd." in row["layout"]["record_set_id"]


def test_cbp_unreadable_as_of_wording_fails_closed():
    """As-of wording the strict pattern cannot read must raise — never be
    treated as 'no note' with a silent retrieval-date fallback."""

    with pytest.raises(ValueError, match="cannot read"):
        parse_cbp_trade_stats(
            _fixture_with_note("FY 2026 data is updated as of the latest revision.")
        )


def test_cbp_displaced_note_fails_closed():
    """When the entry-summaries table carries no note of its own while
    as-of wording exists elsewhere on the page (every live page carries
    other tables' notes, e.g. the Trade Remedy 'All programs updated as
    of:' footnote), the parser must refuse rather than fall back — and
    must never adopt the other table's date."""

    html = CBP_FIXTURE.read_text().replace(
        _FIXTURE_NOTE, "Revenue collection statistics."
    )
    html = html.replace(
        'summary="Trade',
        'summary="Trade remedy"><tbody><tr><td>All programs updated as of: '
        "July 05, 2026.</td></tr></tbody></table>",
    )
    with pytest.raises(ValueError, match="no as-of note of its own"):
        parse_cbp_trade_stats(html.encode("utf-8"))


def test_cbp_conflicting_window_notes_fail_closed():
    with pytest.raises(ValueError, match="conflicting as-of dates"):
        parse_cbp_trade_stats(
            _fixture_with_note(
                "FY 2026 and FY 2025 are updated as of July 27, 2026. "
                "FY 2024 and FY 2023 are updated as of June 01, 2026."
            )
        )


def test_cbp_no_as_of_wording_anywhere_is_a_genuine_no_note_page():
    """Only a page with no as-of wording at all may return the empty note
    that authorizes the retrieval-date fallback."""

    stats = parse_cbp_trade_stats(_fixture_with_note("Revenue collection statistics."))
    assert stats.as_of_note == ""
    assert stats.as_of_date == ""


def test_cbp_entity_encoded_note_wording_is_read_not_fail_open():
    """The r3 probe: ``updated&nbsp;as of`` renders identically to
    ``updated as of`` but is byte-different, and the pre-fix scan saw
    neither sentinel nor note — so an October retrieval emitted all four
    FY2026 cells as completed fiscal years. Entity-encoded wording must
    parse exactly like its rendering, keeping the FYTD labeling."""

    stats = parse_cbp_trade_stats(
        _fixture_with_note("FY 2026 and FY 2025 is updated&nbsp;as of July 27, 2026.")
    )
    assert stats.as_of_date == "2026-07-27"
    rows = build_cbp_entry_fact_rows(
        stats,
        page_sha256="dd" * 32,
        retrieved_at="2026-10-01T00:00:00+00:00",
    )
    assert len(rows) == 4
    for row in rows:
        assert row["period"]["type"] == "fiscal_year_to_date"
        assert row["period"]["as_of"] == "2026-07-27"
        assert row["period"]["as_of_basis"] == "publisher_as_of_note"

    # The fail-closed side of the same hole: entity-encoded as-of wording
    # the strict pattern cannot read must still trip the sentinel and
    # raise — never scan as "no note" and fall back to the retrieval date.
    with pytest.raises(ValueError, match="cannot read"):
        parse_cbp_trade_stats(
            _fixture_with_note("Data is updated&#160;as of the latest revision.")
        )


def test_cbp_impossible_as_of_dates_fail_closed():
    """September 31 must never become a coverage endpoint: the pre-fix
    f-string minted '2026-09-31', which sorts after the FY2026 end and
    classified the in-progress year as complete."""

    with pytest.raises(ValueError, match="impossible calendar date"):
        parse_cbp_trade_stats(
            _fixture_with_note(
                "FY 2026 and FY 2025 is updated as of September 31, 2026."
            )
        )
    with pytest.raises(ValueError, match="impossible calendar date"):
        parse_cbp_trade_stats(
            _fixture_with_note(
                "FY 2026 and FY 2025 is updated as of February 29, 2026."
            )
        )


def test_cbp_fact_builder_refuses_non_date_coverage_endpoints():
    """Defense in depth at the consumer: an impossible as-of date on a
    hand-built stats object, or a malformed retrieval timestamp, must be
    refused — completeness is a calendar comparison, not a string one."""

    stats = parse_cbp_trade_stats(CBP_FIXTURE.read_bytes())
    doctored = CbpEntryStats(
        cells=stats.cells,
        as_of_note=stats.as_of_note,
        as_of_date="2026-09-31",
    )
    with pytest.raises(ValueError, match="not a real calendar date"):
        build_cbp_entry_fact_rows(
            doctored,
            page_sha256="dd" * 32,
            retrieved_at="2026-10-01T00:00:00+00:00",
        )
    noteless = CbpEntryStats(cells=stats.cells, as_of_note="", as_of_date="")
    with pytest.raises(ValueError, match="not a full ISO timestamp"):
        build_cbp_entry_fact_rows(
            noteless,
            page_sha256="dd" * 32,
            retrieved_at="not-a-timestamp",
        )


def test_cbp_note_without_date_is_refused_by_the_fact_builder():
    """Defense in depth at the consumer: stats carrying a note but no
    parsed date can only come from a hand-built object, and the builder
    refuses the retrieval-date fallback for them too."""

    stats = parse_cbp_trade_stats(CBP_FIXTURE.read_bytes())
    inconsistent = CbpEntryStats(
        cells=stats.cells,
        as_of_note="FY 2026 is updated as of an unknown date",
        as_of_date="",
    )
    with pytest.raises(ValueError, match="without a parsed as-of date"):
        build_cbp_entry_fact_rows(
            inconsistent,
            page_sha256="dd" * 32,
            retrieved_at="2026-08-05T00:00:00+00:00",
        )


def test_grain_catalog_is_the_documented_set():
    assert IMPORT_ENTRY_FACT_GRAINS == (
        "national",
        "chapter",
        "country",
        "chapter_country",
    )


def test_facts_refuse_months_without_manifest_coverage():
    with pytest.raises(ValueError, match="No retrieval-manifest entry covers"):
        build_import_entry_fact_rows(
            _margins(),
            retrieval_manifest=[
                entry for entry in _manifest_entries() if entry["month"] != "2026-02"
            ],
            extracted_at="2026-08-05T00:00:00+00:00",
        )


def test_artifact_refuses_empty_retrieval_manifest(tmp_path):
    rows = build_import_entry_fact_rows(
        _margins(),
        retrieval_manifest=_manifest_entries(),
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="empty retrieval manifest"):
        write_consumer_artifact(
            tmp_path / "artifact",
            rows,
            retrieval_manifest=(),
            generator={},
        )


def test_cbp_semantic_keys_are_fiscal_year_invariant():
    from populace.build.us_runtime.us_trade.import_entry_facts import (
        _period_invariant_record_set,
    )

    # FYTD snapshots: the family must drop the year and as-of endpoint but
    # keep the fytd marker, so "latest eligible period" selectors over
    # completed fiscal years can never resolve to a partial-year snapshot.
    stats = parse_cbp_trade_stats(CBP_FIXTURE.read_bytes())
    rows = build_cbp_entry_fact_rows(
        stats,
        page_sha256="dd" * 32,
        retrieved_at="2026-08-05T00:00:00+00:00",
    )
    entries = next(
        row for row in rows if row["layout"]["measure_id"] == "total_entry_summaries"
    )
    fytd_family = _period_invariant_record_set(entries["layout"]["record_set_id"])
    assert fytd_family == "cbp_trade_stats.imports_revenue_collection.fytd"

    # Completed fiscal years: identity must survive the year rolling over.
    (complete,) = build_cbp_entry_fact_rows(
        _fy2025_complete_stats(),
        page_sha256="dd" * 32,
        retrieved_at="2026-08-05T00:00:00+00:00",
    )
    complete_family = _period_invariant_record_set(complete["layout"]["record_set_id"])
    assert complete_family == "cbp_trade_stats.imports_revenue_collection"
    assert complete_family != fytd_family


def test_cbp_and_district_facts_compile_into_ledger_targets(tmp_path):
    stats = parse_cbp_trade_stats(CBP_FIXTURE.read_bytes())
    cbp_rows = build_cbp_entry_fact_rows(
        stats,
        page_sha256="dd" * 32,
        retrieved_at="2026-08-05T00:00:00+00:00",
    )
    district = pd.DataFrame(
        [
            {
                "period": "2026-01",
                "dist_entry": "70",
                "dist_name": "Low Value",
                "con_val_mo": 2_000_000,
                "gen_val_mo": 2_100_000,
                "cal_dut_mo": 0,
                "dut_val_mo": 0,
                "air_val_mo": 0,
                "ves_val_mo": 0,
                "cnt_val_mo": 0,
            }
        ]
    )
    from populace.build.us_runtime.us_trade.import_entry_facts import (
        build_district_entry_fact_rows,
    )

    district_rows = build_district_entry_fact_rows(
        district,
        retrieval_manifest=_manifest_entries(),
        extracted_at="2026-08-05T00:00:00+00:00",
    )
    assert all(
        row["dimensions"] == {"district_of_entry": "70"} for row in district_rows
    )
    assert all("Low Value" in row["label"] for row in district_rows)

    references = [
        LedgerTargetReference(
            name="cbp_total_entry_summaries_fytd2026",
            ledger_selector={
                "source_record_id": (
                    "cbp_trade_stats.imports_revenue_collection.fytd"
                    ".fy2026_as_of_2026_07_27.total_entry_summaries"
                ),
            },
            entity="import_entry",
            measure="entry_summaries",
            period="2026",
            family="trade_imports",
        ),
        LedgerTargetReference(
            name="district70_customs_value_2026_01",
            ledger_selector={
                "source_record_id": (
                    "census_intltrade.imports_district_entry.month_2026_01"
                    ".de70.con_val_mo"
                ),
            },
            entity="import_entry",
            measure="customs_value",
            period="2026-01",
            family="trade_imports",
        ),
    ]
    registry = compile_ledger_target_references(
        cbp_rows + district_rows, references, country="us"
    )
    by_name = {spec.name: spec for spec in registry.specs}
    assert by_name["cbp_total_entry_summaries_fytd2026"].value == 83_133_856
    assert by_name["district70_customs_value_2026_01"].value == 2_000_000
