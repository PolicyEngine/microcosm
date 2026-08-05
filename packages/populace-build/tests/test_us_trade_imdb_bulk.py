"""Tests for the Census bulk IMDB ingest (the primary P1 source)."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from populace.build.ledger_artifact import load_ledger_consumer_artifact
from populace.build.us_trade.census_country_bridge import load_census_country_bridge
from populace.build.us_trade.imdb_bulk import (
    assemble_bulk_margins,
    ensure_imdb_archive,
    imdb_archive_name,
    imdb_archive_url,
    latest_available_imdb_month,
    load_imdb_month,
)
from populace.build.us_trade.import_entry_facts import (
    IMDB_BULK_SOURCE_LEG,
    build_district_entry_fact_rows,
    build_import_entry_fact_rows,
    default_generator_block,
    write_consumer_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_CLI = REPO_ROOT / "tools" / "build_us_import_entry_margins.py"


def _field(line: list[str], start: int, end: int, value: str, *, align: str) -> None:
    """Place ``value`` into the 1-indexed inclusive [start, end] span."""
    width = end - start + 1
    text = value.rjust(width) if align == "right" else value.ljust(width)
    assert len(text) == width, f"{value!r} does not fit {width}"
    line[start - 1 : end] = list(text)


def _detail_line(
    hts10: str,
    cty: str,
    subco: str,
    dist_entry: str,
    dist_unlad: str,
    rate_prov: str,
    year: str,
    month: str,
    measures: dict[str, int],
) -> str:
    line = [" "] * 688
    _field(line, 1, 10, hts10, align="left")
    _field(line, 11, 14, cty, align="left")
    _field(line, 15, 16, subco, align="left")
    _field(line, 17, 18, dist_entry, align="left")
    _field(line, 19, 20, dist_unlad, align="left")
    _field(line, 21, 22, rate_prov, align="left")
    _field(line, 23, 26, year, align="left")
    _field(line, 27, 28, month, align="left")
    spans = {
        "cards_mo": (29, 43),
        "con_qy1_mo": (44, 58),
        "con_qy2_mo": (59, 73),
        "con_val_mo": (74, 88),
        "dut_val_mo": (89, 103),
        "cal_dut_mo": (104, 118),
        "con_cha_mo": (119, 133),
        "con_cif_mo": (134, 148),
        "gen_qy1_mo": (149, 163),
        "gen_qy2_mo": (164, 178),
        "gen_val_mo": (179, 193),
        "gen_cha_mo": (194, 208),
        "gen_cif_mo": (209, 223),
        "air_val_mo": (224, 238),
        "air_wgt_mo": (239, 253),
        "air_cha_mo": (254, 268),
        "ves_val_mo": (269, 283),
        "ves_wgt_mo": (284, 298),
        "ves_cha_mo": (299, 313),
        "cnt_val_mo": (314, 328),
        "cnt_wgt_mo": (329, 343),
        "cnt_cha_mo": (344, 358),
    }
    for name, (start, end) in spans.items():
        _field(line, start, end, str(measures.get(name, 0)), align="right")
    # Year-to-date mirror region (359-688): zero-filled like the published
    # files; the parser never reads it.
    for start in range(359, 689, 15):
        _field(line, start, min(start + 14, 688), "0", align="right")
    return "".join(line)


def _cty_line(
    cty: str, name: str, year: str, month: str, values: dict[str, int]
) -> str:
    line = [" "] * 580
    _field(line, 1, 4, cty, align="left")
    _field(line, 5, 34, name, align="left")
    _field(line, 35, 38, year, align="left")
    _field(line, 39, 40, month, align="left")
    spans = {
        "cards_mo": (41, 55),
        "con_val_mo": (56, 70),
        "dut_val_mo": (71, 85),
        "cal_dut_mo": (86, 100),
        "con_cha_mo": (101, 115),
        "con_cif_mo": (116, 130),
        "gen_val_mo": (131, 145),
        "gen_cha_mo": (146, 160),
        "gen_cif_mo": (161, 175),
        "air_val_mo": (176, 190),
        "air_wgt_mo": (191, 205),
        "air_cha_mo": (206, 220),
        "ves_val_mo": (221, 235),
        "ves_wgt_mo": (236, 250),
        "ves_cha_mo": (251, 265),
        "cnt_val_mo": (266, 280),
    }
    for name_, (start, end) in spans.items():
        _field(line, start, end, str(values.get(name_, 0)), align="right")
    return "".join(line)


def _comm_line(
    hts10: str,
    desc: str,
    unit1: str,
    unit2: str,
    year: str,
    month: str,
    values: dict[str, int],
) -> str:
    line = [" "] * 672
    _field(line, 1, 10, hts10, align="left")
    _field(line, 11, 60, desc, align="left")
    _field(line, 61, 63, unit1, align="left")
    _field(line, 64, 66, unit2, align="left")
    _field(line, 67, 70, year, align="left")
    _field(line, 71, 72, month, align="left")
    spans = {
        "cards_mo": (73, 87),
        "con_qy1_mo": (88, 102),
        "con_qy2_mo": (103, 117),
        "con_val_mo": (118, 132),
        "dut_val_mo": (133, 147),
        "cal_dut_mo": (148, 162),
        "con_cha_mo": (163, 177),
        "con_cif_mo": (178, 192),
        "gen_qy1_mo": (193, 207),
        "gen_qy2_mo": (208, 222),
        "gen_val_mo": (223, 237),
    }
    for name_, (start, end) in spans.items():
        _field(line, start, end, str(values.get(name_, 0)), align="right")
    return "".join(line)


def _de_line(
    code: str, name: str, year: str, month: str, values: dict[str, int]
) -> str:
    line = [" "] * 460
    _field(line, 1, 2, code, align="left")
    _field(line, 3, 32, name, align="left")
    _field(line, 33, 36, year, align="left")
    _field(line, 37, 38, month, align="left")
    spans = {
        "cards_mo": (39, 53),
        "con_val_mo": (54, 68),
        "dut_val_mo": (69, 83),
        "cal_dut_mo": (84, 98),
        "con_cha_mo": (99, 113),
        "con_cif_mo": (114, 128),
        "gen_val_mo": (129, 143),
        "gen_cha_mo": (144, 158),
        "gen_cif_mo": (159, 173),
        "air_val_mo": (174, 188),
        "air_wgt_mo": (189, 203),
        "air_cha_mo": (204, 218),
        "ves_val_mo": (219, 233),
        "ves_wgt_mo": (234, 248),
        "ves_cha_mo": (249, 263),
        "cnt_val_mo": (264, 278),
    }
    for name_, (start, end) in spans.items():
        _field(line, start, end, str(values.get(name_, 0)), align="right")
    return "".join(line)


#: Fixture universe for 2026-01: two commodities, two Schedule C countries
#: (1220 = Canada, 5700 = China per the vendored bridge), two districts,
#: detail split over districts and rate provisions.
_DETAIL_ROWS = (
    (
        "0101210010",
        "1220",
        "00",
        "07",
        "07",
        "10",
        {
            "cards_mo": 1,
            "con_qy1_mo": 2,
            "con_val_mo": 600,
            "dut_val_mo": 0,
            "cal_dut_mo": 0,
            "gen_qy1_mo": 2,
            "gen_val_mo": 600,
            "air_val_mo": 100,
            "ves_val_mo": 500,
            "cnt_val_mo": 400,
        },
    ),
    (
        "0101210010",
        "1220",
        "A ",
        "20",
        "20",
        "61",
        {
            "cards_mo": 1,
            "con_qy1_mo": 1,
            "con_val_mo": 400,
            "dut_val_mo": 400,
            "cal_dut_mo": 25,
            "gen_qy1_mo": 1,
            "gen_val_mo": 400,
            "ves_val_mo": 400,
        },
    ),
    (
        "0101210010",
        "5700",
        "00",
        "07",
        "07",
        "10",
        {
            "cards_mo": 1,
            "con_qy1_mo": 1,
            "con_val_mo": 250,
            "dut_val_mo": 0,
            "cal_dut_mo": 0,
            "gen_qy1_mo": 2,
            "gen_val_mo": 300,
            "air_val_mo": 300,
        },
    ),
    (
        "8471300100",
        "5700",
        "00",
        "20",
        "20",
        "61",
        {
            "cards_mo": 3,
            "con_qy1_mo": 10,
            "con_val_mo": 5000,
            "dut_val_mo": 5000,
            "cal_dut_mo": 250,
            "gen_qy1_mo": 10,
            "gen_val_mo": 5000,
            "ves_val_mo": 5000,
            "cnt_val_mo": 5000,
        },
    ),
    # Year-to-date carrier cell: active earlier in the statistical year,
    # all-zero monthly measures — present in the published union, excluded
    # from the margins table.
    ("8471300100", "1220", "00", "07", "07", "10", {}),
)


def _fixture_zip_bytes(
    *,
    year: str = "2026",
    month: str = "01",
    detail_rows: tuple = _DETAIL_ROWS,
    cty_overrides: dict[str, dict[str, int]] | None = None,
    extra_detail_lines: tuple[str, ...] = (),
) -> bytes:
    detail_lines = [
        _detail_line(hts10, cty, subco, de, du, rp, year, month, measures)
        for hts10, cty, subco, de, du, rp, measures in detail_rows
    ]
    detail_lines += list(extra_detail_lines)
    cty_values = {
        "1220": {
            "con_val_mo": 1000,
            "dut_val_mo": 400,
            "cal_dut_mo": 25,
            "gen_val_mo": 1000,
            "air_val_mo": 100,
            "ves_val_mo": 900,
            "cnt_val_mo": 400,
            "cards_mo": 2,
        },
        "5700": {
            "con_val_mo": 5250,
            "dut_val_mo": 5000,
            "cal_dut_mo": 250,
            "gen_val_mo": 5300,
            "air_val_mo": 300,
            "ves_val_mo": 5000,
            "cnt_val_mo": 5000,
            "cards_mo": 4,
        },
    }
    for code, overrides in (cty_overrides or {}).items():
        cty_values[code].update(overrides)
    cty_lines = [
        _cty_line("1220", "Canada", year, month, cty_values["1220"]),
        _cty_line("5700", "China", year, month, cty_values["5700"]),
    ]
    comm_lines = [
        _comm_line(
            "0101210010",
            "HORSES, LIVE, PUREBRED BREEDING MALE",
            "NO",
            "",
            year,
            month,
            {
                "cards_mo": 3,
                "con_qy1_mo": 4,
                "con_val_mo": 1250,
                "dut_val_mo": 400,
                "cal_dut_mo": 25,
                "gen_qy1_mo": 5,
                "gen_val_mo": 1300,
            },
        ),
        _comm_line(
            "8471300100",
            "PORTABLE AUTOMATIC DATA PROCESSING MACHINES",
            "NO",
            "",
            year,
            month,
            {
                "cards_mo": 3,
                "con_qy1_mo": 10,
                "con_val_mo": 5000,
                "dut_val_mo": 5000,
                "cal_dut_mo": 250,
                "gen_qy1_mo": 10,
                "gen_val_mo": 5000,
            },
        ),
    ]
    de_lines = [
        _de_line(
            "07",
            "OGDENSBURG, NY",
            year,
            month,
            {
                "cards_mo": 2,
                "con_val_mo": 850,
                "dut_val_mo": 0,
                "cal_dut_mo": 0,
                "gen_val_mo": 900,
                "air_val_mo": 400,
                "ves_val_mo": 500,
                "cnt_val_mo": 400,
            },
        ),
        _de_line(
            "20",
            "NEW ORLEANS, LA",
            year,
            month,
            {
                "cards_mo": 4,
                "con_val_mo": 5400,
                "dut_val_mo": 5400,
                "cal_dut_mo": 275,
                "gen_val_mo": 5400,
                "ves_val_mo": 5400,
                "cnt_val_mo": 5000,
            },
        ),
    ]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("IMP_DETL.txt", "\r\n".join(detail_lines) + "\r\n")
        # The country file ships with mixed case in the real archives.
        bundle.writestr("imp_CTY.txt", "\r\n".join(cty_lines) + "\r\n")
        bundle.writestr("IMP_COMM.txt", "\r\n".join(comm_lines) + "\r\n")
        bundle.writestr("IMP_DE.txt", "\r\n".join(de_lines) + "\r\n")
        bundle.writestr("Documentation/IMP_DETL.lay", "reference layout\n")
    return buffer.getvalue()


def _fixture_month(tmp_path: Path, **kwargs):
    archive = tmp_path / "IMDB2601.ZIP"
    archive.write_bytes(_fixture_zip_bytes(**kwargs))
    entry = {
        "source_name": "census_imdb_bulk",
        "month": "2026-01",
        "filename": archive.name,
        "sha256": "ab" * 32,
        "url": imdb_archive_url("2026-01"),
        "retrieved_at": "2026-08-05T12:00:00+00:00",
    }
    return load_imdb_month(archive, "2026-01", entry)


def test_archive_naming_matches_published_pattern():
    assert imdb_archive_name("2025-01") == "IMDB2501.ZIP"
    assert imdb_archive_url("2026-06") == (
        "https://www.census.gov/trade/downloads/2026/Merch/im_m/IMDB2606.ZIP"
    )
    with pytest.raises(ValueError, match="YYYY-MM"):
        imdb_archive_name("202601")


def test_load_parses_detail_and_reconciles_exactly(tmp_path):
    month = _fixture_month(tmp_path)
    assert month.reconciliation_failures == ()
    assert len(month.detail) == 5
    assert set(month.detail["rate_prov"]) == {"10", "61"}
    assert month.detail["con_val_mo"].sum() == 6250
    assert len(month.control_cty) == 2
    assert len(month.control_comm) == 2
    assert len(month.control_de) == 2


def test_assembly_aggregates_margins_and_joins_units(tmp_path):
    month = _fixture_month(tmp_path)
    assembly = assemble_bulk_margins((month,), load_census_country_bridge())
    margins = assembly.margins
    # The YTD-carrier cell (8471300100 x 1220) is excluded: no monthly activity.
    assert len(margins) == 3
    assert not (
        (margins["hts10"] == "8471300100") & (margins["cty_code"] == "1220")
    ).any()
    cell = margins.set_index(["hts10", "cty_code"]).loc[("0101210010", "1220")]
    assert cell["con_val_mo"] == 1000
    assert cell["dut_val_mo"] == 400
    assert cell["cal_dut_mo"] == 25
    assert cell["gen_val_mo"] == 1000
    assert cell["con_qy1_mo"] == 3
    assert cell["cards_mo"] == 2
    assert cell["ves_val_mo"] == 900
    assert cell["iso2"] == "CA"
    assert cell["unit_qy1"] == "NO"
    assert cell["chapter"] == "01"
    china = margins.set_index(["hts10", "cty_code"]).loc[("8471300100", "5700")]
    assert china["iso2"] == "CN"
    totals = assembly.census_totals.set_index("hts10")
    assert totals.loc["0101210010", "con_val_mo"] == 1250
    district = assembly.district_entry.set_index("dist_entry")
    assert district.loc["20", "con_val_mo"] == 5400
    assert district.loc["07", "dist_name"] == "OGDENSBURG, NY"


def test_control_total_mismatch_fails_the_ingest(tmp_path):
    month = _fixture_month(tmp_path, cty_overrides={"1220": {"con_val_mo": 999}})
    assert any(
        "cty_code=1220 con_val_mo: detail sums to 1000, published control "
        "total is 999" in failure
        for failure in month.reconciliation_failures
    )


def test_off_period_rows_fail_the_ingest(tmp_path):
    stray = _detail_line(
        "0101210010",
        "1220",
        "00",
        "07",
        "07",
        "10",
        "2025",
        "12",
        {"con_val_mo": 1},
    )
    month = _fixture_month(tmp_path, extra_detail_lines=(stray,))
    assert any(
        "statistical period other than 2026-01" in failure
        for failure in month.reconciliation_failures
    )


def test_non_schedule_c_country_code_fails_the_ingest(tmp_path):
    stray = _detail_line(
        "0101210010",
        "0500",
        "00",
        "07",
        "07",
        "10",
        "2026",
        "01",
        {"con_val_mo": 1},
    )
    month = _fixture_month(tmp_path, extra_detail_lines=(stray,))
    assert any("outside Schedule C" in f for f in month.reconciliation_failures)


def test_truncated_lines_fail_loudly(tmp_path):
    truncated = _detail_line(
        "0101210010",
        "1220",
        "00",
        "07",
        "07",
        "10",
        "2026",
        "01",
        {"con_val_mo": 1},
    )[:200]
    archive = tmp_path / "IMDB2601.ZIP"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("IMP_DETL.txt", truncated + "\r\n")
        bundle.writestr("imp_CTY.txt", "")
        bundle.writestr("IMP_COMM.txt", "")
        bundle.writestr("IMP_DE.txt", "")
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="blank|shorter|zero rows"):
        load_imdb_month(archive, "2026-01", {})


def test_missing_detail_member_fails_loudly(tmp_path):
    archive = tmp_path / "IMDB2601.ZIP"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("README.txt", "no data here")
    archive.write_bytes(buffer.getvalue())
    with pytest.raises(ValueError, match="exactly one top-level IMP_DETL.txt"):
        load_imdb_month(archive, "2026-01", {})


def test_ensure_archive_downloads_verifies_and_adopts(tmp_path):
    payload = _fixture_zip_bytes()
    calls: list[str] = []

    def fake_fetch(url: str) -> bytes:
        calls.append(url)
        return payload

    path, entry = ensure_imdb_archive(
        "2026-01", tmp_path / "archives", fetch=fake_fetch
    )
    assert path.name == "IMDB2601.ZIP"
    assert calls == [imdb_archive_url("2026-01")]
    assert entry["size_bytes"] == len(payload)
    assert entry["retrieval_note"] == "downloaded by this build"
    import hashlib

    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()

    # Second call adopts the existing file without fetching; a supplied
    # download-manifest timestamp is carried into the entry.
    path2, entry2 = ensure_imdb_archive(
        "2026-01",
        tmp_path / "archives",
        retrieved_at_by_name={"IMDB2601.ZIP": "2026-08-05T11:41:00Z"},
        fetch=fake_fetch,
    )
    assert path2 == path
    assert calls == [imdb_archive_url("2026-01")]
    assert entry2["retrieved_at"] == "2026-08-05T11:41:00Z"
    assert "download manifest" in str(entry2["retrieval_note"])
    assert entry2["sha256"] == entry["sha256"]


def test_ensure_archive_rejects_junk_downloads(tmp_path, monkeypatch):
    import populace.build.us_trade.imdb_bulk as module

    monkeypatch.setattr(module, "_DOWNLOAD_BACKOFF_SECONDS", 0.0)
    with pytest.raises(RuntimeError, match="Could not download a valid IMDB"):
        ensure_imdb_archive(
            "2026-01", tmp_path / "archives", fetch=lambda url: b"not a zip"
        )


def test_latest_available_month_probes_backward():
    statuses = {
        imdb_archive_url("2026-08"): 404,
        imdb_archive_url("2026-07"): 404,
        imdb_archive_url("2026-06"): 200,
    }
    from datetime import UTC, datetime

    month = latest_available_imdb_month(
        now=datetime(2026, 8, 5, tzinfo=UTC), head=lambda url: statuses[url]
    )
    assert month == "2026-06"


def test_bulk_fact_rows_carry_archive_identity_and_stable_record_sets(tmp_path):
    month = _fixture_month(tmp_path)
    assembly = assemble_bulk_margins((month,), load_census_country_bridge())
    rows = build_import_entry_fact_rows(
        assembly.margins,
        retrieval_manifest=assembly.manifest_entries,
        extracted_at="2026-08-05T12:00:00+00:00",
    )
    by_id = {row["lineage"]["source_record_id"]: row for row in rows}
    national = by_id[
        "census_intltrade.imports_hs10.month_2026_01.national.all.con_val_mo"
    ]
    assert national["value"] == 6250
    assert national["source"]["source_file"] == "IMDB2601.ZIP"
    assert national["source"]["source_sha256"] == "ab" * 32
    assert national["lineage"]["source_file_sha256s"] == ["ab" * 32]
    assert "imdb_bulk ingest" in national["source"]["extraction_method"]
    assert "national grain" in national["source"]["extraction_method"]
    chapter = by_id[
        "census_intltrade.imports_hs10.month_2026_01.chapter.ch84.con_val_mo"
    ]
    assert chapter["value"] == 5000
    assert chapter["source"]["source_file"] == "IMDB2601.ZIP"
    assert chapter["source"]["source_sha256"] == "ab" * 32


def test_district_facts_round_trip_with_margin_facts(tmp_path):
    month = _fixture_month(tmp_path)
    assembly = assemble_bulk_margins((month,), load_census_country_bridge())
    rows = build_import_entry_fact_rows(
        assembly.margins,
        retrieval_manifest=assembly.manifest_entries,
        extracted_at="2026-08-05T12:00:00+00:00",
    )
    district_rows = build_district_entry_fact_rows(
        assembly.district_entry,
        retrieval_manifest=assembly.manifest_entries,
        extracted_at="2026-08-05T12:00:00+00:00",
    )
    by_id = {row["lineage"]["source_record_id"]: row for row in district_rows}
    portland = by_id[
        "census_intltrade.imports_district_entry.month_2026_01.de07.con_val_mo"
    ]
    assert portland["value"] == 850
    assert portland["dimensions"] == {
        "district_of_entry": "07",
        "district_name": "OGDENSBURG, NY",
    }
    assert portland["concept_alignment"]["relation"] == "exact"
    duty = by_id[
        "census_intltrade.imports_district_entry.month_2026_01.de20.cal_dut_mo"
    ]
    assert duty["value"] == 275
    assert "concept_alignment" in duty

    manifest = write_consumer_artifact(
        tmp_path / "artifact",
        rows + district_rows,
        retrieval_manifest=assembly.manifest_entries,
        generator=default_generator_block(months=("2026-01",)),
    )
    artifact = load_ledger_consumer_artifact(
        tmp_path / "artifact", expected_facts_sha256=manifest["facts_sha256"]
    )
    assert artifact.fact_row_count == len(rows) + len(district_rows)


def test_source_leg_formats_grain_into_extraction_method():
    assert "{grain}" in IMDB_BULK_SOURCE_LEG.extraction_method
    formatted = IMDB_BULK_SOURCE_LEG.extraction_method.format(grain="chapter")
    assert "chapter grain" in formatted
    assert "populace-minted" in formatted


def test_build_cli_end_to_end_offline(tmp_path):
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "IMDB2601.ZIP").write_bytes(_fixture_zip_bytes())
    download_manifest = tmp_path / "download-manifest.jsonl"
    download_manifest.write_text(
        json.dumps({"file": "IMDB2601.ZIP", "retrieved_at_utc": "2026-08-05T11:41:00Z"})
        + "\n"
    )
    out_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_CLI),
            "--start",
            "2026-01",
            "--end",
            "2026-01",
            "--archive-dir",
            str(archive_dir),
            "--out-dir",
            str(out_dir),
            "--download-manifest",
            str(download_manifest),
            "--skip-cbp",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads((out_dir / "build_report.json").read_text())
    assert report["source"] == "census_imdb_bulk"
    assert report["margin_rows"] == 3
    assert report["district_rows"] == 2
    assert report["reconciliation_failures"] == 0
    assert (out_dir / "margins_hts10_country_month.parquet").exists()
    assert (out_dir / "district_entry_month.parquet").exists()
    assert (out_dir / "detail" / "period=2026-01.parquet").exists()
    artifact = load_ledger_consumer_artifact(
        out_dir / "consumer_artifact",
        expected_facts_sha256=report["facts_sha256"],
    )
    assert artifact.fact_row_count == report["fact_rows"]
    manifest = json.loads((out_dir / "consumer_artifact" / "manifest.json").read_text())
    retrieved = manifest["source_manifest"]["retrievals"][0]
    assert retrieved["filename"] == "IMDB2601.ZIP"
    assert retrieved["retrieved_at"] == "2026-08-05T11:41:00Z"


def test_build_cli_fails_on_control_mismatch(tmp_path):
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "IMDB2601.ZIP").write_bytes(
        _fixture_zip_bytes(cty_overrides={"5700": {"cal_dut_mo": 9}})
    )
    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_CLI),
            "--start",
            "2026-01",
            "--end",
            "2026-01",
            "--archive-dir",
            str(archive_dir),
            "--out-dir",
            str(tmp_path / "out"),
            "--skip-cbp",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 1
    assert "RECONCILIATION FAIL" in result.stderr
