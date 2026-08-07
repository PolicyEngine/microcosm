"""Tests for the Census bulk IMDB ingest (the primary P1 source)."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from populace.build.ledger_artifact import load_ledger_consumer_artifact
from populace.build.us_runtime.us_trade.census_country_bridge import (
    load_census_country_bridge,
)
from populace.build.us_runtime.us_trade.imdb_bulk import (
    assemble_bulk_margins,
    ensure_imdb_archive,
    imdb_archive_name,
    imdb_archive_url,
    latest_available_imdb_month,
    load_imdb_month,
    summarize_imdb_month,
)
from populace.build.us_runtime.us_trade.import_entry_facts import (
    IMDB_BULK_SOURCE_LEG,
    IMDB_DISTRICT_SOURCE_LEG,
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
        "cnt_wgt_mo": (281, 295),
        "cnt_cha_mo": (296, 310),
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
        "gen_cha_mo": (238, 252),
        "gen_cif_mo": (253, 267),
        "air_val_mo": (268, 282),
        "air_wgt_mo": (283, 297),
        "air_cha_mo": (298, 312),
        "ves_val_mo": (313, 327),
        "ves_wgt_mo": (328, 342),
        "ves_cha_mo": (343, 357),
        "cnt_val_mo": (358, 372),
        "cnt_wgt_mo": (373, 387),
        "cnt_cha_mo": (388, 402),
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
        "cnt_wgt_mo": (279, 293),
        "cnt_cha_mo": (294, 308),
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
    # Full-width measures (Japan/semiconductors): every 15-character slice
    # is fully occupied, so a one-character colspec shift in any direction
    # corrupts at least one parsed value and must fail reconciliation.
    (
        "8542310045",
        "5880",
        "00",
        "55",
        "55",
        "61",
        {
            "cards_mo": 999999999999999,
            "con_qy1_mo": 999999999999999,
            "con_qy2_mo": 999999999999999,
            "con_val_mo": 123456789012345,
            "dut_val_mo": 123456789012345,
            "cal_dut_mo": 987654321098765,
            "con_cha_mo": 111111111111111,
            "con_cif_mo": 222222222222222,
            "gen_qy1_mo": 999999999999999,
            "gen_qy2_mo": 999999999999999,
            "gen_val_mo": 123456789012345,
            "gen_cha_mo": 333333333333333,
            "gen_cif_mo": 444444444444444,
            "air_val_mo": 555555555555555,
            "air_wgt_mo": 666666666666666,
            "air_cha_mo": 777777777777777,
            "ves_val_mo": 888888888888888,
            "ves_wgt_mo": 999999999999999,
            "ves_cha_mo": 123123123123123,
            "cnt_val_mo": 456456456456456,
            "cnt_wgt_mo": 789789789789789,
            "cnt_cha_mo": 987987987987987,
        },
    ),
    # Year-to-date carrier cell: active earlier in the statistical year,
    # all-zero monthly measures — present in the published union, excluded
    # from the margins table.
    ("8471300100", "1220", "00", "07", "07", "10", {}),
)

_FULL_WIDTH = _DETAIL_ROWS[4][6]


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
        "5880": dict(_FULL_WIDTH),
    }
    for code, overrides in (cty_overrides or {}).items():
        cty_values[code].update(overrides)
    cty_lines = [
        _cty_line("1220", "Canada", year, month, cty_values["1220"]),
        _cty_line("5700", "China", year, month, cty_values["5700"]),
        _cty_line(
            "5880",
            "Japan",
            year,
            month,
            {k: v for k, v in cty_values["5880"].items() if "qy" not in k},
        ),
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
                "air_val_mo": 400,
                "ves_val_mo": 900,
                "cnt_val_mo": 400,
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
                "ves_val_mo": 5000,
                "cnt_val_mo": 5000,
            },
        ),
        _comm_line(
            "8542310045",
            "ELECTRONIC INTEGRATED CIRCUITS, PROCESSORS",
            "NO",
            "",
            year,
            month,
            dict(_FULL_WIDTH),
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
        _de_line(
            "55",
            "DULUTH, MN",
            year,
            month,
            {k: v for k, v in _FULL_WIDTH.items() if "qy" not in k},
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
    assert len(month.detail) == 6
    assert set(month.detail["rate_prov"]) == {"10", "61"}
    assert month.detail["con_val_mo"].sum() == 6250 + 123_456_789_012_345
    assert len(month.control_cty) == 3
    assert len(month.control_comm) == 3
    assert len(month.control_de) == 3


def test_assembly_aggregates_margins_and_joins_units(tmp_path):
    month = _fixture_month(tmp_path)
    assembly = assemble_bulk_margins(
        (summarize_imdb_month(month),), load_census_country_bridge()
    )
    margins = assembly.margins
    # The YTD-carrier cell (8471300100 x 1220) is excluded: no monthly activity.
    assert len(margins) == 4
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
    # download-manifest timestamp is carried when its hash matches.
    path2, entry2 = ensure_imdb_archive(
        "2026-01",
        tmp_path / "archives",
        retrieved_at_by_sha={
            ("IMDB2601.ZIP", str(entry["sha256"])): "2026-08-05T11:41:00Z"
        },
        fetch=fake_fetch,
    )
    assert path2 == path
    assert calls == [imdb_archive_url("2026-01")]
    assert entry2["retrieved_at"] == "2026-08-05T11:41:00Z"
    assert "download manifest" in str(entry2["retrieval_note"])
    assert entry2["sha256"] == entry["sha256"]
    assert "http_status" not in entry2
    assert entry["http_status"] == 200


def test_ensure_archive_rejects_junk_downloads(tmp_path, monkeypatch):
    import populace.build.us_runtime.us_trade.imdb_bulk as module

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
    assembly = assemble_bulk_margins(
        (summarize_imdb_month(month),), load_census_country_bridge()
    )
    rows = build_import_entry_fact_rows(
        assembly.margins,
        retrieval_manifest=assembly.manifest_entries,
        extracted_at="2026-08-05T12:00:00+00:00",
    )
    by_id = {row["lineage"]["source_record_id"]: row for row in rows}
    national = by_id[
        "census_intltrade.imports_hs10.month_2026_01.national.all.con_val_mo"
    ]
    assert national["value"] == 6250 + 123_456_789_012_345
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
    assembly = assemble_bulk_margins(
        (summarize_imdb_month(month),), load_census_country_bridge()
    )
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
    assert portland["dimensions"] == {"district_of_entry": "07"}
    assert "OGDENSBURG, NY" in portland["label"]
    assert portland["concept_alignment"]["relation"] == "exact"
    duty = by_id[
        "census_intltrade.imports_district_entry.month_2026_01.de20.cal_dut_mo"
    ]
    assert duty["value"] == 275
    assert "concept_alignment" in duty
    # District values come from the IMP_DE control file, and both source
    # members must say so; the margins feed keeps its IMP_DETL identity.
    for row in district_rows:
        assert (
            "IMP_DE district-of-entry control totals"
            in (row["observed_measure"]["source_table"])
        )
        assert (
            row["source"]["source_table"] == (row["observed_measure"]["source_table"])
        )
        assert "IMP_DETL" not in row["observed_measure"]["source_table"]
        assert "district_entry grain" in row["source"]["extraction_method"]
    assert all(
        "IMP_DETL fixed-width detail" in row["observed_measure"]["source_table"]
        for row in rows
    )

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
    assert "{grain}" in IMDB_DISTRICT_SOURCE_LEG.extraction_method
    district = IMDB_DISTRICT_SOURCE_LEG.extraction_method.format(grain="district_entry")
    assert "district_entry grain" in district
    assert "IMP_DE" in IMDB_DISTRICT_SOURCE_LEG.source_table


def test_build_cli_end_to_end_offline(tmp_path):
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "IMDB2601.ZIP").write_bytes(_fixture_zip_bytes())
    import hashlib

    payload = (archive_dir / "IMDB2601.ZIP").read_bytes()
    download_manifest = tmp_path / "download-manifest.jsonl"
    download_manifest.write_text(
        json.dumps(
            {
                "file": "IMDB2601.ZIP",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "retrieved_at_utc": "2026-08-05T11:41:00Z",
            }
        )
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
    assert report["margin_rows"] == 4
    assert report["district_rows"] == 3
    assert report["reconciliation_failures"] == 0
    assert (out_dir / "margins_hts10_country_month.parquet").exists()
    assert (out_dir / "district_entry_month.parquet").exists()
    assert (out_dir / "detail" / "period=2026-01.parquet").exists()
    artifact = load_ledger_consumer_artifact(
        out_dir / "consumer_artifact",
        expected_facts_sha256=report["facts_sha256"],
    )
    assert artifact.fact_row_count == report["fact_rows"]
    manifest_path = out_dir / "consumer_artifact" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    retrieved = manifest["source_manifest"]["retrievals"][0]
    assert retrieved["filename"] == "IMDB2601.ZIP"
    assert retrieved["retrieved_at"] == "2026-08-05T11:41:00Z"
    # The build report pins the consumer manifest's own bytes, and the
    # machine-readable reconciliation evidence + source manifest publish
    # beside the artifacts they certify.
    import hashlib as hashlib_module

    assert report["consumer_manifest_sha256"] == (
        hashlib_module.sha256(manifest_path.read_bytes()).hexdigest()
    )
    evidence = json.loads(
        (out_dir / "reconciliation" / "period=2026-01.json").read_text()
    )
    assert evidence["failure_count"] == 0
    comm_axis = evidence["axes"]["IMP_COMM.txt"]
    assert comm_axis["duplicate_control_key_count"] == 0
    assert comm_axis["measures"]["con_val_mo"]["cells_compared"] > 0
    assert (
        comm_axis["measures"]["con_val_mo"]["detail_total"]
        == (comm_axis["measures"]["con_val_mo"]["published_total"])
    )
    source_manifest_lines = (out_dir / "source_manifest.jsonl").read_text().splitlines()
    assert any("IMDB2601.ZIP" in line for line in source_manifest_lines)
    # Nothing from the build process leaks beside the publication.
    siblings = [path.name for path in out_dir.parent.iterdir()]
    assert not any(".staging-" in name or ".previous-" in name for name in siblings)


def test_build_cli_failure_leaves_prior_publication_untouched(tmp_path):
    """A failed rerun must never disturb a previously published artifact
    set: no partial writes, no deletions, no stale staging residue."""

    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "IMDB2601.ZIP").write_bytes(
        _fixture_zip_bytes(cty_overrides={"5700": {"cal_dut_mo": 9}})
    )
    out_dir = tmp_path / "out"
    prior_files = {
        "build_report.json": b'{"prior": true}\n',
        "margins_hts10_country_month.parquet": b"prior-margins-bytes",
        "detail/period=2020-01.parquet": b"prior-detail-bytes",
        "consumer_artifact/manifest.json": b'{"prior_manifest": true}\n',
    }
    for name, payload in prior_files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
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
            "--skip-cbp",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 1
    assert "nothing was published" in result.stderr
    published = {
        str(path.relative_to(out_dir)): path.read_bytes()
        for path in sorted(out_dir.rglob("*"))
        if path.is_file()
    }
    assert published == prior_files
    siblings = [path.name for path in tmp_path.iterdir()]
    assert not any(".staging-" in name or ".previous-" in name for name in siblings)


def test_build_cli_success_replaces_prior_publication_completely(tmp_path):
    """A successful rerun publishes the complete new set: nothing stale
    survives beside it."""

    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "IMDB2601.ZIP").write_bytes(_fixture_zip_bytes())
    import hashlib as hashlib_module

    payload = (archive_dir / "IMDB2601.ZIP").read_bytes()
    download_manifest = tmp_path / "download-manifest.jsonl"
    download_manifest.write_text(
        json.dumps(
            {
                "file": "IMDB2601.ZIP",
                "sha256": hashlib_module.sha256(payload).hexdigest(),
                "retrieved_at_utc": "2026-08-05T11:41:00Z",
            }
        )
        + "\n"
    )
    out_dir = tmp_path / "out"
    stale = out_dir / "detail" / "period=2019-12.parquet"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale-detail")
    (out_dir / "unrelated-note.txt").write_bytes(b"stale-note")
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
    assert not stale.exists()
    assert not (out_dir / "unrelated-note.txt").exists()
    assert (out_dir / "build_report.json").exists()
    assert (out_dir / "detail" / "period=2026-01.parquet").exists()
    assert (out_dir / "reconciliation" / "period=2026-01.json").exists()
    assert (out_dir / "source_manifest.jsonl").exists()
    siblings = [path.name for path in tmp_path.iterdir()]
    assert not any(".staging-" in name or ".previous-" in name for name in siblings)


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


def test_commodity_and_district_control_drift_fail_the_ingest(tmp_path):
    """Every control axis has teeth, not just the country file."""
    payload = _fixture_zip_bytes()
    import io as io_module
    import zipfile as zipfile_module

    def mutate(member: str, transform) -> bytes:
        source = zipfile_module.ZipFile(io_module.BytesIO(payload))
        buffer = io_module.BytesIO()
        with zipfile_module.ZipFile(buffer, "w") as bundle:
            for name in source.namelist():
                data = source.read(name)
                if name == member:
                    data = transform(data)
                bundle.writestr(name, data)
        return buffer.getvalue()

    # Commodity control drift: corrupt one digit of the horses con_val.
    comm_mutated = mutate(
        "IMP_COMM.txt", lambda data: data.replace(b"1250", b"1251", 1)
    )
    archive = tmp_path / "IMDB2601.ZIP"
    archive.write_bytes(comm_mutated)
    month = load_imdb_month(archive, "2026-01", {})
    assert any(
        "IMP_COMM.txt hts10=0101210010 con_val_mo" in failure
        for failure in month.reconciliation_failures
    )

    # District control drift.
    de_mutated = mutate("IMP_DE.txt", lambda data: data.replace(b"275", b"276", 1))
    archive.write_bytes(de_mutated)
    month = load_imdb_month(archive, "2026-01", {})
    assert any(
        "IMP_DE.txt dist_entry=20 cal_dut_mo" in failure
        for failure in month.reconciliation_failures
    )


def test_duplicate_control_keys_fail_the_ingest(tmp_path):
    """A byte-identical duplicated control row must fail, not pass: sets
    dedupe and index joins replicate, so without an explicit uniqueness
    gate a duplicate-keyed control table reconciles to zero failures."""

    payload = _fixture_zip_bytes()
    import io as io_module
    import zipfile as zipfile_module

    def duplicate_first_line(member: str) -> bytes:
        source = zipfile_module.ZipFile(io_module.BytesIO(payload))
        buffer = io_module.BytesIO()
        with zipfile_module.ZipFile(buffer, "w") as bundle:
            for name in source.namelist():
                data = source.read(name)
                if name == member:
                    first_line = data.split(b"\r\n", 1)[0]
                    data = data + first_line + b"\r\n"
                bundle.writestr(name, data)
        return buffer.getvalue()

    for member, key in (
        ("IMP_COMM.txt", "hts10"),
        ("imp_CTY.txt", "cty_code"),
        ("IMP_DE.txt", "dist_entry"),
    ):
        archive = tmp_path / "IMDB2601.ZIP"
        archive.write_bytes(duplicate_first_line(member))
        month = load_imdb_month(archive, "2026-01", {})
        assert any(
            f"duplicated control key(s) for {key}" in failure
            for failure in month.reconciliation_failures
        ), (member, month.reconciliation_failures)
        axis = month.reconciliation_evidence["axes"][member.replace("imp_", "IMP_")]
        assert axis["duplicate_control_key_count"] == 1
        assert axis["value_comparison"] == "skipped_duplicate_control_keys"


def test_key_sets_must_match_in_both_directions(tmp_path):
    """A key on either side only is named, never zero-compared away."""
    extra_control = _cty_line(
        "3010", "Brazil", "2026", "01", {"con_val_mo": 0, "cards_mo": 0}
    )
    payload = _fixture_zip_bytes()
    import io as io_module
    import zipfile as zipfile_module

    source = zipfile_module.ZipFile(io_module.BytesIO(payload))
    buffer = io_module.BytesIO()
    with zipfile_module.ZipFile(buffer, "w") as bundle:
        for name in source.namelist():
            data = source.read(name)
            if name == "imp_CTY.txt":
                data = data + extra_control.encode("latin-1") + b"\r\n"
            bundle.writestr(name, data)
    archive = tmp_path / "IMDB2601.ZIP"
    archive.write_bytes(buffer.getvalue())
    month = load_imdb_month(archive, "2026-01", {})
    assert any(
        "cty_code=3010: present in the control file but absent from the "
        "detail" in failure
        for failure in month.reconciliation_failures
    )

    stray_detail = _detail_line(
        "0101210010", "3070", "00", "07", "07", "10", "2026", "01", {"con_val_mo": 5}
    )
    month = _fixture_month(tmp_path, extra_detail_lines=(stray_detail,))
    assert any(
        "cty_code=3070: present in the detail but absent from the control" in failure
        for failure in month.reconciliation_failures
    )


def test_shifted_detail_colspecs_fail_reconciliation(tmp_path, monkeypatch):
    """A one-character parse shift cannot survive the control gates.

    The fixture's full-width row occupies every character of each measure
    slice, so shifting the detail colspecs by one column in either
    direction changes at least one parsed value while the control files
    (parsed with correct positions) keep the published totals.
    """
    import populace.build.us_runtime.us_trade.imdb_bulk as module

    for delta in (-1, 1):
        shifted = tuple(
            (name, start + delta, end + delta)
            for name, start, end in module._DETAIL_MEASURES
        )
        monkeypatch.setattr(module, "_DETAIL_MEASURES", shifted)
        archive = tmp_path / f"IMDB2601_{delta}.ZIP"
        archive.write_bytes(_fixture_zip_bytes())
        renamed = tmp_path / f"shift{delta}" / "IMDB2601.ZIP"
        renamed.parent.mkdir()
        archive.rename(renamed)
        try:
            month = load_imdb_month(renamed, "2026-01", {})
            assert month.reconciliation_failures, (
                f"colspec shift {delta:+d} produced no reconciliation failure"
            )
        except ValueError:
            pass  # blank-measure refusal is an equally loud failure
        monkeypatch.undo()


def test_adopted_archive_provenance_is_honest(tmp_path):
    """Adoption records no http_status and binds timestamps to the hash."""
    payload = _fixture_zip_bytes()
    archive_dir = tmp_path / "archives"
    archive_dir.mkdir()
    (archive_dir / "IMDB2601.ZIP").write_bytes(payload)
    import hashlib

    sha = hashlib.sha256(payload).hexdigest()

    _, entry = ensure_imdb_archive(
        "2026-01",
        archive_dir,
        retrieved_at_by_sha={("IMDB2601.ZIP", sha): "2026-08-05T11:41:00Z"},
    )
    assert "http_status" not in entry
    assert entry["retrieved_at"] == "2026-08-05T11:41:00Z"
    assert "sha-matched" in str(entry["retrieval_note"])

    # A wrong recorded hash must not lend its timestamp — and an adopted
    # archive with no manifest match records no retrieval time at all:
    # verified_at covers only this build's byte verification.
    _, entry2 = ensure_imdb_archive(
        "2026-01",
        archive_dir,
        retrieved_at_by_sha={("IMDB2601.ZIP", "00" * 32): "2026-08-05T11:41:00Z"},
    )
    assert "retrieved_at" not in entry2
    assert entry2["verified_at"]
    assert "retrieval time is unknown" in str(entry2["retrieval_note"])


def _build_cli_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_us_import_entry_margins", BUILD_CLI
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _populate(directory: Path, files: dict[str, bytes]) -> None:
    for name, payload in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


_OLD_SET = {"build_report.json": b'{"old": true}\n', "detail/a.parquet": b"old-detail"}
_NEW_SET = {"build_report.json": b'{"new": true}\n', "detail/b.parquet": b"new-detail"}


def _read_set(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _visible_siblings(parent: Path) -> list[str]:
    """Entries beside the publication, minus the persistent lockfile.

    The publisher's advisory lockfile deliberately survives between runs
    (unlinking a flock'd path lets a later opener lock a fresh inode
    while a prior holder still holds the orphan), so residue assertions
    exclude it.
    """
    return sorted(
        path.name
        for path in parent.iterdir()
        if not path.name.endswith(".publish-lock")
    )


def test_publish_atomically_replaces_and_leaves_no_residue(tmp_path):
    """End-to-end replacement through whichever swap path this filesystem
    offers: the new set is live, and no staging, previous, or recovery
    marker survives beside it."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-test"
    out_dir.mkdir()
    staging.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)
    build_cli._publish_atomically(staging, out_dir)
    assert _read_set(out_dir) == _NEW_SET
    assert not staging.exists()
    assert _visible_siblings(tmp_path) == ["out"]


def test_exchange_directories_swaps_in_a_single_syscall(tmp_path):
    """macOS and Linux dev/CI filesystems must take the exchange path —
    the one with no instant at which the published path is missing."""

    build_cli = _build_cli_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "who.txt").write_bytes(b"first")
    (second / "who.txt").write_bytes(b"second")
    exchanged = build_cli._exchange_directories(first, second)
    if not exchanged:
        pytest.skip("no atomic directory exchange on this platform/filesystem")
    assert (first / "who.txt").read_bytes() == b"second"
    assert (second / "who.txt").read_bytes() == b"first"


def _symlink_layout(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """Install ``out`` as a symlink to a populated versioned set."""
    initial_set = tmp_path / ".out.set-initial"
    _populate(initial_set, files)
    (tmp_path / "out").symlink_to(initial_set.name)
    return initial_set


def test_fallback_publish_is_a_windowless_symlink_retarget(tmp_path, monkeypatch):
    """Without an exchange, publication installs a symlink layout and
    every republication retargets it with ONE rename of the public name;
    the spy proves the published path resolves before and after every
    rename the publisher issues — there is no ENOENT window and no
    ordering that removes ``out`` before its replacement is in place."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    out_dir = tmp_path / "out"

    first_staging = tmp_path / ".out.staging-first"
    _populate(first_staging, _OLD_SET)
    build_cli._publish_atomically(first_staging, out_dir)
    assert out_dir.is_symlink()
    assert _read_set(out_dir) == _OLD_SET

    second_staging = tmp_path / ".out.staging-second"
    _populate(second_staging, _NEW_SET)
    real_rename = os.rename
    public_renames: list[tuple[str, str]] = []

    def spying_rename(src, dst, *args, **kwargs):
        assert os.path.exists(out_dir), "public path missing before a rename"
        result = real_rename(src, dst, *args, **kwargs)
        assert os.path.exists(out_dir), "public path missing after a rename"
        if Path(os.fspath(dst)) == out_dir or Path(os.fspath(src)) == out_dir:
            public_renames.append((os.fspath(src), os.fspath(dst)))
        return result

    monkeypatch.setattr(os, "rename", spying_rename)
    build_cli._publish_atomically(second_staging, out_dir)
    monkeypatch.undo()

    assert len(public_renames) == 1
    source, destination = public_renames[0]
    assert ".linktmp-" in Path(source).name
    assert Path(destination) == out_dir
    assert out_dir.is_symlink()
    assert _read_set(out_dir) == _NEW_SET
    siblings = _visible_siblings(tmp_path)
    assert siblings == sorted(["out", os.readlink(out_dir)])


def test_fallback_crash_before_the_retarget_leaves_publication_present(tmp_path):
    """Kill the fallback at the retarget attempt itself: the previous
    publication must still be fully readable under the public name (the
    r2 fallback had already renamed it away at this point — the ENOENT
    window), and recovery must finish the retarget from the marker."""

    initial_set = _symlink_layout(tmp_path, _OLD_SET)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-crash"
    _populate(staging, _NEW_SET)
    script = f"""
import importlib.util, os
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_cli", {str(BUILD_CLI)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._exchange_directories = lambda *args: False

out_dir = Path({str(out_dir)!r})
real_rename = os.rename

def dying_rename(src, dst, *args, **kwargs):
    if Path(os.fspath(dst)) == out_dir:
        os._exit(9)  # dies at the one operation that touches the public name
    return real_rename(src, dst, *args, **kwargs)

os.rename = dying_rename
module._publish_atomically(Path({str(staging)!r}), out_dir)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 9, result.stderr
    # The refutation of the r2 window: the crash point left the previous
    # publication untouched and readable under the public name.
    assert out_dir.is_symlink()
    assert os.readlink(out_dir) == initial_set.name
    assert _read_set(out_dir) == _OLD_SET
    marker = tmp_path / ".out.publish-recovery.json"
    assert json.loads(marker.read_text())["mode"] == "symlink-flip"

    build_cli = _build_cli_module()
    action = build_cli._recover_interrupted_publication(out_dir)
    assert action == "completed the interrupted symlink retarget from the staged set"
    assert out_dir.is_symlink()
    assert _read_set(out_dir) == _NEW_SET
    assert not marker.exists()
    assert not initial_set.exists()
    siblings = _visible_siblings(tmp_path)
    assert siblings == sorted(["out", os.readlink(out_dir)])


@pytest.mark.parametrize("kill_on", ["retarget", "vacate"])
def test_legacy_migration_crash_recovers_forward_to_symlink_layout(tmp_path, kill_on):
    """The one-time legacy migration (real directory, no exchange) is the
    only windowed transition left. Kill it before the window opens
    ('vacate': the publication is still present) and inside the window
    ('retarget': the public name is briefly absent); both states must
    roll forward to the symlink layout at the next build."""

    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-migrate"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)
    predicate = (
        "Path(os.fspath(dst)) == out_dir"
        if kill_on == "retarget"
        else "Path(os.fspath(src)) == out_dir"
    )
    script = f"""
import importlib.util, os
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_cli", {str(BUILD_CLI)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._exchange_directories = lambda *args: False

out_dir = Path({str(out_dir)!r})
real_rename = os.rename

def dying_rename(src, dst, *args, **kwargs):
    if {predicate}:
        os._exit(9)
    return real_rename(src, dst, *args, **kwargs)

os.rename = dying_rename
module._publish_atomically(Path({str(staging)!r}), out_dir)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 9, result.stderr
    marker = tmp_path / ".out.publish-recovery.json"
    assert json.loads(marker.read_text())["mode"] == "migrate"
    if kill_on == "vacate":
        # Killed before the window: the old publication never moved.
        assert out_dir.is_dir() and not out_dir.is_symlink()
        assert _read_set(out_dir) == _OLD_SET
    else:
        # Killed inside the window: the marker-guarded absent state.
        assert not out_dir.exists() and not out_dir.is_symlink()

    build_cli = _build_cli_module()
    action = build_cli._recover_interrupted_publication(out_dir)
    assert action == "completed the interrupted layout migration from the staged set"
    assert out_dir.is_symlink()
    assert _read_set(out_dir) == _NEW_SET
    assert not marker.exists()
    siblings = _visible_siblings(tmp_path)
    assert siblings == sorted(["out", os.readlink(out_dir)])


def test_exchange_interrupted_before_cleanup_reclaims_displaced_set(tmp_path):
    """A hard exit between the exchange and its cleanup used to strand the
    displaced previous publication under ``.staging-*`` with nothing
    recording it. The marker now names the staging path BEFORE the
    exchange — it survives the crash that follows the syscall — and
    recovery reclaims the orphan."""

    build_cli = _build_cli_module()
    if not build_cli._exchange_supported(tmp_path):
        pytest.skip("no atomic directory exchange on this platform/filesystem")
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-exchange"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)
    script = f"""
import importlib.util, os, shutil
from pathlib import Path

spec = importlib.util.spec_from_file_location("build_cli", {str(BUILD_CLI)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

real_rmtree = shutil.rmtree

def dying_rmtree(path, *args, **kwargs):
    if Path(os.fspath(path)).name.startswith(".out.staging-"):
        os._exit(9)  # dies after the exchange, before disposing the old set
    return real_rmtree(path, *args, **kwargs)

shutil.rmtree = dying_rmtree
module._publish_atomically(Path({str(staging)!r}), Path({str(out_dir)!r}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 9, result.stderr
    # The exchange happened: the new set is live and the displaced old
    # set sits under the staging name, which the pre-exchange marker
    # recorded.
    assert _read_set(out_dir) == _NEW_SET
    assert _read_set(staging) == _OLD_SET
    marker = tmp_path / ".out.publish-recovery.json"
    recorded = json.loads(marker.read_text())
    assert recorded["mode"] == "exchange"
    assert recorded["staging"] == staging.name

    action = build_cli._recover_interrupted_publication(out_dir)
    assert action == "removed the displaced set left by an interrupted exchange"
    assert _read_set(out_dir) == _NEW_SET
    assert not staging.exists()
    assert not marker.exists()
    assert _visible_siblings(tmp_path) == ["out"]


def test_publisher_lock_blocks_live_holders_and_dies_with_them(tmp_path):
    """The advisory flock refuses a concurrent publisher while its holder
    lives, releases with the holder's death (no staleness protocol, no
    takeover step to race — the old check-then-unlink takeover let two
    contenders each classify the same lock stale and then delete each
    other's fresh lock), and leaves both directories untouched by a
    refused attempt. A leftover lockfile with no live flock never blocks:
    its contents are diagnostics, not protocol state."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-locked"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)
    lock = tmp_path / ".out.publish-lock"

    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            f"""
import fcntl, os, sys
fd = os.open({str(lock)!r}, os.O_CREAT | os.O_RDWR, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
print("LOCKED", flush=True)
sys.stdin.readline()
""",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="Another publisher holds"):
            build_cli._publish_atomically(staging, out_dir)
        assert _read_set(out_dir) == _OLD_SET
        assert _read_set(staging) == _NEW_SET
    finally:
        holder.stdin.close()
        holder.wait(timeout=60)

    # The holder is dead; the kernel released its lock with it. The
    # stale-looking lockfile (old payload, dead pid) must not block —
    # and there is no unlink step for a second contender to race.
    build_cli._publish_atomically(staging, out_dir)
    assert _read_set(out_dir) == _NEW_SET
    assert lock.exists()


@pytest.mark.parametrize(
    "target",
    ["ABSOLUTE", "out", "../escape", ".other.set-abc"],
    ids=["absolute", "self", "parent-escape", "foreign-basename"],
)
def test_publish_refuses_symlinks_the_publisher_did_not_install(
    tmp_path, target, monkeypatch
):
    """A public name that is a symlink pointing anywhere outside the
    publisher's own ``.out.set-*`` namespace is refused untouched: joining
    an absolute or ``..``-relative target under the parent resolves to the
    target itself, which would ride straight into ``rmtree``; a self-link
    would delete the just-installed public name."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    victim = tmp_path / "victim"
    _populate(victim, {"precious.txt": b"do not delete"})
    resolved = str(victim) if target == "ABSOLUTE" else target
    out_dir = tmp_path / "out"
    out_dir.symlink_to(resolved)
    staging = tmp_path / ".out.staging-refused"
    _populate(staging, _NEW_SET)
    with pytest.raises(RuntimeError, match="not a set name owned by the publisher"):
        build_cli._publish_atomically(staging, out_dir)
    assert _read_set(victim) == {"precious.txt": b"do not delete"}
    assert os.readlink(out_dir) == resolved
    assert _read_set(staging) == _NEW_SET
    assert not (tmp_path / ".out.publish-recovery.json").exists()


def test_marker_survives_short_writes_intact(tmp_path, monkeypatch):
    """A legal short ``os.write`` must not durably truncate the recovery
    marker: the writer loops until every byte lands, so the marker JSON
    parses whole even when the kernel accepts one byte per call."""

    build_cli = _build_cli_module()
    real_write = os.write

    def one_byte_writes(fd, data):
        return real_write(fd, memoryview(data)[:1])

    monkeypatch.setattr(os, "write", one_byte_writes)
    try:
        build_cli._write_marker_durably(
            tmp_path / "out",
            {"mode": "exchange", "out": "out", "staging": ".out.staging-x"},
        )
    finally:
        monkeypatch.undo()
    recorded = json.loads((tmp_path / ".out.publish-recovery.json").read_text())
    assert recorded == {"mode": "exchange", "out": "out", "staging": ".out.staging-x"}


def test_staged_tree_is_fsynced_before_the_first_marker_write(tmp_path, monkeypatch):
    """Recovery assumes a staged set found on disk is complete, so the
    publisher must make the staged *contents* durable — every file and
    directory fsynced — before the first marker-guarded step. The
    parent-directory fsyncs elsewhere persist names, not bytes: without
    this, a power loss can leave the new name durable and the new tree
    hollow after the old set was already reclaimed."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-durable"
    _populate(staging, _NEW_SET)
    staged_inodes = {os.stat(path).st_ino for path in [staging, *staging.rglob("*")]}

    events: list[tuple[str, int | None]] = []
    real_fsync = os.fsync
    real_marker = build_cli._write_marker_durably

    def recording_fsync(fd):
        events.append(("fsync", os.fstat(fd).st_ino))
        return real_fsync(fd)

    def recording_marker(*args, **kwargs):
        events.append(("marker", None))
        return real_marker(*args, **kwargs)

    monkeypatch.setattr(os, "fsync", recording_fsync)
    monkeypatch.setattr(build_cli, "_write_marker_durably", recording_marker)
    try:
        build_cli._publish_atomically(staging, out_dir)
    finally:
        monkeypatch.undo()

    first_marker = next(
        index for index, event in enumerate(events) if event[0] == "marker"
    )
    fsynced_before_marker = {
        inode for kind, inode in events[:first_marker] if kind == "fsync"
    }
    assert staged_inodes <= fsynced_before_marker
    assert _read_set(out_dir) == _NEW_SET


def test_retarget_interrupted_after_the_commit_rolls_forward(tmp_path, monkeypatch):
    """An asynchronous exception arriving AFTER the retarget rename
    commits must not undo it: the handler derives the state from the
    filesystem — undoing here would move the live set out from under the
    just-installed public link, leaving it dangling with no marker. The
    new set stays live, the superseded set is disposed, the marker is
    withdrawn, and the exception propagates."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    initial_set = _symlink_layout(tmp_path, _OLD_SET)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-late"
    _populate(staging, _NEW_SET)
    real_rename = os.rename

    def rename_then_interrupt(src, dst, *args, **kwargs):
        result = real_rename(src, dst, *args, **kwargs)
        if Path(os.fspath(dst)) == out_dir:
            raise KeyboardInterrupt  # delivered after the syscall returned
        return result

    monkeypatch.setattr(os, "rename", rename_then_interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            build_cli._publish_atomically(staging, out_dir)
    finally:
        monkeypatch.undo()
    assert out_dir.is_symlink()
    assert _read_set(out_dir) == _NEW_SET
    assert not initial_set.exists()
    assert not (tmp_path / ".out.publish-recovery.json").exists()
    assert _visible_siblings(tmp_path) == sorted(["out", os.readlink(out_dir)])


def test_migration_interrupted_after_vacating_restores_the_publication(
    tmp_path, monkeypatch
):
    """An asynchronous exception landing right after the migration
    vacates the public name — before any in-process flag could have been
    assigned — must restore the previous publication: the handler derives
    the interruption point from the filesystem, never from control
    flow."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-vacate"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)
    real_rename = os.rename

    def interrupt_after_vacate(src, dst, *args, **kwargs):
        result = real_rename(src, dst, *args, **kwargs)
        if Path(os.fspath(src)) == out_dir:
            raise KeyboardInterrupt  # after the vacating rename returned
        return result

    monkeypatch.setattr(os, "rename", interrupt_after_vacate)
    try:
        with pytest.raises(KeyboardInterrupt):
            build_cli._publish_atomically(staging, out_dir)
    finally:
        monkeypatch.undo()
    assert out_dir.is_dir() and not out_dir.is_symlink()
    assert _read_set(out_dir) == _OLD_SET
    assert _read_set(staging) == _NEW_SET
    assert not (tmp_path / ".out.publish-recovery.json").exists()
    assert _visible_siblings(tmp_path) == sorted(["out", staging.name])


def test_recovery_reuses_the_marker_recorded_temp_link(tmp_path, monkeypatch):
    """Recovery retargets through the marker-recorded ``.linktmp-*`` name
    rather than minting a fresh one: a crash between the symlink and its
    rename then leaves only a link the next recovery pass already knows
    to clear — nothing unrecorded can be orphaned."""

    initial_set = _symlink_layout(tmp_path, _OLD_SET)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-reuse"
    _populate(staging, _NEW_SET)
    # Fabricate the pre-flip crash state the kill test produces live:
    # marker written, staged set parked, public link still on the old set.
    set_name = ".out.set-reused"
    link_name = ".out.linktmp-reused"
    staging.rename(tmp_path / set_name)
    (tmp_path / ".out.publish-recovery.json").write_text(
        json.dumps(
            {
                "mode": "symlink-flip",
                "out": "out",
                "staging": staging.name,
                "set": set_name,
                "link_tmp": link_name,
                "old_set": initial_set.name,
            }
        )
    )
    build_cli = _build_cli_module()
    real_rename = os.rename

    def fail_the_retarget(src, dst, *args, **kwargs):
        if Path(os.fspath(dst)) == out_dir:
            raise OSError("interrupted again")
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", fail_the_retarget)
    try:
        with pytest.raises(OSError, match="interrupted again"):
            build_cli._recover_interrupted_publication(out_dir)
    finally:
        monkeypatch.undo()
    leftovers = [name for name in _visible_siblings(tmp_path) if ".linktmp-" in name]
    assert leftovers == [link_name]

    action = build_cli._recover_interrupted_publication(out_dir)
    assert action == "completed the interrupted symlink retarget from the staged set"
    assert _read_set(out_dir) == _NEW_SET
    assert not initial_set.exists()
    assert [n for n in _visible_siblings(tmp_path) if ".linktmp-" in n] == []


def test_exchange_interrupted_after_the_syscall_preserves_the_marker(
    tmp_path, monkeypatch
):
    """An asynchronous exception can arrive AFTER the exchange syscall
    committed, so the handler must not assume nothing moved: it makes
    whatever happened durable and LEAVES the marker for recovery — which
    resolves the ambiguity either way (one complete set answers to the
    public name; the staging name holds the other for disposal)."""

    build_cli = _build_cli_module()
    if not build_cli._exchange_supported(tmp_path):
        pytest.skip("no atomic directory exchange on this platform/filesystem")
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-lateexchange"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)

    events: list[str] = []
    real_exchange = build_cli._exchange_directories
    real_fsync_dir = build_cli._fsync_dir
    real_remove_marker = build_cli._remove_marker

    def exchange_then_interrupt(source, target):
        assert real_exchange(source, target)
        events.append("exchange")
        raise KeyboardInterrupt  # delivered after the syscall returned

    def recording_fsync_dir(path):
        events.append("fsync_dir")
        return real_fsync_dir(path)

    def recording_remove_marker(target):
        events.append("remove_marker")
        return real_remove_marker(target)

    monkeypatch.setattr(build_cli, "_exchange_directories", exchange_then_interrupt)
    monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
    monkeypatch.setattr(build_cli, "_remove_marker", recording_remove_marker)
    try:
        with pytest.raises(KeyboardInterrupt):
            build_cli._publish_atomically(staging, out_dir)
    finally:
        monkeypatch.undo()
    # The exchange happened; the handler fsynced and did NOT remove the
    # marker — recovery owns the ambiguity.
    assert "remove_marker" not in events[events.index("exchange") :]
    assert "fsync_dir" in events[events.index("exchange") :]
    marker_path = tmp_path / ".out.publish-recovery.json"
    assert marker_path.exists()
    assert _read_set(out_dir) == _NEW_SET
    assert _read_set(staging) == _OLD_SET

    action = build_cli._recover_interrupted_publication(out_dir)
    assert action == "removed the displaced set left by an interrupted exchange"
    assert _read_set(out_dir) == _NEW_SET
    assert not staging.exists()
    assert not marker_path.exists()


def test_marker_removal_bounds_prior_cleanups_durably(tmp_path, monkeypatch):
    """``_remove_marker`` fsyncs the parent BEFORE unlinking the marker,
    so every cleanup since the marker was written (displaced-set
    deletion, temp-link unlinks, rollback renames) is durable before the
    no-recovery-needed commit point can be. On the exchange success path
    the prior ordering was ``backup_deleted -> marker_unlinked`` with no
    boundary — a power loss could keep the durable marker removal and
    lose the deletion, stranding a markerless orphan."""

    build_cli = _build_cli_module()
    if not build_cli._exchange_supported(tmp_path):
        pytest.skip("no atomic directory exchange on this platform/filesystem")
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-boundary"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)

    events: list[str] = []
    real_fsync_dir = build_cli._fsync_dir
    real_rmtree = shutil.rmtree
    real_unlink = Path.unlink

    def recording_fsync_dir(path):
        events.append("fsync_dir")
        return real_fsync_dir(path)

    def recording_rmtree(path, *args, **kwargs):
        events.append("rmtree")
        return real_rmtree(path, *args, **kwargs)

    def recording_unlink(self, *args, **kwargs):
        if self.name.endswith("publish-recovery.json"):
            events.append("marker_unlink")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
    monkeypatch.setattr(shutil, "rmtree", recording_rmtree)
    monkeypatch.setattr(Path, "unlink", recording_unlink)
    try:
        build_cli._publish_atomically(staging, out_dir)
    finally:
        monkeypatch.undo()
    assert _read_set(out_dir) == _NEW_SET
    cleanup = events.index("rmtree")
    marker_unlink = events.index("marker_unlink", cleanup)
    assert "fsync_dir" in events[cleanup + 1 : marker_unlink], (
        "no parent fsync between the displaced-set deletion and the marker unlink"
    )


@pytest.mark.parametrize("mode", ["retarget", "migrate"])
def test_rollback_renames_are_durable_before_the_marker_goes(
    tmp_path, monkeypatch, mode
):
    """The in-process rollback paths rename the parked set back to its
    staging name and then remove the marker: a parent fsync must land in
    between, or a power loss keeping the durable marker removal while
    losing the rename would strand a ``.set-*`` orphan nothing
    records."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-rollback"
    if mode == "retarget":
        _symlink_layout(tmp_path, _OLD_SET)
    else:
        out_dir.mkdir()
        _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)

    events: list[tuple[str, str]] = []
    real_rename = os.rename
    real_fsync_dir = build_cli._fsync_dir
    real_remove_marker = build_cli._remove_marker

    failed: list[bool] = []

    def failing_public_rename(src, dst, *args, **kwargs):
        # Fail only the FIRST public-name rename (the commit attempt);
        # the migration handler's restore rename targets the public name
        # too and must succeed.
        if Path(os.fspath(dst)) == out_dir and not failed:
            failed.append(True)
            raise OSError("simulated rename failure")
        result = real_rename(src, dst, *args, **kwargs)
        events.append(("rename", os.fspath(dst)))
        return result

    def recording_fsync_dir(path):
        events.append(("fsync_dir", os.fspath(path)))
        return real_fsync_dir(path)

    def recording_remove_marker(target):
        events.append(("remove_marker", os.fspath(target)))
        return real_remove_marker(target)

    monkeypatch.setattr(os, "rename", failing_public_rename)
    monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
    monkeypatch.setattr(build_cli, "_remove_marker", recording_remove_marker)
    try:
        with pytest.raises(OSError, match="simulated rename failure"):
            build_cli._publish_atomically(staging, out_dir)
    finally:
        monkeypatch.undo()
    rollback = max(
        index
        for index, (kind, path) in enumerate(events)
        if kind == "rename" and Path(path) == staging
    )
    removal = next(
        index
        for index, (kind, _path) in enumerate(events)
        if kind == "remove_marker" and index > rollback
    )
    assert any(
        kind == "fsync_dir" and Path(path) == tmp_path
        for kind, path in events[rollback + 1 : removal]
    ), "no parent fsync between the rollback rename and the marker removal"
    assert _read_set(staging) == _NEW_SET
    assert _read_set(out_dir) == _OLD_SET
    assert not (tmp_path / ".out.publish-recovery.json").exists()


def test_publisher_lock_refuses_a_symlinked_lockfile(tmp_path):
    """A symlink planted at the lock name must not hand the publisher an
    arbitrary file to truncate and overwrite: the open is no-follow and
    the referent stays byte-for-byte untouched."""

    build_cli = _build_cli_module()
    victim = tmp_path / "victim.json"
    victim.write_bytes(b'{"precious": true}\n')
    (tmp_path / ".out.publish-lock").symlink_to(victim.name)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-lockdodge"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)
    with pytest.raises(RuntimeError, match="refusing to lock through it"):
        build_cli._publish_atomically(staging, out_dir)
    assert victim.read_bytes() == b'{"precious": true}\n'
    assert _read_set(out_dir) == _OLD_SET
    assert _read_set(staging) == _NEW_SET


@pytest.mark.parametrize("mode", ["symlink-flip", "migrate", "exchange", "legacy"])
def test_recovery_refuses_public_names_the_protocol_cannot_have_produced(
    tmp_path, mode
):
    """A valid marker must not let recovery mutate a public name in a
    state its protocol cannot reach: a foreign symlink under a flip or
    migration marker (r5: the migration judged ANY symlink committed and
    discarded the previous set), or a symlink under the exchange/legacy
    protocols, which only ever move real directories. Recovery refuses,
    every named directory survives untouched, and the marker is
    preserved for the operator."""

    build_cli = _build_cli_module()
    victim = tmp_path / "victim"
    _populate(victim, {"precious.txt": b"do not delete"})
    out_dir = tmp_path / "out"
    out_dir.symlink_to(victim.name)
    survivors = {}
    marker_mode = "legacy-two-rename" if mode == "legacy" else mode
    marker: dict[str, object] = {"mode": marker_mode, "out": "out"}
    if mode == "symlink-flip":
        survivors["set"] = tmp_path / ".out.set-pending"
        _populate(survivors["set"], _NEW_SET)
        marker |= {
            "staging": ".out.staging-gone",
            "set": survivors["set"].name,
            "link_tmp": ".out.linktmp-gone",
            "old_set": ".out.set-old",
        }
    elif mode == "migrate":
        survivors["set"] = tmp_path / ".out.set-pending"
        survivors["previous"] = tmp_path / ".out.previous-real"
        _populate(survivors["set"], _NEW_SET)
        _populate(survivors["previous"], _OLD_SET)
        marker |= {
            "staging": ".out.staging-gone",
            "set": survivors["set"].name,
            "link_tmp": ".out.linktmp-gone",
            "previous": survivors["previous"].name,
        }
    elif mode == "exchange":
        survivors["staging"] = tmp_path / ".out.staging-displaced"
        _populate(survivors["staging"], _OLD_SET)
        marker |= {"staging": survivors["staging"].name}
    else:
        survivors["staging"] = tmp_path / ".out.staging-swap"
        survivors["previous"] = tmp_path / ".out.previous-swap"
        _populate(survivors["staging"], _NEW_SET)
        _populate(survivors["previous"], _OLD_SET)
        marker |= {
            "staging": survivors["staging"].name,
            "previous": survivors["previous"].name,
        }
    marker_path = tmp_path / ".out.publish-recovery.json"
    marker_path.write_text(json.dumps(marker))

    with pytest.raises(RuntimeError, match="refusing to recover|not a real directory"):
        build_cli._recover_interrupted_publication(out_dir)
    assert _read_set(victim) == {"precious.txt": b"do not delete"}
    assert os.readlink(out_dir) == victim.name
    for survivor in survivors.values():
        assert survivor.exists()
    assert marker_path.exists()


def test_set_name_is_durable_before_any_rename_touches_the_public_name(
    tmp_path, monkeypatch
):
    """Two renames in one directory carry no ordering guarantee through
    power loss: the set name must be fsynced durable before the public
    name is touched, on both the retarget and the migration — otherwise
    a surviving public link can point at a set name that never made it
    to disk."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    real_rename = os.rename
    real_fsync_dir = build_cli._fsync_dir

    def run(prepare, staging_name):
        events: list[tuple[str, str]] = []
        out_dir = tmp_path / "out"
        prepare(out_dir)
        staging = tmp_path / staging_name
        _populate(staging, _NEW_SET)

        def recording_rename(src, dst, *args, **kwargs):
            events.append(("rename", os.fspath(src), os.fspath(dst)))
            return real_rename(src, dst, *args, **kwargs)

        def recording_fsync_dir(path):
            events.append(("fsync_dir", os.fspath(path), ""))
            return real_fsync_dir(path)

        monkeypatch.setattr(os, "rename", recording_rename)
        monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
        try:
            build_cli._publish_atomically(staging, out_dir)
        finally:
            monkeypatch.setattr(os, "rename", real_rename)
            monkeypatch.setattr(build_cli, "_fsync_dir", real_fsync_dir)
        set_rename = next(
            index
            for index, (kind, _src, dst) in enumerate(events)
            if kind == "rename" and ".set-" in Path(dst).name
        )
        # The FIRST rename that touches the public name in any role, in
        # the whole event stream — the migration's vacating rename has
        # it as the source, and an unsafe mutation BEFORE the set rename
        # must fail the ordering assertion, not escape the window.
        first_public_mutation = next(
            index
            for index, (kind, src, dst) in enumerate(events)
            if kind == "rename" and (Path(src) == out_dir or Path(dst) == out_dir)
        )
        assert first_public_mutation > set_rename, (
            "the public name was touched before the set rename"
        )
        assert any(
            kind == "fsync_dir" and Path(first) == tmp_path
            for kind, first, _ in events[set_rename + 1 : first_public_mutation]
        ), "no parent fsync between the set rename and the first public mutation"
        assert _read_set(out_dir) == _NEW_SET
        shutil.rmtree(out_dir.parent / os.readlink(out_dir))
        out_dir.unlink()

    # Retarget path: an existing symlink layout is republished.
    def prepare_retarget(out_dir):
        _symlink_layout(tmp_path, _OLD_SET)

    run(prepare_retarget, ".out.staging-order-flip")

    # Migration path: a legacy real directory converts on an
    # exchange-less filesystem; the vacating rename also touches the
    # public name, so the fsync must come before it too.
    def prepare_migration(out_dir):
        out_dir.mkdir()
        _populate(out_dir, _OLD_SET)

    run(prepare_migration, ".out.staging-order-migrate")


def test_recovery_heals_the_torn_state_where_the_link_outran_the_set(tmp_path):
    """Power loss can persist the public-name rename while the set rename
    is lost: on disk the public link points at a set name that does not
    exist and the staged set still sits under its staging name. Recovery
    normalizes staging into the set name first, so the committed branch
    heals this torn state instead of discarding anything."""

    build_cli = _build_cli_module()
    initial_set = _symlink_layout(tmp_path, _OLD_SET)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-torn"
    _populate(staging, _NEW_SET)
    set_name = ".out.set-torn"
    # Fabricate the torn state: link retargeted, set rename lost.
    out_dir.unlink()
    out_dir.symlink_to(set_name)
    (tmp_path / ".out.publish-recovery.json").write_text(
        json.dumps(
            {
                "mode": "symlink-flip",
                "out": "out",
                "staging": staging.name,
                "set": set_name,
                "link_tmp": ".out.linktmp-torn",
                "old_set": initial_set.name,
            }
        )
    )
    action = build_cli._recover_interrupted_publication(out_dir)
    assert action == "finished a symlink retarget: disposed of the superseded set"
    assert out_dir.is_symlink()
    assert _read_set(out_dir) == _NEW_SET
    assert not initial_set.exists()
    assert not (tmp_path / ".out.publish-recovery.json").exists()
    assert _visible_siblings(tmp_path) == sorted(["out", set_name])


def test_migration_recovery_reuses_the_marker_recorded_temp_link(tmp_path, monkeypatch):
    """The migration's recovery path retargets through the marker-recorded
    ``.linktmp-*`` name too: a crash mid-recovery leaves only a link the
    next pass already knows to clear."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    set_name = ".out.set-migrate-reuse"
    link_name = ".out.linktmp-migrate-reuse"
    _populate(tmp_path / set_name, _NEW_SET)
    (tmp_path / ".out.publish-recovery.json").write_text(
        json.dumps(
            {
                "mode": "migrate",
                "out": "out",
                "staging": ".out.staging-migrate-reuse",
                "set": set_name,
                "link_tmp": link_name,
                "previous": ".out.previous-migrate-reuse",
            }
        )
    )
    real_rename = os.rename

    def fail_the_retarget(src, dst, *args, **kwargs):
        if Path(os.fspath(dst)) == out_dir:
            raise OSError("interrupted again")
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", fail_the_retarget)
    try:
        with pytest.raises(OSError, match="interrupted again"):
            build_cli._recover_interrupted_publication(out_dir)
    finally:
        monkeypatch.undo()
    leftovers = [name for name in _visible_siblings(tmp_path) if ".linktmp-" in name]
    assert leftovers == [link_name]

    action = build_cli._recover_interrupted_publication(out_dir)
    assert action == "completed the interrupted layout migration from the staged set"
    assert out_dir.is_symlink()
    assert _read_set(out_dir) == _NEW_SET
    assert [n for n in _visible_siblings(tmp_path) if ".linktmp-" in n] == []


@pytest.mark.parametrize("mode", ["migrate", "legacy"])
def test_recovery_refuses_regular_files_at_the_public_name(tmp_path, mode):
    """The symlink guards are not enough: a regular FILE at the public
    name also is not a state these directory-moving protocols can
    produce. Judging it committed (legacy) or vacating it into
    ``previous`` (migration) would destroy real data behind it."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    out_dir.write_bytes(b"a foreign regular file")
    survivors = []
    if mode == "migrate":
        set_dir = tmp_path / ".out.set-pending"
        _populate(set_dir, _NEW_SET)
        survivors.append(set_dir)
        marker: dict[str, object] = {
            "mode": "migrate",
            "out": "out",
            "staging": ".out.staging-gone",
            "set": set_dir.name,
            "link_tmp": ".out.linktmp-gone",
            "previous": ".out.previous-gone",
        }
    else:
        staging = tmp_path / ".out.staging-swap"
        previous = tmp_path / ".out.previous-swap"
        _populate(staging, _NEW_SET)
        _populate(previous, _OLD_SET)
        survivors += [staging, previous]
        marker = {
            "mode": "legacy-two-rename",
            "out": "out",
            "staging": staging.name,
            "previous": previous.name,
        }
    marker_path = tmp_path / ".out.publish-recovery.json"
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="refusing to recover over it"):
        build_cli._recover_interrupted_publication(out_dir)
    assert out_dir.read_bytes() == b"a foreign regular file"
    for survivor in survivors:
        assert survivor.exists()
    assert marker_path.exists()


@pytest.mark.parametrize("mode", ["symlink-flip", "migrate"])
def test_recovery_refuses_indirect_symlinks_at_the_set_name(tmp_path, mode):
    """``is_dir()`` follows symlinks: with ``out -> .out.set-x`` and
    ``.out.set-x -> victim``, the committed judgment would dispose the
    real old publication while the public name resolves into foreign
    territory. A symlink at the SET name refuses outright."""

    build_cli = _build_cli_module()
    victim = tmp_path / "victim"
    _populate(victim, {"precious.txt": b"do not delete"})
    old_set = tmp_path / ".out.set-old"
    _populate(old_set, _OLD_SET)
    set_name = ".out.set-indirect"
    (tmp_path / set_name).symlink_to(victim.name)
    out_dir = tmp_path / "out"
    out_dir.symlink_to(set_name)
    marker: dict[str, object] = {
        "mode": mode,
        "out": "out",
        "staging": ".out.staging-gone",
        "set": set_name,
        "link_tmp": ".out.linktmp-gone",
    }
    previous = tmp_path / ".out.previous-real"
    if mode == "symlink-flip":
        marker["old_set"] = old_set.name
    else:
        _populate(previous, _OLD_SET)
        marker["previous"] = previous.name
    marker_path = tmp_path / ".out.publish-recovery.json"
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(RuntimeError, match="does not hold a real directory"):
        build_cli._recover_interrupted_publication(out_dir)
    assert _read_set(victim) == {"precious.txt": b"do not delete"}
    assert _read_set(old_set) == _OLD_SET
    assert os.readlink(tmp_path / set_name) == victim.name
    if mode == "migrate":
        assert _read_set(previous) == _OLD_SET
    assert marker_path.exists()


def test_flip_recovery_refuses_when_the_link_outran_everything(tmp_path):
    """``out -> set`` with the set gone from BOTH its set and staging
    names is unrecoverable: 'publication intact' would be a lie (the
    link dangles) and clearing the marker would strand the old set
    behind it. Recovery raises and preserves the marker."""

    build_cli = _build_cli_module()
    initial_set = _symlink_layout(tmp_path, _OLD_SET)
    out_dir = tmp_path / "out"
    set_name = ".out.set-vanished"
    out_dir.unlink()
    out_dir.symlink_to(set_name)
    marker_path = tmp_path / ".out.publish-recovery.json"
    marker_path.write_text(
        json.dumps(
            {
                "mode": "symlink-flip",
                "out": "out",
                "staging": ".out.staging-vanished",
                "set": set_name,
                "link_tmp": ".out.linktmp-vanished",
                "old_set": initial_set.name,
            }
        )
    )
    with pytest.raises(RuntimeError, match="cannot be recovered automatically"):
        build_cli._recover_interrupted_publication(out_dir)
    assert _read_set(initial_set) == _OLD_SET
    assert os.readlink(out_dir) == set_name
    assert marker_path.exists()


@pytest.mark.parametrize("mode", ["symlink-flip", "migrate", "exchange", "legacy"])
def test_committed_recovery_fsyncs_before_deleting_the_backup(
    tmp_path, monkeypatch, mode
):
    """Every mode's committed branch must make the OBSERVED state durable
    before any backup is deleted: recovery can run seconds after a live
    crash with the decisive rename still in page cache, and a second
    power loss after the deletion persisted but the rename did not would
    leave the publication lost with the backup already gone."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    if mode == "symlink-flip":
        initial_set = _symlink_layout(tmp_path, _OLD_SET)
        new_set = tmp_path / ".out.set-committed"
        _populate(new_set, _NEW_SET)
        out_dir.unlink()
        out_dir.symlink_to(new_set.name)
        marker: dict[str, object] = {
            "mode": mode,
            "out": "out",
            "staging": ".out.staging-committed",
            "set": new_set.name,
            "link_tmp": ".out.linktmp-committed",
            "old_set": initial_set.name,
        }
        expected_action = "finished a symlink retarget: disposed of the superseded set"
    elif mode == "migrate":
        new_set = tmp_path / ".out.set-committed"
        previous = tmp_path / ".out.previous-committed"
        _populate(new_set, _NEW_SET)
        _populate(previous, _OLD_SET)
        out_dir.symlink_to(new_set.name)
        marker = {
            "mode": mode,
            "out": "out",
            "staging": ".out.staging-committed",
            "set": new_set.name,
            "link_tmp": ".out.linktmp-committed",
            "previous": previous.name,
        }
        expected_action = "finished the layout migration: disposed of the previous set"
    elif mode == "exchange":
        displaced = tmp_path / ".out.staging-displaced"
        out_dir.mkdir()
        _populate(out_dir, _NEW_SET)
        _populate(displaced, _OLD_SET)
        marker = {"mode": mode, "out": "out", "staging": displaced.name}
        expected_action = "removed the displaced set left by an interrupted exchange"
    else:
        swapped_out = tmp_path / ".out.previous-swapped"
        leftover_staging = tmp_path / ".out.staging-leftover"
        out_dir.mkdir()
        _populate(out_dir, _NEW_SET)
        _populate(swapped_out, _OLD_SET)
        _populate(leftover_staging, _NEW_SET)
        marker = {
            "mode": "legacy-two-rename",
            "out": "out",
            "staging": leftover_staging.name,
            "previous": swapped_out.name,
        }
        expected_action = "removed leftover swap directories"
    (tmp_path / ".out.publish-recovery.json").write_text(json.dumps(marker))

    events: list[tuple[str, str]] = []
    real_fsync_dir = build_cli._fsync_dir
    real_rmtree = shutil.rmtree

    def recording_fsync_dir(path):
        events.append(("fsync_dir", os.fspath(path)))
        return real_fsync_dir(path)

    def recording_rmtree(path, *args, **kwargs):
        events.append(("rmtree", os.fspath(path)))
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
    monkeypatch.setattr(shutil, "rmtree", recording_rmtree)
    try:
        action = build_cli._recover_interrupted_publication(out_dir)
    finally:
        monkeypatch.undo()
    assert action == expected_action
    first_rmtree = next(
        index for index, (kind, _path) in enumerate(events) if kind == "rmtree"
    )
    assert any(
        kind == "fsync_dir" and Path(path) == tmp_path
        for kind, path in events[:first_rmtree]
    ), "no parent fsync before the first backup deletion"
    assert _read_set(out_dir) == _NEW_SET


@pytest.mark.parametrize("mode", ["symlink-flip", "migrate"])
@pytest.mark.parametrize("planted", ["set-file", "staging-file"])
def test_recovery_never_treats_files_at_staging_or_set_names_as_sets(
    tmp_path, mode, planted
):
    """The staging and set names may only ever hold real directories: a
    regular file at the set name must refuse (it would be retargeted
    into publication while the old set is deleted), and a regular file
    at the staging name must never be normalized into the set name —
    the intact old publication stays exactly as it is."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    set_name = ".out.set-planted"
    staging_name = ".out.staging-planted"
    old_set_name = None
    if mode == "symlink-flip":
        initial_set = _symlink_layout(tmp_path, _OLD_SET)
        old_set_name = initial_set.name
    else:
        out_dir.mkdir()
        _populate(out_dir, _OLD_SET)
    if planted == "set-file":
        (tmp_path / set_name).write_bytes(b"a foreign regular file")
    else:
        (tmp_path / staging_name).write_bytes(b"a foreign regular file")
    marker: dict[str, object] = {
        "mode": mode,
        "out": "out",
        "staging": staging_name,
        "set": set_name,
        "link_tmp": ".out.linktmp-planted",
    }
    if mode == "symlink-flip":
        marker["old_set"] = old_set_name
    else:
        marker["previous"] = ".out.previous-planted"
    marker_path = tmp_path / ".out.publish-recovery.json"
    marker_path.write_text(json.dumps(marker))

    if planted == "set-file":
        with pytest.raises(RuntimeError, match="does not hold a real directory"):
            build_cli._recover_interrupted_publication(out_dir)
        assert (tmp_path / set_name).read_bytes() == b"a foreign regular file"
        assert marker_path.exists()
    else:
        # The file must not be renamed into the set name; the old
        # publication is judged intact and the foreign file is ignored.
        action = build_cli._recover_interrupted_publication(out_dir)
        assert "intact" in action
        assert (tmp_path / staging_name).read_bytes() == b"a foreign regular file"
        assert not (tmp_path / set_name).exists()
    assert _read_set(out_dir) == _OLD_SET


@pytest.mark.parametrize("old_state", ["missing", "file", "indirect"])
def test_flip_intact_verdict_checks_what_the_link_resolves_to(tmp_path, old_state):
    """'Publication intact' may not rest on the link's target string:
    when the old set behind ``out -> old_set`` is missing, a regular
    file, or an indirect symlink, removing the marker would bless a
    dangling or foreign publication. Recovery raises and preserves the
    marker."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    old_name = ".out.set-old"
    if old_state == "file":
        (tmp_path / old_name).write_bytes(b"a foreign regular file")
    elif old_state == "indirect":
        victim = tmp_path / "victim"
        _populate(victim, {"precious.txt": b"do not delete"})
        (tmp_path / old_name).symlink_to(victim.name)
    out_dir.symlink_to(old_name)
    marker_path = tmp_path / ".out.publish-recovery.json"
    marker_path.write_text(
        json.dumps(
            {
                "mode": "symlink-flip",
                "out": "out",
                "staging": ".out.staging-gone",
                "set": ".out.set-gone",
                "link_tmp": ".out.linktmp-gone",
                "old_set": old_name,
            }
        )
    )
    with pytest.raises(RuntimeError, match="not a real directory"):
        build_cli._recover_interrupted_publication(out_dir)
    assert os.readlink(out_dir) == old_name
    assert marker_path.exists()
    if old_state == "indirect":
        assert _read_set(tmp_path / "victim") == {"precious.txt": b"do not delete"}


@pytest.mark.parametrize("mode", ["migrate", "legacy"])
def test_restore_rename_is_durable_before_the_marker_is_removed(
    tmp_path, monkeypatch, mode
):
    """Restoring ``previous -> out`` must fsync before the marker goes: a
    second power loss that persists the marker deletion but not the
    restore rename would leave the public name absent with no recovery
    state at all."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    previous = tmp_path / ".out.previous-restore"
    _populate(previous, _OLD_SET)
    if mode == "migrate":
        marker: dict[str, object] = {
            "mode": "migrate",
            "out": "out",
            "staging": ".out.staging-gone",
            "set": ".out.set-gone",
            "link_tmp": ".out.linktmp-gone",
            "previous": previous.name,
        }
    else:
        marker = {
            "mode": "legacy-two-rename",
            "out": "out",
            "staging": ".out.staging-gone",
            "previous": previous.name,
        }
    (tmp_path / ".out.publish-recovery.json").write_text(json.dumps(marker))

    events: list[tuple[str, str]] = []
    real_rename = os.rename
    real_fsync_dir = build_cli._fsync_dir
    real_remove_marker = build_cli._remove_marker

    def recording_rename(src, dst, *args, **kwargs):
        events.append(("rename", os.fspath(dst)))
        return real_rename(src, dst, *args, **kwargs)

    def recording_fsync_dir(path):
        events.append(("fsync_dir", os.fspath(path)))
        return real_fsync_dir(path)

    def recording_remove_marker(target):
        events.append(("remove_marker", os.fspath(target)))
        return real_remove_marker(target)

    monkeypatch.setattr(os, "rename", recording_rename)
    monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
    monkeypatch.setattr(build_cli, "_remove_marker", recording_remove_marker)
    try:
        action = build_cli._recover_interrupted_publication(out_dir)
    finally:
        monkeypatch.undo()
    assert action == "restored the previous publication"
    restore = next(
        index
        for index, (kind, path) in enumerate(events)
        if kind == "rename" and Path(path) == out_dir
    )
    removal = next(
        index for index, (kind, _path) in enumerate(events) if kind == "remove_marker"
    )
    assert any(
        kind == "fsync_dir" and Path(path) == tmp_path
        for kind, path in events[restore + 1 : removal]
    ), "no parent fsync between the restore rename and the marker removal"
    assert _read_set(out_dir) == _OLD_SET


def test_migration_live_failure_restore_is_durable_before_the_marker(
    tmp_path, monkeypatch
):
    """The IN-PROCESS migration failure handler restores previous -> out
    and then removes the marker: the restore must be fsynced in between,
    or a power loss after the marker deletion persisted could lose the
    restore with no recovery state left."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-livefail"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)

    events: list[tuple[str, str]] = []
    real_rename = os.rename
    real_fsync_dir = build_cli._fsync_dir
    real_remove_marker = build_cli._remove_marker

    def interrupt_after_vacate(src, dst, *args, **kwargs):
        result = real_rename(src, dst, *args, **kwargs)
        events.append(("rename", os.fspath(dst)))
        if Path(os.fspath(src)) == out_dir:
            raise KeyboardInterrupt
        return result

    def recording_fsync_dir(path):
        events.append(("fsync_dir", os.fspath(path)))
        return real_fsync_dir(path)

    def recording_remove_marker(target):
        events.append(("remove_marker", os.fspath(target)))
        return real_remove_marker(target)

    monkeypatch.setattr(os, "rename", interrupt_after_vacate)
    monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
    monkeypatch.setattr(build_cli, "_remove_marker", recording_remove_marker)
    try:
        with pytest.raises(KeyboardInterrupt):
            build_cli._publish_atomically(staging, out_dir)
    finally:
        monkeypatch.undo()
    restore = next(
        index
        for index, (kind, path) in enumerate(events)
        if kind == "rename" and Path(path) == out_dir
    )
    # The failed exchange attempt withdraws its own marker before the
    # migration starts; the removal that matters is the one AFTER the
    # handler's restore.
    removal = next(
        index
        for index, (kind, _path) in enumerate(events)
        if kind == "remove_marker" and index > restore
    )
    assert any(
        kind == "fsync_dir" and Path(path) == tmp_path
        for kind, path in events[restore + 1 : removal]
    ), "no parent fsync between the handler's restore and the marker removal"
    assert _read_set(out_dir) == _OLD_SET
    assert not (tmp_path / ".out.publish-recovery.json").exists()


def test_interrupted_restore_retry_fsyncs_before_clearing_the_marker(
    tmp_path, monkeypatch
):
    """A recovery pass interrupted right after its restore rename leaves
    the marker in place; the RETRY then finds a healthy-looking public
    directory and takes the 'publication intact' path — which must fsync
    before the marker is cleared, because the restore it is blessing may
    still be sitting in page cache."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    previous = tmp_path / ".out.previous-interrupted"
    _populate(previous, _OLD_SET)
    marker_path = tmp_path / ".out.publish-recovery.json"
    marker_path.write_text(
        json.dumps(
            {
                "mode": "migrate",
                "out": "out",
                "staging": ".out.staging-gone",
                "set": ".out.set-gone",
                "link_tmp": ".out.linktmp-gone",
                "previous": previous.name,
            }
        )
    )
    real_rename = os.rename

    def interrupt_after_restore(src, dst, *args, **kwargs):
        result = real_rename(src, dst, *args, **kwargs)
        if Path(os.fspath(dst)) == out_dir:
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(os, "rename", interrupt_after_restore)
    try:
        with pytest.raises(KeyboardInterrupt):
            build_cli._recover_interrupted_publication(out_dir)
    finally:
        monkeypatch.undo()
    assert marker_path.exists(), "the interrupted pass must leave the marker"
    assert _read_set(out_dir) == _OLD_SET

    events: list[tuple[str, str]] = []
    real_fsync_dir = build_cli._fsync_dir
    real_remove_marker = build_cli._remove_marker

    def recording_fsync_dir(path):
        events.append(("fsync_dir", os.fspath(path)))
        return real_fsync_dir(path)

    def recording_remove_marker(target):
        events.append(("remove_marker", os.fspath(target)))
        return real_remove_marker(target)

    monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
    monkeypatch.setattr(build_cli, "_remove_marker", recording_remove_marker)
    try:
        action = build_cli._recover_interrupted_publication(out_dir)
    finally:
        monkeypatch.undo()
    assert action == "cleared an aborted layout migration (publication intact)"
    removal = next(
        index for index, (kind, _path) in enumerate(events) if kind == "remove_marker"
    )
    assert any(
        kind == "fsync_dir" and Path(path) == tmp_path
        for kind, path in events[:removal]
    ), "no parent fsync before the retry cleared the marker"
    assert not marker_path.exists()
    assert _read_set(out_dir) == _OLD_SET


def test_recovery_retarget_fsyncs_the_set_before_touching_the_public_name(
    tmp_path, monkeypatch
):
    """The set-before-link durability ordering holds on the RECOVERY
    retarget too: after normalizing staging into the set name, a parent
    fsync must land before the rename that retargets the public name."""

    build_cli = _build_cli_module()
    initial_set = _symlink_layout(tmp_path, _OLD_SET)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-reflip"
    _populate(staging, _NEW_SET)
    set_name = ".out.set-reflip"
    (tmp_path / ".out.publish-recovery.json").write_text(
        json.dumps(
            {
                "mode": "symlink-flip",
                "out": "out",
                "staging": staging.name,
                "set": set_name,
                "link_tmp": ".out.linktmp-reflip",
                "old_set": initial_set.name,
            }
        )
    )
    events: list[tuple[str, str, str]] = []
    real_rename = os.rename
    real_fsync_dir = build_cli._fsync_dir

    def recording_rename(src, dst, *args, **kwargs):
        events.append(("rename", os.fspath(src), os.fspath(dst)))
        return real_rename(src, dst, *args, **kwargs)

    def recording_fsync_dir(path):
        events.append(("fsync_dir", os.fspath(path), ""))
        return real_fsync_dir(path)

    monkeypatch.setattr(os, "rename", recording_rename)
    monkeypatch.setattr(build_cli, "_fsync_dir", recording_fsync_dir)
    try:
        action = build_cli._recover_interrupted_publication(out_dir)
    finally:
        monkeypatch.undo()
    assert action == "completed the interrupted symlink retarget from the staged set"
    set_rename = next(
        index
        for index, (kind, _src, dst) in enumerate(events)
        if kind == "rename" and Path(dst).name == set_name
    )
    first_public_mutation = next(
        index
        for index, (kind, src, dst) in enumerate(events)
        if kind == "rename" and (Path(src) == out_dir or Path(dst) == out_dir)
    )
    assert first_public_mutation > set_rename, (
        "recovery touched the public name before the normalization rename"
    )
    assert any(
        kind == "fsync_dir" and Path(first) == tmp_path
        for kind, first, _ in events[set_rename + 1 : first_public_mutation]
    ), "no parent fsync between the normalization rename and the public retarget"
    assert _read_set(out_dir) == _NEW_SET


def test_unreadable_staging_subtree_aborts_before_any_marker(tmp_path, monkeypatch):
    """`_fsync_tree` re-raises traversal failures: an unreadable staged
    subtree aborts publication before any marker is written — silently
    skipping it would publish a set the durability pass never saw."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-unreadable"
    _populate(staging, _NEW_SET)
    sealed = staging / "sealed"
    sealed.mkdir()
    (sealed / "inner.txt").write_bytes(b"unreachable")
    sealed.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            build_cli._publish_atomically(staging, out_dir)
    finally:
        sealed.chmod(0o755)
    assert not (tmp_path / ".out.publish-recovery.json").exists()
    assert not out_dir.exists() and not out_dir.is_symlink()


def test_marker_swap_recovery_restores_previous_when_staging_is_gone(tmp_path):
    """With the staged set gone, recovery must restore the previous
    publication byte-for-byte rather than leaving the path missing."""

    build_cli = _build_cli_module()
    out_dir = tmp_path / "out"
    previous = tmp_path / ".out.previous-deadbeef"
    previous.mkdir()
    _populate(previous, _OLD_SET)
    marker = tmp_path / ".out.publish-recovery.json"
    marker.write_text(
        json.dumps(
            {
                "out": "out",
                "staging": ".out.staging-gone",
                "previous": ".out.previous-deadbeef",
            }
        )
        + "\n"
    )
    action = build_cli._recover_interrupted_publication(out_dir)
    assert action == "restored the previous publication"
    assert _read_set(out_dir) == _OLD_SET
    assert not marker.exists()
    assert _visible_siblings(tmp_path) == ["out"]


def test_symlink_retarget_in_process_failure_restores_staging(tmp_path, monkeypatch):
    """A retarget failure inside the process leaves the previous
    publication untouched under the public name (it was never moved),
    puts the staged set back under its staging name for the caller's
    cleanup, and withdraws the marker."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    initial_set = _symlink_layout(tmp_path, _OLD_SET)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-fails"
    _populate(staging, _NEW_SET)
    real_rename = os.rename

    def failing_rename(src, dst, *args, **kwargs):
        if Path(os.fspath(dst)) == out_dir:
            raise OSError("simulated rename failure")
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated rename failure"):
        build_cli._publish_atomically(staging, out_dir)
    monkeypatch.undo()
    assert out_dir.is_symlink()
    assert os.readlink(out_dir) == initial_set.name
    assert _read_set(out_dir) == _OLD_SET
    assert _read_set(staging) == _NEW_SET
    assert not (tmp_path / ".out.publish-recovery.json").exists()
    siblings = _visible_siblings(tmp_path)
    assert siblings == sorted(["out", initial_set.name, staging.name])


def test_legacy_migration_in_process_failure_restores_previous(tmp_path, monkeypatch):
    """A retarget failure inside the migration rolls straight back: the
    previous publication returns under the public name, the staged set
    returns under its staging name, and the marker is removed."""

    build_cli = _build_cli_module()
    monkeypatch.setattr(build_cli, "_exchange_directories", lambda *args: False)
    out_dir = tmp_path / "out"
    staging = tmp_path / ".out.staging-fails"
    out_dir.mkdir()
    _populate(out_dir, _OLD_SET)
    _populate(staging, _NEW_SET)
    real_rename = os.rename
    failed = []

    def failing_once(src, dst, *args, **kwargs):
        if Path(os.fspath(dst)) == out_dir and not failed:
            failed.append(True)
            raise OSError("simulated rename failure")
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "rename", failing_once)
    with pytest.raises(OSError, match="simulated rename failure"):
        build_cli._publish_atomically(staging, out_dir)
    monkeypatch.undo()
    assert out_dir.is_dir() and not out_dir.is_symlink()
    assert _read_set(out_dir) == _OLD_SET
    assert _read_set(staging) == _NEW_SET
    assert not (tmp_path / ".out.publish-recovery.json").exists()
    siblings = _visible_siblings(tmp_path)
    assert siblings == sorted(["out", staging.name])


def test_cbp_page_retrieval_timestamp_is_captured_at_the_request(tmp_path, monkeypatch):
    """The CBP manifest entry's retrieved_at is the HTTP read moment, not
    a build-start timestamp threaded in from outside — the signature no
    longer even accepts one, and the same entry value feeds the facts."""

    from datetime import UTC, datetime

    build_cli = _build_cli_module()

    class _FakeResponse:
        def read(self):
            return b"<html>cbp page</html>"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(
        build_cli.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(),
    )
    before = datetime.now(UTC).isoformat(timespec="seconds")
    raw, entry = build_cli._archive_cbp_page(tmp_path)
    after = datetime.now(UTC).isoformat(timespec="seconds")
    assert before <= entry["retrieved_at"] <= after
    import hashlib as hashlib_module

    assert entry["sha256"] == hashlib_module.sha256(raw).hexdigest()
    assert (tmp_path / "cbp_newsroom_stats_trade.html").read_bytes() == raw
