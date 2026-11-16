"""VOA's small-area stock tables, CTSOP 1.1, 3.1 and 4.1 (microcosm#1524)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import voa_stock_tables as voa

_KEYS = ["geography", "ba_code", "ecode", "area_name"]


def _zip(tmp_path: Path, member: str, header: list[str], rows: list[list[str]]) -> Path:
    path = tmp_path / f"{Path(member).stem}.zip"
    text = "﻿" + "\n".join(",".join(f'"{v}"' for v in row) for row in [header, *rows])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text + "\n")
    return path


def _ctsop1(tmp_path: Path) -> Path:
    header = [*_KEYS, *(f"band_{b.lower()}" for b in voa.BANDS), "all_properties"]
    rows = [
        [
            "MSOA",
            "",
            "E02000001",
            "One",
            "100",
            "[c]",
            "20",
            "0",
            "0",
            "0",
            "0",
            "0",
            "[z]",
            "120",
        ],
        [
            "MSOA",
            "",
            "W02000001",
            "Two",
            "1,000",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "0",
            "10",
            "1,010",
        ],
        [
            "LAUA",
            "",
            "E06000001",
            "Town",
            "100",
            "[c]",
            "20",
            "0",
            "0",
            "0",
            "0",
            "0",
            "[z]",
            "120",
        ],
    ]
    return _zip(tmp_path, voa.CTSOP1_1_MEMBER, header, rows)


def test_band_counts_read_markers_and_separators(tmp_path: Path) -> None:
    path = _ctsop1(tmp_path)
    assert voa.member_header(path, voa.CTSOP1_1_MEMBER)[:2] == ["geography", "ba_code"]
    table = voa.read_member(path, voa.CTSOP1_1_MEMBER)
    counts = voa.band_counts(table, geography="MSOA", suppressed_value=2.5)
    assert counts.codes == ("E02000001", "W02000001")
    assert counts.counts[0].tolist() == [100, 2.5, 20, 0, 0, 0, 0, 0, 0]
    assert counts.counts[1, 0] == 1000 and counts.counts[1, 8] == 10
    assert counts.all_properties.tolist() == [120, 1010]
    assert counts.suppressed_cells == 1
    assert counts.index["W02000001"] == 1


def test_band_counts_refuse_text_they_cannot_read(tmp_path: Path) -> None:
    table = voa.read_member(_ctsop1(tmp_path), voa.CTSOP1_1_MEMBER)
    table.loc[0, "band_a"] = "about 100"
    with pytest.raises(ValueError, match="Unexpected CTSOP cells"):
        voa.band_counts(table, geography="MSOA", suppressed_value=2.5)
    table = voa.read_member(_ctsop1(tmp_path), voa.CTSOP1_1_MEMBER)
    with pytest.raises(ValueError, match="repeats"):
        voa.band_counts(
            pd.concat([table, table.iloc[:1]]), geography="MSOA", suppressed_value=2.5
        )


_TYPE_COLUMNS = [
    "bungalow_1",
    "bungalow_total",
    "flat_mais_1",
    "flat_mais_total",
    "house_terraced_1",
    "house_terraced_total",
    "house_semi_1",
    "house_semi_total",
    "house_detached_1",
    "house_detached_total",
    "annexe",
    "caravan_houseboat_mobilehome",
    "unknown",
]


def _type_row(area: str, band: str, *, bung, flat, ter, semi, det, other, total):
    def pair(value):
        return [str(value), str(value)]

    return [
        "MSOA" if area.startswith(("E02", "W02")) else "LAUA",
        "",
        area,
        "",
        band,
        *pair(bung),
        *pair(flat),
        *pair(ter),
        *pair(semi),
        *pair(det),
        "0",
        "0",
        str(other),
        str(total),
    ]


def _ctsop3(tmp_path: Path, *, band_a_total: int = 50) -> Path:
    header = [*_KEYS, "band", *_TYPE_COLUMNS, "all_properties"]
    rows = [
        _type_row(
            "E02000001",
            "All",
            bung=10,
            flat=30,
            ter=20,
            semi=20,
            det=10,
            other=10,
            total=100,
        ),
        _type_row(
            "E02000001",
            "A",
            bung=10,
            flat=0,
            ter=20,
            semi=20,
            det=0,
            other=0,
            total=band_a_total,
        ),
        _type_row(
            "E02000001", "B", bung=0, flat=30, ter=0, semi=0, det=10, other=10, total=50
        ),
        # An area whose houses are all bungalows takes its parent's house mix.
        _type_row(
            "E02000002", "All", bung=20, flat=0, ter=0, semi=0, det=0, other=0, total=20
        ),
        _type_row(
            "E02000002", "A", bung=20, flat=0, ter=0, semi=0, det=0, other=0, total=20
        ),
        _type_row(
            "E06000001",
            "All",
            bung=0,
            flat=0,
            ter=10,
            semi=30,
            det=60,
            other=0,
            total=100,
        ),
    ]
    return _zip(tmp_path, voa.CTSOP3_1_MEMBER, header, rows)


def test_type_shares_split_bungalows_and_other_types(tmp_path: Path) -> None:
    table = voa.read_member(_ctsop3(tmp_path), voa.CTSOP3_1_MEMBER)
    shares = voa.type_shares(
        table,
        geography="MSOA",
        suppressed_value=2.5,
        parent_of={"E02000001": "E06000001", "E02000002": "E06000001"},
    )
    assert shares.codes == ("E02000001", "E02000002")
    one = shares.shares[0]
    # Band A: bungalows split by the row's own house mix (semi 1/2, terrace 1/2).
    assert one[0].tolist() == pytest.approx([0.0, 0.5, 0.5, 0.0])
    # Band B: the unknown type spreads in the row's mix (D 1/4, F 3/4).
    assert one[1].tolist() == pytest.approx([0.25, 0.0, 0.0, 0.75])
    # Band C has no stock and takes the area's all-band shares.
    all_band = np.array([12, 24, 24, 30]) / 90
    expected_c = all_band * 90 + 10 * all_band
    assert one[2].tolist() == pytest.approx((expected_c / expected_c.sum()).tolist())
    assert np.allclose(shares.shares.sum(axis=-1), 1.0)
    # E02000002 has bungalows only: they take the authority's house mix.
    assert shares.shares[1][0].tolist() == pytest.approx([0.6, 0.3, 0.1, 0.0])
    assert shares.totals[0, :2].tolist() == [50, 50]
    assert shares.receipt["areas_without_parent_row"] == 0


def test_type_shares_refuse_rows_that_do_not_add_up(tmp_path: Path) -> None:
    table = voa.read_member(_ctsop3(tmp_path, band_a_total=90), voa.CTSOP3_1_MEMBER)
    with pytest.raises(ValueError, match="do not add up"):
        voa.type_shares(table, geography="MSOA", suppressed_value=2.5)


def test_type_columns_must_each_match_a_group() -> None:
    groups = voa.classify_type_columns(
        [*_KEYS, "band", *_TYPE_COLUMNS, "all_properties"]
    )
    assert groups["bungalow_total"] == "total"
    assert groups["house_semi_1"] == "S" and groups["unknown"] == "other"
    with pytest.raises(ValueError, match="match no type group"):
        voa.classify_type_columns([*_TYPE_COLUMNS, "park_home_1"])
    with pytest.raises(ValueError, match="one of the four types"):
        voa.classify_type_columns(
            ["bungalow_1", "flat_1", "house_ter_1", "house_semi_1"]
        )


def test_build_periods_read_spans_and_counts(tmp_path: Path) -> None:
    header = [
        *_KEYS,
        "band",
        "bp_pre_1900",
        "bp_1900_1918",
        "bp_2009",
        "bp_unkw",
        "all_properties",
    ]
    rows = [
        ["MSOA", "", "E02000001", "", "All", "10", "20", "0", "0", "30"],
        ["MSOA", "", "E02000001", "", "A", "10", "[c]", "0", "0", "10"],
        ["MSOA", "", "E02000001", "", "B", "0", "20", "0", "[c]", "20"],
    ]
    table = voa.read_member(
        _zip(tmp_path, voa.CTSOP4_1_MEMBER, header, rows), voa.CTSOP4_1_MEMBER
    )
    periods = voa.build_periods(table, geography="MSOA", suppressed_value=2.5)
    assert periods.spans == ((None, 1899), (1900, 1918), (2009, 2009), (None, None))
    assert periods.counts[0, 0].tolist() == [10, 2.5, 0, 0]
    assert periods.counts[0, 1].tolist() == [0, 20, 0, 2.5]
    assert periods.counts[0, 2:].sum() == 0
    assert periods.receipt["imputed_suppressed_cells"] == 2
    with pytest.raises(ValueError, match="not build periods"):
        voa.classify_build_periods([*header, "bp_recent"])
