"""UK firm public sector employment: an opt-in rescale to ONS headcounts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from functools import cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_public_sector import (
    UK_FIRM_PUBLIC_SECTOR_FILE,
    UK_FIRM_PUBLIC_SECTOR_PERIOD,
    UK_FIRM_PUBLIC_SECTOR_SERIES,
    UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS,
    UK_FIRM_PUBLIC_SECTOR_STATUS,
    UK_FIRM_PUBLIC_SECTOR_STATUSES,
    UKFirmPublicSectorSource,
    read_uk_firm_public_sector_source,
    scale_uk_firm_public_sector_employment,
    uk_firm_public_sector_diagnostics,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURE = PATHS.tests / "fixtures" / "uk" / "firm_public_sector_2025"
CSV = FIXTURE / UK_FIRM_PUBLIC_SECTOR_FILE
PROVENANCE = FIXTURE / (UK_FIRM_PUBLIC_SECTOR_FILE + ".provenance.json")
TOOL = PATHS.repository / "tools" / "extract_uk_firm_public_sector_table.py"

#: ONS Public sector employment, UK: March 2025, Table 1, row Mar 2025
#: (thousands, seasonally adjusted), read off the workbook.
_CENTRAL_THOUSAND = 4020
_LOCAL_THOUSAND = 1977


@cache
def _source() -> UKFirmPublicSectorSource:
    return read_uk_firm_public_sector_source(FIXTURE)


@cache
def _provenance() -> dict:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


@cache
def _load_tool():
    name = "extract_uk_firm_public_sector_table"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _firms(seed: int = 7) -> pd.DataFrame:
    """A small generator-like frame with every ONS legal status."""

    rng = np.random.default_rng(seed)
    status = np.array(
        [1] * 40 + [2] * 10 + [3] * 5 + [4] * 3 + [5] * 12 + [6] * 25 + [7] * 5
    )
    rows = len(status)
    return pd.DataFrame(
        {
            "firm_id": np.arange(1, rows + 1),
            "employment": rng.integers(1, 400, size=rows).astype(np.int64),
            "firm_weight": rng.uniform(0.5, 9.0, size=rows),
            "ons_legal_status": status,
            "turnover": rng.uniform(10.0, 5000.0, size=rows),
        },
        index=np.arange(100, 100 + rows),
    )


def _weighted(frame: pd.DataFrame, status: int, column: str = "employment") -> float:
    rows = frame["ons_legal_status"] == status
    return float((frame.loc[rows, "firm_weight"] * frame.loc[rows, column]).sum())


def test_fixture_reads_as_the_published_headcounts() -> None:
    source = _source()

    assert dict(source.headcount) == {
        5: _CENTRAL_THOUSAND * 1000.0,
        6: _LOCAL_THOUSAND * 1000.0,
    }
    assert dict(source.series) == {5: "G6NQ", 6: "G6NT"}
    assert source.period == UK_FIRM_PUBLIC_SECTOR_PERIOD == "Mar 2025"
    assert source.file == UK_FIRM_PUBLIC_SECTOR_FILE
    assert source.sha256 == hashlib.sha256(CSV.read_bytes()).hexdigest()
    assert read_uk_firm_public_sector_source(CSV) == source
    assert UK_FIRM_PUBLIC_SECTOR_STATUS == "assumed"


def test_fixture_csv_layout() -> None:
    frame = pd.read_csv(CSV, dtype=str, keep_default_na=False)

    assert tuple(frame.columns) == UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS
    assert frame["sector"].tolist() == list(UK_FIRM_PUBLIC_SECTOR_STATUSES)
    assert frame["series_id"].tolist() == list(UK_FIRM_PUBLIC_SECTOR_SERIES.values())
    assert sorted(path.name for path in FIXTURE.iterdir()) == [
        CSV.name,
        PROVENANCE.name,
    ]


def test_provenance_is_well_formed() -> None:
    record = _provenance()
    source = record["source"]

    assert record["kind"] == "uk_firm_public_sector_employment_fixture"
    assert record["accessed"] == "2026-11-22"
    assert "none was filled from memory" in record["method"]
    assert source["landing_page"].startswith("https://www.ons.gov.uk/")
    assert source["url"].startswith("https://www.ons.gov.uk/file?uri=")
    assert source["url"].endswith("/march2025/datasets8.xlsx")
    assert "Public sector employment, UK: March 2025" in source["edition"]
    assert len(source["sha256"]) == 64
    int(source["sha256"], 16)
    assert source["sheet"] == "Table 1"
    assert "seasonally adjusted" in source["table"]
    assert source["series"] == dict(UK_FIRM_PUBLIC_SECTOR_SERIES)
    assert "not the dataset's latest" in record["edition_note"]
    entry = record["files"][UK_FIRM_PUBLIC_SECTOR_FILE]
    payload = CSV.read_bytes()
    assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert entry["bytes"] == len(payload)
    assert entry["columns"] == list(UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS)
    later = record["later_edition"]
    assert len(later["sha256"]) == 64 and later["sha256"] != source["sha256"]
    assert set(later["headcount_thousand"]) == set(UK_FIRM_PUBLIC_SECTOR_STATUSES)


def test_tool_names_the_fixture_and_rebuilds_its_bytes() -> None:
    tool = _load_tool()
    record = _provenance()

    assert tool.CSV_NAME == UK_FIRM_PUBLIC_SECTOR_FILE
    assert tool.PROVENANCE_NAME == PROVENANCE.name
    assert tool.DEFAULT_OUT == FIXTURE
    assert tool.HEADER == UK_FIRM_PUBLIC_SECTOR_SOURCE_COLUMNS
    assert tool.XLSX_URL == record["source"]["url"]
    assert tool.LANDING_PAGE == record["source"]["landing_page"]
    assert tool.PERIOD == UK_FIRM_PUBLIC_SECTOR_PERIOD
    assert {(s, c, i) for s, c, i, _ in tool.SECTORS} == {
        (sector, code, UK_FIRM_PUBLIC_SECTOR_SERIES[sector])
        for sector, code in UK_FIRM_PUBLIC_SECTOR_STATUSES.items()
    }
    # The sheet's shape, cut to the cells the extractor reads.
    sheet = [
        (tool.TITLE_PREFIX + " 1 2 3", None, None, None),
        (None, None, None, " " + tool.UNITS),
        (
            None,
            "Central \ngovernment 4",
            "Local\ngovernment 4 ",
            "Total general government",
        ),
        (None, None, None, None),
        (None, "G6NQ", "G6NT", "G6NW"),
        ("Dec 2024 [r]", 4001, 1986, 5987),
        ("Mar 2025", _CENTRAL_THOUSAND, _LOCAL_THOUSAND, 5997),
        ("Change on quarter", 19, -9, 10),
    ]
    rows = tool.extract_rows(sheet)
    assert tool.to_csv_bytes(rows) == CSV.read_bytes()

    with pytest.raises(ValueError, match="found 0x"):
        tool.extract_rows([row for row in sheet if row[0] != "Mar 2025"])
    swapped = [*sheet]
    swapped[4] = (None, "G6NT", "G6NQ", "G6NW")
    with pytest.raises(ValueError, match="sits under"):
        tool.extract_rows(swapped)


def test_scaling_hits_the_published_totals_with_whole_headcounts() -> None:
    firms = _firms()
    source = _source()

    scaled = scale_uk_firm_public_sector_employment(firms, source)

    for status in (5, 6):
        rows = scaled["ons_legal_status"] == status
        largest_weight = float(scaled.loc[rows, "firm_weight"].max())
        miss = abs(_weighted(scaled, status) - source.headcount[status])
        assert miss <= largest_weight / 2 + 1e-6
        assert miss / source.headcount[status] < 1e-5
        assert int(scaled.loc[rows, "employment"].min()) >= 1
    assert scaled["employment"].dtype == firms["employment"].dtype


def test_scaling_keeps_rows_and_everything_else() -> None:
    firms = _firms()
    before = firms.copy(deep=True)

    scaled = scale_uk_firm_public_sector_employment(firms, _source())

    pd.testing.assert_frame_equal(firms, before)
    assert len(scaled) == len(firms)
    assert scaled.index.equals(firms.index)
    assert list(scaled.columns) == list(firms.columns)
    pd.testing.assert_frame_equal(
        scaled.drop(columns="employment"), firms.drop(columns="employment")
    )
    others = ~firms["ons_legal_status"].isin([5, 6])
    assert others.sum() == 63
    pd.testing.assert_series_equal(
        scaled.loc[others, "employment"], firms.loc[others, "employment"]
    )
    assert _weighted(scaled, 4) == _weighted(firms, 4)
    # One factor per status keeps the order of headcounts within it.
    for status in (5, 6):
        rows = firms["ons_legal_status"] == status
        order = np.argsort(firms.loc[rows, "employment"].to_numpy(), kind="stable")
        assert np.all(np.diff(scaled.loc[rows, "employment"].to_numpy()[order]) >= 0)


def test_scaling_is_deterministic() -> None:
    firms = _firms()
    source = _source()

    first = scale_uk_firm_public_sector_employment(firms, source)
    second = scale_uk_firm_public_sector_employment(firms.copy(), source)
    pd.testing.assert_frame_equal(first, second)


def test_scaling_can_shrink_and_never_goes_below_one() -> None:
    firms = pd.DataFrame(
        {
            "employment": np.array([900_000, 3, 1, 700_000, 2, 5], dtype=np.int64),
            "firm_weight": [10.0, 4.0, 2.0, 8.0, 3.0, 1.0],
            "ons_legal_status": [5, 5, 5, 6, 6, 1],
        }
    )

    scaled = scale_uk_firm_public_sector_employment(firms, _source())

    assert scaled["employment"].tolist()[1:3] == [1, 1]
    assert scaled["employment"].tolist()[4:] == [1, 5]
    assert scaled["employment"].iloc[0] < 900_000
    assert scaled["employment"].iloc[3] < 700_000


def test_scaling_reads_a_named_headcount_column() -> None:
    firms = _firms().rename(columns={"employment": "firm_employees"})
    source = _source()

    scaled = scale_uk_firm_public_sector_employment(
        firms, source, employment_column="firm_employees"
    )

    assert _weighted(scaled, 5, "firm_employees") == pytest.approx(
        source.headcount[5], rel=1e-5
    )
    with pytest.raises(ValueError, match="needs columns"):
        scale_uk_firm_public_sector_employment(firms, source)


def test_diagnostics_report_before_and_after() -> None:
    firms = _firms()
    source = _source()

    report = uk_firm_public_sector_diagnostics(firms, source)
    scaled = scale_uk_firm_public_sector_employment(firms, source)

    assert report["status"] == "assumed"
    assert report["source_sha256"] == source.sha256
    assert report["rows"] == len(firms)
    assert list(report["sectors"]) == list(UK_FIRM_PUBLIC_SECTOR_STATUSES)
    for sector, status in UK_FIRM_PUBLIC_SECTOR_STATUSES.items():
        entry = report["sectors"][sector]
        rows = firms["ons_legal_status"] == status
        assert entry["ons_legal_status"] == status
        assert entry["series_id"] == UK_FIRM_PUBLIC_SECTOR_SERIES[sector]
        assert entry["rows"] == int(rows.sum())
        assert entry["target_weighted_employment"] == source.headcount[status]
        assert entry["weighted_employment_before"] == pytest.approx(
            _weighted(firms, status)
        )
        assert entry["weighted_employment_after"] == pytest.approx(
            _weighted(scaled, status)
        )
        assert entry["scale_factor"] == pytest.approx(
            source.headcount[status] / _weighted(firms, status)
        )
        assert entry["largest_headcount_after"] == int(
            scaled.loc[rows, "employment"].max()
        )
    assert report["total_weighted_employment_after"] == pytest.approx(
        float((scaled["firm_weight"] * scaled["employment"]).sum())
    )
    json.dumps(report)


@pytest.mark.parametrize("status", [5, 6])
def test_scaling_refuses_a_frame_without_a_status(status: int) -> None:
    firms = _firms()
    firms = firms[firms["ons_legal_status"] != status]

    with pytest.raises(ValueError, match=f"no rows with ons_legal_status {status}"):
        scale_uk_firm_public_sector_employment(firms, _source())
    with pytest.raises(ValueError, match=f"no rows with ons_legal_status {status}"):
        uk_firm_public_sector_diagnostics(firms, _source())


def test_scaling_refuses_bad_frames() -> None:
    firms = _firms()
    source = _source()

    with pytest.raises(ValueError, match="needs columns"):
        scale_uk_firm_public_sector_employment(
            firms.drop(columns="firm_weight"), source
        )
    with pytest.raises(ValueError, match="whole numbers"):
        scale_uk_firm_public_sector_employment(
            firms.assign(employment=firms["employment"] + 0.5), source
        )
    with pytest.raises(ValueError, match="firm_weight"):
        scale_uk_firm_public_sector_employment(
            firms.assign(firm_weight=-firms["firm_weight"]), source
        )
    empty = firms.assign(
        employment=firms["employment"].where(firms["ons_legal_status"] != 5, 0)
    )
    with pytest.raises(ValueError, match="cannot scale ons_legal_status 5"):
        scale_uk_firm_public_sector_employment(empty, source)


def test_reader_refuses_malformed_sources(tmp_path) -> None:
    text = CSV.read_text(encoding="utf-8")
    cases = {
        "columns": text.replace("headcount_thousand", "headcount"),
        "once each": text.replace("local_government", "central_government"),
        "ons_legal_status": text.replace(
            "central_government,5", "central_government,6"
        ),
        "series": text.replace("G6NQ", "G7AR"),
        "period": text.replace("Mar 2025", "Jun 2025", 1),
        "not a number": text.replace(str(_CENTRAL_THOUSAND), "x"),
        "positive": text.replace(str(_LOCAL_THOUSAND), "0"),
    }
    for match, payload in cases.items():
        path = tmp_path / UK_FIRM_PUBLIC_SECTOR_FILE
        path.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            read_uk_firm_public_sector_source(tmp_path)
