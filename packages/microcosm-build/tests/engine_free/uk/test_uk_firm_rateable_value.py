"""UK firm rateable value: VOA 2023-list band fixture and the assumed mapping."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.firm_rateable_value import (
    UK_FIRM_RATEABLE_VALUE_FILE,
    UK_FIRM_RATEABLE_VALUE_SOURCE_COLUMNS,
    UK_FIRM_RATEABLE_VALUE_STATUS,
    assign_uk_firm_rateable_value,
    read_uk_firm_rateable_value_source,
    uk_firm_rateable_value_of_top,
    uk_firm_rateable_value_summary,
)
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
FIXTURE = PATHS.tests / "fixtures" / "uk" / "firm_rateable_value_2024"
TOOL = PATHS.repository / "tools" / "extract_uk_firm_rateable_value_tables.py"

#: VOA stock of properties, March 2026, Tables SOP2.1-SOP2.3 ``Total`` rows,
#: all sectors: (hereditaments, total rateable value in GBP thousand).
_TOTALS = {
    "england_and_wales": (2138260, 70853994),
    "england": (2013470, 68437352),
    "wales": (124790, 2416641),
}
#: England, all sectors, read off Table SOP2.2.
_ENGLAND_BANDS = {
    "0 - 999": (144660, 70131),
    "12,001 - 13,999": (77680, 1006689),
    "50,000 - 50,999": (4370, 219463),
    "1,000,000 and over": (6790, 16070230),
}


def _load_tool():
    name = "extract_uk_firm_rateable_value_tables"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _provenance() -> dict:
    return json.loads((FIXTURE / ".provenance.json").read_text(encoding="utf-8"))


def _bands() -> pd.DataFrame:
    return pd.read_csv(
        FIXTURE / UK_FIRM_RATEABLE_VALUE_FILE, dtype=str, keep_default_na=False
    )


def _source():
    return read_uk_firm_rateable_value_source(FIXTURE)


def test_band_fixture_holds_the_published_totals_and_bands() -> None:
    frame = _bands()
    assert tuple(frame.columns) == UK_FIRM_RATEABLE_VALUE_SOURCE_COLUMNS
    everything = frame[frame["sector"] == "all"].set_index(["geography", "rv_band"])
    for geography, (count, value) in _TOTALS.items():
        row = everything.loc[(geography, "Total")]
        assert (int(row["hereditaments"]), int(row["total_rv_gbp_thousand"])) == (
            count,
            value,
        )
    for band, (count, value) in _ENGLAND_BANDS.items():
        row = everything.loc[("england", band)]
        assert (int(row["hereditaments"]), int(row["total_rv_gbp_thousand"])) == (
            count,
            value,
        )


def test_band_fixture_adds_up_within_the_source_rounding() -> None:
    """Bands sum to the ``Total`` row and sectors to ``all``, to the VOA's
    rounding (counts to the nearest ten, values to the nearest thousand)."""

    frame = _bands().astype(
        {"hereditaments": np.int64, "total_rv_gbp_thousand": np.int64}
    )
    numbers = ["hereditaments", "total_rv_gbp_thousand"]
    for (_, sector), rows in frame.groupby(["geography", "sector"]):
        total = rows[rows["rv_band"] == "Total"][numbers].iloc[0]
        summed = rows[rows["rv_band"] != "Total"][numbers].sum()
        assert len(rows) == 19, sector
        assert abs(summed["hereditaments"] - total["hereditaments"]) <= 100
        assert (
            abs(summed["total_rv_gbp_thousand"] - total["total_rv_gbp_thousand"]) < 20
        )
    for _, rows in frame.groupby(["geography", "rv_band"]):
        by_sector = rows.set_index("sector")
        parts = by_sector.loc[["retail", "offices", "industry", "other"], numbers].sum()
        assert abs(parts["hereditaments"] - by_sector.loc["all", "hereditaments"]) <= 40
        assert (
            abs(
                parts["total_rv_gbp_thousand"]
                - by_sector.loc["all", "total_rv_gbp_thousand"]
            )
            < 5
        )


def test_property_type_and_year_fixtures_agree_with_the_band_table() -> None:
    types = pd.read_csv(
        FIXTURE / "voa_property_type.csv", dtype=str, keep_default_na=False
    )
    years = pd.read_csv(FIXTURE / "voa_stock_by_year.csv", dtype=str)
    for geography, (count, value) in _TOTALS.items():
        row = types[(types["geography"] == geography) & (types["level"] == "all")]
        assert row["hereditaments"].tolist() == [str(count)]
        assert row["total_rv_gbp_thousand"].tolist() == [str(value)]
        latest = years[
            (years["geography"] == geography) & (years["as_at"] == "2026-03-31")
        ]
        assert latest["hereditaments"].tolist() == [str(count)]
        assert latest["total_rv_gbp_thousand"].tolist() == [str(value)]
    assert sorted(years["as_at"].unique()) == ["2024-03-31", "2025-03-31", "2026-03-31"]
    assert set(types["level"]) == {"all", "sector", "sub_sector", "shop_type"}
    # A suppressed row keeps its marker and no numbers; it is never a zero.
    suppressed = types[types["marker"] != ""]
    assert suppressed["marker"].tolist() == ["[c]"]
    assert suppressed[["geography", "prop_type_code"]].values.tolist() == [
        ["wales", "CS8"]
    ]
    assert (suppressed["hereditaments"] == "").all()
    assert (types.loc[types["marker"] == "", "hereditaments"] != "").all()
    sectors = types[(types["geography"] == "england") & (types["level"] == "sector")]
    assert sectors["prop_type_code"].tolist() == ["C", "E", "U", "I", "L", "M"]


def test_provenance_records_the_download_and_the_rating_list() -> None:
    record = _provenance()
    tool = _load_tool()
    source = record["source"]
    assert record["accessed"] == "2026-11-22" == tool.ACCESS_DATE
    assert "2023 rating list" in record["rating_list"]
    assert source["url"] == tool.ZIP_URL
    assert source["url"].startswith("https://assets.publishing.service.gov.uk/media/")
    assert source["landing_page"] == tool.LANDING_PAGE
    assert "March 2026" in source["edition"]
    assert len(source["sha256"]) == 64
    assert record["method"]
    assert set(record["files"]) == {
        "voa_rv_band_sector.csv",
        "voa_property_type.csv",
        "voa_stock_by_year.csv",
    }
    for name, entry in record["files"].items():
        payload = (FIXTURE / name).read_bytes()
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
        assert entry["bytes"] == len(payload)
        assert entry["rows"] == payload.count(b"\n") - 1


def _member(header: list[str], rows: list[list[str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return ("﻿" + buffer.getvalue()).encode("utf-8")


def _bundle(path: Path, suppressed: str = "[c]") -> Path:
    """A zip shaped like the VOA bundle, with made-up small tables."""

    sectors = ["all", "retail", "offices", "industry", "other"]
    band_header = ["rv_band"] + [
        f"{kind}_{sector}" for sector in sectors for kind in ("count", "total_rv")
    ]
    band_rows = [
        ["0 - 999", *["20", "8"] * 5],
        ["1,000 and over", *["30", "90"] * 5],
        ["Total", *["50", "98"] * 5],
    ]
    type_header = ["prop_type_code", "description"] + [
        f"{name}_{suffix}"
        for suffix in ("engwal", "eng", "wal")
        for name in ("count", "total_rv", "mean_rv", "median_rv")
    ]
    full = ["50", "98", "2", "1.5"]
    type_rows = [
        ["[z]", "All Properties", *full * 3],
        ["C", "Commercial", *full * 3],
        ["CS", "Shops", *full * 3],
        ["CS8", "Hypermarket and premises", *full * 2, *[suppressed] * 4],
    ]
    year_header = ["geography", "abolished_year", "ba_code", "area_code", "area_name"]
    year_header += ["2023", "2024", "2025", "2026"]
    year_rows = [
        ["ENGWAL", "[z]", "[z]", "K04000001", "England and Wales", "1", "2", "3", "4"],
        ["NATL", "[z]", "[z]", "E92000001", "England", "1", "2", "3", "4"],
        ["NATL", "[z]", "[z]", "W92000004", "Wales", "1", "2", "3", "4"],
    ]
    with zipfile.ZipFile(path, "w") as archive:
        for name in ("table_SOP2_1.csv", "table_SOP2_2.csv", "table_SOP2_3.csv"):
            archive.writestr(name, _member(band_header, band_rows))
        archive.writestr("table_SOP1_2.csv", _member(type_header, type_rows))
        archive.writestr("table_SOP3_1.csv", _member(year_header, year_rows))
        archive.writestr("table_SOP3_2.csv", _member(year_header, year_rows))
    return path


def test_tool_extracts_checks_and_refuses_a_bad_cell(tmp_path: Path, capsys) -> None:
    tool = _load_tool()
    bundle = _bundle(tmp_path / "bundle.zip")
    out = tmp_path / "out"

    assert tool.main(["--zip", str(bundle), "--out", str(out)]) == 0
    assert tool.main(["--zip", str(bundle), "--out", str(out), "--check"]) == 0
    bands = (out / "voa_rv_band_sector.csv").read_text(encoding="utf-8").splitlines()
    assert bands[0] == ",".join(UK_FIRM_RATEABLE_VALUE_SOURCE_COLUMNS)
    assert bands[1] == "england_and_wales,0 - 999,0,999,all,20,8"
    assert 'england,"1,000 and over",1000,,all,30,90' in bands
    assert "wales,Total,,,other,50,98" in bands
    types = (out / "voa_property_type.csv").read_text(encoding="utf-8").splitlines()
    assert "england,CS8,CS,shop_type,Hypermarket and premises,50,98,2,1.5," in types
    assert "wales,CS8,CS,shop_type,Hypermarket and premises,,,,,[c]" in types
    assert "wales,CS,C,sub_sector,Shops,50,98,2,1.5," in types
    years = (out / "voa_stock_by_year.csv").read_text(encoding="utf-8").splitlines()
    assert years[1:4] == [
        "england_and_wales,2024-03-31,2,2",
        "england_and_wales,2025-03-31,3,3",
        "england_and_wales,2026-03-31,4,4",
    ]
    record = json.loads((out / ".provenance.json").read_text(encoding="utf-8"))
    assert record["source"]["sha256"] == hashlib.sha256(bundle.read_bytes()).hexdigest()

    (out / "voa_stock_by_year.csv").write_text("stale\n", encoding="utf-8")
    assert tool.main(["--zip", str(bundle), "--out", str(out), "--check"]) == 1
    assert "voa_stock_by_year.csv" in capsys.readouterr().err

    with pytest.raises(ValueError, match="not a whole number"):
        tool.extract(_bundle(tmp_path / "bad.zip", suppressed="x"))
    with pytest.raises(argparse.ArgumentTypeError, match="YYYY-MM-DD"):
        tool._iso_date("22 November 2026")


def test_source_reads_england_bands_largest_first() -> None:
    source = _source()

    assert UK_FIRM_RATEABLE_VALUE_STATUS == "assumed"
    assert len(source.bands) == 18
    assert source.bands[0] == "1,000,000 and over"
    assert source.bands[-1] == "0 - 999"
    assert source.upper_gbp[0] == float("inf")
    assert source.lower_gbp[0] == 1_000_000.0
    assert source.published_hereditaments == _TOTALS["england"][0]
    assert source.published_total_rv_gbp == _TOTALS["england"][1] * 1e3
    assert abs(source.n_hereditaments - source.published_hereditaments) <= 100
    assert abs(source.total_gbp - source.published_total_rv_gbp) < 20e3
    payload = (FIXTURE / UK_FIRM_RATEABLE_VALUE_FILE).read_bytes()
    assert source.sha256 == hashlib.sha256(payload).hexdigest()


def test_source_refuses_a_malformed_table(tmp_path: Path) -> None:
    frame = _bands()
    with pytest.raises(ValueError, match="columns"):
        frame.drop(columns="sector").to_csv(tmp_path / "a.csv", index=False)
        read_uk_firm_rateable_value_source(tmp_path / "a.csv")
    england_all = (frame["geography"] == "england") & (frame["sector"] == "all")
    no_total = frame[~(england_all & (frame["rv_band"] == "Total"))]
    no_total.to_csv(tmp_path / "b.csv", index=False)
    with pytest.raises(ValueError, match="Total"):
        read_uk_firm_rateable_value_source(tmp_path / "b.csv")
    closed = frame.copy()
    closed.loc[england_all & (closed["band_upper_gbp"] == ""), "band_upper_gbp"] = "9"
    closed.loc[england_all & (closed["rv_band"] == "Total"), "band_upper_gbp"] = ""
    closed.to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="ascend"):
        read_uk_firm_rateable_value_source(tmp_path / "c.csv")


def test_value_of_top_returns_each_band_total_and_stays_in_bounds() -> None:
    source = _source()
    edges = np.cumsum(source.hereditaments)

    np.testing.assert_allclose(
        uk_firm_rateable_value_of_top(source, edges),
        np.cumsum(source.total_rv_gbp),
        rtol=1e-12,
    )
    assert uk_firm_rateable_value_of_top(source, np.array([0.0]))[0] == 0.0
    beyond = uk_firm_rateable_value_of_top(source, np.array([edges[-1] * 3]))[0]
    assert beyond == pytest.approx(source.total_gbp)
    # One hereditament at a time down the first 400,000: values fall with
    # rank, and each sits inside its band.
    ranks = np.arange(0.0, 400_001.0)
    each = np.diff(uk_firm_rateable_value_of_top(source, ranks))
    assert np.all(np.diff(each) <= 1e-3)
    band = np.searchsorted(edges, ranks[1:], side="left")
    lower = np.asarray(source.lower_gbp)[band]
    upper = np.asarray(source.upper_gbp)[band]
    assert np.all(each >= lower - 1e-3)
    assert np.all(each <= upper + 1e-3)


def _firms(n: int, weight: float, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "firm_employees": rng.integers(0, 400, size=n),
            "firm_weight": np.full(n, weight),
            "firm_in_england": rng.random(n) < 0.85,
        }
    )


def test_assignment_follows_headcount_rank_and_conserves_the_stock() -> None:
    source = _source()
    firms = _firms(5_000, 800.0)
    value = assign_uk_firm_rateable_value(firms, source)
    england = firms["firm_in_england"].to_numpy()
    weight = firms["firm_weight"].to_numpy()

    assert value.shape == (len(firms),)
    assert np.all(value[~england] == 0.0)
    # 5,000 x 0.85 x 800 weighted firms exceed the 2.01 million hereditaments,
    # so all of England's rateable value is handed out and the smallest get 0.
    assert weight[england].sum() > source.n_hereditaments
    assert (weight * value).sum() == pytest.approx(source.total_gbp, rel=1e-9)
    with_property = weight[value > 0].sum()
    assert source.n_hereditaments <= with_property < source.n_hereditaments + 800.0
    by_size = firms[england].assign(value=value[england]).sort_values("firm_employees")
    means = by_size.groupby("firm_employees")["value"].agg(["min", "max"])
    # Up to float noise from differencing cumulative totals.
    smaller = means["max"].to_numpy()[:-1]
    assert np.all(means["min"].to_numpy()[1:] >= smaller * (1 - 1e-6) - 1e-3)
    summary = uk_firm_rateable_value_summary(firms, source)
    assert summary["weighted_rateable_value_gbp"] == pytest.approx(source.total_gbp)
    assert summary["weighted_firms_with_property"] == with_property
    assert summary["weighted_firms_in_england"] == weight[england].sum()


def test_assignment_with_fewer_firms_than_hereditaments_takes_the_top() -> None:
    source = _source()
    firms = pd.DataFrame(
        {
            "employment": [5, 900, 40, 40, 0],
            "firm_in_england": [True, True, True, False, True],
        }
    )
    value = assign_uk_firm_rateable_value(firms, source)

    top_band_mean = source.total_rv_gbp[0] / source.hereditaments[0]
    assert value[3] == 0.0
    # Four England firms, unweighted, take the four most valuable
    # hereditaments, all in the flat open top band.
    assert value[[0, 1, 2, 4]] == pytest.approx(np.full(4, top_band_mean))


def test_assignment_is_seeded_and_validates_its_inputs() -> None:
    source = _source()
    firms = _firms(2_000, 1_500.0)
    firms["firm_employees"] = 3

    first = assign_uk_firm_rateable_value(firms, source)
    np.testing.assert_array_equal(first, assign_uk_firm_rateable_value(firms, source))
    other = assign_uk_firm_rateable_value(firms, source, seed=1)
    assert not np.array_equal(first, other)
    assert (firms["firm_weight"] * other).sum() == pytest.approx(source.total_gbp)

    with pytest.raises(ValueError, match="firm_in_england"):
        assign_uk_firm_rateable_value(firms.drop(columns="firm_in_england"), source)
    with pytest.raises(ValueError, match="firm_employees or employment"):
        assign_uk_firm_rateable_value(firms.drop(columns="firm_employees"), source)
    with pytest.raises(ValueError, match="non-negative"):
        assign_uk_firm_rateable_value(firms.assign(firm_weight=-1.0), source)
    empty = assign_uk_firm_rateable_value(firms.assign(firm_in_england=False), source)
    assert np.all(empty == 0.0)
