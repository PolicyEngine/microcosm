"""The high-value homes family (microcosm#1240): stock builder functions and loader."""

from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import high_value_home_stock as hv
from microcosm.build.uk_runtime import high_value_homes
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
TOOLS = PATHS.repository / "tools"


def _load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# Price Paid reduction

_PPD_ROWS = [
    # id, price, date, postcode, type, old/new, duration, paon, saon, street, locality, town, district, county, category, status
    (
        "{1}",
        "250000",
        "2010-03-05 00:00",
        "AB1 2CD",
        "D",
        "N",
        "F",
        "1",
        "",
        "HIGH ST",
        "",
        "TOWN",
        "DIST",
        "CTY",
        "A",
        "A",
    ),
    (
        "{2}",
        "400000",
        "2020-06-10 00:00",
        "AB1 2CD",
        "D",
        "N",
        "F",
        "1",
        "",
        "HIGH ST",
        "",
        "TOWN",
        "DIST",
        "CTY",
        "A",
        "A",
    ),
    (
        "{3}",
        "150000",
        "2015-01-20 00:00",
        "AB1 2CD",
        "F",
        "N",
        "L",
        "2",
        "FLAT 1",
        "HIGH ST",
        "",
        "TOWN",
        "DIST",
        "CTY",
        "A",
        "A",
    ),
    (
        "{4}",
        "150000",
        "2015-01-20 00:00",
        "AB1 2CD",
        "F",
        "N",
        "L",
        "2",
        "FLAT 2",
        "HIGH ST",
        "",
        "TOWN",
        "DIST",
        "CTY",
        "A",
        "A",
    ),
    (
        "{5}",
        "999999",
        "2026-02-01 00:00",
        "AB1 2CD",
        "T",
        "Y",
        "F",
        "9",
        "",
        "NEW RD",
        "",
        "TOWN",
        "DIST",
        "CTY",
        "A",
        "A",
    ),
    (
        "{6}",
        "800000",
        "2021-09-09 00:00",
        "AB1 2CD",
        "O",
        "N",
        "F",
        "3",
        "",
        "HIGH ST",
        "",
        "TOWN",
        "DIST",
        "CTY",
        "A",
        "A",
    ),
    (
        "{7}",
        "700000",
        "2021-09-09 00:00",
        "AB1 2CD",
        "S",
        "N",
        "F",
        "4",
        "",
        "HIGH ST",
        "",
        "TOWN",
        "DIST",
        "CTY",
        "B",
        "A",
    ),
    (
        "{8}",
        "300000",
        "2019-12-31 00:00",
        "ZZ9 9ZZ",
        "S",
        "N",
        "F",
        "5",
        "",
        "FAR ST",
        "",
        "TOWN",
        "DIST",
        "CTY",
        "A",
        "A",
    ),
]


def _ppd_csv(tmp_path: Path) -> Path:
    path = tmp_path / "pp-test.csv"
    lines = [",".join(f'"{field}"' for field in row) for row in _PPD_ROWS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _postcode_index() -> pd.Index:
    return pd.Index(["AB12CD", "EF34GH"])


def test_normalise_postcode_drops_spaces_and_upper_cases() -> None:
    out = hv.normalise_postcode(pd.Series(["ab1 2cd", "EF3  4GH", None]))
    assert list(out[:2]) == ["AB12CD", "EF34GH"]
    assert pd.isna(out[2])


def test_read_price_paid_file_keeps_category_a_residential_and_flags_bulk(
    tmp_path,
) -> None:
    frame = hv.read_price_paid_file(
        _ppd_csv(tmp_path), postcode_index=_postcode_index()
    )
    # Rows 6 (type O) and 7 (category B) are dropped.
    assert len(frame) == 6
    assert set(frame.columns) == {
        "key",
        "yyyymm",
        "price",
        "ptype",
        "postcode_row",
        "newbuild",
        "batch",
    }
    # The two flats share date, price and postcode across two dwellings: bulk.
    assert int(frame["batch"].sum()) == 2
    assert frame.loc[frame["price"] == 150000, "batch"].all()
    # The same dwelling sold twice carries one key.
    keys = frame.loc[frame["price"].isin([250000, 400000]), "key"]
    assert keys.nunique() == 1
    assert int((frame["postcode_row"] < 0).sum()) == 1  # ZZ9 9ZZ is not in the NSPL
    assert frame["yyyymm"].dtype == np.int32
    assert int(frame.loc[frame["price"] == 999999, "newbuild"].iloc[0]) == 1


def test_latest_valuations_collapses_to_one_row_per_dwelling(tmp_path) -> None:
    frame = hv.read_price_paid_file(
        _ppd_csv(tmp_path), postcode_index=_postcode_index()
    )
    out = hv.latest_valuations(frame, stock_yyyymm=202512)
    assert len(out) == 5  # 1 (two sales), flat 1, flat 2, new build, far st
    repeat = out[out["first_yyyymm"] == 201003].iloc[0]
    assert repeat["yyyymm"] == 202006 and repeat["price"] == 400000 and repeat["valued"]
    flats = out[out["first_yyyymm"] == 201501]
    assert (
        len(flats) == 2 and not flats["valued"].any()
    )  # bulk-only dwellings carry no valuation
    newbuild = out[out["first_yyyymm"] == 202602].iloc[0]
    assert not newbuild[
        "in_stock"
    ]  # first registered as a new build after the stock date
    assert list(out["key"]) == sorted(out["key"])


def test_latest_valuations_handles_empty_input() -> None:
    out = hv.latest_valuations(hv._empty_transfers(), stock_yyyymm=202512)
    assert len(out) == 0 and "in_stock" in out.columns


# ----------------------------------------------------------------------------
# HPI


def _hpi_csv(tmp_path: Path) -> Path:
    header = "Date,RegionName,AreaCode," + ",".join(hv.HPI_INDEX_COLUMNS)
    rows = [header]
    for month in range(1, 13):
        # Local authority: detached grows, flats blank; region and nation complete.
        rows.append(
            f"01/{month:02d}/2024,Town,E06000001,{100 + month},,,,{100 + month}"
        )
        rows.append(
            f"01/{month:02d}/2024,Region,E12000001,{90 + month},{90 + month},{90 + month},{90 + month},{90 + month}"
        )
        rows.append(
            f"01/{month:02d}/2024,England,E92000001,{80 + month},{80 + month},{80 + month},{80 + month},{80 + month}"
        )
    rows.append("01/06/2010,Town,E06000001,50,,,,50")
    rows.append("01/06/2010,Region,E12000001,45,45,45,45,45")
    rows.append("01/06/2010,England,E92000001,40,40,40,40,40")
    rows.append("01/04/2026,England,E92000001,120,120,120,120,120")
    rows.append("01/04/2026,Town,E06000001,130,,,,130")
    rows.append("01/04/2026,Region,E12000001,125,125,125,125,125")
    path = tmp_path / "hpi.csv"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_load_hpi_and_uprate_with_fallback_chain(tmp_path) -> None:
    hpi = hv.load_hpi(_hpi_csv(tmp_path))
    assert hpi.area_codes == ("E06000001", "E12000001", "E92000001")
    assert hpi.last_month == (2026, 4)
    basis = hpi.basis_levels(2024)
    la = hpi.area_index["E06000001"]
    assert basis[la, 0] == pytest.approx(106.5)  # detached: mean of 101..112
    assert np.isnan(basis[la, 3])  # flats blank at the local authority
    values, used = hv.uprate_to_basis(
        hpi,
        price=np.array([100.0, 100.0, 100.0]),
        yyyymm=np.array([201006, 201006, 199001]),
        ptype=np.array([0, 3, 0]),
        area_chain=(
            np.array([la, la, la]),
            np.array([hpi.area_index["E12000001"]] * 3),
            np.array([hpi.area_index["E92000001"]] * 3),
        ),
        basis_year=2024,
    )
    assert values[0] == pytest.approx(100.0 * 106.5 / 50.0)  # local authority, detached
    # A flat with no local flat index takes the authority's all-dwellings index
    # before any link further up the chain.
    assert values[1] == pytest.approx(100.0 * 106.5 / 50.0)
    assert used[0] == 0 and used[1] == 1
    assert np.isnan(values[2]) and used[2] == -1  # sale month before the index starts


def test_deflate_thresholds_uses_the_nation_all_dwellings_index(tmp_path) -> None:
    hpi = hv.load_hpi(_hpi_csv(tmp_path))
    basis, factor = hv.deflate_thresholds(
        hpi,
        thresholds=[1_000_000, 2_000_000],
        nation_area_code="E92000001",
        basis_year=2024,
        reference_year=2026,
        reference_month=4,
    )
    assert factor == pytest.approx(86.5 / 120.0)
    assert basis[2_000_000.0] == pytest.approx(2_000_000 * 86.5 / 120.0)


# ----------------------------------------------------------------------------
# Counts and knots


def _stock() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "basis_value": [
                500_000.0,
                1_200_000.0,
                2_500_000.0,
                900_000.0,
                3_000_000.0,
            ],
            "constituency_code": [
                "E14000001",
                "E14000001",
                "E14000001",
                "E14000002",
                "E14000002",
            ],
            "local_authority_code": ["E06000001"] * 5,
        }
    )


def test_count_above_thresholds_sums_weights_per_area() -> None:
    out = hv.count_above_thresholds(
        _stock(),
        thresholds_basis={"over_gbp_1m": 1_000_000.0, "over_gbp_2m": 2_000_000.0},
        weight=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        group_columns=["constituency_code", "local_authority_code"],
    )
    rows = {
        (r.level, r.code, r.threshold_id): r.value for r in out.itertuples(index=False)
    }
    assert rows[("constituency_code", "E14000001", "over_gbp_1m")] == 5.0
    assert rows[("constituency_code", "E14000001", "over_gbp_2m")] == 3.0
    assert rows[("constituency_code", "E14000002", "over_gbp_1m")] == 5.0
    assert rows[("local_authority_code", "E06000001", "over_gbp_2m")] == 8.0


def test_value_knots_winsorise_and_suppress_small_areas() -> None:
    knots, suppressed = hv.value_knots(
        _stock(),
        weight=np.ones(5),
        group_column="constituency_code",
        cap=2_600_000.0,
        minimum_count=3,
        grid=(0.0, 0.5, 1.0),
    )
    assert suppressed == ["E14000002"]
    (area,) = knots
    assert area.code == "E14000001" and area.count == 3
    assert area.quantiles[0] == 500_000.0
    assert area.quantiles[-1] == 2_500_000.0  # below the cap, so untouched
    assert all(a <= b for a, b in zip(area.quantiles, area.quantiles[1:], strict=False))
    capped, _ = hv.value_knots(
        _stock(),
        weight=np.ones(5),
        group_column="local_authority_code",
        cap=1_000_000.0,
        minimum_count=1,
        grid=(1.0,),
    )
    assert capped[0].quantiles == (1_000_000.0,)


def test_rows_sha256_is_order_independent() -> None:
    a = [{"k": 1, "v": "x"}, {"k": 2, "v": "y"}]
    assert hv.rows_sha256(a) == hv.rows_sha256(list(reversed(a)))
    assert hv.rows_sha256(a) != hv.rows_sha256(a[:1])


def test_read_nspl_keeps_terminated_postcodes_and_dedupes() -> None:
    text = (
        "pcds,oa21cd,doterm\nAB1 2CD,E00000001,\nAB1 2CD,E00000002,202001\nEF3 4GH,,\n"
    )
    out = hv.read_nspl_postcode_output_areas(io.StringIO(text))
    assert list(out["postcode"]) == ["AB12CD"]
    assert list(out["oa21cd"]) == ["E00000001"]


# ----------------------------------------------------------------------------
# VOA band mix


def test_voa_band_mix_reads_markers_as_zero_and_is_cumulative() -> None:
    tool = _load_tool("build_uk_voa_band_mix")
    table = pd.DataFrame(
        {
            "geography": ["LSOA", "LSOA", "LAUA"],
            "ecode": ["E01000001", "W01000001", "E06000001"],
            "band_a": ["10", "-", "100"],
            "band_b": ["20", "5", "100"],
            "band_c": ["..", "5", "100"],
            "band_d": ["-", "-", "100"],
            "band_e": ["-", "-", "100"],
            "band_f": ["-", "-", "100"],
            "band_g": ["-", "-", "100"],
            "band_h": ["10", "-", "100"],
            "band_i": ["..", "10", ".."],
            "all_properties": ["40", "20", "800"],
        }
    )
    mix, receipts = tool.band_mix(table)
    assert set(mix) == {"E01000001", "W01000001"}
    england = mix["E01000001"]["cumulative"]
    assert england[0] == pytest.approx(0.25) and england[1] == pytest.approx(0.75)
    assert england[-2] == pytest.approx(1.0) and england[-1] == 1.0
    wales = mix["W01000001"]["cumulative"]
    assert wales[-2] == pytest.approx(0.5) and wales[-1] == 1.0
    assert receipts["lsoa_rows"] == 2 and receipts["band_i_outside_wales_nonzero"] == 0
    assert all(a <= b for a, b in zip(england, england[1:], strict=False))


# ----------------------------------------------------------------------------
# Loader


def _row(
    code: str,
    level: str,
    vintage: str,
    threshold: str,
    measure: str = "owner_occupied_homes",
    value=10.0,
) -> dict:
    return {
        "schema_version": high_value_homes.ROW_SCHEMA_VERSION,
        "aggregate_fact_key": f"{high_value_homes.DERIVED_FACT_KEY_PREFIX}{code}{threshold}{measure}",
        "value": value,
        "geography": {
            "level": level,
            "id": code,
            "vintage": vintage,
            "name": "Somewhere",
        },
        "observed_measure": {
            "source_concept": "x",
            "source_measure_id": measure,
            "unit": "count",
        },
        "dimensions": {
            "microcosm.value_threshold": threshold,
            "uk.dwelling.tenure": "owner_occupied",
        },
        "dimension_labels": {
            "microcosm.value_threshold": "Threshold",
            "uk.dwelling.tenure": "Tenure",
        },
        "dimension_value_labels": {
            "microcosm.value_threshold": {threshold: "over"},
            "uk.dwelling.tenure": {"owner_occupied": "Owner-occupied"},
        },
        "layout": {
            "record_set_spec_id": f"uk.local_geography.high_value_homes.{measure}.{threshold}.v1"
        },
        "aggregation": {"method": "sum"},
        "label": "Owner-occupied dwellings over, Somewhere",
    }


_PROVENANCE = {"supports": {"uk_ew_output_area_2021": {"sha256": "ab" * 32}}}
_CROSSWALK = {
    "levels": {
        "constituency": {"expected_vintage": ["pcon_2024"]},
        "local_authority": {"expected_vintage": {"E": ["lad_2023"], "W": ["lad_2023"]}},
    }
}


def _resource(rows: list[dict], **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "kind": high_value_homes.RESOURCE_KIND,
        "inputs": {"support_sha256": "ab" * 32},
        "inputs_sha256": "cd" * 32,
        "generator": {"tool": "t", "version": "v"},
        "parameters": {
            "thresholds_gbp_at_reference_month": {
                "over_gbp_1m": 1e6,
                "over_gbp_1_5m": 1.5e6,
                "over_gbp_2m": 2e6,
            },
            "frame_basis_thresholds": {
                "over_gbp_1m": 0.9e6,
                "over_gbp_1_5m": 1.35e6,
                "over_gbp_2m": 1.8e6,
            },
            "basis_year": 2024,
            "stock_date": "2025-12-31",
        },
        "row_count": len(rows),
        "rows_sha256": high_value_homes._rows_sha256(rows),
        "rows": rows,
    }
    payload.update(overrides)
    return payload


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loader_accepts_a_well_formed_resource(tmp_path) -> None:
    rows = [
        _row("E14000001", "constituency", "pcon_2024", "over_gbp_2m"),
        _row("W06000001", "local_authority", "lad_2023", "over_gbp_1m"),
    ]
    facts = high_value_homes.load_uk_high_value_home_stock_facts(
        _write(tmp_path, _resource(rows)),
        support_provenance=_PROVENANCE,
        crosswalk=_CROSSWALK,
    )
    assert len(facts.rows) == 2
    thresholds = facts.thresholds
    assert [t.metric_name for t in thresholds] == [
        "high_value_homes/over_1m",
        "high_value_homes/over_1_5m",
        "high_value_homes/over_2m",
    ]
    assert thresholds[2].basis_gbp == 1.8e6
    assert facts.rows_at(
        level="constituency",
        measure_id="owner_occupied_homes",
        threshold_id="over_gbp_2m",
    ) == {"E14000001": 10.0}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p.update(rows_sha256="00" * 32),
            "differ from their recorded digest",
        ),
        (
            lambda p: p["inputs"].update(support_sha256="ff" * 32),
            "other than the committed one",
        ),
        (
            lambda p: p["rows"][0]["geography"].update(vintage="pcon_2010"),
            "carries vintage",
        ),
        (lambda p: p["rows"][0].update(label=""), "requires a label"),
        (lambda p: p["rows"][0].update(value=-1.0), "invalid value"),
        (
            lambda p: p["rows"][0]["dimension_labels"].pop("uk.dwelling.tenure"),
            "lacks a label",
        ),
        (
            lambda p: p["parameters"]["frame_basis_thresholds"].pop("over_gbp_2m"),
            "lacks the basis value",
        ),
    ],
)
def test_loader_refuses_malformed_resources(tmp_path, mutate, message) -> None:
    rows = [_row("E14000001", "constituency", "pcon_2024", "over_gbp_2m")]
    payload = _resource(rows)
    mutate(payload)
    if "rows_sha256" not in message and "differ" not in message:
        payload["rows_sha256"] = high_value_homes._rows_sha256(payload["rows"])
    with pytest.raises(ValueError, match=message):
        high_value_homes.load_uk_high_value_home_stock_facts(
            _write(tmp_path, payload),
            support_provenance=_PROVENANCE,
            crosswalk=_CROSSWALK,
        )


def test_high_value_home_mask_requires_owner_tenure_and_stored_value_above_basis() -> (
    None
):
    values = np.array([2.0e6, 2.0e6, 1.0e6, 0.0])
    tenure = np.array(
        ["OWNED_OUTRIGHT", "RENT_PRIVATELY", "OWNED_WITH_MORTGAGE", "OWNED_OUTRIGHT"]
    )
    mask = high_value_homes.high_value_home_mask(
        values, tenure, threshold_basis_gbp=1.8e6
    )
    assert mask.tolist() == [True, False, False, False]
    mask = high_value_homes.high_value_home_mask(
        values, tenure, threshold_basis_gbp=0.9e6
    )
    assert mask.tolist() == [True, False, True, False]


def test_committed_resource_loads_and_matches_the_crosswalk() -> None:
    facts = high_value_homes.load_uk_high_value_home_stock_facts()
    levels = {row["geography"]["level"] for row in facts.rows}
    assert levels == {"constituency", "local_authority", "region", "country"}
    codes = {
        row["geography"]["id"][0]
        for row in facts.rows
        if row["geography"]["level"] == "constituency"
    }
    assert codes == {"E", "W"}
    assert [t.basis_gbp for t in facts.thresholds] == sorted(
        t.basis_gbp for t in facts.thresholds
    )
    assert all(t.basis_gbp < t.nominal_gbp for t in facts.thresholds)
