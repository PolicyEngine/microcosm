"""DESNZ domestic energy split onto constituencies (microcosm#1095): builder and loader."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from importlib import resources as importlib_resources
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import desnz_constituency_energy as energy
from microcosm.build.uk_runtime.chronicle_feed import load_uk_chronicle_feed
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
TOOLS = PATHS.repository / "tools"


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "build_uk_desnz_constituency_energy_facts",
        TOOLS / "build_uk_desnz_constituency_energy_facts.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------
# Synthetic feed rows and support

_UNITS = {"consumption": "kwh", "meter_count": "count"}


def _fact(record_set: str, fuel: str, metric: str, level: str, code: str, value, **geo):
    period_type = "gas_year" if fuel == "gas" else "calendar_year"
    return {
        "value": value,
        "geography": {"level": level, "id": code, "name": f"Area {code}", **geo},
        "layout": {"record_set_id": record_set},
        "dimensions": {
            "domestic_scope": "all_domestic",
            "fuel": fuel,
            "metric": metric,
            "temperature_adjustment": (
                "weather_corrected" if fuel == "gas" else "as_reported"
            ),
        },
        "dimension_labels": {
            "desnz.subnational.area": "Published geography",
            "domestic_scope": "Domestic scope",
            "fuel": "Fuel",
            "metric": "Metric",
            "temperature_adjustment": "Temperature adjustment",
        },
        "dimension_value_labels": {
            "domestic_scope": {"all_domestic": "All domestic meters"},
            "fuel": {fuel: fuel.title()},
            "metric": {metric: metric.replace("_", " ").title()},
            "temperature_adjustment": {
                "weather_corrected": "Weather corrected",
                "as_reported": "As reported",
            },
        },
        "entity": {"name": "household", "role": "domestic_energy_customer"},
        "observed_measure": {
            "source_concept": energy.CONCEPTS[(fuel, metric)],
            "source_measure_id": f"domestic_{metric}",
            "unit": _UNITS[metric],
        },
        "period": {"type": period_type, "value": 2024},
        "period_coverage": {"source_period_label": "2024"},
        "source": {"source_name": "desnz", "vintage": "2025_publication_2024_data"},
    }


#: Two English authorities (A in two seats, B in one) and one recoded code.
_TOTALS = {"E06000001": 151.0, "E08000038": 30.0}
_SMALL = {"E01000001": 100.0, "E01000002": 50.0, "E01000003": 30.0}


def _feed(totals=_TOTALS, small=_SMALL, *, unallocated=1.0) -> list[dict]:
    rows = []
    for fuel in energy.FUELS:
        for metric in energy.METRICS:
            authority = energy.authority_record_set_id(fuel, 2024)
            small_area = energy.small_area_record_set_id(fuel, 2024)
            for code, value in totals.items():
                rows.append(
                    _fact(authority, fuel, metric, "local_authority", code, value)
                )
            rows.append(_fact(authority, fuel, metric, "region", "E12000001", 1e9))
            for code, value in small.items():
                rows.append(
                    _fact(
                        small_area,
                        fuel,
                        metric,
                        "statistical_scope",
                        code,
                        value,
                        vintage="lsoa_2021",
                    )
                )
            rows.append(
                _fact(
                    small_area,
                    fuel,
                    metric,
                    "statistical_scope",
                    energy.UNALLOCATED_GEOGRAPHY_ID,
                    unallocated,
                    vintage="current",
                )
            )
    return rows


_ALIASES = {"E08000038": "E08000016"}


def _support() -> pd.DataFrame:
    """LSOA E01000001 straddles seats X and Y (60 and 40 households)."""
    return pd.DataFrame(
        {
            "area": ["E00000001", "E00000002", "E00000003", "E00000004"],
            "lsoa_code": ["E01000001", "E01000001", "E01000002", "E01000003"],
            "local_authority_code": [
                "E06000001",
                "E06000001",
                "E06000001",
                "E08000016",
            ],
            "constituency_code": ["E14000001", "E14000002", "E14000001", "E14000002"],
            "region_code": ["E12000001"] * 4,
            "households": np.array([60, 40, 25, 30], dtype=np.int64),
        }
    )


def _series(key=("electricity", "consumption"), **kwargs) -> energy.EnergySeries:
    return energy.select_series(_feed(**kwargs), year=2024, aliases=_ALIASES)[key]


# ----------------------------------------------------------------------------
# Series selection


def test_select_series_reads_four_series_and_aliases_recoded_authorities() -> None:
    series = energy.select_series(_feed(), year=2024, aliases=_ALIASES)
    assert sorted(series) == [
        ("electricity", "consumption"),
        ("electricity", "meter_count"),
        ("gas", "consumption"),
        ("gas", "meter_count"),
    ]
    one = series[("gas", "meter_count")]
    # The region row is not an authority; the recoded code reads as the roster's.
    assert one.authority_totals == {"E06000001": 151.0, "E08000016": 30.0}
    assert one.small_area_values == _SMALL
    assert one.unallocated == 1.0
    assert one.template["geography"]["id"] == "E06000001"
    assert one.template["period"] == {"type": "gas_year", "value": 2024}


def test_select_series_refuses_duplicates_other_units_and_a_missing_series() -> None:
    rows = _feed()
    rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(ValueError, match="two authority rows"):
        energy.select_series(rows, year=2024, aliases=_ALIASES)
    rows = _feed()
    rows[0]["observed_measure"]["unit"] = "gwh"
    with pytest.raises(ValueError, match="is not in kwh"):
        energy.select_series(rows, year=2024, aliases=_ALIASES)
    rows = [
        row
        for row in _feed()
        if row["layout"]["record_set_id"]
        != energy.small_area_record_set_id("gas", 2024)
    ]
    with pytest.raises(ValueError, match="the split needs both"):
        energy.select_series(rows, year=2024, aliases=_ALIASES)


# ----------------------------------------------------------------------------
# The split


def test_split_keeps_every_authority_total_and_splits_straddling_areas_by_households() -> (
    None
):
    split = energy.split_series(_support(), _series())
    # Authority A: keys 100 * 0.6 + 50 = 110 in seat 1, 100 * 0.4 = 40 in seat 2,
    # sharing its published 151; authority B carries its 30 whole into seat 2.
    assert split.seats["E14000001"] == pytest.approx(151.0 * 110 / 150)
    assert split.seats["E14000002"] == pytest.approx(151.0 * 40 / 150 + 30.0)
    by_authority = split.pieces.groupby("local_authority_code")["value"].sum()
    assert by_authority.to_dict() == pytest.approx(
        {"E06000001": 151.0, "E08000016": 30.0}
    )
    receipt = split.receipt
    assert receipt["small_areas_straddling_seats"] == 1
    assert receipt["households_share_in_straddling_small_areas"] == pytest.approx(
        100 / 155, abs=1e-5
    )
    assert receipt["authorities_split_over_seats"] == 1
    assert receipt["seats_of_whole_authorities"] == 0
    assert receipt["unallocated_share_of_authority_total"] == pytest.approx(
        1 / 181, abs=1e-5
    )
    assert receipt["seat_total"] == pytest.approx(receipt["authority_total"])


def test_a_small_area_without_a_row_carries_no_key() -> None:
    small = {"E01000001": 100.0, "E01000003": 30.0}
    split = energy.split_series(
        _support(), _series(totals={"E06000001": 101.0, "E08000038": 30.0}, small=small)
    )
    # E01000002 has no row: authority A's total follows E01000001's households.
    assert split.seats["E14000001"] == pytest.approx(101.0 * 60 / 100)
    assert split.receipt["small_areas_in_support_without_row"] == 1


def test_an_authority_without_a_total_is_left_out() -> None:
    split = energy.split_series(
        _support(),
        _series(
            totals={"E06000001": 151.0}, small={"E01000001": 100.0, "E01000002": 50.0}
        ),
    )
    assert split.receipt["authorities_in_support_without_total"] == ["E08000016"]
    assert split.seats["E14000002"] == pytest.approx(151.0 * 40 / 150)


def _empty_small_area(support, totals, values) -> None:
    support.loc[support["lsoa_code"] == "E01000003", "households"] = 0


def _seat_across_regions(support, totals, values) -> None:
    support.loc[support["area"] == "E00000004", "region_code"] = "E12000002"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda s, t, v: v.update({"E01000002": 0.0, "E01000001": 0.0}),
            "carry a total but no key",
        ),
        (lambda s, t, v: v.update({"E01999999": 1.0}), "not in the atomic supports"),
        (
            lambda s, t, v: t.pop("E08000016"),
            "value in an authority with no published total",
        ),
        (_empty_small_area, "carry a value but no households"),
        (_seat_across_regions, "output areas of two regions"),
        (lambda s, t, v: t.update({"E06000001": 160.0}), "away from the authority"),
        (lambda s, t, v: t.update({"E06000099": 5.0}), "carry a total but no output"),
    ],
)
def test_split_refusals(change, message) -> None:
    support = _support()
    series = _series()
    totals = dict(series.authority_totals)
    values = dict(series.small_area_values)
    change(support, totals, values)
    altered = energy.EnergySeries(
        fuel=series.fuel,
        metric=series.metric,
        authority_totals=totals,
        small_area_values=values,
        unallocated=series.unallocated,
        template=series.template,
    )
    with pytest.raises(ValueError, match=message):
        energy.split_series(support, altered)


def test_support_frame_refuses_an_output_area_in_two_supports() -> None:
    arrays = {
        column: _support()[column].to_numpy() for column in energy.SUPPORT_COLUMNS
    }
    with pytest.raises(ValueError, match="in two atomic supports"):
        energy.support_frame([arrays, arrays])
    with pytest.raises(ValueError, match="lacks columns"):
        energy.support_frame([{"area": arrays["area"]}])


# ----------------------------------------------------------------------------
# Rows


def _rows(**kwargs):
    series = energy.select_series(_feed(**kwargs), year=2024, aliases=_ALIASES)
    splits = {
        key: energy.split_series(_support(), value) for key, value in series.items()
    }
    names = {"E14000001": "Seat One", "E14000002": "Seat Two"}
    return series, energy.seat_fact_rows(
        series, splits, year=2024, seat_names=names, inputs_sha256="ab" * 32
    )


def test_seat_rows_keep_the_published_concept_period_and_dimensions() -> None:
    series, rows = _rows()
    assert len(rows) == 8
    assert rows == sorted(rows, key=lambda row: row["aggregate_fact_key"])
    for row in rows:
        assert row["aggregate_fact_key"].startswith(energy.DERIVED_FACT_KEY_PREFIX)
        assert row["geography"]["level"] == "constituency"
        assert row["geography"]["vintage"] == "pcon_2024"
        assert row["source"]["source_name"] == "microcosm"
        assert row["concept_alignment"]["relation"] == "derived"
        fuel, metric = row["dimensions"]["fuel"], row["dimensions"]["metric"]
        template = series[(fuel, metric)].template
        assert (
            row["observed_measure"]["source_concept"] == energy.CONCEPTS[(fuel, metric)]
        )
        assert row["period"] == template["period"]
        assert row["dimensions"] == template["dimensions"]
        # The publisher's area dimension is a layout group, not a seat dimension.
        assert "desnz.subnational.area" not in row["dimension_labels"]


def test_a_seat_without_a_piece_of_a_fuel_carries_no_row() -> None:
    series = energy.select_series(_feed(), year=2024, aliases=_ALIASES)
    gas_only_a = energy.EnergySeries(
        **{
            **series[("gas", "consumption")].__dict__,
            "authority_totals": {"E06000001": 151.0},
            "small_area_values": {"E01000001": 0.0, "E01000002": 150.0},
        }
    )
    split = energy.split_series(_support(), gas_only_a)
    assert set(split.seats) == {"E14000001"}
    rows = energy.seat_fact_rows(
        {("gas", "consumption"): gas_only_a},
        {("gas", "consumption"): split},
        year=2024,
        seat_names={"E14000001": "Seat One"},
        inputs_sha256="ab" * 32,
    )
    assert [row["geography"]["id"] for row in rows] == ["E14000001"]


def test_seat_rows_need_every_seat_name() -> None:
    series = energy.select_series(_feed(), year=2024, aliases=_ALIASES)
    splits = {
        key: energy.split_series(_support(), value) for key, value in series.items()
    }
    with pytest.raises(ValueError, match="No name for constituency"):
        energy.seat_fact_rows(
            series,
            splits,
            year=2024,
            seat_names={"E14000001": "Seat One"},
            inputs_sha256="ab" * 32,
        )


def test_rows_sha256_is_order_independent() -> None:
    _, rows = _rows()
    assert energy.rows_sha256(rows) == energy.rows_sha256(list(reversed(rows)))


# ----------------------------------------------------------------------------
# The loader

_PROVENANCE = {
    "supports": {
        "uk_ew_output_area_2021": {"sha256": "ab" * 32},
        "uk_scotland_output_area_2022": {"sha256": "cd" * 32},
    }
}
_CROSSWALK = {"levels": {"constituency": {"expected_vintage": ["pcon_2024"]}}}
_FEED_SHA = "ef" * 32


def _resource(rows: list[dict], **overrides) -> dict:
    payload = {
        "schema_version": 1,
        "kind": energy.RESOURCE_KIND,
        "inputs": {
            "support_sha256": {
                "uk_ew_output_area_2021": "ab" * 32,
                "uk_scotland_output_area_2022": "cd" * 32,
            },
            "ledger_feed": {"facts_sha256": _FEED_SHA},
        },
        "inputs_sha256": "12" * 32,
        "generator": {"tool": "t", "version": "v"},
        "parameters": {"year": 2024},
        "row_count": len(rows),
        "rows_sha256": energy.rows_sha256(rows),
        "rows": rows,
    }
    payload.update(overrides)
    return payload


def _load(tmp_path: Path, payload: dict):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return energy.load_uk_desnz_constituency_energy_facts(
        path,
        support_provenance=_PROVENANCE,
        crosswalk=_CROSSWALK,
        feed_facts_sha256=_FEED_SHA,
    )


def test_loader_accepts_a_well_formed_resource(tmp_path) -> None:
    _, rows = _rows()
    facts = _load(tmp_path, _resource(rows))
    assert len(facts.rows) == 8
    values = facts.values(fuel="gas", metric="meter_count", period_value=2024)
    assert values["E14000001"] == pytest.approx(151.0 * 110 / 150, abs=1e-3)
    meters = energy.seat_meter_counts(
        facts,
        year=2024,
        period_types={"electricity": "calendar_year", "gas": "gas_year"},
    )
    assert set(meters["E14000002"]) == {"electricity", "gas"}
    with pytest.raises(ValueError, match="stamped 'gas_year'"):
        energy.seat_meter_counts(
            facts,
            year=2024,
            period_types={"electricity": "calendar_year", "gas": "calendar_year"},
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda p: p.update(rows_sha256="00" * 32),
            "differ from their recorded digest",
        ),
        (
            lambda p: p["inputs"]["support_sha256"].update(
                uk_scotland_output_area_2022="ff" * 32
            ),
            "other than the committed ones",
        ),
        (
            lambda p: p["inputs"]["ledger_feed"].update(facts_sha256="00" * 32),
            "another Chronicle feed",
        ),
        (
            lambda p: p["rows"][0]["geography"].update(vintage="pcon_2010"),
            "carries vintage",
        ),
        (
            lambda p: p["rows"][0]["geography"].update(level="local_authority"),
            "not a constituency row",
        ),
        (
            lambda p: p["rows"][0]["geography"].update(id="N05000001"),
            "Great Britain code",
        ),
        (lambda p: p["rows"][0].update(label=""), "requires a label"),
        (lambda p: p["rows"][0].update(value=-1.0), "invalid value"),
        (
            lambda p: p["rows"][0]["observed_measure"].update(source_concept="x"),
            "names concept",
        ),
        (lambda p: p.pop("parameters"), "lacks parameter 'year'"),
    ],
)
def test_loader_refuses_malformed_resources(tmp_path, mutate, message) -> None:
    _, rows = _rows()
    payload = _resource(rows)
    mutate(payload)
    if "digest" not in message:
        payload["rows_sha256"] = energy.rows_sha256(payload["rows"])
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, payload)


# ----------------------------------------------------------------------------
# The builder tool, end to end on a synthetic feed and supports


def _npz(frame: pd.DataFrame) -> bytes:
    stream = io.BytesIO()
    metadata = {"columns": list(energy.SUPPORT_COLUMNS)}
    np.savez(
        stream,
        metadata_json=np.array(json.dumps(metadata)),
        **{
            column: (
                frame[column].to_numpy(dtype=np.int64)
                if column == "households"
                else frame[column].to_numpy().astype("U9")
            )
            for column in energy.SUPPORT_COLUMNS
        },
    )
    return stream.getvalue()


def test_the_tool_builds_a_resource_the_loader_accepts(tmp_path) -> None:
    tool = _load_tool()
    seat_rows = [
        {
            "geography": {"level": "constituency", "id": code, "name": name},
            "layout": {"record_set_spec_id": tool.SEAT_NAMES_SPEC_ID},
        }
        for code, name in (("E14000001", "Seat One"), ("E14000002", "Seat Two"))
    ]
    lines = [json.dumps(row) + "\n" for row in (*_feed(), *seat_rows)]
    feed = tmp_path / "consumer_facts.jsonl"
    feed.write_text("".join(lines), encoding="utf-8")
    facts_sha = hashlib.sha256(feed.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"facts_sha256": facts_sha}), encoding="utf-8")
    pin = tmp_path / "chronicle_feed.json"
    pin.write_text(
        json.dumps(
            {
                "source_commit": "0" * 40,
                "facts_sha256": facts_sha,
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "fact_row_count": len(lines),
            }
        ),
        encoding="utf-8",
    )
    ew = tmp_path / "ew.npz"
    ew.write_bytes(_npz(_support()))
    scotland_frame = pd.DataFrame(
        {
            "area": ["S00000001"],
            "lsoa_code": ["S01000001"],
            "local_authority_code": ["S12000001"],
            "constituency_code": ["S14000001"],
            "region_code": ["S99999999"],
            "households": np.array([10], dtype=np.int64),
        }
    )
    scotland = tmp_path / "scotland.npz"
    scotland.write_bytes(_npz(scotland_frame))
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "supports": {
                    "uk_ew_output_area_2021": {
                        "sha256": hashlib.sha256(ew.read_bytes()).hexdigest()
                    },
                    "uk_scotland_output_area_2022": {
                        "sha256": hashlib.sha256(scotland.read_bytes()).hexdigest()
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    crosswalk = tmp_path / "crosswalk.json"
    crosswalk.write_text(
        json.dumps(
            {
                "levels": {
                    "constituency": {"expected_vintage": ["pcon_2024"]},
                    "local_authority": {
                        "code_aliases": {"E08000016": {"alias_codes": ["E08000038"]}}
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "facts.json"
    arguments = [
        "--ledger-facts",
        str(feed),
        "--ledger-manifest",
        str(manifest),
        "--ew-support",
        str(ew),
        "--scotland-support",
        str(scotland),
        "--feed-pin",
        str(pin),
        "--support-provenance",
        str(provenance),
        "--crosswalk",
        str(crosswalk),
        "--built-on",
        "2026-10-25",
        "--out",
        str(out),
    ]
    assert tool.main(arguments) == 0
    facts = energy.load_uk_desnz_constituency_energy_facts(
        out,
        support_provenance=json.loads(provenance.read_text()),
        crosswalk=json.loads(crosswalk.read_text()),
        feed_facts_sha256=facts_sha,
    )
    assert len(facts.rows) == 8
    first = out.read_bytes()
    assert tool.main(arguments) == 0
    assert out.read_bytes() == first, "the build is not deterministic"
    # A feed that is not the pinned one is refused before anything is written.
    feed.write_text("".join(lines[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="is not the feed"):
        tool.main(arguments)


# ----------------------------------------------------------------------------
# The committed resource


def _vendored_meter_totals() -> dict[tuple[str, str], float]:
    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("desnz_local_energy_facts.json")
        .read_text(encoding="utf-8")
    )
    totals: dict[tuple[str, str], float] = {}
    for row in payload["rows"]:
        if row["geography"]["level"] != "local_authority":
            continue
        if row["dimensions"]["metric"] != "meter_count":
            continue
        key = (row["geography"]["id"][0], row["dimensions"]["fuel"])
        totals[key] = totals.get(key, 0.0) + float(row["value"])
    return totals


def test_committed_resource_splits_the_vendored_authority_meters_exactly() -> None:
    facts = energy.load_uk_desnz_constituency_energy_facts()
    assert facts.inputs["ledger_feed"]["facts_sha256"] == (
        load_uk_chronicle_feed().facts_sha256
    )
    vendored = _vendored_meter_totals()
    for fuel in energy.FUELS:
        seats = facts.values(fuel=fuel, metric="meter_count", period_value=2024)
        for nation in "EWS":
            seat_total = sum(
                value for code, value in seats.items() if code[0] == nation
            )
            # Seat values carry three decimals, so their roundings add up.
            assert seat_total == pytest.approx(vendored[(nation, fuel)], rel=1e-7)


def test_committed_resource_covers_great_britain_and_the_gas_network() -> None:
    facts = energy.load_uk_desnz_constituency_energy_facts()
    electricity = facts.values(
        fuel="electricity", metric="consumption", period_value=2024
    )
    gas = facts.values(fuel="gas", metric="consumption", period_value=2024)
    assert len(electricity) == 632
    assert {code[0] for code in electricity} == {"E", "W", "S"}
    # Orkney and Shetland has no mains gas network.
    assert set(electricity) - set(gas) == {"S14000051"}
    meters = energy.seat_meter_counts(
        facts,
        year=2024,
        period_types={"electricity": "calendar_year", "gas": "gas_year"},
    )
    shares = [
        counts["gas"] / counts["electricity"]
        for counts in meters.values()
        if "gas" in counts
    ]
    assert all(0.0 < share <= 1.0 for share in shares)
    for receipt in facts.receipts["series"].values():
        assert receipt["seat_total"] == pytest.approx(receipt["authority_total"])


def _crosswalk_regions(level: str) -> tuple[dict[str, str], dict[str, str]]:
    crosswalk = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("local_area_crosswalk.json")
        .read_text(encoding="utf-8")
    )
    entry = crosswalk["levels"][level]
    aliases = {
        str(code): str(area)
        for area, alias in (entry.get("code_aliases") or {}).items()
        for code in alias.get("alias_codes", ())
    }
    return dict(entry["region_code_by_area"]), aliases


def test_committed_resource_keeps_every_region_s_authority_meters() -> None:
    # The split keeps each authority's total and no seat crosses a region, so
    # seats and authorities sum alike in every region: the single-winner rule
    # then rescales the authority cells by one.
    facts = energy.load_uk_desnz_constituency_energy_facts()
    seat_region, _ = _crosswalk_regions("constituency")
    authority_region, aliases = _crosswalk_regions("local_authority")
    payload = json.loads(
        importlib_resources.files("microcosm.build.uk")
        .joinpath("desnz_local_energy_facts.json")
        .read_text(encoding="utf-8")
    )
    for fuel in energy.FUELS:
        by_region: dict[str, float] = {}
        for row in payload["rows"]:
            if row["geography"]["level"] != "local_authority":
                continue
            if row["dimensions"] != {
                **row["dimensions"],
                "fuel": fuel,
                "metric": "meter_count",
            }:
                continue
            code = aliases.get(row["geography"]["id"], row["geography"]["id"])
            region = authority_region[code]
            by_region[region] = by_region.get(region, 0.0) + float(row["value"])
        seats: dict[str, float] = {}
        for code, value in facts.values(
            fuel=fuel, metric="meter_count", period_value=2024
        ).items():
            region = seat_region[code]
            seats[region] = seats.get(region, 0.0) + value
        assert seats.keys() == by_region.keys()
        for region, total in by_region.items():
            assert seats[region] == pytest.approx(total, rel=1e-7), (fuel, region)


def test_the_committed_surface_binds_every_great_britain_seat() -> None:
    from microcosm.build.uk_runtime.ledger_targets import (
        load_uk_local_target_reference_membership,
    )

    targets = load_uk_local_target_reference_membership()["targets"]
    statuses = {}
    for target_id in (
        "desnz.subnational.electricity_kwh",
        "desnz.subnational.gas_kwh",
        "desnz.subnational.gas_connected_households",
    ):
        candidates = targets[target_id]["geography_levels"]["constituency"][
            "candidates"
        ]
        assert {candidate["geography_id"][0] for candidate in candidates} == {
            "E",
            "W",
            "S",
        }
        assert len(candidates) == 632
        statuses[target_id] = {
            candidate["geography_id"]: candidate["status"]
            for candidate in candidates
            if candidate["status"] != "active"
        }
    # Orkney and Shetland has no mains gas, as the two councils have none.
    assert statuses == {
        "desnz.subnational.electricity_kwh": {},
        "desnz.subnational.gas_kwh": {"S14000051": "no_fact_for_area"},
        "desnz.subnational.gas_connected_households": {"S14000051": "signed_deferred"},
    }


def test_the_withholding_families_of_the_energy_rows() -> None:
    # The kWh rows have no higher-grain parent, so they are checked in their
    # regions' blocks, where seats and authorities meet; the gas-connected
    # rows join the census household family and its nation controls.
    from microcosm.build.uk_runtime.diagnostics import uk_local_family_parents

    families = {
        member: (members, parents)
        for members, parents in uk_local_family_parents()
        for member in members
    }
    for target_id in ("desnz.subnational.electricity_kwh", "desnz.subnational.gas_kwh"):
        assert families[target_id] == (frozenset({target_id}), frozenset())
    members, parents = families["desnz.subnational.gas_connected_households"]
    assert "ons.census.households" in members
    assert parents
