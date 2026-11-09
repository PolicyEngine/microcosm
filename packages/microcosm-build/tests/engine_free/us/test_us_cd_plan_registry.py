"""Example tests for the US block -> congressional-district plan registry.

Property-based tests of the same invariants live in
``test_us_cd_plan_registry_properties.py``. The opt-in test at the bottom
checks a real built artifact when ``MICROCOSM_US_CD_PLAN_REGISTRY`` points at
one.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime.block_ladder_sources import US_STATES
from microcosm.build.us_runtime.cd_plan_registry import (
    CD_PLAN_ARRAY_PREFIX,
    PRIMARY_CD_PLAN,
    US_HOUSE_APPORTIONMENT,
    CdBlockAssignment,
    assemble_us_cd_plan_registry,
    cd_block_assignment,
    check_cd_plan_registry_against_block_ladder,
    expected_district_counts,
    load_pinned_us_cd_plan_registry,
    load_us_cd_plan_registry,
    packaged_us_cd_plan_registry_provenance,
    parse_cd_block_assignment,
    replace_state_assignments,
    summarize_cd_plan_registry,
)
from test_support.paths import paths_for

SHA = "0" * 64
_TEST_PATHS = paths_for("microcosm-build")


def _spec(plan: str, apportionment: str = "2020_census") -> dict:
    return {
        "vintage": plan,
        "source": f"test source for {plan}",
        "url": f"https://example.test/{plan}.zip",
        "apportionment": apportionment,
        "source_files": {
            plan: {"url": f"https://example.test/{plan}.zip", "sha256": SHA}
        },
    }


# Delaware (one seat) and Rhode Island (two seats in both apportionments).
DE = [10_001_0401_00_1000 + i for i in range(3)]
RI = [44_001_0301_00_1000 + i for i in range(4)]
BLOCKS = np.asarray(DE + RI, dtype=np.int64)
POPULATION = np.asarray([10, 20, 30, 40, 50, 60, 70], dtype=np.int64)
PLAN_119 = {
    **dict.fromkeys(DE, 1000),
    RI[0]: 4401,
    RI[1]: 4401,
    RI[2]: 4402,
    RI[3]: 4402,
}
PLAN_120 = {**PLAN_119, RI[1]: 4402}


def _payload(**overrides):
    kwargs = {
        "block_geoid": BLOCKS,
        "population": POPULATION,
        "plan_assignments": {
            "119th_congress": cd_block_assignment(PLAN_119, label="119"),
            "120th_congress": cd_block_assignment(PLAN_120, label="120"),
        },
        "plan_sources": {
            "119th_congress": _spec("119th_congress"),
            "120th_congress": _spec("120th_congress"),
        },
    }
    kwargs.update(overrides)
    return assemble_us_cd_plan_registry(**kwargs)


def _write(tmp_path: Path, payload, name: str = "registry.npz") -> Path:
    path = tmp_path / name
    np.savez_compressed(path, **payload)
    return path


# --- parsing -------------------------------------------------------------


@pytest.mark.parametrize(
    ("lines", "label"),
    [
        (
            [
                "BLOCKID|DISTRICT\n",
                "100010401001000|00\n",
                "440010301001000|02\n",
                "440010301001001|ZZ\n",
            ],
            "BAF CD layer",
        ),
        (
            [
                "GEOID, CDFP\n",
                "100010401001000,00\n",
                "440010301001000,02\n",
                "440010301001001,ZZ\n",
            ],
            "118th state file",
        ),
        (
            [
                "﻿GEOID,CDFP\r\n",
                "100010401001000,00\r\n",
                "440010301001000,02\r\n",
                "440010301001001,ZZ\r\n",
            ],
            "119th national file",
        ),
        (
            [
                "GEOID,STATEFP,COUNTYFP,TRACTCE,BLOCKCE,CDFP\n",
                "100010401001000,10,001,040100,1000,00\n",
                "440010301001000,44,001,030100,1000,02\n",
                "440010301001001,44,001,030100,1001,ZZ\n",
            ],
            "120th national file",
        ),
    ],
)
def test_parser_reads_every_census_format(lines, label):
    parsed = parse_cd_block_assignment(lines, label=label)
    assert parsed.block_geoid.tolist() == [100010401001000, 440010301001000]
    assert parsed.district_geoid.tolist() == [1000, 4402]


def test_parser_maps_the_delegate_code_to_district_00():
    parsed = parse_cd_block_assignment(["GEOID,CDFP", "110010001011000,98"], label="DC")
    assert parsed.district_geoid.tolist() == [1100]


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (["GEOID,CDFP", "100010401001000,00", "100010401001000,00"], "more than once"),
        (["GEOID,CDFP", "10001040100100,00"], "15 digits"),
        (["GEOID,CDFP", "100010401001000,00,7"], "fields"),
        (["GEOID,CDFP", "100010401001000,X1"], "two-digit"),
        (
            [
                "GEOID,STATEFP,COUNTYFP,TRACTCE,BLOCKCE,CDFP",
                "100010401001000,44,001,040100,1000,00",
            ],
            "STATEFP",
        ),
        (["GEOID,DISTRICTS", "100010401001000,00"], "none of the columns"),
        (["GEOID,CDFP", "100010401001000,ZZ"], "no district assignments"),
        ([], "empty"),
    ],
)
def test_parser_refuses_malformed_files(lines, message):
    with pytest.raises(ValueError, match=message):
        parse_cd_block_assignment(lines, label="test")


def test_parser_consumes_a_stream_lazily():
    def lines():
        yield "GEOID,CDFP"
        for suffix in range(1000):
            yield f"{100010401000000 + suffix},00"

    parsed = parse_cd_block_assignment(lines(), label="stream")
    assert len(parsed) == 1000
    assert bool(np.all(np.diff(parsed.block_geoid) > 0))


def test_parser_stops_at_a_bad_record_without_reading_ahead():
    # A parser that materialized the stream first would hit the tail's
    # AssertionError instead of reporting the bad record.
    def lines():
        yield "GEOID,CDFP"
        yield "100010401001000,00"
        yield "10001040100100X,00"
        raise AssertionError("the parser read past the bad record")

    with pytest.raises(ValueError, match="15 digits"):
        parse_cd_block_assignment(lines(), label="stream")


# --- apportionment -------------------------------------------------------


def test_apportionment_tables_cover_every_state_and_sum_to_435():
    fips = {state for state, _, _ in US_STATES}
    for census, seats in US_HOUSE_APPORTIONMENT.items():
        assert set(seats) == fips, census
        assert sum(seats.values()) == 435, census


def test_expected_district_counts_give_dc_and_at_large_states_one_district():
    counts = expected_district_counts("2020_census")
    assert counts["11"] == 1  # DC delegate
    assert counts["56"] == 1  # Wyoming at-large
    assert counts["30"] == 2  # Montana gained a seat in 2020
    assert expected_district_counts("2010_census")["30"] == 1
    assert sum(counts.values()) == 436
    with pytest.raises(ValueError, match="Unknown apportionment"):
        expected_district_counts("1990_census")


# --- assembly and loading -------------------------------------------------


def test_round_trip_preserves_every_plan_and_conserves_population(tmp_path):
    registry = load_us_cd_plan_registry(_write(tmp_path, _payload()))
    assert registry.plan_ids == ("119th_congress", "120th_congress")
    assert registry.block_geoid.tolist() == BLOCKS.tolist()
    assert registry.plans["119th_congress"].tolist() == [
        PLAN_119[b] for b in BLOCKS.tolist()
    ]
    assert registry.state_population() == {"10": 60, "44": 220}
    assert registry.district_population("119th_congress") == {
        "1000": 60,
        "4401": 90,
        "4402": 130,
    }
    assert registry.district_population("120th_congress") == {
        "1000": 60,
        "4401": 40,
        "4402": 180,
    }
    summary = summarize_cd_plan_registry(registry)
    assert summary["population"] == 280
    assert summary["plans"]["120th_congress"]["states_differing_from_primary"] == ["44"]
    assert summary["plans"]["119th_congress"]["states_differing_from_primary"] == []
    assert registry.plan_sources["120th_congress"]["districts"] == 3
    assert registry.plan_sources["120th_congress"]["blocks_outside_universe"] == 0


def test_plan_arrays_are_stored_as_int16_and_loaded_as_int64(tmp_path):
    payload = _payload()
    assert payload[f"{CD_PLAN_ARRAY_PREFIX}119th_congress"].dtype == np.int16
    registry = load_us_cd_plan_registry(_write(tmp_path, payload))
    assert registry.plans["119th_congress"].dtype == np.int64


def test_assembly_refuses_a_populated_block_without_a_district():
    gap = {block: district for block, district in PLAN_120.items() if block != RI[3]}
    with pytest.raises(ValueError, match="1 populated block"):
        _payload(
            plan_assignments={
                "119th_congress": cd_block_assignment(PLAN_119, label="119"),
                "120th_congress": cd_block_assignment(gap, label="120"),
            }
        )


def test_assembly_refuses_a_district_in_another_state():
    crossed = {**PLAN_120, DE[0]: 4401}
    with pytest.raises(ValueError, match="another state's district"):
        _payload(
            plan_assignments={
                "119th_congress": cd_block_assignment(PLAN_119, label="119"),
                "120th_congress": cd_block_assignment(crossed, label="120"),
            }
        )


def test_assembly_refuses_a_district_count_off_the_apportionment():
    merged = {**PLAN_120, RI[0]: 4402}  # Rhode Island down to one district
    with pytest.raises(ValueError, match="apportionment"):
        _payload(
            plan_assignments={
                "119th_congress": cd_block_assignment(PLAN_119, label="119"),
                "120th_congress": cd_block_assignment(merged, label="120"),
            }
        )


def test_assembly_refuses_incomplete_plan_metadata():
    for key in ("vintage", "source", "url", "apportionment", "source_files"):
        spec = _spec("120th_congress")
        del spec[key]
        with pytest.raises(ValueError, match="missing"):
            _payload(
                plan_sources={
                    "119th_congress": _spec("119th_congress"),
                    "120th_congress": spec,
                }
            )
    mismatched = {**_spec("120th_congress"), "vintage": "119th_congress"}
    with pytest.raises(ValueError, match="must match"):
        _payload(
            plan_sources={
                "119th_congress": _spec("119th_congress"),
                "120th_congress": mismatched,
            }
        )


def test_blocks_outside_the_universe_are_counted_and_ignored(tmp_path):
    unpopulated = 44_001_0301_00_2000
    puerto_rico = 72_001_9501_00_1000
    extended = {**PLAN_120, unpopulated: 4401, puerto_rico: 7200}
    registry = load_us_cd_plan_registry(
        _write(
            tmp_path,
            _payload(
                plan_assignments={
                    "119th_congress": cd_block_assignment(PLAN_119, label="119"),
                    "120th_congress": cd_block_assignment(extended, label="120"),
                }
            ),
        )
    )
    assert registry.plan_sources["120th_congress"]["blocks_outside_universe"] == 2
    assert unpopulated not in registry.block_geoid.tolist()


def test_loader_pins_the_sha256(tmp_path):
    path = _write(tmp_path, _payload())
    registry = load_us_cd_plan_registry(path)
    assert (
        load_us_cd_plan_registry(path, expected_sha256=registry.sha256).sha256
        == registry.sha256
    )
    with pytest.raises(ValueError, match="expected"):
        load_us_cd_plan_registry(path, expected_sha256="f" * 64)


def test_loader_refuses_an_artifact_edited_after_the_build(tmp_path):
    payload = _payload()
    crossed = payload[f"{CD_PLAN_ARRAY_PREFIX}120th_congress"].copy()
    crossed[0] = 4401  # a Delaware block moved into Rhode Island
    with pytest.raises(ValueError, match="another state's district"):
        load_us_cd_plan_registry(
            _write(
                tmp_path, {**payload, f"{CD_PLAN_ARRAY_PREFIX}120th_congress": crossed}
            )
        )

    metadata = json.loads(str(payload["metadata_json"]))
    metadata["plans"]["120th_congress"]["districts"] = 4
    with pytest.raises(ValueError, match="records 4 districts"):
        load_us_cd_plan_registry(
            _write(
                tmp_path,
                {**payload, "metadata_json": np.asarray(json.dumps(metadata))},
                "districts.npz",
            )
        )

    unsorted = {**payload, "block_geoid": payload["block_geoid"][::-1].copy()}
    with pytest.raises(ValueError, match="sorted and unique"):
        load_us_cd_plan_registry(_write(tmp_path, unsorted, "unsorted.npz"))

    extra = {**payload, "unexpected": np.zeros(1)}
    with pytest.raises(ValueError, match="unexpected arrays"):
        load_us_cd_plan_registry(_write(tmp_path, extra, "extra.npz"))

    missing_plan = dict(payload)
    del missing_plan[f"{CD_PLAN_ARRAY_PREFIX}120th_congress"]
    with pytest.raises(ValueError, match="do not match its arrays"):
        load_us_cd_plan_registry(_write(tmp_path, missing_plan, "missing.npz"))


@pytest.mark.parametrize(
    ("key", "change", "message"),
    [
        ("population", lambda a: a.reshape(-1, 1), "one-dimensional"),
        (
            "population",
            lambda a: np.concatenate([a, a]).reshape(2, -1),
            "one-dimensional",
        ),
        ("block_geoid", lambda a: a[::-1].reshape(-1, 1).copy(), "one-dimensional"),
        ("population", lambda a: np.append(a, 5), "not aligned"),
        (
            f"{CD_PLAN_ARRAY_PREFIX}120th_congress",
            lambda a: a.reshape(-1, 1),
            "one-dimensional",
        ),
        (f"{CD_PLAN_ARRAY_PREFIX}120th_congress", lambda a: a[:-1], "not aligned"),
    ],
)
def test_loader_refuses_arrays_that_are_not_aligned_and_one_dimensional(
    tmp_path, key, change, message
):
    payload = _payload()
    tampered = {**payload, key: change(payload[key])}
    with pytest.raises(ValueError, match=message):
        load_us_cd_plan_registry(_write(tmp_path, tampered))


def test_ladder_differential_refuses_misaligned_ladder_arrays(tmp_path):
    # Extra trailing values must not be silently dropped by the reordering.
    registry = load_us_cd_plan_registry(_write(tmp_path, _payload()))
    ladder_cd = np.asarray([PLAN_119[b] for b in BLOCKS.tolist()], dtype=np.int64)
    with pytest.raises(ValueError, match="not aligned"):
        check_cd_plan_registry_against_block_ladder(
            registry,
            block_geoid=BLOCKS,
            population=np.append(POPULATION, 999),
            congressional_district_geoid=np.append(ladder_cd, 9999),
        )
    with pytest.raises(ValueError, match="one-dimensional"):
        check_cd_plan_registry_against_block_ladder(
            registry,
            block_geoid=BLOCKS.reshape(-1, 1),
            population=POPULATION.reshape(-1, 1),
            congressional_district_geoid=ladder_cd.reshape(-1, 1),
        )


def test_ladder_differential_accepts_agreement_and_refuses_any_difference(tmp_path):
    registry = load_us_cd_plan_registry(_write(tmp_path, _payload()))
    order = np.asarray([6, 0, 3, 1, 5, 2, 4])  # ladder rows in any order
    ladder_cd = np.asarray([PLAN_119[b] for b in BLOCKS.tolist()], dtype=np.int64)
    receipt = check_cd_plan_registry_against_block_ladder(
        registry,
        block_geoid=BLOCKS[order],
        population=POPULATION[order],
        congressional_district_geoid=ladder_cd[order],
    )
    assert receipt == {
        "plan": PRIMARY_CD_PLAN,
        "blocks": 7,
        "population": 280,
        "agreement": "exact",
    }

    changed_cd = ladder_cd.copy()
    changed_cd[4] = 4402
    with pytest.raises(ValueError, match="119th_congress district of 1 block"):
        check_cd_plan_registry_against_block_ladder(
            registry,
            block_geoid=BLOCKS,
            population=POPULATION,
            congressional_district_geoid=changed_cd,
        )
    changed_population = POPULATION.copy()
    changed_population[0] += 1
    with pytest.raises(ValueError, match="population of 1 block"):
        check_cd_plan_registry_against_block_ladder(
            registry,
            block_geoid=BLOCKS,
            population=changed_population,
            congressional_district_geoid=ladder_cd,
        )
    with pytest.raises(ValueError, match="different blocks"):
        check_cd_plan_registry_against_block_ladder(
            registry,
            block_geoid=BLOCKS[1:],
            population=POPULATION[1:],
            congressional_district_geoid=ladder_cd[1:],
        )


def test_replace_state_assignments_swaps_exactly_one_state():
    base = cd_block_assignment(PLAN_120, label="base")
    donor = cd_block_assignment(PLAN_119, label="donor").for_state("44")
    merged = replace_state_assignments(base, donor, state_fips="44")
    assert (
        dict(
            zip(
                merged.block_geoid.tolist(), merged.district_geoid.tolist(), strict=True
            )
        )
        == PLAN_119
    )

    with pytest.raises(ValueError, match="other states"):
        replace_state_assignments(
            base, cd_block_assignment(PLAN_119, label="all"), state_fips="44"
        )
    partial = CdBlockAssignment(donor.block_geoid[:2], donor.district_geoid[:2])
    with pytest.raises(ValueError, match="leaves 2 block"):
        replace_state_assignments(base, partial, state_fips="44")


def test_cd_block_assignment_refuses_duplicates_and_bad_geoids():
    with pytest.raises(ValueError, match="more than once"):
        cd_block_assignment([(DE[0], 1000), (DE[0], 1000)], label="dup")
    with pytest.raises(ValueError, match="15-digit"):
        cd_block_assignment({12345: 1000}, label="short")


# --- builder ---------------------------------------------------------------


def _load_builder():
    path = _TEST_PATHS.repository / "tools" / "build_us_cd_plan_registry_artifact.py"
    spec = importlib.util.spec_from_file_location("build_us_cd_plan_registry", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_builder_pins_every_census_split_note_it_transcribes():
    builder = _load_builder()
    assert (
        set(builder.CENSUS_BEFS)
        == set(builder.CENSUS_BLOCK_SPLITS)
        == set(builder.CENSUS_BLOCK_SPLIT_NOTE_SHA256)
    )
    receipt = packaged_us_cd_plan_registry_provenance()
    for plan, digest in builder.CENSUS_BLOCK_SPLIT_NOTE_SHA256.items():
        note = receipt["plan_sources"][plan]["source_files"][f"{plan}_block_split_note"]
        assert note["sha256"] == digest
        assert (
            receipt["plan_sources"][plan]["block_splits"]
            == (builder.CENSUS_BLOCK_SPLITS[plan])
        )


def test_builder_refuses_a_split_note_that_is_not_the_reviewed_one(
    tmp_path, monkeypatch
):
    builder = _load_builder()
    error_page = tmp_path / "CD119_BlockSplits.pdf"
    error_page.write_text("<html>503 Service Unavailable</html>")
    monkeypatch.setattr(builder, "_download", lambda url, cache_dir: error_page)
    with pytest.raises(SystemExit, match="not the reviewed"):
        builder._plan_from_census_bef("119th_congress", tmp_path, {"10"})


# --- packaged provenance ---------------------------------------------------


def test_packaged_provenance_is_complete_and_matches_the_apportionment():
    receipt = packaged_us_cd_plan_registry_provenance()
    assert re.fullmatch(r"[0-9a-f]{64}", receipt["output_sha256"])
    assert receipt["kind"] == "us_cd_plan_registry"
    assert receipt["block_vintage"] == "2020_tabulation_blocks"
    assert receipt["primary_plan"] == PRIMARY_CD_PLAN
    # The block ladder's universe (US_PUMA_LADDER.md): every populated block
    # of the 50 states and DC, the 2020 apportionment population.
    assert receipt["blocks"] == 5_769_942
    assert receipt["population"] == 331_449_281
    assert receipt["states"] == 51
    assert set(receipt["plans"]) == {
        "117th_congress",
        "118th_congress",
        "119th_congress",
        "120th_congress",
    }
    for plan, summary in receipt["plans"].items():
        spec = receipt["plan_sources"][plan]
        assert spec["vintage"] == plan
        assert summary["districts"] == 436, plan
        assert summary["districts_by_state"] == expected_district_counts(
            spec["apportionment"]
        ), plan
        for entry in spec["source_files"].values():
            assert entry["url"].startswith(
                ("https://www2.census.gov/", "https://www.ncleg.gov/")
            )
            assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    plans = receipt["plans"]
    assert plans["119th_congress"]["states_differing_from_primary"] == []
    # Census's redrawn states for the 118th -> 119th and 119th -> 120th.
    assert plans["118th_congress"]["states_differing_from_primary"] == [
        "01",
        "13",
        "22",
        "36",
        "37",
    ]
    census_120th = receipt["plan_sources"]["120th_congress"]["census_state_files"]
    assert census_120th == ["01", "06", "12", "22", "29", "37", "39", "47", "48", "49"]
    # Missouri reverts to its 2022 map; every other Census redraw stands.
    assert plans["120th_congress"]["states_differing_from_primary"] == [
        state for state in census_120th if state != "29"
    ]
    missouri = receipt["plan_sources"]["120th_congress"]["known_deviations"]["29"]
    assert missouri["from_plan"] == "119th_congress"

    north_carolina = receipt["plan_sources"]["117th_congress"]["known_deviations"]["37"]
    method = north_carolina["method_check"]
    assert method["population_share_agreeing"] >= method["minimum_population_share"]
    official = north_carolina["official_district_population_check"]
    assert official["largest_share"] <= official["maximum_share"]
    assert sorted(official["differences"]) == [f"{d:02d}" for d in range(1, 14)]
    assert set(receipt["plan_sources"]["117th_congress"]["known_deviations"]) == {"37"}
    check = receipt["block_ladder_check"]
    assert check["agreement"] == "exact"
    assert check["plan"] == PRIMARY_CD_PLAN
    assert check["blocks"] == receipt["blocks"]
    assert check["population"] == receipt["population"]
    assert re.fullmatch(r"[0-9a-f]{64}", check["block_ladder_sha256"])


# --- a real artifact (opt-in) -----------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("MICROCOSM_US_CD_PLAN_REGISTRY"),
    reason="set MICROCOSM_US_CD_PLAN_REGISTRY to a built registry to check it",
)
def test_built_registry_matches_its_packaged_receipt():
    registry = load_pinned_us_cd_plan_registry(
        os.environ["MICROCOSM_US_CD_PLAN_REGISTRY"]
    )
    receipt = packaged_us_cd_plan_registry_provenance()
    summary = summarize_cd_plan_registry(registry)
    for key in ("blocks", "population", "states", "primary_plan", "plans"):
        assert summary[key] == receipt[key], key
    ladder_path = os.environ.get("MICROCOSM_US_BLOCK_LADDER")
    if ladder_path:
        with np.load(ladder_path, allow_pickle=False) as ladder:
            check_cd_plan_registry_against_block_ladder(
                registry,
                block_geoid=ladder["block_geoid"],
                population=ladder["population"],
                congressional_district_geoid=ladder["congressional_district_geoid"],
            )
