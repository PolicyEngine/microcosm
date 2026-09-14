"""Opt-in national original-source control; every native pin comes from a guard.

Ordinary CI skips before source path access. The external fixture supplies
``states: {SS: {blocks: pin, total: pin}}``, ``cd: pin``, ``puma: pin`` and
``output_dir``. Each pin is exactly ``{path, sha256, size_bytes}``. There are 104
sources, covering the 50 states plus DC. No native path or population value is
embedded. The independent population oracle retains only per-state digests;
the mapping check calls maintained parsers separately from the adapter result.
This is source normalization, without survey construction or source admission.
"""

import hashlib
import json
import os
import re
import stat
import tempfile
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.us_runtime import atomic_block_api_sources as source
from microcosm.build.us_runtime import atomic_block_sources as source_bytes
from microcosm.build.us_runtime.block_ladder_sources import (
    US_STATES,
    parse_national_cd_bef,
)
from microcosm.build.us_runtime.puma_ladder_sources import (
    parse_tract_to_puma_relationship,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

_STATES = tuple(
    "01 02 04 05 06 08 09 10 11 12 13 15 16 17 18 19 20 "
    "21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 "
    "38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56".split()
)
_SOURCE_IDS = {
    "population": "census-2020-dec-pl-api-P1_001N",
    "district": "census-2025-cd119-NationalCD119.txt",
    "puma": "census-2020-Census-Tract-to-2020-PUMA",
}
_CD_MEMBERS = (
    "01_AL_CD119.txt",
    "13_GA_CD119.txt",
    "22_LA_CD119.txt",
    "36_NY_CD119.txt",
    "37_NC_CD119.txt",
    "NationalCD119.txt",
)
_MAX_SOURCE_BYTES = 64 * 1024**2
_MAX_ACQUIRED_BYTES = 1024**3
# Explicit selected-member cap from the separately reviewed CD metadata probe.
# This is a bound, not a native payload observation by this test's author.
_MAX_CD_MEMBER_BYTES = 163_499_110
_MAX_SELECTED_BYTES = _MAX_ACQUIRED_BYTES + _MAX_CD_MEMBER_BYTES
_MAX_RESPONSE_ROWS = 2_000_000
_BOUNDS = {
    "max_source_bytes": _MAX_SOURCE_BYTES,
    "max_member_bytes": _MAX_CD_MEMBER_BYTES,
    "max_total_geography_bytes": _MAX_SELECTED_BYTES,
    "max_zip_members": 6,
    "max_line_bytes": 64 * 1024,
    "max_response_rows": _MAX_RESPONSE_ROWS,
}


def _require(condition, reason):
    assert condition, "NATIVE_NATIONAL_ATOMIC_" + reason


@pytest.fixture
def _native_case(request):
    try:
        return request.getfixturevalue("guarded_native_atomic_api_national_sources")
    except pytest.FixtureLookupError:
        pytest.skip("requires exact source pins from the external national guard")


def _pin(record, name, filename, maximum):
    _require(
        type(record) is dict and set(record) == {"path", "sha256", "size_bytes"},
        "SOURCE_PIN",
    )
    path, digest, size = record["path"], record["sha256"], record["size_bytes"]
    _require(
        type(path) is str
        and Path(path).is_absolute()
        and Path(path).name == filename
        and "\0" not in path
        and type(digest) is str
        and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        and type(size) is int
        and 0 < size <= maximum,
        "SOURCE_PIN",
    )
    return name, path, digest, size


def _snapshot(case):
    _require(
        type(case) is dict and set(case) == {"states", "cd", "puma", "output_dir"},
        "FIXTURE",
    )
    states = case["states"]
    _require(type(states) is dict and set(states) == set(_STATES), "STATE_ROSTER")
    pins = []
    for state_code in _STATES:
        pair = states[state_code]
        _require(type(pair) is dict and set(pair) == {"blocks", "total"}, "STATE_PAIR")
        for kind, maximum in (("blocks", _MAX_SOURCE_BYTES), ("total", 64 * 1024)):
            pins.append(
                _pin(
                    pair[kind],
                    state_code + "." + kind,
                    f"state-{state_code}.{kind}.P1_001N.json",
                    maximum,
                )
            )
    pins.extend(
        (
            _pin(case["cd"], "cd", "cd119.zip", _MAX_SOURCE_BYTES),
            _pin(
                case["puma"],
                "puma",
                "2020_Census_Tract_to_2020_PUMA.txt",
                4 * 1024**2,
            ),
        )
    )
    _require(
        len(pins) == 104
        and len({pin[1] for pin in pins}) == 104
        and sum(pin[3] for pin in pins) <= _MAX_ACQUIRED_BYTES,
        "SOURCE_ROSTER_OR_BYTES",
    )
    output_dir = case["output_dir"]
    _require(
        type(output_dir) is str
        and "\0" not in output_dir
        and Path(output_dir).is_absolute(),
        "OUTPUT_DIR",
    )
    return tuple(pins), output_dir


def _file_stamp(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_pinned(record):
    _name, path, digest, size = record
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        _require(stat.S_ISREG(before.st_mode) and before.st_size == size, "SOURCE_FILE")
        payload = stream.read(size + 1)
        after = os.fstat(stream.fileno())
    _require(
        len(payload) == size
        and hashlib.sha256(payload).hexdigest() == digest
        and _file_stamp(before) == _file_stamp(after),
        "SOURCE_BYTES",
    )
    return payload


def _source_record(pin):
    return source_bytes.AtomicBlockSourceBytes(
        _read_pinned(pin), pin[2], _CD_MEMBERS if pin[0] == "cd" else ()
    )


def _write_private(directory, filename, payload):
    descriptor, partial = tempfile.mkstemp(prefix=".atomic-national-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, directory / filename, follow_symlinks=False)
    finally:
        os.unlink(partial)


def _digest_row(digest, geoid, population):
    digest.update((geoid + "," + str(population) + "\n").encode("ascii"))


def _original_state_summary(block_payload, total_payload, state_code):
    rows, totals = json.loads(block_payload), json.loads(total_payload)
    _require(
        type(rows) is list
        and 1 < len(rows) <= _MAX_RESPONSE_ROWS + 1
        and rows[0] == ["P1_001N", "state", "county", "tract", "block"]
        and type(totals) is list
        and len(totals) == 2
        and totals[0] == ["P1_001N", "state"]
        and type(totals[1]) is list
        and len(totals[1]) == 2
        and totals[1][1] == state_code
        and type(totals[1][0]) is str
        and re.fullmatch(r"[0-9]{1,16}", totals[1][0]) is not None,
        "ORIGINAL_HEADER_OR_TOTAL",
    )
    expected_total = int(totals[1][0])
    _require(expected_total <= 2**53, "ORIGINAL_TOTAL_DOMAIN")
    positive, seen, zero_count = {}, set(), 0
    for row in rows[1:]:
        _require(
            type(row) is list
            and len(row) == 5
            and all(type(value) is str for value in row),
            "ORIGINAL_ROW",
        )
        population, state, county, tract, block = row
        _require(
            re.fullmatch(r"[0-9]{1,16}", population) is not None
            and state == state_code
            and re.fullmatch(r"[0-9]{3}", county) is not None
            and re.fullmatch(r"[0-9]{6}", tract) is not None
            and re.fullmatch(r"[0-9]{4}", block) is not None,
            "ORIGINAL_DOMAIN",
        )
        count, geoid = int(population), state + county + tract + block
        _require(count <= 2**53 and geoid not in seen, "ORIGINAL_DUPLICATE_OR_COUNT")
        seen.add(geoid)
        if count:
            positive[geoid] = count
        else:
            zero_count += 1
    _require(
        bool(positive) and sum(positive.values()) == expected_total, "ORIGINAL_SUM"
    )
    digest = hashlib.sha256()
    for geoid in sorted(positive):
        _digest_row(digest, geoid, positive[geoid])
    return {
        "state_population": expected_total,
        "block_rows": len(seen),
        "populated_blocks": len(positive),
        "zero_population_blocks": zero_count,
        "ordered_positive_block_population_sha256": digest.hexdigest(),
    }


def _population_responses(pins_by_name, summaries):
    for state_code in _STATES:
        blocks = _source_record(pins_by_name[state_code + ".blocks"])
        total = _source_record(pins_by_name[state_code + ".total"])
        summaries[state_code] = _original_state_summary(
            blocks.payload, total.payload, state_code
        )
        yield state_code, blocks, total
        del blocks, total


def _check_complete_population(support, summaries):
    _require(
        set(support.arrays)
        == {"area", "state", "county", "tract", "puma", "district", "population"},
        "SUPPORT_COLUMNS",
    )
    digests = {state_code: hashlib.sha256() for state_code in _STATES}
    counts, totals = dict.fromkeys(_STATES, 0), dict.fromkeys(_STATES, 0)
    previous = ""
    for geoid, state_code, county, tract, population in zip(
        *(
            support.arrays[name]
            for name in ("area", "state", "county", "tract", "population")
        ),
        strict=True,
    ):
        geoid, state_code = str(geoid), str(state_code)
        count = int(population)
        _require(
            re.fullmatch(r"[0-9]{15}", geoid) is not None
            and previous < geoid
            and state_code in digests
            and state_code == geoid[:2]
            and str(county) == geoid[:5]
            and str(tract) == geoid[:11]
            and count == population
            and 0 < count <= 2**53,
            "OUTPUT_BLOCK_IDENTITY",
        )
        previous = geoid
        _digest_row(digests[state_code], geoid, count)
        counts[state_code] += 1
        totals[state_code] += count
    for state_code in _STATES:
        expected = summaries[state_code]
        _require(
            counts[state_code] == expected["populated_blocks"]
            and totals[state_code] == expected["state_population"]
            and digests[state_code].hexdigest()
            == expected["ordered_positive_block_population_sha256"],
            "COMPLETE_POSITIVE_BLOCK_PRESERVATION",
        )


def _check_complete_mapping_joins(support, pins_by_name, receipt):
    # A separate maintained-parser pass checks every selected output join.
    # This is not an independent interpretation of Census mapping conventions.
    expanded = [0]
    cd_payload, cd_pin = source_bytes._selected_zip_member(
        _source_record(pins_by_name["cd"]),
        "NationalCD119.txt",
        bounds=_BOUNDS,
        expanded=expanded,
    )
    statistics, unassigned = {"source_records": 0, "delegate_records": 0}, set()
    districts = parse_national_cd_bef(
        source_bytes._cd_lines(
            cd_payload,
            max_line_bytes=_BOUNDS["max_line_bytes"],
            statistics=statistics,
            unassigned=unassigned,
        )
    )
    _require(
        not unassigned.intersection(districts)
        and statistics["source_records"] == len(districts) + len(unassigned),
        "CD_SOURCE_RECONCILIATION",
    )
    del cd_payload, unassigned
    puma_payload = _read_pinned(pins_by_name["puma"])
    _require(
        expanded[0] + len(puma_payload) <= _MAX_SELECTED_BYTES, "JOIN_SOURCE_BYTES"
    )
    pumas = parse_tract_to_puma_relationship(
        source_bytes._lines(
            puma_payload, encoding="utf-8", max_line_bytes=_BOUNDS["max_line_bytes"]
        ),
        allowed_state_fips=frozenset(_STATES),
    )
    del puma_payload
    for geoid, district, puma in zip(
        *(support.arrays[name] for name in ("area", "district", "puma")), strict=True
    ):
        block = int(geoid)
        _require(
            block in districts
            and block // 10000 in pumas
            and str(district) == f"{districts[block]:04d}"
            and str(puma) == f"{pumas[block // 10000]:07d}",
            "COMPLETE_MAPPING_JOIN",
        )
    _require(
        all(
            receipt["sources"]["district"][key] == value
            for key, value in cd_pin.items()
        )
        and receipt["sources"]["puma"]["selected_state_tract_mappings"] == len(pumas),
        "MAPPING_RECEIPT",
    )


def test_native_national_api_sources_reconcile_and_round_trip(_native_case):
    _require(tuple(row[0] for row in US_STATES) == _STATES, "MAINTAINED_STATE_ROSTER")
    pins, output_path = _snapshot(_native_case)
    pins_by_name = {pin[0]: pin for pin in pins}
    directory = Path(output_path)
    info = directory.lstat()
    _require(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and info.st_mode & 0o077 == 0,
        "PRIVATE_OUTPUT_DIR",
    )
    summaries = {}
    support_bytes, receipt_bytes = source.assemble_atomic_block_api_sources(
        population_responses=_population_responses(pins_by_name, summaries),
        cd_archive=_source_record(pins_by_name["cd"]),
        tract_to_puma=_source_record(pins_by_name["puma"]),
        state_fips=_STATES,
        source_ids=_SOURCE_IDS,
        **_BOUNDS,
    )
    _require(0 < len(support_bytes) <= RAW_BYTES_MAX_BYTES, "SUPPORT_SIZE")
    _require(0 < len(receipt_bytes) <= 256 * 1024, "RECEIPT_SIZE")
    receipt, support = json.loads(receipt_bytes), decode_atomic_support(support_bytes)
    _check_complete_population(support, summaries)
    _check_complete_mapping_joins(support, pins_by_name, receipt)
    state_totals = {
        state_code: summaries[state_code]["state_population"] for state_code in _STATES
    }
    population_receipts = receipt["sources"]["population"]
    _require(
        receipt["state_fips"] == list(_STATES)
        and receipt["source_ids"] == _SOURCE_IDS
        and receipt["limits"] == _BOUNDS
        and len(population_receipts) == 51
        and [row["state_fips"] for row in population_receipts] == list(_STATES)
        and receipt["reconciliation"]["state_population"] == state_totals
        and receipt["reconciliation"]["output_state_population"] == state_totals
        and receipt["reconciliation"]["population_total"] == sum(state_totals.values())
        and receipt["reconciliation"]["populated_blocks"]
        == sum(value["populated_blocks"] for value in summaries.values())
        and receipt["reconciliation"]["selected_geography_bytes"]
        == sum(pin[3] for pin in pins if pin[0] != "cd")
        + receipt["sources"]["district"]["selected_member_size_bytes"]
        <= _MAX_SELECTED_BYTES
        and receipt["support_sha256"] == hashlib.sha256(support_bytes).hexdigest(),
        "RECONCILIATION",
    )
    source_receipts = {
        "cd": receipt["sources"]["district"],
        "puma": receipt["sources"]["puma"],
    }
    for state_code, state_receipt in zip(_STATES, population_receipts, strict=True):
        for key in (
            "state_population",
            "block_rows",
            "populated_blocks",
            "zero_population_blocks",
        ):
            _require(state_receipt[key] == summaries[state_code][key], "STATE_RECEIPT")
        source_receipts[state_code + ".blocks"] = state_receipt["blocks"]
        source_receipts[state_code + ".total"] = state_receipt["state_total"]
    for name, _path, digest, size in pins:
        _require(
            source_receipts[name]["sha256"] == digest
            and source_receipts[name]["size_bytes"] == size,
            "RECEIPT_SOURCE_PIN",
        )
    _require(
        receipt["sources"]["district"]["zip_members"] == list(_CD_MEMBERS)
        and receipt["sources"]["district"]["crc_checked_members"]
        == ["NationalCD119.txt"]
        and receipt["sources"]["district"]["unselected_member_contents_read"] is False
        and receipt["sources"]["district"]["district_relation"] == "official_tabulation"
        and support.metadata["columns"]["district"]["relation"] == "official_tabulation"
        and support.metadata["columns"]["population"]["basis"] == "2020_census_persons"
        and receipt["exact_input_hashes_verified"] is True
        and all(
            receipt[name] is False
            for name in (
                "publisher_provenance_established",
                "request_origin_verified",
                "source_admission_issued",
                "release_eligible",
            )
        ),
        "AUTHORITY_BOUNDARY",
    )
    support_path = directory / "national-atomic-support.npz"
    _write_private(directory, support_path.name, support_bytes)
    replay_bytes = _read_pinned(
        (
            "support",
            str(support_path),
            hashlib.sha256(support_bytes).hexdigest(),
            len(support_bytes),
        )
    )
    _require(replay_bytes == support_bytes, "SUPPORT_READBACK")
    replay = decode_atomic_support(replay_bytes)
    _require(replay.metadata == support.metadata, "SUPPORT_METADATA_READBACK")
    _require(set(replay.arrays) == set(support.arrays), "SUPPORT_COLUMNS_READBACK")
    for name in support.arrays:
        _require(
            np.array_equal(replay.arrays[name], support.arrays[name]),
            "SUPPORT_ARRAY_READBACK",
        )
    del replay, replay_bytes
    # Final reads reauthenticate all104 pinned originals without retaining them.
    # The guard also checks these pins before and after the entire test process.
    for pin in pins:
        _read_pinned(pin)
    _require(_snapshot(_native_case) == (pins, output_path), "FINAL_FIXTURE_PINS")
    report = canonical_json(
        {
            "protocol": "microcosm.us.native-national-atomic-api-control.v1",
            "status": "source_control_passed",
            "state_fips": list(_STATES),
            "source_count": 104,
            "source_receipt": receipt,
            "source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "population_oracle": summaries,
            "mapping_oracle": "separate maintained-parser pass; complete selected joins",
            "source_read_limits": {
                "unique_acquired_bytes": _MAX_ACQUIRED_BYTES,
                "selected_cd_member_bytes_per_pass": _MAX_CD_MEMBER_BYTES,
                "selected_geography_bytes_per_adapter_pass": _MAX_SELECTED_BYTES,
                "mapping_verification_passes": 1,
            },
            "support_filename": support_path.name,
            "support_sha256": hashlib.sha256(support_bytes).hexdigest(),
            "support_size_bytes": len(support_bytes),
            "exact_support_readback": True,
            "national_coverage": True,
            "survey_frame_created": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    _require(len(report) <= 256 * 1024, "REPORT_SIZE")
    _write_private(directory, "national-atomic-normalization.json", report)
