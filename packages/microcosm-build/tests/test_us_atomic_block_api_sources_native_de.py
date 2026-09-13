"""Opt-in DE original-source control; the external guard provides every pin.

Ordinary CI skips before any source path access. The required fixture is a dict
with keys ``blocks``, ``total``, ``cd``, ``puma``, and ``output_dir``. Each source
entry is exactly ``{path: absolute_string, sha256: lowercase_hex, size_bytes: int}``.
The output directory must already be private and guard-owned. No native path or
native population value is embedded here; this test never creates a survey Frame.
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
from microcosm.build.us_runtime.atomic_block_sources import AtomicBlockSourceBytes
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

_FILES = {
    "blocks": ("state-10.blocks.P1_001N.json", 16 * 1024**2),
    "total": ("state-10.total.P1_001N.json", 64 * 1024),
    "cd": ("cd119.zip", 64 * 1024**2),
    "puma": ("2020_Census_Tract_to_2020_PUMA.txt", 4 * 1024**2),
}
_SOURCE_IDS = {
    "population": "census-2020-dec-pl-api-P1_001N",
    "district": "census-2025-cd119-NationalCD119.txt",
    "puma": "census-2020-Census-Tract-to-2020-PUMA",
}
# Exact published archive roster observed by the separate metadata-only guard.
# Only NationalCD119.txt is selected for decompression and CRC verification.
_CD_MEMBERS = (
    "01_AL_CD119.txt",
    "13_GA_CD119.txt",
    "22_LA_CD119.txt",
    "36_NY_CD119.txt",
    "37_NC_CD119.txt",
    "NationalCD119.txt",
)


def _require(condition, reason):
    # Fixed failure messages avoid dumping native response rows into pytest.
    assert condition, "NATIVE_DE_ATOMIC_" + reason


@pytest.fixture
def _native_case(request):
    try:
        return request.getfixturevalue("guarded_native_atomic_api_de_sources")
    except pytest.FixtureLookupError:
        pytest.skip("requires exact source pins from the external native DE guard")


def _snapshot(case):
    _require(type(case) is dict and set(case) == {*_FILES, "output_dir"}, "FIXTURE")
    records = []
    for name, (filename, maximum) in _FILES.items():
        record = case[name]
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
        records.append((name, path, digest, size))
    output_dir = case["output_dir"]
    _require(type(output_dir) is str and Path(output_dir).is_absolute(), "OUTPUT_DIR")
    return tuple(records), output_dir


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


def _write_private(directory, filename, payload):
    descriptor, partial = tempfile.mkstemp(prefix=".atomic-de-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, directory / filename, follow_symlinks=False)
    finally:
        os.unlink(partial)


def test_native_de_api_sources_reconcile_and_round_trip(_native_case):
    pins, output_path = _snapshot(_native_case)
    directory = Path(output_path)
    info = directory.lstat()
    _require(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and info.st_mode & 0o077 == 0,
        "PRIVATE_OUTPUT_DIR",
    )
    originals = {record[0]: _read_pinned(record) for record in pins}
    records = {
        name: AtomicBlockSourceBytes(
            originals[name], digest, _CD_MEMBERS if name == "cd" else ()
        )
        for name, _path, digest, _size in pins
    }
    support_bytes, receipt_bytes = source.assemble_atomic_block_api_sources(
        population_responses=(("10", records["blocks"], records["total"]),),
        cd_archive=records["cd"],
        tract_to_puma=records["puma"],
        state_fips=("10",),
        source_ids=_SOURCE_IDS,
        max_source_bytes=64 * 1024**2,
        max_member_bytes=512 * 1024**2,
        max_total_geography_bytes=768 * 1024**2,
        max_zip_members=6,
        max_line_bytes=64 * 1024,
        max_response_rows=250_000,
    )
    _require(0 < len(support_bytes) <= RAW_BYTES_MAX_BYTES, "SUPPORT_SIZE")
    _require(0 < len(receipt_bytes) <= 64 * 1024, "RECEIPT_SIZE")
    receipt, support = json.loads(receipt_bytes), decode_atomic_support(support_bytes)
    rows, total_rows = json.loads(originals["blocks"]), json.loads(originals["total"])
    expected_total = int(total_rows[1][0])
    positive, zero_count, seen = {}, 0, set()
    for population, state_code, county, tract, block in rows[1:]:
        geoid = state_code + county + tract + block
        _require(geoid not in seen, "DUPLICATE_ORIGINAL_BLOCK")
        seen.add(geoid)
        count = int(population)
        if count:
            positive[geoid] = count
        else:
            zero_count += 1
    ordered = sorted(positive)
    _require(
        support.arrays["area"].tolist() == ordered
        and support.arrays["population"].tolist() == [positive[key] for key in ordered]
        and set(support.arrays["state"].tolist()) == {"10"}
        and support.arrays["county"].tolist() == [key[:5] for key in ordered]
        and support.arrays["tract"].tolist() == [key[:11] for key in ordered]
        and sum(positive.values()) == expected_total,
        "COMPLETE_BLOCK_PRESERVATION",
    )
    population_receipt = receipt["sources"]["population"]
    _require(
        receipt["state_fips"] == ["10"]
        and receipt["source_ids"] == _SOURCE_IDS
        and len(population_receipt) == 1
        and population_receipt[0]["state_fips"] == "10"
        and population_receipt[0]["block_rows"] == len(seen)
        and population_receipt[0]["populated_blocks"] == len(positive)
        and population_receipt[0]["zero_population_blocks"] == zero_count
        and receipt["reconciliation"]["state_population"] == {"10": expected_total}
        and receipt["reconciliation"]["output_state_population"]
        == {"10": expected_total}
        and receipt["reconciliation"]["population_total"] == expected_total
        and receipt["reconciliation"]["populated_blocks"] == len(positive)
        and receipt["support_sha256"] == hashlib.sha256(support_bytes).hexdigest(),
        "RECONCILIATION",
    )
    source_receipts = {
        "blocks": population_receipt[0]["blocks"],
        "total": population_receipt[0]["state_total"],
        "cd": receipt["sources"]["district"],
        "puma": receipt["sources"]["puma"],
    }
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
    support_path = directory / "de-atomic-support.npz"
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
    # Final source I/O is followed only by immutable comparisons and private
    # report publication; it does not issue source or Population authority.
    for record in pins:
        _require(_read_pinned(record) == originals[record[0]], "FINAL_SOURCE_BYTES")
    _require(_snapshot(_native_case) == (pins, output_path), "FINAL_FIXTURE_PINS")
    report = canonical_json(
        {
            "protocol": "microcosm.us.native-de-atomic-api-control.v1",
            "status": "source_control_passed",
            "state_fips": ["10"],
            "source_receipt": receipt,
            "source_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            "support_filename": support_path.name,
            "support_sha256": hashlib.sha256(support_bytes).hexdigest(),
            "support_size_bytes": len(support_bytes),
            "exact_support_readback": True,
            "national_coverage": False,
            "survey_frame_created": False,
            "source_admission_issued": False,
            "release_eligible": False,
        }
    )
    _require(len(report) <= 64 * 1024, "REPORT_SIZE")
    _write_private(directory, "de-atomic-normalization.json", report)
