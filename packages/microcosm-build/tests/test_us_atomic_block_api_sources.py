"""Invented population-only API responses through real atomic normalization."""

import hashlib
import json
import sys
from dataclasses import replace
from io import BytesIO
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import pytest

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.us_runtime import atomic_block_api_sources as source
from microcosm.build.us_runtime.atomic_block_sources import AtomicBlockSourceBytes

CA_A, CA_B, CA_ZERO = "060010201001000", "060010201001001", "060010202001000"
DC = "110010001001000"
BLOCK_HEADER = ["P1_001N", "state", "county", "tract", "block"]


def _record(payload, members=()):
    return AtomicBlockSourceBytes(payload, hashlib.sha256(payload).hexdigest(), members)


def _json_record(rows):
    return _record(json.dumps(rows, separators=(",", ":")).encode())


def _block(population, geoid):
    return [str(population), geoid[:2], geoid[2:5], geoid[5:11], geoid[11:]]


def _arguments(*, ca_rows=None, ca_total=None, cd_rows=None, puma_rows=None):
    if ca_rows is None:
        ca_rows = [BLOCK_HEADER, _block(2, CA_A), _block(6, CA_B), _block(0, CA_ZERO)]
    if ca_total is None:
        ca_total = [["P1_001N", "state"], ["8", "06"]]
    if cd_rows is None:
        cd_rows = [f"{CA_A},01", f"{CA_B},02", f"{CA_ZERO},ZZ", f"{DC},98"]
    if puma_rows is None:
        puma_rows = [
            "06,001,020100,12345",
            "06,001,020200,99999",
            "11,001,000100,00100",
        ]
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
        archive.writestr(
            ZipInfo(source.sources.CD_MEMBER, date_time=(2020, 1, 1, 0, 0, 0)),
            "GEOID,CDFP\n" + "\n".join(cd_rows) + "\n",
        )
    return {
        "population_responses": (
            ("06", _json_record(ca_rows), _json_record(ca_total)),
            (
                "11",
                _json_record([BLOCK_HEADER, _block(5, DC)]),
                _json_record([["P1_001N", "state"], ["5", "11"]]),
            ),
        ),
        "cd_archive": _record(buffer.getvalue(), (source.sources.CD_MEMBER,)),
        "tract_to_puma": _record(
            (
                "STATEFP,COUNTYFP,TRACTCE,PUMA5CE\n" + "\n".join(puma_rows) + "\n"
            ).encode()
        ),
        "state_fips": ("06", "11"),
        "source_ids": {
            "population": "invented-original-2020-api-P1_001N",
            "district": "invented-original-cd119-bef",
            "puma": "invented-original-tract-puma2020",
        },
    }


def test_exact_api_sources_preserve_blocks_totals_and_nonsecret_request_receipts():
    arguments = _arguments()
    payload, receipt_bytes = source.assemble_atomic_block_api_sources(**arguments)
    support, receipt = decode_atomic_support(payload), json.loads(receipt_bytes)
    assert support.arrays["area"].tolist() == [CA_A, CA_B, DC]
    assert support.arrays["population"].tolist() == [2, 6, 5]
    assert support.arrays["puma"].tolist() == ["0612345", "0612345", "1100100"]
    assert support.arrays["district"].tolist() == ["0601", "0602", "1100"]
    assert support.metadata["columns"]["population"]["basis"] == "2020_census_persons"
    assert support.metadata["columns"]["district"]["relation"] == "official_tabulation"
    assert receipt["reconciliation"]["state_population"] == {"06": 8, "11": 5}
    assert receipt["reconciliation"]["output_state_population"] == {"06": 8, "11": 5}
    assert receipt["reconciliation"]["population_total"] == 13
    assert receipt["reconciliation"]["populated_blocks"] == 3
    assert receipt["support_sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["sources"]["population"][0]["block_rows"] == 3
    assert receipt["sources"]["population"][0]["zero_population_blocks"] == 1
    assert receipt["sources"]["district"]["delegate_records"] == 1
    for (state, blocks, total), provenance in zip(
        arguments["population_responses"], receipt["sources"]["population"], strict=True
    ):
        assert provenance["state_fips"] == state
        for name, record in (("blocks", blocks), ("state_total", total)):
            assert provenance[name]["sha256"] == record.sha256
            assert provenance[name]["size_bytes"] == len(record.payload)
            assert provenance[name]["request"]["endpoint"] == source.ENDPOINT
        assert provenance["blocks"]["request"]["parameters"] == [
            ["get", "P1_001N"],
            ["for", "block:*"],
            ["in", "state:" + state],
            ["in", "county:*"],
            ["in", "tract:*"],
        ]
        assert provenance["state_total"]["request"]["parameters"] == [
            ["get", "P1_001N"],
            ["for", "state:" + state],
        ]
    assert receipt["exact_input_hashes_verified"] is True
    for field in (
        "publisher_provenance_established",
        "request_origin_verified",
        "source_admission_issued",
        "release_eligible",
    ):
        assert receipt[field] is False
    assert b'"key"' not in receipt_bytes
    assert b"YOUR_KEY" not in receipt_bytes
    replay_arguments = {
        **arguments,
        "population_responses": iter(arguments["population_responses"]),
        "source_ids": dict(reversed(list(arguments["source_ids"].items()))),
    }
    assert source.assemble_atomic_block_api_sources(**replay_arguments) == (
        payload,
        receipt_bytes,
    )


def test_reordered_original_rows_have_identical_support_and_distinct_source_pin():
    first = source.assemble_atomic_block_api_sources(**_arguments())
    second = source.assemble_atomic_block_api_sources(
        **_arguments(
            ca_rows=[BLOCK_HEADER, _block(0, CA_ZERO), _block(6, CA_B), _block(2, CA_A)]
        )
    )
    assert first[0] == second[0]
    assert first[1] != second[1]


@pytest.mark.parametrize("which", ("blocks", "total", "cd_archive", "tract_to_puma"))
def test_exact_source_hash_refuses(which):
    arguments = _arguments()
    if which in {"blocks", "total"}:
        first, second = arguments["population_responses"]
        state, blocks, total = first
        if which == "blocks":
            blocks = replace(blocks, sha256="f" * 64)
        else:
            total = replace(total, sha256="f" * 64)
        arguments["population_responses"] = ((state, blocks, total), second)
    else:
        arguments[which] = replace(arguments[which], sha256="f" * 64)
    with pytest.raises(ValueError, match="SOURCE_DIGEST"):
        source.assemble_atomic_block_api_sources(**arguments)


@pytest.mark.parametrize(
    "defect",
    (
        "extra_column",
        "wrong_header",
        "numeric",
        "negative",
        "unicode",
        "null",
        "wrong_state",
        "short_code",
        "duplicate_zero",
        "duplicate_positive",
        "missing_positive",
    ),
)
def test_original_block_response_schema_and_complete_totals_refuse(defect):
    rows = [BLOCK_HEADER.copy(), _block(2, CA_A), _block(6, CA_B), _block(0, CA_ZERO)]
    if defect == "extra_column":
        rows[0].append("unexpected_measure")
        for row in rows[1:]:
            row.append("0")
    elif defect == "wrong_header":
        rows[0][0] = "P1_002N"
    elif defect == "numeric":
        rows[1][0] = 2
    elif defect == "negative":
        rows[-1][0] = "-1"
    elif defect == "unicode":
        rows[1][0] = "٢"
    elif defect == "null":
        rows[1][0] = None
    elif defect == "wrong_state":
        rows[1][1] = "11"
    elif defect == "short_code":
        rows[1][2] = "01"
    elif defect == "duplicate_zero":
        rows.append(rows[-1].copy())
    elif defect == "duplicate_positive":
        rows.append(rows[1].copy())
    else:
        rows.pop(1)
    with pytest.raises(ValueError, match="ATOMIC_BLOCK_API_SOURCES_"):
        source.assemble_atomic_block_api_sources(**_arguments(ca_rows=rows))


@pytest.mark.parametrize(
    "defect", ("wrong_total", "wrong_state", "extra_state", "wrong_header")
)
def test_independent_state_total_refuses(defect):
    rows = [["P1_001N", "state"], ["8", "06"]]
    if defect == "wrong_total":
        rows[1][0] = "9"
    elif defect == "wrong_state":
        rows[1][1] = "11"
    elif defect == "extra_state":
        rows.append(["5", "11"])
    else:
        rows[0].append("unexpected_measure")
    with pytest.raises(ValueError, match="ATOMIC_BLOCK_API_SOURCES_"):
        source.assemble_atomic_block_api_sources(**_arguments(ca_total=rows))


@pytest.mark.parametrize("defect", ("missing", "reordered", "extra_none", "territory"))
def test_exact_state_response_roster_refuses(defect):
    arguments = _arguments()
    records = arguments["population_responses"]
    if defect == "missing":
        arguments["population_responses"] = records[:1]
    elif defect == "reordered":
        arguments["population_responses"] = records[::-1]
    elif defect == "extra_none":
        arguments["population_responses"] = (*records, None)
    else:
        arguments["state_fips"] = ("06", "72")
    with pytest.raises(ValueError, match="(?:SOURCE|STATE)_ROSTER"):
        source.assemble_atomic_block_api_sources(**arguments)


@pytest.mark.parametrize("which", ("blocks", "total", "tract_to_puma"))
def test_original_source_record_mutation_after_check_uses_captured_bytes(which):
    arguments = _arguments()
    expected = source.assemble_atomic_block_api_sources(**arguments)
    if which == "blocks":
        record = arguments["population_responses"][0][1]
        changed = _json_record(
            [BLOCK_HEADER, _block(3, CA_A), _block(5, CA_B), _block(0, CA_ZERO)]
        )
    elif which == "total":
        record = arguments["population_responses"][0][2]
        changed = _json_record([["P1_001N", "state"], ["9", "06"]])
    else:
        record = arguments[which]
        changed = _record(record.payload.replace(b"12345", b"12346"))
    fired = []

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is source.sources._checked_source.__code__
            and frame.f_locals["record"] is record
            and not fired
        ):
            fired.append(True)
            object.__setattr__(record, "payload", changed.payload)
            object.__setattr__(record, "sha256", changed.sha256)
            object.__setattr__(record, "zip_members", ("changed.txt",))

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        actual = source.assemble_atomic_block_api_sources(**arguments)
    finally:
        sys.setprofile(previous)
    assert fired == [True]
    assert actual == expected


@pytest.mark.parametrize(
    "bounds,reason",
    (
        ({"max_source_bytes": 16}, "SOURCE_BYTES"),
        ({"max_total_geography_bytes": 100}, "GEOGRAPHY_BYTES"),
        ({"max_response_rows": 2}, "HEADER_OR_COUNT"),
        ({"max_response_rows": True}, "BOUNDS"),
    ),
)
def test_bounded_source_bytes_and_response_rows_refuse(bounds, reason):
    with pytest.raises(ValueError, match=reason):
        source.assemble_atomic_block_api_sources(**_arguments(), **bounds)


def test_malformed_json_and_support_size_are_refused_before_downstream_decode(
    monkeypatch,
):
    arguments = _arguments()
    first, second = arguments["population_responses"]
    arguments["population_responses"] = (
        (first[0], _record(b"not JSON"), first[2]),
        second,
    )
    with pytest.raises(ValueError, match="RESPONSE_JSON"):
        source.assemble_atomic_block_api_sources(**arguments)
    monkeypatch.setattr(source, "RAW_BYTES_MAX_BYTES", 16)
    decoded = []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is source.decode_atomic_support.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is source.assemble_atomic_block_api_sources.__code__
        ):
            decoded.append(True)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(ValueError, match="SUPPORT_BYTES"):
            source.assemble_atomic_block_api_sources(**_arguments())
    finally:
        sys.setprofile(previous)
    assert not decoded


@pytest.mark.parametrize("mapping", ("cd", "puma"))
def test_positive_block_missing_a_mapping_refuses(mapping):
    arguments = (
        _arguments(cd_rows=[f"{CA_A},ZZ", f"{CA_B},02", f"{DC},98"])
        if mapping == "cd"
        else _arguments(puma_rows=["11,001,000100,00100"])
    )
    with pytest.raises(ValueError, match="US atomic support"):
        source.assemble_atomic_block_api_sources(**arguments)
