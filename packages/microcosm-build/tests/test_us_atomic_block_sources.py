"""Invented source bytes through maintained parsers and atomic normalization."""

import hashlib
import json
import struct
import sys
from dataclasses import replace
from io import BytesIO
from zipfile import ZIP_STORED, ZipFile, ZipInfo
from zlib import crc32

import pytest

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.us_runtime import atomic_block_sources as source
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

CA_A = "060010201001000"
CA_B = "060010201001001"
CA_ZERO = "060010202001000"
DC = "110010001001000"
UNREAD = b"UNSELECTED_SEGMENT_CONTENT"


def _geo_row(level, geoid, population):
    fields = [""] * 97
    fields[2], fields[9], fields[90] = level, geoid, str(population)
    return "|".join(fields) + "\n"


def _zip(entries):
    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_STORED) as archive:
        for name, payload in entries:
            archive.writestr(ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0)), payload)
    return buffer.getvalue()


def _record(payload, members=()):
    return source.AtomicBlockSourceBytes(
        payload, hashlib.sha256(payload).hexdigest(), tuple(members)
    )


def _arguments(*, ca_rows=None, cd_rows=None, puma_rows=None):
    if ca_rows is None:
        ca_rows = [
            _geo_row("040", "06", 8),
            _geo_row("750", CA_A, 2),
            _geo_row("750", CA_B, 6),
            _geo_row("750", CA_ZERO, 0),
        ]
    ca_members = ("ca000012020.pl", "cageo2020.pl")
    ca = _record(
        _zip(((ca_members[0], UNREAD), (ca_members[1], "".join(ca_rows).encode()))),
        ca_members,
    )
    dc = _record(
        _zip(
            (
                (
                    "dcgeo2020.pl",
                    (_geo_row("040", "11", 5) + _geo_row("750", DC, 5)).encode(),
                ),
            )
        ),
        ("dcgeo2020.pl",),
    )
    if cd_rows is None:
        cd_rows = [f"{CA_A},01", f"{CA_B},02", f"{CA_ZERO},ZZ", f"{DC},98"]
    cd = _record(
        _zip(
            ((source.CD_MEMBER, ("GEOID,CDFP\n" + "\n".join(cd_rows) + "\n").encode()),)
        ),
        (source.CD_MEMBER,),
    )
    if puma_rows is None:
        puma_rows = [
            "06,001,020100,12345",
            "06,001,020200,99999",
            "11,001,000100,00100",
            "72,001,000100,00100",
        ]
    return {
        "pl_archives": (("06", ca), ("11", dc)),
        "cd_archive": cd,
        "tract_to_puma": _record(
            (
                "\ufeffSTATEFP,COUNTYFP,TRACTCE,PUMA5CE\n" + "\n".join(puma_rows) + "\n"
            ).encode()
        ),
        "state_fips": ("06", "11"),
        "source_ids": {
            name: "invented-" + name for name in ("district", "population", "puma")
        },
    }


def test_exact_sources_preserve_every_populated_block_and_reconcile_states():
    arguments = _arguments()
    payload, raw_receipt = source.assemble_atomic_block_sources(**arguments)
    support, receipt = decode_atomic_support(payload), json.loads(raw_receipt)
    assert support.arrays["area"].tolist() == [CA_A, CA_B, DC]
    assert support.arrays["population"].tolist() == [2, 6, 5]
    assert support.arrays["state"].tolist() == ["06", "06", "11"]
    assert support.arrays["puma"].tolist() == ["0612345", "0612345", "1100100"]
    assert support.arrays["district"].tolist() == ["0601", "0602", "1100"]
    assert support.metadata["columns"]["district"]["relation"] == "official_tabulation"
    assert receipt["reconciliation"]["state_population"] == {"06": 8, "11": 5}
    assert receipt["reconciliation"]["output_state_population"] == {"06": 8, "11": 5}
    assert receipt["reconciliation"]["population_total"] == 13
    assert receipt["reconciliation"]["populated_blocks"] == 3
    assert receipt["support_sha256"] == hashlib.sha256(payload).hexdigest()
    for (state, record), provenance in zip(
        arguments["pl_archives"], receipt["sources"]["population"], strict=True
    ):
        assert provenance["state_fips"] == state
        assert provenance["sha256"] == record.sha256
        assert provenance["size_bytes"] == len(record.payload)
        assert provenance["zip_members"] == sorted(record.zip_members)
        assert provenance["crc_checked_members"] == [provenance["selected_member"]]
        assert provenance["unselected_member_contents_read"] is False
    assert receipt["sources"]["population"][0]["zero_population_blocks"] == 1
    assert receipt["sources"]["population"][0]["block_rows"] == 3
    assert receipt["sources"]["district"]["unassigned_blocks"] == 1
    assert receipt["sources"]["district"]["delegate_records"] == 1
    assert receipt["sources"]["puma"]["selected_state_tract_mappings"] == 3
    assert receipt["exact_input_hashes_verified"] is True
    for field in (
        "unselected_zip_member_contents_read",
        "publisher_provenance_established",
        "source_admission_issued",
        "release_eligible",
    ):
        assert receipt[field] is False

    consumed = []

    def archives():
        for state, record in arguments["pl_archives"]:
            consumed.append(state)
            yield state, record

    replay = source.assemble_atomic_block_sources(
        **{
            **arguments,
            "pl_archives": archives(),
            "source_ids": dict(reversed(list(arguments["source_ids"].items()))),
        }
    )
    assert consumed == ["06", "11"]
    assert replay == (payload, raw_receipt)


def test_unselected_segment_content_is_not_decompressed_or_crc_checked():
    arguments = _arguments()
    state, record = arguments["pl_archives"][0]
    # Deliberately damage only the stored, unselected member's CRC. The whole
    # archive digest is repinned; selected geography is still exactly intact.
    damaged = record.payload.replace(UNREAD, b"X" + UNREAD[1:])
    assert damaged != record.payload
    with ZipFile(BytesIO(damaged)) as archive:
        # Check only metadata against the bytes this test constructed. The
        # unrelated member is never opened, even to demonstrate the bad CRC.
        assert archive.getinfo("ca000012020.pl").CRC != crc32(b"X" + UNREAD[1:])
    arguments["pl_archives"] = (
        (state, _record(damaged, record.zip_members)),
        arguments["pl_archives"][1],
    )
    payload, receipt = source.assemble_atomic_block_sources(**arguments)
    assert decode_atomic_support(payload).arrays["population"].tolist() == [2, 6, 5]
    assert json.loads(receipt)["unselected_zip_member_contents_read"] is False


@pytest.mark.parametrize("which", ("pl_archives", "cd_archive", "tract_to_puma"))
def test_any_wrong_exact_source_hash_refuses(which):
    arguments = _arguments()
    if which == "pl_archives":
        (state, record), other = arguments[which]
        arguments[which] = ((state, replace(record, sha256="f" * 64)), other)
    else:
        arguments[which] = replace(arguments[which], sha256="f" * 64)
    with pytest.raises(ValueError, match="SOURCE_DIGEST"):
        source.assemble_atomic_block_sources(**arguments)


@pytest.mark.parametrize("which", ("pl_archives", "cd_archive", "tract_to_puma"))
def test_checked_source_bytes_are_retained_after_record_mutation(which):
    arguments = _arguments()
    expected = source.assemble_atomic_block_sources(**arguments)
    if which == "pl_archives":
        record = arguments[which][0][1]
        changed = _arguments(
            ca_rows=[
                _geo_row("040", "06", 8),
                _geo_row("750", CA_A, 3),
                _geo_row("750", CA_B, 5),
                _geo_row("750", CA_ZERO, 0),
            ]
        )[which][0][1]
    elif which == "cd_archive":
        record = arguments[which]
        changed = _arguments(
            cd_rows=[f"{CA_A},03", f"{CA_B},02", f"{CA_ZERO},ZZ", f"{DC},98"]
        )[which]
    else:
        record = arguments[which]
        changed = _arguments(
            puma_rows=[
                "06,001,020100,12346",
                "06,001,020200,99999",
                "11,001,000100,00100",
                "72,001,000100,00100",
            ]
        )[which]
    assert changed.payload != record.payload
    fired = []

    def trace(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is source._checked_source.__code__
            and frame.f_locals["record"] is record
            and not fired
        ):
            fired.append(True)
            object.__setattr__(record, "payload", changed.payload)
            object.__setattr__(record, "sha256", changed.sha256)
            # A changed source container cannot retarget the captured ZIP roster.
            object.__setattr__(record, "zip_members", ("changed.txt",))

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        actual = source.assemble_atomic_block_sources(**arguments)
    finally:
        sys.setprofile(previous)
    assert fired == [True]
    assert actual == expected


def test_support_codec_cap_refuses_before_decoding(monkeypatch):
    assert source.RAW_BYTES_MAX_BYTES == RAW_BYTES_MAX_BYTES == 64 * 1024**2
    # Narrow a resource bound only; real parsers and normalization still execute.
    monkeypatch.setattr(source, "RAW_BYTES_MAX_BYTES", 16)
    decoded = []

    def trace(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is source.decode_atomic_support.__code__
            and frame.f_back is not None
            and frame.f_back.f_code is source.assemble_atomic_block_sources.__code__
        ):
            decoded.append(True)

    previous = sys.getprofile()
    sys.setprofile(trace)
    try:
        with pytest.raises(ValueError, match="ATOMIC_BLOCK_SOURCES_SUPPORT_BYTES"):
            source.assemble_atomic_block_sources(**_arguments())
    finally:
        sys.setprofile(previous)
    assert not decoded


@pytest.mark.parametrize(
    "defect", ("missing", "reordered", "extra", "extra_none", "states")
)
def test_exact_ordered_pl_state_roster_refuses(defect):
    arguments = _arguments()
    records = arguments["pl_archives"]
    if defect == "missing":
        arguments["pl_archives"] = records[:1]
    elif defect == "reordered":
        arguments["pl_archives"] = records[::-1]
    elif defect == "extra":
        arguments["pl_archives"] = (*records, records[0])
    elif defect == "extra_none":
        arguments["pl_archives"] = (*records, None, records[0])
    else:
        arguments["state_fips"] = ("06", "72")
    with pytest.raises(ValueError, match="(?:PL_SOURCE|STATE)_ROSTER"):
        source.assemble_atomic_block_sources(**arguments)


@pytest.mark.parametrize(
    "defect",
    (
        "extra_member",
        "missing_target",
        "duplicate",
        "unsafe_name",
        "nul_name",
        "crc",
        "declared_size",
    ),
)
def test_zip_member_identity_crc_and_declared_size_refuse(defect):
    arguments = _arguments()
    record = arguments["cd_archive"]
    if defect == "extra_member":
        record = replace(record, zip_members=(*record.zip_members, "unexpected.txt"))
    elif defect == "missing_target":
        record = replace(record, zip_members=("different.txt",))
    elif defect == "duplicate":
        with pytest.warns(UserWarning, match="Duplicate name"):
            payload = _zip(((source.CD_MEMBER, b"one"), (source.CD_MEMBER, b"two")))
        record = _record(payload, record.zip_members)
    elif defect == "unsafe_name":
        record = replace(record, zip_members=("../" + source.CD_MEMBER,))
    elif defect == "nul_name":
        name = source.CD_MEMBER + "Xsuffix"
        payload = _zip(((name, b"GEOID,CDFP\n"),))
        payload = payload.replace(
            name.encode(), (source.CD_MEMBER + "\0suffix").encode()
        )
        record = _record(payload, record.zip_members)
    elif defect == "crc":
        payload = record.payload.replace(CA_A.encode(), CA_B.encode(), 1)
        record = _record(payload, record.zip_members)
    else:
        payload = bytearray(record.payload)
        central = payload.index(b"PK\x01\x02")
        struct.pack_into("<I", payload, central + 24, 4096)
        record = _record(bytes(payload), record.zip_members)
        arguments["max_member_bytes"] = 2048
    arguments["cd_archive"] = record
    with pytest.raises(ValueError, match="(?:MEMBER|ZIP_)"):
        source.assemble_atomic_block_sources(**arguments)


@pytest.mark.parametrize(
    "bounds,reason",
    (
        ({"max_source_bytes": 16}, "SOURCE_BYTES"),
        ({"max_member_bytes": 32}, "(?:GEOGRAPHY_BYTES|MEMBER_SIZE)"),
        ({"max_total_geography_bytes": 200}, "TOTAL_GEOGRAPHY_BYTES"),
        ({"max_zip_members": 1}, "SELECTED_MEMBER"),
        ({"max_line_bytes": 16}, "TEXT_LINE"),
        ({"max_member_bytes": True}, "BOUNDS"),
    ),
)
def test_bounds_apply_before_unbounded_parsing(bounds, reason):
    with pytest.raises(ValueError, match=reason):
        source.assemble_atomic_block_sources(**_arguments(), **bounds)


@pytest.mark.parametrize(
    "defect",
    ("negative", "duplicate_zero", "duplicate_state", "short_row", "wrong_total"),
)
def test_pl_silent_skip_cases_and_state_totals_refuse(defect):
    rows = [
        _geo_row("040", "06", 9 if defect == "wrong_total" else 8),
        _geo_row("750", CA_A, 2),
        _geo_row("750", CA_B, 6),
        _geo_row("750", CA_ZERO, -1 if defect == "negative" else 0),
    ]
    if defect == "duplicate_zero":
        rows.append(_geo_row("750", CA_ZERO, 0))
    elif defect == "duplicate_state":
        rows.append(_geo_row("040", "06", 8))
    elif defect == "short_row":
        rows.append("|".join([""] * 96) + "\n")
    with pytest.raises(
        ValueError, match="(?:ATOMIC_BLOCK_SOURCES_PL_|state row records)"
    ):
        source.assemble_atomic_block_sources(**_arguments(ca_rows=rows))


@pytest.mark.parametrize(
    "defect", ("missing_cd", "missing_puma", "conflicting_cd", "conflicting_puma")
)
def test_populated_blocks_cannot_disappear_at_mapping_joins(defect):
    if defect == "missing_cd":
        arguments = _arguments(cd_rows=[f"{CA_A},ZZ", f"{CA_B},02", f"{DC},98"])
    elif defect == "missing_puma":
        arguments = _arguments(puma_rows=["11,001,000100,00100"])
    elif defect == "conflicting_cd":
        arguments = _arguments(
            cd_rows=[f"{CA_A},01", f"{CA_A},ZZ", f"{CA_B},02", f"{DC},98"]
        )
    else:
        arguments = _arguments(
            puma_rows=[
                "06,001,020100,12345",
                "06,001,020100,12346",
                "11,001,000100,00100",
            ]
        )
    with pytest.raises(
        ValueError, match="(?:US atomic support|CD_CONFLICTING_UNASSIGNED|both PUMA)"
    ):
        source.assemble_atomic_block_sources(**arguments)
