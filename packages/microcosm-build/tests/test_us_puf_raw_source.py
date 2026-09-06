"""The PUF raw status slice, exercised entirely on invented bytes.

Nothing here reads a PUF, a capture, a receipt or any genuine microdata. Every
fixture file is written under ``tmp_path`` from literals in this module, and
every fixture definition travels the explicit ``test_fixture`` route, which
refuses the invented-fixture authority's absence and refuses to restate a
packaged source pin. The packaged definition is read only for its own closed
metadata (digests, header lists, domains), never to admit a fixture.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import socket
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import puf_raw_source as puf
from microcosm.build.us_runtime.puf_raw_source import (
    BODY_MAX_BYTES,
    PUF_AGGREGATE_RECIDS,
    PUF_RAW_SOURCE_MAX_BYTES,
    PUF_RAW_SOURCE_NODE,
    RETURN_STATUS_SUMMARY_NODE,
    RETURN_STATUS_TYPE,
    PufRawSourceRefusal,
    USPufRawReturnStatusSummaryKernel,
    USPufRawSourceKernel,
    decode_puf_raw_source,
    decode_return_status,
    encode_return_status,
    fixture_definition,
    load_pinned_puf_bytes,
    packaged_definition,
    puf_raw_source_codecs,
    register_us_puf_raw_source_kernels,
    us_puf_raw_return_status_summary_node,
    us_puf_raw_source_node,
)
from microcosm.graph import (
    ArtifactOutput,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import (
    RAW_BYTES_MAX_BYTES,
    load_csv_tables,
    load_raw_bytes,
)
from microcosm.graph.errors import NodeRejectedError
from microcosm.graph.store import StoreUnavailable

PACKAGED = packaged_definition()
MAIN_HEADER = list(PACKAGED.main.delivered_header)
DEMOGRAPHIC_HEADER = list(PACKAGED.demographic.delivered_header)
HOST_POPULATION = "fixture_host"


# --------------------------------------------------------------------------
# Invented fixture construction
# --------------------------------------------------------------------------


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def header_digest(header) -> str:
    return hashlib.sha256(canonical(list(header))).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).hexdigest()


def csv_bytes(header, records) -> bytes:
    lines = [",".join(header)]
    lines.extend(",".join(record) for record in records)
    return ("\n".join(lines) + "\n").encode("utf-8")


def main_record(
    *, recid, mars="1", dsi="0", s006="12345", flpdyr="2015", flpdmo="12", filler="0"
):
    values = {
        "RECID": recid,
        "MARS": mars,
        "DSI": dsi,
        "S006": s006,
        "FLPDYR": flpdyr,
        "FLPDMO": flpdmo,
    }
    return [values.get(name, filler) for name in MAIN_HEADER]


def demographic_record(
    *,
    recid,
    agedp1="0",
    agedp2="0",
    agedp3="0",
    agerange="2",
    earnsplit="0",
    gender="1",
):
    values = {
        "RECID": recid,
        "AGEDP1": agedp1,
        "AGEDP2": agedp2,
        "AGEDP3": agedp3,
        "AGERANGE": agerange,
        "EARNSPLIT": earnsplit,
        "GENDER": gender,
    }
    return [values[name] for name in DEMOGRAPHIC_HEADER]


DEFAULT_MAIN = [
    main_record(recid="11", mars="1", dsi="0", s006="100"),
    main_record(recid="12", mars="2", dsi="0", s006="250"),
    main_record(recid="13", mars="4", dsi="1", s006="7"),
    main_record(recid=str(PUF_AGGREGATE_RECIDS[0]), mars="0", dsi="0", s006="0"),
]
DEFAULT_DEMOGRAPHIC = [
    demographic_record(recid="11", agerange="4", gender="2"),
    demographic_record(recid="13", agerange="3", earnsplit="1"),
]


def fixture_document(main_payload: bytes, demographic_payload: bytes) -> dict:
    """A fixture definition over invented bytes, on the explicit test route."""

    document = json.loads(json.dumps(dict(PACKAGED.document)))
    document["route"] = "test_fixture"
    document["authority"] = "invented_fixture_nonauthority"
    for name, payload, header, records in (
        ("main", main_payload, MAIN_HEADER, len(DEFAULT_MAIN)),
        (
            "demographic",
            demographic_payload,
            DEMOGRAPHIC_HEADER,
            len(DEFAULT_DEMOGRAPHIC),
        ),
    ):
        entry = document["sources"][name]
        entry["bytes"] = len(payload)
        entry["sha256"] = hashlib.sha256(payload).hexdigest()
        entry["git_blob_sha1"] = git_blob_sha1(payload)
        entry["delivered_header"] = list(header)
        entry["delivered_header_width"] = len(header)
        entry["header_record_canonical_sha256"] = header_digest(header)
        entry["data_records"] = records
    return document


@pytest.fixture
def invented():
    main = csv_bytes(MAIN_HEADER, DEFAULT_MAIN)
    demographic = csv_bytes(DEMOGRAPHIC_HEADER, DEFAULT_DEMOGRAPHIC)
    definition = fixture_definition(fixture_document(main, demographic))
    return main, demographic, definition


def rebuilt(main_records, demographic_records):
    main = csv_bytes(MAIN_HEADER, main_records)
    demographic = csv_bytes(DEMOGRAPHIC_HEADER, demographic_records)
    document = fixture_document(main, demographic)
    document["sources"]["main"]["data_records"] = len(main_records)
    document["sources"]["demographic"]["data_records"] = len(demographic_records)
    return main, demographic, fixture_definition(document)


# --------------------------------------------------------------------------
# The packaged definition and its non-production fixture route
# --------------------------------------------------------------------------


def test_packaged_definition_is_closed_and_reproduces_its_pinned_identities():
    assert PACKAGED.route == "packaged"
    assert len(MAIN_HEADER) == 222 and len(DEMOGRAPHIC_HEADER) == 7
    assert PACKAGED.main.data_records == 207696
    assert PACKAGED.demographic.data_records == 119675
    assert PACKAGED.main.bytes == 126034649
    assert header_digest(MAIN_HEADER) == PACKAGED.main.header_record_canonical_sha256
    assert (
        header_digest(DEMOGRAPHIC_HEADER)
        == PACKAGED.demographic.header_record_canonical_sha256
    )
    document = PACKAGED.document
    assert tuple(document["aggregates"]["recids"]) == PUF_AGGREGATE_RECIDS
    assert document["artifact"]["untyped_main_columns"] == 216
    assert document["artifact"]["period_semantics"] == "unresolved_raw_FLPDYR"
    assert document["artifact"]["weight_units"] == "S006_hundredths"
    assert document["artifact"]["amounts_decoded"] is False
    domains = {field["name"]: field.get("domain") for field in document["fields"]}
    assert domains["GENDER"] == [1, 2]
    assert domains["EARNSPLIT"] == [0, 1, 2, 3]
    assert domains["AGEDP1"] == domains["AGEDP2"] == domains["AGEDP3"] == list(range(7))
    agerange = next(f for f in document["fields"] if f["name"] == "AGERANGE")
    assert agerange["conditional_domain"]["cases"]["0"] == list(range(7))
    assert agerange["conditional_domain"]["cases"]["1"] == [0, 1, 2, 3]
    booklet = document["documentary_sources"]["demographic_booklet"]
    assert booklet["pdf_sha256"].startswith("7bb623c7")
    assert booklet["text_sha256"] == (
        "d2074c3a2e2dabb6d25da4a9a62b8d9302e001cc5d86939edc0bdd09719d8c9a"
    )


def test_packaged_definition_is_importable_from_the_installed_package():
    from importlib import resources

    raw = (
        resources.files("microcosm.build.us")
        .joinpath("puf_2015_raw_source_definition.json")
        .read_bytes()
    )
    assert json.loads(raw.decode("ascii"))["route"] == "packaged"


def test_fixture_route_refuses_production_authority_or_a_packaged_pin(invented):
    main, demographic, _ = invented
    document = fixture_document(main, demographic)
    document["authority"] = PACKAGED.document["authority"]
    with pytest.raises(PufRawSourceRefusal, match="FIXTURE_AUTHORITY"):
        fixture_definition(document)

    document = fixture_document(main, demographic)
    document["sources"]["main"]["sha256"] = PACKAGED.main.sha256
    with pytest.raises(PufRawSourceRefusal, match="FIXTURE_RESTATES_PACKAGED_PIN"):
        fixture_definition(document)


def test_a_fixture_definition_never_reaches_the_packaged_kernel_route(invented):
    _, _, definition = invented
    assert definition.route == "test_fixture"
    assert definition.sha256 != PACKAGED.sha256
    # The packaged route is bound to the packaged bytes, not to a caller's.
    kernel = USPufRawSourceKernel()
    assert kernel.definition.sha256 == PACKAGED.sha256


def test_definition_route_and_header_digest_are_checked(invented):
    main, demographic, _ = invented
    document = fixture_document(main, demographic)
    document["sources"]["main"]["header_record_canonical_sha256"] = "0" * 64
    with pytest.raises(PufRawSourceRefusal, match="DEFINITION_HEADER_DIGEST"):
        fixture_definition(document)

    document = fixture_document(main, demographic)
    document["route"] = "packaged"
    with pytest.raises(PufRawSourceRefusal, match="DEFINITION_ROUTE"):
        fixture_definition(document)


# --------------------------------------------------------------------------
# The pinned raw-byte codec
# --------------------------------------------------------------------------


def write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def test_codec_returns_the_exact_bytes_of_a_matching_regular_file(invented, tmp_path):
    main, _, definition = invented
    path = write(tmp_path / "main.csv", main)
    assert load_pinned_puf_bytes(path, pin=definition.main) == path.read_bytes()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    (
        ("bytes", 3, "CODEC_SIZE"),
        ("sha256", "0" * 64, "CODEC_SHA256"),
        ("git_blob_sha1", "0" * 40, "CODEC_GIT_BLOB_SHA1"),
    ),
)
def test_codec_refuses_a_wrong_length_or_wrong_digest(
    invented, tmp_path, field, value, reason
):
    main, demographic, _ = invented
    document = fixture_document(main, demographic)
    document["sources"]["main"][field] = value
    definition = fixture_definition(document)
    path = write(tmp_path / "main.csv", main)
    with pytest.raises(PufRawSourceRefusal, match=reason):
        load_pinned_puf_bytes(path, pin=definition.main)


def test_codec_refuses_a_mutated_file_of_the_same_length(invented, tmp_path):
    main, _, definition = invented
    mutated = bytearray(main)
    mutated[-2] ^= 0x01
    path = write(tmp_path / "main.csv", bytes(mutated))
    assert path.stat().st_size == definition.main.bytes
    with pytest.raises(PufRawSourceRefusal, match="CODEC_SHA256"):
        load_pinned_puf_bytes(path, pin=definition.main)


def test_codec_refuses_a_directory_a_fifo_a_socket_and_a_symlink(invented, tmp_path):
    main, _, definition = invented
    (tmp_path / "directory").mkdir()
    with pytest.raises(PufRawSourceRefusal, match="CODEC_(OPEN|NOT_REGULAR_FILE)"):
        load_pinned_puf_bytes(tmp_path / "directory", pin=definition.main)

    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(PufRawSourceRefusal, match="CODEC_NOT_REGULAR_FILE"):
        load_pinned_puf_bytes(fifo, pin=definition.main)

    # AF_UNIX paths are short; bind from the directory rather than an absolute
    # temporary path, which macOS refuses at 104 characters.
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind("socket")
            with pytest.raises(
                PufRawSourceRefusal, match="CODEC_(OPEN|NOT_REGULAR_FILE)"
            ):
                load_pinned_puf_bytes(tmp_path / "socket", pin=definition.main)
    finally:
        os.chdir(cwd)

    real = write(tmp_path / "main.csv", main)
    link = tmp_path / "link.csv"
    link.symlink_to(real)
    # O_NOFOLLOW protects the path this function receives. The executor resolves
    # source paths before codec lookup, so under the executor this refusal does
    # not apply to a user's original symlink; see the executor test below.
    with pytest.raises(PufRawSourceRefusal, match="CODEC_OPEN"):
        load_pinned_puf_bytes(link, pin=definition.main)


def test_codec_refuses_a_pin_above_its_private_cap(invented, tmp_path):
    main, demographic, _ = invented
    document = fixture_document(main, demographic)
    document["sources"]["main"]["bytes"] = PUF_RAW_SOURCE_MAX_BYTES + 1
    with pytest.raises(PufRawSourceRefusal, match="DEFINITION_SOURCE_OVER_CAP"):
        fixture_definition(document)


def test_the_shared_raw_bytes_cap_is_untouched(tmp_path):
    assert RAW_BYTES_MAX_BYTES == 64 * 1024 * 1024
    assert PUF_RAW_SOURCE_MAX_BYTES == 128 * 1024 * 1024
    oversized = write(tmp_path / "big.bin", b"\0" * 16)
    assert load_raw_bytes(oversized) == b"\0" * 16
    with pytest.raises(ValueError):
        load_raw_bytes(tmp_path)


# --------------------------------------------------------------------------
# The structural decoder
# --------------------------------------------------------------------------


def test_decoded_slice_types_only_the_declared_fields(invented):
    main, demographic, definition = invented
    status = decode_puf_raw_source(main, demographic, definition)
    assert status.rows == 4
    assert status.typed["RECID"].tolist() == [11, 12, 13, PUF_AGGREGATE_RECIDS[0]]
    assert status.typed["S006"].tolist() == [100, 250, 7, 0]
    assert status.typed["S006"].dtype == np.dtype("<i8")
    assert status.typed["demographic_status"].tolist() == [1, 0, 1, 0]
    assert status.typed["disclosure_aggregate"].tolist() == [0, 0, 0, 1]
    assert status.typed["AGERANGE"].tolist() == [4, -1, 3, -1]
    assert status.typed["GENDER"].tolist() == [2, -1, 1, -1]
    assert status.lexical["RECID"] == ("11", "12", "13", "999996")
    assert status.lexical["GENDER"] == ("2", "", "1", "")
    assert status.facts["matched_keys"] == 2
    assert status.facts["main_unmatched_records"] == 2
    assert status.facts["aggregate_records"] == 1
    assert status.facts["untyped_main_columns"] == 216
    assert status.facts["amounts_decoded"] is False
    assert status.facts["period_semantics"] == "unresolved_raw_FLPDYR"


def test_absent_zero_and_invalid_are_three_distinct_outcomes(invented):
    main, demographic, definition = invented
    status = decode_puf_raw_source(main, demographic, definition)
    # Absent: sentinel and empty lexical spelling, never a zero code.
    assert status.typed["AGEDP1"][1] == -1 and status.lexical["AGEDP1"][1] == ""
    # Documented zero inside a supplied row: a real code.
    assert status.typed["AGEDP1"][0] == 0 and status.lexical["AGEDP1"][0] == "0"
    # A non-decimal token is a refusal, not a zero.
    main_bad, demographic_bad, definition_bad = rebuilt(
        DEFAULT_MAIN,
        [demographic_record(recid="11", agedp1="x"), DEFAULT_DEMOGRAPHIC[1]],
    )
    with pytest.raises(PufRawSourceRefusal, match="TOKEN_GRAMMAR"):
        decode_puf_raw_source(main_bad, demographic_bad, definition_bad)


def test_a_shuffled_missing_extra_or_duplicate_header_is_refused(invented):
    main, demographic, definition = invented
    for header in (
        MAIN_HEADER[1:] + MAIN_HEADER[:1],
        MAIN_HEADER[:-1],
        MAIN_HEADER + ["EXTRA"],
        [MAIN_HEADER[0]] + MAIN_HEADER[1:-1] + [MAIN_HEADER[0]],
    ):
        payload = csv_bytes(header, [])
        with pytest.raises(PufRawSourceRefusal, match="HEADER_MISMATCH"):
            decode_puf_raw_source(payload, demographic, definition)
    with_bom = b"\xef\xbb\xbf" + main
    with pytest.raises(PufRawSourceRefusal, match="HEADER_MISMATCH"):
        decode_puf_raw_source(with_bom, demographic, definition)


def test_width_covers_every_delivered_field_while_content_of_others_is_ignored(
    invented,
):
    main, demographic, definition = invented
    narrow = [record[:-1] for record in DEFAULT_MAIN]
    payload = csv_bytes(MAIN_HEADER, narrow)
    with pytest.raises(PufRawSourceRefusal, match="RECORD_WIDTH"):
        decode_puf_raw_source(payload, demographic, definition)

    noisy = [list(record) for record in DEFAULT_MAIN]
    agir1 = MAIN_HEADER.index("AGIR1")
    for record in noisy:
        record[agir1] = "not a number at all"
    main_noisy, demographic_noisy, definition_noisy = rebuilt(
        noisy, DEFAULT_DEMOGRAPHIC
    )
    status = decode_puf_raw_source(main_noisy, demographic_noisy, definition_noisy)
    assert status.rows == 4


def test_duplicate_and_orphan_keys_are_refused(invented):
    main, demographic, definition = rebuilt(
        DEFAULT_MAIN + [main_record(recid="11")], DEFAULT_DEMOGRAPHIC
    )
    with pytest.raises(PufRawSourceRefusal, match="MAIN_DUPLICATE_LEXICAL_KEY"):
        decode_puf_raw_source(main, demographic, definition)

    main, demographic, definition = rebuilt(
        DEFAULT_MAIN, DEFAULT_DEMOGRAPHIC + [demographic_record(recid="11")]
    )
    with pytest.raises(PufRawSourceRefusal, match="DEMOGRAPHIC_DUPLICATE_LEXICAL_KEY"):
        decode_puf_raw_source(main, demographic, definition)

    main, demographic, definition = rebuilt(
        DEFAULT_MAIN, DEFAULT_DEMOGRAPHIC + [demographic_record(recid="77")]
    )
    with pytest.raises(PufRawSourceRefusal, match="DEMOGRAPHIC_ORPHAN_KEYS"):
        decode_puf_raw_source(main, demographic, definition)


def test_a_leading_zero_demographic_key_is_a_lexical_orphan(invented):
    main, demographic, definition = rebuilt(
        DEFAULT_MAIN,
        [demographic_record(recid="011"), DEFAULT_DEMOGRAPHIC[1]],
    )
    with pytest.raises(PufRawSourceRefusal, match="DEMOGRAPHIC_ORPHAN_KEYS"):
        decode_puf_raw_source(main, demographic, definition)


def test_two_main_keys_differing_only_in_leading_zeros_are_a_numeric_collision():
    main, demographic, definition = rebuilt(
        DEFAULT_MAIN + [main_record(recid="011")], DEFAULT_DEMOGRAPHIC
    )
    with pytest.raises(PufRawSourceRefusal, match="MAIN_NUMERIC_KEY_COLLISION"):
        decode_puf_raw_source(main, demographic, definition)


def test_demographic_order_does_not_change_the_main_row_order():
    main, demographic, definition = rebuilt(
        DEFAULT_MAIN, list(reversed(DEFAULT_DEMOGRAPHIC))
    )
    status = decode_puf_raw_source(main, demographic, definition)
    assert status.lexical["RECID"] == ("11", "12", "13", "999996")
    assert status.typed["demographic_status"].tolist() == [1, 0, 1, 0]
    assert status.typed["AGERANGE"].tolist() == [4, -1, 3, -1]


def test_a_demographic_row_keyed_to_an_aggregate_is_tolerated_typed_and_tagged():
    """Root decision: tolerate, type under the same domains, tag; never infer."""

    aggregate = str(PUF_AGGREGATE_RECIDS[0])
    main, demographic, definition = rebuilt(
        DEFAULT_MAIN,
        DEFAULT_DEMOGRAPHIC + [demographic_record(recid=aggregate, gender="1")],
    )
    status = decode_puf_raw_source(main, demographic, definition)
    assert status.typed["disclosure_aggregate"].tolist() == [0, 0, 0, 1]
    assert status.typed["demographic_status"].tolist() == [1, 0, 1, 1]
    assert status.typed["GENDER"].tolist() == [2, -1, 1, 1]
    assert status.lexical["GENDER"][3] == "1"
    assert status.facts["aggregate_demographic_records"] == 1
    assert status.facts["aggregate_records"] == 1
    # Still four source returns; no person, household or disaggregation appears.
    assert status.rows == 4


def test_aggregate_and_mars_zero_must_agree_in_both_directions():
    main, demographic, definition = rebuilt(
        [main_record(recid=str(PUF_AGGREGATE_RECIDS[1]), mars="1")], []
    )
    with pytest.raises(PufRawSourceRefusal, match="AGGREGATE_MARS_DISAGREEMENT"):
        decode_puf_raw_source(main, demographic, definition)

    main, demographic, definition = rebuilt([main_record(recid="41", mars="0")], [])
    with pytest.raises(PufRawSourceRefusal, match="AGGREGATE_MARS_DISAGREEMENT"):
        decode_puf_raw_source(main, demographic, definition)


@pytest.mark.parametrize(
    ("field", "value"),
    (("FLPDYR", "2011"), ("FLPDMO", "11"), ("MARS", "5"), ("DSI", "2")),
)
def test_a_main_code_outside_its_closed_domain_is_refused(field, value):
    main, demographic, definition = rebuilt(
        [main_record(recid="11", **{field.lower(): value})], []
    )
    with pytest.raises(PufRawSourceRefusal, match="MAIN_DOMAIN"):
        decode_puf_raw_source(main, demographic, definition)


@pytest.mark.parametrize(
    ("field", "value"), (("agedp1", "7"), ("earnsplit", "4"), ("gender", "0"))
)
def test_a_demographic_code_outside_its_closed_domain_is_refused(field, value):
    main, demographic, definition = rebuilt(
        DEFAULT_MAIN, [demographic_record(recid="11", **{field: value})]
    )
    with pytest.raises(PufRawSourceRefusal, match="DEMOGRAPHIC_DOMAIN"):
        decode_puf_raw_source(main, demographic, definition)


def test_agerange_is_conditional_on_dsi():
    dependent = [main_record(recid="13", mars="4", dsi="1")]
    main, demographic, definition = rebuilt(
        dependent, [demographic_record(recid="13", agerange="5")]
    )
    with pytest.raises(PufRawSourceRefusal, match="DEMOGRAPHIC_DOMAIN AGERANGE"):
        decode_puf_raw_source(main, demographic, definition)

    main, demographic, definition = rebuilt(
        dependent, [demographic_record(recid="13", agerange="3")]
    )
    status = decode_puf_raw_source(main, demographic, definition)
    assert status.typed["AGERANGE"].tolist() == [3]

    nondependent = [main_record(recid="13", mars="1", dsi="0")]
    main, demographic, definition = rebuilt(
        nondependent, [demographic_record(recid="13", agerange="5")]
    )
    assert decode_puf_raw_source(main, demographic, definition).typed[
        "AGERANGE"
    ].tolist() == [5]


@pytest.mark.parametrize(
    ("token", "reason"),
    (
        ("", "TOKEN_EMPTY"),
        ("-1", "TOKEN_GRAMMAR"),
        ("+1", "TOKEN_GRAMMAR"),
        ("1.5", "TOKEN_GRAMMAR"),
        ("1e3", "TOKEN_GRAMMAR"),
        (" 1", "TOKEN_GRAMMAR"),
        ("9" * 33, "TOKEN_OVER_WIDTH"),
    ),
)
def test_the_s006_grammar_is_unsigned_decimal_and_bounded(token, reason):
    main, demographic, definition = rebuilt([main_record(recid="11", s006=token)], [])
    with pytest.raises(PufRawSourceRefusal, match=reason):
        decode_puf_raw_source(main, demographic, definition)


def test_s006_is_kept_as_exact_integer_hundredths_and_never_divided():
    main, demographic, definition = rebuilt(
        [main_record(recid="11", s006="123456789")], []
    )
    status = decode_puf_raw_source(main, demographic, definition)
    assert status.typed["S006"].tolist() == [123456789]
    assert status.typed["S006"].dtype.kind == "i"
    assert status.lexical["S006"] == ("123456789",)
    payload = encode_return_status(status, definition)
    header = json.loads(
        payload[
            len(puf.RETURN_STATUS_MAGIC) + 4 : len(puf.RETURN_STATUS_MAGIC)
            + 4
            + int.from_bytes(
                payload[
                    len(puf.RETURN_STATUS_MAGIC) : len(puf.RETURN_STATUS_MAGIC) + 4
                ],
                "big",
            )
        ].decode("ascii")
    )
    assert header["facts"]["weight_units"] == "S006_hundredths"
    assert all(column["dtype"].startswith("<") for column in header["columns"][:14])
    assert not any(
        column.get("dtype", "").startswith("<f") for column in header["columns"]
    )


def test_an_int64_overflowing_recid_is_refused():
    main, demographic, definition = rebuilt([main_record(recid="9" * 25)], [])
    with pytest.raises(PufRawSourceRefusal, match="TOKEN_INT64_OVERFLOW"):
        decode_puf_raw_source(main, demographic, definition)


def test_an_embedded_nul_in_a_selected_token_is_refused():
    main, demographic, definition = rebuilt([main_record(recid="1\x001")], [])
    with pytest.raises(PufRawSourceRefusal, match="TOKEN_EMBEDDED_NUL"):
        decode_puf_raw_source(main, demographic, definition)


# --------------------------------------------------------------------------
# The typed envelope
# --------------------------------------------------------------------------


def test_the_envelope_round_trips_every_column_exactly(invented):
    main, demographic, definition = invented
    status = decode_puf_raw_source(main, demographic, definition)
    payload = encode_return_status(status, definition)
    assert payload.startswith(puf.RETURN_STATUS_MAGIC)
    decoded = decode_return_status(payload)
    assert decoded.rows == status.rows
    for name, array in status.typed.items():
        assert decoded.typed[name].tobytes() == array.tobytes(), name
        assert decoded.typed[name].dtype == array.dtype, name
    assert dict(decoded.lexical) == dict(status.lexical)
    assert dict(decoded.facts) == dict(status.facts)
    assert encode_return_status(decoded, definition) == payload


def test_the_envelope_declares_its_endianness_and_bounds(invented):
    main, demographic, definition = invented
    status = decode_puf_raw_source(main, demographic, definition)
    payload = encode_return_status(status, definition)
    start = len(puf.RETURN_STATUS_MAGIC)
    length = int.from_bytes(payload[start : start + 4], "big")
    header = json.loads(payload[start + 4 : start + 4 + length].decode("ascii"))
    assert header["header_length_endianness"] == "big"
    assert header["body_endianness"] == "little"
    assert header["definition_sha256"] == definition.sha256
    assert header["definition_route"] == "test_fixture"
    widths = {
        field["name"]: field["lexical_width"] for field in definition.document["fields"]
    }
    assert widths == {
        "RECID": 64,
        "S006": 32,
        "FLPDYR": 8,
        "FLPDMO": 4,
        "MARS": 4,
        "DSI": 4,
        "AGEDP1": 4,
        "AGEDP2": 4,
        "AGEDP3": 4,
        "AGERANGE": 4,
        "EARNSPLIT": 4,
        "GENDER": 4,
    }
    lexical = {
        column["name"]: column
        for column in header["columns"]
        if column["kind"].startswith("lexical")
    }
    assert lexical["RECID_lexical"]["width"] == 64
    assert lexical["RECID_lexical"]["bytes"] == status.rows * 64


def test_the_full_slice_body_bound_is_derived_and_stays_under_the_shared_opaque_cap():
    total, sizes = puf._payload_bound(PACKAGED.main.data_records, PACKAGED)
    assert total == PACKAGED.main.data_records * 169
    assert total < BODY_MAX_BYTES == 64 * 1024 * 1024
    assert sizes["RECID_lexical"] == PACKAGED.main.data_records * 64
    # An oversized row count is refused before any body is allocated.
    with pytest.raises(PufRawSourceRefusal, match="BODY_OVER_BOUND"):
        puf.encode_return_status(
            puf.PufRawReturnStatus(typed={}, lexical={}, facts={"rows": 10**9}),
            PACKAGED,
        )


@pytest.mark.parametrize("cut", (0, 1, 40))
def test_a_truncated_tampered_or_mistyped_envelope_is_refused(invented, cut):
    main, demographic, definition = invented
    status = decode_puf_raw_source(main, demographic, definition)
    payload = encode_return_status(status, definition)
    with pytest.raises(PufRawSourceRefusal):
        decode_return_status(payload[: len(payload) - cut - 1])
    with pytest.raises(PufRawSourceRefusal, match="ENVELOPE_MAGIC"):
        decode_return_status(b"not-the-magic" + payload)
    tampered = bytearray(payload)
    tampered[-1] ^= 0x01
    with pytest.raises(PufRawSourceRefusal, match="ENVELOPE_BODY_DIGEST"):
        decode_return_status(bytes(tampered))
    with pytest.raises(PufRawSourceRefusal, match="ENVELOPE_TRAILING_BYTES"):
        decode_return_status(payload + b"\0")
    start = len(puf.RETURN_STATUS_MAGIC)
    with pytest.raises(PufRawSourceRefusal, match="ENVELOPE_HEADER_LENGTH"):
        decode_return_status(
            payload[:start] + (HEADER_OVER := 10**6).to_bytes(4, "big") + payload[start + 4 :]
        )


# --------------------------------------------------------------------------
# The graph
# --------------------------------------------------------------------------


HOST_SOURCE = "fixture_host_tables"


def write_host_source(directory: Path) -> Path:
    """One invented household of engineering context, labelled as invented.

    It creates no PUF person and establishes no return-to-household mapping;
    it exists only so the artifact node has an explicitly named host.
    """

    import pandas as pd

    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"person_id": [1], "person_household_id": [1], "invented_person": [True]}
    ).to_csv(directory / "person.csv", index=False)
    pd.DataFrame({"household_id": [1], "invented_household": [True]}).to_csv(
        directory / "household.csv", index=False
    )
    pd.DataFrame({"household_id": [1], "design_weight": [1.0]}).to_csv(
        directory / "weights.csv", index=False
    )
    (directory / "schema.json").write_text(
        json.dumps(
            {
                "person_entity": "person",
                "group_entities": ["household"],
                "tables": {"person": "person.csv", "household": "household.csv"},
                "dtypes": {
                    "person_id": "int64",
                    "person_household_id": "int64",
                    "household_id": "int64",
                    "invented_person": "bool",
                    "invented_household": "bool",
                    "design_weight": "float64",
                },
                "weights": {
                    "entity": "household",
                    "column": "design_weight",
                    "kind": "design",
                },
                "weights_table": "weights.csv",
            }
        )
    )
    return directory


class FixtureHostKernel(KernelBase):
    """CREATE: the invented one-household engineering context."""

    ref = "us.puf.raw_source.fixture_host@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.BITWISE,
    )

    def run(self, context):
        return KernelResult(
            frame=load_csv_tables(context.sources[HOST_SOURCE]),
            receipt={"invented_engineering_context": True},
        )


def build_graph(definition, *, producer=None, with_consumer=True):
    node = (
        us_puf_raw_source_node(population=HOST_POPULATION, definition=definition)
        if producer is None
        else producer
    )
    nodes = [
        Node(
            HOST_POPULATION,
            FixtureHostKernel.ref,
            structural=StructuralDelta.CREATE,
            sources=(HOST_SOURCE,),
            outputs=(
                Owned("household", "invented_household", "bool"),
                Owned("person", "invented_person", "bool"),
            ),
        ),
        node,
    ]
    if with_consumer:
        nodes.append(
            us_puf_raw_return_status_summary_node(
                population=HOST_POPULATION, producer=node.id
            )
        )
    graph = Graph(
        country="us",
        sources=(
            SourceRef(HOST_SOURCE, "csv-tables"),
            SourceRef(definition.main.source_name, definition.main.codec),
            SourceRef(definition.demographic.source_name, definition.demographic.codec),
        ),
        nodes=tuple(nodes),
    )
    registry = KernelRegistry()
    registry.register(FixtureHostKernel())
    register_us_puf_raw_source_kernels(
        registry,
        definition=definition,
        source_codecs=puf_raw_source_codecs(definition),
    )
    return compile_graph(graph), registry


@pytest.fixture
def graph_fixture(invented, tmp_path):
    main, demographic, definition = invented
    sources = {
        HOST_SOURCE: write_host_source(tmp_path / "host"),
        definition.main.source_name: write(tmp_path / "main.csv", main),
        definition.demographic.source_name: write(
            tmp_path / "demographic.csv", demographic
        ),
    }
    compiled, registry = build_graph(definition)
    return compiled, registry, sources, definition, (main, demographic)



def composed_codecs(definition):
    """Explicit caller composition: csv-tables plus this slice's two codecs.

    Root decision 4: compose the registry at the caller rather than importing
    this module into the generic ``graph_sources`` production registry.
    """

    registry = puf.SourceCodecRegistry()
    registry.register("csv-tables", load_csv_tables)
    puf.register_us_puf_raw_source_codecs(registry, definition=definition)
    return registry


def composed_store(root: Path, definition) -> ContentStore:
    return ContentStore(root, codecs=composed_codecs(definition))


def artifact_bytes(manifest, store, node_id=PUF_RAW_SOURCE_NODE):
    key = manifest.nodes[node_id].opaque_artifacts["return_status"]
    return store.load_bytes(key)


def test_cold_run_stores_an_artifact_byte_identical_to_the_direct_decoder(
    graph_fixture, tmp_path
):
    compiled, registry, sources, definition, (main, demographic) = graph_fixture
    store = composed_store(tmp_path / "store", definition)
    manifest = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    payload = artifact_bytes(manifest, store)
    direct = encode_return_status(
        decode_puf_raw_source(main, demographic, definition), definition
    )
    assert payload == direct
    decoded = decode_return_status(payload)
    assert decoded.rows == 4
    assert decoded.facts["matched_keys"] == 2
    summary = manifest.nodes[RETURN_STATUS_SUMMARY_NODE].receipt[
        "puf_raw_return_status_summary"
    ]
    assert summary["rows"] == 4
    assert summary["aggregate_records"] == 1
    assert summary["amounts_decoded"] is False
    assert summary["period_semantics"] == "unresolved_raw_FLPDYR"
    producer = manifest.nodes[PUF_RAW_SOURCE_NODE].receipt["puf_raw_return_status"]
    assert producer["definition_sha256"] == definition.sha256
    assert producer["artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    # Receipts carry aggregates and digests only, never a key or a token.
    text = str(
        {
            node: dict(manifest.nodes[node].receipt)
            for node in (PUF_RAW_SOURCE_NODE, RETURN_STATUS_SUMMARY_NODE)
        }
    )
    for token in ("999996", '"11"', '"12"', '"13"'):
        assert token not in text


def test_warm_require_hits_every_node_and_never_calls_the_kernel(
    graph_fixture, tmp_path
):
    compiled, registry, sources, definition, _ = graph_fixture
    store = composed_store(tmp_path / "store", definition)
    cold = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    cold_payload = artifact_bytes(cold, store)

    calls = []
    original = USPufRawSourceKernel.run

    def counted(self, context):
        calls.append(context.node.id)
        return original(self, context)

    USPufRawSourceKernel.run = counted
    try:
        warm = run_graph(
            compiled, sources=sources, store=store, kernels=registry, resume="require"
        )
    finally:
        USPufRawSourceKernel.run = original
    assert calls == []
    assert all(node.hit for node in warm.nodes.values())
    assert artifact_bytes(warm, store) == cold_payload


def test_a_fresh_store_recomputes_the_identical_artifact_bytes(
    graph_fixture, tmp_path
):
    compiled, registry, sources, definition, _ = graph_fixture
    store = composed_store(tmp_path / "store", definition)
    cold = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    fresh_store = composed_store(tmp_path / "fresh", definition)
    fresh = run_graph(
        compiled, sources=sources, store=fresh_store, kernels=registry, resume="auto"
    )
    assert not any(node.hit for node in fresh.nodes.values())
    assert artifact_bytes(fresh, fresh_store) == artifact_bytes(cold, store)


def test_changed_source_bytes_change_the_node_key_and_recompute(
    graph_fixture, tmp_path
):
    compiled, registry, sources, definition, _ = graph_fixture
    store = composed_store(tmp_path / "store", definition)
    first = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    changed_main, changed_demographic, changed_definition = rebuilt(
        DEFAULT_MAIN + [main_record(recid="14", mars="2")], DEFAULT_DEMOGRAPHIC
    )
    changed_sources = dict(sources)
    changed_sources[changed_definition.main.source_name] = write(
        tmp_path / "changed_main.csv", changed_main
    )
    changed_compiled, changed_registry = build_graph(changed_definition)
    second = run_graph(
        changed_compiled,
        sources=changed_sources,
        store=store,
        kernels=changed_registry,
        resume="auto",
    )
    assert not second.nodes[PUF_RAW_SOURCE_NODE].hit
    assert (
        second.nodes[PUF_RAW_SOURCE_NODE].key != first.nodes[PUF_RAW_SOURCE_NODE].key
    )
    assert decode_return_status(artifact_bytes(second, store)).rows == 5


def test_a_changed_definition_parameter_changes_the_node_key(graph_fixture, tmp_path):
    compiled, registry, sources, definition, _ = graph_fixture
    store = composed_store(tmp_path / "store", definition)
    first = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    document = json.loads(definition.canonical.decode("ascii"))
    document["join"]["numeric_key_uniqueness_checked"] = False
    restated = fixture_definition(document)
    assert restated.sha256 != definition.sha256
    changed_compiled, changed_registry = build_graph(restated)
    second = run_graph(
        changed_compiled,
        sources=sources,
        store=store,
        kernels=changed_registry,
        resume="auto",
    )
    assert not second.nodes[PUF_RAW_SOURCE_NODE].hit
    assert (
        second.nodes[PUF_RAW_SOURCE_NODE].key != first.nodes[PUF_RAW_SOURCE_NODE].key
    )


def test_a_changed_dependency_version_changes_the_implementation_hash(graph_fixture):
    _, _, _, definition, _ = graph_fixture
    kernel = USPufRawSourceKernel(
        definition=definition, source_codecs=puf_raw_source_codecs(definition)
    )
    baseline = kernel.implementation_hash()

    class Restated(USPufRawSourceKernel):
        capabilities = Capabilities(
            Determinism.DETERMINISTIC,
            numeric=Numeric.BITWISE,
            dependencies=("numpy", "pandas"),
        )

    restated = Restated(
        definition=definition, source_codecs=puf_raw_source_codecs(definition)
    )
    assert restated.implementation_hash() != baseline


def test_an_unregistered_codec_refuses_before_any_kernel_runs(
    graph_fixture, tmp_path
):
    compiled, _, sources, definition, _ = graph_fixture
    registry = KernelRegistry()
    registry.register(FixtureHostKernel())
    register_us_puf_raw_source_kernels(
        registry, definition=definition, source_codecs=puf.SourceCodecRegistry()
    )
    bare = puf.SourceCodecRegistry()
    bare.register("csv-tables", load_csv_tables)
    with pytest.raises((StoreUnavailable, NodeRejectedError)):
        run_graph(
            compiled,
            sources=sources,
            store=ContentStore(tmp_path / "store", codecs=bare),
            kernels=registry,
            resume="forbid",
        )


def test_the_executor_resolves_a_symlinked_source_path_before_the_codec(
    graph_fixture, tmp_path
):
    """Actual executor behavior, not a claim the codec rejects the user's path."""

    compiled, registry, sources, definition, _ = graph_fixture
    linked = dict(sources)
    link = tmp_path / "linked_main.csv"
    link.symlink_to(sources[definition.main.source_name])
    linked[definition.main.source_name] = link
    manifest = run_graph(
        compiled,
        sources=linked,
        store=composed_store(tmp_path / "store", definition),
        kernels=registry,
        resume="forbid",
    )
    assert decode_return_status(
        artifact_bytes(manifest, composed_store(tmp_path / "store", definition))
    ).rows == 4
    # The same symlink handed straight to the codec is refused.
    with pytest.raises(PufRawSourceRefusal, match="CODEC_OPEN"):
        load_pinned_puf_bytes(link, pin=definition.main)


@pytest.mark.parametrize(
    "override",
    (
        {"inputs": (Slice("household", ("invented_household",)),)},
        {"outputs": (Owned("household", "puf_rows", "int64"),)},
        {"params": {"definition": "{}", "extra": "1"}},
        {"artifact_outputs": ()},
        {"sources": ()},
    ),
)
def test_the_kernel_refuses_any_node_shape_but_its_own(
    graph_fixture, tmp_path, override
):
    from dataclasses import replace

    _, _, sources, definition, _ = graph_fixture
    base = us_puf_raw_source_node(population=HOST_POPULATION, definition=definition)
    producer = replace(base, **override)
    compiled, registry = build_graph(
        definition, producer=producer, with_consumer=False
    )
    with pytest.raises((PufRawSourceRefusal, NodeRejectedError)):
        run_graph(
            compiled,
            sources=sources,
            store=composed_store(tmp_path / "store", definition),
            kernels=registry,
            resume="forbid",
        )


def test_the_kernel_refuses_a_create_structural_delta(graph_fixture, tmp_path):
    """A CREATE declaration is legal to the graph and still refused here."""

    _, _, sources, definition, _ = graph_fixture
    create = Node(
        PUF_RAW_SOURCE_NODE,
        USPufRawSourceKernel.ref,
        structural=StructuralDelta.CREATE,
        sources=(definition.main.source_name, definition.demographic.source_name),
        outputs=(Owned("household", "puf_rows", "int64"),),
        params={"definition": definition.params_text},
        artifact_outputs=(ArtifactOutput("return_status", RETURN_STATUS_TYPE),),
    )
    graph = Graph(
        country="us",
        sources=(
            SourceRef(definition.main.source_name, definition.main.codec),
            SourceRef(definition.demographic.source_name, definition.demographic.codec),
        ),
        nodes=(create,),
    )
    registry = KernelRegistry()
    register_us_puf_raw_source_kernels(
        registry,
        definition=definition,
        source_codecs=puf_raw_source_codecs(definition),
    )
    with pytest.raises((PufRawSourceRefusal, NodeRejectedError)):
        run_graph(
            compile_graph(graph),
            sources={
                name: path
                for name, path in sources.items()
                if name != HOST_SOURCE
            },
            store=composed_store(tmp_path / "store", definition),
            kernels=registry,
            resume="forbid",
        )


def test_the_kernel_refuses_a_definition_parameter_it_does_not_hold(
    graph_fixture, tmp_path
):
    from dataclasses import replace

    _, _, sources, definition, _ = graph_fixture
    base = us_puf_raw_source_node(population=HOST_POPULATION, definition=definition)
    document = json.loads(definition.canonical.decode("ascii"))
    document["join"]["numeric_key_uniqueness_checked"] = False
    producer = replace(
        base, params={"definition": fixture_definition(document).params_text}
    )
    compiled, registry = build_graph(
        definition, producer=producer, with_consumer=False
    )
    with pytest.raises((PufRawSourceRefusal, NodeRejectedError)):
        run_graph(
            compiled,
            sources=sources,
            store=composed_store(tmp_path / "store", definition),
            kernels=registry,
            resume="forbid",
        )


def test_the_producer_node_requires_an_explicit_population(invented):
    _, _, definition = invented
    with pytest.raises(PufRawSourceRefusal, match="NODE_POPULATION_REQUIRED"):
        us_puf_raw_source_node(population="", definition=definition)
    with pytest.raises(TypeError):
        us_puf_raw_source_node(definition=definition)


def test_a_malformed_s006_refuses_cleanly_with_no_partial_artifact(
    invented, tmp_path
):
    _, _, _ = invented
    main, demographic, definition = rebuilt(
        [main_record(recid="11", s006="")], []
    )
    sources = {
        HOST_SOURCE: write_host_source(tmp_path / "host"),
        definition.main.source_name: write(tmp_path / "main.csv", main),
        definition.demographic.source_name: write(
            tmp_path / "demographic.csv", demographic
        ),
    }
    compiled, registry = build_graph(definition)
    store = composed_store(tmp_path / "store", definition)
    with pytest.raises((PufRawSourceRefusal, NodeRejectedError)):
        run_graph(
            compiled, sources=sources, store=store, kernels=registry, resume="forbid"
        )
    # No artifact landed for the refused producer.
    direct = encode_return_status(
        decode_puf_raw_source(*rebuilt(DEFAULT_MAIN, DEFAULT_DEMOGRAPHIC)[:2],
                              rebuilt(DEFAULT_MAIN, DEFAULT_DEMOGRAPHIC)[2]),
        rebuilt(DEFAULT_MAIN, DEFAULT_DEMOGRAPHIC)[2],
    )
    with pytest.raises(Exception):
        store.load_bytes(hashlib.sha256(direct).hexdigest())


# --------------------------------------------------------------------------
# Source blindness
# --------------------------------------------------------------------------


def test_the_module_and_definition_carry_no_absolute_or_genuine_source_paths():
    module = Path(puf.__file__)
    text = module.read_text()
    ast.parse(text)
    forbidden = (
        "policyengine-us-data",
        "policyengine_us_data",
        "_buildo-runtime",
        ".cache/microcosm",
        "/Users/",
        "huggingface",
        "puf_2015.csv",
        "demographics_2015.csv",
    )
    for needle in forbidden:
        assert needle not in text, needle
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not node.value.startswith("/"), node.value
    raw = (
        Path(puf.__file__).parents[1]
        / "us"
        / "puf_2015_raw_source_definition.json"
    ).read_text()
    for needle in forbidden:
        assert needle not in raw, needle


def test_the_module_is_classified_in_the_us_runtime_blindness_allowlist():
    blindness = (
        Path(__file__).with_name("test_us_spine_blindness.py").read_text()
    )
    assert '"puf_raw_source.py"' in blindness
