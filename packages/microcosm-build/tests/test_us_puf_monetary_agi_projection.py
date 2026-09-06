"""The thirteen-field PUF monetary projection, on invented bytes only.

Nothing here reads a PUF, a capture or any genuine microdata. Every fixture
delivery is written from literals in this module and travels the explicit
``test_fixture`` route. The two packaged documents are read only for their own
closed metadata, and the accepted twelve-field document is asserted to be
exactly what it was.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import puf_monetary_agi_projection as agi
from microcosm.build.us_runtime import puf_monetary_source as money
from microcosm.build.us_runtime.puf_monetary_agi_projection import (
    AGI_FIELD,
    AGI_PROJECTION_EDITION,
    AGI_PROJECTION_MAGIC,
    AGI_PROJECTION_NODE,
    AGI_PROJECTION_TYPE,
    TWELVE_FIELD_MAGIC,
    USPufMonetaryAgiSourceKernel,
    decode_agi_projection,
    encode_agi_projection,
    fixture_agi_projection_document,
    packaged_agi_projection,
    register_us_puf_monetary_agi_source_kernels,
    us_puf_monetary_agi_source_node,
)
from microcosm.build.us_runtime.puf_monetary_source import (
    AGGREGATE_SENTINEL,
    SUPERSEDED_MAGIC,
    PufMonetaryRefusalError,
    decode_monetary_projection,
    decode_puf_monetary_source,
    encode_monetary_projection,
    fixture_projection_document,
    packaged_projection,
    projection_document_json,
)
from microcosm.build.us_runtime.puf_raw_source import (
    PUF_AGGREGATE_RECIDS,
    decode_puf_raw_source,
    definition_document_json,
    fixture_definition,
    packaged_definition,
)
from microcosm.graph import KernelRegistry, Node
from microcosm.graph.canonical import canonical_json

PACKAGED_DEFINITION = packaged_definition()
PACKAGED_TWELVE = packaged_projection()
PACKAGED_THIRTEEN = packaged_agi_projection()
MAIN_HEADER = list(PACKAGED_DEFINITION.main.delivered_header)
DEMOGRAPHIC_HEADER = list(PACKAGED_DEFINITION.demographic.delivered_header)

#: The accepted twelve-field document, pinned. A change here would move the
#: accepted artifact's identity, which this increment must not do.
REVIEWED_TWELVE_FIELD_SHA256 = (
    "5e4f5943624427376a13e656a6fedd378e2a0ff8276a90bd10bc5149f1291749"
)

ROOT = Path(money.__file__).resolve().parents[4].parents[1]


# --------------------------------------------------------------------------
# Invented fixture construction
# --------------------------------------------------------------------------


def header_digest(header) -> str:
    payload = json.dumps(
        list(header),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    ).hexdigest()


def csv_bytes(header, records) -> bytes:
    lines = [",".join(header)]
    lines.extend(",".join(record) for record in records)
    return ("\n".join(lines) + "\n").encode("utf-8")


def main_record(*, recid, mars="1", s006="12345", flpdyr="2015", amounts=None):
    values = {
        "RECID": recid,
        "MARS": mars,
        "DSI": "0",
        "S006": s006,
        "FLPDYR": flpdyr,
        "FLPDMO": "12",
        **(amounts or {}),
    }
    return [values.get(name, "0") for name in MAIN_HEADER]


def demographic_record(*, recid):
    values = {
        "RECID": recid,
        "AGEDP1": "0",
        "AGEDP2": "0",
        "AGEDP3": "0",
        "AGERANGE": "2",
        "EARNSPLIT": "0",
        "GENDER": "1",
    }
    return [values[name] for name in DEMOGRAPHIC_HEADER]


#: Four invented individual returns and one publisher disclosure aggregate.
#: Every value is invented: a positive return, a loss return with a negative
#: reported AGI, an all-zero return, a small return, and an aggregate whose
#: AGI slot carries an invented fractional token.
DEFAULT_MAIN = [
    main_record(
        recid="11",
        amounts={
            "E00100": "71000",
            "E00200": "50000",
            "E00300": "125",
            "E00400": "40",
            "E00600": "900",
            "E00650": "700",
            "E00900": "12000",
            "E01000": "3400",
            "E01500": "8000",
            "E01700": "6400",
            "E02100": "150",
            "P22250": "1200",
            "P23250": "2200",
        },
    ),
    main_record(
        recid="12",
        mars="2",
        amounts={
            "E00100": "-9300",
            "E00200": "31000",
            "E00900": "-2500",
            "E01000": "-3000",
            "E02100": "-780",
            "P22250": "-3000",
            "P23250": "0",
        },
    ),
    main_record(recid="13", mars="4"),
    main_record(recid="14", amounts={"E00100": "79", "E00200": "77", "E00650": "2"}),
    main_record(
        recid=str(PUF_AGGREGATE_RECIDS[0]),
        mars="0",
        s006="0",
        amounts={
            "E00100": "-999999999.50",
            "E00200": "999999999.12",
            "E01000": "-99999999.75",
            "E00300": "4200",
        },
    ),
]
DEFAULT_DEMOGRAPHIC = [demographic_record(recid="11"), demographic_record(recid="13")]


def fixture_definition_document(main_payload: bytes, demographic_payload: bytes):
    document = definition_document_json(PACKAGED_DEFINITION)
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


def fixture_projection_json(definition, projection):
    document = projection_document_json(projection)
    document["route"] = "test_fixture"
    document["authority"] = "invented_fixture_nonauthority"
    document["raw_source_definition"]["sha256"] = definition.sha256
    document["raw_source_definition"]["bytes"] = len(definition.canonical)
    for name, pin in (
        ("main", definition.main),
        ("demographic", definition.demographic),
    ):
        entry = document["sources"][name]
        entry["sha256"] = pin.sha256
        entry["git_blob_sha1"] = pin.git_blob_sha1
        entry["bytes"] = pin.bytes
        entry["header_record_canonical_sha256"] = pin.header_record_canonical_sha256
        entry["delivered_header_width"] = len(pin.delivered_header)
        entry["data_records"] = pin.data_records
    return document


def build_fixture(main_records=None):
    mains = DEFAULT_MAIN if main_records is None else main_records
    main = csv_bytes(MAIN_HEADER, mains)
    demographic = csv_bytes(DEMOGRAPHIC_HEADER, DEFAULT_DEMOGRAPHIC)
    document = fixture_definition_document(main, demographic)
    document["sources"]["main"]["data_records"] = len(mains)
    definition = fixture_definition(document)
    projection = fixture_agi_projection_document(
        fixture_projection_json(definition, PACKAGED_THIRTEEN), definition
    )
    return main, demographic, definition, projection


@pytest.fixture
def invented():
    return build_fixture()


@pytest.fixture
def decoded(invented):
    main, demographic, definition, projection = invented
    status = decode_puf_raw_source(main, demographic, definition)
    return (
        decode_puf_monetary_source(main, status, projection, definition),
        projection,
        definition,
        status,
    )


def payload_for(decoded_tuple) -> bytes:
    decoded_value, projection, definition, status = decoded_tuple
    return encode_agi_projection(
        decoded_value,
        projection,
        definition,
        status_artifact_sha256="0" * 64,
    )


# --------------------------------------------------------------------------
# The packaged thirteen-field document
# --------------------------------------------------------------------------


def test_the_edition_adds_exactly_one_column_to_the_accepted_twelve():
    assert PACKAGED_TWELVE.sha256 == REVIEWED_TWELVE_FIELD_SHA256
    assert len(PACKAGED_TWELVE.fields) == 12
    assert AGI_FIELD not in PACKAGED_TWELVE.fields
    assert len(PACKAGED_THIRTEEN.fields) == 13
    assert AGI_FIELD in PACKAGED_THIRTEEN.fields
    assert set(PACKAGED_THIRTEEN.fields) - set(PACKAGED_TWELVE.fields) == {AGI_FIELD}
    edition = PACKAGED_THIRTEEN.document["projection_edition"]
    assert edition["edition"] == AGI_PROJECTION_EDITION
    assert tuple(edition["adds"]) == (AGI_FIELD,)
    assert edition["parent_document_canonical_sha256"] == PACKAGED_TWELVE.sha256
    assert tuple(edition["parent_projected_fields"]) == PACKAGED_TWELVE.fields


def test_the_twelve_accepted_columns_are_restated_by_nothing():
    """The edition adds a column; it must not redeclare the other twelve."""
    upstream = {column.field: column for column in PACKAGED_TWELVE.columns}
    for column in PACKAGED_THIRTEEN.columns:
        if column.field == AGI_FIELD:
            continue
        assert column == upstream[column.field]


def test_the_columns_are_in_delivered_order_with_agi_at_its_own_index():
    indices = [column.delivered_index for column in PACKAGED_THIRTEEN.columns]
    assert indices == sorted(indices)
    agi_column = next(c for c in PACKAGED_THIRTEEN.columns if c.field == AGI_FIELD)
    assert agi_column.delivered_index == 73
    assert agi_column.publisher_position == 31
    assert agi_column.publisher_signed_marker is True
    assert agi_column.negative_values == "documented_by_record_layout_marker"
    assert agi_column.dtype == "int64"
    assert agi_column.lexical_width == 13
    assert agi_column.max_digits == 12
    assert agi_column.grammar == "ascii_decimal_signed_integer"


def test_agi_is_the_delivered_field_and_no_component_identity_is_claimed():
    document = PACKAGED_THIRTEEN.document
    entry = next(
        column
        for column in document["projection"]["columns"]
        if column["field"] == AGI_FIELD
    )
    assert entry["derived_combinations"] == "refused"
    assert entry["target_year_use"] == "refused_pending_independent_growth_decision"
    assert entry["source_period_status"]["status"] == "unresolved_raw_FLPDYR"
    codes = {item["code"] for item in entry["unresolved"]}
    assert "component_identity_with_the_projected_amounts" in codes
    assert document["non_claims"]["amount_growth_or_restatement"] is False
    assert document["projection_edition"]["not_derived"].startswith(
        "E00100 is the delivered reported field"
    )
    # The upstream source-only rationale is superseded in writing, not deleted.
    superseded = document["projection_edition"]["supersedes_notable_source_only"]
    assert AGI_FIELD in superseded
    assert superseded[AGI_FIELD]["upstream_statement"]
    assert superseded[AGI_FIELD]["why_it_no_longer_applies"]
    assert AGI_FIELD not in document["layout"]["notable_source_only"]


def test_the_document_names_the_generator_that_built_it_and_its_parent():
    generation = PACKAGED_THIRTEEN.document["generation"]
    tool = ROOT / "tools/build_us_puf_agi_source_layout.py"
    parent = ROOT / "tools/build_us_puf_source_layout.py"
    assert generation["generator"] == "tools/build_us_puf_agi_source_layout.py"
    assert (
        hashlib.sha256(tool.read_bytes()).hexdigest() == generation["generator_sha256"]
    )
    assert (
        hashlib.sha256(parent.read_bytes()).hexdigest()
        == generation["parent_generator_sha256"]
    )
    # The parent generator is still the one the accepted document names.
    assert (
        PACKAGED_TWELVE.document["generation"]["generator_sha256"]
        == generation["parent_generator_sha256"]
    )
    assert generation["reads_microdata"] is False


def test_the_document_is_a_runtime_resource_not_a_country_spec():
    build_root = Path(money.__file__).resolve().parents[1]
    country_package = build_root / "us"
    name = "puf_2015_monetary_agi_source_projection.json"
    assert not (country_package / name).exists()
    assert (build_root / "us_runtime" / name).exists()
    declared = json.loads(
        (country_package / "country_package.json").read_text(encoding="utf-8")
    )["resources"]
    listed = {row["path"] if isinstance(row, dict) else row for row in declared}
    assert name not in listed


def test_an_edition_block_that_does_not_bind_the_parent_is_refused(monkeypatch):
    document = projection_document_json(PACKAGED_THIRTEEN)
    for mutate, reason in (
        (lambda d: d.pop("projection_edition"), "AGI_PROJECTION_EDITION_MISSING"),
        (
            lambda d: d["projection_edition"].__setitem__("edition", "something_else"),
            "AGI_PROJECTION_EDITION",
        ),
        (
            lambda d: d["projection_edition"].__setitem__("adds", ["E00200"]),
            "AGI_PROJECTION_EDITION_ADDS",
        ),
        (
            lambda d: d["projection_edition"].__setitem__(
                "parent_document_canonical_sha256", "0" * 64
            ),
            "AGI_PROJECTION_PARENT_DOCUMENT",
        ),
        (
            lambda d: d["projection_edition"].__setitem__(
                "parent_projected_fields", ["E00200"]
            ),
            "AGI_PROJECTION_PARENT_FIELDS",
        ),
    ):
        candidate = json.loads(json.dumps(document))
        mutate(candidate)
        with pytest.raises(PufMonetaryRefusalError) as error:
            agi._check_agi_edition(
                candidate,
                agi._projection_from_document(
                    candidate, route="packaged", definition=PACKAGED_DEFINITION
                ),
            )
        assert error.value.reason == reason


def test_a_fixture_projection_without_agi_is_refused(invented):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition, PACKAGED_TWELVE)
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_agi_projection_document(document, definition)
    assert error.value.reason == "AGI_PROJECTION_FIELD_MISSING"


# --------------------------------------------------------------------------
# Decoding: the thirteenth column obeys the same universe rules
# --------------------------------------------------------------------------


def test_agi_is_decoded_under_the_ordinary_integer_grammar(decoded):
    decoded_value, _, _, _ = decoded
    known = decoded_value.amount_known
    values = decoded_value.typed[AGI_FIELD]
    assert values.dtype == np.dtype("<i8")
    assert values[known].tolist() == [71000, -9300, 0, 79]
    assert decoded_value.lexical[AGI_FIELD][0] == "71000"
    assert decoded_value.lexical[AGI_FIELD][1] == "-9300"


def test_an_aggregate_agi_slot_holds_the_sentinel_and_keeps_its_token(decoded):
    decoded_value, _, _, _ = decoded
    known = decoded_value.amount_known
    assert known.tolist() == [True, True, True, True, False]
    values = decoded_value.typed[AGI_FIELD]
    assert int(values[~known][0]) == AGGREGATE_SENTINEL
    assert decoded_value.lexical[AGI_FIELD][4] == "-999999999.50"
    pairs = decoded_value.aggregate_lexical_rows(AGI_FIELD)[AGI_FIELD]
    assert pairs == ((PUF_AGGREGATE_RECIDS[0], "-999999999.50"),)


def test_the_facts_carry_thirteen_columns_and_three_row_classes(decoded):
    decoded_value, _, _, _ = decoded
    facts = decoded_value.facts
    assert facts["projected_columns"] == 13
    assert facts["rows"] == 5
    assert facts["amount_known_rows"] == 4
    assert facts["row_class_rows"] == {
        "disclosure_aggregate": 1,
        "demographic_matched": 2,
        "demographic_unmatched": 2,
    }
    classes = facts["column_facts"][AGI_FIELD]
    assert classes["disclosure_aggregate"] == {
        "rows": 1,
        "integer_tokens": 0,
        "fractional_tokens": 1,
        "negative_tokens": 1,
        "max_token_width": 13,
    }
    assert classes["demographic_matched"]["sum"] == 71000
    assert classes["demographic_unmatched"]["sum"] == -9300 + 79


@pytest.mark.parametrize(
    "token,reason",
    [
        ("71000.5", "AMOUNT_TOKEN_GRAMMAR"),
        ("-0", "AMOUNT_TOKEN_NEGATIVE_ZERO"),
        ("+71000", "AMOUNT_TOKEN_GRAMMAR"),
        ("", "AMOUNT_TOKEN_EMPTY"),
        ("1234567890123", "AMOUNT_TOKEN_OVER_DIGITS"),
        (" 71000", "AMOUNT_TOKEN_GRAMMAR"),
    ],
)
def test_an_ordinary_agi_token_keeps_the_unchanged_grammar(token, reason):
    records = [main_record(recid="11", amounts={AGI_FIELD: token})] + DEFAULT_MAIN[1:]
    main, demographic, definition, projection = build_fixture(records)
    status = decode_puf_raw_source(main, demographic, definition)
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(main, status, projection, definition)
    assert error.value.reason == reason


# --------------------------------------------------------------------------
# The artifact: roundtrip, identity and tamper
# --------------------------------------------------------------------------


def test_the_artifact_round_trips_exactly(decoded):
    decoded_value, projection, _, _ = decoded
    payload = payload_for(decoded)
    assert payload.startswith(AGI_PROJECTION_MAGIC)
    read = decode_agi_projection(payload, projection)
    assert sorted(read.typed) == sorted(decoded_value.typed)
    for name, array in decoded_value.typed.items():
        assert np.array_equal(read.typed[name], array)
        assert read.typed[name].dtype == array.dtype
    assert read.lexical == decoded_value.lexical
    assert canonical_json(dict(read.facts)) == canonical_json(dict(decoded_value.facts))
    assert read.amount_known.tolist() == decoded_value.amount_known.tolist()


def test_the_artifact_declares_its_own_type_and_not_the_twelve_field_one(decoded):
    payload = payload_for(decoded)
    header = json.loads(
        payload[
            len(AGI_PROJECTION_MAGIC) + 4 : len(AGI_PROJECTION_MAGIC)
            + 4
            + int.from_bytes(
                payload[len(AGI_PROJECTION_MAGIC) : len(AGI_PROJECTION_MAGIC) + 4],
                "big",
            )
        ].decode("utf-8")
    )
    assert header["type"] == [AGI_PROJECTION_TYPE.name, 1]
    assert AGI_PROJECTION_TYPE.name != money.MONETARY_PROJECTION_TYPE.name


def test_a_twelve_field_payload_is_refused_by_name(invented):
    main, demographic, definition, _ = invented
    twelve = fixture_projection_document(
        fixture_projection_json(definition, PACKAGED_TWELVE), definition
    )
    status = decode_puf_raw_source(main, demographic, definition)
    decoded_twelve = decode_puf_monetary_source(main, status, twelve, definition)
    payload = encode_monetary_projection(
        decoded_twelve, twelve, definition, status_artifact_sha256="0" * 64
    )
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(payload, twelve)
    assert error.value.reason == "ENVELOPE_TWELVE_FIELD_MAGIC"
    for magic in SUPERSEDED_MAGIC:
        with pytest.raises(PufMonetaryRefusalError) as error:
            decode_agi_projection(magic + b"rest")
        assert error.value.reason == "ENVELOPE_TWELVE_FIELD_MAGIC"
    assert TWELVE_FIELD_MAGIC == (money.MONETARY_PROJECTION_MAGIC, *SUPERSEDED_MAGIC)


def test_the_twelve_field_reader_refuses_a_thirteen_field_payload(decoded):
    """The accepted reader must not read this artifact at all."""
    payload = payload_for(decoded)
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(payload)
    assert error.value.reason == "ENVELOPE_MAGIC"


def test_a_tampered_body_byte_is_caught(decoded):
    payload = bytearray(payload_for(decoded))
    payload[-1] ^= 0xFF
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(bytes(payload), decoded[1])
    assert error.value.reason == "ENVELOPE_BODY_DIGEST"


def test_a_header_fact_that_the_arrays_do_not_carry_is_caught(decoded):
    decoded_value, projection, definition, _ = decoded
    payload = payload_for(decoded)
    start = len(AGI_PROJECTION_MAGIC)
    length = int.from_bytes(payload[start : start + 4], "big")
    header = json.loads(payload[start + 4 : start + 4 + length].decode("utf-8"))
    header["facts"]["amount_known_rows"] = 5
    encoded = canonical_json(header)
    forged = b"".join(
        (
            AGI_PROJECTION_MAGIC,
            len(encoded).to_bytes(4, "big"),
            encoded,
            payload[start + 4 + length :],
        )
    )
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(forged, projection)
    assert error.value.reason == "ENVELOPE_FACTS_INCONSISTENT"


def test_a_sentinel_parked_on_an_ordinary_row_is_caught(decoded):
    """Readback re-parses every token, so a forged typed body disagrees."""
    decoded_value, projection, definition, _ = decoded
    typed = {name: np.array(array) for name, array in decoded_value.typed.items()}
    typed[AGI_FIELD][0] = AGGREGATE_SENTINEL
    forged = money.PufMonetarySourceProjection(
        typed=typed,
        lexical=decoded_value.lexical,
        facts=decoded_value.facts,
        column_metadata=decoded_value.column_metadata,
    )
    payload = encode_agi_projection(
        forged, projection, definition, status_artifact_sha256="0" * 64
    )
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(payload, projection)
    assert error.value.reason == "READBACK_LEXICAL_DISAGREEMENT"


def test_trailing_and_truncated_payloads_are_refused(decoded):
    payload = payload_for(decoded)
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(payload + b"\0", decoded[1])
    assert error.value.reason == "ENVELOPE_TRAILING_BYTES"
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(payload[:-1], decoded[1])
    assert error.value.reason == "ENVELOPE_BODY_TRUNCATED"
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(b"not this artifact")
    assert error.value.reason == "ENVELOPE_MAGIC"


def test_a_fixture_payload_needs_its_projection_in_hand(decoded):
    payload = payload_for(decoded)
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(payload)
    assert error.value.reason == "ENVELOPE_PROJECTION_REQUIRED"
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(payload, PACKAGED_THIRTEEN)
    assert error.value.reason == "ENVELOPE_PROJECTION_MISMATCH"


def test_a_status_digest_mismatch_is_refused(decoded):
    payload = payload_for(decoded)
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_agi_projection(
            payload, decoded[1], expected_status_artifact_sha256="1" * 64
        )
    assert error.value.reason == "ENVELOPE_STATUS_ARTIFACT_MISMATCH"


# --------------------------------------------------------------------------
# The kernel and its node
# --------------------------------------------------------------------------


def test_the_kernel_refuses_a_projection_without_agi(invented):
    _, _, definition, _ = invented
    twelve = fixture_projection_document(
        fixture_projection_json(definition, PACKAGED_TWELVE), definition
    )
    with pytest.raises(PufMonetaryRefusalError) as error:
        USPufMonetaryAgiSourceKernel(projection=twelve, definition=definition)
    assert error.value.reason == "AGI_PROJECTION_FIELD_MISSING"


def test_the_node_declares_the_thirteen_field_artifact_type():
    node = us_puf_monetary_agi_source_node(population="invented_host")
    assert node.id == AGI_PROJECTION_NODE
    assert node.kernel == USPufMonetaryAgiSourceKernel.ref
    assert [output.type for output in node.artifact_outputs] == [AGI_PROJECTION_TYPE]
    assert (
        node.artifact_inputs[0].type.name == "microcosm.us.puf_2015_raw_return_status"
    )
    assert not node.outputs and not node.inputs
    assert set(node.params) == {"definition", "projection"}
    assert node.params["projection"] == PACKAGED_THIRTEEN.params_text


def test_the_kernel_identity_differs_from_the_twelve_field_kernel():
    thirteen = USPufMonetaryAgiSourceKernel()
    twelve = money.USPufMonetarySourceKernel()
    assert thirteen.ref != twelve.ref
    assert thirteen.implementation_hash() != twelve.implementation_hash()
    registry = register_us_puf_monetary_agi_source_kernels(KernelRegistry())
    assert registry.get(USPufMonetaryAgiSourceKernel.ref) is not None


def test_a_node_declaration_that_is_not_this_contract_is_refused():
    """A node the kernel did not declare cannot borrow its identity."""

    class _Context:
        node = Node(
            "wrong", USPufMonetaryAgiSourceKernel.ref, population="invented_host"
        )
        params = {}
        artifacts = {}
        sources = {}

    with pytest.raises(PufMonetaryRefusalError) as error:
        USPufMonetaryAgiSourceKernel().run(_Context())
    assert error.value.reason == "NODE_DECLARATION"


# --------------------------------------------------------------------------
# The generator's edit to the frozen module is scoped, including on failure
# --------------------------------------------------------------------------


def load_layout_tool():
    """Import the extension generator by path, without running it.

    Importing only defines names — the module's top level computes paths and
    nothing else — so this reads no booklet, no document and no microdata. The
    two calls that would are not made here; the lane's own
    ``…-generator-reentrancy-*.json`` runs make them against a documents
    directory this repository does not carry.
    """
    path = ROOT / "tools/build_us_puf_agi_source_layout.py"
    spec = importlib.util.spec_from_file_location("_agi_layout_tool_for_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:  # pragma: no cover - import of a path-loaded module
        del sys.modules[spec.name]
        raise
    return module


class _FrozenParentStub:
    """Only the two module globals the extension edits."""

    def __init__(self):
        self.PROJECTION = ("twelve", "declarations")
        self.NOTABLE_SOURCE_ONLY = {AGI_FIELD: "source-only upstream", "E01100": "x"}


def test_the_extension_restores_the_cached_parents_declarations():
    """``load_parent`` caches the module, so the edit must not outlive the call."""
    tool = load_layout_tool()
    parent = _FrozenParentStub()
    projection, notable = parent.PROJECTION, dict(parent.NOTABLE_SOURCE_ONLY)

    with tool.extended_parent(parent, "thirteenth"):
        assert parent.PROJECTION == (*projection, "thirteenth")
        assert AGI_FIELD not in parent.NOTABLE_SOURCE_ONLY

    assert parent.PROJECTION == projection
    assert parent.NOTABLE_SOURCE_ONLY == notable


def test_the_declarations_are_restored_when_the_block_refuses():
    """The failure path is the one that matters: a retry must see twelve.

    Without this, a caller that caught one ``GenerationError`` and tried again
    would be told the frozen parent no longer reproduces the accepted
    twelve-field document — a claim about the accepted artifact produced
    entirely by the extension's own leftover state.
    """
    tool = load_layout_tool()
    parent = _FrozenParentStub()
    projection, notable = parent.PROJECTION, dict(parent.NOTABLE_SOURCE_ONLY)

    with pytest.raises(tool.GenerationError):
        with tool.extended_parent(parent, "thirteenth"):
            raise tool.GenerationError("a refusal raised inside the block")

    assert parent.PROJECTION == projection
    assert parent.NOTABLE_SOURCE_ONLY == notable

    # And again, because a second entry must start from the restored state.
    with tool.extended_parent(parent, "thirteenth"):
        assert len(parent.PROJECTION) == len(projection) + 1
    assert parent.PROJECTION == projection
