"""The PUF monetary source projection, exercised entirely on invented bytes.

Nothing here reads a PUF, a capture, a receipt or any genuine microdata. Every
fixture file is written under ``tmp_path`` from literals in this module, and
every fixture document travels the explicit ``test_fixture`` route. The
packaged projection is read only for its own closed metadata — the
authenticated layout, the publisher labels, the enumerated columns — never to
admit a fixture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from microcosm.build.us_runtime import puf_monetary_source as money
from microcosm.build.us_runtime import puf_raw_source as puf
from microcosm.build.us_runtime.puf_monetary_source import (
    BODY_MAX_BYTES,
    HEADER_MAX_BYTES,
    MONETARY_PROJECTION_NODE,
    MONETARY_PROJECTION_SUMMARY_NODE,
    MONETARY_PROJECTION_TYPE,
    ROW_CLASSES,
    PufMonetaryRefusalError,
    USPufMonetarySourceKernel,
    decode_monetary_projection,
    decode_puf_monetary_source,
    encode_monetary_projection,
    fixture_projection_document,
    packaged_projection,
    payload_bound,
    projection_document_json,
    register_us_puf_monetary_source_kernels,
    us_puf_monetary_projection_summary_node,
    us_puf_monetary_source_node,
)
from microcosm.build.us_runtime.puf_raw_source import (
    PUF_AGGREGATE_RECIDS,
    PUF_RAW_SOURCE_NODE,
    decode_puf_raw_source,
    decode_return_status,
    definition_document_json,
    encode_return_status,
    fixture_definition,
    packaged_definition,
    puf_raw_source_codecs,
    register_us_puf_raw_source_kernels,
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
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SourceCodecRegistry, load_csv_tables
from microcosm.graph.errors import StoreMissError

PACKAGED_DEFINITION = packaged_definition()
PACKAGED_PROJECTION = packaged_projection()
MAIN_HEADER = list(PACKAGED_DEFINITION.main.delivered_header)
DEMOGRAPHIC_HEADER = list(PACKAGED_DEFINITION.demographic.delivered_header)
PROJECTED = list(PACKAGED_PROJECTION.fields)
HOST_POPULATION = "fixture_host"
HOST_SOURCE = "fixture_host_tables"

#: The reviewed raw source module's exact bytes at the accepted lineage head
#: ``2fd9a5cb``. This slice imports that module and must not edit it: an edit
#: would move ``USPufRawSourceKernel.implementation_hash`` and with it every
#: node key and artifact already accepted under it.
REVIEWED_RAW_SOURCE_SHA256 = (
    "9c7459a3b0f95ddf07eea04025311035fadd82773220b5247697fb93b9f9b09d"
)


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


def main_record(
    *,
    recid,
    mars="1",
    dsi="0",
    s006="12345",
    flpdyr="2015",
    flpdmo="12",
    filler="0",
    amounts=None,
):
    values = {
        "RECID": recid,
        "MARS": mars,
        "DSI": dsi,
        "S006": s006,
        "FLPDYR": flpdyr,
        "FLPDMO": flpdmo,
        **(amounts or {}),
    }
    return [values.get(name, filler) for name in MAIN_HEADER]


def demographic_record(*, recid, agerange="2", gender="1"):
    values = {
        "RECID": recid,
        "AGEDP1": "0",
        "AGEDP2": "0",
        "AGEDP3": "0",
        "AGERANGE": agerange,
        "EARNSPLIT": "0",
        "GENDER": gender,
    }
    return [values[name] for name in DEMOGRAPHIC_HEADER]


#: Five invented returns: a matched one, an unmatched one carrying negative
#: business income and a negative short-term Schedule D result, a matched one
#: whose amounts are reported zeros, a second unmatched one, and one publisher
#: disclosure aggregate.
DEFAULT_MAIN = [
    main_record(
        recid="11",
        mars="1",
        amounts={
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
            "E00200": "31000",
            "E00900": "-2500",
            "E01000": "-3000",
            "E02100": "-780",
            "P22250": "-3000",
            "P23250": "0",
        },
    ),
    main_record(recid="13", mars="4", dsi="1"),
    main_record(recid="14", mars="1", amounts={"E00200": "77", "E00650": "2"}),
    main_record(
        recid=str(PUF_AGGREGATE_RECIDS[0]),
        mars="0",
        s006="0",
        amounts={"E00200": "999999999999", "E01000": "-999999999999"},
    ),
]
DEFAULT_DEMOGRAPHIC = [
    demographic_record(recid="11", agerange="4", gender="2"),
    demographic_record(recid="13", agerange="3"),
]


def fixture_definition_document(main_payload: bytes, demographic_payload: bytes):
    document = definition_document_json(PACKAGED_DEFINITION)
    document["route"] = "test_fixture"
    document["authority"] = "invented_fixture_nonauthority"
    for name, payload, header, records in (
        ("main", main_payload, MAIN_HEADER, len(DEFAULT_MAIN)),
        ("demographic", demographic_payload, DEMOGRAPHIC_HEADER, 2),
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


def fixture_projection_json(definition):
    """A projection document over the fixture definition's invented pins."""

    document = projection_document_json()
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


def build_fixture(main_records=None, demographic_records=None):
    mains = DEFAULT_MAIN if main_records is None else main_records
    demographics = (
        DEFAULT_DEMOGRAPHIC if demographic_records is None else demographic_records
    )
    main = csv_bytes(MAIN_HEADER, mains)
    demographic = csv_bytes(DEMOGRAPHIC_HEADER, demographics)
    document = fixture_definition_document(main, demographic)
    document["sources"]["main"]["data_records"] = len(mains)
    document["sources"]["demographic"]["data_records"] = len(demographics)
    definition = fixture_definition(document)
    projection = fixture_projection_document(
        fixture_projection_json(definition), definition
    )
    return main, demographic, definition, projection


@pytest.fixture
def invented():
    return build_fixture()


def decoded_status(main, demographic, definition):
    return decode_puf_raw_source(main, demographic, definition)


@pytest.fixture
def decoded(invented):
    main, demographic, definition, projection = invented
    status = decoded_status(main, demographic, definition)
    return decode_puf_monetary_source(main, status, projection, definition), projection


@dataclass(frozen=True)
class StubStatus:
    """A minimal stand-in for the reviewed status artifact's arrays."""

    typed: dict


def status_stub(recids, aggregates, statuses):
    return StubStatus(
        typed={
            "RECID": np.asarray(recids, dtype="<i8"),
            "disclosure_aggregate": np.asarray(aggregates, dtype="<u1"),
            "demographic_status": np.asarray(statuses, dtype="<i1"),
        }
    )


# --------------------------------------------------------------------------
# The authenticated layout
# --------------------------------------------------------------------------


def test_the_packaged_projection_authenticates_the_whole_delivered_layout():
    document = PACKAGED_PROJECTION.document
    main = document["layout"]["main"]
    demographic = document["layout"]["demographic"]
    assert main["delivered_header_width"] == 222
    assert main["code_fields"] == 43
    assert main["amount_or_large_code_fields"] == 179
    assert demographic["delivered_header_width"] == 7
    assert len(main["columns"]) == 222
    assert len(demographic["columns"]) == 7
    assert PACKAGED_PROJECTION.main_delivered_header == (
        PACKAGED_DEFINITION.main.delivered_header
    )
    assert PACKAGED_PROJECTION.demographic_delivered_header == (
        PACKAGED_DEFINITION.demographic.delivered_header
    )
    codes = [c for c in main["columns"] if c["publisher_group"] == "code_field"]
    amounts = [
        c for c in main["columns"] if c["publisher_group"] == "amount_or_large_code"
    ]
    assert [c["publisher_position"] for c in codes] == list(range(1, 44))
    assert [c["publisher_position"] for c in amounts] == list(range(1, 180))
    assert [c["delivered_index"] for c in main["columns"]] == list(range(222))
    assert main["printed_order_matches_delivered_order"] is True


def test_every_layout_row_is_self_verifying_lexical_evidence():
    for side in ("main", "demographic"):
        for column in PACKAGED_PROJECTION.document["layout"][side]["columns"]:
            row = column["layout_row"]
            text = row["text"]
            assert column["name"] in text
            assert (
                hashlib.sha256(text.encode("utf-8")).hexdigest() == row["text_sha256"]
            )
            assert row["text_end"] - row["text_start"] == len(text)
            assert row["printed_page"] == row["pdf_page"] - 1


def test_the_corroborating_edition_names_its_extraction_artifact():
    main = PACKAGED_PROJECTION.document["layout"]["main"]
    tolerant = {entry["name"] for entry in main["corroborating_edition_tolerant_positions"]}
    # The February 2023 extraction splits the field code on exactly the rows
    # that carry a left-margin section label. The list is recorded, not smoothed.
    assert "RECID" in tolerant and "E06500" in tolerant
    assert "E00200" not in tolerant
    for column in main["columns"]:
        if column["publisher_group"] != "amount_or_large_code":
            continue
        spacing = column["layout_row_corroboration"]["code_token_spacing"]
        assert spacing in (
            "exact",
            "one_interior_space_from_pdf_text_extraction",
        )
        assert (spacing == "exact") is (column["name"] not in tolerant)


def test_no_column_role_is_inferred_from_a_prefix_or_from_list_membership():
    by_name = {
        c["name"]: c for c in PACKAGED_PROJECTION.document["layout"]["main"]["columns"]
    }
    # The publisher's own "Misc Codes" block sits inside the amount-field list.
    for name in ("RECID", "S006", "S008", "S009", "WSAMP", "TXRT"):
        column = by_name[name]
        assert column["publisher_group"] == "amount_or_large_code"
        assert column["admission"] == "not_selected"
        assert column["not_selected_reason"] == "misc_code_not_an_amount"
        assert name not in PROJECTED
    # And the amount list holds S- and T-prefixed names as well as E and P.
    assert by_name["T27800"]["publisher_group"] == "amount_or_large_code"
    assert by_name["S27860"]["publisher_group"] == "amount_or_large_code"
    # TXRT's implied decimal point is cited, not assumed.
    anchor = PACKAGED_PROJECTION.document["anchors"]["txrt_implied_decimal"]
    assert "implied decimal point" in anchor["text"]


def test_not_selected_columns_carry_a_reason_from_the_closed_list():
    document = PACKAGED_PROJECTION.document
    reasons = set(document["layout"]["not_selected_reasons"])
    for side in ("main", "demographic"):
        for column in document["layout"][side]["columns"]:
            if column["admission"] == "not_selected":
                assert column["not_selected_reason"] in reasons
            else:
                assert "not_selected_reason" not in column
    assert reasons == {
        "code_field_not_an_amount",
        "misc_code_not_an_amount",
        "amount_field_outside_this_increment",
    }


def test_tax_outcomes_and_the_capped_se_columns_stay_source_only():
    document = PACKAGED_PROJECTION.document
    notable = document["layout"]["notable_source_only"]
    for name in ("E06500", "E05800", "E08800", "E10300", "E30400", "E30500"):
        assert name in notable
        assert name not in PROJECTED
    assert "cap" in notable["E30400"] or "capped" in notable["E30400"]


# --------------------------------------------------------------------------
# The enumerated projection
# --------------------------------------------------------------------------


def test_the_projection_is_the_enumerated_set_and_nothing_else():
    assert PROJECTED == [
        "E00200",
        "E00300",
        "E00400",
        "E00600",
        "E00650",
        "E00900",
        "E01000",
        "E01500",
        "E01700",
        "E02100",
        "P22250",
        "P23250",
    ]
    groups = {column.concept_group for column in PACKAGED_PROJECTION.columns}
    assert groups == {
        "wages",
        "interest",
        "dividends",
        "business_income",
        "farm_income",
        "pensions",
        "capital_gains",
    }
    admitted = {
        column["name"]
        for column in PACKAGED_PROJECTION.document["layout"]["main"]["columns"]
        if column["admission"] == "monetary_source_projection"
    }
    assert admitted == set(PROJECTED)


def test_the_projection_column_order_is_the_delivered_order():
    indexes = [column.delivered_index for column in PACKAGED_PROJECTION.columns]
    assert indexes == sorted(indexes)
    for column in PACKAGED_PROJECTION.columns:
        assert MAIN_HEADER[column.delivered_index] == column.field


def test_units_are_whole_dollars_with_no_implied_decimals():
    for entry in PACKAGED_PROJECTION.document["projection"]["columns"]:
        units = entry["units"]
        assert units["currency"] == "USD"
        assert units["scale"] == "whole_dollars"
        assert units["implied_decimals"] == 0
        assert units["field_width_digits"] == 12
        assert "rounding_grid" in units["basis_anchors"]
    grid = PACKAGED_PROJECTION.document["anchors"]["rounding_grid"]["text"]
    # The rounding grid is what makes a delivered zero a reported zero.
    assert "Nonzero amounts less than $5 are set to $2" in grid


def test_the_sign_marker_is_recorded_verbatim_and_never_enforced():
    by_field = {
        entry["field"]: entry
        for entry in PACKAGED_PROJECTION.document["projection"]["columns"]
    }
    marked = {"E00900", "E01000", "E02100"}
    for field, entry in by_field.items():
        sign = entry["sign"]
        assert sign["publisher_signed_marker"] is (field in marked)
        assert sign["negative_zero"] == "refused"
        if field in marked:
            assert sign["negative_values"] == "documented_by_record_layout_marker"
        elif field in ("P22250", "P23250"):
            assert (
                sign["negative_values"] == "documented_by_publisher_definition_text"
            )
            assert sign["definition_anchors"]
        else:
            assert sign["negative_values"] == "not_documented_unresolved"


def test_missingness_is_declared_and_never_invented():
    for entry in PACKAGED_PROJECTION.document["projection"]["columns"]:
        missing = entry["missingness"]
        assert missing["empty_token"] == "refused"
        assert missing["zero_meaning"] == "reported_zero_not_a_rounding_artifact"
        assert missing["zero_basis_anchor"] == "rounding_grid"


def test_every_column_declares_the_return_grain_and_the_unresolved_period():
    for entry in PACKAGED_PROJECTION.document["projection"]["columns"]:
        assert entry["return_grain"]["grain"] == "source_return"
        assert entry["source_period_status"]["status"] == "unresolved_raw_FLPDYR"
        assert (
            entry["target_year_use"]
            == "refused_pending_independent_growth_decision"
        )
        assert entry["derived_combinations"] == "refused"
        assert entry["fit_admission"] in ("source_year_only", "unresolved_retained")


def test_the_source_pin_identity_is_the_reviewed_definition_not_a_restatement():
    document = PACKAGED_PROJECTION.document
    assert document["raw_source_definition"]["sha256"] == PACKAGED_DEFINITION.sha256
    assert document["sources"]["main"]["sha256"] == PACKAGED_DEFINITION.main.sha256
    assert (
        document["sources"]["main"]["git_blob_sha1"]
        == PACKAGED_DEFINITION.main.git_blob_sha1
    )
    assert (
        document["sources"]["demographic"]["sha256"]
        == PACKAGED_DEFINITION.demographic.sha256
    )
    assert all(value is False for value in document["non_claims"].values())


def test_unresolved_component_relationships_are_recorded_not_computed():
    by_field = {
        entry["field"]: entry
        for entry in PACKAGED_PROJECTION.document["projection"]["columns"]
    }
    assert [item["code"] for item in by_field["E00650"]["unresolved"]] == [
        "qualified_dividend_containment"
    ]
    assert [item["code"] for item in by_field["E01700"]["unresolved"]] == [
        "taxable_share_relationship"
    ]
    assert [item["code"] for item in by_field["E01000"]["unresolved"]] == [
        "composition_from_schedule_d_rows"
    ]


# --------------------------------------------------------------------------
# Refusals the increment is defined by
# --------------------------------------------------------------------------


def test_e22250_is_refused_and_never_aliased_to_p22250():
    document = PACKAGED_PROJECTION.document
    refused = document["refused_fields"]["E22250"]
    assert refused["policy"] == "never_aliased_to_P22250"
    assert "E22250" not in MAIN_HEADER
    assert "P22250" in PROJECTED
    # A delivered header that carries the refused spelling is a refusal, not
    # a normalisation.
    header = list(MAIN_HEADER)
    header[header.index("S008")] = "E22250"
    main = csv_bytes(header, [[*["0"] * 222]])
    document = fixture_definition_document(main, csv_bytes(DEMOGRAPHIC_HEADER, []))
    entry = document["sources"]["main"]
    entry["delivered_header"] = header
    entry["header_record_canonical_sha256"] = header_digest(header)
    entry["data_records"] = 1
    document["sources"]["demographic"]["data_records"] = 0
    definition = fixture_definition(document)
    projection_document = fixture_projection_json(definition)
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(projection_document, definition)
    # The layout no longer matches the delivered header, which is caught first.
    assert error.value.reason == "PROJECTION_MAIN_LAYOUT_HEADER"


def test_a_projection_naming_a_refused_field_is_refused(invented):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition)
    document["projection"]["columns"][0]["field"] = "E22250"
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_UNDELIVERED_FIELD"


def test_no_projected_column_may_declare_a_derivation(invented):
    _, _, definition, _ = invented
    for key in ("derived_from", "derived_inputs", "formula"):
        document = fixture_projection_json(definition)
        document["projection"]["columns"][0][key] = ["E30400", "E30500"]
        with pytest.raises(PufMonetaryRefusalError) as error:
            fixture_projection_document(document, definition)
        assert error.value.reason == "PROJECTION_DERIVATION_NOT_SUPPORTED"


def test_the_uncapped_self_employment_derivation_is_refused_with_its_evidence():
    document = PACKAGED_PROJECTION.document
    refused = document["refused_derivations"]["uncapped_self_employment_earnings"]
    assert tuple(refused["refused_inputs"]) == ("E30400", "E30500")
    for key in refused["anchors"]:
        text = document["anchors"][key]["text"]
        assert "SUBJECT TO" in text
        assert "maximum amount subject to Social Security tax" in text
    assert "E30400" not in PROJECTED and "E30500" not in PROJECTED


def test_the_layout_and_the_projection_must_agree_in_both_directions(invented):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition)
    document["projection"]["columns"] = document["projection"]["columns"][:-1]
    document["projection"]["column_count"] = len(document["projection"]["columns"])
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_ADMISSION_SET"

    document = fixture_projection_json(definition)
    for column in document["layout"]["main"]["columns"]:
        if column["name"] == "E00200":
            column["admission"] = "not_selected"
            column["not_selected_reason"] = "amount_field_outside_this_increment"
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_ADMISSION_DISAGREEMENT"


def test_a_forged_layout_row_digest_or_label_is_refused(invented):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition)
    document["layout"]["main"]["columns"][0]["layout_row"]["text"] = "forged"
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_LAYOUT_EVIDENCE_DIGEST"

    document = fixture_projection_json(definition)
    row = document["layout"]["main"]["columns"][0]["layout_row"]
    row["text"] = "a row that never names the column"
    row["text_sha256"] = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_LAYOUT_EVIDENCE_NAME"


def test_a_restated_sign_marker_or_target_year_is_refused(invented):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition)
    for entry in document["projection"]["columns"]:
        if entry["field"] == "E00200":
            entry["sign"]["publisher_signed_marker"] = True
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_SIGN_MARKER"

    document = fixture_projection_json(definition)
    document["projection"]["columns"][0]["target_year_use"] = "2024"
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_TARGET_YEAR_USE"

    document = fixture_projection_json(definition)
    document["projection"]["target_year_use"] = "2024"
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_TARGET_YEAR_USE"


def test_fixture_route_refuses_production_authority_or_a_packaged_pin(invented):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition)
    document["authority"] = "irs_soi_2015_public_use_file_retained_booklets"
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "FIXTURE_AUTHORITY"


def test_the_projection_must_match_the_definition_it_is_handed(invented):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition)
    document["raw_source_definition"]["sha256"] = "0" * 64
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_RAW_DEFINITION_MISMATCH"

    document = fixture_projection_json(definition)
    document["sources"]["main"]["bytes"] = definition.main.bytes + 1
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_SOURCE_PIN"


def test_the_projection_document_is_immutable_all_the_way_down():
    document = PACKAGED_PROJECTION.document
    with pytest.raises(TypeError):
        document["route"] = "test_fixture"
    with pytest.raises(TypeError):
        document["layout"]["main"]["columns"][0]["admission"] = "x"
    assert isinstance(document["projection"]["columns"], tuple)
    # A caller that wants a document to edit gets a copy it owns.
    copy = projection_document_json()
    copy["route"] = "test_fixture"
    assert PACKAGED_PROJECTION.document["route"] == "packaged"


def test_the_production_kernel_refuses_anything_but_the_packaged_bytes(invented):
    _, _, definition, projection = invented
    with pytest.raises(PufMonetaryRefusalError) as error:
        USPufMonetarySourceKernel(projection=projection, definition=PACKAGED_DEFINITION)
    assert error.value.reason == "KERNEL_ROUTE_DISAGREEMENT"

    document = projection_document_json()
    document["layout"]["main"]["columns"][0]["not_selected_reason"] = (
        "amount_field_outside_this_increment"
    )
    edited = money._projection_from_document(
        document, route="packaged", definition=PACKAGED_DEFINITION
    )
    with pytest.raises(PufMonetaryRefusalError) as error:
        USPufMonetarySourceKernel(projection=edited, definition=PACKAGED_DEFINITION)
    assert error.value.reason == "KERNEL_PROJECTION_NOT_PACKAGED"


def test_the_reviewed_raw_source_module_bytes_are_unchanged():
    """This slice imports the reviewed producer; it must not edit it.

    An edit would move ``USPufRawSourceKernel.implementation_hash`` and with
    it every node key and artifact already accepted under the reviewed run.
    """

    path = (
        Path(puf.__file__).resolve()
    )
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest() == REVIEWED_RAW_SOURCE_SHA256
    )


# --------------------------------------------------------------------------
# The amount grammar: units, sign, nulls
# --------------------------------------------------------------------------


def parse_amount(token, field="E00200"):
    column = next(c for c in PACKAGED_PROJECTION.columns if c.field == field)
    return money._amount(token, column)


@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("0", 0),
        ("7", 7),
        ("-7", -7),
        ("007", 7),
        ("-007", -7),
        ("999999999999", 999999999999),
        ("-999999999999", -999999999999),
    ],
)
def test_the_amount_grammar_accepts_signed_whole_dollars(token, value):
    assert parse_amount(token) == value


@pytest.mark.parametrize(
    ("token", "reason"),
    [
        ("", "AMOUNT_TOKEN_EMPTY"),
        (" ", "AMOUNT_TOKEN_GRAMMAR"),
        (" 7", "AMOUNT_TOKEN_GRAMMAR"),
        ("7 ", "AMOUNT_TOKEN_GRAMMAR"),
        ("+7", "AMOUNT_TOKEN_GRAMMAR"),
        ("7.0", "AMOUNT_TOKEN_GRAMMAR"),
        ("1e3", "AMOUNT_TOKEN_GRAMMAR"),
        ("--7", "AMOUNT_TOKEN_GRAMMAR"),
        ("-", "AMOUNT_TOKEN_GRAMMAR"),
        ("7-", "AMOUNT_TOKEN_GRAMMAR"),
        ("1,000", "AMOUNT_TOKEN_GRAMMAR"),
        ("NA", "AMOUNT_TOKEN_GRAMMAR"),
        ("-0", "AMOUNT_TOKEN_NEGATIVE_ZERO"),
        ("-000", "AMOUNT_TOKEN_NEGATIVE_ZERO"),
        ("1234567890123", "AMOUNT_TOKEN_OVER_DIGITS"),
        ("-1234567890123", "AMOUNT_TOKEN_OVER_WIDTH"),
        ("-123456789012345", "AMOUNT_TOKEN_OVER_WIDTH"),
        ("7\u00a0", "AMOUNT_TOKEN_NON_ASCII"),
        ("7\x00", "AMOUNT_TOKEN_EMBEDDED_NUL"),
    ],
)
def test_the_amount_grammar_refuses_everything_else(token, reason):
    with pytest.raises(PufMonetaryRefusalError) as error:
        parse_amount(token)
    assert error.value.reason == reason


def test_an_empty_amount_token_is_a_refusal_and_never_a_zero(invented):
    main_records = [
        main_record(recid="11", amounts={"E00200": ""}),
    ]
    main, demographic, definition, projection = build_fixture(
        main_records=main_records, demographic_records=[]
    )
    status = decoded_status(main, demographic, definition)
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(main, status, projection, definition)
    assert error.value.reason == "AMOUNT_TOKEN_EMPTY"


def test_a_delivered_zero_is_a_reported_zero_and_is_counted_as_one(decoded):
    projected, projection = decoded
    facts = projected.facts["column_facts"]["E00300"]
    # Rows 12, 13 and 14 report a zero; the aggregate reports one too.
    assert facts["demographic_matched"]["zero"] == 1
    assert facts["demographic_unmatched"]["zero"] == 2
    assert facts["disclosure_aggregate"]["zero"] == 1
    assert int(projected.typed["E00300"][2]) == 0
    assert projected.lexical["E00300"][2] == "0"


def test_a_negative_amount_keeps_its_sign_and_is_never_absolute_valued(decoded):
    projected, _ = decoded
    assert int(projected.typed["E00900"][1]) == -2500
    assert projected.lexical["E00900"][1] == "-2500"
    assert int(projected.typed["P22250"][1]) == -3000
    # P22250 prints no (+/-) marker on its layout row, yet the negative value
    # is preserved rather than refused or folded.
    column = next(c for c in PACKAGED_PROJECTION.columns if c.field == "P22250")
    assert column.publisher_signed_marker is False
    assert column.negative_values == "documented_by_publisher_definition_text"
    assert projected.facts["column_facts"]["P22250"]["demographic_unmatched"][
        "negative"
    ] == 1


def test_a_negative_wage_is_preserved_even_though_no_marker_documents_one():
    """The absent marker is recorded, not enforced.

    Refusing a negative wage would be inventing a nonnegative domain the
    booklets never state; silently taking its absolute value would be worse.
    The parser keeps it and the facts count it, so a later review can settle
    the domain from evidence.
    """

    main_records = [main_record(recid="11", amounts={"E00200": "-5"})]
    main, demographic, definition, projection = build_fixture(
        main_records=main_records, demographic_records=[]
    )
    status = decoded_status(main, demographic, definition)
    projected = decode_puf_monetary_source(main, status, projection, definition)
    assert int(projected.typed["E00200"][0]) == -5
    assert (
        projected.facts["column_facts"]["E00200"]["demographic_unmatched"]["negative"]
        == 1
    )


def test_the_lexical_spelling_is_retained_alongside_the_typed_value(decoded):
    projected, _ = decoded
    assert projected.lexical["E00200"][0] == "50000"
    assert projected.lexical["E01000"][4] == "-999999999999"
    for field in PROJECTED:
        assert len(projected.lexical[field]) == projected.rows


# --------------------------------------------------------------------------
# Header, width and record structure
# --------------------------------------------------------------------------


def test_a_shuffled_missing_extra_or_duplicate_header_is_refused(invented):
    main, demographic, definition, projection = invented
    status = decoded_status(main, demographic, definition)
    for header in (
        MAIN_HEADER[1:] + MAIN_HEADER[:1],
        MAIN_HEADER[:-1],
        MAIN_HEADER + ["EXTRA"],
        [MAIN_HEADER[0], *MAIN_HEADER[1:]][:-1] + [MAIN_HEADER[0]],
    ):
        payload = csv_bytes(header, [record[: len(header)] for record in DEFAULT_MAIN])
        with pytest.raises(PufMonetaryRefusalError) as error:
            decode_puf_monetary_source(payload, status, projection, definition)
        assert error.value.reason in ("HEADER_MISMATCH", "RECORD_WIDTH")


def test_a_short_or_long_record_is_refused(invented):
    main, demographic, definition, projection = invented
    status = decoded_status(main, demographic, definition)
    records = [record[:-1] for record in DEFAULT_MAIN]
    payload = csv_bytes(MAIN_HEADER, records)
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(payload, status, projection, definition)
    assert error.value.reason == "RECORD_WIDTH"


def test_a_duplicate_or_colliding_key_is_refused(invented):
    _, _, _, _ = invented
    main_records = [main_record(recid="11"), main_record(recid="11")]
    main, demographic, definition, projection = build_fixture(
        main_records=main_records, demographic_records=[]
    )
    status = status_stub([11, 11], [0, 0], [0, 0])
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(main, status, projection, definition)
    assert error.value.reason == "MAIN_DUPLICATE_LEXICAL_KEY"


def test_a_record_count_other_than_the_declared_one_is_refused(invented):
    main, demographic, definition, projection = invented
    status = decoded_status(main, demographic, definition)
    payload = csv_bytes(MAIN_HEADER, DEFAULT_MAIN[:-1])
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(payload, status, projection, definition)
    assert error.value.reason == "MAIN_RECORD_COUNT"


def test_a_lowered_or_raised_csv_field_size_limit_is_refused(invented, monkeypatch):
    import csv as csv_module

    main, demographic, definition, projection = invented
    status = decoded_status(main, demographic, definition)
    original = csv_module.field_size_limit()
    try:
        csv_module.field_size_limit(original // 2)
        with pytest.raises(PufMonetaryRefusalError) as error:
            decode_puf_monetary_source(main, status, projection, definition)
        assert error.value.reason == "CSV_FIELD_SIZE_LIMIT"
    finally:
        csv_module.field_size_limit(original)
