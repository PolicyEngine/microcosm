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
    decode_puf_raw_source,
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

#: The raw producer's resource location was corrected after the monetary slice:
#: runtime source metadata must not enter the country-spec closure. This is a
#: new implementation identity; accepted artifacts at 2fd9a5cb retain their
#: old identity. The raw definition bytes and decoder semantics are unchanged.
REVIEWED_RAW_SOURCE_SHA256 = (
    "c0e31e335dd38a4c767b5cf83ffb09c05576b038608fdc78213e32ee0a65ecf6"
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
#: whose delivered amounts are zeros, a second unmatched one, and one publisher
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
    tolerant = {
        entry["name"] for entry in main["corroborating_edition_tolerant_positions"]
    }
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
    # The rounding rule documents a disclosure edit, not taxpayer reporting.
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
            assert sign["negative_values"] == "documented_by_publisher_definition_text"
            assert sign["definition_anchors"]
        else:
            assert sign["negative_values"] == "not_documented_unresolved"


def test_missingness_is_declared_and_never_invented():
    for entry in PACKAGED_PROJECTION.document["projection"]["columns"]:
        missing = entry["missingness"]
        assert missing["empty_token"] == "refused"
        assert missing["zero_meaning"] == "delivered_zero_in_edited_public_file"
        assert missing["zero_basis_anchor"] == "rounding_grid"


def test_every_column_declares_the_return_grain_and_the_unresolved_period():
    for entry in PACKAGED_PROJECTION.document["projection"]["columns"]:
        assert entry["return_grain"]["grain"] == "source_return"
        assert entry["source_period_status"]["status"] == "unresolved_raw_FLPDYR"
        assert entry["target_year_use"] == "refused_pending_independent_growth_decision"
        assert entry["derived_combinations"] == "refused"
        assert entry["fit_admission"] in (
            "not_admitted_pending_basis_review",
            "unresolved_retained",
        )


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


def test_the_raw_source_module_matches_the_reviewed_packaging_identity():
    """Pin the raw producer's reviewed resource-loader revision.

    This identity includes the relocation into ``us_runtime``. Earlier raw
    runs retain their own implementation identity and accepted artifacts.
    """

    path = Path(puf.__file__).resolve()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == REVIEWED_RAW_SOURCE_SHA256


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
    assert (
        projected.facts["column_facts"]["P22250"]["demographic_unmatched"]["negative"]
        == 1
    )


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


# --------------------------------------------------------------------------
# Retention, the join binding and the disclosure aggregates
# --------------------------------------------------------------------------


def test_every_delivered_record_is_projected_matched_or_not(decoded):
    projected, _ = decoded
    assert projected.rows == len(DEFAULT_MAIN)
    assert projected.facts["main_records"] == len(DEFAULT_MAIN)
    assert list(projected.typed["RECID"]) == [11, 12, 13, 14, PUF_AGGREGATE_RECIDS[0]]
    assert projected.facts["demographic_matched_rows"] == 2
    assert projected.facts["demographic_unmatched_rows"] == 3


def test_the_row_classes_partition_the_rows_exactly_once(decoded):
    projected, _ = decoded
    classes = projected.facts["row_class_rows"]
    assert set(classes) == set(ROW_CLASSES)
    assert sum(classes.values()) == projected.rows
    assert classes["disclosure_aggregate"] == 1
    assert classes["demographic_matched"] == 2
    assert classes["demographic_unmatched"] == 2


def test_an_aggregate_is_never_counted_in_a_per_return_class(decoded):
    projected, _ = decoded
    facts = projected.facts["column_facts"]["E00200"]
    assert facts["disclosure_aggregate"]["rows"] == 1
    assert facts["disclosure_aggregate"]["maximum"] == 999999999999
    # The 12-digit aggregate never reaches either per-return class.
    assert facts["demographic_matched"]["maximum"] == 50000
    assert facts["demographic_unmatched"]["maximum"] == 31000
    assert facts["demographic_matched"]["sum"] == 50000
    assert facts["demographic_unmatched"]["sum"] == 31077


def test_an_aggregate_with_a_demographic_row_is_still_an_aggregate():
    """The reviewed slice tolerates a demographic row keyed to an aggregate.

    The classes must still put it under ``disclosure_aggregate``, so it can
    never be averaged into a per-return statistic.
    """

    recid = str(PUF_AGGREGATE_RECIDS[0])
    main_records = [
        main_record(recid="11"),
        main_record(recid=recid, mars="0", s006="0", amounts={"E00200": "5"}),
    ]
    main, demographic, definition, projection = build_fixture(
        main_records=main_records,
        demographic_records=[demographic_record(recid=recid)],
    )
    status = decoded_status(main, demographic, definition)
    projected = decode_puf_monetary_source(main, status, projection, definition)
    assert projected.facts["demographic_matched_rows"] == 1
    assert projected.facts["row_class_rows"] == {
        "disclosure_aggregate": 1,
        "demographic_matched": 0,
        "demographic_unmatched": 1,
    }


def test_an_unmatched_row_is_absence_not_a_zeroed_amount(decoded):
    projected, _ = decoded
    unmatched = projected.typed["demographic_status"] == 0
    # Row 12 has no demographic record and a nonzero wage; absence of a
    # demographic row says nothing about the amount and zeroes nothing.
    assert bool(unmatched[1])
    assert int(projected.typed["E00200"][1]) == 31000
    assert (
        projected.facts["column_facts"]["E00200"]["demographic_unmatched"]["rows"] == 2
    )


def test_a_status_artifact_with_a_different_recid_sequence_is_refused(invented):
    main, demographic, definition, projection = invented
    reordered = status_stub(
        [12, 11, 13, 14, PUF_AGGREGATE_RECIDS[0]], [0, 0, 0, 0, 1], [0, 1, 1, 0, 0]
    )
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(main, reordered, projection, definition)
    assert error.value.reason == "STATUS_RECID_SEQUENCE"


def test_a_status_artifact_with_different_aggregate_flags_is_refused(invented):
    main, demographic, definition, projection = invented
    forged = status_stub(
        [11, 12, 13, 14, PUF_AGGREGATE_RECIDS[0]], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0]
    )
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(main, forged, projection, definition)
    assert error.value.reason == "STATUS_AGGREGATE_FLAGS"


def test_a_status_artifact_of_the_wrong_length_or_shape_is_refused(invented):
    main, demographic, definition, projection = invented
    short = status_stub([11, 12], [0, 0], [1, 0])
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(main, short, projection, definition)
    assert error.value.reason == "STATUS_ROW_COUNT"

    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_puf_monetary_source(main, object(), projection, definition)
    assert error.value.reason == "STATUS_ARTIFACT_TYPE"


def test_the_carried_demographic_status_is_the_reviewed_artifacts_own(decoded):
    projected, _ = decoded
    assert list(projected.typed["demographic_status"]) == [1, 0, 1, 0, 0]


# --------------------------------------------------------------------------
# The envelope
# --------------------------------------------------------------------------


@pytest.fixture
def envelope(invented):
    main, demographic, definition, projection = invented
    status = decoded_status(main, demographic, definition)
    projected = decode_puf_monetary_source(main, status, projection, definition)
    payload = encode_monetary_projection(
        projected,
        projection,
        definition,
        status_artifact_sha256=hashlib.sha256(
            encode_return_status(status, definition)
        ).hexdigest(),
    )
    return payload, projection, projected


def test_the_envelope_round_trips_every_column_exactly(envelope):
    payload, projection, projected = envelope
    back = decode_monetary_projection(payload, projection)
    assert back.rows == projected.rows
    for name in ("RECID", "disclosure_aggregate", "demographic_status", *PROJECTED):
        assert np.array_equal(back.typed[name], projected.typed[name])
    for field in PROJECTED:
        assert back.lexical[field] == projected.lexical[field]
    assert canonical_json(dict(back.facts)) == canonical_json(dict(projected.facts))


def test_fitting_refusals_survive_readback_and_cannot_be_promoted(envelope):
    payload, projection, projected = envelope
    back = decode_monetary_projection(payload, projection)
    assert back.column_metadata == projected.column_metadata
    for column in projection.columns:
        metadata = back.column_metadata[column.field]
        assert metadata["fit_admission"] == "not_admitted_pending_basis_review"
        assert metadata["unresolved"] == column.unresolved
        assert metadata["period_semantics"] == "unresolved_raw_FLPDYR"
        with pytest.raises(TypeError):
            metadata["fit_admission"] = "source_year_only"

    def forge_admission(header):
        header["facts"]["column_metadata"]["E00200"]["fit_admission"] = (
            "source_year_only"
        )

    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(
            reheader(payload, projection, forge_admission), projection
        )
    assert error.value.reason == "ENVELOPE_FACTS_INCONSISTENT"


def test_readback_can_require_the_actual_status_artifact_digest(envelope, invented):
    payload, projection, _ = envelope
    main, demographic, definition, _ = invented
    digest = hashlib.sha256(
        encode_return_status(decoded_status(main, demographic, definition), definition)
    ).hexdigest()
    decode_monetary_projection(
        payload, projection, expected_status_artifact_sha256=digest
    )
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(
            payload, projection, expected_status_artifact_sha256="0" * 64
        )
    assert error.value.reason == "ENVELOPE_STATUS_ARTIFACT_MISMATCH"


def test_the_envelope_declares_its_type_endianness_and_pins(envelope):
    payload, projection, _ = envelope
    start = len(money.MONETARY_PROJECTION_MAGIC)
    length = int.from_bytes(payload[start : start + 4], "big")
    header = json.loads(payload[start + 4 : start + 4 + length].decode("utf-8"))
    assert header["type"] == [
        MONETARY_PROJECTION_TYPE.name,
        MONETARY_PROJECTION_TYPE.schema_version,
    ]
    assert header["header_length_endianness"] == "big"
    assert header["body_endianness"] == "little"
    assert header["projection_sha256"] == projection.sha256
    assert header["raw_source_definition_sha256"] == projection.raw_definition_sha256
    assert header["facts"]["target_year_use"] == (
        "refused_pending_independent_growth_decision"
    )
    assert header["facts"]["period_semantics"] == "unresolved_raw_FLPDYR"
    assert header["facts"]["amounts_decoded"] is True
    assert header["facts"]["amount_units"] == "whole_usd_source_year"
    for column in header["columns"]:
        if column["kind"] == "typed":
            assert column["dtype"].startswith("<")


def reheader(payload: bytes, projection, mutate):
    start = len(money.MONETARY_PROJECTION_MAGIC)
    length = int.from_bytes(payload[start : start + 4], "big")
    header = json.loads(payload[start + 4 : start + 4 + length].decode("utf-8"))
    bodies = payload[start + 4 + length :]
    mutate(header)
    encoded_header = canonical_json(header)
    return b"".join(
        (
            money.MONETARY_PROJECTION_MAGIC,
            len(encoded_header).to_bytes(4, "big"),
            encoded_header,
            bodies,
        )
    )


def test_a_header_reporting_facts_the_arrays_do_not_carry_is_refused(envelope):
    payload, projection, _ = envelope

    def forge_sum(header):
        header["facts"]["column_facts"]["E00200"]["demographic_matched"]["sum"] = 1

    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(reheader(payload, projection, forge_sum), projection)
    assert error.value.reason == "ENVELOPE_FACTS_INCONSISTENT"

    def forge_negative(header):
        header["facts"]["column_facts"]["P22250"]["demographic_unmatched"][
            "negative"
        ] = 0

    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(
            reheader(payload, projection, forge_negative), projection
        )
    assert error.value.reason == "ENVELOPE_FACTS_INCONSISTENT"

    def forge_classes(header):
        header["facts"]["row_class_rows"]["disclosure_aggregate"] = 0

    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(
            reheader(payload, projection, forge_classes), projection
        )
    assert error.value.reason == "ENVELOPE_FACTS_INCONSISTENT"


def test_a_restated_dtype_kind_or_width_is_refused(envelope):
    payload, projection, _ = envelope

    def as_float(header):
        for column in header["columns"]:
            if column["name"] == "E00200":
                column["dtype"] = "<f8"

    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(reheader(payload, projection, as_float), projection)
    assert error.value.reason == "ENVELOPE_COLUMN_DTYPE"

    def as_typed(header):
        for column in header["columns"]:
            if column["name"] == "E00200_lexical":
                column["kind"] = "typed"

    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(reheader(payload, projection, as_typed), projection)
    assert error.value.reason == "ENVELOPE_COLUMN_KIND"

    def widen(header):
        for column in header["columns"]:
            if column["name"] == "E00200_lexical":
                column["width"] = 14

    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(reheader(payload, projection, widen), projection)
    assert error.value.reason == "ENVELOPE_COLUMN_WIDTH"


def test_a_truncated_or_tampered_envelope_is_refused(envelope):
    payload, projection, _ = envelope
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(payload[:-1], projection)
    assert error.value.reason == "ENVELOPE_BODY_TRUNCATED"
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(payload + b"\0", projection)
    assert error.value.reason == "ENVELOPE_TRAILING_BYTES"
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(b"not-an-envelope", projection)
    assert error.value.reason == "ENVELOPE_MAGIC"
    flipped = bytearray(payload)
    flipped[-1] ^= 0xFF
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(bytes(flipped), projection)
    assert error.value.reason == "ENVELOPE_BODY_DIGEST"


def test_a_payload_from_another_projection_is_refused(envelope):
    payload, projection, _ = envelope
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(payload, PACKAGED_PROJECTION)
    assert error.value.reason == "ENVELOPE_PROJECTION_MISMATCH"
    with pytest.raises(PufMonetaryRefusalError) as error:
        decode_monetary_projection(payload)
    assert error.value.reason == "ENVELOPE_PROJECTION_REQUIRED"


def test_the_full_scale_body_bound_is_derived_and_stays_under_the_cap():
    rows = PACKAGED_DEFINITION.main.data_records
    total, sizes = payload_bound(rows, PACKAGED_PROJECTION)
    per_row = 8 + 1 + 1 + len(PROJECTED) * (8 + 13)
    assert total == rows * per_row
    assert total == 54_416_352
    assert total <= BODY_MAX_BYTES
    assert sizes["E00200_lexical"] == rows * 13


def test_the_full_scale_header_stays_under_its_bound(envelope):
    """The header carries per-column facts for three classes; bound it.

    The fixture's numbers are tiny, so the check widens every integer to a
    twelve-digit magnitude and re-measures — the widest a delivered amount
    field can be — rather than declaring a bound from four invented rows.
    """

    payload, projection, _ = envelope
    start = len(money.MONETARY_PROJECTION_MAGIC)
    length = int.from_bytes(payload[start : start + 4], "big")
    header = json.loads(payload[start + 4 : start + 4 + length].decode("utf-8"))
    header["rows"] = PACKAGED_DEFINITION.main.data_records
    for classes in header["facts"]["column_facts"].values():
        for entry in classes.values():
            for key in entry:
                entry[key] = -999_999_999_999 * 207_696 if key == "sum" else 207_696
    assert len(canonical_json(header)) < HEADER_MAX_BYTES


# --------------------------------------------------------------------------
# The graph: cold, warm and fresh
# --------------------------------------------------------------------------


def write(path: Path, payload: bytes) -> Path:
    path.write_bytes(payload)
    return path


def write_host_source(directory: Path) -> Path:
    """One invented household of engineering context, labelled as invented.

    It creates no PUF person and establishes no return-to-household mapping;
    it exists only so the artifact nodes have an explicitly named host.
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

    ref = "us.puf.monetary_source.fixture_host@1"
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


def build_graph(definition, projection, *, producer=None):
    node = (
        us_puf_monetary_source_node(
            population=HOST_POPULATION,
            definition=definition,
            projection=projection,
        )
        if producer is None
        else producer
    )
    nodes = (
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
        us_puf_raw_source_node(population=HOST_POPULATION, definition=definition),
        node,
        us_puf_monetary_projection_summary_node(
            population=HOST_POPULATION, producer=node.id
        ),
    )
    graph = Graph(
        country="us",
        sources=(
            SourceRef(HOST_SOURCE, "csv-tables"),
            SourceRef(definition.main.source_name, definition.main.codec),
            SourceRef(definition.demographic.source_name, definition.demographic.codec),
        ),
        nodes=nodes,
    )
    registry = KernelRegistry()
    registry.register(FixtureHostKernel())
    register_us_puf_raw_source_kernels(
        registry,
        definition=definition,
        source_codecs=puf_raw_source_codecs(definition),
    )
    register_us_puf_monetary_source_kernels(
        registry,
        projection=projection,
        definition=definition,
        source_codecs=puf_raw_source_codecs(definition),
    )
    return compile_graph(graph), registry


def composed_codecs(definition):
    registry = SourceCodecRegistry()
    registry.register("csv-tables", load_csv_tables)
    puf.register_us_puf_raw_source_codecs(registry, definition=definition)
    return registry


def composed_store(root: Path, definition) -> ContentStore:
    return ContentStore(root, codecs=composed_codecs(definition))


def artifact_bytes(manifest, store, node_id=MONETARY_PROJECTION_NODE):
    key = manifest.nodes[node_id].opaque_artifacts["monetary_projection"]
    return store.load_bytes(key)


@pytest.fixture
def graph_fixture(invented, tmp_path):
    main, demographic, definition, projection = invented
    sources = {
        HOST_SOURCE: write_host_source(tmp_path / "host"),
        definition.main.source_name: write(tmp_path / "main.csv", main),
        definition.demographic.source_name: write(
            tmp_path / "demographic.csv", demographic
        ),
    }
    compiled, registry = build_graph(definition, projection)
    return compiled, registry, sources, definition, projection, (main, demographic)


def test_cold_run_stores_an_artifact_byte_identical_to_the_direct_decoder(
    graph_fixture, tmp_path
):
    compiled, registry, sources, definition, projection, (main, demographic) = (
        graph_fixture
    )
    store = composed_store(tmp_path / "store", definition)
    manifest = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    payload = artifact_bytes(manifest, store)
    status = decode_puf_raw_source(main, demographic, definition)
    direct = encode_monetary_projection(
        decode_puf_monetary_source(main, status, projection, definition),
        projection,
        definition,
        status_artifact_sha256=hashlib.sha256(
            encode_return_status(status, definition)
        ).hexdigest(),
    )
    assert payload == direct
    back = decode_monetary_projection(payload, projection)
    assert back.rows == len(DEFAULT_MAIN)
    summary = manifest.nodes[MONETARY_PROJECTION_SUMMARY_NODE].receipt[
        "puf_monetary_projection_summary"
    ]
    assert summary["rows"] == len(DEFAULT_MAIN)
    assert summary["projected_columns"] == len(PROJECTED)
    assert summary["amount_units"] == "whole_usd_source_year"
    assert summary["period_semantics"] == "unresolved_raw_FLPDYR"
    assert summary["target_year_use"] == ("refused_pending_independent_growth_decision")
    assert summary["negative_values_by_field"]["E00900"] == 1
    assert summary["negative_values_by_field"]["E01000"] == 2
    producer = manifest.nodes[MONETARY_PROJECTION_NODE].receipt[
        "puf_monetary_source_projection"
    ]
    assert producer["projection_sha256"] == projection.sha256
    assert producer["artifact_sha256"] == hashlib.sha256(payload).hexdigest()
    assert list(producer["projected_fields"]) == PROJECTED
    assert producer["target_year_use"] == (
        "refused_pending_independent_growth_decision"
    )
    assert producer["column_metadata"] == summary["column_metadata"]
    assert producer["column_metadata"] == back.column_metadata
    # Receipts carry aggregates and digests only, never a key or a token.
    text = json.dumps(
        {
            node: manifest.nodes[node].receipt
            for node in (MONETARY_PROJECTION_NODE, MONETARY_PROJECTION_SUMMARY_NODE)
        },
        sort_keys=True,
        default=dict,
    )
    for token in ('"50000"', '"-2500"', '"999999999999"', '"11"'):
        assert token not in text


def test_warm_require_hits_every_node_and_never_calls_the_kernel(
    graph_fixture, tmp_path
):
    compiled, registry, sources, definition, projection, _ = graph_fixture
    store = composed_store(tmp_path / "store", definition)
    cold = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    cold_payload = artifact_bytes(cold, store)

    calls = []
    original = money.USPufMonetarySourceKernel.run

    def counted(self, context):
        calls.append(context.node.id)
        return original(self, context)

    money.USPufMonetarySourceKernel.run = counted
    try:
        warm = run_graph(
            compiled, sources=sources, store=store, kernels=registry, resume="require"
        )
    finally:
        money.USPufMonetarySourceKernel.run = original
    assert calls == []
    assert artifact_bytes(warm, store) == cold_payload
    assert (
        warm.nodes[MONETARY_PROJECTION_NODE].key
        == cold.nodes[MONETARY_PROJECTION_NODE].key
    )


def test_a_fresh_store_recomputes_the_identical_artifact_bytes(graph_fixture, tmp_path):
    compiled, registry, sources, definition, projection, _ = graph_fixture
    first = composed_store(tmp_path / "one", definition)
    cold = run_graph(
        compiled, sources=sources, store=first, kernels=registry, resume="forbid"
    )
    second = composed_store(tmp_path / "two", definition)
    with pytest.raises(StoreMissError):
        run_graph(
            compiled, sources=sources, store=second, kernels=registry, resume="require"
        )
    again = run_graph(
        compiled, sources=sources, store=second, kernels=registry, resume="auto"
    )
    assert artifact_bytes(again, second) == artifact_bytes(cold, first)
    assert (
        again.nodes[MONETARY_PROJECTION_NODE].key
        == cold.nodes[MONETARY_PROJECTION_NODE].key
    )


def test_a_changed_projection_parameter_changes_the_node_key(graph_fixture, tmp_path):
    compiled, registry, sources, definition, projection, _ = graph_fixture
    store = composed_store(tmp_path / "store", definition)
    cold = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )
    document = fixture_projection_json(definition)
    # A purely documentary edit still moves the key: the projection travels
    # verbatim as a node parameter.
    document["projection"]["columns"][0]["notes"] = ["an added note"]
    edited = fixture_projection_document(document, definition)
    assert edited.sha256 != projection.sha256
    other, other_registry = build_graph(definition, edited)
    with pytest.raises(StoreMissError):
        run_graph(
            other,
            sources=sources,
            store=store,
            kernels=other_registry,
            resume="require",
        )
    warm = run_graph(
        other, sources=sources, store=store, kernels=other_registry, resume="auto"
    )
    assert (
        warm.nodes[MONETARY_PROJECTION_NODE].key
        != cold.nodes[MONETARY_PROJECTION_NODE].key
    )


def test_changed_source_bytes_change_the_node_key_and_recompute(tmp_path):
    records = list(DEFAULT_MAIN)
    main, demographic, definition, projection = build_fixture(main_records=records)
    sources = {
        HOST_SOURCE: write_host_source(tmp_path / "host"),
        definition.main.source_name: write(tmp_path / "main.csv", main),
        definition.demographic.source_name: write(
            tmp_path / "demographic.csv", demographic
        ),
    }
    compiled, registry = build_graph(definition, projection)
    store = composed_store(tmp_path / "store", definition)
    cold = run_graph(
        compiled, sources=sources, store=store, kernels=registry, resume="forbid"
    )

    changed = list(DEFAULT_MAIN)
    changed[0] = main_record(recid="11", amounts={"E00200": "50001"})
    other_main, other_demographic, other_definition, other_projection = build_fixture(
        main_records=changed
    )
    other_sources = {
        HOST_SOURCE: sources[HOST_SOURCE],
        other_definition.main.source_name: write(
            tmp_path / "main-changed.csv", other_main
        ),
        other_definition.demographic.source_name: write(
            tmp_path / "demographic-changed.csv", other_demographic
        ),
    }
    other_compiled, other_registry = build_graph(other_definition, other_projection)
    other_store = composed_store(tmp_path / "store", other_definition)
    warm = run_graph(
        other_compiled,
        sources=other_sources,
        store=other_store,
        kernels=other_registry,
        resume="auto",
    )
    assert (
        warm.nodes[MONETARY_PROJECTION_NODE].key
        != cold.nodes[MONETARY_PROJECTION_NODE].key
    )
    back = decode_monetary_projection(
        artifact_bytes(warm, other_store), other_projection
    )
    assert int(back.typed["E00200"][0]) == 50001


def test_a_malformed_amount_refuses_cleanly_with_no_partial_artifact(tmp_path):
    records = list(DEFAULT_MAIN)
    records[1] = main_record(recid="12", mars="2", amounts={"E00200": "31000.5"})
    main, demographic, definition, projection = build_fixture(main_records=records)
    sources = {
        HOST_SOURCE: write_host_source(tmp_path / "host"),
        definition.main.source_name: write(tmp_path / "main.csv", main),
        definition.demographic.source_name: write(
            tmp_path / "demographic.csv", demographic
        ),
    }
    compiled, registry = build_graph(definition, projection)
    store = composed_store(tmp_path / "store", definition)
    with pytest.raises(Exception) as error:
        run_graph(
            compiled, sources=sources, store=store, kernels=registry, resume="forbid"
        )
    assert "AMOUNT_TOKEN_GRAMMAR" in str(error.value)


def test_the_producer_node_requires_an_explicit_population(invented):
    _, _, definition, projection = invented
    with pytest.raises(PufMonetaryRefusalError) as error:
        us_puf_monetary_source_node(
            population="", definition=definition, projection=projection
        )
    assert error.value.reason == "NODE_POPULATION_REQUIRED"
    with pytest.raises(PufMonetaryRefusalError) as error:
        us_puf_monetary_projection_summary_node(population="")
    assert error.value.reason == "NODE_POPULATION_REQUIRED"


def test_the_node_factory_refuses_a_projection_for_another_definition(invented):
    _, _, definition, _ = invented
    with pytest.raises(PufMonetaryRefusalError) as error:
        us_puf_monetary_source_node(
            population=HOST_POPULATION,
            definition=definition,
            projection=PACKAGED_PROJECTION,
        )
    assert error.value.reason == "NODE_DEFINITION_MISMATCH"


def test_the_kernel_refuses_any_node_shape_but_its_own(graph_fixture, tmp_path):
    compiled, registry, sources, definition, projection, _ = graph_fixture
    store = composed_store(tmp_path / "store", definition)
    honest = us_puf_monetary_source_node(
        population=HOST_POPULATION, definition=definition, projection=projection
    )
    forged = Node(
        honest.id,
        honest.kernel,
        population=HOST_POPULATION,
        sources=honest.sources,
        params=dict(honest.params),
        artifact_inputs=honest.artifact_inputs,
        artifact_outputs=(
            ArtifactOutput("monetary_projection", MONETARY_PROJECTION_TYPE),
            ArtifactOutput("extra", MONETARY_PROJECTION_TYPE),
        ),
    )
    other, other_registry = build_graph(definition, projection, producer=forged)
    with pytest.raises(Exception) as error:
        run_graph(
            other,
            sources=sources,
            store=store,
            kernels=other_registry,
            resume="forbid",
        )
    assert "NODE_DECLARATION" in str(error.value) or "artifact" in str(error.value)


def test_a_changed_dependency_version_changes_the_implementation_hash(graph_fixture):
    _, registry, _, definition, projection, _ = graph_fixture
    kernel = money.USPufMonetarySourceKernel(
        projection=projection, definition=definition
    )
    before = kernel.implementation_hash()
    original = kernel.capabilities
    kernel.capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        numeric=Numeric.BITWISE,
        dependencies=("numpy", "pandas"),
    )
    try:
        assert kernel.implementation_hash() != before
    finally:
        kernel.capabilities = original
    assert kernel.implementation_hash() == before


# --------------------------------------------------------------------------
# The packaged document's own provenance, and the unresolved-retained path
# --------------------------------------------------------------------------


def test_the_packaged_projection_is_importable_from_the_installed_package():
    from importlib import resources

    payload = (
        resources.files("microcosm.build.us_runtime")
        .joinpath("puf_2015_monetary_source_projection.json")
        .read_bytes()
    )
    assert payload.endswith(b"\n")
    document = json.loads(payload.decode("utf-8"))
    assert canonical_json(document) + b"\n" == payload
    assert hashlib.sha256(canonical_json(document)).hexdigest() == (
        PACKAGED_PROJECTION.sha256
    )


def test_the_packaged_document_names_the_generator_that_produced_it():
    generation = PACKAGED_PROJECTION.document["generation"]
    assert generation["reads_microdata"] is False
    tool = Path(__file__).resolve().parents[3] / generation["generator"]
    if not tool.exists():  # installed wheel: the repo tool is not shipped
        pytest.skip("generator source is not present in this layout")
    assert (
        hashlib.sha256(tool.read_bytes()).hexdigest() == generation["generator_sha256"]
    )


def test_every_declared_dtype_matches_the_dtype_the_envelope_writes(envelope):
    payload, projection, _ = envelope
    start = len(money.MONETARY_PROJECTION_MAGIC)
    length = int.from_bytes(payload[start : start + 4], "big")
    header = json.loads(payload[start + 4 : start + 4 + length].decode("utf-8"))
    written = {
        column["name"]: column["dtype"]
        for column in header["columns"]
        if column["kind"] == "typed"
    }
    for column in projection.columns:
        assert column.dtype == "int64"
        assert written[column.field] == "<i8"
    assert written["RECID"] == "<i8"
    assert written["disclosure_aggregate"] == "<u1"
    assert written["demographic_status"] == "<i1"


def test_a_column_whose_semantics_are_unresolved_is_retained_not_dropped():
    """The mechanism the contract needs, exercised on an invented declaration.

    None of the twelve packaged columns is in this state today: each has a
    verbatim publisher label at a numbered layout position. The path exists so
    that a future selection whose primary semantics the booklets do not settle
    is decoded and kept, marked ``unresolved_retained`` and refused as a fit
    input — rather than invented, or used as a reason to halt the whole
    projection.
    """

    main, demographic, definition, _ = build_fixture()
    document = fixture_projection_json(definition)
    for entry in document["projection"]["columns"]:
        if entry["field"] == "E00400":
            entry["fit_admission"] = "unresolved_retained"
            entry["unresolved"].append(
                {
                    "code": "invented_fixture_unresolved_semantics",
                    "statement": "An invented fixture state, not a publisher claim.",
                }
            )
    projection = fixture_projection_document(document, definition)
    column = next(c for c in projection.columns if c.field == "E00400")
    assert column.fit_admission == "unresolved_retained"
    assert "invented_fixture_unresolved_semantics" in column.unresolved
    status = decoded_status(main, demographic, definition)
    projected = decode_puf_monetary_source(main, status, projection, definition)
    # Retained, decoded and counted like any other column.
    assert int(projected.typed["E00400"][0]) == 40
    assert projected.facts["projected_columns"] == len(PROJECTED)
    # And still refused for a target year.
    assert projected.facts["target_year_use"] == (
        "refused_pending_independent_growth_decision"
    )


@pytest.mark.parametrize("fit_admission", ["target_year_ready", "source_year_only"])
def test_an_unknown_fit_admission_or_admission_value_is_refused(
    invented, fit_admission
):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition)
    document["projection"]["columns"][0]["fit_admission"] = fit_admission
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_FIT_ADMISSION"

    document = fixture_projection_json(definition)
    document["layout"]["main"]["columns"][0]["admission"] = "grown"
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_LAYOUT_ADMISSION"


def test_a_non_claim_flipped_to_true_is_refused(invented):
    _, _, definition, _ = invented
    document = fixture_projection_json(definition)
    document["non_claims"]["amount_growth_or_restatement"] = True
    with pytest.raises(PufMonetaryRefusalError) as error:
        fixture_projection_document(document, definition)
    assert error.value.reason == "PROJECTION_NON_CLAIMS"


def test_the_packaged_document_is_a_runtime_resource_not_a_country_spec():
    """Placement is load-bearing, so assert it rather than assume it.

    ``microcosm/build/us`` is the US **country package**: its
    ``country_package.json`` declares every file it ships, and
    ``spec_engine.load_bundle`` refuses the whole package when an undeclared
    file appears beside the manifest. This document is a runtime source
    adapter's resource, exactly like ``acs_2024_1yr_sources.json``, so it
    lives in ``us_runtime`` and never enters that closure.
    """

    build_root = Path(money.__file__).resolve().parents[1]
    country_package = build_root / "us"
    assert (country_package / "country_package.json").exists()
    assert not (country_package / "puf_2015_monetary_source_projection.json").exists()
    resource = build_root / "us_runtime" / "puf_2015_monetary_source_projection.json"
    assert resource.exists()
    assert (build_root / "us_runtime" / "acs_2024_1yr_sources.json").exists()
    declared = json.loads(
        (country_package / "country_package.json").read_text(encoding="utf-8")
    )["resources"]
    listed = {row["path"] if isinstance(row, dict) else row for row in declared}
    assert "puf_2015_monetary_source_projection.json" not in listed
