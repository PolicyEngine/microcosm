"""Typed money body/selection identity, slicing and adverse-input refusals."""

import struct
from hashlib import sha256

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import asec_current_money as money
from microcosm.build.us_runtime import asec_current_money_resources as loader
from microcosm.build.us_runtime._asec_current_money_codec import (
    current_money_content_sha256,
    encode_current_money,
)
from microcosm.build.us_runtime.asec_current_money import (
    AsecMoneyScope,
    AsecMoneyViews,
    MoneyRefusalError,
    ReadyCurrentMoney,
    SyntheticAsecSource,
    classify_asec_money,
    compile_asec_current_money_spec,
    require_complete_current_money,
    restate_asec_current_money,
)
from microcosm.build.us_runtime.asec_current_money_selection import (
    SELECTED_MAGIC,
    BoundCurrentMoneyBody,
    SelectedCurrentMoney,
    decode_selected_current_money,
    encode_selected_current_money,
    parse_current_money_body,
    select_current_money,
)

PIN = "a" * 64
DIGEST = "b" * 64


def invented_money(person_amounts=None, household_amounts=None):
    """Six invented persons in three single-year households, one per cohort."""
    resources = loader.packaged_current_money_resources(
        inspected_version=money.MICROUNIT_VERSION,
        inspected_module_sha256=money.MICROUNIT_SHA256,
    )
    source = SyntheticAsecSource(
        PIN,
        PIN,
        ((2022, PIN), (2023, PIN), (2024, PIN)),
        (("ED_VAL.archive", PIN), ("ED_VAL.member", PIN)),
        PIN,
    )
    spec = compile_asec_current_money_spec(resources, source)
    scope = AsecMoneyScope(
        (10, 11, 12, 13, 14, 15),
        (100, 101, 102),
        (100, 100, 101, 101, 102, 102),
        (200, 200, 201, 201, 202, 202),
        (2022, 2022, 2023, 2023, 2024, 2024),
        (2022, 2023, 2024),
        ("p0", "p1", "p2", "p3", "p4", "p5"),
        ("h0", "h1", "h2"),
    )
    person_amounts = dict(person_amounts or {})
    household_amounts = dict(household_amounts or {})
    person = pd.DataFrame(
        {
            field.column: np.asarray(
                person_amounts.get(field.name, np.zeros(6)), dtype="float64"
            )
            for field in spec.fields
            if field.entity == "person"
        }
    )
    household = pd.DataFrame(
        {
            field.column: np.asarray(
                household_amounts.get(field.name, np.zeros(3)), dtype="float64"
            )
            for field in spec.fields
            if field.entity == "household"
        }
    )
    views = AsecMoneyViews(person, household, scope)
    ready = require_complete_current_money(
        restate_asec_current_money(classify_asec_money(views, spec, scope), spec), spec
    )
    return spec, ready


def body_of(ready, **overrides):
    return parse_current_money_body(
        encode_current_money(ready),
        expected_header_sha256=overrides.get(
            "expected_header_sha256", money._sha(ready.header)
        ),
        expected_content_sha256=overrides.get(
            "expected_content_sha256", current_money_content_sha256(ready)
        ),
        field_entities=overrides.get("field_entities", field_entities(ready)),
    )


def field_entities(ready):
    return tuple((domain.name, domain.entity) for domain in ready.bindings.spec.fields)


def selected_of(body, persons=(2, 3), households=(1,)):
    return select_current_money(
        body,
        person_positions=np.asarray(persons, dtype="int64"),
        household_positions=np.asarray(households, dtype="int64"),
        prepared_receipt_sha256=DIGEST,
        selection_sha256="c" * 64,
        person_identity_sha256="d" * 64,
        household_identity_sha256="e" * 64,
    )


def test_body_parse_binds_header_and_content_without_minting_readiness():
    _, ready = invented_money({"WSAL_VAL": [1, 2, 3, 4, 5, 6]})
    body = body_of(ready)
    assert type(body) is BoundCurrentMoneyBody
    assert not isinstance(body, money.RestatedAsecMoney)
    assert type(body) is not ReadyCurrentMoney
    assert body.person_rows == 6 and body.household_rows == 3
    assert body.header == ready.header
    assert body.content_sha256 == current_money_content_sha256(ready)
    assert body.entity_of("HTOTVAL") == "household"
    assert body.entity_of("WSAL_VAL") == "person"
    # 2022/2023 rows are restated; 2024 rows keep their nominal bytes.
    assert body.field("WSAL_VAL").amounts.tolist() == pytest.approx(
        ready.field("WSAL_VAL").amounts.tolist()
    )


def test_body_parse_refuses_altered_header_content_and_bytes():
    _, ready = invented_money({"WSAL_VAL": [1, 2, 3, 4, 5, 6]})
    payload = encode_current_money(ready)
    entities = field_entities(ready)
    content = current_money_content_sha256(ready)
    with pytest.raises(MoneyRefusalError, match="BODY_HEADER_BINDING"):
        parse_current_money_body(
            payload,
            expected_header_sha256=PIN,
            expected_content_sha256=content,
            field_entities=entities,
        )
    with pytest.raises(MoneyRefusalError, match="BODY_CONTENT_BINDING"):
        parse_current_money_body(
            payload,
            expected_header_sha256=money._sha(ready.header),
            expected_content_sha256=PIN,
            field_entities=entities,
        )
    tampered = bytearray(payload)
    tampered[-40] ^= 0xFF
    with pytest.raises(MoneyRefusalError, match="PAYLOAD_CHECKSUM"):
        parse_current_money_body(
            bytes(tampered),
            expected_header_sha256=money._sha(ready.header),
            expected_content_sha256=content,
            field_entities=entities,
        )
    with pytest.raises(MoneyRefusalError, match="FIELD_ENTITY_ROSTER"):
        parse_current_money_body(
            payload,
            expected_header_sha256=money._sha(ready.header),
            expected_content_sha256=content,
            field_entities=tuple((name, "tax_unit") for name, _ in entities),
        )
    with pytest.raises(MoneyRefusalError, match="FIELD_ENTITY_ROSTER"):
        parse_current_money_body(
            payload,
            expected_header_sha256=money._sha(ready.header),
            expected_content_sha256=content,
            field_entities=entities[1:],
        )
    # A well-formed but wrongly assigned roster still fails closed on length.
    with pytest.raises(MoneyRefusalError, match="PAYLOAD_LENGTH"):
        parse_current_money_body(
            payload,
            expected_header_sha256=money._sha(ready.header),
            expected_content_sha256=content,
            field_entities=tuple((name, "household") for name, _ in entities),
        )


def test_selection_slices_exact_coordinates_and_preserves_evidence():
    _, ready = invented_money(
        {
            "WSAL_VAL": [1, 2, 3, 4, 5, 6],
            "ANN_VAL": [-1, 0, 7, 0, 0, 0],
            "SEMP_VAL": [0, 0, -5, 0, 0, 0],
        },
        {"HTOTVAL": [11, 12, 13]},
    )
    body = body_of(ready)
    selected = selected_of(body, persons=(2, 3), households=(1,))
    assert type(selected) is SelectedCurrentMoney
    assert not isinstance(selected, money.RestatedAsecMoney)
    assert selected.person_rows == 2 and selected.household_rows == 1
    for name in money.FIELDS:
        parent = body.field(name)
        child = selected.field(name)
        index = [1] if body.entity_of(name) == "household" else [2, 3]
        assert child.amounts.tolist() == parent.amounts[index].tolist()
        assert child.statuses.tolist() == parent.statuses[index].tolist()
        assert child.validity.tolist() == parent.validity[index].tolist()
        assert child.zero_origin.tolist() == parent.zero_origin[index].tolist()
    # A signed self-employment loss survives the slice unclamped.
    assert selected.field("SEMP_VAL").amounts[0] < 0
    # The NIU annuity contributes zero and keeps its declared-NIU status.
    niu = selected_of(body, persons=(0, 1), households=(0,))
    assert niu.field("ANN_VAL").statuses.tolist()[0] == int(
        money.CodebookStatus.DECLARED_NIU
    )
    assert niu.field("ANN_VAL").amounts.tolist() == [0.0, 0.0]


def test_selection_header_binds_parent_and_selection_identities():
    _, ready = invented_money()
    body = body_of(ready)
    selected = selected_of(body)
    header = selected.header_data
    assert header["parent_header_sha256"] == body.header_sha256
    assert header["parent_content_sha256"] == body.content_sha256
    assert header["prepared_receipt_sha256"] == DIGEST
    assert header["selection_sha256"] == "c" * 64
    assert header["person_identity_sha256"] == "d" * 64
    assert header["household_identity_sha256"] == "e" * 64
    assert header["release_eligible"] is False
    assert header["fields"] == list(money.FIELDS)


def test_selection_refuses_unordered_out_of_range_and_untyped_positions():
    _, ready = invented_money()
    body = body_of(ready)
    for persons, match in (
        (np.asarray([3, 2], dtype="int64"), "PERSON_POSITIONS"),
        (np.asarray([0, 6], dtype="int64"), "PERSON_POSITIONS"),
        (np.asarray([0, 0], dtype="int64"), "PERSON_POSITIONS"),
        (np.asarray([], dtype="int64"), "PERSON_POSITIONS"),
        (np.asarray([0, 1], dtype="int32"), "PERSON_POSITIONS"),
    ):
        with pytest.raises(MoneyRefusalError, match=match):
            select_current_money(
                body,
                person_positions=persons,
                household_positions=np.asarray([0], dtype="int64"),
                prepared_receipt_sha256=DIGEST,
                selection_sha256="c" * 64,
                person_identity_sha256="d" * 64,
                household_identity_sha256="e" * 64,
            )
    with pytest.raises(MoneyRefusalError, match="HOUSEHOLD_POSITIONS"):
        select_current_money(
            body,
            person_positions=np.asarray([0, 1], dtype="int64"),
            household_positions=np.asarray([3], dtype="int64"),
            prepared_receipt_sha256=DIGEST,
            selection_sha256="c" * 64,
            person_identity_sha256="d" * 64,
            household_identity_sha256="e" * 64,
        )
    with pytest.raises(MoneyRefusalError, match="SELECTION_BINDING_DIGEST"):
        select_current_money(
            body,
            person_positions=np.asarray([0, 1], dtype="int64"),
            household_positions=np.asarray([0], dtype="int64"),
            prepared_receipt_sha256="not-a-digest",
            selection_sha256="c" * 64,
            person_identity_sha256="d" * 64,
            household_identity_sha256="e" * 64,
        )


def test_selected_codec_round_trips_and_refuses_a_foreign_parent():
    _, ready = invented_money({"WSAL_VAL": [1, 2, 3, 4, 5, 6]})
    body = body_of(ready)
    selected = selected_of(body)
    payload = encode_selected_current_money(selected)
    decoded = decode_selected_current_money(
        payload,
        expected_parent_header_sha256=body.header_sha256,
        expected_parent_content_sha256=body.content_sha256,
        expected_prepared_receipt_sha256=DIGEST,
        expected_selection_sha256="c" * 64,
    )
    assert encode_selected_current_money(decoded) == payload
    assert decoded.header == selected.header
    assert decoded.fields == selected.fields
    with pytest.raises(MoneyRefusalError, match="SELECTED_REPLAY_BINDING"):
        decode_selected_current_money(
            payload,
            expected_parent_header_sha256=PIN,
            expected_parent_content_sha256=body.content_sha256,
            expected_prepared_receipt_sha256=DIGEST,
            expected_selection_sha256="c" * 64,
        )
    with pytest.raises(MoneyRefusalError, match="SELECTED_REPLAY_BINDING"):
        decode_selected_current_money(
            payload,
            expected_parent_header_sha256=body.header_sha256,
            expected_parent_content_sha256=body.content_sha256,
            expected_prepared_receipt_sha256=DIGEST,
            expected_selection_sha256=PIN,
        )


def test_selected_codec_refuses_tampered_bytes_and_short_payloads():
    _, ready = invented_money({"WSAL_VAL": [1, 2, 3, 4, 5, 6]})
    body = body_of(ready)
    payload = encode_selected_current_money(selected_of(body))
    tampered = bytearray(payload)
    tampered[-40] ^= 0xFF
    with pytest.raises(MoneyRefusalError, match="PAYLOAD_CHECKSUM"):
        decode_selected_current_money(
            bytes(tampered),
            expected_parent_header_sha256=body.header_sha256,
            expected_parent_content_sha256=body.content_sha256,
            expected_prepared_receipt_sha256=DIGEST,
            expected_selection_sha256="c" * 64,
        )
    with pytest.raises(MoneyRefusalError, match="PAYLOAD_VERSION"):
        decode_selected_current_money(
            b"X" + payload[1:],
            expected_parent_header_sha256=body.header_sha256,
            expected_parent_content_sha256=body.content_sha256,
            expected_prepared_receipt_sha256=DIGEST,
            expected_selection_sha256="c" * 64,
        )
    truncated = payload[:-8]
    with pytest.raises(MoneyRefusalError, match="PAYLOAD_LENGTH"):
        decode_selected_current_money(
            truncated,
            expected_parent_header_sha256=body.header_sha256,
            expected_parent_content_sha256=body.content_sha256,
            expected_prepared_receipt_sha256=DIGEST,
            expected_selection_sha256="c" * 64,
        )


def test_typed_constructors_are_token_gated():
    _, ready = invented_money()
    body = body_of(ready)
    selected = selected_of(body)
    with pytest.raises(MoneyRefusalError, match="BODY_CONSTRUCTOR_UNAVAILABLE"):
        BoundCurrentMoneyBody(
            body.header, body.content_sha256, body.field_entities, body.fields, 6, 3
        )
    with pytest.raises(MoneyRefusalError, match="SELECTED_CONSTRUCTOR_UNAVAILABLE"):
        SelectedCurrentMoney(selected.header, selected.fields)


@pytest.mark.parametrize(
    "mutation, reason",
    [
        ("null_roster", "FIELD_ENTITY_ROSTER"),
        ("integer_item", "FIELD_ENTITY_ROSTER"),
        ("missing_rows", "SELECTED_HEADER_SCHEMA"),
    ],
)
def test_selected_decoder_sanitizes_malformed_canonical_headers(mutation, reason):
    _, ready = invented_money()
    body = body_of(ready)
    payload = encode_selected_current_money(selected_of(body))
    start = len(SELECTED_MAGIC) + 4
    size = struct.unpack_from("<I", payload, len(SELECTED_MAGIC))[0]
    header = money._parse(payload[start : start + size])
    if mutation == "null_roster":
        header["field_entities"] = None
    elif mutation == "integer_item":
        header["field_entities"][0] = 7
    else:
        del header["person_rows"]
    encoded = money._json(header)
    forged = (
        SELECTED_MAGIC
        + struct.pack("<I", len(encoded))
        + encoded
        + payload[start + size : -32]
    )
    forged += sha256(forged).digest()
    with pytest.raises(MoneyRefusalError, match=reason):
        decode_selected_current_money(
            forged,
            expected_parent_header_sha256=body.header_sha256,
            expected_parent_content_sha256=body.content_sha256,
            expected_prepared_receipt_sha256=DIGEST,
            expected_selection_sha256="c" * 64,
        )


def test_selected_payload_declares_its_own_magic():
    _, ready = invented_money()
    payload = encode_selected_current_money(selected_of(body_of(ready)))
    assert payload.startswith(SELECTED_MAGIC)
    size = struct.unpack_from("<I", payload, len(SELECTED_MAGIC))[0]
    start = len(SELECTED_MAGIC) + 4
    assert sha256(payload[:-32]).digest() == payload[-32:]
    assert payload[start : start + size].startswith(b'{"artifact_kind"')
