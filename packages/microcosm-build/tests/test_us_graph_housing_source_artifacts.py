"""The housing source artifact codecs are strict, self-describing and bounded.

These exercise the codec directly, without a graph run: envelope framing, the
exact five-column inventory, per-column digests, ID identity, the canonical
string and physical boolean representations, and every documented refusal.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import _graph_housing_source_artifacts as codec
from microcosm.build.us_runtime import graph_context
from microcosm.graph import ArtifactType, Numeric, NumericScope, platform_fingerprint
from microcosm.graph.kernel import ArtifactValue
from microcosm.graph.population import dtype_for_token

CANONICAL_STRING = dtype_for_token("string")
KEY_A = "a" * 64
KEY_B = "b" * 64


def _value(payload: bytes, artifact_type: ArtifactType, key: str = KEY_A):
    return ArtifactValue(
        payload,
        artifact_type,
        "c" * 64,
        key,
        NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
    )


def _parent_context() -> ArtifactValue:
    document = {
        "schema_version": 1,
        "entities": {
            entity: {
                "columns": [f"{entity}_id"],
                "rows": 2,
                "id_dtype": "int64",
                "ordered_ids_sha256": "0" * 64,
            }
            for entity in ("person", "household", "tax_unit", "spm_unit", "family")
        },
        "metadata": {},
        "mass_log": [],
        "weight_sources": {"household": "design"},
    }
    document["entities"]["marital_unit"] = document["entities"]["person"]
    return _value(
        json.dumps(document).encode(), graph_context.US_FRAME_CONTEXT_TYPE, KEY_B
    )


def _identities():
    return {
        "person": {
            "full": np.array([1, 2, 3, 4], dtype=np.int64),
            "selected": np.array([1, 3], dtype=np.int64),
        },
        "household": {
            "full": np.array([10, 20], dtype=np.int64),
            "selected": np.array([10], dtype=np.int64),
        },
        "tax_unit": {
            "full": np.array([30, 40], dtype=np.int64),
            "selected": np.array([30], dtype=np.int64),
        },
        "spm_unit": {
            "full": np.array([50, 60], dtype=np.int64),
            "selected": np.array([50], dtype=np.int64),
        },
        "family": {
            "full": np.array([70, 80], dtype=np.int64),
            "selected": np.array([70], dtype=np.int64),
        },
        "marital_unit": {
            "full": np.array([90, 91], dtype=np.int64),
            "selected": np.array([90], dtype=np.int64),
        },
    }


def _projection_payload() -> bytes:
    return codec.encode_projection(
        parent_context=_parent_context(),
        selection={
            "column": "PERIDNUM",
            "phase": "pre_clone",
            "clone_index_column": "person_support_clone_index",
        },
        entities=_identities(),
        person_index=pd.Index([5, 7, 9, 11], name="original_row"),
        weight_sources={"household": "design"},
        mode=codec.FRESH_SOURCE_MODE,
    )


def _source_tables():
    return {
        "person": pd.DataFrame(
            {
                "person_id": pd.Series([1, 3], dtype="int64"),
                "pre_subsidy_rent": pd.Series([1200.0, 0.0], dtype="float64"),
            }
        ),
        "household": pd.DataFrame(
            {
                "household_id": pd.Series([10], dtype="int64"),
                "tenure_type": pd.Series(["RENTED"], dtype=CANONICAL_STRING),
            }
        ),
        "spm_unit": pd.DataFrame(
            {
                "spm_unit_id": pd.Series([50], dtype="int64"),
                "receives_housing_assistance": pd.Series([True], dtype="bool"),
                "spm_unit_tenure_type": pd.Series(["RENTER"], dtype=CANONICAL_STRING),
                "takes_up_housing_assistance_if_eligible": pd.Series(
                    [True], dtype="bool"
                ),
            }
        ),
    }


def _outputs_payload(tables=None) -> bytes:
    return codec.encode_source_outputs(
        _source_tables() if tables is None else tables,
        projection=_value(_projection_payload(), codec.HOUSING_SOURCE_PROJECTION_TYPE),
        frame_context=_parent_context(),
        validation=_value(b"{}", codec.HOUSING_SOURCE_PROJECTION_TYPE),
        mode=codec.FRESH_SOURCE_MODE,
    )


def _split(payload: bytes, magic: bytes):
    start = len(magic)
    length = int.from_bytes(payload[start : start + 4], "big")
    header = json.loads(payload[start + 4 : start + 4 + length])
    return header, payload[start + 4 + length :]


def _reframe(header: dict, body: bytes, magic: bytes) -> bytes:
    encoded = codec.encode_json(header)
    return magic + len(encoded).to_bytes(4, "big") + encoded + body


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_projection_round_trip_binds_parent_selection_and_both_identity_sets():
    document = codec.decode_projection(_projection_payload())
    assert document["mode"] == codec.FRESH_SOURCE_MODE
    assert document["phase"] == codec.HOUSING_PHASE
    assert document["operator"] == "with_us_housing_inputs"
    assert document["family"] == "housing_inputs"
    assert document["parent_context_producer_key"] == KEY_B
    assert document["entities"]["person"]["full"]["rows"] == 4
    assert document["entities"]["person"]["selected"]["rows"] == 2
    assert document["person_index"]["length"] == 4
    assert document["output_family"] == {
        "person": ["pre_subsidy_rent"],
        "household": ["tenure_type"],
        "spm_unit": [
            "receives_housing_assistance",
            "spm_unit_tenure_type",
            "takes_up_housing_assistance_if_eligible",
        ],
    }
    # The projection carries no caller-authored acceptance flag.
    assert "passed" not in document


def test_source_outputs_round_trip_preserves_exact_values_and_dtypes():
    decoded = codec.decode_source_outputs(_outputs_payload())
    expected = _source_tables()
    for entity, table in expected.items():
        pd.testing.assert_frame_equal(decoded.tables[entity], table)
        assert decoded.header["entities"][entity]["rows"] == len(table)
    assert [record["column"] for record in decoded.header["columns"]] == [
        column for _entity, column, _source, _merge in codec.SOURCE_COLUMNS
    ]
    assert decoded.tables["spm_unit"]["receives_housing_assistance"].dtype == np.dtype(
        bool
    )
    tenure = decoded.tables["household"]["tenure_type"].dtype
    assert tenure == CANONICAL_STRING and tenure.storage == CANONICAL_STRING.storage


def test_null_tenure_round_trips_without_hiding_bytes():
    tables = _source_tables()
    tables["household"]["tenure_type"] = pd.Series([pd.NA], dtype=CANONICAL_STRING)
    decoded = codec.decode_source_outputs(_outputs_payload(tables))
    assert decoded.tables["household"]["tenure_type"].isna().all()
    header, _body = _split(_outputs_payload(tables), codec.OUTPUTS_MAGIC)
    assert header["columns"][1]["nulls"] == 1
    assert header["columns"][1]["utf8_bytes"] == 0


def test_merge_evidence_round_trip_carries_the_ordered_source_receipt():
    receipt = {
        "phase": "pre_clone",
        "operator_order": ["with_us_housing_inputs"],
        "suboperators": [],
    }
    payload = codec.encode_merge_evidence(
        projection=_value(_projection_payload(), codec.HOUSING_SOURCE_PROJECTION_TYPE),
        source_outputs=_value(
            _outputs_payload(), codec.HOUSING_SOURCE_OUTPUTS_TYPE, KEY_B
        ),
        pending_context=b"{}",
        merged_rows={"person": 2},
        selected_rows={"person": 2},
        pool_rows={"person": 4},
        merged_cells=[],
        native_cells=[],
        incumbent_cells=[],
        source_preparation=receipt,
        mode=codec.FRESH_SOURCE_MODE,
    )
    document = codec.decode_merge_evidence(payload)
    assert document["scope"] == codec.HOUSING_GATE_SCOPE
    assert document["source_preparation"] == receipt
    assert document["source_outputs_producer_key"] == KEY_B


# ---------------------------------------------------------------------------
# Envelope and framing refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda payload: payload[1:], id="magic"),
        pytest.param(lambda payload: payload[:20], id="truncated-length"),
        pytest.param(lambda payload: payload + b"\x00", id="trailing-bytes"),
    ],
)
def test_source_outputs_envelope_refusals(mutate):
    with pytest.raises(ValueError):
        codec.decode_source_outputs(mutate(_outputs_payload()))


def test_source_outputs_refuses_a_declared_header_longer_than_its_bound():
    payload = _outputs_payload()
    start = len(codec.OUTPUTS_MAGIC)
    tampered = (
        payload[:start]
        + (codec.HEADER_MAX_BYTES + 1).to_bytes(4, "big")
        + payload[start + 4 :]
    )
    with pytest.raises(ValueError, match="exceeds its bound"):
        codec.decode_source_outputs(tampered)


def test_projection_refuses_a_body():
    with pytest.raises(ValueError, match="no body"):
        codec.decode_projection(_projection_payload() + b"x")


# ---------------------------------------------------------------------------
# Header tampering refusals
# ---------------------------------------------------------------------------


def _tampered_outputs(mutate) -> bytes:
    header, body = _split(_outputs_payload(), codec.OUTPUTS_MAGIC)
    body = mutate(header, bytearray(body))
    return _reframe(header, bytes(body), codec.OUTPUTS_MAGIC)


def _drop_column(header, body):
    header["columns"] = header["columns"][:-1]
    return body


def _reorder_columns(header, body):
    header["columns"][2], header["columns"][3] = (
        header["columns"][3],
        header["columns"][2],
    )
    return body


def _extra_field(header, body):
    header["surprise"] = 1
    return body


def _bad_mode(header, body):
    header["mode"] = "complete_incumbent"
    return body


def _bad_phase(header, body):
    header["phase"] = "post_clone"
    return body


def _bad_operator(header, body):
    header["operator"] = "with_us_eligibility_inputs"
    return body


def _bad_schema(header, body):
    header["schema_version"] = 2
    return body


def _wrong_entity(header, body):
    header["entities"]["benefit_unit"] = header["entities"].pop("spm_unit")
    return body


def _bad_digest(header, body):
    header["columns"][0]["values_sha256"] = "0" * 64
    return body


def _malformed_projection_binding(header, body):
    header["projection_sha256"] = "not-a-digest"
    return body


def _malformed_validation_binding(header, body):
    header["validation_producer_key"] = ""
    return body


def _row_mismatch(header, body):
    header["entities"]["person"]["rows"] = 3
    return body


def _bad_string_storage(header, body):
    header["columns"][1]["string_storage"] = "pyarrow"
    return body


def _bad_encoding(header, body):
    header["columns"][0]["encoding"] = "pickle-v1"
    return body


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(_drop_column, id="missing-leaf"),
        pytest.param(_reorder_columns, id="reordered-leaves"),
        pytest.param(_extra_field, id="extra-header-field"),
        pytest.param(_bad_mode, id="unsupported-mode"),
        pytest.param(_bad_phase, id="unsupported-phase"),
        pytest.param(_bad_operator, id="unsupported-operator"),
        pytest.param(_bad_schema, id="unsupported-schema"),
        pytest.param(_wrong_entity, id="unsupported-entity"),
        pytest.param(_bad_digest, id="digest-mismatch"),
        pytest.param(_malformed_projection_binding, id="malformed-projection-binding"),
        pytest.param(_malformed_validation_binding, id="malformed-validation-binding"),
        pytest.param(_row_mismatch, id="row-count-mismatch"),
        pytest.param(_bad_string_storage, id="foreign-string-storage"),
        pytest.param(_bad_encoding, id="pickle-encoding"),
    ],
)
def test_tampered_source_output_headers_are_refused(mutate):
    with pytest.raises(ValueError):
        codec.decode_source_outputs(_tampered_outputs(mutate))


def test_decoded_header_surfaces_every_upstream_binding_for_its_consumer():
    """The codec authenticates bytes; the graph nodes compare producer keys."""
    header = codec.decode_source_outputs(_outputs_payload()).header
    assert header["projection_sha256"] == codec.sha(_projection_payload())
    assert header["projection_producer_key"] == KEY_A
    assert header["frame_context_producer_key"] == KEY_B
    assert header["validation_sha256"] == codec.sha(b"{}")


def test_tampered_source_output_values_are_refused():
    payload = bytearray(_outputs_payload())
    payload[-1] ^= 0xFF
    with pytest.raises(ValueError, match="differ from their digest"):
        codec.decode_source_outputs(bytes(payload))


def test_duplicate_and_negative_source_ids_are_refused():
    header, body = _split(_outputs_payload(), codec.OUTPUTS_MAGIC)
    duplicated = bytearray(body)
    duplicated[8:16] = np.asarray([1], dtype="<i8").tobytes()
    with pytest.raises(ValueError):
        codec.decode_source_outputs(
            _reframe(header, bytes(duplicated), codec.OUTPUTS_MAGIC)
        )


def test_noncanonical_boolean_bytes_are_refused():
    header, body = _split(_outputs_payload(), codec.OUTPUTS_MAGIC)
    offset = (2 + 1 + 1) * 8 + 2 * 8 + header["columns"][1]["value_bytes"]
    tampered = bytearray(body)
    tampered[offset] = 7
    header["columns"][2]["values_sha256"] = codec.sha(
        bytes(tampered[offset : offset + 1])
    )
    with pytest.raises(ValueError):
        codec.decode_source_outputs(
            _reframe(header, bytes(tampered), codec.OUTPUTS_MAGIC)
        )


# ---------------------------------------------------------------------------
# Encode-side representation refusals
# ---------------------------------------------------------------------------


def test_nullable_boolean_source_values_are_refused():
    tables = _source_tables()
    tables["spm_unit"]["receives_housing_assistance"] = pd.Series(
        [True], dtype="boolean"
    )
    with pytest.raises(ValueError, match="physical boolean"):
        _outputs_payload(tables)


def test_object_dtype_tenure_is_refused():
    tables = _source_tables()
    tables["household"]["tenure_type"] = pd.Series(["RENTED"], dtype=object)
    with pytest.raises(ValueError, match="canonical string"):
        _outputs_payload(tables)


def test_foreign_string_storage_is_refused_at_encode():
    tables = _source_tables()
    tables["household"]["tenure_type"] = pd.Series(
        ["RENTED"], dtype=pd.StringDtype(storage="python", na_value=np.nan)
    )
    with pytest.raises(ValueError, match="canonical string"):
        _outputs_payload(tables)


def test_nonfinite_rent_is_refused():
    tables = _source_tables()
    tables["person"]["pre_subsidy_rent"] = pd.Series([np.nan, 0.0], dtype="float64")
    with pytest.raises(ValueError, match="nonfinite"):
        _outputs_payload(tables)


def test_integer_rent_is_refused():
    tables = _source_tables()
    tables["person"]["pre_subsidy_rent"] = pd.Series([1200, 0], dtype="int64")
    with pytest.raises(ValueError, match="float64"):
        _outputs_payload(tables)


def test_missing_leaf_is_refused_at_encode():
    tables = _source_tables()
    tables["spm_unit"] = tables["spm_unit"].drop(columns=["spm_unit_tenure_type"])
    with pytest.raises(ValueError, match="omit"):
        _outputs_payload(tables)


def test_unsupported_mode_is_refused_at_encode():
    with pytest.raises(ValueError, match="Unsupported"):
        codec.encode_source_outputs(
            _source_tables(),
            projection=_value(
                _projection_payload(), codec.HOUSING_SOURCE_PROJECTION_TYPE
            ),
            frame_context=_parent_context(),
            validation=_value(b"{}", codec.HOUSING_SOURCE_PROJECTION_TYPE),
            mode="complete_incumbent",
        )


def test_non_bitwise_artifact_binding_is_refused():
    stale = ArtifactValue(
        b"{}",
        codec.HOUSING_SOURCE_PROJECTION_TYPE,
        "c" * 64,
        KEY_A,
        NumericScope(Numeric.BITWISE),
    )
    with pytest.raises(ValueError, match="platform's bitwise scope"):
        codec.binding(stale)


def test_strict_ids_refuses_non_int64_and_duplicated_identity():
    with pytest.raises(ValueError, match="ordinary int64"):
        codec.strict_ids(pd.Series([1.0, 2.0]), label="person")
    with pytest.raises(ValueError, match="duplicated"):
        codec.strict_ids(pd.Series([1, 1], dtype="int64"), label="person")


# ---------------------------------------------------------------------------
# Cell digests
# ---------------------------------------------------------------------------


def test_cell_digest_separates_values_from_null_pattern():
    values = pd.Series([1.0, np.nan], dtype="float64")
    zeros = pd.Series([1.0, 0.0], dtype="float64")
    assert codec.cell_digest(values, entity="person", column="pre_subsidy_rent") != (
        codec.cell_digest(zeros, entity="person", column="pre_subsidy_rent")
    )
    booleans = pd.Series([True, pd.NA], dtype="boolean")
    filled = pd.Series([True, False], dtype="boolean")
    assert codec.cell_digest(
        booleans, entity="spm_unit", column="receives_housing_assistance"
    ) != codec.cell_digest(
        filled, entity="spm_unit", column="receives_housing_assistance"
    )


def test_cell_digest_refuses_an_unsupported_representation():
    with pytest.raises(ValueError, match="cannot digest"):
        codec.cell_digest(
            pd.Series([1, 2], dtype="int64"), entity="person", column="pre_subsidy_rent"
        )


def test_live_output_roster_is_read_from_the_registry(monkeypatch):
    from microcosm.build.us_runtime import operator_boundary

    families = dict(operator_boundary.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES)
    families["housing_inputs"] = {
        "person": frozenset({"pre_subsidy_rent"}),
        "household": frozenset({"tenure_type"}),
        "spm_unit": frozenset({"receives_housing_assistance"}),
    }
    monkeypatch.setattr(codec, "PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES", families)
    with pytest.raises(ValueError, match="registered housing output family"):
        codec.housing_output_family()


# ---------------------------------------------------------------------------
# Hardening: overflow, domain separation and header/buffer agreement
# ---------------------------------------------------------------------------


def _string_column_offset(header) -> int:
    """Byte offset of the first string column's buffer inside the body."""
    ids = sum(
        header["entities"][entity]["rows"] * 8 for entity in codec.SOURCE_ENTITIES
    )
    return ids + header["columns"][0]["value_bytes"]


def test_wrapping_int64_string_offsets_are_refused():
    """A differenced monotonicity check would wrap; comparison cannot."""
    # Three offsets are needed to express a wrapped step, so widen the table.
    tables = _source_tables()
    tables["household"] = pd.DataFrame(
        {
            "household_id": pd.Series([10, 20], dtype="int64"),
            "tenure_type": pd.Series(["RENTED", "NONE"], dtype=CANONICAL_STRING),
        }
    )
    payload = _outputs_payload(tables)
    header, body = _split(payload, codec.OUTPUTS_MAGIC)
    start = _string_column_offset(header)
    rows = header["entities"]["household"]["rows"]
    utf8 = header["columns"][1]["utf8_bytes"]
    mask_end = start + rows
    offsets_end = mask_end + (rows + 1) * 8
    assert rows == 2
    # Offsets that are non-decreasing only under wrapped int64 arithmetic.
    wrapped = np.array([0, np.iinfo(np.int64).min, utf8], dtype="<i8")
    tampered = bytearray(body)
    tampered[mask_end:offsets_end] = wrapped.tobytes()
    header["columns"][1]["values_sha256"] = codec.sha(
        bytes(tampered[start : start + header["columns"][1]["value_bytes"]])
    )
    with pytest.raises(ValueError, match="offsets are invalid"):
        codec.decode_source_outputs(
            _reframe(header, bytes(tampered), codec.OUTPUTS_MAGIC)
        )


def test_cell_digest_is_domain_separated_by_coordinate_and_dtype():
    values = pd.Series([1.0, 2.0], dtype="float64")
    first = codec.cell_digest(values, entity="person", column="pre_subsidy_rent")
    second = codec.cell_digest(values, entity="person", column="other_column")
    assert first["digest"] != second["digest"]
    booleans = pd.Series([True, False], dtype="bool")
    nullable = pd.Series([True, False], dtype="boolean")
    assert (
        codec.cell_digest(
            booleans, entity="spm_unit", column="receives_housing_assistance"
        )["digest"]
        != codec.cell_digest(
            nullable, entity="spm_unit", column="receives_housing_assistance"
        )["digest"]
    )


@pytest.mark.parametrize(
    ("index", "field", "value"),
    [
        (0, "nulls", 1),
        (0, "encoding", "raw-bool-v1"),
        (0, "value_bytes", 8),
        (2, "encoding", "raw-f8-v1"),
        (2, "nulls", 1),
    ],
)
def test_scalar_records_must_agree_with_their_declared_representation(
    index, field, value
):
    header, body = _split(_outputs_payload(), codec.OUTPUTS_MAGIC)
    header["columns"][index][field] = value
    with pytest.raises(ValueError):
        codec.decode_source_outputs(_reframe(header, body, codec.OUTPUTS_MAGIC))


def test_zero_row_entity_tables_are_refused_on_decode():
    header, body = _split(_outputs_payload(), codec.OUTPUTS_MAGIC)
    header["entities"]["household"]["rows"] = 0
    with pytest.raises(ValueError):
        codec.decode_source_outputs(_reframe(header, body, codec.OUTPUTS_MAGIC))


def _projection_with(**changes):
    payload = _projection_payload()
    header, _body = _split(payload, codec.PROJECTION_MAGIC)
    header.update(changes)
    encoded = codec.encode_json(header)
    return codec.PROJECTION_MAGIC + len(encoded).to_bytes(4, "big") + encoded


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param(
            {
                "selection": {
                    "column": 1,
                    "phase": "pre_clone",
                    "clone_index_column": "x",
                }
            },
            id="non-string-selection",
        ),
        pytest.param({"person_index": {"nonsense": True}}, id="malformed-person-index"),
        pytest.param(
            {"output_family": {"person": "pre_subsidy_rent"}},
            id="malformed-output-family",
        ),
    ],
)
def test_malformed_projection_fields_raise_value_errors(changes):
    with pytest.raises(ValueError):
        codec.decode_projection(_projection_with(**changes))


def test_projection_with_a_foreign_entity_key_is_refused():
    payload = _projection_payload()
    header, _body = _split(payload, codec.PROJECTION_MAGIC)
    header["entities"]["benefit_unit"] = header["entities"].pop("family")
    encoded = codec.encode_json(header)
    with pytest.raises(ValueError, match="no US entity"):
        codec.decode_projection(
            codec.PROJECTION_MAGIC + len(encoded).to_bytes(4, "big") + encoded
        )


def test_merge_evidence_refuses_an_unsupported_mode_at_encode():
    with pytest.raises(ValueError, match="Unsupported"):
        codec.encode_merge_evidence(
            projection=_value(
                _projection_payload(), codec.HOUSING_SOURCE_PROJECTION_TYPE
            ),
            source_outputs=_value(
                _outputs_payload(), codec.HOUSING_SOURCE_OUTPUTS_TYPE
            ),
            pending_context=b"{}",
            merged_rows={},
            selected_rows={},
            pool_rows={},
            merged_cells=[],
            native_cells=[],
            incumbent_cells=[],
            source_preparation={
                "phase": "pre_clone",
                "operator_order": ["with_us_housing_inputs"],
            },
            mode="complete_incumbent",
        )
