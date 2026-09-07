"""Strict finite matrix bytes retain the prepared entity-ID index exactly."""

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.fit.model_input import (
    MATRIX_HEADER_MAX_BYTES,
    RECIPIENT_MATRIX_TYPE,
    decode_recipient_matrix,
    encode_recipient_matrix,
)


def fixture(name=None):
    ids = np.array([97, 11, 301], dtype=np.int64)
    return pd.DataFrame(
        {"age": [40.0, 26.0, 65.0], "indicator": [1.0, 0.0, 1.0]},
        index=pd.Index(ids, name=name),
    ), ids


@pytest.mark.parametrize("name", [None, "household_id"])
def test_matrix_roundtrip_preserves_exact_values_columns_and_index(name):
    features, ids = fixture(name)
    payload = encode_recipient_matrix(features, entity="household", entity_ids=ids)
    decoded = decode_recipient_matrix(payload)
    assert RECIPIENT_MATRIX_TYPE.schema_version == 1
    assert decoded.entity == "household"
    pd.testing.assert_frame_equal(decoded.features, features)
    np.testing.assert_array_equal(decoded.entity_ids, ids)
    assert (
        encode_recipient_matrix(
            decoded.features, entity=decoded.entity, entity_ids=decoded.entity_ids
        )
        == payload
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "range",
        "nullable",
        "object",
        "multi",
        "name",
        "duplicate",
        "order",
        "zero",
        "float32",
        "nan",
        "infinity",
        "big_endian",
        "column",
        "ids_dtype",
    ],
)
def test_encoder_refuses_casts_or_ambiguous_rows(mutation):
    frame, ids = fixture()
    if mutation == "range":
        frame.index = pd.RangeIndex(3)
    elif mutation == "nullable":
        frame.index = pd.Index(ids, dtype="Int64")
    elif mutation == "object":
        frame.index = pd.Index(ids, dtype=object)
    elif mutation == "multi":
        frame.index = pd.MultiIndex.from_arrays([ids, ids])
    elif mutation == "name":
        frame.index.name = ("not", "scalar")
    elif mutation == "duplicate":
        frame.index = pd.Index([97, 97, 301])
        ids = np.array([97, 97, 301], dtype=np.int64)
    elif mutation == "order":
        ids = ids[::-1]
    elif mutation == "zero":
        frame, ids = frame.iloc[:0], ids[:0]
    elif mutation == "float32":
        frame["age"] = frame.age.astype("float32")
    elif mutation == "nan":
        frame.iloc[0, 0] = np.nan
    elif mutation == "infinity":
        frame.iloc[0, 0] = np.inf
    elif mutation == "big_endian":
        frame["age"] = frame.age.to_numpy().astype(">f8")
    elif mutation == "column":
        frame.columns = ["same", "same"]
    elif mutation == "ids_dtype":
        ids = ids.astype("uint64")
    with pytest.raises((ValueError, TypeError)):
        encode_recipient_matrix(frame, entity="household", entity_ids=ids)


@pytest.mark.parametrize(
    "mutation",
    [
        "truncated",
        "trailing",
        "nan",
        "duplicate_ids",
        "reordered_ids",
        "header_bound",
        "zero",
        "noncanonical",
        "wrong_identity",
        "identity_length_float",
        "column_count",
    ],
)
def test_decoder_refuses_malformed_identity_or_body(mutation):
    from microcosm.fit import model_input as codec

    frame, ids = fixture()
    payload = encode_recipient_matrix(frame, entity="household", entity_ids=ids)
    offset = len(codec._MAGIC)
    length = int.from_bytes(payload[offset : offset + 4], "big")
    header = json.loads(payload[offset + 4 : offset + 4 + length])
    body = bytearray(payload[offset + 4 + length :])
    if mutation == "truncated":
        payload = payload[:-1]
    elif mutation == "trailing":
        payload += b"x"
    elif mutation in {"nan", "duplicate_ids", "reordered_ids"}:
        if mutation == "nan":
            body[len(ids) * 8 : len(ids) * 8 + 8] = np.array(
                [np.nan], dtype="<f8"
            ).tobytes()
        elif mutation == "duplicate_ids":
            body[8:16] = body[:8]
        else:
            body[:16] = body[8:16] + body[:8]
        payload = payload[: offset + 4 + length] + bytes(body)
    elif mutation == "header_bound":
        payload = codec._MAGIC + (MATRIX_HEADER_MAX_BYTES + 1).to_bytes(4, "big")
    else:
        if mutation == "zero":
            header["rows"] = 0
        elif mutation == "wrong_identity":
            header["index_identity"]["sha256"] = "f" * 64
        elif mutation == "identity_length_float":
            header["index_identity"]["length"] = float(len(ids))
        elif mutation == "column_count":
            header["columns"].append("extra")
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        if mutation == "noncanonical":
            encoded += b" "
        payload = codec._MAGIC + len(encoded).to_bytes(4, "big") + encoded + bytes(body)
    with pytest.raises((ValueError, TypeError)):
        decode_recipient_matrix(payload)
