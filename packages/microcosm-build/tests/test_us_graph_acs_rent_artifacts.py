"""Placement bytes preserve genuine rows and reject ambiguous identity."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime._graph_acs_rent_artifacts import (
    decode_placement,
    encode_placement,
)
from microcosm.build.us_runtime.housing_inputs import AcsRentRecipientPreparation
from microcosm.graph import (
    ArtifactType,
    ArtifactValue,
    Numeric,
    NumericScope,
    platform_fingerprint,
)


def value(payload=b"{}"):
    return ArtifactValue(
        payload,
        ArtifactType("test.binding", 1),
        "a" * 64,
        "b" * 64,
        NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
    )


def inputs():
    person = pd.DataFrame(
        {
            "person_id": np.array([2, 9, 10], dtype="int64"),
            "person_household_id": np.array([20, 20, 11], dtype="int64"),
            "is_household_head": [True, False, True],
        },
        index=pd.Index([100, 105, 110], name="person_row"),
    )
    household = pd.DataFrame({"household_id": np.array([11, 20], dtype="int64")})
    prepared = AcsRentRecipientPreparation(
        pd.DataFrame({"x": [1.0, 2.0]}, index=pd.Index([11, 20])),
        np.array([2, 0], dtype="int64"),
        person.index,
    )
    return prepared, person, household


def test_placement_roundtrip_uses_household_order_and_original_person_positions():
    prepared, person, household = inputs()
    payload = encode_placement(
        prepared,
        person=person,
        household=household,
        matrix_sha256="c" * 64,
        frame_context=value(),
        encoding=value(),
    )
    decoded = decode_placement(payload)
    np.testing.assert_array_equal(decoded.household_ids, [11, 20])
    np.testing.assert_array_equal(decoded.head_person_ids, [10, 2])
    np.testing.assert_array_equal(decoded.head_positions, [2, 0])
    assert decoded.header["person_index"]["names_repr"] == "('person_row',)"
    assert decoded.header["frame_context_producer_key"] == "b" * 64
    assert not decoded.head_positions.flags.writeable


@pytest.mark.parametrize(
    "damage",
    [
        "truncated",
        "trailing",
        "duplicate_ids",
        "positions",
        "header_type",
        "header_bound",
        "producer",
    ],
)
def test_placement_refuses_malformed_body_and_header(damage):
    from microcosm.build.us_runtime import _graph_acs_rent_artifacts as codec

    prepared, person, household = inputs()
    payload = encode_placement(
        prepared,
        person=person,
        household=household,
        matrix_sha256="c" * 64,
        frame_context=value(),
        encoding=value(),
    )
    offset = len(codec.PLACEMENT_MAGIC)
    size = int.from_bytes(payload[offset : offset + 4], "big")
    header = codec.decode_json(payload[offset + 4 : offset + 4 + size])
    body = bytearray(payload[offset + 4 + size :])
    if damage == "truncated":
        payload = payload[:-1]
    elif damage == "trailing":
        payload += b"bad"
    elif damage == "header_bound":
        payload = codec.PLACEMENT_MAGIC + (65537).to_bytes(4, "big")
    else:
        if damage == "duplicate_ids":
            body[8:16] = body[:8]
        elif damage == "positions":
            body[-8:] = np.array([3], dtype="<i8").tobytes()
        elif damage == "header_type":
            header["household_rows"] = 2.0
        else:
            header["frame_context_producer_key"] = "bad"
        encoded = codec.encode_json(header)
        payload = (
            codec.PLACEMENT_MAGIC + len(encoded).to_bytes(4, "big") + encoded + body
        )
    with pytest.raises(ValueError):
        decode_placement(bytes(payload))


@pytest.mark.parametrize("damage", ["ids_dtype", "wrong_head", "person_index", "scope"])
def test_encoder_refuses_false_placement(damage):
    prepared, person, household = inputs()
    context = value()
    if damage == "ids_dtype":
        household["household_id"] = household.household_id.astype("Int64")
    elif damage == "wrong_head":
        prepared = replace(prepared, head_positions=np.array([1, 0], dtype="int64"))
    elif damage == "person_index":
        person.index = pd.RangeIndex(3)
    else:
        context = replace(
            context, numerics=NumericScope(Numeric.PLATFORM_BITWISE, platform="foreign")
        )
    with pytest.raises(ValueError):
        encode_placement(
            prepared,
            person=person,
            household=household,
            matrix_sha256="c" * 64,
            frame_context=context,
            encoding=value(),
        )
