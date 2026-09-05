"""Strict local rent encoding/placement contracts, with no population selection."""

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.fit._graph_legacy_qrf import decode_json, encode_json, sha
from microcosm.fit.qrf import _index_identity, _IndexIdentity
from microcosm.graph import ArtifactType, Numeric, platform_fingerprint

from .graph_context import _row_identity

ACS_RENT_ENCODING_TYPE = ArtifactType("microcosm.us.acs_rent_encoding", 1)
ACS_RENT_PLACEMENT_TYPE = ArtifactType("microcosm.us.acs_rent_placement", 1)
ACS_RENT_VALIDATION_TYPE = ArtifactType("microcosm.us.acs_rent_validation", 1)
RENT_PHASE = "prepare_multispine_source_inputs_for_clone"
PLACEMENT_MAGIC = b"microcosm.us.acs_rent_placement/1\n"
HEADER_MAX_BYTES = 64 * 1024
_BINDINGS = (
    "matrix_sha256",
    "frame_context_sha256",
    "frame_context_producer_key",
    "encoding_sha256",
    "encoding_producer_key",
)
_PLACEMENT_FIELDS = {
    "schema_version",
    "household_rows",
    "person_rows",
    "person_index",
    "person_identity",
    "household_identity",
    *_BINDINGS,
}


def is_hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def artifact(context, name, expected_type):
    value = context.artifacts.get(name)
    edges = [edge for edge in context.node.artifact_inputs if edge.name == name]
    if (
        value is None
        or len(edges) != 1
        or edges[0].type != expected_type
        or value.type != expected_type
    ):
        raise ValueError(f"ACS rent requires the exact typed {name} edge.")
    binding(value)
    return value


def binding(value):
    if not is_hash(value.producer_key):
        raise ValueError("ACS rent artifact requires its actual producer key.")
    if (
        value.numerics.numeric is not Numeric.PLATFORM_BITWISE
        or value.numerics.platform != platform_fingerprint()
    ):
        raise ValueError(
            "ACS rent artifact requires the current platform's bitwise scope."
        )
    if not isinstance(value.payload, bytes):
        raise ValueError("ACS rent artifacts require immutable bytes.")
    return {"sha256": sha(value.payload), "producer_key": value.producer_key}


def strict_ids(series, *, label, unique=True):
    if series.dtype != np.dtype("int64"):
        raise ValueError(
            f"ACS rent {label} requires ordinary int64 identity without casts."
        )
    values = series.to_numpy(copy=False)
    if (
        values.dtype != np.dtype("<i8")
        or len(values) == 0
        or (values < 0).any()
        or (unique and not series.is_unique)
    ):
        raise ValueError(f"ACS rent {label} identity is empty, duplicated or invalid.")
    return values


def head_association(person, household):
    person_ids = strict_ids(person["person_id"], label="person")
    household_ids = strict_ids(household["household_id"], label="household")
    membership = strict_ids(
        person["person_household_id"], label="membership", unique=False
    )
    flag = person["is_household_head"]
    if str(flag.dtype) not in {"bool", "boolean"}:
        raise ValueError("ACS rent head flags require an explicit boolean dtype.")
    positions = np.flatnonzero(flag.fillna(False).to_numpy(dtype=bool))
    head_households = pd.Index(membership[positions])
    if not head_households.is_unique or set(head_households) != set(household_ids):
        raise ValueError(
            "ACS rent requires exactly one current head per household and no extras."
        )
    order = head_households.get_indexer(household_ids)
    aligned = positions[order]
    return household_ids, person_ids[aligned], aligned


@dataclass(frozen=True)
class RentPlacement:
    header: object
    household_ids: np.ndarray
    head_person_ids: np.ndarray
    head_positions: np.ndarray


def encode_placement(
    prepared, *, person, household, matrix_sha256, frame_context, encoding
):
    ids, heads, positions = head_association(person, household)
    expected = pd.Index(ids)
    if (
        _index_identity(prepared.predictors.index) != _index_identity(expected)
        or not np.array_equal(prepared.head_positions, positions)
        or _index_identity(prepared.person_index) != _index_identity(person.index)
    ):
        raise ValueError(
            "ACS rent preparation differs from actual household/head/person identity."
        )
    if not is_hash(matrix_sha256):
        raise ValueError("ACS rent placement requires its matrix SHA.")
    context_binding = binding(frame_context)
    encoding_binding = binding(encoding)
    header = {
        "schema_version": 1,
        "household_rows": len(ids),
        "person_rows": len(person),
        "person_index": _index_identity(person.index).to_dict(),
        "person_identity": _row_identity(person, "person"),
        "household_identity": _row_identity(household, "household"),
        "matrix_sha256": matrix_sha256,
        "frame_context_sha256": context_binding["sha256"],
        "frame_context_producer_key": context_binding["producer_key"],
        "encoding_sha256": encoding_binding["sha256"],
        "encoding_producer_key": encoding_binding["producer_key"],
    }
    encoded = encode_json(header)
    if len(encoded) > HEADER_MAX_BYTES:
        raise ValueError("ACS rent placement header exceeds its bound.")
    return (
        PLACEMENT_MAGIC
        + len(encoded).to_bytes(4, "big")
        + encoded
        + ids.tobytes()
        + heads.tobytes()
        + positions.tobytes()
    )


def decode_placement(payload: bytes) -> RentPlacement:
    if not isinstance(payload, bytes) or not payload.startswith(PLACEMENT_MAGIC):
        raise ValueError("Invalid ACS rent placement envelope.")
    start = len(PLACEMENT_MAGIC)
    if len(payload) < start + 4:
        raise ValueError("Truncated ACS rent placement length.")
    length = int.from_bytes(payload[start : start + 4], "big")
    start += 4
    if not 0 < length <= HEADER_MAX_BYTES or len(payload) < start + length:
        raise ValueError("ACS rent placement header is truncated or exceeds its bound.")
    header = decode_json(payload[start : start + length])
    if (
        set(header) != _PLACEMENT_FIELDS
        or type(header["schema_version"]) is not int
        or header["schema_version"] != 1
        or any(not is_hash(header[key]) for key in _BINDINGS)
    ):
        raise ValueError("Malformed ACS rent placement schema/binding.")
    h, n = header["household_rows"], header["person_rows"]
    if type(h) is not int or type(n) is not int or not 0 < h <= n:
        raise ValueError("Invalid ACS rent placement dimensions.")
    identity = _IndexIdentity.from_dict(header["person_index"])
    if identity.length != n:
        raise ValueError("ACS rent person index length differs from placement.")
    for key, rows in (("person_identity", n), ("household_identity", h)):
        value = header[key]
        if (
            not isinstance(value, dict)
            or set(value) != {"rows", "id_dtype", "ordered_ids_sha256"}
            or type(value["rows"]) is not int
            or value["rows"] != rows
            or value["id_dtype"] != "int64"
            or not is_hash(value["ordered_ids_sha256"])
        ):
            raise ValueError("ACS rent placement row identity is malformed.")
    body = memoryview(payload)[start + length :]
    if len(body) != h * 24:
        raise ValueError("ACS rent placement vector lengths differ from dimensions.")
    ids, heads, positions = np.frombuffer(body, dtype="<i8").reshape(3, h)
    for values in (ids, heads, positions):
        if (values < 0).any() or len(np.unique(values)) != h:
            raise ValueError(
                "ACS rent placement vectors must be unique and nonnegative."
            )
    if (positions >= n).any():
        raise ValueError("ACS rent placement position exceeds person rows.")
    actual = _row_identity(pd.DataFrame({"household_id": ids}), "household")
    if encode_json(actual) != encode_json(header["household_identity"]):
        raise ValueError("ACS rent placement household IDs differ from its identity.")
    return RentPlacement(MappingProxyType(header), ids, heads, positions)
