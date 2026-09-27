"""Finite model features with an exact, separate entity-ID row index.

V1 is deliberately narrower than a pandas serializer: ordinary int64 Index,
None/string name, positive row count and little-endian float64 columns only.
The producer owns scientific encoding; this codec never casts source values.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit.qrf import _index_identity
from microcosm.graph import ArtifactType

RECIPIENT_MATRIX_TYPE = ArtifactType("microcosm.fit.legacy_recipient_matrix", 1)
MATRIX_HEADER_MAX_BYTES = 64 * 1024
_MAGIC = b"microcosm.fit.legacy_recipient_matrix/1\n"
_FIELDS = {
    "schema_version",
    "entity",
    "rows",
    "columns",
    "index_name",
    "index_identity",
}


@dataclass(frozen=True)
class PreparedRecipientMatrix:
    entity: str
    features: pd.DataFrame
    entity_ids: np.ndarray


def _index(ids, name):
    if name is not None and not isinstance(name, str):
        raise ValueError("Recipient matrix index name must be None or a string.")
    if (
        not isinstance(ids, np.ndarray)
        or ids.dtype != np.dtype("<i8")
        or ids.ndim != 1
        or len(ids) == 0
    ):
        raise ValueError("Recipient matrix requires a nonempty exact int64 ID vector.")
    result = pd.Index(ids, name=name)
    if not result.is_unique:
        raise ValueError("Recipient matrix IDs must be unique.")
    return result


def encode_recipient_matrix(features, *, entity, entity_ids) -> bytes:
    if not isinstance(features, pd.DataFrame):
        raise TypeError("Recipient matrix features must be a DataFrame.")
    if not isinstance(entity, str) or not entity or "." in entity:
        raise ValueError("Recipient matrix entity must be a nonempty entity name.")
    if type(features.index) is not pd.Index or features.index.dtype != np.dtype(
        "int64"
    ):
        raise ValueError(
            "Recipient matrix V1 requires an ordinary int64 Index, without casts."
        )
    index = _index(entity_ids, features.index.name)
    identity = _index_identity(index).to_dict()
    if _index_identity(features.index).to_dict() != identity:
        raise ValueError("Recipient matrix index and ordered entity IDs differ.")
    columns = codec.names(tuple(features.columns), "matrix columns")
    if any(dtype != np.dtype("<f8") for dtype in features.dtypes):
        raise ValueError(
            "Recipient matrix columns must be exact little-endian float64."
        )
    values = features.to_numpy(copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Recipient matrix features must be finite.")
    header = codec.encode_json(
        {
            "schema_version": 1,
            "entity": entity,
            "rows": len(index),
            "columns": list(columns),
            "index_name": index.name,
            "index_identity": identity,
        }
    )
    if len(header) > MATRIX_HEADER_MAX_BYTES:
        raise ValueError("Recipient matrix header exceeds its size bound.")
    return (
        _MAGIC
        + len(header).to_bytes(4, "big")
        + header
        + entity_ids.tobytes(order="C")
        + values.tobytes(order="C")
    )


def decode_recipient_matrix(payload: bytes) -> PreparedRecipientMatrix:
    if not isinstance(payload, bytes) or not payload.startswith(_MAGIC):
        raise ValueError("Invalid recipient matrix envelope.")
    offset = len(_MAGIC)
    if len(payload) < offset + 4:
        raise ValueError("Truncated recipient matrix header length.")
    length = int.from_bytes(payload[offset : offset + 4], "big")
    start = offset + 4
    if length == 0 or length > MATRIX_HEADER_MAX_BYTES or len(payload) < start + length:
        raise ValueError("Recipient matrix header is truncated or exceeds its bound.")
    header = codec.decode_json(payload[start : start + length])
    if (
        set(header) != _FIELDS
        or type(header["schema_version"]) is not int
        or header["schema_version"] != 1
    ):
        raise ValueError("Unsupported recipient matrix schema.")
    entity, rows = header["entity"], header["rows"]
    if (
        not isinstance(entity, str)
        or not entity
        or "." in entity
        or type(rows) is not int
        or rows < 1
    ):
        raise ValueError("Recipient matrix entity/row count is invalid.")
    if not isinstance(header["columns"], list):
        raise ValueError("Recipient matrix columns must be an ordered list.")
    columns = codec.names(tuple(header["columns"]), "matrix columns")
    body = memoryview(payload)[start + length :]
    if len(body) != rows * 8 + rows * len(columns) * 8:
        raise ValueError("Recipient matrix body length differs from its dimensions.")
    ids = np.frombuffer(body[: rows * 8], dtype="<i8")
    index = _index(ids, header["index_name"])
    if codec.encode_json(_index_identity(index).to_dict()) != codec.encode_json(
        header["index_identity"]
    ):
        raise ValueError("Recipient matrix index identity differs from its IDs.")
    values = np.frombuffer(body[rows * 8 :], dtype="<f8").reshape(rows, len(columns))
    if not np.isfinite(values).all():
        raise ValueError("Recipient matrix features must be finite.")
    return PreparedRecipientMatrix(
        entity, pd.DataFrame(values, index=index, columns=columns, copy=False), ids
    )
