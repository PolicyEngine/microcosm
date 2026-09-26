"""Strict local artifact codecs for the legacy per-target graph protocol."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from microcosm.fit import qrf
from microcosm.fit.qrf_target import LegacyQRFTrainingState
from microcosm.frame import EntitySchema, Frame
from microcosm.graph import ROWS_ALL, ArtifactType, Numeric, platform_fingerprint

TRAINING_STATE_TYPE = ArtifactType("microcosm.fit.qrf.legacy_training_state", 1)
APPLY_STATE_TYPE = ArtifactType("microcosm.fit.qrf.legacy_apply_state", 1)
RAW_TARGET_TYPE = ArtifactType("microcosm.fit.qrf.legacy_raw_target", 1)
_RAW_MAGIC = b"microcosm.fit.qrf.legacy_raw_target/1\n"


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encode_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def decode_json(payload: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Repeated legacy graph checkpoint JSON key.")
            result[key] = value
        return result

    def nonfinite(_):
        raise ValueError("Nonfinite legacy graph checkpoint JSON number.")

    value = json.loads(payload, object_pairs_hook=pairs, parse_constant=nonfinite)
    if not isinstance(value, dict) or encode_json(value) != payload:
        raise ValueError("Legacy graph checkpoint must be a canonical JSON object.")
    return value


def names(value, label):
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(name, str) or not name for name in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must be a nonempty tuple of unique names.")
    return value


def table(context, ref):
    node = context.node
    if node.kernel != ref or len(node.inputs) != 1 or node.inputs[0].rows != ROWS_ALL:
        raise ValueError("Legacy QRF requires one all-row encoded input Slice.")
    if node.outputs or node.sources:
        raise ValueError(
            "Legacy QRF target kernels own artifacts only and read declared slices."
        )
    declared = node.inputs[0]
    result = context.tables[declared.entity]
    if not set(declared.columns).issubset(result):
        raise ValueError("Legacy QRF table lacks declared input columns.")
    if not isinstance(context.params.get("phase"), str) or not context.params["phase"]:
        raise ValueError("Legacy QRF node requires its explicit outer phase.")
    return declared, result


def model_frame(context, declared, values):
    """Retain exact table index and resolved weight kind, with minimal links."""
    entity = declared.entity
    table = values.loc[:, list(declared.columns)].copy()
    ids = values[f"{entity}_id"].to_numpy(copy=True)
    table.insert(0, f"{entity}_id", ids)
    weights = context.weights[entity]
    if len(weights) != len(table):
        raise ValueError("Legacy QRF donor weights must align to input rows.")
    if entity == "person":
        table.insert(1, "person_model_unit_id", ids)
        schema = EntitySchema(group_entities=("model_unit",))
        tables = {"person": table, "model_unit": pd.DataFrame({"model_unit_id": ids})}
    else:
        schema = EntitySchema(group_entities=(entity,))
        tables = {
            entity: table,
            "person": pd.DataFrame({"person_id": ids, f"person_{entity}_id": ids}),
        }
    return Frame(tables, schema, {entity: weights})


def artifact(context, name, type_):
    value = context.artifacts[name]
    declared = [item for item in context.node.artifact_inputs if item.name == name]
    if len(declared) != 1 or declared[0].type != type_ or value.type != type_:
        raise ValueError(f"Wrong typed legacy QRF artifact: {name}.")
    if (
        value.numerics.numeric is not Numeric.PLATFORM_BITWISE
        or value.numerics.platform != platform_fingerprint()
    ):
        raise ValueError(
            f"Legacy QRF {name} artifact requires this platform's bitwise scope."
        )
    return value


def _hash(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def history(value, targets, *, models):
    if not isinstance(value, list) or len(value) != len(targets):
        raise ValueError("Legacy QRF history length differs from target prefix.")
    fields = {"target", "sha256", "training_id"} if models else {"target", "sha256"}
    for record, target in zip(value, targets, strict=True):
        if (
            not isinstance(record, dict)
            or set(record) != fields
            or record["target"] != target
        ):
            raise ValueError(
                "Legacy QRF history must bind the exact ordered target prefix."
            )
        if any(not _hash(record[key]) for key in fields - {"target"}):
            raise ValueError("Legacy QRF history has an invalid digest.")
    return value


def read_training(payload):
    packet = decode_json(payload)
    if (
        set(packet) != {"schema_version", "state", "models"}
        or type(packet["schema_version"]) is not int
        or packet["schema_version"] != 1
    ):
        raise ValueError("Invalid legacy QRF graph training checkpoint.")
    state = LegacyQRFTrainingState.from_dict(packet["state"])
    history(packet["models"], state.to_dict()["completed_targets"], models=True)
    return packet, state


def read_application(payload):
    packet = decode_json(payload)
    if (
        set(packet) != {"schema_version", "state", "seed", "models", "raw_targets"}
        or type(packet["schema_version"]) is not int
        or packet["schema_version"] != 1
    ):
        raise ValueError("Invalid legacy QRF graph application checkpoint.")
    if type(packet["seed"]) is not int or packet["seed"] < 0:
        raise ValueError("Invalid legacy QRF application seed.")
    state = qrf.QRFChainState.from_dict(packet["state"])
    history(packet["models"], state.completed_targets, models=True)
    history(packet["raw_targets"], state.completed_targets, models=False)
    return packet, state


def encode_raw_target(values, *, target, index):
    values = np.asarray(values, dtype="<f8")
    if values.shape != (len(index),) or not np.isfinite(values).all():
        raise ValueError("Raw legacy target must be aligned finite float64 draws.")
    metadata = {"target": target, "index": qrf._index_identity(index).to_dict()}
    return _RAW_MAGIC + encode_json(metadata) + b"\n" + values.tobytes()


def read_raw_target(payload: bytes, *, target: str, index: pd.Index) -> np.ndarray:
    """Validate raw target/index binding and return readonly exact float64 bits.

    The caller must obtain bytes from its authenticated graph artifact edge.
    These raw draws are never the decoded or calibrated production columns.
    """
    if not isinstance(payload, bytes) or not payload.startswith(_RAW_MAGIC):
        raise ValueError("Invalid raw legacy target envelope.")
    metadata, separator, body = payload[len(_RAW_MAGIC) :].partition(b"\n")
    expected = {"target": target, "index": qrf._index_identity(index).to_dict()}
    if (
        not separator
        or decode_json(metadata) != expected
        or len(body) != len(index) * 8
    ):
        raise ValueError("Raw legacy target/index/order binding changed.")
    result = np.frombuffer(body, dtype="<f8")
    if not np.isfinite(result).all():
        raise ValueError("Raw legacy target contains nonfinite values.")
    return result
