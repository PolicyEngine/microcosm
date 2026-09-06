"""Shared versioned QRF envelope and input validation for graph kernels.

The payload includes a trusted local pickle. Integrity checks do not make an
untrusted pickle safe to execute; only consume trusted graph-produced bytes.
"""

from __future__ import annotations

import hashlib
import json
import pickle

from microcosm.fit.qrf import FittedRegimeGatedQRF
from microcosm.graph import ROWS_ALL, ArtifactType, ArtifactValue

QRF_MODEL_TYPE = ArtifactType("microcosm.fit.qrf", 1)
_MAGIC = b"microcosm.fit.qrf/1\n"


def _names(value, label):
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(v, str) or not v for v in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{label} must be a nonempty tuple of distinct column names.")
    return value


def _table(context, ref):
    node = context.node
    if node.kernel != ref or len(node.inputs) != 1:
        raise ValueError(f"{ref} requires exactly one declared input Slice.")
    declared = node.inputs[0]
    if declared.rows != ROWS_ALL:
        raise ValueError(f"{ref} requires an all-row input Slice.")
    table = context.tables[declared.entity]
    id_column = f"{declared.entity}_id"
    if id_column not in table or not set(declared.columns).issubset(table.columns):
        raise ValueError(f"{ref} table is missing declared columns or entity IDs.")
    if table[id_column].isna().any() or table[id_column].duplicated().any():
        raise ValueError(f"{ref} requires non-null unique entity IDs.")
    return declared, table, id_column


def _encode_model(model, source_weight_kind):
    payload = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "schema_version": 1,
        "family": "regime_gated_qrf",
        "predictors": model.predictors,
        "targets": model.targets,
        "regimes": model.regimes(),
        "fit_weight_kind": model.weight_kind,
        "source_weight_kind": source_weight_kind,
        "pickle_sha256": hashlib.sha256(payload).hexdigest(),
    }
    header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    return _MAGIC + len(header).to_bytes(8, "big") + header + payload


def load_qrf_model(artifact: ArtifactValue) -> FittedRegimeGatedQRF:
    """Validate and load a trusted graph-produced model (never untrusted pickle)."""
    if not isinstance(artifact, ArtifactValue) or artifact.type != QRF_MODEL_TYPE:
        raise ValueError("Expected a microcosm.fit.qrf version 1 model artifact.")
    data = artifact.payload
    offset = len(_MAGIC)
    if not data.startswith(_MAGIC) or len(data) < offset + 8:
        raise ValueError("Invalid QRF model artifact envelope.")
    size = int.from_bytes(data[offset : offset + 8], "big")
    offset += 8
    if not 0 < size < len(data) - offset:
        raise ValueError("Invalid QRF model artifact header length.")
    try:
        metadata = json.loads(data[offset : offset + size])
    except (ValueError, UnicodeError) as error:
        raise ValueError("Invalid QRF model artifact metadata.") from error
    payload = data[offset + size :]
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
        or metadata.get("family") != "regime_gated_qrf"
        or metadata.get("pickle_sha256") != hashlib.sha256(payload).hexdigest()
        or metadata.get("source_weight_kind")
        not in {"design", "importance", "calibrated"}
        or metadata.get("fit_weight_kind") != "explicit"
    ):
        raise ValueError("Invalid QRF model artifact metadata or payload digest.")
    # The executor verifies the content-store bytes before this trusted loader.
    model = pickle.loads(payload)  # noqa: S301 - trusted local graph artifact only
    if (
        type(model) is not FittedRegimeGatedQRF
        or model.entity is not None
        or model.predictors != metadata.get("predictors")
        or model.targets != metadata.get("targets")
        or model.regimes() != metadata.get("regimes")
        or model.weight_kind != metadata["fit_weight_kind"]
    ):
        raise ValueError("QRF model object does not match its artifact metadata.")
    _names(tuple(model.predictors), "model predictors")
    _names(tuple(model.targets), "model targets")
    if set(model.predictors) & set(model.targets):
        raise ValueError("QRF model predictors and targets overlap.")
    return model
