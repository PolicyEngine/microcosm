"""Integrity checks for source-receipted monetary constraint rows.

This module deliberately knows nothing about country packages or accounting
concepts.  It verifies that a receipt-bearing target is still attached to the
exact prepared direct column and entity-row ordering it was bound against.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
from numpy.typing import ArrayLike

from microcosm.calibrate.target import Target
from microcosm.frame import Frame


class MonetaryBindingIntegrityError(RuntimeError):
    """A monetary receipt cannot safely describe its compiled constraint."""


def monetary_digest(value: object) -> str:
    """Return the SHA-256 of canonical JSON, rejecting non-finite numbers."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def canonical_record_ids(values: ArrayLike) -> np.ndarray:
    """Return unique, non-lossy little-endian int64 record ids."""
    ids = np.asarray(values)
    if ids.ndim != 1 or not ids.size or ids.dtype.kind not in "iu":
        raise ValueError(
            "record_ids must be a nonempty one-dimensional integer vector."
        )
    if ids.dtype.kind == "u" and ids.max() > np.iinfo(np.int64).max:
        raise ValueError("record_ids must fit signed int64 without overflow.")
    ids = ids.astype("<i8", copy=False)
    if np.unique(ids).size != ids.size:
        raise ValueError("record_ids must uniquely identify entity rows.")
    return ids


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hash(value: object, label: str) -> None:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"invalid {label}",
    )


def verify_monetary_binding(target: Target, frame: Frame) -> None:
    """Fail closed if an opted-in monetary target no longer matches its receipt.

    Targets without either receipt key preserve the ordinary compiler behavior.
    A partial receipt is itself an integrity error, rather than an opt-out.
    """
    receipt_keys = {"monetary_binding", "monetary_binding_sha256"}
    if not receipt_keys.intersection(target.metadata):
        return
    try:
        _require(target.filter is None, "monetary filters are unsupported")
        binding = json.loads(target.metadata["monetary_binding"])
        _require(isinstance(binding, dict), "binding must be an object")
        _require(
            monetary_digest(binding) == target.metadata["monetary_binding_sha256"],
            "binding hash differs",
        )
        reference = binding["reference"]
        prepared = binding["prepared"]
        _require(isinstance(reference, dict), "invalid reference receipt")
        _require(isinstance(prepared, dict), "invalid prepared receipt")
        required = {
            "schema_version",
            "basis",
            "n_records",
            "source_identity_sha256",
            "source_values_sha256",
            "values_sha256",
            "record_ids_sha256",
            "bridge",
            "bridge_sha256",
            "readiness",
            "measure_kind",
            "receipt_sha256",
        }
        _require(required <= prepared.keys(), "incomplete prepared receipt")
        _require(prepared["schema_version"] == 2, "unsupported receipt version")
        raw_receipt = {
            key: value for key, value in prepared.items() if key != "receipt_sha256"
        }
        _require(
            monetary_digest(raw_receipt) == prepared["receipt_sha256"],
            "prepared receipt hash differs",
        )
        _require(
            prepared["readiness"] == "ready" and prepared["measure_kind"] == "direct",
            "measure is not ready/direct",
        )
        _require(
            binding["source_assertion"]
            in {"observation", "observed", "derived_observation"},
            "source is not an observation",
        )
        for key in (
            "source_identity_sha256",
            "source_values_sha256",
            "values_sha256",
            "record_ids_sha256",
        ):
            _hash(prepared[key], key)
        _hash(binding["source_identity_sha256"], "source identity")
        basis = prepared["basis"]
        _require(isinstance(basis, dict), "invalid monetary basis")
        fields = (
            "currency",
            "unit",
            "period",
            "temporal_basis",
            "sector",
            "perimeter",
            "valuation",
        )
        _require(
            all(
                isinstance(basis.get(key), str) and basis[key].strip() for key in fields
            ),
            "incomplete monetary basis",
        )
        _require(basis["unit"] == "base_currency", "invalid monetary unit")
        _require(
            basis["temporal_basis"] in {"annual_flow", "closing_stock"},
            "invalid temporal basis",
        )
        bridge = prepared["bridge"]
        _require(isinstance(bridge, dict), "invalid transport bridge")
        _require(
            monetary_digest(bridge) == prepared["bridge_sha256"], "bridge hash differs"
        )
        _require(
            not isinstance(bridge.get("factor"), bool)
            and math.isfinite(bridge["factor"])
            and bridge["factor"] > 0,
            "invalid transport factor",
        )
        _require(
            isinstance(bridge.get("description"), str)
            and bridge["description"].strip(),
            "missing transport description",
        )
        _hash(bridge.get("source_sha256"), "transport source")
        for field in ("name", "entity", "measure", "period", "filter"):
            expected = reference[field]
            if field == "period" and expected is None:
                expected = basis["period"]
            _require(getattr(target, field) == expected, f"reference {field} differs")
        _require(isinstance(target.measure, str), "monetary rows require named columns")
        _require(
            binding["value"] == target.value and binding["source"] == target.source,
            "target value/source differs",
        )
        receipt_metadata = reference["metadata"]
        _require(isinstance(receipt_metadata, dict), "invalid reference metadata")
        _require(
            {
                key: value
                for key, value in target.metadata.items()
                if key not in receipt_keys
            }
            == receipt_metadata,
            "reference metadata differs",
        )
        _require(
            receipt_metadata.get("monetary_target_role") == "calibration"
            and receipt_metadata.get("measure_kind", "direct")
            in {"direct", "direct_column", "prepared_column"}
            and receipt_metadata.get("activation_status") == "active",
            "reference is not a ready direct calibration target",
        )
        table = frame.table(target.entity)
        values = table[target.measure].to_numpy(dtype="<f8")
        ids = canonical_record_ids(
            table[frame.schema.entity_id_column(target.entity)].to_numpy()
        )
        _require(
            type(prepared["n_records"]) is int and prepared["n_records"] > 0,
            "invalid record count",
        )
        _require(
            values.ndim == 1 and len(values) == prepared["n_records"],
            "record length differs",
        )
        _require(np.isfinite(values).all(), "prepared values are non-finite")
        _require(
            hashlib.sha256(values.tobytes()).hexdigest() == prepared["values_sha256"],
            "prepared values differ from frame column",
        )
        _require(
            hashlib.sha256(ids.tobytes()).hexdigest() == prepared["record_ids_sha256"],
            "prepared record IDs/order differ from frame",
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        json.JSONDecodeError,
    ) as exc:
        raise MonetaryBindingIntegrityError(
            f"{target.row_name}: monetary binding integrity failed ({exc})."
        ) from exc
