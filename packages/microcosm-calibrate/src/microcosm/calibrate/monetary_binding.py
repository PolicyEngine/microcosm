"""Integrity checks for source-receipted monetary constraint rows.

This module deliberately knows nothing about country packages or accounting
concepts.  It verifies that a receipt-bearing target is still attached to the
exact prepared direct column and entity-row ordering it was bound against.
The canonical hashes detect accidental drift; provenance authenticity still
belongs to the signed producer or outer artifact manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import date

import numpy as np
from numpy.typing import ArrayLike

from microcosm.calibrate.target import Target
from microcosm.frame import Frame


class MonetaryBindingIntegrityError(RuntimeError):
    """A monetary receipt cannot safely describe its compiled constraint."""


_RECEIPT_KEYS = frozenset({"monetary_binding", "monetary_binding_sha256"})
_BINDING_MARKER_KEYS = frozenset(
    {"monetary_source_activation_status", "monetary_target_role"}
)
_BINDING_KEYS = frozenset(
    {
        "reference",
        "source_reference",
        "value",
        "source",
        "source_assertion",
        "source_identity_sha256",
        "prepared",
    }
)
_REFERENCE_KEYS = frozenset(
    {
        "name",
        "ledger_fact_key",
        "ledger_source_record_id",
        "ledger_selector",
        "value_operation",
        "entity",
        "measure",
        "filter",
        "period",
        "source",
        "family",
        "signed",
        "se",
        "tolerance",
        "notes",
        "metadata",
        "assertion_policy",
        "period_match_policy",
        "uprating_index",
        "uprating_from_period",
        "uprating_to_period",
        # #834: multi-member references (a sum or difference over declared
        # operands) carry their operand roles and member-count guard in the
        # receipt, so the activated reference stays byte-identical to the
        # declaration that compiled it.
        "value_operands",
        "expected_member_count",
    }
)
_PREPARED_KEYS = frozenset(
    {
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
)
_BASIS_KEYS = frozenset(
    {
        "currency",
        "unit",
        "period",
        "temporal_basis",
        "sector",
        "perimeter",
        "valuation",
    }
)
_BRIDGE_KEYS = frozenset({"factor", "description", "source_sha256"})


def monetary_digest(value: object) -> str:
    """Return a canonical-JSON checksum, not a provenance signature."""
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


def _exact_mapping(value: object, keys: frozenset[str], label: str) -> dict:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    _require(
        actual == keys,
        f"{label} keys differ: expected {sorted(keys)}, got {sorted(actual)}",
    )
    return value


def _validate_basis(value: object) -> dict:
    basis = _exact_mapping(value, _BASIS_KEYS, "monetary basis")
    for key in ("sector", "perimeter", "valuation"):
        _require(
            isinstance(basis[key], str) and basis[key].strip(),
            f"invalid monetary basis {key}",
        )
    _require(
        isinstance(basis["currency"], str)
        and re.fullmatch(r"[A-Z]{3}", basis["currency"]) is not None,
        "invalid monetary currency",
    )
    _require(basis["unit"] == "base_currency", "invalid monetary unit")
    period = basis["period"]
    temporal_basis = basis["temporal_basis"]
    if temporal_basis == "annual_flow":
        _require(
            isinstance(period, str) and re.fullmatch(r"[0-9]{4}", period) is not None,
            "invalid annual-flow period",
        )
        date(int(period), 1, 1)
    elif temporal_basis == "closing_stock":
        _require(
            isinstance(period, str)
            and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", period) is not None,
            "invalid closing-stock period",
        )
        date.fromisoformat(period)
    else:
        raise ValueError("invalid temporal basis")
    return basis


def verify_monetary_binding(target: Target, frame: Frame) -> None:
    """Fail closed if an opted-in monetary target no longer matches its receipt.

    Targets without receipt keys or monetary-binding markers preserve the
    ordinary compiler behavior. A partial receipt, or binding markers whose
    receipt was stripped, is an integrity error rather than an opt-out.
    """
    present_receipt_keys = _RECEIPT_KEYS.intersection(target.metadata)
    if not present_receipt_keys:
        remaining_markers = _BINDING_MARKER_KEYS.intersection(target.metadata)
        if remaining_markers:
            raise MonetaryBindingIntegrityError(
                f"{target.row_name}: monetary binding markers remain without the "
                f"required receipt ({sorted(remaining_markers)})."
            )
        return
    try:
        _require(target.filter is None, "monetary filters are unsupported")
        binding = json.loads(target.metadata["monetary_binding"])
        binding = _exact_mapping(binding, _BINDING_KEYS, "binding")
        _require(
            monetary_digest(binding) == target.metadata["monetary_binding_sha256"],
            "binding hash differs",
        )
        reference = _exact_mapping(
            binding["reference"], _REFERENCE_KEYS, "activated reference receipt"
        )
        source_reference = _exact_mapping(
            binding["source_reference"], _REFERENCE_KEYS, "source reference receipt"
        )
        prepared = _exact_mapping(
            binding["prepared"], _PREPARED_KEYS, "prepared receipt"
        )
        _require(
            {key: value for key, value in reference.items() if key != "metadata"}
            == {
                key: value
                for key, value in source_reference.items()
                if key != "metadata"
            },
            "activated reference differs from its source reference",
        )
        source_metadata = source_reference["metadata"]
        _require(isinstance(source_metadata, dict), "invalid source reference metadata")
        expected_reference_metadata = {
            **source_metadata,
            "activation_status": "active",
            "monetary_source_activation_status": source_metadata.get(
                "activation_status", "ready"
            ),
            "monetary_target_role": "calibration",
        }
        _require(
            reference["metadata"] == expected_reference_metadata,
            "activated reference metadata differs from its source reference",
        )
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
        # These identities deliberately describe different evidence: the binding
        # identity receipts the aggregate fact/value, while the prepared identity
        # receipts the entity-level column used to distribute that aggregate.
        _hash(binding["source_identity_sha256"], "target source identity")
        basis = _validate_basis(prepared["basis"])
        bridge = _exact_mapping(prepared["bridge"], _BRIDGE_KEYS, "transport bridge")
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
        for field in (
            "name",
            "entity",
            "measure",
            "period",
            "tolerance",
            "filter",
        ):
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
                if key not in _RECEIPT_KEYS
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
