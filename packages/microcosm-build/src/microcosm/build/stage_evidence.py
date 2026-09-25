"""Portable evidence transport for existing stateful stage adapters.

Capture only their declared checkpoint/evidence hooks after computation. Model
objects, closures and execution timestamps never become numerical artifacts.
Consumers need only this data contract, not a live fitting instance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

from microcosm.graph import ArtifactType

STAGE_EVIDENCE_TYPE = ArtifactType("microcosm.stage-evidence", 1)


def snapshot_stage_evidence(stage: str, transform: object | None) -> dict[str, object]:
    """Snapshot checkpoint, fit-weight and sampling evidence without rerunning."""

    checkpoint = None
    hook = getattr(transform, "checkpoint_metadata", None)
    if callable(hook):
        checkpoint = dict(hook())
        evidence = checkpoint.get("evidence", checkpoint)
    else:
        result = getattr(transform, "last_result", None)
        evidence_hook = getattr(result, "evidence", None)
        evidence = (
            evidence_hook()
            if callable(evidence_hook)
            else result
            if isinstance(result, Mapping)
            else None
        )
    payload: dict[str, object] = {
        "schema_version": STAGE_EVIDENCE_TYPE.schema_version,
        "stage": stage,
        "checkpoint_metadata": checkpoint,
        "evidence": evidence,
        "sampling": getattr(transform, "sampling", None),
    }
    # Do not evaluate a raising property merely to detect whether it exists:
    # missing/unreadable fitting evidence must stay visible to weight audits.
    exposes_records = getattr(type(transform), "fit_weight_records", None) is not None
    exposes_records |= "fit_weight_records" in getattr(transform, "__dict__", {})
    if exposes_records:
        try:
            records = tuple(transform.fit_weight_records or ())
            payload["fit_weight_records"] = [
                {
                    "fit_name": str(record.fit_name),
                    "weight_kind": str(record.weight_kind),
                }
                for record in records
            ]
            payload["fit_weight_records_status"] = "present" if records else "empty"
        except Exception:  # noqa: BLE001 - preserve the existing fail-visible audit
            payload["fit_weight_records"] = []
            payload["fit_weight_records_status"] = "unreadable"
    elif checkpoint is not None and "fit_weight_records" in checkpoint:
        payload["fit_weight_records"] = checkpoint["fit_weight_records"]
        payload["fit_weight_records_status"] = (
            "present" if checkpoint["fit_weight_records"] else "empty"
        )
    return payload


def encode_stage_evidence(payload: Mapping[str, object]) -> bytes:
    """Encode finite JSON; reject nonportable state rather than pickling it."""

    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decode_stage_evidence(payload: bytes, *, stage: str) -> dict[str, object]:
    document = json.loads(payload)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported stage evidence schema.")
    if document.get("stage") != stage:
        raise ValueError("Stored stage evidence has a different stage identity.")
    for key in ("checkpoint_metadata", "evidence", "sampling"):
        if key not in document:
            raise ValueError(f"Stage evidence is missing {key!r}.")
    return document
