"""Portable evidence transport for existing stateful stage adapters.

Capture only their declared checkpoint/evidence hooks after computation. Model
objects, closures and execution timestamps never become numerical artifacts.
Consumers need only this data contract, not a live fitting instance. Once a
stage's evidence is captured, :func:`release_stage_results` lets a run drop
the result the stage kept, output Frame included.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping

from microcosm.graph import ArtifactType

STAGE_EVIDENCE_TYPE = ArtifactType("microcosm.stage-evidence", 1)


def _proxy_chain(transform: object | None) -> Iterator[object]:
    """The object a staged run holds, then each object it wraps.

    A wrapper keeps the object it wraps under ``transform`` in its own
    ``__dict__``. The walk follows that entry and stops at a repeat.
    """

    seen: set[int] = set()
    candidate: object | None = transform
    while candidate is not None and id(candidate) not in seen:
        seen.add(id(candidate))
        yield candidate
        candidate = getattr(candidate, "__dict__", {}).get("transform")


def _fit_weight_hook_holder(transform: object | None) -> object | None:
    """The object declaring ``fit_weight_records``, seen through the proxies.

    Detects the hook without evaluating it (a raising property must count as
    a fitting stage with unreadable records, not vanish), and looks through
    the wrappers a staged run puts around every stage: an observation wrapper
    (telemetry) and a graph source transform both proxy attribute reads
    through ``__getattr__``, which a class-level probe never consults — so a
    probe on the outermost object alone silently loses the block and the
    release-cut weights audit finds no evidence.
    """

    for candidate in _proxy_chain(transform):
        declared = getattr(type(candidate), "fit_weight_records", None) is not None
        if declared or "fit_weight_records" in getattr(candidate, "__dict__", {}):
            return candidate
    return None


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
    holder = _fit_weight_hook_holder(transform)
    if holder is not None:
        try:
            records = tuple(holder.fit_weight_records or ())
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


def release_stage_results(transform: object | None) -> None:
    """Drop the result a stage kept from its last run, once it is snapshotted.

    Call it only after :func:`snapshot_stage_evidence` has read the stage.
    The snapshot is then all a consumer reads, yet a stage keeps its
    ``last_result``, output Frame included, until the run ends. This walks
    the same wrappers as the snapshot and clears ``last_result`` wherever an
    object holds one in its own ``__dict__``. It writes through
    ``object.__setattr__``, so a frozen dataclass is cleared too. A wrapper
    that only proxies the attribute is never written, and everything else
    stays: the wrapped objects, fit-weight records, receipts and sampling.
    """

    for candidate in _proxy_chain(transform):
        if getattr(candidate, "__dict__", {}).get("last_result") is not None:
            object.__setattr__(candidate, "last_result", None)


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
