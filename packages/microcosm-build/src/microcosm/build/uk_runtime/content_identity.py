"""Content identity for UK national frames (#612 increment 3).

The pre-increment-3 descent fences compared Python object identity
(``retained.frame is frame``), which only holds while every stage runs in
one process on the very same objects. Content identity makes the same
guarantee survive a process boundary: two frames carry the same identity
exactly when their schema, tables (column order, dtypes, index, values),
typed weights, mass log, and metadata agree. A checkpoint-rehydrated frame
is content-identical to the frame that was checkpointed, so resumable
staged builds can keep the certified-candidate descent fence.

This is deliberately not
:func:`microcosm.build.outer_stage_runtime.frame_identity`: that identity is
structural only (ids, memberships, provenance columns) and ignores payload
values and weights — sufficient for row-order guarantees, too weak for a
substitution fence.

Two boundaries the digest declares rather than hides. Identities are
comparable only within a pinned environment: the table bytes ride on
``pd.util.hash_pandas_object``, which is not guaranteed stable across
pandas versions — the UK run config pins the environment via
``builder_code_identity``, and that pin is what makes cross-process
comparison sound; do not reuse this as a cross-version artifact identity.
And the digest header carries a version (``:v1``): bump it whenever the
covered surface changes, so digests from different definitions can never
compare equal by accident.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from microcosm.frame import Frame

__all__ = ["uk_frame_content_identity"]

_IDENTITY_HEADER = "microcosm-uk-frame-content-identity:v1"


def uk_frame_content_identity(frame: Frame) -> str:
    """Return a sha256 hex digest over the frame's full content.

    Covers, in deterministic order: the entity set, each entity table's
    column order, dtypes, index, and cell values, each weighted entity's
    typed weight kind and vector, the strata labels, the mass log, and the
    frame metadata. Structural-only changes (a renamed column, a reordered
    column) move the identity just as value changes do. Link tables are
    outside the digest: the UK national schema declares none
    (``validate_uk_national_frame`` enforces linklessness) — extend this
    before reusing it on a linked schema.
    """

    if not isinstance(frame, Frame):
        raise TypeError("content identity requires a microcosm Frame.")
    digest = hashlib.sha256()
    digest.update(_IDENTITY_HEADER.encode("utf-8"))
    for entity in frame.entities:
        table = frame.table(entity)
        digest.update(f"\x00entity\x1f{entity}".encode())
        digest.update(
            json.dumps(
                {
                    "columns": [str(column) for column in table.columns],
                    "dtypes": [str(dtype) for dtype in table.dtypes],
                    "index_dtype": str(table.index.dtype),
                    "rows": int(len(table)),
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        row_hashes = pd.util.hash_pandas_object(table, index=True)
        digest.update(np.ascontiguousarray(row_hashes.to_numpy()).tobytes())
    for entity in frame.weighted_entities:
        weights = frame.weights_for(entity)
        digest.update(f"\x00weights\x1f{entity}\x1f{weights.kind.name}".encode())
        digest.update(np.ascontiguousarray(weights.values, dtype=np.float64).tobytes())
    strata = frame.strata
    digest.update(b"\x00strata")
    digest.update(
        np.ascontiguousarray(
            pd.util.hash_pandas_object(strata, index=True).to_numpy()
        ).tobytes()
    )
    mass_log_payload = [
        {
            "entity": record.entity,
            "old_total": record.old_total,
            "new_total": record.new_total,
            "declared_factor": record.declared_factor,
            "reason": record.reason,
        }
        for record in frame.mass_log
    ]
    digest.update(b"\x00mass_log")
    try:
        digest.update(
            json.dumps(mass_log_payload, sort_keys=True, allow_nan=False).encode(
                "utf-8"
            )
        )
    except ValueError as exc:
        raise ValueError(
            "the frame cannot be content-identified: its mass log carries a "
            "non-finite value."
        ) from exc
    digest.update(b"\x00metadata")
    try:
        digest.update(
            json.dumps(
                _jsonable_metadata(frame.metadata),
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        )
    except ValueError as exc:
        raise ValueError(
            "the frame cannot be content-identified: its metadata carries a "
            "non-finite value."
        ) from exc
    return digest.hexdigest()


def _jsonable_metadata(value: object) -> object:
    """Coerce frozen frame metadata into a canonically serializable shape.

    Set members sort by ``repr`` — the same canonical order
    ``stage_checkpoints._thawed`` uses, so a set that rides a checkpoint
    round trip (JSON has no set type; it returns as a sequence) keeps its
    content identity. Anything outside the JSON-shaped vocabulary is
    refused: digesting an arbitrary object's ``repr`` could fold a memory
    address into the identity, making it differ across processes in exactly
    the dimension the fence exists to make robust.
    """

    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _jsonable_metadata(item) for key, item in value.items()}
    if isinstance(value, str | bool | int | float) or value is None:
        return value
    if isinstance(value, tuple | list | set | frozenset):
        items = [_jsonable_metadata(item) for item in value]
        if isinstance(value, set | frozenset):
            return sorted(items, key=repr)
        return items
    raise TypeError(
        f"frame metadata value of type {type(value).__name__} cannot be "
        "content-identified; metadata must be composed of mappings, "
        "sequences, sets, and scalar values."
    )
