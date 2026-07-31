"""Bounded loading for the input-complete ASEC pre-clone checkpoint.

The checkpoint is produced by the outer-stage runtime after ASEC-only input
enrichment.  This module validates that artifact binding and the loaded
``Frame`` boundary; it does not compute a whole-file digest.  Callers that
pin input files own that separate provenance check.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from populace.build.frame_checkpoint import load_frame_checkpoint
from populace.build.outer_stage_runtime import (
    OUTER_STAGE_CONTEXT_SCHEMA_VERSION,
    FrameIdentity,
    frame_identity,
)
from populace.frame import US_SCHEMA, Frame, WeightKind, Weights

__all__ = ["load_asec_pre_clone_checkpoint"]

_OUTER_STAGE_ARTIFACT_KIND = "populace_outer_stage_frame"
_PRE_CLONE_STAGE = "pre_clone_enrichment"
_PRE_CLONE_STAGE_INDEX = 1
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_BINDING_KEYS = frozenset(
    {
        "artifact_kind",
        "identity",
        "pipeline_sha256",
        "schema_version",
        "stage",
        "stage_index",
    }
)


def load_asec_pre_clone_checkpoint(
    path: str | Path,
) -> tuple[Frame, dict[str, object]]:
    """Load and validate one input-complete ASEC pre-clone checkpoint.

    Args:
        path: Explicit outer-stage Frame checkpoint produced for
            ``pre_clone_enrichment``.

    Returns:
        The restored US ``Frame`` and its canonical, JSON-ready outer-stage
        checkpoint binding.  The metadata contains no newly computed file
        digest.

    Raises:
        FileNotFoundError: If ``path`` does not name a checkpoint file.
        ValueError: If the artifact is not bound to the required outer stage,
            its stored identity differs from the loaded ``Frame``, or the
            frame lacks the required US schema and positive household-only
            typed weights.
    """

    checkpoint_path = Path(path)
    loaded = load_frame_checkpoint(checkpoint_path)
    metadata = _validate_outer_stage_binding(
        loaded.metadata,
        path=checkpoint_path,
    )
    stored_identity = FrameIdentity.from_payload(
        metadata["identity"],
        label="ASEC pre-clone checkpoint identity",
    )
    actual_identity = frame_identity(loaded.frame)
    if actual_identity != stored_identity:
        raise ValueError(
            f"ASEC pre-clone checkpoint {checkpoint_path} Frame identity changed."
        )
    _validate_asec_frame(loaded.frame, path=checkpoint_path)
    metadata["identity"] = stored_identity.to_payload()
    return loaded.frame, metadata


def _validate_outer_stage_binding(
    metadata: dict[str, object],
    *,
    path: Path,
) -> dict[str, object]:
    actual_keys = frozenset(metadata)
    if actual_keys != _BINDING_KEYS:
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} has an incomplete outer-stage "
            f"artifact binding (missing: {sorted(_BINDING_KEYS - actual_keys)}; "
            f"extra: {sorted(actual_keys - _BINDING_KEYS)})."
        )
    if metadata["artifact_kind"] != _OUTER_STAGE_ARTIFACT_KIND:
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} is not an outer-stage Frame artifact."
        )
    if metadata["schema_version"] != OUTER_STAGE_CONTEXT_SCHEMA_VERSION or isinstance(
        metadata["schema_version"], bool
    ):
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} has an unsupported outer-stage "
            "schema version."
        )
    if metadata["stage"] != _PRE_CLONE_STAGE:
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} must be bound to stage "
            f"{_PRE_CLONE_STAGE!r}, got {metadata['stage']!r}."
        )
    if metadata["stage_index"] != _PRE_CLONE_STAGE_INDEX or isinstance(
        metadata["stage_index"], bool
    ):
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} must be bound to stage_index "
            f"{_PRE_CLONE_STAGE_INDEX}, got {metadata['stage_index']!r}."
        )
    pipeline_sha256 = metadata["pipeline_sha256"]
    if not isinstance(pipeline_sha256, str) or not _LOWERCASE_SHA256.fullmatch(
        pipeline_sha256
    ):
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} pipeline_sha256 must be a "
            "lowercase SHA-256 digest."
        )
    return dict(metadata)


def _validate_asec_frame(frame: Frame, *, path: Path) -> None:
    if frame.schema != US_SCHEMA:
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} must use the US entity schema."
        )
    if frame.weighted_entities != ("household",):
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} must carry household weights only; "
            f"got weighted entities {list(frame.weighted_entities)}."
        )
    weights = frame.weights_for("household")
    if not isinstance(weights, Weights) or not isinstance(weights.kind, WeightKind):
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} household weights must be typed."
        )
    values = weights.values
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} household weights must be "
            "strictly positive and finite."
        )
