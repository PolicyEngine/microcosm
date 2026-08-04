"""Identity-bound per-target checkpoints for the ACS transfer QRF chain."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import h5py
import numpy as np

from populace.build.us_runtime.acs_transfer import (
    AcsTransferBankPatternStep,
    AcsTransferTargetCheckpoint,
)
from populace.fit import QRFChainState

ACS_TRANSFER_TARGET_BANK_SCHEMA_VERSION = 1
"""Serialization contract for one ACS-transfer target checkpoint."""

# ACS-transfer target-bank semantic-invalidation ledger.
#
# 1: Initial identity-bound per-target checkpoint format.
# 2: The inherited pool identity now binds the complete canonical take-up
#    contract. Version-1 banks are deliberately stale even though the H5
#    serialization shape is unchanged: correctness takes priority over warmth.
ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION = 2

ACS_TRANSFER_TARGET_BANK_ARTIFACT_KIND = (
    "populace_us_multispine_acs_transfer_target_checkpoint"
)
ACS_TRANSFER_TARGET_BANK_RECEIPT_ARTIFACT_KIND = (
    "populace_us_multispine_acs_transfer_target_bank_provenance"
)

_METADATA_DATASET = "metadata_json"
_RAW_DRAW_BITS_DATASET = "raw_draw_bits"
_TARGETS_DIRNAME = "targets"
_LOWERCASE_SHA256_LENGTH = 64

__all__ = [
    "ACS_TRANSFER_TARGET_BANK_ARTIFACT_KIND",
    "ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION",
    "ACS_TRANSFER_TARGET_BANK_RECEIPT_ARTIFACT_KIND",
    "ACS_TRANSFER_TARGET_BANK_SCHEMA_VERSION",
    "AcsTransferTargetBankStore",
]


class AcsTransferTargetBankStore:
    """Atomic, identity-guarded storage for ordered ACS model targets."""

    def __init__(
        self,
        root: str | Path,
        *,
        identity: Mapping[str, object],
    ) -> None:
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(
                "ACS transfer target-bank root exists but is not a directory: "
                f"{self.root}."
            )
        normalized_identity = _normalize_identity(identity)
        self._identity = normalized_identity
        self._identity_sha256 = _mapping_sha256(normalized_identity)
        self._attempts: dict[int, dict[str, object]] = {}
        self._writes: dict[int, dict[str, object]] = {}
        self._descriptors: dict[int, dict[str, object]] = {}

    @property
    def identity_sha256(self) -> str:
        """SHA-256 of the complete normalized checkpoint identity."""

        return self._identity_sha256

    def target_path(self, target_index: int, model_target: str) -> Path:
        """Return the safe, deterministic path for one ordered model target."""

        if (
            not isinstance(target_index, int)
            or isinstance(target_index, bool)
            or target_index < 0
        ):
            raise ValueError(
                "ACS transfer target_index must be a non-negative integer, got "
                f"{target_index!r}."
            )
        if not isinstance(model_target, str) or not model_target:
            raise ValueError("ACS transfer model_target must be a non-empty string.")
        safe_target = "".join(
            character
            if character.isascii() and (character.isalnum() or character in "-_")
            else "_"
            for character in model_target
        )
        return self.root / _TARGETS_DIRNAME / f"{target_index:03d}__{safe_target}.h5"

    def load_target(
        self,
        *,
        target_index: int,
        total_targets: int,
        entity: str,
        family: str,
        family_targets: tuple[str, ...],
        model_targets: tuple[str, ...],
        model_target: str,
        exported_targets: tuple[str, ...],
        recipient_rows: int,
        expected_states: Mapping[str, QRFChainState],
    ) -> AcsTransferTargetCheckpoint | None:
        """Load one valid target, returning ``None`` for rebuildable artifacts."""

        descriptor = _target_descriptor(
            target_index=target_index,
            total_targets=total_targets,
            entity=entity,
            family=family,
            family_targets=family_targets,
            model_targets=model_targets,
            model_target=model_target,
            exported_targets=exported_targets,
        )
        _validate_recipient_rows(recipient_rows)
        _validate_expected_states(
            expected_states,
            model_targets=model_targets,
            model_target=model_target,
        )
        self._remember_descriptor(descriptor)
        path = self.target_path(target_index, model_target)

        if not path.exists():
            torn_temporaries = sorted(
                str(candidate.resolve())
                for candidate in path.parent.glob(f".{path.name}*.tmp")
                if candidate.exists()
            )
            if torn_temporaries:
                return self._invalid(
                    descriptor,
                    path,
                    reason="incomplete_checkpoint",
                    error=ValueError(
                        "orphan target-checkpoint temporary file(s) remain: "
                        f"{torn_temporaries}"
                    ),
                )
            self._attempts[target_index] = {
                "load_status": "missing",
                "path": str(path.resolve()),
            }
            return None
        if not path.is_file():
            return self._invalid(
                descriptor,
                path,
                reason="incomplete_checkpoint",
                error=ValueError("target checkpoint must be a regular file"),
            )

        try:
            metadata, raw_bits = _read_checkpoint(path)
            observed_identity = metadata.get("identity")
            if observed_identity != self._identity:
                return self._identity_mismatch(
                    descriptor,
                    path,
                    observed_identity=observed_identity,
                )

            _validate_metadata_envelope(
                metadata,
                identity=self._identity,
                identity_sha256=self._identity_sha256,
                descriptor=descriptor,
                recipient_rows=recipient_rows,
            )
            if len(raw_bits) != recipient_rows:
                raise ValueError(
                    "ACS transfer target raw draw row count changed: "
                    f"got {len(raw_bits)}, expected {recipient_rows}."
                )
            actual_raw_sha256 = hashlib.sha256(raw_bits.tobytes()).hexdigest()
            if metadata.get("raw_draw_sha256") != actual_raw_sha256:
                raise ValueError("ACS transfer target raw-draw SHA-256 digest changed.")
            pattern_steps = _load_pattern_steps(
                metadata.get("pattern_steps"),
                expected_states=expected_states,
                model_target=model_target,
            )
            raw_draw = raw_bits.view("<f8").astype(np.float64, copy=False)
            checkpoint = AcsTransferTargetCheckpoint(
                target_index=target_index,
                total_targets=total_targets,
                entity=entity,
                family=family,
                family_targets=family_targets,
                model_targets=model_targets,
                model_target=model_target,
                exported_targets=exported_targets,
                raw_draw=raw_draw,
                pattern_steps=pattern_steps,
            )
            checkpoint_sha256 = _file_sha256(path)
            self._attempts[target_index] = {
                "load_status": "resumed",
                "path": str(path.resolve()),
                "checkpoint_sha256": checkpoint_sha256,
                "size_bytes": path.stat().st_size,
                "raw_draw_sha256": actual_raw_sha256,
                "content_metadata_sha256": metadata["content_metadata_sha256"],
            }
            print(
                "Resumed ACS transfer target "
                f"{target_index + 1}/{total_targets} "
                f"{entity}/{family}/{model_target} from {path}."
            )
            return checkpoint
        except Exception as error:
            return self._invalid(
                descriptor,
                path,
                reason="checkpoint_validation_failed",
                error=error,
            )

    def write_target(self, checkpoint: AcsTransferTargetCheckpoint) -> None:
        """Atomically persist one completed target and its QRF transitions."""

        if not isinstance(checkpoint, AcsTransferTargetCheckpoint):
            raise TypeError(
                "checkpoint must be an AcsTransferTargetCheckpoint, got "
                f"{type(checkpoint).__name__}."
            )
        descriptor = _target_descriptor(
            target_index=checkpoint.target_index,
            total_targets=checkpoint.total_targets,
            entity=checkpoint.entity,
            family=checkpoint.family,
            family_targets=checkpoint.family_targets,
            model_targets=checkpoint.model_targets,
            model_target=checkpoint.model_target,
            exported_targets=checkpoint.exported_targets,
        )
        _validate_pattern_steps_for_write(checkpoint)
        self._remember_descriptor(descriptor)

        draw = np.ascontiguousarray(checkpoint.raw_draw, dtype="<f8")
        raw_bits = draw.view("<u8")
        raw_draw_sha256 = hashlib.sha256(raw_bits.tobytes()).hexdigest()
        metadata: dict[str, object] = {
            "artifact_kind": ACS_TRANSFER_TARGET_BANK_ARTIFACT_KIND,
            "schema_version": ACS_TRANSFER_TARGET_BANK_SCHEMA_VERSION,
            "materializer_version": ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION,
            "identity": self._identity,
            "identity_sha256": self._identity_sha256,
            "target": descriptor,
            "recipient_rows": len(draw),
            "raw_draw_sha256": raw_draw_sha256,
            "pattern_steps": [
                {
                    "pattern": step.pattern,
                    "state_before_sha256": _mapping_sha256(step.state_before.to_dict()),
                    "state_after": step.state_after.to_dict(),
                }
                for step in checkpoint.pattern_steps
            ],
        }
        metadata["content_metadata_sha256"] = _mapping_sha256(metadata)

        path = self.target_path(checkpoint.target_index, checkpoint.model_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.perf_counter()
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
            with h5py.File(temporary_path, mode="w") as h5:
                h5.create_dataset(
                    _METADATA_DATASET,
                    data=np.frombuffer(
                        _canonical_json(metadata).encode("utf-8"),
                        dtype=np.uint8,
                    ),
                    dtype=np.uint8,
                    track_times=False,
                )
                h5.create_dataset(
                    _RAW_DRAW_BITS_DATASET,
                    data=raw_bits,
                    dtype="<u8",
                    track_times=False,
                )
                h5.flush()
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
            _fsync_parent_directory(path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        checkpoint_sha256 = _file_sha256(path)
        self._writes[checkpoint.target_index] = {
            "write_status": "rebuilt",
            "path": str(path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "size_bytes": path.stat().st_size,
            "raw_draw_sha256": raw_draw_sha256,
            "content_metadata_sha256": metadata["content_metadata_sha256"],
            "write_seconds": time.perf_counter() - started_at,
        }
        print(
            "Rebuilt ACS transfer target "
            f"{checkpoint.target_index + 1}/{checkpoint.total_targets} "
            f"{checkpoint.entity}/{checkpoint.family}/{checkpoint.model_target}; "
            f"wrote checkpoint {path}."
        )

    def receipt(self) -> dict[str, object]:
        """Return named load/rebuild evidence for every attempted target."""

        targets: dict[str, dict[str, object]] = {}
        for target_index in sorted(
            set(self._descriptors) | set(self._attempts) | set(self._writes)
        ):
            attempt = dict(
                self._attempts.get(target_index, {"load_status": "not_attempted"})
            )
            written = self._writes.get(target_index)
            source = (
                "rebuilt"
                if written is not None
                else (
                    "checkpoint"
                    if attempt.get("load_status") == "resumed"
                    else "unresolved"
                )
            )
            record: dict[str, object] = {
                "source": source,
                "descriptor": self._descriptors.get(target_index),
                **attempt,
            }
            if written is not None:
                record.update(written)
            targets[str(target_index)] = record
        payload = {
            "artifact_kind": ACS_TRANSFER_TARGET_BANK_RECEIPT_ARTIFACT_KIND,
            "schema_version": ACS_TRANSFER_TARGET_BANK_SCHEMA_VERSION,
            "materializer_version": ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION,
            "root": str(self.root.resolve()),
            "identity": self._identity,
            "identity_sha256": self._identity_sha256,
            "targets": targets,
        }
        normalized = _json_ready(payload)
        if not isinstance(normalized, dict):  # pragma: no cover - fixed mapping
            raise TypeError("ACS transfer bank receipt must normalize to an object.")
        return normalized

    def _remember_descriptor(self, descriptor: Mapping[str, object]) -> None:
        target_index = descriptor["target_index"]
        if not isinstance(target_index, int):  # pragma: no cover - validated earlier
            raise TypeError(
                "ACS transfer target index did not normalize to an integer."
            )
        previous = self._descriptors.get(target_index)
        normalized = dict(descriptor)
        if previous is not None and previous != normalized:
            raise ValueError(
                f"ACS transfer target index {target_index} was reused with a "
                "different descriptor."
            )
        self._descriptors[target_index] = normalized

    def _identity_mismatch(
        self,
        descriptor: Mapping[str, object],
        path: Path,
        *,
        observed_identity: object,
    ) -> None:
        target_index = descriptor["target_index"]
        if not isinstance(target_index, int):  # pragma: no cover - validated earlier
            raise TypeError("ACS transfer target index must be an integer.")
        observed_sha256 = (
            _mapping_sha256(observed_identity)
            if isinstance(observed_identity, Mapping)
            else None
        )
        self._attempts[target_index] = {
            "load_status": "identity_mismatch",
            "path": str(path.resolve()),
            "ignored_checkpoint": {
                "reason": "identity_mismatch",
                "expected_identity_sha256": self._identity_sha256,
                "observed_identity_sha256": observed_sha256,
            },
        }
        print(
            "Ignored stale ACS transfer target "
            f"{target_index + 1}/{descriptor['total_targets']} "
            f"{descriptor['entity']}/{descriptor['family']}/"
            f"{descriptor['model_target']} at {path}: identity "
            f"{observed_sha256!r} != {self._identity_sha256}; rebuilding."
        )
        return None

    def _invalid(
        self,
        descriptor: Mapping[str, object],
        path: Path,
        *,
        reason: str,
        error: Exception,
    ) -> None:
        target_index = descriptor["target_index"]
        if not isinstance(target_index, int):  # pragma: no cover - validated earlier
            raise TypeError("ACS transfer target index must be an integer.")
        failure = {
            "reason": reason,
            "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "path": str(path.resolve()),
        }
        self._attempts[target_index] = {
            "load_status": "invalid_rebuild",
            "path": str(path.resolve()),
            "invalid_checkpoint": failure,
        }
        print(
            "Ignored corrupt ACS transfer target "
            f"{target_index + 1}/{descriptor['total_targets']} "
            f"{descriptor['entity']}/{descriptor['family']}/"
            f"{descriptor['model_target']} at {path}: "
            f"{type(error).__name__}: {error}; rebuilding."
        )
        return None


def _read_checkpoint(path: Path) -> tuple[dict[str, object], np.ndarray]:
    with h5py.File(path, mode="r") as h5:
        if set(h5) != {_METADATA_DATASET, _RAW_DRAW_BITS_DATASET}:
            raise ValueError(
                "ACS transfer target checkpoint datasets must be exactly "
                f"{sorted((_METADATA_DATASET, _RAW_DRAW_BITS_DATASET))}."
            )
        metadata_dataset = h5[_METADATA_DATASET]
        raw_dataset = h5[_RAW_DRAW_BITS_DATASET]
        if metadata_dataset.ndim != 1 or metadata_dataset.dtype != np.dtype(np.uint8):
            raise ValueError(
                "ACS transfer target metadata_json must be one-dimensional uint8."
            )
        if raw_dataset.ndim != 1 or raw_dataset.dtype != np.dtype("<u8"):
            raise ValueError(
                "ACS transfer target raw_draw_bits must be one-dimensional "
                "little-endian uint64."
            )
        metadata_bytes = np.asarray(metadata_dataset, dtype=np.uint8).tobytes()
        raw_bits = np.asarray(raw_dataset, dtype="<u8")
    try:
        metadata = json.loads(metadata_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"ACS transfer target metadata JSON is invalid: {path}."
        ) from error
    if not isinstance(metadata, dict):
        raise ValueError("ACS transfer target metadata JSON must contain an object.")
    return metadata, raw_bits


def _validate_metadata_envelope(
    metadata: Mapping[str, object],
    *,
    identity: Mapping[str, object],
    identity_sha256: str,
    descriptor: Mapping[str, object],
    recipient_rows: int,
) -> None:
    expected_keys = {
        "artifact_kind",
        "schema_version",
        "materializer_version",
        "identity",
        "identity_sha256",
        "target",
        "recipient_rows",
        "raw_draw_sha256",
        "pattern_steps",
        "content_metadata_sha256",
    }
    if set(metadata) != expected_keys:
        raise ValueError(
            "ACS transfer target metadata keys changed: expected "
            f"{sorted(expected_keys)}, got {sorted(metadata)}."
        )
    if (
        metadata.get("artifact_kind") != ACS_TRANSFER_TARGET_BANK_ARTIFACT_KIND
        or metadata.get("schema_version") != ACS_TRANSFER_TARGET_BANK_SCHEMA_VERSION
        or metadata.get("materializer_version")
        != ACS_TRANSFER_TARGET_BANK_MATERIALIZER_VERSION
    ):
        raise ValueError("ACS transfer target checkpoint has an unsupported binding.")
    if metadata.get("identity") != identity:
        raise ValueError("ACS transfer target embedded identity changed.")
    if metadata.get("identity_sha256") != identity_sha256:
        raise ValueError("ACS transfer target identity SHA-256 changed.")
    if _mapping_sha256(identity) != identity_sha256:
        raise ValueError("ACS transfer target expected identity SHA-256 is invalid.")
    if metadata.get("target") != descriptor:
        raise ValueError("ACS transfer target descriptor changed.")
    if metadata.get("recipient_rows") != recipient_rows:
        raise ValueError(
            "ACS transfer target recipient row binding changed: got "
            f"{metadata.get('recipient_rows')!r}, expected {recipient_rows}."
        )
    raw_digest = metadata.get("raw_draw_sha256")
    if not _is_lowercase_sha256(raw_digest):
        raise ValueError("ACS transfer target raw_draw_sha256 is malformed.")
    content_digest = metadata.get("content_metadata_sha256")
    if not _is_lowercase_sha256(content_digest):
        raise ValueError("ACS transfer target content metadata digest is malformed.")
    content = {
        key: value
        for key, value in metadata.items()
        if key != "content_metadata_sha256"
    }
    if _mapping_sha256(content) != content_digest:
        raise ValueError("ACS transfer target content metadata digest changed.")


def _load_pattern_steps(
    value: object,
    *,
    expected_states: Mapping[str, QRFChainState],
    model_target: str,
) -> tuple[AcsTransferBankPatternStep, ...]:
    if not isinstance(value, list):
        raise ValueError("ACS transfer target pattern_steps must be a list.")
    expected_patterns = tuple(expected_states)
    observed_patterns: list[str] = []
    steps: list[AcsTransferBankPatternStep] = []
    for index, raw_step in enumerate(value):
        if not isinstance(raw_step, Mapping):
            raise ValueError(
                f"ACS transfer target pattern_steps[{index}] must be an object."
            )
        expected_keys = {"pattern", "state_before_sha256", "state_after"}
        if set(raw_step) != expected_keys:
            raise ValueError(
                f"ACS transfer target pattern_steps[{index}] keys changed."
            )
        pattern = raw_step.get("pattern")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(
                f"ACS transfer target pattern_steps[{index}] has no pattern name."
            )
        observed_patterns.append(pattern)
        if pattern not in expected_states:
            raise ValueError(
                f"ACS transfer target carries unexpected pattern {pattern!r}."
            )
        state_before = expected_states[pattern]
        expected_before_sha256 = _mapping_sha256(state_before.to_dict())
        if raw_step.get("state_before_sha256") != expected_before_sha256:
            raise ValueError(
                f"ACS transfer target pattern {pattern!r} does not continue the "
                "expected predecessor state."
            )
        state_after_payload = raw_step.get("state_after")
        if not isinstance(state_after_payload, Mapping):
            raise ValueError(
                f"ACS transfer target pattern {pattern!r} lacks state_after."
            )
        state_after = QRFChainState.from_dict(state_after_payload)
        _validate_state_transition(
            state_before,
            state_after,
            model_target=model_target,
            pattern=pattern,
        )
        steps.append(
            AcsTransferBankPatternStep(
                pattern=pattern,
                state_before=state_before,
                state_after=state_after,
            )
        )
    if tuple(observed_patterns) != expected_patterns:
        raise ValueError(
            "ACS transfer target pattern order/membership changed: got "
            f"{observed_patterns}, expected {list(expected_patterns)}."
        )
    return tuple(steps)


def _validate_pattern_steps_for_write(
    checkpoint: AcsTransferTargetCheckpoint,
) -> None:
    if not checkpoint.pattern_steps:
        raise ValueError("ACS transfer target checkpoint requires pattern steps.")
    for step in checkpoint.pattern_steps:
        if not isinstance(step, AcsTransferBankPatternStep):
            raise TypeError(
                "ACS transfer checkpoint pattern_steps must contain "
                "AcsTransferBankPatternStep values."
            )
    patterns = tuple(step.pattern for step in checkpoint.pattern_steps)
    if any(not pattern for pattern in patterns) or len(set(patterns)) != len(patterns):
        raise ValueError(
            "ACS transfer target checkpoint pattern names must be non-empty and unique."
        )
    for step in checkpoint.pattern_steps:
        if tuple(step.state_before.targets) != checkpoint.model_targets:
            raise ValueError(
                f"ACS transfer pattern {step.pattern!r} target order differs from "
                "the checkpoint descriptor."
            )
        _validate_state_transition(
            step.state_before,
            step.state_after,
            model_target=checkpoint.model_target,
            pattern=step.pattern,
        )


def _validate_state_transition(
    state_before: QRFChainState,
    state_after: QRFChainState,
    *,
    model_target: str,
    pattern: str,
) -> None:
    if not isinstance(state_before, QRFChainState) or not isinstance(
        state_after, QRFChainState
    ):
        raise TypeError("ACS transfer pattern states must be QRFChainState values.")
    if state_before.next_target != model_target:
        raise ValueError(
            f"ACS transfer pattern {pattern!r} expected next target "
            f"{state_before.next_target!r}, not {model_target!r}."
        )
    expected_completed = (*state_before.completed_targets, model_target)
    if state_after.completed_targets != expected_completed:
        raise ValueError(
            f"ACS transfer pattern {pattern!r} state_after has completed prefix "
            f"{state_after.completed_targets}, expected {expected_completed}."
        )
    stable_fields = (
        "schema_version",
        "predictors",
        "targets",
        "entity",
        "weight_kind",
        "weight_sha256",
        "n_estimators",
        "zero_atol",
        "max_samples_leaf",
        "max_samples_leaf_kind",
        "seed",
        "fit_n_jobs",
        "donor_index",
    )
    changed = [
        field
        for field in stable_fields
        if getattr(state_after, field) != getattr(state_before, field)
    ]
    if changed:
        raise ValueError(
            f"ACS transfer pattern {pattern!r} state transition changed immutable "
            f"field(s): {changed}."
        )
    if state_after.recipient_index is None:
        raise ValueError(
            f"ACS transfer pattern {pattern!r} state_after has no recipient identity."
        )
    if (
        state_before.recipient_index is not None
        and state_after.recipient_index != state_before.recipient_index
    ):
        raise ValueError(
            f"ACS transfer pattern {pattern!r} recipient identity changed."
        )


def _validate_expected_states(
    expected_states: Mapping[str, QRFChainState],
    *,
    model_targets: tuple[str, ...],
    model_target: str,
) -> None:
    if not isinstance(expected_states, Mapping) or not expected_states:
        raise ValueError("ACS transfer expected_states must be a non-empty mapping.")
    for pattern, state in expected_states.items():
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(
                "ACS transfer expected-state pattern names must be strings."
            )
        if not isinstance(state, QRFChainState):
            raise TypeError(
                f"ACS transfer expected state for {pattern!r} is not a QRFChainState."
            )
        if state.targets != model_targets:
            raise ValueError(
                f"ACS transfer expected state for {pattern!r} has target order "
                f"{state.targets}, expected {model_targets}."
            )
        if state.next_target != model_target:
            raise ValueError(
                f"ACS transfer expected state for {pattern!r} names next target "
                f"{state.next_target!r}, expected {model_target!r}."
            )


def _target_descriptor(
    *,
    target_index: int,
    total_targets: int,
    entity: str,
    family: str,
    family_targets: Sequence[str],
    model_targets: Sequence[str],
    model_target: str,
    exported_targets: Sequence[str],
) -> dict[str, object]:
    if (
        not isinstance(target_index, int)
        or isinstance(target_index, bool)
        or target_index < 0
    ):
        raise ValueError("ACS transfer target_index must be a non-negative integer.")
    if (
        not isinstance(total_targets, int)
        or isinstance(total_targets, bool)
        or total_targets <= 0
        or target_index >= total_targets
    ):
        raise ValueError(
            "ACS transfer total_targets must be positive and exceed target_index."
        )
    for label, value in (
        ("entity", entity),
        ("family", family),
        ("model_target", model_target),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"ACS transfer {label} must be a non-empty string.")
    family_names = _string_tuple(family_targets, label="family_targets")
    model_names = _string_tuple(model_targets, label="model_targets")
    exported_names = _string_tuple(exported_targets, label="exported_targets")
    if model_target not in model_names:
        raise ValueError("ACS transfer model_target is absent from model_targets.")
    return {
        "target_index": target_index,
        "total_targets": total_targets,
        "entity": entity,
        "family": family,
        "family_targets": list(family_names),
        "model_targets": list(model_names),
        "model_target": model_target,
        "exported_targets": list(exported_names),
    }


def _string_tuple(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"ACS transfer {label} must be a string sequence.")
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"ACS transfer {label} must contain non-empty strings.")
    if len(set(result)) != len(result):
        raise ValueError(f"ACS transfer {label} contains duplicates.")
    return result


def _validate_recipient_rows(recipient_rows: int) -> None:
    if (
        not isinstance(recipient_rows, int)
        or isinstance(recipient_rows, bool)
        or recipient_rows < 0
    ):
        raise ValueError("ACS transfer recipient_rows must be a non-negative integer.")


def _normalize_identity(identity: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(identity, Mapping):
        raise TypeError("ACS transfer target-bank identity must be a mapping.")
    normalized = _json_ready(identity)
    if not isinstance(
        normalized, dict
    ):  # pragma: no cover - mapping normalizes to dict
        raise TypeError(
            "ACS transfer target-bank identity must normalize to an object."
        )
    # A canonical JSON round trip ensures tuples, NumPy scalars, and mapping
    # subclasses cannot survive as process-local identity representations.
    restored = json.loads(_canonical_json(normalized))
    if not isinstance(restored, dict):  # pragma: no cover - canonical object
        raise TypeError("ACS transfer target-bank identity must be a JSON object.")
    return restored


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("ACS transfer bank mappings must use string JSON keys.")
        return {key: _json_ready(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_ready(item) for item in value]
        return sorted(normalized, key=_canonical_json)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("ACS transfer bank metadata cannot contain non-finite JSON.")
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _mapping_sha256(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_parent_directory(path: Path) -> None:
    """Persist a completed atomic rename in its containing directory."""

    # O_DIRECTORY is not universal. A read-only directory descriptor is the
    # supported POSIX fallback; failures to open or fsync still propagate.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _is_lowercase_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == _LOWERCASE_SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )
