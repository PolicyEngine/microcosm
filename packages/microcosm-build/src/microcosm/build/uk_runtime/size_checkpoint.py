"""Checkpoint a size run's dense solve and informed L0 search before the draw.

A ``--dataset-households`` run spends hours on the K-cloned pool: the dense
joint solve, then up to ten L0 budget probes, each a full optimisation. The
exact-count draw that follows can refuse (S2, microcosm#355, 2026-09-08:
"degenerate boundary mass" after 4.8 hours, nothing written). This module
persists everything the draw and the refit need — the dense weights and loss
trajectory, the selection's weights, gate probabilities and search receipt,
the protected-carrier mask — with the identity of the pool and the target
surface they were cut from, so a resumed run re-derives the pool and the
surface (minutes), verifies the identity, rebuilds both results through
:func:`microcosm.calibrate.rebuild_calibration_result`, and continues at the
draw. The checkpoint is candidate-run evidence, never a release input.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.uk_runtime.dataset_size import UKSizeSelection
from microcosm.calibrate import (
    CalibrationProblem,
    CalibrationResult,
    TargetSet,
    rebuild_calibration_result,
)
from microcosm.frame import Frame

SIZE_CHECKPOINT_ARRAYS_FILENAME = "size_selection_checkpoint.npz"
SIZE_CHECKPOINT_MANIFEST_FILENAME = "size_selection_checkpoint.json"
SIZE_CHECKPOINT_KIND = "microcosm_uk_size_selection_checkpoint"
SIZE_CHECKPOINT_SCHEMA_VERSION = 1

_DENSE_ARRAYS = (
    "dense_weights",
    "dense_loss_trajectory",
    "initial_weights",
    "target_loss_weights",
    "target_loss_scales",
)
_SELECTION_ARRAYS = (
    "selection_weights",
    "selection_probabilities",
    "selection_loss_trajectory",
    "protected",
)


@dataclass(frozen=True)
class UKSizeCheckpointRestore:
    """A checkpoint's dense solve and selection, rebuilt on the re-derived pool."""

    dense: CalibrationResult
    selection: UKSizeSelection
    receipt: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_digest(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(f"{contiguous.dtype.str}{contiguous.shape}".encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _household_ids_digest(frame: Frame) -> str:
    ids = frame.table("household")["household_id"].to_numpy()
    return _array_digest(np.asarray(ids, dtype=np.int64))


def _target_surface_digest(problem: CalibrationProblem) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(problem.names).encode())
    digest.update(b"\0")
    digest.update(
        np.ascontiguousarray(problem.target_vector, dtype=np.float64).tobytes()
    )
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _normalised(value: Any) -> Any:
    return json.loads(json.dumps(_json_safe(value), sort_keys=True))


def write_uk_size_checkpoint(
    directory: Path,
    *,
    frame: Frame,
    dense: CalibrationResult,
    selection: UKSizeSelection,
    identity: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist ``dense`` and ``selection`` under ``directory`` with their identity.

    ``identity`` is the caller's pin/parameter mapping (input digests, seeds,
    epochs, size, the solve doctrine...); :func:`load_uk_size_checkpoint`
    refuses a resume whose caller identity differs on any key. ``provenance``
    (the writing run's code pin and build id) is recorded and reported on
    resume, not compared. Refuses to overwrite an existing checkpoint. Returns
    a receipt with both file names and digests; the receipt carries nothing
    that differs between identical runs (no timestamp, no absolute path), so
    manifests of identical runs still match.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays_path = directory / SIZE_CHECKPOINT_ARRAYS_FILENAME
    manifest_path = directory / SIZE_CHECKPOINT_MANIFEST_FILENAME
    for existing in (arrays_path, manifest_path):
        if existing.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {existing}.")
    search = selection.selection
    if search.gate_open_probabilities is None:
        raise ValueError("a size checkpoint needs the selection's gate probabilities.")
    n = frame.n("household")
    if len(dense.weights) != n or len(search.weights) != n:
        raise ValueError("checkpoint weights must align with the pool households.")
    np.savez(
        arrays_path,
        dense_weights=np.asarray(dense.weights, dtype=np.float64),
        dense_loss_trajectory=np.asarray(dense.loss_trajectory, dtype=np.float64),
        initial_weights=np.asarray(dense.initial_weights, dtype=np.float64),
        target_loss_weights=np.asarray(dense.target_loss_weights, dtype=np.float64),
        target_loss_scales=np.asarray(dense.target_loss_scales, dtype=np.float64),
        selection_weights=np.asarray(search.weights, dtype=np.float64),
        selection_probabilities=np.asarray(
            search.gate_open_probabilities, dtype=np.float64
        ),
        selection_loss_trajectory=np.asarray(search.loss_trajectory, dtype=np.float64),
        protected=np.asarray(selection.protected, dtype=bool),
    )
    arrays_sha256 = _sha256_file(arrays_path)
    payload = {
        "artifact_kind": SIZE_CHECKPOINT_KIND,
        "schema_version": SIZE_CHECKPOINT_SCHEMA_VERSION,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "arrays_file": SIZE_CHECKPOINT_ARRAYS_FILENAME,
        "arrays_sha256": arrays_sha256,
        "identity": _normalised(identity),
        "provenance": _normalised({} if provenance is None else provenance),
        "pool": {
            "households": int(n),
            "household_ids_sha256": _household_ids_digest(frame),
            "weight_entity": "household",
            "targets": int(dense.problem.n_targets),
            "targets_sha256": _target_surface_digest(dense.problem),
        },
        "dense": {
            "closing_loss": float(dense.final_loss),
            "n_nonzero": int(dense.n_nonzero),
            "l0_lambda": float(dense.l0_lambda),
            "target_loss_cap": float(dense.target_loss_cap),
            "options": _normalised(dense.options),
        },
        "selection": {
            "closing_loss": float(search.final_loss),
            "n_nonzero": int(search.n_nonzero),
            "l0_lambda": float(search.l0_lambda),
            "target_loss_cap": float(search.target_loss_cap),
            "options": _normalised(search.options),
            "households": int(selection.households),
            "epochs": int(selection.epochs),
            "learning_rate": float(selection.learning_rate),
            "seed": int(selection.seed),
            "search_pi_hi": float(selection.search_pi_hi),
            "protected_carriers": int(np.count_nonzero(selection.protected)),
        },
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "arrays_file": SIZE_CHECKPOINT_ARRAYS_FILENAME,
        "arrays_sha256": arrays_sha256,
        "manifest_file": SIZE_CHECKPOINT_MANIFEST_FILENAME,
        "manifest_sha256": _sha256_file(manifest_path),
        "stage": "before_exact_count_draw",
    }


def _check_closing_loss(stage: str, stored: float, recomputed: float) -> None:
    if not math.isclose(stored, recomputed, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"size checkpoint {stage} closing loss {stored!r} disagrees with the "
            f"loss recomputed on this run's compiled system ({recomputed!r})."
        )


def load_uk_size_checkpoint(
    directory: Path,
    *,
    frame: Frame,
    target_set: TargetSet,
    identity: Mapping[str, Any],
) -> UKSizeCheckpointRestore:
    """Rebuild a checkpoint's dense solve and selection on ``frame``.

    Refuses, naming the field, when the arrays' digest, the caller identity,
    the pool's household ids or the compiled target surface differ from what
    the checkpoint recorded, or when the persisted closing losses disagree
    with the losses recomputed on the compiled system.
    """
    directory = Path(directory)
    return load_uk_size_checkpoint_files(
        directory / SIZE_CHECKPOINT_MANIFEST_FILENAME,
        directory / SIZE_CHECKPOINT_ARRAYS_FILENAME,
        frame=frame,
        target_set=target_set,
        identity=identity,
    )


def load_uk_size_checkpoint_files(
    manifest_path: Path,
    arrays_path: Path,
    *,
    frame: Frame,
    target_set: TargetSet,
    identity: Mapping[str, Any],
) -> UKSizeCheckpointRestore:
    """Read the same checkpoint from separately content-verified source paths."""
    manifest_path, arrays_path = Path(manifest_path), Path(arrays_path)
    directory = manifest_path.parent
    for required in (arrays_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(f"size checkpoint file missing: {required}.")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("artifact_kind") != SIZE_CHECKPOINT_KIND:
        raise ValueError(f"{manifest_path} is not a size selection checkpoint.")
    if payload.get("schema_version") != SIZE_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported size checkpoint schema {payload.get('schema_version')!r}."
        )
    arrays_sha256 = _sha256_file(arrays_path)
    if arrays_sha256 != payload.get("arrays_sha256"):
        raise ValueError("size checkpoint arrays do not match their recorded digest.")
    stored_identity = payload.get("identity", {})
    requested_identity = _normalised(identity)
    mismatched = sorted(
        f"{key}: checkpoint {stored_identity.get(key)!r} != run {value!r}"
        for key, value in requested_identity.items()
        if stored_identity.get(key) != value
    )
    missing = sorted(set(requested_identity) - set(stored_identity))
    if mismatched or missing:
        raise ValueError(
            "size checkpoint identity differs from this run: "
            + "; ".join(
                mismatched + [f"{key}: absent in checkpoint" for key in missing]
            )
            + "."
        )
    pool = payload["pool"]
    if pool["households"] != frame.n("household") or pool[
        "household_ids_sha256"
    ] != _household_ids_digest(frame):
        raise ValueError(
            "size checkpoint pool differs from this run's pool "
            f"({pool['households']} checkpointed vs {frame.n('household')} households)."
        )
    with np.load(arrays_path) as arrays:
        stored = {name: np.asarray(arrays[name]) for name in arrays.files}
    for name in (*_DENSE_ARRAYS, *_SELECTION_ARRAYS):
        if name not in stored:
            raise ValueError(f"size checkpoint arrays lack {name!r}.")
    dense_meta = payload["dense"]
    # Rebuild without the loss check first: a surface that moved must be named
    # as such, not as a loss disagreement it would also cause.
    dense = rebuild_calibration_result(
        frame,
        target_set,
        weights=stored["dense_weights"],
        loss_trajectory=stored["dense_loss_trajectory"],
        l0_lambda=float(dense_meta["l0_lambda"]),
        n_nonzero=int(dense_meta["n_nonzero"]),
        target_loss_weights=stored["target_loss_weights"],
        target_loss_scales=stored["target_loss_scales"],
        target_loss_cap=float(dense_meta["target_loss_cap"]),
        options=dense_meta["options"],
    )
    if not np.array_equal(dense.initial_weights, stored["initial_weights"]):
        raise ValueError("size checkpoint initial weights differ from the pool's.")
    if pool["targets"] != dense.problem.n_targets or pool[
        "targets_sha256"
    ] != _target_surface_digest(dense.problem):
        raise ValueError(
            "size checkpoint target surface differs from this run's surface "
            f"({pool['targets']} checkpointed vs {dense.problem.n_targets} rows)."
        )
    _check_closing_loss("dense", float(dense_meta["closing_loss"]), dense.final_loss)
    selection_meta = payload["selection"]
    search = rebuild_calibration_result(
        frame,
        target_set,
        weights=stored["selection_weights"],
        loss_trajectory=stored["selection_loss_trajectory"],
        l0_lambda=float(selection_meta["l0_lambda"]),
        n_nonzero=int(selection_meta["n_nonzero"]),
        target_loss_weights=stored["target_loss_weights"],
        target_loss_scales=stored["target_loss_scales"],
        target_loss_cap=float(selection_meta["target_loss_cap"]),
        options=selection_meta["options"],
        gate_open_probabilities=stored["selection_probabilities"],
    )
    _check_closing_loss(
        "selection", float(selection_meta["closing_loss"]), search.final_loss
    )
    protected = np.asarray(stored["protected"], dtype=bool)
    if int(np.count_nonzero(protected)) != int(selection_meta["protected_carriers"]):
        raise ValueError("size checkpoint protected-carrier mask is inconsistent.")
    selection = UKSizeSelection(
        selection=search,
        protected=protected,
        households=int(selection_meta["households"]),
        epochs=int(selection_meta["epochs"]),
        learning_rate=float(selection_meta["learning_rate"]),
        seed=int(selection_meta["seed"]),
        search_pi_hi=float(selection_meta["search_pi_hi"]),
    )
    receipt = {
        "directory": str(directory),
        "arrays_file": SIZE_CHECKPOINT_ARRAYS_FILENAME,
        "arrays_sha256": arrays_sha256,
        "manifest_file": SIZE_CHECKPOINT_MANIFEST_FILENAME,
        "manifest_sha256": _sha256_file(manifest_path),
        "identity": stored_identity,
        "provenance": payload.get("provenance", {}),
        "dense_closing_loss": float(dense.final_loss),
        "selection_l0_lambda": float(search.l0_lambda),
        "search_pi_hi": float(selection.search_pi_hi),
    }
    return UKSizeCheckpointRestore(dense=dense, selection=selection, receipt=receipt)
