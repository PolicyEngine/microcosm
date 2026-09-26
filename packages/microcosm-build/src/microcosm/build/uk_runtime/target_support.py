"""Per-target weight-stretch anatomy for a UK national calibration attempt.

A national solve compiles one sparse row per target over the household weight
vector (``A @ w`` estimates each target). The staging H5 that carries the final
weights is written only after the terminal battery passes, so an attempt the
battery blocks used to leave nothing from which the per-target anatomy (how
much of a target's fitted mass sits on households the solver stretched) could
be read. :func:`write_uk_target_support_sidecars` writes the compiled system
and both weight vectors beside the diagnostics *before* the battery runs;
:func:`target_support_anatomy` reads them back and decomposes any target into
its carriers, the share of its final aggregate on households stretched beyond
declared ratios of their design weight, and the concentration on its top
carriers. This is the instrument behind microcosm#890's acceptance line (the
share of England bus-fare mass on households stretched more than 3x) and
microcosm#930's measurement.

The sidecars are non-release evidence: nothing here is a shippable artifact,
and the staging H5 posture is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse

#: The compiled constraint matrix ``A`` (targets x households), CSR.
TARGET_SUPPORT_MATRIX_FILE = "target_support_matrix.npz"
#: Row names, target vector, design (initial) and final weights, household ids.
TARGET_SUPPORT_VECTORS_FILE = "target_support_vectors.npz"
#: Shapes and digests of the two sidecars, so a reader can refuse a mismatch.
TARGET_SUPPORT_MANIFEST_FILE = "target_support_manifest.json"
#: Default weight-ratio thresholds for the stretched-mass buckets.
DEFAULT_STRETCH_THRESHOLDS: tuple[float, ...] = (3.0, 5.0)
#: Carriers whose ratio exceeds this fraction of the realised maximum ratio
#: are reported as pinned near the solver's cap.
NEAR_CAP_FRACTION = 0.9


class TargetSupportError(RuntimeError):
    """Raised when the sidecars are inconsistent or a target is unknown."""


def _weight_values(weights: object) -> np.ndarray:
    values = getattr(weights, "values", weights)
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise TargetSupportError(f"weights must be one-dimensional, got {array.shape}.")
    return array


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetSupportSidecars:
    """The compiled system and weights of one calibration attempt."""

    matrix: sparse.csr_array
    names: tuple[str, ...]
    target_vector: np.ndarray
    design_weights: np.ndarray
    final_weights: np.ndarray
    household_ids: np.ndarray | None

    def __post_init__(self) -> None:
        n_rows, n_cols = self.matrix.shape
        if not (len(self.names) == len(self.target_vector) == n_rows):
            raise TargetSupportError(
                f"target rows disagree: matrix {n_rows}, names {len(self.names)}, "
                f"targets {len(self.target_vector)}."
            )
        if not (len(self.design_weights) == len(self.final_weights) == n_cols):
            raise TargetSupportError(
                f"weight vectors disagree with the matrix: matrix {n_cols} columns, "
                f"design {len(self.design_weights)}, final {len(self.final_weights)}."
            )
        if self.household_ids is not None and len(self.household_ids) != n_cols:
            raise TargetSupportError(
                f"household ids ({len(self.household_ids)}) do not match the "
                f"matrix columns ({n_cols})."
            )

    def row(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            without_period = [n for n in self.names if n.split("@", 1)[0] == name]
            if len(without_period) == 1:
                return self.names.index(without_period[0])
            raise TargetSupportError(
                f"target {name!r} is not a compiled row of this attempt "
                f"({len(self.names)} rows)."
            ) from None


def write_uk_target_support_sidecars(
    result: Any, directory: str | Path
) -> dict[str, Path]:
    """Write the solve's compiled system and weight vectors beside the diagnostics.

    ``result`` is a :class:`microcosm.calibrate.CalibrationResult`: its
    ``problem`` carries the CSR matrix, the row names and the target vector;
    ``initial_weights`` and ``weights`` are the design and final household
    weight vectors. Household ids are taken from the calibrated frame's
    household table when it carries one.
    """

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    problem = result.problem
    matrix = sparse.csr_array(problem.matrix)
    names = np.asarray([str(name) for name in problem.names], dtype=object)
    target_vector = np.asarray(problem.target_vector, dtype=float)
    design = _weight_values(result.initial_weights)
    final = _weight_values(result.weights)
    household_ids: np.ndarray | None = None
    frame = getattr(result, "frame", None)
    weight_entity = str(getattr(result, "weight_entity", "household"))
    if frame is not None:
        try:
            table = frame.table(weight_entity)
        except Exception:  # pragma: no cover - a frame without the entity
            table = None
        id_column = f"{weight_entity}_id"
        if table is not None and id_column in table:
            household_ids = np.asarray(table[id_column].to_numpy())
    sidecars = TargetSupportSidecars(
        matrix=matrix,
        names=tuple(str(n) for n in names),
        target_vector=target_vector,
        design_weights=design,
        final_weights=final,
        household_ids=household_ids,
    )
    matrix_path = directory / TARGET_SUPPORT_MATRIX_FILE
    vectors_path = directory / TARGET_SUPPORT_VECTORS_FILE
    manifest_path = directory / TARGET_SUPPORT_MANIFEST_FILE
    sparse.save_npz(matrix_path, sidecars.matrix, compressed=True)
    vectors: dict[str, np.ndarray] = {
        "names": names,
        "target_vector": target_vector,
        "design_weights": design,
        "final_weights": final,
    }
    if household_ids is not None:
        vectors["household_ids"] = household_ids
    np.savez_compressed(vectors_path, **vectors)
    manifest = {
        "kind": "uk_target_support_sidecars",
        "version": 1,
        "weight_entity": weight_entity,
        "targets": int(matrix.shape[0]),
        "records": int(matrix.shape[1]),
        "nonzeros": int(matrix.nnz),
        "household_ids": household_ids is not None,
        "files": {
            TARGET_SUPPORT_MATRIX_FILE: _sha256(matrix_path),
            TARGET_SUPPORT_VECTORS_FILE: _sha256(vectors_path),
        },
        "note": (
            "Non-release evidence written before the terminal battery so a "
            "blocked attempt keeps its per-target weight-stretch anatomy "
            "(tools/diagnose_uk_target_support.py)."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "matrix": matrix_path,
        "vectors": vectors_path,
        "manifest": manifest_path,
    }


def load_uk_target_support_sidecars(directory: str | Path) -> TargetSupportSidecars:
    """Read the sidecars of one attempt, refusing a digest or shape mismatch."""

    directory = Path(directory)
    manifest_path = directory / TARGET_SUPPORT_MANIFEST_FILE
    matrix_path = directory / TARGET_SUPPORT_MATRIX_FILE
    vectors_path = directory / TARGET_SUPPORT_VECTORS_FILE
    for path in (manifest_path, matrix_path, vectors_path):
        if not path.is_file():
            raise TargetSupportError(f"missing target-support sidecar {path}.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("kind") != "uk_target_support_sidecars":
        raise TargetSupportError(f"{manifest_path}: not a target-support manifest.")
    for name, path in (
        (TARGET_SUPPORT_MATRIX_FILE, matrix_path),
        (TARGET_SUPPORT_VECTORS_FILE, vectors_path),
    ):
        expected = str(manifest.get("files", {}).get(name, ""))
        actual = _sha256(path)
        if expected != actual:
            raise TargetSupportError(
                f"{path.name}: sha256 {actual} differs from the manifest's {expected}."
            )
    matrix = sparse.csr_array(sparse.load_npz(matrix_path))
    with np.load(vectors_path, allow_pickle=True) as vectors:
        names = tuple(str(n) for n in vectors["names"])
        target_vector = np.asarray(vectors["target_vector"], dtype=float)
        design = np.asarray(vectors["design_weights"], dtype=float)
        final = np.asarray(vectors["final_weights"], dtype=float)
        household_ids = (
            np.asarray(vectors["household_ids"])
            if "household_ids" in vectors.files
            else None
        )
    sidecars = TargetSupportSidecars(
        matrix=matrix,
        names=names,
        target_vector=target_vector,
        design_weights=design,
        final_weights=final,
        household_ids=household_ids,
    )
    if (int(manifest.get("targets", -1)), int(manifest.get("records", -1))) != (
        matrix.shape[0],
        matrix.shape[1],
    ):
        raise TargetSupportError(
            f"{manifest_path}: declared shape "
            f"({manifest.get('targets')}, {manifest.get('records')}) differs from "
            f"the matrix {matrix.shape}."
        )
    return sidecars


def realised_max_weight_ratio(sidecars: TargetSupportSidecars) -> float | None:
    """The solve's realised maximum final/design weight ratio (positive design)."""

    positive = sidecars.design_weights > 0.0
    if not positive.any():
        return None
    ratios = sidecars.final_weights[positive] / sidecars.design_weights[positive]
    return float(ratios.max())


def target_support_anatomy(
    sidecars: TargetSupportSidecars,
    name: str,
    *,
    stretch_thresholds: Sequence[float] = DEFAULT_STRETCH_THRESHOLDS,
    top: int = 10,
    near_cap_ratio: float | None = None,
) -> dict[str, Any]:
    """Decompose one compiled target into its carriers and stretched mass.

    A carrier is a household with a nonzero compiled value on the row. The
    final aggregate is ``sum(value * final_weight)``; for each threshold ``k``
    the report gives the share of that aggregate carried by households whose
    final weight exceeds ``k`` times their design weight (``stretched_mass``)
    and the share of carriers so stretched (``stretched_carriers``), beside the
    frame-wide share of *all* weight above ``k`` times design, which is the
    baseline the target's share is read against. Carriers are ranked by
    absolute weighted contribution so signed rows report their dominant
    carriers.
    """

    if top < 1:
        raise ValueError("top must be positive.")
    thresholds = tuple(float(t) for t in stretch_thresholds)
    if any(t <= 0 for t in thresholds):
        raise ValueError("stretch thresholds must be positive.")
    row = sidecars.row(name)
    compiled = np.asarray(sidecars.matrix[[row], :].toarray()).ravel()
    design = sidecars.design_weights
    final = sidecars.final_weights
    target_value = float(sidecars.target_vector[row])
    contributions = compiled * final
    design_contributions = compiled * design
    carrier_mask = compiled != 0.0
    carrier_indices = np.flatnonzero(carrier_mask)
    ranked = carrier_indices[
        np.argsort(-np.abs(contributions[carrier_indices]), kind="stable")
    ]
    total = float(contributions.sum())
    design_total = float(design_contributions.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(design > 0.0, final / design, np.nan)
    carrier_ratios = ratios[carrier_mask]
    finite_carrier_ratios = carrier_ratios[np.isfinite(carrier_ratios)]

    def _share(count: int) -> float | None:
        if total == 0.0:
            return None
        return float(contributions[ranked[:count]].sum() / total)

    frame_total_weight = float(final.sum())
    stretched: dict[str, dict[str, float | None]] = {}
    for threshold in thresholds:
        key = f"{threshold:g}x"
        above = np.isfinite(ratios) & (ratios > threshold)
        mass = (
            float(contributions[carrier_mask & above].sum() / total)
            if total != 0.0
            else None
        )
        carriers_above = (
            float((carrier_mask & above).sum() / carrier_mask.sum())
            if carrier_mask.any()
            else None
        )
        frame_share = (
            float(final[above].sum() / frame_total_weight)
            if frame_total_weight > 0.0
            else None
        )
        stretched[key] = {
            "threshold": threshold,
            "stretched_mass": mass,
            "stretched_carriers": carriers_above,
            "frame_weight_share": frame_share,
        }
    carriers: list[dict[str, object]] = []
    for index in ranked[:top]:
        entry: dict[str, object] = {
            "record_index": int(index),
            "compiled_value": float(compiled[index]),
            "design_weight": float(design[index]),
            "final_weight": float(final[index]),
            "weight_ratio": float(ratios[index])
            if np.isfinite(ratios[index])
            else None,
            "weighted_contribution": float(contributions[index]),
            "contribution_share": (
                float(contributions[index] / total) if total != 0.0 else None
            ),
        }
        if sidecars.household_ids is not None:
            value = sidecars.household_ids[index]
            entry["household_id"] = value.item() if hasattr(value, "item") else value
        carriers.append(entry)
    report: dict[str, Any] = {
        "name": sidecars.names[row],
        "target": target_value,
        "design_estimate": design_total,
        "final_estimate": total,
        "design_relative_error": (
            design_total / target_value - 1.0 if target_value else None
        ),
        "final_relative_error": (total / target_value - 1.0 if target_value else None),
        "carrier_count": int(carrier_mask.sum()),
        "carriers_with_nonpositive_design_weight": int(
            (design[carrier_mask] <= 0.0).sum()
        ),
        "top_1_share": _share(1),
        "top_5_share": _share(5),
        f"top_{top}_share": _share(top),
        "carrier_weight_ratio": {
            "mean": (
                float(finite_carrier_ratios.mean())
                if finite_carrier_ratios.size
                else None
            ),
            "median": (
                float(np.median(finite_carrier_ratios))
                if finite_carrier_ratios.size
                else None
            ),
            "p90": (
                float(np.quantile(finite_carrier_ratios, 0.9))
                if finite_carrier_ratios.size
                else None
            ),
            "max": (
                float(finite_carrier_ratios.max())
                if finite_carrier_ratios.size
                else None
            ),
        },
        "stretched": stretched,
        "top_carriers": carriers,
    }
    if near_cap_ratio is not None and finite_carrier_ratios.size:
        report["carrier_share_near_cap"] = float(
            (finite_carrier_ratios > near_cap_ratio).mean()
        )
    return report


def anatomy_for_targets(
    sidecars: TargetSupportSidecars,
    names: Iterable[str],
    *,
    stretch_thresholds: Sequence[float] = DEFAULT_STRETCH_THRESHOLDS,
    top: int = 10,
) -> list[dict[str, Any]]:
    """The anatomy of every named target, with the near-cap fraction resolved."""

    max_ratio = realised_max_weight_ratio(sidecars)
    near_cap = None if max_ratio is None else NEAR_CAP_FRACTION * max_ratio
    return [
        target_support_anatomy(
            sidecars,
            name,
            stretch_thresholds=stretch_thresholds,
            top=top,
            near_cap_ratio=near_cap,
        )
        for name in names
    ]


def verify_against_diagnostics(
    reports: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any],
    *,
    rtol: float = 1e-9,
    atol: float = 0.5,
) -> None:
    """Refuse a sidecar whose recomputed final estimates differ from the diagnostics."""

    recorded: dict[str, float] = {}
    for row in diagnostics.get("targets", ()):
        if not isinstance(row, Mapping):
            continue
        for key in ("name", "target_name"):
            value = row.get(key)
            if isinstance(value, str) and "final_estimate" in row:
                recorded[value] = float(row["final_estimate"])
    for report in reports:
        name = str(report["name"])
        candidates = [name, name.split("@", 1)[0]]
        expected = next((recorded[c] for c in candidates if c in recorded), None)
        if expected is None:
            raise TargetSupportError(
                f"{name}: the diagnostics carry no final_estimate for this row."
            )
        actual = float(report["final_estimate"])
        if abs(actual - expected) > atol + rtol * abs(expected):
            raise TargetSupportError(
                f"{name}: recomputed final estimate {actual} differs from the "
                f"diagnostics' {expected}; the sidecars are not this attempt's."
            )
