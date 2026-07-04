"""Sweep the L2 concentration penalty over a materialized US target frame.

The ESS-vs-accuracy sweep from the L0 paper's future work (issue #285): run
the production sparse calibration stage — L0 selection then refit, exactly as
``tools/build_us_fiscal_refresh_release.py`` runs it — across a grid of
``l2_lambda`` values and penalty arms, and read the frontier off each run's
``calibration_diagnostics.json``.

The expensive part of a release build is PolicyEngine target materialization,
not calibration. This harness skips it by reusing a production run's inputs,
in either of two forms:

- ``--problem-bundle``: an exported compiled problem (scipy CSR constraint
  matrix, target values/names, initial household weights, production target
  loss weights). Each matrix row replays as a Target with a callable measure
  over a synthetic frame, so ``calibrate_l0_refit`` runs unchanged.
- ``--target-frame-checkpoint`` + ``--ledger-facts``: a build's materialized
  frame checkpoint, with targets recompiled through the builder's own loaders.

Either way a sweep point costs one calibrate stage, not one build, and the
input artifact's identity/hash is recorded as provenance in every artifact
this tool writes.

Three penalty arms answer where the penalty does its work:

- ``both``            — same lambda in L0 selection and refit (the primary grid)
- ``selection-only``  — lambda shapes which records are picked; refit unpenalized
- ``refit-only``      — fixed support; lambda spreads the shipped weights

``lambda = 0`` runs once as the shared ``baseline`` point regardless of arms.
Completed points (their ``point.json`` exists) are skipped on rerun, so an
interrupted sweep resumes where it stopped.

Example (production-faithful settings are the defaults):

    uv run tools/sweep_us_l2_lambda.py \\
        --problem-bundle l0_sparse_matrix_bundle.tar.gz \\
        --out sweeps/l2-$(date +%Y%m%d)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import tarfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from populace.calibrate import (
    Target,
    TargetRegistry,
    TargetSet,
    calibrate_l0_refit,
    effective_sample_size,
    write_calibration_diagnostics,
)
from populace.frame import EntitySchema, Frame, WeightKind, Weights

_TOOLS_DIR = Path(__file__).resolve().parent

#: The paper's proposed grid: outer points confirm the flat and
#: over-regularized ends; the action is expected in 1e-3..1e-1 (weight ratios
#: are O(1), so the penalty term is O(lambda) against a ~0.05 target loss).
DEFAULT_LAMBDA_GRID = "0,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1"

ARMS = ("both", "selection-only", "refit-only")

_SUMMARY_COLUMNS = (
    "point_id",
    "arm",
    "l2_lambda",
    "refit_l2_lambda",
    "l2_anchor",
    "n_candidate_households",
    "n_selected_households",
    "l0_lambda",
    "selection_final_loss",
    "selection_effective_sample_size",
    "refit_final_loss",
    "refit_fraction_within_10pct",
    "refit_effective_sample_size",
    "refit_realized_max_weight_ratio",
    "refit_top_1pct_weight_share",
    "seconds",
)


def _load_builder_module():
    """Load the release builder for its checkpoint/registry loaders.

    The sweep runs the same code path as a release build's calibration stage;
    reusing the builder's own loaders (checkpoint frame, ledger-fact registry,
    fiscal target loss weights) is what makes a sweep point production-faithful
    rather than a reimplementation that can drift.
    """
    path = _TOOLS_DIR / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_bundle(archive: Path) -> Path:
    """Extract a bundle tarball next to itself, once; reruns reuse it."""
    stem = archive.name.removesuffix(".tar.gz").removesuffix(".tgz")
    destination = archive.parent / f"{stem}_extracted"
    if not destination.exists():
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(destination, filter="data")
    return destination


def _bundle_directory(root: Path) -> Path:
    """The directory actually holding the bundle files (tarballs often wrap one)."""
    if (root / "constraint_matrix_csr.npz").exists():
        return root
    matches = sorted(root.rglob("constraint_matrix_csr.npz"))
    if not matches:
        raise FileNotFoundError(
            f"{root} does not contain a constraint_matrix_csr.npz bundle."
        )
    return matches[0].parent


def _bundle_row_measure(matrix, row_index: int):
    """A measure callable replaying one compiled constraint row.

    Values align through the frame's ``record_index`` column, so the same
    callable is correct on the full candidate frame and on the refit stage's
    selected-subset frame. Slices the CSR buffers directly: this runs once
    per target per compile (32k+ times on the production surface), and works
    for both ``csr_matrix`` and ``csr_array`` inputs.
    """

    def measure(frame: Frame) -> np.ndarray:
        positions = frame.table("household")["record_index"].to_numpy()
        start, end = matrix.indptr[row_index], matrix.indptr[row_index + 1]
        dense = np.zeros(matrix.shape[1], dtype=np.float64)
        dense[matrix.indices[start:end]] = matrix.data[start:end]
        return dense[positions]

    measure.__name__ = f"bundle_row_{row_index}"
    measure.__qualname__ = measure.__name__
    return measure


def _bundle_target_identities(
    directory: Path, names: list[str]
) -> list[tuple[str, int | str]]:
    """Per-row ``(name, period)`` pairs, preferring the structured row file.

    Falls back to splitting the ``"name@period"`` row labels. The pairs are
    each target's identity in the compiled matrix, so they must be unique.
    """
    rows_path = directory / "target_rows_minimal.jsonl"
    identities: list[tuple[str, int | str]] = []
    if rows_path.exists():
        rows = [
            json.loads(line)
            for line in rows_path.read_text().splitlines()
            if line.strip()
        ]
        if len(rows) == len(names):
            for row in rows:
                name = row.get("name") or row.get("target_name")
                if not name:
                    identities = []
                    break
                identities.append((str(name), row.get("period", 0)))
    if not identities:
        for label in names:
            name, separator, period_text = label.rpartition("@")
            if not separator:
                identities.append((label, 0))
                continue
            try:
                period: int | str = int(period_text)
            except ValueError:
                period = period_text
            identities.append((name, period))
    if len(set(identities)) != len(identities):
        raise ValueError(
            "Bundle target identities (name, period) are not unique; the "
            "compiled matrix cannot be replayed with ambiguous row keys."
        )
    return identities


def load_problem_bundle(
    path: Path,
    *,
    initial_weights_path: Path | None = None,
) -> tuple[Frame, TargetSet, np.ndarray | None, dict[str, object]]:
    """Load an exported compiled-problem bundle for the sweep.

    The bundle is the calibration problem a production run actually solved:
    a scipy CSR constraint matrix (targets x household records), the target
    values and row names, the initial household weights, and optionally the
    production target loss weights. It replays through the ordinary public
    API: a synthetic one-person-per-household frame carries a
    ``record_index`` column and every matrix row becomes a Target with a
    callable measure reading that row — so :func:`calibrate_l0_refit`
    (selection, refit, subsetting, mass conservation) runs exactly as in a
    release build, without PolicyEngine materialization.

    ``initial_weights_path`` replaces the bundle's starting weights with an
    externally supplied vector aligned to the matrix columns — e.g. the
    original survey design weights, for sweeps in the production
    (design-weight-start) regime instead of the bundle's recorded start.

    Returns ``(frame, targets, target_loss_weights_or_None, provenance)``.
    """
    import scipy.sparse

    directory = (
        _extract_bundle(path)
        if path.is_file() and (path.name.endswith(".tar.gz") or path.suffix == ".tgz")
        else path
    )
    directory = _bundle_directory(directory)

    matrix = scipy.sparse.load_npz(directory / "constraint_matrix_csr.npz").tocsr()
    values = np.asarray(np.load(directory / "target_values.npy"), dtype=np.float64)
    names = json.loads((directory / "target_names.json").read_text())
    initial_weights = np.asarray(
        np.load(directory / "initial_weights.npy"), dtype=np.float64
    )
    n_targets, n_records = matrix.shape
    if len(names) != n_targets or values.shape != (n_targets,):
        raise ValueError(
            f"Bundle shapes disagree: matrix has {n_targets} rows, "
            f"{len(names)} names, {values.shape} values."
        )
    if initial_weights.shape != (n_records,):
        raise ValueError(
            f"Bundle initial_weights has shape {initial_weights.shape}, "
            f"expected ({n_records},) to match the matrix columns."
        )
    if not np.isfinite(initial_weights).all() or (initial_weights <= 0).any():
        raise ValueError("Bundle initial_weights must be finite and positive.")
    initial_weights_override = None
    if initial_weights_path is not None:
        initial_weights_override = np.asarray(
            np.load(initial_weights_path), dtype=np.float64
        )
        if initial_weights_override.shape != (n_records,):
            raise ValueError(
                "Override initial weights have shape "
                f"{initial_weights_override.shape}, expected ({n_records},) to "
                "match the matrix columns."
            )
        if (
            not np.isfinite(initial_weights_override).all()
            or (initial_weights_override <= 0).any()
        ):
            raise ValueError("Override initial weights must be finite and positive.")
        initial_weights = initial_weights_override

    loss_weights_path = directory / "target_loss_weights_production_us_fiscal.npy"
    target_loss_weights = (
        np.asarray(np.load(loss_weights_path), dtype=np.float64)
        if loss_weights_path.exists()
        else None
    )
    if target_loss_weights is not None and target_loss_weights.shape != (n_targets,):
        raise ValueError(
            f"Bundle target loss weights have shape {target_loss_weights.shape}, "
            f"expected ({n_targets},)."
        )
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    record_index = np.arange(n_records, dtype="int64")
    frame = Frame(
        {
            "person": pd.DataFrame(
                {
                    "person_id": record_index,
                    "person_household_id": record_index,
                }
            ),
            "household": pd.DataFrame(
                {
                    "household_id": record_index,
                    "record_index": record_index,
                }
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=initial_weights, kind=WeightKind.DESIGN)},
    )
    identities = _bundle_target_identities(directory, names)
    targets = TargetSet(
        tuple(
            Target(
                name=name,
                entity="household",
                value=float(values[index]),
                period=period,
                measure=_bundle_row_measure(matrix, index),
                source="sparse problem bundle",
            )
            for index, (name, period) in enumerate(identities)
        )
    )
    provenance: dict[str, object] = {
        "problem_bundle": {
            "path": str(path.resolve()),
            "sha256": _sha256(path) if path.is_file() else None,
            "n_targets": int(n_targets),
            "n_records": int(n_records),
            "nnz": int(matrix.nnz),
            "has_production_target_loss_weights": target_loss_weights is not None,
            "initial_weights_override": (
                None
                if initial_weights_path is None
                else {
                    "path": str(initial_weights_path.resolve()),
                    "sha256": _sha256(initial_weights_path),
                }
            ),
            "metadata": metadata,
        }
    }
    return frame, targets, target_loss_weights, provenance


def expand_sweep_points(grid: list[float], arms: list[str]) -> list[dict[str, object]]:
    """The sweep's run list: one shared baseline, then one point per (arm, lambda).

    ``lambda = 0`` collapses every arm onto the unpenalized production run, so
    it appears once as ``baseline``. Duplicate lambdas are collapsed; order is
    ascending in lambda so the cheap-to-interpret points land first.
    """
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}; supported: {', '.join(ARMS)}.")
    if not arms:
        raise ValueError("At least one arm is required.")
    cleaned: list[float] = []
    for value in grid:
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"lambda grid values must be finite and non-negative, got {value!r}."
            )
        if value not in cleaned:
            cleaned.append(value)
    if not cleaned:
        raise ValueError("The lambda grid is empty.")

    points: list[dict[str, object]] = []
    for lam in sorted(cleaned):
        if lam == 0.0:
            points.append(
                {
                    "point_id": "baseline",
                    "arm": "baseline",
                    "l2_lambda": 0.0,
                    "refit_l2_lambda": None,
                }
            )
            continue
        for arm in arms:
            l2_lambda, refit_l2_lambda = {
                "both": (lam, None),
                "selection-only": (lam, 0.0),
                "refit-only": (0.0, lam),
            }[arm]
            points.append(
                {
                    "point_id": f"{arm}-{lam:g}",
                    "arm": arm,
                    "l2_lambda": l2_lambda,
                    "refit_l2_lambda": refit_l2_lambda,
                }
            )
    return points


def _progress_logger(
    log: Callable[[str], None], point_id: str, epochs: int
) -> Callable[[dict[str, object]], None]:
    """Log stage transitions and every ~10% of epochs; sweeps run for hours."""
    interval = max(1, epochs // 10)
    last_phase: list[object] = [None]

    def callback(event: dict[str, object]) -> None:
        phase = event.get("phase")
        epoch = event.get("epoch")
        if phase != last_phase[0]:
            last_phase[0] = phase
            log(f"[{point_id}] stage {phase}")
        if isinstance(epoch, int) and (epoch % interval == 0 or epoch == epochs):
            log(
                f"[{point_id}] {phase} epoch {epoch}/{event.get('epochs')} "
                f"loss {event.get('loss'):.6f}"
            )

    return callback


def _summary_row(
    point: dict[str, object],
    result,
    *,
    n_candidate_households: int,
    seconds: float,
    l2_anchor: str,
) -> dict[str, object]:
    l2_lambda = float(point["l2_lambda"])  # type: ignore[arg-type]
    refit_l2_lambda = point["refit_l2_lambda"]
    return {
        "point_id": point["point_id"],
        "arm": point["arm"],
        "l2_lambda": l2_lambda,
        "refit_l2_lambda": (
            l2_lambda if refit_l2_lambda is None else float(refit_l2_lambda)
        ),
        "l2_anchor": l2_anchor,
        "n_candidate_households": int(n_candidate_households),
        "n_selected_households": int(result.selection.n_nonzero),
        "l0_lambda": float(result.l0_lambda),
        "selection_final_loss": float(result.selection.final_loss),
        "selection_effective_sample_size": effective_sample_size(
            result.selection.weights
        ),
        "refit_final_loss": float(result.final_loss),
        "refit_fraction_within_10pct": float(result.fraction_within_10pct),
        "refit_effective_sample_size": float(result.effective_sample_size),
        "refit_realized_max_weight_ratio": float(result.realized_max_weight_ratio),
        "refit_top_1pct_weight_share": float(result.top_1pct_weight_share),
        "seconds": round(float(seconds), 1),
    }


def _write_summary(out_dir: Path, rows: list[dict[str, object]]) -> None:
    (out_dir / "sweep_summary.json").write_text(
        json.dumps({"schema_version": 1, "points": rows}, indent=1, allow_nan=False)
    )
    with (out_dir / "sweep_summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in _SUMMARY_COLUMNS})


def _format_frontier(rows: list[dict[str, object]]) -> str:
    header = (
        f"{'point':<22} {'sel λ2':>8} {'refit λ2':>8} {'kept':>7} "
        f"{'refit loss':>10} {'<10%':>6} {'ESS':>9} {'max w/w0':>9} {'top1%':>6}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['point_id']:<22} {row['l2_lambda']:>8g} "
            f"{row['refit_l2_lambda']:>8g} {row['n_selected_households']:>7} "
            f"{row['refit_final_loss']:>10.6f} "
            f"{row['refit_fraction_within_10pct']:>6.1%} "
            f"{row['refit_effective_sample_size']:>9.0f} "
            f"{row['refit_realized_max_weight_ratio']:>9.2f} "
            f"{row['refit_top_1pct_weight_share']:>6.1%}"
        )
    return "\n".join(lines)


def run_sweep(
    frame: Frame,
    targets: TargetSet,
    points: list[dict[str, object]],
    *,
    out_dir: Path,
    epochs: int,
    learning_rate: float,
    max_weight_ratio: float | None,
    l0_refit_lambda_share: float,
    seed: int,
    target_loss_cap: float,
    l2_anchor: str = "initial",
    registry: TargetRegistry | None = None,
    target_loss_weights: np.ndarray | None = None,
    provenance: dict[str, object] | None = None,
    resume: bool = True,
    log: Callable[[str], None] = print,
) -> list[dict[str, object]]:
    """Run every sweep point and return the frontier rows.

    Each point writes its own ``calibration_diagnostics.json`` (full per-target
    evidence, schema v4 with the weight-concentration scalars) plus a
    ``point.json`` summary row; the aggregate ``sweep_summary.json``/``.csv``
    are rewritten after every point so an interrupted sweep loses at most the
    in-flight point. With ``resume`` (default), points whose ``point.json``
    already exists are reloaded instead of re-run. ``registry`` enriches each
    point's diagnostics with target-registry identity when available (the
    checkpoint input mode); the problem-bundle mode runs without one.
    """
    n_candidate_households = int(frame.n("household"))
    l0_lambda = float(l0_refit_lambda_share) / float(n_candidate_households)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for index, point in enumerate(points, start=1):
        point_id = str(point["point_id"])
        point_dir = out_dir / "points" / point_id
        summary_path = point_dir / "point.json"
        if resume and summary_path.exists():
            rows.append(json.loads(summary_path.read_text()))
            log(f"[{point_id}] already complete, skipping ({index}/{len(points)})")
            continue

        log(
            f"[{point_id}] calibrating ({index}/{len(points)}): "
            f"selection l2={point['l2_lambda']:g}, "
            f"refit l2={'inherit' if point['refit_l2_lambda'] is None else format(point['refit_l2_lambda'], 'g')}"
        )
        point_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        result = calibrate_l0_refit(
            frame,
            targets,
            weight_entity="household",
            epochs=epochs,
            refit_epochs=epochs,
            learning_rate=learning_rate,
            max_weight_ratio=max_weight_ratio,
            seed=seed,
            mass="conserve",
            l0_lambda=l0_lambda,
            l2_lambda=float(point["l2_lambda"]),  # type: ignore[arg-type]
            refit_l2_lambda=(
                None
                if point["refit_l2_lambda"] is None
                else float(point["refit_l2_lambda"])  # type: ignore[arg-type]
            ),
            l2_anchor=l2_anchor,
            target_loss_weights=target_loss_weights,
            target_loss_cap=target_loss_cap,
            progress_callback=_progress_logger(log, point_id, epochs),
        )
        seconds = time.perf_counter() - started

        write_calibration_diagnostics(
            result,
            point_dir / "calibration_diagnostics.json",
            target_registry=registry,
            build={
                "l2_sweep": {
                    "point_id": point_id,
                    "arm": point["arm"],
                    "l2_lambda": float(point["l2_lambda"]),  # type: ignore[arg-type]
                    "refit_l2_lambda": point["refit_l2_lambda"],
                    "l2_anchor": l2_anchor,
                    "epochs": int(epochs),
                    "learning_rate": float(learning_rate),
                    "max_weight_ratio": (
                        None if max_weight_ratio is None else float(max_weight_ratio)
                    ),
                    "l0_refit_lambda_share": float(l0_refit_lambda_share),
                    "seed": int(seed),
                    "target_loss_cap": float(target_loss_cap),
                    **(provenance or {}),
                }
            },
        )
        row = _summary_row(
            point,
            result,
            n_candidate_households=n_candidate_households,
            seconds=seconds,
            l2_anchor=l2_anchor,
        )
        summary_path.write_text(json.dumps(row, indent=1, allow_nan=False))
        rows.append(row)
        _write_summary(out_dir, rows)
        log(
            f"[{point_id}] done in {seconds:.0f}s: kept "
            f"{row['n_selected_households']}, refit loss "
            f"{row['refit_final_loss']:.6f}, ESS "
            f"{row['refit_effective_sample_size']:.0f}"
        )

    _write_summary(out_dir, rows)
    return rows


def _parse_max_weight_ratio(text: str) -> float | None:
    """CLI cap value: a positive float, or 'none'/'uncapped' for no cap."""
    if text.strip().lower() in ("none", "uncapped", ""):
        return None
    return float(text)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--problem-bundle",
        type=Path,
        help=(
            "Exported compiled-problem bundle (a .tar.gz or its extracted "
            "directory) with constraint_matrix_csr.npz, target_values.npy, "
            "target_names.json, initial_weights.npy, and optionally the "
            "production target loss weights. Mutually exclusive with the "
            "checkpoint/ledger inputs."
        ),
    )
    parser.add_argument(
        "--target-frame-checkpoint",
        type=Path,
        help=(
            "target_frame_checkpoint.h5 from a release build (its artifacts/ "
            "directory): the fully materialized calibration frame. The "
            "checkpoint's stored identity is recorded as provenance; it is "
            "not re-derived, so any checkpoint from the intended base "
            "dataset works as-is. Requires --ledger-facts."
        ),
    )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        help=(
            "PolicyEngine Ledger consumer_facts.jsonl used to resolve target "
            "values — the same artifact the release build consumes. Only "
            "used with --target-frame-checkpoint."
        ),
    )
    parser.add_argument(
        "--initial-weights",
        type=Path,
        help=(
            "Optional .npy replacing the problem bundle's starting weights "
            "(aligned to matrix columns) — e.g. original survey design "
            "weights, for production-regime sweeps. Bundle mode only."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--lambda-grid",
        default=DEFAULT_LAMBDA_GRID,
        help=(
            "Comma-separated l2_lambda values. 0 runs once as the shared "
            f"baseline point. Default: {DEFAULT_LAMBDA_GRID}."
        ),
    )
    parser.add_argument(
        "--arms",
        default="both",
        help=(
            "Comma-separated penalty arms for positive lambdas: 'both' "
            "(selection and refit share the lambda; the primary grid), "
            "'selection-only', 'refit-only'. Default: both."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--max-weight-ratio",
        type=_parse_max_weight_ratio,
        default=None,
        help=(
            "Hard per-record weight cap as a multiple of the initial weight, "
            "or 'none' (default) for uncapped. The sweep defaults to "
            "uncapped so the L2 frontier is measured on its own, comparable "
            "to the L0 paper's uncapped probe; pass 5.0 for release-builder "
            "parity."
        ),
    )
    parser.add_argument("--l0-refit-lambda-share", type=float, default=None)
    parser.add_argument(
        "--l2-anchor",
        choices=("initial", "uniform", "design"),
        default="initial",
        help=(
            "Reference weights for the L2 penalty in both stages. 'initial' "
            "(default) anchors each stage at its own starting weights — for "
            "the refit those are the selection stage's concentrated weights, "
            "which a strong penalty pulls toward the SQUARE of (lower ESS). "
            "'uniform' anchors at the mean weight, making the penalty a "
            "direct 1/ESS control. 'design' anchors the refit at the "
            "pre-selection initial weights of the surviving records — 'stay "
            "near the survey design'; identical to 'initial' for the "
            "selection stage, and to 'uniform' end-to-end when the frame "
            "starts from a uniform reset."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--include-congressional-district-targets",
        action="store_true",
        help="Match the production registry: include SOI CD target rows.",
    )
    parser.add_argument(
        "--congressional-district-vintage-crosswalk",
        type=Path,
        help="Required with --include-congressional-district-targets.",
    )
    parser.add_argument(
        "--gate-congressional-district-targets",
        action="store_true",
        help="Compile CD targets as hard rows (builder parity; default off).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-run every point even when its point.json already exists.",
    )
    args = parser.parse_args()
    if args.problem_bundle is None:
        if args.target_frame_checkpoint is None or args.ledger_facts is None:
            parser.error(
                "Provide either --problem-bundle, or --target-frame-checkpoint "
                "together with --ledger-facts."
            )
    elif args.target_frame_checkpoint is not None or args.ledger_facts is not None:
        parser.error(
            "--problem-bundle is mutually exclusive with "
            "--target-frame-checkpoint/--ledger-facts."
        )
    if args.initial_weights is not None and args.problem_bundle is None:
        parser.error("--initial-weights requires --problem-bundle.")
    if (
        args.include_congressional_district_targets
        and args.congressional_district_vintage_crosswalk is None
    ):
        parser.error(
            "--congressional-district-vintage-crosswalk is required when "
            "--include-congressional-district-targets is set."
        )
    return args


def main() -> None:
    args = _parse_args()
    builder = _load_builder_module()
    epochs = (
        builder.DEFAULT_US_FISCAL_CALIBRATION_EPOCHS
        if args.epochs is None
        else args.epochs
    )
    l0_refit_lambda_share = (
        builder.DEFAULT_L0_REFIT_LAMBDA_SHARE
        if args.l0_refit_lambda_share is None
        else args.l0_refit_lambda_share
    )
    points = expand_sweep_points(
        [float(value) for value in args.lambda_grid.split(",")],
        [arm.strip() for arm in args.arms.split(",")],
    )

    if args.problem_bundle is not None:
        print(f"Loading sparse problem bundle {args.problem_bundle} ...")
        frame, targets, target_loss_weights, provenance = load_problem_bundle(
            args.problem_bundle,
            initial_weights_path=args.initial_weights,
        )
        registry = None
        bundle = provenance["problem_bundle"]
        print(
            f"Loaded compiled problem: {bundle['n_targets']:,} targets x "
            f"{bundle['n_records']:,} households ({bundle['nnz']:,} nonzeros)."
        )
        if target_loss_weights is None:
            print(
                "WARNING: bundle carries no production target loss weights; "
                "target rows will weigh equally."
            )
    else:
        print(f"Loading target frame checkpoint {args.target_frame_checkpoint} ...")
        frame, stored_identity, _ = builder._read_target_frame_checkpoint_file(
            args.target_frame_checkpoint
        )
        print(
            f"Loaded {frame.n('household'):,} candidate households "
            f"(checkpoint identity {builder._target_frame_checkpoint_digest(stored_identity)[:12]}...)"
        )

        crosswalk = (
            builder.load_congressional_district_vintage_crosswalk(
                args.congressional_district_vintage_crosswalk
            )
            if args.congressional_district_vintage_crosswalk is not None
            else None
        )
        full_registry = builder.compile_us_fiscal_target_registry(
            builder._load_ledger_facts(args.ledger_facts),
            target_period=builder.PERIOD,
            include_congressional_district_targets=(
                args.include_congressional_district_targets
            ),
            congressional_district_vintage_crosswalk=crosswalk,
        )
        registry, compilation = builder._compile_materialized_target_registry(
            frame,
            full_registry.specs,
            gate_congressional_district_targets=(
                args.gate_congressional_district_targets
            ),
        )
        print(
            f"Compiled {compilation['compiled_candidate_targets']:,} of "
            f"{compilation['declared_targets']:,} declared targets against the frame."
        )
        target_loss_weights = builder._fiscal_target_loss_weights(registry)
        targets = registry.to_target_set()

        provenance = {
            "target_frame_checkpoint": {
                "path": str(args.target_frame_checkpoint.resolve()),
                "identity_sha256": builder._target_frame_checkpoint_digest(
                    stored_identity
                ),
                "stored_identity": stored_identity,
            },
            "ledger_facts_sha256": builder._sha256(args.ledger_facts),
            "target_registry_version": registry.version,
            "declared_targets": compilation["declared_targets"],
            "compiled_candidate_targets": compilation["compiled_candidate_targets"],
        }

    rows = run_sweep(
        frame,
        targets,
        points,
        registry=registry,
        out_dir=args.out,
        epochs=epochs,
        learning_rate=args.learning_rate,
        max_weight_ratio=args.max_weight_ratio,
        l0_refit_lambda_share=l0_refit_lambda_share,
        seed=args.seed,
        target_loss_cap=builder.US_FISCAL_TARGET_LOSS_CAP,
        l2_anchor=args.l2_anchor,
        target_loss_weights=target_loss_weights,
        provenance=provenance,
        resume=not args.no_resume,
    )
    print()
    print(_format_frontier(rows))
    print(f"\nWrote {args.out / 'sweep_summary.json'} and .csv")


if __name__ == "__main__":
    main()
