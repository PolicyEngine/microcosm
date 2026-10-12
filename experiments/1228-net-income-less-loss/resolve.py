"""Re-solve one #1228 arm of the Route A release's national_state calibration.

Reads ``prepare.py``'s outputs from ``--inputs``, swaps in the arm's rows and
loss weights, and calls ``microcosm.calibrate.calibrate`` with the release's
settings (``calibration_diagnostics.json`` options): Adam, 1,500 epochs,
learning rate 0.02, ``mass="conserve"``, ``max_weight_ratio=5``,
``target_loss_cap=1``, no L1/L2, seed 0, from the release's initial weights.

Writes ``<out>/<arm>/weights.npz`` and ``<out>/<arm>/metrics.json``: the final
and initial loss, epochs, seconds, and the arm's estimates of every release
row and every #1228 row, so each arm can be scored on either concept.

Run: ``python resolve.py --inputs DIR --out DIR --arm ARM [--epochs N]``
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

RELEASE_SETTINGS = {
    "weight_entity": "household",
    "method": "adam",
    "epochs": 1500,
    "learning_rate": 0.02,
    "mass": "conserve",
    "max_weight_ratio": 5.0,
    "target_loss_cap": 1.0,
    "l2_lambda": 0.0,
    "seed": 0,
}
ARMS = (
    "release",
    "concept_exact",
    "concept_exact_no_business_count",
    "drop",
    "rebase_positive_part",
)


def load_rows(path: Path) -> tuple[np.ndarray, sparse.csr_array]:
    with np.load(path) as payload:
        return payload["rows"], sparse.csr_array(
            (payload["data"], payload["indices"], payload["indptr"]),
            shape=tuple(payload["shape"]),
        )


def load(inputs: Path):
    matrix = sparse.load_npz(inputs / "target_matrix.npz").tocsr()
    meta = pd.read_parquet(inputs / "targets_meta.parquet")
    concept_rows, concept_matrix = load_rows(inputs / "concept_rows.npz")
    rebase_rows, rebase_matrix = load_rows(inputs / "rebase_rows.npz")
    assert np.array_equal(concept_rows, rebase_rows)
    rebase_meta = pd.read_parquet(inputs / "rebase_meta.parquet")
    assert np.array_equal(rebase_meta["row"].to_numpy(), meta["row"].to_numpy())
    meta["value_rebase_positive_part"] = rebase_meta["value_rebase_positive_part"]
    meta["loss_weight_rebase_positive_part"] = rebase_meta[
        "loss_weight_rebase_positive_part"
    ]
    weights = dict(np.load(inputs / "weights.npz"))
    weights["rows_rebase_positive_part"] = np.arange(matrix.shape[0], dtype=np.int64)
    return matrix, meta, concept_rows, concept_matrix, rebase_matrix, weights


def swap_rows(
    matrix: sparse.csr_array,
    swapped_rows: np.ndarray,
    replacement: sparse.csr_array,
    rows: np.ndarray,
) -> sparse.csr_array:
    """``matrix[rows]`` with ``swapped_rows`` taken from ``replacement``."""

    n_release = matrix.shape[0]
    combined = sparse.csr_array(sparse.vstack([matrix, replacement]).tocsr())
    source = np.arange(n_release, dtype=np.int64)
    source[swapped_rows] = n_release + np.arange(len(swapped_rows), dtype=np.int64)
    return sparse.csr_array(combined[source[rows]])


def csr_row_measure(matrix: sparse.csr_array, row: int, n: int):
    start, stop = int(matrix.indptr[row]), int(matrix.indptr[row + 1])
    indices = matrix.indices[start:stop]
    data = matrix.data[start:stop]

    def measure(frame) -> np.ndarray:
        values = np.zeros(n, dtype=np.float64)
        values[indices] = data
        return values

    measure.__qualname__ = f"target_matrix_row[{row}]"
    return measure


def build_frame(n_households: int, weights: np.ndarray):
    from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

    household_id = np.arange(n_households, dtype=np.int64)
    tables = {
        "household": pd.DataFrame({"household_id": household_id}),
        "person": pd.DataFrame(
            {"person_id": household_id, "person_household_id": household_id}
        ),
    }
    return Frame(
        tables,
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.asarray(weights, np.float64), WeightKind.DESIGN)},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--epochs", type=int, default=RELEASE_SETTINGS["epochs"])
    args = parser.parse_args()

    from microcosm.calibrate import Target, TargetSet, calibrate

    started = time.time()
    matrix, meta, concept_rows, concept_matrix, rebase_matrix, weights = load(
        args.inputs
    )
    rows = weights[f"rows_{args.arm}"].astype(np.int64)
    loss_weights = meta[f"loss_weight_{args.arm}"].to_numpy()[rows]
    assert np.isfinite(loss_weights).all()
    if args.arm.startswith("concept_exact"):
        solve_matrix = swap_rows(matrix, concept_rows, concept_matrix, rows)
    elif args.arm == "rebase_positive_part":
        solve_matrix = swap_rows(matrix, concept_rows, rebase_matrix, rows)
    else:
        solve_matrix = sparse.csr_array(matrix[rows])
    n_households = matrix.shape[1]
    names = meta["name"].to_numpy()[rows]
    value_column = (
        "value_rebase_positive_part" if args.arm == "rebase_positive_part" else "value"
    )
    values = meta[value_column].to_numpy()[rows]
    targets = TargetSet(
        Target(
            name=str(names[position]),
            entity="household",
            measure=csr_row_measure(solve_matrix, position, n_households),
            value=float(values[position]),
            period=2024,
            source="route_a_checkpoint",
        )
        for position in range(len(rows))
    )
    frame = build_frame(n_households, weights["initial"])
    trajectory: list[dict] = []

    def on_progress(event: dict) -> None:
        if event.get("kind") != "calibration_epoch":
            return
        epoch = int(event["epoch"])
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            trajectory.append({"epoch": epoch, "loss": float(event["loss"])})
            elapsed = time.time() - started
            print(
                f"[{args.arm}] epoch {epoch} loss {float(event['loss']):.6f} "
                f"{elapsed:.0f}s",
                flush=True,
            )

    result = calibrate(
        frame,
        targets,
        weight_entity=RELEASE_SETTINGS["weight_entity"],
        method=RELEASE_SETTINGS["method"],
        epochs=args.epochs,
        learning_rate=RELEASE_SETTINGS["learning_rate"],
        mass=RELEASE_SETTINGS["mass"],
        max_weight_ratio=RELEASE_SETTINGS["max_weight_ratio"],
        l2_lambda=RELEASE_SETTINGS["l2_lambda"],
        seed=RELEASE_SETTINGS["seed"],
        target_loss_weights=loss_weights,
        target_loss_cap=RELEASE_SETTINGS["target_loss_cap"],
        progress_callback=on_progress,
    )
    if result.problem.skipped:
        raise SystemExit(f"calibration skipped {len(result.problem.skipped)} targets")
    compiled = [str(name).removesuffix("@2024") for name in result.problem.names]
    if compiled != [str(name) for name in names]:
        raise SystemExit(
            "compiled target rows differ from the arm's rows: "
            f"{compiled[:2]} vs {list(names[:2])}"
        )
    final = np.asarray(result.weights, dtype=np.float64)
    out = args.out / args.arm
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "weights.npz", weights=final, rows=rows)
    release_estimates = matrix @ final
    concept_estimates = concept_matrix @ final
    metrics = {
        "arm": args.arm,
        "n_targets": int(len(rows)),
        "epochs": int(args.epochs),
        "settings": RELEASE_SETTINGS,
        "initial_loss": float(result.initial_loss),
        "final_loss": float(result.final_loss),
        "seconds": time.time() - started,
        "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9,
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "git_sha": os.environ.get("MICROCOSM_GIT_SHA", ""),
        "trajectory": trajectory,
        "release_row_estimates": release_estimates.tolist(),
        "concept_row_estimates": concept_estimates.tolist(),
        "rebase_row_estimates": (rebase_matrix @ final).tolist(),
        "own_rows": rows.tolist(),
        "own_values": values.tolist(),
        "own_estimates": (solve_matrix @ final).tolist(),
        "own_loss_weights": loss_weights.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics))
    print(
        f"[{args.arm}] done: loss {result.initial_loss:.6f} -> "
        f"{result.final_loss:.6f} in {metrics['seconds']:.0f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
