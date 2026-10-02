"""Score the equal-weight solves on the weighted loss.

The weighted grid's solves record both losses; the equal-weight grid's record
only the unweighted one. This scores every equal-weight run's final weights
(``weights.npz`` beside the checkpoint), the published release's weights and
the design weights on the shared module's target-loss weights, the same way
``sweep.py`` scores a weighted run:

* training fit over the run's training rows, weighted as a build calibrating
  on those rows would weight them (the full surface, or one fold's
  complement);
* held-out fit over the run's held-out fold, with each target's full-surface
  weight.

It writes ``results/cross_scores.json``: per run, the weighted capped error
and weighted share within 10% on training and held-out targets, the run's
receipt sha256 and the weights' sha256, plus the full-surface weight vector's
content hash, so analyze.py can set the two grids side by side on one loss.

Run: ``uv run python experiments/us-acs-local-l2-basis-20260928/cross_score.py``
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gradient_scale import RELEASE_WEIGHTS  # noqa: E402
from sweep import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_OUT_ROOT,
    SHARED_TARGET_LOSS_WEIGHTS,
    RunSpec,
    estimates_for,
    fit_block,
    load_inputs,
    loss_vector_sha256,
    loss_weights_for,
    sha256,
    training_rows,
)

RUNS = HERE / "results" / "runs"
OUT = HERE / "results" / "cross_scores.json"


def _measures(block: dict) -> dict:
    overall = block["overall"]
    return {
        "weighted_mean_capped_scaled_error": overall[
            "weighted_mean_capped_scaled_error"
        ],
        "weighted_fraction_within_10pct": overall["weighted_fraction_within_10pct"],
        "mean_capped_scaled_error": overall["mean_capped_scaled_error"],
        "fraction_within_10pct": overall["fraction_within_10pct"],
    }


def main() -> None:
    inputs = load_inputs(DEFAULT_CHECKPOINT, verify=True, need_registry=True)
    all_rows = np.arange(inputs.matrix.shape[0], dtype=np.int64)
    full = loss_weights_for(
        RunSpec(run_id="full", target_weighting="shared"), inputs, all_rows
    )
    fold_weights = {
        fold: loss_weights_for(
            RunSpec(run_id=f"fold{fold}", target_weighting="shared"),
            inputs,
            training_rows(inputs, fold),
        )
        for fold in inputs.folds
    }

    def score(weights: np.ndarray, fold: int | None) -> dict:
        estimates = estimates_for(inputs, weights)
        if fold is None:
            return {
                "train": _measures(
                    fit_block(estimates, inputs.meta, all_rows, loss_weights=full.train)
                )
            }
        train = training_rows(inputs, fold)
        held = inputs.folds[fold]
        return {
            "train": _measures(
                fit_block(
                    estimates,
                    inputs.meta,
                    train,
                    loss_weights=fold_weights[fold].train,
                )
            ),
            "holdout": _measures(
                fit_block(estimates, inputs.meta, held, loss_weights=full.full[held])
            ),
        }

    runs: dict[str, dict] = {}
    for receipt in sorted(RUNS.glob("*.json")):
        payload = json.loads(receipt.read_text())
        spec = payload.get("spec") or {}
        if payload.get("kind") != "calibrate":
            continue
        if spec.get("target_weighting", "equal") != "equal":
            continue
        path = DEFAULT_OUT_ROOT / spec["run_id"] / "weights.npz"
        if not path.exists():
            runs[spec["run_id"]] = {"status": "weights not found", "path": str(path)}
            continue
        weights = np.asarray(np.load(path)["weights"], dtype=np.float64)
        fold = spec.get("holdout_fold")
        runs[spec["run_id"]] = {
            "holdout_fold": fold,
            "receipt_sha256": sha256(receipt),
            "weights_sha256": sha256(path),
            **score(weights, fold),
        }
    published = np.asarray(np.load(RELEASE_WEIGHTS)["weights"], dtype=np.float64)
    references = {
        "published_release": (published, str(RELEASE_WEIGHTS), sha256(RELEASE_WEIGHTS)),
        "design": (inputs.design, "checkpoint households.parquet design_weight", None),
    }
    for label, (weights, source, digest) in references.items():
        runs[label] = {
            "source": source,
            "weights_sha256": digest,
            "full_surface": score(weights, None),
            **{f"fold_{fold}": score(weights, fold) for fold in (0, 1)},
        }
    payload = {
        "function": SHARED_TARGET_LOSS_WEIGHTS,
        "registry_sha256": inputs.registry_receipt["verified_registry_sha256"],
        "full_surface_loss_vector_sha256": loss_vector_sha256(
            inputs.meta["name"].to_numpy(), full.full
        ),
        "runs": runs,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {OUT}: {len(runs)} entries")
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
