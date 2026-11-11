"""Per-record gradient scale of the release's calibration loss at full size.

Adam's step is ``lr * m / (sqrt(v) + eps)`` with ``eps = 1e-8``: a record whose
gradient is well above ``eps`` moves about ``lr`` whatever its size, one near
or below ``eps`` moves in proportion to it. This measures the exact log-weight
gradient of the release's capped-MAPE loss (cap 1.0, uniform target weights)
at the design weights and at the published release's weights, and the split
of its signs, which decides whether the projection parametrization's
same-sign stall can occur at this scale.

Run: ``uv run python experiments/us-acs-local-l2-basis-20260928/gradient_scale.py``
Writes ``results/gradient_scale.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
CHECKPOINT = Path(
    "/Users/maxghenis/PolicyEngine/_build_artifacts/acs-local-l2-basis-20260928/checkpoint"
)
RELEASE_WEIGHTS = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/"
    "run/release/checkpoints/weights_latest.npz"
)
EPS = 1e-8
CAP = 1.0


def main() -> None:
    matrix = sparse.load_npz(CHECKPOINT / "target_matrix.npz").tocsr()
    matrix = matrix.astype(np.float64)
    goals = pd.read_parquet(CHECKPOINT / "targets_meta.parquet")["value"].to_numpy()
    design = pd.read_parquet(CHECKPOINT / "households.parquet")[
        "design_weight"
    ].to_numpy(np.float64)
    scale = np.maximum(np.abs(goals), 1.0)
    out = {"eps": EPS, "loss_cap": CAP, "n_targets": int(len(goals))}
    for label, weights in (
        ("design", design),
        ("release_0923", np.load(RELEASE_WEIGHTS)["weights"].astype(np.float64)),
    ):
        miss = (matrix @ weights - goals) / scale
        # Rows past the cap contribute no gradient to capped MAPE.
        live = np.abs(miss) < CAP
        grad_w = matrix.T @ (np.sign(miss) * live / scale) / len(goals)
        grad = grad_w * weights
        magnitude = np.abs(grad)
        signs = np.sign(grad[grad != 0])
        out[label] = {
            "log_weight_grad_abs_quantiles": {
                q: float(np.quantile(magnitude, float(q)))
                for q in ("0.1", "0.25", "0.5", "0.75", "0.9", "0.99")
            },
            "share_at_or_below_eps": float((magnitude <= EPS).mean()),
            "share_at_least_10_eps": float((magnitude >= 10 * EPS).mean()),
            "share_positive_of_nonzero": float((signs > 0).mean()),
            "share_negative_of_nonzero": float((signs < 0).mean()),
            "live_targets": int(live.sum()),
        }
    (HERE / "results" / "gradient_scale.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
