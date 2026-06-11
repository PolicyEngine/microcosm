"""Bounded-recalibration experiment for the published populace-us weights.

The shipped (unbounded) calibration hit 90.7% of targets within 10% but left a
1.29M max household weight (16 records > 500k, 370 > 100k) — heavy reliance on
a few super-weighted records. solve.py's max_weight_ratio caps each weight at
ratio * its initial weight; the pool's initial weights are near-uniform, so the
ratio behaves like an absolute cap. Sweep ratios and compare loss / within-10%
/ concentration against the unbounded run (loss 0.049, 90.7%, max 1.29M).
"""

import json
import time

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(6)

from populace.calibrate import Target, TargetSet, calibrate
from populace.frame import EntitySchema, Frame, WeightKind, Weights

ART = "/Users/maxghenis/.claude-worktrees/microplex-spec-build/artifacts"


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


surf = np.load(f"{ART}/v5_target_surface_raw.npz", allow_pickle=True)
A = surf["A"].astype(np.float64)
b = surf["b"].astype(np.float64)
w0 = surf["w0"].astype(np.float64)
names = [str(x) for x in surf["names"]]
n_hh, n_t = A.shape
log(
    f"w0: mean {w0.mean():.0f}, p1 {np.percentile(w0, 1):.1f}, "
    f"p50 {np.percentile(w0, 50):.0f}, p99 {np.percentile(w0, 99):.0f}, "
    f"max {w0.max():.0f} -> ratio*w0 cap shape"
)

household = pd.DataFrame({"household_id": np.arange(n_hh, dtype=np.int64)})
person = pd.DataFrame(
    {
        "person_id": np.arange(n_hh, dtype=np.int64),
        "person_household_id": np.arange(n_hh, dtype=np.int64),
    }
)
targets = TargetSet(
    tuple(
        Target(
            name=names[t],
            entity="household",
            aggregation="sum",
            value=float(b[t]),
            measure=(lambda _f, col=A[:, t].copy(): col),
        )
        for t in range(n_t)
    )
)

results = {}
for ratio in (25.0, 50.0, 100.0, 200.0):
    frame = Frame(
        {"person": person.copy(), "household": household.copy()},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=w0.copy(), kind=WeightKind.DESIGN)},
    )
    t0 = time.time()
    res = calibrate(
        frame,
        targets,
        weight_entity="household",
        epochs=3000,
        learning_rate=0.15,
        mass="free",
        max_weight_ratio=ratio,
        seed=0,
    )
    cw = res.frame.resolve_weights("household").values
    row = {
        "ratio": ratio,
        "final_loss": round(res.final_loss, 5),
        "within_10pct": round(res.fraction_within_10pct, 4),
        "weight_sum_M": round(float(cw.sum()) / 1e6, 2),
        "weight_max": round(float(cw.max())),
        "n_gt_100k": int((cw > 1e5).sum()),
        "n_gt_500k": int((cw > 5e5).sum()),
        "at_cap": int((cw >= ratio * w0 * 0.999).sum()),
        "secs": round(time.time() - t0),
    }
    results[str(int(ratio))] = row
    np.save(f"{ART}/bounded_recal_w_ratio{int(ratio)}.npy", cw)
    log(json.dumps(row))

with open(f"{ART}/bounded_recal_results.json", "w") as f:
    json.dump(results, f, indent=1)
log("EXPERIMENT DONE")
