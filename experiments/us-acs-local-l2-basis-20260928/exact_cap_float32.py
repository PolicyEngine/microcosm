"""What float32 does to the softmax cap step at full scale.

Applies the kernel's exact projection to full-scale log-weights (a softmax
run's final weights, moved by a step-sized random perturbation), so every
log-weight is at or below its float32 log cap and the vector meets the total.
Two measurements on each projected vector:

- **The realized weights.** The solve realizes ``total * softmax(log_w)`` in
  float32, whose normalizer is a float32 sum over 1.59M records. Realize the
  same float32 log-weights with a float64 normalizer too; the difference is
  what the float32 softmax alone adds to the cap ratio.
- **microcosm#1078's 32-round loop on that vector.** It needs no cap work, so
  an exact stopping test would pass in the first round. Record the shift the
  loop computes each round, the records it lifts over their float32 log caps,
  and whether the rounds run out.

Run on Modal, off the shared host, from the sweep's volume (it writes
``results/exact_cap_float32.json``)::

    modal run experiments/us-acs-local-l2-basis-20260928/exact_cap_modal.py::float32

or locally against the build artifacts with
``uv run python experiments/us-acs-local-l2-basis-20260928/exact_cap_float32.py``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from microcosm.calibrate import solve

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path("/Users/maxghenis/PolicyEngine/_build_artifacts")
CHECKPOINT = ARTIFACTS / "acs-local-l2-basis-20260928/checkpoint"
RUN_WEIGHTS = (
    ARTIFACTS / "acs-local-l2-basis-20260928/runs/dup_soft_chi_s050_0.03/weights.npz"
)
MAX_WEIGHT_RATIO = 5.0
#: Adam's learning rate in the release's solve: the size of one log-weight step.
STEP = 0.02
TRIALS = 12
SEED = 3


def rounds_32_on(log_w: torch.Tensor, total: float, log_upper: torch.Tensor) -> dict:
    """microcosm#1078's cap step (b115745d1) on a copy, with what each round did."""
    log_w = log_w.clone()
    shifts, lifted, lifted_log_cap = [], [], 0.0
    for _ in range(32):
        shift = math.log(total) - float(torch.logsumexp(log_w, dim=0).item())
        shifts.append(shift)
        log_w.add_(shift)
        over = log_w > log_upper
        lifted.append(int(over.sum()))
        if not lifted[-1]:
            break
        lifted_log_cap = max(lifted_log_cap, float(log_upper[over].abs().max()))
        log_w.clamp_(max=log_upper)
    return {
        "rounds": len(shifts),
        "ran_out": bool(lifted[-1]),
        "shift_min": min(shifts),
        "shift_max": max(shifts),
        "records_lifted_over_their_caps_per_round_max": max(lifted),
        "largest_abs_log_cap_among_lifted_records": lifted_log_cap,
    }


def measure(checkpoint: Path = CHECKPOINT, run_weights: Path = RUN_WEIGHTS) -> dict:
    design = pd.read_parquet(checkpoint / "households.parquet")[
        "design_weight"
    ].to_numpy(np.float64)
    weights = np.load(run_weights)["weights"].astype(np.float64)
    total = float(design.sum())
    cap = torch.tensor(MAX_WEIGHT_RATIO * design, dtype=torch.float64)
    log_upper = torch.log(torch.tensor(MAX_WEIGHT_RATIO * design, dtype=torch.float32))
    rng = np.random.default_rng(SEED)
    trials = []
    for _ in range(TRIALS):
        log_w = torch.tensor(
            np.log(weights) + rng.normal(0.0, STEP, design.size), dtype=torch.float32
        )
        rounds = solve._project_softmax_log_weights_(
            log_w, total, log_upper.to(torch.float64)
        )
        if not bool((log_w <= log_upper).all()):
            raise SystemExit("a projected log-weight passed its float32 log cap")
        loop = rounds_32_on(log_w, total, log_upper)
        float32 = total * torch.softmax(log_w, dim=0)
        exact = log_w.to(torch.float64)
        float64 = total * torch.exp(exact - torch.logsumexp(exact, dim=0))
        trials.append(
            {
                "active_set_rounds": int(rounds),
                "records_at_cap": int((log_w >= log_upper).sum()),
                "float64_shift_still_needed": math.log(total)
                - float(torch.logsumexp(log_w.to(torch.float64), dim=0)),
                "rounds_32_loop": loop,
                "float32_softmax": {
                    "max_cap_ratio_minus_one": solve._max_cap_ratio(float32, cap) - 1.0,
                    "total_error": float(float32.to(torch.float64).sum()) / total - 1.0,
                },
                "float64_normalizer": {
                    "max_cap_ratio_minus_one": solve._max_cap_ratio(float64, cap) - 1.0,
                    "total_error": float(float64.sum()) / total - 1.0,
                },
            }
        )

    def spread(realization: str, key: str) -> dict:
        values = [trial[realization][key] for trial in trials]
        return {"min": min(values), "max": max(values)}

    out = {
        "records": int(design.size),
        "max_weight_ratio": MAX_WEIGHT_RATIO,
        "log_weight_step_sd": STEP,
        "trials": len(trials),
        "seed": SEED,
        "torch": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "float32_softmax": {
            key: spread("float32_softmax", key)
            for key in ("max_cap_ratio_minus_one", "total_error")
        },
        "float64_normalizer": {
            key: spread("float64_normalizer", key)
            for key in ("max_cap_ratio_minus_one", "total_error")
        },
        # How far the float32 softmax's cap ratio sits above its own total
        # error: what is left once the normalizer's error is taken out.
        "float32_cap_ratio_minus_total_error": {
            "min": min(
                t["float32_softmax"]["max_cap_ratio_minus_one"]
                - t["float32_softmax"]["total_error"]
                for t in trials
            ),
            "max": max(
                t["float32_softmax"]["max_cap_ratio_minus_one"]
                - t["float32_softmax"]["total_error"]
                for t in trials
            ),
        },
        "rounds_32_loop_on_the_projected_vectors": {
            "trials_ran_out": sum(t["rounds_32_loop"]["ran_out"] for t in trials),
            "shift": {
                "min": min(t["rounds_32_loop"]["shift_min"] for t in trials),
                "max": max(t["rounds_32_loop"]["shift_max"] for t in trials),
            },
            "float32_log_total_residual": math.log(total)
            - float(torch.tensor(math.log(total), dtype=torch.float32)),
            "records_lifted_over_their_caps_per_round_max": max(
                t["rounds_32_loop"]["records_lifted_over_their_caps_per_round_max"]
                for t in trials
            ),
            "largest_abs_log_cap_among_lifted_records": max(
                t["rounds_32_loop"]["largest_abs_log_cap_among_lifted_records"]
                for t in trials
            ),
        },
        "by_trial": trials,
    }
    return out


def write(out: dict) -> None:
    path = HERE / "results" / "exact_cap_float32.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "by_trial"}, indent=1))


if __name__ == "__main__":
    write(measure())
