"""Measure the Adam calibrator's distance from the exact penalized optimum.

For synthetic surfaces small enough for an exact convex solver, this runs
``microcosm.calibrate.calibrate`` with the chi-square L2 basis under three
mass settings, ``mass="conserve"`` with the default ``"projection"``
parametrization, ``mass="conserve"`` with ``"softmax"``, and ``mass="free"``,
and compares each solve's objective with CLARABEL's optimum of the same
convex program:

    minimize  mean_j |A_j w - b_j| / max(|b_j|, 1)
              + lambda * sum((w - d) ** 2 / d) / sum(d)
    subject to 0 <= w <= 5 d  (and sum(w) = sum(d) under conservation).

The capped-MAPE cap (1.0 in the ACS tool) never binds on these surfaces, so
the capped and uncapped losses coincide at every solution reported here; the
script checks this.

Run (cvxpy is not a workspace dependency):

    uv run --with cvxpy python experiments/us-acs-local-l2-basis-20260928/optimizer_reference.py

Writes ``results/optimizer_reference.csv`` and
``results/optimizer_reference_summary.json`` next to this file.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd

from microcosm.calibrate import Target, TargetSet, calibrate
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

HERE = Path(__file__).resolve().parent
LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0)
CAP = 5.0
LOSS_CAP = 1.0
VARIANTS = (
    ("conserve", "projection"),
    ("conserve", "softmax"),
    ("free", "projection"),
)


def small_problem(seed: int, factors: tuple[float, float]):
    """120 records and six targets; the (1.02, 1.3) factors press on the total."""
    rng = np.random.default_rng(seed)
    n, k = 120, 6
    values = rng.lognormal(0.0, 1.0, (n, k)) * (rng.random((n, k)) < 0.6)
    design = rng.lognormal(2.0, 0.8, n)
    targets = (values * design[:, None]).sum(0) * rng.uniform(*factors, k)
    return values, design, targets


def large_problem(seed: int):
    """3,000 records, 150 sparse targets in both directions, two design strata.

    Four percent of records carry heavier, more dispersed design weights, like
    the ACS local staging's donor spine.
    """
    rng = np.random.default_rng(seed)
    n, k = 3000, 150
    values = rng.lognormal(0.0, 1.0, (n, k)) * (rng.random((n, k)) < 0.05)
    values[:, :10] = rng.lognormal(0.0, 0.5, (n, 10)) * (rng.random((n, 10)) < 0.5)
    heavy = rng.random(n) < 0.04
    design = np.where(heavy, rng.lognormal(4.0, 1.5, n), rng.lognormal(2.0, 0.6, n))
    targets = (values * design[:, None]).sum(0) * rng.normal(1.0, 0.12, k)
    return values, design, targets


def kernel_solve(values, design, targets, lam, mass, parametrization, epochs):
    n, k = values.shape
    frame = Frame(
        {
            "person": pd.DataFrame(
                {"person_id": range(n), "person_household_id": range(n)}
            ),
            "household": pd.DataFrame(
                {"household_id": range(n), **{f"x{j}": values[:, j] for j in range(k)}}
            ),
        },
        EntitySchema(group_entities=("household",)),
        {"household": Weights(values=design, kind=WeightKind.DESIGN)},
    )
    target_set = TargetSet(
        tuple(
            Target(
                name=f"x{j}",
                entity="household",
                value=float(targets[j]),
                measure=f"x{j}",
            )
            for j in range(k)
        )
    )
    result = calibrate(
        frame,
        target_set,
        epochs=epochs,
        learning_rate=0.02,
        mass=mass,
        max_weight_ratio=CAP,
        target_loss_cap=LOSS_CAP,
        l2_lambda=lam,
        l2_basis="chi_square",
        mass_parametrization=parametrization,
        seed=0,
    )
    return result.weights


def objective_parts(values, design, targets, weights):
    scale = np.maximum(np.abs(targets), 1.0)
    miss = np.abs(values.T @ weights - targets) / scale
    distance = float(np.sum((weights - design) ** 2 / design) / design.sum())
    return float(miss.mean()), float(np.minimum(miss, LOSS_CAP).mean()), distance


def reference_solve(values, design, targets, lam, conserve):
    w = cp.Variable(len(design))
    scale = np.maximum(np.abs(targets), 1.0)
    objective = cp.sum(cp.abs(values.T @ w - targets) / scale) / len(targets)
    objective += lam * cp.sum(
        cp.multiply(1.0 / (design * design.sum()), cp.square(w - design))
    )
    constraints = [w >= 0, w <= CAP * design]
    if conserve:
        constraints.append(cp.sum(w) == design.sum())
    cp.Problem(cp.Minimize(objective), constraints).solve(solver=cp.CLARABEL)
    return np.clip(np.asarray(w.value, dtype=np.float64), 0.0, None)


def main() -> None:
    problems = [
        (f"small_s{seed}_{low}", small_problem(seed, (low, high)), 1500)
        for seed in range(8)
        for low, high in ((0.7, 1.4), (1.02, 1.3))
    ] + [(f"large_s{seed}", large_problem(seed), 800) for seed in range(3)]
    rows = []
    started = time.time()
    for name, (values, design, targets), epochs in problems:
        for lam in LAMBDAS:
            references = {}
            for conserve in (True, False):
                ref_w = reference_solve(values, design, targets, lam, conserve)
                references[conserve] = objective_parts(values, design, targets, ref_w)
            for mass, parametrization in VARIANTS:
                weights = kernel_solve(
                    values, design, targets, lam, mass, parametrization, epochs
                )
                loss, capped, distance = objective_parts(
                    values, design, targets, weights
                )
                ref_loss, _, ref_distance = references[mass == "conserve"]
                ess = float(weights.sum() ** 2 / np.square(weights).sum())
                rows.append(
                    {
                        "problem": name,
                        "records": len(design),
                        "targets": len(targets),
                        "epochs": epochs,
                        "lambda": lam,
                        "mass": mass,
                        "parametrization": parametrization,
                        "loss": loss,
                        "loss_cap_binds": capped != loss,
                        "chi_square_distance": distance,
                        "objective": loss + lam * distance,
                        "reference_loss": ref_loss,
                        "reference_chi_square_distance": ref_distance,
                        "reference_objective": ref_loss + lam * ref_distance,
                        "kish_ess": ess,
                        "mass_ratio": float(weights.sum() / design.sum()),
                    }
                )
        print(f"{name} done ({time.time() - started:.0f}s)", flush=True)
    frame = pd.DataFrame(rows)
    frame["excess_objective"] = frame.objective - frame.reference_objective
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    frame.to_csv(out / "optimizer_reference.csv", index=False)

    def nonmonotone(group):
        path = group.sort_values("lambda").chi_square_distance.to_numpy()
        return bool((np.diff(path) > 1e-3).any())

    summary = {}
    for scale_name, subset in (
        ("small", frame[frame.problem.str.startswith("small")]),
        ("large", frame[frame.problem.str.startswith("large")]),
    ):
        by_variant = {}
        for (mass, parametrization), group in subset.groupby(
            ["mass", "parametrization"]
        ):
            paths = group.groupby("problem").apply(nonmonotone, include_groups=False)
            by_variant[f"{mass}/{parametrization}"] = {
                "excess_objective_mean_by_lambda": {
                    str(lam): float(g.excess_objective.mean())
                    for lam, g in group.groupby("lambda")
                },
                "excess_objective_max": float(group.excess_objective.max()),
                "share_of_problems_with_nonmonotone_distance": float(paths.mean()),
                "loss_cap_ever_binds": bool(group.loss_cap_binds.any()),
            }
        summary[scale_name] = by_variant
    (out / "optimizer_reference_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
