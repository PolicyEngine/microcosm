"""Exact optima for the regularization-path tests' own problems.

``test_l2_basis.py`` checks the Adam solve's objective against the exact
optimum of the same convex program, on exactly the problems it generates. This
script computes those optima with CLARABEL and writes them to the test fixture
``packages/microcosm-calibrate/tests/fixtures/l2_basis_path_reference.json``.

The program (``target_loss_cap`` never binds on these surfaces; checked):

    minimize  mean_j |A_j w - b_j| / max(|b_j|, 1)
              + lambda * sum((w - d) ** 2 / d) / sum(d)
    subject to 0 <= w <= 5 d   (and sum(w) = sum(d) under mass="conserve")

Run (cvxpy is not a workspace dependency):

    uv run --with cvxpy python experiments/us-acs-local-l2-basis-20260928/test_path_reference.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cvxpy as cp
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TESTS = REPO / "packages" / "microcosm-calibrate" / "tests"
FIXTURE = TESTS / "fixtures" / "l2_basis_path_reference.json"


def _test_module():
    # The test module imports the repository's test_support package.
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    path = TESTS / "engine_free" / "shared" / "test_l2_basis.py"
    spec = importlib.util.spec_from_file_location("test_l2_basis_for_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _arrays(frame, targets):
    household = frame.table("household")
    names = [target.measure for target in targets]
    values = np.column_stack([household[name].to_numpy(np.float64) for name in names])
    goals = np.array([target.value for target in targets], dtype=np.float64)
    design = frame.resolve_weights("household").values.astype(np.float64)
    return values, goals, design


def solve(values, goals, design, lam, conserve):
    w = cp.Variable(len(design))
    scale = np.maximum(np.abs(goals), 1.0)
    loss = cp.sum(cp.abs(values.T @ w - goals) / scale) / len(goals)
    distance = cp.sum(cp.multiply(1.0 / (design * design.sum()), cp.square(w - design)))
    constraints = [w >= 0, w <= 5.0 * design]
    if conserve:
        constraints.append(cp.sum(w) == design.sum())
    cp.Problem(cp.Minimize(loss + lam * distance), constraints).solve(
        solver=cp.CLARABEL
    )
    weights = np.clip(np.asarray(w.value, dtype=np.float64), 0.0, None)
    miss = np.abs(values.T @ weights - goals) / scale
    if miss.max() >= 10.0:
        raise SystemExit("the test loss cap would bind at the optimum")
    loss_value = float(miss.mean())
    distance_value = float(np.sum((weights - design) ** 2 / design) / design.sum())
    return {
        "objective": loss_value + lam * distance_value,
        "loss": loss_value,
        "chi_square_distance": distance_value,
        "kish_ess": float(weights.sum() ** 2 / np.square(weights).sum()),
    }


def main() -> None:
    module = _test_module()
    entries = {}
    measured = []
    for case in module.PATH_CASES:
        case_id = module.path_case_id(case)
        frame, targets, results = module._solved_path(case_id)
        values, goals, design = _arrays(frame, targets)
        conserve = case["mass"] == "conserve"
        optimum = {
            str(lam): solve(values, goals, design, lam, conserve)
            for lam in module._LAMBDAS
        }
        excess = {}
        for lam, result in zip(module._LAMBDAS, results, strict=True):
            loss, distance = module._objective_parts(frame, targets, result.weights)
            excess[str(lam)] = loss + lam * distance - optimum[str(lam)]["objective"]
        measured.extend(excess.values())
        entries[case_id] = {
            "problem_sha256": module._problem_digest(frame, targets),
            "optimum": optimum,
            "solver_excess_when_generated": excess,
        }
        print(case_id, {k: round(v, 5) for k, v in excess.items()}, flush=True)
    worst = max(measured)
    # Twice the worst measured excess, rounded up to a 1e-3 step, absorbs
    # platform float differences in the Adam path.
    bound = float(np.ceil(2.0 * worst * 1000.0) / 1000.0)
    source = (TESTS / "engine_free" / "shared" / "test_l2_basis.py").read_bytes()
    payload = {
        "schema": "microcosm.l2_basis_path_reference.v1",
        "solver": "cvxpy CLARABEL",
        "program": (
            "mean |A w - b| / max(|b|, 1) + lambda * sum((w - d)^2 / d) / sum(d); "
            "0 <= w <= 5 d; sum(w) = sum(d) under conserve"
        ),
        "generator": "test_l2_basis._problem at the test module's PATH_CASES",
        "solver_settings": "calibrate(epochs=1500, learning_rate=0.02, "
        "max_weight_ratio=5.0, l2_basis='chi_square', seed=0) per case",
        "measured_max_excess_objective": worst,
        "excess_objective_bound": bound,
        "test_module_sha256_when_generated": hashlib.sha256(source).hexdigest(),
        "cases": entries,
    }
    FIXTURE.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"wrote {FIXTURE}")


if __name__ == "__main__":
    main()
