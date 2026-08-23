"""Fit UK childcare take-up rates from a local enhanced-FRS H5."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

TOOL_VERSION = 1
TARGETS = {
    "spending": {
        "tfc": 0.63,
        "extended": 2.5,
        "targeted": 0.6,
        "universal": 1.7,
    },
    "caseload": {
        "tfc": 985,
        "extended": 740,
        "targeted": 130,
        "universal": 490,
    },
}
TARGET_CITATIONS = {
    "tfc": "HMRC Tax-Free Childcare statistics, June 2025 release.",
    "other_childcare": (
        "Department for Education dedicated schools grant national funding "
        "allocations, 2024 to 2025."
    ),
}
INITIAL_PARAMS = np.array([0.5, 0.5, 0.5, 0.5, 15.0, 5.0], dtype=float)
BOUNDS = ((0, 1), (0, 1), (0, 1), (0, 1), (5.0, 30.0), (1.0, 10.0))

SimulationRunner = Callable[
    [Path, np.ndarray, int], tuple[dict[str, float], dict[str, float]]
]


def draw_childcare_inputs(
    benunit_count: int, params: np.ndarray, *, seed: int
) -> dict[str, np.ndarray]:
    """Generate seeded childcare inputs without touching numpy global state."""

    rng = np.random.default_rng(seed)
    tfc, extended, targeted, universal, hours_mean, hours_sd = params
    return {
        "would_claim_tfc": rng.random(benunit_count) < tfc,
        "would_claim_extended_childcare": rng.random(benunit_count) < extended,
        "would_claim_targeted_childcare": rng.random(benunit_count) < targeted,
        "would_claim_universal_childcare": rng.random(benunit_count) < universal,
        "maximum_extended_childcare_hours_usage": np.clip(
            rng.normal(hours_mean, hours_sd, benunit_count), 0, 30
        ),
    }


def fit_childcare_takeup(
    input_h5: Path,
    *,
    seed: int = 42,
    maxiter: int = 5,
    runner: SimulationRunner | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Fit rates and return a deterministic receipt payload."""

    runner = runner or _policyengine_runner

    def objective(params: np.ndarray) -> float:
        spending, caseload = runner(input_h5, params, seed)
        loss = 0.0
        for key, target in TARGETS["spending"].items():
            loss += (spending[key] / target - 1) ** 2
        for key, target in TARGETS["caseload"].items():
            loss += (caseload[key] / target - 1) ** 2
        return float(loss)

    result = minimize(
        objective,
        INITIAL_PARAMS,
        bounds=BOUNDS,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "eps": 1e-2},
    )
    spending, caseload = runner(input_h5, result.x, seed)
    return {
        "tool": "fit_uk_childcare_takeup",
        "tool_version": TOOL_VERSION,
        "input_h5": str(input_h5),
        "input_sha256": _sha256(input_h5),
        "seed": seed,
        "generated_at": generated_at
        if generated_at is not None
        else datetime.now(UTC).date().isoformat(),
        "optimizer": {
            "method": "L-BFGS-B",
            "maxiter": maxiter,
            "success": bool(result.success),
            "loss": float(result.fun),
        },
        "params": {
            "tax_free_childcare": float(result.x[0]),
            "extended_childcare": float(result.x[1]),
            "targeted_childcare": float(result.x[2]),
            "universal_childcare": float(result.x[3]),
            "extended_hours_mean": float(result.x[4]),
            "extended_hours_sd": float(result.x[5]),
        },
        "targets": TARGETS,
        "target_citations": TARGET_CITATIONS,
        "fitted": {"spending": spending, "caseload": caseload},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enhanced-frs-h5", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=5)
    args = parser.parse_args()
    receipt = fit_childcare_takeup(
        args.enhanced_frs_h5, seed=args.seed, maxiter=args.maxiter
    )
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _policyengine_runner(
    input_h5: Path, params: np.ndarray, seed: int
) -> tuple[dict[str, float], dict[str, float]]:
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=str(input_h5))
    benunit_count = sim.calculate("benunit_id").values.shape[0]
    for variable, values in draw_childcare_inputs(
        benunit_count, params, seed=seed
    ).items():
        sim.set_input(variable, 2024, values)
    df = sim.calculate_dataframe(
        [
            "is_child_receiving_tax_free_childcare",
            "is_child_receiving_extended_childcare",
            "is_child_receiving_universal_childcare",
            "is_child_receiving_targeted_childcare",
        ],
        2024,
    )
    spending = {
        "tfc": sim.calculate("tax_free_childcare", 2024).sum() / 1e9,
        "extended": sim.calculate("extended_childcare_entitlement", 2024).sum() / 1e9,
        "targeted": sim.calculate("targeted_childcare_entitlement", 2024).sum() / 1e9,
        "universal": sim.calculate("universal_childcare_entitlement", 2024).sum() / 1e9,
    }
    caseload = {
        "tfc": df["is_child_receiving_tax_free_childcare"].sum() / 1e3,
        "extended": df["is_child_receiving_extended_childcare"].sum() / 1e3,
        "universal": df["is_child_receiving_universal_childcare"].sum() / 1e3,
        "targeted": df["is_child_receiving_targeted_childcare"].sum() / 1e3,
    }
    return spending, caseload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
