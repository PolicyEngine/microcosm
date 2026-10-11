"""M4: re-solve the stored 2026-09-30 national problem with rows masked.

Arms (same doctrine, seed, epochs and code for both, so the comparison is like
with like):
  base   - every bound row (reproduces the build's solve on current code)
  drop4  - obr.vat, obr.fuel_duties_cars and the two ONS energy rows masked;
           family_equal loss weights recomputed over the remaining rows, as a
           register sign-out would

No engine run and no target compilation: the problem artifact already holds the
compiled rows, and the spine H5 supplies the frame whose household axis and
initial weights the problem was built on (both checked exactly).

Usage: python m4_resolve.py solve --arm base|drop4
       python m4_resolve.py report
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PE = Path("/Users/mariajuaristi/Desktop/PolicyEngine")
RUN = PE / "runs/uk-national-main-c5a1cba8"
SPINE = (
    PE
    / "data/ukds/acceptance/national-main-20260930/spine-main-c5a1cba8/spine-main-c5a1cba8.h5"
)
OUT = Path(__file__).resolve().parent
MASKS = {
    "base": set(),
    "drop4": {
        "obr.vat@2025",
        "obr.fuel_duties_cars@2025",
        "ons.household_electricity_expenditure@2025",
        "ons.household_gas_expenditure@2025",
    },
}
DIVISIONS = [
    "food_and_non_alcoholic_beverages_consumption",
    "alcohol_and_tobacco_consumption",
    "clothing_and_footwear_consumption",
    "housing_water_and_electricity_consumption",
    "household_furnishings_consumption",
    "health_consumption",
    "transport_consumption",
    "communication_consumption",
    "recreation_consumption",
    "education_consumption",
    "restaurants_and_hotels_consumption",
    "miscellaneous_consumption",
]


def load_problem():
    from microcosm.calibrate.artifacts import decode_problem

    return decode_problem(
        (RUN / "uk.full.national_problem.problem.artifact").read_bytes()
    )


def solve(arm: str) -> None:
    from microcosm.build.uk_runtime.national_doctrine import (
        uk_national_target_loss_weights,
    )
    from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
    from microcosm.calibrate.solve import calibrate
    from microcosm.calibrate.target import TargetSet

    problem = load_problem()
    frame, _ = load_uk_national_frame(SPINE)
    axis = tuple(frame.table("household")["household_id"].tolist())
    if axis != problem.entity_ids:
        raise SystemExit("spine household axis differs from the problem's")
    initial = frame.weights_for("household")
    if initial.kind != problem.problem.initial_weights.kind or not np.array_equal(
        initial.values, problem.problem.initial_weights.values
    ):
        raise SystemExit(
            "spine household weights differ from the problem's initial weights"
        )
    names = list(problem.problem.names)
    missing = MASKS[arm] - set(names)
    if missing:
        raise SystemExit(f"mask names not in the problem: {sorted(missing)}")
    keep = [i for i, name in enumerate(names) if name not in MASKS[arm]]
    targets = problem.to_target_set()
    families = [str(meta["family"]) for meta in problem.target_metadata]
    binding = problem.bindings
    loss_weights = uk_national_target_loss_weights(
        [families[i] for i in keep], rule=str(binding["target_weight_rule"])
    )
    if arm == "base" and not np.allclose(
        loss_weights, np.asarray(binding["target_loss_weights"], dtype=float)
    ):
        raise SystemExit("recomputed base loss weights differ from the bound ones")
    started = time.time()
    result = calibrate(
        frame,
        TargetSet([targets[i] for i in keep]),
        weight_entity="household",
        epochs=1500,
        learning_rate=0.02,
        seed=0,
        mass="free",
        mass_reason=binding["mass_reason"],
        max_weight_ratio=binding["max_weight_ratio"],
        target_loss_weights=loss_weights,
        target_loss_cap=binding["target_loss_cap"],
    )
    weights = np.asarray(result.weights, dtype=float)
    np.save(OUT / f"weights_{arm}.npy", weights)
    (OUT / f"solve_{arm}.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "masked": sorted(MASKS[arm]),
                "rows": len(keep),
                "families": len(set(families[i] for i in keep)),
                "closing_loss": float(getattr(result, "closing_loss", float("nan"))),
                "seconds": round(time.time() - started, 1),
            },
            indent=1,
        )
    )
    print(f"{arm}: {len(keep)} rows, {time.time() - started:.0f}s")


def kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def composition(h: pd.DataFrame, w: np.ndarray) -> dict:
    fuel = (h.petrol_spending + h.diesel_spending).to_numpy()
    flag = h.has_fuel_consumption.astype(bool).to_numpy()
    gas = h.gas_consumption.to_numpy()
    total = w.sum()
    out = {"households_m": total / 1e6, "kish_ess": kish(w)}
    for column in DIVISIONS + [
        "petrol_spending",
        "diesel_spending",
        "electricity_consumption",
        "gas_consumption",
    ]:
        out[f"{column}_bn"] = float(np.dot(w, h[column].to_numpy(dtype=float)) / 1e9)
    out["consumption_bn"] = sum(out[f"{c}_bn"] for c in DIVISIONS)
    out["gas_connected_share"] = float(w[gas > 0].sum() / total)
    out["petrol_positive_share"] = float(
        w[h.petrol_spending.to_numpy() > 0].sum() / total
    )
    out["diesel_positive_share"] = float(
        w[h.diesel_spending.to_numpy() > 0].sum() / total
    )
    out["fuel_car_zero_fuel_share"] = float(w[flag & (fuel <= 0)].sum() / w[flag].sum())
    return out


def report() -> None:
    from microcosm.calibrate.artifacts import decode_solution

    problem = load_problem()
    names = list(problem.problem.names)
    families = [str(meta["family"]) for meta in problem.target_metadata]
    a = problem.problem.matrix
    b = np.asarray(problem.problem.target_vector, dtype=float)
    w0 = np.asarray(problem.problem.initial_weights.values, dtype=float)
    stored = decode_solution(
        (RUN / "uk.full.dense.solution.artifact").read_bytes()
    ).weights
    arms = {"design": w0, "stored": np.asarray(stored, dtype=float)}
    for arm in MASKS:
        path = OUT / f"weights_{arm}.npy"
        if path.exists():
            arms[arm] = np.load(path)
    household = pd.read_hdf(SPINE, "household")
    masked = MASKS["drop4"]
    common = [i for i, n in enumerate(names) if n not in masked]
    summary: dict = {"arms": {}}
    errors = {}
    for arm, w in arms.items():
        est = a @ w
        rel = (est - b) / np.where(b != 0, np.abs(b), 1.0)
        errors[arm] = rel
        moved = 0.5 * np.abs(w - w0).sum() / w0.sum()
        summary["arms"][arm] = {
            "within_10pct_common": float(np.mean(np.abs(rel[common]) <= 0.10)),
            "past_25pct_common": int(np.sum(np.abs(rel[common]) > 0.25)),
            "median_abs_error_common": float(np.median(np.abs(rel[common]))),
            "share_of_mass_moved_from_design": float(moved),
            "masked_rows_error": {
                names[i]: float(rel[i]) for i, n in enumerate(names) if n in masked
            },
            **composition(household, w),
        }
    if "base" in errors and "drop4" in errors:
        delta = np.abs(errors["drop4"]) - np.abs(errors["base"])
        order = [i for i in np.argsort(-np.abs(delta)) if names[i] not in masked][:20]
        summary["largest_fit_changes_drop4_vs_base"] = [
            {
                "target": names[i],
                "family": families[i],
                "base": float(errors["base"][i]),
                "drop4": float(errors["drop4"][i]),
            }
            for i in order
        ]
        by_family = {}
        for i in common:
            by_family.setdefault(families[i], []).append(delta[i])
        summary["family_mean_abs_error_change"] = {
            k: float(np.mean(v))
            for k, v in sorted(by_family.items(), key=lambda kv: -abs(np.mean(kv[1])))
        }
    if "base" in arms:
        summary["base_vs_stored_max_abs_weight_ratio_diff"] = float(
            np.max(
                np.abs(
                    arms["base"] / np.where(arms["stored"] > 0, arms["stored"], 1) - 1
                )
            )
        )
    (OUT / "m4_report.json").write_text(json.dumps(summary, indent=1))
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k != "largest_fit_changes_drop4_vs_base"
            },
            indent=1,
        )[:6000]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["solve", "report"])
    parser.add_argument("--arm", choices=sorted(MASKS))
    args = parser.parse_args()
    if args.command == "solve":
        solve(args.arm)
    else:
        report()
