"""Stage-only development run of lcfs_consumption on the head-r3 spine (microcosm#1113).

Usage, from the worktree:
  .venv/bin/python <this> <spine.h5> <out.json>

Re-runs the committed lcfs_consumption stage (Group A: road-fuel incidence and
level, recomposed totals) on the finished head-r3 spine, as the #1063 c8 stage
trials did, and records:
- the new receipts and every lcfs stage-health gate run on them;
- the stage's columns before (the spine's head-r3 values) and after, at the
  spine's prior weights.

The spine already holds the stages after lcfs_consumption (the CGT clones
among them), so the run is a development check of the receipts and the
invariants, not a rebuild. No calibration and no engine simulation beyond the
stage's own predictor materialisation.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import lcfs_consumption as _lcfs
from microcosm.build.uk_runtime.lcfs_consumption import (
    UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
    UKLCFSConsumptionStageTransform,
)

# The base tree (before microcosm#1113) has no division-column constant.
UK_LCFS_COICOP_DIVISION_COLUMNS = getattr(
    _lcfs,
    "UK_LCFS_COICOP_DIVISION_COLUMNS",
    tuple(_lcfs.CONSUMPTION_VARIABLE_RENAMES[f"p6{d:02d}"] for d in range(1, 13)),
)
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

TREE = Path.cwd()
PE = Path(os.environ.get("PE", Path.home() / "Desktop" / "PolicyEngine"))
LCFS = PE / "data" / "ukds" / "lcfs_2023_24"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "tool", TREE / "tools" / "verify_uk_identity_stability.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _summary(household, weights) -> dict[str, object]:
    def total(column: str) -> float:
        return float(weights @ household[column].to_numpy(dtype=float))

    fuel = (household["petrol_spending"] + household["diesel_spending"]).to_numpy()
    flagged = household["has_fuel_consumption"].to_numpy(dtype=bool)
    energy = (
        household["electricity_consumption"] + household["gas_consumption"]
    ).to_numpy()
    housing = household["housing_water_and_electricity_consumption"].to_numpy()
    transport = household["transport_consumption"].to_numpy()
    consumption = sum(
        household[column].to_numpy(dtype=float)
        for column in UK_LCFS_COICOP_DIVISION_COLUMNS
    )
    mass = float(weights.sum())
    flagged_mass = float(weights[flagged].sum())
    return {
        "totals": {
            column: total(column) for column in UK_LCFS_CONSUMPTION_TARGET_COLUMNS
        },
        "consumption_total": float(weights @ consumption),
        "fuel_buying_share": float(weights[fuel > 0].sum()) / mass,
        "flagged_share": flagged_mass / mass,
        "flagged_without_fuel_share": (
            float(weights[flagged & ~(fuel > 0)].sum()) / flagged_mass
        ),
        "mean_fuel_among_buyers": float(weights[fuel > 0] @ fuel[fuel > 0])
        / float(weights[fuel > 0].sum()),
        "gas_connected_share": float(
            weights[household["gas_consumption"].to_numpy() > 0].sum()
        )
        / mass,
        "housing_below_energy_share": float(weights[housing < energy - 1e-9].sum())
        / mass,
        "transport_below_fuel_share": float(weights[transport < fuel - 1e-9].sum())
        / mass,
    }


def main() -> None:
    spine, out = Path(sys.argv[1]), Path(sys.argv[2])
    started = time.time()
    frame, _ = load_uk_national_frame(spine)
    frame = _tool()._engine_safe_frame(frame)
    weights = frame.weights_for("household").values.astype(float)
    before = _summary(frame.table("household"), weights)
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    transform = UKLCFSConsumptionStageTransform(
        stage=stage,
        engine=PolicyEngineUKEngine(),
        lcfs_hh_tab_path=LCFS / "dvhh_ukanon_v2_2023.tab",
        lcfs_person_tab_path=LCFS / "dvper_ukanon_202324_2023.tab",
    )
    result = transform(frame)
    after = _summary(result.table("household"), result.weights_for("household").values)
    evidence = transform.checkpoint_metadata()["evidence"]
    gates = {}
    for gate in load_country_spec("uk").gates.gates:
        parameters = dict(gate.parameters)
        if gate.gate != "stage_health" or parameters.get("stage") != "lcfs_consumption":
            continue
        checked = uk_stage_health_gate(
            evidence=evidence,
            stage="lcfs_consumption",
            check=str(parameters["check"]),
            parameters=parameters,
        )
        gates[gate.id] = {"passed": checked.passed, "failures": list(checked.failures)}
    report = {
        "spine": spine.name,
        "households": int(len(weights)),
        "seconds": round(time.time() - started, 1),
        "before_head_r3": before,
        "after_group_a": after,
        "gates": gates,
        "receipts": {
            key: evidence.get(key)
            for key in (
                "road_fuel_incidence",
                "road_fuel_level",
                "recomposed_totals",
                "fuel_litres_audit",
                "division_capture",
                "has_fuel_consumption",
                "donor_floor",
            )
        },
        "support_clip_remainders": {
            column: entry
            for column, entry in evidence["support_clip"]["columns"].items()
            if column
            in (
                "housing_water_excluding_domestic_energy",
                "transport_excluding_road_fuel",
            )
        },
        "fit_weight_records": [
            record.fit_name for record in transform.fit_weight_records
        ],
        "donor_uprating": evidence.get("donor_uprating"),
        "bus_pricing_total": evidence.get("bus_pricing", {}).get("frame")
        if isinstance(evidence.get("bus_pricing"), dict)
        else None,
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    keep = ["household_id", *UK_LCFS_CONSUMPTION_TARGET_COLUMNS, "has_fuel_consumption"]
    old_household = frame.table("household")
    dump = old_household[[c for c in keep if c in old_household]].add_prefix("old_")
    new_household = result.table("household")
    for column in keep:
        dump[f"new_{column}"] = new_household[column].to_numpy()
    for column in (
        "num_vehicles",
        "region",
        "tenure_type",
        "household_is_spi_synthetic",
        "household_is_capital_gains_clone",
    ):
        if column in old_household:
            dump[column] = old_household[column].to_numpy()
    dump["weight"] = weights
    dump.to_parquet(out.with_suffix(".households.parquet"))
    print(json.dumps({"gates": gates, "seconds": report["seconds"]}, indent=1))
    for key in (
        "consumption_total",
        "flagged_without_fuel_share",
        "fuel_buying_share",
        "mean_fuel_among_buyers",
        "gas_connected_share",
        "housing_below_energy_share",
        "transport_below_fuel_share",
    ):
        print(key, before[key], "->", after[key])
    for column in (
        "petrol_spending",
        "diesel_spending",
        "electricity_consumption",
        "gas_consumption",
        "housing_water_and_electricity_consumption",
        "transport_consumption",
    ):
        old, new = before["totals"][column], after["totals"][column]
        print(
            column,
            round(old / 1e9, 3),
            "->",
            round(new / 1e9, 3),
            f"{new / old - 1:+.3%}",
        )
    capture = evidence.get("division_capture")
    if capture:
        for column, entry in capture["divisions"].items():
            print(
                "capture",
                entry["coicop"],
                column,
                round(entry["frame"] / 1e9, 2),
                round(entry["survey_scope"] / 1e9, 2),
                round(entry["capture"], 3),
            )
        print(
            "capture total",
            {
                k: (round(v / 1e9, 2) if abs(v) > 10 else round(v, 4))
                for k, v in capture["total"].items()
            },
        )
    audit = evidence.get("fuel_litres_audit")
    if audit:
        for column, entry in audit["fuels"].items():
            print(
                "litres",
                column,
                round(entry.get("desnz_cars_share", float("nan")), 4),
                round(entry["frame_over_cars_benchmark"], 3),
            )
        print(
            "litres total",
            round(audit["frame_over_cars_benchmark"], 3),
            "obr uniform",
            audit.get("frame_over_obr_uniform_cars_benchmark"),
        )


if __name__ == "__main__":
    main()
