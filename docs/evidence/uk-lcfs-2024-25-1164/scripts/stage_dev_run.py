"""Stage-only development run of lcfs_consumption on a finished spine (microcosm#1164).

Usage, from the tree whose stage code runs (see run_stage_dev_at.sh):
  python <this> <spine.h5> <out.json>

Re-runs the tree's committed lcfs_consumption stage on a finished UK spine
and records, at the spine's prior weights:
- every lcfs stage-health gate run on the stage's receipts;
- the terminal support check: households above each committed support bound;
- the stage's columns before (the spine's) and after;
- the donor's energy split, the household kWh ceilings, the prepayment share
  and the recipients' NEED income bands, where the tree has them.

The LCFS tabs are the ones the tree's own spec pins, read from
$PE/data/ukds/lcfs_<vintage>/, so the base tree runs on its survey year and
this tree on its own. The spine already holds the stages after
lcfs_consumption, so the run is a development check of the receipts and the
invariants, not a rebuild.

Disclosure control (UKDS EUL CD137 cl. 8, CD171 s5.2.1): the JSON written
holds weighted totals and shares and counts, with counts under ten written
"<10"; the support-clip receipts' donor extremes are dropped. The household
dump (unit records) is written beside the JSON, outside the repository.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime import lcfs_consumption as lcfs
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

TREE = Path.cwd()
PE = Path(os.environ.get("PE", Path.home() / "Desktop" / "PolicyEngine"))
DIVISIONS = tuple(lcfs.CONSUMPTION_VARIABLE_RENAMES[f"p6{d:02d}"] for d in range(1, 13))
PREPAYMENT = getattr(lcfs, "UK_LCFS_PREPAYMENT_METER_COLUMN", None)


def _count(value: int) -> int | str:
    """A count, written "<10" when it is a small cell."""

    return int(value) if value == 0 or value >= 10 else "<10"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "tool", TREE / "tools" / "verify_uk_identity_stability.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _lcfs_tabs(stage) -> tuple[Path, Path]:
    def path(role: str) -> Path:
        artifact = next(a for a in stage.artifacts if a.get("role") == role)
        return (
            PE / "data" / "ukds" / f"lcfs_{artifact['vintage']}" / artifact["filename"]
        )

    return path("lcfs_household_tab"), path("lcfs_person_tab")


def _summary(household, weights) -> dict[str, object]:
    def total(column: str) -> float:
        return float(weights @ household[column].to_numpy(dtype=float))

    mass = float(weights.sum())
    fuel = (household["petrol_spending"] + household["diesel_spending"]).to_numpy()
    flagged = household["has_fuel_consumption"].to_numpy(dtype=bool)
    electricity = household["electricity_consumption"].to_numpy(dtype=float)
    gas = household["gas_consumption"].to_numpy(dtype=float)
    housing = household["housing_water_and_electricity_consumption"].to_numpy()
    transport = household["transport_consumption"].to_numpy()
    consumption = sum(household[c].to_numpy(dtype=float) for c in DIVISIONS)
    summary = {
        "totals": {c: total(c) for c in lcfs.UK_LCFS_CONSUMPTION_TARGET_COLUMNS},
        "consumption_total": float(weights @ consumption),
        "fuel_buying_share": float(weights[fuel > 0].sum()) / mass,
        "flagged_share": float(weights[flagged].sum()) / mass,
        "gas_connected_share": float(weights[gas > 0].sum()) / mass,
        "electricity_share_of_energy": float(weights @ electricity)
        / float(weights @ (electricity + gas)),
        "housing_below_energy_share": float(
            weights[housing < electricity + gas - 1e-9].sum()
        )
        / mass,
        "transport_below_fuel_share": float(weights[transport < fuel - 1e-9].sum())
        / mass,
    }
    if PREPAYMENT and PREPAYMENT in household:
        prepays = household[PREPAYMENT].to_numpy(dtype=bool)
        summary["prepayment_share"] = float(weights[prepays].sum()) / mass
    return summary


def _above_committed_bounds(household) -> dict[str, object]:
    """Households above each committed LCFS support bound (the terminal check)."""

    path = (
        TREE
        / "packages/microcosm-build/src/microcosm/build/uk"
        / "lcfs_consumption_support_bounds.json"
    )
    bounds = json.loads(path.read_text(encoding="utf-8"))["bounds"]
    above = {}
    for column, (low, high) in bounds.items():
        values = household[column].to_numpy(dtype=float)
        above[column] = {
            "bound": [low, high],
            "households_below": _count(int((values < low).sum())),
            "households_above": _count(int((values > high).sum())),
        }
    return above


def _need_income_bands(frame, engine, stage) -> dict[str, object] | None:
    """The recipients' weighted shares by NEED income band, on the band income.

    The base tree bands recipients on the household table's gross income when
    the spine carries it and on HBAI net income otherwise; this tree
    materialises the engine's gross income.
    """

    from microcosm.build.uk_runtime.energy_pricing import income_band_ids

    energy = lcfs.lcfs_energy_pricing(stage)
    if energy is None:
        return None
    recipient = lcfs.recipient_predictors(frame, engine)
    weights = frame.weights_for("household").values.astype(float)
    bands = income_band_ids(
        recipient["household_gross_income"].to_numpy(dtype=float), energy.margins
    )
    mass = float(weights.sum())
    predictors = getattr(lcfs, "UK_LCFS_CONSUMPTION_ENGINE_PREDICTORS", ())
    engine_gross = "household_gross_income" in predictors
    on_table = "household_gross_income" in frame.table("household")
    if "income_tax" in predictors:
        concept = "HBAI net income plus the HBAI deductions"
    elif engine_gross:
        concept = "engine gross income"
    else:
        concept = "spine gross income" if on_table else "HBAI net income fallback"
    # A band with fewer than ten recipients is a small cell: no share.
    return {
        "band_income": concept,
        "shares": {
            str(band): float(weights[bands == band].sum()) / mass
            if int((bands == band).sum()) >= 10
            else "<10 households"
            for band in sorted(set(bands.tolist()))
        },
    }


def _sanitized_support_clip(receipt) -> dict[str, object]:
    columns = {}
    for column, entry in (receipt or {}).get("columns", {}).items():
        columns[column] = {
            key: (_count(value) if isinstance(value, int) else value)
            for key, value in entry.items()
            if key not in ("donor_min", "donor_max")
        }
    return {"columns": columns}


def main() -> None:
    spine, out = Path(sys.argv[1]), Path(sys.argv[2])
    started = time.time()
    frame, _ = load_uk_national_frame(spine)
    frame = _tool()._engine_safe_frame(frame)
    weights = frame.weights_for("household").values.astype(float)
    before = _summary(frame.table("household"), weights)
    spec = load_country_spec("uk")
    stage = spec.sources.stage_map()["lcfs_consumption"]
    household_tab, person_tab = _lcfs_tabs(stage)
    engine = PolicyEngineUKEngine()
    transform = lcfs.UKLCFSConsumptionStageTransform(
        stage=stage,
        engine=engine,
        lcfs_hh_tab_path=household_tab,
        lcfs_person_tab_path=person_tab,
    )
    result = transform(frame)
    new_household = result.table("household")
    after = _summary(new_household, result.weights_for("household").values)
    evidence = transform.checkpoint_metadata()["evidence"]
    gates = {}
    for gate in spec.gates.gates:
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
    energy_rake = evidence.get("energy_rake") or {}
    report = {
        "tree_lcfs_tabs": [household_tab.name, person_tab.name],
        "spine": spine.name,
        "households": int(len(weights)),
        "seconds": round(time.time() - started, 1),
        "before_spine": before,
        "after_stage": after,
        "gates": gates,
        "terminal_support": _above_committed_bounds(new_household),
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
                "donor_energy_split",
                PREPAYMENT,
            )
            if key
        },
        "energy_rake_ceiling": energy_rake.get("ceiling"),
        "energy_rake_fit": {
            margin: {
                "max_abs_relative_deviation": block.get("max_abs_relative_deviation"),
                "cells": {
                    cell: {
                        fuel: entry[fuel].get("relative_deviation")
                        for fuel in ("electricity_kwh", "gas_kwh")
                    }
                    for cell, entry in block.get("cells", {}).items()
                },
            }
            for margin, block in (energy_rake.get("fit") or {}).items()
        },
        "energy_rake_sweep_residuals": {
            fuel: (values[-3:] if values else values)
            for fuel, values in (energy_rake.get("sweep_residuals") or {}).items()
        },
        "energy_rake_level_factor": energy_rake.get("level_factor"),
        "gas_connected_share": energy_rake.get("gas_connected_share"),
        "gas_connection_by_region": {
            region: {
                key: entry.get(key)
                for key in ("target_share", "share_before", "share_after", "shortfall")
            }
            for region, entry in (
                (energy_rake.get("gas_connection") or {}).get("by_region") or {}
            ).items()
        },
        "support_clip": _sanitized_support_clip(evidence.get("support_clip")),
        "need_income_bands": _need_income_bands(frame, engine, stage),
        "fit_weight_records": [r.fit_name for r in transform.fit_weight_records],
        "donor_uprating": evidence.get("donor_uprating"),
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    keep = [
        "household_id",
        *lcfs.UK_LCFS_CONSUMPTION_TARGET_COLUMNS,
        "has_fuel_consumption",
        *([PREPAYMENT] if PREPAYMENT and PREPAYMENT in new_household else []),
    ]
    old_household = frame.table("household")
    dump = old_household[[c for c in keep if c in old_household]].add_prefix("old_")
    for column in keep:
        dump[f"new_{column}"] = new_household[column].to_numpy()
    for column in ("num_vehicles", "region", "tenure_type", "accommodation_type"):
        if column in old_household:
            dump[column] = old_household[column].to_numpy()
    dump["weight"] = weights
    dump.to_parquet(out.with_suffix(".households.parquet"))
    print(json.dumps({"gates": gates, "seconds": report["seconds"]}, indent=1))
    print(
        "terminal support above:",
        {
            c: v["households_above"]
            for c, v in report["terminal_support"].items()
            if v["households_above"] != 0
        },
    )
    print("energy ceiling:", report["energy_rake_ceiling"])
    print(
        "electricity share:",
        before["electricity_share_of_energy"],
        "->",
        after["electricity_share_of_energy"],
    )


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
