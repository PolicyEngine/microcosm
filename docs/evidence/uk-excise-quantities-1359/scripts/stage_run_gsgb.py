"""Stage-only runs of lcfs_consumption without and with the gambling top-up (microcosm#1359).

Usage, from the tree (its packages first on PYTHONPATH):
  <venv>/bin/python <this> <tree> <spine.h5> <gsgb consumer facts jsonl> <out.json>

Runs the tree's committed lcfs_consumption stage on a finished spine twice in one
process: as declared, and with top_up_gambling_participation removed, so the
difference is the step's alone. Records aggregates only, at prior weights:

- every LCFS stage-health gate on the declared run, and the participation
  receipt (shares, bands);
- for the Lottery Duty base and one betting and gaming input (every other
  gambling input shares that key), the share of households carrying it and each
  equivalised-income decile's share of households carrying it and of the base,
  in both runs;
- by region, the share of households carrying each key in both runs, beside
  GSGB's published adult rates for the region (B.2, B.4): the out-of-sample
  check, since region is not a target.

Every cell is a weighted share of households resting on far more than ten
source households, which the script checks. Licensed output: it stays under
data/ukds/acceptance.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.household_income import household_level
from microcosm.build.uk_runtime.lcfs_consumption import UKLCFSConsumptionStageTransform
from microcosm.build.uk_runtime.national_frame import (
    load_uk_national_frame,
    uk_time_period,
)
from microcosm.build.uk_runtime.receipt_counts import distinct_count
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

LCFS = Path("/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/lcfs_2024_25")
KEYS = {
    "national_lottery": "national_lottery_ticket_spending",
    "betting_and_gaming": "remote_gaming_net_losses",
}
#: GSGB's region rows: the population each key's activities sit within.
POPULATIONS = {
    "national_lottery": "any_gambling",
    "betting_and_gaming": "excluding_lottery_draw_only",
}
SHEETS = {"past_4_weeks": "Table B.4", "past_12_months": "Table B.2"}
FLOOR = 10


def _tool(tree: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "tool", tree / "tools" / "verify_uk_identity_stability.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _run(stage, frame):
    transform = UKLCFSConsumptionStageTransform(
        stage=stage,
        engine=PolicyEngineUKEngine(),
        lcfs_hh_tab_path=LCFS / "dvhh_uk_anon_v3_2024.tab",
        lcfs_person_tab_path=LCFS / "dvper_uk_anon_2024.tab",
    )
    result = transform(frame)
    return transform, result.table("household")


def _gsgb_regions(path: Path) -> dict:
    rates: dict = {}
    for line in open(path):
        fact = json.loads(line)
        dims = fact["dimensions"]
        if (
            fact["concept_alignment"]["canonical_concept"]
            != "gambling_commission.gsgb.participation"
        ):
            continue
        if (
            dims.get("source_sheet") not in SHEETS.values()
            or dims.get("participation_scope") not in POPULATIONS.values()
        ):
            continue
        geography = fact["geography"]
        rates.setdefault(geography["id"], {"name": geography.get("name")})[
            f"{dims['participation_scope']}:{dims['participation_window']}"
        ] = fact["value"]
    return rates


def main() -> None:
    tree, spine, gsgb, out = (Path(arg) for arg in sys.argv[1:5])
    started = time.time()
    frame, _ = load_uk_national_frame(spine)
    frame = _tool(tree)._engine_safe_frame(frame)
    spec = load_country_spec("uk")
    declared = spec.sources.stage_map()["lcfs_consumption"]
    plain = dataclasses.replace(
        declared,
        operations=tuple(
            op
            for op in declared.operations
            if op.kind != "top_up_gambling_participation"
        ),
    )
    if len(plain.operations) != len(declared.operations) - 1:
        raise SystemExit("the stage does not declare the top-up.")
    transform, after = _run(declared, frame)
    _, before = _run(plain, frame)
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

    household = frame.table("household")
    weight = frame.weights_for("household").values
    mass = float(weight.sum())
    sources = (
        household["source_household_id"].to_numpy()
        if "source_household_id" in household
        else np.arange(len(household))
    )
    engine = PolicyEngineUKEngine()
    variable = "equiv_hbai_household_net_income"
    materialized = engine.materialize(frame, (variable,), uk_time_period(frame))
    income = np.asarray(
        household_level(frame, engine, variable, materialized[variable]), dtype=float
    )
    order = np.argsort(income, kind="stable")
    cumulative = np.cumsum(weight[order]) / mass
    decile = np.empty(len(income), dtype=int)
    decile[order] = np.minimum((cumulative * 10).astype(int), 9) + 1
    region = (
        household["region"].map(lambda v: getattr(v, "name", v)).astype(str).to_numpy()
    )

    def cells(mask_by_group: dict) -> None:
        for name, mask in mask_by_group.items():
            if 0 < distinct_count(sources, mask, weight) < FLOOR:
                raise SystemExit(
                    f"cell {name} rests on fewer than {FLOOR} source households."
                )

    report: dict = {
        "tree": str(tree),
        "spine": str(spine),
        "households": int(len(household)),
        "gates": gates,
        "gambling_participation": evidence.get("gambling_participation"),
        "fit_weight_records": [r.fit_name for r in transform.fit_weight_records][-4:],
        "keys": {},
    }
    for key, column in KEYS.items():
        entry: dict = {"column": column}
        for label, table in (("before", before), ("after", after)):
            values = table[column].to_numpy(dtype=float)
            positive = values > 0
            total = float(weight @ values)
            cells({f"{key}:{label}:positive": positive})
            by_decile = {}
            for d in range(1, 11):
                in_decile = decile == d
                cells(
                    {
                        f"{key}:{label}:decile{d}": in_decile & positive,
                        f"decile{d}": in_decile,
                    }
                )
                by_decile[d] = {
                    "households_carrying": float(
                        weight[in_decile & positive].sum() / weight[in_decile].sum()
                    ),
                    "share_of_base": float(
                        weight[in_decile] @ values[in_decile] / total
                    ),
                }
            by_region = {}
            for r in sorted(set(region)):
                in_region = region == r
                cells({f"{key}:{label}:{r}": in_region & positive, r: in_region})
                by_region[r] = float(
                    weight[in_region & positive].sum() / weight[in_region].sum()
                )
            entry[label] = {
                "households_carrying": float(weight[positive].sum() / mass),
                "frame_total": total,
                "by_decile": by_decile,
                "by_region": by_region,
            }
        report["keys"][key] = entry
    report["gsgb_regions"] = _gsgb_regions(gsgb)
    report["seconds"] = round(time.time() - started, 1)
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {out} in {report['seconds']} s")


if __name__ == "__main__":
    main()
