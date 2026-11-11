"""Stage-only run of nts_bus_travel then lcfs_consumption on a finished spine.

Usage, from the tree whose stage code runs (see run_stage_dev_at.sh):
  python <this> <spine.h5> <out.json>

The NTS recipient income band moved from the engine's household_gross_income
to HBAI gross income (microcosm#1164). The LCFS stage prices bus fares from
the NTS trips, so the two stages run in sequence, each tree's own code on the
tabs its own spec pins ($PE/data/ukds/nts_2002_24/tab/ and
$PE/data/ukds/lcfs_<vintage>/), and the run records at the spine's prior
weights:
- the recipients' NTS income band shares, the bus-use band shares and the
  annual trips per series;
- the bus fare spending and transport totals after the LCFS stage, and every
  NTS and LCFS stage-health gate on the receipts.

Disclosure control (UKDS EUL CD137 cl. 8, CD171 s5.2.1): the JSON holds
weighted shares and totals over the whole frame; the household dump is
written beside it, outside the repository.
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
from microcosm.build.uk_runtime import nts_bus_travel as nts
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

TREE = Path.cwd()
PE = Path(os.environ.get("PE", Path.home() / "Desktop" / "PolicyEngine"))
NTS = PE / "data" / "ukds" / "nts_2002_24" / "tab"


def _tool():
    spec = importlib.util.spec_from_file_location(
        "tool", TREE / "tools" / "verify_uk_identity_stability.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def _artifact_path(stage, role: str, directory: Path) -> Path:
    artifact = next(a for a in stage.artifacts if a.get("role") == role)
    return directory / artifact["filename"]


def _lcfs_dir(stage) -> Path:
    artifact = next(a for a in stage.artifacts if a.get("role") == "lcfs_household_tab")
    return PE / "data" / "ukds" / f"lcfs_{artifact['vintage']}"


def _gates(spec, stage_name: str, evidence) -> dict[str, object]:
    gates = {}
    for gate in spec.gates.gates:
        parameters = dict(gate.parameters)
        if gate.gate != "stage_health" or parameters.get("stage") != stage_name:
            continue
        checked = uk_stage_health_gate(
            evidence=evidence,
            stage=stage_name,
            check=str(parameters["check"]),
            parameters=parameters,
        )
        gates[gate.id] = {"passed": checked.passed, "failures": list(checked.failures)}
    return gates


def _shares(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    mass = float(weights.sum())
    return {
        str(value): float(weights[values == value].sum()) / mass
        for value in sorted(set(values.tolist()))
    }


def main() -> None:
    spine, out = Path(sys.argv[1]), Path(sys.argv[2])
    started = time.time()
    frame, _ = load_uk_national_frame(spine)
    frame = _tool()._engine_safe_frame(frame)
    spec = load_country_spec("uk")
    stages = spec.sources.stage_map()
    engine = PolicyEngineUKEngine()
    nts_stage = stages["nts_bus_travel"]
    derive = nts._operation(nts_stage, nts.CLEAN_NTS_TRAVEL_TABLES_KIND)
    band_parameters = nts._operation(nts_stage, nts.IMPUTE_BUS_USE_BAND_KIND)
    recipient = nts.recipient_predictors(
        frame,
        engine,
        columns=nts.NTSColumns.from_parameters(derive),
        income_band_edges=[float(v) for v in band_parameters["income_band_edges"]],
        region_remap=dict(band_parameters.get("region_remap", {})),
    )
    weight_by_household = dict(
        zip(
            frame.table("household")["household_id"],
            frame.weights_for("household").values,
            strict=True,
        )
    )
    person_weights = np.asarray(
        [weight_by_household[h] for h in recipient["person_household_id"]], dtype=float
    )
    nts_transform = nts.UKNTSBusTravelStageTransform(
        stage=nts_stage,
        engine=engine,
        nts_household_tab_path=_artifact_path(nts_stage, "nts_household_tab", NTS),
        nts_individual_tab_path=_artifact_path(nts_stage, "nts_individual_tab", NTS),
        nts_trip_tab_path=_artifact_path(nts_stage, "nts_trip_tab", NTS),
        nts_stage_tab_path=_artifact_path(nts_stage, "nts_stage_tab", NTS),
        nts_ticket_tab_path=_artifact_path(nts_stage, "nts_ticket_tab", NTS),
    )
    after_nts = nts_transform(frame)
    nts_evidence = nts_transform.checkpoint_metadata()["evidence"]
    person = after_nts.table("person")
    trips = {
        column: float(person_weights @ person[column].to_numpy(dtype=float))
        for column in nts.SERIES_TRIP_COLUMNS.values()
    }
    lcfs_stage = stages["lcfs_consumption"]
    lcfs_dir = _lcfs_dir(lcfs_stage)
    lcfs_transform = lcfs.UKLCFSConsumptionStageTransform(
        stage=lcfs_stage,
        engine=engine,
        lcfs_hh_tab_path=_artifact_path(lcfs_stage, "lcfs_household_tab", lcfs_dir),
        lcfs_person_tab_path=_artifact_path(lcfs_stage, "lcfs_person_tab", lcfs_dir),
    )
    result = lcfs_transform(after_nts)
    lcfs_evidence = lcfs_transform.checkpoint_metadata()["evidence"]
    household = result.table("household")
    weights = result.weights_for("household").values.astype(float)
    report = {
        "spine": spine.name,
        "seconds": round(time.time() - started, 1),
        "nts_income_band_shares_persons": _shares(
            recipient["income_band"].to_numpy(), person_weights
        ),
        "bus_use_band_shares_persons": _shares(
            person[nts.BAND_COLUMN].to_numpy(), person_weights
        ),
        "annual_trips": trips,
        "bus_fare_spending": float(weights @ household["bus_fare_spending"].to_numpy()),
        "transport_consumption": float(
            weights @ household["transport_consumption"].to_numpy()
        ),
        "gates": {
            **_gates(spec, "nts_bus_travel", nts_evidence),
            **_gates(spec, "lcfs_consumption", lcfs_evidence),
        },
    }
    out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float) + "\n")
    dump = household[["household_id", "bus_fare_spending", "transport_consumption"]]
    dump = dump.assign(weight=weights)
    dump.to_parquet(out.with_suffix(".households.parquet"))
    print(json.dumps({k: report[k] for k in report if k != "gates"}, indent=1))
    print({g: v["passed"] for g, v in report["gates"].items()})


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
