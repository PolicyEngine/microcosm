"""Stage-only development run of lcfs_consumption for microcosm#1359.

Usage, from the tree whose stage code runs (PYTHONPATH puts its packages first):
  <venv>/bin/python <this> <tree> <spine.h5> <out.json>

Re-runs the tree's committed lcfs_consumption stage on a finished spine (as the
#1113 stage trials did) and records, for every household column the stage
writes, the sha256 of its bytes, so a base run and a head run on the same spine
can be compared column by column; plus the head's excise_shares receipt and the
stage's own gates. Licensed output: it stays under data/ukds/acceptance.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.lcfs_consumption import (
    UKLCFSConsumptionStageTransform,
)
from microcosm.build.uk_runtime.national_frame import load_uk_national_frame
from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from microcosm.frame.adapters.policyengine_uk import PolicyEngineUKEngine

LCFS = Path("/Users/mariajuaristi/Desktop/PolicyEngine/data/ukds/lcfs_2024_25")


def _tool(tree: Path):
    spec = importlib.util.spec_from_file_location(
        "tool", tree / "tools" / "verify_uk_identity_stability.py"
    )
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def main() -> None:
    tree, spine, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    started = time.time()
    frame, _ = load_uk_national_frame(spine)
    frame = _tool(tree)._engine_safe_frame(frame)
    stage = load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    transform = UKLCFSConsumptionStageTransform(
        stage=stage,
        engine=PolicyEngineUKEngine(),
        lcfs_hh_tab_path=LCFS / "dvhh_uk_anon_v3_2024.tab",
        lcfs_person_tab_path=LCFS / "dvper_uk_anon_2024.tab",
    )
    result = transform(frame)
    household = result.table("household")
    columns = transform.output_columns()
    digests = {
        column: hashlib.sha256(
            np.ascontiguousarray(household[column].to_numpy()).tobytes()
        ).hexdigest()
        for column in columns
    }
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
        "tree": str(tree),
        "spine": str(spine),
        "households": int(len(household)),
        "seconds": round(time.time() - started, 1),
        "fit_weight_records": [record.fit_name for record in transform.fit_weight_records],
        "output_sha256": digests,
        "gates": gates,
        "excise_shares": evidence.get("excise_shares"),
        "excise_quantities": evidence.get("excise_quantities"),
        # Prior-weighted frame totals of every written column (aggregates).
        "weighted_totals": {
            column: float(
                result.weights_for("household").values
                @ household[column].to_numpy(dtype=float)
            )
            for column in columns
        },
    }
    out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(f"wrote {out} in {report['seconds']} s")


if __name__ == "__main__":
    main()
