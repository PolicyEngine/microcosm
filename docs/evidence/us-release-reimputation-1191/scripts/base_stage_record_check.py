"""Check that a base H5 stores a base stage's outputs unchanged.

The base builder records each stage's signal-gate details in its run context.
For the retirement-contribution stage those include the weighted total of each
of the five leaves at the moment the stage finished. This script recomputes
the same totals from the exported base H5 (stored values times the stored
household weight of each person's household) and compares them exactly. Equal
totals show that no later base stage changed the leaves or the weights.

Usage:
    python base_stage_record_check.py BASE_H5 STAGE_RUN_CONTEXT_JSON OUT_JSON
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STAGE = "retirement_contributions_post_clone"
SIGNAL = "retirement_contributions_signal"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    base_path, context_path, out_path = (Path(arg) for arg in sys.argv[1:4])
    context = json.loads(context_path.read_text())
    record = next(r for r in context["stage_records"] if r["stage"] == STAGE)
    recorded = record["metadata"]["signals"][SIGNAL]["details"]["weighted_totals"]
    columns = list(recorded)
    person = pd.read_hdf(base_path, "person", columns=["person_household_id", *columns])
    household = pd.read_hdf(
        base_path, "household", columns=["household_id", "household_weight"]
    ).set_index("household_id")["household_weight"]
    weights = person["person_household_id"].map(household).to_numpy(dtype=np.float64)
    receipt = {
        "base_h5": str(base_path),
        "base_h5_sha256": sha256(base_path),
        "stage_run_context": str(context_path),
        "stage_run_context_sha256": sha256(context_path),
        "stage": STAGE,
        "stage_checkpoint_sha256": record["checkpoint_sha256"],
        "columns": {},
    }
    for column in columns:
        total = float(np.sum(person[column].to_numpy(dtype=np.float64) * weights))
        receipt["columns"][column] = {
            "recorded_at_stage": recorded[column],
            "recomputed_from_h5": total,
            "equal": total == recorded[column],
        }
    Path(out_path).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")
    print(json.dumps({c: v["equal"] for c, v in receipt["columns"].items()}))


if __name__ == "__main__":
    main()
