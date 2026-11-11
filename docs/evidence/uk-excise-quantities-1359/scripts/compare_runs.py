"""Compare a base and a head stage run: which written columns are byte-identical.

Usage: python compare_runs.py <base run JSON> <head run JSON> <out JSON>

Reads the unsanitized run outputs (their per-column digests stay local) and
writes only aggregates: the columns both runs write and whether each is
identical, each differing column's prior-weighted total in both runs, the
columns only the head writes, the fit records and the gate verdicts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    base = json.loads(Path(sys.argv[1]).read_text())
    head = json.loads(Path(sys.argv[2]).read_text())
    shared = [c for c in base["output_sha256"] if c in head["output_sha256"]]
    differing = [c for c in shared if base["output_sha256"][c] != head["output_sha256"][c]]
    report = {
        "households": [base["households"], head["households"]],
        "fit_weight_records": [len(base["fit_weight_records"]), len(head["fit_weight_records"])],
        "shared_columns": len(shared),
        "identical_columns": [c for c in shared if c not in differing],
        "differing_columns": {
            c: {
                "base_weighted_total": base.get("weighted_totals", {}).get(c),
                "head_weighted_total": head["weighted_totals"][c],
            }
            for c in differing
        },
        "head_only_columns": [c for c in head["output_sha256"] if c not in base["output_sha256"]],
        "gates": {
            "base": {k: v["passed"] for k, v in base["gates"].items()},
            "head": {k: v["passed"] for k, v in head["gates"].items()},
        },
    }
    Path(sys.argv[3]).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print("identical", len(report["identical_columns"]), "of", len(shared), "differing", differing)


if __name__ == "__main__":
    main()
