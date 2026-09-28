#!/usr/bin/env python3
"""Compare the declared household-size challenger on separate survey partitions.

Writes aggregate diagnostics only. It does not alter the population build,
publish artifacts, or turn previously inspected data into external evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from microcosm.build.us_runtime import nsece_childcare_sibling_validation
from microcosm.build.us_runtime.childcare_attendance_receipt import (
    attendance_recipe_identity,
)
from microcosm.build.us_runtime.nsece_childcare import load_nsece_childcare
from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
    compare_household_size_matching,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-tsv", type=Path, required=True)
    parser.add_argument("--calendar-tsv", type=Path, required=True)
    parser.add_argument(
        "--partition", choices=("development", "validation"), required=True
    )
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error(
            "Report path must be new; existing evidence will not be overwritten"
        )
    source = load_nsece_childcare(args.household_tsv, args.calendar_tsv)
    report = compare_household_size_matching(source, partition=args.partition)
    root = Path(__file__).resolve().parents[1]
    plan = root / "experiments/us-childcare-attendance/household-size-plan.txt"
    report["plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    report["recipe"] = attendance_recipe_identity()
    report["diagnostic_code_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(nsece_childcare_sibling_validation.__file__))
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for arm in ("legacy", "household_size"):
        failures = [
            c["metric"]
            for c in report[arm]["diagnostic_screen"]["checks"]
            if not c["passed"]
        ]
        print(f"{arm}: {len(failures)}/15 screens failed")
        print(json.dumps(report[arm]["larger_households"], indent=2))


if __name__ == "__main__":
    main()
