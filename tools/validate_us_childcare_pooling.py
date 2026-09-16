#!/usr/bin/env python3
"""Compare fixed partially pooled matching with the existing attendance model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from microcosm.build.us_runtime import (
    nsece_childcare_pooling,
    nsece_childcare_sibling_validation,
)
from microcosm.build.us_runtime.childcare_attendance_receipt import (
    attendance_recipe_identity,
)
from microcosm.build.us_runtime.nsece_childcare import load_nsece_childcare
from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
    assess_sibling_schedules,
    coupling_screen_compatibility,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-tsv", type=Path, required=True)
    parser.add_argument("--calendar-tsv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--compare-moment-fit",
        action="store_true",
        help="Include the separately planned exploratory population-moment objective",
    )
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report path must be new")
    source = load_nsece_childcare(args.household_tsv, args.calendar_tsv)
    plan = (
        Path(__file__).resolve().parents[1]
        / "experiments/us-childcare-attendance/pooled-matching-plan.txt"
    )
    report = {
        "source": source.source_receipt,
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "recipe": attendance_recipe_identity(),
        "diagnostic_code_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(nsece_childcare_pooling.__file__),
                Path(nsece_childcare_sibling_validation.__file__),
            )
        },
        "production_ready": False,
        "production_recipe_changed": False,
        "interpretation": "five household-separated folds on previously inspected development data; no untouched evaluation claim; calendar selection remains unidentified",
    }
    arms = [
        ("legacy", False, "pair_squared_error"),
        ("pooled", True, "pair_squared_error"),
    ]
    if args.compare_moment_fit:
        extra_plan = plan.with_name("pooled-moments-plan.txt")
        report["moment_plan_sha256"] = hashlib.sha256(
            extra_plan.read_bytes()
        ).hexdigest()
        arms.append(("pooled_moments", True, "population_moments"))
    for name, pooled, objective in arms:
        report[name] = assess_sibling_schedules(
            source,
            pooled=pooled,
            include_observed_children=True,
            pooling_fit_objective=objective,
        )
        report[name]["dependence_only_screen_compatibility"] = (
            coupling_screen_compatibility(report[name])
        )
        print(name, json.dumps(report[name]["larger_households"]), flush=True)
        print(
            "joint failures",
            sum(not c["passed"] for c in report[name]["diagnostic_screen"]["checks"]),
            flush=True,
        )
        print(
            "child failures",
            sum(
                not c["passed"]
                for c in report[name]["observed_child_validation"]["screen"]["checks"]
            ),
            flush=True,
        )
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
