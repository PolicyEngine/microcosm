#!/usr/bin/env python3
"""Audit calendar information loss and compare the declared composition model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

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


def calendar_bounds_report(children):
    """Design-weighted identification intervals; no missing outcome imputation."""
    selected = children.loc[
        children.age.between(0, 12) & children.questionnaire_version.eq(1)
    ]
    rows = []
    for name, group in [
        ("all_regular_instrument", selected),
        *selected.groupby("attendance_status"),
    ]:
        weights = group.child_weight.to_numpy()
        rows.append(
            {
                "group": name,
                "children": len(group),
                "child_weight": float(weights.sum()),
                "mean_hours_lower": float(
                    np.average(group.calendar_ece_hours_lower, weights=weights)
                ),
                "mean_hours_upper": float(
                    np.average(group.calendar_ece_hours_upper, weights=weights)
                ),
                "mean_days_lower": float(
                    np.average(group.calendar_ece_days_lower, weights=weights)
                ),
                "mean_days_upper": float(
                    np.average(group.calendar_ece_days_upper, weights=weights)
                ),
                "definitely_in_care_share": float(
                    np.average(group.calendar_ece_hours_lower.gt(0), weights=weights)
                ),
                "possibly_in_care_share": float(
                    np.average(group.calendar_ece_hours_upper.gt(0), weights=weights)
                ),
                "hours_interval_width_le_1_share": float(
                    np.average(
                        (
                            group.calendar_ece_hours_upper
                            - group.calendar_ece_hours_lower
                        ).le(1),
                        weights=weights,
                    )
                ),
            }
        )
    return {
        "interpretation": "regular questionnaire only; measured calendar classifications conditional on published completeness status; identification bounds, not confidence intervals or completed schedules",
        "comparisons": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-tsv", type=Path, required=True)
    parser.add_argument("--calendar-tsv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report path must be new")
    source = load_nsece_childcare(args.household_tsv, args.calendar_tsv)
    plan = (
        Path(__file__).resolve().parents[1]
        / "experiments/us-childcare-attendance/calendar-selection-plan.txt"
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
        "interpretation": "fixed model comparison on previously inspected development data; no independent validation claim",
        "calendar_bounds": calendar_bounds_report(source.children),
    }
    print(json.dumps(report["calendar_bounds"]), flush=True)
    for name, composition in (("pooled_moments", False), ("composition", True)):
        arm = assess_sibling_schedules(
            source,
            pooled=True,
            composition=composition,
            include_observed_children=True,
            pooling_fit_objective="population_moments",
        )
        arm["dependence_only_screen_compatibility"] = coupling_screen_compatibility(arm)
        report[name] = arm
        print(
            name,
            "joint failures",
            sum(not c["passed"] for c in arm["diagnostic_screen"]["checks"]),
            "child failures",
            sum(
                not c["passed"]
                for c in arm["observed_child_validation"]["screen"]["checks"]
            ),
            flush=True,
        )
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
