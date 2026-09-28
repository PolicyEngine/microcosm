#!/usr/bin/env python3
"""Evaluate declared interval-conditioned training and both assumption tilts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import validate_us_childcare_qrf as qrf_tool

from microcosm.build.us_runtime import (
    nsece_childcare_pooling,
    nsece_childcare_qrf,
    nsece_childcare_sibling_validation,
)
from microcosm.build.us_runtime.childcare_attendance_receipt import (
    attendance_recipe_identity,
)
from microcosm.build.us_runtime.nsece_childcare_qrf import (
    QRFChildcareSchedules,
    interval_training_rows,
)
from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
    _household_splits,
    _observed_child_summary,
)


def _records(model, target, complete):
    model.prepare(target)
    for _, child in target.iterrows():
        values, _, probability = model.distribution(child)
        yield (
            float(child.child_weight),
            min(int(child.childcare_household_size), 3),
            bool(complete.loc[child.source_household_id]),
            np.array(
                [
                    float(child.childcare_days_per_week > 0),
                    child.childcare_days_per_week,
                    child.ece_hours_per_week,
                ]
            ),
            np.average(values, axis=0, weights=probability),
            np.average(values**2, axis=0, weights=probability),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-tsv", type=Path, required=True)
    parser.add_argument("--calendar-tsv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report path must be new")
    source, children = qrf_tool.load_source_children(
        args.household_tsv, args.calendar_tsv
    )
    records = {
        name: []
        for name in ("complete_only", "interval", "lower_hours", "higher_hours")
    }
    folds = []
    for fold, (train_mask, target_mask) in enumerate(
        _household_splits(children, seed=271828, validation_seed=None, partition="all")
    ):
        train, target = children.loc[train_mask], children.loc[target_mask]
        complete = target.groupby("source_household_id").attendance_status.agg(
            lambda x: x.eq("complete").all()
        )
        observed = target.loc[target.attendance_status.eq("complete")]
        base = QRFChildcareSchedules(train)
        records["complete_only"].extend(_records(base, observed, complete))
        detail = {
            "fold": fold,
            "evaluated_children": len(observed),
            "household_overlap": 0,
            "completions": {},
        }
        for name, tilt in (("interval", 0), ("lower_hours", -1), ("higher_hours", 1)):
            augmented, audit = interval_training_rows(train, base, tilt=tilt)
            model = QRFChildcareSchedules(augmented, allow_interval_training=True)
            records[name].extend(_records(model, observed, complete))
            detail["completions"][name] = audit
            print("completed fold", fold, name, flush=True)
        folds.append(detail)
    repo = Path(__file__).resolve().parents[1]
    plan = repo / "experiments/us-childcare-attendance/interval-and-paired-plan.txt"
    paths = [
        Path(__file__),
        Path(qrf_tool.__file__),
        Path(nsece_childcare_qrf.__file__),
        Path(nsece_childcare_pooling.__file__),
        Path(nsece_childcare_sibling_validation.__file__),
    ]
    report = {
        "source": source.source_receipt,
        "recipe": attendance_recipe_identity(),
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "diagnostic_code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
        "folds": folds,
        "models": {
            name: _observed_child_summary(rows) for name, rows in records.items()
        },
        "production_ready": False,
        "interpretation": "development comparison; completed training intervals remain modeled under declared coarsening assumptions; no held-out household contributes to completion or refitting",
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for name, result in report["models"].items():
        print(
            name,
            "failed child screens",
            sum(not check["passed"] for check in result["screen"]["checks"]),
            flush=True,
        )


if __name__ == "__main__":
    main()
