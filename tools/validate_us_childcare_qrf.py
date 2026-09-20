#!/usr/bin/env python3
"""Run the declared canonical-QRF marginal diagnostic on licensed local data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime import nsece_childcare_qrf
from microcosm.build.us_runtime.childcare_attendance_receipt import (
    attendance_recipe_identity,
)
from microcosm.build.us_runtime.nsece_childcare import load_nsece_childcare
from microcosm.build.us_runtime.nsece_childcare_pooling import (
    with_childcare_household_size,
)
from microcosm.build.us_runtime.nsece_childcare_qrf import (
    QRF_CHILDCARE_PREDICTORS,
    QRFChildcareSchedules,
)
from microcosm.build.us_runtime.nsece_childcare_sibling_validation import (
    _household_splits,
    _observed_child_summary,
)


def load_source_children(household_tsv, calendar_tsv):
    """Verify source pins and add the declared common household predictors."""
    source = load_nsece_childcare(household_tsv, calendar_tsv)
    raw = pd.read_csv(
        household_tsv,
        sep="\t",
        usecols=["HH4_METH_CASEID", "HH4_HHCOMP_MEMBERS", "HH4_HHCOMP_NUMPARENTS"],
    )
    raw.index = raw.HH4_METH_CASEID.astype(str)
    children = with_childcare_household_size(
        source.children.loc[source.children.age.between(0, 12)]
    )
    children["household_members"] = children.source_household_id.map(
        raw.HH4_HHCOMP_MEMBERS
    )
    children["resident_parent_count"] = children.source_household_id.map(
        raw.HH4_HHCOMP_NUMPARENTS
    )
    return source, children


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-tsv", type=Path, required=True)
    parser.add_argument("--calendar-tsv", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("Report path must be new")
    source, children = load_source_children(args.household_tsv, args.calendar_tsv)
    records, folds, movements = [], [], []
    for fold, (train_mask, target_mask) in enumerate(
        _household_splits(children, seed=271828, validation_seed=None, partition="all")
    ):
        training, target = children.loc[train_mask], children.loc[target_mask]
        complete = target.groupby("source_household_id").attendance_status.agg(
            lambda x: x.eq("complete").all()
        )
        target = target.loc[target.attendance_status.eq("complete")]
        model = QRFChildcareSchedules(training)
        model.prepare(target)
        for _, child in target.iterrows():
            values, _, probability = model.distribution(child)
            actual = np.array(
                [
                    float(child.childcare_days_per_week > 0),
                    child.childcare_days_per_week,
                    child.ece_hours_per_week,
                ]
            )
            records.append(
                (
                    float(child.child_weight),
                    min(int(child.childcare_household_size), 3),
                    bool(complete.loc[child.source_household_id]),
                    actual,
                    np.average(values, axis=0, weights=probability),
                    np.average(values**2, axis=0, weights=probability),
                )
            )
        folds.append(
            {
                "fold": fold,
                "training_households": int(training.source_household_id.nunique()),
                "evaluation_households": int(target.source_household_id.nunique()),
                "household_overlap": 0,
                "evaluated_children": len(target),
                "profiles": len(model.cache),
            }
        )
        movements.extend(model.decoder_movements)
        print("completed fold", fold, flush=True)
    plan = (
        Path(__file__).resolve().parents[1]
        / "experiments/us-childcare-attendance/qrf-calendar-plan.txt"
    )
    report = {
        "source": source.source_receipt,
        "recipe": attendance_recipe_identity(),
        "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "diagnostic_code_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(nsece_childcare_qrf.__file__)]
        },
        "model": {
            "name": "canonical_weighted_QRF_joint_schedule_decoder",
            "predictors": QRF_CHILDCARE_PREDICTORS,
            "trees": 100,
            "seed": 915,
            "integration_points": 128,
            "mean_positive_decoder_code_movement": float(np.mean(movements)),
            "max_positive_decoder_code_movement": float(np.max(movements)),
        },
        "splits": folds,
        "observed_child_validation": _observed_child_summary(records),
        "production_ready": False,
        "interpretation": "marginal development diagnostic only; no dependence fit, no source-selection correction, no population integration or independent validation",
    }
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        "child failures",
        sum(
            not c["passed"]
            for c in report["observed_child_validation"]["screen"]["checks"]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
