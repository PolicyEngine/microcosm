"""Compare the noncalendar bridge before and after thin-cell widening.

Development diagnostic only. It repeats the whole-household masked-calendar
design of ``assess_noncalendar_bridge`` over several split seeds and reports
per-child errors, which the group means in the source-stage report cannot show.
The previous implementation is loaded from a file, e.g.::

    git show e1b5d6c7:packages/microcosm-build/src/microcosm/build/us_runtime/nsece_childcare_bridge.py > /local/previous_bridge.py
    uv run python experiments/us-childcare-attendance/bridge_widening_masked_splits.py \
      --household-tsv /local/39466-0005-Data.tsv \
      --calendar-tsv /local/39466-0004-Data.tsv \
      --previous-bridge /local/previous_bridge.py \
      --report /local/bridge-widening-masked-splits.json

Only aggregates are written; no survey records or donor identities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

from microcosm.build.us_runtime import nsece_childcare_bridge as current_bridge
from microcosm.build.us_runtime.nsece_childcare import (
    NSECEChildcareSource,
    load_nsece_childcare,
)

SPLIT_SEEDS = (161803, 1, 2, 3, 4)
MASKED_COLUMNS = [
    "childcare_attending_days_per_month",
    "childcare_days_per_week",
    "childcare_hours_per_day",
    "ece_hours_per_week",
    "irregular_hours_per_week",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path):
    spec = importlib.util.spec_from_file_location("previous_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(source: NSECEChildcareSource, seed: int):
    children = source.children.copy()
    complete = children.attendance_status.eq("complete")
    holdout = complete & children.source_household_id.map(
        lambda x: (
            int.from_bytes(hashlib.sha256(f"{seed}:{x}".encode()).digest()[:8], "big")
            % 5
            == 0
        )
    )
    truth = children.loc[holdout].copy()
    children.loc[~complete, "regular_hours_per_week"] = np.nan
    children.loc[holdout, "attendance_status"] = "missing_calendar"
    children.loc[holdout, "questionnaire_version"] = 2
    children.loc[holdout, MASKED_COLUMNS] = np.nan
    masked = NSECEChildcareSource(children, source.weights, source.source_receipt)
    return masked, truth, holdout


def _errors(predicted, truth) -> dict:
    weight = truth.child_weight
    regular = truth.regular_hours_per_week > 0

    def average(values, mask=None):
        mask = slice(None) if mask is None else mask
        return float(np.average(values[mask], weights=weight[mask]))

    days = predicted.childcare_days_per_week - truth.childcare_days_per_week
    hours = predicted.ece_hours_per_week - truth.ece_hours_per_week
    daily = predicted.childcare_hours_per_day - truth.childcare_hours_per_day
    return {
        "children": len(truth),
        "mean_days_error": average(days),
        "mean_weekly_hours_error": average(hours),
        "mean_absolute_days_error": average(days.abs()),
        "mean_absolute_days_error_regular_care": average(days.abs(), regular),
        "mean_absolute_weekly_hours_error": average(hours.abs()),
        "mean_absolute_hours_per_day_error_regular_care": average(daily.abs(), regular),
        "share_over_12_hours_per_day": average(
            (predicted.childcare_hours_per_day > 12).astype(float)
        ),
        "measured_share_over_12_hours_per_day": average(
            (truth.childcare_hours_per_day > 12).astype(float)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--household-tsv", type=Path, required=True)
    parser.add_argument("--calendar-tsv", type=Path, required=True)
    parser.add_argument("--previous-bridge", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error("The report path must be new")
    arms = {"previous": _load(args.previous_bridge), "widened": current_bridge}
    source = load_nsece_childcare(args.household_tsv, args.calendar_tsv)
    splits = []
    for seed in SPLIT_SEEDS:
        masked, truth, holdout = _split(source, seed)
        for arm, module in arms.items():
            bridged = module.bridge_nsece_noncalendar_attendance(masked, seed=seed)
            predicted = bridged.children.loc[holdout]
            if not predicted.attendance_status.eq("summary_bridge").all():
                raise ValueError("Masked-calendar bridge has unsupported records")
            splits.append({"split_seed": seed, "arm": arm, **_errors(predicted, truth)})
    metrics = [k for k in splits[0] if k not in ("split_seed", "arm", "children")]
    report = {
        "design": "whole-household masked-calendar test; regular weekly hours remain observed",
        "split_seeds": list(SPLIT_SEEDS),
        "source": source.source_receipt,
        "code_sha256": {
            "previous_bridge": _sha256(args.previous_bridge),
            "widened_bridge": _sha256(Path(current_bridge.__file__)),
            "diagnostic": _sha256(Path(__file__)),
        },
        "splits": splits,
        "mean_over_splits": {
            arm: {
                metric: float(np.mean([s[metric] for s in splits if s["arm"] == arm]))
                for metric in metrics
            }
            for arm in arms
        },
        "production_ready": False,
        "interpretation": (
            "Development evidence on calendars that exist; not a test of the "
            "unobserved days or irregular care of the noncalendar instruments."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["mean_over_splits"], indent=2))


if __name__ == "__main__":
    main()
