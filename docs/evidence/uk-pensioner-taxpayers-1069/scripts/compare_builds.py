"""Compare two UK national builds' fits on the pensioner rows (microcosm#1069).

Reads each build's ``calibration_diagnostics.json`` and, for the rows this
lane touches, writes the target, the solved estimate and the relative error
in both: the SPI State Pension and private pension band rows, the DWP and DfC
State Pension recipient rows and the ``hmrc_itl_age`` rows (the candidate
binds them; the baseline's values come from ``measure_pensioner_taxpayers.py``
on its H5). Also the solve's overall loss and spread. Aggregates only.

    python3 compare_builds.py --baseline <dir> --candidate <dir> \
        --baseline-measures <json> --candidate-measures <json> --output <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PREFIXES = {
    "spi_state_pension_count": "hmrc/state_pension_count_income_band_",
    "spi_state_pension_amount": "hmrc/state_pension_income_band_",
    "spi_private_pension_count": "hmrc/private_pension_income_count_income_band_",
    "spi_private_pension_amount": "hmrc/private_pension_income_income_band_",
    "itl_age": "hmrc.itl.",
    "state_pension_recipients": "dwp.state_pension.recipients",
}
AGE_TARGETS = {
    "hmrc.itl.state_pension_age_taxpayers": "state_pension_age",
    "hmrc.itl.taxpayers_aged_65_and_over": "age_65_and_over",
}
SUMMARY = (
    "initial_loss",
    "final_loss",
    "fraction_within_10pct",
    "n_records",
    "n_nonzero",
    "effective_sample_size",
    "realized_max_weight_ratio",
    "top_1pct_weight_share",
)


def _rows(diagnostics: dict) -> dict[str, dict]:
    out = {}
    for row in diagnostics["targets"]:
        name = row["target_name"]
        if name.startswith("hmrc.itl.") and not name.startswith(tuple(AGE_TARGETS)):
            continue
        for group, prefix in PREFIXES.items():
            if name.startswith(prefix) or (
                group == "state_pension_recipients"
                and name.startswith("dfc_ni.state_pension.recipients")
            ):
                out[row["name"]] = {
                    "group": group,
                    "target": row["target"],
                    "estimate": row["final_estimate"],
                    "relative_error": row["relative_error"],
                    "loss_weight_share": row.get("target_loss_weight_share"),
                    "uprating_factor": row["metadata"].get("uprating_factor"),
                    "uprating_index": row["metadata"].get("uprating_index"),
                }
                break
    return out


def _age_from_measures(measures: dict, name: str) -> float | None:
    target, _, area = name.partition("@")
    area = area.split("@")[0]
    measure = AGE_TARGETS.get(target)
    if measure is None:
        return None
    return measures["taxpayers_by_area"].get(area, {}).get(measure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-measures", required=True, type=Path)
    parser.add_argument("--candidate-measures", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    diagnostics = {
        label: json.loads((path / "calibration_diagnostics.json").read_text())
        for label, path in (("baseline", args.baseline), ("candidate", args.candidate))
    }
    measures = {
        "baseline": json.loads(args.baseline_measures.read_text()),
        "candidate": json.loads(args.candidate_measures.read_text()),
    }
    rows = {label: _rows(d) for label, d in diagnostics.items()}
    names = sorted(set(rows["baseline"]) | set(rows["candidate"]))
    table = []
    for name in names:
        base = rows["baseline"].get(name)
        cand = rows["candidate"].get(name)
        entry = {"name": name, "group": (cand or base)["group"]}
        for label, row in (("baseline", base), ("candidate", cand)):
            if row is not None:
                entry[label] = row
                continue
            value = _age_from_measures(measures[label], name)
            if value is not None and cand is not None:
                # Not bound in this build: its solved weights measured on the
                # candidate's target.
                entry[label] = {
                    "target": cand["target"],
                    "estimate": value,
                    "relative_error": value / cand["target"] - 1,
                    "bound": False,
                }
        table.append(entry)
    groups = {}
    for entry in table:
        group = groups.setdefault(entry["group"], {"baseline": {}, "candidate": {}})
        for label in ("baseline", "candidate"):
            row = entry.get(label)
            if row is None:
                continue
            totals = group[label]
            totals["target"] = totals.get("target", 0.0) + row["target"]
            totals["estimate"] = totals.get("estimate", 0.0) + row["estimate"]
            totals["mean_abs_relative_error"] = totals.get(
                "mean_abs_relative_error", 0.0
            ) + abs(row["relative_error"])
            totals["rows"] = totals.get("rows", 0) + 1
    for group in groups.values():
        for totals in group.values():
            if totals.get("rows"):
                totals["mean_abs_relative_error"] /= totals["rows"]
    report = {
        "summary": {
            label: {key: d.get(key) for key in SUMMARY}
            for label, d in diagnostics.items()
        },
        "target_count": {label: len(d["targets"]) for label, d in diagnostics.items()},
        "groups": groups,
        "rows": table,
        "state_pension_recipients": {
            label: m["state_pension_recipients"] for label, m in measures.items()
        },
        "taxpayers_region_tier_total": {
            label: m["taxpayers_region_tier_total"] for label, m in measures.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=1))
    print(json.dumps(report["groups"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
