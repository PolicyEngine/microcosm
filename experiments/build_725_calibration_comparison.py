#!/usr/bin/env python
"""Compare twin national calibrations on the CGT families and everything else (microcosm#725).

Reads two `calibration_diagnostics.json` files and writes the target-level
comparison the evidence note quotes: every CGT-family target with its target
value, final estimate and relative error on each side where it exists, and a
summary of the non-CGT surface (within 10 % / 25 % counts, targets that cross
either fence on one side only).

    .venv/bin/python experiments/build_725_calibration_comparison.py \
        --control <control diagnostics> --candidate <candidate diagnostics> --out docs/evidence/uk-cgt-725
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_cgt(name: str) -> bool:
    lowered = name.lower()
    return "cgt" in lowered or "capital_gains" in lowered


def _headline(diag: dict) -> dict[str, object]:
    targets = diag["targets"]
    non_cgt = [t for t in targets if not _is_cgt(t["name"])]
    cgt = [t for t in targets if _is_cgt(t["name"])]
    return {
        "targets": len(targets),
        "cgt_targets": len(cgt),
        "skipped": len(diag.get("skipped", [])),
        "initial_loss": diag["initial_loss"],
        "final_loss": diag["final_loss"],
        "fraction_within_10pct": diag["fraction_within_10pct"],
        "effective_sample_size": diag["effective_sample_size"],
        "realized_max_weight_ratio": diag["realized_max_weight_ratio"],
        "top_1pct_weight_share": diag.get("top_1pct_weight_share"),
        "non_cgt_within_10pct": sum(abs(t["relative_error"]) <= 0.10 for t in non_cgt),
        "non_cgt_outside_25pct": sorted(
            t["name"] for t in non_cgt if abs(t["relative_error"]) > 0.25
        ),
        "cgt_within_10pct": sum(abs(t["relative_error"]) <= 0.10 for t in cgt),
        "cgt_within_25pct": sum(abs(t["relative_error"]) <= 0.25 for t in cgt),
        "cgt_outside_25pct": sorted(
            t["name"] for t in cgt if abs(t["relative_error"]) > 0.25
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    control, candidate = _load(args.control), _load(args.candidate)
    by_name = {
        "control": {t["name"]: t for t in control["targets"]},
        "candidate": {t["name"]: t for t in candidate["targets"]},
    }
    names = sorted(set(by_name["control"]) | set(by_name["candidate"]))
    rows = []
    for name in names:
        row = {"name": name, "family": None, "cgt": _is_cgt(name)}
        for side in ("control", "candidate"):
            t = by_name[side].get(name)
            row[f"{side}_target"] = t["target"] if t else None
            row[f"{side}_estimate"] = t["final_estimate"] if t else None
            row[f"{side}_relative_error"] = t["relative_error"] if t else None
            if t:
                row["family"] = t.get("registry", {}).get("family")
        rows.append(row)
    with (args.out / "target-comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    shared = [
        r
        for r in rows
        if r["control_relative_error"] is not None
        and r["candidate_relative_error"] is not None
        and not r["cgt"]
    ]
    crossed_10 = {
        "worse": sorted(
            r["name"]
            for r in shared
            if abs(r["control_relative_error"])
            <= 0.10
            < abs(r["candidate_relative_error"])
        ),
        "better": sorted(
            r["name"]
            for r in shared
            if abs(r["candidate_relative_error"])
            <= 0.10
            < abs(r["control_relative_error"])
        ),
    }
    crossed_25 = {
        "worse": sorted(
            r["name"]
            for r in shared
            if abs(r["control_relative_error"])
            <= 0.25
            < abs(r["candidate_relative_error"])
        ),
        "better": sorted(
            r["name"]
            for r in shared
            if abs(r["candidate_relative_error"])
            <= 0.25
            < abs(r["control_relative_error"])
        ),
    }
    summary = {
        "control": _headline(control),
        "candidate": _headline(candidate),
        "non_cgt_shared_targets": len(shared),
        "non_cgt_crossed_10pct": crossed_10,
        "non_cgt_crossed_25pct": crossed_25,
        "cgt_rows": [
            {
                k: r[k]
                for k in (
                    "name",
                    "control_target",
                    "control_estimate",
                    "control_relative_error",
                    "candidate_target",
                    "candidate_estimate",
                    "candidate_relative_error",
                )
            }
            for r in rows
            if r["cgt"]
        ],
    }
    (args.out / "calibration-comparison.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    for side in ("control", "candidate"):
        print(
            side,
            json.dumps(
                {k: v for k, v in summary[side].items() if not isinstance(v, list)},
                indent=1,
            ),
        )
    print("non-CGT crossed 10%:", crossed_10)
    print("non-CGT crossed 25%:", crossed_25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
