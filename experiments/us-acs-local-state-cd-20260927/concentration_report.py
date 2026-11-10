"""Weight concentration of the evaluation's weight vectors, arm by arm.

Kish ESS nationally, per state and per congressional district, ESS over
distinct households, and the top-1% weight share, for each named weight
vector over one lean checkpoint's households. Uses the tool's own
``weight_origin_summary``.

    uv run python experiments/us-acs-local-state-cd-20260927/concentration_report.py \\
        --lean <checkpoint>/target_frame_lean.h5 \\
        --arm design=<weights.npz>:design --arm state=<weights.npz>:weights ... \\
        --out concentration.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def load_tool():
    path = REPO / "tools" / "build_us_acs_local_release.py"
    spec = importlib.util.spec_from_file_location("build_us_acs_local_release", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        required=True,
        help="label=path.npz:key (weights aligned to the lean household table)",
    )
    parser.add_argument("--focus-state", default="25")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tool = load_tool()
    with pd.HDFStore(args.lean, mode="r") as store:
        households = (
            store.select(
                "household",
                columns=[
                    "household_id",
                    "state_fips",
                    "congressional_district_geoid",
                    "household_spine",
                    "household_source_id",
                ],
            )
            if store.get_storer("household").is_table
            else store["household"]
        )
    origin = {
        "spine": households["household_spine"].to_numpy(),
        "source_id": households["household_source_id"].to_numpy(),
        "state": pd.to_numeric(households["state_fips"])
        .astype(int)
        .map("{:02d}".format)
        .to_numpy(),
        "district": pd.to_numeric(households["congressional_district_geoid"])
        .astype(int)
        .map("{:04d}".format)
        .to_numpy(),
    }
    report: dict[str, dict] = {}
    for arm in args.arm:
        label, rest = arm.split("=", 1)
        path, key = rest.rsplit(":", 1)
        weights = np.load(path)[key]
        if len(weights) != len(households):
            raise SystemExit(f"{label}: {len(weights)} weights, {len(households)} rows")
        summary = tool.cd_surface.weight_origin_summary(weights, **origin)
        focus_districts = {
            code: value
            for code, value in summary["effective_sample_size_by_district"].items()
            if code.startswith(args.focus_state)
        }
        report[label] = {
            "national_ess_rows": summary["effective_sample_size_rows"],
            "national_ess_distinct_households": summary.get(
                "effective_sample_size_distinct_households"
            ),
            "top_1pct_weight_share": summary["top_1pct_weight_share"],
            "weight_share_by_spine": summary.get("weight_share_by_spine"),
            "state_ess": summary["effective_sample_size_by_state_distribution"],
            "district_ess": summary["effective_sample_size_by_district_distribution"],
            f"state_{args.focus_state}_ess": summary[
                "effective_sample_size_by_state"
            ].get(args.focus_state),
            f"state_{args.focus_state}_district_ess": focus_districts,
            "by_state": summary["effective_sample_size_by_state"],
            "by_district": summary["effective_sample_size_by_district"],
        }
    labels = list(report)
    if len(labels) >= 2:
        base, *others = labels
        for other in others:
            for level in ("by_state", "by_district"):
                a, b = report[base][level], report[other][level]
                ratios = np.asarray([b[k] / a[k] for k in a if a[k] > 0 and k in b])
                report[other][f"{level}_ess_ratio_to_{base}"] = {
                    "min": float(ratios.min()),
                    "median": float(np.median(ratios)),
                    "max": float(ratios.max()),
                    "share_lower": float((ratios < 1).mean()),
                }
    args.out.write_text(json.dumps(report, indent=1))
    for label, entry in report.items():
        print(
            label,
            json.dumps(
                {
                    key: value
                    for key, value in entry.items()
                    if key not in ("by_state", "by_district", "weight_share_by_spine")
                },
                indent=None,
                default=str,
            )[:1500],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
