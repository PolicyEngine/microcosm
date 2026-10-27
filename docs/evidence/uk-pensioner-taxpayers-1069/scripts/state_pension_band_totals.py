"""SPI State Pension band rows before and after the re-index (microcosm#1069).

Reads two compile-parity receipts (``uk/ledger_compile_parity_incumbent_2025_
signed_differences.json``: the one on main and this branch's) and prints, for
the count and amount rows, each band's compiled 2025 value in both, the
totals and each band's share of its total. The re-index moves every count
row by one factor, so the 2023-24 total is the new total over that factor,
and the old total's growth from 2023-24 follows. Rows the build holds out of
the objective (the GBP 1m-and-over band, microcosm#1095) are totalled apart.

    python3 state_pension_band_totals.py --before <main receipt> --after <branch receipt> \
        --factor 1.0835478 --output <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUPS = {
    "count": "hmrc/state_pension_count_income_band_",
    "amount": "hmrc/state_pension_income_band_",
}
HELD_OUT = "1_000_000_to_inf"


def _values(path: Path, prefix: str) -> dict[str, float]:
    receipt = json.loads(path.read_text())
    return {
        row["name"].removeprefix(prefix): float(row["ledger_value"])
        for row in receipt["differences"]
        if row["name"].startswith(prefix) and "ledger_value" in row
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", required=True, type=Path)
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--factor", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {"count_factor": args.factor}
    for group, prefix in GROUPS.items():
        before = _values(args.before, prefix)
        after = _values(args.after, prefix)
        if set(before) != set(after):
            raise SystemExit(f"{group}: band sets differ: {set(before) ^ set(after)}")
        bound = [band for band in before if band != HELD_OUT]
        totals = {
            "before_bound": sum(before[b] for b in bound),
            "after_bound": sum(after[b] for b in bound),
            "before_all": sum(before.values()),
            "after_all": sum(after.values()),
        }
        ratios = {band: after[band] / before[band] for band in before}
        entry = {
            "bands": {
                band: {
                    "before": before[band],
                    "after": after[band],
                    "after_over_before": ratios[band],
                    "before_share": before[band] / totals["before_all"],
                    "after_share": after[band] / totals["after_all"],
                }
                for band in sorted(before)
            },
            "totals": totals,
        }
        if group == "count":
            opening = {
                key.replace("after", "opening_2023"): value / args.factor
                for key, value in totals.items()
                if key.startswith("after")
            }
            entry["opening_2023"] = opening
            entry["before_growth_from_2023"] = {
                "bound": totals["before_bound"] / opening["opening_2023_bound"] - 1,
                "all": totals["before_all"] / opening["opening_2023_all"] - 1,
            }
            spread = max(ratios.values()) - min(ratios.values())
            entry["after_over_before_spread"] = spread
        report[group] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    count = report["count"]
    print(json.dumps(count["totals"]), json.dumps(count["opening_2023"]))
    print("old growth from 2023-24:", json.dumps(count["before_growth_from_2023"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
