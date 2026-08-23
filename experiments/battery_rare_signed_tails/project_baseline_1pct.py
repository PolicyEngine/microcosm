"""Project the 1% baseline battery failures onto the frozen 48-red set.

Establishes the "before" side of the lane's 1% verification: which of the
48 frozen full-scale QED reds already manifest at 1%, which cannot (their
carriers are too rare to survive a 1% sample), and whether the one-sided
Keogh leg emits any line at all at 1%. The output JSON is deterministic on
its three frozen inputs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_LEG = re.compile(r"^(?P<leg>[^:]+): ")
_KEOGH_LEG = (
    "person/source_operator_retirement_distributions/"
    "keogh_distributions[clone_0]/positive"
)


def _battery_failures(gates_path: Path) -> list[str]:
    payload = json.loads(gates_path.read_text(encoding="utf-8"))
    battery = payload["terminal_gates"]["gates"]["us_by_origin_battery"]
    failures = battery["failures"]
    if not isinstance(failures, list):
        raise TypeError(f"{gates_path}: battery failures are not a list")
    return failures


def _leg(line: str) -> str:
    match = _LEG.match(line)
    if match is None:
        raise ValueError(f"unparseable failure line: {line!r}")
    return match.group("leg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--realized-regimes",
        type=Path,
        default=Path(__file__).with_name("realized_regimes.json"),
    )
    parser.add_argument("--baseline-1pct-gates", required=True, type=Path)
    parser.add_argument(
        "--full-scale-gates",
        type=Path,
        default=Path(
            "/Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1/"
            "populace_us_2024_stacked_pool.gates.json"
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    regimes = json.loads(args.realized_regimes.read_text(encoding="utf-8"))
    checks = {row["leg_id"]: row for row in regimes["checks"]}
    if len(checks) != 48:
        raise ValueError(f"expected 48 frozen red legs, got {len(checks)}")

    one_pct_lines = _battery_failures(args.baseline_1pct_gates)
    one_pct_by_leg: dict[str, list[str]] = {}
    for line in one_pct_lines:
        one_pct_by_leg.setdefault(_leg(line), []).append(line)

    full_lines = _battery_failures(args.full_scale_gates)
    full_legs = {_leg(line) for line in full_lines}

    present = sorted(set(checks) & set(one_pct_by_leg))
    absent = sorted(set(checks) - set(one_pct_by_leg))
    noise_legs = sorted(set(one_pct_by_leg) - set(checks))

    payload = {
        "title": "1% baseline projection of the frozen 48 red QED checks",
        "inputs": {
            "realized_regimes": str(args.realized_regimes),
            "baseline_1pct_gates": str(args.baseline_1pct_gates),
            "full_scale_gates": str(args.full_scale_gates),
        },
        "baseline_1pct": {
            "failure_lines": len(one_pct_lines),
            "unique_failing_legs": len(one_pct_by_leg),
        },
        "full_scale": {
            "failure_lines": len(full_lines),
            "unique_failing_legs": len(full_legs),
        },
        "frozen_reds_red_at_1pct": {
            "count": len(present),
            "legs": present,
        },
        "frozen_reds_not_red_at_1pct": {
            "count": len(absent),
            "legs": [
                {
                    "leg_id": leg,
                    "ordinal": checks[leg]["ordinal"],
                    "donor_support_starved": checks[leg]["donor_support_starved"],
                }
                for leg in absent
            ],
        },
        "one_pct_only_noise_legs": {
            "count": len(noise_legs),
            "legs": noise_legs,
        },
        "keogh_one_sided_leg": {
            "leg_id": _KEOGH_LEG,
            "lines_at_1pct": one_pct_by_leg.get(_KEOGH_LEG, []),
            "red_at_full_scale": _KEOGH_LEG in full_legs,
            "note": (
                "Native ASEC holds exactly two positive Keogh values in "
                "108,073 rows; a 1% sample almost surely carries neither, so "
                "the 1% gate cannot exercise this leg in either direction. "
                "Gate-level proof of the Keogh repair requires a full-scale "
                "build; the 1% evidence is the support-preserving cap "
                "regression suite plus the frozen bank draw census."
            ),
        },
    }
    rendered = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
