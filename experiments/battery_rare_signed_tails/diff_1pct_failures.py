"""Diff the terminal battery failure lines of two 1% verification builds.

Reads each build's ``*.gates.json``, extracts the ``us_by_origin_battery``
failure list, and reports: lines only in the before build, lines only in the
after build, and the count present in both. Also projects each set against the
frozen full-scale adjudication classes (the 93-line baseline) by leg id so the
1% noise floor is legible.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_LEG = re.compile(r"^(?P<leg>[^:]+): ")


def _battery_failures(gates_path: Path) -> list[str]:
    payload = json.loads(gates_path.read_text(encoding="utf-8"))
    battery = payload["terminal_gates"]["gates"]["us_by_origin_battery"]
    failures = battery["failures"]
    if not isinstance(failures, list):
        raise TypeError(f"{gates_path}: battery failures are not a list")
    return failures


def _legs(lines: list[str]) -> set[str]:
    out = set()
    for line in lines:
        match = _LEG.match(line)
        if match is None:
            raise ValueError(f"unparseable failure line: {line!r}")
        out.add(match.group("leg"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-gates", required=True, type=Path)
    parser.add_argument("--after-gates", required=True, type=Path)
    parser.add_argument(
        "--baseline-gates",
        type=Path,
        default=Path(
            "/Users/maxghenis/PolicyEngine/_buildo-runtime/out/stacked-f025-r1/"
            "populace_us_2024_stacked_pool.gates.json"
        ),
        help="Frozen full-scale f025 gates for baseline-class context.",
    )
    args = parser.parse_args()

    before = _battery_failures(args.before_gates)
    after = _battery_failures(args.after_gates)
    baseline_legs = _legs(_battery_failures(args.baseline_gates))

    before_set, after_set = set(before), set(after)
    only_before = sorted(before_set - after_set)
    only_after = sorted(after_set - before_set)

    print(f"before: {len(before)} failure lines ({args.before_gates})")
    print(f"after:  {len(after)} failure lines ({args.after_gates})")
    print(f"shared: {len(before_set & after_set)}")
    print(f"baseline full-scale legs: {len(baseline_legs)}")

    for title, lines in (("ONLY BEFORE", only_before), ("ONLY AFTER", only_after)):
        print(f"\n== {title} ({len(lines)}) ==")
        for line in lines:
            leg = _LEG.match(line).group("leg")
            tag = "baseline-red" if leg in baseline_legs else "not-baseline-red"
            print(f"[{tag}] {line}")


if __name__ == "__main__":
    main()
