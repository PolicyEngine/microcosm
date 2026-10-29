"""Compare the reform-coverage smoke of two ``probe_report.json`` files.

Built to check that a rerun of the same subsample with different probe code
reproduces a run: per probe, whether the gate's effect, its pass/fail, the
standard error, the effective households and the authority agree, and the
fields only one report has (``noncarrier_effect_households`` is new in the
merged probe). Prints Markdown. Standard library only.

Usage::

    python compare_probe_smoke.py <run probe_report.json> <rerun probe_report.json>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

FIELDS = (
    "effect",
    "passed",
    "authority",
    "standard_error",
    "effective_variance_households",
    "drawn_effect_households",
    "certainty_effect_households",
    "take_all",
)


def _rows(path: Path) -> dict[str, dict]:
    report = json.loads(path.read_text())
    return {
        row["probe"]: row
        for row in report["stages"]["reform_coverage_smoke"].get("probes", [])
    }


def _same(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        return left == right or (math.isnan(left) and math.isnan(right))
    return left == right


def compare(run: dict[str, dict], rerun: dict[str, dict]) -> dict:
    differences: dict[str, dict[str, tuple]] = {}
    for probe in sorted(set(run) & set(rerun)):
        changed = {
            field: (run[probe].get(field), rerun[probe].get(field))
            for field in FIELDS
            if not _same(run[probe].get(field), rerun[probe].get(field))
        }
        if changed:
            differences[probe] = changed
    take_all = {
        probe: {
            "authority": row.get("authority"),
            "drawn_effect_households": row.get("drawn_effect_households"),
            "noncarrier_effect_households": row.get("noncarrier_effect_households"),
        }
        for probe, row in sorted(rerun.items())
        if row.get("take_all")
    }
    return {
        "probes": (len(run), len(rerun)),
        "only_in_run": sorted(set(run) - set(rerun)),
        "only_in_rerun": sorted(set(rerun) - set(run)),
        "differences": differences,
        "take_all": take_all,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("rerun", type=Path)
    args = parser.parse_args(argv)
    result = compare(_rows(args.run), _rows(args.rerun))
    n_run, n_rerun = result["probes"]
    print(f"- probes: {n_run} in the run, {n_rerun} in the rerun")
    print(f"- only in the run: {result['only_in_run'] or 'none'}")
    print(f"- only in the rerun: {result['only_in_rerun'] or 'none'}")
    compared = ", ".join(f"`{field}`" for field in FIELDS)
    if not result["differences"]:
        print(f"- every shared probe agrees exactly on {compared}")
    else:
        print(f"- probes differing on {compared}:")
        for probe, changed in result["differences"].items():
            for field, (left, right) in changed.items():
                print(f"  - `{probe}` {field}: {left!r} -> {right!r}")
    print("\nTake-all probes in the rerun:\n")
    print(
        "| Probe | Authority | Drawn effect households | Non-carrier effect households |"
    )
    print("|---|---|---|---|")
    for probe, row in result["take_all"].items():
        print(
            f"| `{probe}` | {row['authority']} | {row['drawn_effect_households']} "
            f"| {row['noncarrier_effect_households']} |"
        )


if __name__ == "__main__":
    main()
