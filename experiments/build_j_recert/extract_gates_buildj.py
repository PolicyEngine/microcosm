"""Build J verdict extractor — reads a release dir's certification artifacts and
prints the gate table + headline diagnostics + SSI probe for the #368 verdict.

Usage: extract_gates_buildj.py <release_dir>   (dir containing calibration_diagnostics.json etc.)
STAGING/LOCAL ONLY — read-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def find(o, *names):
    """Depth-first search for the first key in `names`; returns its value or None."""
    if isinstance(o, dict):
        for n in names:
            if n in o:
                return o[n]
        for v in o.values():
            r = find(v, *names)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v, *names)
            if r is not None:
                return r
    return None


def main() -> int:
    rd = Path(sys.argv[1])
    diag = load(rd / "calibration_diagnostics.json")
    parity = load(rd / "input_mass_parity.json")
    smoke = load(rd / "reform_coverage_smoke.json")
    coverage = load(rd / "input_coverage.json")

    print(f"=== RELEASE DIR: {rd}")
    if diag:
        print("\n--- HEADLINE DIAGNOSTICS ---")
        for label, keys in [
            ("final_loss", ("final_loss",)),
            ("within_10pct", ("within_10_percent", "within-10%", "within_10pct", "share_within_10_percent")),
            ("ESS", ("effective_sample_size", "ess", "ESS")),
            ("realized_max_weight_ratio", ("realized_max_weight_ratio", "max_weight_ratio")),
            ("n_records", ("n_records", "n_households")),
            ("n_nonzero", ("n_nonzero",)),
        ]:
            print(f"  {label}: {find(diag, *keys)}")
        # gate roll-up under build{}
        print("\n--- STRUCTURAL / SOURCE GATES (passed) ---")
        build = diag.get("build", diag) if isinstance(diag, dict) else {}
        for gname in [
            "target_profile_coverage", "health_input_signal", "base_population",
            "immigration", "degenerate_input", "ecps_parity", "input_mass_reference",
            "hours_worked", "snap_take_up", "eligibility_inputs", "pregnancy",
            "snap_discretionary_exemption", "validation_input_coverage",
        ]:
            g = find(build, gname) or find(build, gname + "_gate") or find(build, gname + "_signal")
            if isinstance(g, dict) and "passed" in g:
                fl = g.get("failures") or []
                print(f"  {gname}: passed={g['passed']}" + (f"  FAILURES={fl}" if fl else ""))
        rg = find(diag, "release_gates")
        if isinstance(rg, dict):
            print(f"  release_gates.passed = {rg.get('passed')}  failures={rg.get('failures')}")

    if parity:
        print("\n--- EXPORT-MASS PARITY ---")
        pg = parity.get("input_mass_parity", parity) if isinstance(parity, dict) else parity
        print(f"  passed = {find(parity, 'passed')}")
        fails = find(parity, "failures")
        print(f"  failures = {fails}")
        cols = find(parity, "columns") or find(parity, "results")
        if isinstance(cols, (list, dict)):
            n = len(cols)
            print(f"  n_columns = {n}")

    if smoke:
        print("\n--- REFORM-COVERAGE SMOKE (SSI probe) ---")
        print(f"  gate passed = {find(smoke, 'passed')}")
        res = find(smoke, "results")
        if isinstance(res, dict):
            for pid, r in res.items():
                print(f"  {pid}: effect={r.get('effect'):+,.0f}  min_abs={r.get('min_abs_effect'):,.0f}  "
                      f"passed={r.get('passed')}  baseline={r.get('baseline_total')}  reform={r.get('reform_total')}")

    if coverage:
        print("\n--- INPUT-COVERAGE (#369) ---")
        print(f"  passed = {find(coverage, 'passed')}")
        cf = find(coverage, "failures")
        print(f"  failures = {cf if cf else '[]'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
