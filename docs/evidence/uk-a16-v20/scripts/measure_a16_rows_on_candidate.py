"""Measure the A16-excluded rows on a calibrated national candidate.

Retirement evidence for the calibration measure-exclusion register: each
excluded reference is compiled in the contract register but pruned from the
calibration surface, so the candidate never solved toward it. This tool
resolves and scores the FULL contract register on the candidate's calibrated
weights (pruning, loudly, only the rows whose measures cannot resolve at all
— the signed unresolvable-counterfactual entries) and reports every excluded
row's relative error. A row inside the 25 % fence at weights that never saw
it is evidence its exclusion can be retired; the retirement itself remains a
signed register change.

Run from the driver tree:  cd <tree> && uv run --no-sync python \
    .../measure_a16_rows_on_candidate.py --tree <tree> ...
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_FAILED_MEASURE = re.compile(
    r"provider (?:failed computing|does not know) ([a-z_]+)\.([A-Za-z0-9_]+)"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--candidate-h5", required=True, type=Path)
    parser.add_argument("--contract-json", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--calibration-year", type=int, default=2025)
    parser.add_argument("--max-prune-rounds", type=int, default=10)
    args = parser.parse_args(argv)

    sys.path.insert(0, str(args.tree / "tools"))
    import score_uk_national_candidate as scorer  # noqa: E402

    from microcosm.build.target_materialization import (  # noqa: E402
        MeasureResolutionError,
    )
    from microcosm.build.uk_runtime.measure_simulation import (  # noqa: E402
        load_uk_calibration_measure_exclusions,
    )
    from microcosm.calibrate import (  # noqa: E402
        TargetRegistry,
        score_targets,
    )

    contract = TargetRegistry.from_json(args.contract_json)
    exclusions = load_uk_calibration_measure_exclusions()
    excluded = {e["name"]: e for e in exclusions}
    year = int(args.calibration_year)
    factory = scorer._default_measure_resolver_factory(args.out.parent, year)

    registry = contract
    pruned: dict[str, str] = {}
    for _round in range(args.max_prune_rounds):
        try:
            frame, _resolution = scorer._scored_frame(
                args.candidate_h5,
                registry,
                year,
                factory,
                band_edge_registry=contract,
            )
        except MeasureResolutionError as error:
            receipt = getattr(error, "receipt", None) or {}
            match = _FAILED_MEASURE.search(str(error))
            if match is not None:
                variable = match.group(2)
                label = f"{match.group(1)}.{variable}"
                names = {
                    str(skip.get("name"))
                    for skip in receipt.get("skips", [])
                    if variable in str(skip.get("reason", ""))
                    or variable == str(skip.get("measure", ""))
                }
            elif "counterfactual" in str(error):
                label = "counterfactual_delta"
                names = {
                    str(skip.get("name"))
                    for skip in receipt.get("skips", [])
                    if "counterfactual" in str(skip.get("reason", ""))
                }
            else:
                raise
            drop = [s for s in registry.specs if s.name in names]
            if not drop:
                raise
            for s in drop:
                pruned[s.name] = label
            registry = TargetRegistry(
                [s for s in registry.specs if s.name not in pruned],
                country=registry.country,
            )
            print(f"prune: {label} unresolvable -> dropped {len(drop)}", flush=True)
            continue
        break
    else:
        raise RuntimeError("resolution still failing after max rounds")

    result = score_targets(frame, registry.to_target_set())
    errors = {d.name: float(d.relative_error) for d in result.diagnostics}
    # The exclusion register keys by spec name; scoring diagnostics key by the
    # compiled row name. Bridge through the contract register.
    row_name = {s.name: s.to_target().row_name for s in contract.specs}
    rows = []
    for name, entry in sorted(excluded.items()):
        rel = errors.get(row_name.get(name, name))
        rows.append(
            {
                "name": name,
                "tracking": entry["tracking"],
                "expires_on": entry["expires_on"],
                "relative_error": rel,
                "unresolvable": pruned.get(name),
                "within_25pct": None if rel is None else abs(rel) <= 0.25,
                "within_10pct": None if rel is None else abs(rel) <= 0.10,
            }
        )
    measured = [r for r in rows if r["relative_error"] is not None]
    within = [r for r in measured if r["within_25pct"]]
    payload = {
        "candidate_h5": str(args.candidate_h5),
        "contract_register": {
            "path": str(args.contract_json),
            "n_specs": len(contract.specs),
        },
        "excluded_entries": len(excluded),
        "measured": len(measured),
        "unresolvable_on_candidate": len(rows) - len(measured),
        "within_25pct": len(within),
        "rows": rows,
    }
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out}: {len(excluded)} excluded entries, {len(measured)} "
        f"measured on the candidate, {len(within)} within 25%"
    )
    for r in sorted(within, key=lambda r: abs(r["relative_error"])):
        print(
            f"  RETIRABLE? {r['name']:64s} {100 * r['relative_error']:+7.1f}%  [{r['tracking']}]"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
