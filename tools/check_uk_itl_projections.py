"""Check a calibrated UK file against HMRC's Income Tax liabilities projections.

Sets the file's Income Tax payers by total-income band (Table 2.5), and of
State Pension age and aged 65 and over in the United Kingdom (Table 2.1) and
in each region-tier area (Table 2.2), beside HMRC's July 2026 projections at
the calibration year's calendar window, and reports each cell's relative miss
against the tolerance ``uk_runtime.itl_differential`` fixes. A cell a bound
target reads is marked in-sample.

    uv run --no-sync python tools/check_uk_itl_projections.py \
        --h5 <national.h5> --out <report.json> [--itl-rows <rows>] [--strict]

``--itl-rows`` replaces the vendored ``hmrc_itl_taxpayer_counts.json`` with a
vendored-format JSON or a JSONL of Chronicle consumer facts. The report holds
aggregates only; keep it on the private lane with the run all the same.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import resources as importlib_resources
from pathlib import Path

from microcosm.build.uk_runtime.hmrc_uprating import UK_HMRC_TAXPAYER_COUNTS_RESOURCE
from microcosm.build.uk_runtime.itl_differential import (
    differential,
    expected_cells,
    people_from_h5,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import vendor_row, vendored_rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path | None) -> tuple[list[dict], str]:
    """The Income Tax liabilities rows, and where they came from."""

    if path is None:
        return (
            vendored_rows(UK_HMRC_TAXPAYER_COUNTS_RESOURCE),
            f"microcosm.build.uk/{UK_HMRC_TAXPAYER_COUNTS_RESOURCE}",
        )
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [vendor_row(json.loads(line)) for line in text.splitlines() if line]
    else:
        rows = list(json.loads(text)["rows"])
    return rows, f"{path} sha256:{_sha256(path)}"


def bound_targets(contract: Path | None) -> set[str]:
    if contract is None:
        text = (
            importlib_resources.files("microcosm.build.uk")
            .joinpath("uk_population_targets.json")
            .read_text(encoding="utf-8")
        )
    else:
        text = contract.read_text(encoding="utf-8")
    return {str(target["target_id"]) for target in json.loads(text)["targets"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--itl-rows", type=Path)
    parser.add_argument(
        "--contract",
        type=Path,
        help="The contract the build read, for in-sample flags (default: this tree's).",
    )
    parser.add_argument(
        "--current-income-for-annual-measures",
        metavar="REASON",
        help="For a file without annual income descriptors (the incumbent).",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any miss.")
    args = parser.parse_args(argv)

    rows, rows_source = load_rows(args.itl_rows)
    cells = expected_cells(
        rows, year=args.year, bound_targets=bound_targets(args.contract)
    )
    report = differential(
        cells,
        people_from_h5(
            args.h5,
            year=args.year,
            scratch=args.out.parent / f"scratch-{args.out.stem}",
            reason=args.current_income_for_annual_measures,
        ),
    )
    receipt = {
        "h5": str(args.h5),
        "h5_sha256": _sha256(args.h5),
        "year": args.year,
        "itl_rows": rows_source,
        **report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    for row in report["cells"]:
        flag = "" if row["within_tolerance"] else "  MISS"
        sample = "in" if row["in_sample"] else "out"
        print(
            f"{row['table']} {row['measure']:<32} {row['key']:<16} {sample:<3} "
            f"{row['expected']:>13,.0f} {row['model']:>13,.0f} {row['residual']:+7.2%}{flag}"
        )
    return 1 if args.strict and report["misses"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
