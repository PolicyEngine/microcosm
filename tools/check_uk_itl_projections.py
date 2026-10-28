"""Check a calibrated UK file against HMRC's Income Tax liabilities projections.

Sets the file's Income Tax payers by total-income band (Table 2.5), and of
State Pension age and aged 65 and over in the United Kingdom (Table 2.1) and
in each region-tier area (Table 2.2), beside HMRC's July 2026 projections at
the calibration year's calendar window, and reports each cell's relative miss
against the tolerance ``uk_runtime.itl_differential`` fixes. A cell a bound
target reads is marked in-sample: bound in the run, from its
``calibration_diagnostics.json`` (``--diagnostics``, a target with a loss
weight above zero), or else declared by the contract.

    uv run --no-sync python tools/check_uk_itl_projections.py \
        --h5 <national.h5> --out <report.json> [--diagnostics <json>] \
        [--itl-rows <rows>] [--strict]

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
    SINGLE_TAX_YEAR,
    dated_people,
    differential,
    expected_cells,
    measure_resolver,
    resolved_people,
    table_of,
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


def run_bound_targets(diagnostics: Path) -> set[str]:
    """The targets a run bound, each by its name without the period.

    A diagnostics row binds when its loss weight is missing or above zero; a
    family's cells are bound one by one, so an excluded or zero-weight band
    stays out of sample however many of its siblings are bound.
    """

    rows = json.loads(diagnostics.read_text(encoding="utf-8"))["targets"]
    bound = set()
    for row in rows:
        weight = row.get("target_loss_weight")
        if weight is None or float(weight) > 0:
            name, _, period = str(row["name"]).rpartition("@")
            bound.add(name if name and period.isdigit() else str(row["name"]))
    return bound


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
        "--diagnostics",
        type=Path,
        help="The run's calibration_diagnostics.json: in-sample means bound in it.",
    )
    parser.add_argument(
        "--current-income-for-annual-measures",
        metavar="REASON",
        help="For a file without annual income descriptors (the incumbent).",
    )
    parser.add_argument(
        "--dated-year",
        type=int,
        help="Also check the dated SPI measures of this tax year (2023) against "
        "Table 2.5's outturn column for it.",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any miss.")
    args = parser.parse_args(argv)

    rows, rows_source = load_rows(args.itl_rows)
    if args.diagnostics is not None:
        bound = run_bound_targets(args.diagnostics)
        in_sample_basis = f"{args.diagnostics} sha256:{_sha256(args.diagnostics)}"
    else:
        bound = bound_targets(args.contract)
        in_sample_basis = f"contract {args.contract or 'of this tree'}"
    cells = expected_cells(rows, year=args.year, bound_targets=bound)
    resolver = measure_resolver(
        args.h5,
        year=args.year,
        scratch=args.out.parent / f"scratch-{args.out.stem}",
        reason=args.current_income_for_annual_measures,
    )
    report = differential(cells, resolved_people(resolver))
    receipt = {
        "h5": str(args.h5),
        "h5_sha256": _sha256(args.h5),
        "year": args.year,
        "itl_rows": rows_source,
        "in_sample_basis": in_sample_basis,
        **report,
    }
    misses = list(report["misses"])
    if args.dated_year is not None:
        table_2_5 = [row for row in rows if table_of(row) == "2.5"]
        dated = differential(
            expected_cells(table_2_5, year=args.dated_year, weights=SINGLE_TAX_YEAR),
            dated_people(resolver, args.dated_year),
        )
        receipt["dated"] = {"year": args.dated_year, **dated}
        misses += [f"dated {miss}" for miss in dated["misses"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    sections = [("", report)]
    if "dated" in receipt:
        sections.append((f"{args.dated_year} ", receipt["dated"]))
    for prefix, section in sections:
        for row in section["cells"]:
            flag = "" if row["within_tolerance"] else "  MISS"
            sample = "in" if row["in_sample"] else "out"
            print(
                f"{prefix}{row['table']} {row['measure']:<32} {row['key']:<16} "
                f"{sample:<3} {row['expected']:>13,.0f} {row['model']:>13,.0f} "
                f"{row['residual']:+7.2%}{flag}"
            )
    return 1 if args.strict and misses else 0


if __name__ == "__main__":
    raise SystemExit(main())
