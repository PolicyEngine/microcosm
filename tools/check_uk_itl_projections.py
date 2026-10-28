"""Check a calibrated UK file against HMRC's Income Tax liabilities projections.

Sets the file's Income Tax payers by total-income band (Table 2.5), and of
State Pension age and aged 65 and over in the United Kingdom (Table 2.1) and
in each region-tier area (Table 2.2), beside HMRC's July 2026 projections at
the calibration year's calendar window, and reports each cell's relative miss
against the tolerance ``uk_runtime.itl_differential`` fixes. It is a check,
not a target, and nothing runs it on its own: run it on a national build.

A cell a bound target reads is marked in-sample. With ``--diagnostics`` the
basis is the run's ``calibration_diagnostics.json``: a row of ``--year``
binds when its loss weight is above zero or missing (the receipt counts the
rows that carry none). Without it the basis is this tree's
``target_references.json`` (``--references`` names another), whose active
references bind at their own period only, so a ``--year`` they do not carry
binds nothing. Table 2.5's bands and Table 2.2's areas (microcosm#1069) are
bound; Table 2.1's United Kingdom counts are not. A cell made of others (the
all-band count, the residents, Table 2.1's United Kingdom counts as the
areas' sum, the count below State Pension age as a difference) is in sample
only when every part is and the file holds nobody it counts outside them
(``outside_parts``). With ``--diagnostics``, each bound cell also carries the
run's own estimate of its target (``final_estimate``) beside the model count
read here, so a check that reads a cell differently from the solve shows.

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
from collections.abc import Iterator, Mapping
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.hmrc_uprating import UK_HMRC_TAXPAYER_COUNTS_RESOURCE
from microcosm.build.uk_runtime.itl_differential import (
    differential,
    expected_cells,
    measure_resolver,
    residual,
    resolved_people,
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
        rows = [
            vendor_row(json.loads(line)) for line in text.splitlines() if line.strip()
        ]
    else:
        rows = list(json.loads(text)["rows"])
    return rows, f"{path} sha256:{_sha256(path)}"


def _rows_at(diagnostics: Path, year: int) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Each diagnostics target row of ``year``, by its name without the period:
    the row's ``target_name`` and ``period``, or, without them, its ``name``
    split at the trailing ``@<year>``. A row of another year, or with no
    period, is left out."""

    for row in json.loads(diagnostics.read_text(encoding="utf-8"))["targets"]:
        name, period = row.get("target_name"), row.get("period")
        if not name or period is None:
            name, _, period = str(row["name"]).rpartition("@")
        if name and str(period).isdigit() and int(period) == year:
            yield str(name), row


def run_bound_targets(diagnostics: Path, year: int) -> set[str]:
    """The targets a run bound at ``year``, each by its name without the period.

    A row binds when its loss weight is missing or above zero. A family's
    cells are bound one by one, so an excluded or zero-weight band stays out
    of sample however many of its siblings are bound.
    """

    bound = set()
    for name, row in _rows_at(diagnostics, year):
        weight = row.get("target_loss_weight")
        if weight is None or float(weight) > 0:
            bound.add(name)
    return bound


def run_unweighted_targets(diagnostics: Path, year: int) -> set[str]:
    """The rows of ``year`` that bind only because they carry no loss weight
    (the diagnostics writer leaves it out when it cannot attribute the loss)."""

    return {
        name
        for name, row in _rows_at(diagnostics, year)
        if row.get("target_loss_weight") is None
    }


def run_estimates(diagnostics: Path, year: int) -> dict[str, float]:
    """The run's own estimate of each target at ``year`` on its solved weights."""

    return {
        name: float(row["final_estimate"])
        for name, row in _rows_at(diagnostics, year)
        if row.get("final_estimate") is not None
    }


def reference_bound_targets(references: Path | None, year: int) -> set[str]:
    """The target references a build of this tree binds at ``year``.

    ``target_references.json`` lists the active references the national
    calibration compiles, each at its period, so a signed-out reference, or
    one of another period, binds nothing.
    """

    if references is None:
        text = (
            importlib_resources.files("microcosm.build.uk")
            .joinpath("target_references.json")
            .read_text(encoding="utf-8")
        )
    else:
        text = references.read_text(encoding="utf-8")
    return {
        str(row["name"])
        for row in json.loads(text)["target_references"]
        if str(row.get("period")).isdigit() and int(row["period"]) == year
    }


def with_run_estimates(
    report: Mapping[str, Any], estimates: Mapping[str, float]
) -> dict[str, Any]:
    """The report with each bound cell's run estimate beside its model count.

    ``run_estimate`` is the run's ``final_estimate`` of the cell's reference
    and ``model_vs_run_estimate`` the model count's relative difference from
    it; both are None for a cell no reference binds or the run did not
    estimate. A difference says the check and the solve read the cell
    differently, or the file's weights are not the solve's.
    """

    cells = []
    for row in report["cells"]:
        estimate = estimates.get(row["reference"]) if row["reference"] else None
        cells.append(
            {
                **row,
                "run_estimate": estimate,
                "model_vs_run_estimate": (
                    residual(row["model"], estimate)
                    if estimate is not None and estimate > 0
                    else None
                ),
            }
        )
    compared = [
        abs(row["model_vs_run_estimate"])
        for row in cells
        if row["model_vs_run_estimate"] is not None
    ]
    return {
        **report,
        "cells": cells,
        "run_estimates": {
            "cells": len(compared),
            "largest_abs_difference": max(compared, default=None),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--itl-rows", type=Path)
    basis = parser.add_mutually_exclusive_group()
    basis.add_argument(
        "--diagnostics",
        type=Path,
        help="The run's calibration_diagnostics.json: in-sample means bound in it.",
    )
    basis.add_argument(
        "--references",
        type=Path,
        help="Without --diagnostics: the target_references.json the build read "
        "(default: this tree's).",
    )
    parser.add_argument(
        "--current-income-for-annual-measures",
        metavar="REASON",
        help="For a file without annual income descriptors (the incumbent).",
    )
    parser.add_argument("--strict", action="store_true", help="Exit 1 on any miss.")
    args = parser.parse_args(argv)

    rows, rows_source = load_rows(args.itl_rows)
    if args.diagnostics is not None:
        bound = run_bound_targets(args.diagnostics, args.year)
        unweighted = run_unweighted_targets(args.diagnostics, args.year)
        in_sample_basis = f"{args.diagnostics} sha256:{_sha256(args.diagnostics)}"
        if unweighted:
            in_sample_basis += (
                f"; {len(unweighted)} rows at {args.year} carry no "
                "target_loss_weight and bind"
            )
    else:
        bound = reference_bound_targets(args.references, args.year)
        in_sample_basis = (
            f"target references {args.references or 'of this tree'} at {args.year}"
        )
    cells = expected_cells(rows, year=args.year, bound_targets=bound)
    resolver = measure_resolver(
        args.h5,
        year=args.year,
        scratch=args.out.parent / f"scratch-{args.out.stem}",
        reason=args.current_income_for_annual_measures,
    )
    report = differential(cells, resolved_people(resolver))
    if args.diagnostics is not None:
        report = with_run_estimates(report, run_estimates(args.diagnostics, args.year))
    receipt = {
        "h5": str(args.h5),
        "h5_sha256": _sha256(args.h5),
        "year": args.year,
        "itl_rows": rows_source,
        "in_sample_basis": in_sample_basis,
        **report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    for row in report["cells"]:
        flag = "" if row["within_tolerance"] else "  MISS"
        sample = "in" if row["in_sample"] else "out"
        print(
            f"{row['table']} {row['measure']:<32} {row['key']:<16} "
            f"{sample:<3} {row['expected']:>13,.0f} {row['model']:>13,.0f} "
            f"{row['residual']:+7.2%}{flag}"
        )
    return 1 if args.strict and report["misses"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
