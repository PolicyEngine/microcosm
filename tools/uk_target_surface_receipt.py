"""Write a receipt of the compiled UK target surface, or diff two receipts.

The receipt is what the ``uk.full.target_compilation`` node compiles, read
through the node's own code path (``compile_uk_full_target_surface``): every
national row after the measure exclusions and every local cell after uprating
and cross-grain reconciliation, with the reconciliation groups' factors, the
partitions, the coverage gaps the doctrine ledger tolerates and the uprating
holds. ``--diff-against`` compares it with an earlier receipt row by row, so a
change to the target surface states exactly which values it moves
(microcosm#1123).

By default only the calibration year is compiled (the node also compiles the
historical validation periods, which never feed the surface); ``--full``
compiles them too.

Usage::

    uv run --no-sync python tools/uk_target_surface_receipt.py \\
        --ledger-facts .codex-work/uk-artifact --ladder <uk_oa_ladder_2021.npz> \\
        --review-date 2026-10-07 --out receipt.json [--diff-against before.json]
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.uk_runtime.cross_grain_declarations import uk_cross_grain_grain
from microcosm.build.uk_runtime.full_targets import load_uk_full_target_inputs
from microcosm.build.uk_runtime.graph_targets import (
    compile_uk_full_target_surface,
    registry_from_payload,
)

RECEIPT_SCHEMA_VERSION = 1


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _national_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    registry = registry_from_payload(payload["national_registry"])
    rows = []
    for spec in registry.specs:
        metadata = spec.metadata
        level = str(
            metadata.get("geography_level") or metadata.get("ledger_geography_level")
        )
        geography_id = str(
            metadata.get("geography_id") or metadata.get("ledger_geography_id")
        )
        rows.append(
            {
                "name": spec.name,
                "contract_target_id": metadata.get("contract_target_id"),
                "family": spec.family,
                "geography_level": level,
                "geography_id": geography_id,
                "grain": uk_cross_grain_grain(metadata, level, geography_id),
                "value": float(spec.value),
                "ledger_fact_period": metadata.get("ledger_fact_period"),
                "uprating_from_period": metadata.get("uprating_from_period"),
                "uprating_index": metadata.get("uprating_index"),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _local_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in payload["surface"]:
        rows.append(
            {
                "name": row["target_name"],
                "contract_target_id": row.get("contract_target_id"),
                "family": row.get("family"),
                "area_type": row.get("area_type"),
                "area_code": row.get("area_code"),
                "metric": row.get("metric"),
                "value": float(row["value"]),
            }
        )
    return sorted(rows, key=lambda row: row["name"])


def _national_reconciliation_summary(
    receipt: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if receipt is None:
        return None
    return {
        "band_bridges": receipt.get("band_bridges"),
        "rows_moved_by_exact_signature": receipt.get("rows_moved_by_exact_signature"),
        "cross_grain_groups": _group_summary(receipt.get("cross_grain") or {}),
    }


#: Census residence-type tables the communal validation reads (#1123,
#: decision 2: no subtraction yet, a receipt only): ONS RM120 for England and
#: Wales, NRS UV101a for Scotland. Areas whose communal residents make up at
#: least STUDENT_COMMUNAL_SHARE of their 18-24 residents are listed as student
#: areas, where the age-band targets count residents the household frame
#: cannot hold.
_COMMUNAL_TABLES = {
    "ons.census2021_residents": (
        "C2021_AGE_24_NAME",
        "C2021_RESTYPE_3_NAME",
        "C_SEX_NAME",
    ),
    "nrs.census2022_residents": ("Age", "Residence Type Indicator", "Sex"),
}
STUDENT_COMMUNAL_SHARE = 0.10


def _age_range(label: str) -> tuple[int, int] | None:
    import re

    numbers = [int(value) for value in re.findall(r"\d+", str(label))]
    if "under 1" in str(label).lower():
        return (0, 0)
    if len(numbers) == 2:
        return (numbers[0], numbers[1])
    if len(numbers) == 1 and "over" in str(label).lower():
        return (numbers[0], 200)
    if len(numbers) == 1:
        return (numbers[0], numbers[0])
    return None


def _communal_validation(ledger_facts: Path) -> dict[str, Any]:
    """Communal-establishment residents per area: a validation receipt only."""

    path = (
        ledger_facts / "consumer_facts.jsonl" if ledger_facts.is_dir() else ledger_facts
    )
    totals: dict[tuple[str, str], dict[str, float]] = {}
    with path.open() as handle:
        for line in handle:
            if (
                "census2021_residents" not in line
                and "census2022_residents" not in line
            ):
                continue
            fact = json.loads(line)
            concept = (fact.get("observed_measure") or {}).get("source_concept")
            keys = _COMMUNAL_TABLES.get(str(concept))
            if keys is None:
                continue
            dimensions = fact.get("dimensions") or {}
            age_label, residence, sex = (str(dimensions.get(key, "")) for key in keys)
            if not sex.lower().startswith("all"):
                continue
            geography = fact.get("geography") or {}
            level = str(geography.get("level"))
            if level not in {"constituency", "local_authority"}:
                continue
            cell = totals.setdefault((level, str(geography.get("id"))), {})
            residence_key = (
                "communal"
                if "communal" in residence.lower()
                else "household"
                if "household" in residence.lower()
                else "all"
            )
            value = float(fact.get("value") or 0.0)
            if age_label.lower() == "total":
                cell[f"{residence_key}_all_ages"] = (
                    cell.get(f"{residence_key}_all_ages", 0.0) + value
                )
                continue
            ages = _age_range(age_label)
            if ages is not None and 18 <= ages[0] and ages[1] <= 24:
                cell[f"{residence_key}_18_24"] = (
                    cell.get(f"{residence_key}_18_24", 0.0) + value
                )
    receipt: dict[str, Any] = {
        "basis": (
            "Census 2021 (England and Wales, RM120) and 2022 (Scotland, UV101a) "
            "usual residents by residence type; Northern Ireland's district "
            "table (CT0105) is not split by area grain here. Validation only: "
            "the age-band targets still count communal residents (#1123)."
        ),
        "student_communal_share": STUDENT_COMMUNAL_SHARE,
    }
    for level in ("constituency", "local_authority"):
        rows = []
        for (cell_level, area), cell in sorted(totals.items()):
            if cell_level != level:
                continue
            communal = cell.get("communal_all_ages", 0.0)
            household = cell.get("household_all_ages", 0.0)
            young_communal = cell.get("communal_18_24", 0.0)
            young_household = cell.get("household_18_24", 0.0)
            all_ages = communal + household
            young = young_communal + young_household
            rows.append(
                {
                    "area": area,
                    "communal_share": communal / all_ages if all_ages else None,
                    "communal_share_18_24": young_communal / young if young else None,
                }
            )
        shares = [
            row["communal_share"] for row in rows if row["communal_share"] is not None
        ]
        receipt[level] = {
            "areas": len(rows),
            "median_communal_share": float(np.median(shares)) if shares else None,
            "max_communal_share": max(shares) if shares else None,
            "student_areas": sorted(
                (
                    {
                        "area": row["area"],
                        "communal_share_18_24": row["communal_share_18_24"],
                    }
                    for row in rows
                    if (row["communal_share_18_24"] or 0.0) >= STUDENT_COMMUNAL_SHARE
                ),
                key=lambda row: -row["communal_share_18_24"],
            ),
        }
    return receipt


def _group_summary(cross_geography: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "inconsistency_id": group["inconsistency_id"],
            "bridge_id": group.get("bridge_id"),
            "winning_grain": group["winning_grain"],
            "legs": [
                {
                    "parent_geography_id": leg["parent_geography_id"],
                    "leg": leg["leg"],
                    "declared_factor": leg["declared_factor"],
                    "relative_shift": leg["relative_shift"],
                }
                for leg in group["legs"]
            ],
        }
        for group in cross_geography.get("groups", ())
    ]


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_uk_full_target_inputs(
        args.ledger_facts,
        calibration_year=args.calibration_year,
        exclusions_evaluated_on=date.fromisoformat(args.review_date),
        include_validation_periods=args.full,
    )
    payload = compile_uk_full_target_surface(inputs, args.ladder)
    cross = payload["cross_geography"]
    national = _national_rows(payload)
    local = _local_rows(payload)
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "code_head": _git_head(),
        "calibration_year": payload["calibration_year"],
        "review_date": args.review_date,
        "mode": "full" if args.full else "calibration_year",
        "source_validation": payload["source_validation"],
        "counts": {
            "national_rows": len(national),
            "local_rows": len(local),
            "national_by_family": dict(
                sorted(Counter(r["family"] for r in national).items())
            ),
            "local_by_family": dict(
                sorted(Counter(r["family"] for r in local).items())
            ),
        },
        "uprating_holds": inputs.get("uprating_holds"),
        "communal_validation": (
            _communal_validation(Path(args.ledger_facts)) if args.communal else None
        ),
        "national_reconciliation": _national_reconciliation_summary(
            inputs.get("national_reconciliation")
        ),
        "uprating_holds_without_index": sorted(
            row["name"]
            for row in national
            if row["uprating_from_period"] is not None and not row["uprating_index"]
        ),
        "cross_grain_groups": _group_summary(cross),
        "partitions": cross.get("partitions"),
        "coverage": cross.get("coverage"),
        "census_household_uprating": {
            level: {"factor": grain.get("factor")}
            for level, grain in (
                payload["census_household_uprating"].get("grains") or {}
            ).items()
        },
        "national": national,
        "local": local,
    }


def diff_receipts(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in ("national", "local"):
        old = {row["name"]: row for row in before[scope]}
        new = {row["name"]: row for row in after[scope]}
        moved = []
        for name in sorted(set(old) & set(new)):
            a, b = old[name]["value"], new[name]["value"]
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0):
                moved.append(
                    {
                        "name": name,
                        "family": new[name].get("family"),
                        "before": a,
                        "after": b,
                        "relative_change": (b - a) / a if a else None,
                    }
                )
        by_family: Counter[str] = Counter(str(row["family"]) for row in moved)
        result[scope] = {
            "added": sorted(set(new) - set(old)),
            "removed": sorted(set(old) - set(new)),
            "moved_count": len(moved),
            "moved_by_family": dict(sorted(by_family.items())),
            "moved": sorted(
                moved,
                key=lambda row: -abs(row["relative_change"] or 0.0),
            ),
        }
    result["grain_changes"] = sorted(
        {
            (row["name"], old_row["grain"], row["grain"])
            for row in after["national"]
            if (old_row := {r["name"]: r for r in before["national"]}.get(row["name"]))
            and old_row.get("grain") != row.get("grain")
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument(
        "--review-date",
        required=True,
        help="Exclusion evaluation date (YYYY-MM-DD); pinned so two receipts compare.",
    )
    parser.add_argument("--calibration-year", type=int, default=2025)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--communal",
        action="store_true",
        help="Add the census communal-establishment validation receipt (#1123).",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diff-against", type=Path)
    parser.add_argument("--diff-out", type=Path)
    args = parser.parse_args(argv)
    receipt = build_receipt(args)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["counts"], sort_keys=True))
    if args.diff_against is not None:
        diff = diff_receipts(json.loads(args.diff_against.read_text()), receipt)
        target = args.diff_out or args.out.with_suffix(".diff.json")
        target.write_text(json.dumps(diff, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    scope: {
                        "moved": diff[scope]["moved_count"],
                        "added": len(diff[scope]["added"]),
                        "removed": len(diff[scope]["removed"]),
                    }
                    for scope in ("national", "local")
                }
                | {"grain_changes": len(diff["grain_changes"])},
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
