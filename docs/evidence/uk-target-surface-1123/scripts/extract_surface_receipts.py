"""Extract the committed evidence from two target-surface receipts (#1123).

The receipts ``tools/uk_target_surface_receipt.py`` writes carry every bound
row and run to several megabytes each, so they stay outside the tree. This
writes the two extracts beside it:

* ``surface-diff.json``: what the surface moves between the receipts. Every
  national row added, removed or moved, and the local cells by target and
  grain (counts, totals before and after, and the spread of the relative
  moves).
* ``reconciliation.json``: how the after surface was reconciled. The national
  band bridges with their bounds, the sum bridges, each fan-out sum bridge at
  each local grain, every cross-grain group's factors, the subset relations,
  the partitions, the uprating holds by kind and the communal validation.

Usage::

    python extract_surface_receipts.py BEFORE.json AFTER.json OUT_DIR
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _spread(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def _local_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["contract_target_id"]), str(row["area_type"])


def surface_diff(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    old = {row["name"]: row for row in before["national"]}
    new = {row["name"]: row for row in after["national"]}
    moved = [
        {
            "name": name,
            "family": new[name].get("family"),
            "before": old[name]["value"],
            "after": new[name]["value"],
            "relative_change": new[name]["value"] / old[name]["value"] - 1.0
            if old[name]["value"]
            else None,
        }
        for name in sorted(set(old) & set(new))
        if old[name]["value"] != new[name]["value"]
    ]
    national = {
        "rows_before": len(old),
        "rows_after": len(new),
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "moved_count": len(moved),
        "moved": moved,
    }

    old_local = {row["name"]: row for row in before["local"]}
    new_local = {row["name"]: row for row in after["local"]}
    cells: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "cells_before": 0,
            "cells_after": 0,
            "added": 0,
            "removed": 0,
            "moved": 0,
            "total_before": 0.0,
            "total_after": 0.0,
            "_changes": [],
        }
    )
    for name, row in old_local.items():
        cell = cells[_local_key(row)]
        cell["cells_before"] += 1
        cell["total_before"] += row["value"]
        if name not in new_local:
            cell["removed"] += 1
    for name, row in new_local.items():
        cell = cells[_local_key(row)]
        cell["cells_after"] += 1
        cell["total_after"] += row["value"]
        previous = old_local.get(name)
        if previous is None:
            cell["added"] += 1
        elif previous["value"] != row["value"]:
            cell["moved"] += 1
            if previous["value"]:
                cell["_changes"].append(row["value"] / previous["value"] - 1.0)
    local_rows = []
    for (target_id, area_type), cell in sorted(cells.items()):
        changes = cell.pop("_changes")
        local_rows.append(
            {
                "contract_target_id": target_id,
                "area_type": area_type,
                **cell,
                "relative_change": _spread(changes) if changes else None,
            }
        )
    return {
        "before": {
            "code_head": before["code_head"],
            "code_modifications": _modifications(before),
            "review_date": before["review_date"],
            "source_validation": before["source_validation"],
        },
        "after": {
            "code_head": after["code_head"],
            "code_modifications": _modifications(after),
            "review_date": after["review_date"],
            "source_validation": after["source_validation"],
        },
        "national": national,
        "local": {
            "rows_before": len(old_local),
            "rows_after": len(new_local),
            "added": len(set(new_local) - set(old_local)),
            "removed": len(set(old_local) - set(new_local)),
            "moved": sum(row["moved"] for row in local_rows),
            "by_target_and_grain": local_rows,
        },
    }


def _modifications(receipt: dict[str, Any]) -> dict[str, Any] | None:
    """The receipt's working-tree record, less the diff text (its digest stays)."""

    modifications = receipt.get("code_modifications")
    if modifications is None:
        return None
    return {key: value for key, value in modifications.items() if key != "diff"}


def reconciliation(after: dict[str, Any]) -> dict[str, Any]:
    partitions = []
    for partition in (after.get("partitions") or {}).get("partitions", ()):
        factors = [cell["declared_factor"] for cell in partition.get("cells", ())]
        partitions.append(
            {key: value for key, value in partition.items() if key not in {"cells"}}
            | {
                "cells": len(factors),
                "declared_factor": _spread(factors) if factors else None,
            }
        )
    communal = after.get("communal_validation") or {}
    return {
        "code_head": after["code_head"],
        "code_modifications": _modifications(after),
        "review_date": after["review_date"],
        "counts": after["counts"],
        "national_reconciliation": after.get("national_reconciliation"),
        "fanout_sum_bridges": after.get("fanout_sum_bridges"),
        "off_surface_cells": after.get("off_surface_cells"),
        "cross_grain_groups": after.get("cross_grain_groups"),
        "subset_relations": after.get("subset_relations"),
        "partitions": partitions,
        "parents_without_members": len(
            (after.get("partitions") or {}).get("parents_without_members", ())
        ),
        "coverage": after.get("coverage"),
        "census_household_uprating": after.get("census_household_uprating"),
        "uprating_holds": after.get("uprating_holds"),
        "communal_validation": {
            "basis": communal.get("basis"),
            "student_communal_share": communal.get("student_communal_share"),
            **{
                grain: {
                    key: value
                    for key, value in (communal.get(grain) or {}).items()
                    if not isinstance(value, (list, dict))
                }
                for grain in ("constituency", "local_authority")
            },
        }
        if communal
        else None,
    }


def main(argv: list[str]) -> int:
    before_path, after_path, out_dir = (Path(value) for value in argv)
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    for name, payload in (
        ("surface-diff.json", surface_diff(before, after)),
        ("reconciliation.json", reconciliation(after)),
    ):
        (out_dir / name).write_text(
            json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
