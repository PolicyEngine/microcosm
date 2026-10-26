"""Write a receipt of the compiled UK target surface, or diff two receipts.

The receipt is what the ``uk.full.target_compilation`` node compiles, read
through the node's own code path (``compile_uk_full_target_surface``): every
national row after the measure exclusions and every local cell after uprating
and cross-grain reconciliation, with the reconciliation groups' factors, the
national band bridges and their bounds, each fan-out sum bridge at each local
grain, the subset relations, the partitions and the uprating holds.
``--diff-against`` compares it with an earlier receipt row by row, so a
change to the target surface states exactly which values it moves
(microcosm#1123).

Inputs, neither of which the repository carries: the Chronicle consumer
artifact at the commit ``uk/chronicle_feed.json`` pins (Chronicle's
``build-bundle --suite uk`` then ``build-consumer-artifact``; the loader
checks its hashes against the pin) and the 2021 output-area ladder
(``uk_oa_ladder_2021.npz``) a UK local build produces.

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
import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.uk_runtime import communal_residents
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


#: The receipt carries the working tree's diff itself up to this size, so a
#: reader can see what an overlay changed; past it, only the diff's digest.
_MAX_RECEIPTED_DIFF_BYTES = 262_144
_CODE_PATHS = ("packages", "tools")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _code_modifications() -> dict[str, Any] | None:
    """What the working tree adds to ``code_head``, so a receipt can be checked.

    A receipt may be run over a tree that is not the commit it names: a
    register signed for the run only, or a tool laid over an older commit
    that does not carry it. ``code_head`` alone would then pass as the
    commit's own receipt. This records every file under ``packages`` and
    ``tools`` that differs from ``code_head``, tracked or not, with the
    SHA-256 of its content, and the diff of the tracked ones (its digest
    always, its text while it is small). An untracked file has no diff: its
    digest is what a reader checks against the commit that later carries it.
    """

    root = _git("rev-parse", "--show-toplevel")
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *_CODE_PATHS)
    diff = _git("diff", "HEAD", "--", *_CODE_PATHS)
    if root is None or status is None or diff is None:
        return None
    files = []
    for line in status.splitlines():
        if not line.strip():
            continue
        path = line[3:].split(" -> ")[-1].strip('"')
        content = Path(root.strip()) / path
        files.append(
            {
                "path": path,
                "status": line[:2].strip(),
                "sha256": (
                    hashlib.sha256(content.read_bytes()).hexdigest()
                    if content.is_file()
                    else None
                ),
            }
        )
    encoded = diff.encode("utf-8")
    return {
        "files": sorted(files, key=lambda entry: entry["path"]),
        "diff_sha256": hashlib.sha256(encoded).hexdigest() if diff else None,
        "diff": diff if len(encoded) <= _MAX_RECEIPTED_DIFF_BYTES else None,
    }


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
        "sum_bridges": receipt.get("sum_bridges"),
        "fanout_sum_controls": {
            bridge_id: {
                key: control.get(key)
                for key in ("higher_target_id", "control", "control_before_alignment")
            }
            for bridge_id, control in (receipt.get("fanout_sum_controls") or {}).items()
        },
        "rows_moved_by_exact_signature": receipt.get("rows_moved_by_exact_signature"),
        "cross_grain_groups": _group_summary(receipt.get("cross_grain") or {}),
    }


#: Areas whose communal residents make up at least STUDENT_COMMUNAL_SHARE of
#: their 18-24 residents are listed as student areas, where the age-band
#: targets count residents the household frame cannot hold. The census
#: residence-type tables, the age bands and the cells the receipt sums live in
#: ``uk_runtime.communal_residents`` (#1123, #1278), which the incumbent
#: evaluation reads too (microcosm#1095).
STUDENT_COMMUNAL_SHARE = 0.10
COMMUNAL_AGE_BANDS = communal_residents.COMMUNAL_AGE_BANDS
_band_key = communal_residents.band_key
_band_of = communal_residents.band_of
_age_range = communal_residents.age_range
#: A band whose communal share is s puts the private-household target a
#: fraction s below the total-resident row the incumbent evaluation gates at
#: 25% (microcosm#1278 sizing of the age-universe change).
SIZING_SHARES = (0.20, 0.25)


def _share_summary(shares: dict[str, float], *, threshold_list: bool) -> dict[str, Any]:
    values = sorted(shares.values())
    summary: dict[str, Any] = {
        "areas": len(values),
        "median_share": float(np.median(values)) if values else None,
        "areas_at_or_above": {
            f"{share:.2f}": sum(value >= share for value in values)
            for share in SIZING_SHARES
        },
    }
    if threshold_list:
        summary["areas_at_or_above_lowest"] = sorted(
            (
                {"area": area, "communal_share": share}
                for area, share in shares.items()
                if share >= SIZING_SHARES[0]
            ),
            key=lambda row: (-row["communal_share"], row["area"]),
        )
    return summary


def _communal_validation(
    ledger_facts: Path, *, region_by_area: dict[str, str] | None = None
) -> dict[str, Any]:
    """Communal-establishment residents per area and age band: a receipt only."""

    path = (
        ledger_facts / "consumer_facts.jsonl" if ledger_facts.is_dir() else ledger_facts
    )
    with path.open() as handle:
        cells = communal_residents.communal_cells(
            json.loads(line)
            for line in handle
            if communal_residents.is_communal_fact_line(line)
        )
    totals = cells.totals
    band_keys = list(communal_residents.BAND_KEYS)
    receipt: dict[str, Any] = {
        "basis": (
            "Census 2021 (England and Wales, RM120) and 2022 (Scotland, UV101a) "
            "usual residents by residence type; Northern Ireland's district "
            "table (CT0105) is approximate (see northern_ireland_local_authority). "
            "Validation and sizing only: the age-band targets still count "
            "communal residents (#1123, #1278)."
        ),
        "student_communal_share": STUDENT_COMMUNAL_SHARE,
        "sizing_shares": list(SIZING_SHARES),
    }
    for level in communal_residents.COMMUNAL_LEVELS:
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
        band_shares = communal_residents.band_shares(cells, level)
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
            "bands": {
                key: _share_summary(band_shares[key], threshold_list=True)
                for key in band_keys
            },
        }
    ni_shares = communal_residents.ni_band_shares(cells)
    ni_counts = communal_residents.ni_band_counts(cells)
    receipt["northern_ireland_local_authority"] = {
        "approximate": True,
        "basis": (
            "NISRA CT0105 communal residents by district (census 2021), each age "
            "band spread evenly over its single years, over NISRA's mid-2025 "
            "district population by single year."
        ),
        "bands": {
            key: _share_summary(ni_shares[key], threshold_list=True)
            for key in band_keys
        },
    }
    if region_by_area:
        region_totals: dict[str, dict[str, float]] = {}
        for (level, area), cell in totals.items():
            if level != "local_authority" or area not in region_by_area:
                continue
            region = region_totals.setdefault(region_by_area[area], {})
            for key, value in cell.items():
                region[key] = region.get(key, 0.0) + value
        for area, bands in ni_counts.items():
            region = region_totals.setdefault(region_by_area.get(area, "N92000002"), {})
            for key, (communal, total) in bands.items():
                region[f"communal_{key}"] = (
                    region.get(f"communal_{key}", 0.0) + communal
                )
                region[f"household_{key}"] = region.get(f"household_{key}", 0.0) + (
                    total - communal
                )
        receipt["region"] = {
            "composed_from": "local authorities (Northern Ireland approximate)",
            "bands": {
                key: {
                    region: (
                        cell.get(f"communal_{key}", 0.0)
                        / (
                            cell.get(f"communal_{key}", 0.0)
                            + cell.get(f"household_{key}", 0.0)
                        )
                    )
                    for region, cell in sorted(region_totals.items())
                    if cell.get(f"communal_{key}", 0.0)
                    + cell.get(f"household_{key}", 0.0)
                }
                for key in band_keys
            },
        }
    return receipt


def _region_by_local_authority() -> dict[str, str]:
    crosswalk = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "packages/microcosm-build/src/microcosm/build/uk/local_area_crosswalk.json"
        ).read_text()
    )
    return dict(crosswalk["levels"]["local_authority"]["region_code_by_area"])


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
    modifications = _code_modifications()
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "code_head": _git_head(),
        # The paths alone, for a reader; ``code_modifications`` carries each
        # file's digest and the diff.
        "code_modified_paths": (
            None
            if modifications is None
            else [entry["path"] for entry in modifications["files"]]
        ),
        "code_modifications": modifications,
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
            _communal_validation(
                Path(args.ledger_facts), region_by_area=_region_by_local_authority()
            )
            if args.communal
            else None
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
        # Each fan-out sum bridge at each local grain, read before any
        # reconciliation: the applied factor and the source-agreement factor
        # the gate tests.
        "fanout_sum_bridges": cross.get("fanout_sum_bridges"),
        # Signed cells that are off the surface and still count in their leg:
        # deferred with a value, or implied from the target's other tier.
        "off_surface_cells": cross.get("off_surface_cells"),
        # The small-cell rule's deferrals: in the frame, never in the solve.
        "small_cell_deferrals": cross.get("small_cell_deferrals"),
        "subset_relations": cross.get("subset_relations"),
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
    parser.add_argument("--ladder", type=Path)
    parser.add_argument(
        "--review-date",
        help="Exclusion evaluation date (YYYY-MM-DD); pinned so two receipts compare.",
    )
    parser.add_argument("--calibration-year", type=int, default=2025)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--communal",
        action="store_true",
        help="Add the census communal-establishment validation receipt (#1123).",
    )
    parser.add_argument(
        "--communal-only",
        action="store_true",
        help=(
            "Write only the communal receipt, without compiling the surface "
            "(#1278 sizing); needs no ladder or review date."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--diff-against", type=Path)
    parser.add_argument("--diff-out", type=Path)
    args = parser.parse_args(argv)
    if args.communal_only:
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "code_head": _git_head(),
            "ledger_facts": Path(args.ledger_facts).name,
            "communal_validation": _communal_validation(
                Path(args.ledger_facts), region_by_area=_region_by_local_authority()
            ),
        }
        args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return 0
    if args.ladder is None or args.review_date is None:
        parser.error("--ladder and --review-date are required unless --communal-only")
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
