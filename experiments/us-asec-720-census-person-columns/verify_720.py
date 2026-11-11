"""Verify the #720 restoration on real inputs (source construction only).

1. Gate before/after: us_reported_coverage_vintage_signal_gate on
   derive_us_cps_carried_inputs of the ASEC raw-stage frame, for route A's
   raw-stage checkpoint (commit 47976be6c, before) and this branch's (after).
2. Column-by-column equality of the restored per-vintage person tables with
   the 2026-08-23 corrected H5s (_buildo-runtime/inputs/asec-720).
3. Structural deltas the restored A_EXPRRP/PTOTVAL/A_ENRLW/A_FTPT cause.

Writes verify_720.json next to this script.
"""

from __future__ import annotations

import gc
import hashlib
import json
import resource
import sys
from pathlib import Path

import pandas as pd

from microcosm.build.us_runtime import (
    derive_us_cps_carried_inputs,
    load_asec_raw_stage_checkpoint,
    us_reported_coverage_vintage_signal_gate,
)
from microcosm.build.us_runtime.asec_census_person_columns import (
    ASEC_CENSUS_PERSON_COLUMN_NAMES,
    ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED,
    restore_asec_census_person_columns,
)
from microcosm.build.us_runtime.asec_pool import load_asec_h5_tables

HERE = Path(__file__).resolve().parent
BEFORE = Path(
    "/Users/maxghenis/PolicyEngine/_recovered/scratch-backup/893/overnight-20260923/"
    "route-a/run-47976be6ce75/base-checkpoints/asec_raw_stage.checkpoint.h5"
)
AFTER = HERE / "checkpoints" / "asec_raw_stage.checkpoint.h5"
STO = Path(
    "/Users/maxghenis/PolicyEngine/policyengine-us-data/policyengine_us_data/storage"
)
ARC = Path("/Users/maxghenis/PolicyEngine/_buildm-runtime/inputs/asec_education")
CORRECTED = Path("/Users/maxghenis/PolicyEngine/_buildo-runtime/inputs/asec-720")
ARCHIVES = {
    2022: "asecpub23csv.zip",
    2023: "asecpub24csv.zip",
    2024: "asecpub25csv.zip",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS


def gate_on(path: Path) -> tuple[dict, pd.DataFrame]:
    frame, metadata = load_asec_raw_stage_checkpoint(path)
    carried = derive_us_cps_carried_inputs(frame)
    gate = us_reported_coverage_vintage_signal_gate(carried)
    keep = [
        "source_year",
        "PERIDNUM",
        "A_EXPRRP",
        "PTOTVAL",
        "A_ENRLW",
        "A_FTPT",
        "LKWEEKS",
        "person_tax_unit_id",
        *[c for c in ASEC_CENSUS_PERSON_COLUMN_NAMES if c.startswith("NOW_")],
    ]
    person = frame.table("person")
    slim = person[[c for c in keep if c in person.columns]].copy()
    rows = {entity: int(frame.n(entity)) for entity in frame.entities}
    sources = metadata["source_receipt"]["metadata"]["sources"]
    result = {
        "checkpoint": str(path),
        "checkpoint_sha256": sha256(path),
        "rows": rows,
        "relationship_recode_source": {
            str(item["year"]): item["relationship_recode_source"] for item in sources
        },
        "census_person_columns": {
            str(item["year"]): item.get("census_person_columns") for item in sources
        },
        "raw_source_mappings": sorted(metadata["raw_source_mappings"]),
        "gate": {
            "passed": bool(gate.passed),
            "failure_count": len(gate.failures),
            "failures": list(gate.failures),
            "vintages": gate.details.get("vintages"),
        },
    }
    del frame, carried, person
    gc.collect()
    return result, slim


def compare_with_corrected() -> dict:
    """Restore each pinned H5 exactly as the pool does; compare to 8/23 files."""

    out = {}
    for year in (2022, 2023, 2024):
        h5 = STO / f"census_cps_{year}.h5"
        person = load_asec_h5_tables(h5)["person"].reset_index(drop=True)
        restored, record = restore_asec_census_person_columns(
            person, income_year=year, source_path=ARC / ARCHIVES[year]
        )
        entry = {
            "h5_sha256": sha256(h5),
            "columns_added": record["columns_added"],
            "columns_verified_equal": record["columns_verified_equal"],
            "joined_person_rows": record["joined_person_rows"],
            "NOW_MCAID_yes_rows": record["member_values"]["NOW_MCAID"]["1"],
            "source_form": record["source_form"],
        }
        if year in (2022, 2023):
            corrected_path = CORRECTED / f"census_cps_{year}.h5"
            with pd.HDFStore(corrected_path, mode="r") as store:
                corrected = store["person"].reset_index(drop=True)
            entry["corrected_h5"] = str(corrected_path)
            entry["corrected_h5_sha256"] = sha256(corrected_path)
            comparison = {}
            for column in ASEC_CENSUS_PERSON_COLUMN_NAMES:
                ours, theirs = restored[column], corrected[column]
                comparison[column] = {
                    "identical": bool(
                        ours.dtype == theirs.dtype
                        and len(ours) == len(theirs)
                        and ours.equals(theirs)
                    ),
                    "dtype": [str(ours.dtype), str(theirs.dtype)],
                }
            entry["added_columns_vs_corrected"] = comparison
            entry["all_added_columns_identical"] = all(
                item["identical"] for item in comparison.values()
            )
            # Every pre-existing column: our table leaves them exactly as the
            # pinned H5 has them, as the corrected file did.
            entry["existing_columns_identical_to_corrected"] = all(
                restored[column].equals(corrected[column])
                and restored[column].dtype == corrected[column].dtype
                for column in person.columns
            )
            entry["corrected_only_columns"] = sorted(
                set(corrected.columns) - set(restored.columns)
            )
            entry["corrected_only_columns_all_in_not_restored"] = set(
                entry["corrected_only_columns"]
            ) <= set(ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED)
            del corrected
        out[str(year)] = entry
        del person, restored
        gc.collect()
    return out


def lkweeks_2022_vs_corrected(after: pd.DataFrame) -> dict:
    """The dropped LKWEEKS is still restored for 2022 by the existing sidecar."""

    with pd.HDFStore(CORRECTED / "census_cps_2022.h5", mode="r") as store:
        corrected = store["person"][["PERIDNUM", "LKWEEKS"]]
    ours = after.loc[after["source_year"] == 2022, ["PERIDNUM", "LKWEEKS"]]
    merged = ours.merge(corrected, on="PERIDNUM", suffixes=("_ours", "_corrected"))
    return {
        "rows_joined": int(len(merged)),
        "equal_rows": int(
            (merged["LKWEEKS_ours"].astype(float) == merged["LKWEEKS_corrected"]).sum()
        ),
    }


def structural_deltas(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    key = ["source_year", "PERIDNUM"]
    merged = before[[*key, "A_EXPRRP"]].merge(
        after[[*key, "A_EXPRRP"]], on=key, suffixes=("_before", "_after")
    )
    out = {"persons_joined": int(len(merged))}
    for year in (2022, 2023, 2024):
        rows = merged[merged["source_year"] == year]
        differ = rows["A_EXPRRP_before"] != rows["A_EXPRRP_after"]
        out[f"A_EXPRRP_changed_{year}"] = int(differ.sum())
        out[f"A_EXPRRP_partner_roommate_13_{year}"] = {
            "before": int((rows["A_EXPRRP_before"] == 13).sum()),
            "after": int((rows["A_EXPRRP_after"] == 13).sum()),
        }
    for name, frame in (("before", before), ("after", after)):
        out[f"tax_units_{name}"] = int(frame["person_tax_unit_id"].nunique())
        out[f"tax_units_by_year_{name}"] = {
            str(year): int(group["person_tax_unit_id"].nunique())
            for year, group in frame.groupby("source_year")
        }
        for column in ("PTOTVAL", "A_ENRLW", "A_FTPT"):
            if column in frame:
                out[f"{column}_null_by_year_{name}"] = {
                    str(year): int(group[column].isna().sum())
                    for year, group in frame.groupby("source_year")
                }
    return out


def main() -> int:
    report: dict[str, object] = {}
    report["before"], before = gate_on(BEFORE)
    print("before gate passed:", report["before"]["gate"]["passed"], flush=True)
    report["after"], after = gate_on(AFTER)
    print("after gate passed:", report["after"]["gate"]["passed"], flush=True)
    report["structural_deltas"] = structural_deltas(before, after)
    report["lkweeks_2022_vs_corrected"] = lkweeks_2022_vs_corrected(after)
    del before, after
    gc.collect()
    report["corrected_h5_comparison"] = compare_with_corrected()
    report["peak_rss_gb"] = peak_rss_gb()
    (HERE / "verify_720.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True)[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
