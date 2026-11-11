"""Compare this branch's ASEC raw-stage frame with the 2026-08-23 corrected one.

The 8/23 frame was built from H5s with 24 appended columns (receipt_720.json);
this branch restores 11 of them in the pool. Every entity table's rows, every
shared column, and the unit assignments should agree; the only expected
differences are the 13 unrestored columns (all-populated there, NaN on the
2022/2023 rows here) and columns other code changes since 8/23 added.
"""

from __future__ import annotations

import gc
import json
import resource
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime.asec_census_person_columns import (
    ASEC_CENSUS_PERSON_COLUMN_NAMES,
    ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED,
)

HERE = Path(__file__).resolve().parent
OURS = HERE / "checkpoints" / "asec_raw_stage.checkpoint.h5"
THEIRS = Path(
    "/Users/maxghenis/PolicyEngine/_buildo-runtime/out/asec-720-source/"
    "checkpoints/asec_raw_stage.checkpoint.h5"
)


def tables(path: Path) -> dict[str, pd.DataFrame]:
    frame = load_frame_checkpoint(path).frame
    return {entity: frame.table(entity) for entity in frame.entities}


def same(left: pd.Series, right: pd.Series) -> bool:
    if left.equals(right):
        return True
    try:
        a = pd.to_numeric(left, errors="raise").to_numpy(dtype=np.float64)
        b = pd.to_numeric(right, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError):
        return bool(
            (
                left.astype("string").fillna("<NA>")
                == right.astype("string").fillna("<NA>")
            ).all()
        )
    return bool(np.array_equal(a, b, equal_nan=True))


def main() -> None:
    ours = tables(OURS)
    theirs = tables(THEIRS)
    report: dict[str, object] = {"ours": str(OURS), "theirs": str(THEIRS)}
    for entity in sorted(set(ours) | set(theirs)):
        left, right = ours.get(entity), theirs.get(entity)
        if left is None or right is None:
            report[entity] = {"present": [left is not None, right is not None]}
            continue
        shared = [c for c in left.columns if c in right.columns]
        differing = [
            c
            for c in shared
            if len(left) != len(right)
            or not same(left[c].reset_index(drop=True), right[c].reset_index(drop=True))
        ]
        report[entity] = {
            "rows": [len(left), len(right)],
            "shared_columns": len(shared),
            "differing_shared_columns": differing,
            "only_ours": sorted(set(left.columns) - set(right.columns)),
            "only_theirs": sorted(set(right.columns) - set(left.columns)),
        }
    person_ours, person_theirs = ours["person"], theirs["person"]
    year = person_ours["source_year"].to_numpy()
    restored = {}
    for column in ASEC_CENSUS_PERSON_COLUMN_NAMES:
        restored[column] = same(person_ours[column], person_theirs[column])
    report["restored_columns_identical_across_all_rows"] = restored
    unrestored = {}
    for column in ASEC_CENSUS_PERSON_COLUMNS_NOT_RESTORED:
        if column not in person_ours or column not in person_theirs:
            unrestored[column] = "absent on one side"
            continue
        is_2024 = year == 2024
        unrestored[column] = {
            "2024_rows_identical": same(
                person_ours.loc[is_2024, column].reset_index(drop=True),
                person_theirs.loc[is_2024, column].reset_index(drop=True),
            ),
            "ours_null_rows_2022_2023": int(
                person_ours.loc[~is_2024, column].isna().sum()
            ),
            "theirs_null_rows_2022_2023": int(
                person_theirs.loc[~is_2024, column].isna().sum()
            ),
        }
    report["unrestored_columns"] = unrestored
    report["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    (HERE / "compare_823_raw_stage.json").write_text(
        json.dumps(report, indent=1, sort_keys=True)
    )
    print(json.dumps(report, indent=1, sort_keys=True))
    del ours, theirs
    gc.collect()


if __name__ == "__main__":
    main()
