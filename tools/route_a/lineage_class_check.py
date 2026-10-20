"""d1712 check: is a stage-001 lineage difference exactly the accepted 9/30 class?

Max ruled on 2026-10-19 (d1712) that a repeat of the 9/30 Linux-vs-Mac
float-rounding difference may be accepted: every entity table has the same
shape and index, totals are equal, and every differing value is within 1e-12
relative. Anything else stops for Max. On 9/30 the only difference was
person.pre_subsidy_rent (313 of 432,523 rows, max relative 6.08e-15), so a
difference in any other column is reported as outside the class.

Usage (with the build worktree's interpreter):
    python lineage_class_check.py LOCAL_FRAME.h5 MODAL_FRAME.h5 OUT.json
Exit 0 = inside the class (or identical), 1 = outside it.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import _frame_tables, load_frame_checkpoint

REL_TOL = 1e-12
ACCEPTED_COLUMNS = {("person", "pre_subsidy_rent")}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tables(path: Path) -> dict[str, pd.DataFrame]:
    return dict(_frame_tables(load_frame_checkpoint(path).frame))


def compare_column(name: str, column: str, a: pd.Series, b: pd.Series) -> dict | None:
    """None when identical; otherwise a summary of the difference."""

    if str(a.dtype) != str(b.dtype):
        return {
            "table": name,
            "column": column,
            "problem": f"dtype {a.dtype} != {b.dtype}",
        }
    av, bv = a.to_numpy(), b.to_numpy()
    if a.dtype.kind == "f":
        same = (av == bv) | (np.isnan(av) & np.isnan(bv))
    else:
        same = np.asarray(a.eq(b) | (a.isna() & b.isna()), dtype=bool)
    if bool(same.all()):
        return None
    differ = ~same
    entry: dict[str, object] = {
        "table": name,
        "column": column,
        "rows_differ": int(differ.sum()),
        "rows": int(len(av)),
    }
    if a.dtype.kind != "f":
        entry["problem"] = "non-float column differs"
        return entry
    if bool((np.isnan(av) != np.isnan(bv)).any()):
        entry["problem"] = "NaN positions differ"
        return entry
    da, db = av[differ], bv[differ]
    absdiff = np.abs(da - db)
    scale = np.maximum(np.abs(da), np.abs(db))
    rel = np.where(scale > 0, absdiff / np.where(scale > 0, scale, 1.0), np.inf)
    sum_a, sum_b = float(np.nansum(av)), float(np.nansum(bv))
    sum_rel = abs(sum_a - sum_b) / max(abs(sum_a), abs(sum_b), 1e-300)
    entry.update(
        {
            "max_abs": float(absdiff.max()),
            "max_rel": float(rel.max()),
            "sum_local": sum_a,
            "sum_modal": sum_b,
            "sum_rel_diff": sum_rel,
        }
    )
    if not (rel.max() <= REL_TOL and sum_rel <= REL_TOL):
        entry["problem"] = f"difference above {REL_TOL} relative"
    return entry


def main() -> int:
    local_path, modal_path, out_path = (Path(p) for p in sys.argv[1:4])
    local, modal = tables(local_path), tables(modal_path)
    problems: list[str] = []
    differences: list[dict] = []
    shapes: dict[str, object] = {}
    if set(local) != set(modal):
        problems.append(f"table sets differ: {sorted(local)} vs {sorted(modal)}")
    for name in sorted(set(local) & set(modal)):
        a, b = local[name], modal[name]
        shapes[name] = {"local": list(a.shape), "modal": list(b.shape)}
        if a.shape != b.shape:
            problems.append(f"{name}: shape {a.shape} != {b.shape}")
            continue
        if not a.index.equals(b.index):
            problems.append(f"{name}: index differs")
            continue
        if list(a.columns) != list(b.columns):
            problems.append(f"{name}: column names or order differ")
            continue
        for column in a.columns:
            entry = compare_column(name, column, a[column], b[column])
            if entry is None:
                continue
            differences.append(entry)
            if "problem" in entry:
                problems.append(f"{name}.{column}: {entry['problem']}")
            elif (name, column) not in ACCEPTED_COLUMNS:
                problems.append(
                    f"{name}.{column}: within tolerance but not the 9/30 column; stop for Max"
                )
    verdict = {
        "ruling": "d1712 (Max 2026-10-19)",
        "rel_tol": REL_TOL,
        "accepted_columns": sorted(f"{t}.{c}" for t, c in ACCEPTED_COLUMNS),
        "local": {"path": str(local_path), "sha256": sha256(local_path)},
        "modal": {"path": str(modal_path), "sha256": sha256(modal_path)},
        "shapes": shapes,
        "differences": differences,
        "problems": problems,
        "inside_accepted_class": not problems,
    }
    Path(out_path).write_text(json.dumps(verdict, indent=1) + "\n")
    print(
        json.dumps({"inside_accepted_class": not problems, "problems": problems[:10]})
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
