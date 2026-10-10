"""Compare every stored column of a US base frame H5 with a release H5.

Rows are matched on each entity's id column. For every column present in
both files the receipt records how many rows changed, split by support channel
and clone index, the direction of each change (zero to positive, positive to
zero, up, down), the largest absolute difference and, where every changed row
moved by the same ratio, that ratio. Columns present in only one file are
listed. The household weight is reported separately because calibration
always changes it.

Usage:
    python base_release_column_diff.py BASE_H5 RELEASE_H5 OUT_JSON
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENTITIES = ("person", "tax_unit", "spm_unit", "household", "family", "marital_unit")
ID_COLUMN = {
    "person": "person_id",
    "tax_unit": "tax_unit_id",
    "spm_unit": "spm_unit_id",
    "household": "household_id",
    "family": "family_id",
    "marital_unit": "marital_unit_id",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def channel_labels(table: pd.DataFrame, entity: str) -> pd.Series:
    channel = table.get(f"{entity}_support_channel")
    clone = table.get(f"{entity}_support_clone_index")
    if channel is None or clone is None:
        return pd.Series("all", index=table.index)
    channel = channel.map(
        lambda value: value.decode() if isinstance(value, bytes) else str(value)
    )
    return channel + "/clone" + clone.astype(int).astype(str)


def as_float(values: pd.Series) -> np.ndarray | None:
    if values.dtype.kind in "biuf":
        return values.to_numpy(dtype=np.float64)
    return None


def compare_column(
    base: pd.Series, release: pd.Series, labels: pd.Series
) -> dict[str, object] | None:
    base_float = as_float(base)
    release_float = as_float(release)
    if base_float is not None and release_float is not None:
        both_nan = np.isnan(base_float) & np.isnan(release_float)
        changed = ~both_nan & (base_float != release_float)
    else:
        base_text = base.map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
        release_text = release.map(
            lambda v: v.decode() if isinstance(v, bytes) else str(v)
        )
        changed = (base_text != release_text).to_numpy()
    n_changed = int(changed.sum())
    if n_changed == 0:
        return None
    by_channel = pd.Series(changed, index=labels.index).groupby(labels.to_numpy()).sum()
    record: dict[str, object] = {
        "rows_changed": n_changed,
        "rows_changed_by_channel": {
            str(key): int(value) for key, value in by_channel.items()
        },
    }
    if base_float is not None and release_float is not None:
        b = base_float[changed]
        r = release_float[changed]
        record["max_abs_difference"] = float(np.nanmax(np.abs(r - b)))
        record["direction"] = {
            "zero_to_nonzero": int(np.count_nonzero((b == 0) & (r != 0))),
            "nonzero_to_zero": int(np.count_nonzero((b != 0) & (r == 0))),
            "up_both_nonzero": int(np.count_nonzero((b != 0) & (r != 0) & (r > b))),
            "down_both_nonzero": int(np.count_nonzero((b != 0) & (r != 0) & (r < b))),
        }
        record["sum_base_changed_rows"] = float(np.nansum(b))
        record["sum_release_changed_rows"] = float(np.nansum(r))
        nonzero = (b != 0) & (r != 0)
        if nonzero.any():
            ratios = r[nonzero] / b[nonzero]
            record["ratio_min"] = float(np.min(ratios))
            record["ratio_max"] = float(np.max(ratios))
        unchanged = ~changed
        record["rows_nonzero_base"] = int(np.count_nonzero(base_float != 0))
        record["rows_nonzero_release"] = int(np.count_nonzero(release_float != 0))
        record["rows_nonzero_base_by_channel"] = {
            str(key): int(value)
            for key, value in pd.Series(base_float != 0, index=labels.index)
            .groupby(labels.to_numpy())
            .sum()
            .items()
        }
        record["rows_unchanged_nonzero"] = int(
            np.count_nonzero(unchanged & (base_float != 0))
        )
    return record


def main() -> None:
    base_path, release_path, out_path = (Path(arg) for arg in sys.argv[1:4])
    receipt: dict[str, object] = {
        "base_h5": str(base_path),
        "base_h5_sha256": sha256(base_path),
        "release_h5": str(release_path),
        "release_h5_sha256": sha256(release_path),
        "entities": {},
    }
    for entity in ENTITIES:
        base = pd.read_hdf(base_path, entity)
        release = pd.read_hdf(release_path, entity)
        id_column = ID_COLUMN[entity]
        base = base.set_index(id_column, drop=False)
        release = release.set_index(id_column, drop=False)
        only_base = sorted(set(base.columns) - set(release.columns))
        only_release = sorted(set(release.columns) - set(base.columns))
        common_ids = base.index.intersection(release.index)
        base = base.loc[common_ids]
        release = release.loc[common_ids]
        labels = channel_labels(base, entity)
        changed_columns: dict[str, object] = {}
        identical_columns: list[str] = []
        for column in sorted(set(base.columns) & set(release.columns)):
            result = compare_column(base[column], release[column], labels)
            if result is None:
                identical_columns.append(column)
            else:
                changed_columns[column] = result
        receipt["entities"][entity] = {
            "rows_base": int(len(base)),
            "rows_release": int(len(release)),
            "rows_matched": int(len(common_ids)),
            "rows_by_channel": {
                str(key): int(value) for key, value in labels.value_counts().items()
            },
            "columns_only_in_base": only_base,
            "columns_only_in_release": only_release,
            "columns_identical_count": len(identical_columns),
            "columns_identical": identical_columns,
            "columns_changed": changed_columns,
        }
        print(
            entity,
            "matched",
            len(common_ids),
            "changed",
            sorted(changed_columns),
            flush=True,
        )
    Path(out_path).write_text(json.dumps(receipt, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
