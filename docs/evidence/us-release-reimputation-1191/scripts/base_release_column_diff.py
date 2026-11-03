"""Compare every stored column of a US base frame H5 with a release H5.

Rows are matched on each entity's id column, which must be unique; rows
present in only one file are counted. For every column present in both files the receipt records how many rows changed, split by support channel
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


def differs(base: pd.Series, release: pd.Series) -> np.ndarray:
    """Row-wise inequality without a lossy conversion.

    Integer and boolean columns are compared in their own dtype (float64
    cannot tell integers above 2**53 apart); floats are compared exactly with
    NaN equal to NaN; everything else is compared as decoded text.
    """

    if base.dtype.kind in "biu" and release.dtype.kind in "biu":
        return base.to_numpy() != release.to_numpy()
    if base.dtype.kind in "biuf" and release.dtype.kind in "biuf":
        left = base.to_numpy(dtype=np.float64)
        right = release.to_numpy(dtype=np.float64)
        return ~(np.isnan(left) & np.isnan(right)) & (left != right)
    left = base.map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    right = release.map(lambda v: v.decode() if isinstance(v, bytes) else str(v))
    return (left != right).to_numpy()


def numeric_values(values: pd.Series) -> np.ndarray | None:
    """Values for magnitude and direction reporting.

    Integer and boolean columns become Python integers (object dtype), so
    differences, sums and comparisons are exact at any magnitude; floats stay
    float64. Other dtypes return ``None``.
    """

    if values.dtype.kind in "biu":
        return np.array([int(value) for value in values.to_numpy()], dtype=object)
    if values.dtype.kind == "f":
        return values.to_numpy(dtype=np.float64)
    return None


def compare_column(
    base: pd.Series, release: pd.Series, labels: pd.Series
) -> dict[str, object] | None:
    changed = differs(base, release)
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
    base_values = numeric_values(base)
    release_values = numeric_values(release)
    if base_values is None or release_values is None:
        return record
    exact = base_values.dtype == object and release_values.dtype == object
    if not exact:
        # Mixed integer and float columns are compared as float64.
        base_values = base_values.astype(np.float64)
        release_values = release_values.astype(np.float64)
    record["arithmetic"] = "exact integer" if exact else "float64"
    b = base_values[changed]
    r = release_values[changed]
    difference = r - b
    record["max_abs_difference"] = (
        int(max(abs(value) for value in difference))
        if exact
        else float(np.nanmax(np.abs(difference)))
    )
    record["direction"] = {
        "zero_to_nonzero": int(np.count_nonzero((b == 0) & (r != 0))),
        "nonzero_to_zero": int(np.count_nonzero((b != 0) & (r == 0))),
        "up_both_nonzero": int(np.count_nonzero((b != 0) & (r != 0) & (r > b))),
        "down_both_nonzero": int(np.count_nonzero((b != 0) & (r != 0) & (r < b))),
    }
    if exact:
        record["sum_base_changed_rows"] = int(sum(b))
        record["sum_release_changed_rows"] = int(sum(r))
    else:
        record["sum_base_changed_rows"] = float(np.nansum(b))
        record["sum_release_changed_rows"] = float(np.nansum(r))
    nonzero = (b != 0) & (r != 0)
    if nonzero.any():
        # Ratios are reported as floats for both kinds.
        ratios = np.asarray(r[nonzero], dtype=np.float64) / np.asarray(
            b[nonzero], dtype=np.float64
        )
        record["ratio_min_float"] = float(np.min(ratios))
        record["ratio_max_float"] = float(np.max(ratios))
    unchanged = ~changed
    base_nonzero = np.asarray(base_values != 0, dtype=bool)
    record["rows_nonzero_base"] = int(np.count_nonzero(base_nonzero))
    record["rows_nonzero_release"] = int(
        np.count_nonzero(np.asarray(release_values != 0, dtype=bool))
    )
    record["rows_nonzero_base_by_channel"] = {
        str(key): int(value)
        for key, value in pd.Series(base_nonzero, index=labels.index)
        .groupby(labels.to_numpy())
        .sum()
        .items()
    }
    record["rows_unchanged_nonzero"] = int(np.count_nonzero(unchanged & base_nonzero))
    return record


def main() -> None:
    base_path, release_path, out_path = (Path(arg) for arg in sys.argv[1:4])
    receipt: dict[str, object] = {
        "base_h5": Path(base_path).name,
        "base_h5_sha256": sha256(base_path),
        "release_h5": Path(release_path).name,
        "release_h5_sha256": sha256(release_path),
        "entities": {},
    }
    for entity in ENTITIES:
        base = pd.read_hdf(base_path, entity)
        release = pd.read_hdf(release_path, entity)
        id_column = ID_COLUMN[entity]
        rows_base, rows_release = int(len(base)), int(len(release))
        if not base[id_column].is_unique or not release[id_column].is_unique:
            raise ValueError(f"{entity}: {id_column} is not unique in both files")
        base = base.set_index(id_column, drop=False)
        release = release.set_index(id_column, drop=False)
        only_base = sorted(set(base.columns) - set(release.columns))
        only_release = sorted(set(release.columns) - set(base.columns))
        common_ids = base.index.intersection(release.index)
        rows_only_base = int(len(base.index.difference(release.index)))
        rows_only_release = int(len(release.index.difference(base.index)))
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
            "rows_base": rows_base,
            "rows_release": rows_release,
            "rows_matched": int(len(common_ids)),
            "rows_only_in_base": rows_only_base,
            "rows_only_in_release": rows_only_release,
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
