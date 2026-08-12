"""UK local-geography target-alignment primitives.

This module owns the area-target alignment contract shared by UK local
builds: a target provider hands Microcosm an explicit area table (HMRC, ONS,
DWP, or other public target files), and :func:`align_area_targets` returns
the metric columns aligned to a canonical area-code order. The code is
deliberately independent of the incumbent UK data package.

The stacked ``areas x households`` matrix representation that used to live
here (the pre-ladder research harness) was removed with microcosm#612
increment 2; the rowwise clone path in
:mod:`microcosm.build.uk_runtime.local_rowwise` is the single UK local-solve
story.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

_AREA_METADATA_COLUMNS = frozenset(
    {
        "area_code",
        "area_index",
        "area_name",
        "area_type",
        "code",
        "country",
        "name",
    }
)


def align_area_targets(
    targets: pd.DataFrame,
    area_codes: Sequence[str],
    *,
    metric_names: Sequence[str] | None = None,
    code_column: str = "code",
) -> pd.DataFrame:
    """Align target rows to ``area_codes`` and return metric columns only."""

    codes = _area_code_tuple(area_codes)
    frame = targets.copy()
    if code_column in frame.columns:
        frame[code_column] = frame[code_column].astype(str)
        frame = frame.set_index(code_column, drop=True)
    else:
        frame.index = frame.index.astype(str)
    if frame.index.has_duplicates:
        duplicates = frame.index[frame.index.duplicated()].unique()
        raise ValueError(
            f"target area code index must be unique; duplicate value(s): "
            f"{list(map(str, duplicates[:5]))}."
        )

    missing = [code for code in codes if code not in frame.index]
    if missing:
        raise ValueError(f"target frame is missing area code(s): {missing[:5]}.")

    if metric_names is None:
        metric_names = [
            col
            for col in frame.columns
            if col not in _AREA_METADATA_COLUMNS
            and pd.api.types.is_numeric_dtype(frame[col])
        ]
    metrics = tuple(str(name) for name in metric_names)
    absent = [name for name in metrics if name not in frame.columns]
    if absent:
        raise ValueError(f"target frame is missing metric column(s): {absent}.")

    aligned = frame.loc[list(codes), list(metrics)].astype(float)
    values = aligned.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("target values must all be finite.")
    return aligned


def _area_code_tuple(area_codes: Sequence[str]) -> tuple[str, ...]:
    codes = tuple(str(code) for code in area_codes)
    if not codes:
        raise ValueError("area_codes must not be empty.")
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    if duplicates:
        raise ValueError(f"area_codes must be unique; duplicates: {duplicates}.")
    return codes
