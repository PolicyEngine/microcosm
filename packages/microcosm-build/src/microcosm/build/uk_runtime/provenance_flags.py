"""UK household provenance flag helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from microcosm.frame import Frame

UK_HOUSEHOLD_PROVENANCE_FLAG_PREFIX = "household_is_"

__all__ = [
    "UK_HOUSEHOLD_PROVENANCE_FLAG_PREFIX",
    "discover_household_provenance_flags",
    "household_provenance_flag_summary",
]


def discover_household_provenance_flags(
    columns: Iterable[str],
    *,
    prefix: str = UK_HOUSEHOLD_PROVENANCE_FLAG_PREFIX,
) -> tuple[str, ...]:
    """Return household provenance flags following the UK naming convention."""

    return tuple(sorted(column for column in columns if str(column).startswith(prefix)))


def household_provenance_flag_summary(frame: Frame) -> Mapping[str, object]:
    """Summarize boolean household provenance flags and weighted shares."""

    household = frame.table("household")
    weights = np.asarray(frame.resolve_weights("household").values, dtype=np.float64)
    flags = discover_household_provenance_flags(household.columns)
    summary: dict[str, object] = {}
    for flag in flags:
        values = household[flag]
        if not _is_bool_series(values):
            raise ValueError(f"{flag} must be boolean to summarize provenance shares.")
        boolean = values.to_numpy(dtype=bool)
        summary[flag] = {
            "count": int(boolean.sum()),
            "share": float(boolean.mean()) if len(boolean) else 0.0,
            "weighted_share": float(np.average(boolean.astype(float), weights=weights))
            if len(boolean)
            else 0.0,
        }
    return {"prefix": UK_HOUSEHOLD_PROVENANCE_FLAG_PREFIX, "flags": summary}


def _is_bool_series(values: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(values):
        return True
    nonmissing = values.dropna()
    return bool(
        nonmissing.empty or nonmissing.map(lambda value: isinstance(value, bool)).all()
    )
