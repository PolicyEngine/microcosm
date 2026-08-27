"""Shared support-clip receipts for UK donor imputation stages."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UKSupportClipReceipt:
    stage: str
    columns: Mapping[str, Mapping[str, object]]

    def evidence(self) -> dict[str, object]:
        return {
            "columns": {
                column: dict(receipt) for column, receipt in self.columns.items()
            }
        }


@dataclass(frozen=True)
class UKSupportClipResult:
    clipped: pd.DataFrame
    receipt: UKSupportClipReceipt


def support_clip_to_donor_with_receipt(
    draws: pd.DataFrame,
    donor: pd.DataFrame,
    *,
    columns: Sequence[str],
    stage: str,
    exempt: set[str] | None = None,
) -> UKSupportClipResult:
    """Clip draws to donor support and return a structured receipt."""

    clipped = draws.copy()
    exempt = exempt or set()
    receipts: dict[str, dict[str, object]] = {}
    for column in columns:
        if column in exempt:
            if column in clipped:
                receipts[column] = {
                    "exempt": True,
                    "rows_considered": int(len(clipped)),
                }
            continue
        if column not in clipped or column not in donor:
            continue
        values = pd.to_numeric(donor[column], errors="coerce")
        finite = values[np.isfinite(values)]
        if finite.empty:
            continue
        lower = float(finite.min())
        upper = float(finite.max())
        draw_values = pd.to_numeric(clipped[column], errors="coerce")
        receipts[column] = {
            "donor_min": lower,
            "donor_max": upper,
            "clipped_low_rows": int((draw_values < lower).sum()),
            "clipped_high_rows": int((draw_values > upper).sum()),
            "rows_considered": int(len(clipped)),
        }
        clipped[column] = clipped[column].clip(lower=lower, upper=upper)
    return UKSupportClipResult(
        clipped=clipped,
        receipt=UKSupportClipReceipt(stage=stage, columns=receipts),
    )
