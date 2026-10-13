"""Recover Schedule E line 32 and its passive subset for the processed PUF donor.

PolicyEngine-US 2.2.1 reads ``partnership_income`` plus ``s_corp_income`` as
Schedule E line 32, total partnership and S corporation income or (loss), and
``passive_partnership_s_corp_income`` as its section 469 passive subset (the
net investment income leg of section 1411(c)(1)(A)(ii)).

The processed 2024 PUF does not carry line 32. Its ``partnership_s_corp_income``
array is the archived pipeline's nonpassive-only sum ``(E25980 - E25960) +
(E26190 - E26180)`` in TY2015 dollars. This module therefore reads line 32 and
the passive columns from the pinned TY2015 source rows and aligns them to the
processed tax units the same way :mod:`puf_source_agi` aligns E00100.

Field meanings come from the IRS SOI General Description Booklet for the 2014
Public Use Tax File, whose annotated Schedule E Part II maps line 28 columns to
fields (the TY2015 file carries the same codes):

====  ==========================================  ===========  ===========
col   Schedule E line 28 column                   partnership  S corp
====  ==========================================  ===========  ===========
(f)   passive loss allowed (Form 8582)            E25920       E26160
(g)   passive income from Schedule K-1            E25940       E26170
(h)   nonpassive loss from Schedule K-1           E25960       E26180
(i)   section 179 expense deduction (Form 4562)   E26110       E26100
(j)   nonpassive income from Schedule K-1         E25980       E26190
====  ==========================================  ===========  ===========

Line 32 (E26270) combines line 30, columns (g) plus (j), with line 31, columns
(f), (h) and (i). The passive subset is (g) minus (f), which already applies
the section 469 limitation because column (f) is the loss *allowed*. Section
179 sits under the form's "Nonpassive Income and Loss" heading, so it belongs
to the nonpassive remainder.

The IRS rounds every PUF amount independently (booklet disclosure step 9), so
E26270 equals the sum of the ten column fields only to within the documented
rounding of each field. :func:`schedule_e_line_32_rounding_bound` encodes that
bound and the alignment refuses any regular record outside it.

The four IRS disclosure aggregate rows were replaced by synthetic records whose
amount columns the archive calibrated one at a time to each row's totals. No
per-record line 32 exists for them, so each aggregate row's line-32 and passive
totals are allocated over its synthetic records in proportion to their
processed nonpassive amounts. That preserves every bucket's published totals
and each record's sign pattern.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.puf_source_agi import (
    PUF_AGGREGATE_RECIDS,
    source_year_puf_layout,
)

PUF_SCHEDULE_E_LINE_32 = "E26270"
PUF_PARTNERSHIP_PASSIVE_INCOME = "E25940"
PUF_PARTNERSHIP_PASSIVE_LOSS_ALLOWED = "E25920"
PUF_PARTNERSHIP_NONPASSIVE_INCOME = "E25980"
PUF_PARTNERSHIP_NONPASSIVE_LOSS = "E25960"
PUF_PARTNERSHIP_SECTION_179 = "E26110"
PUF_S_CORP_PASSIVE_INCOME = "E26170"
PUF_S_CORP_PASSIVE_LOSS_ALLOWED = "E26160"
PUF_S_CORP_NONPASSIVE_INCOME = "E26190"
PUF_S_CORP_NONPASSIVE_LOSS = "E26180"
PUF_S_CORP_SECTION_179 = "E26100"

#: Schedule E line 28 column fields that add into line 30.
PUF_SCHEDULE_E_INCOME_LEGS = (
    PUF_PARTNERSHIP_PASSIVE_INCOME,
    PUF_PARTNERSHIP_NONPASSIVE_INCOME,
    PUF_S_CORP_PASSIVE_INCOME,
    PUF_S_CORP_NONPASSIVE_INCOME,
)
#: Schedule E line 28 column fields that add into line 31 (subtracted).
PUF_SCHEDULE_E_LOSS_LEGS = (
    PUF_PARTNERSHIP_PASSIVE_LOSS_ALLOWED,
    PUF_PARTNERSHIP_NONPASSIVE_LOSS,
    PUF_PARTNERSHIP_SECTION_179,
    PUF_S_CORP_PASSIVE_LOSS_ALLOWED,
    PUF_S_CORP_NONPASSIVE_LOSS,
    PUF_S_CORP_SECTION_179,
)
PUF_SCHEDULE_E_PART_II_LEGS = (
    *PUF_SCHEDULE_E_INCOME_LEGS,
    *PUF_SCHEDULE_E_LOSS_LEGS,
)
PUF_SOURCE_YEAR_SCHEDULE_E_REQUIRED_COLUMNS = (
    "RECID",
    "MARS",
    "S006",
    PUF_SCHEDULE_E_LINE_32,
    *PUF_SCHEDULE_E_PART_II_LEGS,
)

#: Donor output names this module supplies, at tax-unit grain.
SCHEDULE_E_LINE_32_OUTPUT = "partnership_income"
SCHEDULE_E_PASSIVE_OUTPUT = "passive_partnership_s_corp_income"


def schedule_e_line_32_rounding_bound(values: Any) -> np.ndarray:
    """Return the largest rounding error the PUF allows on each stored amount.

    The 2014 booklet's disclosure step 9 rounds amounts above $100,000 in
    absolute value to four significant digits, amounts from $10,000 to
    $100,000 to the nearest $100, amounts from $5 to $10,000 to the nearest
    $10, and sets nonzero amounts below $5 to $2. The bound is half of the
    stored amount's rounding unit (3 for the $2 floor). A stored amount of
    exactly $10,000 or $100,000 can be the coarser band's rounding of a value
    just below it, so those two values take the coarser half-unit.
    """

    magnitude = np.abs(np.asarray(values, dtype=np.float64))
    bound = np.zeros_like(magnitude)
    above = magnitude > 100_000.0
    bound[above] = 0.5 * 10.0 ** (np.floor(np.log10(magnitude[above])) - 3.0)
    bound[(magnitude > 10_000.0) & ~above] = 50.0
    bound[(magnitude >= 5.0) & (magnitude <= 10_000.0)] = 5.0
    bound[(magnitude > 0.0) & (magnitude < 5.0)] = 3.0
    bound[(magnitude == 10_000.0) | (magnitude == 100_000.0)] = 50.0
    return bound


def schedule_e_partnership_s_corp_columns(
    frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Return line 32, its passive subset and nonpassive remainder per record.

    Args:
        frame: Source rows carrying E26270 and the ten line 28 column fields.

    Returns:
        ``line_32`` (E26270 as stored), ``passive`` ((g) minus (f) for both
        entity types), ``nonpassive`` ((j) minus (h) minus (i) for both entity
        types, which is what the processed artifact's archived column omits
        only by (i)), ``archived_nonpassive`` (the processed artifact's
        ``(E25980 - E25960) + (E26190 - E26180)``) and ``rounding_bound``.

    Raises:
        ValueError: If a field is missing or nonfinite, a line 28 column field
            is negative, or E26270 differs from the column fields by more than
            the booklet's per-field rounding.
    """

    missing = [
        column
        for column in (PUF_SCHEDULE_E_LINE_32, *PUF_SCHEDULE_E_PART_II_LEGS)
        if column not in frame
    ]
    if missing:
        raise ValueError(f"TY2015 PUF source is missing column(s): {missing}.")
    values = {
        column: pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        for column in (PUF_SCHEDULE_E_LINE_32, *PUF_SCHEDULE_E_PART_II_LEGS)
    }
    for column, array in values.items():
        if not np.isfinite(array).all():
            raise ValueError(f"TY2015 PUF {column} must be finite.")
    negative_legs = [
        column for column in PUF_SCHEDULE_E_PART_II_LEGS if (values[column] < 0).any()
    ]
    if negative_legs:
        raise ValueError(
            "TY2015 PUF Schedule E line 28 column fields are stored as "
            f"nonnegative amounts; got negative values in {negative_legs}."
        )
    passive = (
        values[PUF_PARTNERSHIP_PASSIVE_INCOME]
        - values[PUF_PARTNERSHIP_PASSIVE_LOSS_ALLOWED]
        + values[PUF_S_CORP_PASSIVE_INCOME]
        - values[PUF_S_CORP_PASSIVE_LOSS_ALLOWED]
    )
    archived_nonpassive = (
        values[PUF_PARTNERSHIP_NONPASSIVE_INCOME]
        - values[PUF_PARTNERSHIP_NONPASSIVE_LOSS]
        + values[PUF_S_CORP_NONPASSIVE_INCOME]
        - values[PUF_S_CORP_NONPASSIVE_LOSS]
    )
    nonpassive = (
        archived_nonpassive
        - values[PUF_PARTNERSHIP_SECTION_179]
        - values[PUF_S_CORP_SECTION_179]
    )
    line_32 = values[PUF_SCHEDULE_E_LINE_32]
    rounding_bound = schedule_e_line_32_rounding_bound(line_32)
    for column in PUF_SCHEDULE_E_PART_II_LEGS:
        rounding_bound = rounding_bound + schedule_e_line_32_rounding_bound(
            values[column]
        )
    outside = np.abs(line_32 - (passive + nonpassive)) > rounding_bound
    if outside.any():
        raise ValueError(
            "TY2015 PUF E26270 differs from its Schedule E line 28 column "
            "fields by more than the booklet's per-field rounding on "
            f"{int(outside.sum())} record(s)."
        )
    return {
        "line_32": line_32,
        "passive": passive,
        "nonpassive": nonpassive,
        "archived_nonpassive": archived_nonpassive,
        "rounding_bound": rounding_bound,
    }


def _schedule_e_source_frame(
    source: str | Path | pd.DataFrame | Mapping[str, Sequence[Any]],
) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        frame = pd.read_csv(
            Path(source),
            usecols=lambda name: name in PUF_SOURCE_YEAR_SCHEDULE_E_REQUIRED_COLUMNS,
        )
    else:
        frame = pd.DataFrame(source).copy()
    missing = [
        column
        for column in PUF_SOURCE_YEAR_SCHEDULE_E_REQUIRED_COLUMNS
        if column not in frame
    ]
    if missing:
        raise ValueError(f"TY2015 PUF source is missing column(s): {missing}.")
    frame = frame.loc[:, PUF_SOURCE_YEAR_SCHEDULE_E_REQUIRED_COLUMNS].copy()
    for column in PUF_SOURCE_YEAR_SCHEDULE_E_REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["RECID", "MARS", "S006"]].isna().any().any():
        raise ValueError("TY2015 PUF RECID, MARS, and S006 must be numeric.")
    return frame


def _processed_tax_unit_nonpassive(
    values: Sequence[Any],
    *,
    label: str,
    length: int,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != length or not np.isfinite(array).all():
        raise ValueError(
            f"{label} must be finite and aligned one-for-one with tax_unit_id."
        )
    return array


def processed_puf_tax_unit_partnership_s_corp_income(
    arrays: Mapping[str, Sequence[Any]],
) -> np.ndarray:
    """Sum the processed artifact's person-grain nonpassive column to tax units.

    Returns the archived ``partnership_s_corp_income`` per processed
    ``tax_unit_id``, in stored tax-unit order, for the alignment check in
    :func:`source_year_puf_partnership_s_corp_income`.
    """

    for column in ("tax_unit_id", "person_tax_unit_id", "partnership_s_corp_income"):
        if column not in arrays:
            raise ValueError(f"Processed PUF arrays are missing {column!r}.")
    person_tax_unit_id = np.asarray(arrays["person_tax_unit_id"]).astype(np.int64)
    values = np.asarray(arrays["partnership_s_corp_income"], dtype=np.float64)
    if values.shape != person_tax_unit_id.shape:
        raise ValueError(
            "Processed PUF partnership_s_corp_income must align with "
            "person_tax_unit_id."
        )
    tax_unit_id = np.asarray(arrays["tax_unit_id"]).astype(np.int64)
    totals = pd.Series(values).groupby(person_tax_unit_id, sort=False).sum()
    missing = ~pd.Index(tax_unit_id).isin(totals.index)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} processed PUF tax unit(s) have no person rows."
        )
    return totals.reindex(tax_unit_id).to_numpy(dtype=np.float64)


def source_year_puf_partnership_s_corp_income(
    source: str | Path | pd.DataFrame | Mapping[str, Sequence[Any]],
    *,
    processed_tax_unit_ids: Sequence[Any],
    processed_tax_unit_weights: Sequence[Any],
    processed_nonpassive_partnership_s_corp_income: Sequence[Any],
    receipt: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    """Return Schedule E line 32 and its passive subset per processed tax unit.

    Args:
        source: The pinned TY2015 PUF CSV, or a table with the same columns.
        processed_tax_unit_ids: Processed PUF ``tax_unit_id`` in stored order.
        processed_tax_unit_weights: Processed PUF tax-unit weights.
        processed_nonpassive_partnership_s_corp_income: The processed
            artifact's ``partnership_s_corp_income`` summed to tax units. It
            must equal the archived nonpassive sum on every regular record,
            which proves the RECID alignment and that the processed column
            still has the archived meaning and dollar basis.
        receipt: Optional mutable sink for weighted totals and the synthetic
            allocation ratios.

    Returns:
        ``partnership_income`` (Schedule E line 32) and
        ``passive_partnership_s_corp_income`` (its passive subset), both in
        TY2015 dollars, the dollar basis of the processed column they replace.

    Raises:
        ValueError: If the layout does not align, the processed column departs
            from the archived nonpassive sum, a record breaks the line-32
            identity beyond rounding, or a synthetic bucket cannot carry its
            aggregate row's totals.
    """

    frame = _schedule_e_source_frame(source)
    layout = source_year_puf_layout(
        frame,
        processed_tax_unit_ids=processed_tax_unit_ids,
        processed_tax_unit_weights=processed_tax_unit_weights,
    )
    weights = layout.processed_weights
    processed_nonpassive = _processed_tax_unit_nonpassive(
        processed_nonpassive_partnership_s_corp_income,
        label="Processed PUF partnership_s_corp_income",
        length=len(weights),
    )
    regular_count = len(layout.regular)
    regular_columns = schedule_e_partnership_s_corp_columns(layout.regular)
    regular_processed = processed_nonpassive[:regular_count]
    # The processed artifact stores person-grain float values that sum to the
    # tax unit; allow only float accumulation error, never a dollar.
    tolerance = 1e-9 * np.maximum(1.0, np.abs(regular_columns["archived_nonpassive"]))
    drift = np.abs(regular_processed - regular_columns["archived_nonpassive"])
    if (drift > tolerance).any():
        raise ValueError(
            "Processed PUF partnership_s_corp_income no longer equals the "
            "archived nonpassive sum (E25980 - E25960) + (E26190 - E26180) on "
            f"{int((drift > tolerance).sum())} regular record(s); the donor "
            "cannot replace it with TY2015 Schedule E line 32."
        )

    line_32 = np.empty(len(weights), dtype=np.float64)
    passive = np.empty(len(weights), dtype=np.float64)
    line_32[:regular_count] = regular_columns["line_32"]
    passive[:regular_count] = regular_columns["passive"]

    aggregate_columns = schedule_e_partnership_s_corp_columns(
        layout.aggregate.loc[list(PUF_AGGREGATE_RECIDS)]
    )
    synthetic_receipt: dict[str, dict[str, float]] = {}
    for index, recid in enumerate(PUF_AGGREGATE_RECIDS):
        rows = layout.synthetic_rows[recid]
        population = layout.aggregate_population[recid]
        bucket_weights = weights[rows]
        if not np.isclose(bucket_weights.sum(), round(population), rtol=0.0, atol=1e-9):
            raise ValueError(
                f"Processed PUF synthetic weights for aggregate RECID {recid} "
                "do not preserve the aggregate-row population."
            )
        per_return_nonpassive = float(aggregate_columns["archived_nonpassive"][index])
        per_return_line_32 = float(aggregate_columns["line_32"][index])
        per_return_passive = float(aggregate_columns["passive"][index])
        bucket_nonpassive = processed_nonpassive[rows]
        processed_total = float(bucket_weights @ bucket_nonpassive)
        expected_total = population * per_return_nonpassive
        if per_return_nonpassive == 0.0 or not np.isclose(
            processed_total,
            expected_total,
            rtol=1e-9,
            atol=1e-3,
        ):
            raise ValueError(
                f"Processed PUF synthetic records for aggregate RECID {recid} "
                "do not carry the aggregate row's nonpassive total, so its "
                "line-32 and passive totals cannot be allocated over them."
            )
        line_32_ratio = per_return_line_32 / per_return_nonpassive
        passive_ratio = per_return_passive / per_return_nonpassive
        line_32[rows] = bucket_nonpassive * line_32_ratio
        passive[rows] = bucket_nonpassive * passive_ratio
        synthetic_receipt[str(recid)] = {
            "records": float(rows.stop - rows.start),
            "population": population,
            "line_32_per_return": per_return_line_32,
            "passive_per_return": per_return_passive,
            "archived_nonpassive_per_return": per_return_nonpassive,
            "line_32_to_nonpassive_ratio": line_32_ratio,
            "passive_to_nonpassive_ratio": passive_ratio,
        }

    if not (np.isfinite(line_32).all() and np.isfinite(passive).all()):
        raise ValueError("Recovered TY2015 Schedule E values must be finite.")
    if receipt is not None:
        receipt.update(
            {
                "source": "TY2015 PUF E26270 and Schedule E line 28 columns",
                "dollar_basis": "TY2015 (the processed column's basis)",
                "weighted_line_32": float(weights @ line_32),
                "weighted_passive": float(weights @ passive),
                "weighted_processed_nonpassive": float(weights @ processed_nonpassive),
                "weighted_line_32_positive": float(weights @ np.maximum(line_32, 0)),
                "weighted_line_32_negative": float(weights @ np.minimum(line_32, 0)),
                "weighted_passive_positive": float(weights @ np.maximum(passive, 0)),
                "weighted_passive_negative": float(weights @ np.minimum(passive, 0)),
                "weighted_line_32_nonzero_count": float(weights[line_32 != 0].sum()),
                "weighted_processed_nonpassive_nonzero_count": float(
                    weights[processed_nonpassive != 0].sum()
                ),
                "synthetic_buckets": synthetic_receipt,
            }
        )
    return {
        SCHEDULE_E_LINE_32_OUTPUT: line_32,
        SCHEDULE_E_PASSIVE_OUTPUT: passive,
    }


__all__ = [
    "PUF_SCHEDULE_E_INCOME_LEGS",
    "PUF_SCHEDULE_E_LINE_32",
    "PUF_SCHEDULE_E_LOSS_LEGS",
    "PUF_SCHEDULE_E_PART_II_LEGS",
    "PUF_SOURCE_YEAR_SCHEDULE_E_REQUIRED_COLUMNS",
    "SCHEDULE_E_LINE_32_OUTPUT",
    "SCHEDULE_E_PASSIVE_OUTPUT",
    "schedule_e_line_32_rounding_bound",
    "processed_puf_tax_unit_partnership_s_corp_income",
    "schedule_e_partnership_s_corp_columns",
    "source_year_puf_partnership_s_corp_income",
]
