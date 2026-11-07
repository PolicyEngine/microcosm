"""US alias for the shared weighted per-column input-mass helper.

The implementation was promoted to :mod:`microcosm.build.input_mass` when the
UK terminal battery adopted the same #278 input-mass parity gate (#609); the
computation is schema-driven and never carried US-specific logic. This module
remains so existing US call sites keep their import path, and it holds the one
US comparison rule the shared gate can't know: how to read Schedule E line 32
against a reference built before the entity split.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult, input_mass_parity_gate
from microcosm.build.input_mass import input_mass_totals
from microcosm.frame import Frame

__all__ = [
    "US_LINE_32_COLUMNS",
    "UsLine32Carriers",
    "us_input_mass_parity_gate",
    "us_input_mass_totals",
    "us_line_32_carriers",
    "us_line_32_counts_both",
    "us_line_32_on_reference_terms",
]

us_input_mass_totals = input_mass_totals

#: The two columns microcosm#1261 split Schedule E line 32 into.
US_LINE_32_COLUMNS: tuple[str, str] = ("partnership_income", "s_corp_income")


@dataclass(frozen=True)
class UsLine32Carriers:
    """Which line 32 columns have records that hold an amount.

    A weighted total can't tell records that cancel from no records at all, so
    the line 32 rule reads these instead. :func:`us_line_32_carriers` reads them
    from the frames whose totals the gate compares.

    Attributes:
        candidate_partnership_income: A candidate record with nonzero weight
            holds a nonzero ``partnership_income``.
        reference_s_corp_income: A reference record, at any weight, holds a
            nonzero ``s_corp_income``.
    """

    candidate_partnership_income: bool
    reference_s_corp_income: bool


def _carries(frame: Frame, column: str, *, weighted: bool) -> bool:
    """Whether a record of ``frame`` holds a nonzero ``column``.

    Values convert as :func:`microcosm.build.input_mass.input_mass_totals`
    converts them, so a column whose total differs from zero always has a
    carrier. With ``weighted``, only a record under a nonzero weight counts.
    """
    for entity in frame.entities:
        table = frame.table(entity)
        if column not in table.columns:
            continue
        values = table[column]
        if pd.api.types.is_bool_dtype(values):
            numeric = values.fillna(False).to_numpy(dtype=np.float64)
        elif pd.api.types.is_numeric_dtype(values):
            numeric = pd.to_numeric(values, errors="coerce")
            numeric = numeric.fillna(0.0).to_numpy(dtype=np.float64)
        else:
            return False
        carriers = numeric != 0.0
        if weighted:
            weights = np.asarray(frame.resolve_weights(entity).values, dtype=np.float64)
            carriers &= weights != 0.0
        return bool(carriers.any())
    return False


def us_line_32_carriers(candidate: Frame, reference: Frame) -> UsLine32Carriers:
    """Read which line 32 columns have carriers in a candidate and its reference.

    The candidate's ``partnership_income`` counts only records with nonzero
    weight, because the gate guards the mass the weighted build keeps. The
    reference's ``s_corp_income`` counts every record, because one that holds an
    amount shows the build split the line, whatever its weight.
    """
    return UsLine32Carriers(
        candidate_partnership_income=_carries(
            candidate, "partnership_income", weighted=True
        ),
        reference_s_corp_income=_carries(reference, "s_corp_income", weighted=False),
    )


def _nonzero(total: float) -> bool:
    return bool(np.isfinite(total)) and total != 0.0


def us_line_32_counts_both(
    candidate_totals: Mapping[str, float],
    reference_totals: Mapping[str, float],
    carriers: UsLine32Carriers,
) -> bool:
    """Whether the candidate's ``partnership_income`` counts both line 32 columns.

    It does when the candidate measures ``partnership_income`` and a record with
    nonzero weight carries it, and no reference record carries ``s_corp_income``.

    Raises:
        ValueError: If a nonzero total has no carrier. A record carries every
            nonzero total, so the carriers then came from frames other than the
            ones the totals came from.
    """
    partnership = candidate_totals.get("partnership_income", 0.0)
    if _nonzero(partnership) and not carriers.candidate_partnership_income:
        raise ValueError(
            f"partnership_income: the candidate totals {partnership:.6g} but no "
            "candidate record with nonzero weight carries it; read the carriers "
            "from the frames the totals came from."
        )
    s_corp = reference_totals.get("s_corp_income", 0.0)
    if _nonzero(s_corp) and not carriers.reference_s_corp_income:
        raise ValueError(
            f"s_corp_income: the reference totals {s_corp:.6g} but no reference "
            "record carries it; read the carriers from the frames the totals "
            "came from."
        )
    return (
        "partnership_income" in candidate_totals
        and carriers.candidate_partnership_income
        and not carriers.reference_s_corp_income
    )


def us_line_32_on_reference_terms(
    candidate_totals: Mapping[str, float],
    reference_totals: Mapping[str, float],
    carriers: UsLine32Carriers,
) -> dict[str, float]:
    """Return candidate totals that read Schedule E line 32 as the reference does.

    A reference built before microcosm#1261 holds partnership and S corporation
    income together in ``partnership_income`` and no ``s_corp_income`` record.
    One built before microcosm#1251 holds only the nonpassive part of that
    income. A candidate built after #1261 splits the income between the two
    columns, so its ``partnership_income`` alone can fall short of that
    reference however well the build keeps the income. Against a reference with
    no record that carries ``s_corp_income``, or no such column, the candidate's
    ``partnership_income`` counts both columns.

    Any reference record with an ``s_corp_income`` amount means the reference
    carries the split, even when those amounts cancel, and the gate compares
    each column alone. So the rule stops once a reference carries the split, and
    a frame compared with itself never drifts.

    A candidate with no ``partnership_income`` column, or none on a record with
    nonzero weight, stays as it is, and the input-mass gate refuses it. S
    corporation income can't stand in for a lost column, even when it alone
    lands inside the band. Records that carry partnership income count however
    their amounts net, so when they cancel, the gate compares the combined line.

    Against a reference without the split, the gate reads only the total, so it
    can't catch a wrong division of the income between the two columns.
    """
    totals = dict(candidate_totals)
    if us_line_32_counts_both(candidate_totals, reference_totals, carriers):
        totals["partnership_income"] += totals.get("s_corp_income", 0.0)
    return totals


def us_input_mass_parity_gate(
    candidate_totals: Mapping[str, float],
    reference_totals: Mapping[str, float],
    *,
    carriers: UsLine32Carriers,
    **gate_terms: Any,
) -> GateResult:
    """Run the input-mass gate with Schedule E line 32 on the reference's terms.

    When the rule counts both columns, the gate's details say so, because the
    mass and drift under ``partnership_income`` then measure both.
    """
    gate = input_mass_parity_gate(
        us_line_32_on_reference_terms(candidate_totals, reference_totals, carriers),
        reference_totals,
        **gate_terms,
    )
    if not us_line_32_counts_both(candidate_totals, reference_totals, carriers):
        return gate
    return replace(
        gate,
        details={
            **gate.details,
            "partnership_income_counts": list(US_LINE_32_COLUMNS),
        },
    )
