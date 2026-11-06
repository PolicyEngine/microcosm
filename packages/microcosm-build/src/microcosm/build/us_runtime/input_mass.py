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
from dataclasses import replace
from typing import Any

from microcosm.build.gates import GateResult, input_mass_parity_gate
from microcosm.build.input_mass import input_mass_totals

__all__ = [
    "us_input_mass_parity_gate",
    "us_input_mass_totals",
    "us_line_32_on_reference_terms",
]

us_input_mass_totals = input_mass_totals


def us_line_32_on_reference_terms(
    candidate_totals: Mapping[str, float],
    reference_totals: Mapping[str, float],
) -> dict[str, float]:
    """Return candidate totals that read Schedule E line 32 as the reference does.

    A reference built before microcosm#1261 holds partnership and S corporation
    income together in ``partnership_income`` and zero in ``s_corp_income``. One
    built before microcosm#1251 holds only the nonpassive part of that income. A
    candidate built after #1261 splits the income between the two columns, so its
    ``partnership_income`` alone can fall short of that reference however well the
    build keeps the income. Against a reference with a zero ``s_corp_income``
    total, or no such column, the candidate's ``partnership_income`` counts both
    columns.

    A reference with a nonzero ``s_corp_income`` total gets each column compared
    alone, so the rule stops applying once the reference carries the split, and a
    frame compared with itself never drifts.

    A candidate with no ``partnership_income`` column or mass stays as it is, and
    the input-mass gate refuses it. S corporation income can't stand in for a
    zeroed column, even when it alone lands inside the band.

    Against a reference without the split, the gate reads only the total, so it
    can't catch a wrong division of the income between the two columns.
    """
    totals = dict(candidate_totals)
    if (
        totals.get("partnership_income", 0.0) != 0.0
        and reference_totals.get("s_corp_income", 0.0) == 0.0
    ):
        totals["partnership_income"] += totals.get("s_corp_income", 0.0)
    return totals


def us_input_mass_parity_gate(
    candidate_totals: Mapping[str, float],
    reference_totals: Mapping[str, float],
    **gate_terms: Any,
) -> GateResult:
    """Run the input-mass gate with Schedule E line 32 on the reference's terms.

    When the rule moves ``partnership_income``, the gate's details say so, because
    the mass and drift under that name then measure both columns.
    """
    totals = us_line_32_on_reference_terms(candidate_totals, reference_totals)
    gate = input_mass_parity_gate(totals, reference_totals, **gate_terms)
    if totals == dict(candidate_totals):
        return gate
    return replace(
        gate,
        details={
            **gate.details,
            "partnership_income_counts": ["partnership_income", "s_corp_income"],
        },
    )
