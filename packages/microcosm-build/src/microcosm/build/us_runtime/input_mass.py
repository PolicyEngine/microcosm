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

from microcosm.build.input_mass import input_mass_totals

__all__ = ["us_input_mass_totals", "us_line_32_on_reference_terms"]

us_input_mass_totals = input_mass_totals


def us_line_32_on_reference_terms(
    candidate_totals: Mapping[str, float],
    reference_totals: Mapping[str, float],
    *,
    minimum_reference_total: float,
) -> dict[str, float]:
    """Return candidate totals that read Schedule E line 32 as the reference does.

    A reference built before microcosm#1261 holds all of line 32 in
    ``partnership_income`` and carries no ``s_corp_income`` mass. A candidate
    built after it splits the same income between the two columns, so its
    ``partnership_income`` alone falls short of that reference however well the
    build keeps line 32. Against such a reference, the candidate's
    ``partnership_income`` counts both columns.

    A reference with ``s_corp_income`` mass above ``minimum_reference_total``
    gets each column compared alone, so the rule stops applying once the
    reference carries the split. A candidate without ``partnership_income``
    stays without it, and the input-mass gate refuses the absent column.
    """
    totals = dict(candidate_totals)
    if (
        "partnership_income" in totals
        and abs(reference_totals.get("s_corp_income", 0.0)) <= minimum_reference_total
    ):
        totals["partnership_income"] += totals.get("s_corp_income", 0.0)
    return totals
