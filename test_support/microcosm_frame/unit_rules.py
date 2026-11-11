"""Hypothesis strategies for benefit-unit rules, shared across test groups."""

# ruff: noqa: F401

from __future__ import annotations

from hypothesis import strategies as st

from microcosm.frame.unit_construction import (
    BenefitUnitRule,
    DependentChildRule,
    FinancialIndependenceTest,
    SplitParentsPolicy,
    UnparentedChildPlacement,
)


@st.composite
def unit_rules(draw, *, refusing: bool = False) -> BenefitUnitRule:
    """A unit rule; non-refusing unless ``refusing``, so any valid frame builds."""

    max_age = draw(st.integers(0, 30))
    test = None
    if draw(st.booleans()):
        test = FinancialIndependenceTest(
            min_age=draw(st.integers(0, max_age)),
            min_usual_weekly_hours=draw(
                st.floats(min_value=0.5, max_value=168.0, allow_nan=False)
            ),
        )
    placements = list(UnparentedChildPlacement)
    splits = list(SplitParentsPolicy)
    if not refusing:
        placements.remove(UnparentedChildPlacement.REFUSE)
        splits.remove(SplitParentsPolicy.REFUSE)
    return BenefitUnitRule(
        entity=draw(st.sampled_from(["family", "benefit_unit"])),
        dependent_child=DependentChildRule(
            max_age=max_age, financial_independence=test
        ),
        unparented_child=draw(st.sampled_from(placements)),
        split_parents=draw(st.sampled_from(splits)),
        provenance={"max_age": "generated"},
    )


__all__ = [name for name in globals() if not name.startswith("__")]
