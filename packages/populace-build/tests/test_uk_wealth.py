"""Contract tests for the UK household-wealth imputation plan.

Mirrors ``test_uk_bus.py`` (plan assembly + donor citations), for the
WAS-anchored wealth holdings — in particular the cash_isa /
stocks_and_shares_isa split and the back-compat fold into corporate_wealth.
"""

from __future__ import annotations

import pytest

from populace.build.uk import (
    UK_WEALTH_DONORS,
    UK_WEALTH_NONNEGATIVE_SOURCE_OUTPUTS,
    UK_WEALTH_SOURCE_MANIFEST,
    UK_WEALTH_STAGE_NAMES,
    uk_wealth_plan,
)

EXPECTED_STAGES = {"household_wealth"}
ISA_OUTPUTS = {"cash_isa", "stocks_and_shares_isa"}


def _noop_implementations() -> dict:
    return {name: (lambda frame: frame) for name in UK_WEALTH_STAGE_NAMES}


class TestUkWealthManifest:
    def test_manifest_is_uk(self) -> None:
        assert UK_WEALTH_SOURCE_MANIFEST.country == "uk"
        assert UK_WEALTH_SOURCE_MANIFEST.version >= 1
        assert set(UK_WEALTH_STAGE_NAMES) == EXPECTED_STAGES

    def test_isa_outputs_present_and_nonnegative(self) -> None:
        outputs = {
            output
            for stage in UK_WEALTH_SOURCE_MANIFEST.stages
            for output in stage.outputs
        }
        assert ISA_OUTPUTS <= outputs
        assert ISA_OUTPUTS <= UK_WEALTH_NONNEGATIVE_SOURCE_OUTPUTS

    def test_stage_imputes_then_clips(self) -> None:
        for stage in UK_WEALTH_SOURCE_MANIFEST.stages:
            kinds = [op.kind for op in stage.operations]
            assert "fit_weighted_qrf" in kinds
            assert "support_clip" in kinds

    def test_investment_isa_folded_into_corporate_wealth(self) -> None:
        # Back-compat: investment ISAs remain part of corporate_wealth.
        folds = [
            op
            for stage in UK_WEALTH_SOURCE_MANIFEST.stages
            for op in stage.operations
            if op.kind == "fold_into"
        ]
        assert any(
            op.parameters.get("output") == "corporate_wealth"
            and op.parameters.get("component") == "stocks_and_shares_isa"
            for op in folds
        )


class TestUkWealthPlan:
    def test_plan_assembles_with_donor_citations(self) -> None:
        plan = uk_wealth_plan(_noop_implementations())
        assert tuple(stage.name for stage in plan.stages) == UK_WEALTH_STAGE_NAMES
        donor_stages = dict(plan.donors())
        assert set(donor_stages) == set(UK_WEALTH_DONORS)
        for spec in donor_stages.values():
            assert spec.source.startswith("https://")

    def test_missing_stage_refuses_to_assemble(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            uk_wealth_plan({})

    def test_unknown_stage_refuses_to_assemble(self) -> None:
        implementations = _noop_implementations()
        implementations["not_a_stage"] = lambda frame: frame
        with pytest.raises(ValueError, match="Unknown stage"):
            uk_wealth_plan(implementations)
