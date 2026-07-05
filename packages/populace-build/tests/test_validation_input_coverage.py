"""US validation-input coverage gate: no validation row scores a structural zero.

Replays the invisible-gap class the gate exists to catch (populace#252/#253):
a validation config scores a provision whose effect is driven by a pure-input
leaf that no source stage produces, so the row validates as a silent zero. The
plant-a-missing-leaf test proves the gate fails loudly in exactly that case; the
registry-consistency test proves the seed entries are real provision input
leaves against the live PolicyEngine-US graph (so the registry cannot rot).
"""

from __future__ import annotations

import importlib.util

import pytest

from populace.build.us_runtime import (
    US_VALIDATION_PROVISION_INPUT_LEAVES,
    ValidationInputLeaf,
    assert_validation_leaf_registry_current,
    us_source_stage_outputs,
    us_validation_input_coverage_gate,
)
from populace.build.us_runtime.validation_input_coverage import (
    us_validation_input_leaf_requirements,
    us_validation_input_leaf_reviewed_exclusions,
)

policyengine_us_installed = importlib.util.find_spec("policyengine_us") is not None
requires_us = pytest.mark.skipif(
    not policyengine_us_installed,
    reason="requires the policyengine-us [us] extra (build environment)",
)


class TestUsSourceStageOutputs:
    def test_reads_shipped_manifest_outputs(self) -> None:
        outputs = us_source_stage_outputs()
        # A representative PUF income leaf declared by the tax-detail stage.
        assert "employment_income_before_lsr" in outputs
        assert "student_loan_interest" in outputs
        # The two invisible-gap leaves are NOT declared by any stage — the
        # whole reason the gate needs their reviewed exclusions.
        assert "qualified_tuition_expenses" not in outputs
        assert "qualified_passenger_vehicle_loan_interest" not in outputs


class TestUsValidationInputCoverageGate:
    def test_shipped_config_passes_with_known_gaps_allowlisted(self) -> None:
        # The shipped state: both critical leaves are missing but each carries a
        # reviewed exclusion with its tracking issue, so the gate passes.
        result = us_validation_input_coverage_gate()
        assert result.passed, result.failures
        assert set(result.details["reviewed_exclusions"]) == {
            "qualified_tuition_expenses",
            "qualified_passenger_vehicle_loan_interest",
        }
        assert result.details["missing"] == []

    def test_planted_missing_leaf_fails_loudly(self) -> None:
        # Plant a NEW validation row whose provision keys on an un-imputed,
        # un-allowlisted input leaf. This is the #252/#253 pattern: the gate
        # must fail and name the variable and the affected row.
        planted = us_validation_input_leaf_requirements()
        planted["some_new_unimputed_input"] = ["obbba_new_untested_provision"]

        from populace.build.gates import source_stage_input_coverage_gate

        result = source_stage_input_coverage_gate(
            planted,
            declared_outputs=us_source_stage_outputs(),
            reviewed_exclusions=us_validation_input_leaf_reviewed_exclusions(),
            name="us_validation_input_coverage",
        )
        assert not result.passed
        assert any("some_new_unimputed_input" in line for line in result.failures)
        assert any("obbba_new_untested_provision" in line for line in result.failures)
        assert result.details["missing"] == ["some_new_unimputed_input"]

    def test_shipped_gate_fails_when_registry_gains_an_unimputed_leaf(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same plant, but through the public gate entry point: appending a
        # registry entry with no reason for a leaf no stage produces must make
        # us_validation_input_coverage_gate() itself fail. This is the guard a
        # future validation row (a new provision keying on an un-imputed leaf)
        # trips in CI instead of shipping a silent zero.
        from populace.build.us_runtime import validation_input_coverage as module

        planted = (
            *module.US_VALIDATION_PROVISION_INPUT_LEAVES,
            ValidationInputLeaf(
                leaf="some_new_unimputed_input",
                provision_variables=("some_new_provision",),
                validation_rows=("obbba_new_untested_provision",),
                # No reason: this leaf is expected to be present, not a tracked gap.
            ),
        )
        monkeypatch.setattr(
            module, "US_VALIDATION_PROVISION_INPUT_LEAVES", planted
        )

        result = us_validation_input_coverage_gate()
        assert not result.passed
        assert result.details["missing"] == ["some_new_unimputed_input"]
        assert any(
            "obbba_new_untested_provision" in line for line in result.failures
        )

    def test_leaf_becoming_a_declared_output_flags_stale_exclusion(self) -> None:
        # If a known-gap leaf is later produced by a stage, its reviewed
        # exclusion is stale and the gate flags it so the register cannot rot.
        result = us_validation_input_coverage_gate(
            source_stage_outputs=[
                *us_source_stage_outputs(),
                "qualified_tuition_expenses",
            ],
        )
        assert not result.passed
        assert any("Stale reviewed exclusions" in line for line in result.failures)
        assert "qualified_tuition_expenses" in result.details["stale_exclusions"]

    def test_removing_the_gap_reason_makes_the_row_fail(self) -> None:
        # A registry entry with no reason (not a tracked gap) that is still
        # unproduced must fail: this is the guard against re-shipping the #252
        # silent zero once the leaf is expected to be present.
        requirements = us_validation_input_leaf_requirements()

        from populace.build.gates import source_stage_input_coverage_gate

        result = source_stage_input_coverage_gate(
            requirements,
            declared_outputs=us_source_stage_outputs(),
            reviewed_exclusions={},  # no leaf allowlisted
            name="us_validation_input_coverage",
        )
        assert not result.passed
        assert set(result.details["missing"]) == {
            "qualified_tuition_expenses",
            "qualified_passenger_vehicle_loan_interest",
        }


class TestValidationInputLeafRegistry:
    def test_every_entry_names_rows_and_provisions(self) -> None:
        assert US_VALIDATION_PROVISION_INPUT_LEAVES
        for entry in US_VALIDATION_PROVISION_INPUT_LEAVES:
            assert entry.leaf
            assert entry.provision_variables
            assert entry.validation_rows

    def test_entry_requires_provision_variables(self) -> None:
        with pytest.raises(ValueError, match="provision_variables is required"):
            ValidationInputLeaf(
                leaf="x", provision_variables=(), validation_rows=("r",)
            )

    def test_entry_requires_validation_rows(self) -> None:
        with pytest.raises(ValueError, match="validation_rows is required"):
            ValidationInputLeaf(
                leaf="x", provision_variables=("v",), validation_rows=()
            )

    def test_registry_leaves_are_provision_inputs_when_configs_are_present(
        self,
    ) -> None:
        # Every registered leaf must be a validation-config row id that actually
        # exists in the shipped configs, so a failure names a real row.
        import json
        from importlib.resources import files

        row_ids: set[str] = set()
        for filename, key in (
            ("obbba_reforms.json", "reforms"),
            ("tax_expenditure_reforms.json", "reforms"),
            ("soi_baseline_levels.json", "levels"),
        ):
            payload = json.loads((files("populace.build.us") / filename).read_text())
            row_ids.update(row["id"] for row in payload.get(key, ()))
        for entry in US_VALIDATION_PROVISION_INPUT_LEAVES:
            for row in entry.validation_rows:
                assert row in row_ids, (
                    f"{entry.leaf} registered under unknown validation row {row!r}"
                )

    @requires_us
    def test_registry_is_current_against_live_engine_graph(self) -> None:
        # The anti-rot check: each registered leaf is a pure input leaf and a
        # dependency of its provision variable per the live PolicyEngine-US
        # graph. Runs only where the [us] extra is installed.
        assert_validation_leaf_registry_current()
