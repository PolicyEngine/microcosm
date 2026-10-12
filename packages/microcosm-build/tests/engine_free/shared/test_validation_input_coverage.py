"""Tests split from packages/microcosm-build/tests/test_validation_input_coverage.py."""

# ruff: noqa: F403, F405
from test_support.microcosm_build.validation_input_coverage import *


class TestUsSourceStageOutputs:
    def test_reads_shipped_manifest_outputs(self) -> None:
        outputs = us_source_stage_outputs()
        # A representative PUF income leaf declared by the tax-detail stage.
        assert "employment_income_before_lsr" in outputs
        assert "student_loan_interest" in outputs
        assert "fsla_overtime_premium" in outputs
        # Tuition and qualifying auto-loan interest are declared source-stage
        # outputs and may no longer hide behind exclusions.
        assert "qualified_tuition_expenses" in outputs
        assert "qualified_passenger_vehicle_loan_interest" in outputs
        assert "household_vehicles_owned" in outputs
        assert "household_vehicles_value" in outputs
        assert "traditional_401k_contributions_desired" in outputs
        assert "roth_401k_contributions_desired" in outputs
        assert "traditional_ira_contributions_desired" in outputs
        assert "roth_ira_contributions_desired" in outputs
        assert "self_employed_pension_contributions_desired" in outputs
        assert "takes_up_ssi_if_eligible" in outputs
        assert "takes_up_head_start_if_eligible" in outputs
        assert "weeks_unemployed" in outputs
        assert "casualty_loss" in outputs
        assert "domestic_production_ald" in outputs
        assert "unreimbursed_business_employee_expenses" in outputs
        assert "spm_unit_pre_subsidy_childcare_expenses" in outputs
        assert "spm_unit_energy_subsidy" in outputs
        assert "child_support_received" in outputs
        assert "child_support_expense" in outputs
        assert "disability_benefits" in outputs
        assert "educator_expense" in outputs
        assert "other_health_insurance_premiums" in outputs
        assert "farm_operations_income" in outputs
        assert "farm_rent_income" in outputs
        assert set(US_QBI_OUTPUT_COLUMNS) <= outputs


class TestUsValidationInputCoverageGate:
    def test_shipped_config_passes_without_reviewed_exclusions(self) -> None:
        result = us_validation_input_coverage_gate()
        assert result.passed, result.failures
        assert result.details["reviewed_exclusions"] == {}
        assert result.details["missing"] == []
        requirements = us_validation_input_leaf_requirements()
        assert requirements["tip_income"] == ["obbba_no_tax_on_tips"]
        assert requirements["treasury_tipped_occupation_code"] == [
            "obbba_no_tax_on_tips"
        ]
        assert requirements["fsla_overtime_premium"] == ["obbba_no_tax_on_overtime"]
        assert requirements["qualified_passenger_vehicle_loan_interest"] == [
            "obbba_auto_loan_interest"
        ]
        for leaf in (
            "traditional_401k_contributions_desired",
            "roth_401k_contributions_desired",
            "traditional_ira_contributions_desired",
            "roth_ira_contributions_desired",
            "self_employed_pension_contributions_desired",
        ):
            assert requirements[leaf] == ["soi_savers_credit"]
        assert requirements["casualty_loss"] == ["obbba_casualty_loss_limit"]
        assert requirements["unreimbursed_business_employee_expenses"] == [
            "obbba_misc_itemized_deductions"
        ]
        assert requirements["spm_unit_pre_subsidy_childcare_expenses"] == [
            "obbba_cdcc",
            "soi_cdcc",
            "te_cdcc",
        ]

    def test_planted_missing_leaf_fails_loudly(self) -> None:
        # Plant a NEW validation row whose provision keys on an un-imputed,
        # un-allowlisted input leaf. This is the #252/#253 pattern: the gate
        # must fail and name the variable and the affected row.
        planted = us_validation_input_leaf_requirements()
        planted["some_new_unimputed_input"] = ["obbba_new_untested_provision"]

        from microcosm.build.gates import source_stage_input_coverage_gate

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
        from microcosm.build.us_runtime import validation_input_coverage as module

        planted = (
            *module.US_VALIDATION_PROVISION_INPUT_LEAVES,
            ValidationInputLeaf(
                leaf="some_new_unimputed_input",
                provision_variables=("some_new_provision",),
                validation_rows=("obbba_new_untested_provision",),
                # No reason: this leaf is expected to be present, not a tracked gap.
            ),
        )
        monkeypatch.setattr(module, "US_VALIDATION_PROVISION_INPUT_LEAVES", planted)

        result = us_validation_input_coverage_gate()
        assert not result.passed
        assert result.details["missing"] == ["some_new_unimputed_input"]
        assert any("obbba_new_untested_provision" in line for line in result.failures)

    def test_removing_tuition_from_outputs_makes_the_row_fail(self) -> None:
        requirements = us_validation_input_leaf_requirements()

        from microcosm.build.gates import source_stage_input_coverage_gate

        result = source_stage_input_coverage_gate(
            requirements,
            declared_outputs=us_source_stage_outputs() - {"qualified_tuition_expenses"},
            reviewed_exclusions={},
            name="us_validation_input_coverage",
        )
        assert not result.passed
        assert set(result.details["missing"]) == {"qualified_tuition_expenses"}

    def test_removing_casualty_loss_makes_the_obbba_row_fail(self) -> None:
        requirements = us_validation_input_leaf_requirements()

        from microcosm.build.gates import source_stage_input_coverage_gate

        result = source_stage_input_coverage_gate(
            requirements,
            declared_outputs=us_source_stage_outputs() - {"casualty_loss"},
            reviewed_exclusions={},
            name="us_validation_input_coverage",
        )
        assert not result.passed
        assert set(result.details["missing"]) == {"casualty_loss"}

    def test_removing_misc_expense_makes_the_obbba_row_fail(self) -> None:
        requirements = us_validation_input_leaf_requirements()
        leaf = "unreimbursed_business_employee_expenses"

        from microcosm.build.gates import source_stage_input_coverage_gate

        result = source_stage_input_coverage_gate(
            requirements,
            declared_outputs=us_source_stage_outputs() - {leaf},
            reviewed_exclusions={},
            name="us_validation_input_coverage",
        )
        assert not result.passed
        assert set(result.details["missing"]) == {leaf}

    def test_removing_childcare_makes_cdcc_rows_fail(self) -> None:
        requirements = us_validation_input_leaf_requirements()
        leaf = "spm_unit_pre_subsidy_childcare_expenses"

        from microcosm.build.gates import source_stage_input_coverage_gate

        result = source_stage_input_coverage_gate(
            requirements,
            declared_outputs=us_source_stage_outputs() - {leaf},
            reviewed_exclusions={},
            name="us_validation_input_coverage",
        )
        assert not result.passed
        assert set(result.details["missing"]) == {leaf}
