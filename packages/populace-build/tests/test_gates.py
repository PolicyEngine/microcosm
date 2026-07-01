"""The acceptance gates: each failure mode named, each pass earned.

The aggregate-vs-admin tests replay the two real incidents (net-STCG sign
flip, investment-interest blow-up) as the cases the gate must catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from populace.build import (
    FitWeightRecord,
    GateReport,
    GateResult,
    TargetCoverageRequirement,
    aggregate_admin_gate,
    default_valued_columns_gate,
    enum_domain_gate,
    export_surface_gate,
    formula_owned_export_gate,
    input_mass_parity_gate,
    macro_realism_gate,
    nonconstant_columns_gate,
    nonnegative_columns_gate,
    parity_gate,
    per_family_fit_gate,
    relative_error_loss,
    source_coverage_gate,
    source_stage_input_coverage_gate,
    support_gate,
    target_profile_coverage_gate,
    target_surface_gate,
    weights_audit_gate,
)
from populace.calibrate import TargetSpec


class TestGateResultInvariants:
    def test_pass_with_failures_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot pass with failures"):
            GateResult(name="x", passed=True, failures=("oops",))

    def test_fail_without_reason_is_refused(self) -> None:
        with pytest.raises(ValueError, match="cannot fail without naming"):
            GateResult(name="x", passed=False)

    def test_report_aggregates(self) -> None:
        report = GateReport(
            (
                GateResult(name="a", passed=True),
                GateResult(name="b", passed=False, failures=("broke",)),
            )
        )
        assert not report.passed
        assert report.failures == ("[b] broke",)
        manifest = report.to_manifest()
        assert manifest["passed"] is False
        assert manifest["gates"]["b"]["failures"] == ["broke"]


class TestParityGate:
    def test_gap_fails_with_the_variable_named(self) -> None:
        result = parity_gate(
            {"snap": 0.1, "net_worth": 0.0},
            {"snap": 0.12, "net_worth": 0.9},
        )
        assert not result.passed
        assert "net_worth" in result.failures[0]

    def test_full_parity_passes(self) -> None:
        result = parity_gate(
            {"snap": 0.1, "net_worth": 0.5},
            {"snap": 0.12, "net_worth": 0.9},
        )
        assert result.passed
        assert result.details["gaps"] == 0

    def test_candidate_extras_are_not_failures(self) -> None:
        result = parity_gate({"snap": 0.1, "tips": 0.004}, {"snap": 0.12})
        assert result.passed

    def test_known_gap_is_exempt_and_recorded(self) -> None:
        result = parity_gate(
            {"snap": 0.1}, {"snap": 0.1, "rare_var": 0.01}, known_gaps=["rare_var"]
        )
        assert result.passed
        assert result.details["exempted"] == ["rare_var"]
        assert result.details["stale_exemptions"] == []
        assert result.details["dormant_exemptions"] == []

    def test_stale_exemption_fails_once_the_candidate_produces_the_layer(self) -> None:
        # The takes_up_medicaid_if_eligible rot case: the exemption predates
        # the candidate producing the layer, and must fail the gate the build
        # after production starts.
        result = parity_gate(
            {"snap": 0.1, "takes_up_medicaid_if_eligible": 0.8},
            {"snap": 0.1, "takes_up_medicaid_if_eligible": 0.85},
            known_gaps=["takes_up_medicaid_if_eligible"],
        )
        assert not result.passed
        assert result.details["stale_exemptions"] == ["takes_up_medicaid_if_eligible"]
        assert "takes_up_medicaid_if_eligible" in result.failures[0]
        assert "remove the exemption" in result.failures[0]

    def test_stale_exemption_fails_even_when_the_reference_lacks_the_layer(
        self,
    ) -> None:
        # Candidate-populated is what makes an exemption a lie; a reference
        # that never carried the layer does not rescue it.
        result = parity_gate(
            {"snap": 0.1, "new_var": 0.3},
            {"snap": 0.1},
            known_gaps=["new_var"],
        )
        assert not result.passed
        assert result.details["stale_exemptions"] == ["new_var"]

    def test_dormant_exemption_is_reported_not_failed(self) -> None:
        # An exemption for a layer this reference never populates is dormant
        # (different reference vintages populate different layers).
        result = parity_gate(
            {"snap": 0.1},
            {"snap": 0.1, "dropped_var": 0.0},
            known_gaps=["dropped_var", "absent_var"],
        )
        assert result.passed
        assert result.details["dormant_exemptions"] == ["absent_var", "dropped_var"]

    def test_stale_exemption_does_not_mask_real_gaps(self) -> None:
        # A stale exemption and an unexempted gap both fail, each named.
        result = parity_gate(
            {"produced_var": 0.5},
            {"produced_var": 0.4, "missing_var": 0.9},
            known_gaps=["produced_var"],
        )
        assert not result.passed
        assert result.details["gaps"] == 1
        assert result.details["stale_exemptions"] == ["produced_var"]
        assert any("missing_var" in failure for failure in result.failures)
        assert any("produced_var" in failure for failure in result.failures)


class TestSupportGate:
    def test_within_donor_support_passes(self) -> None:
        result = support_gate(
            {"stcg": np.asarray([-50.0, 0.0, 120.0])},
            {"stcg": (-100.0, 200.0)},
        )
        assert result.passed

    def test_escape_fails_with_ranges_shown(self) -> None:
        result = support_gate(
            {"stcg": np.asarray([-500.0, 0.0])}, {"stcg": (-100.0, 200.0)}
        )
        assert not result.passed
        assert "outside donor support" in result.failures[0]

    def test_missing_range_declaration_fails(self) -> None:
        result = support_gate({"stcg": np.asarray([1.0])}, {})
        assert not result.passed
        assert "no donor range declared" in result.failures[0]


class TestAggregateAdminGate:
    def _stcg_anchor(self) -> TargetSpec:
        return TargetSpec(
            name="puf/net_short_term_capital_gains",
            entity="household",
            measure="net_short_term_capital_gains",
            value=-76.8e9,
            signed=True,
            source="IRS SOI (PUF P22250, uprated)",
            family="irs_soi",
        )

    def test_sign_flip_is_caught(self) -> None:
        # The v2 incident: calibration drove net STCG to -$3.9T... and a
        # positive build would be just as wrong. Flip vs the signed anchor:
        result = aggregate_admin_gate(
            {"puf/net_short_term_capital_gains": +3.9e12},
            [self._stcg_anchor()],
        )
        assert not result.passed
        assert "sign flip" in result.failures[0]

    def test_order_of_magnitude_blowup_is_caught(self) -> None:
        # The investment-interest incident: $33.5T against tens of billions.
        anchor = TargetSpec(
            name="soi/investment_interest_expense",
            entity="household",
            measure="investment_interest_expense",
            value=24.0e9,
            source="IRS SOI Table 1.4",
            family="irs_soi",
        )
        result = aggregate_admin_gate(
            {"soi/investment_interest_expense": 33.5e12}, [anchor]
        )
        assert not result.passed
        assert "relative miss" in result.failures[0]

    def test_within_tolerance_passes(self) -> None:
        result = aggregate_admin_gate(
            {"puf/net_short_term_capital_gains": -80e9},
            [self._stcg_anchor()],
        )
        assert result.passed

    def test_declared_but_unmeasured_anchor_fails(self) -> None:
        result = aggregate_admin_gate({}, [self._stcg_anchor()])
        assert not result.passed
        assert "did not measure" in result.failures[0]

    def test_spec_tolerance_overrides_default(self) -> None:
        anchor = TargetSpec(
            name="census/population",
            entity="household",
            measure="household_count",
            value=334e6,
            tolerance=1e6,  # absolute
            source="Census",
            family="census",
        )
        tight_miss = aggregate_admin_gate({"census/population": 340e6}, [anchor])
        assert not tight_miss.passed  # 6M off against a 1M tolerance


class TestPerFamilyFitGate:
    def test_broad_family_miss_cannot_hide(self) -> None:
        names = [f"good/t{i}" for i in range(20)] + [f"bad/t{i}" for i in range(6)]
        errors = [0.01] * 20 + [0.5] * 6
        result = per_family_fit_gate(names, errors)
        assert not result.passed
        assert result.failures and "bad" in result.failures[0]
        # the global average would have looked fine:
        global_share = np.mean(
            [abs(e) <= result.details["hard_within"] for e in errors]
        )
        assert global_share > 0.7

    def test_near_misses_are_diagnostic_not_hard_failures(self) -> None:
        names = [f"cbo/t{i}" for i in range(6)]
        errors = [0.1041] * 6
        result = per_family_fit_gate(names, errors)
        assert result.passed
        assert result.details["family_within_shares"]["cbo"] == 0.0
        assert result.details["family_hard_within_shares"]["cbo"] == 1.0

    def test_small_families_report_but_do_not_gate(self) -> None:
        names = (
            ["good/t0"] * 0
            + [f"good/t{i}" for i in range(10)]
            + [
                "tiny/t0",
                "tiny/t1",
            ]
        )
        errors = [0.01] * 10 + [0.9, 0.9]
        result = per_family_fit_gate(names, errors, min_family_size=5)
        assert result.passed
        assert result.details["family_within_shares"]["tiny"] == 0.0
        assert result.details["family_hard_within_shares"]["tiny"] == 0.0

    def test_misalignment_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            per_family_fit_gate(["a/b"], [0.1, 0.2])


class TestRelativeErrorLoss:
    def test_matches_the_calibrator_formula(self) -> None:
        est = np.asarray([110.0, 90.0])
        tgt = np.asarray([100.0, 100.0])
        expected = np.abs((est - tgt) / np.maximum(np.abs(tgt), 1.0)).mean()
        assert relative_error_loss(est, tgt) == pytest.approx(expected)

    def test_accepts_target_loss_weights(self) -> None:
        est = np.asarray([110.0, 90.0])
        tgt = np.asarray([100.0, 100.0])
        weights = np.asarray([10.0, 1.0])
        residual = np.abs((est - tgt) / np.maximum(np.abs(tgt), 1.0))
        expected = np.average(residual, weights=weights)

        assert relative_error_loss(
            est,
            tgt,
            target_loss_weights=weights,
        ) == pytest.approx(expected)

    def test_accepts_target_loss_scales_and_caps_each_row(self) -> None:
        est = np.asarray([1_000.0, 50.0])
        tgt = np.asarray([0.0, 100.0])
        scales = np.asarray([100.0, 100.0])
        expected = np.asarray([10.0, 0.5]).mean()

        assert relative_error_loss(
            est,
            tgt,
            target_loss_scales=scales,
            target_loss_cap=10.0,
        ) == pytest.approx(expected)

    def test_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must align"):
            relative_error_loss(np.zeros(2), np.zeros(3))

    def test_weight_shape_mismatch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="target_loss_weights"):
            relative_error_loss(
                np.zeros(2),
                np.zeros(2),
                target_loss_weights=np.zeros(3),
            )


class TestZeroValuedAnchor:
    def test_zero_anchor_gates_on_absolute_scale(self) -> None:
        anchor = TargetSpec(
            name="soi/net_operating_loss_carryforward_count",
            entity="household",
            measure="net_operating_loss_carryforward_count",
            value=0.0,
            source="IRS SOI (no filers expected)",
            family="irs_soi",
        )
        assert aggregate_admin_gate({anchor.name: 0.0}, [anchor]).passed
        assert aggregate_admin_gate({anchor.name: 0.3}, [anchor]).passed
        assert not aggregate_admin_gate({anchor.name: 0.8}, [anchor]).passed


class TestExportedNonzeroGate:
    def test_all_zero_column_fails_with_remedy_named(self) -> None:
        from populace.build import exported_nonzero_gate

        result = exported_nonzero_gate({"snap": 0.1, "net_worth": 0.0})
        assert not result.passed
        assert "net_worth" in result.failures[0]
        assert "populate it or remove it upstream" in result.failures[0]

    def test_fully_populated_export_passes(self) -> None:
        from populace.build import exported_nonzero_gate

        result = exported_nonzero_gate({"snap": 0.1, "tips": 0.004})
        assert result.passed
        assert result.details["columns_checked"] == 2

    def test_exemption_needs_a_reason(self) -> None:
        from populace.build import exported_nonzero_gate

        with pytest.raises(ValueError, match="needs a reason"):
            exported_nonzero_gate({"x": 0.0}, exemptions={"x": ""})

    def test_documented_exemption_passes_and_is_recorded(self) -> None:
        from populace.build import exported_nonzero_gate

        result = exported_nonzero_gate(
            {"x": 0.0},
            exemptions={"x": "no NOL filers in 2024 vintage (SOI table 1.4)"},
        )
        assert result.passed
        assert "x" in result.details["exempted"]

    def test_unused_exemptions_are_surfaced(self) -> None:
        from populace.build import exported_nonzero_gate

        result = exported_nonzero_gate(
            {"y": 0.5}, exemptions={"gone_var": "documented"}
        )
        assert result.passed
        assert result.details["unused_exemptions"] == ["gone_var"]


class TestNonconstantColumnsGate:
    def test_constant_bool_column_fails_with_remedy_named(self) -> None:
        result = nonconstant_columns_gate(
            {"takes_up_aca_if_eligible": np.asarray([True, True, True])},
            ["takes_up_aca_if_eligible"],
        )
        assert not result.passed
        assert "takes_up_aca_if_eligible" in result.failures[0]
        assert "regenerate upstream source inputs" in result.failures[0]
        assert result.details["constant_values"] == {"takes_up_aca_if_eligible": True}

    def test_nonconstant_columns_pass(self) -> None:
        result = nonconstant_columns_gate(
            {
                "takes_up_aca_if_eligible": np.asarray([True, False]),
                "selected_marketplace_plan_benchmark_ratio": np.asarray([1.0, 0.7]),
            },
            [
                "takes_up_aca_if_eligible",
                "selected_marketplace_plan_benchmark_ratio",
            ],
        )
        assert result.passed
        assert result.details["unique_counts"] == {
            "selected_marketplace_plan_benchmark_ratio": 2,
            "takes_up_aca_if_eligible": 2,
        }

    def test_missing_required_column_fails(self) -> None:
        result = nonconstant_columns_gate({}, ["takes_up_aca_if_eligible"])
        assert not result.passed
        assert "required nonconstant column missing" in result.failures[0]

    def test_reviewed_exclusion_needs_mapping_reason(self) -> None:
        with pytest.raises(TypeError, match="mapping from name to reason"):
            nonconstant_columns_gate(
                {},
                ["takes_up_aca_if_eligible"],
                reviewed_exclusions=["takes_up_aca_if_eligible"],  # type: ignore[arg-type]
            )


class TestDefaultValuedColumnsGate:
    def test_column_stuck_at_engine_default_fails(self) -> None:
        # The constant-40 hours incident: populated, plausible-looking, and
        # identical to the engine default for every record.
        result = default_valued_columns_gate(
            {"weekly_hours_worked_before_lsr": np.asarray([40.0, 40.0, 40.0])},
            {"weekly_hours_worked_before_lsr": 40.0},
        )
        assert not result.passed
        assert "weekly_hours_worked_before_lsr" in result.failures[0]
        assert "engine default" in result.failures[0]
        assert result.details["default_valued_columns"] == {
            "weekly_hours_worked_before_lsr": 40.0
        }

    def test_constant_bool_at_default_fails(self) -> None:
        result = default_valued_columns_gate(
            {"takes_up_snap_if_eligible": np.asarray([True, True])},
            {"takes_up_snap_if_eligible": True},
        )
        assert not result.passed

    def test_constant_enum_name_at_default_fails(self) -> None:
        result = default_valued_columns_gate(
            {"ssn_card_type": np.asarray(["CITIZEN", "CITIZEN"])},
            {"ssn_card_type": "CITIZEN"},
        )
        assert not result.passed

    def test_column_with_signal_passes(self) -> None:
        result = default_valued_columns_gate(
            {"weekly_hours_worked_before_lsr": np.asarray([0.0, 25.0, 40.0])},
            {"weekly_hours_worked_before_lsr": 40.0},
        )
        assert result.passed
        assert result.details["columns_checked"] == 1

    def test_constant_but_not_default_passes_and_is_reported(self) -> None:
        result = default_valued_columns_gate(
            {"pension_contribution_rate": np.asarray([0.05, 0.05])},
            {"pension_contribution_rate": 0.0},
        )
        assert result.passed
        assert result.details["constant_nondefault_columns"] == {
            "pension_contribution_rate": 0.05
        }

    def test_bool_column_never_matches_numeric_default(self) -> None:
        # True == 1 in Python; a flag stuck at True must not be excused by a
        # numeric default of 1.0.
        result = default_valued_columns_gate(
            {"some_flag": np.asarray([True, True])},
            {"some_flag": 1.0},
        )
        assert result.passed
        assert result.details["constant_nondefault_columns"] == {"some_flag": True}

    def test_column_without_declared_default_is_skipped(self) -> None:
        result = default_valued_columns_gate(
            {"HRSWK": np.asarray([0, 0, 0])},
            {},
        )
        assert result.passed
        assert result.details["columns_checked"] == 0

    def test_reviewed_exclusion_accepts_known_degenerate_column(self) -> None:
        result = default_valued_columns_gate(
            {"weekly_hours_worked_before_lsr": np.asarray([40.0, 40.0])},
            {"weekly_hours_worked_before_lsr": 40.0},
            reviewed_exclusions={
                "weekly_hours_worked_before_lsr": "tracked in populace#242"
            },
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "weekly_hours_worked_before_lsr": "tracked in populace#242"
        }

    def test_stale_exclusion_for_column_with_signal_fails(self) -> None:
        result = default_valued_columns_gate(
            {"weekly_hours_worked_before_lsr": np.asarray([0.0, 40.0])},
            {"weekly_hours_worked_before_lsr": 40.0},
            reviewed_exclusions={
                "weekly_hours_worked_before_lsr": "tracked in populace#242"
            },
        )
        assert not result.passed
        assert "Stale reviewed exclusions" in result.failures[0]

    def test_dormant_exclusion_for_absent_column_passes(self) -> None:
        # Different release lines persist different column sets; an exclusion
        # for a column not on this surface must not fail the gate.
        result = default_valued_columns_gate(
            {"employment_income": np.asarray([0.0, 52_000.0])},
            {"employment_income": 0.0},
            reviewed_exclusions={
                "weekly_hours_worked_before_lsr": "tracked in populace#242"
            },
        )
        assert result.passed
        assert result.details["dormant_exclusions"] == [
            "weekly_hours_worked_before_lsr"
        ]

    def test_reviewed_exclusion_needs_mapping_reason(self) -> None:
        with pytest.raises(TypeError, match="mapping from name to reason"):
            default_valued_columns_gate(
                {},
                {},
                reviewed_exclusions=["weekly_hours_worked_before_lsr"],  # type: ignore[arg-type]
            )


class TestNonnegativeColumnsGate:
    def test_negative_exported_value_fails_with_column_named(self) -> None:
        result = nonnegative_columns_gate(
            {"auto_loan_interest": [0.0, 125.0, -9.0]},
            ["auto_loan_interest"],
        )
        assert not result.passed
        assert "auto_loan_interest" in result.failures[0]
        assert "below zero" in result.failures[0]
        assert result.details["negative_counts"] == {"auto_loan_interest": 1}

    def test_nonnegative_required_column_passes(self) -> None:
        result = nonnegative_columns_gate(
            {"auto_loan_interest": [0.0, 125.0, float("nan")]},
            ["auto_loan_interest"],
        )
        assert result.passed
        assert result.details["minima"] == {"auto_loan_interest": 0.0}

    def test_missing_required_column_fails(self) -> None:
        result = nonnegative_columns_gate({}, ["auto_loan_interest"])
        assert not result.passed
        assert "required non-negative column missing" in result.failures[0]

    def test_reviewed_exclusion_needs_mapping_reason(self) -> None:
        with pytest.raises(TypeError, match="mapping from name to reason"):
            nonnegative_columns_gate(
                {},
                ["auto_loan_interest"],
                reviewed_exclusions=["auto_loan_interest"],  # type: ignore[arg-type]
            )

    def test_sliceable_column_is_scanned_in_chunks(self) -> None:
        class SliceableColumn:
            shape = (5,)

            def __init__(self) -> None:
                self.slices: list[slice] = []

            def __getitem__(self, key: slice) -> np.ndarray:
                self.slices.append(key)
                return np.asarray([0.0, -1.0, 2.0, 3.0, 4.0])[key]

        column = SliceableColumn()
        result = nonnegative_columns_gate(
            {"auto_loan_interest": column},
            ["auto_loan_interest"],
            chunk_size=2,
        )
        assert not result.passed
        assert column.slices == [slice(0, 2), slice(2, 4), slice(4, 5)]
        assert result.details["negative_counts"] == {"auto_loan_interest": 1}


class TestFormulaOwnedExportGate:
    def test_formula_owned_column_fails_with_remedy_named(self) -> None:
        result = formula_owned_export_gate(
            ["person_id", "employment_income", "ssi"],
            ["ssi", "income_tax"],
            structural_columns=["person_id"],
        )
        assert not result.passed
        assert result.failures == (
            "ssi: formula-owned engine output exported as an input; "
            "remove it upstream before export.",
        )
        assert result.details["offenders"] == ["ssi"]

    def test_structural_overlap_is_exempted(self) -> None:
        result = formula_owned_export_gate(
            ["person_id", "employment_income"],
            ["person_id", "income_tax"],
            structural_columns=["person_id"],
        )
        assert result.passed
        assert result.details["structural_exemptions"] == ["person_id"]


class TestExportSurfaceGate:
    def test_reference_export_surface_match_passes(self) -> None:
        result = export_surface_gate(
            ["age", "employment_income", "household_id"],
            ["age", "employment_income"],
            candidate_name="populace-uk",
            reference_name="eFRS",
            allowed_extra_columns=["household_id"],
        )
        assert result.passed
        assert result.details["allowed_extra_columns"] == ["household_id"]

    def test_missing_reference_column_fails(self) -> None:
        result = export_surface_gate(
            ["age"],
            ["age", "employment_income"],
            candidate_name="populace-uk",
            reference_name="eFRS",
        )
        assert not result.passed
        assert "employment_income" in result.failures[0]

    def test_unreviewed_extra_column_fails(self) -> None:
        result = export_surface_gate(
            ["age", "employment_income", "raw_frs_serial"],
            ["age", "employment_income"],
            candidate_name="populace-uk",
            reference_name="eFRS",
        )
        assert not result.passed
        assert "raw_frs_serial" in result.failures[0]

    def test_reviewed_missing_reference_column_passes_and_is_recorded(self) -> None:
        result = export_surface_gate(
            ["age"],
            ["age", "legacy_efrs_marker"],
            reviewed_exclusions={
                "legacy_efrs_marker": "reference-only audit field, not a PE input"
            },
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "legacy_efrs_marker": "reference-only audit field, not a PE input"
        }

    def test_reviewed_exclusion_needs_a_reason(self) -> None:
        with pytest.raises(ValueError, match="need reasons"):
            export_surface_gate(["age"], ["age", "x"], reviewed_exclusions={"x": ""})

    def test_reviewed_exclusion_list_is_refused(self) -> None:
        with pytest.raises(TypeError, match="mapping from name to reason"):
            export_surface_gate(
                ["age"],
                ["age", "x"],
                reviewed_exclusions=["x"],  # type: ignore[arg-type]
            )


class TestInputMassParityGate:
    def test_mass_within_tolerance_passes(self) -> None:
        result = input_mass_parity_gate(
            {"student_loan_interest": 18e9, "charitable_cash_donations": 300e9},
            {"student_loan_interest": 22e9, "charitable_cash_donations": 340e9},
            candidate_name="sparse",
            reference_name="dense",
        )
        assert result.passed
        assert result.details["columns_checked"] == 2

    def test_zeroed_input_base_fails(self) -> None:
        result = input_mass_parity_gate(
            {"traditional_ira_contributions": 0.0},
            {"traditional_ira_contributions": 17.4e9},
            candidate_name="sparse",
            reference_name="dense",
        )
        assert not result.passed
        assert "traditional_ira_contributions" in result.failures[0]
        assert "-100.0%" in result.failures[0]

    def test_absent_column_fails(self) -> None:
        result = input_mass_parity_gate(
            {},
            {"tax_unit_childcare_expenses": 81.2e9},
            candidate_name="sparse",
            reference_name="dense",
        )
        assert not result.passed
        assert "absent" in result.failures[0]
        assert result.details["worst_drifts"]["tax_unit_childcare_expenses"] == -1.0

    def test_drift_beyond_tolerance_fails_in_both_directions(self) -> None:
        result = input_mass_parity_gate(
            {"a": 40.0, "b": 160.0},
            {"a": 100.0, "b": 100.0},
            relative_tolerance=0.5,
        )
        assert not result.passed
        assert len(result.failures) == 2

    def test_reference_below_floor_is_skipped(self) -> None:
        result = input_mass_parity_gate(
            {"tiny": 0.0, "material": 2e9, "negative_material": -2e9},
            {"tiny": 5e8, "material": 2e9, "negative_material": -2e9},
            relative_tolerance=0.5,
            minimum_reference_total=1e9,
        )
        assert result.passed
        assert result.details["columns_below_reference_floor"] == 1
        assert result.details["columns_checked"] == 2

    def test_signed_reference_mass_uses_absolute_denominator(self) -> None:
        result = input_mass_parity_gate(
            {"rental_income": -30.0},
            {"rental_income": -100.0},
            relative_tolerance=0.5,
        )
        assert not result.passed
        assert "+70.0%" in result.failures[0]

    def test_reviewed_exclusion_passes_and_is_recorded(self) -> None:
        result = input_mass_parity_gate(
            {"care_expenses": 0.0},
            {"care_expenses": 5e9},
            reviewed_exclusions={
                "care_expenses": "pre-existing zero-mass input, tracked in #26"
            },
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "care_expenses": "pre-existing zero-mass input, tracked in #26"
        }

    def test_unused_reviewed_exclusion_is_reported(self) -> None:
        result = input_mass_parity_gate(
            {"a": 1e10},
            {"a": 1e10},
            reviewed_exclusions={"gone_column": "kept for a retired reference"},
        )
        assert result.passed
        assert result.details["unused_reviewed_exclusions"] == ["gone_column"]

    def test_reviewed_exclusion_needs_a_reason(self) -> None:
        with pytest.raises(ValueError, match="need reasons"):
            input_mass_parity_gate({}, {"a": 1.0}, reviewed_exclusions={"a": ""})

    def test_candidate_only_columns_are_recorded_not_failed(self) -> None:
        result = input_mass_parity_gate(
            {"a": 1e10, "new_input": 5e9},
            {"a": 1e10},
        )
        assert result.passed
        assert result.details["candidate_only_columns"] == ["new_input"]

    def test_negative_tolerance_is_refused(self) -> None:
        with pytest.raises(ValueError, match="relative_tolerance"):
            input_mass_parity_gate({}, {}, relative_tolerance=-0.1)

    def test_non_finite_totals_are_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            input_mass_parity_gate({"a": float("nan")}, {"a": 1e10})


class TestTargetSurfaceGate:
    def test_candidate_can_be_strict_superset_of_reference(self) -> None:
        result = target_surface_gate(
            ["ons/population", "hmrc/income_tax", "dwp/universal_credit"],
            ["ons/population", "hmrc/income_tax"],
            candidate_name="populace-uk",
            reference_name="eFRS",
        )
        assert result.passed
        assert result.details["extra_candidate_targets"] == ["dwp/universal_credit"]

    def test_missing_reference_target_fails(self) -> None:
        result = target_surface_gate(
            ["ons/population"],
            ["ons/population", "hmrc/income_tax"],
            candidate_name="populace-uk",
            reference_name="eFRS",
        )
        assert not result.passed
        assert "hmrc/income_tax" in result.failures[0]

    def test_reviewed_missing_reference_target_passes_and_is_recorded(self) -> None:
        result = target_surface_gate(
            ["ons/population"],
            ["ons/population", "legacy/benefit_count"],
            reviewed_exclusions={
                "legacy/benefit_count": "retired eFRS target replaced by DWP admin target"
            },
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "legacy/benefit_count": "retired eFRS target replaced by DWP admin target"
        }

    def test_reviewed_exclusion_list_is_refused(self) -> None:
        with pytest.raises(TypeError, match="mapping from name to reason"):
            target_surface_gate(
                ["ons/population"],
                ["ons/population", "hmrc/income_tax"],
                reviewed_exclusions=["hmrc/income_tax"],  # type: ignore[arg-type]
            )


class TestTargetProfileCoverageGate:
    def test_required_target_concept_must_be_present(self) -> None:
        result = target_profile_coverage_gate(
            ["nation/irs/adjusted gross income/total"],
            [
                TargetCoverageRequirement(
                    requirement_id="income_tax",
                    label="Federal income tax",
                    accepted_measures=("income_tax",),
                    accepted_names=("nation/irs/income_tax_total",),
                )
            ],
        )
        assert not result.passed
        assert "income_tax" in result.failures[0]

    def test_requirement_can_match_structured_target_metadata(self) -> None:
        result = target_profile_coverage_gate(
            [
                {
                    "name": "nation/treasury/individual_income_tax",
                    "measure": "income_tax",
                    "family": "treasury",
                }
            ],
            [
                TargetCoverageRequirement(
                    requirement_id="income_tax",
                    label="Federal income tax",
                    accepted_measures=("income_tax",),
                )
            ],
        )
        assert result.passed
        assert result.details["matches_by_requirement"] == {
            "income_tax": ["nation/treasury/individual_income_tax"]
        }

    def test_requirement_metadata_must_match_when_declared(self) -> None:
        requirement = TargetCoverageRequirement(
            requirement_id="jct_salt",
            label="JCT SALT expenditure",
            accepted_names=("nation/jct/salt_deduction_expenditure",),
            required_metadata=(
                ("kind", "neutralize_variable"),
                ("output_variable", "income_tax"),
            ),
        )
        missing_metadata = target_profile_coverage_gate(
            ["nation/jct/salt_deduction_expenditure"], [requirement]
        )
        assert not missing_metadata.passed

        with_metadata = target_profile_coverage_gate(
            [
                {
                    "name": "nation/jct/salt_deduction_expenditure",
                    "kind": "neutralize_variable",
                    "output_variable": "income_tax",
                }
            ],
            [requirement],
        )
        assert with_metadata.passed

    def test_required_metadata_keys_must_be_present(self) -> None:
        requirement = TargetCoverageRequirement(
            requirement_id="snap_state_benefits",
            label="SNAP state benefit totals",
            accepted_families=("usda_snap",),
            required_metadata_keys=("state_fips",),
        )
        missing_key = target_profile_coverage_gate(
            [
                {
                    "name": "usda_snap.fy2024.national_benefits.total_benefits",
                    "family": "usda_snap",
                    "metadata": {"target_role": "snap_total"},
                }
            ],
            [requirement],
        )
        assert not missing_key.passed

        with_key = target_profile_coverage_gate(
            [
                {
                    "name": "usda_snap.fy2024.state_benefits.wro.ca.total_benefits",
                    "family": "usda_snap",
                    "metadata": {"target_role": "snap_total", "state_fips": "06"},
                }
            ],
            [requirement],
        )
        assert with_key.passed

    def test_required_metadata_keys_reject_empty_key(self) -> None:
        with pytest.raises(ValueError, match="non-empty keys"):
            TargetCoverageRequirement(
                requirement_id="bad",
                label="Bad requirement",
                accepted_families=("usda_snap",),
                required_metadata_keys=("",),
            )

    def test_minimum_match_count_is_enforced(self) -> None:
        requirement = TargetCoverageRequirement(
            requirement_id="state_income_tax",
            label="State income tax",
            accepted_name_substrings=("/state_income_tax",),
            min_matches=2,
        )
        fail = target_profile_coverage_gate(
            ["state/CA/state_income_tax"], [requirement]
        )
        assert not fail.passed
        ok = target_profile_coverage_gate(
            ["state/CA/state_income_tax", "state/NY/state_income_tax"],
            [requirement],
        )
        assert ok.passed

    def test_reviewed_exclusion_needs_mapping_reason(self) -> None:
        with pytest.raises(TypeError, match="mapping from name to reason"):
            target_profile_coverage_gate(
                [],
                [
                    TargetCoverageRequirement(
                        requirement_id="income_tax",
                        label="Federal income tax",
                        accepted_measures=("income_tax",),
                    )
                ],
                reviewed_exclusions=["income_tax"],  # type: ignore[arg-type]
            )


class TestMacroRealismGate:
    def test_metrics_inside_bands_pass(self) -> None:
        result = macro_realism_gate(
            {"income_tax_to_gdp": 0.09},
            {"income_tax_to_gdp": (0.07, 0.11)},
        )
        assert result.passed

    def test_missing_and_out_of_band_metrics_fail(self) -> None:
        result = macro_realism_gate(
            {"income_tax_to_gdp": 0.04},
            {
                "income_tax_to_gdp": (0.07, 0.11),
                "agi_to_gdp": (0.50, 0.70),
            },
        )
        assert not result.passed
        assert any("income_tax_to_gdp" in failure for failure in result.failures)
        assert any("agi_to_gdp" in failure for failure in result.failures)

    def test_bad_band_is_refused(self) -> None:
        with pytest.raises(ValueError, match="low > high"):
            macro_realism_gate({"x": 1.0}, {"x": (2.0, 1.0)})


class TestEnumDomainGate:
    def test_valid_enum_names_pass(self) -> None:
        result = enum_domain_gate(
            {"race": ["WHITE", "BLACK", "HISPANIC", "OTHER"]},
            {"race": ("WHITE", "BLACK", "HISPANIC", "OTHER")},
        )
        assert result.passed
        assert result.details["columns_checked"] == 1

    def test_raw_source_codes_fail_with_examples(self) -> None:
        result = enum_domain_gate(
            {"race": [0, 1, 10, 11]},
            {"race": ("WHITE", "BLACK", "HISPANIC", "OTHER")},
        )
        assert not result.passed
        assert result.failures and "race" in result.failures[0]
        assert result.details["invalid_counts"] == {"race": 4}
        assert "10" in result.details["invalid_examples"]["race"]

    def test_enum_class_domains_are_supported(self) -> None:
        import enum

        class Race(enum.Enum):
            WHITE = "white"
            BLACK = "black"

        result = enum_domain_gate({"race": [Race.WHITE, "BLACK"]}, {"race": Race})
        assert result.passed


class TestSourceCoverageGate:
    def test_hard_targets_must_be_active_or_reviewed(self) -> None:
        coverage = (
            {
                "family_id": "population_age_sex",
                "role": "hard_target",
                "package_aliases": ("census-pep-state-age-sex",),
            },
        )
        result = source_coverage_gate(coverage)
        assert not result.passed
        assert "census-pep-state-age-sex" in result.failures[0]

    def test_reviewed_hard_target_exclusion_passes_and_is_recorded(self) -> None:
        coverage = (
            {
                "family_id": "population_age_sex",
                "role": "hard_target",
                "package_aliases": ("census-pep-state-age-sex",),
            },
        )
        result = source_coverage_gate(
            coverage,
            reviewed_exclusions={
                "census-pep-state-age-sex": "state-age targets not in this smoke build"
            },
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "census-pep-state-age-sex": "state-age targets not in this smoke build"
        }
        assert result.details["coverage_summary"]["hard_target"] == {
            "families": 1,
            "package_aliases": 1,
            "covered_package_aliases": 0,
            "missing_package_aliases": 0,
            "reviewed_excluded_package_aliases": 1,
        }

    def test_reviewed_hard_target_exclusion_requires_reason(self) -> None:
        coverage = (
            {
                "family_id": "population_age_sex",
                "role": "hard_target",
                "package_aliases": ("census-pep-state-age-sex",),
            },
        )
        result = source_coverage_gate(
            coverage,
            reviewed_exclusions={"census-pep-state-age-sex": ""},
        )
        assert not result.passed
        assert "requires a non-empty string reason" in result.failures[0]
        assert "census-pep-state-age-sex" in result.details["missing_hard_targets"]
        assert result.details["reviewed_exclusions"] == {}
        assert result.details["coverage_summary"]["hard_target"] == {
            "families": 1,
            "package_aliases": 1,
            "covered_package_aliases": 0,
            "missing_package_aliases": 1,
            "reviewed_excluded_package_aliases": 0,
        }

    def test_reviewed_hard_target_exclusion_reason_must_be_string(self) -> None:
        coverage = (
            {
                "family_id": "population_age_sex",
                "role": "hard_target",
                "package_aliases": ("census-pep-state-age-sex",),
            },
        )
        result = source_coverage_gate(
            coverage,
            reviewed_exclusions={"census-pep-state-age-sex": None},
        )
        assert not result.passed
        assert "requires a non-empty string reason" in result.failures[0]
        assert result.details["reviewed_exclusions"] == {}

    def test_reviewed_hard_target_exclusion_requires_mapping(self) -> None:
        coverage = (
            {
                "family_id": "population_age_sex",
                "role": "hard_target",
                "package_aliases": ("census-pep-state-age-sex",),
            },
        )
        with pytest.raises(TypeError, match="mapping from alias to reason"):
            source_coverage_gate(
                coverage,
                reviewed_exclusions=("census-pep-state-age-sex",),
            )

    def test_validation_only_family_cannot_be_a_hard_target(self) -> None:
        coverage = (
            {
                "family_id": "census_cps_spm",
                "role": "validation_only",
                "package_aliases": ("census-cps-spm-2024",),
            },
        )
        result = source_coverage_gate(
            coverage,
            active_target_families=("census_cps_spm",),
        )
        assert not result.passed
        assert "validation-only" in result.failures[0]
        assert result.details["validation_only_families"]["census_cps_spm"][
            "activated_as_hard_target"
        ]

    def test_source_gaps_are_reported_without_failing(self) -> None:
        coverage = (
            {
                "family_id": "usda_wic",
                "role": "source_gap",
                "missing_source_packages": ("USDA FNS WIC program data",),
            },
        )
        result = source_coverage_gate(coverage)
        assert result.passed
        assert result.details["source_gaps"] == {
            "usda_wic": ("USDA FNS WIC program data",)
        }
        assert result.details["coverage_summary"]["source_gap"] == {
            "families": 1,
            "missing_source_packages": 1,
        }

    def test_named_source_coverage_gate_surfaces_manifest_summary(self) -> None:
        coverage = (
            {
                "family_id": "population_age_sex",
                "role": "hard_target",
                "package_aliases": ("census-pep-state-age-sex",),
            },
        )
        result = source_coverage_gate(
            coverage,
            active_target_aliases=("census-pep-state-age-sex",),
            name="us_source_coverage",
        )
        manifest = GateReport((result,)).to_manifest()
        gate = manifest["gates"]["us_source_coverage"]
        assert gate["passed"]
        assert gate["details"]["coverage_summary"]["hard_target"] == {
            "families": 1,
            "package_aliases": 1,
            "covered_package_aliases": 1,
            "missing_package_aliases": 0,
            "reviewed_excluded_package_aliases": 0,
        }


class TestFitWeightRecord:
    def test_accepts_the_resolved_weight_kinds(self) -> None:
        # design/importance/calibrated (WeightKind), none (unweighted), and
        # explicit (a DataFrame vector fit) — the vocabulary populace.fit's
        # resolved_weight_kind can emit.
        for kind in ("design", "importance", "calibrated", "none", "explicit"):
            record = FitWeightRecord("some_fit", kind)
            assert record.fit_name == "some_fit"
            assert record.weight_kind == kind

    def test_unknown_weight_kind_is_refused(self) -> None:
        # A fit recorded with an uninterpretable weight kind is unauditable —
        # the same class of failure as an imputation with no declared support.
        with pytest.raises(ValueError, match="weight kind"):
            FitWeightRecord("some_fit", "weighted")

    def test_empty_fit_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="fit_name"):
            FitWeightRecord("", "design")


class TestWeightsAuditGate:
    def test_weighted_fits_pass_and_are_recorded(self) -> None:
        result = weights_audit_gate(
            [
                FitWeightRecord("puf_tax_detail_support", "design"),
                FitWeightRecord("cps_reweight", "calibrated"),
            ]
        )
        assert result.passed
        # The manifest surface: every production fit's resolved kind is on the
        # record, which GateReport.to_manifest() then serializes.
        assert result.details["resolved_weight_kinds"] == {
            "cps_reweight": "calibrated",
            "puf_tax_detail_support": "design",
        }
        assert result.details["unweighted_fits"] == []

    def test_explicit_dataframe_weight_kind_is_weighted_and_passes(self) -> None:
        # A DataFrame fit weighted by a caller-supplied vector resolves to
        # "explicit": weighted, so it is not flagged as an unweighted risk.
        result = weights_audit_gate([FitWeightRecord("df_donor", "explicit")])
        assert result.passed
        assert result.details["resolved_weight_kinds"] == {"df_donor": "explicit"}
        assert result.details["unweighted_fits"] == []

    def test_unlisted_unweighted_fit_fails_with_the_fit_named(self) -> None:
        # The guarantee: a fit that resolved to no weights (the $201T-scale
        # landmine of the legacy stack) blocks the release unless explicitly
        # allowed with a reason.
        result = weights_audit_gate(
            [
                FitWeightRecord("puf_tax_detail_support", "design"),
                FitWeightRecord("scf_net_worth_donor", "none"),
            ]
        )
        assert not result.passed
        assert "scf_net_worth_donor" in result.failures[0]
        assert "unweighted" in result.failures[0]
        assert result.details["unweighted_fits"] == ["scf_net_worth_donor"]

    def test_allowlisted_unweighted_fit_passes_and_is_recorded(self) -> None:
        result = weights_audit_gate(
            [FitWeightRecord("weakly_informative_donor", "none")],
            allowed_unweighted={
                "weakly_informative_donor": (
                    "CPS person weights are weakly informative here; weighting "
                    "is free but unweighted is a reviewed choice (issue #300)."
                )
            },
        )
        assert result.passed
        assert result.details["allowed_unweighted"] == {
            "weakly_informative_donor": (
                "CPS person weights are weakly informative here; weighting "
                "is free but unweighted is a reviewed choice (issue #300)."
            )
        }
        # An allowed-unweighted fit is not reported as a live unweighted risk.
        assert result.details["unweighted_fits"] == []

    def test_allowlist_entry_needs_a_reason(self) -> None:
        # An undocumented allow entry is just a silent unweighted fit with
        # extra steps — the same rule every reviewed-exclusion gate enforces.
        with pytest.raises(ValueError, match="need reasons"):
            weights_audit_gate(
                [FitWeightRecord("donor", "none")],
                allowed_unweighted={"donor": ""},
            )

    def test_allowlist_must_be_a_mapping(self) -> None:
        with pytest.raises(TypeError, match="mapping from name to reason"):
            weights_audit_gate(
                [FitWeightRecord("donor", "none")],
                allowed_unweighted=["donor"],  # type: ignore[arg-type]
            )

    def test_unused_allowlist_entry_is_reported(self) -> None:
        # A weighted fit does not consume its allow entry; the stale entry is
        # surfaced so the register cannot rot.
        result = weights_audit_gate(
            [FitWeightRecord("donor", "design")],
            allowed_unweighted={"retired_donor": "kept for a fit no longer run"},
        )
        assert result.passed
        assert result.details["unused_allowed_unweighted"] == ["retired_donor"]

    def test_allow_entry_for_a_now_weighted_fit_is_reported_as_unused(self) -> None:
        # The allow entry names a fit that is present but resolved weighted;
        # the exemption is no longer live and is reported, not silently kept.
        result = weights_audit_gate(
            [FitWeightRecord("donor", "design")],
            allowed_unweighted={"donor": "used to be unweighted, now design"},
        )
        assert result.passed
        assert result.details["unused_allowed_unweighted"] == ["donor"]

    def test_duplicate_fit_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="[Dd]uplicate fit name"):
            weights_audit_gate(
                [
                    FitWeightRecord("donor", "design"),
                    FitWeightRecord("donor", "none"),
                ]
            )

    def test_audit_round_trips_through_the_release_manifest(self) -> None:
        result = weights_audit_gate(
            [
                FitWeightRecord("puf_tax_detail_support", "design"),
                FitWeightRecord("scf_net_worth_donor", "none"),
            ]
        )
        manifest = GateReport((result,)).to_manifest()
        gate = manifest["gates"]["weights_audit"]
        assert gate["passed"] is False
        assert gate["details"]["resolved_weight_kinds"] == {
            "puf_tax_detail_support": "design",
            "scf_net_worth_donor": "none",
        }
        assert gate["details"]["unweighted_fits"] == ["scf_net_worth_donor"]

    def test_empty_fit_list_passes_but_records_nothing(self) -> None:
        result = weights_audit_gate([])
        assert result.passed
        assert result.details["resolved_weight_kinds"] == {}
        assert result.details["fits_checked"] == 0


class TestSourceStageInputCoverageGate:
    """A validation row's provision-critical input leaf must be produced.

    Replays the #252/#253 class: a validation config scores a provision whose
    effect is driven by a pure-input leaf (``qualified_tuition_expenses``,
    ``qualified_passenger_vehicle_loan_interest``). If no source stage produces
    that leaf, the row validates as a structural zero and nobody notices until
    an external benchmark exposes it. This gate makes that fail the build.
    """

    def test_undeclared_required_leaf_fails_with_variable_named(self) -> None:
        # The #253 signature: education-credit validation needs
        # qualified_tuition_expenses, but no stage declares it.
        result = source_stage_input_coverage_gate(
            {"qualified_tuition_expenses": ("soi_education_credits",)},
            declared_outputs=[
                "employment_income_before_lsr",
                "short_term_capital_gains",
            ],
        )
        assert not result.passed
        assert "qualified_tuition_expenses" in result.failures[0]
        assert "soi_education_credits" in result.failures[0]
        assert result.details["missing"] == ["qualified_tuition_expenses"]

    def test_produced_required_leaf_passes(self) -> None:
        result = source_stage_input_coverage_gate(
            {"student_loan_interest": ("soi_student_loan_interest",)},
            declared_outputs=["student_loan_interest", "home_mortgage_interest"],
        )
        assert result.passed
        assert result.details["required_leaves"] == 1
        assert result.details["missing"] == []

    def test_multiple_consumers_of_a_missing_leaf_are_all_named(self) -> None:
        result = source_stage_input_coverage_gate(
            {
                "qualified_passenger_vehicle_loan_interest": (
                    "obbba_auto_loan_interest",
                    "te_auto_loan_interest",
                )
            },
            declared_outputs=["auto_loan_interest"],
        )
        assert not result.passed
        assert "obbba_auto_loan_interest" in result.failures[0]
        assert "te_auto_loan_interest" in result.failures[0]

    def test_reviewed_exclusion_passes_and_is_recorded(self) -> None:
        result = source_stage_input_coverage_gate(
            {
                "qualified_passenger_vehicle_loan_interest": (
                    "obbba_auto_loan_interest",
                )
            },
            declared_outputs=["auto_loan_interest"],
            reviewed_exclusions={
                "qualified_passenger_vehicle_loan_interest": (
                    "OBBBA auto-loan qualifying-interest input not yet imputed "
                    "(PolicyEngine/populace#252)."
                )
            },
        )
        assert result.passed
        assert result.details["reviewed_exclusions"] == {
            "qualified_passenger_vehicle_loan_interest": (
                "OBBBA auto-loan qualifying-interest input not yet imputed "
                "(PolicyEngine/populace#252)."
            )
        }
        assert result.details["missing"] == []

    def test_reviewed_exclusion_needs_a_reason(self) -> None:
        with pytest.raises(ValueError, match="need reasons"):
            source_stage_input_coverage_gate(
                {"x": ("row",)},
                declared_outputs=[],
                reviewed_exclusions={"x": ""},
            )

    def test_reviewed_exclusion_list_is_refused(self) -> None:
        with pytest.raises(TypeError, match="mapping from name to reason"):
            source_stage_input_coverage_gate(
                {"x": ("row",)},
                declared_outputs=[],
                reviewed_exclusions=["x"],  # type: ignore[arg-type]
            )

    def test_stale_exclusion_for_produced_leaf_fails(self) -> None:
        # An exclusion for a leaf a stage now produces is stale: the register
        # must not rot, exactly like the other gates' stale-exclusion checks.
        result = source_stage_input_coverage_gate(
            {"student_loan_interest": ("soi_student_loan_interest",)},
            declared_outputs=["student_loan_interest"],
            reviewed_exclusions={
                "student_loan_interest": "was missing, tracked in #999"
            },
        )
        assert not result.passed
        assert "Stale reviewed exclusions" in result.failures[0]
        assert result.details["stale_exclusions"] == ["student_loan_interest"]

    def test_exclusion_for_unrequired_leaf_is_reported_not_failed(self) -> None:
        # A dormant exclusion for a leaf no row requires is surfaced but does
        # not fail the gate (different release lines validate different rows).
        result = source_stage_input_coverage_gate(
            {"student_loan_interest": ("soi_student_loan_interest",)},
            declared_outputs=["student_loan_interest"],
            reviewed_exclusions={"some_other_leaf": "documented but not required"},
        )
        assert result.passed
        assert result.details["dormant_exclusions"] == ["some_other_leaf"]

    def test_details_map_each_missing_leaf_to_its_consumers(self) -> None:
        result = source_stage_input_coverage_gate(
            {
                "qualified_tuition_expenses": ("soi_education_credits",),
                "student_loan_interest": ("soi_student_loan_interest",),
            },
            declared_outputs=["student_loan_interest"],
        )
        assert not result.passed
        assert result.details["missing_consumers"] == {
            "qualified_tuition_expenses": ["soi_education_credits"]
        }
