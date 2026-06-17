"""The acceptance gates: each failure mode named, each pass earned.

The aggregate-vs-admin tests replay the two real incidents (net-STCG sign
flip, investment-interest blow-up) as the cases the gate must catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from populace.build import (
    GateReport,
    GateResult,
    TargetCoverageRequirement,
    aggregate_admin_gate,
    enum_domain_gate,
    export_surface_gate,
    formula_owned_export_gate,
    macro_realism_gate,
    nonconstant_columns_gate,
    nonnegative_columns_gate,
    parity_gate,
    per_family_fit_gate,
    relative_error_loss,
    source_coverage_gate,
    support_gate,
    target_profile_coverage_gate,
    target_surface_gate,
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
        expected = (((est - tgt) / (tgt + 1.0)) ** 2).mean()
        assert relative_error_loss(est, tgt) == pytest.approx(expected)

    def test_accepts_target_loss_weights(self) -> None:
        est = np.asarray([110.0, 90.0])
        tgt = np.asarray([100.0, 100.0])
        weights = np.asarray([10.0, 1.0])
        residual = ((est - tgt) / (tgt + 1.0)) ** 2
        expected = np.average(residual, weights=weights)

        assert relative_error_loss(
            est,
            tgt,
            target_loss_weights=weights,
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
