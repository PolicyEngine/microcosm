from __future__ import annotations

import numpy as np
import pytest

from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate


def _passed(result) -> bool:
    assert result.name == "stage_health"
    return result.passed


def test_support_clip_gate_requires_receipted_columns_and_wires_thresholds() -> None:
    evidence = {
        "stage": "was_wealth",
        "support_clip": {
            "columns": {
                "cash_isa": {
                    "donor_min": 0.0,
                    "donor_max": 100.0,
                    "clipped_low_rows": 1,
                    "clipped_high_rows": 0,
                    "rows_considered": 2,
                }
            }
        },
    }
    parameters = {
        "stage": "was_wealth",
        "check": "support_clip",
        "columns": ["cash_isa"],
        "max_clipped_low_rows_by_column": {"cash_isa": 1},
        "max_clipped_high_rows_by_column": {"cash_isa": 0},
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="was_wealth",
            check="support_clip",
            parameters=parameters,
        )
    )

    failed = uk_stage_health_gate(
        evidence=evidence,
        stage="was_wealth",
        check="support_clip",
        parameters={
            **parameters,
            "max_clipped_low_rows_by_column": {"cash_isa": 0},
        },
    )
    assert failed.passed is False
    assert "clipped_low_rows" in failed.failures[0]


def test_realization_gate_target_and_deviation_parameters_are_live() -> None:
    evidence = {
        "stage": "salary_sacrifice",
        "headcount_receipt": {
            "target": 10.0,
            "realization_deviation": 0.1,
            "cap_bound": False,
        },
    }
    parameters = {
        "stage": "salary_sacrifice",
        "check": "realization_target",
        "target": 10.0,
        "maximum_abs_realization_deviation": 0.1,
        "allow_cap_bound": False,
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="salary_sacrifice",
            check="realization_target",
            parameters=parameters,
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="salary_sacrifice",
        check="realization_target",
        parameters={**parameters, "target": 11.0},
    ).passed
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="salary_sacrifice",
        check="realization_target",
        parameters={**parameters, "maximum_abs_realization_deviation": 0.09},
    ).passed


def test_student_loan_stock_parameter_is_live() -> None:
    evidence = {
        "stage": "student_loans",
        "plans": {
            "PLAN_2": {
                "stock": 100.0,
                "final_england_count": 98.0,
                "realization_deviation": -0.02,
            }
        },
    }
    parameters = {
        "stage": "student_loans",
        "check": "student_loan_plans",
        "stocks": {"PLAN_2": 100.0},
        "maximum_abs_realization_deviation": 0.02,
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="student_loans",
            check="student_loan_plans",
            parameters=parameters,
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="student_loans",
        check="student_loan_plans",
        parameters={**parameters, "stocks": {"PLAN_2": 99.0}},
    ).passed


def test_cgt_incidence_mass_threshold_is_live() -> None:
    evidence = {
        "stage": "cgt_incidence_clone",
        "mass_by_clone_flag": {"false": 100.0, "true": 99.0},
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="cgt_incidence_clone",
            check="cgt_incidence_mass",
            parameters={
                "stage": "cgt_incidence_clone",
                "check": "cgt_incidence_mass",
                "maximum_relative_mass_imbalance": 0.01,
            },
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="cgt_incidence_clone",
        check="cgt_incidence_mass",
        parameters={
            "stage": "cgt_incidence_clone",
            "check": "cgt_incidence_mass",
            "maximum_relative_mass_imbalance": 0.009,
        },
    ).passed


def test_cgt_incidence_mass_accepts_float_roundoff_at_zero_policy_tolerance() -> None:
    original = 100.0
    clone = np.nextafter(original, np.inf)

    result = uk_stage_health_gate(
        evidence={
            "stage": "cgt_incidence_clone",
            "mass_by_clone_flag": {"false": original, "true": clone},
        },
        stage="cgt_incidence_clone",
        check="cgt_incidence_mass",
        parameters={
            "stage": "cgt_incidence_clone",
            "check": "cgt_incidence_mass",
            "maximum_relative_mass_imbalance": 0.0,
        },
    )

    assert _passed(result)
    assert result.details["relative_imbalance"] > 0.0


def test_spi_support_channel_parameters_are_live() -> None:
    evidence = {
        "stage": "spi_support_channel",
        "spi_prior_mass_share": 0.5,
        "household_weight_kind": "importance",
        "spi_households": 10,
    }
    parameters = {
        "stage": "spi_support_channel",
        "check": "spi_support_channel",
        "spi_prior_mass_share": 0.5,
        "absolute_tolerance": 0.0,
        "household_weight_kind": "importance",
        "minimum_spi_households": 10,
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="spi_support_channel",
            check="spi_support_channel",
            parameters=parameters,
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="spi_support_channel",
        check="spi_support_channel",
        parameters={**parameters, "minimum_spi_households": 11},
    ).passed


def test_spi_income_identity_parameters_are_live() -> None:
    evidence = {
        "stage": "hmrc_spi_income_spine",
        "spi_prior": {"mass_share": 0.5},
        "targets": {"count": 2},
        "post_draw_identity": {"exact": True, "rows_checked": 3},
    }
    parameters = {
        "stage": "hmrc_spi_income_spine",
        "check": "spi_income_spine",
        "spi_prior_mass_share": 0.5,
        "absolute_tolerance": 0.0,
        "minimum_identity_rows": 3,
        "minimum_target_count": 2,
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="hmrc_spi_income_spine",
            check="spi_income_spine",
            parameters=parameters,
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="hmrc_spi_income_spine",
        check="spi_income_spine",
        parameters={**parameters, "minimum_target_count": 3},
    ).passed


def test_source_signal_structural_zero_parameter_is_live() -> None:
    evidence = {
        "stage": "frs_hmrc_spine_leaves",
        "source_signal_rows": {"gift_aid": 0, "employment_income": 2},
        "structural_zero_columns": ["gift_aid"],
    }
    parameters = {
        "stage": "frs_hmrc_spine_leaves",
        "check": "source_signal",
        "minimum_signal_rows": 1,
        "structural_zero_columns": ["gift_aid"],
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="frs_hmrc_spine_leaves",
            check="source_signal",
            parameters=parameters,
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="frs_hmrc_spine_leaves",
        check="source_signal",
        parameters={**parameters, "structural_zero_columns": []},
    ).passed


def test_cgt_band_donor_support_handles_open_upper_bound() -> None:
    evidence = {
        "stage": "cgt_band_donors",
        "bands": [
            {
                "lower_limit": 12300.0,
                "donor_count": 1,
                "realized_min_gain": 12300.0,
                "realized_max_gain": 1_000_000_000.0,
            }
        ],
    }
    parameters = {
        "stage": "cgt_band_donors",
        "check": "cgt_band_donor_support",
        "support_bounds_resource": "cgt_band_donor_support_bounds.json",
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="cgt_band_donors",
            check="cgt_band_donor_support",
            parameters=parameters,
        )
    )

    failed = uk_stage_health_gate(
        evidence={
            **evidence,
            "bands": [{**evidence["bands"][0], "realized_min_gain": 12_299.0}],
        },
        stage="cgt_band_donors",
        check="cgt_band_donor_support",
        parameters=parameters,
    )
    assert failed.passed is False
    assert "falls below" in failed.failures[0]


def test_age_tail_relative_deviation_parameter_is_live() -> None:
    evidence = {
        "stage": "uk_age_tail_disaggregation",
        "achieved_weighted": {"MALE": {"80_84": 90.0}},
        "band_populations": {"MALE:80_84": 100.0},
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="age_tail",
            check="age_tail_targets",
            parameters={
                "stage": "age_tail",
                "check": "age_tail_targets",
                "maximum_relative_deviation": 0.1,
            },
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="age_tail",
        check="age_tail_targets",
        parameters={
            "stage": "age_tail",
            "check": "age_tail_targets",
            "maximum_relative_deviation": 0.09,
        },
    ).passed


def test_cgt_summary_allocation_receipt_must_be_finite_and_non_negative() -> None:
    parameters = {
        "stage": "hmrc_cgt_gains_spine",
        "check": "cgt_imputation_summary",
        "minimum_band_rows": 1,
    }

    def evidence(error: float, released: float) -> dict:
        return {
            "stage": "hmrc_cgt_gains_spine",
            "rows": [{"gain_lower_bound": 12300.0}],
            "taxpayer_mass": 1.0,
            "published_taxpayer_mass": 1.0,
            "remainder_mass": 0.0,
            "allocation": {
                "rake": {"ipf_max_abs_margin_error": error, "ipf_zero_seed_cells": 0},
                "fallback_released_mass": released,
            },
        }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence(0.01, 250.0),
            stage="hmrc_cgt_gains_spine",
            check="cgt_imputation_summary",
            parameters=parameters,
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence(0.01, -1.0),
        stage="hmrc_cgt_gains_spine",
        check="cgt_imputation_summary",
        parameters=parameters,
    ).passed
    with pytest.raises(ValueError):
        uk_stage_health_gate(
            evidence=evidence(float("nan"), 0.0),
            stage="hmrc_cgt_gains_spine",
            check="cgt_imputation_summary",
            parameters=parameters,
        )


def test_cgt_summary_minimum_rows_parameter_is_live() -> None:
    evidence = {
        "stage": "hmrc_cgt_gains_spine",
        "rows": [{"gain_lower_bound": 12300.0}],
        "taxpayer_mass": 1.0,
        "published_taxpayer_mass": 1.0,
        "remainder_mass": 0.0,
    }

    assert _passed(
        uk_stage_health_gate(
            evidence=evidence,
            stage="hmrc_cgt_gains_spine",
            check="cgt_imputation_summary",
            parameters={
                "stage": "hmrc_cgt_gains_spine",
                "check": "cgt_imputation_summary",
                "minimum_band_rows": 1,
            },
        )
    )
    assert not uk_stage_health_gate(
        evidence=evidence,
        stage="hmrc_cgt_gains_spine",
        check="cgt_imputation_summary",
        parameters={
            "stage": "hmrc_cgt_gains_spine",
            "check": "cgt_imputation_summary",
            "minimum_band_rows": 2,
        },
    ).passed


def test_support_clip_gate_fails_closed_on_a_missing_allowance() -> None:
    """An undeclared allowance skipped the comparison entirely, so a stage
    clipping every row passed a release-blocking gate — the green-by-absence
    class the #787 review named. A non-exempt column now needs both bounds
    pinned, or the gate says so.
    """

    evidence = {
        "stage": "was_wealth",
        "support_clip": {
            "columns": {
                "cash_isa": {
                    "donor_min": 0.0,
                    "donor_max": 100.0,
                    "clipped_low_rows": 0,
                    "clipped_high_rows": 0,
                    "rows_considered": 2,
                }
            }
        },
    }
    result = uk_stage_health_gate(
        evidence=evidence,
        stage="was_wealth",
        check="support_clip",
        parameters={
            "stage": "was_wealth",
            "check": "support_clip",
            "columns": ["cash_isa"],
            "max_clipped_low_rows_by_column": {},
            "max_clipped_high_rows_by_column": {},
        },
    )
    assert not _passed(result)
    assert any("no clipped_low_rows allowance" in f for f in result.failures)
    assert any("no clipped_high_rows allowance" in f for f in result.failures)


def _latent_receipt() -> dict:
    row = {"target": 0.5, "realized": 0.51, "tolerance": 0.05, "rows": 1000}
    return {
        "stage": "uc_deduction_attributes",
        "coherence_violation_count": 0,
        "incidence_by_region": {"LONDON": dict(row)},
        "latent_rate_bands": {"AT_25": dict(row)},
        "combination_shares": {"ADVANCE_ONLY": dict(row)},
    }


def _latent_gate(evidence: dict):
    return uk_stage_health_gate(
        evidence=evidence,
        stage="uc_deduction_attributes",
        check="latent_attribute_realization",
        parameters={
            "stage": "uc_deduction_attributes",
            "check": "latent_attribute_realization",
        },
    )


def test_latent_attribute_realization_passes_a_coherent_in_band_receipt() -> None:
    result = _latent_gate(_latent_receipt())

    assert _passed(result)
    assert result.details["cells_checked"] == 3
    assert result.details["coherence_violation_count"] == 0


def test_latent_attribute_realization_fails_on_coherence_violations() -> None:
    evidence = _latent_receipt()
    evidence["coherence_violation_count"] = 2

    assert not _latent_gate(evidence).passed


def test_latent_attribute_realization_fails_when_the_count_is_missing() -> None:
    evidence = _latent_receipt()
    del evidence["coherence_violation_count"]

    assert not _latent_gate(evidence).passed


def test_latent_attribute_realization_fails_beyond_the_declared_tolerance() -> None:
    evidence = _latent_receipt()
    evidence["latent_rate_bands"]["AT_25"]["realized"] = 0.56

    assert not _latent_gate(evidence).passed


def test_latent_attribute_realization_caps_a_widened_producer_tolerance() -> None:
    # 1,000 rows at a 0.5 share give a four-sigma band of ~0.063; a producer
    # that declares 1.0 must not widen the pass rule.
    evidence = _latent_receipt()
    evidence["incidence_by_region"]["LONDON"].update(
        {"tolerance": 1.0, "realized": 0.6}
    )

    assert not _latent_gate(evidence).passed


def test_latent_attribute_realization_fails_on_empty_blocks_and_zero_rows() -> None:
    empty = _latent_receipt()
    empty["combination_shares"] = {}
    assert not _latent_gate(empty).passed

    zero_rows = _latent_receipt()
    zero_rows["incidence_by_region"]["LONDON"]["rows"] = 0
    assert not _latent_gate(zero_rows).passed


def _asset_type_evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "stage": "hmrc_cgt_asset_type_spine",
        "residential": {
            "count_target_individuals_basis": 202_630.0,
            "gains_target_individuals_basis": 12.24e9,
            "expected_count": 202_630.0,
            "expected_gains": 12.24e9,
            "achieved_count": 202_620.0,
            "achieved_gains": 11.9e9,
            "achieved_rows": 3_377,
            "max_liable_weight": 60.0,
            "count_bernoulli_sigma": 2_000.0,
            "gains_bernoulli_sigma": 0.15e9,
        },
        "asset_type": {
            "achieved_gains_share": {
                "listed_shares": 0.094,
                "unlisted_shares": 0.528,
                "other_financial_assets": 0.282,
                "agricultural_commercial_industrial_land_buildings": 0.044,
                "other_non_financial_assets": 0.052,
            }
        },
        "value_counts": {
            "none": 4_043,
            "sub_aea": 1_309,
            "residential_land_buildings": 3_377,
        },
    }
    evidence.update(overrides)
    return evidence


def test_cgt_asset_type_summary_holds_the_residential_realisation() -> None:
    parameters = {
        "stage": "hmrc_cgt_asset_type_spine",
        "check": "cgt_asset_type_summary",
        "maximum_relative_deviation": 0.05,
        "maximum_gains_sigma": 3.0,
        "maximum_solve_relative_error": 1e-6,
    }

    passed = uk_stage_health_gate(
        stage="hmrc_cgt_asset_type_spine",
        check="cgt_asset_type_summary",
        evidence=_asset_type_evidence(),
        parameters=parameters,
    )
    assert passed.passed
    assert passed.details["residential_count_gap"] == 10.0
    assert passed.details["residential_gains_bound"] == 0.05 * 12.24e9

    # A small frame: the noise floor is wider than the band and governs.
    noisy = _asset_type_evidence(
        residential={
            **_asset_type_evidence()["residential"],
            "achieved_gains": 9.0e9,
            "gains_bernoulli_sigma": 1.5e9,
        }
    )
    assert uk_stage_health_gate(
        stage="hmrc_cgt_asset_type_spine",
        check="cgt_asset_type_summary",
        evidence=noisy,
        parameters=parameters,
    ).passed

    # A count more than one person off the expectation is a broken draw.
    off_count = _asset_type_evidence(
        residential={
            **_asset_type_evidence()["residential"],
            "achieved_count": 202_500.0,
        }
    )
    failed = uk_stage_health_gate(
        stage="hmrc_cgt_asset_type_spine",
        check="cgt_asset_type_summary",
        evidence=off_count,
        parameters=parameters,
    )
    assert not failed.passed
    assert any("count gap" in failure for failure in failed.failures)

    drifted = _asset_type_evidence(
        residential={
            **_asset_type_evidence()["residential"],
            "achieved_gains": 10.0e9,
        }
    )
    failed = uk_stage_health_gate(
        stage="hmrc_cgt_asset_type_spine",
        check="cgt_asset_type_summary",
        evidence=drifted,
        parameters=parameters,
    )
    assert not failed.passed
    assert any("gains gap" in failure for failure in failed.failures)

    bad_share = _asset_type_evidence(
        asset_type={"achieved_gains_share": {"listed_shares": 1.5}}
    )
    failed = uk_stage_health_gate(
        stage="hmrc_cgt_asset_type_spine",
        check="cgt_asset_type_summary",
        evidence=bad_share,
        parameters=parameters,
    )
    assert not failed.passed

    with pytest.raises(ValueError, match="residential"):
        uk_stage_health_gate(
            stage="hmrc_cgt_asset_type_spine",
            check="cgt_asset_type_summary",
            evidence={"stage": "hmrc_cgt_asset_type_spine"},
            parameters=parameters,
        )
