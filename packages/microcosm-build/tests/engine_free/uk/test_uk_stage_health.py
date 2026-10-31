from __future__ import annotations

import json

import numpy as np
import pytest

from microcosm.build.uk_runtime.stage_health import uk_stage_health_gate
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


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


def _student_loan_receipt(**overrides):
    """A PLAN_2 receipt the walk produced: 3 of 6 eligible rows taken, 2 skipped."""
    receipt = {
        "stock": 100.0,
        "reported_count": 70.0,
        "reported_england_count": 60.0,
        "shortfall": 40.0,
        "eligible_rows": 6,
        "eligible_mass": 60.0,
        "topped_up_rows": 3,
        "topped_up_mass": 39.0,
        "rows_skipped_for_weight": 2,
        "lightest_skipped_weight": 5.0,
        "realization_gap": -1.0,
        "pool_exhausted": False,
        "final_england_count": 99.0,
        "stock_attainment": 0.99,
    }
    receipt.update(overrides)
    return receipt


def test_student_loan_gate_holds_the_walk_bound_and_the_stock() -> None:
    """The gate checks what the walk controls and fails closed (microcosm#1049)."""
    parameters = {
        "stage": "student_loans",
        "check": "student_loan_plans",
        "stocks": {"PLAN_2": 100.0},
        "maximum_stock_relative_deviation": 0.02,
    }

    def run(receipt, params=parameters):
        return uk_stage_health_gate(
            evidence={"stage": "student_loans", "plans": {"PLAN_2": receipt}},
            stage="student_loans",
            check="student_loan_plans",
            parameters=params,
        )

    passed = run(_student_loan_receipt())
    assert _passed(passed)
    assert passed.details["plans"]["PLAN_2"]["regime"] == "walked_to_stock"
    assert passed.details["worst_abs_realization_gap"] == 1.0
    # The declared stock is live.
    assert not run(
        _student_loan_receipt(), {**parameters, "stocks": {"PLAN_2": 99.0}}
    ).passed
    # The walk bound: a gap at or beyond the lightest skipped weight.
    assert not run(_student_loan_receipt(lightest_skipped_weight=1.0)).passed
    # A fit claimed without a skipped person to bound it.
    assert not run(
        _student_loan_receipt(rows_skipped_for_weight=0, lightest_skipped_weight=None)
    ).passed
    # The final count against the stock, at the declared tolerance.
    assert not run(
        _student_loan_receipt(),
        {**parameters, "maximum_stock_relative_deviation": 0.005},
    ).passed
    # Self-consistency: a tampered gap or final count.
    assert not run(_student_loan_receipt(realization_gap=-2.0)).passed
    assert not run(_student_loan_receipt(final_england_count=98.0)).passed
    # A pool receipted as exhausted must have been taken whole ...
    exhausted = _student_loan_receipt(
        shortfall=140.0,
        stock=200.0,
        eligible_rows=6,
        eligible_mass=60.0,
        topped_up_rows=6,
        topped_up_mass=60.0,
        rows_skipped_for_weight=0,
        lightest_skipped_weight=None,
        realization_gap=-80.0,
        pool_exhausted=True,
        final_england_count=120.0,
        stock_attainment=0.6,
    )
    whole = run(exhausted, {**parameters, "stocks": {"PLAN_2": 200.0}})
    assert _passed(whole)
    assert whole.details["plans"]["PLAN_2"]["regime"] == "pool_exhausted"
    assert whole.details["plans"]["PLAN_2"]["stock_attainment"] == 0.6
    # ... and fails when it was not.
    assert not run(
        {
            **exhausted,
            "topped_up_rows": 5,
            "topped_up_mass": 55.0,
            "realization_gap": -85.0,
            "final_england_count": 115.0,
        },
        {**parameters, "stocks": {"PLAN_2": 200.0}},
    ).passed
    # A plan at or above its stock is left alone; topping it up fails.
    reported = _student_loan_receipt(
        shortfall=0.0,
        reported_england_count=105.0,
        topped_up_rows=0,
        topped_up_mass=0.0,
        rows_skipped_for_weight=0,
        lightest_skipped_weight=None,
        realization_gap=0.0,
        final_england_count=105.0,
        stock_attainment=1.05,
    )
    above = run(reported)
    assert _passed(above)
    assert above.details["plans"]["PLAN_2"]["regime"] == "reported_at_or_above_stock"
    assert not run(
        {
            **reported,
            "topped_up_rows": 1,
            "topped_up_mass": 2.0,
            "realization_gap": 2.0,
            "final_england_count": 107.0,
        }
    ).passed
    # A partial receipt fails closed.
    with pytest.raises(ValueError, match="pool_exhausted"):
        run({k: v for k, v in _student_loan_receipt().items() if k != "pool_exhausted"})
    with pytest.raises(ValueError, match="eligible_rows"):
        run(_student_loan_receipt(eligible_rows=-1))


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


def _support_split_receipt() -> dict[str, object]:
    """A licensed-scale support split receipt: every column met its rule."""

    published = {0: 17_000.0, 37_700: 5_000.0, 50_000: 11_000.0, 100_000: 3_000.0}
    rows = []
    for lower, count in published.items():
        support = 2 * 2.0 * count
        rows.append(
            {
                "income_lower_bound": float(lower),
                "published_top_band_taxpayers": count,
                "suppressed_cells": 1 if lower == 100_000 else 0,
                "support_mass": support,
                "pool_households": 4_000,
                "pool_mass": 4_000_000.0,
                "households_selected": 70,
                "copies_created": 1_120,
                "selected_mass": support + 500.0,
                "wealth_threshold": 2_500_000.0,
                "heaviest_selected_weight": 1_900.0,
                "heaviest_copy_weight": 59.4,
                "pool_exhausted": False,
            }
        )
    return {
        "stage": "cgt_support_split",
        "bands": rows,
        "totals": {
            "published_top_band_taxpayers": sum(published.values()),
            "suppressed_cells": 1,
            "support_mass": sum(row["support_mass"] for row in rows),
            "households_selected": 280,
            "copies_created": 4_480,
            "selected_mass": sum(row["selected_mass"] for row in rows),
            "households_before": 26_768,
            "households_after": 31_248,
            "zero_weight_excluded": 0,
        },
        "mass": {"old_total": 29_422_433.0, "new_total": 29_422_433.0},
        "parameters": {
            "clone_split_factor": 2,
            "headroom": 2.0,
            "maximum_copy_weight": 60.0,
        },
    }


_SUPPORT_SPLIT_PARAMETERS = {
    "stage": "cgt_support_split",
    "check": "cgt_support_split",
    "clone_split_factor": 2,
    "headroom": 2.0,
    "maximum_copy_weight": 60.0,
    "maximum_relative_mass_deviation": 1e-9,
}


def _support_split_gate(evidence: dict[str, object]):
    return uk_stage_health_gate(
        evidence=evidence,
        stage="cgt_support_split",
        check="cgt_support_split",
        parameters=_SUPPORT_SPLIT_PARAMETERS,
    )


def test_cgt_support_split_gate_passes_a_conforming_receipt() -> None:
    result = _support_split_gate(_support_split_receipt())

    assert _passed(result)
    assert result.details["exhausted_columns"] == []
    assert result.details["copies_created"] == 4_480


def test_cgt_support_split_gate_tolerates_a_recorded_exhausted_column() -> None:
    """A column lighter than its support mass selects its whole pool and says so."""

    evidence = _support_split_receipt()
    rows = [dict(row) for row in evidence["bands"]]
    rows[-1].update(
        {
            "pool_households": 3,
            "pool_mass": 4.5,
            "households_selected": 3,
            "copies_created": 0,
            "selected_mass": 4.5,
            "heaviest_selected_weight": 1.7,
            "heaviest_copy_weight": 1.7,
            "pool_exhausted": True,
        }
    )
    totals = dict(evidence["totals"])
    totals["households_selected"] = 210 + 3
    totals["copies_created"] = 3_360
    totals["selected_mass"] = sum(row["selected_mass"] for row in rows)
    result = _support_split_gate({**evidence, "bands": rows, "totals": totals})

    assert _passed(result)
    assert result.details["exhausted_columns"] == [100_000.0]


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (
            lambda e: e["mass"].__setitem__("new_total", 29_422_433.0 * (1 + 1e-6)),
            "mass deviation",
        ),
        (
            lambda e: e["bands"][0].__setitem__("heaviest_copy_weight", 60.5),
            "exceeds the maximum",
        ),
        (
            lambda e: e["bands"][0].__setitem__("support_mass", 68_001.0),
            "differs from",
        ),
        (
            lambda e: e["bands"][1].__setitem__("selected_mass", 19_000.0),
            "falls short",
        ),
        (
            lambda e: e["bands"][2].__setitem__("pool_exhausted", True),
            "did not select its whole pool",
        ),
        (
            lambda e: e["totals"].__setitem__("copies_created", 4_481),
            "differs from the column sum",
        ),
        (
            lambda e: e["parameters"].__setitem__("headroom", 1.5),
            "differs from the gate's",
        ),
    ],
)
def test_cgt_support_split_gate_fails_closed(mutate, fragment: str) -> None:
    evidence = _support_split_receipt()
    mutate(evidence)
    result = _support_split_gate(evidence)

    assert result.passed is False
    assert fragment in " ".join(result.failures)


def test_cgt_support_split_gate_refuses_a_missing_field() -> None:
    evidence = _support_split_receipt()
    del evidence["bands"][0]["heaviest_copy_weight"]

    with pytest.raises(ValueError, match="heaviest_copy_weight"):
        _support_split_gate(evidence)


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
                "rake": {
                    "ipf_max_abs_margin_error": error,
                    "gains_margin_max_abs_error": error,
                    "ipf_zero_seed_cells": 0,
                },
                "fallback_released_mass": released,
                "remainder": {
                    "persons": 3,
                    "mass": 300.0,
                    "annual_exempt_amount": 3000.0,
                    "min_amount": 12.5,
                    "max_amount": 2990.0,
                },
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


def test_cgt_summary_remainder_must_stay_inside_the_exempt_range() -> None:
    """The sub-AEA remainder receipt (microcosm#970) is fenced on its range."""
    parameters = {
        "stage": "hmrc_cgt_gains_spine",
        "check": "cgt_imputation_summary",
        "minimum_band_rows": 1,
    }

    def evidence(remainder: dict) -> dict:
        return {
            "stage": "hmrc_cgt_gains_spine",
            "rows": [{"gain_lower_bound": 12300.0}],
            "taxpayer_mass": 1.0,
            "published_taxpayer_mass": 1.0,
            "remainder_mass": 0.0,
            "allocation": {
                "rake": {
                    "ipf_max_abs_margin_error": 0.0,
                    "gains_margin_max_abs_error": 0.0,
                    "ipf_zero_seed_cells": 0,
                },
                "fallback_released_mass": 0.0,
                "remainder": remainder,
            },
        }

    def verdict(remainder: dict):
        return uk_stage_health_gate(
            evidence=evidence(remainder),
            stage="hmrc_cgt_gains_spine",
            check="cgt_imputation_summary",
            parameters=parameters,
        )

    inside = {
        "persons": 2,
        "mass": 200.0,
        "annual_exempt_amount": 3000.0,
        "min_amount": 1.0,
        "max_amount": 3000.0,
    }
    assert _passed(verdict(inside))
    assert not verdict({**inside, "max_amount": 3000.5}).passed
    assert not verdict({**inside, "min_amount": 0.0}).passed
    assert not verdict({**inside, "mass": -1.0}).passed
    # An empty remainder carries zero amounts and passes.
    assert _passed(
        verdict(
            {
                "persons": 0,
                "mass": 0.0,
                "annual_exempt_amount": 3000.0,
                "min_amount": 0.0,
                "max_amount": 0.0,
            }
        )
    )
    with pytest.raises(ValueError):
        verdict({**inside, "mass": float("nan")})


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


def test_support_clip_gate_holds_the_declared_donor_floor() -> None:
    """microcosm#1063 c9: where the gate declares a donor floor, the receipt
    must show it applied, no negative donor row left and no clip range below it."""

    def evidence(**floor_changes):
        return {
            "stage": "lcfs_consumption",
            "support_clip": {
                "columns": {
                    "housing_water_and_electricity_consumption": {
                        "donor_min": 0.0,
                        "donor_max": 9_000.0,
                        "clipped_low_rows": 0,
                        "clipped_high_rows": 0,
                        "rows_considered": 2,
                    }
                }
            },
            "donor_floor": {
                "floor": 0.0,
                "rows_raised": 250,
                "remaining_negative_rows": 0,
                "columns": {
                    "housing_water_and_electricity_consumption": {
                        "rows_raised": 250,
                    }
                },
                **floor_changes,
            },
        }

    parameters = {
        "stage": "lcfs_consumption",
        "check": "support_clip",
        "columns": ["housing_water_and_electricity_consumption"],
        "max_clipped_low_rows_by_column": {
            "housing_water_and_electricity_consumption": 0
        },
        "max_clipped_high_rows_by_column": {
            "housing_water_and_electricity_consumption": 0
        },
        "donor_floor": 0.0,
    }

    def gate(evidence_payload, **parameter_changes):
        return uk_stage_health_gate(
            evidence=evidence_payload,
            stage="lcfs_consumption",
            check="support_clip",
            parameters={**parameters, **parameter_changes},
        )

    passed = gate(evidence())
    assert _passed(passed)
    assert passed.details["donor_floor_rows_raised"] == 250
    assert any(
        "donor_floor receipt" in f
        for f in gate({**evidence(), "donor_floor": None}).failures
    )
    assert any(
        "negative consumption row" in f
        for f in gate(evidence(remaining_negative_rows=3)).failures
    )
    assert any("not the declared" in f for f in gate(evidence(floor=-1.0)).failures)
    below = evidence()
    below["support_clip"]["columns"]["housing_water_and_electricity_consumption"][
        "donor_min"
    ] = -50.0
    assert any("below the donor floor" in f for f in gate(below).failures)
    # Without a declared floor (the WAS clip gate) the receipt is not required.
    undeclared = {**parameters}
    del undeclared["donor_floor"]
    assert _passed(
        uk_stage_health_gate(
            evidence={**evidence(), "donor_floor": None},
            stage="lcfs_consumption",
            check="support_clip",
            parameters=undeclared,
        )
    )


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


def _wealth_coherence_evidence() -> dict:
    return {
        "stage": "was_wealth",
        "tenure_coherence": {
            "main_residence_mortgage_off_mortgaged_tenure_rows": 0,
            "main_residence_value_off_owner_tenure_rows": 0,
            "owner_share_without_main_residence_value": 0.001,
            "donor_owner_share_without_main_residence_value": 0.0,
        },
        "identities": {
            "property_wealth_violation_rows": 0,
            "corporate_wealth_violation_rows": 0,
            "gross_financial_wealth_violation_rows": 0,
            "net_financial_wealth_violation_rows": 0,
            "mortgage_debt_violation_rows": 0,
            "share_split_violation_rows": 0,
            "secured_debt_violation_rows": 0,
            "property_wealth_capped_rows": 3,
        },
    }


_WEALTH_COHERENCE_PARAMETERS = {
    "stage": "was_wealth",
    "check": "wealth_coherence",
    "maximum_owner_share_without_main_residence_excess": 0.005,
}


def _wealth_coherence(evidence: dict):
    return uk_stage_health_gate(
        evidence=evidence,
        stage="was_wealth",
        check="wealth_coherence",
        parameters=_WEALTH_COHERENCE_PARAMETERS,
    )


def test_wealth_coherence_gate_passes_a_coherent_receipt() -> None:
    result = _wealth_coherence(_wealth_coherence_evidence())

    assert _passed(result)
    assert result.details["main_residence_mortgage_off_mortgaged_tenure_rows"] == 0
    # The donor-range cap is recorded, never a failure.
    assert result.details["capped_rows"] == {"property_wealth_capped_rows": 3}


@pytest.mark.parametrize(
    ("block", "key"),
    [
        ("tenure_coherence", "main_residence_mortgage_off_mortgaged_tenure_rows"),
        ("tenure_coherence", "main_residence_value_off_owner_tenure_rows"),
        ("identities", "property_wealth_violation_rows"),
        ("identities", "corporate_wealth_violation_rows"),
        ("identities", "gross_financial_wealth_violation_rows"),
        ("identities", "net_financial_wealth_violation_rows"),
        ("identities", "mortgage_debt_violation_rows"),
        ("identities", "share_split_violation_rows"),
        ("identities", "secured_debt_violation_rows"),
    ],
)
def test_wealth_coherence_gate_requires_every_count_and_requires_it_zero(
    block: str, key: str
) -> None:
    evidence = _wealth_coherence_evidence()
    evidence[block][key] = 2
    failed = _wealth_coherence(evidence)
    assert not _passed(failed)
    assert f"{key} is 2, expected 0" in failed.failures[0]

    del evidence[block][key]
    missing = _wealth_coherence(evidence)
    assert not _passed(missing)
    assert f"receipt is missing {key}" in missing.failures[0]


def test_wealth_coherence_gate_bounds_owners_without_a_main_residence() -> None:
    evidence = _wealth_coherence_evidence()
    evidence["tenure_coherence"]["owner_share_without_main_residence_value"] = 0.084
    failed = _wealth_coherence(evidence)
    assert not _passed(failed)
    assert "no main-residence value" in failed.failures[0]

    # The bound is on the excess over the donor's own share.
    evidence["tenure_coherence"]["donor_owner_share_without_main_residence_value"] = (
        0.08
    )
    assert _passed(_wealth_coherence(evidence))


def test_wealth_coherence_gate_fails_closed_without_its_receipt_blocks() -> None:
    result = _wealth_coherence({"stage": "was_wealth", "support_clip": {}})

    assert not _passed(result)
    assert any("missing tenure_coherence" in f for f in result.failures)
    assert any("missing identities" in f for f in result.failures)


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


def _residential_bands() -> list[dict[str, object]]:
    """Two gain bands whose achieved masses sum to the residential receipt's."""

    return [
        {
            "gain_lower_bound": 0.0,
            "gain_upper_bound": 250_000.0,
            "liable_rows": 2_900,
            "achieved_rows": 2_600,
            "achieved_count": 190_000.0,
            "achieved_gains": 8.0e9,
        },
        {
            "gain_lower_bound": 250_000.0,
            "gain_upper_bound": None,
            "liable_rows": 900,
            "achieved_rows": 777,
            "achieved_count": 12_630.0,
            "achieved_gains": 4.24e9,
        },
    ]


def _asset_type_evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "stage": "hmrc_cgt_asset_type_spine",
        "residential": {
            "source_stage": "cgt_residential_split",
            "count_target_individuals_basis": 202_630.0,
            "gains_target_individuals_basis": 12.24e9,
            "achieved_count": 202_630.0,
            "achieved_gains": 12.24e9,
            "achieved_rows": 3_377,
            "count_relative_error": 0.0,
            "gains_relative_error": 0.0,
            "max_liable_weight": 60.0,
            "bands": _residential_bands(),
        },
        "asset_type": {
            "achieved_gains_share": {
                "listed_shares": 0.094,
                "unlisted_shares": 0.528,
                "other_financial_assets": 0.282,
                "agricultural_commercial_industrial_land_buildings": 0.044,
                "other_non_financial_assets": 0.052,
            },
            "share_fit_converged": True,
        },
        "badr": _badr_receipt(),
        "value_counts": {
            "none": 4_043,
            "sub_aea": 1_309,
            "residential_land_buildings": 3_377,
        },
    }
    evidence.update(overrides)
    return evidence


def _badr_receipt(**overrides: object) -> dict[str, object]:
    receipt: dict[str, object] = {
        "lifetime_limit": 1_000_000.0,
        "bands": [
            {"lower_bound": 0, "upper_bound": 10_000, "skipped": True},
            {
                "lower_bound": 100_000,
                "upper_bound": 250_000,
                "skipped": False,
                "qualifying_amount": "net_gain",
                "count_target": 11_000.0,
                "gains_target": 1.866e9,
                "expected_count": 11_000.0,
                "expected_gains": 1.866e9,
                "achieved_count": 11_050.0,
                "achieved_gains": 1.87e9,
                "max_pool_weight": 1_496.0,
                "pool_min_gain": 101_000.0,
                "pool_max_gain": 249_000.0,
            },
            {
                "lower_bound": 1_000_000,
                "upper_bound": None,
                "skipped": False,
                "qualifying_amount": "lifetime_limit",
                "count_target": 6_787.0,
                "gains_target": 6.787e9,
                "expected_count": 6_787.0,
                "expected_gains": 6.787e9,
                "achieved_count": 6_800.0,
                "achieved_gains": 6.8e9,
                "max_pool_weight": 873.0,
                "pool_min_gain": 1_000_000.0,
                "pool_max_gain": 169.0e6,
            },
        ],
        "totals": {
            "achieved_count": 17_850.0,
            "achieved_gains": 8.67e9,
            "relief_rate_tax": 0.86e9,
        },
        "invariants": {
            "claimants_outside_pool": 0,
            "qualifying_above_gain": 0,
            "qualifying_above_limit": 0,
            "qualifying_outside_band": 0,
            "residential_overlap": 0,
            "sub_aea_claimants": 0,
        },
    }
    receipt.update(overrides)
    return receipt


def test_cgt_asset_type_summary_holds_the_residential_identities() -> None:
    parameters = {
        "stage": "hmrc_cgt_asset_type_spine",
        "check": "cgt_asset_type_summary",
        "maximum_solve_relative_error": 1e-6,
    }

    def gate(evidence: dict[str, object]):
        return uk_stage_health_gate(
            stage="hmrc_cgt_asset_type_spine",
            check="cgt_asset_type_summary",
            evidence=evidence,
            parameters=parameters,
        )

    def with_residential(**changes: object) -> dict[str, object]:
        return _asset_type_evidence(
            residential={**_asset_type_evidence()["residential"], **changes}
        )

    passed = gate(_asset_type_evidence())
    assert passed.passed
    assert passed.details["residential_count_relative_error"] == 0.0
    assert passed.details["residential_gains_relative_error"] == 0.0
    assert passed.details["residential_gain_bands"] == 2

    # The arms carry Table 8a at design weights: a departure beyond the solve
    # tolerance is not a realisation gap but a broken split.
    failed = gate(with_residential(achieved_count=202_500.0))
    assert not failed.passed
    assert any("on the arms" in failure for failure in failed.failures)
    failed = gate(with_residential(achieved_gains=11.9e9))
    assert not failed.passed
    assert any("gains on the arms" in failure for failure in failed.failures)

    # The receipt must come from the split stage and restate it by band.
    failed = gate(with_residential(source_stage="hmrc_cgt_asset_type_spine"))
    assert not failed.passed
    assert any("cgt_residential_split" in failure for failure in failed.failures)
    bands = _residential_bands()
    bands[0]["achieved_gains"] = 7.0e9
    failed = gate(with_residential(bands=bands))
    assert not failed.passed
    assert any("sums to" in failure for failure in failed.failures)
    failed = gate(with_residential(bands=[]))
    assert not failed.passed
    assert any("no gain bands" in failure for failure in failed.failures)

    bad_share = _asset_type_evidence(
        asset_type={"achieved_gains_share": {"listed_shares": 1.5}}
    )
    failed = gate(bad_share)
    assert not failed.passed

    with pytest.raises(ValueError, match="residential"):
        uk_stage_health_gate(
            stage="hmrc_cgt_asset_type_spine",
            check="cgt_asset_type_summary",
            evidence={"stage": "hmrc_cgt_asset_type_spine"},
            parameters=parameters,
        )


def _residential_split_evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "stage": "cgt_residential_split",
        "mass": {"old_total": 28_600_000.0, "new_total": 28_600_000.0},
        "households_before": 58_288,
        "households_after": 60_600,
        "arms_created": 2_312,
        "households_by_liable_gainers": {"1": 2_300, "2": 4},
        "arm_weights_exact": True,
        "identities": {
            "count_target_individuals_basis": 202_630.0,
            "gains_target_individuals_basis": 12.24e9,
            "expected_count": 202_630.0,
            "expected_gains": 12.24e9,
            "achieved_count": 202_630.0,
            "achieved_gains": 12.24e9,
            "count_solve_relative_error": 1e-15,
            "gains_solve_relative_error": 7e-10,
            "count_identity_relative_error": 0.0,
            "gains_identity_relative_error": 0.0,
        },
        "bands": [
            {
                "gain_lower_bound": 0.0,
                "gain_upper_bound": 250_000.0,
                "expected_count": 190_000.0,
                "expected_gains": 8.0e9,
                "achieved_count": 190_000.0,
                "achieved_gains": 8.0e9,
            },
            {
                "gain_lower_bound": 250_000.0,
                "gain_upper_bound": None,
                "expected_count": 12_630.0,
                "expected_gains": 4.24e9,
                "achieved_count": 12_630.0,
                "achieved_gains": 4.24e9,
            },
        ],
        "arm_weights": {
            "residential_arms": 2_312,
            "minimum": 0.04,
            "maximum": 700.0,
            "below_one_household": 300,
            "below_one_tenth": 12,
        },
        "concentration": {
            "top_arms_share_of_achieved_gains": 0.12,
            "largest_liable_stake_share_of_gains_target": 0.89,
        },
    }
    evidence.update(overrides)
    return evidence


def test_cgt_residential_split_holds_the_identities_and_the_arm_structure() -> None:
    parameters = {
        "stage": "cgt_residential_split",
        "check": "cgt_residential_split",
        "maximum_relative_mass_deviation": 1e-9,
        "maximum_solve_relative_error": 1e-6,
        "maximum_identity_relative_error": 1e-9,
        "maximum_liable_gainers_per_household": 3,
    }

    def gate(evidence: dict[str, object]):
        return uk_stage_health_gate(
            stage="cgt_residential_split",
            check="cgt_residential_split",
            evidence=evidence,
            parameters=parameters,
        )

    def with_identities(**changes: object) -> dict[str, object]:
        return _residential_split_evidence(
            identities={**_residential_split_evidence()["identities"], **changes}
        )

    passed = gate(_residential_split_evidence())
    assert _passed(passed)
    assert passed.details["arms_created"] == 2_312
    assert passed.details["arm_below_one_household"] == 300
    assert passed.details["top_arms_share_of_achieved_gains"] == 0.12

    # Mass created or lost, arms not the declared products, or an arm count
    # that does not follow from the households by liable gainers.
    failed = gate(
        _residential_split_evidence(
            mass={"old_total": 28_600_000.0, "new_total": 28_600_100.0}
        )
    )
    assert not failed.passed and "mass deviation" in " ".join(failed.failures)
    failed = gate(_residential_split_evidence(arm_weights_exact=False))
    assert not failed.passed and "declared products" in " ".join(failed.failures)
    failed = gate(_residential_split_evidence(arms_created=2_311))
    assert not failed.passed and "imply 2312" in " ".join(failed.failures)
    failed = gate(
        _residential_split_evidence(households_by_liable_gainers={"1": 2_300, "4": 1})
    )
    assert not failed.passed and "above the declared maximum" in " ".join(
        failed.failures
    )
    # The solve must reach the targets and the arms must realise it exactly.
    failed = gate(with_identities(count_solve_relative_error=1e-3))
    assert not failed.passed and "solve error" in " ".join(failed.failures)
    failed = gate(with_identities(gains_identity_relative_error=1e-6))
    assert not failed.passed and "departs from its expectation" in " ".join(
        failed.failures
    )
    bands = _residential_split_evidence()["bands"]
    bands[1] = {**bands[1], "achieved_gains": 4.0e9}
    failed = gate(_residential_split_evidence(bands=bands))
    assert not failed.passed and "gain band from 250000.0" in " ".join(failed.failures)
    failed = gate(_residential_split_evidence(bands=[]))
    assert not failed.passed and "no gain bands" in " ".join(failed.failures)


def test_cgt_asset_type_summary_holds_every_badr_band_to_the_walk_bound() -> None:
    parameters = {
        "stage": "hmrc_cgt_asset_type_spine",
        "check": "cgt_asset_type_summary",
        "maximum_solve_relative_error": 1e-6,
    }

    def gate(evidence: dict[str, object]):
        return uk_stage_health_gate(
            stage="hmrc_cgt_asset_type_spine",
            check="cgt_asset_type_summary",
            evidence=evidence,
            parameters=parameters,
        )

    passed = gate(_asset_type_evidence())
    assert passed.passed
    assert passed.details["badr_bands_checked"] == 2

    def with_band(index: int, **changes: object) -> dict[str, object]:
        receipt = _badr_receipt()
        bands = [dict(band) for band in receipt["bands"]]  # type: ignore[union-attr]
        bands[index].update(changes)
        return _asset_type_evidence(badr=_badr_receipt(bands=bands))

    # More than the band pool's largest weight off the expected count.
    failed = gate(with_band(1, achieved_count=11_000.0 + 1_500.0))
    assert not failed.passed
    assert any("count gap" in failure for failure in failed.failures)
    # Qualifying gains beyond max weight x (2 max gain - min gain).
    failed = gate(with_band(1, achieved_gains=1.866e9 + 0.6e9))
    assert not failed.passed
    assert any("walk's bound" in failure for failure in failed.failures)
    # The top band's gains must be exactly the limit times the realised count.
    failed = gate(with_band(2, achieved_gains=6.8e9 + 1.0e6))
    assert not failed.passed
    assert any("lifetime limit" in failure for failure in failed.failures)
    # A solve that missed its published target.
    failed = gate(with_band(1, expected_gains=1.9e9))
    assert not failed.passed
    assert any("solve error" in failure for failure in failed.failures)
    # A skipped band is not held to anything.
    assert gate(with_band(0, achieved_count=1.0e9)).passed

    broken = _badr_receipt()
    broken["invariants"] = {**broken["invariants"], "residential_overlap": 2}  # type: ignore[dict-item]
    failed = gate(_asset_type_evidence(badr=broken))
    assert not failed.passed
    assert any("residential_overlap" in failure for failure in failed.failures)

    unconverged = _asset_type_evidence()
    unconverged["asset_type"] = {
        **unconverged["asset_type"],  # type: ignore[dict-item]
        "share_fit_converged": False,
    }
    failed = gate(unconverged)
    assert not failed.passed
    assert any("did not converge" in failure for failure in failed.failures)

    no_bands = gate(_asset_type_evidence(badr=_badr_receipt(bands=[])))
    assert not no_bands.passed
    missing = _asset_type_evidence()
    del missing["badr"]
    with pytest.raises(ValueError, match="badr"):
        gate(missing)


def _anchor_evidence(**overrides: object) -> dict[str, object]:
    evidence: dict[str, object] = {
        "stage": "cgt_incidence_anchor",
        "liable_mass": 551_600.0,
        "transferred_mass": 11_921_000.0,
        "pair_count": 60_000,
        "max_pair_relative_error": 0.0,
        "targets": {"sub_exempt": 43_000.0, "loss": 136_000.0},
        "before": {
            "sub_exempt": 10_800_000.0,
            "loss": 1_300_000.0,
            "liable": 159_000.0,
        },
        "after": {"sub_exempt": 43_000.0, "loss": 136_000.0, "liable": 159_000.0},
        "mass_by_clone_flag": {"false": 40_000_000.0, "true": 338_000.0},
    }
    evidence.update(overrides)
    return evidence


_ANCHOR_PARAMETERS = {
    "stage": "cgt_incidence_anchor",
    "check": "cgt_incidence_anchor",
    "maximum_relative_composition_error": 1e-6,
    "maximum_pair_relative_error": 0.0,
    "minimum_pair_count": 1,
}


def _anchor_gate(evidence: dict[str, object]):
    return uk_stage_health_gate(
        stage="cgt_incidence_anchor",
        check="cgt_incidence_anchor",
        evidence=evidence,
        parameters=_ANCHOR_PARAMETERS,
    )


def test_cgt_incidence_anchor_gate_holds_the_composition_and_the_pairs() -> None:
    passed = _anchor_gate(_anchor_evidence())
    assert passed.passed
    assert passed.details["sub_exempt_relative_error"] == 0.0
    assert passed.details["liable_clone_mass"] == 159_000.0
    assert passed.details["pair_count"] == 60_000

    # A group already at or below its target stays where it was.
    untouched = _anchor_evidence(
        targets={"sub_exempt": 43_000.0, "loss": 2_000_000.0},
        after={"sub_exempt": 43_000.0, "loss": 1_300_000.0, "liable": 159_000.0},
        transferred_mass=10_757_000.0,
    )
    assert _anchor_gate(untouched).passed

    def fails(match: str, **overrides: object) -> None:
        result = _anchor_gate(_anchor_evidence(**overrides))
        assert not result.passed
        assert any(match in failure for failure in result.failures), result.failures

    fails(
        "misses its target",
        after={"sub_exempt": 43_100.0, "loss": 136_000.0, "liable": 159_000.0},
        transferred_mass=11_920_900.0,
    )
    fails(
        "liable clone mass moved",
        after={"sub_exempt": 43_000.0, "loss": 136_000.0, "liable": 158_000.0},
    )
    fails("pair mass error", max_pair_relative_error=1e-9)
    fails("pairs is below", pair_count=0)
    fails(
        "but moved to",
        targets={"sub_exempt": 43_000.0, "loss": 2_000_000.0},
        after={"sub_exempt": 43_000.0, "loss": 1_200_000.0, "liable": 159_000.0},
        transferred_mass=10_857_000.0,
    )
    fails(
        "clone mass rose",
        before={"sub_exempt": 40_000.0, "loss": 1_300_000.0, "liable": 159_000.0},
        transferred_mass=1_161_000.0,
    )
    fails("disagrees with the transferred mass", transferred_mass=1.0)
    fails("exceeds original mass", mass_by_clone_flag={"false": 1.0, "true": 2.0})
    fails("liable mass must be positive", liable_mass=0.0)

    with pytest.raises(ValueError, match="targets"):
        _anchor_gate(
            {
                "stage": "cgt_incidence_anchor",
                "liable_mass": 1.0,
                "transferred_mass": 0.0,
                "max_pair_relative_error": 0.0,
                "pair_count": 1,
            }
        )
    with pytest.raises(ValueError, match="pair_count"):
        _anchor_gate(_anchor_evidence(pair_count=1.5))


def _gate_parameters(gate_id: str) -> dict:
    gates = json.loads(
        (_TEST_PATHS.package / "src/microcosm/build/uk/gates.json").read_text("utf-8")
    )
    entry = next(g for g in gates["gates"] if g["id"] == gate_id)
    assert entry["gate"] == "stage_health"
    assert entry["population_fact_check"] is True
    assert entry["evidence_absent_blocks"] is True
    return dict(entry["parameters"])


def test_bus_pricing_gate_recomputes_every_price_from_the_vendored_rows() -> None:
    """The lcfs bus_pricing receipt is fact-checked at stage time (microcosm#930 C6)."""

    import copy

    import pandas as pd

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.bus_fare_pricing import (
        BUS_IN_LONDON,
        OTHER_LOCAL_BUS,
        bus_fare_prices,
        price_bus_journeys,
    )
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_VENDORED_RESOURCES,
        bus_pricing_operation,
    )

    parameters = _gate_parameters("uk_stage_lcfs_consumption_bus_pricing")
    assert parameters["check"] == "bus_pricing"
    declared = bus_pricing_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    prices = bus_fare_prices(declared, allowed_resources=UK_LCFS_VENDORED_RESOURCES)
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5],
            "region": ["LONDON", "SOUTH_EAST", "WALES", "SCOTLAND", "NORTHERN_IRELAND"],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 12, 21, 31, 41, 51],
            "person_household_id": [1, 1, 2, 3, 4, 5],
            "bus_in_london_trips": [100.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            "other_local_bus_trips": [0.0, 10.0, 40.0, 30.0, 20.0, 25.0],
            "bus_pass_eligible": [False, True, False, False, False, True],
        }
    )
    _, _, receipt = price_bus_journeys(
        person,
        household,
        prices=prices,
        trips_columns={
            BUS_IN_LONDON: "bus_in_london_trips",
            OTHER_LOCAL_BUS: "other_local_bus_trips",
        },
        eligibility_column="bus_pass_eligible",
        household_weights=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        raw_household_fares=np.array([500.0, 600.0, 700.0, 800.0, 900.0]),
    )
    receipt = {"chain_conditioned_on": "raw_draw", **prices.receipt, **receipt}

    def run(ev):
        return uk_stage_health_gate(
            evidence={"stage": "lcfs_consumption", "bus_pricing": ev},
            stage="lcfs_consumption",
            check="bus_pricing",
            parameters=parameters,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    assert passed.details["areas_fact_checked"] == 1 + len(
        {a.label for a in prices.other_by_region.values()}
    )
    assert set(passed.details["frame_implied_over_published_boardings"]) >= {
        "london_series",
        "england_outside_london",
    }
    tampered = copy.deepcopy(receipt)
    tampered["prices"]["london_series"]["yield_per_fare_paying_boarding"] *= 1.01
    result = run(tampered)
    assert not result.passed and any(
        "yield_per_fare_paying_boarding" in f for f in result.failures
    ), result.failures
    tampered = copy.deepcopy(receipt)
    tampered["prices"]["england_outside_london"]["boardings"]["value"] *= 1.01
    result = run(tampered)
    assert not result.passed and any(
        "boardings" in f and "not the vendored" in f for f in result.failures
    ), result.failures
    tampered = copy.deepcopy(receipt)
    tampered["prices"]["england_outside_london"]["calendar_price_factor"] *= 1.01
    result = run(tampered)
    assert not result.passed and any(
        "calendar_price_factor" in f for f in result.failures
    ), result.failures
    tampered = copy.deepcopy(receipt)
    tampered["chain_conditioned_on"] = "priced"
    result = run(tampered)
    assert not result.passed and any("raw draw" in f for f in result.failures)
    tampered = copy.deepcopy(receipt)
    tampered["unpriced_regions"] = ["WALES", "SCOTLAND"]
    result = run(tampered)
    assert not result.passed and any("unpriced regions" in f for f in result.failures)
    with pytest.raises(ValueError, match="bus_pricing must be an object"):
        run(None)


def test_road_fuel_level_gate_recomputes_the_level_from_the_vendored_row() -> None:
    """The lcfs road_fuel_level receipt is fact-checked at stage time (microcosm#1113)."""

    import pandas as pd

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.calibration_run import UK_SPINE_GATE_SCOPE
    from microcosm.build.uk_runtime.lcfs_consumption import (
        level_road_fuel,
        road_fuel_level,
        road_fuel_level_operation,
    )

    gate_id = "uk_stage_lcfs_consumption_road_fuel_level"
    assert gate_id in UK_SPINE_GATE_SCOPE
    parameters = _gate_parameters(gate_id)
    assert parameters["check"] == "road_fuel_level"
    declared = road_fuel_level_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    draws = pd.DataFrame(
        {"petrol_spending": [900.0, 0.0, 400.0], "diesel_spending": [0.0, 700.0, 0.0]}
    )
    _, receipt = level_road_fuel(
        draws,
        level=road_fuel_level(declared, other_fuels_share=0.004),
        weights=np.array([1.0e7, 1.5e7, 2.0e7]),
    )

    def run(ev):
        return uk_stage_health_gate(
            evidence={"stage": "lcfs_consumption", "road_fuel_level": ev},
            stage="lcfs_consumption",
            check="road_fuel_level",
            parameters=parameters,
        )

    def failing(ev, fragment: str) -> None:
        result = run(ev)
        assert not result.passed and any(fragment in f for f in result.failures), (
            fragment,
            result.failures,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    assert passed.details["level"] == pytest.approx(receipt["level"])
    failing(
        {**receipt, "published": receipt["published"] * 1.01}, "road-fuel published"
    )
    failing({**receipt, "level": receipt["level"] * 1.01}, "road-fuel level")
    failing({**receipt, "frame_after": receipt["frame_after"] * 0.99}, "levelled")
    failing({**receipt, "factor": receipt["factor"] * 1.01}, "does not reproduce")
    failing({**receipt, "source_record_id": "elsewhere"}, "declared row")
    failing({**receipt, "other_fuels_share": 0.2}, "other-fuels share")
    with pytest.raises(ValueError, match="road_fuel_level must be an object"):
        run(None)


def test_consumption_basis_gate_holds_every_spend_level_to_calendar_2024() -> None:
    """Every LCFS spend column is on the H5's year, calendar 2024 (#1113)."""

    import copy

    from microcosm.build.uk_runtime.calibration_run import UK_SPINE_GATE_SCOPE

    gate_id = "uk_stage_lcfs_consumption_basis"
    assert gate_id in UK_SPINE_GATE_SCOPE
    parameters = _gate_parameters(gate_id)
    assert parameters == {
        "stage": "lcfs_consumption",
        "check": "consumption_basis",
        "period_type": "calendar_year",
        "period_value": 2024,
    }
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_DIARY_PRICE_COLUMNS,
    )

    evidence = {
        "stage": "lcfs_consumption",
        "donor_uprating": {
            "from_period": 2023,
            "to_period": 2024,
            "fiscal_year_bridge": {"start_year_weight": 0.75, "next_year_weight": 0.25},
            "columns": {
                column: {"basis": "engine_parameter", "from_window": "fiscal_year"}
                for column in UK_LCFS_DIARY_PRICE_COLUMNS
            },
        },
        "energy_pricing": {
            "period_type": "calendar_year",
            "period_value": 2024,
            "level": {"period_type": "calendar_year", "period_value": 2024},
        },
        "bus_pricing": {"price_year": 2024},
        "road_fuel_level": {"period_type": "calendar_year", "period_value": 2024},
    }

    def run(ev):
        return uk_stage_health_gate(
            evidence=ev,
            stage="lcfs_consumption",
            check="consumption_basis",
            parameters=parameters,
        )

    def failing(mutate, fragment: str) -> None:
        tampered = copy.deepcopy(evidence)
        mutate(tampered)
        result = run(tampered)
        assert not result.passed and any(fragment in f for f in result.failures), (
            fragment,
            result.failures,
        )

    passed = run(evidence)
    assert passed.passed, passed.failures
    assert passed.details["steps"]["energy_level"] == ("calendar_year", 2024)
    assert passed.details["steps"]["donor_uprating"] == ("calendar_year", 2024)
    failing(lambda e: e["donor_uprating"].update(to_period=2025), "donor uprating")
    # A diary column moved by a full year's step lands a quarter past the
    # calendar year, whatever the receipt's target year says.
    failing(
        lambda e: e["donor_uprating"]["columns"]["health_consumption"].pop(
            "from_window"
        ),
        "a quarter past calendar 2024",
    )
    failing(
        lambda e: e["donor_uprating"]["columns"].pop("bus_fare_spending"),
        "a quarter past calendar 2024",
    )
    failing(
        lambda e: e["donor_uprating"]["fiscal_year_bridge"].update(
            start_year_weight=1.0
        ),
        "calendar-year index",
    )
    failing(
        lambda e: e["donor_uprating"].pop("fiscal_year_bridge"), "calendar-year index"
    )
    failing(
        lambda e: e["energy_pricing"].update(period_type="fiscal_year"), "energy prices"
    )
    failing(
        lambda e: e["energy_pricing"]["level"].update(period_type="fiscal_year"),
        "energy level",
    )
    failing(lambda e: e["bus_pricing"].update(price_year=None), "bus fares")
    failing(lambda e: e["road_fuel_level"].update(period_value=2025), "road-fuel level")
    failing(lambda e: e.pop("bus_pricing"), "no bus_pricing receipt")
    with pytest.raises(ValueError, match="calendar year"):
        uk_stage_health_gate(
            evidence=evidence,
            stage="lcfs_consumption",
            check="consumption_basis",
            parameters={**parameters, "period_type": "fiscal_year"},
        )


def test_road_fuel_incidence_gate_holds_every_flagged_household_positive() -> None:
    """The lcfs road_fuel_incidence receipt is checked at stage time (microcosm#1113)."""

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.calibration_run import UK_SPINE_GATE_SCOPE
    from microcosm.build.uk_runtime.lcfs_consumption import (
        road_fuel_incidence_operation,
    )

    gate_id = "uk_stage_lcfs_consumption_road_fuel_incidence"
    assert gate_id in UK_SPINE_GATE_SCOPE
    parameters = _gate_parameters(gate_id)
    assert parameters["check"] == "road_fuel_incidence"
    declared = road_fuel_incidence_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    receipt = {
        **{key: declared[key] for key in ("rule", "seed", "salt", "n_estimators")},
        "regimes": {
            "road_fuel_total": "positive_only",
            "petrol_share_of_road_fuel": "zero_inflated_positive",
        },
        "flagged_households": 100,
        "flagged_zero_before": 34,
        "flagged_zero_share_before": 0.343,
        "flagged_zero_after": 0,
        "unflagged_with_fuel": 0,
        "mean_positive_before": 1500.0,
        "mean_redrawn": 900.0,
    }

    def run(ev):
        return uk_stage_health_gate(
            evidence={"stage": "lcfs_consumption", "road_fuel_incidence": ev},
            stage="lcfs_consumption",
            check="road_fuel_incidence",
            parameters=parameters,
        )

    def failing(ev, fragment: str) -> None:
        result = run(ev)
        assert not result.passed and any(fragment in f for f in result.failures), (
            fragment,
            result.failures,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    assert passed.details["flagged_zero_share_before"] == 0.343
    failing({**receipt, "flagged_zero_after": 1}, "flagged_zero_after is 1")
    failing({**receipt, "unflagged_with_fuel": 2}, "unflagged_with_fuel is 2")
    failing({**receipt, "salt": "other"}, "the redraw ran with salt")
    failing({**receipt, "n_estimators": 4}, "the redraw ran with n_estimators")
    failing(
        {**receipt, "regimes": {"road_fuel_total": "zero_inflated_positive"}},
        "fitted regime",
    )
    failing({**receipt, "flagged_zero_share_before": 0.6}, "outside [0, 0.5]")
    failing({**receipt, "flagged_zero_before": 3.0}, "is not a count")
    # Nothing to redraw needs no fitted model.
    nothing = {**receipt, "flagged_zero_before": 0, "regimes": None}
    assert run({**nothing, "flagged_zero_share_before": 0.0}).passed
    with pytest.raises(ValueError, match="road_fuel_incidence must be an object"):
        run(None)


def test_recomposed_totals_gate_holds_totals_above_their_components() -> None:
    """The lcfs recomposed_totals receipt is checked at stage time (microcosm#1113)."""

    import copy

    import pandas as pd

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.calibration_run import UK_SPINE_GATE_SCOPE
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_COICOP_DIVISION_COLUMNS,
        recompose_from_remainder_operation,
        recompose_parent_totals,
    )

    gate_id = "uk_stage_lcfs_consumption_recomposed_totals"
    assert gate_id in UK_SPINE_GATE_SCOPE
    parameters = _gate_parameters(gate_id)
    assert parameters["check"] == "recomposed_totals"
    declared = recompose_from_remainder_operation(
        load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
    )
    drawn = pd.DataFrame(
        {
            "domestic_energy_consumption": [2000.0, 900.0],
            "liquid_fuel_consumption": [0.0, 0.0],
            "solid_fuel_consumption": [0.0, 0.0],
            "petrol_spending": [800.0, 0.0],
            "diesel_spending": [0.0, 700.0],
            "bus_fare_spending": [100.0, 0.0],
        }
    )
    draws = pd.DataFrame(
        {
            "housing_water_and_electricity_consumption": [7000.0, 600.0],
            "electricity_consumption": [900.0, 600.0],
            "gas_consumption": [700.0, 0.0],
            "liquid_fuel_consumption": [0.0, 0.0],
            "solid_fuel_consumption": [0.0, 0.0],
            "transport_consumption": [2000.0, 1000.0],
            "petrol_spending": [1500.0, 0.0],
            "diesel_spending": [0.0, 900.0],
            "bus_fare_spending": [0.0, 250.0],
        }
    )
    housing = "housing_water_and_electricity_consumption"
    food = "food_and_non_alcoholic_beverages_consumption"
    # Every other division at its donor's level; food far above it, so food
    # alone is levelled (microcosm#1278) and the two totals keep their draws.
    means = dict.fromkeys(UK_LCFS_COICOP_DIVISION_COLUMNS, 100.0)
    means.update({housing: 4600.0, "transport_consumption": 2300.0, food: 1.0})
    for division in UK_LCFS_COICOP_DIVISION_COLUMNS:
        if division not in draws:
            draws[division] = [100.0, 100.0]
    draws[food] = [1000.0, 1000.0]
    _, receipt = recompose_parent_totals(
        draws,
        drawn=drawn,
        parameters=declared,
        weights=np.array([2.0, 1.0]),
        donor_support={
            "housing_water_and_electricity_consumption": (0.0, 6000.0),
            "transport_consumption": (0.0, 1e9),
        },
        donor_division_means=means,
    )
    assert receipt["division_levelling"]["divisions"][food]["factor"] < 1.0
    assert receipt["parents"][housing]["levelling_factor"] == 1.0

    def run(ev):
        return uk_stage_health_gate(
            evidence={"stage": "lcfs_consumption", "recomposed_totals": ev},
            stage="lcfs_consumption",
            check="recomposed_totals",
            parameters=parameters,
        )

    def failing(mutate, fragment: str) -> None:
        tampered = copy.deepcopy(receipt)
        mutate(tampered)
        result = run(tampered)
        assert not result.passed and any(fragment in f for f in result.failures), (
            fragment,
            result.failures,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    assert (
        passed.details["published_fuels_total"] == receipt["published_fuels"]["total"]
    )
    # 7000 - 2000 + 1600 is held at 6000 (microcosm#1164).
    assert receipt["parents"][housing]["rows_held_at_maximum"] == 1
    assert receipt["parents"][housing]["weighted_held_mass"] == 2.0 * 600.0
    failing(
        lambda r: r["parents"][housing].update(weighted_held_mass=0.0),
        "is not its remainder plus components",
    )
    failing(
        lambda r: r["parents"][housing].update(maximum_total="none"),
        "declared maximum_total",
    )
    failing(
        lambda r: r["parents"][housing].update(rows_held_at_maximum=-1),
        "rows held at the maximum",
    )
    failing(
        lambda r: r["parents"][housing].update(minimum_remainder=-1.0),
        "negative remainder",
    )
    failing(
        lambda r: r["parents"][housing].update(rows_below_components=1),
        "below its components",
    )
    failing(
        lambda r: r["parents"]["transport_consumption"].update(weighted_total=1.0),
        "is not its remainder plus components",
    )
    failing(
        lambda r: r["parents"][housing].update(weighted_floored_mass=0.0),
        "is not its draw less the drawn parts",
    )
    failing(
        lambda r: r["parents"][housing].update(components=["electricity_consumption"]),
        "components",
    )
    failing(
        lambda r: r["parents"]["transport_consumption"].update(
            drawn_subtracts=["petrol_spending"]
        ),
        "drawn_subtracts",
    )
    failing(lambda r: r["parents"].pop("transport_consumption"), "not the declared")
    failing(lambda r: r["published_fuels"].update(total=0.0), "published fuel spend")
    # The division levelling (microcosm#1278).
    failing(lambda r: r.pop("division_levelling"), "records no division levelling")
    failing(
        lambda r: r["division_levelling"].update(rule="to_donor_totals"),
        "not the declared levelling",
    )
    failing(
        lambda r: r["division_levelling"]["divisions"].pop(food),
        "not the declared",
    )
    failing(
        lambda r: r["division_levelling"]["divisions"][food].update(factor=1.5),
        "is not in [0, 1]",
    )
    failing(
        lambda r: r["division_levelling"]["divisions"]["transport_consumption"].update(
            factor=0.5
        ),
        "is not above the shared",
    )
    failing(
        lambda r: r["parents"][housing].update(levelling_factor=1.5),
        "is not in [0, 1]",
    )
    failing(
        lambda r: r["parents"][housing].update(levelling_factor=0.5),
        "is not its division's",
    )
    with pytest.raises(ValueError, match="recomposed_totals must be an object"):
        run(None)


def test_bus_support_pricing_gate_recomputes_the_factors_from_the_vendored_rows() -> (
    None
):
    """The ETB bus-support pricing receipt is fenced at stage time (microcosm#930)."""

    import copy

    import pandas as pd

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.bus_support_pricing import (
        bus_support_pricing_operation,
        price_bus_support,
    )
    from microcosm.build.uk_runtime.etb_services import (
        UK_ETB_SERVICES_VENDORED_RESOURCES,
    )

    parameters = _gate_parameters("uk_stage_etb_services_support_pricing")
    assert parameters["check"] == "bus_support_pricing"
    declared = bus_support_pricing_operation(
        load_country_spec("uk").sources.stage_map()["etb_services"]
    )
    household = pd.DataFrame(
        {
            "household_id": [1, 2, 3, 4, 5],
            "region": ["LONDON", "SOUTH_EAST", "WALES", "SCOTLAND", "NORTHERN_IRELAND"],
        }
    )
    person = pd.DataFrame(
        {
            "person_id": [11, 12, 21, 31, 41, 51],
            "person_household_id": [1, 1, 2, 3, 4, 5],
            "bus_in_london_trips": [100.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            "other_local_bus_trips": [0.0, 10.0, 40.0, 30.0, 20.0, 25.0],
            "bus_pass_eligible": [False, True, False, False, True, True],
        }
    )
    _, _, receipt = price_bus_support(
        declared,
        person=person,
        household=household,
        household_weights=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        raw_support=np.array([50.0, 60.0, 70.0, 80.0, 90.0]),
        allowed_resources=UK_ETB_SERVICES_VENDORED_RESOURCES,
    )

    def run(ev):
        return uk_stage_health_gate(
            evidence={"stage": "etb_services", "bus_support_pricing": ev},
            stage="etb_services",
            check="bus_support_pricing",
            parameters=parameters,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    assert passed.details["areas_fact_checked"] == 3
    assert set(passed.details["priced_over_published"]) == {
        "london",
        "england_outside_london",
        "scotland",
    }
    tampered = copy.deepcopy(receipt)
    tampered["by_area"]["scotland"]["other_support_per_boarding"] *= 1.01
    result = run(tampered)
    assert not result.passed and any(
        "scotland other_support_per_boarding" in f for f in result.failures
    ), result.failures
    tampered = copy.deepcopy(receipt)
    tampered["by_area"]["london"]["net_support"]["value"] *= 1.01
    result = run(tampered)
    assert not result.passed and any("london net_support" in f for f in result.failures)
    tampered = copy.deepcopy(receipt)
    tampered["raw_draw_regions"] = ["WALES"]
    result = run(tampered)
    assert not result.passed and any("raw-draw regions" in f for f in result.failures)
    tampered = copy.deepcopy(receipt)
    tampered["applied"] = False
    result = run(tampered)
    assert not result.passed and any("applied pricing" in f for f in result.failures)
    tampered = copy.deepcopy(receipt)
    del tampered["by_area"]["scotland"]
    result = run(tampered)
    assert not result.passed and any(
        "no support area 'scotland'" in f for f in result.failures
    )
    with pytest.raises(ValueError, match="bus_support_pricing must be an object"):
        run(None)


def _spi_income_band_donor_evidence() -> dict[str, object]:
    plan = {
        200_000: (1_436, 359_000.0),
        500_000: (244, 61_000.0),
        1_000_000: (120, 20_000.0),
        2_000_000: (120, 10_000.0),
    }
    rows = [
        {
            "lower_bound": lower,
            "expected_donor_households": count,
            "donor_households": count,
            "carriers": count,
            "planned_donor_weight": taxpayers / count,
            "donor_weight": taxpayers / count,
            "weighted_taxpayers": taxpayers,
            "published_taxpayers": taxpayers,
        }
        for lower, (count, taxpayers) in plan.items()
    ]
    return {
        "stage": "spi_income_band_donors",
        "minimum_donors_per_band": 120,
        "maximum_donor_weight": 250.0,
        "seating_scale": 1.0,
        "mass_scale": 1.0,
        "sample_fraction": 1.0,
        "donor_count": 1_920,
        "reallocated_mass": 450_000.0,
        "mass": {"old_total": 28_600_000.0, "new_total": 28_600_000.0},
        "bands": rows,
        "funding": [
            {
                "stratum": "LONDON",
                "donor_households": 1_200,
                "incumbent_mass": 3_700_000.0,
                "donor_mass": 300_000.0,
                "factor": 1.0 - 300_000.0 / 3_700_000.0,
            },
            {
                "stratum": "SOUTH_EAST",
                "donor_households": 720,
                "incumbent_mass": 3_900_000.0,
                "donor_mass": 150_000.0,
                "factor": 1.0 - 150_000.0 / 3_900_000.0,
            },
            {
                "stratum": "WALES",
                "donor_households": 0,
                "incumbent_mass": 1_400_000.0,
                "donor_mass": 0.0,
                "factor": 1.0,
            },
        ],
    }


_SPI_INCOME_BAND_DONOR_GATE_PARAMETERS = {
    "stage": "spi_income_band_donors",
    "check": "spi_income_band_donor_support",
    "minimum_donors_per_band": 120,
    "maximum_donor_weight": 250.0,
    "minimum_funding_factor": 0.5,
    "maximum_band_taxpayer_deviation": 0.5,
    "maximum_relative_mass_deviation": 1e-9,
    "band_lower_bounds": [200_000, 500_000, 1_000_000, 2_000_000],
}


def _spi_income_band_donor_gate(evidence: dict[str, object]):
    return uk_stage_health_gate(
        evidence=evidence,
        stage="spi_income_band_donors",
        check="spi_income_band_donor_support",
        parameters=_SPI_INCOME_BAND_DONOR_GATE_PARAMETERS,
    )


def test_spi_income_band_donor_support_gate_checks_every_reserved_band() -> None:
    evidence = _spi_income_band_donor_evidence()
    result = _spi_income_band_donor_gate(evidence)
    assert _passed(result)
    assert result.details["donor_count"] == 1_920
    assert result.details["heaviest_donor_weight"] == 250.0
    rows = evidence["bands"]
    # A band whose copy lost its carrier fails.
    broken = [dict(row) for row in rows]
    broken[-1]["carriers"] = 119
    result = _spi_income_band_donor_gate({**evidence, "bands": broken})
    assert not result.passed and "119 carriers" in " ".join(result.failures)
    # A missing band fails against the declared four.
    result = _spi_income_band_donor_gate({**evidence, "bands": rows[:-1]})
    assert not result.passed and "differ from the declared" in " ".join(result.failures)
    # A full stack whose weights do not sum to the published mass fails.
    off = [dict(row) for row in rows]
    off[0]["donor_weight"] = 1.0
    result = _spi_income_band_donor_gate({**evidence, "bands": off})
    assert not result.passed and "differ from the published" in " ".join(
        result.failures
    )
    # A donor above the maximum weight fails, as does a band seated below the
    # minimum at full scale.
    heavy = [dict(row) for row in rows]
    heavy[0].update(donor_households=718, carriers=718, donor_weight=500.0)
    failures = " ".join(
        _spi_income_band_donor_gate({**evidence, "bands": heavy}).failures
    )
    assert "exceeds the maximum 250.0" in failures
    assert "the plan seats 1436" in failures


def test_spi_income_band_donor_support_gate_holds_the_mass_conserved() -> None:
    evidence = _spi_income_band_donor_evidence()
    # Given donors added on top of the incoming mass (the pre-#1063 stage),
    # the gate refuses.
    added = {**evidence, "mass": {"old_total": 28_600_000.0, "new_total": 29_050_000.0}}
    result = _spi_income_band_donor_gate(added)
    assert not result.passed and "funded from the incumbent households" in " ".join(
        result.failures
    )
    # A stratum scaled below the funding floor fails.
    funding = [dict(row) for row in evidence["funding"]]
    funding[0]["factor"] = 0.49
    result = _spi_income_band_donor_gate({**evidence, "funding": funding})
    assert not result.passed and "outside [0.5, 1]" in " ".join(result.failures)
    # Funding rows that do not account for the reallocated mass fail.
    funding = [dict(row) for row in evidence["funding"]]
    funding[1]["donor_mass"] = 100_000.0
    result = _spi_income_band_donor_gate({**evidence, "funding": funding})
    assert not result.passed and "funding strata carry donor mass" in " ".join(
        result.failures
    )
    # The receipt must be the reviewed rule's.
    result = _spi_income_band_donor_gate({**evidence, "maximum_donor_weight": 500.0})
    assert not result.passed and "differs from the gate's 250.0" in " ".join(
        result.failures
    )
    with pytest.raises(ValueError, match="funding must be a non-empty list"):
        _spi_income_band_donor_gate({**evidence, "funding": []})


def test_spi_income_band_donor_support_gate_scales_only_off_the_full_sample() -> None:
    evidence = _spi_income_band_donor_evidence()
    # Given a small frame: fewer, lighter donors. The bands fall short of the
    # published mass by construction and the gate passes on the receipt.
    scale = 0.5
    rows = [
        {
            **row,
            "donor_households": 2,
            "carriers": 2,
            "donor_weight": row["planned_donor_weight"] * scale,
        }
        for row in evidence["bands"]
    ]
    donor_mass = sum(row["donor_weight"] * 2 for row in rows)
    scaled = {
        **evidence,
        "seating_scale": 8 / 1_920,
        "mass_scale": scale,
        "sample_fraction": 0.01,
        "donor_count": 8,
        "reallocated_mass": donor_mass,
        "bands": rows,
        "funding": [
            {
                "stratum": "LONDON",
                "donor_households": 8,
                "incumbent_mass": 2.0 * donor_mass,
                "donor_mass": donor_mass,
                "factor": 0.5,
            }
        ],
    }
    assert _passed(_spi_income_band_donor_gate(scaled))
    # The same receipt on a full-sample build fails: a full build seats and
    # funds the donors at full scale.
    result = _spi_income_band_donor_gate({**scaled, "sample_fraction": 1.0})
    assert not result.passed and "at full scale" in " ".join(result.failures)
    result = _spi_income_band_donor_gate({**scaled, "mass_scale": 0.0})
    assert not result.passed and "outside (0, 1]" in " ".join(result.failures)


def test_spi_support_channel_checks_the_declared_pension_age_share() -> None:
    evidence = {
        "stage": "spi_support_channel",
        "spi_prior_mass_share": 0.5,
        "pension_age_spi_prior_mass_share": 0.25,
        "household_weight_kind": "importance",
        "spi_households": 10,
    }
    parameters = {
        "stage": "spi_support_channel",
        "check": "spi_support_channel",
        "spi_prior_mass_share": 0.5,
        "pension_age_spi_prior_mass_share": 0.25,
        "absolute_tolerance": 0.0,
        "household_weight_kind": "importance",
        "minimum_spi_households": 10,
    }

    def gate(evidence, parameters):
        return uk_stage_health_gate(
            evidence=evidence,
            stage="spi_support_channel",
            check="spi_support_channel",
            parameters=parameters,
        )

    passed = gate(evidence, parameters)
    assert _passed(passed)
    assert passed.details["pension_age_spi_prior_mass_share"] == 0.25
    drifted = gate(evidence, {**parameters, "pension_age_spi_prior_mass_share": 0.5})
    assert not drifted.passed
    assert "pension_age_spi_prior_mass_share 0.25 != declared 0.5" in str(
        drifted.failures
    )
    missing = gate(
        {k: v for k, v in evidence.items() if k != "pension_age_spi_prior_mass_share"},
        parameters,
    )
    assert not missing.passed


def test_pension_credit_take_up_checks_each_band_against_its_rate() -> None:
    def band(name, rate, realized, *, exceed=False, units=10, scope="great_britain"):
        return {
            "band": name,
            "rate": rate,
            "realized_take_up": realized,
            "reporters_exceed_rate": exceed,
            "entitled_units": units,
            "scope": scope,
        }

    parameters = {
        "stage": "pension_credit_take_up",
        "check": "pension_credit_take_up",
        "maximum_take_up_deviation": 0.05,
        "minimum_entitled_units": 1,
    }

    def gate(*bands):
        return uk_stage_health_gate(
            evidence={"stage": "pension_credit_take_up", "bands": list(bands)},
            stage="pension_credit_take_up",
            check="pension_credit_take_up",
            parameters=parameters,
        )

    assert _passed(
        gate(
            band("guarantee_credit", 0.69, 0.70),
            band("savings_credit_only", 0.37, 0.36),
        )
    )
    assert not gate(band("guarantee_credit", 0.69, 0.60)).passed
    assert _passed(gate(band("guarantee_credit", 0.69, 0.80, exceed=True)))
    assert not gate(band("guarantee_credit", 0.69, 0.50, exceed=True)).passed
    assert not gate(band("guarantee_credit", 0.69, None, units=0)).passed
    assert not gate().passed
    # DWP's rates cover Great Britain: a band measured UK-wide fails even at
    # its rate (uk-data#510).
    assert not gate(band("guarantee_credit", 0.69, 0.69, scope="united_kingdom")).passed
    # The realised rates are population facts: a synthetic smoke fixture records
    # the failure without blocking, every other posture blocks on it.
    committed = _gate_parameters("uk_stage_pension_credit_take_up")
    assert committed["check"] == "pension_credit_take_up"


def test_uc_take_up_gate_holds_each_family_type_to_dwp() -> None:
    def family(dwp, before, after, reachable, realized):
        return {
            "dwp_caseload": dwp,
            "caseload_before": before,
            "caseload_after": after,
            "reachable_pools_dwp_caseload": reachable,
            "reachable_pools_caseload_after": realized,
        }

    # 0.05 until microcosm#1418.
    parameters = {
        "stage": "uc_take_up",
        "check": "uc_take_up",
        "maximum_caseload_deviation": 0.055,
    }

    def gate(families, **counts):
        return uk_stage_health_gate(
            evidence={
                "stage": "uc_take_up",
                "families": families,
                "reporters_not_claiming": counts.get("reporters", 0),
                "changed_outside_draw": counts.get("outside", 0),
            },
            stage="uc_take_up",
            check="uc_take_up",
            parameters=parameters,
        )

    good = {
        "SINGLE": family(3_000.0, 3_300.0, 3_020.0, 2_500.0, 2_510.0),
        # No reachable pool: the realized check does not apply.
        "LONE_PARENT": family(2_100.0, 1_400.0, 1_500.0, 0.0, 0.0),
    }
    assert _passed(gate(good))
    # A reporter not claiming, or a unit outside the draw changed, fails.
    assert not gate(good, reporters=1).passed
    assert not gate(good, outside="<3").passed
    # The reachable pools miss DWP's count by more than the tolerance.
    assert not gate(
        {**good, "SINGLE": family(3_000.0, 3_300.0, 3_020.0, 2_500.0, 2_700.0)}
    ).passed
    # The family type moves further from DWP than before.
    assert not gate(
        {**good, "LONE_PARENT": family(2_100.0, 1_400.0, 1_100.0, 0.0, 0.0)}
    ).passed
    # A caseload resting on fewer than three sources cannot be checked: it is
    # flagged, never failed (María's ruling of 2026-10-26); one missing fails.
    thin = gate({**good, "SINGLE": family(3_000.0, "<3", 1.0, 0.0, 0.0)})
    assert thin.passed, thin.failures
    assert thin.details["family_types_unchecked_small_cells"] == ["SINGLE"]
    assert gate(good).details["family_types_unchecked_small_cells"] == []
    assert not gate({**good, "SINGLE": family(3_000.0, None, 1.0, 0.0, 0.0)}).passed
    assert not gate({}).passed
    # The caseloads are population facts: a synthetic smoke fixture records the
    # failure without blocking, every other posture blocks on it.
    committed = _gate_parameters("uk_stage_uc_take_up")
    assert committed == parameters


def test_spi_benefit_coherence_gate_holds_the_structural_zeros() -> None:
    gates = json.loads(
        (_TEST_PATHS.package / "src/microcosm/build/uk/gates.json").read_text("utf-8")
    )
    entry = next(
        g for g in gates["gates"] if g["id"] == "uk_stage_spi_benefit_coherence"
    )
    assert entry["criticality"] == "release_blocking"
    assert entry["evidence_absent_blocks"] is True
    parameters = dict(entry["parameters"])

    def evidence():
        return {
            "stage": "spi_benefit_coherence",
            "zeroed": {
                column: {"rows_reporting_before": 3, "rows_reporting_after": 0}
                for column in parameters["zeroed_columns"]
            },
            "restored": {
                column: {"rows_differing_from_twin_after": 0}
                for column in parameters["restored_columns"]
            },
            "benefits_in_own_right": {"mismatches_after": 0, "spi_rows_changed": 2},
            "universal_credit_take_up": {
                "reporters_not_claiming": 0,
                "outside_population_non_reporters_claiming": 0,
                "spi_claiming_before": 5,
                "spi_claiming_after": 4,
            },
            "base_rows_unchanged": True,
        }

    def gate(payload):
        return uk_stage_health_gate(
            evidence=payload,
            stage="spi_benefit_coherence",
            check="spi_benefit_coherence",
            parameters=parameters,
        )

    assert _passed(gate(evidence()))
    broken = []
    payload = evidence()
    payload["zeroed"]["sda_reported"]["rows_reporting_after"] = 1
    broken.append(payload)
    payload = evidence()
    del payload["zeroed"]["ssmg_reported"]
    broken.append(payload)
    payload = evidence()
    payload["restored"]["iidb_reported"]["rows_differing_from_twin_after"] = 2
    broken.append(payload)
    payload = evidence()
    payload["benefits_in_own_right"]["mismatches_after"] = 1
    broken.append(payload)
    payload = evidence()
    payload["base_rows_unchanged"] = False
    broken.append(payload)
    for key in ("reporters_not_claiming", "outside_population_non_reporters_claiming"):
        payload = evidence()
        payload["universal_credit_take_up"][key] = 1
        broken.append(payload)
    for payload in broken:
        assert not gate(payload).passed


def test_child_benefit_take_up_holds_claims_by_age_and_the_opt_out_share() -> None:
    parameters = {
        "stage": "child_benefit_take_up",
        "check": "child_benefit_take_up",
        "maximum_claim_rate_deviation": 0.02,
        "maximum_age_claim_rate_deviation": 0.05,
        "minimum_age_child_rows": 200,
        "maximum_opt_out_share_deviation": 0.01,
        "minimum_eligible_family_units": 1,
    }

    def age(value, published, realized, *, rows=1_000, clipped=False):
        return {
            "age": value,
            "published_rate": published,
            "realized_rate": realized,
            "eligible_child_rows": rows,
            "clipped": clipped,
        }

    def gate(*, realized=0.868, ages=None, share=0.0905, exhausted=False, units=100):
        evidence = {
            "stage": "child_benefit_take_up",
            "claims": {
                "eligible_family_units": units,
                "target_rate": 0.871,
                "realized_rate": realized,
                "ages": ages
                if ages is not None
                else [age(0, 0.688, 0.69), age(1, 0.768, 0.75)],
            },
            "opt_outs": {
                "target_share": 0.0907,
                "realized_share": share,
                "pool_exhausted": exhausted,
            },
        }
        return uk_stage_health_gate(
            evidence=evidence,
            stage="child_benefit_take_up",
            check="child_benefit_take_up",
            parameters=parameters,
        )

    result = gate()
    assert _passed(result)
    assert result.details["ages_measured"] == 2
    assert result.details["largest_age_deviation"]["age"] == 1
    # The overall claimed share is held to the published rates at the frame's
    # age mix.
    assert "against 0.8710" in " ".join(gate(realized=0.90).failures)
    # An age is held where it has the rows to measure and was not clipped.
    assert "at age 1" in " ".join(gate(ages=[age(1, 0.768, 0.70)]).failures)
    assert _passed(gate(ages=[age(1, 0.768, 0.70, rows=50)]))
    clipped = gate(ages=[age(19, 0.497, 0.60, clipped=True)])
    assert _passed(clipped) and clipped.details["clipped_ages"] == [19]
    # The opted-out share may fall short only where the charged families ran
    # out, and never exceeds the published share.
    assert "opted out against" in " ".join(gate(share=0.05).failures)
    assert _passed(gate(share=0.05, exhausted=True))
    assert not gate(share=0.12, exhausted=True).passed
    assert not gate(units=0).passed
    assert not gate(ages=[]).passed


_PROPERTY_COMPONENTS_PARAMETERS = {
    "stage": "property_components",
    "check": "property_components",
    "maximum_rows_receipts_below_profit": 0,
    "maximum_rows_with_profit_above_band": 0,
    "maximum_band_error_in_boundary_weights": 1.0,
}


def _property_components_evidence() -> dict:
    # Three bounded bands and the remainder; the £90k band's allowance is the
    # two straddlers' weights, 9,000 + 8,000, and its error 600 sits inside.
    return {
        "stage": "property_components",
        "receipts": {
            "rows_receipts_below_profit": 0,
            "rows_with_profit_above_their_band": 0,
            "largest_landlord_weight": 9_500.0,
            "top_band_multiplier_bounds": [1.0, 5.0],
            "bands_top_down": [
                {
                    "lower_bound": 100_000,
                    "target_landlords": 30_000.0,
                    "walk_landlords_weight": 30_400.0,
                    "boundary_allowance_weight": 9_000.0,
                    "profit_multiplier": 1.0,
                    "multiplier_at_floor": True,
                },
                {
                    "lower_bound": 90_000,
                    "target_landlords": 7_600.0,
                    "walk_landlords_weight": 7_000.0,
                    "boundary_allowance_weight": 17_000.0,
                },
                {
                    "lower_bound": 10_000,
                    "target_landlords": 650_000.0,
                    "walk_landlords_weight": 652_000.0,
                    "boundary_allowance_weight": 8_000.0 + 7_500.0,
                },
                {
                    "lower_bound": 0,
                    "target_landlords": 900_000.0,
                    "walk_landlords_weight": 950_000.0,
                    "boundary_allowance_weight": 7_500.0,
                },
            ],
        },
    }


def _property_components(evidence: dict):
    return uk_stage_health_gate(
        evidence=evidence,
        stage="property_components",
        check="property_components",
        parameters=_PROPERTY_COMPONENTS_PARAMETERS,
    )


def test_property_components_gate_passes_and_reports_the_floor() -> None:
    result = _property_components(_property_components_evidence())

    assert _passed(result)
    # The remainder band is never checked, and a multiplier at its floor of 1
    # is reported, not failed.
    assert [row["lower_bound"] for row in result.details["band_errors"]] == [
        100_000,
        90_000,
        10_000,
    ]
    assert result.details["top_band_multiplier_at_floor"] is True
    assert result.details["rows_with_profit_above_their_band"] == 0


def test_property_components_gate_holds_a_band_to_its_own_rounding_allowance() -> None:
    # An empty £90k band passed the old test of twice the largest landlord
    # weight (19,000 against a 7,600 target); against its own straddlers'
    # weights it fails.
    evidence = _property_components_evidence()
    band = evidence["receipts"]["bands_top_down"][1]
    band["walk_landlords_weight"] = 0.0
    band["boundary_allowance_weight"] = 7_000.0
    failed = _property_components(evidence)
    assert not _passed(failed)
    assert "band from 90000" in failed.failures[0]
    assert "rounding allowance 7000.0" in failed.failures[0]

    # The allowance is the receipt's, so a missing one fails closed.
    del band["boundary_allowance_weight"]
    with pytest.raises(ValueError, match="rounding allowance"):
        _property_components(evidence)


@pytest.mark.parametrize(
    ("key", "fragment"),
    [
        ("rows_receipts_below_profit", "receipts below profit"),
        ("rows_with_profit_above_their_band", "band below their profit"),
    ],
)
def test_property_components_gate_requires_the_zero_counts(key, fragment) -> None:
    evidence = _property_components_evidence()
    evidence["receipts"][key] = 3
    failed = _property_components(evidence)
    assert not _passed(failed)
    assert fragment in failed.failures[0]


def test_property_components_gate_fails_a_top_band_at_its_cap() -> None:
    evidence = _property_components_evidence()
    top = evidence["receipts"]["bands_top_down"][0]
    top["profit_multiplier"] = 5.0
    top["multiplier_at_floor"] = False
    failed = _property_components(evidence)
    assert not _passed(failed)
    assert "reached its cap 5.0" in failed.failures[0]
    assert failed.details["top_band_multiplier_at_floor"] is False


def test_property_components_gate_caps_a_reported_allowance_at_two_weights() -> None:
    # The stage's allowance is two straddlers at most. A receipt that reported
    # more would widen the gate, so the cap of twice the largest landlord
    # weight (19,000 here) applies before the band's error is judged.
    evidence = _property_components_evidence()
    band = evidence["receipts"]["bands_top_down"][2]
    band["walk_landlords_weight"] = band["target_landlords"] + 25_000.0
    band["boundary_allowance_weight"] = 30_000.0
    failed = _property_components(evidence)
    assert not _passed(failed)
    assert "rounding allowance 19000.0" in failed.failures[0]
    assert failed.details["band_errors"][2]["allowance_weight"] == 19_000.0


def test_consumption_basis_gate_checks_the_recorded_bridge_factor() -> None:
    # The receipt records before, after, base and factor for each bridged
    # column; the gate holds factor to after / base (the #1167 re-review's nit).
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_DIARY_PRICE_COLUMNS,
    )

    parameters = _gate_parameters("uk_stage_lcfs_consumption_basis")
    columns = {
        column: {
            "basis": "engine_parameter",
            "from_window": "fiscal_year",
            "before": 1.5,
            "after": 1.6,
            "base": 1.525,
            "factor": 1.6 / 1.525,
        }
        for column in UK_LCFS_DIARY_PRICE_COLUMNS
    }
    evidence = {
        "stage": "lcfs_consumption",
        "donor_uprating": {
            "from_period": 2023,
            "to_period": 2024,
            "fiscal_year_bridge": {"start_year_weight": 0.75, "next_year_weight": 0.25},
            "columns": columns,
        },
        "energy_pricing": {
            "period_type": "calendar_year",
            "period_value": 2024,
            "level": {"period_type": "calendar_year", "period_value": 2024},
        },
        "bus_pricing": {"price_year": 2024},
        "road_fuel_level": {"period_type": "calendar_year", "period_value": 2024},
    }

    def run(ev):
        return uk_stage_health_gate(
            evidence=ev,
            stage="lcfs_consumption",
            check="consumption_basis",
            parameters=parameters,
        )

    assert run(evidence).passed
    first = UK_LCFS_DIARY_PRICE_COLUMNS[0]
    evidence["donor_uprating"]["columns"][first]["factor"] = 1.6 / 1.5
    result = run(evidence)
    assert not result.passed
    assert any("not after / base" in f and first in f for f in result.failures)


def _packaged_parameters(gate_id: str) -> dict:
    gates = json.loads(
        (_TEST_PATHS.package / "src/microcosm/build/uk/gates.json").read_text()
    )["gates"]
    (gate,) = [entry for entry in gates if entry["id"] == gate_id]
    return gate["parameters"]


def _purchases_receipt() -> dict:
    def kind(name, target, fitted, shortfall=0.0, value=1.0e9):
        return {
            "property_type": name,
            "target_transactions": target,
            "fitted_transactions": fitted,
            "target_value": value,
            "fitted_value": value * (1.0 - shortfall),
            "value_shortfall_share": shortfall,
        }

    return {
        "stage": "property_purchases",
        "kinds": [
            kind("additional_residential", 1000.0, 1010.0),
            kind("non_residential", 200.0, 196.0, shortfall=0.1),
        ],
        "derivation": {
            "residential_individuals_transaction_share": 0.9,
            "residential_individuals_value_share": 0.8,
        },
    }


def _gate(gate_id: str, evidence: dict):
    parameters = _packaged_parameters(gate_id)
    return uk_stage_health_gate(
        evidence=evidence,
        stage=parameters["stage"],
        check=parameters["check"],
        parameters=parameters,
    )


def test_property_purchases_gate_holds_each_kind_to_hmrc_within_the_contract() -> None:
    assert _passed(_gate("uk_stage_property_purchases_fit", _purchases_receipt()))

    missing = _purchases_receipt()
    missing["kinds"] = missing["kinds"][:1]
    assert (
        "draws kinds" in _gate("uk_stage_property_purchases_fit", missing).failures[0]
    )

    far = _purchases_receipt()
    far["kinds"][0]["fitted_transactions"] = 1100.0
    assert "transactions" in _gate("uk_stage_property_purchases_fit", far).failures[0]

    short = _purchases_receipt()
    short["kinds"][1].update(value_shortfall_share=0.25, fitted_value=0.75e9)
    assert (
        "short of HMRC" in _gate("uk_stage_property_purchases_fit", short).failures[0]
    )

    unrecorded = _purchases_receipt()
    unrecorded["kinds"][1]["fitted_value"] = 1.0e9
    assert (
        "recorded shortfall"
        in (_gate("uk_stage_property_purchases_fit", unrecorded).failures[0])
    )

    no_share = _purchases_receipt()
    no_share["derivation"]["residential_individuals_value_share"] = 1.2
    assert (
        "not a share" in _gate("uk_stage_property_purchases_fit", no_share).failures[0]
    )


def test_property_purchases_gate_flags_a_kind_on_fewer_than_three_buyers() -> None:
    """A kind whose buyers rest on fewer than three sources prints "<3": it is
    flagged unchecked, never failed (María's ruling of 2026-10-26)."""
    thin = _purchases_receipt()
    name = thin["kinds"][0]["property_type"]
    thin["kinds"][0].update(
        fitted_transactions="<3", fitted_value="<3", value_shortfall_share="<3"
    )
    result = _gate("uk_stage_property_purchases_fit", thin)
    assert result.passed, result.failures
    assert result.details["kinds_unchecked_small_cells"] == [name]
    clean = _gate("uk_stage_property_purchases_fit", _purchases_receipt())
    assert clean.details["kinds_unchecked_small_cells"] == []


def _home_values_receipt() -> dict:
    regions = [f"E1200000{index}" for index in range(1, 10)] + ["W92000004"]
    return {
        "stage": "regional_home_values",
        "rows": 100,
        # Counts as the receipt prints them (receipt_counts.rounded_count).
        "rows_rewritten_by_region": {region: 10 for region in regions},
        "sources_at_cap": 0,
        "knots_basis_year": 2024,
        "ranking": {"rank_basis": "weighted_midpoint_within_frs_region"},
        "weighted_value_before": 1.0e9,
        "weighted_value_after": 1.2e9,
    }


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda r: r["rows_rewritten_by_region"].pop("W92000004"), "rewrites regions"),
        (lambda r: r["rows_rewritten_by_region"].update(E12000007=0), "no row"),
        (lambda r: r.update(rows=40), "more rows"),
        (lambda r: r.update(sources_at_cap=1), "value cap"),
        # A band past the limit fails as plainly as the count it stands for.
        (lambda r: r.update(sources_at_cap="<3"), "value cap"),
        (lambda r: r.update(knots_basis_year=2023), "basis year"),
        (lambda r: r["ranking"].update(rank_basis="national"), "ranking basis"),
        (lambda r: r.update(weighted_value_after=0.0), "not positive"),
    ],
)
def test_regional_home_values_gate_holds_the_ranking_contract(mutate, fragment) -> None:
    assert _passed(
        _gate("uk_stage_regional_home_values_ranking", _home_values_receipt())
    )
    receipt = _home_values_receipt()
    mutate(receipt)
    result = _gate("uk_stage_regional_home_values_ranking", receipt)
    assert not result.passed
    assert any(fragment in failure for failure in result.failures), result.failures


def test_regional_home_values_gate_flags_a_region_of_fewer_than_three_rows() -> None:
    """A region the receipt prints "<3" rewrote one or two rows: not empty,
    and flagged rather than failed (María's ruling of 2026-10-26)."""
    receipt = _home_values_receipt()
    receipt["rows_rewritten_by_region"]["W92000004"] = "<3"
    result = _gate("uk_stage_regional_home_values_ranking", receipt)
    assert result.passed, result.failures
    assert result.details["regions_rewritten_under_floor"] == ["W92000004"]
    assert result.details["sources_at_cap_unchecked"] is False


def test_regional_home_values_gate_reads_a_banded_count_at_its_bounds() -> None:
    """A region the receipt prints "3-9" rewrote three to nine rows: not empty,
    above the floor, so not flagged, and read at three in the sum check. A
    rounded count is read at the lowest count that rounds to it, and the
    stage's own row count at the highest."""
    gate_id = "uk_stage_regional_home_values_ranking"
    receipt = _home_values_receipt()
    receipt["rows_rewritten_by_region"]["W92000004"] = "3-9"
    result = _gate(gate_id, receipt)
    assert result.passed, result.failures
    assert result.details["regions_rewritten_under_floor"] == []
    # Nine regions printed 10 (five rows at the least) and one of at least
    # three: 48 rows at the least, against a stage of 44 rows at the most.
    receipt["rows"] = 40
    assert "more rows" in _gate(gate_id, receipt).failures[0]
    receipt["rows"] = 50
    assert _gate(gate_id, receipt).passed
    # Ten regions of 15 rows print 20 each; a stage of 150 rows is not short.
    rounded = _home_values_receipt()
    rounded["rows_rewritten_by_region"] = {
        region: 20 for region in rounded["rows_rewritten_by_region"]
    }
    rounded["rows"] = 150
    assert _gate(gate_id, rounded).passed
    # A stage whose own count is banded is read at the band's top.
    small = _home_values_receipt()
    small["rows_rewritten_by_region"] = {
        region: "<3" for region in small["rows_rewritten_by_region"]
    }
    small["rows"] = 20
    assert _gate(gate_id, small).passed
    small["rows"] = "3-9"
    assert "more rows" in _gate(gate_id, small).failures[0]


def _legacy_closure_receipt() -> dict:
    return {
        "stage": "uc_legacy_closure_take_up",
        "p_basis": "spine_design_weights_great_britain",
        "great_britain_weighted_share_claiming_at_closure": 0.8,
        "combinations": {
            "housing_benefit": {
                "rate": 0.75,
                "rate_source": "combination",
                "rate_unreachable": False,
                "support": "30+",
            },
            "jsa_income": {
                "rate": 0.6,
                "rate_source": "all_households",
                "rate_unreachable": "<3",
                "support": "<3",
            },
        },
    }


def test_legacy_closure_gate_reads_what_the_receipt_prints() -> None:
    gate_id = "uk_stage_uc_legacy_closure_take_up_contract"
    assert _passed(_gate(gate_id, _legacy_closure_receipt()))
    # A thin drawn cohort prints its share withheld.
    withheld = _legacy_closure_receipt()
    withheld["great_britain_weighted_share_claiming_at_closure"] = "<3"
    flagged = _gate(gate_id, withheld)
    assert _passed(flagged)
    # What it could not read is flagged (María's ruling of 2026-10-26).
    assert flagged.details["share_claiming_at_closure_withheld"] is True
    assert flagged.details["unreachable_flags_withheld"] == 1
    for mutate, fragment in (
        (lambda r: r.update(p_basis="calibrated_weights"), "measured on"),
        (lambda r: r["combinations"]["housing_benefit"].update(rate=1.2), "rate 1.2"),
        (
            lambda r: r["combinations"]["jsa_income"].update(rate_source="guess"),
            "takes its rate",
        ),
        (lambda r: r["combinations"]["jsa_income"].update(support="3"), "support band"),
        (
            lambda r: r["combinations"]["jsa_income"].update(rate_unreachable="2"),
            "unreachable",
        ),
        (lambda r: r.update(combinations={}), "no combinations"),
    ):
        receipt = _legacy_closure_receipt()
        mutate(receipt)
        result = _gate(gate_id, receipt)
        assert not result.passed
        assert any(fragment in failure for failure in result.failures), result.failures


def test_the_1308_fact_gates_record_without_blocking_a_synthetic_smoke_build() -> None:
    """A synthetic fixture is not the population HMRC and the knots describe (#1308).

    The legacy closure gate holds the receipt's own contract, which a synthetic
    fixture can meet, so it blocks a smoke build as it blocks every posture.
    """

    gates = {
        entry["id"]: entry
        for entry in json.loads(
            (_TEST_PATHS.package / "src/microcosm/build/uk/gates.json").read_text()
        )["gates"]
    }
    for gate_id in (
        "uk_stage_regional_home_values_ranking",
        "uk_stage_property_purchases_fit",
    ):
        assert gates[gate_id]["population_fact_check"] is True
    contract = gates["uk_stage_uc_legacy_closure_take_up_contract"]
    assert contract.get("population_fact_check", False) is False


def test_excise_quantities_gate_recomputes_every_level_from_the_vendored_rows(
    monkeypatch,
) -> None:
    """The lcfs excise_quantities receipt is fact-checked at stage time (#1359)."""

    import copy

    import pandas as pd

    from microcosm.build.uk_runtime.lcfs_consumption import (
        excise_levels,
        level_excise_quantities,
    )
    from test_support.microcosm_build.uk_excise_facts import (
        excise_declaration,
        patch_excise_facts,
    )

    declaration = excise_declaration()
    patch_excise_facts(monkeypatch, declaration)
    resolved = excise_levels(declaration)
    tobacco, on_trade, gambling, lottery_share, betting_share = (
        "tobacco_share_of_alcohol_and_tobacco",
        "on_trade_alcohol_share_of_restaurants_and_hotels",
        "gambling_stakes_share_of_recreation",
        "national_lottery_share_of_gambling_stakes",
        "betting_and_gaming_share_of_other_gambling_stakes",
    )
    draws = pd.DataFrame(
        {
            "alcohol_and_tobacco_consumption": [1000.0, 500.0, 0.0, 800.0],
            "restaurants_and_hotels_consumption": [2000.0, 0.0, 600.0, 400.0],
            "recreation_consumption": [3000.0, 1000.0, 0.0, 2000.0],
            tobacco: [0.25, 1.0, 0.0, 0.0],
            on_trade: [0.1, 0.0, 0.5, 0.0],
            gambling: [0.02, 0.0, 0.0, 0.05],
            lottery_share: [0.5, 0.0, 0.0, 0.0],
            betting_share: [0.0, 0.0, 0.0, 0.75],
        }
    )
    _, quantities = level_excise_quantities(
        draws,
        levels=resolved,
        weights=np.array([1.0e3, 2.0e3, 3.0e3, 4.0e3]),
    )
    receipt = {
        "period_type": "fiscal_year",
        "period_value": 2024,
        "weeks_per_year": 52,
        **quantities,
    }
    shares = {
        name: {
            "frame": {"draws_outside_unit_interval": 0},
            "donor": {"positive_support": "30+", "total_beside_lines": residual},
        }
        for name, residual in (
            (tobacco, 0.004),
            (on_trade, None),
            (gambling, None),
            (lottery_share, None),
            (betting_share, None),
        )
    }
    parameters = {
        "stage": "lcfs_consumption",
        "check": "excise_quantities",
        "maximum_relative_deviation": 1e-9,
        "maximum_parent_residual_share": 0.01,
        "required_support_band": "30+",
    }

    def run(excise, excise_shares=shares):
        return uk_stage_health_gate(
            evidence={
                "stage": "lcfs_consumption",
                "excise_quantities": excise,
                "excise_shares": excise_shares,
            },
            stage="lcfs_consumption",
            check="excise_quantities",
            parameters=parameters,
        )

    def failing(change, fragment: str) -> None:
        excise, excise_shares = copy.deepcopy(receipt), copy.deepcopy(shares)
        change(excise, excise_shares)
        result = run(excise, excise_shares)
        assert not result.passed and any(fragment in f for f in result.failures), (
            fragment,
            result.failures,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    assert passed.details["levels"]["beer_litres"] == pytest.approx(2.0e5 / 0.04)
    beer = lambda e, s: e["outputs"]["beer_litres"]  # noqa: E731
    failing(lambda e, s: beer(e, s).update(published=2.1), "published")
    failing(lambda e, s: beer(e, s).update(annual_level=1.0), "annual_level")
    failing(lambda e, s: beer(e, s).update(source_record_id="x"), "declared row")
    failing(
        lambda e, s: beer(e, s).update(
            frame_annual_total=beer(e, s)["frame_annual_total"] * 0.99
        ),
        "is not the level",
    )
    failing(
        lambda e, s: beer(e, s).update(factor=beer(e, s)["factor"] * 1.01),
        "does not reproduce",
    )
    failing(
        lambda e, s: beer(e, s).update(positive_share=0.99), "where its spending is"
    )
    # A betting and gaming base positive where the National Lottery's stakes
    # are, not where its own are.
    failing(
        lambda e, s: e["outputs"]["remote_gaming_net_losses"].update(
            positive_share=e["keys"]["national_lottery"]["positive_share"]
        ),
        "where its spending is",
    )
    failing(lambda e, s: e["outputs"].pop("wine_litres"), "not the declared")
    failing(lambda e, s: e.update(weeks_per_year=52.18), "weeks_per_year")
    failing(
        lambda e, s: e["draught_shares"]["beer_draught_share"].update(share=0.3),
        "beer_draught_share",
    )
    failing(lambda e, s: e["draught_shares"].clear(), "draught shares")
    lottery = lambda e, s: e["outputs"]["national_lottery_ticket_spending"]  # noqa: E731
    failing(
        lambda e, s: lottery(e, s).update(
            frame_annual_total=lottery(e, s)["frame_annual_total"] * 1.01
        ),
        "national_lottery_ticket_spending",
    )
    failing(lambda e, s: e.update(period_value=2025), "period_value")
    failing(
        lambda e, s: e.update(alcohol_volume_weights={"rule": "other"}),
        "volume weights",
    )
    failing(
        lambda e, s: s[tobacco]["frame"].update(draws_outside_unit_interval="<3"),
        "left [0, 1]",
    )
    failing(
        lambda e, s: s[on_trade]["donor"].update(positive_support="10-29"),
        "rests on",
    )
    failing(
        lambda e, s: s[tobacco]["donor"].update(total_beside_lines=0.02),
        "beside",
    )
    # A residual withheld for resting on under three donors is under the ceiling.
    withheld = copy.deepcopy(shares)
    withheld[tobacco]["donor"]["total_beside_lines"] = "<3"
    flagged = run(receipt, withheld)
    assert flagged.passed
    assert flagged.details["residuals_unchecked_small_cells"] == [tobacco]
    with pytest.raises(ValueError, match="excise_quantities must be an object"):
        run(None)


def test_gambling_participation_gate_recomputes_the_rates_and_checks_the_shares(
    monkeypatch,
) -> None:
    """The lcfs gambling_participation receipt is fact-checked at stage time (#1359)."""

    import copy

    from microcosm.build.uk_runtime.gambling_participation import (
        top_up_gambling_participation,
    )
    from test_support.microcosm_build.uk_gambling_participation_facts import (
        participation_declaration,
        participation_inputs,
        patch_gambling_participation_facts,
    )

    patch_gambling_participation_facts(monkeypatch, participation_declaration())
    inputs = participation_inputs()
    topped = top_up_gambling_participation(**inputs)
    weights = inputs["weights"]
    receipt = copy.deepcopy(dict(topped.receipt))
    # The levelling's keys, as level_excise_quantities reads them.
    excise = {
        "keys": {
            key: {"positive_share": float(weights[values > 0].sum() / weights.sum())}
            for key, values in topped.keys.items()
        }
    }
    parameters = {
        "stage": "lcfs_consumption",
        "check": "gambling_participation",
        "maximum_relative_deviation": 1e-9,
        "maximum_stratum_shortfall": 0.01,
        "amount_scale_bounds": [0.5, 2.0],
        "required_support_band": "30+",
    }

    def run(participation, levelled=excise):
        return uk_stage_health_gate(
            evidence={
                "stage": "lcfs_consumption",
                "gambling_participation": participation,
                "excise_quantities": levelled,
            },
            stage="lcfs_consumption",
            check="gambling_participation",
            parameters=parameters,
        )

    def failing(change, fragment: str) -> None:
        participation, levelled = copy.deepcopy(receipt), copy.deepcopy(excise)
        change(participation, levelled)
        result = run(participation, levelled)
        assert not result.passed and any(fragment in f for f in result.failures), (
            fragment,
            result.failures,
        )

    passed = run(receipt)
    assert passed.passed, passed.failures
    lottery = lambda r, e: r["keys"]["national_lottery"]  # noqa: E731
    shipped = next(
        label
        for label, entry in receipt["keys"]["national_lottery"]["strata"].items()
        if entry != {"withheld": True}
    )
    stratum = lambda r, e: lottery(r, e)["strata"][shipped]  # noqa: E731
    failing(
        lambda r, e: r["facts"]["rates"]["national_lottery"]["regular"].update(
            {"MALE:18": 0.9}
        ),
        "not the vendored",
    )
    failing(lambda r, e: r["facts"].update(rows_read=1), "rows_read")
    failing(
        lambda r, e: r["settings"]["theta"].update(national_lottery=0.5),
        "not the declared",
    )
    failing(
        lambda r, e: stratum(r, e).update(
            after_regular=stratum(r, e)["regular_target"] - 0.05
        ),
        "more than 0.01",
    )
    failing(
        lambda r, e: stratum(r, e).update(before=1.0),
        "but moved",
    )
    failing(lambda r, e: stratum(r, e).pop("after_annual"), "no shares")
    failing(
        lambda r, e: lottery(r, e).update(
            after_annual=lottery(r, e)["annual_target"] - 0.05
        ),
        "falls short",
    )
    failing(
        lambda r, e: lottery(r, e)["stake_shares"].update(seen=0.9),
        "do not split",
    )
    # A stake draw far off the seen households' mean, which the anchoring
    # would otherwise hide.
    failing(lambda r, e: lottery(r, e).update(amount_scale=3.0), "amount scale")
    failing(lambda r, e: lottery(r, e).update(added_without_stake="<3"), "no stake")
    failing(
        lambda r, e: lottery(r, e)["propensity"].update(positive_support="10-29"),
        "rest on",
    )
    failing(
        lambda r, e: e["keys"]["national_lottery"].update(positive_share=0.1),
        "excise levelling",
    )
    failing(lambda r, e: r["keys"].pop("betting_and_gaming"), "tops up no")
    # A key withheld nationally is named unchecked, never failed (María's
    # ruling of 2026-10-26).
    national = copy.deepcopy(receipt)
    national["keys"]["national_lottery"] = {
        **{
            name: national["keys"]["national_lottery"][name]
            for name in ("added_without_stake", "propensity", "strata")
        },
        "withheld": True,
    }
    result = run(national)
    assert result.passed, result.failures
    assert result.details["keys_unchecked_small_cells"] == ["national_lottery"]
    assert passed.details["keys_unchecked_small_cells"] == []
    # A withheld stratum is counted, not checked.
    withheld = copy.deepcopy(receipt)
    withheld["keys"]["national_lottery"]["strata"][shipped] = {"withheld": True}
    result = run(withheld)
    assert result.passed
    assert result.details["strata_withheld"] == passed.details["strata_withheld"] + 1
    with pytest.raises(ValueError, match="gambling_participation must be an object"):
        run(None)
    for bounds in (None, [0.5], [2.0, 0.5], [0.0, 2.0], [True, 2.0]):
        parameters["amount_scale_bounds"] = bounds
        with pytest.raises(ValueError, match="amount_scale_bounds"):
            run(receipt)


def _apprentice_receipt(**overrides):
    """An is_apprentice receipt at the contract's 2024 rate (microcosm#1373)."""

    from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract

    rate = load_uk_take_up_contract().rate("apprentice_stock_19_to_24", 2024)
    receipt = {
        "build_year": 2024,
        "output": "is_apprentice",
        "rate_key": "apprentice_stock_19_to_24",
        "rate": rate,
        "ages": [21, 24],
        "eligible_statuses": ["FT_EMPLOYED", "PT_EMPLOYED"],
        "population_mass": 1_000_000.0,
        "eligible_mass": 600_000.0,
        "target": rate * 1_000_000.0,
        "cap_bound": False,
        "population_support": "30+",
        "eligible_support": "30+",
        "flagged_support": "30+",
        "realized_mass": rate * 1_000_000.0 + 500.0,
        "realization_deviation": 500.0,
        "implied_relief": 200_000_000.0,
    }
    receipt.update(overrides)
    return {"stage": "frs_person_draws", "apprentice_receipt": receipt}


def test_apprentice_gate_holds_the_draw_to_the_contract_rate() -> None:
    gate_id = "uk_stage_frs_person_draws_apprentices"
    assert _passed(_gate(gate_id, _apprentice_receipt()))
    rate = _apprentice_receipt()["apprentice_receipt"]["rate"]
    far = rate * 1_000_000.0 * 1.06
    for receipt, fragment in (
        (
            _apprentice_receipt(rate_key="apprenticeship_participation_19_to_24"),
            "reads",
        ),
        (_apprentice_receipt(rate=0.05992), "not the contract's"),
        (_apprentice_receipt(target=1.0), "not the rate times"),
        (_apprentice_receipt(flagged_support="12"), "printed band"),
        (
            _apprentice_receipt(
                realized_mass=far, realization_deviation=far - rate * 1_000_000.0
            ),
            "away from it",
        ),
        (_apprentice_receipt(realization_deviation=0.0), "realized minus target"),
        (_apprentice_receipt(cap_bound=True), "cap_bound is True"),
        (_apprentice_receipt(implied_relief=400_000_000.0), "above the"),
        (_apprentice_receipt(implied_relief=None), "no implied relief"),
    ):
        result = _gate(gate_id, receipt)
        assert not result.passed
        assert any(fragment in failure for failure in result.failures), result.failures


def test_apprentice_gate_flags_its_small_cells_without_failing() -> None:
    """A "<3" figure is named unchecked, never failed (María's ruling, 10-26)."""

    gate_id = "uk_stage_frs_person_draws_apprentices"
    thin_flags = _gate(
        gate_id,
        _apprentice_receipt(
            flagged_support="<3",
            realized_mass="<3",
            realization_deviation="<3",
            implied_relief="<3",
        ),
    )
    assert _passed(thin_flags)
    assert thin_flags.details["figures_unchecked_small_cells"] == [
        "realized_mass",
        "realization_deviation",
        "implied_relief",
    ]
    thin_population = _gate(
        gate_id,
        _apprentice_receipt(
            population_support="<3",
            population_mass="<3",
            target="<3",
            realization_deviation="<3",
        ),
    )
    assert _passed(thin_population)
    assert thin_population.details["figures_unchecked_small_cells"] == [
        "population_mass",
        "target",
        "realization_deviation",
    ]
    assert (
        _gate(gate_id, _apprentice_receipt()).details["figures_unchecked_small_cells"]
        == []
    )
    # What the receipt still prints is checked beside a withheld figure.
    off_rate = _gate(
        gate_id,
        _apprentice_receipt(eligible_support="<3", eligible_mass="<3", rate=0.05992),
    )
    assert not off_rate.passed
    assert any("not the contract's" in failure for failure in off_rate.failures)


def test_apprentice_gate_accepts_a_cap_bound_draw_that_flags_every_employee() -> None:
    gate_id = "uk_stage_frs_person_draws_apprentices"
    rate = _apprentice_receipt()["apprentice_receipt"]["rate"]
    short = _apprentice_receipt(
        eligible_mass=20_000.0,
        cap_bound=True,
        realized_mass=20_000.0,
        realization_deviation=20_000.0 - rate * 1_000_000.0,
    )
    assert _passed(_gate(gate_id, short))

    partial = _apprentice_receipt(
        eligible_mass=20_000.0,
        cap_bound=True,
        realized_mass=10_000.0,
        realization_deviation=10_000.0 - rate * 1_000_000.0,
    )
    result = _gate(gate_id, partial)
    assert not result.passed
    assert "every eligible employee" in result.failures[0]


def test_apprentice_gate_is_a_population_fact_check() -> None:
    """A synthetic fixture is not the population DfE's rate describes (#1373)."""

    gates = {
        entry["id"]: entry
        for entry in json.loads(
            (_TEST_PATHS.package / "src/microcosm/build/uk/gates.json").read_text()
        )["gates"]
    }
    entry = gates["uk_stage_frs_person_draws_apprentices"]
    assert entry["population_fact_check"] is True
    assert entry["criticality"] == "release_blocking"
    assert entry["evidence_absent_blocks"] is True


def test_assembled_stage_gates_read_only_stages_run_by_the_assembled_boundary() -> None:
    """An assembled-phase stage gate reads only a stage that has run by the
    assembled boundary; a later stage's gate belongs to the transferred phase,
    or the boundary offers no receipt for it and the gate fails closed (the
    re-review of microcosm#1422)."""
    from microcosm.build import load_country_spec
    from microcosm.build.uk_runtime.spine_build import UK_SPINE_ASSEMBLED_FINAL_STAGE

    spec = load_country_spec("uk")
    order = [stage.stage for stage in spec.sources.stages]
    boundary = order.index(UK_SPINE_ASSEMBLED_FINAL_STAGE)
    stage_gates = [entry for entry in spec.gates.gates if entry.gate == "stage_health"]
    assert {str(entry.parameters["stage"]) for entry in stage_gates} <= set(order)
    late = sorted(
        entry.id
        for entry in stage_gates
        if entry.phase == "assembled"
        and order.index(str(entry.parameters["stage"])) > boundary
    )
    assert late == []
