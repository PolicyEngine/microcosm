from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from populace.build.source_manifest import SourceOperationSpec
from populace.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
    run_source_stage,
)
from populace.build.us_runtime import US_SOURCE_MANIFEST
from populace.build.us_runtime.puf_aggregate_records import (
    AGGREGATE_RECIDS,
    SYNTHETIC_RECID_START,
    derive_puf_policyengine_variables,
    disaggregate_puf_aggregate_records,
)
from populace.build.us_runtime.source_runtime import (
    aggregate_us_person_to_tax_unit_from_manifest,
    calibrate_us_binary_assignment_from_manifest,
    calibrate_us_binary_assignment_joint_targets_from_manifest,
    derive_us_child_support_from_manifest,
    derive_us_disability_benefits_from_manifest,
    derive_us_other_health_insurance_from_manifest,
    derive_us_puf_policyengine_variables_from_manifest,
    disaggregate_us_puf_aggregate_records_from_manifest,
    impute_us_child_support_to_puf_support_from_manifest,
    impute_us_disability_benefits_to_puf_support_from_manifest,
    impute_us_other_health_insurance_to_puf_support_from_manifest,
    us_source_operation_handlers,
)


def _make_runtime_mini_puf() -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows: list[dict[str, float | int]] = []
    donor_specs = [
        (999996, -18_000_000.0, -500_000.0),
        (999997, 250_000.0, 9_500_000.0),
        (999998, 12_000_000.0, 90_000_000.0),
        (999999, 125_000_000.0, 650_000_000.0),
    ]
    next_recid = 1
    for _bucket_recid, low, high in donor_specs:
        for index in range(25):
            agi = float(rng.uniform(low, high))
            sign = -1.0 if agi < 0 else 1.0
            abs_agi = abs(agi)
            rows.append(
                {
                    "RECID": next_recid,
                    "S006": 100,
                    "MARS": 2 if index % 3 == 0 else 1,
                    "XTOT": 2 if index % 3 == 0 else 1,
                    "DSI": 0,
                    "EIC": index % 4,
                    "E00100": agi,
                    "E00200": abs_agi * 0.08,
                    "P23250": abs_agi * rng.uniform(0.05, 0.35) * sign,
                    "P22250": abs_agi * rng.uniform(0.01, 0.12) * sign,
                    "E00650": abs_agi * 0.04,
                    "E00300": abs_agi * 0.03,
                    "E26270": abs_agi * rng.uniform(0.01, 0.15) * sign,
                    "E00900": abs_agi * 0.03 * sign,
                    "E02100": abs_agi * 0.01 * sign,
                    "E27200": abs_agi * 0.005 * sign,
                    "E00400": abs_agi * 0.01,
                    "E00600": abs_agi * 0.05,
                }
            )
            next_recid += 1

    aggregate_rows = [
        (999996, -5_000_000.0),
        (999997, 5_000_000.0),
        (999998, 30_000_000.0),
        (999999, 300_000_000.0),
    ]
    for recid, agi in aggregate_rows:
        sign = -1.0 if agi < 0 else 1.0
        abs_agi = abs(agi)
        rows.append(
            {
                "RECID": recid,
                "S006": 2_000,
                "MARS": 0,
                "XTOT": 1,
                "DSI": 0,
                "EIC": 0,
                "E00100": agi,
                "E00200": abs_agi * 0.08,
                "P23250": abs_agi * 0.30 * sign,
                "P22250": abs_agi * 0.08 * sign,
                "E00650": abs_agi * 0.04,
                "E00300": abs_agi * 0.03,
                "E26270": abs_agi * 0.10 * sign,
                "E00900": abs_agi * 0.03 * sign,
                "E02100": abs_agi * 0.01 * sign,
                "E27200": abs_agi * 0.005 * sign,
                "E00400": abs_agi * 0.01,
                "E00600": abs_agi * 0.05,
            }
        )
    result = pd.DataFrame(rows)
    result["E03230"] = np.where(result["RECID"] % 5 == 0, 1_500.0, 0.0)
    result["E87530"] = np.where(result["RECID"] % 7 == 0, 4_000.0, 0.0)
    result["E00800"] = np.where(result["RECID"] % 13 == 0, 3_000.0, 0.0)
    result["E03500"] = np.where(result["RECID"] % 17 == 0, 2_000.0, 0.0)
    result["E20500"] = np.where(result["RECID"] % 11 == 0, 8_000.0, 0.0)
    result["E03240"] = np.where(result["RECID"] % 9 == 0, 6_000.0, 0.0)
    result["E03220"] = np.where(result["RECID"] % 7 == 0, 300.0, 0.0)
    result["E20400"] = np.where(result["RECID"] % 3 == 0, 2_500.0, 0.0)
    result["E58990"] = np.where(result["RECID"] % 19 == 0, 5_000.0, 0.0)
    return result


def test_us_child_support_handlers_are_in_shared_registry() -> None:
    handlers = us_source_operation_handlers()

    assert (
        handlers["derive_child_support_inputs"] is derive_us_child_support_from_manifest
    )
    assert (
        handlers["impute_child_support_to_puf_support"]
        is impute_us_child_support_to_puf_support_from_manifest
    )


def test_us_disability_benefits_handlers_are_in_shared_registry() -> None:
    handlers = us_source_operation_handlers()

    assert (
        handlers["derive_disability_benefits"]
        is derive_us_disability_benefits_from_manifest
    )
    assert (
        handlers["impute_disability_benefits_to_puf_support"]
        is impute_us_disability_benefits_to_puf_support_from_manifest
    )


def test_us_other_health_insurance_handlers_are_in_shared_registry() -> None:
    handlers = us_source_operation_handlers()

    assert (
        handlers["derive_other_health_insurance_premiums"]
        is derive_us_other_health_insurance_from_manifest
    )
    assert (
        handlers["impute_other_health_insurance_premiums_to_puf_support"]
        is impute_us_other_health_insurance_to_puf_support_from_manifest
    )


def _make_aca_people() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5, 6, 7],
            "tax_unit_id": [10, 10, 20, 30, 40, 50, 60],
            "has_marketplace_health_coverage_at_interview": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
            "reported_has_subsidized_marketplace_health_coverage_at_interview": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        }
    )


def _make_aca_tax_units() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tax_unit_id": [10, 20, 30, 40, 50, 60],
            "state_fips": ["01", "01", "01", "01", "02", "02"],
            "tax_unit_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "household_weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "stable_tax_unit_draw": [0.90, 0.95, 0.10, 0.20, 0.80, 0.30],
            "aca_take_up_rate": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "is_aca_ptc_eligible": [True, True, True, True, True, True],
            "health_insurance_premiums_without_medicare_part_b": [
                0.0,
                300.0,
                200.0,
                0.0,
                500.0,
                100.0,
            ],
            "assigned_aca_ptc": [0.0, 200.0, 200.0, 0.0, 0.0, 100.0],
            "weighted_assigned_aca_ptc": [0.0, 200.0, 200.0, 0.0, 0.0, 100.0],
            "slcsp": [0.0, 500.0, 500.0, 500.0, 0.0, 500.0],
        }
    )


def _make_aca_state_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "02"],
            "target": [2.0, 1.0],
        }
    )


def _make_aca_bronze_targets() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "state_fips": ["01", "02"],
            "target": [1.0, 1.0],
        }
    )


def test_us_puf_manifest_prefix_runs_aggregate_disaggregation() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"]
    mini_puf = _make_runtime_mini_puf()

    result = run_source_stage(
        stage,
        tables={"puf_tax_unit": mini_puf},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="disaggregate_aggregate_records",
    )

    expected = disaggregate_puf_aggregate_records(
        derive_puf_policyengine_variables(
            mini_puf,
            qualified_tuition_primary_source="E03230",
            qualified_tuition_optional_source="E87530",
            alimony_income_source="E00800",
            alimony_expense_source="E03500",
            casualty_loss_source="E20500",
            domestic_production_ald_source="E03240",
            educator_expense_source="E03220",
            unreimbursed_business_employee_expenses_source="E20400",
            farm_operations_income_source="E02100",
            farm_rent_income_source="E27200",
            investment_income_elected_form_4952_source="E58990",
        ),
        seed=42,
    )
    pd.testing.assert_frame_equal(result, expected)
    assert not result["RECID"].isin(AGGREGATE_RECIDS).any()
    assert (result["RECID"] >= SYNTHETIC_RECID_START).any()
    assert "ordinary_dividend_income" not in result.columns
    assert "dividend_income" not in result.columns
    assert "qualified_dividend_income" in result.columns
    assert "non_qualified_dividend_income" in result.columns
    assert np.allclose(result["qualified_dividend_income"], result["E00650"])
    assert np.allclose(
        result["non_qualified_dividend_income"],
        result["E00600"] - result["E00650"],
    )
    assert np.allclose(
        result["qualified_tuition_expenses"],
        np.maximum(result["E03230"], result["E87530"]),
    )
    assert (result["qualified_tuition_expenses"] >= 0.0).all()
    assert np.allclose(result["alimony_income"], result["E00800"])
    assert np.allclose(result["alimony_expense"], result["E03500"])
    assert np.allclose(result["casualty_loss"], result["E20500"])
    assert (result["casualty_loss"] >= 0.0).all()
    assert np.allclose(result["domestic_production_ald"], result["E03240"])
    assert (result["domestic_production_ald"] >= 0.0).all()
    assert np.allclose(result["educator_expense"], result["E03220"])
    assert (result["educator_expense"] >= 0.0).all()
    assert np.allclose(
        result["unreimbursed_business_employee_expenses"], result["E20400"]
    )
    assert np.allclose(result["farm_operations_income"], result["E02100"])
    assert np.allclose(result["farm_rent_income"], result["E27200"])
    assert np.allclose(result["investment_income_elected_form_4952"], result["E58990"])
    assert (result["investment_income_elected_form_4952"] >= 0.0).all()


def test_us_puf_manifest_prefix_uses_build_seed() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"]
    mini_puf = _make_runtime_mini_puf()

    first = run_source_stage(
        stage,
        tables={"puf_tax_unit": mini_puf},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=1, target_year=2024),
        stop_after="disaggregate_aggregate_records",
    )
    second = run_source_stage(
        stage,
        tables={"puf_tax_unit": mini_puf},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=2, target_year=2024),
        stop_after="disaggregate_aggregate_records",
    )

    assert not first.equals(second)


def test_us_puf_manifest_rejects_invalid_dividend_sources() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["puf_tax_detail"]
    mini_puf = _make_runtime_mini_puf()
    mini_puf.loc[mini_puf.index[0], "E00650"] = (
        mini_puf.loc[mini_puf.index[0], "E00600"] + 1.0
    )

    with pytest.raises(SourceRuntimeError, match="qualified dividends above ordinary"):
        run_source_stage(
            stage,
            tables={"puf_tax_unit": mini_puf},
            operation_handlers=us_source_operation_handlers(),
            config=SourceRuntimeConfig(seed=42, target_year=2024),
            stop_after="derive_puf_policyengine_variables",
        )


def test_us_puf_handler_validates_packaged_spec_shape() -> None:
    mini_puf = _make_runtime_mini_puf()
    bad_operation = SourceOperationSpec.from_mapping(
        {
            "kind": "disaggregate_aggregate_records",
            "method": "donor_template_calibration",
            "spec": "puf_aggregate_record_disaggregation",
            "replace_records": [999999],
            "weight": "s006",
            "amount_columns": "irs_puf_amount_columns",
        }
    )

    with pytest.raises(SourceRuntimeError, match="replace_records"):
        disaggregate_us_puf_aggregate_records_from_manifest(
            mini_puf,
            bad_operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={},
            ),
        )


def test_us_puf_handler_rejects_unknown_parameters() -> None:
    mini_puf = _make_runtime_mini_puf()
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "disaggregate_aggregate_records",
            "method": "donor_template_calibration",
            "spec": "puf_aggregate_record_disaggregation",
            "replace_records": [999996, 999997, 999998, 999999],
            "weight": "s006",
            "amount_columns": "irs_puf_amount_columns",
            "seed_from_build_config": True,
            "seed": 7,
        }
    )

    with pytest.raises(SourceRuntimeError, match="unsupported parameter"):
        disaggregate_us_puf_aggregate_records_from_manifest(
            mini_puf,
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={},
            ),
        )


def test_us_puf_policyengine_variable_handler_rejects_unknown_parameters() -> None:
    mini_puf = _make_runtime_mini_puf()
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "derive_puf_policyengine_variables",
            "ordinary_dividend_source": "E00600",
            "qualified_dividend_source": "E00650",
            "unsupported": "field",
        }
    )

    with pytest.raises(SourceRuntimeError, match="unsupported parameter"):
        derive_us_puf_policyengine_variables_from_manifest(
            mini_puf,
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={},
            ),
        )


def test_us_puf_handler_requires_build_seed() -> None:
    mini_puf = _make_runtime_mini_puf()
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "disaggregate_aggregate_records",
            "method": "donor_template_calibration",
            "spec": "puf_aggregate_record_disaggregation",
            "replace_records": [999996, 999997, 999998, 999999],
            "weight": "s006",
            "amount_columns": "irs_puf_amount_columns",
            "seed_from_build_config": False,
        }
    )

    with pytest.raises(SourceRuntimeError, match="seed_from_build_config=true"):
        disaggregate_us_puf_aggregate_records_from_manifest(
            mini_puf,
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={},
            ),
        )


def test_us_aca_take_up_manifest_prefix_aggregates_assigns_and_calibrates() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]

    result = run_source_stage(
        stage,
        tables={
            "cps_person": _make_aca_people(),
            "tax_unit": _make_aca_tax_units(),
            "cms_aca_aptc_recipients_by_state": _make_aca_state_targets(),
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="calibrate_binary_assignment",
    )

    assigned = result.set_index("tax_unit_id")["takes_up_aca_if_eligible"]
    assert bool(assigned.loc[20]) is True
    assert bool(assigned.loc[30]) is True
    assert bool(assigned.loc[60]) is True
    assert assigned.sum() == 3
    assert result.groupby("state_fips")["takes_up_aca_if_eligible"].sum().to_dict() == {
        "01": 2,
        "02": 1,
    }
    assert result["takes_up_aca_if_eligible"].nunique() == 2


def test_us_aca_take_up_assignment_is_seed_stable_without_draw_column() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
    tax_units = _make_aca_tax_units().drop(columns=["stable_tax_unit_draw"])
    tax_units["aca_take_up_rate"] = 0.5

    first = run_source_stage(
        stage,
        tables={"cps_person": _make_aca_people(), "tax_unit": tax_units},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=1, target_year=2024),
        stop_after="assign_binary_from_rate",
    )
    second = run_source_stage(
        stage,
        tables={"cps_person": _make_aca_people(), "tax_unit": tax_units},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=1, target_year=2024),
        stop_after="assign_binary_from_rate",
    )
    different_seed = run_source_stage(
        stage,
        tables={"cps_person": _make_aca_people(), "tax_unit": tax_units},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=2, target_year=2024),
        stop_after="assign_binary_from_rate",
    )

    pd.testing.assert_series_equal(
        first["stable_tax_unit_draw"],
        second["stable_tax_unit_draw"],
    )
    assert not first["stable_tax_unit_draw"].equals(
        different_seed["stable_tax_unit_draw"]
    )


def test_us_aca_selected_plan_ratio_uses_neutral_defaults_and_clips() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]

    result = run_source_stage(
        stage,
        tables={
            "cps_person": _make_aca_people(),
            "tax_unit": _make_aca_tax_units(),
            "cms_aca_aptc_recipients_by_state": _make_aca_state_targets(),
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="support_clip",
    )

    ratios = result.set_index("tax_unit_id")[
        "selected_marketplace_plan_benchmark_ratio"
    ]
    assert ratios.loc[10] == 1.0
    assert ratios.loc[20] == 1.0
    assert ratios.loc[30] == 0.8
    assert ratios.loc[50] == 1.0
    assert ratios.loc[60] == 0.5
    assert ratios.nunique() > 1


def test_us_aca_marketplace_stage_runs_through_expression_calibration() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]

    result = run_source_stage(
        stage,
        tables={
            "cps_person": _make_aca_people(),
            "tax_unit": _make_aca_tax_units(),
            "cms_aca_aptc_recipients_by_state": _make_aca_state_targets(),
            "cms_aca_bronze_aptc_consumers_by_state": _make_aca_bronze_targets(),
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
    )

    below_benchmark = (result["selected_marketplace_plan_benchmark_ratio"] < 1.0) & (
        result["assigned_aca_ptc"] > 0
    )
    assert below_benchmark.groupby(result["state_fips"]).sum().to_dict() == {
        "01": 1,
        "02": 1,
    }


def test_us_aca_take_up_calibration_can_remove_non_anchor_assignments() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
    tax_units = _make_aca_tax_units()
    tax_units["aca_take_up_rate"] = 1.0
    targets = pd.DataFrame({"state_fips": ["01", "02"], "target": [1.0, 0.0]})

    result = run_source_stage(
        stage,
        tables={
            "cps_person": _make_aca_people(),
            "tax_unit": tax_units,
            "cms_aca_aptc_recipients_by_state": targets,
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="calibrate_binary_assignment",
    )

    assigned = result.set_index("tax_unit_id")["takes_up_aca_if_eligible"]
    assert bool(assigned.loc[20]) is True
    assert result.groupby("state_fips")["takes_up_aca_if_eligible"].sum().to_dict() == {
        "01": 1,
        "02": 0,
    }


def test_us_aca_take_up_calibration_respects_eligibility_domain() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
    tax_units = _make_aca_tax_units()
    tax_units["stable_tax_unit_draw"] = [0.01, 0.95, 0.10, 0.20, 0.80, 0.30]
    tax_units["is_aca_ptc_eligible"] = [False, True, True, True, True, True]
    targets = pd.DataFrame({"state_fips": ["01", "02"], "target": [2.0, 1.0]})

    result = run_source_stage(
        stage,
        tables={
            "cps_person": _make_aca_people(),
            "tax_unit": tax_units,
            "cms_aca_aptc_recipients_by_state": targets,
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="calibrate_binary_assignment",
    )

    assigned = result.set_index("tax_unit_id")["takes_up_aca_if_eligible"]
    assert bool(assigned.loc[10]) is False
    assert result.groupby("state_fips")["takes_up_aca_if_eligible"].sum().to_dict() == {
        "01": 2,
        "02": 1,
    }


def test_us_aca_take_up_calibration_uses_declared_weights() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
    tax_units = _make_aca_tax_units()
    tax_units["tax_unit_weight"] = [3.0, 1.0, 2.0, 1.0, 5.0, 2.0]
    tax_units["stable_tax_unit_draw"] = [0.99, 0.95, 0.10, 0.20, 0.80, 0.30]
    targets = pd.DataFrame({"state_fips": ["01", "02"], "target": [3.0, 2.0]})

    result = run_source_stage(
        stage,
        tables={
            "cps_person": _make_aca_people(),
            "tax_unit": tax_units,
            "cms_aca_aptc_recipients_by_state": targets,
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="calibrate_binary_assignment",
    )

    weighted = (
        result["takes_up_aca_if_eligible"].astype(float) * result["tax_unit_weight"]
    )
    assert weighted.groupby(result["state_fips"]).sum().to_dict() == {
        "01": 3.0,
        "02": 2.0,
    }


def test_us_aca_take_up_calibration_uses_soi_ptc_targets() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
    people = _make_aca_people()
    people["reported_has_subsidized_marketplace_health_coverage_at_interview"] = False
    tax_units = _make_aca_tax_units()
    tax_units["assigned_aca_ptc"] = [1000.0, 900.0, 100.0, 100.0, 500.0, 100.0]
    tax_units["weighted_assigned_aca_ptc"] = tax_units["assigned_aca_ptc"]
    cms_targets = pd.DataFrame({"state_fips": ["01", "02"], "target": [4.0, 2.0]})
    soi_return_targets = pd.DataFrame(
        {"state_fips": ["01", "02"], "target": [2.0, 1.0]}
    )
    soi_amount_targets = pd.DataFrame(
        {"state_fips": ["01", "02"], "target": [200.0, 100.0]}
    )

    result = run_source_stage(
        stage,
        tables={
            "cps_person": people,
            "tax_unit": tax_units,
            "cms_aca_aptc_recipients_by_state": cms_targets,
            "irs_soi_premium_tax_credit_returns_by_state": soi_return_targets,
            "irs_soi_premium_tax_credit_amount_by_state": soi_amount_targets,
        },
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="support_clip",
    )

    assigned = result.set_index("tax_unit_id")["takes_up_aca_if_eligible"]
    assert assigned.to_dict() == {
        10: False,
        20: False,
        30: True,
        40: True,
        50: False,
        60: True,
    }
    assigned_amount = result["assigned_aca_ptc"] * result[
        "takes_up_aca_if_eligible"
    ].astype(float)
    assert assigned_amount.groupby(result["state_fips"]).sum().to_dict() == {
        "01": 200.0,
        "02": 100.0,
    }
    assigned_returns = result["household_weight"] * result[
        "takes_up_aca_if_eligible"
    ].astype(float)
    assert assigned_returns.groupby(result["state_fips"]).sum().to_dict() == {
        "01": 2.0,
        "02": 1.0,
    }


def test_us_aca_take_up_calibration_requires_declared_weight_column() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
    tax_units = _make_aca_tax_units().drop(columns=["tax_unit_weight"])

    with pytest.raises(SourceRuntimeError, match="tax_unit_weight"):
        run_source_stage(
            stage,
            tables={
                "cps_person": _make_aca_people(),
                "tax_unit": tax_units,
                "cms_aca_aptc_recipients_by_state": _make_aca_state_targets(),
            },
            operation_handlers=us_source_operation_handlers(),
            config=SourceRuntimeConfig(seed=42, target_year=2024),
            stop_after="calibrate_binary_assignment",
        )


def test_us_aca_reported_anchor_does_not_override_ineligibility() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
    people = _make_aca_people()
    tax_units = _make_aca_tax_units()
    tax_units.loc[tax_units["tax_unit_id"] == 20, "is_aca_ptc_eligible"] = False

    result = run_source_stage(
        stage,
        tables={"cps_person": people, "tax_unit": tax_units},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=42, target_year=2024),
        stop_after="assign_binary_from_rate",
    )

    assigned = result.set_index("tax_unit_id")["takes_up_aca_if_eligible"]
    assert bool(assigned.loc[20]) is False


def test_us_aca_take_up_assignment_requires_declared_eligibility_column() -> None:
    stage = US_SOURCE_MANIFEST.stage_map()["aca_marketplace_inputs"]
    tax_units = _make_aca_tax_units().drop(columns=["is_aca_ptc_eligible"])

    with pytest.raises(SourceRuntimeError, match="is_aca_ptc_eligible"):
        run_source_stage(
            stage,
            tables={"cps_person": _make_aca_people(), "tax_unit": tax_units},
            operation_handlers=us_source_operation_handlers(),
            config=SourceRuntimeConfig(seed=42, target_year=2024),
            stop_after="assign_binary_from_rate",
        )


def test_us_aca_tax_unit_aggregation_rejects_wrong_operation() -> None:
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "aggregate_person_to_tax_unit",
            "person_table": "cps_person",
            "tax_unit_table": "tax_unit",
            "person_tax_unit_id": "tax_unit_id",
            "tax_unit_id": "tax_unit_id",
            "aggregates": ["has_marketplace_health_coverage_at_interview"],
            "operation": "sum",
        }
    )

    with pytest.raises(SourceRuntimeError, match="operation='any'"):
        aggregate_us_person_to_tax_unit_from_manifest(
            None,
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={
                    "cps_person": _make_aca_people(),
                    "tax_unit": _make_aca_tax_units(),
                },
            ),
        )


def _make_person_grain_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "state_fips": ["01", "01", "02", "02"],
            "takes_up_medicaid_if_eligible": [True, False, False, False],
            "person_weight": [1.0, 1.0, 1.0, 1.0],
            "stable_person_draw": [0.1, 0.4, 0.2, 0.9],
        }
    )


def _make_person_grain_targets() -> pd.DataFrame:
    return pd.DataFrame({"state_fips": ["01", "02"], "target": [2.0, 1.0]})


def test_us_binary_calibration_requires_declared_draw_column() -> None:
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "calibrate_binary_assignment",
            "variable": "takes_up_medicaid_if_eligible",
            "targets": ["cms_medicaid_enrollment_by_state"],
            "weight": "person_weight",
        }
    )

    with pytest.raises(
        SourceRuntimeError,
        match="US binary calibration requires a non-empty 'draw' parameter",
    ):
        calibrate_us_binary_assignment_from_manifest(
            _make_person_grain_frame(),
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={
                    "cms_medicaid_enrollment_by_state": (_make_person_grain_targets()),
                },
            ),
        )


def test_us_binary_calibration_errors_are_not_aca_labeled() -> None:
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "calibrate_binary_assignment",
            "variable": "takes_up_medicaid_if_eligible",
            "targets": ["cms_medicaid_enrollment_by_state"],
            "weight": "person_weight",
            "draw": "stable_person_draw",
        }
    )

    with pytest.raises(SourceRuntimeError) as excinfo:
        calibrate_us_binary_assignment_from_manifest(
            _make_person_grain_frame().drop(columns=["person_weight"]),
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={
                    "cms_medicaid_enrollment_by_state": (_make_person_grain_targets()),
                },
            ),
        )

    message = str(excinfo.value)
    assert "US binary calibration frame" in message
    assert "person_weight" in message
    assert "ACA" not in message


def test_us_binary_calibration_runs_at_person_grain_with_declared_draw() -> None:
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "calibrate_binary_assignment",
            "variable": "takes_up_medicaid_if_eligible",
            "targets": ["cms_medicaid_enrollment_by_state"],
            "weight": "person_weight",
            "draw": "stable_person_draw",
        }
    )

    result = calibrate_us_binary_assignment_from_manifest(
        _make_person_grain_frame(),
        operation,
        context=SourceRuntimeContext(
            config=SourceRuntimeConfig(seed=42),
            tables={
                "cms_medicaid_enrollment_by_state": _make_person_grain_targets(),
            },
        ),
    )

    assigned = result.set_index("person_id")["takes_up_medicaid_if_eligible"]
    assert assigned.to_dict() == {1: True, 2: True, 3: True, 4: False}


def test_us_joint_binary_calibration_requires_declared_draw_column() -> None:
    operation = SourceOperationSpec.from_mapping(
        {
            "kind": "calibrate_binary_assignment_joint_targets",
            "variable": "takes_up_medicaid_if_eligible",
            "count_targets": ["count_targets_by_state"],
            "amount_targets": ["amount_targets_by_state"],
            "count_weight": "person_weight",
            "amount_weight": "person_weight",
        }
    )

    with pytest.raises(
        SourceRuntimeError,
        match="US joint binary calibration requires a non-empty 'draw' parameter",
    ):
        calibrate_us_binary_assignment_joint_targets_from_manifest(
            _make_person_grain_frame(),
            operation,
            context=SourceRuntimeContext(
                config=SourceRuntimeConfig(seed=42),
                tables={
                    "count_targets_by_state": _make_person_grain_targets(),
                    "amount_targets_by_state": _make_person_grain_targets(),
                },
            ),
        )
