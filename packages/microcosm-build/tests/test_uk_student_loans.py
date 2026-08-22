from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.student_loans import (
    PLAN_PRIORITY,
    STUDENT_LOAN_ENUM_DOMAIN,
    _assert_student_loans_stage_parameters,
    assign_student_loan_plans,
    load_slc_liable_stocks,
)


def _stocks(*, plan_2: float, plan_5: float, year: int = 2025):
    return {
        "plans": {
            "plan_2": {"liable": {str(year): plan_2}},
            "plan_5": {"liable": {str(year): plan_5}},
        }
    }


def _frame(
    *,
    ages,
    repayments,
    regions=None,
    education=None,
    weights=None,
):
    n = len(ages)
    ids = np.arange(1, n + 1, dtype="int64")
    regions = regions or ["LONDON"] * n
    education = education or ["TERTIARY"] * n
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "age": ages,
                "student_loan_repayments": repayments,
                "highest_education": education,
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame({"household_id": ids, "region": regions}),
        household_weights=np.ones(n) if weights is None else weights,
        time_period="2024",
    )


@lru_cache
def _stage():
    return load_country_spec("uk").sources.stage_map()["student_loans"]


def _drift(operation_index: int, parameter: str):
    stage = _stage()
    operations = list(stage.operations)
    operation = operations[operation_index]
    operations[operation_index] = SourceOperationSpec(
        operation.kind,
        {**operation.parameters, parameter: "__drift__"},
    )
    return replace(stage, operations=tuple(operations))


def test_reported_cohort_boundaries_and_country_independence() -> None:
    # At 2025: these ages imply start years 2011, 2012, 2022, and 2023.
    result = assign_student_loan_plans(
        _frame(
            ages=[32, 31, 21, 20],
            repayments=[100.0] * 4,
            regions=["WALES", "SCOTLAND", "NORTHERN_IRELAND", "WALES"],
        ),
        stocks=_stocks(plan_2=0, plan_5=0),
        year=2025,
    )

    assert result.frame.table("person")["student_loan_plan"].tolist() == [
        "PLAN_1",
        "PLAN_2",
        "PLAN_2",
        "PLAN_5",
    ]


def test_topups_apply_eligibility_gates_and_plan5_priority() -> None:
    result = assign_student_loan_plans(
        _frame(
            ages=[20, 20, 20, 31, 31, 40],
            repayments=[0.0] * 6,
            regions=["LONDON", "WALES", "LONDON", "LONDON", "LONDON", "LONDON"],
            education=["TERTIARY", "TERTIARY", "GCSE", "TERTIARY", "GCSE", "TERTIARY"],
        ),
        stocks=_stocks(plan_2=1, plan_5=1),
        year=2025,
    )
    plans = result.frame.table("person")["student_loan_plan"].tolist()

    assert plans == ["PLAN_5", "NONE", "NONE", "PLAN_2", "NONE", "NONE"]
    assert tuple(result.plans) == PLAN_PRIORITY
    assert "PLAN_4" not in plans


def test_rate_rule_receipt_uses_weighted_shortfall() -> None:
    result = assign_student_loan_plans(
        _frame(
            ages=[31, 31],
            repayments=[0.0, 0.0],
            weights=[2.0, 2.0],
        ),
        stocks=_stocks(plan_2=2, plan_5=0),
        year=2025,
    )
    receipt = result.plans["PLAN_2"]

    assert receipt.shortfall == 2.0
    assert receipt.eligible_mass == 4.0
    assert receipt.rate == 0.5
    assert receipt.final_england_count == receipt.topped_up_mass


def test_calibration_year_changes_cohort_assignment() -> None:
    frame = _frame(ages=[31], repayments=[100.0])

    at_2025 = assign_student_loan_plans(
        frame, stocks=_stocks(plan_2=0, plan_5=0), year=2025
    )
    at_2036 = assign_student_loan_plans(
        frame, stocks=_stocks(plan_2=0, plan_5=0, year=2036), year=2036
    )

    assert at_2025.frame.table("person").student_loan_plan.iloc[0] == "PLAN_2"
    assert at_2036.frame.table("person").student_loan_plan.iloc[0] == "PLAN_5"
    assert at_2036.calibration_year == 2036


def test_committed_stocks_pin_full_2025_to_2030_series() -> None:
    stocks = load_slc_liable_stocks()["plans"]

    assert stocks["plan_2"]["liable"] == {
        "2025": 8_940_000,
        "2026": 9_710_000,
        "2027": 10_360_000,
        "2028": 10_615_000,
        "2029": 10_600_000,
        "2030": 10_525_000,
    }
    assert stocks["plan_5"]["above_threshold"]["2030"] == 1_235_000


def test_values_match_policyengine_enum_when_available() -> None:
    module = pytest.importorskip(
        "policyengine_uk.variables.gov.hmrc.student_loans.student_loan_plan"
    )
    engine_names = set(module.StudentLoanPlan.__members__)

    assert set(STUDENT_LOAN_ENUM_DOMAIN) <= engine_names
    assert "PLAN_4" in engine_names
    assert "PLAN_4" not in STUDENT_LOAN_ENUM_DOMAIN


@pytest.mark.parametrize(
    "operation_index,parameter",
    [
        *[
            (0, name)
            for name in (
                "year_rule",
                "start_year_formula",
                "reported_repayment_test",
                "reported_country_gate",
                "plan_1_before",
                "plan_5_from",
                "enum_domain",
                "plan_4_imputation",
            )
        ],
        *[
            (1, name)
            for name in (
                "priority",
                "resource",
                "stock_series",
                "year_rule",
                "age_min",
                "age_max",
                "cohort_start_min",
                "eligible_region_exclusions",
                "highest_education",
                "seed",
                "salt",
            )
        ],
        *[
            (2, name)
            for name in (
                "priority",
                "resource",
                "stock_series",
                "year_rule",
                "age_min",
                "age_max",
                "cohort_start_min",
                "cohort_start_max_exclusive",
                "eligible_region_exclusions",
                "highest_education",
                "seed",
                "salt",
            )
        ],
    ],
)
def test_manifest_drift_assert_covers_every_reviewed_parameter(
    operation_index: int, parameter: str
) -> None:
    with pytest.raises((ValueError, KeyError), match="drifted|priority"):
        _assert_student_loans_stage_parameters(
            _drift(operation_index, parameter),
            stocks=load_slc_liable_stocks(),
            year=2025,
        )


def test_stock_drift_assert_rejects_2025_change() -> None:
    stocks = _stocks(plan_2=1, plan_5=10_000)
    with pytest.raises(ValueError, match="PLAN_2.*drifted"):
        _assert_student_loans_stage_parameters(_stage(), stocks=stocks, year=2025)


def test_drift_assert_rejects_extra_keys_and_operations() -> None:
    with pytest.raises(ValueError, match="drifted"):
        _assert_student_loans_stage_parameters(
            _drift(2, "undeclared_extra_key"),
            stocks=load_slc_liable_stocks(),
            year=2025,
        )
    stage = _stage()
    extra = replace(stage, operations=(*stage.operations, stage.operations[-1]))
    with pytest.raises(ValueError, match="operation order drifted"):
        _assert_student_loans_stage_parameters(
            extra, stocks=load_slc_liable_stocks(), year=2025
        )
