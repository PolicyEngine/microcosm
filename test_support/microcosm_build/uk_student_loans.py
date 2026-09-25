# ruff: noqa: F401
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


__all__ = [name for name in globals() if not name.startswith("__")]
