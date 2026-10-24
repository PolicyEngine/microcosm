"""ASEC childcare restoration and PUF-half QRF treatment."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.childcare as module
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime.childcare import (
    US_CHILDCARE_OUTPUT_COLUMNS,
    derive_us_childcare_from_manifest,
    impute_us_childcare_to_puf_support_from_manifest,
    us_childcare_signal_gate,
    us_childcare_stage_spec,
    with_us_childcare_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_CHILDCARE_OUTPUT_COLUMNS[0]


def _person_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.arange(1, 7, dtype="int64"),
            "person_household_id": [10, 10, 20, 30, 40, 50],
            "person_tax_unit_id": [100, 100, 200, 300, 400, 500],
            "person_spm_unit_id": [1_000, 1_000, 2_000, 3_000, 4_000, 5_000],
            "person_family_id": [10_000, 10_000, 20_000, 30_000, 40_000, 50_000],
            "person_marital_unit_id": [
                100_000,
                100_000,
                200_000,
                300_000,
                400_000,
                500_000,
            ],
            "SPM_CHILDCAREXPNS": [800.0, 800.0, 0.0, 1_200.0, 0.0, 0.0],
            "WSAL_VAL": [50_000.0, 20_000.0, 0.0, 35_000.0, 10_000.0, 0.0],
            "SEMP_VAL": [0.0, 0.0, 20_000.0, 0.0, 5_000.0, 0.0],
            "employment_income_before_lsr": [
                50_000.0,
                20_000.0,
                0.0,
                35_000.0,
                10_000.0,
                0.0,
            ],
            "self_employment_income_before_lsr": [
                0.0,
                0.0,
                20_000.0,
                0.0,
                5_000.0,
                0.0,
            ],
            "age": [35, 33, 45, 29, 55, 70],
            "is_female": [False, True, True, False, True, False],
            "has_esi": [True, True, False, True, False, False],
            "tax_unit_role_input": [
                "HEAD",
                "SPOUSE",
                "HEAD",
                "HEAD",
                "HEAD",
                "HEAD",
            ],
            "social_security_retirement": [0.0, 0.0, 0.0, 0.0, 0.0, 15_000.0],
            "social_security_disability": [0.0] * 6,
            "social_security_dependents": [0.0] * 6,
            "social_security_survivors": [0.0] * 6,
        }
    )


def _frame() -> Frame:
    person = _person_source()
    ids = {
        "household": [10, 20, 30, 40, 50],
        "tax_unit": [100, 200, 300, 400, 500],
        "spm_unit": [1_000, 2_000, 3_000, 4_000, 5_000],
        "family": [10_000, 20_000, 30_000, 40_000, 50_000],
        "marital_unit": [100_000, 200_000, 300_000, 400_000, 500_000],
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": np.asarray(values, dtype="int64")})
        for entity, values in ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = [
        "JOINT",
        "SINGLE",
        "SINGLE",
        "SINGLE",
        "SINGLE",
    ]
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(5, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    operation = next(
        operation
        for operation in us_childcare_stage_spec().operations
        if operation.kind == "derive_childcare_inputs"
    )
    return derive_us_childcare_from_manifest(frame, operation, None)


__all__ = [name for name in globals() if not name.startswith("__")]
