"""ASEC child-support restoration and retired PUF-half joint QRF treatment."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.child_support as module
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime.child_support import (
    CHILD_SUPPORT_ARCHIVED_PUF_IMPUTATION_URL,
    CHILD_SUPPORT_ARCHIVED_PUF_OUTPUTS_URL,
    CHILD_SUPPORT_EXPENSE_ARCHIVED_DERIVATION_URL,
    CHILD_SUPPORT_RECEIVED_ARCHIVED_DERIVATION_URL,
    US_CHILD_SUPPORT_OUTPUT_COLUMNS,
    derive_us_child_support_from_manifest,
    impute_us_child_support_to_puf_support_from_manifest,
    us_child_support_signal_gate,
    us_child_support_stage_spec,
    with_us_child_support_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_RECEIVED, _EXPENSE = US_CHILD_SUPPORT_OUTPUT_COLUMNS


def _person_source() -> pd.DataFrame:
    count = 10
    return pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype="int64"),
            "person_household_id": np.arange(1, count + 1, dtype="int64") * 10,
            "person_tax_unit_id": np.arange(1, count + 1, dtype="int64") * 100,
            "person_spm_unit_id": np.arange(1, count + 1, dtype="int64") * 1_000,
            "person_family_id": np.arange(1, count + 1, dtype="int64") * 10_000,
            "person_marital_unit_id": (
                np.arange(1, count + 1, dtype="int64") * 100_000
            ),
            "CSP_VAL": [3_600.0, *([0.0] * 9)],
            "CHSP_VAL": [2_400.0, *([0.0] * 9)],
            "WSAL_VAL": np.linspace(10_000.0, 100_000.0, count),
            "SEMP_VAL": np.zeros(count),
            "employment_income_before_lsr": np.linspace(10_000.0, 100_000.0, count),
            "self_employment_income_before_lsr": np.zeros(count),
            "age": np.arange(25, 25 + count),
            "is_female": np.asarray([False, True] * 5),
            "has_esi": np.asarray([True, False] * 5),
            "tax_unit_role_input": ["HEAD"] * count,
            "social_security_retirement": np.zeros(count),
            "social_security_disability": np.zeros(count),
            "social_security_dependents": np.zeros(count),
            "social_security_survivors": np.zeros(count),
        }
    )


def _frame() -> Frame:
    person = _person_source()
    count = len(person)
    ids = {
        "household": person["person_household_id"].to_numpy(),
        "tax_unit": person["person_tax_unit_id"].to_numpy(),
        "spm_unit": person["person_spm_unit_id"].to_numpy(),
        "family": person["person_family_id"].to_numpy(),
        "marital_unit": person["person_marital_unit_id"].to_numpy(),
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": values}) for entity, values in ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = ["SINGLE"] * count
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.arange(1.0, count + 1.0),
                WeightKind.DESIGN,
            )
        },
    )


def _derive(person: pd.DataFrame) -> pd.DataFrame:
    operation = next(
        operation
        for operation in us_child_support_stage_spec().operations
        if operation.kind == "derive_child_support_inputs"
    )
    return derive_us_child_support_from_manifest(person, operation, None)


__all__ = [name for name in globals() if not name.startswith("__")]
