"""ASEC disability-benefit restoration and PUF-half QRF treatment."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.disability_benefits as module
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime.disability_benefits import (
    DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL,
    DISABILITY_BENEFITS_ARCHIVED_PUF_IMPUTATION_URL,
    DISABILITY_BENEFITS_ARCHIVED_PUF_OUTPUTS_URL,
    DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL,
    US_DISABILITY_BENEFITS_OUTPUT_COLUMNS,
    US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS,
    derive_us_disability_benefits_from_manifest,
    impute_us_disability_benefits_to_puf_support_from_manifest,
    us_disability_benefits_signal_gate,
    us_disability_benefits_stage_spec,
    with_us_disability_benefits,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.release_input_coverage import (
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]
_PREDICTORS = (
    "age",
    "is_male",
    "has_esi",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "employment_income",
    "self_employment_income",
    "social_security",
)


def _person_source() -> pd.DataFrame:
    count = 100
    first_amount = np.zeros(count)
    first_code = np.zeros(count)
    second_amount = np.zeros(count)
    second_code = np.zeros(count)
    first_amount[0] = 3_600.0
    first_code[0] = 2.0
    # A workers'-compensation amount in the other slot must remain excluded.
    second_amount[0] = 1_200.0
    second_code[0] = 1.0
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
            "DIS_VAL1": first_amount,
            "DIS_SC1": first_code,
            "DIS_VAL2": second_amount,
            "DIS_SC2": second_code,
            "WSAL_VAL": np.linspace(0.0, 99_000.0, count),
            "SEMP_VAL": np.zeros(count),
            "employment_income_before_lsr": np.linspace(0.0, 99_000.0, count),
            "self_employment_income_before_lsr": np.zeros(count),
            "age": np.arange(20, 20 + count),
            "is_female": np.tile([False, True], count // 2),
            "has_esi": np.tile([True, False], count // 2),
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
                np.ones(count, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _derive(person: pd.DataFrame) -> pd.DataFrame:
    operation = next(
        operation
        for operation in us_disability_benefits_stage_spec().operations
        if operation.kind == "derive_disability_benefits"
    )
    return derive_us_disability_benefits_from_manifest(person, operation, None)


__all__ = [name for name in globals() if not name.startswith("__")]
