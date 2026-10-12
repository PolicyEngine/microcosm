"""ASEC workers' compensation restoration and PUF-half QRF treatment."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.workers_compensation as module
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.workers_compensation import (
    US_WORKERS_COMPENSATION_OUTPUT_COLUMNS,
    US_WORKERS_COMPENSATION_REQUIRED_SOURCE_COLUMNS,
    WORKERS_COMPENSATION_ARCHIVED_DERIVATION_URL,
    WORKERS_COMPENSATION_ARCHIVED_PUF_IMPUTATION_URL,
    WORKERS_COMPENSATION_ARCHIVED_PUF_OUTPUTS_URL,
    WORKERS_COMPENSATION_ARCHIVED_SOURCE_COLUMNS_URL,
    derive_us_workers_compensation_from_manifest,
    impute_us_workers_compensation_to_puf_support_from_manifest,
    us_workers_compensation_signal_gate,
    us_workers_compensation_stage_spec,
    with_us_workers_compensation,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


_OUTPUT = US_WORKERS_COMPENSATION_OUTPUT_COLUMNS[0]
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
ROOT = _TEST_PATHS.repository


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _person_source() -> pd.DataFrame:
    count = 100
    workers_compensation = np.zeros(count)
    workers_compensation[0] = 3_600.0
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
            "WC_VAL": workers_compensation,
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
        for operation in us_workers_compensation_stage_spec().operations
        if operation.kind == "derive_workers_compensation"
    )
    return derive_us_workers_compensation_from_manifest(person, operation, None)


def _stacked_workers_compensation_frame() -> Frame:
    direct = with_us_workers_compensation(_frame(), seed=0, time_period=2024)
    expanded = clone_us_frame_for_puf_support(direct)
    tables = {entity: expanded.table(entity).copy() for entity in expanded.entities}
    person = tables["person"]
    physical_asec = person["person_source_id"].le(50)
    person["person_spine_source_id"] = person["person_source_id"]
    person["person_support_channel"] = np.where(physical_asec, "asec", "acs")
    person.loc[~physical_asec, "WC_VAL"] = np.nan
    return Frame(
        tables,
        expanded.schema,
        {entity: expanded.weights_for(entity) for entity in expanded.weighted_entities},
        expanded.strata,
        mass_log=expanded.mass_log,
        metadata=expanded.metadata,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
