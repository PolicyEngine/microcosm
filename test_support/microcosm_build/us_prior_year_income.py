"""Archived adjacent-ASEC prior-year-income restoration."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.prior_year_income as module
from microcosm.build.gates import GateReport
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.prior_year_income import (
    PRIOR_YEAR_INCOME_ARCHIVED_DERIVATION_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_FINALIZER_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_FORMULA_OUTPUT_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_PUF_IMPUTATION_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_PUF_OUTPUTS_URL,
    PRIOR_YEAR_INCOME_ARCHIVED_PUF_SPLICE_URL,
    US_PRIOR_YEAR_INCOME_NONCONSTANT_PERSON_COLUMNS,
    US_PRIOR_YEAR_INCOME_OUTPUT_COLUMNS,
    US_PRIOR_YEAR_INCOME_PERSISTED_OUTPUT_COLUMNS,
    derive_us_prior_year_income_from_manifest,
    impute_us_prior_year_income_to_puf_support_from_manifest,
    us_prior_year_income_signal_gate,
    us_prior_year_income_source_reconciliation_gate,
    us_prior_year_income_stage_spec,
    with_us_prior_year_income_inputs,
)
from microcosm.build.us_runtime.puf_support import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
)
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine


def _frame(person: pd.DataFrame) -> Frame:
    person = person.reset_index(drop=True).copy()
    n = len(person)
    ids = np.arange(1, n + 1, dtype=np.int64)
    defaults: dict[str, object] = {
        "person_id": ids,
        "person_household_id": ids,
        "person_tax_unit_id": ids + 100,
        "person_spm_unit_id": ids + 200,
        "person_family_id": ids + 300,
        "person_marital_unit_id": ids + 400,
        "tax_unit_role_input": np.full(n, "HEAD", dtype=object),
        "age": np.linspace(25, 65, n),
        "is_female": ids % 2 == 0,
        "has_esi": ids % 3 == 0,
        "employment_income_before_lsr": np.zeros(n),
        "self_employment_income_before_lsr": np.zeros(n),
        "social_security_retirement": np.zeros(n),
        "social_security_disability": np.zeros(n),
        "social_security_dependents": np.zeros(n),
        "social_security_survivors": np.zeros(n),
    }
    for column, values in defaults.items():
        if column not in person:
            person[column] = values
    person["employment_income_before_lsr"] = pd.to_numeric(
        person.get("WSAL_VAL", person["employment_income_before_lsr"]),
        errors="coerce",
    ).replace({-1: 0, -9999: 0})
    person["self_employment_income_before_lsr"] = pd.to_numeric(
        person.get("SEMP_VAL", person["self_employment_income_before_lsr"]),
        errors="coerce",
    ).replace({-1: 0, -9999: 0})
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": ids, "state_fips": np.resize([6, 36], n)}
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": ids + 100,
                "filing_status_input": np.resize(["SINGLE", "JOINT"], n),
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids + 200}),
        "family": pd.DataFrame({"family_id": ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.linspace(1.0, 2.0, n), WeightKind.DESIGN)},
    )


def _source_frame() -> Frame:
    return _frame(
        pd.DataFrame(
            {
                "source_year": [2022, 2023, 2024, 2023, 2024, 2023, 2024, 2024],
                "PERIDNUM": ["A", "A", "A", "B", "B", "C", "C", "D"],
                "WSAL_VAL": [100, 200, 300, 10, 400, 300, 500, 600],
                "SEMP_VAL": [-20, 30, 40, 20, -5, -9999, 50, 60],
                "I_ERNVAL": [0, 0, 0, 1, 0, 0, 0, 0],
                "I_SEVAL": [0, 0, 0, 0, 0, 0, 0, 0],
            }
        )
    )


class _Fitted:
    def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        n = len(test)
        self_employment = np.zeros(n, dtype=np.float64)
        self_employment[:2] = [125.0, -25.0]
        return pd.DataFrame(
            {
                "employment_income_last_year": np.arange(n) + 1_000.0,
                "self_employment_income_last_year": self_employment,
            },
            index=test.index,
        )


class _QRF:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def fit(
        self,
        training: pd.DataFrame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: np.ndarray,
    ) -> _Fitted:
        self.calls.append(
            {
                "kwargs": self.kwargs,
                "training": training.copy(),
                "predictors": predictors,
                "targets": targets,
                "weights": weights.copy(),
            }
        )
        return _Fitted()


def _signal_frame() -> Frame:
    n = 100
    return _frame(
        pd.DataFrame(
            {
                "self_employment_income_last_year": np.where(
                    np.arange(n) < 4,
                    np.resize([10_000.0, -2_000.0], n),
                    0.0,
                ),
                "previous_year_income_available": np.arange(n) < 20,
            }
        )
    )


def _signal_frame_with_availability_rows(rows: int) -> Frame:
    frame = _signal_frame()
    person = frame.table("person").copy()
    person["previous_year_income_available"] = np.arange(len(person)) < rows
    return module._replace_person_table(frame, person)


def _with_stack_manifest(
    frame: Frame,
    manifest: object,
) -> Frame:
    return Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata={"us_stacked_spine_manifest": manifest},
    )


__all__ = [name for name in globals() if not name.startswith("__")]
