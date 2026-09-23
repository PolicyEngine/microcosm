"""Measured SIPP voluntary-filing source-stage tests."""

# ruff: noqa: F401

from __future__ import annotations

import hashlib
import importlib.util
import io
import urllib.request
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.voluntary_filing as module
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.release_input_coverage import (
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.voluntary_filing import (
    SIPP_2023_VOLUNTARY_FILING_DONOR_REVISION,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    SIPP_2023_VOLUNTARY_FILING_DONOR_URL,
    SIPP_VOLUNTARY_FILING_MODEL_PREDICTORS,
    SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS,
    US_VOLUNTARY_FILING_NONCONSTANT_TAX_UNIT_COLUMNS,
    US_VOLUNTARY_FILING_OUTPUT_COLUMNS,
    US_VOLUNTARY_FILING_STAGE_NAME,
    VOLUNTARY_FILING_ARCHIVED_DERIVATION_URL,
    VOLUNTARY_FILING_ARCHIVED_PARAMETERS_URL,
    VOLUNTARY_FILING_SIPP_DICTIONARY_URL,
    fetch_sipp_2023_voluntary_filing_donor,
    impute_us_voluntary_filing,
    load_sipp_2023_voluntary_filing_donor,
    us_voluntary_filing_signal_gate,
    us_voluntary_filing_stage_spec,
    us_voluntary_filing_summary,
    with_us_voluntary_filing_input,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_VOLUNTARY_FILING_OUTPUT_COLUMNS[0]


def _source_row(
    ssuid: int,
    pnum: int,
    *,
    month: int = 12,
    weight: float = 10.0,
    age: float = 40.0,
    sex: int = 1,
    spouse: float = np.nan,
    filing: float = 1.0,
    filing_status: int = 1,
    will_file: float = np.nan,
    will_file_status: int = 0,
    dependent: float = np.nan,
    monthly_wages: float = 1_000.0,
) -> dict[str, object]:
    row: dict[str, object] = {
        "SSUID": ssuid,
        "PNUM": pnum,
        "MONTHCODE": month,
        "WPFINWGT": weight,
        "TAGE": age,
        "ESEX": sex,
        "EPNSPOUSE": spouse,
        "AFILING": filing_status,
        "EFILING": filing,
        "AWILLFILE": will_file_status,
        "EWILLFILE": will_file,
        "EDEPCLM": dependent,
    }
    for job in range(1, 8):
        row[f"TJB{job}_MSUM"] = monthly_wages if job == 1 else 0.0
    return row


def _write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "pu2023.csv"
    pd.DataFrame(rows, columns=SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS).to_csv(
        path, sep="|", index=False
    )
    return path


def _synthetic_source_rows() -> list[dict[str, object]]:
    return [
        # A non-December duplicate must not enter either target or predictors.
        _source_row(1, 101, month=11, monthly_wages=999_999.0),
        # Reciprocal spouses report the same filing target and collapse once.
        _source_row(1, 101, spouse=102, age=40, monthly_wages=1_000.0),
        _source_row(
            1,
            102,
            spouse=101,
            age=38,
            sex=2,
            monthly_wages=500.0,
        ),
        # Household context is counted before the filing-response filter.
        _source_row(
            1,
            103,
            age=10,
            filing=np.nan,
            filing_status=0,
            monthly_wages=0.0,
        ),
        # A directly reported no/not-planning response supplies the false class.
        _source_row(
            2,
            101,
            age=70,
            sex=2,
            filing=2,
            will_file=2,
            will_file_status=1,
            monthly_wages=200.0,
        ),
        # Claimed dependents are not standalone receiver tax units.
        _source_row(3, 101, age=21, dependent=1, monthly_wages=300.0),
        # Imputed filing and will-file answers are not measured targets.
        _source_row(4, 101, filing_status=2),
        _source_row(
            5,
            101,
            filing=2,
            will_file=1,
            will_file_status=2,
        ),
        # Invalid survey weights are excluded after source-unit construction.
        _source_row(6, 101, weight=0.0),
    ]


def _donor(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(918)
    income = rng.gamma(2.0, 18_000.0, n)
    target = (np.arange(n) % 4) != 0
    return pd.DataFrame(
        {
            "source_tax_unit_key": [f"unit:{i}" for i in range(n)],
            "employment_income": income,
            "reference_age": rng.integers(18, 90, n).astype(np.float64),
            "reference_is_female": rng.integers(0, 2, n).astype(np.float64),
            "reference_is_married": rng.integers(0, 2, n).astype(np.float64),
            "count_under_18": rng.integers(0, 5, n).astype(np.float64),
            _OUTPUT: target,
            "tax_unit_weight": rng.uniform(100.0, 2_000.0, n),
        }
    )


def _frame(n_households: int = 12) -> Frame:
    people: list[dict[str, object]] = []
    person_id = 1
    for household_id in range(1, n_households + 1):
        tax_unit_id = household_id + 100
        people.append(
            {
                "person_id": person_id,
                "person_household_id": household_id,
                "person_tax_unit_id": tax_unit_id,
                "person_spm_unit_id": household_id + 200,
                "person_family_id": household_id + 300,
                "person_marital_unit_id": household_id + 400,
                "age": float(25 + household_id % 50),
                "is_female": bool(household_id % 2),
                "tax_unit_role_input": "HEAD",
                "employment_income_before_lsr": float(household_id * 3_000),
                "A_LINENO": 1,
            }
        )
        person_id += 1
        if household_id % 3 == 0:
            people.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": tax_unit_id,
                    "person_spm_unit_id": household_id + 200,
                    "person_family_id": household_id + 300,
                    "person_marital_unit_id": household_id + 400,
                    "age": float(22 + household_id % 40),
                    "is_female": not bool(household_id % 2),
                    "tax_unit_role_input": "SPOUSE",
                    "employment_income_before_lsr": 2_000.0,
                    "A_LINENO": 2,
                }
            )
            person_id += 1
        if household_id % 2 == 0:
            people.append(
                {
                    "person_id": person_id,
                    "person_household_id": household_id,
                    "person_tax_unit_id": tax_unit_id,
                    "person_spm_unit_id": household_id + 200,
                    "person_family_id": household_id + 300,
                    "person_marital_unit_id": household_id + 400,
                    "age": 8.0,
                    "is_female": False,
                    "tax_unit_role_input": "DEPENDENT",
                    "employment_income_before_lsr": 0.0,
                    "A_LINENO": 3,
                }
            )
            person_id += 1
    person = pd.DataFrame(people)
    household_ids = np.arange(1, n_households + 1, dtype=np.int64)
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": household_ids}),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": household_ids + 100,
                "filing_status_input": np.where(
                    household_ids % 3 == 0, "JOINT", "SINGLE"
                ),
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 200}),
        "family": pd.DataFrame({"family_id": household_ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": household_ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.linspace(1.0, 2.0, n_households), WeightKind.DESIGN)},
    )


def _replace_tax_unit(frame: Frame, **columns: np.ndarray) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column, values in columns.items():
        tables["tax_unit"][column] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


class _ChunkedResponse(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.read_sizes: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [name for name in globals() if not name.startswith("__")]
