"""ASEC retirement-distribution restoration and PUF-half QRF treatment."""

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

import microcosm.build.us_runtime.retirement_distributions as module
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.asec_pool import load_asec_h5_tables
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.release_input_coverage import (
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.retirement_distributions import (
    RETIREMENT_DISTRIBUTIONS_ARCHIVED_DERIVATION_URL,
    RETIREMENT_DISTRIBUTIONS_ARCHIVED_PARAMETERS_URL,
    US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS,
    US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS,
    derive_us_retirement_distributions_from_manifest,
    us_retirement_distributions_signal_gate,
    us_retirement_distributions_stage_spec,
    us_retirement_distributions_summary,
    with_us_retirement_distribution_inputs,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository

_OUTPUTS = US_RETIREMENT_DISTRIBUTION_OUTPUT_COLUMNS
_PUF_QRF_OUTPUTS = (
    "taxable_401k_distributions",
    "taxable_403b_distributions",
    "keogh_distributions",
    "taxable_sep_distributions",
)
_SLOT_SUFFIXES = ("1", "2", "1_YNG", "2_YNG")


def _person_source() -> pd.DataFrame:
    count = 8
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype="int64"),
            "person_household_id": np.arange(10, 10 + count, dtype="int64"),
            "person_tax_unit_id": np.arange(100, 100 + count, dtype="int64"),
            "person_spm_unit_id": np.arange(1_000, 1_000 + count, dtype="int64"),
            "person_family_id": np.arange(10_000, 10_000 + count, dtype="int64"),
            "person_marital_unit_id": np.arange(
                100_000, 100_000 + count, dtype="int64"
            ),
            "age": [65, 66, 67, 68, 69, 70, 71, 40],
            "is_female": [False, True, False, True, False, True, False, True],
            "has_esi": [False] * count,
            "tax_unit_role_input": ["HEAD"] * count,
            "employment_income_before_lsr": [50_000.0] * count,
            "self_employment_income_before_lsr": [0.0] * count,
            "social_security_retirement": [10_000.0] * count,
            "social_security_disability": [0.0] * count,
            "social_security_dependents": [0.0] * count,
            "social_security_survivors": [0.0] * count,
        }
    )
    for suffix in _SLOT_SUFFIXES:
        person[f"DST_SC{suffix}"] = 0
        person[f"DST_VAL{suffix}"] = 0.0
    person["DST_SC1"] = np.arange(1, count + 1) % 8
    person["DST_VAL1"] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 0.0]
    # A second 401(k) slot proves that repeated account codes sum by person.
    person.loc[0, ["DST_SC2", "DST_VAL2"]] = [1, 50.0]
    return person


def _frame() -> Frame:
    person = _person_source()
    identifiers = {
        "household": person["person_household_id"].to_numpy(),
        "tax_unit": person["person_tax_unit_id"].to_numpy(),
        "spm_unit": person["person_spm_unit_id"].to_numpy(),
        "family": person["person_family_id"].to_numpy(),
        "marital_unit": person["person_marital_unit_id"].to_numpy(),
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": values})
        for entity, values in identifiers.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = ["SINGLE"] * len(person)
    # Total = 10,000. Rare outputs receive smaller weights so the fixture
    # reproduces plausible population shares while retaining every account.
    weights = np.asarray([1_000, 100, 100, 1_000, 1, 100, 100, 7_599], dtype=float)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.DESIGN)},
    )


def _operation(kind: str = "derive_retirement_distributions"):
    return next(
        operation
        for operation in us_retirement_distributions_stage_spec().operations
        if operation.kind == kind
    )


def _derive(frame: pd.DataFrame) -> pd.DataFrame:
    return derive_us_retirement_distributions_from_manifest(
        frame,
        _operation(),
        None,
    )


def _stacked_frame() -> Frame:
    direct = with_us_retirement_distribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )
    stacked = clone_us_frame_for_puf_support(direct)
    person = stacked.table("person")
    source_record = np.tile(np.arange(8, dtype=np.int64), 2)
    person["person_spine_source_id"] = source_record
    person["person_support_channel"] = np.where(
        source_record < 4,
        "asec",
        "acs",
    )
    acs = person["person_support_channel"].eq("acs")
    person.loc[
        acs,
        list(US_RETIREMENT_DISTRIBUTION_REQUIRED_SOURCE_COLUMNS),
    ] = np.nan
    return stacked


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [name for name in globals() if not name.startswith("__")]
