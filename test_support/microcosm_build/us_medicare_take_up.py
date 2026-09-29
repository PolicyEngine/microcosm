"""Measured ASEC Medicare take-up restoration."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import (
    MEDICARE_TAKE_UP_ARCHIVED_CLONE_URL,
    MEDICARE_TAKE_UP_ARCHIVED_DERIVATION_URL,
    MEDICARE_TAKE_UP_ARCHIVED_EXPORT_URL,
    MEDICARE_TAKE_UP_ARCHIVED_SOURCE_COLUMNS_URL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_DONORS,
    US_MEDICARE_TAKE_UP_NONCONSTANT_PERSON_COLUMNS,
    US_MEDICARE_TAKE_UP_OUTPUT_COLUMNS,
    US_MEDICARE_TAKE_UP_REQUIRED_SOURCE_COLUMNS,
    US_MEDICARE_TAKE_UP_STAGE_NAME,
    US_PUF_SUPPORT_STAGE_NAME,
    US_STAGE_NAMES,
    clone_us_frame_for_puf_support,
    derive_us_medicare_take_up_from_manifest,
    load_release_input_coverage_manifest,
    us_medicare_take_up_signal_gate,
    us_medicare_take_up_stage_spec,
    us_medicare_take_up_summary,
    us_release_reform_coverage_probes,
    with_us_medicare_take_up_input,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository

_OUTPUT = "takes_up_medicare_if_eligible"
_SOURCE = "MCARE"


def _frame(
    source_codes: list[object] | None = None,
    *,
    output: list[object] | None = None,
    weights: list[float] | None = None,
) -> Frame:
    codes = source_codes or [1, 2, 2, 2, 0]
    count = len(codes)
    household_ids = np.arange(1, count + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": household_ids,
            "person_household_id": household_ids,
            "person_tax_unit_id": household_ids + 100,
            "person_spm_unit_id": household_ids + 200,
            "person_family_id": household_ids + 300,
            "person_marital_unit_id": household_ids + 400,
            _SOURCE: codes,
            "age": [70, 66, 40, 20, 10][:count],
        }
    )
    if output is not None:
        person[_OUTPUT] = output
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": household_ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 100}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 200}),
        "family": pd.DataFrame({"family_id": household_ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": household_ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(weights or [1.0] * count, dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _operation() -> SourceOperationSpec:
    return next(
        operation
        for operation in us_medicare_take_up_stage_spec().operations
        if operation.kind == "derive_medicare_take_up"
    )


def _stacked_frame() -> Frame:
    derived = with_us_medicare_take_up_input(_frame(), seed=0, time_period=2024)
    stacked = clone_us_frame_for_puf_support(derived)
    person = stacked.table("person")
    source_record = np.tile(np.arange(5, dtype=np.int64), 2)
    person["person_spine_source_id"] = source_record
    person["person_support_channel"] = np.where(
        source_record < 3,
        "asec",
        "acs",
    )
    person.loc[person["person_support_channel"].eq("acs"), _SOURCE] = np.nan
    return stacked


def _derive(person: pd.DataFrame) -> pd.DataFrame:
    return derive_us_medicare_take_up_from_manifest(person, _operation(), None)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [name for name in globals() if not name.startswith("__")]
