"""ASEC relationship-input restoration and partner-source exclusion."""

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

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import (
    US_DONORS,
    US_PUF_SUPPORT_STAGE_NAME,
    US_RELATIONSHIP_INPUTS_NONCONSTANT_PERSON_COLUMNS,
    US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS,
    US_RELATIONSHIP_INPUTS_REQUIRED_SOURCE_COLUMNS,
    US_RELATIONSHIP_INPUTS_STAGE_NAME,
    US_STAGE_NAMES,
    derive_us_relationship_inputs_from_manifest,
    load_release_input_coverage_manifest,
    us_relationship_inputs_signal_gate,
    us_relationship_inputs_stage_spec,
    us_relationship_inputs_summary,
    us_release_reform_coverage_probes,
    with_us_relationship_inputs,
)
from microcosm.build.us_runtime.asec_pool import _with_relationship_recode
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")

ROOT = _TEST_PATHS.repository

_HEAD = "is_household_head"
_SEPARATED = "is_separated"
_SURVIVING = "is_surviving_spouse"
_UNMARRIED_PARTNER = "is_unmarried_partner_of_household_head"
_OUTPUTS = (_HEAD, _SEPARATED, _SURVIVING)


def _person(rows: list[dict[str, object]]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        record: dict[str, object] = {
            "person_id": index + 1,
            "PH_SEQ": 100 + index,
            "P_SEQ": 1,
            "A_MARITL": 7,
        }
        record.update(row)
        records.append(record)
    return pd.DataFrame(records)


def _frame(rows: list[dict[str, object]], weights: list[float] | None = None) -> Frame:
    person = _person(rows)
    household_ids, person_household_ids = np.unique(
        person["PH_SEQ"].to_numpy(dtype=np.int64), return_inverse=True
    )
    person_household_ids = household_ids[person_household_ids]
    person["person_household_id"] = person_household_ids
    person["person_tax_unit_id"] = person_household_ids + 1_000
    person["person_spm_unit_id"] = person_household_ids + 2_000
    person["person_family_id"] = person_household_ids + 3_000
    person["person_marital_unit_id"] = np.arange(len(person)) + 4_000
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": household_ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": household_ids + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 2_000}),
        "family": pd.DataFrame({"family_id": household_ids + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(len(person)) + 4_000}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray(weights or [1.0] * len(household_ids), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _operation():
    return next(
        operation
        for operation in us_relationship_inputs_stage_spec().operations
        if operation.kind == "derive_relationship_inputs"
    )


def _known_gap(name: str) -> dict[str, object]:
    payload = json.loads(
        files("microcosm.build.us")
        .joinpath("ecps_parity_known_gaps.json")
        .read_text(encoding="utf-8")
    )
    return payload["known_gaps"][name]


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [name for name in globals() if not name.startswith("__")]
