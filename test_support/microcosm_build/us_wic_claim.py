"""Source-backed WIC claim-input restoration tests."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from copy import deepcopy
from importlib.metadata import version
from importlib.resources import files

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceOperationSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime import (
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_DONORS,
    US_PREGNANCY_STAGE_NAME,
    US_PUF_SUPPORT_STAGE_NAME,
    US_STAGE_NAMES,
    US_WIC_CLAIM_NONCONSTANT_PERSON_COLUMNS,
    US_WIC_CLAIM_OUTPUT_COLUMNS,
    US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS,
    US_WIC_CLAIM_STAGE_NAME,
    WIC_CLAIM_ARCHIVED_DERIVATION_URL,
    WIC_CLAIM_ARCHIVED_PARAMETERS_URL,
    WIC_CLAIM_ARCHIVED_RANDOMNESS_URL,
    WIC_CLAIM_FNS_SOURCE_URL,
    clone_us_frame_for_puf_support,
    derive_us_wic_claim_from_manifest,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
    us_wic_claim_signal_gate,
    us_wic_claim_stage_spec,
    us_wic_claim_summary,
    with_us_wic_claim_input,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
)
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.wic_claim import _stable_person_keys
from microcosm.frame import US_SCHEMA, EntitySchema, Frame, WeightKind, Weights

_OUTPUT = "takes_up_wic_if_eligible"
_RATES = {
    "pregnant": 0.456,
    "postpartum": 0.689,
    "breastfeeding": 0.663,
    "infant": 0.784,
    "child": 0.460,
    "none": 0.0,
}


def _frame(rows: list[dict[str, object]]) -> Frame:
    records: list[dict[str, object]] = []
    for index, overrides in enumerate(rows, start=1):
        record: dict[str, object] = {
            "person_id": index,
            "person_household_id": index,
            "person_tax_unit_id": index + 10_000,
            "person_spm_unit_id": index + 20_000,
            "person_family_id": index + 30_000,
            "person_marital_unit_id": index + 40_000,
            "age": 30.0,
            "is_female": False,
            "is_pregnant": False,
            "own_children_in_household": 0,
            "source_year": 2024,
            "source_household_id": index,
            "source_person_id": index,
        }
        record.update(overrides)
        records.append(record)
    person = pd.DataFrame.from_records(records)
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": np.sort(person["person_household_id"].unique())}
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": np.sort(person["person_tax_unit_id"].unique())}
        ),
        "spm_unit": pd.DataFrame(
            {"spm_unit_id": np.sort(person["person_spm_unit_id"].unique())}
        ),
        "family": pd.DataFrame(
            {"family_id": np.sort(person["person_family_id"].unique())}
        ),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.sort(person["person_marital_unit_id"].unique())}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(tables["household"]), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


def _replace_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


def _context(seed: int = 0) -> SourceRuntimeContext:
    return SourceRuntimeContext(
        config=SourceRuntimeConfig(seed=seed, target_year=2024),
        tables={},
    )


def _operation() -> SourceOperationSpec:
    return next(
        operation
        for operation in us_wic_claim_stage_spec().operations
        if operation.kind == "derive_wic_claim"
    )


def _derive(person: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    return derive_us_wic_claim_from_manifest(
        person,
        _operation(),
        _context(seed),
    )


def _plausible_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    next_id = 1

    for _ in range(120):
        rows.append(
            {
                "person_id": next_id,
                "is_female": True,
                "is_pregnant": True,
                "age": 27,
            }
        )
        next_id += 1

    for family_number in range(120):
        family_id = 100_000 + family_number
        rows.append(
            {
                "person_id": next_id,
                "person_family_id": family_id,
                "is_female": True,
                "own_children_in_household": 1,
                "age": 27,
            }
        )
        next_id += 1
        rows.append(
            {
                "person_id": next_id,
                "person_family_id": family_id,
                "age": 0,
            }
        )
        next_id += 1

    for _ in range(240):
        rows.append({"person_id": next_id, "age": 3})
        next_id += 1

    for _ in range(7_500):
        rows.append({"person_id": next_id, "age": 35})
        next_id += 1
    return rows


__all__ = [name for name in globals() if not name.startswith("__")]
