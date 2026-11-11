"""Exact ASEC repair and PUF-half treatment for weeks unemployed."""

# ruff: noqa: F401

from __future__ import annotations

import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.weeks_unemployed as module
from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.source_runtime import (
    SourceRuntimeConfig,
    SourceRuntimeContext,
    SourceRuntimeError,
)
from microcosm.build.us_runtime.weeks_unemployed import (
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_CRC32,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SIZE_BYTES,
    ASEC_2023_WEEKS_UNEMPLOYED_POSITIVE_ROWS,
    ASEC_2023_WEEKS_UNEMPLOYED_RAW_ROWS,
    ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_COLUMNS,
    ASEC_2023_WEEKS_UNEMPLOYED_UNIQUE_KEYS,
    ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_SOURCE_SHARE,
    ASEC_2023_WEEKS_UNEMPLOYED_WEIGHTED_WEEKS,
    ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SHA256,
    ASEC_2023_WEEKS_UNEMPLOYED_ZIP_SIZE_BYTES,
    WEEKS_UNEMPLOYED_DERIVE_PARAMETERS,
    WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS,
    WEEKS_UNEMPLOYED_READ_PARAMETERS,
    derive_us_weeks_unemployed_from_manifest,
    fetch_asec_2023_weeks_unemployed_source,
    fill_asec_2022_weeks_unemployed_source,
    impute_us_weeks_unemployed_to_puf_support_from_manifest,
    load_asec_2023_weeks_unemployed_source,
    us_weeks_unemployed_signal_gate,
    us_weeks_unemployed_stage_spec,
    with_us_weeks_unemployed,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = "weeks_unemployed"
_PREFIX = "weeks_unemployed_predictor_"
_REQUIRED_PREDICTORS = (
    "age",
    "is_male",
    "tax_unit_is_joint",
    "is_tax_unit_head",
    "is_tax_unit_spouse",
    "is_tax_unit_dependent",
)


def _person_csv() -> bytes:
    source = pd.DataFrame(
        {
            "PH_SEQ": [101, 102, 103],
            "P_SEQ": [1, 1, 1],
            "A_LINENO": [1, 1, 1],
            "PERIDNUM": [f"{value:022d}" for value in (1, 2, 3)],
            "LKWEEKS": [0, 4, -1],
            "A_FNLWGT": [100, 200, 300],
        }
    )
    return source.to_csv(index=False).encode()


def _zip_bytes(member: bytes | None = None) -> bytes:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(ASEC_2023_WEEKS_UNEMPLOYED_MEMBER, member or _person_csv())
    return payload.getvalue()


def _pins(payload: bytes, member: bytes) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        info = archive.getinfo(ASEC_2023_WEEKS_UNEMPLOYED_MEMBER)
    return {
        "expected_zip_size_bytes": len(payload),
        "expected_zip_sha256": hashlib.sha256(payload).hexdigest(),
        "expected_member_size_bytes": len(member),
        "expected_member_crc32": f"{info.CRC:08x}",
        "expected_member_sha256": hashlib.sha256(member).hexdigest(),
    }


def _mini_source(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    member = _person_csv()
    payload = _zip_bytes(member)
    path = tmp_path / "asecpub23csv.zip"
    path.write_bytes(payload)
    return path, _pins(payload, member)


def _stage_spec() -> SourceStageSpec:
    return SourceStageSpec(
        stage="weeks_unemployed_input",
        survey="Census CPS ASEC",
        source="official fixture",
        grain="person",
        artifacts=(),
        operations=(
            SourceOperationSpec("read_table", WEEKS_UNEMPLOYED_READ_PARAMETERS),
            SourceOperationSpec(
                "derive_weeks_unemployed", WEEKS_UNEMPLOYED_DERIVE_PARAMETERS
            ),
            SourceOperationSpec(
                "impute_weeks_unemployed_to_puf_support",
                WEEKS_UNEMPLOYED_PUF_IMPUTATION_PARAMETERS,
            ),
        ),
        outputs=(_OUTPUT,),
        nonnegative_outputs=(_OUTPUT,),
    )


def _operation(kind: str) -> SourceOperationSpec:
    spec = _stage_spec()
    return next(operation for operation in spec.operations if operation.kind == kind)


def _frame(*, channels: bool = True) -> Frame:
    person_count = 4 if channels else 2
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, person_count + 1, dtype=np.int64),
            "person_household_id": np.arange(11, 11 + person_count, dtype=np.int64),
            "person_tax_unit_id": np.arange(21, 21 + person_count, dtype=np.int64),
            "person_spm_unit_id": np.arange(31, 31 + person_count, dtype=np.int64),
            "person_family_id": np.arange(41, 41 + person_count, dtype=np.int64),
            "person_marital_unit_id": np.arange(51, 51 + person_count, dtype=np.int64),
            "source_year": np.resize([2022, 2023], person_count),
            "source_household_id": np.arange(101, 101 + person_count, dtype=np.int64),
            "P_SEQ": np.ones(person_count, dtype=np.int64),
            "A_LINENO": np.ones(person_count, dtype=np.int64),
            "PERIDNUM": [f"{value:022d}" for value in range(1, person_count + 1)],
            "LKWEEKS": np.resize([2, 4], person_count),
            "age": np.arange(30, 30 + person_count, dtype=np.int64),
            "is_female": np.resize([False, True], person_count),
            "tax_unit_role_input": np.resize(["HEAD", "SPOUSE"], person_count),
            "unemployment_compensation": np.resize([100.0, 0.0], person_count),
        }
    )
    if channels:
        person["person_support_channel"] = [
            "asec",
            "asec",
            "puf_tax_detail",
            "puf_tax_detail",
        ]
    entity_ids = {
        "household": person["person_household_id"].to_numpy(),
        "tax_unit": person["person_tax_unit_id"].to_numpy(),
        "spm_unit": person["person_spm_unit_id"].to_numpy(),
        "family": person["person_family_id"].to_numpy(),
        "marital_unit": person["person_marital_unit_id"].to_numpy(),
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": ids})
        for entity, ids in entity_ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = np.resize(
        ["SINGLE", "JOINT"], person_count
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(person_count, dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


def _gate_frame() -> Frame:
    asec_rows = 1_000
    puf_rows = 2_000
    person_count = asec_rows + puf_rows
    asec_weeks = np.zeros(asec_rows, dtype=np.float64)
    asec_weeks[:30] = 17.0
    puf_weeks = np.zeros(puf_rows, dtype=np.float64)
    puf_weeks[:12] = 16.0
    unemployment_compensation = np.zeros(person_count, dtype=np.float64)
    unemployment_compensation[asec_rows : asec_rows + 12] = 100.0
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, person_count + 1, dtype=np.int64),
            "person_household_id": np.arange(
                10_001, 10_001 + person_count, dtype=np.int64
            ),
            "person_tax_unit_id": np.arange(
                20_001, 20_001 + person_count, dtype=np.int64
            ),
            "person_spm_unit_id": np.arange(
                30_001, 30_001 + person_count, dtype=np.int64
            ),
            "person_family_id": np.arange(
                40_001, 40_001 + person_count, dtype=np.int64
            ),
            "person_marital_unit_id": np.arange(
                50_001, 50_001 + person_count, dtype=np.int64
            ),
            "person_support_channel": ["asec"] * asec_rows
            + ["puf_tax_detail"] * puf_rows,
            "LKWEEKS": np.concatenate(
                [asec_weeks, np.zeros(puf_rows, dtype=np.float64)]
            ),
            _OUTPUT: np.concatenate([asec_weeks, puf_weeks]),
            "unemployment_compensation": unemployment_compensation,
        }
    )
    entity_links = {
        "household": "person_household_id",
        "tax_unit": "person_tax_unit_id",
        "spm_unit": "person_spm_unit_id",
        "family": "person_family_id",
        "marital_unit": "person_marital_unit_id",
    }
    tables = {
        entity: pd.DataFrame(
            {f"{entity}_id": person[column].to_numpy(dtype=np.int64)}
        )
        for entity, column in entity_links.items()
    }
    tables["person"] = person
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(person_count, dtype=np.float64), WeightKind.DESIGN
            )
        },
    )


def _stacked_gate_frame() -> Frame:
    frame = _gate_frame()
    person = frame.table("person").copy()
    asec_rows = 1_000
    asec_native_rows = 500
    acs_native_rows = 1_000
    person["person_support_channel"] = ["asec"] * asec_rows + ["acs"] * 2_000
    person["person_support_clone_index"] = np.concatenate(
        [
            np.zeros(asec_native_rows, dtype=np.int64),
            np.ones(asec_rows - asec_native_rows, dtype=np.int64),
            np.zeros(acs_native_rows, dtype=np.int64),
            np.ones(2_000 - acs_native_rows, dtype=np.int64),
        ]
    )
    person["person_spine_source_id"] = np.concatenate(
        [
            np.arange(asec_native_rows, dtype=np.int64),
            np.arange(asec_native_rows, dtype=np.int64),
            np.arange(10_000, 10_000 + acs_native_rows, dtype=np.int64),
            np.arange(10_000, 10_000 + acs_native_rows, dtype=np.int64),
        ]
    )
    # Spine assembly writes the assembly-unique source ID beside the raw one
    # (the ACS offset above keeps it unique); a support clone keeps its
    # source's ID.
    person["person_source_id"] = person["person_spine_source_id"]
    weeks = np.zeros(len(person), dtype=np.float64)
    weeks[:18] = 17.0
    weeks[asec_native_rows : asec_native_rows + 12] = 17.0
    weeks[asec_rows + acs_native_rows : asec_rows + acs_native_rows + 12] = 16.0
    person[_OUTPUT] = weeks
    source = np.full(len(person), np.nan, dtype=np.float64)
    source[:asec_rows] = 0.0
    source[:18] = 17.0
    person["LKWEEKS"] = source
    unemployment_compensation = np.zeros(len(person), dtype=np.float64)
    unemployment_compensation[
        asec_native_rows : asec_native_rows + 12
    ] = 100.0
    person["unemployment_compensation"] = unemployment_compensation
    return module._replace_person_table(frame, person)




















class _CapturingQRF:
    instances: list[_CapturingQRF] = []
    prediction = pd.DataFrame({_OUTPUT: [1.6, 80.4]})

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.training: pd.DataFrame | None = None
        self.predictors: list[str] = []
        self.weights: np.ndarray | None = None
        _CapturingQRF.instances.append(self)

    def fit(
        self,
        training: pd.DataFrame,
        predictors: list[str],
        targets: list[str],
        *,
        weights: np.ndarray,
    ) -> _CapturingQRF:
        assert targets == [_OUTPUT]
        self.training = training.copy()
        self.predictors = predictors
        self.weights = weights.copy()
        return self

    def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        assert list(test.columns) == self.predictors
        return self.prediction.copy()


def _imputation_table(*, include_uc: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "person_support_channel": [
                "asec",
                "asec",
                "puf_tax_detail",
                "puf_tax_detail",
            ],
            "person_weight": [2.0, 3.0, 4.0, 5.0],
            _OUTPUT: [0.0, 6.0, 0.0, 6.0],
        }
    )
    for index, predictor in enumerate(_REQUIRED_PREDICTORS):
        frame[_PREFIX + predictor] = np.arange(4) + index
    if include_uc:
        frame[_PREFIX + "unemployment_compensation"] = [0.0, 100.0, 50.0, 0.0]
    return frame

__all__ = [name for name in globals() if not name.startswith("__")]
