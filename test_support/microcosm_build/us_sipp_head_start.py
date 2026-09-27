"""Strict measured-SIPP Head Start take-up tests."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.sipp_head_start as module
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.sipp_head_start import (
    HEAD_START_SIPP_DICTIONARY_URL,
    SIPP_2023_HEAD_START_DONOR_REVISION,
    SIPP_2023_HEAD_START_DONOR_SHA256,
    SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
    SIPP_2023_HEAD_START_DONOR_URL,
    SIPP_HEAD_START_FIT_PARAMETERS,
    SIPP_HEAD_START_MODEL_PREDICTORS,
    SIPP_HEAD_START_READ_PARAMETERS,
    SIPP_HEAD_START_SOURCE_COLUMNS,
    US_SIPP_HEAD_START_NONCONSTANT_PERSON_COLUMNS,
    US_SIPP_HEAD_START_OUTPUT_COLUMNS,
    US_SIPP_HEAD_START_REQUIRED_SOURCE_COLUMNS,
    US_SIPP_HEAD_START_STAGE_NAME,
    impute_us_sipp_head_start,
    load_sipp_2023_head_start_donor,
    us_sipp_head_start_signal_gate,
    us_sipp_head_start_summary,
    with_us_sipp_head_start_input,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_SIPP_HEAD_START_OUTPUT_COLUMNS[0]


def _load_tail_fixtures():
    path = Path(__file__).with_name("us_tail_clone_fixtures.py")
    spec = importlib.util.spec_from_file_location("us_tail_clone_fixtures", path)
    fixtures = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fixtures)
    return fixtures


_TAIL = _load_tail_fixtures()


def _source_row(
    ssuid: str,
    pnum: int,
    *,
    month: int = 12,
    age: int = 4,
    head_start_status: int = 1,
    head_start_answer: float = 2.0,
    screen_status: int = 1,
    screen: float = 1.0,
    grade_status: int = 1,
    grade: float = 21.0,
    end_month_status: int = 1,
    end_month: float = 12.0,
    weight: float = 100.0,
) -> dict[str, object]:
    row: dict[str, object] = {column: 0.0 for column in SIPP_HEAD_START_SOURCE_COLUMNS}
    row.update(
        {
            "SSUID": ssuid,
            "PNUM": pnum,
            "MONTHCODE": month,
            "WPFINWGT": weight,
            "TAGE": age,
            "ESEX": 2 if pnum % 2 else 1,
            "EED_SCRNR": screen,
            "AED_SCRNR": screen_status,
            "EEDGRADE": grade,
            "AEDGRADE": grade_status,
            "EEDEMONTH": end_month,
            "AEDMONTH": end_month_status,
            "EEDHEADST": head_start_answer,
            "AEDHEADST": head_start_status,
        }
    )
    return row


def _write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "pu2023.csv"
    pd.DataFrame(rows, columns=SIPP_HEAD_START_SOURCE_COLUMNS).to_csv(
        path, sep="|", index=False
    )
    return path


def _donor() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [3.0, 4.0, 5.0, 4.0],
            "is_female": [0.0, 1.0, 0.0, 1.0],
            "household_size": [2.0, 3.0, 4.0, 5.0],
            "count_under_18": [1.0, 2.0, 3.0, 4.0],
            "count_under_6": [1.0, 1.0, 2.0, 2.0],
            "household_employment_income": [0.0, 10_000.0, 20_000.0, 30_000.0],
            _OUTPUT: [False, True, False, True],
            "sipp_weight": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _frame(
    source_ids: list[int],
    *,
    ages: list[int] | None = None,
    female: list[bool] | None = None,
    channels: list[str] | None = None,
    output: list[object] | None = None,
) -> Frame:
    n = len(source_ids)
    ids = np.arange(1, n + 1, dtype=np.int64)
    if ages is None:
        ages = [4] * n
    if female is None:
        female = [i % 2 == 0 for i in range(n)]
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids + 100,
            "person_spm_unit_id": ids + 200,
            "person_family_id": ids + 300,
            "person_marital_unit_id": ids + 400,
            "person_source_id": source_ids,
            "age": ages,
            "is_female": female,
            "employment_income_before_lsr": np.arange(n, dtype=np.float64) * 100.0,
        }
    )
    if channels is not None:
        person["person_support_channel"] = channels
    if output is not None:
        person[_OUTPUT] = output
    tables = {
        "person": person,
        "household": pd.DataFrame({"household_id": ids}),
        "tax_unit": pd.DataFrame({"tax_unit_id": ids + 100}),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids + 200}),
        "family": pd.DataFrame({"family_id": ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
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


class _FakeQRF:
    instances: list[_FakeQRF] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.weights: object = None
        self.training: pd.DataFrame | None = None
        self.receiver: pd.DataFrame | None = None
        self.__class__.instances.append(self)

    def fit(
        self,
        training: pd.DataFrame,
        *,
        predictors: list[str],
        targets: list[str],
        weights: object,
    ) -> _FakeQRF:
        assert predictors == list(SIPP_HEAD_START_MODEL_PREDICTORS)
        assert targets == [_OUTPUT]
        self.training = training.copy()
        self.weights = weights
        return self

    def predict(self, receiver: pd.DataFrame) -> pd.DataFrame:
        self.receiver = receiver.copy()
        return pd.DataFrame(
            {_OUTPUT: receiver["is_female"].to_numpy(dtype=bool)},
            index=receiver.index,
        )


@pytest.fixture(autouse=True)
def _clear_fake() -> None:
    _FakeQRF.instances.clear()




















def _historical_tail_frame() -> Frame:
    """A non-assembled PUF-support frame whose source 10 has a tail copy."""

    native = _frame(
        [10, 20, 30],
        ages=[4, 5, 40],
        female=[True, False, True],
    )
    native = _replace_person(
        native, native.table("person").drop(columns=["person_source_id"])
    )
    cloned = clone_us_frame_for_puf_support(native)
    tailed = _TAIL.with_capital_gains_tail_copies(cloned, [1])
    person = tailed.table("person").copy()
    # Source IDs are the pre-clone person IDs; restate them as the test's
    # source labels so assertions read in source-person terms.
    person["person_source_id"] = person["person_source_id"].map({1: 10, 2: 20, 3: 30})
    return _replace_person(tailed, person)

__all__ = [name for name in globals() if not name.startswith("__")]
