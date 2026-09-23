"""Source-faithful SIPP SSI disability-criteria stage tests."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.ssi_disability_criteria as module
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.ssi_disability_criteria import (
    SIPP_2023_SSI_DISABILITY_DONOR_REVISION,
    SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
    SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    SIPP_2023_SSI_DISABILITY_DONOR_URL,
    SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS,
    SIPP_SSI_DISABILITY_FIT_PARAMETERS,
    SIPP_SSI_DISABILITY_MODEL_PREDICTORS,
    SIPP_SSI_DISABILITY_READ_PARAMETERS,
    SIPP_SSI_DISABILITY_SOURCE_COLUMNS,
    SSI_DISABILITY_ARCHIVED_CPS_URL,
    SSI_DISABILITY_ARCHIVED_EXTENDED_CPS_URL,
    SSI_DISABILITY_ARCHIVED_SIPP_URL,
    SSI_DISABILITY_ARCHIVED_SOURCE_IMPUTE_URL,
    US_SSI_DISABILITY_CRITERIA_NONCONSTANT_PERSON_COLUMNS,
    US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS,
    US_SSI_DISABILITY_CRITERIA_STAGE_NAME,
    impute_us_ssi_disability_criteria,
    load_sipp_2023_ssi_disability_donor,
    us_ssi_disability_criteria_signal_gate,
    us_ssi_disability_criteria_stage_spec,
    us_ssi_disability_criteria_summary,
    with_us_ssi_disability_criteria,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OUTPUT = US_SSI_DISABILITY_CRITERIA_OUTPUT_COLUMNS[0]


def _source_row(
    ssuid: str,
    pnum: int,
    *,
    month: int = 12,
    age: float = 40.0,
    received_ssi: float = 2.0,
    reason: float = np.nan,
    receipt_allocation: float = 1.0,
    reason_allocation: float = 1.0,
    weight: float = 100.0,
    assets: float = 0.0,
    monthly_earnings: float = 0.0,
    difficulty_seeing: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        column: 0.0 for column in SIPP_SSI_DISABILITY_SOURCE_COLUMNS
    }
    row.update(
        {
            "SSUID": ssuid,
            "PNUM": pnum,
            "MONTHCODE": month,
            "SPANEL": 2023,
            "SWAVE": 1,
            "WPFINWGT": weight,
            "TAGE": age,
            "ESEX": 2,
            "EMS": 2,
            "TVAL_BANK": assets,
            "TVAL_STMF": 0.0,
            "TVAL_BOND": 0.0,
            "TINC_BANK": 0.0,
            "TINC_STMF": 0.0,
            "TINC_BOND": 0.0,
            "TINC_RENT": 0.0,
            "TPTOTINC": monthly_earnings,
            "TJB1_MSUM": monthly_earnings,
            "TSSSAMT": 0.0,
            "RSSI_YRYN": received_ssi,
            "ESSI_BRSN": reason,
            "ASSI_YRYN": receipt_allocation,
            "ASSI_BRSN": reason_allocation,
            "ESSRSN2YN": 2.0,
            "EDISANY": 2.0,
            "ESELFCARE": 2.0,
            "EHEARING": 2.0,
            "ESEEING": 1.0 if difficulty_seeing else 2.0,
            "EERRANDS": 2.0,
            "EAMBULAT": 2.0,
            "ECOGNIT": 2.0,
        }
    )
    return row


def _write_source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "pu2023.csv"
    pd.DataFrame(rows, columns=SIPP_SSI_DISABILITY_SOURCE_COLUMNS).to_csv(
        path,
        sep="|",
        index=False,
    )
    return path


def _donor(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(747)
    donor = pd.DataFrame(index=np.arange(n))
    donor["age"] = np.arange(n, dtype=np.float64)
    donor["is_female"] = rng.integers(0, 2, n)
    donor["is_married"] = rng.integers(0, 2, n)
    donor["employment_income"] = rng.gamma(2.0, 5_000.0, n)
    donor["interest_income"] = rng.gamma(1.0, 100.0, n)
    donor["dividend_income"] = rng.gamma(1.0, 100.0, n)
    donor["rental_income"] = rng.normal(0.0, 500.0, n)
    donor["bank_account_assets"] = rng.gamma(1.0, 1_000.0, n)
    donor["stock_assets"] = rng.gamma(1.0, 1_000.0, n)
    donor["bond_assets"] = rng.gamma(1.0, 200.0, n)
    donor["count_under_18"] = rng.integers(0, 5, n)
    for predictor in SIPP_SSI_DISABILITY_DIFFICULTY_PREDICTORS:
        donor[predictor] = rng.integers(0, 2, n)
    donor["social_security_disability"] = rng.integers(0, 2, n) * 6_000.0
    donor["has_disability_income"] = rng.integers(0, 2, n)
    donor[_OUTPUT] = np.arange(n) % 5 == 0
    donor["household_weight"] = np.arange(1, n + 1, dtype=np.float64)
    return donor.loc[
        :,
        [*SIPP_SSI_DISABILITY_MODEL_PREDICTORS, _OUTPUT, "household_weight"],
    ]


def _frame(n: int = 20) -> Frame:
    ids = np.arange(1, n + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids,
            "person_tax_unit_id": ids + 100,
            "person_spm_unit_id": ids + 200,
            "person_family_id": ids + 300,
            "person_marital_unit_id": ids + 400,
            "age": np.full(n, 40.0),
            "is_female": ids % 2 == 0,
            "A_MARITL": np.full(n, 5),
            "employment_income_before_lsr": np.zeros(n),
            # Deliberately omit the aggregate leaves: the receiver must use
            # the complete measured PolicyEngine component pairs.
            "taxable_interest_income": np.arange(n, dtype=np.float64),
            "tax_exempt_interest_income": np.full(n, 2.0),
            "qualified_dividend_income": np.arange(n, dtype=np.float64) * 3.0,
            "non_qualified_dividend_income": np.full(n, 4.0),
            "rental_income": np.zeros(n),
            "bank_account_assets": np.where(np.isin(ids, [2, 3]), 100.0, 0.0),
            "stock_assets": np.zeros(n),
            "bond_assets": np.zeros(n),
            "PEDISDRS": np.where(ids == 2, 1, 2),
            "PEDISEAR": np.full(n, 2),
            "PEDISEYE": np.full(n, 2),
            "PEDISOUT": np.full(n, 2),
            "PEDISPHY": np.full(n, 2),
            "PEDISREM": np.full(n, 2),
            # The archived signal coercion is strictly >0, not generic truthiness.
            "social_security_disability": np.where(ids == 3, -10.0, 0.0),
            "disability_benefits": np.zeros(n),
            "SSI_VAL": np.where(ids == 1, 1_200.0, 0.0),
        }
    )
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


def _replace_person(frame: Frame, **columns: Any) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for column, values in columns.items():
        tables["person"][column] = values
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )


class _FakeQRF:
    instances: list[_FakeQRF] = []
    predict_receivers: list[pd.DataFrame] = []
    predict_start_offsets: list[int] = []

    def __init__(self, *, n_estimators: int, seed: int) -> None:
        self.n_estimators = n_estimators
        self.seed = seed
        self.training: pd.DataFrame | None = None
        self.weights: object = None
        self.receiver: pd.DataFrame | None = None
        self.draw_offset = 0
        self.__class__.instances.append(self)

    def fit(
        self,
        training: pd.DataFrame,
        *,
        predictors: list[str],
        targets: list[str],
        weights: object,
    ) -> _FakeQRF:
        assert predictors == list(SIPP_SSI_DISABILITY_MODEL_PREDICTORS)
        assert targets == [_OUTPUT]
        self.training = training.copy()
        self.weights = weights
        return self

    def predict(self, receiver: pd.DataFrame) -> pd.DataFrame:
        self.receiver = receiver.copy()
        self.__class__.predict_receivers.append(receiver.copy())
        self.__class__.predict_start_offsets.append(self.draw_offset)
        self.draw_offset += len(receiver)
        return pd.DataFrame(
            {_OUTPUT: receiver["bank_account_assets"].to_numpy() > 0.0},
            index=receiver.index,
        )


@pytest.fixture(autouse=True)
def _clear_fake_instances() -> None:
    _FakeQRF.instances.clear()
    _FakeQRF.predict_receivers.clear()
    _FakeQRF.predict_start_offsets.clear()


__all__ = [name for name in globals() if not name.startswith("__")]
