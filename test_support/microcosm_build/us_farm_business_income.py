"""Signed ASEC/PUF farm-business input restoration contracts."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.puf_support as puf_support_module
from microcosm.build.us_runtime.cps_carried import derive_us_cps_carried_inputs
from microcosm.build.us_runtime.farm_business_income import (
    FARM_BUSINESS_INCOME_ARCHIVED_CPS_FARM_INCOME_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_DERIVATION_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_EXPORT_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_IMPUTATION_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_OVERRIDE_URL,
    FARM_BUSINESS_INCOME_ARCHIVED_PUF_ARTIFACT_URL,
    US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS,
    derive_us_farm_business_income_from_puf,
    us_farm_business_income_signal_gate,
    us_farm_business_income_stage_spec,
)
from microcosm.build.us_runtime.puf_aggregate_records import (
    _reconcile_puf_farm_business_income_from_sources,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    clone_us_frame_for_puf_support,
    impute_us_puf_tax_detail_support,
    puf_tax_unit_donor_from_arrays,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_OPERATIONS, _RENT = US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS


def _asec_frame() -> Frame:
    count = 20
    ids = np.arange(1, count + 1, dtype=np.int64)
    farm_operations = np.zeros(count, dtype=np.float64)
    farm_operations[:4] = [500.0, -200.0, 800.0, -300.0]
    person = pd.DataFrame(
        {
            "person_id": ids,
            "person_household_id": ids + 100,
            "person_tax_unit_id": ids + 200,
            "person_spm_unit_id": ids + 300,
            "person_family_id": ids + 400,
            "person_marital_unit_id": ids + 500,
            "A_AGE": np.arange(30, 30 + count),
            "A_SEX": np.tile([1, 2], count // 2),
            "WSAL_VAL": np.arange(count) * 1_000.0,
            "SEMP_VAL": np.arange(count) * 100.0,
            "FRSE_VAL": farm_operations,
            "OI_VAL": np.zeros(count),
            "OI_OFF": np.zeros(count),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {"household_id": ids + 100, "state_fips": np.full(count, 6)}
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": ids + 200,
                "filing_status_input": ["SINGLE"] * count,
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": ids + 300}),
        "family": pd.DataFrame({"family_id": ids + 400}),
        "marital_unit": pd.DataFrame({"marital_unit_id": ids + 500}),
    }
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


def _imputed_frame(monkeypatch: pytest.MonkeyPatch) -> Frame:
    asec = derive_us_cps_carried_inputs(_asec_frame())
    expanded = clone_us_frame_for_puf_support(asec)
    predictions = {
        _OPERATIONS: np.asarray([700.0, -400.0, 900.0, -500.0, *([0.0] * 16)]),
        _RENT: np.asarray([300.0, -100.0, 600.0, -200.0, *([0.0] * 16)]),
    }

    class FakeFitted:
        def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
            return pd.DataFrame(predictions, index=test.index)

    class FakeQRF:
        def __init__(self, *, n_estimators: int, seed: int) -> None:
            assert n_estimators == 100
            assert seed == 11

        def fit(
            self,
            frame: Frame,
            predictors: list[str],
            outcomes: list[str],
            *,
            weights: str,
        ) -> FakeFitted:
            assert predictors == []
            assert outcomes == list(US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS)
            assert weights == "design"
            return FakeFitted()

    monkeypatch.setattr(puf_support_module, "QRF", FakeQRF)
    donor = pd.DataFrame(
        {
            _OPERATIONS: [10.0, -20.0, 0.0, 30.0],
            _RENT: [-5.0, 15.0, 0.0, 25.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    return impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=(),
        person_outputs=US_FARM_BUSINESS_INCOME_OUTPUT_COLUMNS,
        tax_unit_outputs=(),
        seed=11,
    )


__all__ = [name for name in globals() if not name.startswith("__")]
