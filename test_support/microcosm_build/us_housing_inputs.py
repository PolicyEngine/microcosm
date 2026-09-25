"""Archived CPS/ACS housing-input restoration."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.housing_inputs as module
from microcosm.build.us_runtime.housing_inputs import (
    ACS_2022_RENT_ARTIFACT_SHA256,
    HOUSING_INPUTS_ARCHIVED_ACS_DERIVATION_URL,
    HOUSING_INPUTS_ARCHIVED_CPS_RENT_URL,
    HOUSING_INPUTS_ARCHIVED_CPS_SPM_URL,
    HOUSING_INPUTS_ARCHIVED_PUF_IMPUTATION_URL,
    HOUSING_TAKE_UP_ARCHIVED_DERIVATION_URL,
    HOUSING_TAKE_UP_ARCHIVED_HUD_ETL_URL,
    HOUSING_TAKE_UP_ARCHIVED_PARAMETER_URL,
    US_HOUSING_INPUTS_OUTPUT_COLUMNS,
    US_HOUSING_NONCONSTANT_HOUSEHOLD_COLUMNS,
    US_HOUSING_NONCONSTANT_PERSON_COLUMNS,
    US_HOUSING_NONCONSTANT_SPM_UNIT_COLUMNS,
    derive_us_housing_inputs,
    impute_us_housing_assistance_to_puf_support,
    load_acs_2022_rent_donor,
    us_housing_inputs_signal_gate,
    us_housing_inputs_stage_spec,
    with_us_housing_inputs,
)
from microcosm.build.us_runtime.l0_refit_export import (
    US_RELEASE_REQUIRED_HOUSEHOLD_NONCONSTANT_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_PERSON_SOURCE_COLUMNS,
    US_RELEASE_REQUIRED_SPM_UNIT_SOURCE_COLUMNS,
)
from microcosm.build.us_runtime.puf_support import (
    BASE_ASEC_SUPPORT_CHANNEL,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    clone_us_frame_for_puf_support,
    support_channel_column,
)
from microcosm.build.us_runtime.release_input_coverage import (
    RESTORED_REFERENCE_ECPS_REQUIRED_INPUTS,
    load_release_input_coverage_manifest,
    us_release_reform_coverage_probes,
)
from microcosm.build.us_runtime.take_up_contract import load_take_up_contract
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _frame() -> Frame:
    n = 20
    household_ids = np.arange(1, n + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": household_ids,
            "person_household_id": household_ids,
            "person_tax_unit_id": household_ids + 100,
            "person_spm_unit_id": household_ids + 200,
            "person_family_id": household_ids + 300,
            "person_marital_unit_id": household_ids + 400,
            "is_household_head": np.ones(n, dtype=bool),
            "tax_unit_role_input": ["HEAD"] * n,
            "age": np.linspace(25, 75, n),
            "is_female": household_ids % 2 == 0,
            "has_esi": household_ids % 3 == 0,
            "employment_income_before_lsr": household_ids * 2_000.0,
            "self_employment_income_before_lsr": household_ids * 100.0,
            "social_security_retirement": np.where(household_ids > 15, 12_000.0, 0.0),
            "social_security_disability": np.zeros(n),
            "social_security_survivors": np.zeros(n),
            "social_security_dependents": np.zeros(n),
            "taxable_private_pension_income": np.where(
                household_ids > 15, 4_000.0, 0.0
            ),
            "tax_exempt_private_pension_income": np.zeros(n),
            "SPM_CAPHOUSESUB": np.where(household_ids == 1, 5_000.0, 0.0),
            "SPM_TENMORTSTATUS": np.resize(np.array([3, 1, 2]), n),
        }
    )
    h_tenure = np.where(household_ids <= 4, 2, np.where(household_ids == 5, 3, 1))
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "state_fips": np.where(household_ids <= 10, 6, 36),
                "H_TENURE": h_tenure,
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": household_ids + 100,
                "filing_status_input": ["SINGLE"] * n,
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": household_ids + 200}),
        "family": pd.DataFrame({"family_id": household_ids + 300}),
        "marital_unit": pd.DataFrame({"marital_unit_id": household_ids + 400}),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.full(n, 1_000_000.0),
                WeightKind.DESIGN,
            )
        },
    )


def _donor(n: int = 60) -> pd.DataFrame:
    rows = np.arange(n, dtype=np.float64)
    donor = pd.DataFrame(
        {
            predictor: rows + position
            for position, predictor in enumerate(module.ACS_RENT_PREDICTORS)
        }
    )
    donor["is_household_head"] = 1.0
    donor["tenure_type"] = np.resize(
        np.array(["NONE", "OWNED_WITH_MORTGAGE", "RENTED"]), n
    )
    donor["state_code_str"] = np.resize(np.array(["06", "36", "48"]), n)
    donor["rent"] = np.where(donor["tenure_type"] == "RENTED", 12_000.0, 0.0)
    donor["rent_is_allocated"] = False
    donor["real_estate_taxes"] = np.where(
        donor["tenure_type"] == "RENTED", 0.0, 4_000.0
    )
    donor["real_estate_taxes_is_allocated"] = False
    donor["household_weight"] = np.linspace(1.0, 2.0, n)
    return donor


class _RentFitted:
    def predict(self, test: pd.DataFrame, **kwargs) -> pd.DataFrame:
        rented = test["tenure_type__RENTED"] > 0
        return pd.DataFrame(
            {"rent": np.where(rented, 12_000.0, 0.0)},
            index=test.index,
        )


class _RentQRF:
    def __init__(self, **_kwargs: object) -> None:
        pass

    def fit(self, *_args: object, **_kwargs: object) -> _RentFitted:
        return _RentFitted()


def _write_tiny_acs(path: Path) -> None:
    arrays = {
        "person_id": np.array([1, 2, 3, 4]),
        "person_household_id": np.array([10, 10, 20, 30]),
        "is_household_head": np.array([True, False, True, True]),
        "age": np.array([40, 38, 50, 60]),
        "is_male": np.array([True, False, True, False]),
        "employment_income": np.array([50_000, 20_000, 0, 10_000]),
        "self_employment_income": np.array([0, 0, 5_000, 0]),
        "social_security": np.array([0, 0, 10_000, 12_000]),
        "taxable_private_pension_income": np.array([0, 0, 3_000, 4_000]),
        "rent": np.array([12_000, 0, 0, 18_000]),
        "rent_is_allocated": np.array([False, False, True, False]),
        "real_estate_taxes": np.array([0, 0, 8_000, 4_000]),
        "real_estate_taxes_is_allocated": np.array([False, False, False, True]),
        "household_id": np.array([10, 20, 30]),
        "household_weight": np.array([100.0, 200.0, 300.0]),
        "state_fips": np.array([6, 36, 48]),
        "tenure_type": np.array([b"RENTED", b"OWNED_OUTRIGHT", b"OWNED_WITH_MORTGAGE"]),
    }
    with h5py.File(path, "w") as h5:
        for name, values in arrays.items():
            h5[name] = values


__all__ = [name for name in globals() if not name.startswith("__")]
