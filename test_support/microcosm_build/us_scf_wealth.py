"""US SIPP/SCF wealth and SSI countable-resource tests (#49/#356/#368/#374).

The three asset leaves ``bank_account_assets`` / ``stock_assets`` /
``bond_assets`` are what ``ssi_countable_resources`` sums; with them absent the
SSI resource-limit reform class scores $0 (the #356 failure). This stage
draws them from one SIPP or SCF source per household reference person and
restores signed household ``net_worth`` from the retired pipeline's direct SCF
anchor.
"""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.scf_wealth as scf_wealth_runtime
from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.us_runtime import (
    FINANCIAL_ASSET_BLEND_AUDIT_KEY,
    FINANCIAL_ASSET_SOURCE_SCF_PROBABILITY,
    SCF_FINANCIAL_ASSET_TARGET_COMPONENTS,
    SCF_NET_WORTH_TARGET_COMPONENTS,
    SCF_WEALTH_PREDICTORS,
    SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN,
    SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS,
    US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS,
    US_SCF_NET_WORTH_OUTPUT_COLUMNS,
    US_SCF_WEALTH_NONCONSTANT_HOUSEHOLD_COLUMNS,
    US_SCF_WEALTH_STAGE_NAME,
    fetch_scf_2022_summary_extract,
    financial_asset_source_is_scf,
    impute_us_scf_financial_assets,
    impute_us_scf_net_worth,
    impute_us_sipp_financial_assets,
    impute_us_sipp_scf_financial_assets,
    load_scf_2022_financial_asset_donor,
    us_scf_wealth_signal_gate,
    us_scf_wealth_stage_spec,
    us_scf_wealth_summary,
    with_us_scf_wealth_inputs,
)
from microcosm.build.us_runtime.scf_wealth import (
    _household_head_mask,
    _recipient_cps_race,
    _replace_sentinels,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

TIME_PERIOD = 2024

_DONOR_WEIGHT_COLUMN = "scf_weight"


# --------------------------------------------------------------------------- #
# Fixtures                                                                      #
# --------------------------------------------------------------------------- #
def _raw_scf_summary() -> pd.DataFrame:
    """A tiny SCF-summary-extract-shaped table (the columns the loader reads)."""

    rng = np.random.default_rng(0)
    n = 400
    liq = rng.gamma(2.0, 3_000.0, n)
    net_worth = rng.lognormal(12.0, 1.1, n)
    indebted = rng.random(n) < 0.10
    net_worth[indebted] = -rng.gamma(2.0, 20_000.0, indebted.sum())
    return pd.DataFrame(
        {
            "liq": liq,
            "stocks": rng.gamma(1.0, 5_000.0, n),
            "nmmf": rng.gamma(1.0, 4_000.0, n),
            "bond": np.where(rng.random(n) < 0.05, rng.gamma(1.0, 9_000.0, n), 0.0),
            "networth": net_worth,
            "wgt": rng.uniform(500.0, 2_000.0, n),
            "age": rng.integers(20, 85, n).astype(float),
            "hhsex": rng.integers(1, 3, n).astype(float),
            "racecl5": rng.integers(1, 6, n).astype(float),
            "married": (rng.random(n) < 0.5).astype(float),
            "kids": rng.integers(0, 4, n).astype(float),
            "wageinc": rng.gamma(2.0, 20_000.0, n),
            "intdivinc": rng.gamma(1.0, 1_000.0, n),
            "ssretinc": rng.gamma(1.0, 8_000.0, n),
        }
    )


def _donor_table() -> pd.DataFrame:
    """A ready-made donor table (as the loader would emit) for impute tests."""

    rng = np.random.default_rng(1)
    n = 400
    frame = pd.DataFrame({p: rng.normal(0.0, 1.0, n) for p in SCF_WEALTH_PREDICTORS})
    frame["age"] = rng.integers(18, 90, n).astype(float)
    frame["is_female"] = (rng.random(n) < 0.5).astype(float)
    frame["cps_race"] = rng.integers(1, 8, n).astype(float)
    frame["is_married"] = (rng.random(n) < 0.5).astype(float)
    frame["own_children_in_household"] = rng.integers(0, 4, n).astype(float)
    frame["employment_income"] = rng.gamma(2.0, 15_000.0, n)
    frame["interest_dividend_income"] = rng.gamma(1.0, 900.0, n)
    frame["social_security_pension_income"] = rng.gamma(1.0, 7_000.0, n)
    frame["bank_account_assets"] = rng.gamma(2.0, 3_000.0, n)
    frame["stock_assets"] = np.where(
        rng.random(n) < 0.25, rng.gamma(1.0, 20_000.0, n), 0.0
    )
    frame["bond_assets"] = np.where(
        rng.random(n) < 0.04, rng.gamma(1.0, 9_000.0, n), 0.0
    )
    net_worth = rng.lognormal(12.0, 1.1, n)
    indebted = rng.random(n) < 0.12
    net_worth[indebted] = -rng.gamma(2.0, 20_000.0, indebted.sum())
    frame["net_worth"] = net_worth
    frame[_DONOR_WEIGHT_COLUMN] = rng.uniform(500.0, 2_000.0, n)
    return frame


def _sipp_donor_table() -> pd.DataFrame:
    """Small low-liquid-asset donor; unit tests never read ``pu2023.csv``."""

    rng = np.random.default_rng(374)
    n = 400
    frame = pd.DataFrame(
        {
            predictor: rng.normal(size=n)
            for predictor in SIPP_FINANCIAL_ASSET_MODEL_PREDICTORS
        }
    )
    frame["age"] = rng.integers(18, 90, n).astype(float)
    frame["is_female"] = rng.integers(0, 2, n).astype(float)
    frame["is_married"] = rng.integers(0, 2, n).astype(float)
    frame["count_under_18"] = rng.integers(0, 4, n).astype(float)
    frame["count_under_6"] = rng.integers(0, 2, n).astype(float)
    frame["household_size"] = rng.integers(1, 6, n).astype(float)
    frame["employment_income"] = rng.gamma(1.5, 10_000.0, n)
    frame["social_security"] = rng.gamma(0.8, 4_000.0, n)
    frame["retirement_income"] = rng.gamma(0.8, 5_000.0, n)
    frame["non_ssi_income"] = (
        frame["employment_income"]
        + frame["social_security"]
        + frame["retirement_income"]
    )
    frame["bank_account_assets"] = np.where(
        rng.random(n) < 0.65,
        rng.uniform(50.0, 1_500.0, n),
        0.0,
    )
    frame["stock_assets"] = np.where(
        rng.random(n) < 0.10,
        rng.uniform(50.0, 1_000.0, n),
        0.0,
    )
    frame["bond_assets"] = np.where(
        rng.random(n) < 0.04,
        rng.uniform(25.0, 500.0, n),
        0.0,
    )
    frame[SIPP_FINANCIAL_ASSET_DONOR_WEIGHT_COLUMN] = rng.uniform(1.0, 4.0, n)
    for target in US_SCF_FINANCIAL_ASSET_OUTPUT_COLUMNS:
        frame[f"{target}_is_observed"] = True
    return frame


def _person_rows(n_households: int = 60) -> pd.DataFrame:
    """A raw-ASEC-shaped recipient person table: two persons per household."""

    records: list[dict] = []
    rng = np.random.default_rng(2)
    person_id = 1
    for household in range(1, n_households + 1):
        # Head (line 1) then a second member (line 2).
        records.append(
            {
                "person_id": person_id,
                "person_household_id": household,
                "PH_SEQ": household,
                "A_LINENO": 1,
                "age": float(rng.integers(30, 85)),
                "is_female": bool(rng.integers(0, 2)),
                "PRDTRACE": int(rng.integers(1, 7)),
                "PRDTHSP": int(rng.integers(0, 2)),
                "A_MARITL": int(rng.integers(1, 8)),
                "PEPAR1": -1,
                "PEPAR2": -1,
                "employment_income_before_lsr": float(rng.gamma(2.0, 12_000.0)),
                "taxable_interest_income": float(rng.gamma(1.0, 500.0)),
                "social_security_retirement": float(rng.gamma(1.0, 6_000.0)),
            }
        )
        person_id += 1
        records.append(
            {
                "person_id": person_id,
                "person_household_id": household,
                "PH_SEQ": household,
                "A_LINENO": 2,
                "age": float(rng.integers(1, 60)),
                "is_female": bool(rng.integers(0, 2)),
                "PRDTRACE": int(rng.integers(1, 7)),
                "PRDTHSP": int(rng.integers(0, 2)),
                "A_MARITL": int(rng.integers(1, 8)),
                "PEPAR1": -1,
                "PEPAR2": -1,
                "employment_income_before_lsr": float(rng.gamma(1.0, 3_000.0)),
                "taxable_interest_income": 0.0,
                "social_security_retirement": 0.0,
            }
        )
        person_id += 1
    return pd.DataFrame(records)


def _us_frame(
    person: pd.DataFrame,
    *,
    weights: list[float] | None = None,
    household_extra: dict[str, object] | None = None,
) -> Frame:
    person = person.copy()
    n = len(person)
    household_ids = person["person_household_id"].to_numpy()
    unique_households = np.unique(household_ids)
    person["person_tax_unit_id"] = person["person_household_id"] + 1_000
    person["person_spm_unit_id"] = person["person_household_id"] + 2_000
    person["person_family_id"] = person["person_household_id"] + 3_000
    person["person_marital_unit_id"] = np.arange(n, dtype="int64") + 4_000
    household = pd.DataFrame({"household_id": unique_households})
    for column, values in (household_extra or {}).items():
        household[column] = values
    tables = {
        "person": person,
        "household": household,
        "tax_unit": pd.DataFrame({"tax_unit_id": unique_households + 1_000}),
        "spm_unit": pd.DataFrame({"spm_unit_id": unique_households + 2_000}),
        "family": pd.DataFrame({"family_id": unique_households + 3_000}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": np.arange(n, dtype="int64") + 4_000}
        ),
    }
    w = weights or [1.0] * len(unique_households)
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                values=np.asarray(w, dtype=np.float64), kind=WeightKind.DESIGN
            )
        },
    )


@pytest.fixture(scope="module")
def sipp_scf_blend_case() -> dict[str, object]:
    """Fast synthetic case shared by the #374 block-blend tests."""

    person = _person_rows(240)
    scf_donor = _donor_table()
    sipp_donor = _sipp_donor_table()
    seed = 23
    n_estimators = 8
    blended = impute_us_sipp_scf_financial_assets(
        person,
        scf_donor,
        sipp_donor,
        seed=seed,
        time_period=TIME_PERIOD,
        n_estimators=n_estimators,
    )
    scf = impute_us_scf_financial_assets(
        person,
        scf_donor,
        seed=seed,
        n_estimators=n_estimators,
    )
    sipp = impute_us_sipp_financial_assets(
        person,
        sipp_donor,
        seed=seed,
        n_estimators=n_estimators,
    )
    return {
        "person": person,
        "scf_donor": scf_donor,
        "sipp_donor": sipp_donor,
        "seed": seed,
        "n_estimators": n_estimators,
        "blended": blended,
        "scf": scf,
        "sipp": sipp,
    }


# --------------------------------------------------------------------------- #
# Manifest declaration                                                          #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Donor loading                                                                 #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Predictor construction                                                        #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Imputation (head-carry)                                                        #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Frame integration                                                             #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Gate + summary                                                                #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Provisioning helper                                                           #
# --------------------------------------------------------------------------- #

__all__ = [name for name in globals() if not name.startswith("__")]
