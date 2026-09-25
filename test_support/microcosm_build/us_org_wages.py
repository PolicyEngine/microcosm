"""Exact CPS ORG, occupation, and FLSA overtime stage contracts."""

# ruff: noqa: F401

from __future__ import annotations

import gzip
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.us_runtime import (
    ORG_2024_DONOR_CONTENT_SHA256,
    ORG_PREDICTORS,
    US_ORG_WAGES_OUTPUT_COLUMNS,
    derive_flsa_overtime_premium,
    derive_us_org_occupation_inputs,
    fetch_org_2024_donor,
    load_org_2024_donor,
    us_org_wages_signal_gate,
    us_org_wages_stage_spec,
)
from microcosm.build.us_runtime import org_wages as module
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSVariableMetadataIndex,
)
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


def _load_fiscal_builder_module():
    root = _TEST_PATHS.repository
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release_hours_org", path
    )
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(builder)
    return builder


def _person(n: int = 1_000) -> pd.DataFrame:
    household = np.arange(1, n + 1, dtype=np.int64)
    occupation = np.full(n, 20, dtype=np.int16)
    occupation[:180] = 0
    occupation[180:420] = 53
    occupation[420:423] = 52
    occupation[423:443] = 8
    occupation[443:447] = 41
    occupation[447:737] = 1
    employment = np.zeros(n)
    employment[400:] = 52_000.0
    return pd.DataFrame(
        {
            "person_id": np.arange(1, n + 1, dtype=np.int64),
            "person_household_id": household,
            "person_tax_unit_id": household + 10_000,
            "person_spm_unit_id": household + 20_000,
            "person_family_id": household + 30_000,
            "person_marital_unit_id": household + 40_000,
            "age": np.resize(np.arange(18, 78), n),
            "is_female": np.arange(n) % 2 == 0,
            "PRDTRACE": np.resize(np.asarray([1, 1, 2, 4]), n),
            "PRDTHSP": np.arange(n) % 7 == 0,
            "POCCU2": occupation,
            "employment_income_before_lsr": employment,
            "self_employment_income_before_lsr": 0.0,
            "weekly_hours_worked_before_lsr": np.where(employment > 0, 40.0, 0.0),
            "hours_worked_last_week": np.where(
                np.arange(n) >= 950, 50.0, np.where(employment > 0, 40.0, 0.0)
            ),
            "weeks_worked": np.where(employment > 0, 52.0, 0.0),
        }
    )


def _frame(person: pd.DataFrame) -> Frame:
    household_ids = person["person_household_id"].to_numpy()
    n = len(person)
    return Frame(
        {
            "person": person,
            "household": pd.DataFrame(
                {"household_id": household_ids, "state_fips": np.resize([6, 36], n)}
            ),
            "tax_unit": pd.DataFrame(
                {"tax_unit_id": person["person_tax_unit_id"].to_numpy()}
            ),
            "spm_unit": pd.DataFrame(
                {"spm_unit_id": person["person_spm_unit_id"].to_numpy()}
            ),
            "family": pd.DataFrame(
                {"family_id": person["person_family_id"].to_numpy()}
            ),
            "marital_unit": pd.DataFrame(
                {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
            ),
        },
        US_SCHEMA,
        {"household": Weights(np.ones(n), WeightKind.DESIGN)},
    )


def _donor(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return pd.DataFrame(
        {
            "employment_income": rng.uniform(10_000, 120_000, n),
            "weekly_hours_worked": rng.uniform(20, 60, n),
            "age": rng.integers(18, 80, n),
            "is_female": rng.integers(0, 2, n),
            "is_hispanic": rng.integers(0, 2, n),
            "race_wbho": rng.integers(1, 5, n),
            "state_fips": np.resize([6, 36], n),
            "hourly_wage": rng.uniform(8, 80, n),
            "is_paid_hourly": rng.integers(0, 2, n),
            "sample_weight": rng.uniform(1, 5, n),
        }
    )


def _plausible_surface() -> Frame:
    person = _person()
    person["cps_race"] = person["PRDTRACE"]
    person["is_hispanic"] = person["PRDTHSP"].ne(0)
    carried = derive_us_org_occupation_inputs(person)
    for column in carried:
        person[column] = carried[column]
    person["hourly_wage"] = 0.0
    person.loc[400:949, "hourly_wage"] = 25.0
    person["is_paid_hourly"] = False
    person.loc[400:649, "is_paid_hourly"] = True
    person["is_union_member_or_covered"] = False
    person.loc[700:759, "is_union_member_or_covered"] = True
    person["fsla_overtime_premium"] = 0.0
    # These 50 are non-exempt code-20 workers with 50 hours and positive wages.
    person.loc[950:999, "fsla_overtime_premium"] = 52_000 / 11
    # These 50 carry the usual-hours leg: reference week at the threshold but a
    # 45-hour usual week (share 1/19) — valid carriers under the two-signal
    # estimator.
    person.loc[900:949, "weekly_hours_worked_before_lsr"] = 45.0
    person.loc[900:949, "fsla_overtime_premium"] = 52_000 / 19
    return _frame(person)


class _ConstantQRF:
    """Deterministic stand-in: 25.0/hr, everyone paid hourly."""

    def __init__(self, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        return self

    def predict(self, features):
        return pd.DataFrame(
            {
                "hourly_wage": np.full(len(features), 25.0),
                "is_paid_hourly": np.ones(len(features)),
            },
            index=features.index,
        )


__all__ = [name for name in globals() if not name.startswith("__")]
