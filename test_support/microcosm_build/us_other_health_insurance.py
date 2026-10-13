"""ASEC other-health-insurance restoration and ESI source exclusion."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
import json
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import microcosm.build.us_runtime.other_health_insurance as module
from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import US_SOURCE_MANIFEST
from microcosm.build.us_runtime.other_health_insurance import (
    OTHER_HEALTH_INSURANCE_ARCHIVED_DERIVATION_URL,
    OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_IMPUTATION_URL,
    OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_OUTPUTS_URL,
    OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_PREDICTORS_URL,
    OTHER_HEALTH_INSURANCE_ARCHIVED_PUF_SPLICE_URL,
    US_OTHER_HEALTH_INSURANCE_NONCONSTANT_PERSON_COLUMNS,
    US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS,
    US_OTHER_HEALTH_INSURANCE_STAGE_OUTPUT_COLUMNS,
    US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS,
    US_SE_HEALTH_MEDICARE_AGE_THRESHOLD,
    US_SE_HEALTH_SELF_EMPLOYMENT_INCOME_SOURCES,
    attribute_us_se_health_premiums,
    attribute_us_se_health_premiums_from_manifest,
    derive_us_other_health_insurance_from_asec,
    derive_us_other_health_insurance_from_manifest,
    impute_us_other_health_insurance_to_puf_support_from_manifest,
    us_other_health_insurance_signal_gate,
    us_other_health_insurance_stage_spec,
    with_us_other_health_insurance_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

_TEST_PATHS = paths_for("microcosm-build")


_REPORTED, _OTHER = US_OTHER_HEALTH_INSURANCE_OUTPUT_COLUMNS
ROOT = _TEST_PATHS.repository
_PREDICTORS = (
    "age",
    "is_male",
    "has_esi",
    "tax_unit_is_joint",
    "tax_unit_count_dependents",
    "employment_income",
    "self_employment_income",
    "social_security",
)
_REPORTED_VALUES = np.asarray(
    [0.0, 1_000.0, 0.0, 2_000.0, 0.0, 3_000.0, 0.0, 4_000.0, 0.0, 0.0]
)
# Person 5 (reported premium 3,000, age 35) is the frame's self-employment
# carrier so the attribution surface is nonzero on every integration path.
_SELF_EMPLOYMENT_VALUES = np.asarray(
    [0.0, 0.0, 0.0, 0.0, 0.0, 20_000.0, 0.0, 0.0, 0.0, 0.0]
)


def _frame() -> Frame:
    count = len(_REPORTED_VALUES)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1, dtype="int64"),
            "person_household_id": np.arange(101, 101 + count, dtype="int64"),
            "person_tax_unit_id": np.arange(201, 201 + count, dtype="int64"),
            "person_spm_unit_id": np.arange(301, 301 + count, dtype="int64"),
            "person_family_id": np.arange(401, 401 + count, dtype="int64"),
            "person_marital_unit_id": np.arange(501, 501 + count, dtype="int64"),
            _REPORTED: _REPORTED_VALUES,
            "age": np.arange(30, 30 + count),
            "is_female": np.tile([True, False], count // 2),
            # Person 5 (the self-employment carrier) stays outside measured
            # employer coverage so the 162(l)(2)(B) guard leaves the frame's
            # attribution surface nonzero.
            "has_esi": np.asarray(
                [False, True, False, True, False, False, False, True, False, True]
            ),
            "tax_unit_role_input": ["HEAD"] * count,
            "employment_income_before_lsr": np.arange(count) * 5_000.0,
            "self_employment_income_before_lsr": _SELF_EMPLOYMENT_VALUES,
            "sstb_self_employment_income_before_lsr": np.zeros(count),
            "social_security_retirement": np.zeros(count),
            "social_security_disability": np.zeros(count),
            "social_security_dependents": np.zeros(count),
            "social_security_survivors": np.zeros(count),
        }
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": person["person_household_id"],
                "state_fips": np.full(count, 6),
            }
        ),
        "tax_unit": pd.DataFrame(
            {
                "tax_unit_id": person["person_tax_unit_id"],
                "filing_status_input": ["SINGLE"] * count,
            }
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": person["person_spm_unit_id"]}),
        "family": pd.DataFrame({"family_id": person["person_family_id"]}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"]}
        ),
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


class _ZeroPremiumEngine:
    def materialize(
        self,
        frame: Frame,
        variables: list[str],
        *,
        period: int,
    ) -> dict[str, np.ndarray]:
        assert period == 2024
        assert variables == list(
            module.US_OTHER_HEALTH_INSURANCE_MODELED_PREMIUM_VARIABLES
        )
        return {
            variable: np.zeros(frame.n("tax_unit"), dtype=np.float64)
            for variable in variables
        }


_PREMIUMS_OUTPUT, _FLAG_OUTPUT = US_SE_HEALTH_ATTRIBUTION_OUTPUT_COLUMNS


def _attribution_source(
    *,
    reported: list[float],
    self_employment: list[float],
    sstb: list[float] | None = None,
    age: list[float] | None = None,
    ssdi: list[float] | None = None,
    esi: list[bool] | None = None,
) -> pd.DataFrame:
    count = len(reported)
    return pd.DataFrame(
        {
            _REPORTED: reported,
            "self_employment_income_before_lsr": self_employment,
            "sstb_self_employment_income_before_lsr": (
                sstb if sstb is not None else [0.0] * count
            ),
            "age": age if age is not None else [40.0] * count,
            "social_security_disability": (ssdi if ssdi is not None else [0.0] * count),
            "has_esi": esi if esi is not None else [False] * count,
        }
    )


__all__ = [name for name in globals() if not name.startswith("__")]
