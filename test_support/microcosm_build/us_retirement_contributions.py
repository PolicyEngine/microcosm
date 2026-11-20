"""Retirement-contribution fixture frames for the contributions tests.

``_person_source``, ``_frame`` and ``_stacked_frame`` moved here verbatim from
``tests/engine_free/us/test_us_retirement_contributions.py``, so the d2283
PR 3 differential (``tests/engine_free/us/test_us_pr3_route_a_differential.py``)
grades the same pre-clone and stacked frames as that file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.retirement_contributions import (
    with_us_retirement_contribution_inputs,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _person_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3, 4], dtype="int64"),
            "person_household_id": np.asarray([10, 20, 30, 40], dtype="int64"),
            "person_tax_unit_id": np.asarray([100, 200, 300, 400], dtype="int64"),
            "person_spm_unit_id": np.asarray([1_000, 2_000, 3_000, 4_000]),
            "person_family_id": np.asarray([10_000, 20_000, 30_000, 40_000]),
            "person_marital_unit_id": np.asarray([100_000, 200_000, 300_000, 400_000]),
            # Wage-only, self-employment-only, both, and neither.
            "RETCB_VAL": [1.0, 1.0, 1.0, 1.0],
            "WSAL_VAL": [50_000.0, 0.0, 50_000.0, 0.0],
            "SEMP_VAL": [0.0, 30_000.0, 30_000.0, 0.0],
            "employment_income_before_lsr": [50_000.0, 0.0, 50_000.0, 0.0],
            "self_employment_income_before_lsr": [0.0, 30_000.0, 30_000.0, 0.0],
            "age": [35, 45, 55, 25],
            "is_female": [False, True, False, True],
            "has_esi": [True, False, True, False],
            "tax_unit_role_input": ["HEAD", "HEAD", "HEAD", "HEAD"],
            "social_security_retirement": [0.0, 0.0, 2_000.0, 0.0],
            "social_security_disability": [0.0, 0.0, 0.0, 0.0],
            "social_security_dependents": [0.0, 0.0, 0.0, 0.0],
            "social_security_survivors": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _frame() -> Frame:
    person = _person_source()
    ids = {
        "household": [10, 20, 30, 40],
        "tax_unit": [100, 200, 300, 400],
        "spm_unit": [1_000, 2_000, 3_000, 4_000],
        "family": [10_000, 20_000, 30_000, 40_000],
        "marital_unit": [100_000, 200_000, 300_000, 400_000],
    }
    tables = {
        entity: pd.DataFrame({f"{entity}_id": np.asarray(values, dtype="int64")})
        for entity, values in ids.items()
    }
    tables["person"] = person
    tables["tax_unit"]["filing_status_input"] = [
        "SINGLE",
        "SINGLE",
        "SINGLE",
        "SINGLE",
    ]
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([1.0, 1.0, 1.0, 1.0]),
                WeightKind.DESIGN,
            )
        },
    )


def _stacked_frame() -> Frame:
    direct = with_us_retirement_contribution_inputs(
        _frame(),
        seed=0,
        time_period=2024,
    )
    stacked = clone_us_frame_for_puf_support(direct)
    person = stacked.table("person")
    source_record = np.tile(np.arange(4, dtype=np.int64), 2)
    person["person_spine_source_id"] = source_record
    person["person_support_channel"] = np.where(
        source_record < 2,
        "asec",
        "acs",
    )
    acs = person["person_support_channel"].eq("acs")
    person.loc[acs, ["RETCB_VAL", "WSAL_VAL", "SEMP_VAL"]] = np.nan
    return stacked


__all__ = [name for name in globals() if not name.startswith("__")]
