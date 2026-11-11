"""ASEC-measured CDCC adult-care inputs (PolicyEngine/microcosm#451 item 1)."""

# ruff: noqa: F401

from __future__ import annotations

import importlib.util
from importlib.metadata import version

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import acs_transfer as acs_transfer_module
from microcosm.build.us_runtime import multispine_pool as multispine_pool_module
from microcosm.build.us_runtime.adult_care import (
    US_ADULT_CARE_CHILD_QUALIFYING_AGE_LIMIT,
    US_ADULT_CARE_EARNED_INCOME_SOURCES,
    US_ADULT_CARE_OUTPUT_COLUMNS,
    US_ADULT_CARE_REQUIRED_SOURCE_COLUMNS,
    derive_us_adult_care_from_manifest,
    us_adult_care_signal_gate,
    us_adult_care_stage_spec,
    with_us_adult_care_inputs,
)
from microcosm.build.us_runtime.puf_support import clone_us_frame_for_puf_support
from microcosm.build.us_runtime.source_runtime import us_source_operation_handlers
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

_FLAG, _EXPENSE = US_ADULT_CARE_OUTPUT_COLUMNS
_DONOR_CHILDCARE = 4_000.0


def _frame() -> Frame:
    """25 people: two statute-eligible units, donor units, and filler singles.

    Unit 201: married head (earned) + spouse with measured PEDISDRS == 1 and
    no earnings — the 21(b)(1)(C) prong whose work test passes only through
    the 21(d)(2) deeming rule.
    Unit 202: single head (earned) + disabled adult dependent — 21(b)(1)(B).
    Unit 203: two working parents + a child under 13 with positive measured
    childcare — the paid-usage donor.
    Unit 204: two working parents + a child under 13 with zero childcare —
    holds the measured usage rate strictly inside (0, 1).
    Units 205+: working singles diluting the weighted flag share into the
    plausibility band.
    """

    rows: list[dict[str, object]] = []

    def _person(
        unit: int,
        role: str,
        *,
        age: float,
        pedisdrs: float = 2.0,
        employment: float = 0.0,
        student: bool = False,
    ) -> None:
        rows.append(
            {
                "person_tax_unit_id": unit,
                "tax_unit_role_input": role,
                "age": age,
                "PEDISDRS": pedisdrs,
                "employment_income_before_lsr": employment,
                "self_employment_income_before_lsr": 0.0,
                "sstb_self_employment_income_before_lsr": 0.0,
                "is_full_time_college_student": student,
                "person_support_channel": "asec",
            }
        )

    _person(201, "HEAD", age=45, employment=80_000.0)
    _person(201, "SPOUSE", age=44, pedisdrs=1.0)
    _person(202, "HEAD", age=50, employment=40_000.0)
    _person(202, "DEPENDENT", age=25, pedisdrs=1.0)
    _person(203, "HEAD", age=35, employment=60_000.0)
    _person(203, "SPOUSE", age=34, employment=30_000.0)
    _person(203, "DEPENDENT", age=6)
    _person(204, "HEAD", age=36, employment=55_000.0)
    _person(204, "SPOUSE", age=35, employment=25_000.0)
    _person(204, "DEPENDENT", age=4)
    for offset in range(15):
        _person(205 + offset, "HEAD", age=30 + offset, employment=20_000.0)

    person = pd.DataFrame(rows)
    count = len(person)
    person.insert(0, "person_id", np.arange(1, count + 1, dtype="int64"))
    person["person_household_id"] = person["person_tax_unit_id"] + 800
    person["person_spm_unit_id"] = person["person_tax_unit_id"] + 400
    person["person_family_id"] = person["person_tax_unit_id"] + 600
    person["person_marital_unit_id"] = np.arange(701, 701 + count, dtype="int64")

    tax_unit_ids = person["person_tax_unit_id"].drop_duplicates().to_numpy()
    spm_unit_ids = person["person_spm_unit_id"].drop_duplicates().to_numpy()
    household_ids = person["person_household_id"].drop_duplicates().to_numpy()
    childcare = np.where(
        spm_unit_ids == 603,
        _DONOR_CHILDCARE,
        0.0,
    )
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": household_ids,
                "state_fips": np.full(len(household_ids), 6),
            }
        ),
        "tax_unit": pd.DataFrame({"tax_unit_id": tax_unit_ids}),
        "spm_unit": pd.DataFrame(
            {
                "spm_unit_id": spm_unit_ids,
                "spm_unit_pre_subsidy_childcare_expenses": childcare,
            }
        ),
        "family": pd.DataFrame(
            {"family_id": person["person_family_id"].drop_duplicates().to_numpy()}
        ),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": person["person_marital_unit_id"].to_numpy()}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.ones(len(household_ids), dtype=np.float64),
                WeightKind.DESIGN,
            )
        },
    )


__all__ = [name for name in globals() if not name.startswith("__")]
