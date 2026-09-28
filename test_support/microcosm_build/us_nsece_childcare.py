# ruff: noqa: F401
"""Synthetic source-code and Frame/export contracts; no NSECE microdata in CI."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import childcare_attendance_stage as stage
from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
)
from microcosm.build.us_runtime.childcare_attendance_receipt import (
    assert_bound_childcare_attendance,
    restore_native_childcare_receipt,
)
from microcosm.build.us_runtime.childcare_population import (
    harmonize_asec_childcare_predictors,
)
from microcosm.build.us_runtime.nsece_childcare import (
    NSECE_CALENDAR_BLOCKS,
    NSECE_CHILD_INDICES,
    NSECE_PROVIDER_INDICES,
    NSECEChildcareSource,
    assert_childcare_attendance_exportable,
    derive_nsece_childcare,
    load_nsece_childcare,
    nsece_childcare_calendar_columns,
    nsece_childcare_household_columns,
    nsece_childcare_validation_report,
    with_us_nsece_childcare_attendance,
)
from microcosm.build.us_runtime.release_input_coverage import (
    ReleaseInputColumn,
    ReleaseInputCoverageManifest,
    us_release_input_coverage_gate,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from test_support.paths import paths_for

MONTH, DAYS, HOURS = US_CHILDCARE_ATTENDANCE_COLUMNS


REPOSITORY_ROOT = paths_for("microcosm-build").repository


def _raw(n=1):
    household = pd.DataFrame(
        -9.0, index=range(n), columns=nsece_childcare_household_columns()
    )
    calendar = pd.DataFrame(
        -9.0, index=range(n), columns=nsece_childcare_calendar_columns()
    )
    for table in (household, calendar):
        table["HH4_METH_CASEID"] = np.arange(1, n + 1)
    household["HH4_METH_QUEXVERSION"] = 1
    household["HH4_REGION"] = 1
    household["HH4_PARWORK_STATUS"] = 2
    household["HH4_RPARENT"] = 1
    household["HH4_METH_WEIGHT"] = 100.0
    household["HH4_ECON_INCOME_ANNUAL"] = 40_000
    for child in NSECE_CHILD_INDICES:
        household[f"HHC4_METH_WEIGHT_{child}"] = np.nan
        household[f"HH4_MISSING_STATUS_CC_{child}"] = 0
    household["HHC4_AGE_AT_USAGE_1"] = 36
    household["HHC4_METH_WEIGHT_1"] = 100.0
    household["HH4_MISSING_STATUS_CC_1"] = 2
    for provider in NSECE_PROVIDER_INDICES:
        household[f"HH4_TYPEOFCARE_AGG_1_{provider}"] = -8
    household["HH4_TYPEOFCARE_AGG_1_1"] = 4
    for block in range(1, NSECE_CALENDAR_BLOCKS + 1):
        calendar[f"HH4_CHCAL_R_1_{block}"] = 0
    return household, calendar


def _care(calendar, *, row=0, day=0, hours=4, provider=1):
    start = day * 96 + 9 * 4 + 1
    columns = [f"HH4_CHCAL_R_1_{b}" for b in range(start, start + hours * 4)]
    calendar.loc[row, columns] = provider


def _frame(*, parent_observed=True):
    person = pd.DataFrame(
        {"person_id": [1, 2], "person_source_id": ["adult", "child"], "age": [30, 3]}
    )
    tables = {}
    for entity in US_SCHEMA.group_entities:
        person[f"person_{entity}_id"] = 1
        tables[entity] = pd.DataFrame({f"{entity}_id": [1]})
    if parent_observed:
        for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
            person[column] = [0.0, np.nan]
    tables["person"] = person
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([100.0]), WeightKind.DESIGN)},
        metadata={"existing_receipt": "preserved"},
    )


def _source():
    hh, cal = _raw()
    for day in range(5):
        _care(cal, day=day, hours=8)
    return derive_nsece_childcare(hh, cal)


def _asec_frame():
    frame = _frame()
    tables = {e: frame.table(e).copy() for e in frame.entities}
    tables["person"]["A_LINENO"] = [1, 2]
    tables["person"]["PEPAR1"] = [-1, 1]
    tables["person"]["PEPAR2"] = [-1, -1]
    tables["person"]["hours_worked_last_week"] = [40, 0]
    tables["person"]["PTOTVAL"] = [40000, 0]
    tables["person"]["source_year"] = 2023
    tables["household"]["state_fips"] = 25
    return Frame(
        tables,
        frame.schema,
        {"household": frame.weights_for("household")},
        metadata=frame.metadata,
    )


def _replace(frame, *, people=None, metadata=None):
    tables = {e: frame.table(e).copy() for e in frame.entities}
    if people is not None:
        tables["person"] = people
    return Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        metadata=frame.metadata if metadata is None else metadata,
    )


def _candidate():
    return with_us_nsece_childcare_attendance(
        _frame(), _source(), seed=915, match_columns=("age",)
    )


def _sibling_source():

    rows = []
    for household in range(100):
        size = 1 if household % 2 else 3
        for child in range(size):
            days = 5.0 if size == 1 else 0.0
            rows.append(
                {
                    "donor_id": f"h{household}:c{child}",
                    "source_household_id": f"h{household}",
                    "age": 3,
                    "childcare_household_size": size,
                    "region": 1,
                    "parent_work_status": 2,
                    "income_band": 1,
                    "attendance_status": "complete",
                    "household_weight": 1.0,
                    "child_weight": 1.0,
                    MONTH: 22.0 if days else 0.0,
                    DAYS: days,
                    HOURS: 8.0 if days else 0.0,
                }
            )
    children = pd.DataFrame(rows)
    return NSECEChildcareSource(
        children, Weights(np.ones(len(children)), WeightKind.DESIGN), {}
    )


def _paired_sensitivity_fixture():
    source = _source()
    source.children["attendance_status"] = "summary_bridge"
    source.children["regular_hours_per_week"] = 12.0
    source.children["irregular_hours_per_week"] = 1.0
    source.children[HOURS] = 13 / source.children[DAYS]
    source.children["ece_hours_per_week"] = 13.0
    child = source.children.iloc[0]
    people = pd.DataFrame(
        {"age": [child.age], **{c: [child[c]] for c in (MONTH, DAYS, HOURS)}}
    )
    for c in (MONTH, DAYS, HOURS):
        people[f"{c}_source"] = f"donor:{child.donor_id}"
    return people, source


__all__ = [name for name in globals() if not name.startswith("__")]
