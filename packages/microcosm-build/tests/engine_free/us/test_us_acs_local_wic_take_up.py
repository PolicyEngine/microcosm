"""ACS local WIC take-up (microcosm#1154, the design of microcosm#1156)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.acs_local_wic_take_up import (
    ACS_LOCAL_WIC_TAKE_UP_ISSUE,
    US_WIC_TAKE_UP_COLUMN,
    require_complete_us_wic_take_up,
    with_acs_local_wic_take_up,
)
from microcosm.build.us_runtime.wic_claim import (
    WIC_CLAIM_FNS_SOURCE_URL,
    with_us_wic_claim_input,
)
from test_support.microcosm_build.us_wic_claim import (
    _frame,
    _plausible_rows,
    _replace_person,
)

WIC = US_WIC_TAKE_UP_COLUMN


def _acs_frame():
    return _frame(_plausible_rows())


def test_the_acs_rows_draw_with_the_national_generator() -> None:
    frame = _acs_frame()
    drawn, receipt = with_acs_local_wic_take_up(frame, seed=7, time_period=2024)
    expected = with_us_wic_claim_input(frame, seed=7, time_period=2024)
    person = drawn.table("person")
    assert person[WIC].dtype == bool
    pd.testing.assert_series_equal(person[WIC], expected.table("person")[WIC])
    assert receipt["issue"] == ACS_LOCAL_WIC_TAKE_UP_ISSUE
    assert receipt["design"] == "microcosm#1156"
    assert (receipt["seed"], receipt["source_year"]) == (7, 2024)
    assert receipt["person_rows"] == len(person)
    assert receipt["output_columns"] == [WIC]
    assert receipt["category_rates_source"] == WIC_CLAIM_FNS_SOURCE_URL
    assert set(receipt["category_weighted_take_up_shares"]) == set(
        receipt["category_counts"]
    )
    again, _ = with_acs_local_wic_take_up(frame, seed=7, time_period=2024)
    pd.testing.assert_series_equal(again.table("person")[WIC], person[WIC])


def test_an_all_null_column_from_the_pooled_schema_is_drawn() -> None:
    frame = _acs_frame()
    nulls = _replace_person(frame, frame.table("person").assign(**{WIC: np.nan}))
    drawn, _ = with_acs_local_wic_take_up(nulls, seed=0, time_period=2024)
    require_complete_us_wic_take_up(drawn.table("person"), what="test")


def test_existing_decisions_are_never_redrawn() -> None:
    frame = _acs_frame()
    stored = _replace_person(frame, frame.table("person").assign(**{WIC: True}))
    with pytest.raises(ValueError, match="already carries"):
        with_acs_local_wic_take_up(stored, seed=0, time_period=2024)


@pytest.mark.parametrize(
    "column", ["source_year", "source_household_id", "source_person_id"]
)
def test_the_draw_needs_the_loaders_source_identity(column: str) -> None:
    frame = _acs_frame()
    person = frame.table("person").drop(columns=column)
    with pytest.raises(ValueError, match="source identity"):
        with_acs_local_wic_take_up(
            _replace_person(frame, person), seed=0, time_period=2024
        )


def test_a_partial_source_identity_is_refused() -> None:
    frame = _acs_frame()
    person = frame.table("person").copy()
    person["source_person_id"] = person["source_person_id"].astype(float)
    person.loc[person.index[0], "source_person_id"] = np.nan
    with pytest.raises(ValueError, match="complete source identity"):
        with_acs_local_wic_take_up(
            _replace_person(frame, person), seed=0, time_period=2024
        )


def test_donor_rows_are_refused() -> None:
    frame = _acs_frame()
    person = frame.table("person").copy()
    person.loc[person.index[0], "source_year"] = 2023
    with pytest.raises(ValueError, match="ACS rows only"):
        with_acs_local_wic_take_up(
            _replace_person(frame, person), seed=0, time_period=2024
        )


@pytest.mark.parametrize(
    ("values", "match"),
    [
        (None, "requires person column"),
        ([True, np.nan], "missing on 1 of 2 person rows"),
        (pd.Series([True, False], dtype=object), "must be boolean"),
    ],
)
def test_an_incomplete_column_is_refused(values, match: str) -> None:
    person = pd.DataFrame({"person_id": [1, 2]})
    if values is not None:
        person[WIC] = values
    with pytest.raises(ValueError, match=match):
        require_complete_us_wic_take_up(person, what="test")


def test_a_complete_boolean_column_passes() -> None:
    require_complete_us_wic_take_up(
        pd.DataFrame({"person_id": [1, 2], WIC: [True, False]}), what="test"
    )
