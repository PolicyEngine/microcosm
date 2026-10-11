"""ACS participation uses the existing category generator, never default fill."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime.wic_claim import (
    US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS,
    require_complete_us_wic_claim_input,
    with_acs_wic_claim_input,
    with_us_wic_claim_input,
)
from test_support.microcosm_build.us_wic_claim import _frame, _replace_person

OUTPUT = "takes_up_wic_if_eligible"


def test_acs_generation_matches_existing_category_rates_and_is_deterministic():
    frame = _frame(
        [{"age": 0}, {"age": 3}, {"is_female": True, "is_pregnant": True}, {}] * 32
    )
    before = frame.person.copy(deep=True)
    result = with_acs_wic_claim_input(frame, seed=19, time_period=2024)
    expected = with_us_wic_claim_input(frame, seed=19, time_period=2024)
    pd.testing.assert_series_equal(result.person[OUTPUT], expected.person[OUTPUT])
    assert result.person[OUTPUT].dtype == bool
    assert result.person[OUTPUT].nunique() == 2
    assert not result.person.loc[
        result.person.age.eq(30) & ~result.person.is_pregnant, OUTPUT
    ].any()
    pd.testing.assert_frame_equal(frame.person, before)
    assert with_acs_wic_claim_input(result, seed=19, time_period=2024) is result
    reordered = _replace_person(frame, frame.person.iloc[::-1].reset_index(drop=True))
    actual = with_acs_wic_claim_input(reordered, seed=19, time_period=2024)
    pd.testing.assert_series_equal(
        actual.person.set_index("person_id")[OUTPUT].sort_index(),
        result.person.set_index("person_id")[OUTPUT].sort_index(),
    )


@pytest.mark.parametrize("column", US_WIC_CLAIM_REQUIRED_SOURCE_COLUMNS[:-1])
def test_missing_demographic_inputs_fail_before_acs_wic_generation(column):
    frame = _frame([{}])
    missing = _replace_person(frame, frame.person.drop(columns=column))
    with pytest.raises(SourceRuntimeError, match="source column"):
        with_acs_wic_claim_input(missing, seed=0, time_period=2024)


@pytest.mark.parametrize(
    "column", ["source_year", "source_household_id", "source_person_id"]
)
def test_missing_acs_identity_is_not_replaced_with_generated_person_ids(column):
    frame = _frame([{}])
    with pytest.raises(ValueError, match="source identity"):
        with_acs_wic_claim_input(
            _replace_person(frame, frame.person.drop(columns=column)),
            seed=0,
            time_period=2024,
        )
    with pytest.raises(ValueError, match="complete source identity"):
        with_acs_wic_claim_input(
            _replace_person(frame, frame.person.assign(**{column: [None]})),
            seed=0,
            time_period=2024,
        )


@pytest.mark.parametrize(
    "values", [[None], [np.nan], [pd.NA], ["False"], [1], [2], [np.inf]]
)
def test_incomplete_or_non_boolean_participation_is_refused(values):
    with pytest.raises(ValueError, match="WIC participation"):
        require_complete_us_wic_claim_input(pd.DataFrame({OUTPUT: values}))


def test_absent_participation_is_refused_and_complete_decisions_are_unchanged():
    with pytest.raises(ValueError, match="requires person column"):
        require_complete_us_wic_claim_input(pd.DataFrame({"person_id": [1]}))
    person = pd.DataFrame({OUTPUT: [False, True]})
    before = person.copy(deep=True)
    require_complete_us_wic_claim_input(person)
    pd.testing.assert_frame_equal(person, before)
