"""Behavioral contracts for the opt-in childcare attendance donor primitive.

All donors below are synthetic test records, not survey estimates.
"""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from microcosm.build.us_runtime.childcare_attendance import (
    US_CHILDCARE_ATTENDANCE_COLUMNS,
    impute_us_childcare_attendance,
)
from microcosm.frame import WeightKind, Weights

MONTH, DAYS, HOURS = US_CHILDCARE_ATTENDANCE_COLUMNS


def _donors():
    return pd.DataFrame(
        {
            "donor_id": ["none", "part", "full"],
            "age": [3, 3, 3],
            MONTH: [0, 13, 22],
            DAYS: [0.0, 3.0, 5.0],
            HOURS: [0.0, 4.0, 8.0],
        }
    )


def _people(count=1):
    return pd.DataFrame(
        {"person_source_id": [f"c:{i}" for i in range(count)], "age": [3] * count}
    )


def _impute(person, donor=None, weights=None, **kwargs):
    donor = _donors() if donor is None else donor
    return impute_us_childcare_attendance(
        person,
        donor,
        donor_weights=Weights(
            np.ones(len(donor)) if weights is None else np.asarray(weights),
            WeightKind.DESIGN,
        ),
        match_columns=kwargs.pop("match_columns", ("age",)),
        seed=kwargs.pop("seed", 42),
        **kwargs,
    )


def test_joint_draw_respects_survey_weights_and_includes_nonparticipants():
    result = _impute(_people(1000), weights=[1, 0, 4])
    # Independent population behavior: 80% care, with sampling tolerance.
    assert 0.75 < (result[DAYS] > 0).mean() < 0.85
    assert set(result[HOURS]) == {0, 8}
    assert set(zip(result[MONTH], result[DAYS], result[HOURS], strict=True)) == {
        (0, 0, 0),
        (22, 5, 8),
    }
    assert str(result[MONTH].dtype) == "Int64"


def test_observed_zero_and_partial_positive_are_constraints_not_missing():
    person = _people(2).assign(**{DAYS: [0.0, 3.0]})
    original = person.copy(deep=True)
    result = _impute(person)
    assert result[MONTH].tolist() == [0, 13]
    assert result[HOURS].tolist() == [0, 4]
    assert result[f"{DAYS}_source"].tolist() == ["observed", "observed"]
    assert_frame_equal(person, original)
    assert_frame_equal(result, _impute(result))


def test_expense_and_parental_work_are_not_participation_flags():
    people = _people(2).assign(
        spm_unit_pre_subsidy_childcare_expenses=[0, 5000],
        parent_hours_worked=[0, 40],
    )
    cared_for = _impute(people, weights=[0, 0, 1])
    assert (cared_for[DAYS] == 5).all()
    nonparticipants = _impute(people, weights=[1, 0, 0])
    assert (nonparticipants[DAYS] == 0).all()


def test_age_domain_and_person_level_values_leave_adults_and_older_children_null():
    people = _people(4).assign(age=[3, 35, 13, 16])
    people.loc[3, [MONTH, DAYS, HOURS]] = [13, 3, 4]
    result = _impute(people, weights=[0, 1, 0])
    assert result.loc[0, DAYS] == 3
    assert result.loc[[1, 2], list(US_CHILDCARE_ATTENDANCE_COLUMNS)].isna().all().all()
    assert result.loc[3, DAYS] == 3  # preserve observed older-child care


def test_clone_fanout_and_order_and_chunk_invariance():
    people = _people(8)
    clones = people.iloc[:3].copy()
    people = pd.concat([people, clones], ignore_index=True)
    people["person_id"] = np.arange(len(people))
    donor = _donors()
    expected = _impute(people, donor, weights=[1, 2, 3])
    shuffled = _impute(
        people.sample(frac=1, random_state=12),
        donor.iloc[::-1],
        weights=[3, 2, 1],
    ).sort_index()
    assert_frame_equal(expected, shuffled)
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        assert (expected.groupby("person_source_id")[column].nunique() == 1).all()
    # Keep all clones of a source person in the same chunk.
    chunks = [
        people[people.person_source_id < "c:4"],
        people[people.person_source_id >= "c:4"],
    ]
    actual = pd.concat(
        [_impute(chunk, weights=[1, 2, 3]) for chunk in chunks]
    ).sort_index()
    assert_frame_equal(expected, actual)


def test_observations_on_one_clone_constrain_all_clones():
    people = pd.concat([_people(), _people()], ignore_index=True)
    people.loc[1, DAYS] = 3
    result = _impute(people)
    assert result[DAYS].tolist() == [3, 3]
    assert result[HOURS].tolist() == [4, 4]
    assert result.loc[1, f"{DAYS}_source"] == "observed"


def test_complete_clone_observations_need_no_donor_support():
    people = pd.concat([_people(), _people()], ignore_index=True)
    people.loc[1, [MONTH, DAYS, HOURS]] = [17, 4, 6]
    result = _impute(people)
    assert result[MONTH].tolist() == [17, 17]
    assert result.loc[0, f"{MONTH}_source"] == "source_person:c:0"


@pytest.mark.parametrize("column,value", [(DAYS, 5), ("age", 4)])
def test_conflicting_clone_records_raise(column, value):
    people = pd.concat([_people(), _people()], ignore_index=True).assign(**{DAYS: 3})
    people.loc[1, column] = value
    with pytest.raises(ValueError, match="clone .* disagree"):
        _impute(people)


def test_no_compatible_support_raises_without_overwriting_observed_values():
    person = _people().assign(**{DAYS: 4})
    with pytest.raises(ValueError, match="No compatible positive-weight"):
        _impute(person)
    assert person[DAYS].tolist() == [4]


def test_exact_covariate_matching_does_not_relax_to_another_group():
    people = _people().assign(parent_activity="working")
    donor = _donors().assign(parent_activity=["not_working", "working", "not_working"])
    result = _impute(people, donor, match_columns=("age", "parent_activity"))
    assert result[DAYS].tolist() == [3]
    with pytest.raises(ValueError, match="No compatible positive-weight"):
        _impute(
            people, donor, weights=[1, 0, 1], match_columns=("age", "parent_activity")
        )


@pytest.mark.parametrize(
    "column,value,message",
    [
        (DAYS, -1, "nonnegative"),
        (DAYS, 8, "calendar bounds"),
        (HOURS, 25, "calendar bounds"),
        (HOURS, np.inf, "finite"),
        (MONTH, 13.5, "integral"),
        (DAYS, np.nan, "complete"),
        (HOURS, 0, "nonattendance"),
        ("age", 13, "aged 0–12"),
        ("age", np.nan, "whole-year ages"),
    ],
)
def test_invalid_donor_records_raise(column, value, message):
    donor = _donors()
    donor[column] = donor[column].astype(float)
    donor.loc[1, column] = value
    with pytest.raises(ValueError, match=message):
        _impute(_people(), donor)


def test_missing_matching_fields_and_duplicate_donor_ids_raise():
    with pytest.raises(ValueError, match="matching fields must be complete"):
        _impute(
            _people().assign(region=pd.NA),
            _donors().assign(region="NE"),
            match_columns=("age", "region"),
        )
    donor = _donors()
    donor.loc[1, "donor_id"] = "none"
    with pytest.raises(ValueError, match="donors cannot contain clones"):
        _impute(_people(), donor)


def test_weights_must_be_typed_and_aligned():
    with pytest.raises(ValueError, match="align"):
        _impute(_people(), weights=[1, 1])
    with pytest.raises(TypeError, match="typed survey Weights"):
        impute_us_childcare_attendance(
            _people(),
            _donors(),
            donor_weights=np.ones(3),
            match_columns=("age",),
            seed=1,
        )


@pytest.mark.requires_us
def test_engine_accepts_child_inputs_and_derives_weekly_hours():
    from policyengine_us import Simulation

    result = _impute(_people(), weights=[0, 0, 1])
    year = 2026
    child = {"age": {year: 3}}
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        value = result.loc[0, column]
        child[column] = {year: int(value) if column == MONTH else float(value)}
    simulation = Simulation(
        situation={
            "people": {"parent": {"age": {year: 30}}, "child": child},
            "households": {"household": {"members": ["parent", "child"]}},
        }
    )
    assert simulation.calculate("childcare_hours_per_week", year).tolist() == [0, 40]
    assert simulation.calculate(MONTH, year).tolist() == [0, 22]
    # No mutation of the engine defaults by the dataset preparation function.
    for column in US_CHILDCARE_ATTENDANCE_COLUMNS:
        assert simulation.tax_benefit_system.variables[column].default_value == 0
