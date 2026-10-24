from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime.frs_education import (
    EDUCQUAL_MAP,
    add_frs_education,
    derive_current_education,
    derive_frs_education,
)
from microcosm.build.uk_runtime.frs_spine import WEEKS_IN_YEAR


def test_current_education_cascade_keeps_dead_post_secondary_shadowed() -> None:
    result = derive_current_education(
        fted=np.array([2, 1, 1, 1, 1, 1]),
        typeed2=np.array([9, 1, 2, 5, 7, 9]),
        age=np.array([20, 4, 10, 16, 19, 20]),
    )

    assert result.tolist() == [
        "NOT_IN_EDUCATION",
        "PRE_PRIMARY",
        "PRIMARY",
        "LOWER_SECONDARY",
        "UPPER_SECONDARY",
        "TERTIARY",
    ]


def test_educqual_map_pins_the_corrected_frs_value_labels() -> None:
    # Signed difference vs the incumbent, whose map was inverted (it read
    # low codes as school-level and 17-21 as degrees). Labels verified
    # against the FRS 2023-24 data dictionary (UKDS SN 9367, DOI
    # 10.5255/UKDA-SN-9367-2, adult table) and corroborated by the raw
    # aggregates: code 1 (Doctorate) is ~1.8% of adults at the highest mean
    # earnings; 18/19 are near-empty baccalaureates; the GCSE band (36-82)
    # carries the mass. Code 87 is undocumented in the dictionary and is
    # deliberately unmapped, falling to the fillna default.
    assert EDUCQUAL_MAP[1] == "TERTIARY"
    assert EDUCQUAL_MAP[3] == "TERTIARY"
    assert EDUCQUAL_MAP[11] == "POST_SECONDARY"
    assert EDUCQUAL_MAP[17] == "UPPER_SECONDARY"
    assert EDUCQUAL_MAP[26] == "POST_SECONDARY"
    assert EDUCQUAL_MAP[36] == "LOWER_SECONDARY"
    assert EDUCQUAL_MAP[82] == "LOWER_SECONDARY"
    assert EDUCQUAL_MAP[85] == "NOT_COMPLETED_PRIMARY"
    assert EDUCQUAL_MAP[86] == "LOWER_SECONDARY"
    assert 87 not in EDUCQUAL_MAP
    assert set(EDUCQUAL_MAP) == set(range(1, 87))


def test_training_qyp_ema_and_benefits_in_own_right() -> None:
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "age": [19, 17, 20],
            "universal_credit_reported": [0, 1, 0],
            "jsa_contrib_reported": [0, 0, 0],
            "jsa_income_reported": [0, 0, 0],
            "esa_contrib_reported": [0, 0, 0],
            "esa_income_reported": [0, 0, 0],
        }
    )
    raw = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "fted": [1, 2, 2],
            "typeed2": [7, 0, 0],
            "educqual": [17, 86, 999],
            # A child-table 17-year-old outside education is a trainee; the
            # adult-table record's TRAINEE answer is not read as approved
            # training (uk-data#561).
            "trainee": [2, 1, 1],
            "_frs_table": ["child", "child", "adult"],
            "slos": [np.nan, np.nan, 2],
            "educleft": [np.nan, np.nan, 18],
            # 2023-24-shaped vintage: no adema pair, eduma/edumaamt instead.
            "eduma": [1, 1, 0],
            "edumaamt": [2.0, -1.0, 5.0],
            "chema": [0, 1, 1],
            "chemaamt": [0.0, 3.0, -1.0],
        }
    )

    result = derive_frs_education(person, raw)

    assert result["is_in_approved_training"].tolist() == [False, True, False]
    assert result["age_started_or_accepted_current_education_or_training"].tolist() == [
        18,
        17,
        1000,
    ]
    assert result[
        "is_before_universal_credit_qualifying_young_person_terminal_date"
    ].tolist() == [True, False, False]
    # fill_with_mean semantics: a participant's sentinel −1 receives the
    # participant mean (2.0 adult / 3.0 child); a non-participant keeps a
    # positive reported amount (no flag-gating of reported values).
    assert result["adult_ema"].tolist() == [
        2.0 * WEEKS_IN_YEAR,
        2.0 * WEEKS_IN_YEAR,
        5.0 * WEEKS_IN_YEAR,
    ]
    assert result["child_ema"].tolist() == [
        0.0,
        3.0 * WEEKS_IN_YEAR,
        3.0 * WEEKS_IN_YEAR,
    ]
    assert result["receives_benefits_in_own_right"].tolist() == [False, True, False]


def test_adult_ema_prefers_the_adema_pair_when_present() -> None:
    person = pd.DataFrame({"person_id": [1, 2], "age": [17, 18]})
    raw = pd.DataFrame(
        {
            "person_id": [1, 2],
            "fted": [2, 2],
            "typeed2": [0, 0],
            "educqual": [3, 3],
            # adema pair present: it must win over a contradictory eduma pair
            # (the incumbent aliases eduma into adema only when adema is
            # absent).
            "trainee": [2, 2],
            "_frs_table": ["child", "adult"],
            "slos": [np.nan, 2],
            "educleft": [np.nan, 18],
            "adema": [1, 1],
            "ademaamt": [4.0, -1.0],
            "eduma": [1, 1],
            "edumaamt": [9.0, 9.0],
            "chema": [0, 0],
            "chemaamt": [0.0, 0.0],
        }
    )

    result = derive_frs_education(person, raw)

    assert result["adult_ema"].tolist() == [
        4.0 * WEEKS_IN_YEAR,
        4.0 * WEEKS_IN_YEAR,
    ]


def test_approved_training_is_trainee_on_child_table_records_only() -> None:
    """TRAINEE = 1 flags a child-table record and never an adult-table one.

    The FRS counts a 16- to 19-year-old as a dependent child only in full-time
    non-advanced education or unwaged government training, so a child-table
    trainee is in approved training. An adult-table record is not a dependant,
    and its answer does not say the training is unwaged (uk-data#561).
    """

    person = pd.DataFrame({"person_id": [1, 2, 3, 4, 5], "age": [16, 19, 18, 17, 40]})
    raw = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "fted": [2, 2, 2, 2, 2],
            "typeed2": [0, 0, 0, 0, 0],
            "educqual": [86, 86, 86, 86, 86],
            # 1 yes, 2 no; a blank and an out-of-range code are not a yes.
            "trainee": [1, 1, 1, np.nan, 1],
            "_frs_table": ["child", "child", "adult", "child", "adult"],
            "slos": [np.nan, np.nan, 2, np.nan, 2],
            "educleft": [np.nan, np.nan, 17, np.nan, 16],
            "chema": [0, 0, 0, 0, 0],
            "chemaamt": [0.0, 0.0, 0.0, 0.0, 0.0],
            "eduma": [0, 0, 0, 0, 0],
            "edumaamt": [0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )

    result = derive_frs_education(person, raw)

    assert result["is_in_approved_training"].tolist() == [
        True,
        True,
        False,
        False,
        False,
    ]
    # The entry age and the age-19 terminal-date flag follow the training flag.
    assert result["age_started_or_accepted_current_education_or_training"].tolist() == [
        16,
        18,
        1000,
        1000,
        1000,
    ]
    assert result[
        "is_before_universal_credit_qualifying_young_person_terminal_date"
    ].tolist() == [False, True, False, False, False]


@pytest.mark.parametrize("column", ["trainee", "_frs_table"])
def test_approved_training_refuses_rows_without_its_columns(column: str) -> None:
    """A missing column refuses; it never reads as nobody in training."""

    person = pd.DataFrame({"person_id": [1], "age": [17]})
    raw = pd.DataFrame(
        {
            "person_id": [1],
            "fted": [2],
            "typeed2": [0],
            "educqual": [86],
            "trainee": [1],
            "_frs_table": ["child"],
            "slos": [np.nan],
            "educleft": [np.nan],
        }
    )

    with pytest.raises(KeyError, match="approved training cannot be derived"):
        derive_frs_education(person, raw.drop(columns=[column]))


@pytest.mark.parametrize("table", ["adult", "child"])
def test_the_stage_refuses_a_tab_without_trainee(tmp_path, table: str) -> None:
    """Each tab is checked on its own, before the two are concatenated.

    After the concat a tab without TRAINEE would read as blanks beside the
    other tab's answers, which is a silent "nobody on this table trains".
    """

    artifacts = []
    for name in ("adult", "child"):
        columns = ["SERNUM", "BENUNIT", "PERSON", "TRAINEE"]
        if name == table:
            columns.remove("TRAINEE")
        row = ["1", "1", "1" if name == "adult" else "2", "2"][: len(columns)]
        path = tmp_path / f"{name}.tab"
        path.write_text("\t".join(columns) + "\n" + "\t".join(row) + "\n")
        artifacts.append(
            {
                "table": name,
                "locator": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            }
        )

    with pytest.raises(KeyError, match=f"The FRS {table} tab has no TRAINEE column"):
        add_frs_education(None, tmp_path, stage=SimpleNamespace(artifacts=artifacts))


def test_student_loan_answers_are_read_from_adult_records_only() -> None:
    """SLOS = 1 reports an outstanding loan and EDUCLEFT the age full-time
    education ended, on adult-table records; a child record, a no, a blank and
    an age outside 10 to 90 read as no loan and 0 (microcosm#868)."""

    person = pd.DataFrame({"person_id": [1, 2, 3, 4, 5], "age": [30, 25, 15, 40, 50]})
    raw = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "fted": [2, 2, 1, 2, 2],
            "typeed2": [0, 0, 5, 0, 0],
            "educqual": [3, 3, 86, 86, 86],
            "trainee": [2, 2, 2, 2, 2],
            "_frs_table": ["adult", "adult", "child", "adult", "adult"],
            "slos": [1, 2, 1, np.nan, 1],
            "educleft": [21, 22, 15, 97, -1],
        }
    )

    result = derive_frs_education(person, raw)

    assert result["reports_outstanding_student_loan"].tolist() == [
        True,
        False,
        False,
        False,
        True,
    ]
    assert result["age_left_full_time_education"].tolist() == [21, 22, 0, 0, 0]
    assert result["age_left_full_time_education"].dtype == np.int64


@pytest.mark.parametrize("column", ["slos", "educleft"])
def test_student_loan_answers_refuse_rows_without_their_columns(column: str) -> None:
    """A missing column refuses rather than read nobody as a borrower."""

    person = pd.DataFrame({"person_id": [1], "age": [30]})
    raw = pd.DataFrame(
        {
            "person_id": [1],
            "fted": [2],
            "typeed2": [0],
            "educqual": [3],
            "trainee": [2],
            "_frs_table": ["adult"],
            "slos": [1],
            "educleft": [21],
        }
    )

    with pytest.raises(KeyError, match="student-loan borrowers and course dates"):
        derive_frs_education(person, raw.drop(columns=[column]))
