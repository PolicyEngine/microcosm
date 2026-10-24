from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from microcosm.build.uk_runtime.frs_employment import (
    CHILD_EMPLOYMENT_STATUS,
    FRS_EMPSTATI_EMPLOYMENT_STATUS,
    derive_employment_status_from_frs,
    derive_frs_employment,
)

# EMPSTATI ("Adult - Employment Status - ILO definition") value labels in the
# UKDS FRS 2024-25 data dictionary (SN 9563, adult table), with the status each
# label means. The incumbent's uk-data#526 pins the same table.
DATA_DICTIONARY = {
    1: ("Full-time Employee", "FT_EMPLOYED"),
    2: ("Part-time Employee", "PT_EMPLOYED"),
    3: ("Full-time Self-Employed", "FT_SELF_EMPLOYED"),
    4: ("Part-time Self-Employed", "PT_SELF_EMPLOYED"),
    5: ("Unemployed", "UNEMPLOYED"),
    6: ("Retired", "RETIRED"),
    7: ("Student", "STUDENT"),
    8: ("Looking after family/home", "CARER"),
    9: ("Permanently sick/disabled", "LONG_TERM_DISABLED"),
    10: ("Temporarily sick/injured", "SHORT_TERM_DISABLED"),
    11: ("Other Inactive", "OTHER_INACTIVE"),
}
ADULT_CODES = sorted(DATA_DICTIONARY)

adult_rows = st.tuples(st.just(True), st.sampled_from(ADULT_CODES))
# Child-table people have no EMPSTATI (NaN once aligned to adult.tab); the
# code must not matter for them.
child_rows = st.tuples(
    st.just(False),
    st.one_of(st.just(0), st.just(np.nan), st.integers(-9, 99)),
)
people = st.lists(st.one_of(adult_rows, child_rows), max_size=60)
unknown_adult_codes = st.one_of(
    st.just(np.nan),
    st.integers(-99, 0),
    st.integers(12, 999),
    st.floats(0.5, 11.5).filter(lambda x: not float(x).is_integer()),
)


def _derive(rows):
    is_adult = [adult for adult, _ in rows]
    codes = [code for _, code in rows]
    return derive_employment_status_from_frs(codes, is_adult)


def _expected(adult, code):
    return DATA_DICTIONARY[code][1] if adult else "CHILD"


def test_employment_maps_status_sector_and_sic() -> None:
    # Person 6 is not in adult.tab (a child-table person), so every adult.tab
    # column aligns to NaN for them.
    person = pd.DataFrame({"person_id": [1, 2, 3, 4, 5, 6]})
    adult = pd.DataFrame(
        {
            "person_id": [5, 4, 3, 2, 1],
            "empstati": [8, 1.0, 11, 10, 9],
            "mjobsect": [1, np.nan, 2, 1, 0],
            "sic": [20, 7, np.nan, 84.9, -5],
        }
    )

    result = derive_frs_employment(person, adult)

    assert result["employment_status"].tolist() == [
        "LONG_TERM_DISABLED",
        "SHORT_TERM_DISABLED",
        "OTHER_INACTIVE",
        "FT_EMPLOYED",
        "CARER",
        "CHILD",
    ]
    assert result["employment_sector"].tolist() == [
        "NOT_EMPLOYED",
        "PRIVATE",
        "PUBLIC",
        "NOT_EMPLOYED",
        "PRIVATE",
        "NOT_EMPLOYED",
    ]
    assert result["sic_industry_division"].tolist() == [0, 84, 0, 7, 20, 0]


def test_other_inactive_is_not_long_term_disabled() -> None:
    result = derive_employment_status_from_frs([9, 11], [True, True])
    assert result.tolist() == ["LONG_TERM_DISABLED", "OTHER_INACTIVE"]


def test_only_code_11_moves_from_the_truncated_map() -> None:
    # The map this replaces zipped codes 0-11 with 11 statuses and sent the
    # unmatched code 11 to LONG_TERM_DISABLED; codes 1-10 keep their status.
    truncated = {
        code: status for code, (_, status) in DATA_DICTIONARY.items() if code != 11
    } | {11: "LONG_TERM_DISABLED"}
    moved = {
        code
        for code in ADULT_CODES
        if FRS_EMPSTATI_EMPLOYMENT_STATUS[code] != truncated[code]
    }
    assert moved == {11}


@pytest.mark.parametrize("code", range(12))
def test_every_code_from_0_to_11(code) -> None:
    assert derive_employment_status_from_frs([code], [False]).tolist() == ["CHILD"]
    if code == 0:
        with pytest.raises(ValueError, match="EMPSTATI"):
            derive_employment_status_from_frs([code], [True])
    else:
        label, status = DATA_DICTIONARY[code]
        assert derive_employment_status_from_frs([code], [True]).tolist() == [status], (
            label
        )


def test_code_table_is_the_data_dictionary() -> None:
    assert dict(FRS_EMPSTATI_EMPLOYMENT_STATUS) == {
        code: status for code, (_, status) in DATA_DICTIONARY.items()
    }


def test_code_table_is_immutable() -> None:
    with pytest.raises(TypeError):
        FRS_EMPSTATI_EMPLOYMENT_STATUS[12] = "OTHER_INACTIVE"  # type: ignore[index]


def test_adult_codes_and_child_rows_cover_each_status_once() -> None:
    statuses = [*FRS_EMPSTATI_EMPLOYMENT_STATUS.values(), CHILD_EMPLOYMENT_STATUS]
    assert CHILD_EMPLOYMENT_STATUS == "CHILD"
    assert len(set(statuses)) == len(statuses) == 12


@pytest.mark.parametrize("code", [0, -1, 12, 11.5, np.nan, "x"])
def test_unknown_adult_code_fails_the_build(code) -> None:
    with pytest.raises(ValueError, match="EMPSTATI"):
        derive_employment_status_from_frs([1, code], [True, True])


@pytest.mark.parametrize("code", [0, 12, np.nan])
def test_adult_record_with_unknown_or_blank_code_fails_the_stage(code) -> None:
    person = pd.DataFrame({"person_id": [1, 2]})
    adult = pd.DataFrame(
        {"person_id": [1, 2], "empstati": [1, code], "mjobsect": [1, 1], "sic": [1, 1]}
    )

    with pytest.raises(ValueError, match="EMPSTATI"):
        derive_frs_employment(person, adult)


def _digits_outside_code_list(message: str) -> list[str]:
    return re.findall(r"\d", re.sub(r"\[.*?\]", "", message))


def test_unknown_code_message_names_codes_and_prints_no_count() -> None:
    # Adults are not survey households, so even a count of 10 or more could
    # describe fewer than 10 households: the refusal never prints one.
    with pytest.raises(ValueError) as few:
        derive_employment_status_from_frs([12, 12, np.nan, 11.5], [True] * 4)
    message = str(few.value)
    assert "['11.5', '12', 'blank or non-numeric']" in message
    assert _digits_outside_code_list(message) == []

    with pytest.raises(ValueError) as many:
        derive_employment_status_from_frs([13] * 25, [True] * 25)
    assert "['13']" in str(many.value)
    assert _digits_outside_code_list(str(many.value)) == []


def test_mismatched_lengths_fail() -> None:
    with pytest.raises(ValueError, match="same length"):
        derive_employment_status_from_frs([1, 2], [True])


@given(people)
def test_each_row_maps_on_its_own(rows) -> None:
    result = _derive(rows)
    assert result.tolist() == [_expected(adult, code) for adult, code in rows]


@given(people, st.randoms(use_true_random=False))
def test_mapping_commutes_with_row_order(rows, rng) -> None:
    order = list(range(len(rows)))
    rng.shuffle(order)
    assert _derive([rows[i] for i in order]).tolist() == [
        _derive(rows)[i] for i in order
    ]


@given(people, unknown_adult_codes, st.integers(0, 60))
def test_any_unknown_adult_code_fails_the_build(rows, bad_code, position) -> None:
    rows = list(rows)
    rows.insert(min(position, len(rows)), (True, bad_code))
    with pytest.raises(ValueError, match="EMPSTATI"):
        _derive(rows)


@given(st.lists(unknown_adult_codes, min_size=1, max_size=30), people)
def test_unknown_code_message_never_prints_a_count(bad_codes, rows) -> None:
    rows = [*rows, *((True, code) for code in bad_codes)]
    with pytest.raises(ValueError) as error:
        _derive(rows)
    assert _digits_outside_code_list(str(error.value)) == []


@given(people)
def test_stage_marks_adults_by_adult_tab_membership(rows) -> None:
    # The stage aligns adult.tab by person_id: people missing from it get NaN
    # codes and are CHILD; everyone in it is mapped strictly.
    person = pd.DataFrame({"person_id": np.arange(len(rows)) + 101})
    adult = pd.DataFrame(
        {
            "person_id": [
                pid
                for pid, (is_adult, _) in zip(person["person_id"], rows, strict=True)
                if is_adult
            ],
            "empstati": [code for is_adult, code in rows if is_adult],
        }
    ).assign(mjobsect=0, sic=0)

    result = derive_frs_employment(person, adult.iloc[::-1])

    assert result["employment_status"].tolist() == [
        _expected(is_adult, code) for is_adult, code in rows
    ]


@pytest.mark.parametrize("missing", ["empstati", "mjobsect", "sic"])
def test_employment_missing_fail_loud_columns_raise(missing: str) -> None:
    person = pd.DataFrame({"person_id": [1]})
    adult = pd.DataFrame(
        {"person_id": [1], "empstati": [1], "mjobsect": [1], "sic": [10]}
    ).drop(columns=[missing])

    with pytest.raises(KeyError):
        derive_frs_employment(person, adult)
