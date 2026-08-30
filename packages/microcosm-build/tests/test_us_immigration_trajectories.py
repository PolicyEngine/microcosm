"""Feasible immigration support, status transitions, and retained event clocks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.immigration import (
    US_IMMIGRATION_FACT_COLUMNS,
    derive_us_immigration_facts,
    us_immigration_composition_gate,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _acs_people(
    *,
    count: int = 512,
    country: int = 303,
    entry_year: int = 2005,
    age: int = 30,
    lawful_evidence: bool = False,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.arange(1, count + 1),
            "CIT": 5,
            "POBP": country,
            "YOEP": entry_year,
            "age": age,
            "SSIP": 1.0 if lawful_evidence else 0.0,
        }
    )


def _derive(person: pd.DataFrame, *, year: int = 2024) -> pd.DataFrame:
    return derive_us_immigration_facts(person, seed=19, time_period=year)


def _residual_status_support(output: pd.DataFrame) -> pd.Series:
    # The lossy PE enum omits some humanitarian categories, so UNDOCUMENTED
    # there is not proof that the neutral facts retain a residual branch.
    assigned_status_columns = [
        "immigration_is_lawful_permanent_resident",
        "immigration_was_admitted_as_refugee",
        "immigration_has_been_granted_asylum",
        "immigration_is_currently_paroled",
        "immigration_has_deportation_or_removal_withheld",
        "immigration_was_granted_conditional_entry",
        "immigration_is_current_cuban_haitian_entrant",
        "immigration_is_current_cofa_resident",
        "immigration_is_current_tps",
        "immigration_is_current_daca",
        "immigration_is_amerasian_immigrant",
        "immigration_is_battered_alien_or_qualifying_relative",
        "immigration_is_severe_trafficking_victim_or_qualifying_relative",
        "immigration_has_other_lawful_status",
    ]
    return output["immigration_is_alien"] & ~output[assigned_status_columns].any(axis=1)


def test_recent_ordinary_immigrants_have_lpr_and_non_lpr_support() -> None:
    # Citizenship/origin/entry alone do not rule out a recent immigrant visa.
    output = _derive(_acs_people(country=301, entry_year=2023))
    lpr = output["immigration_is_lawful_permanent_resident"]

    assert lpr.any()
    assert (~lpr).any()
    assert not output["immigration_has_lawful_status_evidence"].any()


@pytest.mark.parametrize(
    "column,value", [("CAID", 1), ("IHSFLG", 1), ("SPM_CAPHOUSESUB", 1000)]
)
def test_suggestive_proxy_receipt_preserves_lawful_and_residual_support(
    column: str, value: int
) -> None:
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 1025),
            "PRCITSHP": 5,
            "PENATVTY": 301,
            "PEINUSYR": 24,
            "A_AGE": 40,
            column: value,
        }
    )
    observed = _derive(person)

    assert observed["immigration_has_lawful_status_evidence"].all()
    assert observed["immigration_has_other_lawful_status"].any()
    assert _residual_status_support(observed).any()

    person[column] = 0
    absent = _derive(person)
    # A survey benefit/work proxy is not an individual legal finding. It
    # must not determine any status or history draw in either direction.
    status_columns = [
        column
        for column in US_IMMIGRATION_FACT_COLUMNS
        if column != "immigration_has_lawful_status_evidence" and "ssn" not in column
    ]
    pd.testing.assert_frame_equal(observed[status_columns], absent[status_columns])


@pytest.mark.parametrize("entry_year", [1977, 1981])
def test_pre_1982_entry_is_not_lawful_status_evidence(entry_year: int) -> None:
    output = _derive(_acs_people(country=301, entry_year=entry_year, age=70))

    assert not output["immigration_has_lawful_status_evidence"].any()
    assert output["immigration_has_other_lawful_status"].any()
    assert _residual_status_support(output).any()


@pytest.mark.parametrize(
    "country,entry_year,age,history",
    [
        (301, 1977, 70, "immigration_was_granted_conditional_entry"),
        (247, 1995, 55, "immigration_is_amerasian_immigrant"),
        (200, 2015, 40, "immigration_has_iraqi_or_afghan_special_immigrant_history"),
        (301, 2015, 40, "immigration_is_battered_alien_or_qualifying_relative"),
        (
            301,
            2015,
            40,
            "immigration_is_severe_trafficking_victim_or_qualifying_relative",
        ),
    ],
)
def test_humanitarian_history_support_does_not_require_a_positive_survey_proxy(
    country: int, entry_year: int, age: int, history: str
) -> None:
    person = _acs_people(country=country, entry_year=entry_year, age=age)
    absent = _derive(person)
    person["SSIP"] = 1000
    observed = _derive(person)

    assert not absent["immigration_has_lawful_status_evidence"].any()
    assert observed["immigration_has_lawful_status_evidence"].all()
    assert absent[history].any()
    assert (~absent[history]).any()
    pd.testing.assert_series_equal(absent[history], observed[history])


def test_amerasian_support_has_no_empirical_2000_arrival_cutoff() -> None:
    # Pub. L. 100-202 section 584(a)(1)(B), as amended, permits departure
    # from Vietnam on/after March 22, 1988; it sets no 2000 upper cutoff.
    # https://www.govinfo.gov/content/pkg/USCODE-2024-title8/html/USCODE-2024-title8-chap12-subchapI-sec1101.htm
    output = _derive(_acs_people(country=247, entry_year=2015, age=55))

    assert output["immigration_is_amerasian_immigrant"].any()
    assert (~output["immigration_is_amerasian_immigrant"]).any()


def test_proxy_independent_status_support_preserves_clone_and_weight_invariance() -> (
    None
):
    person = _acs_people(country=301, entry_year=2015)
    person["SSIP"] = np.where(np.arange(len(person)) % 2, 1000, 0)
    person["source_year"] = 2024
    person["source_household_id"] = person["person_id"]
    person["source_person_id"] = 1
    person["person_weight"] = 1.0
    expected = _derive(person)
    assert expected["immigration_has_other_lawful_status"].any()
    assert _residual_status_support(expected).any()

    clones = pd.concat([person, person], ignore_index=True)
    clones["person_id"] = np.arange(1, len(clones) + 1)
    clones["person_weight"] = np.geomspace(1e-9, 1e12, len(clones))
    actual = _derive(clones)
    columns = list(US_IMMIGRATION_FACT_COLUMNS)
    for start in (0, len(person)):
        pd.testing.assert_frame_equal(
            actual.iloc[start : start + len(person)][columns].reset_index(drop=True),
            expected[columns].reset_index(drop=True),
        )


@pytest.mark.parametrize("country", [301, 327, 332, 140])
def test_refugee_asylum_support_does_not_use_top_country_lists(country: int) -> None:
    # 8 USC 1157/1158 do not turn empirical leading-origin flow tables into
    # exhaustive nationality rules. Cuba/Haiti may also have CHE history;
    # that separate path must not erase refugee/asylum support.
    # https://www.govinfo.gov/content/pkg/USCODE-2024-title8/html/USCODE-2024-title8-chap12-subchapII-partI-sec1158.htm
    columns = [
        "immigration_was_admitted_as_refugee",
        "immigration_has_been_granted_asylum",
    ]
    expected = _derive(_acs_people(country=200, entry_year=2015))[columns]
    actual = _derive(_acs_people(country=country, entry_year=2015))[columns]

    pd.testing.assert_frame_equal(actual, expected)
    assert actual.any().all()
    assert (~actual).any().all()


def test_withholding_support_is_not_limited_to_leading_asylum_nationalities() -> None:
    output = _derive(_acs_people(country=301, entry_year=2015))
    withholding = output["immigration_has_deportation_or_removal_withheld"]

    assert withholding.any()
    assert (~withholding).any()


_INCIDENT_AUTHORIZATION_COLUMNS = [
    "immigration_is_lawful_permanent_resident",
    "immigration_was_admitted_as_refugee",
    "immigration_has_been_granted_asylum",
    "immigration_is_current_cofa_resident",
    "immigration_has_deportation_or_removal_withheld",
    "immigration_is_current_tps",
]


def test_status_categories_with_incident_authorization_keep_it() -> None:
    person = pd.concat(
        [
            _acs_people(country=301, entry_year=2015),
            _acs_people(country=511, entry_year=2015),
            _acs_people(country=315, entry_year=1995, age=50),
        ],
        ignore_index=True,
    )
    person["person_id"] = np.arange(1, len(person) + 1)
    output = _derive(person)

    for column in _INCIDENT_AUTHORIZATION_COLUMNS:
        assert output[column].any(), column
        assert output.loc[
            output[column], "immigration_has_employment_authorization"
        ].all()


def test_citizens_are_authorized_to_work_without_an_ead_grant() -> None:
    person = _acs_people(count=4)
    person["CIT"] = [1, 2, 3, 4]
    output = _derive(person)

    assert output["immigration_is_us_citizen"].all()
    assert output["immigration_has_employment_authorization"].all()
    assert output["immigration_ssn_is_valid_for_employment"].all()


@pytest.mark.parametrize(
    "country,entry_year,lawful_evidence,status",
    [
        (303, 2005, False, "immigration_is_current_daca"),
        (315, 2023, False, "immigration_is_currently_paroled"),
        (327, 2015, False, "immigration_is_current_cuban_haitian_entrant"),
        (301, 2015, True, "immigration_has_other_lawful_status"),
    ],
)
def test_discretionary_or_unspecified_status_does_not_guarantee_authorization(
    country: int, entry_year: int, lawful_evidence: bool, status: str
) -> None:
    # 8 CFR 274a.12(c)(11)/(33) require a separate authorization grant for
    # ordinary parole/DACA. Unspecified lawful status or CHE alone does not
    # identify one of the incident-authorization categories either.
    # https://www.ecfr.gov/current/title-8/section-274a.12
    output = _derive(
        _acs_people(
            count=2048,
            country=country,
            entry_year=entry_year,
            lawful_evidence=lawful_evidence,
        )
    )
    only_discretionary = output[status] & ~output[_INCIDENT_AUTHORIZATION_COLUMNS].any(
        axis=1
    )
    authorization = output.loc[
        only_discretionary, "immigration_has_employment_authorization"
    ]

    assert only_discretionary.any()
    assert authorization.any()
    assert (~authorization).any()


@pytest.mark.parametrize(
    "country,history",
    [
        (200, "immigration_has_qualifying_afghan_humanitarian_parole_history"),
        (164, "immigration_has_qualifying_ukrainian_humanitarian_parole_history"),
    ],
)
def test_current_qualifying_programme_parole_is_authorized_incident_to_parole(
    country: int, history: str
) -> None:
    output = _derive(_acs_people(country=country, entry_year=2023))
    selected = output[history] & output["immigration_is_currently_paroled"]

    assert selected.any()
    assert output.loc[selected, "immigration_was_paroled"].all()
    assert output.loc[selected, "immigration_has_employment_authorization"].all()


@pytest.mark.parametrize(
    "country,entry_year,first_month,last_month,history",
    [
        (200, 2021, 2021 * 12 + 6, 2021 * 12 + 11, "afghan"),
        (200, 2023, 2023 * 12, 2023 * 12 + 8, "afghan"),
        (164, 2022, 2022 * 12 + 1, 2022 * 12 + 11, "ukrainian"),
        (164, 2024, 2024 * 12, 2024 * 12 + 8, "ukrainian"),
    ],
)
def test_primary_programme_parole_respects_boundary_months(
    country: int, entry_year: int, first_month: int, last_month: int, history: str
) -> None:
    output = _derive(_acs_people(country=country, entry_year=entry_year))
    selected = output[
        f"immigration_has_qualifying_{history}_humanitarian_parole_history"
    ]
    entry = output.loc[selected, "immigration_us_entry_month_index"]
    event = output.loc[selected, "immigration_parole_start_month_index"]

    assert selected.any()
    assert (~selected).any()
    assert output.loc[selected, "immigration_was_paroled"].all()
    assert entry.between(first_month, last_month).all()
    assert entry.nunique() > 1  # Preserve month uncertainty, not one invented date.
    np.testing.assert_array_equal(event, entry)


@pytest.mark.parametrize(
    "country,entry_year,history",
    [
        (200, 2020, "afghan"),
        (200, 2024, "afghan"),
        (164, 2021, "ukrainian"),
        (164, 2025, "ukrainian"),
    ],
)
def test_primary_programme_support_does_not_assert_unknown_family_exceptions(
    country: int, entry_year: int, history: str
) -> None:
    # Later-entry relatives can qualify under the statutes, but this narrow
    # primary-cohort generator has no qualifying-family-link source facts.
    # These false prototype flags are not evidence that a family case fails.
    output = _derive(
        _acs_people(country=country, entry_year=entry_year), year=max(2024, entry_year)
    )

    assert not output[
        f"immigration_has_qualifying_{history}_humanitarian_parole_history"
    ].any()


@pytest.mark.parametrize(
    "arrival_code,lower,upper", [(27, 2020, 2021), (28, 2022, 2024)]
)
def test_asec_afghan_parole_intersects_programme_window_with_source_interval(
    arrival_code: int, lower: int, upper: int
) -> None:
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 513),
            "PRCITSHP": 5,
            "PENATVTY": 200,
            "PEINUSYR": arrival_code,
            "A_AGE": 30,
        }
    )
    output = _derive(person)
    selected = output["immigration_has_qualifying_afghan_humanitarian_parole_history"]
    entry = output.loc[selected, "immigration_us_entry_month_index"]

    assert selected.any()
    assert entry.between(
        max(lower * 12, 2021 * 12 + 6), min(upper * 12 + 11, 2023 * 12 + 8)
    ).all()
    assert (output["immigration_us_entry_year_lower_bound"] == lower).all()
    assert (output["immigration_us_entry_year_upper_bound"] == upper).all()
    assert output["immigration_entry_year_is_interval_midpoint"].all()
    other_entry = output.loc[~selected, "immigration_us_entry_month_index"]
    if arrival_code == 27:
        assert other_entry.lt(2021 * 12 + 6).any()
    else:
        assert other_entry.gt(2023 * 12 + 8).any()


def test_ukrainian_primary_window_extension_is_not_backdated_to_2023() -> None:
    person = _acs_people(country=164, entry_year=2023)
    before = _derive(person, year=2023)
    after = _derive(person, year=2024)
    history = "immigration_has_qualifying_ukrainian_humanitarian_parole_history"

    assert before[history].any()
    assert (
        before.loc[before[history], "immigration_parole_start_month_index"]
        <= 2023 * 12 + 8
    ).all()
    assert (
        after.loc[after[history], "immigration_parole_start_month_index"]
        > 2023 * 12 + 8
    ).any()


@pytest.mark.parametrize("country,history", [(200, "afghan"), (164, "ukrainian")])
def test_programme_parole_history_precedes_any_asserted_lpr_adjustment(
    country: int, history: str
) -> None:
    output = _derive(_acs_people(country=country, entry_year=2023))
    adjusted = output[f"immigration_adjusted_to_lpr_from_qualifying_{history}_parole"]

    assert adjusted.any()
    assert output.loc[adjusted, "immigration_was_paroled"].all()
    assert (
        output.loc[adjusted, "immigration_parole_start_month_index"]
        <= output.loc[adjusted, "immigration_lpr_status_start_month_index"]
    ).all()
    assert not output.loc[adjusted, "immigration_is_currently_paroled"].any()


def test_programme_bounds_and_authorization_grants_preserve_clone_invariance() -> None:
    people = pd.concat(
        [
            _acs_people(country=200, entry_year=2021, count=64),
            _acs_people(country=164, entry_year=2024, count=64),
            _acs_people(country=315, entry_year=2023, count=64),
        ],
        ignore_index=True,
    )
    people["source_year"] = 2024
    people["source_person_id"] = np.arange(1, len(people) + 1)
    expected = _derive(people)
    clones = pd.concat([people, people], ignore_index=True)
    clones["person_id"] = np.arange(1, len(clones) + 1)
    clones["person_weight"] = np.geomspace(1e-6, 1e9, len(clones))
    actual = _derive(clones)
    columns = list(US_IMMIGRATION_FACT_COLUMNS)

    for start in (0, len(people)):
        pd.testing.assert_frame_equal(
            actual.iloc[start : start + len(people)][columns].reset_index(drop=True),
            expected[columns].reset_index(drop=True),
        )


def test_current_daca_does_not_survive_lpr_adjustment() -> None:
    # USCIS's active-DACA population excludes people who obtained LPR status:
    # https://www.uscis.gov/sites/default/files/document/data/active_daca_recipients_fy2026_q2_v1.xlsx
    output = _derive(_acs_people())
    daca = output["immigration_is_current_daca"]
    lpr = output["immigration_is_lawful_permanent_resident"]

    assert daca.any()
    assert lpr.any()
    assert not (daca & lpr).any()


@pytest.mark.parametrize("year,age", [(2024, 44), (2024, 60), (2025, 45)])
def test_daca_excludes_every_definitely_pre_1981_birth(year: int, age: int) -> None:
    # 8 CFR 236.22(b)(7): born on or after June 16, 1981. Source age and
    # survey year are not an exact DOB; even the latest possible birth year
    # for these rows precedes 1981.
    # https://www.ecfr.gov/current/title-8/section-236.22#p-236.22(b)(7)
    output = _derive(
        _acs_people(entry_year=1977 if age == 60 else 1990, age=age),
        year=year,
    )

    assert not output["immigration_is_current_daca"].any()


@pytest.mark.parametrize("year,age", [(2024, 42), (2024, 43), (2025, 44)])
def test_daca_preserves_support_when_the_unknown_birthdate_can_meet_cutoff(
    year: int, age: int
) -> None:
    # Someone aged 43 somewhere within 2024 can have been born in 1980 or
    # 1981. Do not invent January 1 as their DOB and exclude possible births
    # on/after June 16, 1981. The source-supported cohort retains both draws.
    output = _derive(_acs_people(entry_year=1990, age=age), year=year)
    daca = output["immigration_is_current_daca"]

    assert daca.any()
    assert (~daca).any()


def test_daca_birth_cohort_uses_the_age_observation_year_not_later_build_year() -> None:
    # The pooled source retains age 44 observed in 2024 (born no later than
    # 1980); assigning this source to a 2026 build must not turn that into a
    # possible 1982 birth. No exact date is needed to exclude this cohort.
    person = _acs_people(entry_year=1990, age=44)
    person["source_year"] = 2024
    output = _derive(person, year=2026)

    assert not output["immigration_is_current_daca"].any()


_EXPORTED_TRANSITION_EVENTS = (
    (
        "immigration_adjusted_to_lpr_from_refugee_status",
        "immigration_was_admitted_as_refugee",
        "immigration_refugee_admission_month_index",
    ),
    (
        "immigration_adjusted_to_lpr_from_asylum_status",
        "immigration_has_been_granted_asylum",
        "immigration_asylum_grant_month_index",
    ),
    (
        "immigration_adjusted_to_lpr_from_withholding_status",
        "immigration_has_deportation_or_removal_withheld",
        "immigration_withholding_start_month_index",
    ),
    (
        "immigration_adjusted_to_lpr_from_conditional_entry_status",
        "immigration_was_granted_conditional_entry",
        "immigration_conditional_entry_month_index",
    ),
    (
        "immigration_adjusted_to_lpr_from_cuban_haitian_entrant_status",
        "immigration_is_current_cuban_haitian_entrant",
        "immigration_cuban_haitian_entrant_start_month_index",
    ),
    (
        "immigration_adjusted_to_lpr_from_amerasian_status",
        "immigration_is_amerasian_immigrant",
        "immigration_amerasian_admission_month_index",
    ),
    (
        "immigration_adjusted_to_lpr_from_parole_for_at_least_one_year",
        "immigration_was_paroled",
        "immigration_parole_start_month_index",
    ),
)


def test_every_asserted_history_to_lpr_transition_follows_its_event() -> None:
    cohorts = (
        _acs_people(country=200, entry_year=2007, lawful_evidence=True),
        _acs_people(country=207, entry_year=2007, lawful_evidence=True),
        _acs_people(country=412, entry_year=1978, age=70, lawful_evidence=True),
        _acs_people(country=327, entry_year=2015, lawful_evidence=True),
        _acs_people(country=247, entry_year=1995, age=45, lawful_evidence=True),
        _acs_people(country=315, entry_year=2023),
    )
    person = pd.concat(cohorts, ignore_index=True)
    person["person_id"] = np.arange(1, len(person) + 1)
    output = _derive(person)
    lpr = output["immigration_is_lawful_permanent_resident"]
    lpr_month = output["immigration_lpr_status_start_month_index"]

    for transition, history, event in _EXPORTED_TRANSITION_EVENTS:
        adjusted = output[transition]
        assert adjusted.any(), transition
        assert (lpr[adjusted] & output.loc[adjusted, history]).all(), transition
        assert (output.loc[adjusted, event] <= lpr_month[adjusted]).all(), transition


def test_former_parole_survives_lpr_adjustment_without_current_parole() -> None:
    output = _derive(_acs_people(country=315, entry_year=2023))
    adjusted = output["immigration_adjusted_to_lpr_from_parole_for_at_least_one_year"]
    lpr = output["immigration_is_lawful_permanent_resident"]
    current_parole = output["immigration_is_currently_paroled"]

    assert adjusted.any()
    assert not (lpr & current_parole).any()
    assert output.loc[adjusted, "immigration_was_paroled"].all()
    assert (output.loc[adjusted, "immigration_parole_period_years"] >= 1).all()
    assert (output.loc[adjusted, "immigration_parole_start_month_index"] > 0).all()
    assert (
        output.loc[adjusted, "immigration_first_qualified_status_start_month_index"]
        <= output.loc[adjusted, "immigration_parole_start_month_index"]
    ).all()

    # In the ordinary-parole-only path, adjustment must not restart the
    # qualified-status clock at the later LPR event.
    only_parole = (
        adjusted
        & ~output["immigration_has_been_granted_asylum"]
        & ~output["immigration_was_admitted_as_refugee"]
        & ~output["immigration_is_battered_alien_or_qualifying_relative"]
        & ~output["immigration_is_severe_trafficking_victim_or_qualifying_relative"]
    )
    assert only_parole.any()
    np.testing.assert_array_equal(
        output.loc[only_parole, "immigration_first_qualified_status_start_month_index"],
        output.loc[only_parole, "immigration_parole_start_month_index"],
    )


def test_parole_period_support_straddles_one_year_threshold_stably() -> None:
    person = _acs_people(count=2048, country=315, entry_year=2023)
    output = _derive(person)
    repeated = _derive(person)
    history = output["immigration_was_paroled"]
    period = output["immigration_parole_period_years"]
    short = history & (period < 1)
    at_least_one_year = history & (period >= 1)
    current = output["immigration_is_currently_paroled"]

    assert short.any()
    assert at_least_one_year.any()
    assert set(period[history].unique()) == {0, 2}
    assert (period[~history] == 0).all()
    assert not current[short].any()
    assert (current <= at_least_one_year).all()
    np.testing.assert_array_equal(
        period,
        repeated["immigration_parole_period_years"],
    )

    same_year = _derive(person, year=2023)
    same_year_short = same_year["immigration_was_paroled"] & (
        same_year["immigration_parole_period_years"] < 1
    )
    short_without_lpr = (
        same_year_short & ~same_year["immigration_is_lawful_permanent_resident"]
    )
    assert short_without_lpr.any()
    assert same_year.loc[short_without_lpr, "immigration_is_currently_paroled"].all()

    adjusted_from_one_year_parole = output[
        "immigration_adjusted_to_lpr_from_parole_for_at_least_one_year"
    ]
    assert adjusted_from_one_year_parole.any()
    assert not adjusted_from_one_year_parole[short].any()
    assert (adjusted_from_one_year_parole <= at_least_one_year).all()


def test_expired_parole_retains_history_and_event_month() -> None:
    # A retained grant in 2023 is no longer current in 2026; its event still
    # belongs to a person's immigration history regardless of grant length.
    output = _derive(_acs_people(country=315, entry_year=2023), year=2026)
    history = output["immigration_was_paroled"]

    assert history.any()
    assert not output["immigration_is_currently_paroled"].any()
    assert (output.loc[history, "immigration_parole_start_year"] == 2023).all()
    assert (output.loc[history, "immigration_parole_start_month_index"] > 0).all()
    assert (output.loc[~history, "immigration_parole_start_month_index"] == 0).all()


def test_transition_fixes_preserve_clone_and_weight_invariance() -> None:
    person = _acs_people(country=315, entry_year=2023, count=64)
    person["source_year"] = 2024
    person["source_household_id"] = person["person_id"]
    person["source_person_id"] = 1
    person["person_weight"] = 1.0
    expected = _derive(person)

    clones = pd.concat([person, person], ignore_index=True)
    clones["person_id"] = np.arange(1, len(clones) + 1)
    clones["person_weight"] = np.geomspace(1e-9, 1e12, len(clones))
    actual = _derive(clones)
    columns = list(US_IMMIGRATION_FACT_COLUMNS)
    pd.testing.assert_frame_equal(
        actual.iloc[: len(person)][columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        actual.iloc[len(person) :][columns].reset_index(drop=True),
        expected[columns].reset_index(drop=True),
    )


def _frame(person: pd.DataFrame) -> Frame:
    person = person.copy()
    ids = person["person_id"].to_numpy()
    for entity in US_SCHEMA.group_entities:
        person[US_SCHEMA.membership_column(entity)] = ids
    return Frame(
        {
            "person": person,
            **{
                entity: pd.DataFrame({US_SCHEMA.id_column(entity): ids})
                for entity in US_SCHEMA.group_entities
            },
        },
        US_SCHEMA,
        {"household": Weights(values=np.ones(len(person)), kind=WeightKind.DESIGN)},
    )


@pytest.mark.parametrize(
    "flag,label",
    [
        ("immigration_is_current_daca", "current DACA and LPR"),
        ("immigration_is_currently_paroled", "current parole and LPR"),
    ],
)
def test_gate_rejects_incompatible_current_lpr_statuses(flag: str, label: str) -> None:
    output = _derive(_acs_people(country=315, entry_year=2023))
    row = output.index[output["immigration_is_lawful_permanent_resident"]][0]
    output.loc[row, flag] = True
    result = us_immigration_composition_gate(_frame(output))

    assert not result.passed
    assert any(label in failure for failure in result.failures)


def test_gate_rejects_history_event_after_its_asserted_lpr_adjustment() -> None:
    output = _derive(_acs_people(country=200, entry_year=2007, lawful_evidence=True))
    transition = "immigration_adjusted_to_lpr_from_asylum_status"
    row = output.index[output[transition]][0]
    after_lpr = output.loc[row, "immigration_lpr_status_start_month_index"] + 1
    output.loc[row, "immigration_asylum_grant_month_index"] = after_lpr
    output.loc[row, "immigration_asylum_grant_year"] = after_lpr // 12
    result = us_immigration_composition_gate(_frame(output))

    assert not result.passed
    assert any(transition in failure for failure in result.failures)
