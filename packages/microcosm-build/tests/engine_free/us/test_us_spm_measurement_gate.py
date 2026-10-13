"""The release gate on SPM measurement composition (microcosm#1263).

Every frame is invented (``test_support.microcosm_build.us_spm_measurement_gate``).
The property tests beside this file pin the gate's invariants; the engine
scenario test pins its rule against the installed engine.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.spm_composition import check_spm_composition
from microcosm.build.us_runtime.spm_measurement_gate import (
    SPM_MEASUREMENT_GATE_NAME,
    check_release_spm_measurement,
    engine_scopes_spm_universe,
)
from microcosm.build.us_runtime.spm_universe_source import (
    UNIVERSE_INPUT,
    attach_spm_universe_status,
)
from test_support.microcosm_build.us_spm_measurement_gate import (
    ASEC,
    Household,
    release_frame,
)

#: A family, a group-quarters adult, a group-quarters teen carrying the role,
#: and a donor-spine household whose only member is a role-carrying teen.
_MEASURABLE = [
    Household([(41, True), (39, True), (9, False)]),
    Household([(67, True)], kind=3),
    Household([(16, True)], kind=2),
    Household([(16, True)], spine=ASEC),
]
#: A group-quarters child: no role can classify a member under 15.
_GQ_CHILD = Household([(11, True)], kind=2)
#: A housing unit with no classified adult: a data defect under any engine.
_ZERO_ADULT_HOUSING_UNIT = Household([(16, False), (12, False)])


def _grade(households, *, scopes=False, **frame_options):
    return check_release_spm_measurement(
        release_frame(households, **frame_options), engine_scopes_universe=scopes
    )


def test_the_gate_name_is_the_contract_key():
    from microcosm.data.contract import US_SPM_MEASUREMENT_EVIDENCE_KEY

    assert SPM_MEASUREMENT_GATE_NAME == US_SPM_MEASUREMENT_EVIDENCE_KEY


@pytest.mark.parametrize(
    ("variables", "scopes"),
    [
        ({"age", "spm_unit_spm_threshold"}, False),
        ({"age", UNIVERSE_INPUT}, True),
    ],
)
def test_an_engine_scopes_by_universe_only_when_it_defines_the_declaration(
    variables, scopes
):
    assert engine_scopes_spm_universe(variables) is scopes


def test_a_measurable_release_passes_and_reports_aggregates():
    result = _grade(_MEASURABLE)

    assert result.passed, result.failures
    assert result.failures == ()
    details = result.details
    assert details["measured_scope"] == "all_units"
    assert details["n_units"] == details["n_measured_units"] == 4
    assert details["n_units_by_universe"] == {"INCLUDED": 2, "OUTSIDE": 2}
    assert details["n_measured_units_without_classified_adult"] == 0
    assert details["n_units_without_classified_adult_by_universe"] == {
        "INCLUDED": 0,
        "OUTSIDE": 0,
    }
    assert details["role_null_values"] == 0
    assert details["role_true_persons"] == 5


def test_a_group_quarters_child_fails_under_an_engine_that_measures_every_unit():
    """The release of 2026-09-23 had 1,946 of these; every state region failed."""

    result = _grade([*_MEASURABLE, _GQ_CHILD])

    assert not result.passed
    assert result.details["n_measured_units_without_classified_adult"] == 1
    assert result.details["n_units_without_classified_adult_by_universe"] == {
        "INCLUDED": 0,
        "OUTSIDE": 1,
    }
    assert result.details["unclassified_units_by_oldest_member"] == {
        "under_15": 1,
        "15_to_17": 0,
        "unknown": 0,
    }
    assert "SPM_COMPOSITION_REQUIRED" in result.failures[0]


def test_a_group_quarters_child_passes_once_the_engine_scopes_and_the_release_declares():
    frame = attach_spm_universe_status(release_frame([*_MEASURABLE, _GQ_CHILD])).frame

    result = check_release_spm_measurement(frame, engine_scopes_universe=True)

    assert result.passed, result.failures
    assert result.details["measured_scope"] == "included_units"
    assert result.details["n_measured_units"] == 2
    # Reported, not excused: the residue stays on the record.
    assert result.details["n_units_without_classified_adult_by_universe"] == {
        "INCLUDED": 0,
        "OUTSIDE": 1,
    }


def test_a_scoping_engine_refuses_a_release_that_declares_no_universe():
    result = _grade(_MEASURABLE, scopes=True)

    assert not result.passed
    assert any("SPM_UNIVERSE_REQUIRED" in line for line in result.failures)


@pytest.mark.parametrize("scopes", [False, True])
def test_a_zero_adult_housing_unit_fails_under_every_engine(scopes):
    frame = attach_spm_universe_status(
        release_frame([*_MEASURABLE, _ZERO_ADULT_HOUSING_UNIT])
    ).frame

    result = check_release_spm_measurement(frame, engine_scopes_universe=scopes)

    assert not result.passed
    assert result.details["n_units_without_classified_adult_by_universe"] == {
        "INCLUDED": 1,
        "OUTSIDE": 0,
    }


def test_a_release_without_the_role_column_fails():
    """The head/spouse fallback is not a release remedy."""

    result = _grade(_MEASURABLE, role=False)

    assert not result.passed
    assert result.details["role_column_present"] is False
    assert "is_spm_independent_minor_role" in result.failures[0]


def test_a_null_role_fails_rather_than_reading_as_false():
    frame = release_frame(_MEASURABLE)
    person = frame.table("person")
    person["is_spm_independent_minor_role"] = person[
        "is_spm_independent_minor_role"
    ].astype(object)
    person.loc[0, "is_spm_independent_minor_role"] = None

    result = check_release_spm_measurement(frame, engine_scopes_universe=False)

    assert not result.passed
    assert result.details["role_null_values"] == 1


#: An adult, and a group-quarters 16-year-old whose stored role is False. Read
#: correctly, the teen's unit has no classified adult; read as True, it has one.
_TEEN_WITHOUT_ROLE = [Household([(41, True)]), Household([(16, False)], kind=2)]

_ACCEPTED_ROLE_ENCODINGS = {
    "numpy_bool": lambda role: role.to_numpy(dtype=bool),
    "nullable_boolean": lambda role: role.astype("boolean"),
    "uint8": lambda role: role.astype(np.uint8),
    "int64": lambda role: role.astype(np.int64),
    "float64": lambda role: role.astype(np.float64),
    "object_bool": lambda role: role.astype(object),
    "object_int": lambda role: pd.Series([int(v) for v in role], dtype=object),
}
_REFUSED_ROLE_ENCODINGS = {
    "text_digits": lambda role: pd.Series(["1" if v else "0" for v in role]),
    "object_text_digits": lambda role: pd.Series(
        ["1" if v else "0" for v in role], dtype=object
    ),
    "text_words": lambda role: pd.Series([str(bool(v)) for v in role], dtype=object),
    "bytes_digits": lambda role: pd.Series(
        [b"1" if v else b"0" for v in role], dtype=object
    ),
    "object_bool_and_text": lambda role: pd.Series([True, "0"], dtype=object),
    "two_for_true": lambda role: role.astype(np.int64) * 2,
    "fraction": lambda role: role.astype(np.float64) * 0.5,
}


def _with_role_encoding(encode):
    frame = release_frame(_TEEN_WITHOUT_ROLE)
    person = frame.table("person")
    # A Series keeps its dtype on assignment; the person index is 0..n-1.
    person["is_spm_independent_minor_role"] = encode(
        person["is_spm_independent_minor_role"].astype(bool)
    )
    return frame


@pytest.mark.parametrize("encoding", sorted(_ACCEPTED_ROLE_ENCODINGS))
def test_every_accepted_role_encoding_reads_the_same_role(encoding):
    """A Boolean, or a number equal to 0 or 1, in any column type that can
    hold one: each reads the teen's False as False."""

    frame = _with_role_encoding(_ACCEPTED_ROLE_ENCODINGS[encoding])

    result = check_release_spm_measurement(frame, engine_scopes_universe=False)

    assert result.details["role_non_boolean_values"] == 0
    assert result.details["role_true_persons"] == 1
    assert result.details["n_units_without_classified_adult"] == 1
    assert not result.passed


@pytest.mark.parametrize("encoding", sorted(_REFUSED_ROLE_ENCODINGS))
def test_a_role_that_is_not_boolean_is_refused_before_it_can_cast_to_true(encoding):
    """Text and bytes cast to True whatever they spell: a stored ``"0"`` would
    hand the teen the role and the release would pass. The gate refuses the
    column instead of grading it."""

    frame = _with_role_encoding(_REFUSED_ROLE_ENCODINGS[encoding])

    result = check_release_spm_measurement(frame, engine_scopes_universe=False)

    assert not result.passed
    assert result.details["role_non_boolean_values"] >= 1
    # Refused, not graded: no composition was computed from the column.
    assert "role_true_persons" not in result.details
    assert "n_units_without_classified_adult" not in result.details


def test_a_stored_declaration_must_equal_the_source():
    """Storing INCLUDED on a group-quarters unit would hide it from a scoping engine."""

    result = _grade([*_MEASURABLE, _GQ_CHILD], declare="INCLUDED")

    assert not result.passed
    assert any("differs from the source declaration" in f for f in result.failures)


def test_a_bytes_encoded_declaration_reads_as_its_member_name():
    frame = attach_spm_universe_status(release_frame(_MEASURABLE)).frame
    units = frame.table("spm_unit")
    units[UNIVERSE_INPUT] = [value.encode() for value in units[UNIVERSE_INPUT]]

    assert check_release_spm_measurement(frame, engine_scopes_universe=True).passed


def test_an_undeclarable_universe_fails_with_the_producers_code():
    frame = release_frame(_MEASURABLE)
    frame.table("household")["household_spine"] = "puf_tax_detail"

    result = check_release_spm_measurement(frame, engine_scopes_universe=False)

    assert not result.passed
    assert result.details["universe_refusal"] == "SPM_UNIVERSE_UNDECLARED_SPINE"
    assert "cannot be declared" in result.failures[0]
    assert "SPM_UNIVERSE_UNDECLARED_SPINE" in result.failures[0]


def test_a_universe_refusal_names_no_record():
    """The producer's own message lists example household ids. The gate keeps
    the code and drops the examples."""

    crowded = [*_MEASURABLE, Household([(30, True), (31, True)], kind=2)]

    result = _grade(crowded)

    assert result.failures == (
        "The source SPM measurement universe cannot be declared for this "
        "release (SPM_UNIVERSE_GQ_MULTI_PERSON). Run "
        "microcosm.build.us_runtime.spm_universe_source.classify_spm_universe "
        "on the frame for the records behind it.",
    )
    assert result.details["universe_refusal"] == "SPM_UNIVERSE_GQ_MULTI_PERSON"


def test_an_unjoinable_membership_fails_without_naming_an_id():
    frame = release_frame(_MEASURABLE)
    frame.table("person").loc[0, "person_spm_unit_id"] = 555_444_333

    result = check_release_spm_measurement(frame, engine_scopes_universe=False)

    assert not result.passed
    assert "cannot be evaluated" in result.failures[0]
    assert "555444333" not in json.dumps(
        {"failures": result.failures, "details": result.details}
    )


def test_the_count_agrees_with_the_preflight_composition_check():
    households = [*_MEASURABLE, _GQ_CHILD, _ZERO_ADULT_HOUSING_UNIT]
    frame = release_frame(households)

    gate = check_release_spm_measurement(frame, engine_scopes_universe=False)
    preflight = check_spm_composition(frame)

    assert (
        gate.details["n_units_without_classified_adult"]
        == preflight.details["n_units_without_classified_adult"]
        == 2
    )


def test_the_verdict_names_no_unit_and_no_age():
    """Aggregates only: a gate summary is published beside the release."""

    result = _grade([*_MEASURABLE, _GQ_CHILD, _ZERO_ADULT_HOUSING_UNIT])
    text = json.dumps({"failures": result.failures, "details": result.details})

    for unit_id in range(987_001, 987_007):
        assert str(unit_id) not in text
    assert "member_age_bands" not in text
