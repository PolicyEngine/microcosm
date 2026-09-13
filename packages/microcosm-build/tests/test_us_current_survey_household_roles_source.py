"""Actual retained source owners over the maintained tiny invented fixture.

These cases require a separate bounded runtime proposal. They replace only
fixture pins before issuance and never substitute an issuer or checked accessor.
"""

import sys

import pytest
from test_us_current_asec_demographics import _demographic_arguments

from microcosm.build.us_runtime import current_survey_household_roles as roles


def test_actual_sources_qualify_named_reference_roles_and_leave_gq_unbound(
    tmp_path, monkeypatch
):
    prepared = roles.source.prepare_authenticated_survey_population(
        **_demographic_arguments(tmp_path, monkeypatch)
    )
    before = roles.source._frame_identity(prepared.checked_view().frame)
    qualified = roles.qualify_current_survey_household_roles(prepared)
    table = qualified.rows
    acs = table.loc[table[roles.SURVEY_COLUMN].eq("acs")]
    asec = table.loc[table[roles.SURVEY_COLUMN].eq("asec")]
    assert len(acs) == 5 and len(asec) == 4
    assert acs[roles.CODE_KNOWN_COLUMN].all() and asec[roles.CODE_KNOWN_COLUMN].all()
    assert asec[roles.VALUE_KNOWN_COLUMN].all()
    assert int(asec[roles.VALUE_COLUMN].sum()) == 2
    gq = acs.loc[acs[roles.CODE_COLUMN].isin((37, 38))]
    housing = acs.loc[~acs.index.isin(gq.index)]
    assert len(gq) == 2 and gq[roles.VALUE_COLUMN].isna().all()
    assert not gq[roles.VALUE_KNOWN_COLUMN].any()
    assert gq[roles.UNBOUND_REASON_COLUMN].eq(5).all()
    assert housing[roles.VALUE_KNOWN_COLUMN].all()
    assert int(housing[roles.VALUE_COLUMN].sum()) == 2
    assert not table[roles.ALLOCATION_KNOWN_COLUMN].any()
    assert roles.source._frame_identity(qualified.source_frame) == before
    assert roles.source._frame_identity(prepared.checked_view().frame) == before
    fresh = roles.qualify_current_survey_household_roles(prepared)
    assert roles.household_role_projection_seal(
        fresh
    ) == roles.household_role_projection_seal(qualified)


@pytest.mark.parametrize("borrow_number", [1, 2])
@pytest.mark.parametrize(
    "mutation", ["callable", "code", "defaults", "period", "column", "owner_code"]
)
def test_first_and_final_preparation_borrows_refuse_live_changes(
    tmp_path, monkeypatch, borrow_number, mutation
):
    prepared = roles.source.prepare_authenticated_survey_population(
        **_demographic_arguments(tmp_path, monkeypatch)
    )
    checked = type(prepared)._checked
    checked_code = checked.__code__
    qualify_code = roles.qualify_current_survey_household_roles.__code__
    seen, fired = [], []

    def replacement(*args, **kwargs):
        raise AssertionError("mutated executable must not run")

    def profile(frame, event, arg):
        if (
            event != "return"
            or frame.f_code is not checked_code
            or frame.f_back is None
            or frame.f_back.f_code is not qualify_code
        ):
            return
        seen.append(True)
        if len(seen) != borrow_number:
            return
        fired.append(True)
        if mutation == "callable":
            monkeypatch.setattr(roles, "_project", replacement)
        elif mutation == "code":
            monkeypatch.setattr(roles._project, "__code__", replacement.__code__)
        elif mutation == "defaults":
            monkeypatch.setattr(roles._project, "__defaults__", (None,))
        elif mutation == "period":
            monkeypatch.setattr(roles, "ACS_OBSERVATION_YEAR", 2099)
        elif mutation == "column":
            monkeypatch.setattr(roles, "VALUE_COLUMN", "unreviewed_role")
        else:
            monkeypatch.setattr(checked, "__code__", replacement.__code__)

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            ValueError, match="CURRENT_SURVEY_HOUSEHOLD_ROLES_IMPLEMENTATION_CHANGED"
        ):
            roles.qualify_current_survey_household_roles(prepared)
    finally:
        sys.setprofile(previous)
    assert len(seen) == borrow_number and fired == [True]


def test_last_acs_owner_callback_cannot_change_a_column_alias(tmp_path, monkeypatch):
    prepared = roles.source.prepare_authenticated_survey_population(
        **_demographic_arguments(tmp_path, monkeypatch)
    )
    lookup_code = roles.source.acs_catalogue._lookup.__code__
    qualify_code = roles.qualify_current_survey_household_roles.__code__
    fired = []

    def profile(frame, event, arg):
        if (
            event == "return"
            and frame.f_code is lookup_code
            and frame.f_back is not None
            and frame.f_back.f_code is qualify_code
        ):
            fired.append(True)
            monkeypatch.setattr(roles, "CODE_COLUMN", "unreviewed_source_code")

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        with pytest.raises(
            ValueError, match="CURRENT_SURVEY_HOUSEHOLD_ROLES_IMPLEMENTATION_CHANGED"
        ):
            roles.qualify_current_survey_household_roles(prepared)
    finally:
        sys.setprofile(previous)
    assert fired == [True]
