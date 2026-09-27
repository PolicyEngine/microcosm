"""Actual retained source owners over the maintained tiny invented fixture.

These cases require a separate bounded runtime proposal. They replace only
fixture pins before issuance and never substitute an issuer or checked accessor.
"""

import json
import sys

import pytest
from test_us_current_asec_demographics import _demographic_arguments

from microcosm.build.us_runtime import current_survey_household_roles as roles


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("native", "ASEC_NATIVE_JOIN"),
        ("line", "ASEC_COORDINATE_DISAGREE"),
        ("household", "ASEC_COORDINATE_DISAGREE"),
    ],
)
def test_actual_asec_owner_rejects_detached_mismatched_join_coordinates(
    tmp_path, monkeypatch, defect, reason
):
    prepared = roles.source.prepare_authenticated_survey_population(
        **_demographic_arguments(tmp_path, monkeypatch)
    )
    entry = prepared._checked()
    state = entry[2]
    before = roles.source._frame_identity(state.frame)
    origins = roles._origins(state.frame, json.loads(entry[1]))
    keys = roles._asec_keys(origins)
    observed, _ = roles._asec_observed(state, prepared)
    expected = roles._asec_selected(origins, keys, observed)
    changed = origins.copy(deep=True)
    row = changed.index[changed.source.eq("asec")][0]
    if defect == "native":
        changed.loc[row, "native_person_id"] = (
            int(observed.array("person_id").max()) + 1
        )
    elif defect == "line":
        old = int(changed.loc[row, "native_line_numeric_original"])
        changed.loc[row, "native_line_numeric_original"] = "2" if old == 1 else "1"
    else:
        old = int(changed.loc[row, "raw_native_household_id"])
        changed.loc[row, "raw_native_household_id"] = "2" if old == 1 else "1"
    # Only a detached consumer coordinate is changed. The real preparation,
    # owner buffers, member bytes and issuance accessors remain untouched.
    with pytest.raises(ValueError, match=reason):
        roles._asec_selected(changed, keys, observed)
    assert roles._asec_selected(origins, keys, observed) == expected
    observed.validate()
    assert prepared._checked() is entry
    assert roles.source._frame_identity(state.frame) == before


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("household", "ACS_ROSTER_HOUSEHOLD_SCOPE"),
        ("line", "ACS_ROSTER_COVERAGE"),
    ],
)
def test_actual_acs_roster_rejects_detached_unpublished_retained_scope(
    tmp_path, monkeypatch, defect, reason
):
    prepared = roles.source.prepare_authenticated_survey_population(
        **_demographic_arguments(tmp_path, monkeypatch)
    )
    entry = prepared._checked()
    state = entry[2]
    before = roles.source._frame_identity(state.frame)
    origins = roles._origins(state.frame, json.loads(entry[1]))
    _, serials = roles._acs_keys(origins)
    roster, owner = roles._acs_roster(state, serials)
    expected = roles._acs_states(serials, roster)
    changed = {serial: set(lines) for serial, lines in serials.items()}
    if defect == "household":
        invented = "2024HU9999999"
        assert invented not in changed
        changed[invented] = {1}
    else:
        serial = sorted(changed)[0]
        absent = next(
            line
            for line in range(1, roles.MAX_HOUSEHOLD_MEMBERS + 1)
            if (serial, line) not in roster
        )
        changed[serial].add(absent)
    # The retained scope is the adversarial input; the roster still comes from
    # the real pinned catalogue and no issuer/classifier is substituted.
    with pytest.raises(ValueError, match=reason):
        roles._acs_states(changed, roster)
    assert roles._acs_states(serials, roster) == expected
    assert roles.source.acs_catalogue._lookup(state.catalogues[0]) is owner
    assert prepared._checked() is entry
    assert roles.source._frame_identity(state.frame) == before


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
    "mutation",
    [
        "callable",
        "code",
        "defaults",
        "period",
        "column",
        "owner_code",
        "shared_private",
        "shared_public",
    ],
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
        elif mutation in {"shared_private", "shared_public"}:
            name = (
                "_UNBOUND_DEMOGRAPHICS"
                if mutation == "shared_private"
                else "UNBOUND_DEMOGRAPHIC_COLUMNS"
            )
            monkeypatch.setattr(
                roles.demographic_contract,
                name,
                getattr(roles.demographic_contract, name)[1:],
            )
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
