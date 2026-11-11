"""Properties of the SPM measurement release gate over generated releases.

Each test draws a release-shaped frame: ACS housing units of one to four
people, one-person ACS group-quarters records and ASEC donor households, each
one SPM unit, with any ages and roles. For every such frame:

- oracle: the gate passes exactly when every unit the engine measures has a
  member satisfying ``age >= 18 or (age >= 15 and role)``, restated
  independently of the implementation; the engine measures every unit unless
  it scopes by universe and the release declares one, and then only the units
  the source places inside the universe;
- conservation: the unclassified count splits exactly into the two universes,
  and the units split exactly into them too;
- containment: the measured set always contains every in-universe unit, so a
  scoping engine never reports more measured unclassified units than one that
  measures everything;
- row order: permuting the person rows changes no detail;
- age floor: giving the role to anyone under 15 changes no detail;
- differential: the unclassified count equals the preflight composition
  check's.
"""

from __future__ import annotations

import numpy as np
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.us_runtime.spm_composition import (  # noqa: E402
    check_spm_composition,
)
from microcosm.build.us_runtime.spm_measurement_gate import (  # noqa: E402
    check_release_spm_measurement,
)
from microcosm.build.us_runtime.spm_universe_source import (  # noqa: E402
    INCLUDED,
    OUTSIDE,
    attach_spm_universe_status,
)
from test_support.microcosm_build.us_spm_measurement_gate import (  # noqa: E402
    ACS,
    ASEC,
    Household,
    expected_unclassified,
    release_frame,
)

_MEMBER = st.tuples(st.integers(0, 90), st.booleans())


@st.composite
def _households(draw) -> list[Household]:
    households = []
    for _ in range(draw(st.integers(min_value=1, max_value=10))):
        shape = draw(st.sampled_from(("housing_unit", "group_quarters", "donor")))
        if shape == "group_quarters":
            households.append(
                Household([draw(_MEMBER)], kind=draw(st.sampled_from((2, 3))))
            )
        else:
            households.append(
                Household(
                    draw(st.lists(_MEMBER, min_size=1, max_size=4)),
                    spine=ACS if shape == "housing_unit" else ASEC,
                )
            )
    return households


def _gate(households, *, scopes, declare, **options):
    frame = release_frame(households, **options)
    if declare:
        frame = attach_spm_universe_status(frame).frame
    return check_release_spm_measurement(frame, engine_scopes_universe=scopes)


@settings(max_examples=150, deadline=None)
@given(households=_households(), scopes=st.booleans(), declare=st.booleans())
def test_the_gate_passes_exactly_when_every_measured_unit_has_an_adult(
    households, scopes, declare
):
    result = _gate(households, scopes=scopes, declare=declare)

    if scopes and not declare:
        # A scoping engine cannot measure an undeclared release at all.
        assert not result.passed
        return
    unclassified = expected_unclassified(households)
    measured = [
        not scopes or household.universe == INCLUDED for household in households
    ]
    oracle_failures = sum(u and m for u, m in zip(unclassified, measured, strict=True))
    assert result.passed is (oracle_failures == 0)
    assert result.details["n_measured_units_without_classified_adult"] == (
        oracle_failures
    )
    assert result.details["n_measured_units"] == sum(measured)


@settings(max_examples=150, deadline=None)
@given(households=_households())
def test_counts_split_exactly_into_the_two_universes(households):
    details = _gate(households, scopes=False, declare=False).details

    by_universe = details["n_units_without_classified_adult_by_universe"]
    assert set(by_universe) == {INCLUDED, OUTSIDE}
    assert sum(by_universe.values()) == details["n_units_without_classified_adult"]
    assert sum(details["n_units_by_universe"].values()) == details["n_units"]
    assert details["n_units_by_universe"][OUTSIDE] == sum(
        household.universe == OUTSIDE for household in households
    )
    assert (
        sum(details["unclassified_units_by_oldest_member"].values())
        == (details["n_units_without_classified_adult"])
    )


@settings(max_examples=150, deadline=None)
@given(households=_households())
def test_a_scoping_engine_measures_a_subset_containing_every_included_unit(
    households,
):
    everything = _gate(households, scopes=False, declare=True).details
    scoped = _gate(households, scopes=True, declare=True).details

    assert scoped["n_measured_units"] == everything["n_units_by_universe"][INCLUDED]
    assert scoped["n_measured_units"] <= everything["n_measured_units"]
    assert (
        scoped["n_measured_units_without_classified_adult"]
        == everything["n_units_without_classified_adult_by_universe"][INCLUDED]
        <= everything["n_measured_units_without_classified_adult"]
    )


@settings(max_examples=100, deadline=None)
@given(households=_households(), data=st.data())
def test_person_row_order_changes_nothing(households, data):
    persons = sum(len(household.members) for household in households)
    order = data.draw(st.permutations(range(persons)))

    original = _gate(households, scopes=False, declare=False)
    permuted = _gate(households, scopes=False, declare=False, person_order=order)

    assert permuted.passed is original.passed
    assert dict(permuted.details) == dict(original.details)


@settings(max_examples=100, deadline=None)
@given(households=_households())
def test_the_role_never_counts_below_age_fifteen(households):
    lifted = [
        Household(
            [(age, role or age < 15) for age, role in household.members],
            kind=household.kind,
            spine=household.spine,
        )
        for household in households
    ]

    original = _gate(households, scopes=False, declare=False).details
    changed = _gate(lifted, scopes=False, declare=False).details

    assert {
        key: value for key, value in changed.items() if key != "role_true_persons"
    } == {key: value for key, value in original.items() if key != "role_true_persons"}


@settings(max_examples=100, deadline=None)
@given(households=_households())
def test_the_count_equals_the_preflight_composition_check(households):
    frame = release_frame(households)

    gate = check_release_spm_measurement(frame, engine_scopes_universe=False)
    preflight = check_spm_composition(frame)

    assert (
        gate.details["n_units_without_classified_adult"]
        == (preflight.details["n_units_without_classified_adult"])
    )
    assert gate.passed is (preflight.status == "PASS")
    assert np.isfinite(gate.details["n_units"])
