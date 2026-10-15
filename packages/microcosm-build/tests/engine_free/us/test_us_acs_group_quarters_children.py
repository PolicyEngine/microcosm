"""The interim drop of ACS group-quarters children (microcosm#1263, d1484).

Max ruled on 2026-10-15 that the one-person ACS group-quarters records of
children under 15, whom no SPM independence role can classify, leave the ACS
local population until the engine scopes SPM by measurement universe. Every
frame here is invented (``test_support.microcosm_build.us_acs_spm_role``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.us_runtime.acs_inputs import (  # noqa: E402
    ACS_GROUP_QUARTERS_CHILD_RULE,
    with_acs_spm_independence_role,
    without_acs_group_quarters_children,
)
from microcosm.build.us_runtime.spm_composition import (  # noqa: E402
    check_spm_composition,
)
from test_support.microcosm_build.us_acs_spm_role import (  # noqa: E402
    OTHER_MEMBER_CODES,
    acs_frame,
)

#: A family with a child, a group-quarters child, a group-quarters teen, a
#: group-quarters adult and a housing unit headed by a 15-year-old.
_HOUSEHOLDS = [
    [(20, 40.0), (25, 9.0)],
    [(37, 10.0)],
    [(38, 16.0)],
    [(37, 70.0)],
    [(20, 15.0)],
]
_KINDS = [1, 3, 2, 3, 1]


def _person_ids(frame) -> list[int]:
    return sorted(frame.table("person")["person_id"].tolist())


def test_only_group_quarters_children_under_15_leave():
    frame = acs_frame(_HOUSEHOLDS, typehugq=_KINDS)

    result = without_acs_group_quarters_children(frame)

    # Person 3 is the group-quarters ten-year-old; the housing-unit child
    # (person 2) and every other group-quarters record stay.
    assert _person_ids(result.frame) == [1, 2, 4, 5, 6]
    assert result.frame.table("household")["household_id"].tolist() == [1, 3, 4, 5]
    assert result.frame.table("spm_unit")["spm_unit_id"].tolist() == [1, 3, 4, 5]
    assert result.frame.table("tax_unit")["tax_unit_id"].tolist() == [1, 3, 4, 5]
    assert result.frame.weights_for("household").values.tolist() == [
        10.0,
        30.0,
        40.0,
        50.0,
    ]


def test_the_receipt_counts_and_discloses_the_drop():
    frame = acs_frame(_HOUSEHOLDS, typehugq=_KINDS)

    provenance = without_acs_group_quarters_children(frame).provenance

    assert provenance["households_before"] == 5
    assert provenance["persons_before"] == 6
    assert provenance["group_quarters_households"] == 3
    assert provenance["households_dropped"] == 1
    assert provenance["persons_dropped"] == 1
    assert provenance["household_weight_dropped"] == 20.0
    assert provenance["household_weight_share_dropped"] == pytest.approx(20.0 / 150.0)
    assert provenance["rule"] == ACS_GROUP_QUARTERS_CHILD_RULE
    assert provenance["status"] == "interim"
    assert "d1484" in provenance["decision"]
    assert "spm_unit_spm_universe_status" in provenance["flip_condition"]
    assert "SEHSD-WP2020-09" in provenance["authority"]


def test_a_frame_without_group_quarters_children_is_returned_unchanged():
    frame = acs_frame([[(20, 40.0), (25, 3.0)], [(37, 15.0)]], typehugq=[1, 2])

    result = without_acs_group_quarters_children(frame)

    assert result.frame is frame
    assert result.provenance["households_dropped"] == 0


def test_after_the_drop_every_unit_has_a_classified_adult():
    """The point of the drop: under the role rule, the staged ACS spine passes
    the composition check that the engine's SPM measurement would apply."""

    frame = acs_frame(_HOUSEHOLDS, typehugq=_KINDS)
    assert (
        check_spm_composition(with_acs_spm_independence_role(frame).frame).details[
            "n_units_without_classified_adult"
        ]
        == 1
    )

    dropped = without_acs_group_quarters_children(frame).frame
    composition = check_spm_composition(with_acs_spm_independence_role(dropped).frame)

    assert composition.status == "PASS"


@pytest.mark.parametrize(
    ("households", "kinds", "match"),
    [
        ([[(37, 10.0), (37, 11.0)]], [3], "more than one"),
        ([[(20, 40.0)]], [4], "TYPEHUGQ in \\{1, 2, 3\\}"),
        ([[(37, np.nan)]], [3], "observed age"),
    ],
)
def test_it_refuses_rather_than_guessing(households, kinds, match):
    frame = acs_frame(households, typehugq=kinds)

    with pytest.raises(ValueError, match=match):
        without_acs_group_quarters_children(frame)


def test_it_refuses_a_frame_without_the_record_type():
    with pytest.raises(ValueError, match="TYPEHUGQ"):
        without_acs_group_quarters_children(acs_frame([[(37, 10.0)]]))


@st.composite
def _spine(draw) -> tuple[list[list[tuple[int, float]]], list[int]]:
    """Valid ACS spines: housing units with one head, or one-person GQ."""

    households, kinds = [], []
    for _ in range(draw(st.integers(min_value=1, max_value=10))):
        if draw(st.booleans()):
            code = draw(st.sampled_from((37, 38)))
            households.append([(code, float(draw(st.integers(0, 99))))])
            kinds.append(draw(st.sampled_from((2, 3))))
            continue
        members = [(20, float(draw(st.integers(15, 99))))]
        if draw(st.booleans()):
            members.append(
                (draw(st.sampled_from((21, 23))), float(draw(st.integers(15, 99))))
            )
        members.extend(
            (code, float(age))
            for code, age in draw(
                st.lists(
                    st.tuples(st.sampled_from(OTHER_MEMBER_CODES), st.integers(0, 99)),
                    max_size=3,
                )
            )
        )
        households.append(members)
        kinds.append(1)
    if all(
        kind in (2, 3) and members[0][1] < 15
        for members, kind in zip(households, kinds, strict=True)
    ):
        # Frame.select keeps at least one person; a spine of nothing but
        # group-quarters children is outside the domain the drop runs on.
        households.append([(20, 40.0)])
        kinds.append(1)
    return households, kinds


@settings(max_examples=150, deadline=None)
@given(spine=_spine())
def test_the_drop_removes_exactly_the_group_quarters_children(spine):
    households, kinds = spine
    frame = acs_frame(households, typehugq=kinds)
    person = frame.table("person")
    kind_by_household = dict(zip(range(1, len(kinds) + 1), kinds, strict=True))
    person_kind = person["person_household_id"].map(kind_by_household)
    expected_dropped = set(
        person.loc[person_kind.isin((2, 3)) & (person["age"] < 15), "person_id"]
    )

    result = without_acs_group_quarters_children(frame)

    kept = set(result.frame.table("person")["person_id"])
    assert set(person["person_id"]) - kept == expected_dropped
    # Housing units are never touched.
    assert set(person.loc[person_kind.eq(1), "person_id"]) <= kept
    # Weight is only ever removed with a dropped household, never moved.
    before = frame.weights_for("household").values.sum()
    after = result.frame.weights_for("household").values.sum()
    assert after + result.provenance["household_weight_dropped"] == pytest.approx(
        before
    )
    assert result.provenance["persons_dropped"] == len(expected_dropped)


@settings(max_examples=150, deadline=None)
@given(spine=_spine())
def test_after_the_drop_no_unit_lacks_a_classified_adult(spine):
    households, kinds = spine

    dropped = without_acs_group_quarters_children(
        acs_frame(households, typehugq=kinds)
    ).frame
    derived = with_acs_spm_independence_role(dropped)

    assert derived.provenance["group_quarters_units_without_classified_adult"] == 0
    assert check_spm_composition(derived.frame).status == "PASS"
    assert isinstance(dropped.table("person"), pd.DataFrame)
