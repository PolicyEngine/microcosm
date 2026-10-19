"""Properties of the ACS SPM independence role over generated ACS spines.

Each test draws valid ACS households (housing units with exactly one
RELSHIPP=20 reference person, at most one spouse and any other members, or
one-person group-quarters records) and checks, for every such input:

- the rule: the role is a complete numpy Boolean equal to the RELSHIPP rule,
  whatever the row order;
- age blindness: changing any age (keeping heads and spouses 15 or older)
  never changes the role;
- composition: only group-quarters persons under 15 are left without a
  classified adult, and the independent composition checker agrees;
- differential: the role equals the loader's own CPS relationship encoding
  (``A_EXPRRP`` 1-4, or a group-quarters record's sole person), the analog the
  ASEC validation compared against the Census SPM fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from microcosm.build.us_runtime import acs_pums  # noqa: E402
from microcosm.build.us_runtime.acs_inputs import (  # noqa: E402
    with_acs_spm_independence_role,
)
from microcosm.build.us_runtime.spm_composition import (  # noqa: E402
    check_spm_composition,
)
from microcosm.build.us_runtime.spm_role_source import NATIVE_SPM_ROLE  # noqa: E402
from test_support.microcosm_build.us_acs_spm_role import (  # noqa: E402
    OTHER_MEMBER_CODES,
    ROLE_CODES,
    acs_frame,
    role_by_person,
)


@st.composite
def _households(draw) -> list[list[tuple[int, float]]]:
    """Valid ACS households: housing units with one head, or one-person GQ."""

    households = []
    for _ in range(draw(st.integers(min_value=1, max_value=8))):
        if draw(st.booleans()) and draw(st.booleans()):
            code = draw(st.sampled_from((37, 38)))
            households.append([(code, draw(st.integers(0, 99)))])
            continue
        members = [(20, draw(st.integers(15, 99)))]
        if draw(st.booleans()):
            members.append((draw(st.sampled_from((21, 23))), draw(st.integers(15, 99))))
        members.extend(
            draw(
                st.lists(
                    st.tuples(st.sampled_from(OTHER_MEMBER_CODES), st.integers(0, 99)),
                    max_size=5,
                )
            )
        )
        households.append(members)
    return households


def _person_count(households) -> int:
    return sum(len(members) for members in households)


@settings(max_examples=150, deadline=None)
@given(households=_households(), data=st.data())
def test_role_is_the_relshipp_rule_under_any_order(households, data) -> None:
    order = data.draw(st.permutations(range(_person_count(households))))
    typehugq = [
        2 if members[0][0] == 37 else 3 if members[0][0] == 38 else 1
        for members in households
    ]
    frame = acs_frame(households, typehugq=typehugq, order=list(order))

    result = with_acs_spm_independence_role(frame)

    person = result.frame.table("person")
    role = person[NATIVE_SPM_ROLE]
    assert role.dtype == np.dtype(bool)
    assert not role.isna().any()
    assert role.tolist() == person["RELSHIPP"].isin(ROLE_CODES).tolist()
    holders = role.groupby(person["person_spm_unit_id"]).sum()
    assert holders.between(1, 2).all()
    assert result.provenance["role_true_persons"] == int(role.sum())
    assert sum(result.provenance["role_branches"].values()) == int(role.sum())
    unordered = with_acs_spm_independence_role(acs_frame(households))
    assert (
        role_by_person(result.frame)
        .sort_index()
        .equals(role_by_person(unordered.frame).sort_index())
    )


@settings(max_examples=150, deadline=None)
@given(households=_households(), data=st.data())
def test_role_never_reads_age(households, data) -> None:
    # Heads and spouses of housing units stay 15 or older, so no unit is
    # refused; every other age, a group-quarters person's included, is free.
    reaged = [
        [
            (
                code,
                data.draw(
                    st.integers(15, 99) if code in (20, 21, 23) else st.integers(0, 99)
                ),
            )
            for code, _age in members
        ]
        for members in households
    ]

    first = with_acs_spm_independence_role(acs_frame(households))
    second = with_acs_spm_independence_role(acs_frame(reaged))

    assert role_by_person(first.frame).equals(role_by_person(second.frame))


@settings(max_examples=150, deadline=None)
@given(households=_households())
def test_only_group_quarters_children_lack_a_classified_adult(households) -> None:
    result = with_acs_spm_independence_role(acs_frame(households))

    gq_children = sum(
        1 for members in households if members[0][0] in (37, 38) and members[0][1] < 15
    )
    assert result.provenance["group_quarters_units_without_classified_adult"] == (
        gq_children
    )
    composition = check_spm_composition(result.frame)
    assert composition.details["n_units_without_classified_adult"] == gq_children
    assert composition.details["role_source"] == "source_column"


@settings(max_examples=150, deadline=None)
@given(households=_households(), data=st.data())
def test_role_agrees_with_the_loader_cps_relationship_encoding(
    households, data
) -> None:
    """Differential: the ACS rule versus the loader's own ``A_EXPRRP`` recode.

    In CPS codes the ASEC-validated analog is ``A_EXPRRP`` 1/2 (reference
    person) or 3/4 (spouse), plus the sole person of a group-quarters record
    (``A_EXPRRP`` 14 in a household with no reference person).
    """

    frame = acs_frame(households)
    person = frame.table("person")
    raw = pd.DataFrame(
        {
            "household_id": person["person_household_id"].to_numpy(),
            "SPORDER": person.groupby("person_household_id").cumcount() + 1,
            "AGEP": person["age"].astype(int),
            "RELSHIPP": person["RELSHIPP"],
            "SEX": [data.draw(st.sampled_from((1, 2))) for _ in range(len(person))],
        }
    )
    married = raw["RELSHIPP"].isin((21, 23)).groupby(raw["household_id"]).transform(
        "any"
    ) & raw["RELSHIPP"].isin((20, 21, 23))
    raw["MAR"] = np.where(married, 1, 5)
    encoded = acs_pums._with_structural_columns(raw)
    has_reference = (
        encoded["RELSHIPP"].eq(20).groupby(encoded["household_id"]).transform("any")
    )
    sole = encoded.groupby("household_id")["household_id"].transform("size").eq(1)
    oracle = encoded["A_EXPRRP"].isin((1, 2, 3, 4)) | (
        encoded["A_EXPRRP"].eq(14) & sole & ~has_reference
    )

    result = with_acs_spm_independence_role(frame)

    assert result.frame.table("person")[NATIVE_SPM_ROLE].tolist() == oracle.tolist()
