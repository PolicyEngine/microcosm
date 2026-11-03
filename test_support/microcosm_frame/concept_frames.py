"""Hypothesis strategies for valid concept frames, shared across test groups.

A generated frame satisfies every contract
:func:`microcosm.frame.concepts.validate_concept_tables` checks: ids are
unique and non-contiguous (so nothing may assume positional ids), every
household has one reference person among its members, partners point at
each other within the household, parents are distinct co-resident persons
who are not the partner, the parent graph is acyclic, and the cross-concept
consistency rules hold. Values are drawn from each concept's declared
domain and bounds, so a test built on these frames exercises the schema as
declared rather than a hand-picked example.
"""

# ruff: noqa: F401

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st

from microcosm.frame.concepts import (
    ATTAINMENT_DOMAIN,
    CONCEPTS,
    ENROLLMENT_DOMAIN,
    MARITAL_STATUS_DOMAIN,
    RENTED_TENURES,
    SEX_DOMAIN,
    TENURE_DOMAIN,
    Concept,
)

_MONEY = 1e7


def _amount(draw, item: Concept) -> float:
    signed = item.monetary is not None and item.monetary.signed
    value = draw(
        st.floats(
            min_value=-_MONEY if signed else 0.0,
            max_value=_MONEY,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    # Round to cents: survey amounts are whole cents, and it keeps share
    # round trips well inside float tolerance.
    return round(value, 2)


@st.composite
def concept_frames(
    draw,
    *,
    min_households: int = 1,
    max_households: int = 4,
    max_members: int = 5,
    min_id: int = 0,
) -> dict[str, pd.DataFrame]:
    """A valid concept frame with every concept column present.

    Person and household ids are drawn from ``[min_id, 10**9]``. The
    concept-frame contract accepts any unique integer id, so a negative
    ``min_id`` checks that no consumer reads an id as a sentinel such as -1;
    ids near zero are then mixed in, because a draw from a wide range almost
    never yields one exactly.
    """

    ids = st.integers(min_id, 10**9)
    if min_id < 0:
        ids |= st.integers(max(min_id, -3), 3)
    n_households = draw(st.integers(min_households, max_households))
    sizes = [draw(st.integers(1, max_members)) for _ in range(n_households)]
    n_persons = sum(sizes)
    person_ids = draw(
        st.lists(ids, min_size=n_persons, max_size=n_persons, unique=True)
    )
    household_ids = draw(
        st.lists(
            ids,
            min_size=n_households,
            max_size=n_households,
            unique=True,
        )
    )

    rows: list[dict[str, object]] = []
    households: list[dict[str, object]] = []
    cursor = 0
    for household_id, size in zip(household_ids, sizes, strict=True):
        members = person_ids[cursor : cursor + size]
        cursor += size
        # Rank members; parents always outrank children, which keeps the
        # parent graph acyclic by construction.
        order = draw(st.permutations(members))
        rank = {person_id: index for index, person_id in enumerate(order)}
        reference = draw(st.sampled_from(members))
        partner: dict[int, int] = {}
        pool = list(order)
        while len(pool) >= 2 and draw(st.booleans()):
            first, second = pool.pop(0), pool.pop(0)
            partner[first], partner[second] = second, first
        tenure = draw(st.sampled_from(TENURE_DOMAIN))
        household = {
            "household_id": household_id,
            "reference_person_id": reference,
            "tenure": tenure,
        }
        for item in CONCEPTS:
            if item.entity == "household" and item.monetary is not None:
                household[item.name] = _amount(draw, item)
        if tenure not in RENTED_TENURES:
            household["rent"] = 0.0
        if tenure != "owned_with_mortgage":
            household["mortgage_interest"] = 0.0
            household["mortgage_principal"] = 0.0
        households.append(household)

        for person_id in members:
            elders = [
                other
                for other in members
                if rank[other] < rank[person_id] and partner.get(person_id) != other
            ]
            parents = draw(
                st.lists(st.sampled_from(elders), max_size=2, unique=True)
                if elders
                else st.just([])
            )
            enrollment = draw(st.sampled_from(ENROLLMENT_DOMAIN))
            weeks = draw(st.integers(0, 53))
            row: dict[str, object] = {
                "person_id": person_id,
                "person_household_id": household_id,
                "age": draw(st.integers(0, 130)),
                "sex": draw(st.sampled_from(SEX_DOMAIN)),
                "legal_marital_status": draw(st.sampled_from(MARITAL_STATUS_DOMAIN)),
                "partner_person_id": partner.get(person_id),
                "parent_1_person_id": parents[0] if parents else None,
                "parent_2_person_id": parents[1] if len(parents) == 2 else None,
                "weeks_worked": weeks,
                "usual_weekly_hours": 0.0
                if weeks == 0
                else draw(st.floats(min_value=0.5, max_value=168.0, allow_nan=False)),
                "has_disability": draw(st.booleans()),
                "educational_attainment": draw(st.sampled_from(ATTAINMENT_DOMAIN)),
                "education_enrollment": enrollment,
                "enrolled_full_time": enrollment != "not_enrolled"
                and draw(st.booleans()),
                "take_up_seed": draw(
                    st.floats(
                        min_value=0.0,
                        max_value=1.0,
                        exclude_max=True,
                        allow_nan=False,
                    )
                ),
            }
            for item in CONCEPTS:
                if item.entity == "person" and item.monetary is not None:
                    row[item.name] = _amount(draw, item)
            rows.append(row)

    person = pd.DataFrame(rows)
    for column in ("partner_person_id", "parent_1_person_id", "parent_2_person_id"):
        person[column] = pd.array(person[column].tolist(), dtype="Int64")
    person["age"] = person["age"].astype(np.int64)
    person["weeks_worked"] = person["weeks_worked"].astype(np.int64)
    person["has_disability"] = person["has_disability"].astype(bool)
    person["enrolled_full_time"] = person["enrolled_full_time"].astype(bool)
    household = pd.DataFrame(households)
    household["reference_person_id"] = pd.array(
        household["reference_person_id"].tolist(), dtype="Int64"
    )
    return {"person": person, "household": household}


#: A share parameter anywhere on [0, 1], endpoints included.
shares = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


def assert_concepts_round_trip(
    decoded: dict[str, pd.DataFrame],
    original: dict[str, pd.DataFrame],
    concept_ids: tuple[str, ...],
) -> None:
    """Assert every listed concept decoded back to its original values.

    Amounts compare to within float rounding of a share split; everything
    else compares exactly, dtype kind included.
    """

    by_id = {item.id: item for item in CONCEPTS}
    for concept_id in concept_ids:
        item = by_id[concept_id]
        actual = decoded[item.entity][item.name]
        expected = original[item.entity][item.name]
        assert len(actual) == len(expected), concept_id
        if item.dtype == "float":
            np.testing.assert_allclose(
                actual.to_numpy(dtype=np.float64),
                expected.to_numpy(dtype=np.float64),
                rtol=1e-12,
                atol=1e-6,
                err_msg=concept_id,
            )
        else:
            assert actual.tolist() == expected.tolist(), concept_id


__all__ = [name for name in globals() if not name.startswith("__")]
