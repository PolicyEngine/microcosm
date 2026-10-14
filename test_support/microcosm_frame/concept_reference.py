"""A row-by-row reference for ``ConceptMapping.encode``, for differential tests.

The production encoder is vectorized and resolves pointers through row
positions. This one walks records one at a time with plain dictionaries,
straight from each transform's documented meaning, so a differential test
that compares the two catches a transform that computes the wrong thing
(the wrong share leaf, a shared take-up key, a dropped predicate clause,
allocation to every member) even where no round trip can see it.
"""

# ruff: noqa: F401

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from microcosm.frame.concept_mapping import (
    AllocateToReferencePerson,
    ConceptMapping,
    CoresidentChildCount,
    Fraction,
    Identity,
    InputBinding,
    Positive,
    Predicate,
    Product,
    Recode,
    RelationshipRole,
    Role,
    Scale,
    ScaledSum,
    Share,
    Sum,
    TakeUpThreshold,
)
from microcosm.frame.concepts import CONCEPT_BY_ID, derive_take_up_draws


def _present(value: object) -> bool:
    return value is not None and not pd.isna(value)


def reference_values(
    binding: InputBinding,
    tables: Mapping[str, pd.DataFrame],
    shares: Mapping[str, float],
    rates: Mapping[str, float],
) -> list[object]:
    """What ``binding`` should write, computed one record at a time."""

    persons = tables["person"].to_dict("records")
    households = {
        row["household_id"]: row for row in tables["household"].to_dict("records")
    }
    by_id = {row["person_id"]: row for row in persons}
    transform = binding.transform

    def value(row: dict, concept_id: str) -> object:
        item = CONCEPT_BY_ID[concept_id]
        if item.entity == "person":
            return row[item.name]
        household = (
            row if "person_id" not in row else households[row["person_household_id"]]
        )
        return household[item.name]

    def reference_of(row: dict) -> object:
        return households[row["person_household_id"]]["reference_person_id"]

    def is_parent_of(row: dict, child: dict, max_age: int | None) -> bool:
        if max_age is not None and child["age"] > max_age:
            return False
        return any(
            _present(child[column]) and child[column] == row["person_id"]
            for column in ("parent_1_person_id", "parent_2_person_id")
        )

    rows = (
        persons
        if binding.concept_entity == "person"
        else list(tables["household"].to_dict("records"))
    )
    out: list[object] = []
    for row in rows:
        concepts = binding.concepts
        if isinstance(transform, Identity):
            out.append(value(row, concepts[0]))
        elif isinstance(transform, Recode):
            out.append(dict(transform.pairs)[value(row, concepts[0])])
        elif isinstance(transform, Share):
            share = shares[transform.parameter]
            factor = (1.0 - share) if transform.complement else share
            out.append(float(value(row, concepts[0])) * factor)
        elif isinstance(transform, Fraction):
            out.append(float(value(row, concepts[0])) * shares[transform.parameter])
        elif isinstance(transform, Scale):
            out.append(float(value(row, concepts[0])) * transform.factor)
        elif isinstance(transform, Positive):
            out.append(float(value(row, concepts[0])) > 0)
        elif isinstance(transform, Sum):
            out.append(sum(float(value(row, concept_id)) for concept_id in concepts))
        elif isinstance(transform, ScaledSum):
            total = sum(float(value(row, concept_id)) for concept_id in concepts)
            out.append(total * transform.factor)
        elif isinstance(transform, Product):
            first, second = (float(value(row, concept_id)) for concept_id in concepts)
            out.append(first * second)
        elif isinstance(transform, AllocateToReferencePerson):
            household = households[row["person_household_id"]]
            name = CONCEPT_BY_ID[concepts[0]].name
            mine = row["person_id"] == household["reference_person_id"]
            out.append(float(household[name]) if mine else 0.0)
        elif isinstance(transform, Predicate):
            out.append(
                all(
                    value(row, concept_id) in allowed
                    for concept_id, allowed in transform.clauses
                )
            )
        elif isinstance(transform, RelationshipRole):
            role = transform.role
            if role is Role.REFERENCE_PERSON:
                out.append(row["person_id"] == reference_of(row))
            elif role in (Role.HAS_PARTNER, Role.NO_PARTNER):
                has = _present(row["partner_person_id"])
                out.append(has if role is Role.HAS_PARTNER else not has)
            elif role is Role.PARENT_OF_CORESIDENT_CHILD:
                out.append(
                    any(
                        is_parent_of(row, child, transform.max_child_age)
                        for child in persons
                    )
                )
            else:
                head = by_id.get(reference_of(row))
                partner = head["partner_person_id"] if head is not None else None
                flag = _present(partner) and partner == row["person_id"]
                if role is Role.UNMARRIED_PARTNER_OF_REFERENCE_PERSON:
                    flag = flag and row["legal_marital_status"] != "married"
                out.append(flag)
        elif isinstance(transform, CoresidentChildCount):
            out.append(
                sum(
                    1
                    for child in persons
                    for column in ("parent_1_person_id", "parent_2_person_id")
                    if _present(child[column])
                    and child[column] == row["person_id"]
                    and (transform.max_age is None or child["age"] <= transform.max_age)
                )
            )
        elif isinstance(transform, TakeUpThreshold):
            seed = np.asarray([value(row, concepts[0])], dtype=np.float64)
            draw = derive_take_up_draws(seed, transform.program)[0]
            out.append(bool(draw < rates[transform.program]))
        else:  # pragma: no cover - a new transform needs a reference here
            raise AssertionError(f"no reference for {transform!r}")
    return out


def assert_encode_matches_reference(
    mapping: ConceptMapping,
    tables: Mapping[str, pd.DataFrame],
    shares: Mapping[str, float],
    rates: Mapping[str, float],
) -> int:
    """Assert every executable binding's column equals the reference.

    Returns how many bindings were compared.
    """

    encoded = mapping.encode(tables, shares=shares, take_up_rates=rates).tables
    compared = 0
    for binding in mapping.bindings:
        if not mapping.is_executable(binding):
            continue
        column = encoded[binding.concept_entity].get(binding.engine_input)
        assert column is not None, binding.engine_input
        expected = reference_values(binding, tables, shares, rates)
        actual = column.tolist()
        assert len(actual) == len(expected), binding.engine_input
        for got, want in zip(actual, expected, strict=True):
            if isinstance(want, float):
                assert np.isclose(float(got), want, rtol=1e-12, atol=1e-9), (
                    binding.engine_input,
                    got,
                    want,
                )
            else:
                assert got == want, (binding.engine_input, got, want)
        compared += 1
    return compared


__all__ = [name for name in globals() if not name.startswith("__")]
