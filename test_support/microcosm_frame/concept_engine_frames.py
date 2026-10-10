"""Run encoded concept inputs through a real rules engine and back.

Builds the smallest valid engine Frame around a mapping's encoded person and
household inputs (one group unit of each kind per household), materializes
the input variables through the adapter, and returns engine tables in the
shape :meth:`ConceptMapping.decode` reads. What comes back is what the
engine actually loaded, so a decode of it tests the mapping against the
engine rather than against itself.
"""

# ruff: noqa: F401

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.concept_mapping import (
    AllocateToReferencePerson,
    ConceptMapping,
    CoresidentChildCount,
    EncodedInputs,
    Fraction,
    Identity,
    InputBinding,
    Positive,
    Predicate,
    Product,
    Recode,
    RelationshipRole,
    Scale,
    ScaledSum,
    Share,
    Sum,
    TakeUpThreshold,
)
from microcosm.frame.concepts import concept
from microcosm.frame.schema import EntitySchema

_IDS = {"person": ("person_id", "person_household_id"), "household": ("household_id",)}


def sort_concept_tables(tables: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Order rows as a Frame requires: households by id, persons within them."""

    return {
        "person": tables["person"]
        .sort_values(["person_household_id", "person_id"])
        .reset_index(drop=True),
        "household": tables["household"]
        .sort_values("household_id")
        .reset_index(drop=True),
    }


def frame_from_inputs(
    encoded: EncodedInputs,
    schema: EntitySchema,
    context: Mapping[str, Mapping[str, object]] | None = None,
) -> Frame:
    """A Frame holding the encoded inputs, one unit per household per group.

    ``context`` adds constant non-concept columns an engine cannot run
    without (the UK loader needs a household ``region``), keyed by entity.
    """

    person = encoded.tables["person"].copy()
    household = encoded.tables["household"].copy()
    for entity, columns in (context or {}).items():
        table = person if entity == "person" else household
        for column, value in columns.items():
            table[column] = value
    tables: dict[str, pd.DataFrame] = {"person": person, "household": household}
    for group in schema.group_entities:
        if group == "household":
            continue
        person[f"person_{group}_id"] = person["person_household_id"].to_numpy()
        tables[group] = pd.DataFrame(
            {f"{group}_id": household["household_id"].to_numpy()}
        )
    weights = {
        "household": Weights(values=np.ones(len(household)), kind=WeightKind.DESIGN)
    }
    return Frame(tables, schema, weights)


def engine_round_trip(
    mapping: ConceptMapping,
    adapter,
    encoded: EncodedInputs,
    schema: EntitySchema,
    period: int,
    context: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Materialize every encoded input through ``adapter``; return its tables."""

    frame = frame_from_inputs(encoded, schema, context)
    names = {
        entity: [column for column in table.columns if column not in _IDS[entity]]
        for entity, table in encoded.tables.items()
    }
    values = adapter.materialize(frame, [*names["person"], *names["household"]], period)
    out: dict[str, pd.DataFrame] = {}
    for entity, table in encoded.tables.items():
        loaded = table.loc[:, list(_IDS[entity])].copy()
        for name in names[entity]:
            loaded[name] = _plain(values[name])
        out[entity] = loaded
    return out


def assert_engine_round_trip(
    mapping: ConceptMapping,
    adapter,
    tables: Mapping[str, pd.DataFrame],
    schema: EntitySchema,
    period: int,
    *,
    shares: Mapping[str, float],
    take_up_rates: Mapping[str, float],
    context: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Encode, load through the real engine, read back, decode, compare.

    Every encoded input must come back as written, and every invertible
    concept must decode to its original value. The PolicyEngine engines store
    amounts as float32, so amounts compare to about seven significant digits.
    """

    tables = sort_concept_tables(tables)
    encoded = mapping.encode(tables, shares=shares, take_up_rates=take_up_rates)
    loaded = engine_round_trip(mapping, adapter, encoded, schema, period, context)
    for entity, table in encoded.tables.items():
        for column in table.columns:
            _assert_float32_close(loaded[entity][column], table[column])
    decoded = mapping.decode(loaded)
    for concept_id in mapping.invertible_concepts():
        item = concept(concept_id)
        _assert_float32_close(
            decoded[item.entity][item.name], tables[item.entity][item.name]
        )


def _assert_float32_close(actual: pd.Series, expected: pd.Series) -> None:
    if expected.dtype.kind in "fiub" and actual.dtype.kind in "fiub":
        np.testing.assert_allclose(
            actual.to_numpy(dtype=np.float64),
            expected.to_numpy(dtype=np.float64),
            rtol=2e-7,
            atol=1e-2,
            err_msg=str(expected.name),
        )
    else:
        assert actual.tolist() == expected.tolist(), expected.name


def expected_input_dtypes(binding: InputBinding) -> set[str]:
    """The engine dtype kinds a binding's transform can produce."""

    transform = binding.transform
    if isinstance(transform, Predicate | RelationshipRole | TakeUpThreshold | Positive):
        return {"bool"}
    if isinstance(transform, CoresidentChildCount):
        return {"int"}
    if isinstance(
        transform,
        Share
        | Fraction
        | Scale
        | Sum
        | ScaledSum
        | Product
        | AllocateToReferencePerson,
    ):
        return {"float"}
    if isinstance(transform, Recode):
        images = {engine_value for _, engine_value in transform.pairs}
        return {"bool"} if images <= {True, False} else {"str"}
    assert isinstance(transform, Identity), transform
    source = concept(binding.concepts[0]).dtype
    return {"int", "float"} if source in ("int", "float") else {source}


def _plain(values: np.ndarray) -> np.ndarray:
    """Engine arrays as plain NumPy: enum members become their names."""

    values = np.asarray(values)
    if values.dtype == object:
        return np.asarray(
            [getattr(value, "name", value) for value in values.tolist()], dtype=object
        )
    return values


__all__ = [name for name in globals() if not name.startswith("__")]
