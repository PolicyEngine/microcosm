"""Invented parity for the streamed in-process population seal.

Every fixture here is invented in this file: no genuine microdata, no source
admission, no store, archive or native read, and no tax engine.  The property
under test is narrow and exact — streaming the seal's parts into the unchanged
``_digest`` must produce the byte-identical digest that the pre-change expanded
tuple produced — so each test compares the live implementation against
``_expanded_parts`` below, which is a verbatim copy of the pre-change tuple.
Mutation coverage is here too, because an optimization that preserved a digest
by *losing* a part would also pass a parity-only test.
"""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_full_puf_enrichment as placement
from microcosm.frame import EntitySchema, Frame, MassChangeRecord, WeightKind, Weights
from microcosm.graph import store as store_ops
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import MassRecord, Population


def _expanded_parts(population):
    """The exact pre-change tuple: every part expanded before hashing."""

    placement.require(type(population) is Population, "FULL_PUF_POPULATION_REQUIRED")
    frame = population.frame
    return (
        canonical_json(
            {
                "schema": asdict(frame.schema),
                "entities": frame.entities,
                "links": frame.links,
                "metadata": store_ops._encode_frame_metadata(frame.metadata),
                "mass_log": [asdict(r) for r in frame.mass_log],
                "version": population.version,
                "owners": sorted(population.owners.items()),
                "weight_kind": tuple(population.weight_kind.items()),
                "mass_ledger": [asdict(r) for r in population.mass_ledger],
            }
        ),
        *(placement._table_stamp(frame.table(e)).encode() for e in frame.entities),
        *placement._axis_parts(frame.strata.index),
        canonical_json(store_ops._axis_name_payload(frame.strata.name)),
        *placement._series_parts(frame.strata),
        *(
            canonical_json(
                (
                    e,
                    frame.weights_for(e).kind,
                    str(frame.weights_for(e).values.dtype),
                    frame.weights_for(e).values.shape,
                )
            )
            + frame.weights_for(e).values.tobytes()
            for e in frame.weighted_entities
        ),
        *(
            canonical_json((e, str(a.dtype), a.shape)) + a.tobytes()
            for e, a in population.design_weights.items()
        ),
    )


def _expanded_population_stamp(population):
    return placement._digest(_expanded_parts(population))


def _object_array(values):
    """An object ndarray built element-wise, so numpy never re-types a leaf."""

    array = np.empty(len(values), dtype=object)
    for position, value in enumerate(values):
        array[position] = value
    return array


def _person_table(
    *,
    person_id=(1, 2, 3, 4),
    household_of=(10, 10, 20, 30),
    amount=(-0.0, 1.5, float("inf"), 2.0),
    hidden=(7, 8, 9, 10),
    mask=(False, True, False, True),
    text=("a", "b", "c", "d"),
    leaves=("x", 5, None, 2.5),
    index=None,
):
    table = pd.DataFrame(
        {
            "person_id": np.array(person_id, dtype=np.int64),
            "person_household_id": np.array(household_of, dtype=np.int64),
            "amount": np.array(amount, dtype=np.float64),
            # Hidden bytes under the null mask are part of this physical seal.
            "nullable": pd.arrays.IntegerArray(
                np.array(hidden, dtype=np.int64), np.array(mask, dtype=np.bool_)
            ),
            "text": pd.array(
                list(text), dtype=pd.StringDtype(storage="python", na_value=pd.NA)
            ),
        },
        index=pd.RangeIndex(len(person_id)) if index is None else index,
    )
    table["leaf"] = _object_array(leaves)
    return table


def _household_table(rent=(0.5, 1.5, 2.5)):
    return pd.DataFrame(
        {
            "household_id": np.array([10, 20, 30], dtype=np.int64),
            "rent": np.array(rent, dtype=np.float64),
        }
    )


def _strata(labels=("s1", "s1", "s2", "s2"), index=None):
    return pd.Series(
        _object_array(labels),
        index=pd.RangeIndex(len(labels)) if index is None else index,
    )


def _mass_log():
    return (
        MassChangeRecord(
            entity="household",
            old_total=6.0,
            new_total=6.0,
            declared_factor=None,
            reason="invented",
        ),
    )


def _mass_ledger(after_total=4.0):
    return (
        MassRecord(
            node_id="invented_node",
            operation="rewrite",
            policy="preserve",
            before_total=4.0,
            after_total=after_total,
            before_by_stratum=(("s1", 2.0), ("s2", 2.0)),
            after_by_stratum=(("s1", 2.0), ("s2", 2.0)),
            entity="person",
        ),
    )


def _frame(
    *,
    person=None,
    household=None,
    strata=None,
    weights=None,
    metadata=None,
    mass_log=None,
):
    return Frame(
        {
            "person": _person_table() if person is None else person,
            "household": _household_table() if household is None else household,
        },
        EntitySchema(group_entities=("household",)),
        (
            {"household": Weights(np.array([1.0, 2.0, 3.0]), WeightKind.DESIGN)}
            if weights is None
            else weights
        ),
        _strata() if strata is None else strata,
        mass_log=_mass_log() if mass_log is None else mass_log,
        metadata={"note": "invented"} if metadata is None else metadata,
    )


def _population(*, frame=None, version="v1", design=None, mass_ledger=None):
    return Population.from_frame(
        _frame() if frame is None else frame,
        version,
        mass_ledger=_mass_ledger() if mass_ledger is None else mass_ledger,
        design_weights=(
            {"person": np.array([1.0, 2.0, 3.0, 4.0])} if design is None else design
        ),
    )


def _empty_population():
    person = pd.DataFrame(
        {
            "person_id": np.array([], dtype=np.int64),
            "person_household_id": np.array([], dtype=np.int64),
            "amount": np.array([], dtype=np.float64),
            "nullable": pd.arrays.IntegerArray(
                np.array([], dtype=np.int64), np.array([], dtype=np.bool_)
            ),
            "text": pd.array(
                [], dtype=pd.StringDtype(storage="python", na_value=pd.NA)
            ),
        }
    )
    person["leaf"] = _object_array(())
    household = pd.DataFrame(
        {
            "household_id": np.array([], dtype=np.int64),
            "rent": np.array([], dtype=np.float64),
        }
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {},
        pd.Series(_object_array(()), index=person.index),
    )
    return Population.from_frame(frame, "v1")


def _labelled_index_population():
    index = pd.Index([11, 12, 13, 14], name="row")
    return _population(
        frame=_frame(person=_person_table(index=index), strata=_strata(index=index))
    )


def _reversed_row_population():
    return _population(
        frame=_frame(
            person=_person_table(
                person_id=(4, 3, 2, 1),
                household_of=(30, 20, 10, 10),
                amount=(2.0, float("inf"), 1.5, -0.0),
                hidden=(10, 9, 8, 7),
                mask=(True, False, True, False),
                text=("d", "c", "b", "a"),
                leaves=(2.5, None, 5, "x"),
            ),
            strata=_strata(labels=("s2", "s2", "s1", "s1")),
        ),
        design={"person": np.array([4.0, 3.0, 2.0, 1.0])},
    )


def _no_design_population():
    return _population(design={})


def _wide_object_strata_population():
    """The retention case the streaming change targets: many object leaves."""

    rows = 512
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, rows + 1, dtype=np.int64),
            "person_household_id": np.arange(1, rows + 1, dtype=np.int64),
            "amount": np.linspace(-1.0, 1.0, rows),
        }
    )
    household = pd.DataFrame({"household_id": np.arange(1, rows + 1, dtype=np.int64)})
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.ones(rows), WeightKind.DESIGN)},
        pd.Series(_object_array([f"s{position}" for position in range(rows)])),
    )
    return Population.from_frame(frame, "v1")


_POPULATIONS = {
    "default": _population,
    "empty": _empty_population,
    "labelled-index": _labelled_index_population,
    "reversed-rows": _reversed_row_population,
    "no-design-weights": _no_design_population,
    "wide-object-strata": _wide_object_strata_population,
}


@pytest.mark.parametrize("name", sorted(_POPULATIONS))
def test_streamed_population_stamp_matches_the_expanded_tuple(name):
    population = _POPULATIONS[name]()
    assert placement._population_stamp(population) == _expanded_population_stamp(
        population
    )


@pytest.mark.parametrize("name", sorted(_POPULATIONS))
def test_streamed_parts_are_the_expanded_parts_part_for_part(name, monkeypatch):
    """Same parts, same order, same bytes — and never materialized at once."""

    population = _POPULATIONS[name]()
    captured = {}
    original = placement._digest

    def recording_digest(parts):
        captured["one_shot"] = iter(parts) is parts
        captured["materialized"] = isinstance(parts, (tuple, list, set, frozenset))
        materialized = tuple(parts)
        captured["parts"] = materialized
        return original(materialized)

    monkeypatch.setattr(placement, "_digest", recording_digest)
    digest = placement._population_stamp(population)
    monkeypatch.undo()

    assert captured["one_shot"] is True
    assert captured["materialized"] is False
    assert captured["parts"] == _expanded_parts(population)
    assert digest == _expanded_population_stamp(population)


def _mutations():
    """(label, population) pairs that must every one move the seal."""

    yield (
        "visible-value",
        _population(
            frame=_frame(person=_person_table(amount=(-0.0, 1.5, float("inf"), 2.5)))
        ),
    )
    yield (
        "negative-zero",
        _population(
            frame=_frame(person=_person_table(amount=(0.0, 1.5, float("inf"), 2.0)))
        ),
    )
    yield (
        "infinity-sign",
        _population(
            frame=_frame(person=_person_table(amount=(-0.0, 1.5, float("-inf"), 2.0)))
        ),
    )
    yield (
        "hidden-masked-data",
        _population(frame=_frame(person=_person_table(hidden=(7, 999, 9, 10)))),
    )
    yield (
        "null-mask",
        _population(frame=_frame(person=_person_table(mask=(False, True, True, True)))),
    )
    yield (
        "string-value",
        _population(frame=_frame(person=_person_table(text=("a", "b", "c", "z")))),
    )
    yield (
        "object-leaf-type",
        _population(frame=_frame(person=_person_table(leaves=("x", 5.0, None, 2.5)))),
    )
    yield "row-order", _reversed_row_population()
    yield (
        "strata-label",
        _population(frame=_frame(strata=_strata(labels=("s1", "s1", "s2", "s3")))),
    )
    yield (
        "strata-order",
        _population(frame=_frame(strata=_strata(labels=("s2", "s1", "s2", "s1")))),
    )
    yield "axis-labels", _labelled_index_population()
    yield (
        "group-table-value",
        _population(frame=_frame(household=_household_table(rent=(0.5, 1.5, 3.5)))),
    )
    yield (
        "weights",
        _population(
            frame=_frame(
                weights={
                    "household": Weights(np.array([1.0, 2.0, 4.0]), WeightKind.DESIGN)
                }
            )
        ),
    )
    yield (
        "weight-kind",
        _population(
            frame=_frame(
                weights={
                    "household": Weights(
                        np.array([1.0, 2.0, 3.0]), WeightKind.IMPORTANCE
                    )
                }
            )
        ),
    )
    yield "metadata", _population(frame=_frame(metadata={"note": "changed"}))
    yield (
        "frame-mass-log",
        _population(
            frame=_frame(mass_log=(replace(_mass_log()[0], reason="different"),))
        ),
    )
    yield (
        "design-weights",
        _population(design={"person": np.array([1.0, 2.0, 3.0, 5.0])}),
    )
    yield "no-design-weights", _no_design_population()
    yield "version", _population(version="v2")
    yield "mass-ledger", _population(mass_ledger=_mass_ledger(after_total=5.0))


@pytest.mark.parametrize("label,mutated", list(_mutations()), ids=lambda value: value)
def test_every_mutation_moves_the_streamed_and_expanded_stamps_together(label, mutated):
    base = _population()
    baseline = placement._population_stamp(base)
    assert baseline == _expanded_population_stamp(base)

    moved = placement._population_stamp(mutated)
    assert moved == _expanded_population_stamp(mutated)
    assert moved != baseline, label


def test_the_stamp_still_refuses_a_bare_frame():
    with pytest.raises(ValueError, match="FULL_PUF_POPULATION_REQUIRED"):
        placement._population_stamp(_frame())
