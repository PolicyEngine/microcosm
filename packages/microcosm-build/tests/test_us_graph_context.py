"""Production graph metadata must follow the exact population it describes."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime.graph_context import (
    US_FRAME_CONTEXT_TYPE,
    encode_us_frame_context,
    us_frame_from_context,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import ArtifactValue, ContentStore, KernelContext, Node, Numeric
from microcosm.graph.canonical import canonical_json
from microcosm.graph.kernel import NumericScope


def _frame() -> Frame:
    tables = {
        entity: pd.DataFrame({US_SCHEMA.entity_id_column(entity): [10, 20]})
        for entity in US_SCHEMA.entities
    }
    for entity in US_SCHEMA.group_entities:
        tables["person"][f"person_{entity}_id"] = [10, 20]
    tables["person"]["age"] = [30, 60]
    tables["person"]["income"] = [0.0, np.nan]
    tables["household"]["state_fips"] = [6, 36]
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([2.0, 3.0]), WeightKind.DESIGN)},
        pd.Series(["asec", "acs"]),
        metadata={"us_spine_assembly_manifest": {"native": ["age"], "revision": 1}},
    )


def _context(frame: Frame, *, payload: bytes | None = None) -> KernelContext:
    artifact = ArtifactValue(
        encode_us_frame_context(frame) if payload is None else payload,
        US_FRAME_CONTEXT_TYPE,
        "a" * 64,
        "b" * 64,
        NumericScope(Numeric.BITWISE),
    )
    return KernelContext(
        node=Node(id="stage", population="us", kernel="test.stage@1"),
        tables={entity: frame.table(entity) for entity in frame.entities},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        params={},
        rng=np.random.default_rng(0),
        artifacts={"frame_context": artifact},
    )


def test_round_trip_preserves_metadata_order_and_observed_zero_vs_null():
    source = _frame()
    restored = us_frame_from_context(_context(source))
    # Metadata mappings have no normative key order; Frame's frozen mapping
    # representation retains insertion order, whereas JSON canonicalizes it.
    assert canonical_json(restored.metadata) == canonical_json(source.metadata)
    for entity in source.entities:
        pd.testing.assert_frame_equal(restored.table(entity), source.table(entity))
    pd.testing.assert_series_equal(restored.strata, source.strata)
    assert restored.weights_for("household").kind is WeightKind.DESIGN
    np.testing.assert_array_equal(
        restored.weights_for("household").values,
        source.weights_for("household").values,
    )
    assert restored.table("person").loc[0, "income"] == 0.0
    assert pd.isna(restored.table("person").loc[1, "income"])


def test_context_artifact_contains_no_population_values_or_weights():
    document = json.loads(encode_us_frame_context(_frame()))
    assert "tables" not in document
    assert "weights" not in document
    assert "strata" not in document
    assert document["entities"]["person"]["rows"] == 2
    assert "values" not in document["entities"]["person"]


@pytest.mark.parametrize("group", US_SCHEMA.group_entities)
@pytest.mark.parametrize("index_kind", ("offset", "named", "array"))
def test_context_v1_refuses_unsupported_group_index_before_publishing(
    group, index_kind
):
    source = _frame()
    tables = {entity: source.table(entity).copy() for entity in source.entities}
    indices = {
        "offset": pd.RangeIndex(10, 12),
        "named": pd.RangeIndex(2, name="original_group_row"),
        "array": pd.Index(np.arange(2, dtype=np.int64)),
    }
    tables[group].index = indices[index_kind]
    # This is a valid Frame. Its group pandas index is not its entity ID.
    frame = Frame(
        tables,
        source.schema,
        {"household": source.weights_for("household")},
        source.strata,
    )
    with pytest.raises(
        ValueError, match=f"{group} requires a default unnamed RangeIndex"
    ):
        encode_us_frame_context(frame)


def test_canonical_group_indices_survive_real_graph_store_codec(tmp_path):
    source = _frame()
    source.person.index = pd.Index([91, 44], name="original_person_row")
    source = Frame(
        {entity: source.table(entity) for entity in source.entities},
        source.schema,
        {"household": source.weights_for("household")},
        source.strata.set_axis(source.person.index),
    )
    store = ContentStore(tmp_path)
    store.put_frame("f" * 64, source)
    restored = store.load_frame("f" * 64)
    assert encode_us_frame_context(restored) == encode_us_frame_context(source)
    for entity in source.entities:
        pd.testing.assert_frame_equal(restored.table(entity), source.table(entity))


def test_metadata_changes_change_context_bytes():
    first = _frame()
    second = Frame(
        {entity: first.table(entity) for entity in first.entities},
        first.schema,
        {"household": first.weights_for("household")},
        first.strata,
        metadata={"us_spine_assembly_manifest": {"native": ["age"], "revision": 2}},
    )
    assert encode_us_frame_context(first) != encode_us_frame_context(second)


def test_reconstruction_cannot_read_undeclared_population_columns():
    source = _frame()
    context = _context(source)
    tables = dict(context.tables)
    tables["person"] = tables["person"].drop(columns="income")
    restored = us_frame_from_context(replace(context, tables=tables))
    assert "income" not in restored.table("person")
    assert "income" in source.table("person")


def test_row_reordering_refuses_context_from_another_population_order():
    context = _context(_frame())
    tables = dict(context.tables)
    tables["household"] = tables["household"].iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="household.*identity"):
        us_frame_from_context(replace(context, tables=tables))


def test_wrong_artifact_type_refused():
    from microcosm.graph import ArtifactType

    context = _context(_frame())
    artifact = replace(
        context.artifacts["frame_context"], type=ArtifactType("wrong", 1)
    )
    with pytest.raises(ValueError, match="type"):
        us_frame_from_context(replace(context, artifacts={"frame_context": artifact}))


@pytest.mark.parametrize("mutation", ["schema", "extra", "unknown_entity", "nan"])
def test_malformed_payload_refused(mutation):
    frame = _frame()
    document = json.loads(encode_us_frame_context(frame))
    if mutation == "schema":
        document["schema_version"] = 999
    elif mutation == "extra":
        document["unbound_metadata"] = {}
    elif mutation == "unknown_entity":
        document["entities"]["unrelated"] = document["entities"]["person"]
    else:
        document["metadata"]["unbounded"] = float("nan")
    with pytest.raises(ValueError):
        us_frame_from_context(_context(frame, payload=json.dumps(document).encode()))


def test_restored_metadata_and_tables_do_not_mutate_input_context():
    source = _frame()
    context = _context(source)
    restored = us_frame_from_context(context)
    restored.table("person").loc[0, "age"] = 99
    with pytest.raises(AttributeError):
        restored.metadata["us_spine_assembly_manifest"]["native"].append("income")
    assert source.table("person").loc[0, "age"] == 30
    assert source.metadata["us_spine_assembly_manifest"]["native"] == ("age",)


def test_separately_constructed_contexts_are_byte_identical():
    assert encode_us_frame_context(_frame()) == encode_us_frame_context(_frame())


@pytest.mark.parametrize("key", ["timings", "hostname", "path", "run_id"])
def test_run_observations_cannot_enter_normative_context(key):
    source = _frame()
    frame = Frame(
        {entity: source.table(entity) for entity in source.entities},
        source.schema,
        {"household": source.weights_for("household")},
        source.strata,
        metadata={"us_spine_assembly_manifest": {key: "run-specific"}},
    )
    with pytest.raises(ValueError, match="run-level"):
        encode_us_frame_context(frame)


def test_unknown_metadata_cannot_silently_rekey_nodes():
    source = _frame()
    frame = Frame(
        {entity: source.table(entity) for entity in source.entities},
        source.schema,
        {"household": source.weights_for("household")},
        source.strata,
        metadata={"new_unaudited_receipt": {}},
    )
    with pytest.raises(ValueError, match="undeclared metadata"):
        encode_us_frame_context(frame)


def test_mass_history_and_all_resolved_entity_weights_round_trip():
    from microcosm.frame import MassChangeRecord

    source = _frame()
    frame = Frame(
        {entity: source.table(entity) for entity in source.entities},
        source.schema,
        {"household": source.weights_for("household")},
        source.strata,
        mass_log=(MassChangeRecord("household", 10.0, 5.0, 0.5, "harmonize"),),
        metadata=source.metadata,
    )
    restored = us_frame_from_context(_context(frame))
    assert restored.mass_log == frame.mass_log
    assert restored.weighted_entities == frame.weighted_entities
    for entity in frame.entities:
        actual = restored.resolve_weights(entity)
        expected = frame.resolve_weights(entity)
        assert actual.kind == expected.kind
        np.testing.assert_array_equal(actual.values, expected.values)
