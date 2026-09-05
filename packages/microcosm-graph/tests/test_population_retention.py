"""Opt-in lazy populations preserve history without retaining every Frame."""

import gc
import importlib.util
import sys
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.frame import Frame
from microcosm.graph import ContentStore, compile_graph, run_graph
from microcosm.graph.manifest import PopulationView

if "_toy" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "_toy", Path(__file__).with_name("_toy.py")
    )
    sys.modules[spec.name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[spec.name])
toy = sys.modules["_toy"]


def run(compiled, sources, store, *, mode="lazy", resume="auto", kernels=None):
    return run_graph(
        compiled,
        sources=sources,
        store=store,
        kernels=kernels or toy.toy_registry(),
        population_retention=mode,
        resume=resume,
    )


def assert_frames_equal(left, right):
    assert left.schema == right.schema
    assert left.entities == right.entities
    assert left.metadata == right.metadata
    assert left.mass_log == right.mass_log
    pd.testing.assert_series_equal(left.strata, right.strata)
    for entity in left.entities:
        pd.testing.assert_frame_equal(left.table(entity), right.table(entity))
    assert left.weighted_entities == right.weighted_entities
    for entity in left.weighted_entities:
        assert left.weights_for(entity).kind == right.weights_for(entity).kind
        np.testing.assert_array_equal(
            left.weights_for(entity).values, right.weights_for(entity).values
        )


@pytest.mark.parametrize("graph_factory", [toy.small_graph, toy.full_graph])
def test_history_keys_and_cache_policies_match_eager(tmp_path, graph_factory):
    compiled, sources = compile_graph(graph_factory()), toy.toy_sources(tmp_path)
    eager_store, lazy_store = (
        ContentStore(tmp_path / "eager"),
        ContentStore(tmp_path / "lazy"),
    )
    for resume in ("auto", "auto", "require", "forbid"):
        eager = run(compiled, sources, eager_store, mode="eager", resume=resume)
        lazy = run(compiled, sources, lazy_store, resume=resume)
        assert eager.key == lazy.key
        assert eager.mass_ledgers == lazy.mass_ledgers
        assert set(eager.populations) == set(lazy.populations)
        for node_id in eager.nodes:
            assert eager.nodes[node_id].hit == lazy.nodes[node_id].hit
            assert (
                eager.nodes[node_id]._content_payload()
                == lazy.nodes[node_id]._content_payload()
            )
        for version in eager.populations:
            assert_frames_equal(eager.population(version), lazy.population(version))


def test_lookup_is_lazy_weakly_memoized_and_not_triggered_by_manifest_operations(
    tmp_path, monkeypatch
):
    store = ContentStore(tmp_path / "cache")
    manifest = run(compile_graph(toy.small_graph()), toy.toy_sources(tmp_path), store)
    loads, original = [], store.load_frame

    def load(*args, **kwargs):
        loads.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "load_frame", load)
    assert tuple(manifest.populations) == ("survey",)
    assert len(manifest.populations) == 1
    assert "survey" in manifest.populations
    assert "missing" not in manifest.populations
    assert manifest.key
    assert manifest.to_json()
    assert replace(manifest, host="different").key == manifest.key
    assert not loads
    view = manifest.population("survey")
    assert type(view) is PopulationView and isinstance(view, Frame)
    assert manifest.populations["survey"] is view
    assert len(loads) == 1
    with pytest.raises(TypeError):
        manifest.populations["survey"] = view
    reference = weakref.ref(view)
    del view
    gc.collect()
    assert reference() is None
    assert manifest.population("survey").person.shape[0] > 0
    assert len(loads) == 2


def test_invalid_retention_refused_before_execution(tmp_path):
    with pytest.raises(ValueError, match="population_retention"):
        run(
            compile_graph(toy.small_graph()),
            {},
            ContentStore(tmp_path / "cache"),
            mode="invalid",
        )


def test_structural_writer_copies_only_one_column_at_a_time(tmp_path, monkeypatch):
    import microcosm.graph.executor as executor

    original = executor._series_for_column
    references, counts = [], []

    def series(*args):
        gc.collect()
        counts.append(sum(ref() is not None for ref in references))
        value = original(*args)
        references.append(weakref.ref(value))
        return value

    monkeypatch.setattr(executor, "_series_for_column", series)
    compiled = compile_graph(toy.small_graph(nodes=(toy.CREATE,)))
    run_graph(
        compiled,
        sources=toy.toy_sources(tmp_path),
        store=ContentStore(tmp_path / "cache"),
        kernels=toy.toy_registry(),
    )
    assert len(counts) > 5
    assert max(counts) == 0


def test_late_branch_nullable_patch_and_unpatched_snapshot_reuse(tmp_path):
    graph = toy.small_graph(
        nodes=(
            toy.CREATE,
            toy.reweight_node("first", base="survey", to_kind="importance"),
            toy.patch_node(
                "patch_income", "income", "float64", 99.0, population="first"
            ),
            toy.absent_node("absent", "unknown", population="first"),
            toy.select_node("child", base="first"),
            toy.select_node("last", base="survey"),
        )
    )
    compiled, sources = compile_graph(graph), toy.toy_sources(tmp_path)
    eager = run(compiled, sources, ContentStore(tmp_path / "eager"), mode="eager")
    lazy = run(compiled, sources, ContentStore(tmp_path / "lazy"))
    assert compiled.order.index("last") > compiled.order.index("first")
    assert tuple(lazy.populations) == tuple(eager.populations)
    for version in eager.populations:
        assert_frames_equal(eager.population(version), lazy.population(version))
    first = lazy.population("first").person
    assert (first.loc[first.is_adult, "income"] == 99.0).all()
    pd.testing.assert_series_equal(
        first.loc[~first.is_adult, "income"],
        lazy.population("survey").person.loc[~first.is_adult, "income"],
    )
    assert lazy.population("first").person.unknown.isna().all()
    assert not (lazy.population("survey").person.income == 99.0).all()
    references = lazy.populations._references
    assert references["survey"].key == lazy.nodes["survey"].frame_key
    assert references["first"].key != lazy.nodes["first"].frame_key


@pytest.mark.parametrize("factory", [toy.entrant_expand_node, toy.entrant_person_node])
def test_expansion_history_retains_strata_weights_and_mass(tmp_path, factory):
    expand, claim = factory()
    compiled = compile_graph(toy.small_graph(nodes=(toy.CREATE, expand, claim)))
    sources = toy.toy_sources(tmp_path)
    eager = run(compiled, sources, ContentStore(tmp_path / "eager"), mode="eager")
    lazy = run(compiled, sources, ContentStore(tmp_path / "lazy"))
    assert eager.key == lazy.key
    assert eager.mass_ledgers == lazy.mass_ledgers
    for version in eager.populations:
        assert_frames_equal(eager.population(version), lazy.population(version))


def test_live_metadata_descriptor_survives_laziness_without_mutating_store_codec(
    tmp_path,
):
    from microcosm.graph import (
        Capabilities,
        Determinism,
        KernelBase,
        KernelResult,
        StructuralDelta,
    )

    class MetadataSource(KernelBase):
        ref = "test.metadata-source@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def run(self, context):
            frame = toy.read_toy_frame(context.sources["survey"])
            return KernelResult(
                frame=Frame(
                    {entity: frame.table(entity) for entity in frame.entities},
                    frame.schema,
                    {
                        entity: frame.weights_for(entity)
                        for entity in frame.weighted_entities
                    },
                    frame.strata,
                    metadata={
                        "source": {
                            "labels": ["original", "local"],
                            "years": {2024, 2025},
                        },
                        "nullable": None,
                    },
                )
            )

    graph = toy.small_graph(
        nodes=(
            replace(toy.CREATE, kernel=MetadataSource.ref),
            toy.derive("value", ("age",), "value"),
        )
    )
    compiled, sources = compile_graph(graph), toy.toy_sources(tmp_path)
    eager_store, lazy_store = (
        ContentStore(tmp_path / "eager"),
        ContentStore(tmp_path / "lazy"),
    )
    registry = toy.toy_registry()
    registry.register(MetadataSource())
    for resume in ("auto", "require", "forbid"):
        eager = run(
            compiled,
            sources,
            eager_store,
            mode="eager",
            resume=resume,
            kernels=registry,
        )
        lazy = run(compiled, sources, lazy_store, resume=resume, kernels=registry)
        left, right = eager.population("survey"), lazy.population("survey")
        assert_frames_equal(left, right)
        if resume != "require":
            assert right.metadata["source"]["labels"] == ("original", "local")
            assert right.metadata["source"]["years"] == frozenset({2024, 2025})
            with pytest.raises(TypeError):
                right.metadata["source"]["labels"] = ()
        else:
            # The pre-existing Frame codec loses source metadata on cache hits.
            # Lazy retention must match that path rather than invent provenance.
            assert not right.metadata


def test_retired_population_objects_are_collectible_on_a_linear_history(
    tmp_path, monkeypatch
):
    from microcosm.graph.population import Population

    nodes = [toy.CREATE]
    base = "survey"
    for i in range(12):
        name = f"version_{i:02}"
        nodes.append(toy.select_node(name, base=base))
        base = name
    compiled, sources = (
        compile_graph(toy.small_graph(nodes=nodes)),
        toy.toy_sources(tmp_path),
    )
    original = Population.__post_init__
    references, counts, table_references, table_counts = [], [], [], []

    def initialize(self):
        original(self)
        references.append(weakref.ref(self))
        table_references.append(weakref.ref(self.frame.table("person")))
        counts.append(sum(ref() is not None for ref in references))
        table_counts.append(
            len({id(ref()) for ref in table_references if ref() is not None})
        )

    monkeypatch.setattr(Population, "__post_init__", initialize)
    eager = run(compiled, sources, ContentStore(tmp_path / "eager"), mode="eager")
    eager_peak, eager_tables = max(counts), max(table_counts)
    references.clear()
    table_references.clear()
    counts.clear()
    table_counts.clear()
    lazy = run(compiled, sources, ContentStore(tmp_path / "lazy"))
    assert eager_peak >= 12 and eager_tables >= 12
    assert max(counts) <= 4 and max(table_counts) <= 4
    gc.collect()
    assert not any(ref() is not None for ref in references)
    assert not any(ref() is not None for ref in table_references)
    assert len(lazy.populations) == len(eager.populations) == 13
    for version in eager.populations:
        assert_frames_equal(eager.population(version), lazy.population(version))


def test_require_can_create_only_missing_auxiliary_attachment(tmp_path):
    import shutil

    store = ContentStore(tmp_path / "cache")
    registry = toy.toy_registry()
    compiled, sources = compile_graph(toy.small_graph()), toy.toy_sources(tmp_path)
    cold = run(compiled, sources, store, kernels=registry)
    key = cold.populations._references["survey"].key
    assert key != cold.nodes["survey"].frame_key
    shutil.rmtree(store.object_path(key))
    calls = toy.total_calls(registry)
    warm = run(compiled, sources, store, resume="require", kernels=registry)
    assert toy.total_calls(registry) == calls
    assert all(node.hit for node in warm.nodes.values())
    assert warm.key == cold.key
    assert store.has(key)
    assert_frames_equal(cold.population("survey"), warm.population("survey"))


def test_lazy_lookup_verifies_payload_and_node_binding(tmp_path):
    from microcosm.graph import StoreCorruptError

    store = ContentStore(tmp_path / "cache")
    manifest = run(compile_graph(toy.small_graph()), toy.toy_sources(tmp_path), store)
    reference = manifest.populations._references["survey"]
    value_file = next(store.object_path(reference.key).rglob("values.npy"))
    value_file.write_bytes(value_file.read_bytes() + b"corruption")
    with pytest.raises(StoreCorruptError):
        manifest.population("survey")


def test_lazy_failed_gate_preserves_only_realized_versions_on_every_cache_path(
    tmp_path,
):
    spec = importlib.util.spec_from_file_location(
        "_retention_gate_fixture",
        Path(__file__).with_name("test_gate_artifact_failures.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    compiled, sources, store, registry, gate, consumer = module.fixture(tmp_path)
    cold = None
    for resume in ("auto", "auto", "require", "forbid"):
        current = run(compiled, sources, store, resume=resume, kernels=registry)
        if cold is None:
            cold = current
        assert current.key == cold.key
        assert tuple(current.populations) == ("survey",)
        assert "age_copy" in current.population("survey").person
        assert "unavailable" not in current.population("survey").person
        assert current.nodes["adults"].receipt["execution"]["state"] == "unreached"
        assert current.nodes["adults"].frame_key is None
        assert consumer.calls == 0
    assert gate.calls == 2


@pytest.mark.parametrize("structural", [False, True])
def test_tolerance_recomputation_snapshots_actual_values_and_metadata(
    tmp_path, structural
):
    import shutil

    from microcosm.graph import (
        Capabilities,
        Determinism,
        KernelBase,
        KernelResult,
        Node,
        Owned,
        Slice,
        StructuralDelta,
    )
    from microcosm.graph.executor import _cache_record_key
    from microcosm.graph.kernel import Numeric, Tolerance

    class Drifting(KernelBase):
        ref = "test.retention-drifting@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC,
            structural=StructuralDelta.CREATE if structural else StructuralDelta.NONE,
            numeric=Numeric.TOLERANCE_BOUND,
            tolerance=Tolerance(atol=0.1),
        )
        value = 1.0

        def implementation_hash(self):
            return "d" * 64

        def run(self, context):
            if structural:
                frame = toy.read_toy_frame(context.sources["survey"])
                tables = {entity: frame.table(entity) for entity in frame.entities}
                tables["person"] = tables["person"].assign(income=self.value)
                return KernelResult(
                    frame=Frame(
                        tables,
                        frame.schema,
                        {
                            entity: frame.weights_for(entity)
                            for entity in frame.weighted_entities
                        },
                        frame.strata,
                        metadata={"emitted": self.value},
                    )
                )
            return KernelResult(
                columns={
                    ("person", "drift"): pd.Series(
                        self.value,
                        index=pd.Index(
                            context.tables["person"].person_id, name="person_id"
                        ),
                        name="drift",
                        dtype="float64",
                    )
                }
            )

    node = (
        replace(toy.CREATE, kernel=Drifting.ref)
        if structural
        else Node(
            "drift",
            Drifting.ref,
            population="survey",
            inputs=(Slice("person", ("age",)),),
            outputs=(Owned("person", "drift", "float64"),),
        )
    )
    nodes = (node,) if structural else (toy.CREATE, node)
    compiled, sources = (
        compile_graph(toy.small_graph(nodes=nodes)),
        toy.toy_sources(tmp_path),
    )
    eager_store, lazy_store = (
        ContentStore(tmp_path / "eager"),
        ContentStore(tmp_path / "lazy"),
    )
    kernel = Drifting()
    registry = toy.toy_registry()
    registry.register(kernel)
    eager = run(compiled, sources, eager_store, mode="eager", kernels=registry)
    lazy = run(compiled, sources, lazy_store, kernels=registry)
    original_key = lazy.key
    for store in (eager_store, lazy_store):
        shutil.rmtree(store.object_path(_cache_record_key(lazy.nodes[node.id].key)))
    kernel.value = 1.05
    eager_new = run(compiled, sources, eager_store, mode="eager", kernels=registry)
    lazy_new = run(compiled, sources, lazy_store, kernels=registry)
    assert lazy_new.key == eager_new.key == original_key
    assert_frames_equal(eager_new.population("survey"), lazy_new.population("survey"))
    target = "income" if structural else "drift"
    assert lazy_new.population("survey").person[target].iloc[0] == 1.05
    assert lazy.population("survey").person[target].iloc[0] == 1.0
    assert_frames_equal(eager.population("survey"), lazy.population("survey"))
    assert (
        lazy_new.populations._references["survey"].key
        != lazy.populations._references["survey"].key
    )


def test_payload_signature_refuses_replacement_even_with_valid_store_metadata(tmp_path):
    from microcosm.graph import StoreCorruptError

    store = ContentStore(tmp_path / "cache")
    compiled, sources = (
        compile_graph(toy.small_graph(nodes=(toy.CREATE,))),
        toy.toy_sources(tmp_path),
    )
    manifest = run(compiled, sources, store)
    reference = manifest.populations._references["survey"]
    frame = store.load_frame(reference.key)
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].assign(income=999.0)
    replacement = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
    )
    store.put_frame(
        reference.key, replacement, node_key=reference.node_key, verify_existing=False
    )
    with pytest.raises(StoreCorruptError, match="changed after execution"):
        manifest.population("survey")


def test_lazy_view_refuses_different_node_binding(tmp_path):
    from microcosm.graph import StoreCorruptError

    store = ContentStore(tmp_path / "cache")
    manifest = run(compile_graph(toy.small_graph()), toy.toy_sources(tmp_path), store)
    reference = manifest.populations._references["survey"]
    frame = store.load_frame(reference.key)
    store.put_frame(reference.key, frame, node_key="f" * 64, verify_existing=False)
    with pytest.raises(StoreCorruptError, match="different node"):
        manifest.population("survey")


def test_snapshot_publication_failure_leaves_no_partial_attachment_or_staging(
    tmp_path, monkeypatch
):
    import microcosm.graph.attachments as attachments

    store = ContentStore(tmp_path / "cache")
    original = attachments._write_frame

    def fail(root, frame):
        original(root, frame)
        raise OSError("snapshot write interrupted")

    monkeypatch.setattr(attachments, "_write_frame", fail)
    with pytest.raises(OSError, match="snapshot write interrupted"):
        run(compile_graph(toy.small_graph()), toy.toy_sources(tmp_path), store)
    assert not list(store.tmp.iterdir())
    for path in store.objects.glob("*/*"):
        assert (path / "meta.json").is_file()
