"""The composed transport skeleton, its CLI driver and its charter properties.

Everything runs on the engine-free toy package in
``test_support/microcosm_build/transport_composed.py``: six invented donor
records, invented facts for both Chronicle roles, a toy RuleSpec tree and
three pure-Python rules engines (two person engines in the skeleton, a
family engine on the test extension) routed by one
``simulate.rules_by_ref@1`` kernel in one ``run_graph``.

Invariants (each a property over every input the strategy draws, or an
executed check):

- Purity: the graph is a function of the spec's resources and the prepared
  config; composing twice gives equal ``graph_to_json``.
- Resource locality: adding a key to one spec resource re-keys exactly the
  nodes that select that resource whole (or its digest) and their
  descendants. A3 (a calibration reference) and C2 (a ``gates.json``
  threshold, a hold-out reference) are executed instances.
- A1 determinism, A2 memoization, A4 inert prose, A5 (one changed RuleSpec
  byte re-keys exactly the rules nodes and their descendants).
- Hold-out ancestry: no calibration ancestor reads the hold-out facts, the
  family engine, or a family-engine input or output column.
- B1: a non-structural node that owns a structural or engine column is
  refused.
- Extension: an extension adds nodes without re-keying any skeleton node;
  one that would add a member to a calibration base version is refused.
- E3: ``--resume require`` refuses a cold store and creates nothing.
- Differentials: each routed rules node equals the stock single-engine
  ``simulate.rules@1``; preparation's CREATE inventory equals the
  independent composition of the CREATE functions; the registry's schema
  equals CREATE's frame schema; declared node keys equal executed keys.
- Real spec: the packaged country spec refuses with every activation gap
  named.
"""

from __future__ import annotations

import ast
import copy
import json
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from microcosm.build.transport import cli as cli_module
from microcosm.build.transport.cli import (
    EXPORT_FILENAME,
    execute_transport_graph,
    prepare_create_outputs,
    prepare_transport_build,
    through,
)
from microcosm.build.transport.compose import (
    TransportGraphConfig,
    compose_transport_graph,
    load_transport_spec,
    validate_transport_activation,
)
from microcosm.build.transport.terminal_kernels import materialize_export
from microcosm.frame.kernels import SimulateRulesKernel
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    GraphError,
    KernelBase,
    KernelRegistry,
    Node,
    NodeRejectedError,
    Owned,
    Slice,
    StoreMissError,
    compile_graph,
    graph_to_json,
    run_graph,
)
from microcosm.graph.keys import node_key, source_content_key
from test_support.microcosm_build.transport_composed import (
    FAMILY_INPUTS,
    PROGRAMS,
    SHARED_PATH,
    make_composed_fixture,
)
from test_support.microcosm_build.transport_graph import descendants
from test_support.paths import paths_for

TRANSPORT = paths_for("microcosm-build").package / "src/microcosm/build/transport"
PROPERTY = settings(
    max_examples=6,
    deadline=None,
    database=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
EXECUTED = settings(PROPERTY, max_examples=2)
CALIBRATION = "nz.calibrate"
RULES = "simulate.rules_by_ref@1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _keys(manifest) -> dict[str, str]:
    return {name: receipt.key for name, receipt in manifest.nodes.items()}


def _artifacts(manifest, store) -> dict:
    return {
        (name, artifact): store.load_bytes(key)
        for name, receipt in manifest.nodes.items()
        for artifact, key in receipt.opaque_artifacts.items()
    }


def _declared_keys(graph, fixture, exported: Path | None = None) -> dict[str, str]:
    """Node keys from the public key formula, without executing anything.

    The exported dataset is an outer input; without a file its identity is
    held fixed, which suffices to compare two declarations.
    """
    compiled = compile_graph(graph)
    sources = {
        name: source_content_key(name, path)
        for name, path in fixture.source_mapping(exported).items()
    }
    for source in graph.sources:
        sources.setdefault(source.name, "outer-input-held-fixed")
    keys: dict[str, str] = {}
    for name in compiled.order:
        kernel = fixture.registry.kernels.get(graph.node(name).kernel)
        keys[name] = node_key(
            compiled,
            name,
            keys,
            kernel.implementation_hash(),
            sources,
            kernel_capabilities=kernel.capabilities,
        )
    return keys


def _export(root: Path, fixture, graph) -> Path:
    """Write the skeleton H5 from a separate store's export preparation."""
    store = ContentStore(root / "preparation", codecs=fixture.registry.codecs)
    compiled = compile_graph(graph)
    prepared = run_graph(
        compile_graph(through(graph, "nz.export.prepare")),
        sources=fixture.sources,
        store=store,
        kernels=fixture.registry.kernels,
    )
    receipt = prepared.nodes["nz.export.prepare"]
    descriptor = json.loads(
        store.load_bytes(receipt.opaque_artifacts["export_descriptor"])
    )
    exported = root / EXPORT_FILENAME
    version = compiled.versions["nz.export.prepare"]
    materialize_export(prepared.populations[version], descriptor, exported)
    return exported


def _full_run(root: Path, fixture, graph, *, store=None, exported=None, resume="auto"):
    """One ``run_graph`` of the whole graph; the H5 is written beforehand.

    Preparing the H5 in a separate store keeps checkpoint hits from hiding
    which nodes of the full run miss. Reusing one H5's bytes keeps the HDF5
    writer's timestamps out of determinism comparisons.
    """
    root.mkdir(parents=True, exist_ok=True)
    exported = _export(root, fixture, graph) if exported is None else exported
    store = (
        ContentStore(root / "store", codecs=fixture.registry.codecs)
        if store is None
        else store
    )
    manifest = run_graph(
        compile_graph(graph),
        sources=fixture.source_mapping(exported),
        store=store,
        kernels=fixture.registry.kernels,
        resume=resume,
    )
    return manifest, store, exported


def _copied_store(base, root: Path, fixture) -> ContentStore:
    shutil.copytree(base.root, root / "store-copy")
    return ContentStore(root / "store-copy", codecs=fixture.registry.codecs)


def _misses(manifest) -> set[str]:
    return {name for name, receipt in manifest.nodes.items() if not receipt.hit}


def _changed(before, after) -> set[str]:
    return {
        name for name in after.nodes if before.nodes[name].key != after.nodes[name].key
    }


def _selectors(spec: dict, resource: str) -> set[str]:
    """Nodes whose resolved params change when ``resource``'s bytes change.

    A whole-resource selection (empty path) or a digest selection moves
    with any edit; a path selection moves only with its own subtree.
    """
    found = set()
    for row in spec["resources"]["transport_graph"]["nodes"]:
        stack = list(row.get("params", {}).values())
        while stack:
            value = stack.pop()
            if isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, dict) and value.get("resource") == resource:
                if value.get("encoding") == "sha256" or not value.get("path"):
                    found.add(row["id"])
    return found


# ---------------------------------------------------------------------------
# Shared fixtures (built once; tests that edit inputs copy what they change)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    return make_composed_fixture(tmp_path_factory.mktemp("composed"))


@pytest.fixture(scope="module")
def cold(composed, tmp_path_factory):
    """One timed cold run of the extended graph."""
    graph = composed.graph(extended=True)
    started = time.perf_counter()
    manifest, store, exported = _full_run(
        tmp_path_factory.mktemp("cold"), composed, graph
    )
    return graph, manifest, store, exported, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Composition invariants
# ---------------------------------------------------------------------------


@PROPERTY
# Stripped: graph/decl.py accepts an empty description or text with a
# non-whitespace character, so a whitespace-only draw such as " " is invalid.
@given(prose=st.text(alphabet="abc XYZ.", max_size=24).map(str.strip))
def test_composition_is_pure_and_prose_moves_no_key(composed, prose):
    graph = composed.graph(extended=True)
    again = compose_transport_graph(
        copy.deepcopy(composed.spec), composed.extended_config()
    )
    assert graph_to_json(graph) == graph_to_json(again)
    edited = replace(
        graph,
        nodes=tuple(
            replace(node, description=prose, citation=prose) for node in graph.nodes
        ),
        sources=tuple(replace(source, description=prose) for source in graph.sources),
    )
    assert _declared_keys(edited, composed) == _declared_keys(graph, composed)


def _selected_resources(spec) -> list[str]:
    names = set()
    for row in spec["resources"]["transport_graph"]["nodes"]:
        stack = list(row.get("params", {}).values())
        while stack:
            value = stack.pop()
            if isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, dict) and "resource" in value:
                names.add(value["resource"])
    return sorted(names)


@PROPERTY
@given(data=st.data(), note=st.text(alphabet="abcdef", min_size=1, max_size=8))
def test_one_resource_edit_rekeys_exactly_its_selectors_and_descendants(
    composed, data, note
):
    """The graph binds each resource only through the nodes that select it."""
    resource = data.draw(st.sampled_from(_selected_resources(composed.spec)))
    spec = copy.deepcopy(composed.spec)
    spec["resources"][resource]["zz_hypothesis_note"] = note
    graph = composed.graph(extended=True)
    edited = compose_transport_graph(spec, composed.extended_config())
    before = _declared_keys(graph, composed)
    after = _declared_keys(edited, composed)
    moved = {name for name in before if before[name] != after[name]}
    assert moved == descendants(graph, _selectors(spec, resource))


def test_extension_adds_nodes_without_rekeying_the_skeleton(composed):
    skeleton = composed.graph()
    extended = composed.graph(extended=True)
    first = _declared_keys(skeleton, composed)
    second = _declared_keys(extended, composed)
    assert set(second) - set(first) == {
        "nz.test.scn",
        "nz.test.scn.rules",
        "nz.test.validate",
        "nz.test.validate.compare",
    }
    assert {name: second[name] for name in first} == first


def test_an_extension_member_of_a_calibration_base_is_refused(composed):
    def member_of_open(spec, config, skeleton):
        return (
            Node(
                "nz.test.member",
                "transport.unit_attributes@1",
                population="nz.open",
                inputs=(Slice("person", ("age",)),),
                outputs=(Owned("person", "toy_extra", "int64"),),
            ),
        )

    config = replace(composed.config, extensions=(member_of_open,))
    with pytest.raises(ValueError, match="re-key skeleton node 'nz.calibrate'"):
        compose_transport_graph(composed.spec, config)


def test_skeleton_refuses_a_kernel_outside_its_vocabulary(composed):
    spec = copy.deepcopy(composed.spec)
    spec["resources"]["transport_graph"]["nodes"][-1]["kernel"] = "takeup.gap@1"
    with pytest.raises(ValueError, match="through an extension"):
        compose_transport_graph(spec, composed.config)


def test_rules_nodes_take_engine_ref_and_period_only_from_their_binding(composed):
    spec = copy.deepcopy(composed.spec)
    row = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row.get("rules_binding") == "main_benefit"
    )
    row["params"] = {"period": "2030"}
    with pytest.raises(ValueError, match="may set only variables"):
        compose_transport_graph(spec, composed.config)
    graph = composed.graph()
    routed = [node for node in graph.nodes if node.kernel == RULES]
    assert {node.params["engine_ref"] for node in routed} == {
        composed.registry.engine_refs["main_benefit"],
        composed.registry.engine_refs["pension"],
    }


# ---------------------------------------------------------------------------
# Executed charter properties
# ---------------------------------------------------------------------------


def test_a1_a2_three_engines_in_one_run_under_sixty_seconds(composed, cold, tmp_path):
    graph, first, store, exported, seconds = cold
    assert seconds < 60
    routed = [node for node in graph.nodes if node.kernel == RULES]
    assert len({node.params["engine_ref"] for node in routed}) == 3
    assert all(not first.nodes[node.id].hit for node in routed)
    assert all(engine.calls for engine in composed.engines_by_binding.values())
    gates = [
        receipt for receipt in first.nodes.values() if receipt.receipt.get("outcome")
    ]
    assert gates and all(receipt.receipt["outcome"] == "pass" for receipt in gates)
    assert all(
        composed.registry.kernels.get(node.kernel).capabilities.role.value != "release"
        for node in graph.nodes
    )
    second, second_store, _ = _full_run(
        tmp_path / "second", composed, graph, exported=exported
    )
    assert _keys(first) == _keys(second)
    assert _artifacts(first, store) == _artifacts(second, second_store)
    reloaded = ContentStore(second_store.root, codecs=composed.registry.codecs)
    warm, _, _ = _full_run(
        tmp_path / "warm",
        composed,
        graph,
        store=reloaded,
        exported=exported,
        resume="require",
    )
    assert _misses(warm) == set()
    assert _keys(warm) == _keys(first)


def test_declared_keys_equal_executed_keys(composed, cold):
    graph, manifest, _, exported, _ = cold
    assert _declared_keys(graph, composed, exported) == _keys(manifest)


def test_a4_inert_fields_execute_as_store_hits(composed, cold, tmp_path):
    graph, before, store, exported, _ = cold
    edited = replace(
        graph,
        nodes=tuple(
            replace(node, description="reviewed", citation="toy")
            for node in graph.nodes
        ),
    )
    after, _, _ = _full_run(
        tmp_path,
        composed,
        edited,
        store=_copied_store(store, tmp_path, composed),
        exported=exported,
    )
    assert _keys(after) == _keys(before)
    assert _misses(after) == set()


@EXECUTED
@given(marker=st.integers(min_value=1, max_value=10**6))
def test_a3_calibration_reference_edit_is_descendant_exact(
    composed, cold, tmp_path_factory, marker
):
    graph, before, store, _, _ = cold
    root = tmp_path_factory.mktemp("a3")
    spec = copy.deepcopy(composed.spec)
    spec["resources"]["target_references"]["target_references"][0]["metadata"][
        "review_marker"
    ] = marker
    changed = replace(composed, spec=spec)
    edited = changed.graph(extended=True)
    after, _, _ = _full_run(
        root, changed, edited, store=_copied_store(store, root, composed)
    )
    expected = descendants(edited, {"nz.targets.compile"})
    assert _changed(before, after) == expected
    assert _misses(after) == expected
    calibration_side_hits = {
        name
        for name in after.nodes
        if name.startswith(
            ("nz.transport.", "nz.geo.", "nz.takeup.", "nz.rules.", "nz.encode.")
        )
    }
    assert calibration_side_hits and calibration_side_hits.isdisjoint(expected)


@EXECUTED
@given(byte=st.integers(min_value=0, max_value=255), position=st.integers(0, 30))
def test_a5_one_rulespec_byte_rekeys_exactly_rules_nodes_and_descendants(
    composed, cold, tmp_path_factory, byte, position
):
    graph, before, store, _, _ = cold
    root = tmp_path_factory.mktemp("a5")
    fixture = composed.relocated(root / "fixture")
    assert _declared_keys(fixture.graph(extended=True), fixture) == _declared_keys(
        graph, composed
    )
    shared = fixture.sources["rulespec_nz"] / SHARED_PATH
    payload = bytearray(shared.read_bytes())
    index = position % len(payload)
    payload[index] = byte if byte != payload[index] else (byte + 1) % 256
    shared.write_bytes(bytes(payload))
    changed = fixture.reprepare()
    edited = changed.graph(extended=True)
    after, _, _ = _full_run(
        root, changed, edited, store=_copied_store(store, root, composed)
    )
    bound = {node.id for node in edited.nodes if node.kernel == RULES}
    assert len(bound) == 3
    expected = descendants(edited, bound)
    assert _changed(before, after) == expected
    assert _misses(after) == expected


@EXECUTED
@given(which=st.sampled_from(("gate", "holdout")), marker=st.integers(1, 10**6))
def test_c2_gate_and_holdout_edits_move_no_calibration_key(
    composed, cold, tmp_path_factory, which, marker
):
    graph, before, store, exported, _ = cold
    root = tmp_path_factory.mktemp("c2")
    spec = copy.deepcopy(composed.spec)
    if which == "gate":
        gate = spec["resources"]["gates"]["gates"][0]["parameters"]
        gate["maximum_max_to_median_ratio"] = 4.0 + marker
        start, reuse = {"nz.gates.terminal"}, None
    else:
        spec["resources"]["holdout_references"]["target_references"][0]["metadata"][
            "review_marker"
        ] = marker
        start, reuse = {"nz.test.validate.compare"}, exported
    changed = replace(composed, spec=spec)
    edited = changed.graph(extended=True)
    after, _, _ = _full_run(
        root,
        changed,
        edited,
        store=_copied_store(store, root, composed),
        exported=reuse,
    )
    expected = descendants(edited, start)
    assert _changed(before, after) == expected
    assert _misses(after) == expected
    calibration = set(compile_graph(through(edited, CALIBRATION)).order)
    assert calibration.isdisjoint(expected)


def test_holdout_and_family_semantics_have_no_calibration_ancestry(composed):
    graph = composed.graph(extended=True)
    family_ref = composed.registry.engine_refs["family_housing"]
    family_columns = set(FAMILY_INPUTS) | set(PROGRAMS["family_housing"]["outputs"])
    ancestors = through(graph, CALIBRATION).nodes
    for node in ancestors:
        assert "nz_holdout_facts" not in node.sources
        assert node.params.get("engine_ref") != family_ref
        columns = {column for item in node.inputs for column in item.columns}
        columns |= {item.column for item in node.outputs}
        assert columns.isdisjoint(family_columns), node.id
    # The hold-out source and the family engine are in the graph, elsewhere.
    assert any("nz_holdout_facts" in node.sources for node in graph.nodes)
    assert any(node.params.get("engine_ref") == family_ref for node in graph.nodes)
    calibration = composed.sources["nz_calibration_facts"].read_text().splitlines()
    holdout = composed.sources["nz_holdout_facts"].read_text().splitlines()
    assert {json.loads(row)["aggregate_fact_key"] for row in calibration}.isdisjoint(
        json.loads(row)["aggregate_fact_key"] for row in holdout
    )


class _IllegalOwner(KernelBase):
    ref = "test.illegal_owner@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        raise AssertionError("A refused declaration never reaches its kernel.")


@pytest.mark.parametrize(
    "column", ["person_id", "person_household_id", "person_family_id"]
)
def test_b1_a_non_structural_node_cannot_own_a_structural_column(
    composed, tmp_path, column
):
    graph = composed.graph()
    compile_graph(graph)
    kernels = KernelRegistry()
    for ref in composed.registry.kernels.refs():
        kernels.register(composed.registry.kernels.get(ref))
    kernels.register(_IllegalOwner())
    mutation = Node(
        "nz.test.illegal",
        _IllegalOwner.ref,
        population="nz.terminal",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", column, "int64"),),
    )
    mutated = replace(graph, nodes=(*graph.nodes, mutation))
    with pytest.raises(NodeRejectedError, match="structural column"):
        run_graph(
            compile_graph(through(mutated, mutation.id)),
            sources=composed.sources,
            store=ContentStore(tmp_path / "store", codecs=composed.registry.codecs),
            kernels=kernels,
        )


@pytest.mark.parametrize(
    ("population", "entity", "column"),
    [
        ("nz.open", "person", "toy_main_eligible"),
        ("nz.as", "family", "input_rent"),
    ],
)
def test_b1_a_second_owner_of_an_engine_or_family_input_column_is_refused(
    composed, population, entity, column
):
    graph = composed.graph()
    mutation = Node(
        "nz.test.second_owner",
        _IllegalOwner.ref,
        population=population,
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned(entity, column, "float64"),),
    )
    with pytest.raises(GraphError, match="is owned by both"):
        compile_graph(replace(graph, nodes=(*graph.nodes, mutation)))


# ---------------------------------------------------------------------------
# Differentials
# ---------------------------------------------------------------------------


def test_routed_rules_nodes_equal_the_stock_single_engine_kernel(
    composed, cold, tmp_path
):
    graph, combined, store, _, _ = cold
    for node in (node for node in graph.nodes if node.kernel == RULES):
        reference = node.params["engine_ref"]
        kernels = KernelRegistry()
        for ref in composed.registry.kernels.refs():
            kernels.register(composed.registry.kernels.get(ref))
        kernels.register(
            SimulateRulesKernel(reference, composed.registry.engines[reference])
        )
        subset = through(graph, node.id)
        subset = replace(
            subset,
            nodes=tuple(
                replace(item, kernel="simulate.rules@1") if item.id == node.id else item
                for item in subset.nodes
            ),
        )
        direct_store = ContentStore(tmp_path / node.id, codecs=composed.registry.codecs)
        direct = run_graph(
            compile_graph(subset),
            sources=composed.sources,
            store=direct_store,
            kernels=kernels,
        )
        for owned in node.outputs:
            key = (owned.entity, owned.column)
            np.testing.assert_array_equal(
                store.load_column(combined.nodes[node.id].artifacts[key]),
                direct_store.load_column(direct.nodes[node.id].artifacts[key]),
            )


def test_prepared_create_inventory_equals_independent_composition(composed):
    assert (
        prepare_create_outputs(composed.spec, composed.sources)
        == composed.config.create_outputs
    )


def test_registry_schema_equals_the_create_frame_schema(composed, cold):
    graph, manifest, store, _, _ = cold
    frame = store.load_frame(manifest.nodes["nz.create"].frame_key)
    assert composed.registry.schema == frame.schema
    assert dict(composed.registry.nesting) == {"family": "household"}


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------


def _inventory(root: Path) -> dict:
    return {
        path.relative_to(root): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_e3_require_refuses_a_cold_store_and_creates_nothing(composed, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(StoreMissError, match="cold"):
        execute_transport_graph(
            composed.graph(),
            composed.registry,
            composed.sources,
            out=out,
            endpoints=composed.endpoints,
            resume="require",
        )
    assert not out.exists()


def test_cli_checkpoints_write_only_under_out_and_resume_warm(composed, tmp_path):
    graph = composed.graph(extended=True)
    out = tmp_path / "out"
    options = {
        "out": out,
        "endpoints": composed.endpoints,
        "pending_outputs": composed.pending_outputs,
    }
    inputs = _inventory(composed.root)
    manifest = execute_transport_graph(
        graph,
        composed.registry,
        composed.sources,
        out=out,
        endpoints=composed.endpoints,
        pending_outputs=composed.pending_outputs,
    )
    assert _inventory(composed.root) == inputs
    assert set(manifest.nodes) == {node.id for node in graph.nodes}
    assert json.loads((out / "graph.json").read_text()) == json.loads(
        graph_to_json(graph)
    )
    assert set(json.loads((out / "manifest.json").read_text())["nodes"]) == set(
        manifest.nodes
    )
    status = json.loads((out / "build-status.json").read_text())
    assert status["local_only"] is True
    assert status["not_yet_produced"] == list(composed.pending_outputs)
    assert {"graph", "skeleton_dataset", "calibration_diagnostics", "explorer"} <= set(
        status["produced"]
    )
    for name in (
        "checkpoint-geography.json",
        "checkpoint-calibration.json",
        "checkpoint-export-prepared.json",
        "checkpoint-terminal-0.json",
        "checkpoint-terminal-1.json",
        "explorer.html",
        "calibration-diagnostics.json",
        EXPORT_FILENAME,
    ):
        assert (out / name).is_file(), name
    direct, _, _ = _full_run(
        tmp_path / "direct", composed, graph, exported=out / EXPORT_FILENAME
    )
    assert _keys(direct) == _keys(manifest)
    written = _inventory(out)
    warm = execute_transport_graph(
        graph,
        composed.registry,
        composed.sources,
        out=out,
        endpoints=composed.endpoints,
        pending_outputs=composed.pending_outputs,
        resume="require",
    )
    assert _misses(warm) == set()
    assert _keys(warm) == _keys(manifest)
    assert _inventory(out)[Path(EXPORT_FILENAME)] == written[Path(EXPORT_FILENAME)]
    # A deleted H5 is rewritten from the warm store and passes readback; a
    # corrupted one under an unchanged descriptor is refused, not replaced.
    (out / EXPORT_FILENAME).unlink()
    rewritten = execute_transport_graph(
        graph, composed.registry, composed.sources, **options
    )
    assert rewritten.nodes["nz.export.readback"].receipt["outcome"] == "pass"
    (out / EXPORT_FILENAME).write_bytes(b"not an h5 file")
    with pytest.raises(ValueError, match="cannot be read"):
        execute_transport_graph(graph, composed.registry, composed.sources, **options)


def test_cli_refuses_an_output_directory_that_overlaps_an_input(composed):
    for out in (composed.sources["rulespec_nz"] / "out", composed.root):
        with pytest.raises(ValueError, match="overlaps"):
            execute_transport_graph(
                composed.graph(),
                composed.registry,
                composed.sources,
                out=out,
                endpoints=composed.endpoints,
            )
    assert not (composed.sources["rulespec_nz"] / "out").exists()


def test_cli_refuses_a_supplied_export_path(composed, tmp_path):
    sources = {**composed.sources, "nz_exported_dataset": tmp_path / "elsewhere.h5"}
    with pytest.raises(ValueError, match="written by the run"):
        execute_transport_graph(
            composed.graph(),
            composed.registry,
            sources,
            out=tmp_path / "out",
            endpoints=composed.endpoints,
        )


def test_prepare_builds_the_same_graph_as_the_fixture(composed):
    prepared = prepare_transport_build(
        composed.spec,
        sources=composed.sources,
        engines_by_binding=composed.engines_by_binding,
        extensions=(composed.extended_config().extensions),
    )
    assert graph_to_json(prepared.graph) == graph_to_json(composed.graph(extended=True))
    assert prepared.endpoints == composed.endpoints
    assert prepared.pending_outputs == composed.pending_outputs


def test_cli_arguments():
    args = cli_module.parse_args(
        ["--country", "xx", "--out", "o", "--source", "a=one", "--source", "b=two"]
    )
    assert args.local_only is True
    assert dict(args.source) == {"a": Path("one"), "b": Path("two")}
    for argv in (
        ["--out", "o"],
        ["--country", "xx", "--spec-dir", "d", "--out", "o"],
        ["--country", "xx", "--out", "o", "--source", "a=1", "--source", "a=2"],
        ["--country", "xx", "--out", "o", "--source", "missing-separator"],
    ):
        with pytest.raises(SystemExit):
            cli_module.parse_args(argv)


# ---------------------------------------------------------------------------
# The packaged country spec: activation gaps refuse, named
# ---------------------------------------------------------------------------

_GAPS = (
    "benefit_unit_rule.dependent_child.age_limit_years",
    "placeholder reference precal_references.",
    "placeholder reference target_references.",
    "placeholder reference holdout_references.",
    "receipt_contract.json",
    "axiom_rules_bindings.engine.commit",
    "axiom_rules_bindings.engine.wheel_sha256",
    "scenario S4 rent_stock_factor",
    "transport_graph.json",
)


def test_packaged_spec_refuses_with_every_activation_gap_named():
    spec = load_transport_spec("nz")
    with pytest.raises(ValueError) as error:
        validate_transport_activation(spec)
    for gap in _GAPS:
        assert gap in str(error.value), gap
    config = TransportGraphConfig(
        engine_refs={"x": "y"}, create_outputs=(Owned("person", "age", "int64"),)
    )
    with pytest.raises(ValueError, match="not activated"):
        compose_transport_graph(spec, config)


def test_packaged_spec_refuses_before_reading_any_source(tmp_path):
    missing = tmp_path / "absent"
    with pytest.raises(ValueError, match="not activated"):
        prepare_transport_build(
            load_transport_spec("nz"), sources={"rulespec_nz": missing}
        )


def test_cli_main_refuses_the_packaged_spec_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "out"
    assert cli_module.main(["--country", "nz", "--out", str(out)]) == 1
    message = capsys.readouterr().err
    for gap in _GAPS:
        assert gap in message, gap
    assert not out.exists()


def test_packaged_holdout_references_are_disjoint_from_calibration_references():
    spec = load_transport_spec("nz")

    def names(resource):
        rows = spec["resources"][resource]["target_references"]
        return {row["name"] for row in rows}

    calibration = names("target_references") | names("precal_references")
    assert names("holdout_references").isdisjoint(calibration)


# ---------------------------------------------------------------------------
# Source rules for the new modules
# ---------------------------------------------------------------------------


def test_transport_python_has_no_policy_literal_or_remote_path():
    """No target, threshold or rate value is assigned in the transport Python.

    ``test_transport_graph.py`` already refuses float literals and the word
    for any country; this also refuses integer-valued policy assignments,
    and keeps the driver's imports local: no country runtime, no network or
    hub client and no staged-bundle publication.
    """
    number = r"[-+]?\d+(?:\.\d+)?"
    patterns = (
        rf"\b(?:target_value|threshold|max_age|age_limit\w*|\w+_threshold|\w+_rate)\s*[=:]\s*{number}\b",
        rf"[\"'](?:target_value|threshold|max_age|\w+_threshold|\w+_rate|rate)[\"']\s*:\s*{number}\b",
        rf"\brate\s*=\s*{number}\b",
    )
    for path in sorted(TRANSPORT.glob("*.py")):
        source = path.read_text()
        for pattern in patterns:
            assert re.search(pattern, source) is None, (path.name, pattern)
    remote = (
        "uk_runtime",
        "us_runtime",
        "huggingface",
        "requests",
        "urllib",
        "http",
        "boto",
    )
    for name in ("compose.py", "cli.py", "registry.py"):
        tree = ast.parse((TRANSPORT / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                imported = modules
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
                imported = [alias.name for alias in node.names]
            else:
                continue
            assert not any(part in module for module in modules for part in remote), (
                name
            )
            assert "publish_staged_bundle" not in imported, name
