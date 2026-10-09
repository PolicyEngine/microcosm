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
import hashlib
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
from microcosm.build.transport import compose as compose_module
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
    SourceRef,
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
    _blueprint,
    make_composed_fixture,
    toy_extension,
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


# Inventory changes no selectors; the blueprint needs one household column
# to construct its input slices, without running CREATE during collection.
SELECTED_RESOURCES = _selected_resources(
    {
        "resources": {
            "transport_graph": _blueprint((Owned("household", "rent", "float64"),))
        }
    }
)


@pytest.mark.parametrize("resource", SELECTED_RESOURCES)
@PROPERTY
@given(note=st.text(alphabet="abcdef", min_size=1, max_size=8))
def test_one_resource_edit_rekeys_exactly_its_selectors_and_descendants(
    composed, resource, note
):
    """The graph binds each resource only through the nodes that select it."""
    assert set(SELECTED_RESOURCES) == set(_selected_resources(composed.spec))
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


@pytest.mark.parametrize(
    "violation",
    ["prepared", "embedded_ref", "holdout_resource", "holdout_source", "graph_digest"],
)
def test_compose_refuses_contaminated_calibration_ancestry(composed, violation):
    spec = copy.deepcopy(composed.spec)
    row = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["id"] == "nz.targets.problem"
    )
    if violation == "holdout_source":
        row["sources"] = ["nz_holdout_facts"]
    else:
        row["params"]["contamination"] = {
            "prepared": {"prepared": "engine_refs"},
            "embedded_ref": json.dumps(
                {"nested": [composed.registry.engine_refs["family_housing"]]}
            ),
            "holdout_resource": {
                "resource": "holdout_references",
                "encoding": "sha256",
            },
            "graph_digest": {"resource": "transport_graph", "encoding": "sha256"},
        }[violation]
    with pytest.raises(ValueError, match="calibration ancestry|graph resource.*digest"):
        compose_transport_graph(spec, composed.config)


@pytest.mark.parametrize(
    "literal",
    [
        {"resource": "holdout_references", "encoding": "sha256"},
        {"prepared": "engine_refs"},
    ],
)
def test_selected_lists_keep_selector_shaped_objects_literal(composed, literal):
    spec = copy.deepcopy(composed.spec)
    spec["resources"]["calibration"]["literal"] = [[literal]]
    result = compose_module._param(
        spec,
        {"resource": "calibration", "path": ["literal"]},
        composed.config.engine_refs,
    )
    assert result == ((literal,),)


@pytest.mark.parametrize("gap", ["mass", "scenario", "resource"])
def test_activation_names_generic_selected_gaps_before_donor_read(
    composed, monkeypatch, gap
):
    spec = copy.deepcopy(composed.spec)
    if gap == "mass":
        spec["resources"]["mass_references"]["target_references"][0]["metadata"][
            "activation_status"
        ] = "placeholder"
        match = "placeholder reference mass_references"
    elif gap == "scenario":
        spec["resources"]["scenarios"]["scenarios"] = [
            {"id": "SX", "knobs": {"other_factor": None}}
        ]
        match = "scenario SX other_factor"
    else:
        spec["resources"]["transport_graph"]["nodes"][0]["params"]["missing"] = [
            {"resource": "absent_resource", "encoding": "sha256"}
        ]
        match = "absent_resource.json"

    def unread(*args, **kwargs):
        pytest.fail("activation must refuse before CREATE reads the donor")

    monkeypatch.setattr(cli_module, "prepare_create_outputs", unread)
    with pytest.raises(ValueError, match=match):
        prepare_transport_build(
            spec,
            sources=composed.sources,
            engines_by_binding=composed.engines_by_binding,
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


def test_cli_fresh_builds_have_identical_h5_bytes_and_every_node_key(
    composed, tmp_path
):
    graph = composed.graph(extended=True)
    first_out, second_out = tmp_path / "first", tmp_path / "second"
    first = execute_transport_graph(
        graph,
        composed.registry,
        composed.sources,
        out=first_out,
        endpoints=composed.endpoints,
    )
    # HDF5 object timestamps have second resolution; cross that boundary.
    time.sleep(1.1)
    second = execute_transport_graph(
        graph,
        composed.registry,
        composed.sources,
        out=second_out,
        endpoints=composed.endpoints,
    )
    assert (first_out / EXPORT_FILENAME).read_bytes() == (
        second_out / EXPORT_FILENAME
    ).read_bytes()
    assert _keys(first) == _keys(second)


def test_cli_checkpoints_write_only_under_out_and_resume_warm(composed, tmp_path):
    extension = compose_module.TransportExtension(
        toy_extension,
        checkpoints=("nz.test.scn.rules", "nz.test.validate.compare"),
    )
    graph = compose_transport_graph(
        composed.spec, replace(composed.config, extensions=(extension,))
    )
    endpoints = compose_module.transport_endpoints(
        composed.spec, extensions=(extension,)
    )
    out = tmp_path / "out"
    options = {
        "out": out,
        "endpoints": endpoints,
        "pending_outputs": composed.pending_outputs,
    }
    inputs = _inventory(composed.root)
    manifest = execute_transport_graph(
        graph,
        composed.registry,
        composed.sources,
        out=out,
        endpoints=endpoints,
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
        "checkpoint-extension-0.json",
        "checkpoint-extension-1.json",
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
        endpoints=endpoints,
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
    compose_module.validate_transport_calibration_ancestry(spec)


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


def _write_local_transport_spec(root, spec):
    directory = root / spec["country"]
    directory.mkdir(parents=True)
    names = []
    for name, document in spec["resources"].items():
        filename = name + ".json"
        names.append(filename)
        (directory / filename).write_text(json.dumps(document), encoding="utf-8")
    (directory / "country_package.json").write_text(
        json.dumps(
            {
                "country": spec["country"],
                "policy": "Synthetic transport test package.",
                "resources": names,
            }
        ),
        encoding="utf-8",
    )
    return directory


def test_cli_main_require_refuses_before_preparing_or_reading_sources(
    tmp_path, monkeypatch, capsys
):
    calls = []

    def prepare(*args, **kwargs):
        calls.append("prepare")
        raise ValueError("CREATE probe ran")

    monkeypatch.setattr(cli_module, "prepare_transport_build", prepare)
    out = tmp_path / "out"
    assert (
        cli_module.main(["--country", "nz", "--out", str(out), "--resume", "require"])
        == 1
    )
    assert "cold transport graph store" in capsys.readouterr().err
    assert calls == []
    assert not out.exists()


@pytest.mark.parametrize("relation", ["inside", "contains", "same"])
def test_cli_main_refuses_output_overlap_with_spec_dir(
    tmp_path, relation, monkeypatch, capsys
):
    spec_dir = tmp_path / "spec"
    spec_dir.mkdir()
    out = {"inside": spec_dir / "out", "contains": tmp_path, "same": spec_dir}[relation]
    calls = []

    def load(source):
        calls.append(source)
        raise ValueError("spec was read")

    monkeypatch.setattr(cli_module, "load_transport_spec", load)
    assert cli_module.main(["--spec-dir", str(spec_dir), "--out", str(out)]) == 1
    assert "--out overlaps --spec-dir" in capsys.readouterr().err
    assert calls == []


@pytest.mark.parametrize("filename", ["../escape", "nested/file", ".", "..", ""])
def test_cli_destination_refuses_escaping_names(tmp_path, filename):
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="simple filename"):
        cli_module._destination(out, filename)


@pytest.mark.parametrize("filename", ["graph.json", ".graph-store"])
def test_cli_destination_refuses_symlinks(tmp_path, filename):
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "outside"
    target.write_bytes(b"preserved")
    (out / filename).symlink_to(target)
    with pytest.raises(ValueError, match="outside --out"):
        cli_module._destination(out, filename)
    assert target.read_bytes() == b"preserved"


@pytest.mark.parametrize("missing", ["unknown_variable", "unit_rule", "kernel"])
def test_cli_main_turns_key_errors_into_refusals(
    composed, tmp_path, monkeypatch, capsys, missing
):
    spec = copy.deepcopy(composed.spec)
    rows = spec["resources"]["transport_graph"]["nodes"]
    if missing == "unknown_variable":
        spec["resources"]["axiom_rules_bindings"]["bindings"][0]["variables"].append(
            "unknown_variable"
        )
    elif missing == "unit_rule":
        create = next(row for row in rows if row["kernel"] == "transport.create@1")
        create["params"]["renamed_unit_rule"] = create["params"].pop("unit_rule")
    else:
        next(row for row in rows if row["id"] == "nz.targets.problem").pop("kernel")
    monkeypatch.setattr(cli_module, "load_transport_spec", lambda source: spec)
    build_registry = cli_module.build_transport_registry

    def injected_registry(bindings, root, *, unit_rule, engines_by_binding=None):
        return build_registry(
            bindings,
            root,
            unit_rule=unit_rule,
            engines_by_binding=composed.engines_by_binding,
        )

    monkeypatch.setattr(cli_module, "build_transport_registry", injected_registry)
    out = tmp_path / "out"
    argv = ["--country", "nz", "--out", str(out)]
    for name, path in composed.sources.items():
        argv.extend(["--source", f"{name}={path}"])
    assert cli_module.main(argv) == 1
    error = capsys.readouterr().err
    assert "transport build refused:" in error
    assert missing in error
    assert "Traceback" not in error
    assert not (out / EXPORT_FILENAME).exists()


def test_cli_main_and_path_loader_success(composed, tmp_path, monkeypatch, capsys):
    # Real Path loading, registry preparation and execution; only the injected
    # engine seam replaces Axiom with the fixture's pure-Python adapters.
    directory = _write_local_transport_spec(tmp_path / "spec", composed.spec)
    loaded = load_transport_spec(directory)
    assert loaded["country"] == composed.spec["country"]
    assert loaded["resources"] == composed.spec["resources"]
    for name in loaded["resources"]:
        assert (
            loaded["resource_hashes"][name]
            == hashlib.sha256((directory / (name + ".json")).read_bytes()).hexdigest()
        )
    build_registry = cli_module.build_transport_registry
    observe_warm = []
    calls = []

    def injected_registry(bindings, root, *, unit_rule, engines_by_binding=None):
        registry = build_registry(
            bindings,
            root,
            unit_rule=unit_rule,
            engines_by_binding=composed.engines_by_binding,
        )
        if observe_warm:
            for ref in registry.kernels.refs():
                kernel = registry.kernels.get(ref)
                run = kernel.run

                def counted(context, _run=run, _ref=ref):
                    calls.append(_ref)
                    return _run(context)

                monkeypatch.setattr(kernel, "run", counted)
        return registry

    monkeypatch.setattr(cli_module, "build_transport_registry", injected_registry)
    out = tmp_path / "out"
    argv = ["--spec-dir", str(directory), "--out", str(out)]
    for name, path in composed.sources.items():
        argv.extend(["--source", f"{name}={path}"])
    assert cli_module.main(argv) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert "Transport skeleton written" in captured.out
    assert "Not yet produced:" in captured.out
    assert (out / EXPORT_FILENAME).is_file()
    assert json.loads((out / "manifest.json").read_bytes())["nodes"]
    observe_warm.append(True)
    assert cli_module.main([*argv, "--resume", "require"]) == 0
    assert capsys.readouterr().err == ""
    assert calls == []


def test_cli_main_invalid_arguments_are_a_refusal(capsys):
    assert cli_module.main(["--country", "nz"]) == 1
    assert "transport build refused:" in capsys.readouterr().err


def test_cli_main_help_succeeds(capsys):
    assert cli_module.main(["--help"]) == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert captured.err == ""


def test_cli_main_refuses_store_symlink_before_loading_spec(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "out"
    out.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (target / "preserved").write_bytes(b"preserved")
    (out / ".graph-store").symlink_to(target, target_is_directory=True)
    calls = []

    def load(source):
        calls.append(source)
        raise ValueError("spec was read")

    monkeypatch.setattr(cli_module, "load_transport_spec", load)
    assert cli_module.main(["--country", "nz", "--out", str(out)]) == 1
    assert "outside --out" in capsys.readouterr().err
    assert calls == []
    assert (target / "preserved").read_bytes() == b"preserved"


def _plant_store_link(out, link, target):
    """Make ``<out>/.graph-store/<link>`` a link to ``target``."""

    path = out / ".graph-store" / link
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


@pytest.mark.parametrize("link", ["objects", "tmp", "objects/aa"])
def test_cli_store_root_refuses_links_inside_the_store_tree(tmp_path, link):
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _plant_store_link(out, link, outside)
    with pytest.raises(ValueError, match="is a link"):
        cli_module._contained_store_root(out)
    assert list(outside.iterdir()) == []


def test_cli_store_root_accepts_a_plain_store_tree(tmp_path):
    out = tmp_path / "out"
    (out / ".graph-store" / "objects" / "aa").mkdir(parents=True)
    (out / ".graph-store" / "tmp").mkdir()
    assert cli_module._contained_store_root(out) == out / ".graph-store"
    assert cli_module._contained_store_root(tmp_path / "fresh") == (
        tmp_path / "fresh" / ".graph-store"
    )


@pytest.mark.parametrize("link", ["objects", "tmp", "objects/aa"])
def test_cli_execution_refuses_a_redirected_store_and_writes_nothing_outside(
    composed, tmp_path, link
):
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _plant_store_link(out, link, outside)
    with pytest.raises(ValueError, match="is a link"):
        execute_transport_graph(
            composed.graph(extended=True),
            composed.registry,
            composed.sources,
            out=out,
            endpoints=composed.endpoints,
        )
    assert list(outside.iterdir()) == []


def test_cli_main_refuses_a_nested_store_link_before_loading_spec(
    tmp_path, monkeypatch, capsys
):
    out = tmp_path / "out"
    out.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _plant_store_link(out, "objects", outside)
    calls = []

    def load(source):
        calls.append(source)
        raise ValueError("spec was read")

    monkeypatch.setattr(cli_module, "load_transport_spec", load)
    assert cli_module.main(["--country", "nz", "--out", str(out)]) == 1
    assert "is a link" in capsys.readouterr().err
    assert calls == []
    assert list(outside.iterdir()) == []


def test_create_inventory_cache_reuses_warm_without_running_create(
    composed, tmp_path, monkeypatch
):
    store = ContentStore(tmp_path / "inventory")
    calls = []
    create = cli_module.TRANSPORT_CREATE.run

    def counted(context):
        calls.append(context.node.id)
        return create(context)

    monkeypatch.setattr(cli_module.TRANSPORT_CREATE, "run", counted)
    first = prepare_create_outputs(
        composed.spec, composed.sources, inventory_store=store
    )
    assert calls == ["nz.create"]
    for resume in ("auto", "require"):
        assert (
            prepare_create_outputs(
                composed.spec, composed.sources, inventory_store=store, resume=resume
            )
            == first
        )
    assert calls == ["nz.create"]
    # Source paths are descriptive; moving identical bytes retains the hit.
    copied = tmp_path / "moved-facts.jsonl"
    copied.write_bytes(composed.sources["nz_calibration_facts"].read_bytes())
    moved = {**composed.sources, "nz_calibration_facts": copied}
    assert (
        prepare_create_outputs(
            composed.spec, moved, inventory_store=store, resume="require"
        )
        == first
    )
    assert calls == ["nz.create"]
    assert (
        prepare_create_outputs(
            composed.spec, composed.sources, inventory_store=store, resume="forbid"
        )
        == first
    )
    assert calls == ["nz.create", "nz.create"]


def test_create_inventory_cache_binds_actual_source_bytes(
    composed, tmp_path, monkeypatch
):
    store = ContentStore(tmp_path / "inventory")
    calls = []
    create = cli_module.TRANSPORT_CREATE.run

    def counted(context):
        calls.append(context.node.id)
        return create(context)

    monkeypatch.setattr(cli_module.TRANSPORT_CREATE, "run", counted)
    first = prepare_create_outputs(
        composed.spec, composed.sources, inventory_store=store
    )
    # An inert extra newline preserves facts and column inventory, but changes
    # source bytes. A cache keyed only by declared pins would silently hit.
    changed = tmp_path / "changed-facts.jsonl"
    changed.write_bytes(composed.sources["nz_calibration_facts"].read_bytes() + b"\n")
    sources = {**composed.sources, "nz_calibration_facts": changed}
    with pytest.raises(StoreMissError, match="this CREATE inventory"):
        prepare_create_outputs(
            composed.spec, sources, inventory_store=store, resume="require"
        )
    assert calls == ["nz.create"]
    assert (
        prepare_create_outputs(composed.spec, sources, inventory_store=store) == first
    )
    assert calls == ["nz.create", "nz.create"]


def test_prepare_does_not_create_inventory_store_before_activation(tmp_path):
    inventory = tmp_path / "inventory"
    with pytest.raises(ValueError, match="not activated"):
        prepare_transport_build(
            load_transport_spec("nz"), sources={}, inventory_store=inventory
        )
    assert not inventory.exists()


def test_prepare_build_reuses_inventory_without_running_create(
    composed, tmp_path, monkeypatch
):
    inventory = tmp_path / "inventory"
    calls = []
    create = cli_module.TRANSPORT_CREATE.run

    def counted(context):
        calls.append(context.node.id)
        return create(context)

    monkeypatch.setattr(cli_module.TRANSPORT_CREATE, "run", counted)
    options = {
        "sources": composed.sources,
        "engines_by_binding": composed.engines_by_binding,
        "inventory_store": inventory,
    }
    first = prepare_transport_build(composed.spec, **options)
    second = prepare_transport_build(composed.spec, **options, resume="require")
    assert graph_to_json(first.graph) == graph_to_json(second.graph)
    assert calls == ["nz.create"]


@pytest.mark.parametrize(
    "node_id",
    [
        "../escape",
        "nested/node",
        "nested\\node",
        "..",
        "unsafe..node",
        "unsafe\x00node",
    ],
)
@pytest.mark.parametrize("extension", [False, True])
def test_compose_refuses_unsafe_node_ids(composed, node_id, extension):
    spec = copy.deepcopy(composed.spec)
    config = composed.config
    if extension:

        def factory(spec, config, skeleton):
            return (replace(skeleton.node("nz.as"), id=node_id),)

        config = replace(config, extensions=(factory,))
    else:
        rows = spec["resources"]["transport_graph"]["nodes"]
        row = next(item for item in rows if item["id"] == "nz.targets.problem")
        row["id"] = node_id
        for item in rows:
            for edge in item.get("artifact_inputs", ()):
                if edge["producer"] == "nz.targets.problem":
                    edge["producer"] = node_id
    with pytest.raises(ValueError, match="safe filename"):
        compose_transport_graph(spec, config)


def test_extension_factories_receive_independent_spec_copies(composed):
    before = copy.deepcopy(composed.spec)
    observed = []

    def first(spec, config, skeleton):
        spec["resources"]["calibration"]["private_edit"] = "extension-local"
        return ()

    def second(spec, config, skeleton):
        observed.append(copy.deepcopy(spec))
        return ()

    compose_transport_graph(
        composed.spec, replace(composed.config, extensions=(first, second))
    )
    assert composed.spec == before
    assert observed == [before]


@pytest.mark.parametrize(
    "selector",
    [
        {"resource": "transport_graph", "encoding": "sha256"},
        {"prepared": "spec_fingerprint"},
    ],
)
def test_compose_refuses_global_declaration_identities(composed, selector):
    spec = copy.deepcopy(composed.spec)
    # Put it outside calibration ancestry, proving the global refusal.
    row = next(
        item
        for item in spec["resources"]["transport_graph"]["nodes"]
        if item["id"] == "nz.package"
    )
    row["params"]["forbidden_identity"] = selector
    with pytest.raises(
        ValueError, match="graph resource.*digest|whole-spec fingerprint"
    ):
        compose_transport_graph(spec, composed.config)


@pytest.mark.parametrize("prefixed", [False, True, "escaped"])
@pytest.mark.parametrize("adapter", ["python", "axiom"])
@pytest.mark.parametrize("selected", [False, True])
def test_public_declaration_check_refuses_unit_ref_without_config(
    composed, selected, adapter, prefixed
):
    spec = copy.deepcopy(composed.spec)
    reference = composed.registry.engine_refs["family_housing"]
    bindings = spec["resources"]["axiom_rules_bindings"]["bindings"]
    if adapter == "axiom":
        from microcosm.frame.adapters.axiom import _ENGINE_REF_SCHEMA

        binding = next(row for row in bindings if row["id"] == "family_housing")
        reference = json.dumps(
            {"format": _ENGINE_REF_SCHEMA, "module": binding["rulespec_path"]}
        )
    if prefixed == "escaped":
        reference = f"prefix {json.dumps(reference)} suffix"
    elif prefixed:
        reference = f"prefix {reference} suffix"
    row = next(
        item
        for item in spec["resources"]["transport_graph"]["nodes"]
        if item["id"] == "nz.targets.problem"
    )
    if selected:
        # Selector-shaped objects in the selected data are literal, but an
        # actual embedded forbidden engine reference still contaminates it.
        spec["resources"]["calibration"]["literal_ref"] = [{"nested": reference}]
        row["params"]["contamination"] = {
            "resource": "calibration",
            "path": ["literal_ref"],
        }
    else:
        row["params"]["contamination"] = json.dumps({"nested": [reference]})
    validator = getattr(compose_module, "validate_transport_calibration_ancestry", None)
    assert callable(validator), "composition must expose its declaration ancestry check"
    with pytest.raises(ValueError, match="calibration ancestry"):
        validator(spec)


def test_extension_rules_helper_uses_the_binding_period_and_ref(composed):
    spec = copy.deepcopy(composed.spec)
    bindings = spec["resources"]["axiom_rules_bindings"]
    family = next(row for row in bindings["bindings"] if row["id"] == "family_housing")
    previous = family["period"]
    family["period"] = "custom-bound-period"
    bindings["periods"][family["period"]] = dict(bindings["periods"][previous])
    fixture = composed.reprepare(spec)
    unbound = Node(
        "nz.test.helper.rules",
        "simulate.rules_by_ref@1",
        population="nz.as",
        sources=("rulespec_nz",),
        inputs=(Slice("person", ("age",)), Slice("family", FAMILY_INPUTS)),
        outputs=tuple(
            Owned("family", name, row["dtype"])
            for name, row in PROGRAMS["family_housing"]["outputs"].items()
        ),
    )
    bound = compose_module.transport_rules_node(
        spec, fixture.config, "family_housing", unbound
    )
    assert bound.params["engine_ref"] == fixture.config.engine_refs["family_housing"]
    assert bound.params["period"] == "custom-bound-period"
    assert bound.params["variables"] == tuple(family["variables"])
    assert bound.inputs == unbound.inputs
    assert bound.outputs == unbound.outputs
    assert bound.sources == unbound.sources
    # The fixture extension itself must also take the period from the binding.
    assert (
        fixture.graph(extended=True).node("nz.test.scn.rules").params["period"]
        == "custom-bound-period"
    )
    with pytest.raises(ValueError, match="may set only variables"):
        compose_module.transport_rules_node(
            spec,
            fixture.config,
            "family_housing",
            replace(unbound, params={"period": previous}),
        )


def test_extension_sources_rekey_only_their_own_nodes(composed, tmp_path):
    from microcosm.build.transport.codecs import FACTS_SOURCE_CODEC

    def factory(spec, config, skeleton):
        return tuple(
            replace(node, sources=("scenario_facts",))
            if node.id == "nz.test.validate.compare"
            else node
            for node in toy_extension(spec, config, skeleton)
        )

    extension = compose_module.TransportExtension(
        factory, sources=(SourceRef("scenario_facts", FACTS_SOURCE_CODEC),)
    )
    config = replace(composed.config, extensions=(extension,))
    extra = tmp_path / "scenario-facts.jsonl"
    extra.write_bytes(composed.sources["nz_holdout_facts"].read_bytes())
    fixture = replace(composed, sources={**composed.sources, "scenario_facts": extra})
    skeleton = _declared_keys(composed.graph(), fixture)
    graph = compose_transport_graph(composed.spec, config)
    first = _declared_keys(graph, fixture)
    assert {source.name for source in graph.sources} == {
        source.name for source in composed.graph().sources
    } | {"scenario_facts"}
    assert {name: first[name] for name in skeleton} == skeleton
    before_source = source_content_key("scenario_facts", extra)
    extra.write_bytes(extra.read_bytes() + b"\n")
    assert source_content_key("scenario_facts", extra) != before_source
    second = _declared_keys(graph, fixture)
    assert {name for name in first if first[name] != second[name]} == {
        "nz.test.validate.compare"
    }
    assert {name: second[name] for name in skeleton} == skeleton


@pytest.mark.parametrize("collision", ["skeleton", "extension"])
def test_extension_sources_cannot_replace_declared_inputs(composed, collision):
    def empty(spec, config, skeleton):
        return ()

    source = (
        composed.graph().sources[0]
        if collision == "skeleton"
        else SourceRef("scenario_facts", "raw-bytes-v1")
    )
    extension = compose_module.TransportExtension(empty, sources=(source,))
    extensions = (extension,) if collision == "skeleton" else (extension, extension)
    with pytest.raises(ValueError, match="must have a new name"):
        compose_transport_graph(
            composed.spec, replace(composed.config, extensions=extensions)
        )


@pytest.mark.parametrize(
    "checkpoint,match",
    [
        ("missing.checkpoint", "extended graph"),
        ("nz.export.readback", "precede export readback"),
    ],
)
def test_extension_checkpoints_are_checked_before_execution(
    composed, checkpoint, match
):
    extension = compose_module.TransportExtension(
        lambda spec, config, skeleton: (), checkpoints=(checkpoint,)
    )
    with pytest.raises(ValueError, match=match):
        compose_transport_graph(
            composed.spec, replace(composed.config, extensions=(extension,))
        )


def test_extension_checkpoints_are_returned_in_declaration_order(composed):
    extension = compose_module.TransportExtension(
        toy_extension, checkpoints=("nz.test.scn.rules", "nz.test.validate.compare")
    )
    extensions = (extension,)
    endpoints = compose_module.transport_endpoints(composed.spec, extensions=extensions)
    assert endpoints["checkpoints"] == extension.checkpoints
    duplicate = replace(
        extension, checkpoints=("nz.test.scn.rules", "nz.test.scn.rules")
    )
    with pytest.raises(ValueError, match="distinct nodes"):
        compose_module.transport_endpoints(composed.spec, extensions=(duplicate,))


@pytest.mark.parametrize("failure_type", [ImportError, NotImplementedError])
def test_cli_main_refuses_unavailable_rules_engine(
    composed, tmp_path, monkeypatch, capsys, failure_type
):
    monkeypatch.setattr(cli_module, "load_transport_spec", lambda source: composed.spec)

    def unavailable(*args, **kwargs):
        raise failure_type("rules engine unavailable")

    monkeypatch.setattr(cli_module, "build_transport_registry", unavailable)
    out = tmp_path / "out"
    argv = ["--country", "nz", "--out", str(out)]
    for name, path in composed.sources.items():
        argv.extend(["--source", f"{name}={path}"])
    assert cli_module.main(argv) == 1
    assert (
        "transport build refused: rules engine unavailable" in capsys.readouterr().err
    )
    assert not (out / EXPORT_FILENAME).exists()
