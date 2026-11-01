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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.reference_profiles import reference_document_sha256
from microcosm.build.transport import cli as cli_module
from microcosm.build.transport import compose as compose_module
from microcosm.build.transport import target_kernels as target_kernels_module
from microcosm.build.transport.artifact_types import COMPARISON_TYPE
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
    skeleton_create_declaration,
    validate_transport_activation,
)
from microcosm.build.transport.population_kernels import TRANSPORT_CURRENCY
from microcosm.build.transport.takeup_kernels import GAP_TYPE
from microcosm.build.transport.target_kernels import CALIBRATION_PREPARE
from microcosm.build.transport.terminal_kernels import materialize_export
from microcosm.frame.kernels import SimulateRulesKernel
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    GraphError,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    NodeRejectedError,
    Owned,
    Slice,
    SourceRef,
    StoreMissError,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    graph_to_json,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.keys import node_key, source_content_key
from test_support.microcosm_build.chronicle_feed import write_pinned_artifact
from test_support.microcosm_build.reference_factor_transport import (
    factor_transport_fixture,
)
from test_support.microcosm_build.reference_intervals import (
    INCOME_COLUMN,
    UNIVERSE_COLUMN,
    interval_bindings,
    interval_profile_fixture,
)
from test_support.microcosm_build.transport_composed import (
    FAMILY_INPUTS,
    PROGRAMS,
    SHARED_PATH,
    _blueprint,
    make_composed_fixture,
    toy_extension,
)
from test_support.microcosm_build.transport_entitlement import (
    make_entitlement_fixture,
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


# The pre-S5 implementation is intentionally retained for differential checks.
def _precalibration_reference_selections(value: object) -> Iterator[Mapping]:
    """Every explicit selection inside a declared parameter value."""

    if isinstance(value, Mapping):
        yield value
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _precalibration_reference_selections(item)


def _precalibration_reference_ancestors(
    predecessors: Mapping[str, Sequence[str]], roots: set[str]
) -> set[str]:
    ancestors = set()
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in ancestors:
            continue
        if name not in predecessors:
            raise ValueError(f"Calibration endpoint {name!r} is not a declared node.")
        ancestors.add(name)
        pending.extend(predecessors[name])
    return ancestors


def _precalibration_reference_embeds(
    value: object,
    references: set[str],
    unit_modules: frozenset[str] = frozenset(),
) -> bool:
    if not references and not unit_modules:
        return False
    if isinstance(value, Mapping):
        if unit_modules:
            from microcosm.build.transport.registry import PYTHON_ENGINE_REF_FORMAT
            from microcosm.frame.adapters.axiom import _ENGINE_REF_SCHEMA

            module = None
            if value.get("format") == _ENGINE_REF_SCHEMA:
                module = value.get("module")
            elif value.get("format") == PYTHON_ENGINE_REF_FORMAT:
                binding = value.get("binding")
                if isinstance(binding, Mapping):
                    module = binding.get("rulespec_path")
            if isinstance(module, str) and module in unit_modules:
                return True
        return any(
            _precalibration_reference_embeds(item, references, unit_modules)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, list | tuple):
        return any(
            _precalibration_reference_embeds(item, references, unit_modules)
            for item in value
        )
    if not isinstance(value, str):
        return False
    if any(
        reference in value or json.dumps(reference, ensure_ascii=False)[1:-1] in value
        for reference in references
    ):
        return True
    try:
        decoded = json.loads(value)
    except (ValueError, TypeError):
        decoder = json.JSONDecoder()
        for index, character in enumerate(value):
            if character not in {"{", '"'}:
                continue
            try:
                embedded, _ = decoder.raw_decode(value, index)
            except ValueError:
                continue
            if _precalibration_reference_embeds(embedded, references, unit_modules):
                return True
        return False
    return decoded != value and _precalibration_reference_embeds(
        decoded, references, unit_modules
    )


def _precalibration_reference_guard(
    spec: Mapping[str, object],
    *,
    graph: Graph | None = None,
    config: TransportGraphConfig | None = None,
) -> None:
    """Refuse hold-out or benefit-unit engine semantics upstream of calibration.

    Without a graph, check the declaration's structural-base membership and
    artifact ancestry. This permits a packaged declaration check before donor
    preparation. Canonical engine-reference documents are matched to the
    benefit-unit bindings' module paths even before references are prepared.
    A composed graph also checks its exact compiled predecessors and the
    prepared references, including references embedded in JSON text.
    Benefit-unit attributes remain allowed; excluded columns are the unit
    rules bindings' outputs and the group encoder's declared outputs.
    """
    if not compose_module._has_resource(spec, compose_module.GRAPH_RESOURCE):
        return
    document = compose_module.transport_graph_document(spec)
    rows = {
        row["id"]: row
        for row in (
            compose_module._mapping(item, "transport node")
            for item in compose_module._sequence(
                document.get("nodes"), "transport nodes"
            )
        )
    }
    unit_entity = compose_module.transport_resource(spec, "benefit_unit_rule").get(
        "entity"
    )
    bindings = compose_module.transport_resource(spec, "axiom_rules_bindings").get(
        "bindings", ()
    )
    unit_bindings = {
        row["id"]: row
        for row in compose_module._sequence(bindings, "axiom_rules_bindings.bindings")
        if isinstance(row, Mapping) and row.get("entity") == unit_entity
    }
    unit_modules = frozenset(
        row["rulespec_path"]
        for row in unit_bindings.values()
        if isinstance(row.get("rulespec_path"), str)
    )
    forbidden_columns = {
        (unit_entity, variable)
        for row in unit_bindings.values()
        for variable in row.get("variables", ())
    }
    for row in rows.values():
        if row.get("kernel") == "concepts.encode_groups@1":
            forbidden_columns.update(
                (output["entity"], output["column"])
                for output in row.get("outputs", ())
                if output.get("entity") == unit_entity
            )
    references = {
        row["engine_ref"]
        for row in unit_bindings.values()
        if isinstance(row.get("engine_ref"), str) and row["engine_ref"]
    }
    if config is not None:
        references.update(
            config.engine_refs[name]
            for name in unit_bindings
            if name in config.engine_refs
        )
    endpoints = compose_module._mapping(
        document.get("endpoints"), "transport endpoints"
    )
    roots = {
        row["id"]
        for row in rows.values()
        if isinstance(row.get("kernel"), str) and row["kernel"].startswith("calibrate.")
    }
    if isinstance(endpoints.get("calibration"), str):
        roots.add(endpoints["calibration"])
    predecessors = {name: set() for name in rows}
    for name, row in rows.items():
        base = row.get("base")
        if isinstance(base, str):
            predecessors[name].add(base)
            predecessors[name].update(
                item["id"] for item in rows.values() if item.get("population") == base
            )
        elif isinstance(row.get("population"), str):
            predecessors[name].add(row["population"])
        predecessors[name].update(
            edge["producer"] for edge in row.get("artifact_inputs", ())
        )
    for name in _precalibration_reference_ancestors(predecessors, roots):
        row = rows[name]
        reason = None
        if row.get("rules_binding") in unit_bindings:
            reason = "benefit-unit rules binding"
        for value in compose_module._mapping(
            row.get("params", {}), "node params"
        ).values():
            for selection in _precalibration_reference_selections(value):
                if selection.get("prepared") == "engine_refs":
                    reason = "prepared engine_refs map"
                if isinstance(selection.get("resource"), str):
                    resource = compose_module._stem(selection["resource"])
                    if resource == "holdout_references":
                        reason = "holdout_references selection"
                    if (
                        resource == compose_module.GRAPH_RESOURCE
                        and selection.get("encoding") == "sha256"
                    ):
                        reason = "graph resource's own digest"
                    if selection.get(
                        "encoding", "value"
                    ) != "sha256" and compose_module._has_resource(spec, resource):
                        selected = compose_module.transport_resource(spec, resource)
                        for component in selection.get("path", ()):
                            if (
                                not isinstance(selected, Mapping)
                                or component not in selected
                            ):
                                selected = None
                                break
                            selected = selected[component]
                        if _precalibration_reference_embeds(
                            selected, references, unit_modules
                        ):
                            reason = "selected data embedding a benefit-unit engine reference"
        columns = {
            (item["entity"], column)
            for item in row.get("inputs", ())
            for column in item["columns"]
        }
        columns.update(
            (item["entity"], item["column"]) for item in row.get("outputs", ())
        )
        if columns & forbidden_columns:
            reason = "benefit-unit engine input or output column"
        if any(
            isinstance(source, str) and "holdout" in source.lower()
            for source in row.get("sources", ())
        ):
            reason = "hold-out source"
        if _precalibration_reference_embeds(
            row.get("params", {}), references, unit_modules
        ):
            reason = "embedded benefit-unit engine reference"
        if reason:
            raise ValueError(
                f"Transport calibration ancestry: node {name!r} carries {reason}."
            )
    if graph is None:
        return
    compiled = compile_graph(graph)
    roots.update(
        node.id for node in graph.nodes if node.kernel.startswith("calibrate.")
    )
    for node in graph.nodes:
        if node.kernel == "concepts.encode_groups@1":
            forbidden_columns.update(
                (item.entity, item.column)
                for item in node.outputs
                if item.entity == unit_entity
            )
    for name in _precalibration_reference_ancestors(compiled.predecessors, roots):
        node = graph.node(name)
        columns = {
            (item.entity, column) for item in node.inputs for column in item.columns
        }
        columns.update((item.entity, item.column) for item in node.outputs)
        if (
            any("holdout" in source.lower() for source in node.sources)
            or _precalibration_reference_embeds(node.params, references, unit_modules)
            or columns & forbidden_columns
        ):
            raise ValueError(
                f"Transport calibration ancestry: node {name!r} carries hold-out or benefit-unit engine semantics."
            )


# ---------------------------------------------------------------------------
# Shared fixtures (built once; tests that edit inputs copy what they change)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    return make_composed_fixture(tmp_path_factory.mktemp("composed"))


@pytest.fixture(scope="module")
def calibration_ancestry_composed(tmp_path_factory):
    """Prepare real engine references without unrelated gate binding identities."""
    from microcosm.build.transport import registry as registry_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry_module, "register_gate_kernel", lambda *args: None)
        return make_composed_fixture(tmp_path_factory.mktemp("calibration-ancestry"))


@pytest.fixture(scope="module")
def feed_compilation_config(composed):
    """Compose every feed compiler, with an optional explicit gap pin."""

    def make_config(*, gap_pin=None):
        def extension(spec, config, skeleton):
            nodes = toy_extension(spec, config, skeleton)
            compare = next(node for node in nodes if node.kernel == "takeup.compare@1")
            references = json.loads(compare.params["references"])
            params = {
                **compare.params,
                "annualisation_factors": canonical_json(
                    {row["name"]: 1 for row in references["target_references"]}
                ).decode("utf-8"),
                "scenario": "toy",
            }
            if gap_pin is not None:
                params["chronicle_feed_pin"] = gap_pin
            gap = replace(
                compare,
                id="nz.test.validate.gap",
                kernel="takeup.gap@1",
                params=params,
                artifact_outputs=(ArtifactOutput("gap", GAP_TYPE),),
            )
            return (*nodes, gap)

        return replace(composed.config, extensions=(extension,))

    return make_config


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


def test_composition_binds_feed_pin_to_skeleton_preparation_and_extensions(
    composed, feed_compilation_config
):
    rows = [
        json.loads(line)
        for line in composed.sources["nz_calibration_facts"].read_text().splitlines()
    ]
    _artifact, pin = write_pinned_artifact(composed.root / "pinned-facts", rows)
    spec = copy.deepcopy(composed.spec)
    spec["resources"]["chronicle_feed"] = pin
    config = feed_compilation_config()
    graph = compose_transport_graph(spec, config)
    compilers = [
        node
        for node in graph.nodes
        if node.kernel
        in {
            "transport.create@1",
            "targets.compile@1",
            "takeup.compare@1",
            "takeup.gap@1",
        }
    ]
    assert {node.kernel for node in compilers} == {
        "transport.create@1",
        "targets.compile@1",
        "takeup.compare@1",
        "takeup.gap@1",
    }
    for node in compilers:
        assert json.loads(node.params["chronicle_feed_pin"]) == pin
    declaration = skeleton_create_declaration(spec)
    assert dict(declaration.params) == dict(graph.node(declaration.id).params)
    changed = copy.deepcopy(spec)
    changed["resources"]["chronicle_feed"]["manifest_sha256"] = "0" * 64
    altered = compose_transport_graph(changed, config)
    for node in compilers:
        assert node.params != altered.node(node.id).params
    for node in graph.nodes:
        if node not in compilers:
            assert node.params == altered.node(node.id).params


def test_composition_refuses_a_compiler_pin_that_overrides_the_country_pin(composed):
    spec = copy.deepcopy(composed.spec)
    spec["resources"]["chronicle_feed"] = {"manifest_sha256": "1" * 64}
    declaration = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["kernel"] == "targets.compile@1"
    )
    declaration["params"]["chronicle_feed_pin"] = "{}"
    with pytest.raises(ValueError, match="feed pin differs"):
        compose_transport_graph(spec, composed.config)


def test_composition_refuses_a_gap_pin_that_overrides_the_country_pin(
    composed, feed_compilation_config
):
    spec = copy.deepcopy(composed.spec)
    spec["resources"]["chronicle_feed"] = {"manifest_sha256": "1" * 64}
    with pytest.raises(ValueError, match=r"takeup\.gap@1 Chronicle feed pin differs"):
        compose_transport_graph(spec, feed_compilation_config(gap_pin="{}"))


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


def test_a1_a2_three_engines_in_one_run(composed, cold, tmp_path, record_property):
    graph, first, store, exported, seconds = cold
    record_property("cold_run_seconds", seconds)
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


def test_holdout_and_family_semantics_have_no_calibration_ancestry(
    calibration_ancestry_composed,
):
    composed = calibration_ancestry_composed
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
@pytest.mark.parametrize("guard", ["reference", "absent", "empty"])
def test_compose_refuses_contaminated_calibration_ancestry(
    calibration_ancestry_composed, monkeypatch, violation, guard
):
    composed = calibration_ancestry_composed
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
    with monkeypatch.context() as reference:
        reference.setattr(
            compose_module,
            "validate_transport_calibration_ancestry",
            _precalibration_reference_guard,
        )
        expected = _precalibration_outcome(
            lambda document: compose_transport_graph(document, composed.config), spec
        )
    _precalibration_guard_variant(spec, guard, monkeypatch)
    with pytest.raises(
        ValueError, match="calibration ancestry|graph resource.*digest"
    ) as error:
        compose_transport_graph(spec, composed.config)
    assert (type(error.value), str(error.value)) == expected


def _precalibration_stage(spec, binding, columns=None):
    row = next(
        item
        for item in spec["resources"]["axiom_rules_bindings"]["bindings"]
        if item["id"] == binding
    )
    return {
        "id": f"toy.precal.{binding}",
        "rules_binding": binding,
        "rulespec_path": row["rulespec_path"],
        "outputs": [
            {
                "entity": row["entity"],
                "column": column,
                "purpose": "classification"
                if column.endswith("eligible")
                else "calibration_measure",
            }
            for column in (row["variables"] if columns is None else columns)
        ],
        "citation": "Synthetic pre-calibration method card.",
    }


def _precalibration_spec(composed, binding="family_housing"):
    spec = copy.deepcopy(composed.spec)
    nodes = spec["resources"]["transport_graph"]["nodes"]
    calibration = next(row for row in nodes if row["id"] == CALIBRATION)
    if binding == "family_housing":
        encoder = next(row for row in nodes if row["id"] == "nz.as.inputs")
        encoder["population"] = "nz.open"
        nodes.append(
            {
                "id": "nz.precal.family",
                "kernel": RULES,
                "population": "nz.open",
                "sources": ["rulespec_nz"],
                "inputs": [{"entity": "family", "columns": list(FAMILY_INPUTS)}],
                "outputs": [
                    {"entity": "family", "column": name, "dtype": output["dtype"]}
                    for name, output in PROGRAMS[binding]["outputs"].items()
                ],
                "rules_binding": binding,
            }
        )
        columns = ["toy_family_weekly_amount"]
        entity = "family"
    else:
        columns = list(PROGRAMS[binding]["outputs"])
        entity = "person"
    measure = next(column for column in columns if not column.endswith("eligible"))
    calibration["inputs"].append({"entity": entity, "columns": [measure]})
    spec["resources"]["precalibration_stages"] = {
        "schema_version": 1,
        "stages": [_precalibration_stage(spec, binding, columns)],
    }
    return spec


def _precalibration_outcome(guard, spec, **kwargs):
    try:
        guard(spec, **kwargs)
    except (TypeError, ValueError) as error:
        return type(error), str(error)
    return None


def _precalibration_guard_variant(spec, guard, monkeypatch):
    if guard == "empty":
        spec["resources"]["precalibration_stages"] = {"schema_version": 1, "stages": []}
    if guard == "reference":
        monkeypatch.setattr(
            compose_module,
            "validate_transport_calibration_ancestry",
            _precalibration_reference_guard,
        )


@pytest.mark.parametrize("guard", ["reference", "absent", "empty"])
def test_precalibration_ancestry_unlisted_person_differential(
    calibration_ancestry_composed, monkeypatch, guard
):
    composed = calibration_ancestry_composed
    spec = copy.deepcopy(composed.spec)
    original = graph_to_json(composed.graph())
    _precalibration_guard_variant(spec, guard, monkeypatch)
    graph = compose_transport_graph(spec, composed.config)
    assert graph_to_json(graph) == original
    assert (
        graph.node("nz.rules.main_benefits").params["engine_ref"]
        == (composed.config.engine_refs["main_benefit"])
    )
    assert _precalibration_outcome(
        compose_module.validate_transport_calibration_ancestry,
        spec,
        graph=graph,
        config=composed.config,
    ) == _precalibration_outcome(
        _precalibration_reference_guard, spec, graph=graph, config=composed.config
    )


def test_precalibration_ancestry_listed_unit_stage_composes(
    calibration_ancestry_composed,
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    graph = compose_transport_graph(spec, composed.config)
    declared = {("family", "toy_family_weekly_amount")}
    actual = {
        (item.entity, column)
        for item in graph.node(CALIBRATION).inputs
        for column in item.columns
        if item.entity == "family"
    }
    assert actual == declared
    assert {item.column for item in graph.node("nz.precal.family").outputs} == set(
        PROGRAMS["family_housing"]["outputs"]
    )
    assert len(declared) < len(graph.node("nz.precal.family").outputs)


def test_precalibration_ancestry_listed_person_stage_composes(
    calibration_ancestry_composed,
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed, "main_benefit")
    graph = compose_transport_graph(spec, composed.config)
    assert Slice("person", ("toy_main_payment",)) in graph.node(CALIBRATION).inputs
    assert (
        Slice(
            "person",
            (
                "take_up_seed",
                "toy_main_eligible",
                "toy_main_payment",
                "toy_pension_eligible",
                "toy_pension_payment",
            ),
        )
        in graph.node("nz.takeup.assign").inputs
    )


@pytest.mark.parametrize("nested", [False, True])
def test_precalibration_ancestry_selector_stage_in_both_passes(
    calibration_ancestry_composed, nested
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    graph = compose_transport_graph(spec, composed.config)
    row = next(
        item
        for item in spec["resources"]["transport_graph"]["nodes"]
        if item["id"] == "nz.precal.family"
    )
    del row["rules_binding"]
    row["kernel"] = "simulate.solve_zero@1"
    selector = {"binding": "family_housing", "field": "engine_ref"}
    row["params"] = {"engine_ref": [{"nested": selector}] if nested else selector}
    compose_module.validate_transport_calibration_ancestry(spec, config=composed.config)
    reference = composed.config.engine_refs["family_housing"]
    graph = replace(
        graph,
        nodes=tuple(
            replace(
                node,
                kernel="simulate.solve_zero@1",
                params={
                    "engine_ref": json.dumps({"nested": [reference]})
                    if nested
                    else reference
                },
            )
            if node.id == row["id"]
            else node
            for node in graph.nodes
        ),
    )
    compose_module.validate_transport_calibration_ancestry(
        spec, graph=graph, config=composed.config
    )


def test_precalibration_ancestry_unlisted_unit_stage_refuses(
    calibration_ancestry_composed,
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    spec["resources"]["precalibration_stages"]["stages"] = [
        _precalibration_stage(spec, "main_benefit")
    ]
    with pytest.raises(ValueError, match="not a declared pre-calibration stage"):
        compose_transport_graph(spec, composed.config)


@pytest.mark.parametrize("contamination", ["holdout_source", "holdout_selection"])
@pytest.mark.parametrize("compiled", [False, True])
def test_precalibration_ancestry_holdout_fed_stage_refuses(
    calibration_ancestry_composed, contamination, compiled
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    graph = compose_transport_graph(spec, composed.config)
    if compiled:
        graph = replace(
            graph,
            nodes=tuple(
                replace(
                    node,
                    sources=("nz_holdout_facts",)
                    if contamination == "holdout_source"
                    else node.sources,
                )
                if node.id == "nz.as.inputs"
                else node
                for node in graph.nodes
            ),
        )
        if contamination == "holdout_selection":
            row = next(
                item
                for item in spec["resources"]["transport_graph"]["nodes"]
                if item["id"] == "nz.as.inputs"
            )
            # Its declaration lies after calibration, while the compiled
            # population makes it a predecessor. Selection provenance belongs
            # to the declaration; resolved parameter objects stay literal.
            row["population"] = "nz.as"
            row["params"]["holdout"] = {
                "resource": "holdout_references",
                "encoding": "json",
            }
    else:
        row = next(
            item
            for item in spec["resources"]["transport_graph"]["nodes"]
            if item["id"] == "nz.as.inputs"
        )
        if contamination == "holdout_source":
            row["sources"] = ["nz_holdout_facts"]
        else:
            row["params"]["holdout"] = {
                "resource": "holdout_references",
                "encoding": "json",
            }
    reason = "hold-out source|holdout_references selection"
    with pytest.raises(ValueError, match=f"node 'nz.as.inputs'.*({reason})"):
        compose_module.validate_transport_calibration_ancestry(
            spec, graph=graph if compiled else None, config=composed.config
        )


@pytest.mark.parametrize("module", [False, True])
def test_precalibration_ancestry_holdout_output_or_module_refuses(
    calibration_ancestry_composed, module
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    holdout = spec["resources"]["holdout_references"]["target_references"][0]
    holdout["entity"] = "family"
    if module:
        holdout["measure"] = "toy_family_holdout"
        bindings = spec["resources"]["axiom_rules_bindings"]["bindings"]
        extra = copy.deepcopy(
            next(row for row in bindings if row["id"] == "family_housing")
        )
        extra.update(id="family_evaluation", variables=[holdout["measure"]])
        bindings.append(extra)
    else:
        holdout["measure"] = "toy_family_weekly_amount"
    with pytest.raises(ValueError, match="precalibration_stages.*holdout"):
        compose_module.validate_transport_calibration_ancestry(spec)


@pytest.mark.parametrize("binding", ["family_housing", "main_benefit"])
@pytest.mark.parametrize("compiled", [False, True])
def test_precalibration_ancestry_undeclared_stage_variable_refuses(
    calibration_ancestry_composed, binding, compiled
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed, binding)
    graph = compose_transport_graph(spec, composed.config)
    stage = spec["resources"]["precalibration_stages"]["stages"][0]
    if binding == "main_benefit":
        stage["outputs"] = [
            row for row in stage["outputs"] if row["column"] != "toy_main_eligible"
        ]
        entity, variable = "person", "toy_main_eligible"
    else:
        entity, variable = "family", "toy_family_eligible"
    if compiled:
        # Keep the declaration clean: only this extension consumes the omitted
        # variable, so refusal must come from the compiled predecessor pass.
        if binding == "main_benefit":
            nodes = spec["resources"]["transport_graph"]["nodes"]
            for row in nodes:
                if "inputs" in row:
                    row["inputs"] = [
                        {
                            **item,
                            "columns": [
                                column
                                for column in item["columns"]
                                if column != variable
                            ],
                        }
                        for item in row["inputs"]
                        if any(column != variable for column in item["columns"])
                    ]
            graph = replace(
                graph,
                nodes=tuple(
                    replace(
                        node,
                        inputs=tuple(
                            Slice(
                                item.entity,
                                tuple(
                                    column
                                    for column in item.columns
                                    if column != variable
                                ),
                            )
                            for item in node.inputs
                            if any(column != variable for column in item.columns)
                        ),
                    )
                    for node in graph.nodes
                ),
            )
        consumer = Node(
            "nz.precal.extension.reader",
            "transport.probe@1",
            population="nz.open",
            inputs=(Slice(entity, (variable,)),),
        )
        graph = replace(graph, nodes=(*graph.nodes, consumer))
        name = consumer.id
    else:
        name = "nz.takeup.assign" if binding == "main_benefit" else "nz.targets.problem"
        row = next(
            item
            for item in spec["resources"]["transport_graph"]["nodes"]
            if item["id"] == name
        )
        if binding == "family_housing":
            row["inputs"].append({"entity": entity, "columns": [variable]})
    with pytest.raises(
        ValueError,
        match=f"node '{name}' carries undeclared pre-calibration stage output",
    ):
        compose_module.validate_transport_calibration_ancestry(
            spec, graph=graph if compiled else None, config=composed.config
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_precalibration_ancestry_shared_encode_output_refuses(
    calibration_ancestry_composed, compiled
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    graph = compose_transport_graph(spec, composed.config)
    if compiled:
        reader = Node(
            "nz.precal.extension.reader",
            "transport.probe@1",
            population="nz.open",
            inputs=(Slice("family", ("input_rent",)),),
        )
        graph = replace(graph, nodes=(*graph.nodes, reader))
        name = reader.id
    else:
        name = "nz.targets.problem"
        row = next(
            item
            for item in spec["resources"]["transport_graph"]["nodes"]
            if item["id"] == name
        )
        row["inputs"].append({"entity": "family", "columns": ["input_rent"]})
    with pytest.raises(ValueError, match=f"node '{name}' carries"):
        compose_module.validate_transport_calibration_ancestry(
            spec, graph=graph if compiled else None, config=composed.config
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "resource_key",
        "stage_key",
        "output_key",
        "duplicate_id",
        "empty_id",
        "unsafe_id",
        "missing_binding",
        "path_mismatch",
        "duplicate_binding",
        "schema_bool",
        "schema_float",
        "schema_version",
        "purpose",
        "variable",
        "wrong_entity",
        "empty_outputs",
        "duplicate_output",
        "stages_type",
    ],
)
def test_precalibration_ancestry_stage_schema_refuses(
    calibration_ancestry_composed, mutation
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    resource = spec["resources"]["precalibration_stages"]
    stage = resource["stages"][0]
    output = stage["outputs"][0]
    if mutation == "resource_key":
        resource["unknown"] = True
    elif mutation == "stage_key":
        stage["unknown"] = True
    elif mutation == "output_key":
        output["unknown"] = True
    elif mutation in {"duplicate_id", "duplicate_binding"}:
        other = copy.deepcopy(stage)
        if mutation == "duplicate_binding":
            other["id"] += ".again"
        resource["stages"].append(other)
    elif mutation in {"empty_id", "unsafe_id"}:
        stage["id"] = "" if mutation == "empty_id" else "../unsafe"
    elif mutation == "missing_binding":
        stage["rules_binding"] = "absent"
    elif mutation == "path_mismatch":
        stage["rulespec_path"] = "toy/other.yaml"
    elif mutation.startswith("schema_"):
        resource["schema_version"] = {
            "schema_bool": True,
            "schema_float": 1.0,
            "schema_version": 2,
        }[mutation]
    elif mutation == "purpose":
        output["purpose"] = "evaluation"
    elif mutation == "variable":
        output["column"] = "not_a_binding_variable"
    elif mutation == "wrong_entity":
        output["entity"] = "person"
    elif mutation == "empty_outputs":
        stage["outputs"] = []
    elif mutation == "duplicate_output":
        stage["outputs"].append(dict(output))
    else:
        resource["stages"] = {}
    with pytest.raises((TypeError, ValueError), match="precalibration_stages"):
        compose_module.validate_transport_calibration_ancestry(spec)


@pytest.mark.parametrize("compiled", [False, True])
def test_precalibration_ancestry_unlisted_selector_refuses(
    calibration_ancestry_composed, compiled
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed, "main_benefit")
    graph = compose_transport_graph(spec, composed.config)
    name = "nz.precal.unlisted.solve"
    if compiled:
        graph = replace(
            graph,
            nodes=(
                *graph.nodes,
                Node(
                    name,
                    "simulate.solve_zero@1",
                    population="nz.open",
                    params={
                        "engine_ref": composed.config.engine_refs["family_housing"]
                    },
                ),
            ),
        )
    else:
        spec["resources"]["transport_graph"]["nodes"].append(
            {
                "id": name,
                "kernel": "simulate.solve_zero@1",
                "population": "nz.open",
                "params": {
                    "engine_ref": {"binding": "family_housing", "field": "engine_ref"}
                },
            }
        )
    with pytest.raises(
        ValueError, match=f"node '{name}' carries not a declared pre-calibration stage"
    ):
        compose_module.validate_transport_calibration_ancestry(
            spec, graph=graph if compiled else None, config=composed.config
        )


@pytest.mark.parametrize("compiled", [False, True])
def test_precalibration_ancestry_mixed_unit_bindings_refuses(
    calibration_ancestry_composed, compiled
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    bindings = spec["resources"]["axiom_rules_bindings"]["bindings"]
    other = copy.deepcopy(
        next(row for row in bindings if row["id"] == "family_housing")
    )
    other.update(
        id="other_family",
        rulespec_path="toy/other_family.yaml",
        variables=["other_result"],
    )
    bindings.append(other)
    config = replace(
        composed.config,
        engine_refs={
            **composed.config.engine_refs,
            "other_family": "synthetic-other-family-engine-ref",
        },
    )
    graph = compose_transport_graph(spec, config)
    name = "nz.precal.family"
    if compiled:
        graph = replace(
            graph,
            nodes=tuple(
                replace(
                    node,
                    params={**node.params, "other": config.engine_refs["other_family"]},
                )
                if node.id == name
                else node
                for node in graph.nodes
            ),
        )
    else:
        row = next(
            item
            for item in spec["resources"]["transport_graph"]["nodes"]
            if item["id"] == name
        )
        row["params"] = {
            "components": [{"binding": "other_family", "field": "engine_ref"}]
        }
    with pytest.raises(
        ValueError,
        match=f"node '{name}' carries mixes listed and unlisted unit bindings",
    ):
        compose_module.validate_transport_calibration_ancestry(
            spec, graph=graph if compiled else None, config=config
        )


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


def _assert_activation_refuses_before_preparation(
    spec, composed, tmp_path, monkeypatch, match
):
    spies = []
    for owner, name in (
        (cli_module, "build_transport_registry"),
        (cli_module, "prepare_create_outputs"),
        (cli_module, "source_content_key"),
        (cli_module.TRANSPORT_CREATE, "run"),
    ):
        spy = Mock(side_effect=AssertionError(f"Activation entered {name}."))
        monkeypatch.setattr(owner, name, spy)
        spies.append(spy)
    with pytest.raises(ValueError, match=match):
        validate_transport_activation(spec)
    inventory = tmp_path / "inventory"
    with pytest.raises(ValueError, match=match):
        prepare_transport_build(
            spec,
            sources=composed.sources,
            engines_by_binding=composed.engines_by_binding,
            inventory_store=inventory,
        )
    for spy in spies:
        spy.assert_not_called()
    assert not inventory.exists()


@pytest.mark.parametrize(
    ("encoding", "resource", "reason"),
    [
        ("value", {"factor": None}, "not activated"),
        ("json", {"factor": None}, "not activated"),
        ("value", {}, "no selected path"),
        ("json", {"factor": 1}, "json selection must be an object"),
        ("value", {"factor": {}}, "value selection is an object"),
    ],
)
def test_activation_refuses_invalid_selected_values_before_preparation(
    composed, tmp_path, monkeypatch, encoding, resource, reason
):
    spec = copy.deepcopy(composed.spec)
    spec["resources"]["currency"] = resource
    currency = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["kernel"] == TRANSPORT_CURRENCY.ref
    )
    currency["params"]["probe"] = [
        {"resource": "currency", "path": ["factor"], "encoding": encoding}
    ]
    _assert_activation_refuses_before_preparation(
        spec, composed, tmp_path, monkeypatch, reason
    )
    with pytest.raises(ValueError) as error:
        validate_transport_activation(spec)
    assert "currency" in str(error.value)
    assert "['factor']" in str(error.value)


@pytest.mark.parametrize("name", [None, "", 7, ["currency"]])
def test_activation_refuses_a_malformed_resource_name_before_preparation(
    composed, tmp_path, monkeypatch, name
):
    spec = copy.deepcopy(composed.spec)
    currency = next(
        row
        for row in spec["resources"]["transport_graph"]["nodes"]
        if row["kernel"] == TRANSPORT_CURRENCY.ref
    )
    currency["params"]["probe"] = [
        {"resource": name, "path": ["factor"], "encoding": "value"}
    ]
    _assert_activation_refuses_before_preparation(
        spec, composed, tmp_path, monkeypatch, "must name a nonempty string"
    )


@pytest.mark.parametrize("gap", ["empty", "seed", "judgment", "rate"])
def test_activation_refuses_invalid_receipt_contract_before_preparation(
    composed, tmp_path, monkeypatch, gap
):
    spec = copy.deepcopy(composed.spec)
    receipt = spec["resources"]["receipt_contract"]
    if gap == "empty":
        receipt["receipts"] = {}
    elif gap == "seed":
        receipt["receipts"]["seed_column"] = None
    else:
        key = "judgment_column" if gap == "judgment" else "rate"
        receipt["receipts"]["programs"][0][key] = None
    _assert_activation_refuses_before_preparation(
        spec, composed, tmp_path, monkeypatch, "receipt_contract.*receipts"
    )


def test_activation_accepts_optional_json_nulls(composed):
    spec = copy.deepcopy(composed.spec)
    spec["resources"]["receipt_contract"]["receipts"]["programs"][0][
        "payment_column"
    ] = None
    validate_transport_activation(spec)


def _transport_spec_with_reference_profile(composed, suffix=""):
    spec = copy.deepcopy(composed.spec)
    row = spec["resources"]["target_references"]["target_references"][0]
    selector = {
        "aggregate_fact_key": "atom",
        "source_name": "publisher",
        "source_measure_id": "count",
        "source_concept": "population",
        "entity_name": row["entity"],
        "record_set_id": "population",
        "period_type": "calendar_year",
        "period_value": 2023,
        "geography_level": "country",
        "geography_id": "toy",
        "geography_vintage": "unstated",
        "dimensions": ["category"],
        "dimension_values": {"category": 1},
    }
    row.pop("ledger_fact_key")
    row["ledger_selector"] = selector
    row["metadata"]["activation_status"] = "requires_reference_profile_resolution"
    total = {key: value for key, value in selector.items() if key != "dimension_values"}
    total.update(aggregate_fact_key="total", dimensions=[])
    transformation = {
        "reference_names": [row["name"]],
        "operation": "allocated_residual_by_ratio",
        "value_operands": [
            {"role": "base", "expected_member_count": 1},
            {"role": "total", **total},
            {
                "role": "partition",
                "member_selectors": [selector],
                "allocation_member_keys": ["atom"],
            },
        ],
    }
    spec["resources"]["reference_profiles"] = {
        "schema_version": 1,
        "reference_resource": "target_references" + suffix,
        "interval_resource": "profile_intervals" + suffix,
        "dimensions": {
            "residual": {
                "default": "central",
                "profiles": {"central": {"transformations": [transformation]}},
            },
            "aging_index": {
                "default": "central",
                "profiles": {
                    "central": {"deferred": True, "transformations": []},
                    "low": {"available": False, "transformations": []},
                },
            },
        },
    }
    spec["resources"]["profile_intervals"] = {"bands": []}
    return spec, row


@pytest.mark.parametrize("suffix", ["", ".json"])
def test_preflight_and_composition_use_the_same_resolved_reference_view(
    composed, suffix
):
    spec, row = _transport_spec_with_reference_profile(composed, suffix)
    validate_transport_activation(spec)
    graph = compose_transport_graph(spec, composed.config)
    compiler = next(
        node
        for node in graph.nodes
        if node.kernel == "targets.compile@1"
        and json.loads(node.params["references"])["target_references"][0]["name"]
        == row["name"]
    )
    references = json.loads(compiler.params["references"])
    receipt = json.loads(compiler.params["reference_resolution"])
    assert (
        references["target_references"][0]["metadata"]["activation_status"] == "active"
    )
    assert (
        compiler.params["references_sha256"]
        == hashlib.sha256(canonical_json(references)).hexdigest()
    )
    assert (
        receipt["effective_references_sha256"] == compiler.params["references_sha256"]
    )
    assert (
        row["metadata"]["activation_status"] == "requires_reference_profile_resolution"
    )
    with pytest.raises(ValueError, match="unavailable"):
        compose_module.reference_profile_view(spec, {"aging_index": "low"})


def test_preflight_checks_the_selected_variant_reference_status(composed):
    spec, row = _transport_spec_with_reference_profile(composed)
    compiler = copy.deepcopy(
        next(
            node
            for node in spec["resources"]["transport_graph"]["nodes"]
            if node["kernel"] == "targets.compile@1"
            and node["params"]["references"]["resource"] == "target_references"
        )
    )
    compiler["id"] = "{variant}.targets"
    spec["resources"]["entitlement_graph"] = {
        "schema_version": 1,
        "variant_nodes": [compiler],
        "variant_gap": "{variant}.targets",
    }
    variant = {
        "id": "candidate",
        "tier": "calibration",
        "knobs": {"residual": "central"},
    }
    spec["resources"]["scenarios"]["scenarios"] = [variant]
    validate_transport_activation(spec)
    profiles = spec["resources"]["reference_profiles"]["dimensions"]["residual"][
        "profiles"
    ]
    profiles["unresolved"] = {"transformations": []}
    variant["knobs"]["residual"] = "unresolved"
    with pytest.raises(
        ValueError,
        match="placeholder reference target_references\\." + re.escape(row["name"]),
    ):
        validate_transport_activation(spec)


def test_factor_context_is_read_without_composition_feed_io(tmp_path, monkeypatch):
    _, pin, _, _, spec = factor_transport_fixture(tmp_path)
    spec["resources"]["chronicle_feed"] = pin

    def no_feed_io(*args, **kwargs):
        raise AssertionError("Composition must not load the consumer artifact.")

    monkeypatch.setattr(
        "microcosm.build.ledger_artifact.load_ledger_consumer_artifact", no_feed_io
    )
    view = compose_module.reference_profile_view(spec)
    assert view[2]["selected_factors"]["toy_factor"]["operational_factor"] == {
        "numerator": 1100000,
        "denominator": 1000000,
    }
    assert all(
        row["metadata"]["activation_status"] == "requires_prerequisites"
        for row in view[0]["target_references"]
    )
    assert compose_module.transport_resource(spec, "target_references") == view[0]


def test_factor_context_ignores_feed_pin_prose(tmp_path):
    _, pin, _, _, spec = factor_transport_fixture(tmp_path)
    spec["resources"]["chronicle_feed"] = pin
    before = compose_module.reference_profile_view(spec)
    pin["build"] = "Reviewed prose edit"
    pin["source_repo"] = "Reviewed repository description"
    assert compose_module.reference_profile_view(spec) == before


@pytest.mark.parametrize(
    "defect",
    ["missing", "wrong_feed", "wrong_commit", "wrong_manifest", "wrong_factor"],
)
def test_preflight_refuses_missing_or_incompatible_default_factor_context(
    tmp_path, defect
):
    _, pin, _, _, spec = factor_transport_fixture(tmp_path)
    spec["resources"]["chronicle_feed"] = pin
    if defect == "missing":
        spec["resources"].pop("factor_context")
    elif defect == "wrong_feed":
        spec["resources"]["chronicle_feed"] = {**pin, "facts_sha256": "0" * 64}
    elif defect == "wrong_commit":
        pin["source_commit"] = "d" * 40
    elif defect == "wrong_manifest":
        pin["manifest_sha256"] = "0" * 64
    else:
        spec["resources"]["factor_context"]["factors"][0]["operational_factor"][
            "numerator"
        ] += 1
    with pytest.raises(
        ValueError, match="missing resource|incompatible|factor differs"
    ):
        validate_transport_activation(spec)


def test_preflight_resolves_empty_knobs_only_once(tmp_path, monkeypatch):
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    original = compose_module.reference_profile_view
    selections = []

    def recording_view(spec, selection=None):
        selections.append(selection)
        return original(spec, selection)

    monkeypatch.setattr(compose_module, "reference_profile_view", recording_view)
    with pytest.raises(ValueError, match="Transport spec is not activated"):
        validate_transport_activation(spec)
    assert selections
    assert all(selection is None for selection in selections)


def test_preflight_validates_scenario_factor_context_before_activation(tmp_path):
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    profiles = spec["resources"]["reference_profiles"]["dimensions"]["index"][
        "profiles"
    ]
    profiles["low"]["transformations"][0]["factor_id"] = "missing-scenario-factor"
    assert compose_module.reference_profile_view(spec)[2]["selection"] == {
        "index": "central"
    }
    spec["resources"]["scenarios"]["scenarios"].append(
        {"id": "variant", "knobs": {"index": "low"}}
    )
    with pytest.raises(ValueError, match="Missing selected factor"):
        validate_transport_activation(spec)


def test_declared_ageing_scenarios_bind_selected_evidence_and_preserve_locality():
    spec = load_transport_spec("nz")
    profiles = spec["resources"]["reference_profiles"]
    receipts = {}
    for scenario in spec["resources"]["scenarios"]["scenarios"]:
        receipt = compose_module.reference_profile_view(spec, scenario["knobs"])[2]
        receipts[scenario["id"]] = receipt
        profile = {"V1": "low", "V2": "high"}.get(scenario["id"], "central")
        factor_id = f"ird_ty2024_to_ty2027_{profile}"
        assert receipt["selection"] == {
            "aging_index": profile,
            "census_residual_allocation": (
                "small_households" if scenario["id"] == "V4" else "mar"
            ),
            "family_discrepancy": "central",
        }
        assert set(receipt["selected_factors"]) == {factor_id}
        entry = next(
            entry
            for entry in spec["resources"]["reference_aging"]["factors"]
            if entry["declaration"]["id"] == factor_id
        )
        assert receipt["selected_factors"][factor_id] == entry
        chosen = {
            dimension: {
                "name": name,
                "declaration": profiles["dimensions"][dimension]["profiles"][name],
            }
            for dimension, name in receipt["selection"].items()
        }
        assert (
            receipt["selected_declarations_sha256"]
            == hashlib.sha256(canonical_json(chosen)).hexdigest()
        )
    profiles["dimensions"]["aging_index"]["profiles"]["high"]["notes"] = (
        "Reviewed edit confined to the unselected high profile."
    )
    for scenario in spec["resources"]["scenarios"]["scenarios"]:
        revised = compose_module.reference_profile_view(spec, scenario["knobs"])[2]
        if scenario["id"] == "V2":
            assert (
                revised["reference_selection_sha256"]
                != receipts["V2"]["reference_selection_sha256"]
            )
        else:
            assert revised == receipts[scenario["id"]]


def _interval_prepare_spec():
    base, profiles, context, intervals = interval_profile_fixture()
    table = intervals["tables"][0]
    spec = {
        "country": "xx",
        "resources": {
            "target_references": base,
            "reference_profiles": profiles,
            "factor_context": context,
            "profile_intervals": intervals,
            "calibration_bindings": interval_bindings(table),
        },
    }
    row = {
        "id": "prepare",
        "kernel": "calibration.prepare@1",
        "population": "open",
        "params": {
            "bindings": {"resource": "calibration_bindings", "encoding": "json"},
            "references": {"resource": "target_references", "encoding": "json"},
            "references_sha256": {
                "resource": "target_references",
                "encoding": "sha256",
            },
        },
        "outputs": [
            {"entity": "person", "column": band["indicator"], "dtype": "bool"}
            for band in table["bands"]
        ],
    }
    return spec, row


_INTERVAL_CONFIG = TransportGraphConfig(
    engine_refs={}, create_outputs=(Owned("person", "age", "int64"),)
)


def _prepare_node(spec, row, knobs, *, interval_knobs="same"):
    params = {
        name: compose_module._resource_param(spec, value, profile_selection=knobs)
        for name, value in row["params"].items()
    }
    if interval_knobs != "same":
        params["intervals"] = compose_module._resource_param(
            spec,
            {"resource": "profile_intervals", "encoding": "json"},
            profile_selection=interval_knobs,
        )
    return compose_module._node(
        spec,
        row,
        _INTERVAL_CONFIG,
        {},
        resolved_params=params,
        profile_selection=knobs,
    )


def _run_prepare(node, incomes):
    tables = {
        "person": pd.DataFrame(
            {
                "person_id": range(len(incomes)),
                INCOME_COLUMN: incomes,
                UNIVERSE_COLUMN: ["included"] * len(incomes),
            }
        )
    }
    return CALIBRATION_PREPARE.run(
        KernelContext(
            node=node,
            tables=tables,
            weights={},
            strata=pd.Series([], dtype=object, name="stratum"),
            params=node.params,
            rng=np.random.default_rng(0),
            sources={},
            artifacts={},
        )
    )


def test_composition_supplies_each_variant_its_own_effective_interval_companion():
    spec, row = _interval_prepare_spec()
    incomes = [104_999, 107_000, 112_000, 114_000, 116_000]
    first_band = {}
    for profile, knobs in (
        ("central", None),
        ("low", {"index": "low"}),
        ("high", {"index": "high"}),
    ):
        node = _prepare_node(spec, row, knobs)
        view = compose_module.reference_profile_view(spec, knobs)
        assert node.params["intervals"] == canonical_json(view[1]).decode("utf-8")
        assert json.loads(node.params["reference_resolution"]) == view[2]
        result = _run_prepare(node, incomes)
        first_band[profile] = result.columns["person", "toy_income_band_b0"].tolist()
        evidence = json.loads(result.artifacts["preparation"])
        assert (
            evidence["effective_intervals_sha256"]
            == view[2]["effective_intervals_sha256"]
        )
    # 100,000 source cents become 105,000, 110,000 and 115,000: each variant
    # regenerates its indicators from its own cuts rather than reusing S0's.
    assert first_band == {
        "low": [True, False, False, False, False],
        "central": [True, True, False, False, False],
        "high": [True, True, True, True, False],
    }


def test_composition_refuses_a_supplied_interval_document_from_another_profile():
    spec, row = _interval_prepare_spec()
    row = copy.deepcopy(row)
    row["params"]["intervals"] = {"resource": "profile_intervals", "encoding": "json"}
    node = _prepare_node(spec, row, {"index": "low"})
    assert (
        json.loads(node.params["intervals"])["tables"][0]["scaling"]["factor_id"]
        == "toy_low"
    )
    with pytest.raises(ValueError, match="differs from its effective interval"):
        _prepare_node(spec, row, {"index": "low"}, interval_knobs=None)


def test_declared_scenarios_scale_national_cuts_with_their_own_factor():
    from fractions import Fraction

    spec = load_transport_spec("nz")
    source = spec["resources"]["reference_intervals"]
    by_id = {table["id"]: table for table in source["tables"]}
    seen = {}
    for scenario in spec["resources"]["scenarios"]["scenarios"]:
        _, intervals, receipt = compose_module.reference_profile_view(
            spec, scenario["knobs"]
        )
        profile = {"V1": "low", "V2": "high"}.get(scenario["id"], "central")
        tables = {table["id"]: table for table in intervals["tables"]}
        national = tables["ird_taxable_income_national_ty2024"]
        assert national["scaling"]["factor_id"] == f"ird_ty2024_to_ty2027_{profile}"
        assert (
            tables["ird_taxable_income_age_ty2024"]
            == by_id["ird_taxable_income_age_ty2024"]
        )
        factor = national["scaling"]["operational_factor"]
        rational = Fraction(factor["numerator"], factor["denominator"])
        for band, original in zip(
            national["bands"],
            by_id["ird_taxable_income_national_ty2024"]["bands"],
            strict=True,
        ):
            if original.get("upper_cent") is not None:
                upper = band["upper_cent"]
                assert Fraction(upper["numerator"], upper["denominator"]) == (
                    original["upper_cent"] * rational
                )
        seen.setdefault(profile, set()).add(receipt["effective_intervals_sha256"])
    assert all(len(hashes) == 1 for hashes in seen.values())
    assert len(set().union(*seen.values())) == 3


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
    # The dependent-child rule is settled from the Social Security Act 2018
    # Schedule 2, so it is no longer an activation gap.
    assert "benefit_unit_rule.dependent_child" not in str(error.value)
    for resource in ("precal_references", "target_references", "holdout_references"):
        for reference in compose_module.transport_resource(spec, resource)[
            "target_references"
        ]:
            if reference["metadata"].get("activation_status", "active") != "active":
                assert f"placeholder reference {resource}.{reference['name']}" in str(
                    error.value
                )
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


def test_cli_main_refuses_the_packaged_spec_and_writes_nothing(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(cli_module, "_today", lambda: date(2026, 10, 24))
    out = tmp_path / "out"
    assert cli_module.main(["--country", "nz", "--out", str(out)]) == 1
    message = capsys.readouterr().err
    for gap in _GAPS:
        assert gap in message, gap
    assert not out.exists()


@pytest.mark.parametrize("guard", ["reference", "absent", "empty"])
def test_packaged_holdout_references_are_disjoint_from_calibration_references(
    monkeypatch, guard
):
    spec = load_transport_spec("nz")

    def names(resource):
        rows = spec["resources"][resource]["target_references"]
        return {row["name"] for row in rows}

    calibration = names("target_references") | names("precal_references")
    assert names("holdout_references").isdisjoint(calibration)
    _precalibration_guard_variant(spec, guard, monkeypatch)
    assert _precalibration_outcome(_precalibration_reference_guard, spec) is None
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


_AFTER_FACTOR_EXPIRY = date(2100, 1, 1)


@pytest.mark.parametrize("resume", ["auto", "forbid"])
@pytest.mark.parametrize("selection", ["default", "scenario"])
def test_cli_main_refuses_expired_factors_before_create(
    tmp_path, monkeypatch, capsys, resume, selection
):
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    if selection == "scenario":
        spec["resources"]["reference_profiles"]["dimensions"]["index"]["profiles"][
            "central"
        ]["transformations"] = []
        spec["resources"]["scenarios"]["scenarios"].append(
            {"id": "variant", "knobs": {"index": "low"}}
        )
    monkeypatch.setattr(cli_module, "_today", lambda: _AFTER_FACTOR_EXPIRY)
    monkeypatch.setattr(cli_module, "load_transport_spec", lambda source: spec)
    later = _mock_steps_after_the_currency_check(monkeypatch)
    out = tmp_path / "out"
    assert (
        cli_module.main(["--country", "xx", "--out", str(out), "--resume", resume]) == 1
    )
    error = capsys.readouterr().err
    assert "toy_factor" in error
    assert "reviewed re-declaration" in error
    for step in later:
        step.assert_not_called()
    assert not out.exists()


def _mock_steps_after_the_currency_check(monkeypatch):
    """Activation, the registry, the CREATE inventory and CREATE, each refusing."""
    steps = []
    for owner, name in (
        (cli_module, "validate_transport_activation"),
        (cli_module, "build_transport_registry"),
        (cli_module, "prepare_create_outputs"),
        (cli_module.TRANSPORT_CREATE, "run"),
    ):
        step = Mock(side_effect=AssertionError(f"Must refuse before {name}."))
        monkeypatch.setattr(owner, name, step)
        steps.append(step)
    return steps


@pytest.mark.parametrize("resume", ["auto", "forbid"])
@pytest.mark.parametrize("selection", ["default", "scenario"])
def test_preparation_refuses_expired_factors_for_library_callers(
    tmp_path, monkeypatch, resume, selection
):
    # prepare_transport_build is exported: a caller that skips main must meet
    # the same run-entry check, before activation, the registry or CREATE.
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    if selection == "scenario":
        spec["resources"]["reference_profiles"]["dimensions"]["index"]["profiles"][
            "central"
        ]["transformations"] = []
        spec["resources"]["scenarios"]["scenarios"].append(
            {"id": "variant", "knobs": {"index": "low"}}
        )
    monkeypatch.setattr(cli_module, "_today", lambda: _AFTER_FACTOR_EXPIRY)
    later = _mock_steps_after_the_currency_check(monkeypatch)
    with pytest.raises(ValueError, match="toy_factor") as error:
        cli_module.prepare_transport_build(spec, sources={}, resume=resume)
    assert "reviewed re-declaration" in str(error.value)
    for step in later:
        step.assert_not_called()


def test_preparation_lets_require_reach_stored_work_after_expiry(tmp_path, monkeypatch):
    # require skips the currency check and goes on to activation and the store.
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    monkeypatch.setattr(cli_module, "_today", lambda: _AFTER_FACTOR_EXPIRY)
    reached = Mock(side_effect=LookupError("activation reached"))
    monkeypatch.setattr(cli_module, "validate_transport_activation", reached)
    with pytest.raises(LookupError, match="activation reached"):
        cli_module.prepare_transport_build(spec, sources={}, resume="require")
    reached.assert_called_once_with(spec)


def test_preparation_checks_resume_before_currency(tmp_path, monkeypatch):
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    monkeypatch.setattr(cli_module, "_today", lambda: _AFTER_FACTOR_EXPIRY)
    with pytest.raises(ValueError, match="resume must be"):
        cli_module.prepare_transport_build(spec, sources={}, resume="sometimes")


def test_cli_factor_currency_decision_allows_historical_replay(tmp_path, monkeypatch):
    # The factor fixture has parked measurement rows and no runnable skeleton.
    # Exercise the CLI decision directly; the warm-store test below verifies
    # actual require execution with the engine-free composed skeleton.
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    assert compose_module.reference_profile_view(spec)[2]["selected_factors"]
    monkeypatch.setattr(cli_module, "_today", lambda: _AFTER_FACTOR_EXPIRY)
    cli_module._require_current_run_factors(spec, resume="require")


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
    monkeypatch.setattr(cli_module, "_today", lambda: _AFTER_FACTOR_EXPIRY)
    assert cli_module.main([*argv, "--resume", "require"]) == 0
    assert capsys.readouterr().err == ""
    assert calls == []


def test_cli_run_date_is_the_utc_date(monkeypatch):
    # 11:30pm in New York on 3 November is already 4 November in UTC, where CI runs.
    instant = datetime(2026, 11, 4, 3, 30, tzinfo=UTC)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return instant.astimezone(tz or timezone(timedelta(hours=-5)))

    monkeypatch.setattr(cli_module, "datetime", Clock)
    assert cli_module._today() == date(2026, 11, 4)


def test_cli_currency_check_walks_the_views_activation_walks(tmp_path, monkeypatch):
    # Activation resolves scenario views only when both resources exist and
    # validates their shapes; the run-entry check shares that walk.
    _, _, _, _, spec = factor_transport_fixture(tmp_path)
    del spec["resources"]["scenarios"]
    assert len(compose_module.reference_profile_views(spec)) == 1
    monkeypatch.setattr(cli_module, "_today", lambda: date(2050, 1, 1))
    cli_module._require_current_run_factors(spec, resume="auto")
    monkeypatch.setattr(cli_module, "_today", lambda: _AFTER_FACTOR_EXPIRY)
    with pytest.raises(ValueError, match="toy_factor"):
        cli_module._require_current_run_factors(spec, resume="auto")
    spec["resources"]["scenarios"] = {"scenarios": {"id": "malformed"}}
    with pytest.raises(TypeError) as activation:
        compose_module.validate_transport_activation(spec)
    with pytest.raises(TypeError) as run_entry:
        cli_module._require_current_run_factors(spec, resume="auto")
    assert str(run_entry.value) == str(activation.value) == "scenarios must be a list."


def test_cli_require_replays_a_warm_store_after_selected_factors_expire(
    composed, tmp_path, monkeypatch, capsys
):
    # End to end: warm a real store while a selected factor is current, then
    # replay it past valid_until. require verifies and runs nothing; a new run
    # of the same spec refuses on that factor before preparation work.
    facts, pin, _, _, factors = factor_transport_fixture(tmp_path / "factor")
    spec = copy.deepcopy(composed.spec)
    spec["resources"].update(
        factor_target_references=factors["resources"]["target_references"],
        factor_context=factors["resources"]["factor_context"],
        factor_feed=pin,
        reference_profiles={
            **factors["resources"]["reference_profiles"],
            "reference_resource": "factor_target_references",
        },
    )
    profile_view = compose_module.reference_profile_view

    def measured_view(spec, selection=None):
        # As in factor_transport_fixture(activate=True), supply a synthetic
        # reviewed measurement closure. The real resolver still selects and
        # binds the factors; composition must attach its rehashed receipt.
        # Copies, so the resolver's own return values are never edited.
        references, intervals, receipt = profile_view(spec, selection)
        references, receipt = copy.deepcopy(references), copy.deepcopy(receipt)
        for row in references["target_references"]:
            row["metadata"]["activation_status"] = "active"
        receipt["effective_references_sha256"] = reference_document_sha256(references)
        receipt.pop("receipt_sha256")
        receipt["receipt_sha256"] = reference_document_sha256(receipt)
        return references, intervals, receipt

    monkeypatch.setattr(compose_module, "reference_profile_view", measured_view)
    declaration = spec["resources"]["transport_graph"]
    compiler_id = "nz.factor.targets.compile"
    declaration["sources"].append(
        {
            **next(
                source
                for source in declaration["sources"]
                if source["name"] == "nz_calibration_facts"
            ),
            "name": "factor_facts",
        }
    )
    declaration["nodes"].append(
        {
            "id": compiler_id,
            "kernel": "targets.compile@1",
            "population": "nz.open",
            "sources": ["factor_facts"],
            "params": {
                "country": factors["country"],
                "references": {
                    "resource": "factor_target_references",
                    "encoding": "json",
                },
                "references_sha256": {
                    "resource": "factor_target_references",
                    "encoding": "sha256",
                },
                "chronicle_feed_pin": {"resource": "factor_feed", "encoding": "json"},
            },
        }
    )
    # This factor-bound compiler is an explicit CLI terminal endpoint, so its
    # target surface must actually be compiled and cached during the warm run.
    declaration["endpoints"]["terminal"].append(compiler_id)
    compiler = compose_transport_graph(spec, composed.config).node(compiler_id)
    assert compiler.kernel == "targets.compile@1"
    assert "reference_resolution" in compiler.params
    resolution = json.loads(compiler.params["reference_resolution"])
    assert resolution["selected_factors"]
    assert json.loads(compiler.params["chronicle_feed_pin"]) == pin
    directory = _write_local_transport_spec(tmp_path / "spec", spec)
    build_registry = cli_module.build_transport_registry
    calls = []
    replay = Mock(
        wraps=target_kernels_module.reference_factors_module.replay_reference_factor
    )
    monkeypatch.setattr(
        target_kernels_module.reference_factors_module,
        "replay_reference_factor",
        replay,
    )

    def injected_registry(bindings, root, *, unit_rule, engines_by_binding=None):
        registry = build_registry(
            bindings,
            root,
            unit_rule=unit_rule,
            engines_by_binding=composed.engines_by_binding,
        )
        for ref in registry.kernels.refs():
            kernel = registry.kernels.get(ref)
            run = kernel.run

            def counted(context, _run=run):
                calls.append(context.node.id)
                return _run(context)

            monkeypatch.setattr(kernel, "run", counted)
        return registry

    monkeypatch.setattr(cli_module, "build_transport_registry", injected_registry)
    out = tmp_path / "out"
    argv = ["--spec-dir", str(directory), "--out", str(out)]
    for name, path in {**composed.sources, "factor_facts": facts}.items():
        argv.extend(["--source", f"{name}={path}"])
    monkeypatch.setattr(cli_module, "_today", lambda: date(2050, 1, 1))
    assert cli_module.main(argv) == 0
    assert capsys.readouterr().err == ""
    assert calls.count(compiler_id) == 1
    replay.assert_called_once()
    factor = resolution["selected_factors"]["toy_factor"]
    # The replay ran against the authenticated factor feed: the facts an
    # independent load of the pinned factor_facts source yields.
    fed_facts = (
        target_kernels_module.ledger_artifact_module.load_ledger_consumer_artifact(
            facts, feed_pin=pin, lexical_floats=True
        ).facts
    )
    assert fed_facts
    replay.assert_called_once_with(
        factor["declaration"],
        fed_facts,
        factor["declaration"]["feed_identity"],
        factor["operational_factor"],
        factor["receipt"],
    )
    warm_node = json.loads((out / "manifest.json").read_bytes())["nodes"][compiler_id]
    assert warm_node["receipt"]["n_targets"] == 2
    exported = (out / EXPORT_FILENAME).read_bytes()
    calls.clear()
    replay.reset_mock()
    monkeypatch.setattr(cli_module, "_today", lambda: _AFTER_FACTOR_EXPIRY)
    assert cli_module.main([*argv, "--resume", "require"]) == 0
    assert capsys.readouterr().err == ""
    # require reads back the stored run's export and rewrites nothing.
    assert (out / EXPORT_FILENAME).read_bytes() == exported
    assert calls == []
    replay.assert_not_called()
    stored_node = json.loads((out / "manifest.json").read_bytes())["nodes"][compiler_id]
    assert stored_node["hit"] is True
    assert stored_node["key"] == warm_node["key"]
    assert stored_node["opaque_artifacts"] == warm_node["opaque_artifacts"]
    later = _mock_steps_after_the_currency_check(monkeypatch)
    assert cli_module.main([*argv, "--resume", "auto"]) == 1
    error = capsys.readouterr().err
    assert "toy_factor" in error
    assert "reviewed re-declaration" in error
    assert calls == []
    for step in later:
        step.assert_not_called()


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


@pytest.mark.parametrize("target", ["nz.export.prepare", "absent.node"])
def test_extension_gate_evidence_cannot_target_other_skeleton_nodes(composed, target):
    extension = compose_module.TransportExtension(
        toy_extension,
        gate_inputs=(
            (
                target,
                ArtifactInput(
                    "cell_diagnostics",
                    "nz.test.validate.compare",
                    "comparison",
                    COMPARISON_TYPE,
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="only a skeleton battery"):
        compose_transport_graph(
            composed.spec, replace(composed.config, extensions=(extension,))
        )


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


_PRECALIBRATION_PROPERTY_DOMAINS = {
    "mixed": {
        "kinds": ("encode", "unit", "person", "selector", "holdout", "filter", "plain"),
        "holdout_sources": True,
    },
    # Without hold-out data most graphs pass the other checks, so the
    # producer attribution of rule 3 is exercised in every run.
    "rules_only": {
        "kinds": ("person", "selector", "filter", "plain"),
        "holdout_sources": False,
    },
    # Population versions with shared names: a column a stage leaves
    # undeclared is often owned again in another version, read there or
    # rewritten, so version attribution decides most outcomes.
    "versions": {
        "kinds": ("person", "filter", "plain"),
        "holdout_sources": False,
        "eager": True,
    },
    # Writer chains: masked owners keep the values below them, and an EXPAND
    # overlays a cell that a claim owns, so a read often has several writers.
    "writers": {
        "kinds": ("person", "filter", "plain", "expand"),
        "holdout_sources": False,
        "eager": True,
        "masks": True,
        "focused": True,
    },
}

_PRECALIBRATION_STRUCTURAL_KINDS = {"filter", "expand"}


def _precalibration_edges(draw, nodes, version, writers, focused=False):
    """Dependencies, reads and artifact edges from earlier nodes.

    A read is chosen from the node its column resolves to first in
    ``version``, and names every writer whose values it receives. When
    ``focused``, only nodes outside every binding pass artifacts.
    """
    dependencies, reads, artifacts = [], [], []
    for predecessor, earlier in enumerate(nodes):
        if earlier["kind"] in _PRECALIBRATION_STRUCTURAL_KINDS:
            continue
        edge = draw(st.integers(min_value=0, max_value=3))
        if edge == 1 and not (focused and earlier["binding"] is not None):
            dependencies.append(predecessor)
            artifacts.append(predecessor)
        if edge > 1:
            columns = [
                column
                for column in earlier["outputs"]
                if writers(version, column)[:1] == [predecessor]
            ]
            if columns and edge == 2:
                columns = [draw(st.sampled_from(columns))]
            if columns:
                dependencies.append(predecessor)
            reads.extend(
                (writer, column)
                for column in columns
                for writer in writers(version, column)
            )
    return dependencies, reads, artifacts


@st.composite
def _precalibration_ancestry_dags(
    draw, kinds, holdout_sources, eager=False, masks=False, focused=False
):
    """Small DAGs with independent binding declarations and column edges.

    A new binding may also declare another binding's variable name, which only
    that other binding's node produces; selector nodes add a derived column
    that is not a binding variable; any binding node may carry hold-out data.
    A filter node opens a population version on an earlier one and may read
    columns from its base. A binding or plain node may own a ``shared`` column
    that a node in another version also owns, and a person node in a filter
    version may rewrite a person column it sees from below or, with
    ``masks``, own it again on masked rows. An EXPAND opens a version too and
    overlays a cell that the next node claims: a new cell, the derived output
    of the claim's binding when it has one, or else possibly a person column
    the EXPAND sees. A structural node is now and then a person binding's
    node, which makes it a stage node when that binding is listed. Each read,
    rewrite and claim names every writer whose values it receives, as the
    executor tracks them: the owner in the reader's version (a filter's base,
    or below a rewrite's or claim's own version), and the writers below it
    while it keeps earlier values, with an EXPAND's cell.
    With ``focused``, a listed stage withholds at most one of its columns and
    stage nodes pass no artifacts, so a refusal usually has one cause.
    """
    kinds = draw(
        st.lists(st.sampled_from(kinds), min_size=1, max_size=9 if eager else 7)
    )
    likely = st.sampled_from((True, True, True, False)) if eager else st.booleans()
    nodes = []
    bindings = {}
    # Version None is CREATE; a structural node's index names its version. A
    # version a structural node branches from takes no new members, so
    # structural edges point back.
    bases = {None: None}
    closed = set()
    owned = {}
    entities = {}
    # Owners that keep earlier values: rewrites, masked owners and claims.
    keeps = set()
    # The cell each EXPAND writes, by its version.
    expanded = {}
    # Versions a read resolved through: a later owner there would re-route it.
    frozen = set()

    def writers(version, column):
        found = []
        while True:
            if (version, column) in owned:
                found.append(owned[version, column])
                if (version, column) not in keeps:
                    return found
            if column in expanded.get(version, ()):
                found.append(version)
            if version is None:
                return found
            version = bases[version]

    def freeze(version, column):
        while (version, column) not in owned:
            frozen.add((version, column))
            version = bases[version]

    def member_version():
        return draw(st.sampled_from(sorted(set(bases) - closed, key=str)))

    def seen(version):
        return sorted({column for _, column in owned if writers(version, column)})

    def structural_node(kind, base):
        # A structural node reads its inputs from its base; a focused one only
        # columns that no binding's node writes.
        columns = [
            column
            for column in seen(base)
            if not focused
            or all(nodes[writer]["binding"] is None for writer in writers(base, column))
        ]
        chosen = (
            draw(
                st.lists(
                    st.sampled_from(columns),
                    unique=True,
                    min_size=1 if eager else 0,
                    max_size=2,
                )
            )
            if columns
            else []
        )
        for column in chosen:
            freeze(base, column)
        return {
            "kind": kind,
            "binding": None,
            "entity": "person",
            "version": None,
            "base": base,
            "outputs": [],
            "cells": [],
            "reads": [
                (writer, column)
                for column in chosen
                for writer in writers(base, column)
            ],
            "rewrites": [],
            "claims": [],
            "masks": [],
            # A version depends on its base and every member of it.
            "dependencies": [
                *([] if base is None else [base]),
                *(
                    member
                    for member, row in enumerate(nodes)
                    if row["kind"] not in _PRECALIBRATION_STRUCTURAL_KINDS
                    and row["version"] == base
                ),
            ],
            "artifacts": [],
            "holdout_selection": False,
            "holdout_source": False,
        }

    for kind in kinds:
        index = len(nodes)
        if kind in _PRECALIBRATION_STRUCTURAL_KINDS:
            base = draw(st.sampled_from(sorted(bases, key=str)))
            node = structural_node(kind, base)
            bases[index] = base
            closed.add(base)
            nodes.append(node)
            available = [
                name for name, row in bindings.items() if row["entity"] == "person"
            ]
            if available and draw(st.integers(min_value=0, max_value=7)) == 0:
                node["binding"] = draw(st.sampled_from(available))
            if kind == "filter":
                continue
            # The claim follows the EXPAND, often as a person binding's node.
            claim, binding = index + 1, None
            if draw(st.booleans()):
                if available and draw(st.booleans()):
                    binding = draw(st.sampled_from(available))
                else:
                    binding = f"binding_{claim}"
                    bindings[binding] = {
                        "entity": "person",
                        "variables": [],
                        "derived": [],
                    }
            overlaid = [column for column in seen(base) if entities[column] == "person"]
            if binding is None and overlaid and draw(likely):
                cell = draw(st.sampled_from(overlaid))
            else:
                cell = f"cell_{index}"
                entities[cell] = "person"
                if binding is not None:
                    bindings[binding]["derived"].append(cell)
            node["cells"] = [cell]
            expanded[index] = {cell}
            owned[index, cell] = claim
            keeps.add((index, cell))
            nodes.append(
                {
                    "kind": "claim",
                    "binding": binding,
                    "entity": "person",
                    "version": index,
                    "outputs": [cell],
                    "cells": [],
                    "reads": [],
                    "rewrites": [],
                    "claims": [
                        (writer, cell)
                        for writer in writers(index, cell)
                        if writer != claim
                    ],
                    "masks": [],
                    "dependencies": [index],
                    "artifacts": [],
                    "holdout_selection": False,
                    "holdout_source": False,
                }
            )
            continue
        version = member_version()
        binding = None
        entity = "family" if kind in {"encode", "unit"} else "person"
        if kind in {"unit", "person", "selector"}:
            if kind == "selector":
                entity = draw(st.sampled_from(("person", "family")))
            available = [
                name for name, row in bindings.items() if row["entity"] == entity
            ]
            if kind == "selector" and available and draw(st.booleans()):
                binding = draw(st.sampled_from(available))
            else:
                binding = f"binding_{index}"
                shared = [
                    variable
                    for row in bindings.values()
                    if row["entity"] == entity
                    for variable in row["variables"]
                ]
                bindings[binding] = {"entity": entity, "variables": [], "derived": []}
                if shared and draw(st.booleans()):
                    bindings[binding]["variables"].append(draw(st.sampled_from(shared)))
        dependencies, reads, artifacts = _precalibration_edges(
            draw, nodes, version, writers, focused
        )
        for _, column in reads:
            freeze(version, column)
        outputs = [f"value_{index}"]
        if binding is not None:
            outputs.append(f"private_{index}")
        # A node may not own a column it reads, one its version owns, or one an
        # earlier read in its version resolved below it.
        taken = owned.keys() | frozen
        common = f"shared_{entity}"
        if (
            (binding is not None or kind == "plain")
            and (version, common) not in taken
            and draw(likely)
        ):
            outputs.append(common)
        rewrites, masked = [], []
        if version is not None and entity == "person":
            # The incumbent comes from below the version; CREATE has none. A
            # rewritten or masked column joins the node's binding variables,
            # which a derived output or an EXPAND cell may never be.
            columns = [
                column
                for column in seen(bases[version])
                if entities[column] == "person"
                and not column.startswith(("derived_", "cell_"))
                and (version, column) not in taken
                and column not in outputs
                and column not in {read for _, read in reads}
            ]
            if columns and draw(likely):
                for column in draw(
                    st.lists(
                        st.sampled_from(columns),
                        unique=True,
                        min_size=1,
                        max_size=2 if eager else 1,
                    )
                ):
                    # A masked owner leaves the incumbent's values outside its
                    # mask, and a rewrite receives them, so both keep them.
                    if masks and draw(likely):
                        masked.append(column)
                    else:
                        rewrites.extend(
                            (writer, column)
                            for writer in writers(bases[version], column)
                        )
                    keeps.add((version, column))
                    outputs.append(column)
        if binding is not None:
            bindings[binding]["variables"].extend(
                column
                for column in outputs
                if column not in bindings[binding]["variables"]
            )
        if kind == "selector":
            outputs.append(f"derived_{index}")
            bindings[binding]["derived"].append(f"derived_{index}")
        for column in outputs:
            owned[version, column] = index
            entities.setdefault(column, entity)
        nodes.append(
            {
                "kind": kind,
                "binding": binding,
                "entity": entity,
                "version": version,
                "outputs": outputs,
                "cells": [],
                "reads": reads,
                "rewrites": rewrites,
                "claims": [],
                "masks": masked,
                "dependencies": [
                    *([] if version is None else [version]),
                    *dependencies,
                ],
                "artifacts": artifacts,
                "holdout_selection": kind == "holdout" and draw(st.booleans()),
                "holdout_source": holdout_sources
                and binding is not None
                and draw(st.sampled_from((False, False, False, True))),
            }
        )
    version = member_version()
    dependencies, reads, artifacts = _precalibration_edges(
        draw, nodes, version, writers, focused
    )
    nodes.append(
        {
            "kind": "calibrate",
            "binding": None,
            "entity": "person",
            "version": version,
            "outputs": [],
            "cells": [],
            "reads": reads,
            "rewrites": [],
            "claims": [],
            "masks": [],
            "dependencies": [*([] if version is None else [version]), *dependencies],
            "artifacts": artifacts,
            "holdout_selection": False,
            "holdout_source": False,
        }
    )
    relevant = _precalibration_reference_relevant(nodes)
    # Columns whose values a later writer passes on: under a mask, or as a cell.
    passed = {column for node in nodes for column in node["masks"] + node["cells"]}
    declared = {}
    for binding, row in bindings.items():
        if not draw(likely):
            continue
        # Derived columns alone are a complete declaration.
        columns = row["variables"] + row["derived"]
        if not focused:
            declared[binding] = draw(st.sets(st.sampled_from(columns), min_size=1))
            continue
        # A focused stage declares no derived output that its nodes upstream do
        # not produce, and withholds at most one column, often one passed on.
        produced = {
            column
            for index in relevant
            if nodes[index]["binding"] == binding
            for column in nodes[index]["outputs"]
        }
        columns = [
            column
            for column in columns
            if column in row["variables"] or column in produced
        ]
        if not columns:
            continue
        declared[binding] = set(columns)
        if len(columns) > 1 and draw(likely):
            candidates = [column for column in columns if column in passed]
            if not candidates or not draw(likely):
                candidates = columns
            declared[binding].remove(draw(st.sampled_from(candidates)))
    return nodes, bindings, declared


def _precalibration_reference_relevant(nodes):
    """Calibration and its ancestors: the nodes the stage rules govern."""
    relevant = {len(nodes) - 1}
    for index in range(len(nodes) - 1, -1, -1):
        if index in relevant:
            relevant.update(nodes[index]["dependencies"])
    return relevant


def _precalibration_reference_missing_derived(model, binding):
    """Declared derived columns that no node of the stage upstream produces."""
    nodes, bindings, declared = model
    relevant = _precalibration_reference_relevant(nodes)
    # A node produces what it owns.
    produced = {
        column
        for index in relevant
        if nodes[index]["binding"] == binding
        for column in nodes[index]["outputs"]
    }
    return declared[binding] & set(bindings[binding]["derived"]) - produced


def _precalibration_reference_accepts(model):
    """The stage rules on generated kinds, column reads and artifact edges, upstream of calibration."""
    nodes, _, declared = model
    relevant = _precalibration_reference_relevant(nodes)
    if any(_precalibration_reference_missing_derived(model, name) for name in declared):
        return False
    for index in relevant:
        node = nodes[index]
        binding = node["binding"]
        if node["kind"] == "holdout" or node["holdout_source"]:
            return False
        # A stage changes columns, not the population.
        if node["kind"] in _PRECALIBRATION_STRUCTURAL_KINDS and binding in declared:
            return False
        if (
            binding is not None
            and node["entity"] == "family"
            and binding not in declared
        ):
            return False
        if node["kind"] == "encode":
            consumers = [
                nodes[consumer]
                for consumer in relevant
                if any(producer == index for producer, _ in nodes[consumer]["reads"])
            ]
            if not consumers or any(
                consumer["binding"] not in declared or consumer["entity"] != "family"
                for consumer in consumers
            ):
                return False
        # A rewrite receives its incumbent and a claim its cell's writers, so
        # both are reads too.
        for producer, column in node["reads"] + node["rewrites"] + node["claims"]:
            source_binding = nodes[producer]["binding"]
            if (
                source_binding in declared
                and source_binding != binding
                and column not in declared[source_binding]
            ):
                return False
        # A stage's artifacts are never declared outputs.
        for producer in node["artifacts"]:
            source_binding = nodes[producer]["binding"]
            if source_binding in declared and source_binding != binding:
                return False
    return True


def _precalibration_synthetic_graph(model):
    """Render the same model as declarations and compiled extension nodes."""
    nodes, bindings, declared = model
    references = {name: f"synthetic-engine:{name}:end" for name in bindings}
    seed = Owned("person", "seed", "float64")
    # CREATE also loads the boolean mask that masked owners read.
    created = (
        (seed, Owned("person", "keep", "bool"))
        if any(item["masks"] for item in nodes)
        else (seed,)
    )
    trace = ArtifactType("synthetic.precalibration.trace", 1)
    graph_nodes = [
        Node(
            "p1.create",
            "synthetic.create@1",
            structural=StructuralDelta.CREATE,
            outputs=created,
            sources=("synthetic_donor",),
        )
    ]
    rows = [
        {
            "id": "p1.create",
            "kernel": "synthetic.create@1",
            "outputs": [
                {"entity": output.entity, "column": output.column} for output in created
            ],
        }
    ]

    def version_id(version):
        return "p1.create" if version is None else f"p1.node_{version}"

    for index, item in enumerate(nodes):
        name = "p1.calibrate" if item["kind"] == "calibrate" else f"p1.node_{index}"
        inputs = {}
        # A read names each of its writers; the node reads the column once.
        for producer, column in item["reads"]:
            columns = inputs.setdefault(nodes[producer]["entity"], [])
            if column not in columns:
                columns.append(column)
        if item["masks"]:
            inputs.setdefault("person", []).append("keep")
        if item["kind"] in _PRECALIBRATION_STRUCTURAL_KINDS:
            base = version_id(item["base"])
            kernel = f"synthetic.{item['kind']}@1"
            params = {}
            prepared_params = {}
            if item["binding"] is not None:
                params["engine_ref"] = {
                    "binding": item["binding"],
                    "field": "engine_ref",
                }
                prepared_params["engine_ref"] = references[item["binding"]]
            if item["cells"]:
                cells = tuple(("person", cell, "float64") for cell in item["cells"])
                params["expand_cells"] = [list(cell) for cell in cells]
                prepared_params["expand_cells"] = cells
            graph_nodes.append(
                Node(
                    name,
                    kernel,
                    structural=StructuralDelta(item["kind"]),
                    base=base,
                    inputs=tuple(
                        Slice(entity, tuple(columns))
                        for entity, columns in inputs.items()
                    ),
                    params=prepared_params,
                )
            )
            rows.append(
                {
                    "id": name,
                    "kernel": kernel,
                    "structural": item["kind"],
                    "base": base,
                    "inputs": [
                        {"entity": entity, "columns": columns}
                        for entity, columns in inputs.items()
                    ],
                    **({"params": params} if params else {}),
                }
            )
            continue
        kernel = {
            "encode": "concepts.encode_groups@1",
            "unit": RULES,
            "person": RULES,
            "selector": "simulate.solve_zero@1",
            "holdout": "synthetic.holdout@1",
            "plain": "synthetic.plain@1",
            "claim": "synthetic.claim@1",
            "calibrate": "calibrate.synthetic@1",
        }[item["kind"]]
        rewritten = {column for _, column in item["rewrites"]}
        binding = item["binding"]
        params = {}
        prepared_params = {}
        if binding is not None:
            prepared_params["engine_ref"] = references[binding]
            if item["kind"] == "selector":
                params["engine_ref"] = {"binding": binding, "field": "engine_ref"}
        if item["claims"]:
            claimed = list(
                dict.fromkeys(f"person.{cell}" for _, cell in item["claims"])
            )
            params["materialized_expand_outputs"] = claimed
            prepared_params["materialized_expand_outputs"] = tuple(claimed)
        sources = ("synthetic_holdout_facts",) if item["holdout_source"] else ()
        if item["kind"] == "holdout":
            if item["holdout_selection"]:
                params["selection"] = {"resource": "holdout_references"}
                # Resource selectors disappear during parameter preparation.
                prepared_params["selection"] = "{}"
            else:
                sources = ("synthetic_holdout_facts",)
        # A column read is its own edge; an artifact carries the rest.
        edges = tuple(
            ArtifactInput(f"edge_{producer}", f"p1.node_{producer}", "trace", trace)
            for producer in item["artifacts"]
        )
        graph_nodes.append(
            Node(
                name,
                kernel,
                population=version_id(item["version"]),
                inputs=tuple(
                    Slice(entity, tuple(columns)) for entity, columns in inputs.items()
                ),
                outputs=tuple(
                    Owned(
                        item["entity"],
                        column,
                        "float64",
                        rewrite=column in rewritten,
                        **({"rows": "keep"} if column in item["masks"] else {}),
                    )
                    for column in item["outputs"]
                ),
                params=prepared_params,
                sources=sources,
                artifact_inputs=edges,
                artifact_outputs=(ArtifactOutput("trace", trace),),
            )
        )
        row = {
            "id": name,
            "kernel": kernel,
            "population": version_id(item["version"]),
            "inputs": [
                {"entity": entity, "columns": columns}
                for entity, columns in inputs.items()
            ],
            "outputs": [
                {
                    "entity": item["entity"],
                    "column": column,
                    **({"rewrite": True} if column in rewritten else {}),
                    **({"rows": "keep"} if column in item["masks"] else {}),
                }
                for column in item["outputs"]
            ],
            "params": params,
            "sources": list(sources),
            "artifact_inputs": [
                {"producer": f"p1.node_{producer}"} for producer in item["artifacts"]
            ],
        }
        if binding is not None and item["kind"] != "selector":
            row["rules_binding"] = binding
        rows.append(row)
    spec = {
        "country": "synthetic",
        "resources": {
            "benefit_unit_rule": {"entity": "family"},
            "axiom_rules_bindings": {
                "bindings": [
                    {
                        "id": name,
                        "rulespec_path": f"synthetic/{name}.yaml",
                        "entity": row["entity"],
                        "variables": row["variables"],
                    }
                    for name, row in bindings.items()
                ]
            },
            "holdout_references": {"target_references": []},
            "precalibration_stages": {
                "schema_version": 1,
                "stages": [
                    {
                        "id": f"stage.{name}",
                        "rules_binding": name,
                        "rulespec_path": f"synthetic/{name}.yaml",
                        "outputs": [
                            {
                                "entity": bindings[name]["entity"],
                                "column": column,
                                "purpose": "calibration_measure",
                            }
                            for column in bindings[name]["variables"]
                            if column in columns
                        ],
                        "citation": "Synthetic stage contract.",
                        **(
                            {
                                "derived_outputs": [
                                    {
                                        "entity": bindings[name]["entity"],
                                        "column": column,
                                        "purpose": "calibration_measure",
                                    }
                                    for column in bindings[name]["derived"]
                                    if column in columns
                                ]
                            }
                            if columns & set(bindings[name]["derived"])
                            else {}
                        ),
                    }
                    for name, columns in declared.items()
                ],
            },
            "transport_graph": {
                "schema_version": 1,
                "country": "synthetic",
                "nodes": rows,
                "endpoints": {"calibration": "p1.calibrate"},
            },
        },
    }
    graph = Graph(
        "synthetic",
        (
            SourceRef("synthetic_donor", "synthetic@1"),
            SourceRef("synthetic_holdout_facts", "synthetic@1"),
        ),
        tuple(graph_nodes),
    )
    config = TransportGraphConfig(engine_refs=references, create_outputs=created)
    return spec, graph, config


def _precalibration_property_outcomes(model):
    """Acceptance by the guard and by today's guard, in both passes."""
    spec, graph, config = _precalibration_synthetic_graph(model)
    compiled = compile_graph(graph)
    # The composed pass gives each read the executor's writers.
    assert _guard_writers(graph, compiled) == _executor_writers(graph, compiled)
    extension_spec = copy.deepcopy(spec)
    extension_spec["resources"]["transport_graph"]["nodes"] = [
        spec["resources"]["transport_graph"]["nodes"][0],
        {
            "id": "p1.calibrate",
            "kernel": "calibrate.synthetic@1",
            "population": "p1.create",
        },
    ]
    # These compiled nodes stand for a code extension's, which have no
    # declaration, and a selected hold-out resource leaves no marker in
    # compiled params. Keep those rows so this path still checks their
    # provenance. (Entitlement rows are declarations: see the example tests.)
    extension_spec["resources"]["transport_graph"]["nodes"].extend(
        row
        for row, node in zip(
            spec["resources"]["transport_graph"]["nodes"][1:], model[0], strict=True
        )
        if node["holdout_selection"]
    )
    if not model[2]:
        extension_spec = spec
    outcomes = {}
    for label, document, compiled in (
        ("declaration", spec, None),
        ("compiled extensions", extension_spec, graph),
    ):
        for guard_name, guard in (
            ("guard", compose_module.validate_transport_calibration_ancestry),
            ("today", _precalibration_reference_guard),
        ):
            try:
                guard(document, graph=compiled, config=config)
            except ValueError:
                outcomes[label, guard_name] = False
            else:
                outcomes[label, guard_name] = True
    return outcomes


@pytest.mark.parametrize("domain", sorted(_PRECALIBRATION_PROPERTY_DOMAINS))
@settings(max_examples=2000, deadline=None, database=None)
@given(data=st.data())
def test_precalibration_stages_calibration_ancestry_property(domain, data):
    model = data.draw(
        _precalibration_ancestry_dags(**_PRECALIBRATION_PROPERTY_DOMAINS[domain])
    )
    nodes, bindings, declared = model
    outcomes = _precalibration_property_outcomes(model)
    labels = ("declaration", "compiled extensions")
    for label in labels:
        actual = outcomes[label, "guard"]
        if not declared:
            # I1 with no stages: exactly today's guard. Its declaration pass
            # follows artifact and structural edges only, not column reads.
            assert actual == outcomes[label, "today"], (label, model)
            continue
        assert actual == _precalibration_reference_accepts(model), (label, model)
        if all(bindings[name]["entity"] == "person" for name in declared):
            # I1: person stages only restrict today's guard.
            assert outcomes[label, "today"] or not actual, (label, model)
    if not declared:
        return
    dropped = data.draw(st.sampled_from(sorted(declared)))
    smaller = (
        nodes,
        bindings,
        {name: columns for name, columns in declared.items() if name != dropped},
    )
    reduced = _precalibration_property_outcomes(smaller)
    for label in labels:
        assert reduced[label, "guard"] == (
            _precalibration_reference_accepts(smaller)
            if smaller[2]
            else reduced[label, "today"]
        ), (label, smaller)
        if bindings[dropped]["entity"] == "person":
            # I2: dropping a person stage never refuses what listing it admitted.
            assert reduced[label, "guard"] or not outcomes[label, "guard"], (
                label,
                model,
                dropped,
            )
        elif smaller[2] and not _precalibration_reference_missing_derived(
            model, dropped
        ):
            # I2: dropping a unit stage never admits what listing it refused,
            # unless the refusal is that stage's own declaration error. Dropping
            # the last stage restores today's guard exactly (checked above),
            # and today's guard does not follow column-only paths to a
            # resource selection or, in its declaration pass, at all.
            assert outcomes[label, "guard"] or not reduced[label, "guard"], (
                label,
                model,
                dropped,
            )


@pytest.mark.parametrize("shared_module", [False, True])
def test_precalibration_ancestry_stage_own_selected_reference_is_allowed(
    calibration_ancestry_composed, shared_module
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    reference = composed.config.engine_refs["family_housing"]
    config = composed.config
    if shared_module:
        bindings = spec["resources"]["axiom_rules_bindings"]["bindings"]
        other = copy.deepcopy(
            next(item for item in bindings if item["id"] == "family_housing")
        )
        other.update(id="other_family", variables=["other_result"])
        bindings.append(other)
        other_reference = json.loads(reference)
        other_reference["binding"].update(id="other_family", variables=["other_result"])
        config = replace(
            config,
            engine_refs={
                **config.engine_refs,
                "other_family": canonical_json(other_reference).decode(),
            },
        )
    graph = compose_transport_graph(spec, config)
    spec["resources"]["stage_engine_data"] = {"nested": [reference]}
    row = next(
        item
        for item in spec["resources"]["transport_graph"]["nodes"]
        if item["id"] == "nz.precal.family"
    )
    row["params"] = {"context": {"resource": "stage_engine_data", "encoding": "json"}}
    compose_module.validate_transport_calibration_ancestry(spec, config=config)
    compose_module.validate_transport_calibration_ancestry(
        spec, graph=graph, config=config
    )


@pytest.mark.parametrize(
    "literal",
    [
        {"resource": "holdout_references", "encoding": "sha256"},
        {"prepared": "engine_refs"},
        {"binding": "family_housing", "field": "engine_ref"},
    ],
)
def test_precalibration_ancestry_person_stage_preserves_selected_literal_objects(
    calibration_ancestry_composed, literal
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed, "main_benefit")
    spec["resources"]["calibration"]["literal"] = {"items": [[literal]]}
    row = next(
        item
        for item in spec["resources"]["transport_graph"]["nodes"]
        if item["id"] == "nz.targets.problem"
    )
    row["params"]["literal"] = {
        "resource": "calibration",
        "path": ["literal"],
        "encoding": "json",
    }
    graph = compose_transport_graph(spec, composed.config)
    assert json.loads(graph.node(row["id"]).params["literal"]) == {"items": [[literal]]}


@pytest.mark.parametrize("first", ["nz.targets.problem", "nz.as.inputs"])
def test_precalibration_ancestry_refusal_follows_declaration_order(
    calibration_ancestry_composed, first
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    document = spec["resources"]["transport_graph"]
    for row in document["nodes"]:
        if row["id"] in {"nz.targets.problem", "nz.as.inputs"}:
            row["sources"] = ["nz_holdout_facts"]
    document["nodes"].sort(key=lambda row: row["id"] != first)
    outcomes = [
        _precalibration_outcome(
            compose_module.validate_transport_calibration_ancestry,
            spec,
            config=composed.config,
        )
        for _ in range(3)
    ]
    assert outcomes == [
        (
            ValueError,
            f"Transport calibration ancestry: node {first!r} carries hold-out source.",
        )
    ] * len(outcomes)


def test_precalibration_ancestry_schema_checked_before_graph(
    calibration_ancestry_composed,
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    del spec["resources"]["transport_graph"]
    spec["resources"]["precalibration_stages"]["stages"][0]["outputs"][0]["purpose"] = (
        "evaluation"
    )
    with pytest.raises(ValueError, match="precalibration_stages"):
        compose_module.validate_transport_calibration_ancestry(spec)


@pytest.mark.parametrize("compiled", [False, True])
def test_precalibration_ancestry_postcalibration_stage_nodes_are_not_checked(
    calibration_ancestry_composed, compiled
):
    """A stage node after calibration is an output; its hold-out sibling is allowed."""
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    for row in spec["resources"]["transport_graph"]["nodes"]:
        if row["id"] in {"nz.as.inputs", "nz.precal.family"}:
            row["population"] = "nz.as"
        if row["id"] == CALIBRATION:
            row["inputs"] = [
                item for item in row["inputs"] if item["entity"] != "family"
            ]
    graph = compose_transport_graph(spec, composed.config)
    if compiled:
        graph = replace(
            graph,
            nodes=tuple(
                replace(node, sources=("nz_holdout_facts",))
                if node.id == "nz.as.inputs"
                else node
                for node in graph.nodes
            ),
        )
    else:
        row = next(
            item
            for item in spec["resources"]["transport_graph"]["nodes"]
            if item["id"] == "nz.as.inputs"
        )
        row["sources"] = ["nz_holdout_facts"]
    compose_module.validate_transport_calibration_ancestry(
        spec, graph=graph if compiled else None, config=composed.config
    )


def test_precalibration_ancestry_stage_binding_after_calibration_is_not_checked(
    calibration_ancestry_composed,
):
    """A binding listed before calibration may also run after it, beside hold-out data."""
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    nodes = spec["resources"]["transport_graph"]["nodes"]
    holdout = {
        "id": "nz.postcal.holdout",
        "kernel": "transport.probe@1",
        "population": "nz.as",
        "sources": ["nz_holdout_facts"],
        "params": {"holdout": {"resource": "holdout_references", "encoding": "json"}},
    }
    nodes.extend(
        [
            holdout,
            {
                "id": "nz.postcal.family",
                "kernel": RULES,
                "population": "nz.as",
                "rules_binding": "family_housing",
                "artifact_inputs": [{"producer": holdout["id"]}],
            },
        ]
    )
    compose_module.validate_transport_calibration_ancestry(spec, config=composed.config)
    # The same node before calibration refuses: the check is not vacuous.
    holdout["population"] = "nz.open"
    with pytest.raises(ValueError, match="node 'nz.postcal.holdout' carries"):
        compose_module.validate_transport_calibration_ancestry(
            spec, config=composed.config
        )


def _stage_artifact_case(entity, reader):
    """A stage node writes an artifact; calibration or another stage node reads it."""
    binding = _synthetic_binding("u", entity, ["u_out"])
    trace = ArtifactType("synthetic.trace", 1)
    selector = {"engine_ref": {"binding": "u", "field": "engine_ref"}}
    # Calibration reads the artifact itself, or a second stage node turns it
    # into the declared column that calibration reads.
    writer = "probe" if reader == "calibration" else "own"
    rows = [
        {
            "id": "probe",
            "kernel": "synthetic.probe@1",
            "population": "create",
            "params": selector,
            **(
                {"outputs": [{"entity": entity, "column": "u_out"}]}
                if writer == "probe"
                else {}
            ),
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "create",
            "inputs": [{"entity": entity, "columns": ["u_out"]}],
        },
    ]
    edge = ArtifactInput("trace", "probe", "trace", trace)
    nodes = [
        Node(
            "probe",
            "synthetic.probe@1",
            population="create",
            params={"engine_ref": "u-ref"},
            outputs=(Owned(entity, "u_out", "float64"),) if writer == "probe" else (),
            artifact_outputs=(ArtifactOutput("trace", trace),),
        )
    ]
    if writer == "probe":
        rows[1]["artifact_inputs"] = [{"name": "trace", "producer": "probe"}]
    else:
        rows.insert(
            1,
            {
                "id": "own",
                "kernel": "synthetic.probe@1",
                "population": "create",
                "params": selector,
                "outputs": [{"entity": entity, "column": "u_out"}],
                "artifact_inputs": [{"name": "trace", "producer": "probe"}],
            },
        )
        nodes.append(
            Node(
                "own",
                "synthetic.probe@1",
                population="create",
                params={"engine_ref": "u-ref"},
                outputs=(Owned(entity, "u_out", "float64"),),
                artifact_inputs=(edge,),
            )
        )
    nodes.append(
        Node(
            "cal",
            "calibrate.synthetic@1",
            population="create",
            inputs=(Slice(entity, ("u_out",)),),
            artifact_inputs=(edge,) if writer == "probe" else (),
        )
    )
    stage = _synthetic_stage(binding, ["u_out"])
    return _synthetic_stage_spec([binding], [stage], rows), nodes


@pytest.mark.parametrize("reader", ["calibration", "own_stage"])
@pytest.mark.parametrize("entity", ["family", "person"])
def test_precalibration_ancestry_stage_artifact_is_not_a_declared_output(
    entity, reader
):
    """Outputs are columns, so only a stage's own nodes may read its artifacts."""
    spec, nodes = _stage_artifact_case(entity, reader)
    outcomes = _synthetic_outcomes(spec, nodes, {"u": "u-ref"})
    if reader == "own_stage":
        assert outcomes == {"declaration": None, "composed": None}
        return
    message = (
        "Transport calibration ancestry: node 'cal' carries undeclared "
        "pre-calibration stage output."
    )
    assert outcomes == {
        "declaration": (ValueError, message),
        "composed": (ValueError, message),
    }


@pytest.mark.parametrize("entity", ["family", "person"])
def test_precalibration_ancestry_composed_stage_artifact_read_refuses(entity):
    """Only the composed graph reads the stage artifact, as an extension node can."""
    spec, nodes = _stage_artifact_case(entity, "calibration")
    rows = spec["resources"]["transport_graph"]["nodes"]
    del next(row for row in rows if row["id"] == "cal")["artifact_inputs"]
    assert _synthetic_outcomes(spec, nodes, {"u": "u-ref"}) == {
        "declaration": None,
        "composed": (
            ValueError,
            "Transport calibration ancestry: node 'cal' carries undeclared "
            "pre-calibration stage output.",
        ),
    }


def _variant_calibration_case(post, feeds_variant):
    """A node after the central fit, and a variant calibration that may read it."""
    tax = _synthetic_binding("tax", "person", ["tax_out"])
    family = _synthetic_binding("fam", "family", ["fam_out"])
    fit = ArtifactType("synthetic.fit", 1)
    holdout = post == "holdout"
    entity, column = ("person", "post_value") if holdout else ("family", "fam_out")
    post_row = {
        "id": "post",
        "kernel": "synthetic.holdout@1" if holdout else RULES,
        "population": "create",
        "outputs": [{"entity": entity, "column": column}],
        "artifact_inputs": [{"producer": "cal"}],
        **(
            {"sources": ["synthetic_holdout_facts"]}
            if holdout
            else {"rules_binding": "fam"}
        ),
    }
    rows = [
        {"id": "cal", "kernel": "calibrate.synthetic@1", "population": "create"},
        post_row,
        {
            "id": "variant",
            "kernel": "calibrate.variant@1",
            "population": "create",
            "artifact_inputs": [{"producer": "post"}] if feeds_variant else [],
        },
    ]
    nodes = [
        Node(
            "cal",
            "calibrate.synthetic@1",
            population="create",
            artifact_outputs=(ArtifactOutput("fit", fit),),
        ),
        Node(
            "post",
            post_row["kernel"],
            population="create",
            params={} if holdout else {"engine_ref": "fam-ref"},
            sources=("synthetic_holdout_facts",) if holdout else (),
            outputs=(Owned(entity, column, "float64"),),
            artifact_inputs=(ArtifactInput("fit", "cal", "fit", fit),),
            artifact_outputs=(ArtifactOutput("fit", fit),),
        ),
        Node(
            "variant",
            "calibrate.variant@1",
            population="create",
            artifact_inputs=(
                (ArtifactInput("fit", "post", "fit", fit),) if feeds_variant else ()
            ),
        ),
    ]
    stages = [_synthetic_stage(tax, ["tax_out"])]
    return _synthetic_stage_spec([tax, family], stages, rows), nodes


@pytest.mark.parametrize("feeds_variant", [False, True])
@pytest.mark.parametrize("post", ["holdout", "unlisted_unit"])
def test_precalibration_ancestry_node_after_central_fit_feeding_a_variant_is_checked(
    post, feeds_variant
):
    """After the central fit is not after calibration when a variant reads it."""
    spec, nodes = _variant_calibration_case(post, feeds_variant)
    reason = (
        "hold-out source"
        if post == "holdout"
        else ("not a declared pre-calibration stage")
    )
    expected = (
        (ValueError, f"Transport calibration ancestry: node 'post' carries {reason}.")
        if feeds_variant
        else None
    )
    assert _synthetic_outcomes(spec, nodes, {"tax": "tax-ref", "fam": "fam-ref"}) == {
        "declaration": expected,
        "composed": expected,
    }


def _precalibration_inventory(engine_refs, encoding):
    """The prepared engine_refs map, spelled the ways a parameter can carry it."""
    inventory = {"engine_refs": dict(sorted(engine_refs.items()))}
    canonical = canonical_json(inventory).decode()
    deep = canonical
    for _ in range(6):
        deep = json.dumps(deep)
    return {
        "string": canonical,
        "tuple": (canonical,),
        "spaced": json.dumps(inventory, indent=2),
        "nested": json.dumps({"receipt": json.dumps(inventory)}),
        "embedded": f"inventory={json.dumps(inventory)};",
        # A JSON string spelling every brace as an escape: no literal brace.
        "escaped": json.dumps(canonical).replace("{", "\\u007b"),
        # The key itself spelled with an escape.
        "escaped_key": canonical.replace('"engine_refs"', '"\\u0065ngine_refs"'),
        # Six layers of string encoding.
        "deep": deep,
    }[encoding]


_PRECALIBRATION_INVENTORY_ENCODINGS = [
    "string",
    "tuple",
    "spaced",
    "nested",
    "embedded",
    "escaped",
    "escaped_key",
    "deep",
]


@pytest.mark.parametrize("name", ["nz.targets.problem", "nz.precal.family"])
@pytest.mark.parametrize("encoding", _PRECALIBRATION_INVENTORY_ENCODINGS)
def test_precalibration_ancestry_compiled_prepared_map_still_refuses(
    calibration_ancestry_composed, name, encoding
):
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    graph = compose_transport_graph(spec, composed.config)
    prepared = _precalibration_inventory(composed.config.engine_refs, encoding)
    graph = replace(
        graph,
        nodes=tuple(
            replace(node, params={**node.params, "inventory": prepared})
            if node.id == name
            else node
            for node in graph.nodes
        ),
    )
    with pytest.raises(
        ValueError, match=f"node '{name}' carries prepared engine_refs map"
    ):
        compose_module.validate_transport_calibration_ancestry(
            spec, graph=graph, config=composed.config
        )


@pytest.mark.parametrize(
    "encoding",
    ["string", "spaced", "nested", "embedded", "escaped", "escaped_key", "deep"],
)
def test_precalibration_ancestry_declared_prepared_map_refuses(
    calibration_ancestry_composed, encoding
):
    """A listed stage node may not carry every engine in its declaration either."""
    composed = calibration_ancestry_composed
    spec = _precalibration_spec(composed)
    row = next(
        item
        for item in spec["resources"]["transport_graph"]["nodes"]
        if item["id"] == "nz.precal.family"
    )
    row["params"] = {
        "inventory": _precalibration_inventory(composed.config.engine_refs, encoding)
    }
    with pytest.raises(
        ValueError, match="node 'nz.precal.family' carries prepared engine_refs map"
    ):
        compose_module.validate_transport_calibration_ancestry(
            spec, config=composed.config
        )


_SYNTHETIC_SEED = Owned("person", "seed", "float64")


def _synthetic_stage_spec(bindings, stages, rows, *, holdout=()):
    return {
        "country": "synthetic",
        "resources": {
            "benefit_unit_rule": {"entity": "family"},
            "axiom_rules_bindings": {"bindings": bindings},
            "holdout_references": {"target_references": list(holdout)},
            "precalibration_stages": {"schema_version": 1, "stages": stages},
            "transport_graph": {
                "schema_version": 1,
                "country": "synthetic",
                "nodes": [{"id": "create", "kernel": "synthetic.create@1"}, *rows],
                "endpoints": {"calibration": "cal"},
            },
        },
    }


def _synthetic_binding(name, entity, variables):
    return {
        "id": name,
        "entity": entity,
        "rulespec_path": f"synthetic/{name}.yaml",
        "variables": list(variables),
    }


def _synthetic_stage(binding, outputs, derived=None):
    stage = {
        "id": f"stage.{binding['id']}",
        "rules_binding": binding["id"],
        "rulespec_path": binding["rulespec_path"],
        "outputs": [
            {
                "entity": binding["entity"],
                "column": column,
                "purpose": "calibration_measure",
            }
            for column in outputs
        ],
        "citation": "Synthetic stage contract.",
    }
    if derived is not None:
        stage["derived_outputs"] = [
            {"entity": entity, "column": column, "purpose": "calibration_measure"}
            for entity, column in derived
        ]
    return stage


def _synthetic_graph(nodes, create_params=None):
    """A synthetic graph: CREATE, then the given nodes."""
    return Graph(
        "synthetic",
        (
            SourceRef("donor", "synthetic@1"),
            SourceRef("synthetic_holdout_facts", "synthetic@1"),
        ),
        (
            Node(
                "create",
                "synthetic.create@1",
                structural=StructuralDelta.CREATE,
                sources=("donor",),
                outputs=(_SYNTHETIC_SEED,),
                params=create_params or {},
            ),
            *nodes,
        ),
    )


def _synthetic_outcomes(spec, nodes, engine_refs, create_params=None):
    """The guard's outcome in the declaration pass and the composed pass."""
    graph = _synthetic_graph(nodes, create_params)
    config = TransportGraphConfig(
        engine_refs=engine_refs, create_outputs=(_SYNTHETIC_SEED,)
    )
    return {
        label: _precalibration_outcome(
            compose_module.validate_transport_calibration_ancestry,
            spec,
            graph=graph if label == "composed" else None,
            config=config,
        )
        for label in ("declaration", "composed")
    }


def _overlap_case(a_produces_shared):
    """Bindings A and B both declare family.shared; each stage exposes only its public column."""
    bindings = [
        _synthetic_binding(name, "family", ["shared", f"{name}_public"])
        for name in ("A", "B")
    ]
    stages = [_synthetic_stage(row, [f"{row['id']}_public"]) for row in bindings]
    rows = [
        {
            "id": "b_source",
            "kernel": RULES,
            "population": "create",
            "rules_binding": "B",
            "outputs": [{"entity": "family", "column": "shared"}],
        },
        {
            "id": "b",
            "kernel": "simulate.solve_zero@1",
            "population": "create",
            "params": {"ref": {"binding": "B", "field": "engine_ref"}},
            "inputs": [{"entity": "family", "columns": ["shared"]}],
            "outputs": [{"entity": "family", "column": "B_public"}],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "create",
            "inputs": [{"entity": "family", "columns": ["B_public"]}],
        },
    ]
    nodes = [
        Node(
            "b_source",
            RULES,
            population="create",
            params={"engine_ref": "B-ref"},
            outputs=(Owned("family", "shared", "float64"),),
        ),
        Node(
            "b",
            "simulate.solve_zero@1",
            population="create",
            params={"engine_ref": "B-ref"},
            inputs=(Slice("family", ("shared",)),),
            outputs=(Owned("family", "B_public", "float64"),),
        ),
        Node(
            "cal",
            "calibrate.synthetic@1",
            population="create",
            inputs=(Slice("family", ("B_public",)),),
        ),
    ]
    if a_produces_shared:
        # A also runs upstream and owns a column of the same name, read by B.
        rows[0] = {**rows[0], "rules_binding": "A"}
        nodes[0] = replace(nodes[0], params={"engine_ref": "A-ref"})
    return _synthetic_stage_spec(bindings, stages, rows), nodes


@pytest.mark.parametrize("drop_unused", [False, True])
def test_precalibration_ancestry_unused_stage_does_not_restrict_by_name(drop_unused):
    """Rule 3 restricts what a stage's own nodes produce, not every matching name."""
    spec, nodes = _overlap_case(a_produces_shared=False)
    if drop_unused:
        spec["resources"]["precalibration_stages"]["stages"].pop(0)
    outcomes = _synthetic_outcomes(spec, nodes, {"A": "A-ref", "B": "B-ref"})
    assert outcomes == {"declaration": None, "composed": None}


def test_precalibration_ancestry_stage_output_read_by_another_stage_refuses():
    spec, nodes = _overlap_case(a_produces_shared=True)
    outcomes = _synthetic_outcomes(spec, nodes, {"A": "A-ref", "B": "B-ref"})
    message = (
        "Transport calibration ancestry: node 'b' carries undeclared "
        "pre-calibration stage output."
    )
    assert outcomes == {
        "declaration": (ValueError, message),
        "composed": (ValueError, message),
    }


def _version_member(name, kernel, version, *, binding=None, inputs=(), outputs=()):
    """A declaration row and its node in a population version.

    ``outputs`` are ``(column, rewrite)`` pairs on the person entity.
    """
    row = {"id": name, "kernel": kernel, "population": version}
    params = {}
    if binding is not None:
        row["rules_binding"] = binding
        params["engine_ref"] = f"{binding}-ref"
    if inputs:
        row["inputs"] = [{"entity": "person", "columns": list(inputs)}]
    if outputs:
        row["outputs"] = [
            {"entity": "person", "column": column, "rewrite": rewrite}
            for column, rewrite in outputs
        ]
    node = Node(
        name,
        kernel,
        population=version,
        params=params,
        inputs=(Slice("person", tuple(inputs)),) if inputs else (),
        outputs=tuple(
            Owned("person", column, "float64", rewrite=rewrite)
            for column, rewrite in outputs
        ),
    )
    return row, node


def _version_filter(name, base, *, inputs=()):
    row = {"id": name, "kernel": "synthetic.filter@1", "structural": "filter"}
    row["base"] = base
    if inputs:
        row["inputs"] = [{"entity": "person", "columns": list(inputs)}]
    node = Node(
        name,
        "synthetic.filter@1",
        structural=StructuralDelta.FILTER,
        base=base,
        inputs=(Slice("person", tuple(inputs)),) if inputs else (),
    )
    return row, node


def _version_case(shape):
    """Stages A and B each own a person.shared, in different population versions.

    A produces its private ``shared`` in CREATE. A FILTER makes ``v2``, where
    B's ``b`` reads ``shared`` and publishes only ``B_public`` to calibration.
    The shape says where the ``shared`` that ``b`` reads comes from.
    """
    bindings = [
        _synthetic_binding(name, "person", ["shared", f"{name}_public"])
        for name in ("A", "B")
    ]
    stages = [_synthetic_stage(row, [f"{row['id']}_public"]) for row in bindings]
    deep = shape.startswith("deep_")
    reader = "v3" if deep else "v2"
    members = [
        _version_member(
            "a_source",
            RULES,
            "create",
            binding="A",
            outputs=(("shared", False), ("A_public", False)),
        ),
        _version_filter(
            "v2",
            "create",
            inputs=("shared",) if shape.startswith("structural") else (),
        ),
    ]
    if shape in {"own", "structural", "deep_own"}:
        # B's own column of the same name, in a version above A's.
        members.append(
            _version_member(
                "b_source", RULES, "v2", binding="B", outputs=(("shared", False),)
            )
        )
    if shape in {"rewrite", "structural_shadowed"}:
        # A node outside both stages owns v2's shared: as a rewrite, its
        # incumbent is A's private column; otherwise it shadows A's.
        members.append(
            _version_member(
                "adjust",
                "synthetic.adjust@1",
                "v2",
                outputs=(("shared", shape == "rewrite"),),
            )
        )
    if deep:
        members.append(_version_filter("v3", "v2"))
    members.extend(
        [
            _version_member(
                "b",
                RULES,
                reader,
                binding="B",
                inputs=("shared",),
                outputs=(("B_public", False),),
            ),
            _version_member(
                "cal", "calibrate.synthetic@1", reader, inputs=("B_public",)
            ),
        ]
    )
    rows, nodes = zip(*members, strict=True)
    return _synthetic_stage_spec(bindings, stages, list(rows)), list(nodes)


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "compiled_only"])
@pytest.mark.parametrize(
    ("shape", "refuses"),
    [
        ("own", None),
        ("deep_own", None),
        ("inherited", "b"),
        ("deep_inherited", "b"),
        ("rewrite", "adjust"),
        ("structural", "v2"),
        ("structural_shadowed", "v2"),
    ],
)
def test_precalibration_ancestry_reads_follow_population_versions(
    shape, refuses, declared
):
    """Rule 3 follows each read to its producer in the reader's version lineage.

    B's own ``shared`` in a later version is not A's private column of the same
    name. A read that resolves to A's column refuses: an inherited read (also
    two versions up), a rewrite's incumbent and a structural node's input, which
    comes from its base even when its own version owns a column of that name.
    ``compiled_only`` declares just calibration, as for extension nodes, so the
    composed pass decides alone.
    """
    spec, nodes = _version_case(shape)
    if not declared:
        spec["resources"]["transport_graph"]["nodes"][1:] = [
            {"id": "cal", "kernel": "calibrate.synthetic@1", "population": "create"}
        ]
    outcome = (
        None
        if refuses is None
        else (
            ValueError,
            f"Transport calibration ancestry: node {refuses!r} carries undeclared "
            "pre-calibration stage output.",
        )
    )
    outcomes = _synthetic_outcomes(spec, nodes, {"A": "A-ref", "B": "B-ref"})
    assert outcomes == {
        "declaration": outcome if declared else None,
        "composed": outcome,
    }


@pytest.mark.parametrize("shape", ["inherited", "own"])
def test_precalibration_ancestry_producer_without_a_version_counts_everywhere(shape):
    """A declared producer with no population may be in the reader's version.

    It counts beside the reader's own version's owner too (``own``): the
    declaration alone cannot place it.
    """
    spec, nodes = _version_case(shape)
    rows = {row["id"]: row for row in spec["resources"]["transport_graph"]["nodes"]}
    del rows["a_source"]["population"]
    message = (
        "Transport calibration ancestry: node 'b' carries undeclared "
        "pre-calibration stage output."
    )
    outcomes = _synthetic_outcomes(spec, nodes, {"A": "A-ref", "B": "B-ref"})
    assert outcomes == {
        "declaration": (ValueError, message),
        "composed": (ValueError, message),
    }


def test_precalibration_ancestry_cyclic_versions_count_every_producer():
    """A declared base cycle never resolves, so every producer counts."""
    spec, _ = _version_case("inherited")
    rows = spec["resources"]["transport_graph"]["nodes"]
    {row["id"]: row for row in rows}["v2"]["base"] = "v3"
    rows.append(
        {
            "id": "v3",
            "kernel": "synthetic.filter@1",
            "structural": "filter",
            "base": "v2",
        }
    )
    assert _precalibration_outcome(
        compose_module.validate_transport_calibration_ancestry, spec
    ) == (
        ValueError,
        "Transport calibration ancestry: node 'b' carries undeclared "
        "pre-calibration stage output.",
    )


def test_precalibration_ancestry_unresolved_version_counts_every_producer():
    """A declared read with no version counts every producer of its column.

    The composed graph still places ``b`` in ``v2``; the declaration alone
    cannot, so it refuses (and a composed run checks the declaration first).
    """
    spec, nodes = _version_case("own")
    rows = {row["id"]: row for row in spec["resources"]["transport_graph"]["nodes"]}
    del rows["b"]["population"]
    message = (
        "Transport calibration ancestry: node 'b' carries undeclared "
        "pre-calibration stage output."
    )
    outcomes = _synthetic_outcomes(spec, nodes, {"A": "A-ref", "B": "B-ref"})
    assert outcomes == {
        "declaration": (ValueError, message),
        "composed": (ValueError, message),
    }


def _writer_case(shape):
    """A's private ``shared`` reaches B's ``b`` in ``v2`` only through ``adjust``.

    A produces ``shared`` in CREATE and a FILTER makes ``v2``, where ``adjust``,
    outside both stages, owns ``shared`` again. A full owner replaces A's
    values (``shadowed``). A masked owner keeps them on the rows outside its
    mask (``masked``, and two versions up in ``deep_masked``), and a rewrite
    passes them on: ``rewrite_read_first`` declares ``b`` before ``adjust``,
    so ``b`` is the first node that receives them.
    """
    bindings = [
        _synthetic_binding(name, "person", ["shared", f"{name}_public"])
        for name in ("A", "B")
    ]
    stages = [_synthetic_stage(row, [f"{row['id']}_public"]) for row in bindings]
    reader = "v3" if shape == "deep_masked" else "v2"
    masked = shape.endswith("masked")
    adjust_row = {
        "id": "adjust",
        "kernel": "synthetic.adjust@1",
        "population": "v2",
        "outputs": [{"entity": "person", "column": "shared"}],
    }
    adjust = Node(
        "adjust",
        "synthetic.adjust@1",
        population="v2",
        outputs=(Owned("person", "shared", "float64"),),
    )
    if masked:
        adjust_row["inputs"] = [{"entity": "person", "columns": ["keep"]}]
        adjust_row["outputs"][0]["rows"] = "keep"
        adjust = replace(
            adjust,
            inputs=(Slice("person", ("keep",)),),
            outputs=(Owned("person", "shared", "float64", rows="keep"),),
        )
    if shape == "rewrite_read_first":
        adjust_row["outputs"][0]["rewrite"] = True
        adjust = replace(
            adjust, outputs=(Owned("person", "shared", "float64", rewrite=True),)
        )
    members = [
        _version_member(
            "a_source",
            RULES,
            "create",
            binding="A",
            outputs=(("shared", False), ("A_public", False)),
        ),
        (
            {
                "id": "flags",
                "kernel": "synthetic.flags@1",
                "population": "create",
                "outputs": [{"entity": "person", "column": "keep"}],
            },
            Node(
                "flags",
                "synthetic.flags@1",
                population="create",
                outputs=(Owned("person", "keep", "bool"),),
            ),
        ),
        _version_filter("v2", "create"),
        (adjust_row, adjust),
    ]
    if shape == "deep_masked":
        members.append(_version_filter("v3", "v2"))
    members.extend(
        [
            _version_member(
                "b",
                RULES,
                reader,
                binding="B",
                inputs=("shared",),
                outputs=(("B_public", False),),
            ),
            _version_member(
                "cal", "calibrate.synthetic@1", reader, inputs=("B_public",)
            ),
        ]
    )
    if shape == "rewrite_read_first":
        members.insert(3, members.pop(4))
    rows, nodes = zip(*members, strict=True)
    return _synthetic_stage_spec(bindings, stages, list(rows)), list(nodes)


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "compiled_only"])
@pytest.mark.parametrize(
    ("shape", "refuses"),
    [
        ("shadowed", None),
        ("masked", "b"),
        ("deep_masked", "b"),
        ("rewrite_read_first", "b"),
    ],
)
def test_precalibration_ancestry_reads_follow_every_writer(shape, refuses, declared):
    """A column read counts every writer whose values it receives, as the executor does.

    The writers continue below an owner that keeps earlier values: a masked
    owner leaves A's values outside its mask, and a rewrite receives them. A
    full owner that is not a rewrite stops the chain.
    """
    spec, nodes = _writer_case(shape)
    if not declared:
        spec["resources"]["transport_graph"]["nodes"][1:] = [
            {"id": "cal", "kernel": "calibrate.synthetic@1", "population": "create"}
        ]
    outcome = (
        None
        if refuses is None
        else (
            ValueError,
            f"Transport calibration ancestry: node {refuses!r} carries undeclared "
            "pre-calibration stage output.",
        )
    )
    outcomes = _synthetic_outcomes(spec, nodes, {"A": "A-ref", "B": "B-ref"})
    assert outcomes == {
        "declaration": outcome if declared else None,
        "composed": outcome,
    }


def _expand_case(shape):
    """A's ``cell`` reaches calibration through an EXPAND that overlays it.

    ``a_source`` owns A's ``cell`` and ``other`` in CREATE. The EXPAND, no
    stage's node, overlays ``cell`` on its entrants, and ``claim`` names
    ``cell`` in ``materialized_expand_outputs``, so it receives the values of
    both without a ``Slice``. ``claim`` also owns ``other`` outright, which
    replaces A's ``other``. Calibration reads ``cell``, ``other`` and CREATE's
    ``seed``. Stage A lists ``cell`` except in the ``private`` shapes and
    ``claim_in_stage``, where ``claim`` is one of A's own nodes. In
    ``stage_expand`` the EXPAND is one of A's nodes too.
    ``private_read_first`` declares calibration before ``claim``, so
    calibration is the first node that receives A's values. In ``two_claims``
    the EXPAND overlays ``other`` too and ``claim`` claims it after ``cell``,
    so the second claim receives A's private ``other``.
    """
    binding = _synthetic_binding("A", "person", ["cell", "other", "A_public"])
    public = not shape.startswith("private") and shape != "claim_in_stage"
    stage = _synthetic_stage(binding, ["A_public", "cell"] if public else ["A_public"])
    cells = (("person", "cell", "float64"),)
    claims = ("person.cell",)
    if shape == "two_claims":
        cells += (("person", "other", "float64"),)
        claims += ("person.other",)
    expand_params = {"expand_cells": [list(cell) for cell in cells]}
    claim_params = {"materialized_expand_outputs": list(claims)}
    if shape == "stage_expand":
        expand_params["engine_ref"] = {"binding": "A", "field": "engine_ref"}
    if shape == "claim_in_stage":
        claim_params["engine_ref"] = {"binding": "A", "field": "engine_ref"}
    rows = [
        {
            "id": "a_source",
            "kernel": RULES,
            "population": "create",
            "rules_binding": "A",
            "outputs": [
                {"entity": "person", "column": column}
                for column in ("cell", "other", "A_public")
            ],
        },
        {
            "id": "a_expand",
            "kernel": "synthetic.expand@1",
            "structural": "expand",
            "base": "create",
            "params": expand_params,
        },
        {
            "id": "claim",
            "kernel": "synthetic.claim@1",
            "population": "a_expand",
            "params": claim_params,
            "outputs": [
                {"entity": "person", "column": "cell"},
                {"entity": "person", "column": "other"},
            ],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "a_expand",
            "inputs": [{"entity": "person", "columns": ["cell", "other", "seed"]}],
        },
    ]
    nodes = [
        Node(
            "a_source",
            RULES,
            population="create",
            params={"engine_ref": "A-ref"},
            outputs=tuple(
                Owned("person", column, "float64")
                for column in ("cell", "other", "A_public")
            ),
        ),
        Node(
            "a_expand",
            "synthetic.expand@1",
            structural=StructuralDelta.EXPAND,
            base="create",
            params={
                "expand_cells": cells,
                **({"engine_ref": "A-ref"} if shape == "stage_expand" else {}),
            },
        ),
        Node(
            "claim",
            "synthetic.claim@1",
            population="a_expand",
            params={
                "materialized_expand_outputs": claims,
                **({"engine_ref": "A-ref"} if shape == "claim_in_stage" else {}),
            },
            outputs=(
                Owned("person", "cell", "float64"),
                Owned("person", "other", "float64"),
            ),
        ),
        Node(
            "cal",
            "calibrate.synthetic@1",
            population="a_expand",
            inputs=(Slice("person", ("cell", "other", "seed")),),
        ),
    ]
    spec = _synthetic_stage_spec([binding], [stage], rows)
    # Values only composition resolves: the declaration cannot read them.
    spec["resources"]["synthetic_expand"] = {
        "cells": [list(cell) for cell in cells],
        "claims": ["person.cell"],
    }
    if shape == "selected_cells":
        expand_params["expand_cells"] = {
            "resource": "synthetic_expand",
            "path": ["cells"],
        }
    if shape == "selected_claims":
        claim_params["materialized_expand_outputs"] = {
            "resource": "synthetic_expand",
            "path": ["claims"],
        }
    if shape == "private_read_first":
        declared = spec["resources"]["transport_graph"]["nodes"]
        declared[3:] = declared[:2:-1]
        nodes[2:] = nodes[:1:-1]
    return spec, nodes


def _stage_output_refusal(node):
    return (
        ValueError,
        f"Transport calibration ancestry: node {node!r} carries undeclared "
        "pre-calibration stage output.",
    )


def _structural_stage_refusal(node, kind):
    return (
        ValueError,
        f"Transport calibration ancestry: node {node!r} of pre-calibration stage "
        f"binding 'A' is structural ({kind}); a stage changes columns, not the "
        "population.",
    )


@pytest.mark.parametrize(
    ("shape", "outcomes"),
    [
        ("public", {"declaration": None, "composed": None}),
        (
            "private",
            {
                "declaration": _stage_output_refusal("claim"),
                "composed": _stage_output_refusal("claim"),
            },
        ),
        # The claim keeps the values below it, so calibration receives them.
        (
            "private_read_first",
            {
                "declaration": _stage_output_refusal("cal"),
                "composed": _stage_output_refusal("cal"),
            },
        ),
        (
            "claim_in_stage",
            {
                "declaration": _stage_output_refusal("cal"),
                "composed": _stage_output_refusal("cal"),
            },
        ),
        # Cells only composition can tell: the declaration counts the EXPAND
        # as a writer of every column read through it.
        ("selected_cells", {"declaration": None, "composed": None}),
        # Claims only composition can tell: every output is claimed, so the
        # claim receives A's private ``other`` from below.
        (
            "selected_claims",
            {
                "declaration": _stage_output_refusal("claim"),
                "composed": _stage_output_refusal("claim"),
            },
        ),
        (
            "stage_expand",
            {
                "declaration": _structural_stage_refusal("a_expand", "expand"),
                "composed": _structural_stage_refusal("a_expand", "expand"),
            },
        ),
        # Every claim counts, not only the first: the second receives A's
        # private ``other``.
        (
            "two_claims",
            {
                "declaration": _stage_output_refusal("claim"),
                "composed": _stage_output_refusal("claim"),
            },
        ),
    ],
)
def test_precalibration_ancestry_expand_claims_read_their_cells(shape, outcomes):
    """An EXPAND overlays its cells, and a claim receives them without a ``Slice``.

    The overlay keeps the values below it, so a claim and each later reader of
    a cell count the EXPAND and every writer below it. An EXPAND that is a
    stage's own node refuses.
    """
    spec, nodes = _expand_case(shape)
    assert _synthetic_outcomes(spec, nodes, {"A": "A-ref"}) == outcomes


@pytest.mark.parametrize(
    ("node", "params", "message"),
    [
        ("a_expand", {}, "EXPAND node 'a_expand' needs tuple "),
        (
            "claim",
            {"materialized_expand_outputs": ("cell",)},
            "Node 'claim' params['materialized_expand_outputs'] must be a tuple ",
        ),
    ],
    ids=["expand_cells", "materialized_expand_outputs"],
)
def test_precalibration_ancestry_malformed_compiled_expand_params_refuse(
    node, params, message
):
    """The composed pass reads EXPAND cells and claims as the executor does."""
    spec, nodes = _expand_case("public")
    nodes = [
        replace(item, params=params) if item.id == node else item for item in nodes
    ]
    outcome = _synthetic_outcomes(spec, nodes, {"A": "A-ref"})
    assert outcome["declaration"] is None
    assert outcome["composed"][0] is ValueError
    assert outcome["composed"][1].startswith(
        f"Transport calibration ancestry: {message}"
    )


_STRUCTURAL_KINDS = ("create", "filter", "expand", "reweight")


def _structural_stage_case(kind, *, bound=True, ancestor=True):
    """Stage A publishes ``A_public``; a ``kind`` node may be one of A's nodes.

    ``change`` (CREATE itself for ``create``) carries A's engine reference
    when ``bound``. As an ancestor it opens the version calibration reads;
    otherwise calibration stays in CREATE. The EXPAND would overlay the unit
    membership ``person_family_id``, which calibration receives without a
    ``Slice``. Returns the spec, the nodes and CREATE's params.
    """
    binding = _synthetic_binding("A", "person", ["A_public"])
    version = "change" if ancestor and kind != "create" else "create"
    rows = [
        {
            "id": "a_source",
            "kernel": RULES,
            "population": "create",
            "rules_binding": "A",
            "outputs": [{"entity": "person", "column": "A_public"}],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": version,
            "inputs": [{"entity": "person", "columns": ["A_public", "seed"]}],
        },
    ]
    nodes = [
        Node(
            "a_source",
            RULES,
            population="create",
            params={"engine_ref": "A-ref"},
            outputs=(Owned("person", "A_public", "float64"),),
        ),
        Node(
            "cal",
            "calibrate.synthetic@1",
            population=version,
            inputs=(Slice("person", ("A_public", "seed")),),
        ),
    ]
    params, prepared, transition = {}, {}, {}
    if bound:
        params["engine_ref"] = {"binding": "A", "field": "engine_ref"}
        prepared["engine_ref"] = "A-ref"
    if kind != "create":
        if kind == "expand":
            cells = (("person", "person_family_id", "int64"),)
            params["expand_cells"] = [list(cell) for cell in cells]
            prepared["expand_cells"] = cells
        if kind == "reweight":
            transition = {
                "weights": WeightTransition("person", "importance", mass="free"),
                "mass": "free",
            }
        rows.insert(
            1,
            {
                "id": "change",
                "kernel": f"synthetic.{kind}@1",
                "structural": kind,
                "base": "create",
                **({"params": params} if params else {}),
            },
        )
        nodes.insert(
            1,
            Node(
                "change",
                f"synthetic.{kind}@1",
                structural=StructuralDelta(kind),
                base="create",
                params=prepared,
                **transition,
            ),
        )
    spec = _synthetic_stage_spec(
        [binding], [_synthetic_stage(binding, ["A_public"])], rows
    )
    if kind != "create":
        return spec, nodes, None
    create = spec["resources"]["transport_graph"]["nodes"][0]
    create["structural"] = "create"
    if bound:
        create["rules_binding"] = "A"
    return spec, nodes, prepared


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "compiled_only"])
@pytest.mark.parametrize("kind", _STRUCTURAL_KINDS)
def test_precalibration_ancestry_stage_nodes_change_no_population(kind, declared):
    """A stage node upstream of calibration is never structural.

    A stage declares only columns, but a structural node also writes what
    every later kernel of its version receives without a ``Slice``: ids,
    memberships, weights, strata, the frame metadata and the mass log.
    ``compiled_only`` declares just calibration, so the composed pass decides
    alone.
    """
    spec, nodes, create_params = _structural_stage_case(kind)
    if not declared:
        spec["resources"]["transport_graph"]["nodes"][:] = [
            {"id": "create", "kernel": "synthetic.create@1"},
            {"id": "cal", "kernel": "calibrate.synthetic@1", "population": "create"},
        ]
    outcome = _structural_stage_refusal(
        "create" if kind == "create" else "change", kind
    )
    outcomes = _synthetic_outcomes(spec, nodes, {"A": "A-ref"}, create_params)
    assert outcomes == {
        "declaration": outcome if declared else None,
        "composed": outcome,
    }


@pytest.mark.parametrize(
    ("kind", "bound", "ancestor"),
    [
        *((kind, False, True) for kind in _STRUCTURAL_KINDS),
        *((kind, True, False) for kind in _STRUCTURAL_KINDS[1:]),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_precalibration_ancestry_other_structural_nodes_pass(kind, bound, ancestor):
    """A structural node outside every stage, or no ancestor of calibration, passes."""
    spec, nodes, create_params = _structural_stage_case(
        kind, bound=bound, ancestor=ancestor
    )
    outcomes = _synthetic_outcomes(spec, nodes, {"A": "A-ref"}, create_params)
    assert outcomes == {"declaration": None, "composed": None}


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "compiled_only"])
@pytest.mark.parametrize("bound", [True, False], ids=["bound", "unbound"])
def test_precalibration_ancestry_checks_the_calibration_node_itself(bound, declared):
    """Calibration is in its own ancestry, so a stage's structural calibration refuses.

    Calibration here is a REWEIGHT that carries A's engine reference itself
    when ``bound``; unbound, it is the usual calibration and passes.
    """
    spec, nodes, _ = _structural_stage_case("create", bound=False)
    rows = spec["resources"]["transport_graph"]["nodes"]
    if declared:
        rows[-1] = {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "structural": "reweight",
            "base": "create",
            "inputs": [{"entity": "person", "columns": ["A_public", "seed"]}],
        }
        if bound:
            rows[-1]["params"] = {"engine_ref": {"binding": "A", "field": "engine_ref"}}
    else:
        rows[:] = [
            {"id": "create", "kernel": "synthetic.create@1"},
            {"id": "cal", "kernel": "calibrate.synthetic@1", "population": "create"},
        ]
    nodes[-1] = Node(
        "cal",
        "calibrate.synthetic@1",
        structural=StructuralDelta.REWEIGHT,
        base="create",
        inputs=(Slice("person", ("A_public", "seed")),),
        params={"engine_ref": "A-ref"} if bound else {},
        weights=WeightTransition("person", "importance", mass="free"),
        mass="free",
    )
    outcome = _structural_stage_refusal("cal", "reweight") if bound else None
    assert _synthetic_outcomes(spec, nodes, {"A": "A-ref"}) == {
        "declaration": outcome if declared else None,
        "composed": outcome,
    }


class _MembershipCreate(KernelBase):
    """Person 1 in household 10."""

    ref = "synthetic.membership_create@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        import pandas as pd

        from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

        person = {
            "person_id": np.array([1], dtype="int64"),
            "person_household_id": np.array([10], dtype="int64"),
            "mask": np.array([True]),
        }
        return KernelResult(
            frame=Frame(
                {
                    "person": pd.DataFrame(person),
                    "household": pd.DataFrame(
                        {"household_id": np.array([10], dtype="int64")}
                    ),
                },
                EntitySchema(group_entities=("household",)),
                {"household": Weights(np.array([1.0]), WeightKind.DESIGN)},
                pd.Series(["s"], name="stratum"),
            )
        )


class _MembershipExpand(KernelBase):
    """Clones person 1 into person 2 of a new household 20, halving the weights."""

    ref = "synthetic.membership_expand@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.EXPAND
    )

    def run(self, context):
        import pandas as pd

        from microcosm.frame import WeightKind, Weights

        return KernelResult(
            expand={
                "person": pd.Series(
                    [1], index=pd.Index([2], name="person_id"), dtype="int64"
                ),
                "household": pd.Series(
                    [10], index=pd.Index([20], name="household_id"), dtype="int64"
                ),
            },
            columns={
                ("person", "person_household_id"): pd.Series(
                    [10, 20], index=pd.Index([1, 2], name="person_id"), dtype="int64"
                )
            },
            weights=Weights(np.array([0.5, 0.5]), WeightKind.DESIGN),
        )


class _MembershipCalibrate(KernelBase):
    """Records the household membership it receives without a ``Slice``."""

    ref = "calibrate.membership@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def __init__(self):
        self.received = []

    def run(self, context):
        import pandas as pd

        person = context.tables["person"]
        self.received.append(person["person_household_id"].tolist())
        return KernelResult(
            columns={
                ("person", "result"): pd.Series(
                    person["person_household_id"].to_numpy(),
                    index=pd.Index(person["person_id"], name="person_id"),
                    dtype="int64",
                )
            }
        )


def test_precalibration_ancestry_refuses_a_stage_membership_calibration_receives(
    tmp_path,
):
    """Round 4's case: a stage's EXPAND writes a membership calibration receives.

    Stage A's EXPAND clones person 1 into a new household and overlays
    ``person_household_id``, and calibration declares no ``Slice``. Run, the
    graph hands calibration the stage's membership while the executor's
    writer lists hold no read of it, so only the structural rule can refuse
    it. Both passes do, at the EXPAND.
    """
    cells = (("person", "person_household_id", "int64"),)
    weights = {"expand_weight_entity": "household", "expand_weight_kind": "design"}
    nodes = (
        Node(
            "create",
            _MembershipCreate.ref,
            structural=StructuralDelta.CREATE,
            sources=("donor",),
            outputs=(Owned("person", "mask", "bool"),),
        ),
        Node(
            "stage_expand",
            _MembershipExpand.ref,
            structural=StructuralDelta.EXPAND,
            base="create",
            params={"engine_ref": "A-ref", "expand_cells": cells, **weights},
        ),
        Node(
            "cal",
            _MembershipCalibrate.ref,
            population="stage_expand",
            outputs=(Owned("person", "result", "int64"),),
        ),
    )
    binding = _synthetic_binding("A", "person", ["public"])
    rows = [
        {
            "id": "stage_expand",
            "kernel": _MembershipExpand.ref,
            "structural": "expand",
            "base": "create",
            "params": {
                "engine_ref": {"binding": "A", "field": "engine_ref"},
                "expand_cells": [list(cell) for cell in cells],
                **weights,
            },
        },
        {
            "id": "cal",
            "kernel": _MembershipCalibrate.ref,
            "population": "stage_expand",
            "outputs": [{"entity": "person", "column": "result"}],
        },
    ]
    spec = _synthetic_stage_spec(
        [binding], [_synthetic_stage(binding, ["public"])], rows
    )
    spec["resources"]["benefit_unit_rule"] = {"entity": "household"}
    graph = Graph("synthetic", (SourceRef("donor", "raw-bytes-v1"),), nodes)
    compiled = compile_graph(graph)
    config = TransportGraphConfig(
        engine_refs={"A": "A-ref"}, create_outputs=nodes[0].outputs
    )
    for candidate in (None, graph):
        assert _precalibration_outcome(
            compose_module.validate_transport_calibration_ancestry,
            spec,
            graph=candidate,
            config=config,
        ) == _structural_stage_refusal("stage_expand", "expand")
    assert _executor_writers(graph, compiled) == {
        "create": set(),
        "stage_expand": set(),
        "cal": set(),
    }
    calibrate = _MembershipCalibrate()
    kernels = KernelRegistry()
    for kernel in (_MembershipCreate(), _MembershipExpand(), calibrate):
        kernels.register(kernel)
    donor = tmp_path / "donor.txt"
    donor.write_text("synthetic")
    run_graph(
        compiled,
        sources={"donor": donor},
        store=ContentStore(tmp_path / "store"),
        kernels=kernels,
    )
    assert calibrate.received == [[10, 20]]


_MASS_RECORD = {
    "entity": "household",
    "old_total": 1.0,
    "new_total": 1.0,
    "declared_factor": None,
    "reason": "7",
}


def _mass_receipt(context):
    """A conserving household mass record whose reason is 7, when asked for."""
    if not context.params.get("append"):
        return {}
    return {"frame_mass_log_append": [dict(_MASS_RECORD)]}


class _MassLogCreate(KernelBase):
    """Person 1 in household 10, whose ``x`` is 1."""

    ref = "synthetic.mass_log_create@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        import pandas as pd

        from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

        return KernelResult(
            frame=Frame(
                {
                    "person": pd.DataFrame(
                        {
                            "person_id": np.array([1], dtype="int64"),
                            "person_household_id": np.array([10], dtype="int64"),
                        }
                    ),
                    "household": pd.DataFrame(
                        {
                            "household_id": np.array([10], dtype="int64"),
                            "x": np.array([1], dtype="int64"),
                        }
                    ),
                },
                EntitySchema(group_entities=("household",)),
                {"household": Weights(np.array([1.0]), WeightKind.DESIGN)},
                pd.Series(["s"], name="stratum"),
            )
        )


class _MassLogColumn(KernelBase):
    """Owns one person column, 7, and appends the mass record when asked."""

    ref = "synthetic.mass_log_column@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        import pandas as pd

        (owned,) = context.node.outputs
        return KernelResult(
            columns={
                (owned.entity, owned.column): pd.Series(
                    [7], index=pd.Index([1], name="person_id"), dtype="int64"
                )
            },
            receipt=_mass_receipt(context),
        )


class _MassLogCalibrate(KernelBase):
    """Weights the household by the reason of the last mass record it receives."""

    ref = "calibrate.mass_log@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
    )

    def __init__(self):
        self.received = []

    def run(self, context):
        from microcosm.frame import WeightKind, Weights

        log = context.frame_mass_log
        self.received.append([record.reason for record in log])
        return KernelResult(
            weights=Weights(
                np.array([float(log[-1].reason) if log else 1.0]),
                WeightKind.IMPORTANCE,
            ),
            receipt=_mass_receipt(context),
        )


@pytest.mark.parametrize("restored", [False, True], ids=["cold", "restored"])
@pytest.mark.parametrize("appender", [None, "a", "b", "cal"])
def test_transport_run_refuses_a_frame_mass_log_append(tmp_path, appender, restored):
    """A transport run takes the frame mass log from CREATE only.

    A node's receipt can append to the mass log (``frame_mass_log_append``),
    and every later structural node of its lineage receives the log without a
    ``Slice``. Round 5's case: stage A's node ``a`` owns only an unpublished
    column, yet the record it appends sets calibration's weight. No
    declaration shows the log, so both guard passes admit the graph, and a
    plain run delivers the value. The transport runner refuses any node that
    appends, a stage's (``a``) or not (``b``), calibration included, whether
    it ran or was restored from a store such a plain run wrote.
    """

    def params(name, **values):
        return {**values, **({"append": True} if name == appender else {})}

    nodes = (
        Node(
            "create",
            _MassLogCreate.ref,
            structural=StructuralDelta.CREATE,
            sources=("donor",),
            outputs=(Owned("household", "x", "int64"),),
        ),
        Node(
            "a",
            _MassLogColumn.ref,
            population="create",
            params=params("a", engine_ref="A-ref"),
            outputs=(Owned("person", "private", "int64"),),
        ),
        Node(
            "b",
            _MassLogColumn.ref,
            population="create",
            params=params("b"),
            outputs=(Owned("person", "b_out", "int64"),),
        ),
        Node(
            "cal",
            _MassLogCalibrate.ref,
            structural=StructuralDelta.REWEIGHT,
            base="create",
            params=params("cal"),
            inputs=(Slice("household", ("x",)),),
            weights=WeightTransition("household", "importance", mass="free"),
            mass="free",
        ),
    )
    binding = _synthetic_binding("A", "person", ["public"])
    rows = [
        {
            "id": "a",
            "kernel": _MassLogColumn.ref,
            "population": "create",
            "params": {"engine_ref": {"binding": "A", "field": "engine_ref"}},
            "outputs": [{"entity": "person", "column": "private"}],
        },
        {
            "id": "b",
            "kernel": _MassLogColumn.ref,
            "population": "create",
            "outputs": [{"entity": "person", "column": "b_out"}],
        },
        {
            "id": "cal",
            "kernel": _MassLogCalibrate.ref,
            "structural": "reweight",
            "base": "create",
            "inputs": [{"entity": "household", "columns": ["x"]}],
        },
    ]
    spec = _synthetic_stage_spec(
        [binding], [_synthetic_stage(binding, ["public"])], rows
    )
    graph = Graph("synthetic", (SourceRef("donor", "raw-bytes-v1"),), nodes)
    config = TransportGraphConfig(
        engine_refs={"A": "A-ref"}, create_outputs=nodes[0].outputs
    )
    for candidate in (None, graph):
        assert (
            _precalibration_outcome(
                compose_module.validate_transport_calibration_ancestry,
                spec,
                graph=candidate,
                config=config,
            )
            is None
        )
    calibrate = _MassLogCalibrate()
    kernels = KernelRegistry()
    for kernel in (_MassLogCreate(), _MassLogColumn(), calibrate):
        kernels.register(kernel)
    donor = tmp_path / "donor.txt"
    donor.write_text("synthetic")
    store = ContentStore(tmp_path / "store")
    if restored:
        run_graph(
            compile_graph(graph), sources={"donor": donor}, store=store, kernels=kernels
        )
        assert calibrate.received == [["7"] if appender in {"a", "b"} else []]
        calibrate.received.clear()

    def run():
        return cli_module._run(
            graph,
            SimpleNamespace(kernels=kernels),
            {"donor": donor},
            store,
            "require" if restored else "auto",
        )

    if appender is None:
        manifest = run()
        assert all(receipt.hit for receipt in manifest.nodes.values()) is restored
    else:
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Transport node {appender!r} changed the frame mass log it received"
            ),
        ):
            run()
    # Calibration never runs on an appended log under the runner.
    ran = not restored and appender in {None, "cal"}
    assert calibrate.received == ([[]] if ran else [])


def _guard_writers(graph, compiled):
    """Each node's reads as the composed pass resolves them, as (entity, column, writer)."""
    return compose_module._precalibration_compiled_reads(
        graph, compiled, *compose_module._precalibration_expansions(graph)
    )


def _executor_writers(graph, compiled):
    """Each node's reads from the executor's own static writer lists."""
    from microcosm.graph.executor import _input_writers

    return {
        node.id: {
            (*coordinate, writer)
            for coordinate, writers in _input_writers(compiled, node.id).items()
            for writer in writers
        }
        for node in graph.nodes
    }


@pytest.mark.parametrize(
    ("case", "shape"),
    [
        *((_writer_case, shape) for shape in ("shadowed", "masked", "deep_masked")),
        (_writer_case, "rewrite_read_first"),
        *(
            (_expand_case, shape)
            for shape in ("public", "private", "claim_in_stage", "two_claims")
        ),
    ],
    ids=lambda value: getattr(value, "__name__", value),
)
def test_precalibration_compiled_writers_match_the_executor(case, shape):
    """The composed pass and the executor give every read the same writers.

    The pass reimplements the executor's writer chain (``_writers_of``)
    because the guard runs before execution; this differential holds them
    together on the writer and EXPAND examples.
    """
    _, nodes = case(shape)
    graph = _synthetic_graph(nodes)
    compiled = compile_graph(graph)
    assert _guard_writers(graph, compiled) == _executor_writers(graph, compiled)


@pytest.mark.parametrize("source", ["skeleton", "extended", "entitlement"])
def test_precalibration_compiled_writers_match_the_executor_on_composed_graphs(
    request, source
):
    """The same differential on the composed NZ graphs and the entitlement graph."""
    if source == "entitlement":
        fixture = request.getfixturevalue("precalibration_entitlement")
        graph = compose_transport_graph(fixture.spec, fixture.config)
    else:
        graph = request.getfixturevalue("composed").graph(extended=source == "extended")
    compiled = compile_graph(graph)
    assert _guard_writers(graph, compiled) == _executor_writers(graph, compiled)


def _derived_case(*, derived, outputs=(), producer_upstream=True):
    """A person-binding gross-up solve whose output is not a binding variable.

    By default the stage lists no binding variable, as NZ's Schedule 1 stage
    does: the tax stays inside the solve and only the gross-up flows on.
    """
    binding = _synthetic_binding("tax", "person", ["tax_before_credits"])
    stage = _synthetic_stage(binding, outputs, derived)
    population = "create" if producer_upstream else "after"
    rows = [
        {
            "id": "gross",
            "kernel": "simulate.solve_zero@1",
            "population": population,
            "params": {"engine_ref": {"binding": "tax", "field": "engine_ref"}},
            "outputs": [{"entity": "person", "column": "benefit_gross"}],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "create",
            "inputs": [
                {
                    "entity": "person",
                    "columns": ["benefit_gross" if producer_upstream else "seed"],
                }
            ],
        },
    ]
    nodes = [
        Node(
            "gross",
            "simulate.solve_zero@1",
            population="create",
            params={"engine_ref": "tax-ref"},
            outputs=(Owned("person", "benefit_gross", "float64"),),
        ),
        Node(
            "cal",
            "calibrate.synthetic@1",
            population="create",
            inputs=(Slice("person", (rows[1]["inputs"][0]["columns"][0],)),),
        ),
    ]
    if not producer_upstream:
        rows.insert(
            1,
            {
                "id": "after",
                "kernel": "transport.boundary@1",
                "structural": "filter",
                "base": "cal",
            },
        )
        nodes = [nodes[1]]
    return _synthetic_stage_spec([binding], [stage], rows), nodes


def test_precalibration_ancestry_derived_output_flows_to_calibration():
    """NZ's w2b gross-up: a selector stage node's own column is a declared output."""
    spec, nodes = _derived_case(derived=[("person", "benefit_gross")])
    assert _synthetic_outcomes(spec, nodes, {"tax": "tax-ref"}) == {
        "declaration": None,
        "composed": None,
    }


def test_precalibration_ancestry_undeclared_derived_column_refuses():
    spec, nodes = _derived_case(derived=None, outputs=["tax_before_credits"])
    message = (
        "Transport calibration ancestry: node 'cal' carries undeclared "
        "pre-calibration stage output."
    )
    assert _synthetic_outcomes(spec, nodes, {"tax": "tax-ref"}) == {
        "declaration": (ValueError, message),
        "composed": (ValueError, message),
    }


@pytest.mark.parametrize("producer_upstream", [False, True])
def test_precalibration_ancestry_derived_output_needs_an_upstream_producer(
    producer_upstream,
):
    """A derived row naming a column no upstream stage node produces refuses."""
    # Upstream, the real column is declared too, so only the phantom row fails.
    column = "never_produced" if producer_upstream else "benefit_gross"
    derived = [("person", "benefit_gross")] if producer_upstream else []
    spec, nodes = _derived_case(
        derived=[*derived, ("person", column)], producer_upstream=producer_upstream
    )
    message = (
        f"precalibration_stages derived output person.{column} of 'stage.tax' is "
        "not produced by a stage node upstream of calibration."
    )
    assert _synthetic_outcomes(spec, nodes, {"tax": "tax-ref"}) == {
        "declaration": (ValueError, message),
        "composed": (ValueError, message),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "key",
        "binding_variable",
        "purpose",
        "duplicate",
        "holdout",
        "empty_column",
        "no_outputs",
    ],
)
def test_precalibration_ancestry_derived_output_schema_refuses(mutation):
    spec, _ = _derived_case(derived=[("person", "benefit_gross")])
    stage = spec["resources"]["precalibration_stages"]["stages"][0]
    row = stage["derived_outputs"][0]
    if mutation == "key":
        row["unknown"] = True
    elif mutation == "binding_variable":
        row["column"] = "tax_before_credits"
    elif mutation == "purpose":
        row["purpose"] = "evaluation"
    elif mutation == "duplicate":
        stage["derived_outputs"].append(dict(row))
    elif mutation == "holdout":
        spec["resources"]["holdout_references"]["target_references"].append(
            {"entity": "person", "measure": "benefit_gross"}
        )
    elif mutation == "no_outputs":
        stage["derived_outputs"] = []
    else:
        row["column"] = ""
    with pytest.raises(ValueError, match="precalibration_stages"):
        compose_module.validate_transport_calibration_ancestry(spec)


@pytest.mark.parametrize("compiled", [False, True])
def test_precalibration_ancestry_stage_node_holdout_source_refuses(compiled):
    """The hold-out check covers a stage's own nodes, not only their ancestors."""
    spec, nodes = _derived_case(derived=[("person", "benefit_gross")])
    rows = spec["resources"]["transport_graph"]["nodes"]
    if compiled:
        nodes = [
            replace(node, sources=("synthetic_holdout_facts",))
            if node.id == "gross"
            else node
            for node in nodes
        ]
    else:
        next(row for row in rows if row["id"] == "gross")["sources"] = [
            "synthetic_holdout_facts"
        ]
    outcomes = _synthetic_outcomes(spec, nodes, {"tax": "tax-ref"})
    label = "composed" if compiled else "declaration"
    assert outcomes[label] == (
        ValueError,
        "Transport calibration ancestry: node 'gross' carries hold-out source.",
    )


@pytest.mark.parametrize("guard", ["absent", "empty"])
@pytest.mark.parametrize("malformed", ["unit_entity", "binding_entity", "missing"])
def test_precalibration_ancestry_malformed_entities_match_today(
    monkeypatch, guard, malformed
):
    """Without stages, even a malformed entity gets today's exact outcome."""
    spec = _synthetic_stage_spec(
        [], [], [{"id": "cal", "kernel": "calibrate.synthetic@1"}]
    )
    resources = spec["resources"]
    if malformed == "unit_entity":
        resources["benefit_unit_rule"]["entity"] = []
    elif malformed == "binding_entity":
        resources["axiom_rules_bindings"]["bindings"] = [
            {"id": "odd", "entity": ["family"], "variables": ["x"]}
        ]
    else:
        del resources["benefit_unit_rule"]["entity"]
        resources["axiom_rules_bindings"]["bindings"] = [
            {"id": "odd", "variables": ["x"], "rulespec_path": "odd.yaml"}
        ]
    if guard == "absent":
        del resources["precalibration_stages"]
    expected = _precalibration_outcome(_precalibration_reference_guard, spec)
    assert (
        _precalibration_outcome(
            compose_module.validate_transport_calibration_ancestry, spec
        )
        == expected
    )


def _shape_case():
    """A listed unit stage and one reader of its declared output, well formed."""
    binding = _synthetic_binding("u", "family", ["u_out", "u_priv"])
    rows = [
        {
            "id": "probe",
            "kernel": "synthetic.probe@1",
            "population": "create",
            "params": {"engine_ref": {"binding": "u", "field": "engine_ref"}},
            "outputs": [
                {"entity": "family", "column": "u_out"},
                {"entity": "family", "column": "u_priv"},
            ],
        },
        {
            "id": "mid",
            "kernel": "synthetic.mid@1",
            "population": "create",
            "inputs": [{"entity": "family", "columns": ["u_out"]}],
            "outputs": [{"entity": "person", "column": "mid_out"}],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "create",
            "inputs": [{"entity": "person", "columns": ["mid_out"]}],
        },
    ]
    binding["engine_ref"] = "u-ref"
    stage = _synthetic_stage(binding, ["u_out"])
    spec = _synthetic_stage_spec([binding], [stage], rows)
    # Data a non-stage node may select; it embeds the stage's engine reference.
    spec["resources"]["probe_refs"] = {"refs": {"u": "u-ref"}}
    return spec


def _shape_rows(spec):
    return {row["id"]: row for row in spec["resources"]["transport_graph"]["nodes"]}


def _shape_outcome(guard, spec):
    """Any refusal, so today's guard compares even where it raises KeyError."""
    try:
        guard(spec)
    except Exception as error:
        return type(error), str(error)
    return None


#: A malformed field the declaration pass reads, by the node that carries it.
_MALFORMED_NODE_FIELDS = {
    "string_columns": ("mid", "inputs", [{"entity": "family", "columns": "u_priv"}]),
    "missing_columns": ("mid", "inputs", [{"entity": "family"}]),
    "string_inputs": ("mid", "inputs", "u_out"),
    "output_without_entity": ("mid", "outputs", [{"column": "mid_out"}]),
    "list_output_rows": (
        "mid",
        "outputs",
        [{"entity": "person", "column": "mid_out", "rows": ["keep"]}],
    ),
    "null_output_rows": (
        "mid",
        "outputs",
        [{"entity": "person", "column": "mid_out", "rows": None}],
    ),
    "unknown_structural": ("mid", "structural", "explode"),
    "list_structural": ("mid", "structural", ["expand"]),
    "edge_without_producer": ("cal", "artifact_inputs", [{"name": "trace"}]),
    "string_edges": ("cal", "artifact_inputs", "probe"),
    "missing_kernel": ("probe", "kernel", None),
    "list_population": ("mid", "population", ["create"]),
    "list_rules_binding": ("mid", "rules_binding", ["u"]),
    "list_params": ("probe", "params", ["engine_ref"]),
    "string_path": (
        "mid",
        "params",
        {"rows": {"resource": "probe_refs", "path": "refs"}},
    ),
    "string_sources": ("mid", "sources", "synthetic_holdout_facts"),
    "list_resource": ("mid", "params", {"rows": {"resource": ["holdout_references"]}}),
    "unknown_encoding": (
        "mid",
        "params",
        {"rows": {"resource": "probe_refs", "encoding": ["sha256"]}},
    ),
    "list_binding_selector": (
        "mid",
        "params",
        {"ref": {"binding": ["u"], "field": "engine_ref"}},
    ),
}


def test_precalibration_ancestry_shape_case_is_accepted():
    assert (
        _shape_outcome(
            compose_module.validate_transport_calibration_ancestry, _shape_case()
        )
        is None
    )


@pytest.mark.parametrize("case", sorted(_MALFORMED_NODE_FIELDS))
def test_precalibration_ancestry_malformed_node_field_refuses_naming_the_node(case):
    """With stages, a malformed field refuses before it can drop a read or edge."""
    node, field, value = _MALFORMED_NODE_FIELDS[case]
    spec = _shape_case()
    _shape_rows(spec)[node][field] = value
    guard = compose_module.validate_transport_calibration_ancestry
    assert _shape_outcome(guard, spec) == (
        ValueError,
        f"Transport calibration ancestry: node {node!r} has malformed {field}.",
    )
    # Without stages the guard is today's, malformed declarations included.
    spec["resources"]["precalibration_stages"]["stages"] = []
    assert _shape_outcome(guard, spec) == _shape_outcome(
        _precalibration_reference_guard, spec
    )


@pytest.mark.parametrize(
    ("case", "value", "reason"),
    [
        (
            "string_columns",
            [{"entity": "family", "columns": ["u_priv"]}],
            "undeclared pre-calibration stage output",
        ),
        (
            "string_path",
            {"rows": {"resource": "probe_refs", "path": ["refs"]}},
            "not a declared pre-calibration stage",
        ),
        ("string_sources", ["synthetic_holdout_facts"], "hold-out source"),
        (
            "list_resource",
            {"rows": {"resource": "holdout_references"}},
            "holdout_references selection",
        ),
    ],
)
def test_precalibration_ancestry_malformed_spelling_hid_a_refusal(case, value, reason):
    """Read character by character, these spellings used to be accepted."""
    node, field, _ = _MALFORMED_NODE_FIELDS[case]
    spec = _shape_case()
    _shape_rows(spec)[node][field] = value
    assert _shape_outcome(
        compose_module.validate_transport_calibration_ancestry, spec
    ) == (
        ValueError,
        f"Transport calibration ancestry: node {node!r} carries {reason}.",
    )


def test_precalibration_ancestry_repeated_node_id_refuses():
    """A repeated id would hide the first node's read of an undeclared output."""
    spec = _shape_case()
    rows = spec["resources"]["transport_graph"]["nodes"]
    _shape_rows(spec)["mid"]["inputs"] = [{"entity": "family", "columns": ["u_priv"]}]
    rows.insert(
        rows.index(_shape_rows(spec)["cal"]),
        {"id": "mid", "kernel": "synthetic.mid@1", "population": "create"},
    )
    assert _shape_outcome(
        compose_module.validate_transport_calibration_ancestry, spec
    ) == (
        ValueError,
        "Transport calibration ancestry: node 'mid' is declared twice.",
    )


@pytest.mark.parametrize("name", [None, ["mid"], 3])
def test_precalibration_ancestry_non_text_node_id_refuses(name):
    spec = _shape_case()
    _shape_rows(spec)["mid"]["id"] = name
    assert _shape_outcome(
        compose_module.validate_transport_calibration_ancestry, spec
    ) == (
        ValueError,
        f"Transport calibration ancestry: node id {name!r} is not text.",
    )


@pytest.mark.parametrize(
    ("case", "change"),
    [
        ("string_variables", {"variables": "u_out_total"}),
        ("list_entity", {"entity": ["family"]}),
        ("missing_rulespec_path", {"rulespec_path": None}),
    ],
)
def test_precalibration_ancestry_malformed_binding_refuses(case, change):
    """A string variable list would match stage outputs by substring."""
    spec = _shape_case()
    (binding,) = spec["resources"]["axiom_rules_bindings"]["bindings"]
    binding.update(change)
    assert _shape_outcome(
        compose_module.validate_transport_calibration_ancestry, spec
    ) == (
        ValueError,
        "precalibration_stages needs rules binding 'u' to name its entity, "
        "rulespec_path and variables.",
    )


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("u", "needs distinct rules binding ids; 'u' repeats."),
        (None, "needs text rules binding ids, not None."),
    ],
    ids=["repeated", "non_text"],
)
def test_precalibration_ancestry_binding_id_refuses(name, message):
    spec = _shape_case()
    bindings = spec["resources"]["axiom_rules_bindings"]["bindings"]
    bindings.append({**_synthetic_binding("u", "person", ["other"]), "id": name})
    assert _shape_outcome(
        compose_module.validate_transport_calibration_ancestry, spec
    ) == (ValueError, f"precalibration_stages {message}")


@pytest.mark.parametrize("endpoint", [["cal"], None, ""])
def test_precalibration_ancestry_malformed_calibration_endpoint_refuses(endpoint):
    """A present endpoint names a node: null would drop a non-calibrate root."""
    spec = _shape_case()
    rows = _shape_rows(spec)
    rows["cal"]["kernel"] = "synthetic.fit@1"
    rows["mid"]["sources"] = ["synthetic_holdout_facts"]
    guard = compose_module.validate_transport_calibration_ancestry
    assert _shape_outcome(guard, spec) == (
        ValueError,
        "Transport calibration ancestry: node 'mid' carries hold-out source.",
    )
    spec["resources"]["transport_graph"]["endpoints"]["calibration"] = endpoint
    assert _shape_outcome(guard, spec) == (
        ValueError,
        "Transport calibration ancestry: the calibration endpoint must name a node.",
    )


@pytest.mark.parametrize(
    "note",
    [
        "the engine_refs map",
        '{"note": "engine_refs"}',
        '["engine_refs"]',
        '"\\u0065ngine_refs"',
    ],
)
def test_precalibration_ancestry_engine_refs_text_is_not_the_map(note):
    """Only the map's key, followed by a colon, marks the inventory."""
    spec = _shape_case()
    _shape_rows(spec)["mid"]["params"] = {"note": note}
    assert (
        _shape_outcome(compose_module.validate_transport_calibration_ancestry, spec)
        is None
    )


@pytest.mark.parametrize("scope", ["helper", "person_stage"])
def test_precalibration_ancestry_inventory_scan_is_linear(scope):
    """A harmless parameter full of escaped quotes costs one pass per layer.

    Decoding JSON from every quote made the inventory scan quadratic, four
    times the cost per doubling: on an idle host 32 KB took 0.5 s and 128 KB
    7.5 s. Decoding layer by layer takes milliseconds at 128 KB. The guard case
    lists only a person stage, because the engine-reference scan that predates
    the stages (``_embeds_engine_ref``) still decodes JSON from every quote of
    a parameter when a benefit-unit engine reference is looked for.
    """
    note = "prefix" + r"\"" * 64_000
    binding = {**_synthetic_binding("p", "person", ["p_out"]), "engine_ref": "p-ref"}
    rows = [
        {
            "id": "probe",
            "kernel": "synthetic.probe@1",
            "population": "create",
            "params": {"engine_ref": {"binding": "p", "field": "engine_ref"}},
            "outputs": [{"entity": "person", "column": "p_out"}],
        },
        {
            "id": "mid",
            "kernel": "synthetic.mid@1",
            "population": "create",
            "params": {"note": note},
            "inputs": [{"entity": "person", "columns": ["p_out"]}],
            "outputs": [{"entity": "person", "column": "mid_out"}],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "create",
            "inputs": [{"entity": "person", "columns": ["mid_out"]}],
        },
    ]
    spec = _synthetic_stage_spec(
        [binding], [_synthetic_stage(binding, ["p_out"])], rows
    )
    guard = compose_module.validate_transport_calibration_ancestry
    started = time.perf_counter()
    if scope == "helper":
        assert not compose_module._carries_engine_ref_inventory(note)
    else:
        assert _shape_outcome(guard, spec) is None
    assert time.perf_counter() - started < 1


def test_precalibration_ancestry_composed_stage_scan_is_linear(monkeypatch):
    """The composed pass reads stage bindings in one pass per escape layer.

    Round 5's case: a person stage, a node with a harmless parameter full of
    escaped quotes, then calibration. Matching each binding's engine
    reference by decoding JSON from every quote made this composed path
    quadratic (1.8 s at 16 KB on a loaded host). It now decodes no JSON from
    an offset of any parameter here, so the count is exact, not timed.
    """

    class Counted(json.JSONDecoder):
        offsets = 0

        def raw_decode(self, text, idx=0):
            type(self).offsets += 1
            return super().raw_decode(text, idx)

    monkeypatch.setattr(json, "JSONDecoder", Counted)
    note = "prefix" + r"\"" * 64_000
    binding = {**_synthetic_binding("p", "person", ["p_out"]), "engine_ref": "p-ref"}
    rows = [
        {
            "id": "probe",
            "kernel": "synthetic.probe@1",
            "population": "create",
            "params": {"engine_ref": {"binding": "p", "field": "engine_ref"}},
            "outputs": [{"entity": "person", "column": "p_out"}],
        },
        {
            "id": "mid",
            "kernel": "synthetic.mid@1",
            "population": "create",
            "params": {"note": note},
            "inputs": [{"entity": "person", "columns": ["p_out"]}],
            "outputs": [{"entity": "person", "column": "mid_out"}],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "create",
            "inputs": [{"entity": "person", "columns": ["mid_out"]}],
        },
    ]
    nodes = [
        Node(
            "probe",
            "synthetic.probe@1",
            population="create",
            params={"engine_ref": "p-ref"},
            outputs=(Owned("person", "p_out", "float64"),),
        ),
        Node(
            "mid",
            "synthetic.mid@1",
            population="create",
            params={"note": note},
            inputs=(Slice("person", ("p_out",)),),
            outputs=(Owned("person", "mid_out", "float64"),),
        ),
        Node(
            "cal",
            "calibrate.synthetic@1",
            population="create",
            inputs=(Slice("person", ("mid_out",)),),
        ),
    ]
    spec = _synthetic_stage_spec(
        [binding], [_synthetic_stage(binding, ["p_out"])], rows
    )
    started = time.perf_counter()
    outcomes = _synthetic_outcomes(spec, nodes, {"p": "p-ref"})
    elapsed = time.perf_counter() - started
    assert outcomes == {"declaration": None, "composed": None}
    assert Counted.offsets == 0
    assert elapsed < 1


@pytest.mark.parametrize(
    ("carrier", "refused"), [("a_source", True), ("cal", True), ("post", False)]
)
def test_precalibration_ancestry_unreadable_composed_bindings_refuse_upstream(
    carrier, refused
):
    """A parameter still escaped after 32 layers refuses upstream of calibration.

    Its node's stage bindings cannot be read, so on stage A's node or on the
    calibration node the composed pass refuses. ``post`` reads calibration's
    output, so none of its bindings is used and it faces only the shape
    checks; round 6 found it refused before the ancestry was known.
    """
    binding = _synthetic_binding("A", "person", ["A_public"])
    rows = [
        {
            "id": "a_source",
            "kernel": RULES,
            "population": "create",
            "rules_binding": "A",
            "outputs": [{"entity": "person", "column": "A_public"}],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "create",
            "inputs": [{"entity": "person", "columns": ["A_public", "seed"]}],
            "outputs": [{"entity": "person", "column": "cal_out"}],
        },
        {
            "id": "post",
            "kernel": "synthetic.post@1",
            "population": "create",
            "inputs": [{"entity": "person", "columns": ["cal_out"]}],
        },
    ]
    nodes = [
        Node(
            "a_source",
            RULES,
            population="create",
            params={"engine_ref": "A-ref"},
            outputs=(Owned("person", "A_public", "float64"),),
        ),
        Node(
            "cal",
            "calibrate.synthetic@1",
            population="create",
            inputs=(Slice("person", ("A_public", "seed")),),
            outputs=(Owned("person", "cal_out", "float64"),),
        ),
        Node(
            "post",
            "synthetic.post@1",
            population="create",
            inputs=(Slice("person", ("cal_out",)),),
        ),
    ]
    deep = _escape_layers('"note"', 33)
    nodes = [
        replace(node, params={**node.params, "note": deep})
        if node.id == carrier
        else node
        for node in nodes
    ]
    spec = _synthetic_stage_spec(
        [binding], [_synthetic_stage(binding, ["A_public"])], rows
    )
    refusal = (
        ValueError,
        f"Transport calibration ancestry: node {carrier!r} has a parameter "
        "still JSON-escaped after 32 layers, so its stage bindings cannot be "
        "read.",
    )
    assert _synthetic_outcomes(spec, nodes, {"A": "A-ref"}) == {
        "declaration": None,
        "composed": refusal if refused else None,
    }


def test_precalibration_ancestry_composed_stage_reads_escaped_reference_text():
    """A node carrying its reference's escaped text after a backslash is a member.

    Round 6: the reference is a newline, a backslash and ``n``, and ``mid``'s
    note is a backslash, then the reference's JSON-escaped text. Decoding
    pairs the note's backslash with the text's first, so no decoded layer
    holds the reference. ``_embeds_engine_ref`` matches the escaped text as
    written, and so must the composed pass, or ``mid`` leaves stage p and its
    derived output refuses as unproduced.
    """
    reference = "\n\\n"
    note = "\\" + json.dumps(reference, ensure_ascii=False)[1:-1]
    assert compose_module._embeds_engine_ref(note, {reference})
    assert compose_module._embedded_engine_refs(note, {reference}) == {reference}
    binding = {
        **_synthetic_binding("p", "person", ["p_out", "p_priv"]),
        "engine_ref": reference,
    }
    rows = [
        {
            "id": "probe",
            "kernel": "synthetic.probe@1",
            "population": "create",
            "params": {"engine_ref": {"binding": "p", "field": "engine_ref"}},
            "outputs": [
                {"entity": "person", "column": "p_out"},
                {"entity": "person", "column": "p_priv"},
            ],
        },
        {
            "id": "mid",
            "kernel": "synthetic.mid@1",
            "population": "create",
            "params": {"period": {"binding": "p", "field": "period"}, "note": note},
            "inputs": [{"entity": "person", "columns": ["p_priv"]}],
            "outputs": [{"entity": "person", "column": "mid_out"}],
        },
        {
            "id": "cal",
            "kernel": "calibrate.synthetic@1",
            "population": "create",
            "inputs": [{"entity": "person", "columns": ["mid_out"]}],
        },
    ]
    nodes = [
        Node(
            "probe",
            "synthetic.probe@1",
            population="create",
            params={"engine_ref": reference},
            outputs=(
                Owned("person", "p_out", "float64"),
                Owned("person", "p_priv", "float64"),
            ),
        ),
        Node(
            "mid",
            "synthetic.mid@1",
            population="create",
            params={"period": "2026", "note": note},
            inputs=(Slice("person", ("p_priv",)),),
            outputs=(Owned("person", "mid_out", "float64"),),
        ),
        Node(
            "cal",
            "calibrate.synthetic@1",
            population="create",
            inputs=(Slice("person", ("mid_out",)),),
        ),
    ]
    spec = _synthetic_stage_spec(
        [binding],
        [_synthetic_stage(binding, ["p_out"], [("person", "mid_out")])],
        rows,
    )
    assert _synthetic_outcomes(spec, nodes, {"p": reference}) == {
        "declaration": None,
        "composed": None,
    }


_SCANNED_REFS = (
    "p-ref",
    '{"format":"synthetic","module":"a/b.yaml"}',
    '{"module":"\U0001f600/\\\\ \\"q\\""}',
    # Round 6: decoding after a stray backslash loses it (see below).
    "\n\\n",
)
# Characters whose escapes start with a backslash, and some that don't.
_ESCAPE_CHARACTERS = '\\"\nn{\u00e9'


@st.composite
def _spelled_engine_ref(draw):
    """An engine reference behind up to four random JSON spellings."""
    reference = draw(
        st.sampled_from(_SCANNED_REFS)
        | st.text(alphabet=_ESCAPE_CHARACTERS, min_size=1, max_size=6)
    )
    value = reference
    for _ in range(draw(st.integers(min_value=0, max_value=4))):
        step = draw(
            st.sampled_from(("dumps", "ascii", "object", "list", "content", "text"))
        )
        if step == "dumps":
            value = json.dumps(value, ensure_ascii=False)
        elif step == "ascii":
            value = json.dumps(value)
        elif step == "object":
            key = draw(st.text(max_size=3))
            value = json.dumps({key: value}, ensure_ascii=draw(st.booleans()))
        elif step == "list":
            value = json.dumps([draw(st.integers()), value])
        elif step == "content":
            # A string's content as JSON writes it, without its quotes.
            value = json.dumps(value, ensure_ascii=draw(st.booleans()))[1:-1]
        else:
            text = st.text(max_size=4) | st.text(
                alphabet=_ESCAPE_CHARACTERS, max_size=4
            )
            # A stray backslash pairs with the value's first escape.
            stray = draw(st.sampled_from(("", "\\")))
            value = draw(text) + stray + value + draw(text)
    return reference, value


@settings(max_examples=300, deadline=None)
@given(case=_spelled_engine_ref(), as_key=st.booleans())
def test_precalibration_stage_scan_finds_what_json_decoding_finds(case, as_key):
    """The layered scan finds every reference ``_embeds_engine_ref`` finds.

    ``_embeds_engine_ref`` decodes JSON from every quote and brace; the
    composed pass's membership scan decodes escapes layer by layer instead.
    Over fixed and random references, and nested JSON, ASCII-escaped
    (surrogate pairs included), object, list, quoteless and embedded-in-text
    spellings, the layered scan never misses a reference the decoding scan
    finds.
    """
    reference, value = case
    params = {value: 1} if as_key else {"note": [value]}
    found = compose_module._embedded_engine_refs(params, {reference})
    assert found is not None
    if compose_module._embeds_engine_ref(params, {reference}):
        assert reference in found


@settings(max_examples=300, deadline=None)
@given(
    reference=st.text(alphabet=_ESCAPE_CHARACTERS, min_size=1, max_size=4),
    prefix=st.text(alphabet=_ESCAPE_CHARACTERS, max_size=3),
    suffix=st.text(alphabet=_ESCAPE_CHARACTERS, max_size=3),
)
@example(reference="\n\\n", prefix="\\", suffix="")
@example(reference="\n\\n\u00e9", prefix="\\", suffix="")
def test_precalibration_stage_scan_counts_escaped_text_anywhere(
    reference, prefix, suffix
):
    """A reference's JSON-escaped text counts in any surrounding text.

    Decoding alone can miss it: after a stray backslash the text's first
    escape pairs with that backslash, so no decoded layer holds the reference
    (round 6's example, a newline, a backslash and ``n``). ``_embeds_engine_ref``
    matches the text as written, and so does the layered scan, in the text
    ``json.dumps`` writes without ASCII escapes, as ``_embeds_engine_ref``
    matches it. That holds within the 32-layer bound: a parameter with any
    string still escaped after it gives ``None``, whatever was found. These
    strings, at most 14 characters, cannot reach it.
    """
    value = prefix + json.dumps(reference, ensure_ascii=False)[1:-1] + suffix
    assert compose_module._embeds_engine_ref(value, {reference})
    found = compose_module._embedded_engine_refs({"note": value}, {reference})
    assert found is not None
    assert reference in found


def _escape_layers(text, layers):
    """``text`` behind ``layers`` layers of JSON escapes, one decode each.

    Each quote becomes ``\\u0022``, then each backslash ``\\u005c`` once per
    further layer, so the length grows by five characters a layer, not twofold
    as with ``json.dumps``.
    """
    text = text.replace('"', "\\u0022")
    for _ in range(layers - 1):
        text = text.replace("\\", "\\u005c")
    return text


@pytest.mark.parametrize(
    ("note", "outcome"),
    [
        # The inventory at the deepest layer the scan decodes, and one deeper.
        (_escape_layers('{"engine_refs":{}}', 32), "inventory"),
        (_escape_layers('{"engine_refs":{}}', 33), "inventory"),
        # Harmless text: decoded through 32 layers, or refused one deeper.
        (_escape_layers('{"note":"x"}', 32), None),
        (_escape_layers('{"note":"x"}', 33), "inventory"),
        # A chain of escaped backslashes 40,000 layers deep, refused quickly.
        ("\\u005c" + "u005c" * 40_000 + "u0022x", "inventory"),
    ],
    ids=["key_32", "key_33", "harmless_32", "harmless_33", "chain_40000"],
)
def test_precalibration_ancestry_inventory_scan_stops_at_32_layers(note, outcome):
    """The scan decodes at most 32 layers; a string escaped deeper refuses.

    Each layer is a pass over the string, so a crafted chain one layer per
    escaped backslash cost a pass per layer before the bound: quadratic.
    """
    spec = _shape_case()
    _shape_rows(spec)["mid"]["params"] = {"note": note}
    started = time.perf_counter()
    assert _shape_outcome(
        compose_module.validate_transport_calibration_ancestry, spec
    ) == (
        None
        if outcome is None
        else (
            ValueError,
            "Transport calibration ancestry: node 'mid' carries prepared "
            "engine_refs map.",
        )
    )
    assert time.perf_counter() - started < 1


@pytest.mark.parametrize("node", ["probe", "mid"])
def test_precalibration_ancestry_selected_prepared_map_refuses(node):
    """Selected resource data carrying the inventory is the inventory."""
    spec = _shape_case()
    spec["resources"]["probe_inventory"] = {"engine_refs": {"u": "u-ref"}}
    _shape_rows(spec)[node].setdefault("params", {})["inventory"] = {
        "resource": "probe_inventory",
        "encoding": "json",
    }
    assert _shape_outcome(
        compose_module.validate_transport_calibration_ancestry, spec
    ) == (
        ValueError,
        f"Transport calibration ancestry: node {node!r} carries prepared "
        "engine_refs map.",
    )


@pytest.fixture(scope="module")
def precalibration_entitlement(tmp_path_factory):
    """The entitlement fixture, with real extension rows and variant fits."""
    from microcosm.build.transport import registry as registry_module

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(registry_module, "register_gate_kernel", lambda *args: None)
        return make_entitlement_fixture(
            tmp_path_factory.mktemp("precalibration-entitlement")
        )


def _entitlement_stage_spec(fixture, *, listed=True):
    """The fixture's spec, listing its person main-benefit binding as a stage."""
    spec = copy.deepcopy(fixture.spec)
    resources = spec["resources"]
    binding = next(
        row
        for row in resources["axiom_rules_bindings"]["bindings"]
        if row["id"] == "main_benefit"
    )
    resources["precalibration_stages"] = {
        "schema_version": 1,
        "stages": [
            {
                "id": "stage.main",
                "rules_binding": binding["id"],
                "rulespec_path": binding["rulespec_path"],
                "outputs": [
                    {
                        "entity": binding["entity"],
                        "column": variable,
                        "purpose": "calibration_measure",
                    }
                    for variable in binding["variables"]
                ],
                "citation": "Synthetic test stage.",
            }
        ]
        if listed
        else [],
    }
    return spec


def _entitlement_variant_targets(spec):
    """V1's target compiler: an extension row feeding the V1 variant fit."""
    scenario = next(
        row for row in spec["resources"]["scenarios"]["scenarios"] if row["id"] == "V1"
    )
    return next(row for row in scenario["nodes"] if row["id"] == "nz.v.V1.targets")


def _select_holdout(row, encoding):
    key = "references" if encoding == "json" else "references_sha256"
    row["params"][key] = {"resource": "holdout_references", "encoding": encoding}
    row["inputs"] = [{"entity": "person", "columns": ["receives_main"]}]


@pytest.mark.parametrize("encoding", ["json", "sha256"])
def test_precalibration_ancestry_extension_holdout_selection_refuses(
    precalibration_entitlement, encoding
):
    """An entitlement row's selection is read from its declaration.

    Its parameters are resolved before the composed graph exists, so the
    resolved node carries no marker of the hold-out resource it selected.
    """
    fixture = precalibration_entitlement
    spec = _entitlement_stage_spec(fixture)
    compose_transport_graph(spec, fixture.config)
    _select_holdout(_entitlement_variant_targets(spec), encoding)
    with pytest.raises(
        ValueError,
        match="node 'nz.v.V1.targets' carries holdout_references selection",
    ):
        compose_transport_graph(spec, fixture.config)


def test_precalibration_ancestry_extension_selection_without_stages_is_todays(
    precalibration_entitlement,
):
    fixture = precalibration_entitlement
    spec = _entitlement_stage_spec(fixture, listed=False)
    _select_holdout(_entitlement_variant_targets(spec), "json")
    graph = compose_transport_graph(spec, fixture.config)
    assert _precalibration_outcome(
        compose_module.validate_transport_calibration_ancestry,
        spec,
        graph=graph,
        config=fixture.config,
    ) == _precalibration_outcome(
        _precalibration_reference_guard, spec, graph=graph, config=fixture.config
    )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            {"params": {"references": {"resource": ["holdout_references"]}}},
            "node 'nz.v.V1.targets' has malformed params.",
        ),
        ({"id": "nz.targets.compile"}, "node 'nz.targets.compile' is declared twice."),
        ({"id": None}, "node id None is not text."),
    ],
    ids=["malformed_params", "repeated_id", "non_text_id"],
)
def test_precalibration_ancestry_malformed_extension_row_refuses(
    precalibration_entitlement, change, message
):
    """A standalone check refuses an extension row it cannot read."""
    fixture = precalibration_entitlement
    spec = _entitlement_stage_spec(fixture)
    graph = compose_transport_graph(spec, fixture.config)
    _entitlement_variant_targets(spec).update(change)
    assert _precalibration_outcome(
        compose_module.validate_transport_calibration_ancestry,
        spec,
        graph=graph,
        config=fixture.config,
    ) == (ValueError, f"Transport calibration ancestry: {message}")
