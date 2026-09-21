"""Bounded host wiring and terminal state tests; no invented owner issuance."""

import inspect
import itertools
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_puf55_original_placement import SEEDS, fixture, routes

from microcosm.build.us_runtime import graph_puf55_original_host as original
from microcosm.build.us_runtime import graph_us_survey_enrichment as host
from microcosm.graph import (
    ArtifactOutput,
    ArtifactType,
    Graph,
    Node,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
)
from microcosm.graph import population as populations

fragment, values = original.fragment, original.values


@pytest.mark.parametrize("seed", [False, True, 1.5, "73", {}, np.int64(73)])
def test_application_seed_option_refuses_before_parent_access(seed):
    with pytest.raises(ValueError, match="ORIGINAL_APPLICATION_SEED"):
        host.Boundary(
            object(), groups=(), n_estimators=2, original_application_seed=seed
        )


@pytest.mark.parametrize(
    "entry,callee",
    [("run_us_survey_enrichment", "_construct"), ("_construct", "Boundary")],
)
def test_original_seed_is_optional_and_forwarded_with_other_options(
    entry, callee, monkeypatch
):
    assert (
        inspect.signature(getattr(host, entry))
        .parameters["original_application_seed"]
        .default
        is None
    )
    seen = []

    def stop(run, **kwargs):
        seen.append(kwargs)
        raise RuntimeError("stop before source admission")

    monkeypatch.setattr(host, callee, stop)
    with pytest.raises(RuntimeError, match="stop before source admission"):
        getattr(host, entry)(
            object(),
            groups=("veterans_benefits",),
            n_estimators=2,
            full_original_amount_donors=True,
            canonical_state_input=True,
            original_application_seed=73,
        )
    assert seen[0]["original_application_seed"] == 73
    assert (
        seen[0]["full_original_amount_donors"]
        is seen[0]["canonical_state_input"]
        is True
    )


@pytest.mark.parametrize(
    "state,canonical,race,sex,immigration,spm",
    itertools.product((False, True), repeat=6),
)
def test_terminal_uses_actual_selected_node_version(
    state, canonical, race, sex, immigration, spm, monkeypatch
):
    # Metadata-only host descriptor, not a source/PUF/enrichment owner.
    qualified = object()
    monkeypatch.setattr(
        host, "_canonical_enabled", lambda q: q is qualified and canonical
    )
    monkeypatch.setattr(
        host, "_canonical_ids", lambda q: ("canonical.version", "canonical.attach")
    )
    ids = (
        host.state_graph.NODE,
        "canonical.attach",
        host.race_graph.ATTACH_NODE,
        host.sex_graph.ATTACH_NODE,
        host.immigration_graph.ATTACH_NODE,
        host.spm_graph.ATTACH_NODE,
        host.hours_graph.ATTACH_NODE,
    )
    nodes = tuple(
        Node(
            n,
            "fixture.terminal@1",
            population="actual.version." + str(i),
            artifact_outputs=(ArtifactOutput("attachment", ArtifactType("test", 1)),),
        )
        for i, n in enumerate(ids)
    )
    b = SimpleNamespace(
        canonical_state_input=state,
        qualified=qualified,
        race=object() if race else None,
        sex=object() if sex else None,
        immigration_transfer=object() if immigration else None,
        spm=object() if spm else None,
        nodes=nodes,
    )
    selected = host._receiving_terminal(b)
    expected = next(
        n
        for flag, n in zip(
            (state, canonical, race, sex, immigration, spm, True), nodes, strict=True
        )
        if flag
    )
    assert selected is expected
    assert host._original_after(selected).producer == expected.id
    assert selected.population.startswith("actual.version.")


def _descriptive_binding():
    """Exercise retained-value mechanics, explicitly bypassing no source issuer.

    This constructs an internal descriptive fixture directly, never calls the
    production admission constructor, and cannot satisfy the real host registry.
    """
    fixed, inputs, _ = fixture()
    declarations = routes(fixed)
    terminal = Node(
        "invented.complete.terminal",
        "fixture.terminal@1",
        population=inputs.receiving.version,
        artifact_outputs=(ArtifactOutput("attachment", ArtifactType("test", 1)),),
    )
    after = host._original_after(terminal)
    binding = object.__new__(original.Binding)
    binding.terminal, binding.after = terminal, after
    binding.seeds = dict(SEEDS)
    binding.routes, binding.fixed = declarations, fixed
    binding.fixed_stamp = original.fixed_graph.values.fixed_input_stamp(fixed)
    binding.template = replace(inputs, receiving=inputs.arm_one)
    binding.template_stamp = values._stamp(binding.template)
    binding.placement_nodes = fragment.original_placement_nodes(
        fixed,
        binding.template,
        declarations,
        after=after,
        receiving_version=terminal.population,
        **SEEDS,
    )
    binding.source_nodes = ()
    binding.apply_nodes = original.application.original_application_nodes(
        fixed, declarations, population=terminal.population, **SEEDS
    )
    binding.nodes = (*binding.apply_nodes, *binding.placement_nodes)
    binding.node_stamp = tuple(
        original.codec.encode_json(original._node_payload(n)) for n in binding.nodes
    )
    binding.config = (
        original.configuration(),
        tuple(sorted(SEEDS.items())),
        declarations,
        terminal,
        after,
    )
    parent = SimpleNamespace(routes=declarations, seed=31)
    b = SimpleNamespace(
        parent_entry=(None, None, SimpleNamespace(boundary=parent)),
        declaration=(terminal, *binding.nodes),
    )
    b.pure = binding.pure  # Only descriptive mechanics, explicitly no live admission.
    binding.host = b
    binding.receiving = binding.receiving_stamp = binding.kept = binding.kept_stamp = (
        None
    )
    return binding, inputs


def test_terminal_observer_captures_complete_late_values_and_refuses_replacement():
    b, inputs = _descriptive_binding()
    inputs.receiving.frame.person["invented_late_leaf"] = np.arange(20, dtype="float64")
    # Rebuild exact owners for the deliberately added late writer.
    owners = dict(inputs.receiving.owners)
    owners["person", "invented_late_leaf"] = b.terminal.id
    late = populations.Population.from_frame(
        inputs.receiving.frame, inputs.receiving.version, owners
    )
    with pytest.raises(ValueError, match="TERMINAL_NOT_OBSERVED"):
        b.inputs()
    b.observe_terminal("unrelated", late)
    assert b.receiving is None
    b.observe_terminal(b.terminal.id, late)
    pd.testing.assert_series_equal(
        b.kept.frame.person.invented_late_leaf, late.frame.person.invented_late_leaf
    )
    assert b.inputs().receiving is late
    with pytest.raises(ValueError, match="TERMINAL_OBSERVER"):
        b.observe_terminal(b.terminal.id, late)
    late.frame.person.loc[0, "invented_late_leaf"] += 1
    with pytest.raises(ValueError, match="TERMINAL_CHANGED"):
        b.pure()


def test_independent_reconstruction_requires_exact_completed_terminal():
    b, inputs = _descriptive_binding()
    b.observe_terminal(b.terminal.id, inputs.receiving)
    expected = b.reconstruct(b.placement_nodes[0], inputs.receiving, {}, {})
    original.physical.replay.same_replayed_population(expected, b.kept)
    earlier = replace(inputs.receiving, version="invented.earlier")
    with pytest.raises((ValueError, AssertionError)):
        b.reconstruct(b.placement_nodes[0], earlier, {}, {})


def test_whole55_extension_declarations_compile_after_late_terminal():
    """Actual apply/placement declarations; metadata suppliers are not models."""
    b, inputs = _descriptive_binding()
    frame = inputs.receiving.frame
    # Like the actual host CREATE, own no entity ID or membership column: the
    # executor projects those structurally and the compiler refuses a Slice that
    # names them (the first genuine Stage B run failed on exactly that).
    structural = {(e, frame.schema.entity_id_column(e)) for e in frame.entities} | {
        ("person", frame.schema.membership_column(g))
        for g in frame.schema.group_entities
    }
    create = Node(
        inputs.receiving.version,
        "fixture.create@1",
        structural=StructuralDelta.CREATE,
        sources=("invented",),
        mass="free",
        outputs=tuple(
            Owned(e, c, populations.token_for_dtype(frame.table(e)[c].dtype))
            for e in frame.entities
            for c in frame.table(e)
            if (e, c) not in structural
        ),
    )
    assert not any(
        (s.entity, c) in structural
        for n in b.placement_nodes
        for s in n.inputs
        for c in s.columns
    )
    # External prefix producers are declaration-only placeholders, explicitly no
    # source authority/fitted model/cache-hit claim. Every actual110 apply node
    # and its fixed/matrix/training/55-step edges are compiled unchanged.
    extension = (*b.apply_nodes, *b.placement_nodes)
    local = {n.id for n in extension} | {create.id, b.terminal.id}
    suppliers = {}
    for node in extension:
        for edge in node.artifact_inputs:
            if edge.producer not in local:
                suppliers.setdefault(edge.producer, {})[edge.artifact] = edge.type
    prefix = tuple(
        Node(
            name,
            "fixture.metadata@1",
            population=create.id,
            artifact_outputs=tuple(ArtifactOutput(n, t) for n, t in outputs.items()),
        )
        for name, outputs in suppliers.items()
    )
    compiled = compile_graph(
        Graph(
            "invented-declaration-only",
            (SourceRef("invented", "frame-store"),),
            (create, b.terminal, *prefix, *extension),
        )
    )
    assert len(b.apply_nodes) == 110
    assert compiled.order.index(b.terminal.id) < compiled.order.index(
        fragment.KEEP_NODE
    )
    assert compiled.versions[fragment.ATTACH_NODE] == fragment.KEEP_NODE
    assert all(
        compiled.order.index(n.id) < compiled.order.index(fragment.ATTACH_NODE)
        for n in b.apply_nodes
    )


def test_real_host_rejects_unissued_parent_even_when_original_option_enabled():
    with pytest.raises(ValueError, match="UNISSUED_PUF55_RUN"):
        host.Boundary(object(), groups=(), n_estimators=2, original_application_seed=73)


def test_original_source_and_callable_changes_are_in_live_host_seal(monkeypatch):
    before = host._live()
    monkeypatch.setattr(original.values, "SINGLETON_OUTPUTS", ())
    assert host._live() != before


def test_new_host_module_has_no_provenance_owner_exemption():
    import test_us_spine_blindness as scanner

    name = "graph_puf55_original_host.py"
    assert name not in scanner._SOURCE_SPINE_PROVENANCE_OWNERS
    assert (
        scanner._non_owner_source_spine_accesses(
            name, (scanner._US_RUNTIME / name).read_text()
        )
        == ()
    )
    assert scanner._non_owner_source_spine_accesses(
        name, 'x = table["person_spine_source_id"]'
    )


@pytest.mark.parametrize("which", [0, 1])
def test_placement_context_accepts_actual_executor_projection(which):
    from microcosm.graph.executor import _project_context

    b, inputs = _descriptive_binding()
    b.observe_terminal(b.terminal.id, inputs.receiving)
    node = b.placement_nodes[which]
    b.host.context = lambda context: b.pure()  # Descriptive fixture, no authority.
    b.host.keys = ((node.id, "f" * 64),)
    b.host.paths = ()
    projected = _project_context(
        node,
        b.receiving if which == 0 else b.kept,
        key="f" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    b.context(projected)


@pytest.mark.parametrize(
    "defect", ["table", "dtype", "axis", "weights", "strata", "params"]
)
def test_placement_context_refuses_altered_executor_projection(defect):
    b, inputs = _descriptive_binding()
    b.observe_terminal(b.terminal.id, inputs.receiving)
    node = b.placement_nodes[0]
    b.host.context = lambda context: b.pure()  # Only the projected-value check.
    b.host.keys = ((node.id, "f" * 64),)
    b.host.paths = ()
    projected = original.executor._project_context(
        node,
        b.receiving,
        key="f" * 64,
        sources={},
        tolerances={},
        numerics={},
    )
    if defect in ("table", "dtype", "axis"):
        table = projected.tables["person"].copy(deep=True)
        if defect == "table":
            table.loc[0, "person_id"] += 123
        elif defect == "dtype":
            table["person_id"] = table.person_id.astype("float64")
        else:
            table = table.iloc[::-1]
        projected = replace(projected, tables={**projected.tables, "person": table})
    elif defect == "weights":
        entity = next(iter(projected.weights))
        weight = projected.weights[entity]
        projected = replace(
            projected,
            weights={
                **projected.weights,
                entity: replace(weight, values=weight.values + 1),
            },
        )
    elif defect == "strata":
        strata = projected.strata.copy(deep=True)
        strata.iloc[0] = "invented_changed_stratum"
        projected = replace(projected, strata=strata)
    else:
        projected = replace(projected, params={})
    with pytest.raises(ValueError, match="PLACEMENT_CONTEXT_VALUES"):
        b.context(projected)


@pytest.mark.parametrize("defect", [None, "column", "artifact", "receipt"])
def test_placement_kernel_brackets_host_callback_with_result_seal(monkeypatch, defect):
    # This isolates the callback fence, not model decoding or host admission.
    # Actual strict full55 envelopes have separate scoped tests.
    from microcosm.graph import KernelResult

    result = KernelResult(
        columns={
            ("person", "invented"): pd.Series(
                [1.0], index=pd.Index([7], name="person_id")
            )
        },
        artifacts={"placement": b"invented"},
        receipt={"invented": 1},
    )
    calls = []

    def context_fence(context):
        calls.append(context)
        if len(calls) == 2:
            if defect == "column":
                result.columns["person", "invented"].iloc[0] += 1
            elif defect == "artifact":
                result.artifacts["placement"] = b"changed"
            elif defect == "receipt":
                result.receipt["invented"] = 2

    b = SimpleNamespace(
        context=context_fence,
        fixed=None,
        inputs=lambda: None,
        routes=(),
        after=None,
        seeds=SEEDS,
    )
    monkeypatch.setattr(fragment, "original_placement_result", lambda *a, **kw: result)
    context = SimpleNamespace(node=None, artifacts={})
    kernel = original.PlacementKernel(b)
    if defect is None:
        assert kernel.run(context) is result
    else:
        with pytest.raises(ValueError, match="FINAL_PLACEMENT_RESULT"):
            kernel.run(context)
    assert len(calls) == 2
