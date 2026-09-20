"""State host wiring and genuine small atomic-prefix composition, not PUF issuance."""

import inspect
import itertools
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_graph_current_survey_state import (
    _FixtureKeepAll,
    _FixtureLegacyState,
    _with_households,
    genuine_atomic,  # noqa: F401 -- imported module-scoped genuine fixture
    parent,
)

from microcosm.build.us_runtime import graph_us_survey_enrichment as host
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    NumericScope,
    Owned,
    Slice,
    StructuralDelta,
    compile_graph,
    run_graph,
)

binding = host.state_graph


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}, np.bool_(True)])
def test_state_option_is_strict_before_parent_access(value):
    with pytest.raises(ValueError, match="CANONICAL_STATE_OPTION"):
        host.Boundary(object(), groups=(), n_estimators=2, canonical_state_input=value)


@pytest.mark.parametrize("options", itertools.product((False, True), repeat=5))
@pytest.mark.parametrize(
    "entry,callee",
    [("run_us_survey_enrichment", "_construct"), ("_construct", "Boundary")],
)
def test_all_independent_options_are_forwarded(options, entry, callee, monkeypatch):
    names = (
        "health_completion",
        "demographic_inputs",
        "race_hispanic_inputs",
        "full_original_amount_donors",
        "canonical_state_input",
    )
    expected = dict(zip(names, options, strict=True))
    seen = []

    def stop(run, **kwargs):
        seen.append((run, kwargs))
        raise RuntimeError("stop before authority")

    monkeypatch.setattr(host, callee, stop)
    run = object()
    with pytest.raises(RuntimeError, match="stop before authority"):
        getattr(host, entry)(
            run, groups=("workers_compensation",), n_estimators=2, **expected
        )
    assert seen[0][0] is run
    assert all(seen[0][1][name] is value for name, value in expected.items())
    assert seen[0][1]["groups"] == ("workers_compensation",)


def test_state_default_off_and_disabled_nodes():
    for entry in (host.Boundary, host._construct, host.run_us_survey_enrichment):
        assert (
            inspect.signature(entry).parameters["canonical_state_input"].default
            is False
        )
    assert (
        host.Boundary._state_nodes(SimpleNamespace(canonical_state_input=False)) == ()
    )


@pytest.mark.parametrize("options", itertools.product((False, True), repeat=4))
def test_state_terminal_follows_every_prior_option(options):
    spm, immigration, sex, race = options
    edge = host._state_after_edge(*options)
    expected = (
        host.race_graph.ATTACH_NODE
        if race
        else host.sex_graph.ATTACH_NODE
        if sex
        else host.immigration_graph.ATTACH_NODE
        if immigration
        else host.spm_graph.ATTACH_NODE
        if spm
        else host.hours_graph.ATTACH_NODE
    )
    assert edge.producer == expected and edge.artifact == "attachment"


def test_state_mapping_type_and_helper_changes_are_sealed(monkeypatch):
    before = host._live()

    class ChangedMembership(dict):
        def __contains__(self, item):
            return True

    with monkeypatch.context() as patch:
        patch.setattr(
            binding,
            "US_STATE_FIPS_TO_POSTAL",
            ChangedMembership(binding.US_STATE_FIPS_TO_POSTAL),
        )
        assert host._live() != before
    monkeypatch.setattr(binding, "bind_state_fips", lambda table: None)
    assert host._live() != before


class _PriorReader(KernelBase):
    ref = "fixture.prior_state_reader@1"
    capabilities = Capabilities(Determinism.DETERMINISTIC)

    def run(self, context):
        return KernelResult(artifacts={"attachment": b"prior"})


class _Version(KernelBase):
    """Exercise the actual host pure result without pretending to issue a PUF owner."""

    ref = host.CurrentSurveyStateVersionKernel.ref
    capabilities = host.CurrentSurveyStateVersionKernel.capabilities

    def run(self, context):
        return host._state_version_result(
            context.node,
            context.tables["person"][["person_id", "person_support_clone_index"]],
            context.artifacts,
        )


def _artifacts(node, manifest, store):
    result = {}
    for edge in node.artifact_inputs:
        record = manifest.node(edge.producer)
        key = record.opaque_artifacts[edge.artifact]
        result[edge.name] = ArtifactValue(
            payload=store.load_bytes(key),
            type=edge.type,
            key=key,
            producer_key=record.key,
            numerics=NumericScope(Numeric.BITWISE),
        )
    return result


@pytest.mark.parametrize("incumbent", [None, "same", "different", "nullable"])
def test_genuine_prefix_terminal_version_cold_required_and_reconstruction(
    genuine_atomic,  # noqa: F811 -- pytest injects the imported fixture
    incumbent,
):
    run = genuine_atomic.run
    before = run.geography_population
    stamp = parent.reconstruction._population_stamp(before)
    receiving = before.frame
    version = before.version
    registry = KernelRegistry()
    for kernel in run.kernels.as_mapping().values():
        registry.register(kernel)
    added = []
    if incumbent is not None:
        table = receiving.table("household").copy(deep=True)
        dtype = "Int64" if incumbent == "nullable" else "int64"
        table["state_fips"] = pd.array([99] * len(table), dtype=dtype)
        receiving = _with_households(receiving, table)
        legacy = Node(
            "fixture.legacy_state",
            _FixtureLegacyState.ref,
            population=version,
            inputs=(Slice("household", binding.INPUTS),),
            outputs=(Owned("household", "state_fips", dtype),),
            params={"kind": incumbent},
        )
        carrier = Node(
            "fixture.state_carrier",
            _FixtureKeepAll.ref,
            structural=StructuralDelta.FILTER,
            base=version,
            inputs=(Slice("person", ("person_support_clone_index",)),),
        )
        added.extend((legacy, carrier))
        registry.register(_FixtureLegacyState())
        registry.register(_FixtureKeepAll())
        version = carrier.id
    # This reader deliberately sees the carried original, including a wrong value.
    # Its declaration is unchanged when the final canonical rewrite is added.
    reader = Node(
        "fixture.prior_state_reader",
        _PriorReader.ref,
        population=version,
        inputs=(
            Slice(
                "household", (*binding.INPUTS, *(("state_fips",) if incumbent else ()))
            ),
        ),
        artifact_outputs=(ArtifactOutput("attachment", host.ATTACHMENT_TYPE),),
    )
    added.append(reader)
    registry.register(_PriorReader())
    edge = ArtifactInput(
        "previous_attachment", reader.id, "attachment", host.ATTACHMENT_TYPE
    )
    nodes = host._state_nodes(receiving, receiving_version=version, after=edge)
    if incumbent:
        unsafe = binding.state_binding_node(receiving, population=version, after=edge)
        with pytest.raises(ValueError, match="(?i)cycle"):
            compile_graph(
                replace(
                    run.compiled.graph,
                    nodes=(*run.compiled.graph.nodes, *added, unsafe),
                )
            )
    graph = compile_graph(
        replace(run.compiled.graph, nodes=(*run.compiled.graph.nodes, *added, *nodes))
    )
    registry.register(_Version())
    registry.register(binding.CurrentSurveyStateKernel())
    snapshots = {}
    manifest = run_graph(
        graph,
        sources=run.sources,
        store=run.store,
        kernels=registry,
        _population_observer=lambda name, value: snapshots.setdefault(name, value),
    )
    incoming = snapshots[reader.id]
    for node in nodes:
        artifacts = _artifacts(node, manifest, run.store)
        persisted = {
            a.name: run.store.load_bytes(
                manifest.node(node.id).opaque_artifacts[a.name]
            )
            for a in node.artifact_outputs
        }
        expected = host._state_expected_population(incoming, node, artifacts, persisted)
        parent.reconstruction.replay.same_replayed_population(
            expected, snapshots[node.id]
        )
        with pytest.raises(ValueError, match="STATE_RESULT_ARTIFACT"):
            host._state_expected_population(
                incoming, node, artifacts, {**persisted, "unclaimed": b"bad"}
            )
        incoming = expected
    final = snapshots[binding.NODE]
    base = snapshots[reader.id]
    for entity in final.frame.entities:
        cols = [
            c
            for c in base.frame.table(entity)
            if (entity, c) != ("household", "state_fips")
        ]
        pd.testing.assert_frame_equal(
            final.frame.table(entity)[cols],
            base.frame.table(entity)[cols],
            check_exact=True,
        )
    pd.testing.assert_series_equal(
        final.frame.table("household").set_index("household_id").state_fips,
        binding.bind_state_fips(base.frame.table("household")),
        check_exact=True,
    )
    np.testing.assert_array_equal(
        final.frame.weights_for("household").values,
        base.frame.weights_for("household").values,
    )
    assert len(final.mass_ledger) == len(base.mass_ledger) + 1
    assert final.mass_ledger[:-1] == base.mass_ledger
    assert graph.graph.node(reader.id) == reader
    warm = run_graph(
        graph, sources=run.sources, store=run.store, kernels=registry, resume="require"
    )
    assert all(record.hit for record in warm.nodes.values())
    assert warm.key == manifest.key
    parent.survey._same_frame(final.frame, warm.population(host.STATE_VERSION_NODE))
    assert parent.reconstruction._population_stamp(before) == stamp


def test_state_wrapper_checks_retained_projection_and_final_pure_fence():
    """Orchestration only; this stand-in is never registered as a live owner."""
    from test_us_graph_current_survey_state import _table

    table = _table()
    table.loc[1, ["assigned_state_fips", "survey_observed_state"]] = "36"
    receiving = SimpleNamespace(table=lambda entity: table)
    nodes = host._state_nodes(
        receiving,
        receiving_version="prior",
        after=ArtifactInput("previous", "previous", "attachment", host.ATTACHMENT_TYPE),
    )
    calls = []
    boundary = SimpleNamespace(
        context=lambda context: calls.append("context"),
        pure=lambda: calls.append("pure"),
        run=SimpleNamespace(population=SimpleNamespace(frame=receiving)),
    )
    node = nodes[-1]
    gate = binding.canonical_json(
        {"outcome": "pass", "scope": "atomic_geography_mapping_integrity"}
    )
    artifacts = {
        edge.name: ArtifactValue(
            payload=gate if edge.name == "geography_validation" else b"version",
            type=edge.type,
            key="a" * 64,
            producer_key="b" * 64,
            numerics=NumericScope(Numeric.BITWISE),
        )
        for edge in node.artifact_inputs
    }
    context = SimpleNamespace(
        node=node,
        tables={"household": table},
        artifacts=artifacts,
        sources={},
        params=node.params,
    )
    result = host.CurrentSurveyCanonicalStateKernel(boundary).run(context)
    assert calls == ["context", "pure"] and result.columns[
        "household", "state_fips"
    ].tolist() == [6, 36]
    context.tables["household"] = table.copy(deep=True)
    context.tables["household"].loc[0, "assigned_state_fips"] = "01"
    with pytest.raises(ValueError, match="STATE_RECEIVING_HOUSEHOLDS"):
        host.CurrentSurveyCanonicalStateKernel(boundary).run(context)


def test_state_version_wrapper_preserves_structural_context(genuine_atomic):  # noqa: F811
    """Real prefix Frame; wrapper ordering only, no stand-in authority issuance."""
    frame = genuine_atomic.run.geography_population.frame
    edge = ArtifactInput("prior", "prior", "attachment", host.ATTACHMENT_TYPE)
    node = host._state_version_node(receiving_version="prior", after=edge)
    columns = [*host._structural_columns(frame, "person"), "person_support_clone_index"]
    calls = []
    boundary = SimpleNamespace(
        context=lambda context: calls.append("context"),
        pure=lambda: calls.append("pure"),
        run=SimpleNamespace(population=SimpleNamespace(frame=frame)),
    )
    context = SimpleNamespace(
        node=node,
        tables={"person": frame.person[columns]},
        sources={},
        artifacts={
            "prior": ArtifactValue(
                payload=b"prior",
                type=edge.type,
                key="a" * 64,
                producer_key="b" * 64,
                numerics=NumericScope(Numeric.BITWISE),
            )
        },
    )
    result = host.CurrentSurveyStateVersionKernel(boundary).run(context)
    assert result.keep.all() and len(result.keep) == len(frame.person)
    assert calls == ["context", "pure"]
    context.tables["person"] = frame.person[columns].copy(deep=True)
    context.tables["person"].loc[0, "person_household_id"] += 1
    with pytest.raises(ValueError, match="STATE_VERSION_PERSONS"):
        host.CurrentSurveyStateVersionKernel(boundary).run(context)


def test_combined_child_then_state_preserves_later_writes_and_both_ledgers(
    tmp_path, monkeypatch
):
    from test_us_native_child_support import (
        _full_original_child_source_and_donor,
        _run_child_source_graph,
    )

    donor = _full_original_child_source_and_donor(tmp_path, monkeypatch)
    graph_dir = tmp_path / "combined-graph"
    graph_dir.mkdir()
    _run_child_source_graph(graph_dir, monkeypatch, donor, -999.0, canonical_state=True)


@pytest.mark.parametrize("options", itertools.product((False, True), repeat=4))
@pytest.mark.parametrize("child", [False, True])
def test_host_state_nodes_compose_after_optional_child(options, child):
    from test_us_graph_current_survey_state import _table, _view

    spm, immigration, sex, race = options
    qualified = SimpleNamespace(
        groups=(
            SimpleNamespace(
                spec=SimpleNamespace(
                    key="child_support" if child else "workers_compensation"
                )
            ),
        )
    )
    nodes = host.Boundary._state_nodes(
        SimpleNamespace(
            canonical_state_input=True,
            qualified=qualified,
            run=SimpleNamespace(population=SimpleNamespace(frame=_view(_table()))),
            spm=object() if spm else None,
            immigration_transfer=object() if immigration else None,
            sex=object() if sex else None,
            race=object() if race else None,
        )
    )
    assert nodes[0].base == (
        host.CHILD_VERSION_NODE if child else host.parent.attach.FILTER_NODE
    )
    assert nodes[0].artifact_inputs[0] == (
        ArtifactInput(
            "child_support_attachment",
            host.CHILD_ATTACH_NODE,
            "attachment",
            host.ATTACHMENT_TYPE,
        )
        if child
        else host._state_after_edge(*options)
    )
    assert nodes[1].population == nodes[0].id


def test_state_keep_all_reconstruction_explicitly_refuses_mutated_group_axis():
    """A normal Frame refuses orphans; emulate mutation after its construction."""
    from test_us_native_child_support import _child_family

    from microcosm.graph import population as population_ops

    _, receiving = _child_family()
    edge = ArtifactInput("prior", "prior", "attachment", host.ATTACHMENT_TYPE)
    node = host._state_version_node(receiving_version="prior", after=edge)
    incoming = population_ops.Population.from_frame(receiving, node.base)
    artifacts = {
        "prior": ArtifactValue(
            payload=b"prior",
            type=edge.type,
            key="a" * 64,
            producer_key="b" * 64,
            numerics=NumericScope(Numeric.BITWISE),
        )
    }
    person = receiving.person[["person_id", "person_support_clone_index"]]
    result = host._state_version_result(node, person, artifacts)
    membership = receiving.schema.membership_column("household")
    first, second = receiving.person[membership].drop_duplicates().iloc[:2]
    receiving.person.loc[receiving.person[membership].eq(first), membership] = second
    with pytest.raises(ValueError, match="STATE_VERSION_AXIS_CHANGED"):
        host._state_expected_population(incoming, node, artifacts, result.artifacts)
