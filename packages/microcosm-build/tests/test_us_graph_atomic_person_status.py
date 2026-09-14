"""Optional status on the actual common host, over finite invented sources."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest
from test_us_current_survey_person_status_source import _source_arguments
from test_us_graph_atomic_survey_population import _support_payload

from microcosm.build.us_runtime import graph_atomic_survey_financial as runner
from microcosm.build.us_runtime import graph_current_survey_person_status as graph


@pytest.fixture(scope="module")
def status_host(tmp_path_factory):
    root = tmp_path_factory.mktemp("atomic-person-status")
    with pytest.MonkeyPatch.context() as patch:
        arguments = _source_arguments(root, patch)
        payload, source_ids = _support_payload()
        support = root / "invented-block-support.npz"
        support.write_bytes(payload)
        config = runner.reconstruction.AtomicSurveyReconstruction(
            support_path=str(support),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        call = {
            **arguments,
            "store_root": root / "store",
            "geography_config": config,
            "demographic_conditioning": True,
            "n_estimators": 2,
            "return_values": True,
        }
        base = runner.run_atomic_survey_financial(**call, person_status=False)
        cold = runner.run_atomic_survey_financial(**call, person_status=True)
        warm = runner.run_atomic_survey_financial(
            **call, person_status=True, resume="require"
        )
        yield SimpleNamespace(base=base, cold=cold, warm=warm, call=call)
        for run in (base, cold, warm):
            run.checked_view()


def test_optional_23_node_host_preserves_default_19_and_nine_node_prefix(status_host):
    case = status_host
    assert len(case.base.compiled.order) == 19
    assert len(case.cold.compiled.order) == len(case.warm.compiled.order) == 23
    assert case.cold.manifest.key == case.warm.manifest.key
    assert all(record.hit for record in case.warm.manifest.nodes.values())
    assert not set(graph.BIND_COLUMNS) & set(
        case.base.financial_population.frame.person
    )
    assert runner._run_entry(case.base)[2].person_status_boundary is None
    for run in (case.cold, case.warm):
        boundary = runner._run_entry(run)[2].person_status_boundary
        runner.atomic.same_replayed_population(
            case.base.financial_population,
            boundary.complement(run.financial_population),
        )
        for name in (
            "allocated_population",
            "observed_population",
            "expanded_population",
            "geography_population",
            "clone_population",
        ):
            runner.atomic.same_replayed_population(
                getattr(case.base.prefix, name), getattr(run.prefix, name)
            )
        assert len(run.prefix.compiled.order) == 9
        for node in case.base.prefix.compiled.graph.nodes:
            assert run.compiled.graph.node(node.id) == node
        assert graph.BIND_NODE in run.compiled.predecessors[runner.financial.DONOR_NODE]
        assert (
            graph.BIND_NODE
            in run.compiled.order[
                : run.compiled.order.index(runner.financial.ATTACH_NODE)
            ]
        )
        doc = json.loads(run.checked_view().payload)
        assert doc["person_status"]["node_count"] == 4
        assert doc["person_status"]["source_artifact_required"]
        assert doc["person_status"]["descriptive_only"] and not doc["release_eligible"]
        assert set(graph.BIND_COLUMNS) <= set(doc["owned_columns"])


@pytest.mark.parametrize("column", ["survey_vision_difficulty", "employment_income"])
def test_common_host_refuses_status_only_and_financial_only_drift(status_host, column):
    run = status_host.cold
    people = run.financial_population.frame.person
    # Check the original financial complement remains identical for a status-only
    # mutation; acceptance still fails because the full reconstruction includes it.
    if column == "employment_income":
        column = runner.values.OUTPUTS[0]
    saved = people[column].copy(deep=True)
    complement_columns = [c for c in people if c not in graph.BIND_COLUMNS]
    complement = people.loc[:, complement_columns].copy(deep=True)
    try:
        people.loc[0, column] = (
            True if column in graph.BIND_COLUMNS else float(saved.iloc[0]) + 11
        )
        if column in graph.BIND_COLUMNS:
            pd.testing.assert_frame_equal(
                people.loc[:, complement_columns], complement, check_exact=True
            )
        with pytest.raises(ValueError):
            run.checked_view()
    finally:
        people[column] = saved
    run.checked_view()


@pytest.mark.parametrize("defect", ["copied_run", "status_owner", "qualifier_receipt"])
def test_common_host_retains_live_issued_owner_and_writers(status_host, defect):
    run = status_host.cold
    boundary = runner._run_entry(run)[2].person_status_boundary
    if defect == "copied_run":
        with pytest.raises(ValueError):
            replace(run).checked_view()
        return
    if defect == "status_owner":
        owners = dict(run.financial_population.owners)
        owners["person", graph.BIND_COLUMNS[0]] = "unissued"
        detached = replace(run.financial_population, owners=owners)
        with pytest.raises(ValueError, match="PERSON_STATUS_WRITER"):
            boundary.complement(detached)
        return
    saved = boundary.qualified.receipt
    try:
        object.__setattr__(boundary.qualified, "receipt", b"{}")
        with pytest.raises(ValueError):
            run.checked_view()
    finally:
        object.__setattr__(boundary.qualified, "receipt", saved)
    run.checked_view()


def test_final_status_source_io_cannot_mutate_only_financial_complement(
    status_host, monkeypatch
):
    from pathlib import Path

    run = status_host.cold
    boundary = runner._run_entry(run)[2].person_status_boundary
    column = runner.values.OUTPUTS[0]
    saved = run.financial_population.frame.person[column].copy(deep=True)
    read = Path.read_bytes
    changed = []

    def mutate(path):
        payload = read(path)
        if not changed:
            changed.append(True)
            run.financial_population.frame.person.loc[0, column] += 1
        return payload

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", mutate)
            boundary.requalify()
        assert changed
        with pytest.raises(ValueError):
            runner._pure_run(run, runner._run_entry(run))
    finally:
        run.financial_population.frame.person[column] = saved
    run.checked_view()


@pytest.mark.parametrize("defect", ["declaration", "binding", "artifact_roster"])
def test_ordering_adapter_refuses_changes_before_original_donor(status_host, defect):
    from microcosm.graph import ArtifactValue
    from microcosm.graph.artifact_edges import numeric_scope
    from microcosm.graph.executor import _project_context

    run = status_host.cold
    node = run.compiled.graph.node(runner.financial.DONOR_NODE)
    kernel = run.kernels.get(node.kernel)
    graph_edge = graph.binding_edge()
    assert node == replace(
        kernel.original_node,
        artifact_inputs=(*kernel.original_node.artifact_inputs, graph_edge),
    )
    artifacts = {}
    for edge in node.artifact_inputs:
        record = run.manifest.node(edge.producer)
        artifacts[edge.name] = ArtifactValue(
            run.store.load_bytes(record.opaque_artifacts[edge.artifact]),
            edge.type,
            record.opaque_artifacts[edge.artifact],
            record.key,
            numeric_scope(
                run.kernels.get(
                    run.compiled.graph.node(edge.producer).kernel
                ).capabilities
            ),
        )
    incoming = runner.Population.from_frame(
        run.prefix.preparation.checked_view().frame, runner.survey.CREATE_NODE
    )
    context = _project_context(
        node,
        incoming,
        key=run.manifest.node(node.id).key,
        sources={},
        tolerances={},
        numerics={},
        artifacts=artifacts,
    )
    if defect == "declaration":
        context = replace(
            context, node=replace(node, params={**node.params, "changed": True})
        )
    elif defect == "binding":
        artifacts[graph_edge.name] = replace(
            artifacts[graph_edge.name], payload=b"changed"
        )
        context = replace(context, artifacts=artifacts)
    else:
        artifacts.pop(graph_edge.name)
        context = replace(context, artifacts=artifacts)
    with pytest.raises(ValueError, match="PERSON_STATUS_DONOR_"):
        kernel.run(context)


@pytest.fixture(scope="module", params=[False, True], ids=["property39", "tax42"])
def extended_status_host(tmp_path_factory, request):
    import test_us_graph_atomic_property_financial as fixtures

    with pytest.MonkeyPatch.context() as patch:

        def status_demographics(path, monkeypatch, *, unknown=False, zero=True):
            assert not unknown and zero is False
            return _source_arguments(path, monkeypatch)

        # Compose source constructors before any preparation is issued. The
        # existing property helper retains status columns while updating money.
        patch.setattr(fixtures, "_demographic_arguments", status_demographics)
        generator = fixtures.property_host_source.__wrapped__(tmp_path_factory)
        case = next(generator)
        try:
            pins = runner._property_module().sources.routing.coverage._MEMBER_PINS
            patch.setattr(graph.source.student, "_MEMBER_PINS", pins)
            options = replace(case.options, completion_routing=True)
            call = {
                **case.call,
                "property_income": options,
                "rebase_property_taxes": request.param,
                "person_status": True,
            }
            cold = runner.run_atomic_survey_financial(**call)
            warm = runner.run_atomic_survey_financial(**call, resume="require")
            yield SimpleNamespace(cold=cold, warm=warm, tax=request.param)
            cold.checked_view()
            warm.checked_view()
        finally:
            generator.close()


def test_optional_status_preserved_through_property_and_tax_hosts(extended_status_host):
    case = extended_status_host
    assert (
        len(case.cold.compiled.order)
        == len(case.warm.compiled.order)
        == (42 if case.tax else 39)
    )
    assert case.cold.manifest.key == case.warm.manifest.key
    assert all(record.hit for record in case.warm.manifest.nodes.values())
    for run in (case.cold, case.warm):
        state = runner._run_entry(run)[2]
        boundary = state.person_status_boundary
        columns = list(graph.BIND_COLUMNS)
        pd.testing.assert_frame_equal(
            state.legacy_financial_population.frame.person[columns],
            run.financial_population.frame.person[columns],
            check_exact=True,
        )
        status_parent = (
            state.property_population if case.tax else run.financial_population
        )
        assert boundary.complement(status_parent).version == status_parent.version
        assert all(
            status_parent.owners["person", c] == graph.BIND_NODE for c in columns
        )
        current_owner = (
            runner._tax_module().RECEIVING_NODE if case.tax else graph.BIND_NODE
        )
        assert all(
            run.financial_population.owners["person", c] == current_owner
            for c in columns
        )
        if case.tax:
            assert run.financial_population.version == current_owner
            assert status_parent is not run.financial_population
            # A post-FILTER population must not masquerade as the original
            # writer's version merely because all status values were carried.
            with pytest.raises(ValueError, match="PERSON_STATUS_WRITER"):
                boundary.complement(run.financial_population)
        run.checked_view()
        # The pure check runs both during issuance and after final source I/O.
        # A genuine FILTER owner is accepted only with the retained complete
        # population seal; neither stale ownership nor status-only drift passes.
        populations = (
            (status_parent, run.financial_population)
            if case.tax
            else (run.financial_population,)
        )
        column = "survey_status_original_age"
        for population in populations:
            people = population.frame.person
            saved_values = people[column].copy(deep=True)
            saved_owners = population.owners
            for defect in ("value", "owner"):
                try:
                    if defect == "value":
                        people.loc[people.index[0], column] = (
                            int(saved_values.iloc[0]) + 1
                        )
                    else:
                        changed = dict(saved_owners)
                        changed["person", column] = (
                            graph.BIND_NODE
                            if saved_owners["person", column] != graph.BIND_NODE
                            else "unissued_status_writer"
                        )
                        object.__setattr__(population, "owners", changed)
                    with pytest.raises(ValueError):
                        runner._pure_run(run, runner._run_entry(run))
                finally:
                    people[column] = saved_values.copy(deep=True)
                    object.__setattr__(population, "owners", saved_owners)
                runner._pure_run(run, runner._run_entry(run))
