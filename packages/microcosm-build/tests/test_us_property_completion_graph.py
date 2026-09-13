"""Invented source/projection and actual export boundary; no models/engine."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_us_current_property_income_sources import _detached, actual  # noqa: F401

from microcosm.build.us_runtime import current_property_completion_routing as routing
from microcosm.build.us_runtime import graph_current_survey_property as graph
from microcosm.graph import (
    Capabilities,
    Determinism,
    Graph,
    Node,
    SourceRef,
    StructuralDelta,
    compile_graph,
    describe,
    explain_html,
)
from microcosm.graph.manifest import NodeReceipt, RunManifest

OPTIONS = dict(scales=(1.0, 1.0, 1.0, 1.0), atol=1e-10, rtol=1e-12, n_estimators=2)


def _pins(live):
    result = {}
    for edge in graph.host.current_survey_host_edges():
        record = live.manifest.node(edge.producer)
        key = record.opaque_artifacts[edge.artifact]
        result[edge.name] = {
            "producer_key": record.key,
            "artifact_key": key,
            "payload_sha256": graph.codec.sha(live.store.load_bytes(key)),
        }
    return result


def test_disabled_options_preserve_original_canonical_bytes():
    old = graph.codec.encode_json(OPTIONS)
    assert graph.PropertyIncomeOptions(**OPTIONS).to_bytes() == old
    assert (
        graph.PropertyIncomeOptions(**OPTIONS, completion_routing=False).to_bytes()
        == old
    )
    enabled = graph.PropertyIncomeOptions(**OPTIONS, completion_routing=True)
    assert enabled.model_document() == OPTIONS
    assert enabled.to_bytes() == graph.codec.encode_json(
        {**OPTIONS, "completion_routing": True}
    )


@pytest.mark.parametrize("bad", [0, 1, None, "true", np.bool_(True)])
def test_non_boolean_option_refuses(bad):
    with pytest.raises(ValueError, match="COMPLETION_ROUTING_OPTION"):
        graph.PropertyIncomeOptions(**OPTIONS, completion_routing=bad)


def test_declaration_does_not_run_routing_and_only_source_node_changes(
    actual,  # noqa: F811
    monkeypatch,
):
    live, qualified, _ = actual

    def forbidden(*args, **kwargs):
        raise AssertionError("routing must not run during declaration")

    monkeypatch.setattr(routing, "build_property_completion_routing", forbidden)
    off, on = [
        graph.current_survey_property_nodes(
            qualified,
            live.clone_population.frame,
            host_pins=_pins(live),
            options=graph.PropertyIncomeOptions(**OPTIONS, completion_routing=flag),
        )
        for flag in (False, True)
    ]
    assert len(off) == len(on) == 16
    assert off[1:] == on[1:]
    assert (
        replace(on[0], params=off[0].params, artifact_outputs=off[0].artifact_outputs)
        == off[0]
    )
    assert dict(on[0].params) == {**dict(off[0].params), "completion_routing": True}
    assert on[0].artifact_outputs == (
        *off[0].artifact_outputs,
        routing.property_completion_artifact_output(),
    )
    assert len(off[0].artifact_outputs) == 3


def test_projection_preserves_base_bytes_and_verifies_expected_private_artifact(actual):  # noqa: F811
    live, qualified, _ = actual
    before = graph.sources.property_income_sources_seal(qualified)
    clone_before = graph.sources._frame_seal(live.clone_population.frame)
    off, on = [
        graph._projection_result(
            qualified,
            live.clone_population.frame,
            graph.PropertyIncomeOptions(**OPTIONS, completion_routing=flag),
        )
        for flag in (False, True)
    ]
    assert dict(off.artifacts) == graph._projection_values(qualified)
    assert all(on.artifacts[k] == v for k, v in off.artifacts.items())
    assert {
        k: v for k, v in on.receipt.items() if k != "completion_routing"
    } == off.receipt
    assert on.receipt["completion_routing"][
        "private_artifact_sha256"
    ] == graph.codec.sha(on.artifacts["completion_routing"])
    artifacts = {(graph.PROJECTION_NODE, n): v for n, v in on.artifacts.items()}
    expected = graph._checked_projection_result(
        qualified,
        live.clone_population.frame,
        graph.PropertyIncomeOptions(**OPTIONS, completion_routing=True),
        artifacts,
    )
    assert dict(expected.artifacts) == dict(on.artifacts)
    assert expected.receipt == on.receipt
    assert graph.sources.property_income_sources_seal(qualified) == before
    assert graph.sources._frame_seal(live.clone_population.frame) == clone_before


@pytest.mark.parametrize(
    "fault", ["private_bytes", "missing_private", "unexpected_private", "base_bytes"]
)
def test_actual_reconstruction_refuses_artifact_mutation_before_model_reads(
    actual,  # noqa: F811
    monkeypatch,
    fault,
):
    live, qualified, _ = actual
    options = graph.PropertyIncomeOptions(
        **OPTIONS, completion_routing=fault != "unexpected_private"
    )
    result = graph._projection_result(qualified, live.clone_population.frame, options)
    artifacts = {(graph.PROJECTION_NODE, n): v for n, v in result.artifacts.items()}
    key = (graph.PROJECTION_NODE, "completion_routing")
    if fault == "missing_private":
        del artifacts[key]
    elif fault == "base_bytes":
        artifacts[graph.PROJECTION_NODE, "projection"] += b" "
    else:
        artifacts[key] = b"altered or undeclared private bytes"

    def forbidden(*args):
        raise AssertionError("model artifact decoding is out of this test's scope")

    monkeypatch.setattr(graph, "_read_draws", forbidden)
    with pytest.raises(ValueError, match="PROJECTION_ARTIFACTS"):
        graph.reconstruct_property_results(
            qualified,
            live.clone_population.frame,
            host_pins=_pins(live),
            options=options,
            artifacts=artifacts,
            legacy_matrix_producer_key="a" * 64,
        )


def _large_identity(qualified, frame):
    value, clone = _detached(qualified), _detached(frame)
    original = value.origins.index[
        value.origins.source.eq("asec") & value.origins.native_person_id.eq(107)
    ][0]
    native = 2**53 + 17
    value.origins.loc[original, "native_person_id"] = native
    value.shared_predictors.origins.loc[original, "native_person_id"] = native
    document = json.loads(value.origin_document)
    columns = document["persons"]["columns"]
    for row in document["persons"]["rows"]:
        if row[columns.index("person_id")] == int(original):
            row[columns.index("selected_receiving_person_id")] = native
    object.__setattr__(value, "origin_document", graph.codec.encode_json(document))
    for table in (
        value.asec_interest_values.person,
        value.asec_routing_values.person,
        value.asec_dividend_values.person,
        value.donor_basis.person,
    ):
        table.loc[original, "native_person_id"] = native
    p = routing.provenance
    for source in (value.source_frame, value.shared_predictors.source_frame):
        source.person.loc[
            source.person.person_id.eq(original), p.spine_source_id_column("person")
        ] = native
    clone.person.loc[
        clone.person[p.support_source_id_column("person")].eq(original),
        p.spine_source_id_column("person"),
    ] = native
    return value, clone, native


def test_actual_explain_and_describe_expose_summary_not_private_rows(actual):  # noqa: F811
    live, qualified, _ = actual
    value, frame, native = _large_identity(qualified, live.clone_population.frame)
    result = graph._projection_result(
        value, frame, graph.PropertyIncomeOptions(**OPTIONS, completion_routing=True)
    )
    private = result.artifacts["completion_routing"]
    assert str(native).encode() in private
    summary = result.receipt["completion_routing"]["summary"]
    assert summary["routes_mutually_exclusive"] and summary["reasons_overlap"]
    # Small descriptive graph uses the real production source result, not a toy
    # replacement serializer. No source/graph execution or authority is claimed.
    create = Node(
        "source",
        "fixture.create@1",
        sources=("invented",),
        outputs=(graph.Owned("person", "age", "int64"),),
        structural=StructuralDelta.CREATE,
    )
    node = Node(
        graph.PROJECTION_NODE,
        graph.CurrentSurveyPropertyProjectionKernel.ref,
        population="source",
        artifact_outputs=tuple(
            graph.ArtifactOutput(
                n,
                routing.PROPERTY_COMPLETION_TYPE
                if n == "completion_routing"
                else graph.PROJECTION_TYPE,
            )
            for n in result.artifacts
        ),
    )
    compiled = compile_graph(
        Graph("completion_privacy", (SourceRef("invented", "fixture"),), (create, node))
    )

    def receipt(node, key, body, opaque):
        return NodeReceipt(
            key=key,
            hit=False,
            seed=1,
            kernel_ref=node.kernel,
            kernel_impl_hash="f" * 64,
            capabilities=Capabilities(
                Determinism.DETERMINISTIC, structural=node.structural
            ),
            receipt=body,
            opaque_artifacts=opaque,
        )

    manifest = RunManifest(
        "completion_privacy",
        {
            "source": receipt(create, "a" * 64, {}, {}),
            node.id: receipt(
                node,
                "b" * 64,
                result.receipt,
                {n: graph.codec.sha(v) for n, v in result.artifacts.items()},
            ),
        },
    )
    public = (
        json.dumps(dict(result.receipt))
        + explain_html(compiled, manifest)
        + describe(compiled, node.id, manifest)
    )
    assert str(native) not in public
    # Short fixture IDs also occur in legitimate counts and hashes. The exact
    # large native sentinel and closed schema below test disclosure, not digit
    # coincidence in otherwise safe aggregate text.
    assert '"completion_routing"' in public
    assert "carry_known_components" in public
    for name in (
        "native_person_id",
        "original_person_id",
        "clone_person_id",
        "tables",
        "literal_status",
    ):
        assert name not in json.dumps(summary)
    assert private.decode() not in public


@pytest.mark.parametrize("fault", ["label", "column", "non_numeric", "non_finite"])
def test_public_summary_refuses_unreviewed_or_non_numeric_fields(actual, fault):  # noqa: F811
    live, qualified, _ = actual
    value = routing.build_property_completion_routing(
        qualified, live.clone_population.frame
    )
    table = value.summary.copy(deep=True)
    if fault == "label":
        table.rename(index={"all": "native_person_9007199254741009"}, inplace=True)
    elif fault == "column":
        table["native_person_id"] = 2**53 + 17
    elif fault == "non_numeric":
        table["person_count"] = table.person_count.astype(str)
    else:
        table.loc["all", "design_weighted_person_mass"] = np.inf
    with pytest.raises(ValueError, match="PUBLIC_"):
        routing.property_completion_public_summary(replace(value, summary=table))


def test_kernel_identity_declares_routing_implementation(actual, monkeypatch):  # noqa: F811
    live, _, _ = actual
    kernel = graph.CurrentSurveyPropertyProjectionKernel(
        live.preparation,
        live.allocated_population,
        live.clone_population,
        host_pins=_pins(live),
        options=graph.PropertyIncomeOptions(**OPTIONS),
    )
    original = graph.source_hash
    calls = []

    def capture(*modules, **kwargs):
        calls.append(modules)
        return original(*modules, **kwargs)

    monkeypatch.setattr(graph, "source_hash", capture)
    first = kernel.implementation_hash()
    assert any(routing in modules for modules in calls)
    kernel.options = graph.PropertyIncomeOptions(**OPTIONS, completion_routing=True)
    assert kernel.implementation_hash() == first


def test_retained_option_serialization_rejects_non_boolean_mutation():
    options = graph.PropertyIncomeOptions(**OPTIONS, completion_routing=True)
    object.__setattr__(options, "completion_routing", 1)
    with pytest.raises(ValueError, match="COMPLETION_ROUTING_OPTION"):
        options.to_bytes()


@pytest.mark.parametrize("bad_base", [False, True])
def test_attachment_checks_three_edges_without_routing_before_numeric_reconstruction(
    actual,  # noqa: F811
    monkeypatch,
    bad_base,
):
    live, qualified, _ = actual
    base = graph._projection_values(qualified)
    artifacts = {(graph.PROJECTION_NODE, n): p for n, p in base.items()}
    if bad_base:
        artifacts[graph.PROJECTION_NODE, "projection"] += b" "
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("attachment must not recompute routing")

    def inspect_numeric(*args, **kwargs):
        calls.append(kwargs)
        assert dict(kwargs["source_result"].artifacts) == base
        assert kwargs["options"].completion_routing is True
        return {graph.ATTACH_NODE: "numeric seam reached"}

    monkeypatch.setattr(routing, "build_property_completion_routing", forbidden)
    monkeypatch.setattr(graph, "_reconstruct_property_values", inspect_numeric)
    options = graph.PropertyIncomeOptions(**OPTIONS, completion_routing=True)
    if bad_base:
        with pytest.raises(ValueError, match="PROJECTION_ARTIFACTS"):
            graph._reconstruct_attachment_results(
                qualified,
                live.clone_population.frame,
                options=options,
                artifacts=artifacts,
                legacy_matrix_producer_key="a" * 64,
            )
        assert not calls
    else:
        assert graph._reconstruct_attachment_results(
            qualified,
            live.clone_population.frame,
            options=options,
            artifacts=artifacts,
            legacy_matrix_producer_key="a" * 64,
        ) == {graph.ATTACH_NODE: "numeric seam reached"}
        assert len(calls) == 1


@pytest.mark.parametrize(
    "name",
    [
        "build_property_completion_routing",
        "property_completion_public_summary",
        "_PUBLIC_ROUTES",
        "_PUBLIC_REASONS",
        "PROPERTY_COMPLETION_TYPE",
        "_COMPONENT_FIELDS",
    ],
)
def test_retained_live_marker_binds_completion_functions_and_contracts(
    monkeypatch, name
):
    from microcosm.build.us_runtime import graph_atomic_survey_financial as runner

    options = graph.PropertyIncomeOptions(**OPTIONS, completion_routing=True)
    before = runner._live(options, True)
    original = getattr(routing, name)
    if name in (
        "build_property_completion_routing",
        "property_completion_public_summary",
    ):

        def changed(*args, **kwargs):
            return None
    elif name == "PROPERTY_COMPLETION_TYPE":
        changed = graph.ArtifactType("invented.other_completion", 1)
    elif name == "_COMPONENT_FIELDS":
        changed = (*original, ("invented", "FIELD", "known"))
    else:
        changed = original | {"invented_unreviewed_label"}
    monkeypatch.setattr(routing, name, changed)
    assert runner._live(options, True) != before
    monkeypatch.setattr(routing, name, original)
    assert runner._live(options, True) == before
