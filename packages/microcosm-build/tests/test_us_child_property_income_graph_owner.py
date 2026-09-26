"""Real small survey owners plus real graph; invented receiving host only.

No actual country financial/tax/PUF host acceptance is asserted by these tests.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import test_us_child_property_income_source_owner as source_fixture
from test_us_child_property_income_graph import (
    ORDER_BYTES,
    ORDER_TYPE,
    _options,
    _ordering,
    _receiving_frame,
)

from microcosm.build.us_runtime import graph_child_property_income as graph
from microcosm.build.us_runtime import graph_survey_population as survey_graph
from microcosm.frame import WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    NumericScope,
    Owned,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    compile_graph,
    run_graph,
)

actual = source_fixture.actual


def _artifact(manifest, store, edge):
    receipt = manifest.node(edge.producer)
    key = receipt.opaque_artifacts[edge.artifact]
    numeric = receipt.capabilities.numeric
    return ArtifactValue(
        store.load_bytes(key),
        edge.type,
        key,
        receipt.key,
        NumericScope(
            numeric,
            platform=graph.platform_fingerprint()
            if numeric is Numeric.PLATFORM_BITWISE
            else None,
        ),
    )


def _run(actual, tmp_path):
    preparation = actual.partial
    entry = preparation._checked()
    origins = graph._origins(entry)
    qualified = graph.child.qualify_child_property_sources(preparation)
    receiving = _receiving_frame(qualified, origins)
    create_id, allocated_id = "test.child.receiving", "test.child.allocated"
    allocated_weights = Weights(
        receiving.weights_for("household").values, WeightKind.IMPORTANCE
    )

    class Receiving(KernelBase):
        ref = "test.child.receiving@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def run(self, context):
            return KernelResult(frame=receiving, artifacts={"ordering": ORDER_BYTES})

    class Allocate(KernelBase):
        ref = "test.child.allocate@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.REWEIGHT
        )

        def run(self, context):
            return KernelResult(weights=allocated_weights)

    create = Node(
        create_id,
        Receiving.ref,
        sources=(graph.SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(
                entity,
                col,
                graph.populations.token_for_dtype(receiving.table(entity)[col].dtype),
            )
            for entity in receiving.entities
            for col in receiving.table(entity)
            if col != receiving.schema.entity_id_column(entity)
            and not (
                entity == receiving.schema.person_entity
                and col
                in {
                    receiving.schema.membership_column(group)
                    for group in receiving.schema.group_entities
                }
            )
        ),
        artifact_outputs=(ArtifactOutput("ordering", ORDER_TYPE),),
    )
    allocate = Node(
        allocated_id,
        Allocate.ref,
        base=create_id,
        structural=StructuralDelta.REWEIGHT,
        weights=WeightTransition("household", "importance", "conserve"),
        mass="conserve",
    )
    refs = (SourceRef(graph.SOURCE_NAME, survey_graph.SOURCE_CODEC),)
    registry = KernelRegistry()
    registry.register(Receiving())
    registry.register(Allocate())
    store = ContentStore(
        tmp_path / "store",
        codecs=survey_graph.survey_population_source_codecs(
            snapshot_root=tmp_path / "unused-codec-snapshot"
        ),
    )
    sources = {graph.SOURCE_NAME: entry[2].root}
    captured = {}
    prefix = run_graph(
        compile_graph(Graph("us", refs, (create, allocate))),
        sources=sources,
        store=store,
        kernels=registry,
        _population_observer=lambda name, pop: captured.update({name: pop}),
    )
    parent = captured[allocated_id]
    stamp = graph.child.physical._population_stamp(parent)
    calls = SimpleNamespace(count=0, on_call=None)

    def require_parent():
        calls.count += 1
        preparation._checked()
        assert graph.child.physical._population_stamp(parent) == stamp
        if calls.on_call is not None:
            calls.on_call(calls.count)
        return parent

    edge = ArtifactInput("ordering", create_id, "ordering", ORDER_TYPE)
    value = _artifact(prefix, store, edge)
    pins = {
        "ordering": {
            "producer_key": value.producer_key,
            "artifact_key": value.key,
            "payload_sha256": graph._sha(value.payload),
        }
    }
    boundary = graph.ChildPropertyBoundary(
        preparation,
        parent,
        require_parent=require_parent,
        options=_options(only_positive=True),
        host_edges=(edge,),
        host_pins=pins,
    )
    for kernel in boundary.kernels():
        registry.register(kernel)
    compiled = compile_graph(Graph("us", refs, (create, allocate, *boundary.nodes)))
    captured = {}
    cold = run_graph(
        compiled,
        sources=sources,
        store=store,
        kernels=registry,
        _population_observer=lambda name, pop: captured.update({name: pop}),
    )
    assert all(not cold.node(node.id).hit for node in boundary.nodes)

    def evidence(manifest, observed):
        inputs = {
            edge.name: _artifact(manifest, store, edge)
            for edge in boundary.nodes[-1].artifact_inputs
        }
        verification = _artifact(
            manifest,
            store,
            ArtifactInput(
                "verification", graph.VERIFY, "verification", graph.VERIFICATION_TYPE
            ),
        )
        return dict(
            population=observed[graph.VERIFY],
            artifacts=inputs,
            support_populations={
                name: observed[name] for name in (graph.DONOR, graph.RECIPIENT)
            },
            verification=verification,
        )

    def check(manifest, observed):
        return boundary.verify_materialized(**evidence(manifest, observed))

    check(cold, captured)
    warm_capture = {}
    calls_before = calls.count
    warm = run_graph(
        compiled,
        sources=sources,
        store=store,
        kernels=registry,
        resume="require",
        _population_observer=lambda name, pop: warm_capture.update({name: pop}),
    )
    assert all(item.hit for item in warm.nodes.values())
    assert calls.count == calls_before  # Required replay skips every kernel.
    check(warm, warm_capture)
    assert calls.count > calls_before  # Retained host check remains mandatory.
    return SimpleNamespace(
        boundary=boundary,
        cold=cold,
        warm=warm,
        captured=warm_capture,
        store=store,
        calls=calls,
        check=check,
        evidence=evidence,
        qualified=qualified,
        parent=parent,
    )


def test_actual_owner_full_donor_and_actual_graph_complete_identical_child_clones(
    actual, tmp_path
):
    result = _run(actual, tmp_path)
    donor = result.captured[graph.DONOR].frame.person
    assert len(donor) == 1
    assert (
        donor[graph.WEIGHT].iloc[0]
        == result.qualified.donor_projection.donors[graph.WEIGHT].iloc[0]
    )
    assert tuple(result.qualified.donor_projection.donors.donor_key)[0] not in set(
        result.qualified.recipients.recipient_key
    )
    output = result.captured[graph.VERIFY].frame.person
    original = output[graph.provenance.support_source_id_column("person")]
    for identity in result.qualified.recipients.index[
        result.qualified.recipients.eligible
    ]:
        copies = output.loc[original == identity]
        assert len(copies) == 2 and copies[graph.IMPUTED].all()
        np.testing.assert_array_equal(
            copies[list(graph.child.TARGETS)].iloc[0],
            copies[list(graph.child.TARGETS)].iloc[1],
        )
    for name in result.parent.frame.person:
        if name not in graph.child.TARGETS:
            pd.testing.assert_series_equal(
                result.parent.frame.person[name], output[name], check_exact=True
            )
    actual.partial._checked()


@pytest.mark.parametrize("when", ["first", "final"])
def test_actual_owner_callbacks_cannot_change_the_completed_population(
    actual, tmp_path, when
):
    result = _run(actual, tmp_path)
    baseline = result.calls.count
    output = result.captured[graph.VERIFY]
    count = baseline + (1 if when == "first" else 2)
    before = output.frame.person.loc[0, "unrelated_amount"]

    def mutate(n):
        if n == count:
            output.frame.person.loc[0, "unrelated_amount"] += 1.0

    result.calls.on_call = mutate
    with pytest.raises(ValueError):
        result.check(result.warm, result.captured)
    assert output.frame.person.loc[0, "unrelated_amount"] == before + 1.0
    assert result.boundary.revoked
    result.calls.on_call = None
    with pytest.raises(ValueError, match="REVOKED"):
        result.check(result.warm, result.captured)
    actual.partial._checked()


@pytest.mark.parametrize("when", ["first", "final"])
@pytest.mark.parametrize(
    "surface",
    [
        "artifact_replace",
        "support_add",
        "support_remove",
        "verification_payload",
        "verification_producer",
    ],
)
def test_actual_owner_callbacks_cannot_change_materialized_evidence(
    actual, tmp_path, when, surface
):
    result = _run(actual, tmp_path)
    inputs = result.evidence(result.warm, result.captured)
    before = graph.child.physical._population_stamp(inputs["population"])
    target_call = result.calls.count + (1 if when == "first" else 2)
    mutations = []

    def mutate(count):
        if count != target_call:
            return
        mutations.append(surface)
        if surface == "artifact_replace":
            value = inputs["artifacts"]["draws"]
            inputs["artifacts"]["draws"] = replace(value, payload=value.payload + b" ")
        elif surface == "support_add":
            inputs["support_populations"]["unexpected"] = inputs["support_populations"][
                graph.DONOR
            ]
        elif surface == "support_remove":
            del inputs["support_populations"][graph.RECIPIENT]
        elif surface == "verification_payload":
            value = inputs["verification"]
            object.__setattr__(value, "payload", value.payload + b" ")
        else:
            # Keep the descriptor internally valid: changed provenance must
            # still be detected across either original owner borrow.
            value = inputs["verification"]
            producer = "f" * 64
            assert value.producer_key != producer
            object.__setattr__(value, "producer_key", producer)
            object.__setattr__(
                value, "key", graph.opaque_artifact_key(producer, "verification")
            )

    result.calls.on_call = mutate
    with pytest.raises(ValueError, match="EVIDENCE_CHANGED|SUPPORT_POPULATION_ROSTER"):
        result.boundary.verify_materialized(**inputs)
    assert mutations == [surface]
    assert graph.child.physical._population_stamp(inputs["population"]) == before
    assert result.boundary.revoked
    result.calls.on_call = None
    with pytest.raises(ValueError, match="REVOKED"):
        # Fresh valid descriptors cannot resurrect the boundary.
        result.check(result.warm, result.captured)
    actual.partial._checked()


def test_copied_source_owner_cannot_construct_the_country_boundary(actual):
    preparation = object.__new__(type(actual.partial))
    object.__setattr__(preparation, "payload", actual.partial.payload)
    qualified = actual.partial_values
    origins = graph._origins(actual.partial._checked())
    parent = graph.populations.Population.from_frame(
        _receiving_frame(qualified, origins), "test.allocated"
    )
    edge, _, pins = _ordering()
    with pytest.raises(ValueError, match="UNISSUED_OR_CHANGED"):
        graph.ChildPropertyBoundary(
            preparation,
            parent,
            require_parent=lambda: parent,
            options=_options(only_positive=True),
            host_edges=(edge,),
            host_pins=pins,
        )
    actual.partial._checked()
