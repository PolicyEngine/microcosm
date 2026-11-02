"""Composition's proof of a measurement producer (G8-r2c ruling R2C-1).

``compose.measurement_proofs`` satisfies ``measurement:<id>`` only when, for
every calibration node, exactly one node upstream of it declares
``params.measurement_id: <id>`` and writes the scaled interval table's income and
universe columns on that table's entity. Each mutant below breaks one clause
and must leave the rows parked, with the activation gap naming the
measurement.
"""

from __future__ import annotations

import copy
import json

import pytest

from microcosm.build.transport.compose import (
    _reference_activation_gaps,
    measurement_proofs,
    reference_profile_view,
)
from test_support.microcosm_build.reference_intervals import (
    INCOME_COLUMN,
    SUMMED_MEASUREMENT,
    UNIVERSE_COLUMN,
    summed_profile_fixture,
)

GROUPS = [[f"n{index}" for index in range(5)], [f"n{index}" for index in range(5, 9)]]


def _graph(entity="person", *, producer_entity=None, measured=True):
    """CREATE, the producer, preparation, the problem and calibration.

    Preparation reads the income and universe columns on the producer's
    entity, so the producer always sits upstream of calibration.
    """
    producer_entity = entity if producer_entity is None else producer_entity
    measure = {
        "id": "toy.measure",
        "kernel": "measure.combine@1",
        "population": "toy.create",
        "inputs": [{"entity": producer_entity, "columns": ["toy_wages"]}],
        "outputs": [
            {"entity": producer_entity, "column": INCOME_COLUMN, "dtype": "int64"},
            {"entity": producer_entity, "column": UNIVERSE_COLUMN, "dtype": "str"},
        ],
    }
    if measured:
        measure["params"] = {"measurement_id": SUMMED_MEASUREMENT}
    return {
        "schema_version": 1,
        "country": "toy",
        "nodes": [
            {
                "id": "toy.create",
                "kernel": "transport.create@1",
                "structural": "create",
            },
            measure,
            {
                "id": "toy.prepare",
                "kernel": "calibration.prepare@1",
                "population": "toy.create",
                "inputs": [
                    {
                        "entity": producer_entity,
                        "columns": [INCOME_COLUMN, UNIVERSE_COLUMN],
                    }
                ],
                "outputs": [
                    {"entity": entity, "column": "toy_income_band_b0", "dtype": "bool"}
                ],
            },
            {
                "id": "toy.problem",
                "kernel": "targets.problem@1",
                "population": "toy.create",
                "artifact_inputs": [{"name": "prepared", "producer": "toy.prepare"}],
            },
            {
                "id": "toy.calibrate",
                "kernel": "calibrate.ordered_adam@1",
                "population": "toy.create",
                "artifact_inputs": [{"name": "problem", "producer": "toy.problem"}],
            },
        ],
        "endpoints": {"calibration": "toy.calibrate"},
    }


def _spec(graph, *, entity="person"):
    base, profiles, context, intervals = summed_profile_fixture(GROUPS, entity=entity)
    return {
        "country": "toy",
        "resources": {
            "reference_profiles": profiles,
            "target_references": base,
            "factor_context": context,
            "profile_intervals": intervals,
            "transport_graph": graph,
        },
    }


def _nodes(graph):
    return {row["id"]: row for row in graph["nodes"]}


def test_one_upstream_producer_proves_and_activates_the_rows():
    spec = _spec(_graph())
    proofs, refusals = measurement_proofs(spec)
    assert refusals == {}
    assert proofs == {
        SUMMED_MEASUREMENT: {
            "measurement_id": SUMMED_MEASUREMENT,
            "producers": {"toy.calibrate": "toy.measure"},
            "columns": [f"person.{INCOME_COLUMN}", f"person.{UNIVERSE_COLUMN}"],
        }
    }
    references, _, receipt = reference_profile_view(spec)
    assert receipt["measurement_proofs"] == proofs
    assert {
        row["metadata"]["activation_status"] for row in references["target_references"]
    } == {"active"}
    assert list(_reference_activation_gaps(references, "target_references")) == []


def test_a_third_entity_table_passes_the_proof():
    """A table on a non-household, non-person entity, as ``tax_unit`` is."""
    spec = _spec(_graph("tax_unit"), entity="tax_unit")
    proofs, refusals = measurement_proofs(spec)
    assert refusals == {}
    assert proofs[SUMMED_MEASUREMENT]["columns"] == [
        f"tax_unit.{INCOME_COLUMN}",
        f"tax_unit.{UNIVERSE_COLUMN}",
    ]


def _no_producer(graph):
    del _nodes(graph)["toy.measure"]["params"]


def _after_calibration(graph):
    # The producer reads the calibration artifact, so nothing upstream of
    # calibration can read what it writes; preparation reads a CREATE column.
    nodes = _nodes(graph)
    nodes["toy.measure"]["artifact_inputs"] = [
        {"name": "fit", "producer": "toy.calibrate"}
    ]
    nodes["toy.prepare"]["inputs"] = [{"entity": "person", "columns": ["toy_wages"]}]


def _two_producers(graph):
    second = copy.deepcopy(_nodes(graph)["toy.measure"])
    second["id"] = "toy.measure.again"
    second["outputs"] = [
        {"entity": "person", "column": "toy_other_income", "dtype": "int64"}
    ]
    graph["nodes"].insert(2, second)
    _nodes(graph)["toy.prepare"]["inputs"][0]["columns"].append("toy_other_income")


def _wrong_column(graph):
    # The producer writes another income column; CREATE supplies the one
    # the table names.
    nodes = _nodes(graph)
    nodes["toy.measure"]["outputs"][0]["column"] = "toy_other_cents"


def _income_without_universe(graph):
    nodes = _nodes(graph)
    nodes["toy.measure"]["outputs"] = nodes["toy.measure"]["outputs"][:1]
    graph["nodes"].insert(
        2,
        {
            "id": "toy.universe",
            "kernel": "measure.combine@1",
            "population": "toy.create",
            "outputs": [
                {"entity": "person", "column": UNIVERSE_COLUMN, "dtype": "str"}
            ],
        },
    )


@pytest.mark.parametrize(
    ("mutant", "graph", "entity", "reason"),
    [
        ("no_producer", _no_producer, "person", "0 nodes declaring"),
        ("after_calibration", _after_calibration, "person", "0 nodes declaring"),
        ("two_producers", _two_producers, "person", "2 nodes declaring"),
        ("wrong_column", _wrong_column, "person", "does not write"),
        (
            "income_without_universe",
            _income_without_universe,
            "person",
            "does not write",
        ),
        ("wrong_entity", None, "person", "does not write"),
        ("wrong_entity_tax_unit", None, "tax_unit", "does not write"),
    ],
)
def test_each_proof_mutant_leaves_rows_parked_and_names_the_measurement(
    mutant, graph, entity, reason
):
    if graph is None:
        document = _graph(entity, producer_entity="household")
    else:
        document = _graph(entity)
        graph(document)
    spec = _spec(document, entity=entity)
    proofs, refusals = measurement_proofs(spec)
    assert proofs == {}
    assert reason in refusals[SUMMED_MEASUREMENT]
    references, _, receipt = reference_profile_view(spec)
    assert "measurement_proofs" not in receipt
    rows = references["target_references"]
    assert {row["metadata"]["activation_status"] for row in rows} == {
        "requires_prerequisites"
    }
    gaps = list(
        _reference_activation_gaps(references, "target_references", refusals=refusals)
    )
    assert len(gaps) == len(rows)
    assert all(f"measurement:{SUMMED_MEASUREMENT} (" in gap for gap in gaps)
    assert all(reason in gap for gap in gaps)


def test_every_calibration_node_needs_its_own_upstream_producer():
    graph = _graph()
    graph["nodes"].append(
        {
            "id": "toy.calibrate.variant",
            "kernel": "calibrate.ordered_adam@1",
            "population": "toy.create",
        }
    )
    proofs, refusals = measurement_proofs(_spec(graph))
    assert proofs == {}
    assert "toy.calibrate.variant" in refusals[SUMMED_MEASUREMENT]


def test_a_spec_without_profiles_or_scaling_proves_nothing():
    assert measurement_proofs({"country": "toy", "resources": {}}) == ({}, {})
    spec = _spec(_graph())
    for profile in spec["resources"]["reference_profiles"]["dimensions"]["index"][
        "profiles"
    ].values():
        profile["transformations"] = []
    assert measurement_proofs(spec) == ({}, {})


def test_proof_result_is_bound_into_the_selection_identity():
    proved, _, proved_receipt = reference_profile_view(_spec(_graph()))
    document = _graph()
    _no_producer(document)
    parked, _, parked_receipt = reference_profile_view(_spec(document))
    assert (
        proved_receipt["reference_selection_sha256"]
        != parked_receipt["reference_selection_sha256"]
    )
    assert (
        json.loads(
            proved["target_references"][0]["metadata"][
                "reference_prerequisites_remaining"
            ]
        )
        == []
    )
    assert parked != proved


def _compiled_toy(*, second_producer=False):
    """A compiled graph: CREATE, one or two producers, preparation, calibration."""
    from microcosm.graph import (
        Graph,
        Node,
        Owned,
        Slice,
        SourceRef,
        StructuralDelta,
        compile_graph,
    )

    def producer(name, column):
        return Node(
            id=name,
            kernel="measure.combine@1",
            population="toy.create",
            params={"measurement_id": SUMMED_MEASUREMENT},
            outputs=(Owned("person", column, "int64"),),
        )

    nodes = [
        Node(
            id="toy.create",
            kernel="transport.create@1",
            structural=StructuralDelta.CREATE,
            sources=("toy_donor",),
            outputs=(Owned("person", "person_id", "int64"),),
        ),
        producer("toy.measure", INCOME_COLUMN),
    ]
    read = [INCOME_COLUMN]
    if second_producer:
        nodes.append(producer("toy.measure.again", "toy_other_cents"))
        read.append("toy_other_cents")
    nodes += [
        Node(
            id="toy.prepare",
            kernel="calibration.prepare@1",
            population="toy.create",
            inputs=(Slice("person", tuple(read)),),
            outputs=(Owned("person", "toy_income_band_b0", "bool"),),
        ),
        Node(
            id="toy.calibrate",
            kernel="calibrate.ordered_adam@1",
            population="toy.create",
            inputs=(Slice("person", ("toy_income_band_b0",)),),
        ),
    ]
    graph = Graph(
        country="toy",
        sources=(SourceRef("toy_donor", "toy-codec-v1"),),
        nodes=tuple(nodes),
    )
    return graph, compile_graph(graph)


def _forced_proof(monkeypatch, producer):
    from microcosm.build.transport import compose as compose_module

    proof = {
        SUMMED_MEASUREMENT: {
            "measurement_id": SUMMED_MEASUREMENT,
            "producers": {"toy.calibrate": producer},
            "columns": [],
        }
    }
    monkeypatch.setattr(compose_module, "measurement_proofs", lambda spec: (proof, {}))
    monkeypatch.setattr(
        compose_module,
        "transport_endpoints",
        lambda spec: {"calibration": "toy.calibrate"},
    )
    return compose_module._check_measurement_proofs


def test_the_compiled_graph_confirms_the_declared_producer(monkeypatch):
    check = _forced_proof(monkeypatch, "toy.measure")
    graph, compiled = _compiled_toy()
    check({}, graph, compiled)


@pytest.mark.parametrize(
    ("producer", "second", "found"),
    [("toy.elsewhere", False, "1 nodes"), ("toy.measure", True, "2 nodes")],
)
def test_composition_refuses_a_proof_the_compiled_graph_contradicts(
    monkeypatch, producer, second, found
):
    check = _forced_proof(monkeypatch, producer)
    graph, compiled = _compiled_toy(second_producer=second)
    with pytest.raises(ValueError, match=f"contradicts the declared proof; {found}"):
        check({}, graph, compiled)


def test_a_non_string_measurement_id_param_proves_nothing():
    document = _graph()
    _nodes(document)["toy.measure"]["params"] = {
        "measurement_id": {"resource": "factor_context", "path": [], "encoding": "json"}
    }
    proofs, refusals = measurement_proofs(_spec(document))
    assert proofs == {}
    assert "not a string" in refusals[SUMMED_MEASUREMENT]
