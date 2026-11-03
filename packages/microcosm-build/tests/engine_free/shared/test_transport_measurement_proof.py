"""Composition's proof of a measurement producer (G8-r2c ruling R2C-1).

``compose.measurement_proofs`` satisfies ``measurement:<id>`` only when, for
every calibration node, exactly one node upstream of it declares
``params.measurement_id: <id>`` and writes the scaled interval table's income and
universe columns on that table's entity. Each mutant below breaks one clause
and must leave the rows parked, with the activation gap naming the
measurement. Composition checks the same rule against the compiled graph, and
for a calibration node that a transport extension adds, which no declared row
names, the compiled graph alone proves the producer.
"""

from __future__ import annotations

import copy
import json
import re

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.compose import (
    _reference_activation_gaps,
    compose_transport_graph,
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


def _filtered(graph):
    """Open a ``filter`` version on CREATE and move calibration into it."""
    nodes = _nodes(graph)
    position = graph["nodes"].index(nodes["toy.prepare"])
    graph["nodes"].insert(
        position,
        {
            "id": "toy.filter",
            "kernel": "transport.boundary@1",
            "structural": "filter",
            "base": "toy.create",
        },
    )
    for name in ("toy.prepare", "toy.problem", "toy.calibrate"):
        nodes[name]["population"] = "toy.filter"
    return position + 1


def _rewriter(name, column, dtype, *, population="toy.filter", rewrite=True):
    output = {"entity": "person", "column": column, "dtype": dtype}
    if rewrite:
        output["rewrite"] = True
    return {
        "id": name,
        "kernel": "toy.rewrite@1",
        "population": population,
        "inputs": [{"entity": "person", "columns": [column]}],
        "outputs": [output],
    }


def _later_version_rewrite(graph):
    """The review's item 2c: an undeclared later-version rewrite of the income."""
    position = _filtered(graph)
    graph["nodes"].insert(position, _rewriter("toy.overwrite", INCOME_COLUMN, "int64"))


def _writer_before_the_producer(graph):
    """An undeclared CREATE-version writer; the producer rewrites it later."""
    _filtered(graph)
    nodes = _nodes(graph)
    nodes["toy.measure"]["population"] = "toy.filter"
    nodes["toy.measure"]["outputs"][0]["rewrite"] = True
    graph["nodes"].remove(nodes["toy.measure"])
    graph["nodes"].insert(
        graph["nodes"].index(nodes["toy.prepare"]), nodes["toy.measure"]
    )
    graph["nodes"].insert(
        1,
        _rewriter(
            "toy.early",
            INCOME_COLUMN,
            "int64",
            population="toy.create",
            rewrite=False,
        ),
    )
    graph["nodes"][1]["inputs"] = [{"entity": "person", "columns": ["toy_wages"]}]


def _universe_rewrite(graph):
    position = _filtered(graph)
    graph["nodes"].insert(position, _rewriter("toy.overwrite", UNIVERSE_COLUMN, "str"))


def _expand_writer(graph, cells=None):
    """An EXPAND version that may write the income, with calibration in it."""
    nodes = _nodes(graph)
    graph["nodes"].insert(
        graph["nodes"].index(nodes["toy.prepare"]),
        {
            "id": "toy.expand",
            "kernel": "transport.boundary@1",
            "structural": "expand",
            "base": "toy.create",
            "params": {
                "expand_cells": (
                    [["person", INCOME_COLUMN, "int64"]] if cells is None else cells
                )
            },
        },
    )
    for name in ("toy.prepare", "toy.problem", "toy.calibrate"):
        nodes[name]["population"] = "toy.expand"


def _expand_with_resource_cells(graph):
    """An EXPAND whose cells only composition can tell writes every column."""
    _expand_writer(graph, {"resource": "toy_cells"})


@pytest.mark.parametrize(
    ("mutant", "graph", "entity", "reason"),
    [
        (
            "expand_cells",
            _expand_writer,
            "person",
            f"writers ['toy.expand', 'toy.measure'] of person.{INCOME_COLUMN} upstream",
        ),
        (
            "expand_resource_cells",
            _expand_with_resource_cells,
            "person",
            f"writers ['toy.expand', 'toy.measure'] of person.{INCOME_COLUMN} upstream",
        ),
        (
            "later_version_rewrite",
            _later_version_rewrite,
            "person",
            "writers ['toy.measure', 'toy.overwrite'] of person."
            f"{INCOME_COLUMN} upstream",
        ),
        (
            "writer_before_the_producer",
            _writer_before_the_producer,
            "person",
            f"writers ['toy.early', 'toy.measure'] of person.{INCOME_COLUMN} upstream",
        ),
        (
            "universe_rewrite",
            _universe_rewrite,
            "person",
            "writers ['toy.measure', 'toy.overwrite'] of person."
            f"{UNIVERSE_COLUMN} upstream",
        ),
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


def _compiled_toy(
    *, declared=True, second_producer=False, second_writer=False, expand_cells=None
):
    """A compiled graph: CREATE, one or two producers, preparation, calibration.

    Without ``declared``, the first producer declares no measurement. With
    ``second_writer``, a ``filter`` version holds an undeclared rewrite
    of the income column, and preparation and calibration live in it. With
    ``expand_cells``, preparation and calibration live in an EXPAND version
    that declares those cells.
    """
    from microcosm.graph import (
        Graph,
        Node,
        Owned,
        Slice,
        SourceRef,
        StructuralDelta,
        compile_graph,
    )

    def producer(name, column, declares=True):
        return Node(
            id=name,
            kernel="measure.combine@1",
            population="toy.create",
            params={"measurement_id": SUMMED_MEASUREMENT} if declares else {},
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
        producer("toy.measure", INCOME_COLUMN, declared),
    ]
    read = [INCOME_COLUMN]
    if second_producer:
        nodes.append(producer("toy.measure.again", "toy_other_cents"))
        read.append("toy_other_cents")
    population = "toy.create"
    if second_writer:
        population = "toy.filter"
        nodes += [
            Node(
                id="toy.filter",
                kernel="transport.boundary@1",
                structural=StructuralDelta.FILTER,
                base="toy.create",
            ),
            Node(
                id="toy.overwrite",
                kernel="toy.rewrite@1",
                population="toy.filter",
                inputs=(Slice("person", (INCOME_COLUMN,)),),
                outputs=(Owned("person", INCOME_COLUMN, "int64", rewrite=True),),
            ),
        ]
    if expand_cells is not None:
        population = "toy.expand"
        nodes.append(
            Node(
                id="toy.expand",
                kernel="transport.boundary@1",
                structural=StructuralDelta.EXPAND,
                base="toy.create",
                params={"expand_cells": expand_cells},
            )
        )
    nodes += [
        Node(
            id="toy.prepare",
            kernel="calibration.prepare@1",
            population=population,
            inputs=(Slice("person", tuple(read)),),
            outputs=(Owned("person", "toy_income_band_b0", "bool"),),
        ),
        Node(
            id="toy.calibrate",
            kernel="calibrate.ordered_adam@1",
            population=population,
            inputs=(Slice("person", ("toy_income_band_b0",)),),
        ),
    ]
    graph = Graph(
        country="toy",
        sources=(SourceRef("toy_donor", "toy-codec-v1"),),
        nodes=tuple(nodes),
    )
    return graph, compile_graph(graph)


def _forced_proof(monkeypatch, producer, columns=(f"person.{INCOME_COLUMN}",)):
    """Patch in a proof; with no producer it names no calibration node."""
    from microcosm.build.transport import compose as compose_module

    proof = {
        SUMMED_MEASUREMENT: {
            "measurement_id": SUMMED_MEASUREMENT,
            "producers": {} if producer is None else {"toy.calibrate": producer},
            "columns": list(columns),
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


def test_the_compiled_check_alone_refuses_a_second_writer(monkeypatch):
    """The declared check never sees this graph: a forced proof drives it."""
    check = _forced_proof(monkeypatch, "toy.measure")
    graph, compiled = _compiled_toy(second_writer=True)
    with pytest.raises(ValueError) as refused:
        check({}, graph, compiled)
    assert str(refused.value) == (
        f"Measurement {SUMMED_MEASUREMENT!r}: the compiled graph contradicts the "
        f"declared proof; measurement {SUMMED_MEASUREMENT!r}: calibration node "
        "'toy.calibrate' has writers ['toy.measure', 'toy.overwrite'] of "
        f"person.{INCOME_COLUMN} upstream; only its producer 'toy.measure' may "
        "write it."
    )


def test_the_compiled_check_alone_counts_an_expand_as_a_writer(monkeypatch):
    check = _forced_proof(monkeypatch, "toy.measure")
    graph, compiled = _compiled_toy(expand_cells=(("person", INCOME_COLUMN, "int64"),))
    with pytest.raises(ValueError, match=r"writers \['toy.expand', 'toy.measure'\]"):
        check({}, graph, compiled)


def test_the_compiled_check_passes_an_expand_of_another_column(monkeypatch):
    check = _forced_proof(monkeypatch, "toy.measure")
    graph, compiled = _compiled_toy(expand_cells=(("person", "toy_other", "int64"),))
    check({}, graph, compiled)


def test_a_non_string_measurement_id_param_proves_nothing():
    document = _graph()
    _nodes(document)["toy.measure"]["params"] = {
        "measurement_id": {"resource": "factor_context", "path": [], "encoding": "json"}
    }
    proofs, refusals = measurement_proofs(_spec(document))
    assert proofs == {}
    assert "not a string" in refusals[SUMMED_MEASUREMENT]


# ---------------------------------------------------------------------------
# R2C-15: a measurement has one writer
# ---------------------------------------------------------------------------


def test_a_later_version_rewrite_names_the_measurement_node_column_and_writers():
    """The review's item 2c refuses in the declared check, naming every fact."""
    document = _graph()
    _later_version_rewrite(document)
    proofs, refusals = measurement_proofs(_spec(document))
    assert proofs == {}
    assert refusals == {
        SUMMED_MEASUREMENT: (
            f"measurement {SUMMED_MEASUREMENT!r}: calibration node "
            "'toy.calibrate' has writers ['toy.measure', 'toy.overwrite'] of "
            f"person.{INCOME_COLUMN} upstream; only its producer 'toy.measure' "
            "may write it"
        )
    }


def test_a_second_writer_refuses_without_rewrite():
    """A later writer refuses whether or not its output says ``rewrite``."""
    document = _graph()
    position = _filtered(document)
    document["nodes"].insert(
        position, _rewriter("toy.overwrite", INCOME_COLUMN, "int64", rewrite=False)
    )
    proofs, refusals = measurement_proofs(_spec(document))
    assert proofs == {}
    assert "['toy.measure', 'toy.overwrite']" in refusals[SUMMED_MEASUREMENT]


def test_a_same_named_column_on_another_entity_is_not_a_second_writer():
    document = _graph()
    position = _filtered(document)
    other = _rewriter("toy.household", INCOME_COLUMN, "int64", rewrite=False)
    other["inputs"] = [{"entity": "person", "columns": ["toy_wages"]}]
    other["outputs"][0]["entity"] = "household"
    document["nodes"].insert(position, other)
    _nodes(document)["toy.prepare"]["inputs"].append(
        {"entity": "household", "columns": [INCOME_COLUMN]}
    )
    proofs, refusals = measurement_proofs(_spec(document))
    assert refusals == {}
    assert proofs[SUMMED_MEASUREMENT]["producers"] == {"toy.calibrate": "toy.measure"}


def test_a_writer_outside_the_calibration_closure_is_not_a_second_writer():
    """A later version that reads the fit rewrites the column after calibration."""
    document = _graph()
    document["nodes"] += [
        {
            "id": "toy.after",
            "kernel": "transport.boundary@1",
            "structural": "filter",
            "base": "toy.create",
            "artifact_inputs": [{"name": "fit", "producer": "toy.calibrate"}],
        },
        _rewriter("toy.overwrite", INCOME_COLUMN, "int64", population="toy.after"),
    ]
    proofs, refusals = measurement_proofs(_spec(document))
    assert refusals == {}
    assert proofs[SUMMED_MEASUREMENT]["producers"] == {"toy.calibrate": "toy.measure"}


@given(
    cells=st.lists(
        st.sampled_from([INCOME_COLUMN, UNIVERSE_COLUMN, "toy_other"]), unique=True
    )
)
@settings(
    max_examples=30,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_an_expand_proves_exactly_when_its_cells_avoid_the_measured_columns(cells):
    document = _graph()
    _expand_writer(document, [["person", column, "int64"] for column in cells])
    proofs, refusals = measurement_proofs(_spec(document))
    proved = not {INCOME_COLUMN, UNIVERSE_COLUMN} & set(cells)
    assert (SUMMED_MEASUREMENT in proofs) is proved
    assert (SUMMED_MEASUREMENT in refusals) is not proved


def _chain(draws):
    """CREATE, then nodes chained by artifacts, all upstream of calibration.

    Each draw is (declares the id, writes the income, writes the universe).
    """
    nodes = [
        {"id": "toy.create", "kernel": "transport.create@1", "structural": "create"}
    ]
    previous = None
    for index, (declares, income, universe) in enumerate(draws):
        row = {
            "id": f"toy.n{index}",
            "kernel": "toy.chain@1",
            "population": "toy.create",
            "outputs": [
                {"entity": "person", "column": column, "dtype": dtype}
                for column, dtype, wanted in (
                    (INCOME_COLUMN, "int64", income),
                    (UNIVERSE_COLUMN, "str", universe),
                    (f"toy_step_{index}", "bool", True),
                )
                if wanted
            ],
        }
        if declares:
            row["params"] = {"measurement_id": SUMMED_MEASUREMENT}
        if previous is not None:
            row["inputs"] = [{"entity": "person", "columns": [previous]}]
        previous = f"toy_step_{index}"
        nodes.append(row)
    nodes.append(
        {
            "id": "toy.calibrate",
            "kernel": "calibrate.ordered_adam@1",
            "population": "toy.create",
            "inputs": [{"entity": "person", "columns": [previous]}],
        }
    )
    return {
        "schema_version": 1,
        "country": "toy",
        "nodes": nodes,
        "endpoints": {"calibration": "toy.calibrate"},
    }


@given(
    draws=st.lists(
        st.tuples(st.booleans(), st.booleans(), st.booleans()),
        min_size=1,
        max_size=4,
    )
)
@settings(
    max_examples=60,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
# Random draws rarely reach these: one declarer alone, then with a second
# writer after it and before it.
@example(draws=[(True, True, True)])
@example(draws=[(True, True, True), (False, True, False)])
@example(draws=[(False, False, True), (True, True, True)])
def test_a_chain_proves_exactly_when_one_declarer_is_the_only_writer(draws):
    proofs, refusals = measurement_proofs(_spec(_chain(draws)))
    declarers = [index for index, draw in enumerate(draws) if draw[0]]
    income = [index for index, draw in enumerate(draws) if draw[1]]
    universe = [index for index, draw in enumerate(draws) if draw[2]]
    proved = len(declarers) == 1 and income == universe == declarers
    assert (SUMMED_MEASUREMENT in proofs) is proved
    assert (SUMMED_MEASUREMENT in refusals) is not proved
    if proved:
        assert proofs[SUMMED_MEASUREMENT]["producers"] == {
            "toy.calibrate": f"toy.n{declarers[0]}"
        }


def test_a_calibration_endpoint_without_a_calibrate_kernel_is_a_root():
    """Only the endpoint clause of the roots finds this calibration node."""
    document = _graph()
    _nodes(document)["toy.calibrate"]["kernel"] = "toy.fit@1"
    proofs, refusals = measurement_proofs(_spec(document))
    assert refusals == {}
    assert proofs[SUMMED_MEASUREMENT]["producers"] == {"toy.calibrate": "toy.measure"}


@pytest.fixture(scope="module")
def composed_toy(tmp_path_factory):
    from test_support.microcosm_build.transport_composed import (
        make_composed_fixture,
    )

    return make_composed_fixture(tmp_path_factory.mktemp("composed"))


def _composed_spec(composed_toy, *, overwrite, expand=False):
    """The composed toy with summed rows and a producer upstream of calibration.

    ``transport.currency@1`` stands in for the producer's kernel, because
    composition checks the skeleton vocabulary and runs no kernel.
    """
    spec = copy.deepcopy(composed_toy.spec)
    base, profiles, context, intervals = summed_profile_fixture(GROUPS)
    references = spec["resources"]["target_references"]
    references["target_references"] += base["target_references"]
    for key, value in base["hierarchy"].items():
        references["hierarchy"].setdefault(key, {}).update(value)
    spec["resources"].update(
        reference_profiles=profiles,
        factor_context=context,
        profile_intervals=intervals,
    )
    graph = spec["resources"]["transport_graph"]
    nodes = _nodes(graph)
    calibration = graph["endpoints"]["calibration"]
    (create,) = (
        row["id"] for row in graph["nodes"] if row.get("structural") == "create"
    )
    opened = nodes[calibration]["base"]
    graph["nodes"].insert(
        1,
        {
            "id": "toy.measure",
            "kernel": "transport.currency@1",
            "population": create,
            "inputs": [{"entity": "person", "columns": ["age"]}],
            "outputs": [
                {"entity": "person", "column": INCOME_COLUMN, "dtype": "Int64"},
                {"entity": "person", "column": UNIVERSE_COLUMN, "dtype": "string"},
            ],
            "params": {"measurement_id": SUMMED_MEASUREMENT},
        },
    )
    if overwrite:
        graph["nodes"].insert(
            graph["nodes"].index(nodes[opened]) + 1,
            {
                **_rewriter("toy.overwrite", INCOME_COLUMN, "Int64", population=opened),
                "kernel": "transport.currency@1",
            },
        )
    if expand:
        graph["nodes"].insert(
            graph["nodes"].index(nodes[calibration]),
            {
                "id": "toy.expand",
                "kernel": "transport.boundary@1",
                "structural": "expand",
                "base": opened,
                "params": {"expand_cells": [["person", INCOME_COLUMN, "Int64"]]},
            },
        )
        nodes[calibration]["base"] = "toy.expand"
    nodes[calibration].setdefault("inputs", []).append(
        {"entity": "person", "columns": [INCOME_COLUMN, UNIVERSE_COLUMN]}
    )
    return spec, calibration


def test_composition_refuses_a_later_version_rewrite_of_a_measured_column(
    composed_toy,
):
    spec, calibration = _composed_spec(composed_toy, overwrite=False)
    assert measurement_proofs(spec)[0][SUMMED_MEASUREMENT]["producers"] == {
        calibration: "toy.measure"
    }
    compose_transport_graph(spec, composed_toy.config)
    spec, calibration = _composed_spec(composed_toy, overwrite=True)
    with pytest.raises(ValueError) as refused:
        compose_transport_graph(spec, composed_toy.config)
    message = str(refused.value)
    assert message.startswith("Transport spec is not activated:")
    assert f"awaits measurement:{SUMMED_MEASUREMENT}" in message
    assert (
        f"calibration node {calibration!r} has writers ['toy.measure', "
        f"'toy.overwrite'] of person.{INCOME_COLUMN} upstream"
    ) in message


def test_composition_refuses_an_expand_of_a_measured_column(composed_toy):
    spec, calibration = _composed_spec(composed_toy, overwrite=False, expand=True)
    with pytest.raises(ValueError) as refused:
        compose_transport_graph(spec, composed_toy.config)
    message = str(refused.value)
    assert message.startswith("Transport spec is not activated:")
    assert (
        f"calibration node {calibration!r} has writers ['toy.expand', "
        f"'toy.measure'] of person.{INCOME_COLUMN} upstream"
    ) in message


# ---------------------------------------------------------------------------
# A calibration node that a transport extension adds
# ---------------------------------------------------------------------------

EXTENSION_PREFIX = (
    f"Measurement {SUMMED_MEASUREMENT!r}: the composed graph proves no producer "
    "for extension calibration node 'toy.calibrate'; "
)


@pytest.mark.parametrize(
    "shape", [{}, {"expand_cells": (("person", "toy_other", "int64"),)}]
)
def test_the_compiled_graph_proves_an_extension_calibration_producer(
    monkeypatch, shape
):
    check = _forced_proof(monkeypatch, None)
    graph, compiled = _compiled_toy(**shape)
    check({}, graph, compiled, extension_nodes={"toy.calibrate"})


def test_a_skeleton_calibration_node_still_needs_its_declared_producer(monkeypatch):
    """The extension rule applies only to the nodes an extension adds."""
    check = _forced_proof(monkeypatch, None)
    graph, compiled = _compiled_toy()
    with pytest.raises(ValueError, match="and the declaration proved None"):
        check({}, graph, compiled)


def test_only_an_extension_calibration_node_leaves_the_declared_comparison(
    monkeypatch,
):
    """Extension ids of other nodes leave the skeleton's checks as they were."""
    check = _forced_proof(monkeypatch, "toy.measure")
    graph, compiled = _compiled_toy(second_writer=True)
    with pytest.raises(ValueError) as refused:
        check({}, graph, compiled, extension_nodes={"toy.filter", "toy.overwrite"})
    assert str(refused.value).startswith(
        f"Measurement {SUMMED_MEASUREMENT!r}: the compiled graph contradicts the "
        "declared proof; "
    )
    assert "writers ['toy.measure', 'toy.overwrite']" in str(refused.value)


@pytest.mark.parametrize(
    ("shape", "columns", "reason"),
    [
        ({"declared": False}, [INCOME_COLUMN], "0 nodes declaring"),
        ({"second_producer": True}, [INCOME_COLUMN], "2 nodes declaring"),
        (
            {},
            [INCOME_COLUMN, UNIVERSE_COLUMN],
            "its one declarer 'toy.measure' does not write "
            f"['person.{UNIVERSE_COLUMN}']",
        ),
        (
            {"expand_cells": (("person", INCOME_COLUMN, "int64"),)},
            [INCOME_COLUMN],
            f"writers ['toy.expand', 'toy.measure'] of person.{INCOME_COLUMN}",
        ),
    ],
)
def test_composition_refuses_an_extension_calibration_without_one_writer(
    monkeypatch, shape, columns, reason
):
    check = _forced_proof(monkeypatch, None, [f"person.{name}" for name in columns])
    graph, compiled = _compiled_toy(**shape)
    with pytest.raises(ValueError) as refused:
        check({}, graph, compiled, extension_nodes={"toy.calibrate"})
    assert str(refused.value).startswith(EXTENSION_PREFIX)
    assert reason in str(refused.value)


def test_an_extension_calibration_refuses_a_later_version_rewrite(monkeypatch):
    """The NZ hub's G7 in miniature: a filter version rewrites the income."""
    check = _forced_proof(monkeypatch, None)
    graph, compiled = _compiled_toy(second_writer=True)
    with pytest.raises(ValueError) as refused:
        check({}, graph, compiled, extension_nodes={"toy.calibrate"})
    assert str(refused.value) == (
        f"{EXTENSION_PREFIX}measurement {SUMMED_MEASUREMENT!r}: calibration node "
        "'toy.calibrate' has writers ['toy.measure', 'toy.overwrite'] of "
        f"person.{INCOME_COLUMN} upstream; only its producer 'toy.measure' may "
        "write it."
    )


def _compiled_chain(draws):
    """:func:`_chain`'s draws as a compiled graph.

    A version holds one owner per column, so a node that writes a column an
    earlier node wrote opens a ``filter`` version and rewrites it there.
    """
    from microcosm.graph import (
        Graph,
        Node,
        Owned,
        Slice,
        SourceRef,
        StructuralDelta,
        compile_graph,
    )

    nodes = [
        Node(
            id="toy.create",
            kernel="transport.create@1",
            structural=StructuralDelta.CREATE,
            sources=("toy_donor",),
            outputs=(Owned("person", "person_id", "int64"),),
        )
    ]
    population, written, previous = "toy.create", set(), None
    for index, (declares, income, universe) in enumerate(draws):
        measured = [
            (column, dtype)
            for column, dtype, wanted in (
                (INCOME_COLUMN, "int64", income),
                (UNIVERSE_COLUMN, "string", universe),
            )
            if wanted
        ]
        if written & {column for column, _ in measured}:
            nodes.append(
                Node(
                    id=f"toy.v{index}",
                    kernel="transport.boundary@1",
                    structural=StructuralDelta.FILTER,
                    base=population,
                )
            )
            population = f"toy.v{index}"
        nodes.append(
            Node(
                id=f"toy.n{index}",
                kernel="toy.chain@1",
                population=population,
                params={"measurement_id": SUMMED_MEASUREMENT} if declares else {},
                inputs=() if previous is None else (Slice("person", (previous,)),),
                outputs=(
                    *(
                        Owned("person", column, dtype, rewrite=column in written)
                        for column, dtype in measured
                    ),
                    Owned("person", f"toy_step_{index}", "bool"),
                ),
            )
        )
        written |= {column for column, _ in measured}
        previous = f"toy_step_{index}"
    nodes.append(
        Node(
            id="toy.calibrate",
            kernel="calibrate.ordered_adam@1",
            population=population,
            inputs=(Slice("person", (previous,)),),
        )
    )
    graph = Graph(
        country="toy",
        sources=(SourceRef("toy_donor", "toy-codec-v1"),),
        nodes=tuple(nodes),
    )
    return graph, compile_graph(graph)


@given(
    draws=st.lists(
        st.tuples(st.booleans(), st.booleans(), st.booleans()),
        min_size=1,
        max_size=4,
    )
)
@settings(
    max_examples=60,
    database=None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
# One declarer alone; without the universe; with a second writer after it and
# before it; with two declarers.
@example(draws=[(True, True, True)])
@example(draws=[(True, True, False)])
@example(draws=[(True, True, True), (False, True, False)])
@example(draws=[(False, False, True), (True, True, True)])
@example(draws=[(True, True, False), (True, False, True)])
def test_an_extension_chain_proves_exactly_when_the_declared_chain_does(draws):
    """The declared proof and the compiled extension proof agree, refusal for refusal.

    The declared check reads :func:`_chain`'s rows. The compiled check reads
    the same draws compiled, as for a calibration node that an extension adds.
    """
    proofs, refusals = measurement_proofs(_spec(_chain(draws)))
    graph, compiled = _compiled_chain(draws)
    columns = [f"person.{INCOME_COLUMN}", f"person.{UNIVERSE_COLUMN}"]
    with pytest.MonkeyPatch.context() as patch:
        check = _forced_proof(patch, None, columns)
        try:
            check({}, graph, compiled, extension_nodes={"toy.calibrate"})
        except ValueError as error:
            refusal = str(error)
        else:
            refusal = None
    assert (refusal is None) is (SUMMED_MEASUREMENT in proofs)
    if refusal is not None:
        reason = refusals[SUMMED_MEASUREMENT]
        missing = re.fullmatch(
            r"producer ('[^']+') (does not write \[.*\]) on the interval "
            "table's entity",
            reason,
        )
        if missing is not None:
            reason = f"its one declarer {missing[1]} {missing[2]}"
        assert refusal == f"{EXTENSION_PREFIX}{reason}."


@pytest.fixture(scope="module")
def nz_toy(tmp_path_factory):
    """NZ's synthetic graph, whose V1, V3 and VHF scenarios each calibrate."""
    from test_support.microcosm_build.transport_nz_toy import make_nz_toy_fixture

    return make_nz_toy_fixture(tmp_path_factory.mktemp("nz-toy") / "fixture")


def _nz_proof(nz_toy):
    proofs, refusals = measurement_proofs(nz_toy.spec)
    assert refusals == {}
    ((binding, proof),) = proofs.items()
    return binding, proof


def _v3_without_declaration(rows, nodes, binding, columns):
    """M1: V3's stand-in keeps its outputs and drops the measurement id."""
    del rows["nz.v.V3.income.standin"]["params"]["measurement_id"]


def _v3_second_declarer(rows, nodes, binding, columns):
    """M2: V3's earnings node declares the measurement too."""
    rows["nz.v.V3.earnings.cents"]["params"]["measurement_id"] = binding


def _v3_universe_elsewhere(rows, nodes, binding, columns):
    """M4: V3's universe column moves from the stand-in to the earnings node."""
    standin = rows["nz.v.V3.income.standin"]
    (moved,) = [row for row in standin["outputs"] if row["column"] == columns[1]]
    standin["outputs"].remove(moved)
    rows["nz.v.V3.earnings.cents"]["outputs"].append(moved)


def _v3_later_version_rewrite(rows, nodes, binding, columns):
    """G7: a filter version inside V3 rewrites both measured columns.

    Preparation, the compiled targets and the problem move into it, and V3's
    calibration opens from it.
    """
    standin = rows["nz.v.V3.income.standin"]
    dtypes = {row["column"]: row["dtype"] for row in standin["outputs"]}
    moved = [
        rows[name]
        for name in (
            "nz.v.V3.calibration.prepare",
            "nz.v.V3.targets.compile",
            "nz.v.V3.targets.problem",
        )
    ]
    for row in moved:
        nodes.remove(row)
        row["population"] = "nz.v.V3.review"
    rows["nz.v.V3.calibrate"]["base"] = "nz.v.V3.review"
    position = nodes.index(standin) + 1
    nodes[position:position] = [
        {
            "id": "nz.v.V3.review",
            "kernel": "transport.boundary@1",
            "structural": "filter",
            "base": "nz.v.V3",
        },
        {
            "id": "nz.v.V3.review.rewrite",
            "kernel": "transport.currency@1",
            "population": "nz.v.V3.review",
            "inputs": [{"entity": "person", "columns": list(columns)}],
            "outputs": [
                {
                    "entity": "person",
                    "column": column,
                    "dtype": dtypes[column],
                    "rewrite": True,
                }
                for column in columns
            ],
        },
        *moved,
    ]


def test_composition_proves_each_extension_calibration_producer(nz_toy):
    """The declaration names the skeleton's calibration node; compose proves V3's."""
    _, proof = _nz_proof(nz_toy)
    graph = compose_transport_graph(copy.deepcopy(nz_toy.spec), nz_toy.config)
    calibration = {
        node.id for node in graph.nodes if node.kernel.startswith("calibrate.")
    }
    assert "nz.v.V3.calibrate" in calibration - set(proof["producers"])


@pytest.mark.parametrize(
    ("mutant", "reason"),
    [
        (
            _v3_without_declaration,
            "0 nodes declaring measurement_id {binding!r} sit upstream of "
            "calibration node 'nz.v.V3.calibrate'",
        ),
        (
            _v3_second_declarer,
            "2 nodes declaring measurement_id {binding!r} sit upstream of "
            "calibration node 'nz.v.V3.calibrate'",
        ),
        (
            _v3_universe_elsewhere,
            "its one declarer 'nz.v.V3.income.standin' does not write "
            "['person.{universe}']",
        ),
        (
            _v3_later_version_rewrite,
            "calibration node 'nz.v.V3.calibrate' has writers "
            "['nz.v.V3.income.standin', 'nz.v.V3.review.rewrite'] of "
            "person.{income} upstream; only its producer 'nz.v.V3.income.standin' "
            "may write it",
        ),
    ],
)
def test_composition_refuses_a_scenario_calibration_the_declaration_never_sees(
    nz_toy, mutant, reason
):
    binding, proof = _nz_proof(nz_toy)
    income, universe = (name.split(".", 1)[1] for name in proof["columns"])
    spec = copy.deepcopy(nz_toy.spec)
    (scenario,) = (
        row for row in spec["resources"]["scenarios"]["scenarios"] if row["id"] == "V3"
    )
    rows = {row["id"]: row for row in scenario["nodes"]}
    mutant(rows, scenario["nodes"], binding, (income, universe))
    # Scenario rows sit outside the declared graph, so the declaration still
    # proves the measurement and only the extension pass can refuse.
    assert binding in measurement_proofs(spec)[0]
    with pytest.raises(ValueError) as refused:
        compose_transport_graph(spec, nz_toy.config)
    message = str(refused.value)
    assert message.startswith(
        f"Measurement {binding!r}: the composed graph proves no producer for "
        "extension calibration node 'nz.v.V3.calibrate'; "
    )
    assert reason.format(binding=binding, income=income, universe=universe) in message
