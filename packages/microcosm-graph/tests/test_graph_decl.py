"""Compile-time contracts of the frozen declaration interface.

These are the halves of charter properties B1, C1, C3, and D4 that hold
before any executor exists: ownership is total and exclusive at compile
time, declaration order carries no meaning, predecessors are exactly the
owners of declared inputs, and non-finite parameters are rejected.
"""

from __future__ import annotations

import pytest

from microcosm.graph import (
    Graph,
    GraphError,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
)

SRC = SourceRef("survey", "frame-h5", description="a synthetic survey")
CREATE = Node(
    "survey",
    "source.frame@1",
    sources=("survey",),
    structural=StructuralDelta.CREATE,
    outputs=(
        Owned("person", "age", "int64"),
        Owned("person", "income", "float64"),
        Owned("person", "is_adult", "boolean"),
    ),
)


def _fit(node_id: str, predictors: tuple[str, ...], target: str) -> Node:
    return Node(
        node_id,
        "fit.qrf@1",
        inputs=(Slice("person", predictors),),
        outputs=(Owned("person", target, "float64"),),
        params={"n_estimators": 100},
    )


def _with(node: Node, **overrides: object) -> Node:
    return Node(**{**{f: getattr(node, f) for f in node.normative()}, **overrides})


def test_order_is_canonical_under_declaration_permutation() -> None:
    a = _fit("fit_a", ("age",), "a")
    b = _fit("fit_b", ("age", "a"), "b")
    c = _fit("fit_c", ("age",), "c")
    forward = compile_graph(Graph("toy", (SRC,), (CREATE, a, b, c)))
    shuffled = compile_graph(Graph("toy", (SRC,), (c, b, CREATE, a)))
    assert forward.order == shuffled.order == ("survey", "fit_a", "fit_c", "fit_b")


def test_predecessors_are_exactly_the_owners_of_declared_inputs() -> None:
    a = _fit("fit_a", ("age",), "a")
    b = _fit("fit_b", ("age", "a"), "b")
    c = _fit("fit_c", ("age",), "c")
    compiled = compile_graph(Graph("toy", (SRC,), (CREATE, a, b, c)))
    assert compiled.predecessors["fit_b"] == ("fit_a", "survey")
    assert compiled.predecessors["fit_c"] == ("survey",)
    assert compiled.owners[("survey", "person", "b")] == "fit_b"
    assert compiled.owners[("survey", "person", "age")] == "survey"


def test_removing_a_leaf_changes_no_other_predecessor_set() -> None:
    a = _fit("fit_a", ("age",), "a")
    b = _fit("fit_b", ("age", "a"), "b")
    c = _fit("fit_c", ("age",), "c")
    full = compile_graph(Graph("toy", (SRC,), (CREATE, a, b, c)))
    without_c = compile_graph(Graph("toy", (SRC,), (CREATE, a, b)))
    for node_id in ("survey", "fit_a", "fit_b"):
        assert full.predecessors[node_id] == without_c.predecessors[node_id]


def test_two_owners_of_one_cell_are_rejected() -> None:
    a = _fit("fit_a", ("age",), "a")
    dup = _fit("fit_dup", ("age",), "a")
    with pytest.raises(GraphError, match="owned by both"):
        compile_graph(Graph("toy", (SRC,), (CREATE, a, dup)))


def test_reading_an_unowned_column_is_rejected() -> None:
    orphan = _fit("fit_x", ("nobody_makes_this",), "x")
    with pytest.raises(GraphError, match="no node owns"):
        compile_graph(Graph("toy", (SRC,), (CREATE, orphan)))


def test_create_nodes_declare_what_they_load() -> None:
    with pytest.raises(GraphError, match="must declare every column"):
        Node(
            "survey",
            "source.frame@1",
            sources=("survey",),
            structural=StructuralDelta.CREATE,
        )


def test_cycles_are_rejected() -> None:
    a = _fit("fit_a", ("b",), "a")
    b = _fit("fit_b", ("a",), "b")
    with pytest.raises(GraphError, match="Cycle"):
        compile_graph(Graph("toy", (SRC,), (CREATE, a, b)))


def test_a_node_cannot_own_a_column_it_reads() -> None:
    with pytest.raises(GraphError, match="both reads and owns"):
        _fit("fit_a", ("a",), "a")


def test_non_finite_parameters_are_rejected() -> None:
    with pytest.raises(GraphError, match="not finite"):
        Node("bad", "k@1", params={"p": float("nan")})


def test_callable_parameters_are_rejected() -> None:
    with pytest.raises(GraphError, match="parameters are"):
        Node("bad", "k@1", params={"p": len})


def test_row_masks_must_be_declared_inputs() -> None:
    with pytest.raises(GraphError, match="must be one of the node's input columns"):
        Node(
            "masked",
            "k@1",
            inputs=(Slice("person", ("age",), rows="is_adult"),),
            outputs=(Owned("person", "x", "float64"),),
        )
    masked = Node(
        "masked",
        "k@1",
        inputs=(Slice("person", ("age", "is_adult"), rows="is_adult"),),
        outputs=(Owned("person", "x", "float64", rows="is_adult"),),
    )
    assert compile_graph(Graph("toy", (SRC,), (CREATE, masked))).order == (
        "survey",
        "masked",
    )


def test_descriptive_fields_are_outside_the_normative_projection() -> None:
    node = _fit("fit_a", ("age",), "a")
    described = _with(node, description="words", citation="a source")
    assert node.normative() == described.normative()
    assert "description" not in node.normative()


def test_graph_needs_a_create_node() -> None:
    with pytest.raises(GraphError, match="CREATE"):
        compile_graph(Graph("toy", (SRC,), (_fit("fit_a", ("age",), "a"),)))


def test_filtered_versions_route_carried_columns_through_the_filter() -> None:
    subset = Node(
        "adults",
        "select.rows@1",
        structural=StructuralDelta.FILTER,
        base="survey",
        inputs=(Slice("person", ("is_adult",)),),
    )
    orphan = _fit("fit_a", ("age",), "a")
    with pytest.raises(GraphError, match="omits population"):
        compile_graph(Graph("toy", (SRC,), (CREATE, subset, orphan)))
    placed = _with(orphan, population="adults")
    downstream = _with(_fit("fit_b", ("age", "a"), "b"), population="adults")
    compiled = compile_graph(Graph("toy", (SRC,), (CREATE, subset, placed, downstream)))
    assert compiled.predecessors["adults"] == ("survey",)
    # ``age`` reaches fit_a through the filter's artifact, so the filter is the
    # predecessor; the CREATE node is reached only transitively.
    assert compiled.predecessors["fit_a"] == ("adults",)
    assert compiled.predecessors["fit_b"] == ("adults", "fit_a")
    assert compiled.versions["fit_a"] == "adults"
    assert compiled.order == ("survey", "adults", "fit_a", "fit_b")


def test_structural_nodes_depend_on_every_node_of_their_base() -> None:
    a = _fit("fit_a", ("age",), "a")
    subset = Node(
        "adults",
        "select.rows@1",
        structural=StructuralDelta.FILTER,
        base="survey",
        inputs=(Slice("person", ("is_adult",)),),
    )
    compiled = compile_graph(
        Graph("toy", (SRC,), (CREATE, _with(a, population="survey"), subset))
    )
    assert compiled.predecessors["adults"] == ("fit_a", "survey")


def test_rewrites_own_a_carried_column_in_a_versioned_population() -> None:
    subset = Node(
        "adults",
        "select.rows@1",
        structural=StructuralDelta.FILTER,
        base="survey",
        inputs=(Slice("person", ("is_adult",)),),
    )
    rewriter = Node(
        "recode_income",
        "k@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "income", "float64", rewrite=True),),
        population="adults",
    )
    reader = Node(
        "uses_income",
        "k@1",
        inputs=(Slice("person", ("income",)),),
        outputs=(Owned("person", "y", "float64"),),
        population="adults",
    )
    compiled = compile_graph(Graph("toy", (SRC,), (CREATE, subset, rewriter, reader)))
    assert compiled.owners[("adults", "person", "income")] == "recode_income"
    assert compiled.predecessors["recode_income"] == ("adults",)
    assert compiled.predecessors["uses_income"] == ("adults", "recode_income")


def test_rewrites_need_a_base_with_a_matching_declared_dtype() -> None:
    in_create = Node(
        "recode_income",
        "k@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "income", "float64", rewrite=True),),
    )
    with pytest.raises(GraphError, match="needs a version with a base"):
        compile_graph(Graph("toy", (SRC,), (CREATE, in_create)))

    subset = Node(
        "adults",
        "select.rows@1",
        structural=StructuralDelta.FILTER,
        base="survey",
        inputs=(Slice("person", ("is_adult",)),),
    )
    wrong_dtype = Node(
        "recode_income",
        "k@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "income", "string", rewrite=True),),
        population="adults",
    )
    with pytest.raises(GraphError, match="incumbent is declared 'float64'"):
        compile_graph(Graph("toy", (SRC,), (CREATE, subset, wrong_dtype)))

    unknown = Node(
        "recode_ghost",
        "k@1",
        inputs=(Slice("person", ("age",)),),
        outputs=(Owned("person", "ghost", "float64", rewrite=True),),
        population="adults",
    )
    with pytest.raises(GraphError, match="which no node defines"):
        compile_graph(Graph("toy", (SRC,), (CREATE, subset, unknown)))


def test_a_filter_cannot_read_a_column_nobody_defines() -> None:
    subset = Node(
        "adults",
        "select.rows@1",
        structural=StructuralDelta.FILTER,
        base="survey",
        inputs=(Slice("person", ("ghost",)),),
    )
    with pytest.raises(GraphError, match="no node owns"):
        compile_graph(Graph("toy", (SRC,), (CREATE, subset)))


def _expand(node_id: str, **overrides: object) -> Node:
    return Node(
        node_id,
        "clone.persons@1",
        base="survey",
        structural=StructuralDelta.EXPAND,
        mass="free",
        **overrides,  # type: ignore[arg-type]
    )


def test_entrants_are_an_expand_declaration_that_cannot_conserve_mass() -> None:
    """Amendment 11: entrants add rows without lineage, so they add mass."""
    admitted = _expand("cohort", entrants=True)
    assert admitted.entrants is True
    assert "entrants" in admitted.normative()
    with pytest.raises(GraphError, match="cannot declare mass='conserve'"):
        _expand("cohort", entrants=True, mass="conserve")
    with pytest.raises(GraphError, match="only an EXPAND node"):
        Node("fit", "fit.qrf@1", entrants=True)
    with pytest.raises(GraphError, match="must be a boolean"):
        _expand("cohort", entrants="yes")


def test_mass_partition_must_be_declared_by_every_create_node() -> None:
    """Amendment 12: partitions exist from the first version, with a partition dtype."""
    periodic = Node(
        "survey",
        "source.frame@1",
        sources=("survey",),
        structural=StructuralDelta.CREATE,
        outputs=(Owned("person", "age", "int64"), Owned("person", "period", "int64")),
    )
    graph = Graph("toy", (SRC,), (periodic,), mass_partition=("person", "period"))
    compiled = compile_graph(graph)
    assert compiled.graph.normative() == {"mass_partition": ("person", "period")}
    assert Graph("toy", (SRC,), (periodic,)).normative() == {"mass_partition": None}
    with pytest.raises(GraphError, match="does not declare"):
        compile_graph(
            Graph("toy", (SRC,), (CREATE,), mass_partition=("person", "period"))
        )
    floating = Node(
        "survey",
        "source.frame@1",
        sources=("survey",),
        structural=StructuralDelta.CREATE,
        outputs=(Owned("person", "period", "float64"),),
    )
    with pytest.raises(GraphError, match="partition column must be one of"):
        compile_graph(
            Graph("toy", (SRC,), (floating,), mass_partition=("person", "period"))
        )
    with pytest.raises(GraphError, match="pair of strings"):
        Graph("toy", (SRC,), (periodic,), mass_partition=("person",))  # type: ignore[arg-type]
