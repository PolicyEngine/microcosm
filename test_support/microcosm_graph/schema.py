"""Compiler-only graph fixtures for schema and Orrery adapter tests."""

from __future__ import annotations

from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Graph,
    Node,
    Owned,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightUpdate,
    compile_graph,
)


def compiled_graph():
    """Return a graph covering carried, rewritten, and structural fields."""

    summary_type = ArtifactType("invented.summary", 1)
    graph = Graph(
        "invented",
        sources=(SourceRef("survey", "unused@1", "Recorded survey tables"),),
        nodes=(
            Node(
                "base",
                "unused.create@1",
                structural=StructuralDelta.CREATE,
                sources=("survey",),
                outputs=(
                    Owned("person", "age", "int64"),
                    Owned("person", "keep", "boolean"),
                ),
                artifact_outputs=(ArtifactOutput("summary", summary_type),),
                description="Load the declared source tables",
            ),
            Node(
                "filtered",
                "unused.filter@1",
                structural=StructuralDelta.FILTER,
                base="base",
                inputs=(Slice("person", ("keep",)),),
                description="Retain selected people",
            ),
            Node(
                "rewrite",
                "unused.rewrite@1",
                population="filtered",
                inputs=(Slice("person", ("age", "keep"), rows="keep"),),
                outputs=(
                    Owned(
                        "person",
                        "age",
                        "int64",
                        rows="keep",
                        rewrite=True,
                    ),
                ),
                description="Replace age for selected people",
                citation="Example method",
            ),
            Node(
                "expanded",
                "unused.expand@1",
                structural=StructuralDelta.EXPAND,
                base="filtered",
                inputs=(Slice("person", ("age",)),),
            ),
            Node(
                "reweighted",
                "unused.reweight@1",
                structural=StructuralDelta.REWEIGHT,
                base="expanded",
                weights=WeightUpdate("person", "design", "normalize sample weights"),
                mass="declared",
                artifact_inputs=(
                    ArtifactInput("source_summary", "base", "summary", summary_type),
                ),
            ),
        ),
    )
    return compile_graph(graph)


def compiled_default_population_graph():
    """Return a legal graph whose ordinary node omits its sole population."""

    return compile_graph(
        Graph(
            "invented",
            sources=(SourceRef("survey", "unused@1"),),
            nodes=(
                Node(
                    "base",
                    "unused.create@1",
                    structural=StructuralDelta.CREATE,
                    sources=("survey",),
                    outputs=(Owned("person", "age", "int64"),),
                ),
                Node(
                    "consumer",
                    "unused.consume@1",
                    inputs=(Slice("person", ("age",)),),
                ),
            ),
        )
    )
