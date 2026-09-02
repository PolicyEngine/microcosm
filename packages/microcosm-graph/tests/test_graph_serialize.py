"""Canonical, lossless graph declaration JSON contracts."""

from __future__ import annotations

import json

import pytest

import microcosm.graph as graph_api
from microcosm.graph import (
    Graph,
    Node,
    Owned,
    Ownership,
    Slice,
    SourceRef,
    StructuralDelta,
    WeightTransition,
    graph_from_json,
    graph_to_json,
)


def _graph() -> Graph:
    source = Node(
        "survey",
        "source.csv@1",
        outputs=(
            Owned("person", "age", "int64"),
            Owned("person", "selected", "boolean"),
        ),
        params={"columns": ("age", "selected"), "revision": 1},
        structural=StructuralDelta.CREATE,
        sources=("fixture",),
        description="load fixture",
        citation="synthetic",
    )
    absent = Node(
        "absent",
        "absent@1",
        inputs=(Slice("person", ("age", "selected"), rows="selected"),),
        outputs=(
            Owned(
                "person",
                "score",
                "float64",
                rows="selected",
                ownership=Ownership.ABSENT,
            ),
        ),
        params={"nested": (True, None, (2.5, "x"))},
        population="survey",
    )
    pool = Node(
        "pool",
        "reweight@1",
        inputs=(Slice("person", ("age",)),),
        structural=StructuralDelta.REWEIGHT,
        base="survey",
        weights=WeightTransition("person", "importance", mass="free"),
        mass="free",
    )
    return Graph(
        "toy",
        (SourceRef("fixture", "csv-tables", description="pinned table"),),
        (source, absent, pool),
    )


def test_graph_json_round_trip_is_lossless_and_canonical() -> None:
    graph = _graph()
    text = graph_to_json(graph)
    restored = graph_from_json(text)

    assert restored == graph
    assert graph_to_json(restored) == text
    assert ": " not in text
    assert ", " not in text
    assert not text.endswith("\n")
    assert restored.nodes[0].structural is StructuralDelta.CREATE
    assert restored.nodes[1].outputs[0].ownership is Ownership.ABSENT
    assert restored.nodes[1].params["nested"] == (True, None, (2.5, "x"))
    assert restored.nodes[2].weights == WeightTransition(
        "person", "importance", mass="free"
    )

    payload = json.loads(text)
    assert payload["sources"][0]["codec"] == "csv-tables"
    assert payload["nodes"][1]["params"]["nested"] == [True, None, [2.5, "x"]]
    assert list(payload) == ["country", "nodes", "sources"]


def test_graph_from_json_rejects_shape_enum_and_parameter_drift() -> None:
    payload = json.loads(graph_to_json(_graph()))

    with_extra = dict(payload, schema_version=1)
    with pytest.raises(ValueError, match="graph fields"):
        graph_from_json(json.dumps(with_extra))

    payload["nodes"][0]["structural"] = "invented"
    with pytest.raises(ValueError):
        graph_from_json(json.dumps(payload))

    payload = json.loads(graph_to_json(_graph()))
    payload["nodes"][0]["params"]["revision"] = {"not": "a Param"}
    with pytest.raises(TypeError, match="legal graph parameter"):
        graph_from_json(json.dumps(payload))

    with pytest.raises(ValueError, match="non-finite"):
        graph_from_json(graph_to_json(_graph()).replace("2.5", "NaN"))


def test_graph_serializers_are_public_without_replacing_frozen_graph() -> None:
    assert graph_api.Graph is Graph
    assert graph_api.graph_to_json is graph_to_json
    assert graph_api.graph_from_json is graph_from_json
