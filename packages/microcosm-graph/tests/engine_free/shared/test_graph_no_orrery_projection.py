"""Microcosm keeps no copy of the Orrery projection.

The collector builds the Orrery graph files from a build's saved schema,
execution overlay and upstream files, with its own copy of the projection.
"""

import importlib.util

import pytest

import microcosm.graph as graph


@pytest.mark.parametrize(
    "module_name", ["microcosm.graph.orrery", "microcosm.graph.orrery_evidence"]
)
def test_the_graph_package_has_no_projection_module(module_name):
    assert importlib.util.find_spec(module_name) is None


def test_the_graph_package_exports_no_projection_function():
    projection_functions = {"orrery_document", "orrery_json"}

    assert not projection_functions & set(graph.__all__)
    assert not any(hasattr(graph, name) for name in projection_functions)
