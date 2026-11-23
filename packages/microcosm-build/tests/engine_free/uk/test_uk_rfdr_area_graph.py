"""The post-geography Rural Fuel Duty Relief node in the UK full-build graph (#1418)."""

from __future__ import annotations

import functools
import json

import pytest

from microcosm.build.uk_runtime.atomic_area_support import (
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig
from microcosm.build.uk_runtime.graph_population import _rfdr_area_nodes
from microcosm.build.uk_runtime.rfdr_area_layer import (
    RFDR_AREA_LAYER_SOURCE,
    RFDR_COLUMN,
)
from microcosm.graph import ContentStore, compile_graph, run_graph
from microcosm.graph.errors import NodeRejectedError
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    TOY_RFDR_AREAS,
    toy_rfdr_area_layer,
    toy_support_sources,
    write_toy_rfdr_area_layer,
    write_toy_supports,
)
from test_support.microcosm_build.uk_full_population_graph import (
    graph_and_registry,
)
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)


def _atomic(k: int, tmp_path, mode: str = "area", *, full_scale: bool = True):
    payloads, paths = write_toy_supports(tmp_path / "supports")
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    graph, registry = graph_and_registry(
        k,
        geography_assignment="atomic",
        definition=definition,
        rfdr_localisation_mode=mode,
        full_scale=full_scale,
    )
    return graph, registry, paths


@pytest.mark.parametrize("k", [1, 3])
def test_rfdr_node_flags_each_clone_by_its_area_and_replays(
    k, toy_ladder, tmp_path
) -> None:
    _, ladder_path = toy_ladder
    graph, registry, paths = _atomic(k, tmp_path)
    compiled = compile_graph(graph)
    order = compiled.order
    # The node reads the assigned area and precedes the geography gates.
    assert order.index("uk.full.geography.assign") < order.index(
        "uk.full.geography.rfdr"
    )
    assert order.index("uk.full.geography.rfdr") < order.index("uk.full.geography.gate")
    sources = {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
        **write_toy_rfdr_area_layer(tmp_path / "rfdr"),
    }
    store = ContentStore(tmp_path / "store")
    first = run_graph(compiled, sources=sources, store=store, kernels=registry)
    household = first.population("uk.full.expand").table("household")
    flagged = {code for codes in TOY_RFDR_AREAS.values() for code in codes}
    expected = household["atomic_area_code"].astype(str).isin(flagged)
    assert expected.any() and not expected.all()
    assert household[RFDR_COLUMN].astype(bool).tolist() == expected.tolist()
    # Northern Ireland has no area in the scheme.
    ni = household["atomic_area_system"].astype(str) == "uk_ni_data_zone_2021"
    assert not household.loc[ni, RFDR_COLUMN].astype(bool).any()
    receipt = first.nodes["uk.full.geography.rfdr"].receipt
    assert set(receipt["rows_by_system"]) == {
        "uk_ew_output_area_2021",
        "uk_ni_data_zone_2021",
        "uk_scotland_output_area_2022",
    }
    _, fresh, _ = _atomic(k, tmp_path / "fresh")
    replay = run_graph(
        compiled, sources=sources, store=store, kernels=fresh, resume="require"
    )
    assert all(node.hit for node in replay.nodes.values())


def test_none_mode_adds_no_rfdr_node(tmp_path) -> None:
    graph, _, _ = _atomic(1, tmp_path, mode="none")
    assert "uk.full.geography.rfdr" not in compile_graph(graph).order
    assert "uk_rfdr_area_layer" not in {source.name for source in graph.sources}


def test_area_rfdr_flags_need_the_atomic_assignment() -> None:
    with pytest.raises(ValueError, match="atomic geography assignment"):
        graph_and_registry(
            1, geography_assignment="legacy", rfdr_localisation_mode="area"
        )
    with pytest.raises(ValueError, match="RFDR localisation must be one of"):
        graph_and_registry(1, rfdr_localisation_mode="region")


def test_the_node_refuses_a_pool_without_its_columns() -> None:
    cells = {("household", "atomic_area_system"): "string"}
    with pytest.raises(ValueError, match="'atomic_area_code'"):
        _rfdr_area_nodes(cells, common={"time_period": "2023"})


def test_only_a_full_scale_build_refuses_a_system_with_none_flagged() -> None:
    # Vahid's review and the independent review of microcosm#1420: a sampled
    # rung can miss the 41 English and Welsh output areas by chance.
    cells = {
        ("household", "atomic_area_system"): "string",
        ("household", "atomic_area_code"): "string",
    }
    ((full,), _) = _rfdr_area_nodes(dict(cells), common={"time_period": "2023"})
    ((sampled,), _) = _rfdr_area_nodes(
        dict(cells), common={"time_period": "2023"}, strict=False
    )
    assert full.params["strict_flags"] is True
    assert sampled.params["strict_flags"] is False


def test_config_follows_the_geography_assignment() -> None:
    assert UKFullBuildConfig(calibration_year=2025).rfdr_localisation == "area"
    legacy = UKFullBuildConfig(
        calibration_year=2025,
        geography_assignment="legacy",
        property_localisation="none",
    )
    assert legacy.rfdr_localisation == "none"
    with pytest.raises(ValueError, match="atomic geography assignment"):
        UKFullBuildConfig(
            calibration_year=2025,
            geography_assignment="legacy",
            property_localisation="none",
            rfdr_localisation="area",
        )


@pytest.mark.parametrize("full_scale", [True, False])
def test_the_rfdr_node_is_strict_exactly_at_full_scale(full_scale, tmp_path) -> None:
    """The re-review of microcosm#1420, C: the graph sets the switch from the
    build's own scale, not only a direct call."""

    graph, _, _ = _atomic(1, tmp_path, full_scale=full_scale)
    assert graph.node("uk.full.geography.rfdr").params["strict_flags"] is full_scale


@pytest.mark.parametrize("full_scale", [True, False])
def test_the_rfdr_kernel_refuses_an_unflagged_system_only_at_full_scale(
    full_scale, toy_ladder, tmp_path
) -> None:
    """The re-review of microcosm#1420, C: the kernel honours the node's
    switch. England and Wales's flagged area is one no household is placed
    in: a full-scale run refuses, a sampled rung flags none there and runs."""

    _, ladder_path = toy_ladder
    graph, registry, paths = _atomic(1, tmp_path, full_scale=full_scale)
    layer = toy_rfdr_area_layer()
    layer["areas"]["uk_ew_output_area_2021"] = ["E00099999"]
    layer_path = tmp_path / "unplaced-rfdr-area-layer.json"
    layer_path.write_text(json.dumps(layer), encoding="utf-8")
    sources = {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
        RFDR_AREA_LAYER_SOURCE: layer_path,
    }
    run = functools.partial(
        run_graph,
        compile_graph(graph),
        sources=sources,
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
    )
    if full_scale:
        with pytest.raises(
            NodeRejectedError, match="flags none of the 2 uk_ew_output_area_2021"
        ):
            run()
        return
    household = run().population("uk.full.expand").table("household")
    england_wales = (
        household["atomic_area_system"].astype(str) == "uk_ew_output_area_2021"
    )
    assert england_wales.any()
    assert not household.loc[england_wales, RFDR_COLUMN].astype(bool).any()
