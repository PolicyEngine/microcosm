"""The post-geography area-BRMA node in the UK full-build graph (microcosm#1262)."""

from __future__ import annotations

import pytest

from microcosm.build.uk_runtime.atomic_area_support import (
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig
from microcosm.build.uk_runtime.graph_population import _brma_area_nodes
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    TOY_AREA_BRMAS,
    toy_support_sources,
    write_toy_brma_area_layer,
    write_toy_supports,
)
from test_support.microcosm_build.uk_full_population_graph import (
    graph_and_registry,
)
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)


def _atomic(k: int, tmp_path, mode: str = "area"):
    payloads, paths = write_toy_supports(tmp_path / "supports")
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    graph, registry = graph_and_registry(
        k,
        geography_assignment="atomic",
        definition=definition,
        brma_localisation_mode=mode,
    )
    return graph, registry, paths, definition


@pytest.mark.parametrize("k", [1, 3])
def test_brma_node_reads_each_clone_area_and_replays(k, toy_ladder, tmp_path) -> None:
    _, ladder_path = toy_ladder
    graph, registry, paths, definition = _atomic(k, tmp_path)
    compiled = compile_graph(graph)
    order = compiled.order
    # The node reads the derived area codes and precedes the geography gates.
    assert order.index("uk.full.geography.derive") < order.index(
        "uk.full.geography.brma"
    )
    assert order.index("uk.full.geography.brma") < order.index("uk.full.geography.gate")
    assert order.index("uk.full.geography.brma") < order.index("uk.full.geography_gate")
    sources = {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
        **write_toy_brma_area_layer(tmp_path / "brma"),
    }
    store = ContentStore(tmp_path / "store")
    first = run_graph(compiled, sources=sources, store=store, kernels=registry)
    household = first.population("uk.full.expand").table("household")
    receipt = first.nodes["uk.full.geography.brma"].receipt
    system = household["atomic_area_system"].astype(str)
    great_britain = system != "uk_ni_data_zone_2021"
    for area_column, rows in (
        ("lsoa_code", system == "uk_ew_output_area_2021"),
        ("oa_code", system == "uk_scotland_output_area_2022"),
    ):
        for code, brma in zip(
            household.loc[rows, area_column].astype(str),
            household.loc[rows, "brma"].astype(str),
            strict=True,
        ):
            assert brma == TOY_AREA_BRMAS.get(code, "MAIDSTONE")
    # Northern Ireland keeps the spine's regional draw.
    assert set(household.loc[~great_britain, "brma"].astype(str)) == {"BELFAST"}
    assert "MAIDSTONE" not in set(household.loc[great_britain, "brma"].astype(str))
    assert set(receipt["rows_by_grain"]) == {
        "ew_lsoa21",
        "northern_ireland",
        "scotland_oa22",
    }
    _, fresh, _, _ = _atomic(k, tmp_path / "fresh")
    replay = run_graph(
        compiled, sources=sources, store=store, kernels=fresh, resume="require"
    )
    assert all(node.hit for node in replay.nodes.values())


def test_brma_follows_the_area_not_the_clone(toy_ladder, tmp_path) -> None:
    """Every clone in one area gets that area's BRMA, whatever its source."""

    _, ladder_path = toy_ladder
    graph, registry, paths, _ = _atomic(3, tmp_path)
    sources = {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
        **write_toy_brma_area_layer(tmp_path / "brma"),
    }
    run = run_graph(
        compile_graph(graph),
        sources=sources,
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
    )
    household = run.population("uk.full.expand").table("household")
    per_area = household.groupby(household["oa_code"].astype(str))["brma"].nunique()
    assert (per_area == 1).all()


def test_none_mode_adds_no_brma_node(tmp_path) -> None:
    graph, _, _, _ = _atomic(1, tmp_path, mode="none")
    assert "uk.full.geography.brma" not in compile_graph(graph).order
    assert "uk_brma_area_layer" not in {source.name for source in graph.sources}


def test_area_brmas_need_the_atomic_assignment() -> None:
    with pytest.raises(ValueError, match="atomic geography assignment"):
        graph_and_registry(
            1, geography_assignment="legacy", brma_localisation_mode="area"
        )
    with pytest.raises(ValueError, match="BRMA localisation must be one of"):
        graph_and_registry(1, brma_localisation_mode="spread")


def test_the_node_refuses_a_pool_without_its_columns() -> None:
    cells = {
        ("household", column): "string"
        for column in ("atomic_area_system", "lsoa_code", "oa_code", "region")
    }
    with pytest.raises(ValueError, match="'brma'"):
        _brma_area_nodes(cells, common={"time_period": "2023"})


def test_config_follows_the_geography_assignment() -> None:
    assert UKFullBuildConfig(calibration_year=2025).brma_localisation == "area"
    legacy = UKFullBuildConfig(
        calibration_year=2025,
        geography_assignment="legacy",
        property_localisation="none",
    )
    assert legacy.brma_localisation == "none"
    with pytest.raises(ValueError, match="atomic geography assignment"):
        UKFullBuildConfig(
            calibration_year=2025,
            geography_assignment="legacy",
            property_localisation="none",
            brma_localisation="area",
        )


def test_brma_targets_need_area_brmas():
    """microcosm#1305 review: without area BRMAs the rows would bind on the spine's draw."""
    assert UKFullBuildConfig(
        calibration_year=2025, geography_levels=("la", "brma")
    ).geography_levels == ("la", "brma")
    with pytest.raises(ValueError, match="BRMA targets need the area BRMAs"):
        UKFullBuildConfig(
            calibration_year=2025,
            geography_assignment="legacy",
            geography_levels=("la", "brma"),
        )
