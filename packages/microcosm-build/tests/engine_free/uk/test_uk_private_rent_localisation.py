"""Private rents read off the assigned BRMA's list of rents (microcosm#1262)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.uk_runtime import private_rent_localisation as prl
from microcosm.build.uk_runtime.atomic_area_support import (
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.graph_build import UKFullBuildConfig
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_build import uk_full_population_graph as population
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    TOY_BRMA_RENT_LEVELS,
    toy_private_rent_resources,
    toy_support_sources,
    write_toy_brma_area_layer,
    write_toy_private_rent_resources,
    write_toy_supports,
)
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)

W = prl.WEEKS_IN_YEAR


def _quantiles() -> prl.RentQuantiles:
    return prl.load_rent_quantiles(
        json.dumps(toy_private_rent_resources()["quantiles"]).encode()
    )


def _counts() -> dict:
    return prl.load_private_renter_counts(
        json.dumps(toy_private_rent_resources()["counts"]).encode()
    )


def _rows(**overrides) -> dict:
    """Four London private renters (two FRS, one SPI, one FRS clone pair) and others."""

    base = {
        "spine_key": np.array(["a", "b", "c", "d", "d", "e", "f"], dtype=object),
        "region": np.array(
            [
                "LONDON",
                "LONDON",
                "LONDON",
                "LONDON",
                "LONDON",
                "LONDON",
                "NORTHERN_IRELAND",
            ]
        ),
        "tenure_type": np.array(
            ["RENT_PRIVATELY"] * 5 + ["OWNED_OUTRIGHT", "RENT_PRIVATELY"]
        ),
        "rent": np.array(
            [10_000.0, 20_000.0, 15_000.0, 30_000.0, 30_000.0, 0.0, 8_000.0]
        ),
        "num_bedrooms": np.array([2, 2, 2, 2, 2, 3, 2]),
        "brma": np.array(
            [
                "CENTRAL_LONDON",
                "INNER_EAST_LONDON",
                "CENTRAL_LONDON",
                "CENTRAL_LONDON",
                "INNER_EAST_LONDON",
                "CENTRAL_LONDON",
                "BELFAST",
            ],
            dtype=object,
        ),
        "channel": np.array(["frs", "frs", "spi", "frs", "frs", "frs", "frs"]),
        "weight": np.array([1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0]),
    }
    base.update(overrides)
    return base


def _ties(keys) -> dict[str, float]:
    return {str(key): 0.5 for key in keys}


def _localise(**overrides):
    rows = _rows(**overrides)
    return prl.localise_rents(
        **rows,
        tie_uniform=_ties(rows["spine_key"]),
        quantiles=_quantiles(),
        counts=_counts(),
    )


# ----------------------------------------------------------------------------
# Cells


def test_rent_categories_follow_the_homes_bedrooms() -> None:
    assert prl.rent_categories([0, 1, 2, 3, 4, 7]).tolist() == [
        "B",
        "B",
        "C",
        "D",
        "E",
        "E",
    ]
    for bad in ([np.nan], [-1], [1.5]):
        with pytest.raises(ValueError, match="bedroom count"):
            prl.rent_categories(bad)


def test_redraw_mask_takes_great_britain_private_renters_with_a_rent() -> None:
    mask = prl.redraw_mask(
        ["LONDON", "LONDON", "WALES", "NORTHERN_IRELAND", "SCOTLAND"],
        [
            "RENT_PRIVATELY",
            "RENT_FROM_COUNCIL",
            "RENT_PRIVATELY",
            "RENT_PRIVATELY",
            "RENT_PRIVATELY",
        ],
        [100.0, 100.0, 0.0, 100.0, 50.0],
    )
    assert mask.tolist() == [True, False, False, False, True]


# ----------------------------------------------------------------------------
# Ranks


def test_frs_renters_rank_within_their_cell_and_spi_renters_land_among_them() -> None:
    rows = _rows()
    mask = prl.redraw_mask(rows["region"], rows["tenure_type"], rows["rent"])
    spine = prl._spine_rows(
        spine_key=rows["spine_key"][mask],
        region=rows["region"][mask],
        category=prl.rent_categories(rows["num_bedrooms"][mask]),
        rent=rows["rent"][mask],
        channel=rows["channel"][mask],
        weight=rows["weight"][mask],
    )
    u, details = prl.rent_uniforms(spine, _ties(spine.index))
    # FRS reference: a (10k, w1), b (20k, w1), d (30k, w1 summed over its clones).
    assert u["a"] == pytest.approx(1 / 6)
    assert u["b"] == pytest.approx(3 / 6)
    assert u["d"] == pytest.approx(5 / 6)
    # The SPI renter at 15k sits above a and below b, so F(x-) = F(x) = 1/3.
    assert u["c"] == pytest.approx(1 / 3)
    assert details == {"fallback_cells": 0}


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from([5_000.0, 8_000.0, 12_000.0, 20_000.0]),
            st.floats(min_value=0.1, max_value=5.0),
            st.floats(min_value=0.0, max_value=0.999),
        ),
        min_size=2,
        max_size=25,
    )
)
def test_ranks_preserve_order_and_bracket_each_rent(entries) -> None:
    keys = [f"k{i}" for i in range(len(entries))]
    spine = pd.DataFrame(
        {
            "region": "LONDON",
            "category": "C",
            "rent": [rent for rent, _, _ in entries],
            "channel": "frs",
            "weight": [weight for _, weight, _ in entries],
        },
        index=pd.Index(keys, name="key"),
    )
    ties = {key: v for key, (_, _, v) in zip(keys, entries, strict=True)}
    u, _ = prl.rent_uniforms(spine, ties)
    total = spine["weight"].sum()
    for key in keys:
        x = spine.loc[key, "rent"]
        below = spine.loc[spine["rent"] < x, "weight"].sum() / total
        at_or_below = spine.loc[spine["rent"] <= x, "weight"].sum() / total
        assert below - 1e-12 <= u[key] <= at_or_below + 1e-12
    for i in keys:
        for j in keys:
            if spine.loc[i, "rent"] < spine.loc[j, "rent"]:
                assert u[i] <= u[j] + 1e-12


def test_a_cell_without_frs_renters_ranks_against_its_region() -> None:
    rows = _rows(num_bedrooms=np.array([2, 2, 4, 2, 2, 3, 2]))
    _, details = prl.localise_rents(
        **rows,
        tie_uniform=_ties(rows["spine_key"]),
        quantiles=_quantiles(),
        counts=_counts(),
    )
    assert details["fallback_cells"] == 1


# ----------------------------------------------------------------------------
# Scales and the re-draw


def test_region_scale_holds_the_frs_mean_on_the_census_mix() -> None:
    rows = _rows()
    mask = prl.redraw_mask(rows["region"], rows["tenure_type"], rows["rent"])
    spine = prl._spine_rows(
        spine_key=rows["spine_key"][mask],
        region=rows["region"][mask],
        category=prl.rent_categories(rows["num_bedrooms"][mask]),
        rent=rows["rent"][mask],
        channel=rows["channel"][mask],
        weight=rows["weight"][mask],
    )
    scales = prl.region_scales(spine, counts=_counts(), quantiles=_quantiles())
    quantiles = _quantiles()
    level = (
        W
        * 0.5
        * (
            quantiles.means[("CENTRAL_LONDON", "C")]
            + quantiles.means[("INNER_EAST_LONDON", "C")]
        )
    )
    frs_mean = (10_000.0 + 20_000.0 + 30_000.0) / 3
    assert scales == {"LONDON": pytest.approx(frs_mean / level)}


def test_localise_rents_reads_each_row_off_its_brma_at_its_rank() -> None:
    new, details = _localise()
    quantiles = _quantiles()
    level = (
        W
        * 0.5
        * (
            quantiles.means[("CENTRAL_LONDON", "C")]
            + quantiles.means[("INNER_EAST_LONDON", "C")]
        )
    )
    scale = 20_000.0 / level

    def expected(brma: str, u: float) -> float:
        return (
            W
            * scale
            * float(np.interp(u, quantiles.grid, quantiles.knots[(brma, "C")]))
        )

    assert new[0] == pytest.approx(expected("CENTRAL_LONDON", 1 / 6))
    assert new[1] == pytest.approx(expected("INNER_EAST_LONDON", 3 / 6))
    assert new[2] == pytest.approx(expected("CENTRAL_LONDON", 1 / 3))
    # The two clones of d share a rank and differ only through their BRMA.
    assert new[3] / new[4] == pytest.approx(
        TOY_BRMA_RENT_LEVELS["CENTRAL_LONDON"]
        / TOY_BRMA_RENT_LEVELS["INNER_EAST_LONDON"]
    )
    # The owner and the Northern Irish renter keep their rents bit for bit.
    assert new[5] == 0.0 and new[6] == 8_000.0
    assert details["rows_redrawn"] == 5 and details["spine_households_redrawn"] == 4


def test_localise_rents_does_not_depend_on_row_order() -> None:
    rows = _rows()
    order = np.array([6, 4, 2, 0, 5, 3, 1])
    permuted = {key: value[order] for key, value in rows.items()}
    new, _ = _localise()
    new_permuted, _ = prl.localise_rents(
        **permuted,
        tie_uniform=_ties(rows["spine_key"]),
        quantiles=_quantiles(),
        counts=_counts(),
    )
    np.testing.assert_array_equal(new_permuted, new[order])


def test_localise_rents_does_not_depend_on_k() -> None:
    """Three clones per household at a third of the weight give the same rents."""

    rows = _rows()
    tripled = {key: np.repeat(value, 3) for key, value in rows.items()}
    tripled["weight"] = tripled["weight"] / 3.0
    new, _ = _localise()
    new_tripled, _ = prl.localise_rents(
        **tripled,
        tie_uniform=_ties(rows["spine_key"]),
        quantiles=_quantiles(),
        counts=_counts(),
    )
    np.testing.assert_allclose(new_tripled, np.repeat(new, 3), rtol=1e-12)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "rent": np.array(
                    [10_000.0, 20_000.0, 15_000.0, 30_000.0, 31_000.0, 0.0, 8_000.0]
                )
            },
            "disagree",
        ),
        ({"brma": np.array(["MARS"] * 7, dtype=object)}, "No list of rents"),
        ({"channel": np.array(["spi"] * 7)}, "No FRS private renter"),
        ({"rent": np.array([10.0, 20.0, 15.0, 30.0, 30.0, 0.0, 8.0])}, "outside"),
    ],
)
def test_localise_rents_fails_closed(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        _localise(**overrides)


def test_weeks_in_year_is_the_spines() -> None:
    from microcosm.build.uk_runtime import frs_spine

    assert prl.WEEKS_IN_YEAR == frs_spine.WEEKS_IN_YEAR


# ----------------------------------------------------------------------------
# The graph


def _graph(k: int, tmp_path, *, mode: str = "brma_list"):
    payloads, paths = write_toy_supports(tmp_path / "supports")
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    graph, registry = population.graph_and_registry(
        k,
        geography_assignment="atomic",
        definition=definition,
        brma_localisation_mode="area",
        private_rent_localisation_mode=mode,
    )
    return graph, registry, paths


def _sources(tmp_path, ladder_path, paths) -> dict:
    return {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
        **write_toy_brma_area_layer(tmp_path / "brma"),
        **write_toy_private_rent_resources(tmp_path / "rent"),
    }


@pytest.mark.parametrize("k", [1, 3])
def test_private_rent_node_rewrites_the_renter_and_replays(
    k, toy_ladder, tmp_path
) -> None:
    _, ladder_path = toy_ladder
    graph, registry, paths = _graph(k, tmp_path)
    compiled = compile_graph(graph)
    order = compiled.order
    assert order.index("uk.full.geography.brma") < order.index("uk.full.private_rent")
    assert order.index("uk.full.private_rent") < order.index("uk.full.geography_gate")
    sources = _sources(tmp_path, ladder_path, paths)
    store = ContentStore(tmp_path / "store")
    run = run_graph(compiled, sources=sources, store=store, kernels=registry)
    household = run.population("uk.full.expand").table("household")
    renters = household[household["tenure_type"].astype(str) == "RENT_PRIVATELY"]
    others = household[household["tenure_type"].astype(str) != "RENT_PRIVATELY"]
    assert len(renters) == k and set(renters["brma"].astype(str)) == {"LOTHIAN"}
    # One FRS renter in its cell: its rank is its tie-break and its region's
    # scale returns its own rent as the region mean, so every clone reads the
    # same rent and the region's mean rent is unchanged.
    assert renters["rent"].nunique() == 1
    assert float(renters["rent"].iloc[0]) > 0
    assert set(others["rent"]) == {0.0}
    receipt = run.nodes["uk.full.private_rent"].receipt
    assert receipt["weights"] == "unchanged"
    assert receipt["regions_scaled"] == 1
    _, fresh, _ = _graph(k, tmp_path / "fresh")
    replay = run_graph(
        compiled, sources=sources, store=store, kernels=fresh, resume="require"
    )
    assert all(node.hit for node in replay.nodes.values())


def test_london_clones_read_their_own_areas_list(
    toy_ladder, tmp_path, monkeypatch
) -> None:
    original = population.source_frame

    def london_renter():
        frame = original()
        household = frame.table("household")
        household.loc[0, "tenure_type"] = "RENT_PRIVATELY"
        household.loc[0, "rent"] = 20_000.0
        household.loc[0, "num_bedrooms"] = 2
        return frame

    monkeypatch.setattr(population, "source_frame", london_renter)
    _, ladder_path = toy_ladder
    graph, registry, paths = _graph(6, tmp_path)
    run = run_graph(
        compile_graph(graph),
        sources=_sources(tmp_path, ladder_path, paths),
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
    )
    household = run.population("uk.full.expand").table("household")
    london = household[household["region"].astype(str) == "LONDON"]
    by_brma = london.groupby(london["brma"].astype(str))["rent"].agg(["min", "max"])
    assert set(by_brma.index) <= {"CENTRAL_LONDON", "INNER_EAST_LONDON"}
    assert (by_brma["min"] == by_brma["max"]).all()
    if len(by_brma) == 2:
        ratio = (
            by_brma.loc["CENTRAL_LONDON", "min"]
            / by_brma.loc["INNER_EAST_LONDON", "min"]
        )
        assert ratio == pytest.approx(
            TOY_BRMA_RENT_LEVELS["CENTRAL_LONDON"]
            / TOY_BRMA_RENT_LEVELS["INNER_EAST_LONDON"]
        )


def test_none_mode_adds_no_private_rent_node(tmp_path) -> None:
    graph, _, _ = _graph(1, tmp_path, mode="none")
    assert "uk.full.private_rent" not in compile_graph(graph).order


def test_private_rents_need_area_brmas() -> None:
    with pytest.raises(ValueError, match="need the area BRMAs"):
        population.graph_and_registry(
            1,
            geography_assignment="legacy",
            private_rent_localisation_mode="brma_list",
        )


def test_config_follows_the_brma_mode() -> None:
    assert (
        UKFullBuildConfig(calibration_year=2025).private_rent_localisation
        == "brma_list"
    )
    no_brma = UKFullBuildConfig(calibration_year=2025, brma_localisation="none")
    assert no_brma.private_rent_localisation == "none"
    with pytest.raises(ValueError, match="need the area BRMAs"):
        UKFullBuildConfig(
            calibration_year=2025,
            brma_localisation="none",
            private_rent_localisation="brma_list",
        )
