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
    assert details == {"fallback_cells": 0, "great_britain_cells": 0}


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


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_renters_on_the_census_mix_keep_their_regions_frs_level(seed) -> None:
    """Placed on the census mix, the weighted redrawn mean is the FRS mean.

    London's toy counts put a different share of each category in Central
    London. Renters placed in those shares, with ranks independent of where
    they land, keep the region's weighted FRS mean rent within Monte-Carlo
    noise: the scale's level is the expected rent on that mix.
    """

    rng = np.random.default_rng(seed)
    n = 4_000
    bedrooms = rng.choice([1, 2, 3, 4], size=n, p=[0.3, 0.35, 0.25, 0.1])
    categories = prl.rent_categories(bedrooms)
    central = {"B": 0.7, "C": 0.5, "D": 0.3, "E": 0.2}
    brma = np.where(
        rng.random(n) < np.vectorize(central.get)(categories),
        "CENTRAL_LONDON",
        "INNER_EAST_LONDON",
    ).astype(object)
    rent = rng.lognormal(np.log(15_000.0), 0.4, size=n)
    weight = rng.uniform(0.5, 1.5, size=n)
    keys = np.array([f"k{i}" for i in range(n)], dtype=object)
    new, _ = prl.localise_rents(
        spine_key=keys,
        region=np.full(n, "LONDON"),
        tenure_type=np.full(n, "RENT_PRIVATELY"),
        rent=rent,
        num_bedrooms=bedrooms,
        brma=brma,
        channel=np.full(n, "frs"),
        weight=weight,
        tie_uniform={str(key): float(rng.random()) for key in keys},
        quantiles=_quantiles(),
        counts=_counts(),
    )
    frs_mean = float(np.average(rent, weights=weight))
    # The unmutated ratio sits within half a percent; one category's mix used
    # for every category moves it by about three (microcosm#1305 re-review).
    assert float(np.average(new, weights=weight)) == pytest.approx(frs_mean, rel=0.01)


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


def _graph(k: int, tmp_path, *, mode: str = "brma_list", full_scale: bool = True):
    payloads, paths = write_toy_supports(tmp_path / "supports")
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    graph, registry = population.graph_and_registry(
        k,
        geography_assignment="atomic",
        definition=definition,
        brma_localisation_mode="area",
        private_rent_localisation_mode=mode,
        full_scale=full_scale,
    )
    return graph, registry, paths


@pytest.mark.parametrize("full_scale", [True, False])
def test_the_rent_node_is_strict_exactly_at_full_scale(full_scale, tmp_path) -> None:
    """microcosm#1305 re-review: the sampled-rung fallback is pinned to the rung."""

    graph, _, _ = _graph(1, tmp_path, full_scale=full_scale)
    params = graph.node("uk.full.private_rent").params
    assert params["strict_cells"] is full_scale


@pytest.mark.parametrize(
    "fractions,full_scale",
    [((1.0, 1.0), True), ((0.5, 1.0), False), ((1.0, 0.5), False)],
)
def test_the_build_config_sets_full_scale_from_both_sample_fractions(
    fractions, full_scale, monkeypatch
) -> None:
    from microcosm.build.uk_runtime import graph_build

    seen = {}

    class StopBuildError(Exception):
        pass

    def capture(*args, **kwargs):
        seen.update(kwargs)
        raise StopBuildError

    from microcosm.build.uk_runtime.graph import uk_spine_graph

    monkeypatch.setattr(graph_build, "append_uk_population_nodes", capture)
    sample, source = fractions
    with pytest.raises(StopBuildError):
        graph_build.uk_full_graph(
            UKFullBuildConfig(
                calibration_year=2025,
                sample_fraction=sample,
                source_sample_fraction=source,
            ),
            spine=uk_spine_graph(source_mode="split"),
        )
    assert seen["full_scale"] is full_scale


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
    # One FRS renter in its cell: its rank u is its tie-break, and the region's
    # scale sets the census-mix mean to its own spine rent of 9,000, so every
    # clone reads 9,000 * (0.5 + u) off the linear toy list: one rent for all
    # clones, and not the spine's.
    assert renters["rent"].nunique() == 1
    rent = float(renters["rent"].iloc[0])
    assert 4_500.0 < rent < 13_500.0
    assert rent != pytest.approx(9_000.0)
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
    # Six clones, placed in both toy London BRMAs (two and four at this seed):
    # each BRMA's clones read one rent, in the ratio of the two lists.
    assert len(london) == 6
    by_brma = london.groupby(london["brma"].astype(str))["rent"].agg(["min", "max"])
    assert set(by_brma.index) == {"CENTRAL_LONDON", "INNER_EAST_LONDON"}
    assert (by_brma["min"] == by_brma["max"]).all()
    assert not np.isclose(london["rent"].astype(float), 20_000.0).any()
    ratio = (
        by_brma.loc["CENTRAL_LONDON", "min"] / by_brma.loc["INNER_EAST_LONDON", "min"]
    )
    assert ratio == pytest.approx(
        TOY_BRMA_RENT_LEVELS["CENTRAL_LONDON"]
        / TOY_BRMA_RENT_LEVELS["INNER_EAST_LONDON"]
    )


@pytest.mark.parametrize("full_scale", [True, False])
def test_the_rent_kernel_reads_its_strictness_off_the_node(
    full_scale, toy_ladder, tmp_path, monkeypatch
) -> None:
    """The third review of #1305: the kernel, not just the node, honours it.

    Hard-coding the kernel's strictness passed every test; the kernel must
    hand ``localise_rents`` the node's own ``strict_cells``.
    """
    seen: list[bool] = []
    original = prl.localise_rents

    def recording(**kwargs):
        seen.append(kwargs["strict"])
        return original(**kwargs)

    monkeypatch.setattr(prl, "localise_rents", recording)
    _, ladder_path = toy_ladder
    graph, registry, paths = _graph(1, tmp_path, full_scale=full_scale)
    run_graph(
        compile_graph(graph),
        sources=_sources(tmp_path, ladder_path, paths),
        store=ContentStore(tmp_path / "store"),
        kernels=registry,
    )
    assert seen == [full_scale]


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


def _regional_rows(*, wales_frs: int, wales_rent: float = 9_000.0) -> dict:
    """Twelve London FRS renters in category C, one SPI-only London D renter,
    ``wales_frs`` Welsh FRS renters at ``wales_rent`` and one Welsh SPI one."""

    london = 12
    n = london + 1 + wales_frs + 1
    region = ["LONDON"] * (london + 1) + ["WALES"] * (wales_frs + 1)
    brma = [
        "CENTRAL_LONDON" if index % 2 else "INNER_EAST_LONDON"
        for index in range(london + 1)
    ] + ["CARDIFF"] * (wales_frs + 1)
    rent = (
        [18_000.0 + 500.0 * index for index in range(london)]
        + [25_000.0]
        + [wales_rent] * wales_frs
        + [9_500.0]
    )
    channel = ["frs"] * london + ["spi"] + ["frs"] * wales_frs + ["spi"]
    return {
        "spine_key": np.array([f"k{index}" for index in range(n)], dtype=object),
        "region": np.array(region),
        "tenure_type": np.full(n, "RENT_PRIVATELY"),
        "rent": np.asarray(rent, dtype=float),
        "num_bedrooms": np.array([2] * london + [3] + [2] * (wales_frs + 1)),
        "brma": np.array(brma, dtype=object),
        "channel": np.array(channel),
        "weight": np.ones(n),
    }


def _localise_rows(rows: dict, **kwargs):
    return prl.localise_rents(
        **rows,
        tie_uniform=_ties(rows["spine_key"]),
        quantiles=_quantiles(),
        counts=_counts(),
        **kwargs,
    )


def test_a_sampled_rung_ranks_a_region_without_frs_renters_on_great_britain():
    """microcosm#1305 review: a small rung's sample can leave a region bare.

    Wales holds only the SPI-channel renter here. A full-scale build refuses;
    a sampled rung ranks and scales it on Great Britain's FRS renters on
    Wales's census mix, and the details count it once: London's SPI-only
    category falls back on London's renters, Wales's cell on Great Britain's.
    """

    rows = _regional_rows(wales_frs=0)
    with pytest.raises(
        ValueError, match="No FRS private renter carries weight in WALES"
    ):
        _localise_rows(rows)
    new, details = _localise_rows(rows, strict=False)
    assert (details["fallback_cells"], details["great_britain_cells"]) == (1, 1)
    assert details["regions_scaled_on_great_britain"] == 1
    assert new[-1] > 0 and new[-1] != pytest.approx(rows["rent"][-1])


def test_a_sampled_region_with_a_few_frs_renters_ranks_on_great_britain():
    """microcosm#1305 re-review: two FRS renters are sampling noise.

    Their implausible rents put Wales's own scale outside the band. A
    full-scale build computes it and refuses; a sampled rung ranks and scales
    the region on Great Britain's renters, and counts it.
    """

    rows = _regional_rows(wales_frs=2, wales_rent=100.0)
    with pytest.raises(ValueError, match="WALES's scale on the lists is outside"):
        _localise_rows(rows)
    new, details = _localise_rows(rows, strict=False)
    assert details["great_britain_cells"] == 1
    assert details["regions_scaled_on_great_britain"] == 1
    assert (new[-3:] > 1_000).all()
    # Ten FRS renters are enough: the region keeps its own scale.
    rows = _regional_rows(wales_frs=prl.MIN_SAMPLED_REGION_RENTERS)
    _, details = _localise_rows(rows, strict=False)
    assert details["great_britain_cells"] == 0
    assert details["regions_scaled_on_great_britain"] == 0
    assert (
        prl.sampled_thin_regions(
            pd.DataFrame(
                {
                    "region": ["WALES"],
                    "channel": ["frs"],
                    "weight": [1.0],
                }
            ),
            strict=True,
        )
        == frozenset()
    )


def test_the_size_identity_moves_with_the_rent_pins():
    """microcosm#1305 review: a size checkpoint binds the rent resources it ran on."""
    from argparse import Namespace

    from microcosm.build.uk_runtime.rowwise_posture import UK_ROWWISE_DENSE_POSTURE
    from microcosm.build.uk_runtime.size_checkpoint import uk_size_checkpoint_identity

    def identity(quantiles_sha256: str) -> dict:
        args = Namespace(
            _posture=UK_ROWWISE_DENSE_POSTURE,
            ledger_facts_sha256="f",
            ledger_manifest_sha256="m",
            seed=7,
            selection_seed=None,
            n_clones=2,
            dataset_households=2,
            epochs=2,
            learning_rate=0.02,
            sample_fraction=1.0,
            sample_seed=1,
            source_lineage_modulus=None,
            target_weight_rule="grain_equal",
            engine_blocks=1,
            measure_exclusions=None,
            private_rent_localisation="brma_list",
        )
        pins = {
            "dataset": {"sha256": "d"},
            "ladder": {"sha256": "l"},
            prl.QUANTILES_SOURCE: {"sha256": quantiles_sha256},
            prl.COUNTS_SOURCE: {"sha256": "c"},
        }
        return uk_size_checkpoint_identity(args, pins=pins, source_year=2024)

    assert identity("q1") != identity("q2")
    assert identity("q1")["private_rent_localisation"]["resource_pins"][
        prl.QUANTILES_SOURCE
    ] == {"sha256": "q1"}
