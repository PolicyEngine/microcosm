"""Per-clone home value and band localisation (microcosm#1240, fixes 3 and 4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import property_localisation as pl
from microcosm.build.uk_runtime.atomic_area_support import (
    uk_atomic_assignment_definition,
)
from microcosm.build.uk_runtime.household_lineage import (
    uk_household_clone_paths,
    without_geographic_branch,
)
from microcosm.graph import ContentStore, compile_graph, run_graph
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    toy_property_resources,
    toy_support_sources,
    write_toy_property_resources,
    write_toy_supports,
)
from test_support.microcosm_build.uk_full_population_graph import (
    graph_and_registry,
)
from test_support.microcosm_build.uk_ladder_rowwise_clone import (
    toy_ladder as toy_ladder,
)

STREAM = ("sha256-u53-v1", pl.PROPERTY_STREAM_EXPERIMENT, 0, 7)


def _knots() -> pl.ValueKnots:
    return pl.load_value_knots(json.dumps(toy_property_resources()["knots"]).encode())


def _mix() -> pl.BandMix:
    return pl.load_band_mix(json.dumps(toy_property_resources()["band_mix"]).encode())


# ----------------------------------------------------------------------------
# Resources


def test_resource_loaders_validate_shape() -> None:
    knots = _knots()
    assert knots.cap == 2_000_000 and knots.grid[0] == 0.0 and knots.grid[-1] == 1.0
    mix = _mix()
    assert all(values[-1] == 1.0 for values in mix.cumulative.values())
    bad = toy_property_resources()["knots"]
    bad["levels"]["constituency"][next(iter(bad["levels"]["constituency"]))][
        "quantiles"
    ][2] = 5.0e6
    with pytest.raises(ValueError, match="outside"):
        pl.load_value_knots(json.dumps(bad).encode())
    bad_mix = toy_property_resources()["band_mix"]
    next(iter(bad_mix["lsoa"].values()))["cumulative"][-1] = 0.9
    with pytest.raises(ValueError, match="malformed"):
        pl.load_band_mix(json.dumps(bad_mix).encode())


# ----------------------------------------------------------------------------
# Uniforms


def _pool(k: int) -> pd.DataFrame:
    # Three spine households (two London owners, one London renter), K clones each.
    rows = []
    for key, region, tenure, value, weight in (
        ("a", "LONDON", "OWNED_OUTRIGHT", 500_000.0, 2.0),
        ("b", "LONDON", "OWNED_WITH_MORTGAGE", 300_000.0, 2.0),
        ("c", "LONDON", "RENT_PRIVATELY", 0.0, 2.0),
    ):
        for clone in range(k):
            rows.append(
                {
                    "key": key,
                    "region": region,
                    "tenure_type": tenure,
                    "value": value,
                    "weight": weight / k,
                    "clone": clone,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("k", [1, 3])
def test_owner_uniforms_rank_within_region_and_are_clone_invariant(k) -> None:
    pool = _pool(k)
    u, receipt = pl.owner_uniforms(
        spine_key=pool["key"].to_numpy(dtype=object),
        region=pool["region"].to_numpy(),
        tenure_type=pool["tenure_type"].to_numpy(),
        value=pool["value"].to_numpy(),
        weight=pool["weight"].to_numpy(),
        stream=STREAM,
    )
    by_key = pd.Series(u).groupby(pool["key"]).agg(["min", "max", "nunique"])
    assert (by_key["nunique"] == 1).all()  # one uniform per spine household
    # Two equal-weight owners: mid-ranks 0.25 (lower value) and 0.75.
    assert by_key.loc["b", "min"] == pytest.approx(0.25)
    assert by_key.loc["a", "min"] == pytest.approx(0.75)
    assert 0.0 <= by_key.loc["c", "min"] < 1.0  # renter: identity uniform
    assert receipt["ranked_owner_households_by_region"] == {"LONDON": 2}
    assert receipt["spine_households"] == 3 and receipt["pool_rows"] == 3 * k


def test_owner_uniforms_do_not_depend_on_k() -> None:
    draws = []
    for k in (1, 2, 5):
        pool = _pool(k)
        u, _ = pl.owner_uniforms(
            spine_key=pool["key"].to_numpy(dtype=object),
            region=pool["region"].to_numpy(),
            tenure_type=pool["tenure_type"].to_numpy(),
            value=pool["value"].to_numpy(),
            weight=pool["weight"].to_numpy(),
            stream=STREAM,
        )
        draws.append(pd.Series(u).groupby(pool["key"]).first().to_dict())
    assert draws[0] == draws[1] == draws[2]


def test_support_copies_take_a_keyed_uniform_inside_the_top_stratum() -> None:
    pool = _pool(2)
    support = (pool["key"] == "b").to_numpy()
    u, receipt = pl.owner_uniforms(
        spine_key=pool["key"].to_numpy(dtype=object),
        region=pool["region"].to_numpy(),
        tenure_type=pool["tenure_type"].to_numpy(),
        value=pool["value"].to_numpy(),
        weight=pool["weight"].to_numpy(),
        stream=STREAM,
        support_copy=support,
        support_stratum_lower=0.9,
    )
    assert np.all(u[support] >= 0.9) and np.all(u[support] < 1.0)
    assert receipt["support_copies"] == 1
    # The support copy is ranked beside the other owner (review finding 1 of
    # microcosm#1270): b (300k) then a (500k), equal spine weights, so a's
    # mid-rank is 0.75 and the copy's own draw replaces its 0.25.
    assert pd.Series(u).groupby(pool["key"]).first()["a"] == pytest.approx(0.75)


def test_support_copies_keep_the_stratum_mass_at_the_selected_share() -> None:
    # Four thousand equal-weight owners in one region: the stage selects the
    # top five per cent and copies each six ways; the pool's uniforms must
    # leave exactly that share of the owner mass at or above the stratum.
    from microcosm.build.uk_runtime.high_value_home_support import (
        select_high_value_support_households,
    )

    n = 4_000
    ids = np.arange(1, n + 1, dtype=np.int64)
    values = 100_000.0 + 1_000.0 * np.arange(n, dtype=float)
    region = np.full(n, "LONDON", dtype=object)
    tenure = np.full(n, "OWNED_OUTRIGHT", dtype=object)
    weights = np.ones(n, dtype=float)
    selected, _ = select_high_value_support_households(
        household_ids=ids,
        weights=weights,
        region=region,
        tenure_type=tenure,
        value=values,
        stratum_lower=0.95,
    )
    assert selected.sum() == pytest.approx(0.05 * n, abs=2)
    copies = 6
    rows = []
    for index in range(n):
        if selected[index]:
            for copy in range(copies):
                rows.append(
                    (f"{ids[index]}#{copy}", values[index], 1.0 / copies, copy > 0)
                )
        else:
            rows.append((str(ids[index]), values[index], 1.0, False))
    pool = pd.DataFrame(rows, columns=["key", "value", "weight", "support"])
    u, receipt = pl.owner_uniforms(
        spine_key=pool["key"].to_numpy(dtype=object),
        region=np.full(len(pool), "LONDON", dtype=object),
        tenure_type=np.full(len(pool), "OWNED_OUTRIGHT", dtype=object),
        value=pool["value"].to_numpy(),
        weight=pool["weight"].to_numpy(),
        stream=STREAM,
        support_copy=pool["support"].to_numpy(),
        support_stratum_lower=0.95,
    )
    total = float(pool["weight"].sum())
    mass_top = float(pool["weight"].to_numpy()[u >= 0.95].sum()) / total
    mass_top_1 = float(pool["weight"].to_numpy()[u >= 0.99].sum()) / total
    assert mass_top == pytest.approx(0.05, abs=0.003)
    assert mass_top_1 == pytest.approx(0.01, abs=0.003)
    assert receipt["support_copies"] == int(selected.sum()) * (copies - 1)


def test_owner_uniforms_refuse_clones_that_disagree() -> None:
    pool = _pool(2)
    pool.loc[0, "value"] = 999.0
    with pytest.raises(ValueError, match="different stored home values"):
        pl.owner_uniforms(
            spine_key=pool["key"].to_numpy(dtype=object),
            region=pool["region"].to_numpy(),
            tenure_type=pool["tenure_type"].to_numpy(),
            value=pool["value"].to_numpy(),
            weight=pool["weight"].to_numpy(),
            stream=STREAM,
        )


# ----------------------------------------------------------------------------
# Maps


def test_localise_values_interpolates_caps_and_leaves_others_untouched() -> None:
    knots = _knots()
    seat = next(iter(knots.quantiles))
    values = np.array([400_000.0, 250_000.0, 0.0, 300_000.0, 700_000.0])
    tenure = np.array(
        [
            "OWNED_OUTRIGHT",
            "OWNED_WITH_MORTGAGE",
            "RENT_PRIVATELY",
            "OWNED_OUTRIGHT",
            "OWNED_OUTRIGHT",
        ]
    )
    codes = np.array([seat, seat, seat, "S14000001", seat])
    u = np.array([0.5, 1.0, 0.2, 0.9, 0.0])
    new_value, new_wealth, receipt = pl.localise_values(
        u=u,
        constituency_code=codes,
        tenure_type=tenure,
        value=values,
        property_wealth=values + 10.0,
        knots=knots,
    )
    assert new_value[0] == pytest.approx(100_000 + 1_900_000 * 0.5)
    assert new_value[1] == pytest.approx(2_000_000.0)  # the cap
    assert (
        new_value[2] == 0.0 and new_value[3] == 300_000.0
    )  # renter, Scotland untouched
    assert new_value[4] == pytest.approx(100_000.0)
    assert new_wealth[0] == pytest.approx(new_value[0] + 10.0)
    assert receipt["rows_rewritten"] == 3 and receipt["rows_at_cap"] == 1
    assert receipt["rows_untouched"] == {
        "not_owner_occupied": 1,
        "zero_value_owner": 0,
        "outside_england_wales": 1,
    }
    with pytest.raises(ValueError, match="No value knots"):
        pl.localise_values(
            u=u[:1],
            constituency_code=np.array(["E14999999"]),
            tenure_type=tenure[:1],
            value=values[:1],
            property_wealth=values[:1],
            knots=knots,
        )


def test_localise_bands_is_monotone_in_the_uniform_and_skips_scotland() -> None:
    mix = _mix()
    lsoa = next(iter(mix.cumulative))
    u = np.array([0.0, 0.124, 0.126, 0.5, 0.999, 0.3, 0.999])
    codes = np.array([lsoa] * 5 + ["S01000001", lsoa])
    band = np.array(["D", "D", "D", "D", "D", "A", "C"])
    tenure = np.array(["OWNED_OUTRIGHT"] * 6 + ["RENT_PRIVATELY"])
    new_band, receipt = pl.localise_bands(
        u=u, lsoa_code=codes, band=band, tenure_type=tenure, mix=mix
    )
    assert list(new_band[:5]) == ["A", "A", "B", "E", "H"]
    assert new_band[5] == "A"
    # An England renter keeps the spine's band whatever its uniform says.
    assert new_band[6] == "C"
    assert (
        receipt["rows_rewritten"] == 5
        and receipt["rows_untouched_outside_england_wales"] == 1
        and receipt["rows_untouched_renters_england_wales"] == 1
    )
    assert pl.monotone_violations(u[:5], new_band[:5], codes[:5]) == 0
    assert receipt["band_mix_after"]["I"] == 0


# ----------------------------------------------------------------------------
# Lineage


def test_spine_keys_drop_the_geographic_branch_only() -> None:
    household = pd.DataFrame(
        {
            "source_household_id": np.array([1, 1, 2], dtype="int64"),
            "household_support_channel": pd.array(
                ["frs", "frs", "spi"], dtype="string"
            ),
            "household_support_clone_index": np.array([0, 0, 1], dtype="int64"),
            "household_is_cgt_support_copy": [False, False, False],
            "cgt_support_copy_index": np.array([0, 0, 0], dtype="int64"),
            "household_is_capital_gains_clone": [False, False, True],
            "household_is_cgt_residential_clone": [False, False, False],
            "cgt_residential_clone_index": np.array([0, 0, 0], dtype="int64"),
            "household_clone_index": np.array([0, 1, 2], dtype="int64"),
        }
    )
    ids, paths = uk_household_clone_paths(household)
    assert ids == [1, 1, 2]
    assert paths[1] == (("geographic_support", 1),)
    assert paths[2] == (
        ("spi_support_channel", 1),
        ("cgt_incidence_clone", 1),
        ("geographic_support", 2),
    )
    spine = without_geographic_branch(paths)
    assert spine[0] == spine[1] == ()
    keys = pl.spine_household_keys(household, source="frs", source_vintage="v")
    assert keys[0] == keys[1] != keys[2]


# ----------------------------------------------------------------------------
# Graph


@pytest.mark.parametrize("k", [1, 3])
def test_property_nodes_rewrite_values_and_bands_per_clone_and_replay(
    k, toy_ladder, tmp_path
) -> None:
    _, ladder_path = toy_ladder
    payloads, paths = write_toy_supports(tmp_path / "supports")
    resource_paths = write_toy_property_resources(tmp_path / "property")
    definition = uk_atomic_assignment_definition(payloads, seed=7)
    graph, registry = graph_and_registry(
        k,
        geography_assignment="atomic",
        definition=definition,
        property_localisation_mode="price_paid",
    )
    compiled = compile_graph(graph)
    order = compiled.order
    assert order.index("uk.full.geography.derive") < order.index(
        "uk.full.property.value"
    )
    assert order.index("uk.full.property.value") < order.index("uk.full.property.band")
    assert order.index("uk.full.property.band") < order.index("uk.full.geography_gate")
    sources = {
        "uk_ladder": ladder_path,
        "fixture": ladder_path,
        **toy_support_sources(paths),
        **resource_paths,
    }
    store = ContentStore(tmp_path / "store")
    first = run_graph(compiled, sources=sources, store=store, kernels=registry)
    household = first.population("uk.full.expand").table("household")
    value_receipt = first.nodes["uk.full.property.value"].receipt
    band_receipt = first.nodes["uk.full.property.band"].receipt
    england_wales = household["constituency_code"].astype(str).str[0].isin(["E", "W"])
    owners = household["tenure_type"].isin(
        ["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE"]
    ) & (household["main_residence_value"] > 0)
    # Every England-and-Wales owner clone reads its value off the toy knots.
    rewritten = household[england_wales & owners]
    assert len(rewritten) == 2 * k and value_receipt["rows_rewritten"] == 2 * k
    assert (rewritten["main_residence_value"] >= 100_000).all()
    assert (rewritten["main_residence_value"] <= 2_000_000).all()
    # Clones of one spine household share one uniform, so one value.
    per_source = rewritten.groupby("source_household_id")[
        "main_residence_value"
    ].nunique()
    assert (per_source == 1).all()
    # Scotland and Northern Ireland keep the spine's values and bands.
    untouched = household[~england_wales]
    assert set(untouched["main_residence_value"]) <= {0.0}
    assert band_receipt["rows_untouched_outside_england_wales"] == len(untouched)
    assert band_receipt["monotone_violations"] == 0
    assert set(household.loc[england_wales, "council_tax_band"]) <= set("ABCDEFGH")
    assert value_receipt["weights"] == "unchanged"
    _, fresh = graph_and_registry(
        k,
        geography_assignment="atomic",
        definition=definition,
        property_localisation_mode="price_paid",
    )
    replay = run_graph(
        compiled, sources=sources, store=store, kernels=fresh, resume="require"
    )
    assert all(receipt.hit for receipt in replay.nodes.values())


def test_property_localisation_needs_the_atomic_assignment() -> None:
    with pytest.raises(ValueError, match="atomic geography assignment"):
        graph_and_registry(
            1, geography_assignment="legacy", property_localisation_mode="price_paid"
        )
