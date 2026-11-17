"""Per-clone home value and band localisation (microcosm#1240, fixes 3 and 4)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import property_localisation as pl
from microcosm.build.uk_runtime.household_lineage import (
    uk_household_clone_paths,
    without_geographic_branch,
)
from test_support.microcosm_build.uk_atomic_support_fixtures import (
    toy_property_resources,
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


_SEAT = "E14000001"
_OTHER_SEAT = "E14000002"


def _slices(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, object]]:
    return pl.owner_slice_uniforms(
        key=frame["key"].to_numpy(dtype=object),
        cell=frame["cell"].to_numpy(dtype=object),
        stratum=frame["clone"].to_numpy(dtype=np.int64),
        tenure_type=frame["tenure_type"].to_numpy(dtype=object),
        value=frame["value"].to_numpy(dtype=float),
        weight=frame["weight"].to_numpy(dtype=float),
        stream=STREAM,
    )


def _frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["key", "cell", "clone", "tenure_type", "value", "weight"]
    )


def test_each_owner_takes_a_point_inside_its_own_weight_slice() -> None:
    # One English seat: owners by value b (weight 2), c (1), a (1), so their
    # slices of the seat's weight are [0, 0.5), [0.5, 0.75) and [0.75, 1).
    frame = _frame(
        [
            ("a", _SEAT, 0, "OWNED_OUTRIGHT", 300_000.0, 1.0),
            ("b", _SEAT, 0, "OWNED_WITH_MORTGAGE", 100_000.0, 2.0),
            ("c", _SEAT, 0, "OWNED_OUTRIGHT", 200_000.0, 1.0),
            ("renter", _SEAT, 0, "RENT_PRIVATELY", 0.0, 1.0),
            ("scot", "S14000001", 0, "OWNED_OUTRIGHT", 250_000.0, 1.0),
        ]
    )
    u, receipt = _slices(frame)
    by_key = dict(zip(frame["key"], u, strict=True))
    assert 0.0 <= by_key["b"] < 0.5
    assert 0.5 <= by_key["c"] < 0.75
    assert 0.75 <= by_key["a"] < 1.0
    assert 0.0 <= by_key["renter"] < 1.0 and 0.0 <= by_key["scot"] < 1.0
    assert receipt == {
        "ranked_rows": 3,
        "cells": 1,
        "cells_without_weight": 0,
        "rank_basis": pl.OWNER_RANK_BASIS,
    }
    assert pl.OWNER_RANK_BASIS == "weighted_slice_within_constituency_and_clone"


def test_constituencies_and_clones_rank_apart() -> None:
    # Two seats and two clones: each (seat, clone) cell holds two equal owners,
    # so in every cell the lower value takes [0, 0.5) and the higher [0.5, 1),
    # whatever the other cells hold.
    rows = []
    for seat in (_SEAT, _OTHER_SEAT):
        for clone in (0, 1):
            low = 100_000.0 * (1 + clone) + (5_000.0 if seat == _OTHER_SEAT else 0.0)
            rows.append(
                (f"{seat}-{clone}-low", seat, clone, "OWNED_OUTRIGHT", low, 1.0)
            )
            rows.append(
                (f"{seat}-{clone}-high", seat, clone, "OWNED_OUTRIGHT", 9 * low, 3.0)
            )
    frame = _frame(rows)
    u, receipt = _slices(frame)
    low = frame["key"].str.endswith("low").to_numpy()
    assert np.all((u[low] >= 0.0) & (u[low] < 0.25))
    assert np.all((u[~low] >= 0.25) & (u[~low] < 1.0))
    assert receipt["cells"] == 4


def test_clone_zero_takes_the_one_clone_uniforms() -> None:
    rng = np.random.default_rng(5)
    n = 40
    seats = np.where(np.arange(n) % 3 == 0, _OTHER_SEAT, _SEAT)
    values = rng.uniform(1e5, 2e6, n)
    weights = rng.uniform(0.5, 3.0, n)
    one = _frame(
        [
            (f"h{i}", seats[i], 0, "OWNED_OUTRIGHT", values[i], weights[i])
            for i in range(n)
        ]
    )
    k = 3
    rows = []
    for clone in range(k):
        for i in range(n):
            key = f"h{i}" if clone == 0 else f"h{i}#{clone}"
            seat = seats[i] if clone == 0 else (_SEAT, _OTHER_SEAT)[(i + clone) % 2]
            value = values[i] if clone == 0 else rng.uniform(1e5, 2e6)
            rows.append((key, seat, clone, "OWNED_OUTRIGHT", value, weights[i] / k))
    many = _frame(rows)
    u_one, _ = _slices(one)
    u_many, _ = _slices(many)
    clone_zero = (many["clone"] == 0).to_numpy()
    np.testing.assert_allclose(u_many[clone_zero], u_one, rtol=0, atol=1e-12)


def test_the_uniforms_do_not_depend_on_row_order() -> None:
    rng = np.random.default_rng(11)
    frame = _frame(
        [
            (
                f"h{i}",
                (_SEAT, _OTHER_SEAT)[i % 2],
                i % 3,
                "OWNED_OUTRIGHT",
                float(rng.integers(1, 6)) * 1e5,
                rng.uniform(0.5, 2.0),
            )
            for i in range(60)
        ]
    )
    u, _ = _slices(frame)
    shuffled = frame.sample(frac=1.0, random_state=3).reset_index(drop=True)
    u_shuffled, _ = _slices(shuffled)
    expected = dict(zip(frame["key"], u, strict=True))
    assert [expected[key] for key in shuffled["key"]] == list(u_shuffled)


def test_owners_weighted_uniforms_are_uniform_even_in_small_cells() -> None:
    # Two thousand seats of two equal owners each: a mid-rank would put every
    # owner at 0.25 or 0.75; the slice points spread the owner weight evenly,
    # so each share above a threshold is the threshold's complement.
    rows = []
    for seat in range(2_000):
        code = f"E14{seat:06d}"
        rows.append((f"s{seat}-a", code, 0, "OWNED_OUTRIGHT", 1e5, 1.0))
        rows.append((f"s{seat}-b", code, 0, "OWNED_OUTRIGHT", 2e5, 1.0))
    u, _ = _slices(_frame(rows))
    for threshold in (0.5, 0.9, 0.99):
        assert float(np.mean(u >= threshold)) == pytest.approx(1 - threshold, abs=0.02)
    assert float(np.max(u)) > 0.99


def test_a_cell_without_weight_takes_equal_slices() -> None:
    frame = _frame(
        [
            ("a", _SEAT, 0, "OWNED_OUTRIGHT", 100_000.0, 0.0),
            ("b", _SEAT, 0, "OWNED_OUTRIGHT", 200_000.0, 0.0),
        ]
    )
    u, receipt = _slices(frame)
    assert 0.0 <= u[0] < 0.5 <= u[1] < 1.0
    assert receipt["cells_without_weight"] == 1


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda f: f.assign(key=["a", "a"]), "own key"),
        (lambda f: f.assign(weight=[1.0, -1.0]), "non-negative"),
        (lambda f: f.assign(weight=[1.0, np.nan]), "non-negative"),
    ],
)
def test_owner_slice_uniforms_refuse(change, match) -> None:
    frame = _frame(
        [
            ("a", _SEAT, 0, "OWNED_OUTRIGHT", 100_000.0, 1.0),
            ("b", _SEAT, 0, "OWNED_OUTRIGHT", 200_000.0, 1.0),
        ]
    )
    with pytest.raises(ValueError, match=match):
        _slices(change(frame))
    with pytest.raises(ValueError, match="align"):
        pl.owner_slice_uniforms(
            key=np.array(["a", "b"], dtype=object),
            cell=np.array([_SEAT], dtype=object),
            stratum=np.zeros(2, dtype=np.int64),
            tenure_type=np.array(["OWNED_OUTRIGHT"] * 2, dtype=object),
            value=np.ones(2),
            weight=np.ones(2),
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


def test_the_value_half_is_localise_values_values_bit_for_bit() -> None:
    """The WAS chain reads its home values through the value half of
    localise_values (microcosm#931): the same values, bit for bit, and the
    rows it rewrote."""

    knots = _knots()
    seats = sorted(knots.quantiles)
    rng = np.random.default_rng(931)
    n = 64
    codes = np.asarray(
        [seats[i % len(seats)] for i in range(n - 8)]
        + ["S14000001"] * 4
        + ["N05000001"] * 4,
        dtype=object,
    )
    tenure = np.resize(
        np.asarray(["OWNED_OUTRIGHT", "OWNED_WITH_MORTGAGE", "RENT_PRIVATELY"]), n
    )
    values = np.where(rng.uniform(size=n) < 0.2, 0.0, rng.uniform(50_000, 900_000, n))
    u = rng.uniform(size=n)

    new_value, target = pl.localise_main_residence_values(
        u=u, constituency_code=codes, tenure_type=tenure, value=values, knots=knots
    )
    whole, _, receipt = pl.localise_values(
        u=u,
        constituency_code=codes,
        tenure_type=tenure,
        value=values,
        property_wealth=values + 5.0,
        knots=knots,
    )

    assert new_value.view(np.int64).tolist() == whole.view(np.int64).tolist()
    assert int(target.sum()) == receipt["rows_rewritten"]
    assert not target[codes == "S14000001"].any()
    assert (new_value[~target] == values[~target]).all()


def test_the_value_half_refuses_a_seat_without_knots_by_a_count_free_text() -> None:
    with pytest.raises(ValueError, match="No value knots") as caught:
        pl.localise_main_residence_values(
            u=np.array([0.5]),
            constituency_code=np.array(["E14999999"]),
            tenure_type=np.array(["OWNED_OUTRIGHT"]),
            value=np.array([1.0]),
            knots=_knots(),
        )
    # The text names no constituency: a row's constituency is its place.
    assert "E14999999" not in str(caught.value)
    assert not any(character.isdigit() for character in str(caught.value))


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


def _scotland_mix() -> pl.BandMix:
    return pl.load_scotland_band_mix(
        json.dumps(toy_property_resources()["scotland_band_mix"]).encode()
    )


def test_scotland_band_mix_needs_every_band_of_every_council_area() -> None:
    resource = toy_property_resources()["scotland_band_mix"]
    mix = pl.load_scotland_band_mix(json.dumps(resource).encode())
    # Scotland's bands run A to H, so band I adds nothing.
    assert all(values[-2] == values[-1] == 1.0 for values in mix.cumulative.values())
    assert mix.as_at == pl.SCOTLAND_BAND_PERIOD
    short = {**resource, "rows": resource["rows"][1:]}
    with pytest.raises(ValueError, match="lacks bands"):
        pl.load_scotland_band_mix(json.dumps(short).encode())
    with pytest.raises(ValueError, match="not the vendored CTAXBASE facts"):
        pl.load_scotland_band_mix(json.dumps({**resource, "kind": "other"}).encode())
    # Only the September 2025 rows are read.
    stale = {
        **resource,
        "rows": [
            {**row, "period": {"type": "month", "value": "2024-09"}}
            for row in resource["rows"]
        ],
    }
    with pytest.raises(ValueError, match="no council areas"):
        pl.load_scotland_band_mix(json.dumps(stale).encode())


def test_localise_bands_draws_scottish_owners_from_their_council_area() -> None:
    """With the CTAXBASE mix a Scottish owner draws its band from its council
    area's shares at its uniform; a Scottish renter and Northern Ireland keep
    theirs (microcosm#1095 B12)."""
    mix, scotland = _mix(), _scotland_mix()
    lsoa = next(iter(mix.cumulative))
    area = next(iter(scotland.cumulative))
    u = np.array([0.0, 0.124, 0.126, 0.999, 0.6, 0.6, 0.3])
    codes = np.array(["S01000001"] * 5 + ["N00000001", lsoa])
    areas = np.array([area] * 5 + ["N09000001", "E06000001"])
    band = np.array(["D"] * 7)
    tenure = np.array(
        ["OWNED_OUTRIGHT"] * 4 + ["RENT_PRIVATELY"] + ["OWNED_OUTRIGHT"] * 2
    )
    new_band, receipt = pl.localise_bands(
        u=u,
        lsoa_code=codes,
        band=band,
        tenure_type=tenure,
        mix=mix,
        scotland_mix=scotland,
        local_authority_code=areas,
    )
    # The toy council area splits A to H in eighths, so no owner reads I.
    assert list(new_band[:4]) == ["A", "A", "B", "H"]
    assert new_band[4] == "D" and new_band[5] == "D"
    # England and Wales draw as before.
    assert new_band[6] == "C"
    scottish = receipt["scotland"]
    assert scottish["rows_rewritten"] == 4 and scottish["rows_untouched_renters"] == 1
    assert scottish["council_areas"] == 1 and scottish["band_mix_after"]["I"] == 0
    assert scottish["as_at"] == "2025-09"
    # Scotland's owners leave the untouched count; the renter and
    # Northern Ireland stay in it.
    assert receipt["rows_untouched_outside_england_wales"] == 2
    assert pl.monotone_violations(u[:4], new_band[:4], areas[:4]) == 0
    with pytest.raises(ValueError, match="No Scottish band mix"):
        pl.localise_bands(
            u=u,
            lsoa_code=codes,
            band=band,
            tenure_type=tenure,
            mix=mix,
            scotland_mix=scotland,
            local_authority_code=np.where(areas == area, "S12999999", areas),
        )
    with pytest.raises(ValueError, match="council areas"):
        pl.localise_bands(
            u=u,
            lsoa_code=codes,
            band=band,
            tenure_type=tenure,
            mix=mix,
            scotland_mix=scotland,
        )


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
