"""Private rents read off the assigned area's list of rents (microcosm#1262).

The UK spine carries each private renter's rent from its source household
(the FRS report, or the SPI housing shell's draw from FRS donors), which
describes its FRS region, not the district the household is placed in. After
geography, ``uk.full.private_rent`` keeps each private renter's rank among
the FRS renters of its region and bedroom size and reads that quantile off
the list of rents of the BRMA its assigned area sits in
(``uk/brma_rent_quantiles.json``), scaled per FRS region so the region's mean
rent stays at the FRS level (María's rulings of 2026-10-15/16): the lists set
where rents sit within a region and their spread within a BRMA, not the
level.

Everything here is pure and depends on the standard library, numpy and
pandas only, so the build tool can load this module by path; the graph
kernel supplies the spine keys and the keyed tie-break uniforms.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "CATEGORIES",
    "COUNTS_SOURCE",
    "GRID",
    "METHOD_VERSION",
    "PRIVATE_RENT_LOCALISATION_MODES",
    "QUANTILES_KIND",
    "QUANTILES_METHOD",
    "QUANTILES_SCHEMA_VERSION",
    "QUANTILES_SOURCE",
    "STREAM_EXPERIMENT",
    "WEEKS_IN_YEAR",
    "RentQuantiles",
    "knot_mean",
    "load_private_renter_counts",
    "load_rent_quantiles",
    "localise_rents",
    "quantile",
    "redraw_mask",
    "region_scales",
    "rent_categories",
    "rent_uniforms",
]

#: ``brma_list`` re-draws each Great Britain private renter's rent from its
#: assigned BRMA's list of rents; ``none`` keeps the spine's rents.
PRIVATE_RENT_LOCALISATION_MODES = ("none", "brma_list")
METHOD_VERSION = "uk-private-rent-localisation-v1"
STREAM_EXPERIMENT = "uk-post-clone-private-rent-v1"
#: Graph source names: the committed rent quantiles and the census counts of
#: private-rented households by region, BRMA and bedrooms (microcosm#1087).
QUANTILES_SOURCE = "uk_brma_rent_quantiles"
COUNTS_SOURCE = "uk_brma_private_renters"
#: The spine's annualisation of weekly amounts (``frs_spine.WEEKS_IN_YEAR``).
WEEKS_IN_YEAR = 365.25 / 7
#: A region's scale outside this band means a resource or the frame is wrong.
SCALE_BAND = (0.25, 4.0)
PRIVATE_TENURE = "RENT_PRIVATELY"
NORTHERN_IRELAND = "NORTHERN_IRELAND"
FRS_CHANNEL = "frs"

QUANTILES_KIND = "uk_brma_rent_quantiles"
QUANTILES_SCHEMA_VERSION = 1
QUANTILES_METHOD = "uk-brma-rent-quantiles-v1"
#: LHA categories by the home's bedrooms: B one (or a studio), C two, D three,
#: E four or more. Category A, a room in shared accommodation, places no one.
CATEGORIES = ("B", "C", "D", "E")
#: Probabilities of the quantile knots; the quantile function is linear
#: between knots and flat beyond the end knots.
GRID = (
    0.005,
    0.01,
    0.025,
    0.05,
    *(round(0.10 + 0.05 * step, 2) for step in range(17)),
    0.95,
    0.975,
    0.99,
    0.995,
)


def knot_mean(grid: Sequence[float], knots: Sequence[float]) -> float:
    """The mean of the quantile function the knots define (flat tails)."""

    p = np.asarray(grid, dtype=float)
    q = np.asarray(knots, dtype=float)
    inner = float(np.sum(np.diff(p) * (q[1:] + q[:-1]) / 2.0))
    return float(p[0] * q[0] + inner + (1.0 - p[-1]) * q[-1])


def quantile(grid: np.ndarray, knots: np.ndarray, u: np.ndarray) -> np.ndarray:
    """The quantile function at ``u``: linear between knots, flat beyond them."""

    return np.interp(np.asarray(u, dtype=float), grid, knots)


@dataclass(frozen=True)
class RentQuantiles:
    """Weekly rent knots by BRMA and category, with their implied means."""

    grid: np.ndarray
    knots: Mapping[tuple[str, str], np.ndarray]
    means: Mapping[tuple[str, str], float]
    method: str


def load_rent_quantiles(payload: bytes) -> RentQuantiles:
    """Load and validate ``uk/brma_rent_quantiles.json``."""

    data = json.loads(payload.decode("utf-8"))
    if (
        data.get("kind") != QUANTILES_KIND
        or data.get("schema_version") != QUANTILES_SCHEMA_VERSION
    ):
        raise ValueError("The rent quantiles resource is not a schema-1 resource.")
    if data.get("method") != QUANTILES_METHOD:
        raise ValueError(
            f"The rent quantiles method is {data.get('method')!r}; "
            f"this build reads {QUANTILES_METHOD!r}."
        )
    if data.get("unit") != "gbp_per_week":
        raise ValueError("The rent quantiles must be weekly pounds.")
    grid = np.asarray(data.get("grid") or (), dtype=float)
    if grid.shape != (len(GRID),) or not np.allclose(grid, GRID):
        raise ValueError("The rent quantiles grid is not the declared grid.")
    if tuple(data.get("categories") or ()) != CATEGORIES:
        raise ValueError("The rent quantiles must carry categories B to E.")
    knots: dict[tuple[str, str], np.ndarray] = {}
    means: dict[tuple[str, str], float] = {}
    brmas = data.get("brmas") or {}
    if not brmas:
        raise ValueError("The rent quantiles resource carries no BRMAs.")
    for brma, cells in brmas.items():
        for category in CATEGORIES:
            entry = cells.get(category)
            if entry is None:
                raise ValueError(f"{brma} has no category {category} rents.")
            values = np.asarray(entry["quantiles"], dtype=float)
            if values.shape != grid.shape:
                raise ValueError(f"{brma} {category} knots do not match the grid.")
            if np.any(~np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"{brma} {category} knots must be positive.")
            if np.any(np.diff(values) < 0):
                raise ValueError(f"{brma} {category} knots must not decrease.")
            mean = knot_mean(grid, values)
            if abs(mean - float(entry["mean"])) > 1e-3:
                raise ValueError(f"{brma} {category} mean is not its knots' mean.")
            knots[(str(brma), category)] = values
            means[(str(brma), category)] = mean
    return RentQuantiles(
        grid=grid, knots=knots, means=means, method=str(data["method"])
    )


def load_private_renter_counts(
    payload: bytes,
) -> dict[tuple[str, str], dict[str, float]]:
    """Census private-rented households by (FRS region, category) and BRMA."""

    data = json.loads(payload.decode("utf-8"))
    cells = data.get("cells") or {}
    if not cells:
        raise ValueError("The private-renter count table carries no cells.")
    out: dict[tuple[str, str], dict[str, float]] = {}
    for region, categories in cells.items():
        for category in CATEGORIES:
            counts = {
                str(brma): float(count)
                for brma, count in (categories.get(category) or {}).items()
                if float(count) > 0
            }
            out[(str(region), category)] = counts
    return out


# ----------------------------------------------------------------------------
# Who moves, and in which cell


def redraw_mask(region, tenure_type, rent) -> np.ndarray:
    """Great Britain private renters with a positive rent, in either channel."""

    region = np.asarray(region).astype(str)
    tenure = np.asarray(tenure_type).astype(str)
    rent = np.asarray(rent, dtype=float)
    return (region != NORTHERN_IRELAND) & (tenure == PRIVATE_TENURE) & (rent > 0)


def rent_categories(num_bedrooms) -> np.ndarray:
    """The home's LHA category: 0-1 bedrooms B, 2 C, 3 D, 4 or more E."""

    values = np.asarray(num_bedrooms, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            "A private renter's bedroom count must be a non-negative number."
        )
    if np.any(values != np.floor(values)):
        raise ValueError("A private renter's bedroom count must be a whole number.")
    out = np.full(values.shape, "E", dtype=object)
    out[values <= 1] = "B"
    out[values == 2] = "C"
    out[values == 3] = "D"
    return out


def _spine_rows(*, spine_key, region, category, rent, channel, weight) -> pd.DataFrame:
    """One row per spine household: its clones' summed weight and shared cell."""

    frame = pd.DataFrame(
        {
            "key": np.asarray(spine_key, dtype=object),
            "region": np.asarray(region).astype(str),
            "category": np.asarray(category, dtype=object),
            "rent": np.asarray(rent, dtype=float),
            "channel": np.asarray(channel).astype(str),
            "weight": np.asarray(weight, dtype=float),
        }
    )
    grouped = frame.groupby("key", sort=True)
    spine = grouped.agg(
        region=("region", "first"),
        regions=("region", "nunique"),
        category=("category", "first"),
        categories=("category", "nunique"),
        rent=("rent", "first"),
        rent_min=("rent", "min"),
        rent_max=("rent", "max"),
        channel=("channel", "first"),
        channels=("channel", "nunique"),
        weight=("weight", "sum"),
    )
    if (
        (spine["regions"] != 1).any()
        or (spine["categories"] != 1).any()
        or (spine["channels"] != 1).any()
        or (spine["rent_max"] - spine["rent_min"]).abs().max() > 1e-9
    ):
        raise ValueError(
            "Pool clones of one spine household disagree on region, bedrooms, "
            "channel or rent."
        )
    return spine[["region", "category", "rent", "channel", "weight"]]


# ----------------------------------------------------------------------------
# Ranks and scales


def rent_uniforms(
    spine: pd.DataFrame, tie_uniform: Mapping[str, float]
) -> tuple[pd.Series, dict[str, int]]:
    """Each spine household's rank among its cell's FRS renters.

    ``u = F(x-) + v (F(x) - F(x-))`` with ``F`` the weighted distribution of
    the FRS-channel rents of the household's region and category and ``v`` its
    keyed tie-break uniform, so FRS renters' ranks are uniform and SPI-channel
    renters land where their rent sits among them. A cell with no FRS renter
    ranks against its region's FRS renters of every size.
    """

    reference = spine[(spine["channel"] == FRS_CHANNEL) & (spine["weight"] > 0)]
    u = pd.Series(np.nan, index=spine.index, dtype=float)
    fallback_cells = 0
    for (region, category), rows in spine.groupby(["region", "category"], sort=True):
        pool = reference[
            (reference["region"] == region) & (reference["category"] == category)
        ]
        if pool.empty:
            pool = reference[reference["region"] == region]
            fallback_cells += 1
        if pool.empty or float(pool["weight"].sum()) <= 0:
            raise ValueError(f"No FRS private renter carries weight in {region}.")
        order = np.argsort(pool["rent"].to_numpy(), kind="stable")
        values = pool["rent"].to_numpy()[order]
        cumulative = np.concatenate(
            [[0.0], np.cumsum(pool["weight"].to_numpy()[order])]
        )
        total = cumulative[-1]
        x = rows["rent"].to_numpy()
        below = cumulative[np.searchsorted(values, x, side="left")] / total
        at_or_below = cumulative[np.searchsorted(values, x, side="right")] / total
        v = np.asarray([float(tie_uniform[key]) for key in rows.index], dtype=float)
        u.loc[rows.index] = np.clip(below + v * (at_or_below - below), 0.0, 1.0)
    if u.isna().any():
        raise ValueError("A private renter was left without a rank.")
    return u, {"fallback_cells": fallback_cells}


def region_scales(
    spine: pd.DataFrame,
    *,
    counts: Mapping[tuple[str, str], Mapping[str, float]],
    quantiles: RentQuantiles,
) -> dict[str, float]:
    """Each region's scale on the lists, holding its FRS mean rent.

    ``s_r * sum_c p(c|r) sum_b pi(b|r,c) mu(b,c) = m_r``: ``m_r`` and ``p(c|r)``
    are the FRS-channel renters' weighted mean rent and category shares,
    ``pi`` the census private-rented households by BRMA within the region and
    bedroom band, ``mu`` each list's mean annual rent.
    """

    reference = spine[(spine["channel"] == FRS_CHANNEL) & (spine["weight"] > 0)]
    scales: dict[str, float] = {}
    for region, rows in reference.groupby("region", sort=True):
        weight = rows["weight"].to_numpy()
        mean_rent = float(np.average(rows["rent"].to_numpy(), weights=weight))
        shares = rows.groupby("category")["weight"].sum() / weight.sum()
        level = 0.0
        for category, share in shares.items():
            cell = counts.get((region, str(category))) or {}
            total = sum(cell.values())
            if total <= 0:
                raise ValueError(f"No census private renters for {region} {category}.")
            for brma, households in cell.items():
                key = (brma, str(category))
                if key not in quantiles.means:
                    raise ValueError(f"No list of rents for {brma} {category}.")
                level += (
                    share * households / total * quantiles.means[key] * WEEKS_IN_YEAR
                )
        if level <= 0:
            raise ValueError(f"The lists give {region} no rent level.")
        scale = mean_rent / level
        if not np.isfinite(scale) or not SCALE_BAND[0] <= scale <= SCALE_BAND[1]:
            raise ValueError(
                f"{region}'s scale on the lists is outside {SCALE_BAND}; a resource "
                "or the frame's rents are wrong."
            )
        scales[str(region)] = float(scale)
    return scales


def localise_rents(
    *,
    spine_key,
    region,
    tenure_type,
    rent,
    num_bedrooms,
    brma,
    channel,
    weight,
    tie_uniform: Mapping[str, float],
    quantiles: RentQuantiles,
    counts: Mapping[tuple[str, str], Mapping[str, float]],
) -> tuple[np.ndarray, dict[str, int]]:
    """Each Great Britain private renter's rent off its assigned BRMA's list.

    The rent is ``WEEKS_IN_YEAR * s_r * Q_bc(u)``: ``u`` the household's rank
    among its region and category's FRS renters (shared by every pool clone
    of a spine household), ``Q_bc`` its assigned BRMA and category's weekly
    quantiles, ``s_r`` its region's scale. Every other row keeps its rent bit
    for bit. Returns exact counts; callers round them before any receipt.
    """

    rent = np.asarray(rent, dtype=float)
    out = rent.copy()
    mask = redraw_mask(region, tenure_type, rent)
    details = {"rows_redrawn": int(mask.sum()), "spine_households_redrawn": 0}
    if not mask.any():
        return out, {**details, "fallback_cells": 0, "regions": 0}
    rows = np.flatnonzero(mask)
    keys = np.asarray(spine_key, dtype=object)[rows]
    categories = rent_categories(np.asarray(num_bedrooms)[rows])
    regions = np.asarray(region).astype(str)[rows]
    spine = _spine_rows(
        spine_key=keys,
        region=regions,
        category=categories,
        rent=rent[rows],
        channel=np.asarray(channel)[rows],
        weight=np.asarray(weight, dtype=float)[rows],
    )
    u, rank_details = rent_uniforms(spine, tie_uniform)
    scales = region_scales(spine, counts=counts, quantiles=quantiles)
    missing = sorted(set(regions) - set(scales))
    if missing:
        raise ValueError(f"No FRS private renter to scale the regions {missing}.")
    cells = pd.DataFrame(
        {
            "brma": np.asarray(brma).astype(str)[rows],
            "category": categories,
            "u": u.reindex(keys).to_numpy(dtype=float),
            "scale": [scales[code] for code in regions],
        }
    )
    for (code, category), group in cells.groupby(["brma", "category"], sort=True):
        knots = quantiles.knots.get((str(code), str(category)))
        if knots is None:
            raise ValueError(f"No list of rents for {code} {category}.")
        positions = group.index.to_numpy()
        out[rows[positions]] = (
            WEEKS_IN_YEAR
            * group["scale"].to_numpy()
            * np.interp(group["u"].to_numpy(), quantiles.grid, knots)
        )
    return out, {
        **details,
        "spine_households_redrawn": int(len(spine)),
        "fallback_cells": int(rank_details["fallback_cells"]),
        "regions": len(scales),
    }
