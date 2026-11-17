"""Owner ranks, home-value knots and band mixes by area (microcosm#1240, microcosm#931).

The spine's WAS chain reads each England and Wales owner's main residence off
its assigned constituency's knots (``uk/high_value_home_value_knots.json``,
winsorised at the declared cap), at a keyed point inside the owner's weighted
slice of its constituency and geographic clone (``area_home_values``, G8).
This module holds what that step reads: the knots loader, the owner mask, the
slices with their keyed draws (:func:`owner_slice_uniforms`) and the read
itself (:func:`localise_main_residence_values`). ``high_value_home_support`` takes its
stratum from :data:`SUPPORT_STRATUM_LOWER`, and ``tools/uk_private_rent_pool.py``
keys households with :func:`spine_household_keys`.

These functions first served the dense line's two post-geography nodes,
``uk.full.property.value`` and ``uk.full.property.band`` (microcosm#1240), which
microcosm#931 retired. No stage reads the band mixes or the band draw now
(:func:`load_band_mix`, :func:`load_scotland_band_mix`, :func:`localise_bands`),
nor :func:`localise_values`.

Everything here is pure: callers hand these functions arrays and resources and
write the results back.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.atomic_household_identity import household_draw_key
from microcosm.build.uk_runtime.household_lineage import (
    uk_household_clone_paths,
    without_geographic_branch,
)
from microcosm.build.uk_runtime.tenure_constants import (
    UK_TENURE_OWNED_OUTRIGHT,
    UK_TENURE_OWNED_WITH_MORTGAGE,
    UK_TENURE_TYPE_TO_CATEGORY,
)
from microcosm.graph.randomness import keyed_uniform

__all__ = [
    "BAND_MIX_SOURCE",
    "COUNCIL_TAX_BANDS",
    "METHOD_VERSION",
    "OWNER_RANK_BASIS",
    "PROPERTY_LOCALISATION_MODES",
    "PROPERTY_STREAM_EXPERIMENT",
    "RANK_BASIS_COLUMN",
    "SCOTLAND_BAND_MIX_RESOURCE",
    "SCOTLAND_BAND_MIX_SOURCE",
    "SCOTLAND_BAND_PERIOD",
    "SUPPORT_STRATUM_LOWER",
    "QUANTILES_SOURCE",
    "BandMix",
    "ValueKnots",
    "load_band_mix",
    "load_scotland_band_mix",
    "load_value_knots",
    "localise_bands",
    "localise_main_residence_values",
    "localise_values",
    "owner_slice_uniforms",
    "spine_household_keys",
]

QUANTILES_SOURCE = "uk_price_paid_quantiles"
BAND_MIX_SOURCE = "uk_voa_band_mix"
#: The Scottish Government's chargeable dwellings by council-tax band per
#: council area (CTAXBASE), vendored from Chronicle (microcosm#1095 B12).
SCOTLAND_BAND_MIX_SOURCE = "uk_scotgov_band_mix"
SCOTLAND_BAND_MIX_RESOURCE = "scotgov_chargeable_dwellings_by_band.json"
SCOTLAND_BAND_PERIOD = "2025-09"
_SCOTLAND_BAND_CONCEPT = "scotgov.chargeable_dwellings_band_"
_SCOTTISH_COUNCIL_AREA_PREFIX = "S12"
PROPERTY_LOCALISATION_MODES: tuple[str, ...] = ("none", "price_paid")
PROPERTY_STREAM_EXPERIMENT = "uk-post-clone-property-localisation-v1"
METHOD_VERSION = "uk-property-localisation-v1"
#: Owner households at or above this within-region rank are the high-value
#: support stratum (microcosm#1240 fix 1, ruled top 5%).
SUPPORT_STRATUM_LOWER = 0.95
#: The WAS chain's segment-2 draw of the main residence, which its area home
#: values keep beside the value they read off the owner's constituency and
#: the regional property uprating divides by (``area_home_values``,
#: microcosm#931). The release boundary drops it.
RANK_BASIS_COLUMN = "main_residence_value_rank_basis"
COUNCIL_TAX_BANDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
KNOTS_KIND = "uk_high_value_home_value_knots"
BAND_MIX_KIND = "uk_voa_council_tax_band_mix_lsoa"
_OWNER_CATEGORIES = frozenset({UK_TENURE_OWNED_OUTRIGHT, UK_TENURE_OWNED_WITH_MORTGAGE})
#: The process names of the owners' keyed draws: the tie-break among equal
#: values and the point inside each row's slice. The spine keys each row on its
#: own id (``area_home_values``, microcosm#931).
_TIE_BREAK = "tie_break"
_SLICE = "slice"
_IDENTITY = "identity"
#: The rank basis :func:`owner_slice_uniforms` reports, which the home-value
#: ranking gate declares (microcosm#931).
OWNER_RANK_BASIS = "weighted_slice_within_constituency_and_clone"


# ----------------------------------------------------------------------------
# Resources


@dataclass(frozen=True)
class ValueKnots:
    """Per-constituency weighted quantiles of the basis value."""

    grid: np.ndarray
    quantiles: Mapping[str, np.ndarray]
    cap: float
    basis_year: int
    inputs_sha256: str
    method: str


@dataclass(frozen=True)
class BandMix:
    """Per-LSOA cumulative band shares, A to I."""

    cumulative: Mapping[str, np.ndarray]
    source_sha256: str
    as_at: str


def load_value_knots(payload: bytes) -> ValueKnots:
    data = json.loads(payload.decode("utf-8"))
    if data.get("kind") != KNOTS_KIND or data.get("schema_version") != 1:
        raise ValueError("The value knots resource is not a schema-1 knots resource.")
    grid = np.asarray(data["grid"], dtype=float)
    if grid.ndim != 1 or len(grid) < 2 or grid[0] != 0.0 or grid[-1] != 1.0:
        raise ValueError("The value knots grid must run from 0 to 1.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("The value knots grid must increase strictly.")
    cap = float(data["cap"])
    if not cap > 0:
        raise ValueError("The value knots cap must be positive.")
    levels = data.get("levels") or {}
    seats = levels.get("constituency") or {}
    if not seats:
        raise ValueError("The value knots resource carries no constituency knots.")
    quantiles: dict[str, np.ndarray] = {}
    for code, entry in seats.items():
        values = np.asarray(entry["quantiles"], dtype=float)
        if values.shape != grid.shape:
            raise ValueError(f"Constituency {code} knots do not match the grid.")
        if np.any(~np.isfinite(values)) or np.any(values < 0) or np.any(values > cap):
            raise ValueError(f"Constituency {code} knots are outside [0, cap].")
        if np.any(np.diff(values) < 0):
            raise ValueError(f"Constituency {code} knots are not non-decreasing.")
        quantiles[str(code)] = values
    return ValueKnots(
        grid=grid,
        quantiles=quantiles,
        cap=cap,
        basis_year=int(data["basis_year"]),
        inputs_sha256=str(data.get("inputs_sha256") or ""),
        method=str(data.get("method") or ""),
    )


def load_band_mix(payload: bytes) -> BandMix:
    data = json.loads(payload.decode("utf-8"))
    if data.get("kind") != BAND_MIX_KIND or data.get("schema_version") != 1:
        raise ValueError("The band mix resource is not a schema-1 VOA band mix.")
    if tuple(data.get("bands") or ()) != COUNCIL_TAX_BANDS:
        raise ValueError("The band mix resource must carry bands A to I in order.")
    cumulative: dict[str, np.ndarray] = {}
    for code, entry in (data.get("lsoa") or {}).items():
        values = np.asarray(entry["cumulative"], dtype=float)
        if values.shape != (len(COUNCIL_TAX_BANDS),):
            raise ValueError(f"LSOA {code} carries {len(values)} cumulative shares.")
        if np.any(np.diff(values) < -1e-9) or values[-1] != 1.0 or values[0] < 0:
            raise ValueError(f"LSOA {code} cumulative shares are malformed.")
        cumulative[str(code)] = values
    if not cumulative:
        raise ValueError("The band mix resource carries no LSOA rows.")
    source = data.get("source") or {}
    return BandMix(
        cumulative=cumulative,
        source_sha256=str(source.get("sha256") or ""),
        as_at=str(source.get("as_at") or ""),
    )


def load_scotland_band_mix(payload: bytes) -> BandMix:
    """Per-council-area cumulative band shares from the vendored CTAXBASE rows.

    Scotland's bands run A to H, so a council area's share of band I is zero
    and its cumulative shares reach one at H. Every council area must carry
    every band once, at ``SCOTLAND_BAND_PERIOD``.
    """
    data = json.loads(payload.decode("utf-8"))
    if (
        data.get("kind") != "uk_vendored_ledger_facts"
        or data.get("resource") != SCOTLAND_BAND_MIX_RESOURCE
    ):
        raise ValueError(
            "The Scottish band mix resource is not the vendored CTAXBASE facts."
        )
    scottish_bands = COUNCIL_TAX_BANDS[:-1]
    counts: dict[str, dict[str, float]] = {}
    for row in data.get("rows") or ():
        concept = str(row.get("concept") or "")
        geography = row.get("geography") or {}
        period = row.get("period") or {}
        if (
            not concept.startswith(_SCOTLAND_BAND_CONCEPT)
            or geography.get("level") != "local_authority"
            or str(period.get("value")) != SCOTLAND_BAND_PERIOD
        ):
            continue
        band = concept[len(_SCOTLAND_BAND_CONCEPT) :].upper()
        if band not in scottish_bands:
            raise ValueError(f"Unknown Scottish council-tax band {band!r}.")
        cells = counts.setdefault(str(geography.get("id")), {})
        if band in cells:
            raise ValueError(f"Council area {geography.get('id')} repeats band {band}.")
        value = float(row["value"])
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"Council area {geography.get('id')} has a bad band count."
            )
        cells[band] = value
    cumulative: dict[str, np.ndarray] = {}
    for code, cells in sorted(counts.items()):
        if set(cells) != set(scottish_bands):
            raise ValueError(
                f"Council area {code} lacks bands {sorted(set(scottish_bands) - set(cells))}."
            )
        values = np.asarray([cells[band] for band in scottish_bands], dtype=float)
        if not values.sum() > 0:
            raise ValueError(f"Council area {code} has no chargeable dwellings.")
        shares = np.cumsum(values) / values.sum()
        shares[-1] = 1.0
        cumulative[code] = np.append(shares, 1.0)
    if not cumulative:
        raise ValueError("The Scottish band mix resource carries no council areas.")
    feed = data.get("source_fact_feed") or {}
    return BandMix(
        cumulative=cumulative,
        source_sha256=str(feed.get("facts_sha256") or ""),
        as_at=SCOTLAND_BAND_PERIOD,
    )


# ----------------------------------------------------------------------------
# Keys and uniforms


def spine_household_keys(
    household: pd.DataFrame, *, source: str, source_vintage: str
) -> np.ndarray:
    """Each household's key over its clone path without the geographic
    branch, so a record's geographic clones share one.

    ``tools/uk_private_rent_pool.py`` matches the pool rows of dense runs
    built before microcosm#931 to their spine households with it.
    """
    source_ids, paths = uk_household_clone_paths(household)
    spine_paths = without_geographic_branch(paths)
    keys = [
        household_draw_key(
            source=source,
            source_vintage=source_vintage,
            source_household_id=source_id,
            clone_path=path,
        )
        for source_id, path in zip(source_ids, spine_paths, strict=True)
    ]
    return np.asarray(keys, dtype=object)


def owner_mask(tenure_type: np.ndarray) -> np.ndarray:
    categories = np.array(
        [
            UK_TENURE_TYPE_TO_CATEGORY.get(str(key), "")
            for key in np.asarray(tenure_type)
        ],
        dtype=object,
    )
    return np.isin(categories, list(_OWNER_CATEGORIES))


def owner_slice_uniforms(
    *,
    key: np.ndarray,
    cell: np.ndarray,
    stratum: np.ndarray,
    tenure_type: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
    stream: tuple,
) -> tuple[np.ndarray, dict[str, object]]:
    """The uniform each owner row reads its constituency's knots at.

    The ranked rows are the owners with a positive ``value`` whose ``cell``
    (the assigned constituency) is in England or Wales. They are ranked within
    their cell and ``stratum`` (the geographic clone index): in order of
    value, ties broken by a uniform keyed on the row's ``key``, each row holds
    a slice of its cell's weight as wide as its own weight, and takes a second
    keyed uniform point inside that slice (María's ruling of 17 Nov 2026). So
    a cell's weighted values follow its knots in expectation however few rows
    it holds, and its top row can reach the top of the knots. Geographic clone
    0's rows form the same cells as a one-clone frame's rows, so they take the
    same uniforms, to rounding. A cell whose ranked rows carry no weight gives
    each row an equal slice. Every other row takes an identity-keyed uniform,
    which no caller reads. A row's uniform depends on its own key and its
    cell's rows, never on the table's row order.

    Refused, by name: inputs that do not align; a key that two rows share;
    weights that are not finite or are negative.
    """

    key = np.asarray(key, dtype=object).astype(str)
    cell = np.asarray(cell, dtype=object).astype(str)
    stratum = np.asarray(stratum, dtype=np.int64)
    tenure_type = np.asarray(tenure_type)
    value = np.asarray(value, dtype=float)
    weight = np.asarray(weight, dtype=float)
    count = len(key)
    if not (
        len(cell)
        == len(stratum)
        == len(tenure_type)
        == len(value)
        == len(weight)
        == count
    ):
        raise ValueError("owner_slice_uniforms inputs must align.")
    if len(np.unique(key)) != count:
        raise ValueError("owner_slice_uniforms keys each row on its own key.")
    if not np.isfinite(weight).all() or np.any(weight < 0):
        raise ValueError("owner_slice_uniforms reads finite, non-negative weights.")
    ranked = (
        owner_mask(tenure_type)
        & np.isfinite(value)
        & (value > 0)
        & _england_wales(cell)
    )
    u = np.empty(count, dtype=float)
    u[~ranked] = keyed_uniform(
        stream=stream, keys=[(k, _IDENTITY) for k in key[~ranked]]
    )
    rows = np.flatnonzero(ranked)
    ranked_keys = key[rows]
    tie_break = np.asarray(
        keyed_uniform(stream=stream, keys=[(k, _TIE_BREAK) for k in ranked_keys])
    )
    point = np.asarray(
        keyed_uniform(stream=stream, keys=[(k, _SLICE) for k in ranked_keys])
    )
    _, cell_id = np.unique(cell[rows], return_inverse=True)
    _, key_id = np.unique(ranked_keys, return_inverse=True)
    ranked_stratum = stratum[rows]
    order = np.lexsort((key_id, tie_break, value[rows], ranked_stratum, cell_id))
    boundary = np.ones(len(order), dtype=bool)
    boundary[1:] = (cell_id[order][1:] != cell_id[order][:-1]) | (
        ranked_stratum[order][1:] != ranked_stratum[order][:-1]
    )
    starts = np.flatnonzero(boundary)
    ends = np.append(starts[1:], len(order))
    below_one = np.nextafter(1.0, 0.0)
    unweighted = 0
    for begin, end in zip(starts, ends, strict=True):
        members = order[begin:end]
        cell_weight = weight[rows[members]]
        upper = np.cumsum(cell_weight)
        if upper[-1] <= 0:
            cell_weight = np.ones(len(members), dtype=float)
            upper = np.cumsum(cell_weight)
            unweighted += 1
        lower = np.concatenate(([0.0], upper[:-1]))
        u[rows[members]] = np.minimum(
            (lower + point[members] * cell_weight) / upper[-1], below_one
        )
    if np.any(~np.isfinite(u)) or np.any(u < 0) or np.any(u >= 1):
        raise ValueError("Property uniforms must lie in [0, 1).")
    receipt = {
        "ranked_rows": int(len(rows)),
        "cells": int(len(starts)),
        "cells_without_weight": int(unweighted),
        "rank_basis": OWNER_RANK_BASIS,
    }
    return u, receipt


# ----------------------------------------------------------------------------
# Maps


def _england_wales(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes).astype(str)
    return np.char.startswith(codes, "E") | np.char.startswith(codes, "W")


def localise_main_residence_values(
    *,
    u: np.ndarray,
    constituency_code: np.ndarray,
    tenure_type: np.ndarray,
    value: np.ndarray,
    knots: ValueKnots,
) -> tuple[np.ndarray, np.ndarray]:
    """Read owner rows' values off their constituency's knots at ``u``.

    The value half of :func:`localise_values`, which the spine's WAS chain
    reads its home values through (``area_home_values``, microcosm#931).
    Returns every row's value and the rows it rewrote: the owner-occupied rows
    with a positive value whose constituency is in England or Wales. Every
    other row keeps its value bit for bit. A rewritten row's constituency
    without knots fails closed, by a text that names no constituency: a row's
    constituency is its household's place.
    """
    u = np.asarray(u, dtype=float)
    value = np.asarray(value, dtype=float)
    codes = np.asarray(constituency_code).astype(str)
    target = owner_mask(tenure_type) & (value > 0) & _england_wales(codes)
    new_value = value.copy()
    if not set(codes[target]) <= set(knots.quantiles):
        raise ValueError(
            "No value knots for a constituency an England or Wales owner row is "
            "assigned to: the knots resource carries every constituency the "
            "area draw places owners in."
        )
    for code in sorted(set(codes[target])):
        rows = np.flatnonzero(target & (codes == code))
        new_value[rows] = np.interp(u[rows], knots.grid, knots.quantiles[code])
    if np.any(~np.isfinite(new_value)) or np.any(new_value < 0):
        raise ValueError("Localised home values must be finite and non-negative.")
    return new_value, target


def localise_values(
    *,
    u: np.ndarray,
    constituency_code: np.ndarray,
    tenure_type: np.ndarray,
    value: np.ndarray,
    property_wealth: np.ndarray,
    knots: ValueKnots,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Read owner rows' values off their constituency's knots at ``u``.

    Returns the new values (:func:`localise_main_residence_values`'), the new
    property wealth (the old total moved by the value's change, so the wealth
    identity holds) and a receipt. Rows that are not owner-occupied, carry no
    value or sit outside England and Wales are untouched; a constituency
    without knots fails closed.
    """
    u = np.asarray(u, dtype=float)
    value = np.asarray(value, dtype=float)
    wealth = np.asarray(property_wealth, dtype=float)
    codes = np.asarray(constituency_code).astype(str)
    new_value, target = localise_main_residence_values(
        u=u, constituency_code=codes, tenure_type=tenure_type, value=value, knots=knots
    )
    delta = new_value - value
    new_wealth = wealth + delta
    if np.any(~np.isfinite(new_wealth)):
        raise ValueError("Localised property wealth must be finite.")
    new_wealth = np.maximum(new_wealth, 0.0)
    receipt = {
        "rows": int(len(value)),
        "rows_rewritten": int(target.sum()),
        "rows_untouched": {
            "not_owner_occupied": int((~owner_mask(tenure_type)).sum()),
            "zero_value_owner": int((owner_mask(tenure_type) & (value <= 0)).sum()),
            "outside_england_wales": int(
                (owner_mask(tenure_type) & (value > 0) & ~_england_wales(codes)).sum()
            ),
        },
        "rows_at_cap": int((target & (new_value >= knots.cap)).sum()),
        "cap": knots.cap,
        "constituencies": int(len(set(codes[target]))),
        "value_sum_before": float(value[target].sum()),
        "value_sum_after": float(new_value[target].sum()),
        "method": METHOD_VERSION,
    }
    return new_value, new_wealth, receipt


def localise_bands(
    *,
    u: np.ndarray,
    lsoa_code: np.ndarray,
    band: np.ndarray,
    tenure_type: np.ndarray,
    mix: BandMix,
    scotland_mix: BandMix | None = None,
    local_authority_code: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Draw each England-and-Wales owner's band from its LSOA's cumulative shares at
    ``u``, and with ``scotland_mix`` each Scottish owner's from its council area's
    (microcosm#1095 B12).

    Renters keep the spine's band: the VOA and CTAXBASE mixes count every
    dwelling, and drawing a social tenant from them would hand the owners' share
    of the high bands to households the retired value node left untouched. Without
    ``scotland_mix`` Scotland keeps its bands, as Northern Ireland always does.
    No stage has called this function since microcosm#931.
    """
    u = np.asarray(u, dtype=float)
    codes = np.asarray(lsoa_code).astype(str)
    current = np.asarray(band).astype(str)
    owner = owner_mask(tenure_type)
    england_wales = _england_wales(codes)
    target = england_wales & owner
    missing = sorted(set(codes[target]) - set(mix.cumulative))
    if missing:
        raise ValueError(f"No VOA band mix for LSOA {missing[:5]}.")
    new_band = current.copy()
    _draw_bands(new_band, target, codes, mix, u)
    scotland = np.zeros(len(current), dtype=bool)
    areas = np.full(len(current), "", dtype=object)
    if scotland_mix is not None:
        if local_authority_code is None:
            raise ValueError("A Scottish band mix needs the rows' council areas.")
        areas = np.asarray(local_authority_code).astype(str)
        scotland = (
            np.char.startswith(areas, _SCOTTISH_COUNCIL_AREA_PREFIX) & ~england_wales
        )
        missing = sorted(set(areas[scotland & owner]) - set(scotland_mix.cumulative))
        if missing:
            raise ValueError(f"No Scottish band mix for council area {missing[:5]}.")
        _draw_bands(new_band, scotland & owner, areas, scotland_mix, u)
    scottish_owner = scotland & owner
    changed = target & (new_band != current)
    receipt = {
        "rows": int(len(current)),
        "rows_rewritten": int(target.sum()),
        "rows_changed": int(changed.sum()),
        "rows_untouched_outside_england_wales": int(
            (~england_wales & ~scottish_owner).sum()
        ),
        "rows_untouched_renters_england_wales": int((england_wales & ~owner).sum()),
        "lsoas": int(len(set(codes[target]))),
        "band_mix_before": {
            b: int((current[target] == b).sum()) for b in COUNCIL_TAX_BANDS
        },
        "band_mix_after": {
            b: int((new_band[target] == b).sum()) for b in COUNCIL_TAX_BANDS
        },
        "voa_as_at": mix.as_at,
        "method": METHOD_VERSION,
    }
    if scotland_mix is not None:
        receipt["scotland"] = {
            "rows_rewritten": int(scottish_owner.sum()),
            "rows_changed": int((scottish_owner & (new_band != current)).sum()),
            "rows_untouched_renters": int((scotland & ~owner).sum()),
            "council_areas": int(len(set(areas[scottish_owner]))),
            "band_mix_before": {
                b: int((current[scottish_owner] == b).sum()) for b in COUNCIL_TAX_BANDS
            },
            "band_mix_after": {
                b: int((new_band[scottish_owner] == b).sum()) for b in COUNCIL_TAX_BANDS
            },
            "as_at": scotland_mix.as_at,
        }
    return new_band, receipt


def _draw_bands(
    new_band: np.ndarray,
    target: np.ndarray,
    codes: np.ndarray,
    mix: BandMix,
    u: np.ndarray,
) -> None:
    """Write each target row's band from its area's cumulative shares at ``u``."""
    for code in sorted(set(codes[target])):
        rows = np.flatnonzero(target & (codes == code))
        index = np.searchsorted(mix.cumulative[code], u[rows], side="right")
        index = np.minimum(index, len(COUNCIL_TAX_BANDS) - 1)
        new_band[rows] = np.asarray(COUNCIL_TAX_BANDS, dtype=object)[index]


def monotone_violations(u: np.ndarray, band: np.ndarray, lsoa_code: np.ndarray) -> int:
    """Count LSOAs where a higher uniform drew a lower band (expected zero)."""
    frame = pd.DataFrame(
        {
            "u": np.asarray(u, dtype=float),
            "band": np.asarray(band).astype(str),
            "lsoa": np.asarray(lsoa_code).astype(str),
        }
    )
    order = {b: i for i, b in enumerate(COUNCIL_TAX_BANDS)}
    frame["ordinal"] = frame["band"].map(order).fillna(-1)
    violations = 0
    for _, group in frame.groupby("lsoa", sort=False):
        ordered = group.sort_values("u")["ordinal"].to_numpy()
        if np.any(np.diff(ordered) < 0):
            violations += 1
    return int(violations)


def spine_keys_are_clone_invariant(
    keys: Sequence[str], clone_index: Sequence[int]
) -> bool:
    """True when every spine key appears once per clone index it carries."""
    frame = pd.DataFrame({"key": list(keys), "clone": list(clone_index)})
    return bool(
        (frame.groupby("key")["clone"].nunique() == frame.groupby("key").size()).all()
    )
