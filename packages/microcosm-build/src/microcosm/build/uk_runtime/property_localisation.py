"""Per-clone home value and council-tax band from the assigned area (microcosm#1240).

The UK spine imputes ``main_residence_value`` from WAS and carries
``council_tax_band`` from the FRS before any household has an area, so every
geographic clone of a household used to carry the same value and band
wherever it landed. Two post-geography nodes in the dense graph fix that:

- ``uk.full.property.value`` keeps each owner household's rank among the
  owner households of its FRS region (a weighted mid-rank of the stored
  value, ties broken by an identity-keyed uniform; the same uniform for every
  pool clone of one spine household) and reads the clone's value off its
  assigned constituency's uprated latest-sale distribution
  (``uk/high_value_home_value_knots.json``, winsorised at the declared cap).
  Non-owner rows and rows outside England and Wales keep their value.
- ``uk.full.property.band`` draws the clone's band from its assigned LSOA's
  VOA band mix (``uk/voa_council_tax_band_mix_lsoa.json``) at the same
  uniform, so value and band agree by construction. Rows outside England and
  Wales keep their band.

Everything here is pure: the kernels in ``graph_population`` hand these
functions arrays and resources and write the results back.
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
    "PROPERTY_LOCALISATION_MODES",
    "PROPERTY_STREAM_EXPERIMENT",
    "SUPPORT_STRATUM_LOWER",
    "QUANTILES_SOURCE",
    "BandMix",
    "ValueKnots",
    "load_band_mix",
    "load_value_knots",
    "localise_bands",
    "localise_values",
    "owner_uniforms",
    "spine_household_keys",
]

QUANTILES_SOURCE = "uk_price_paid_quantiles"
BAND_MIX_SOURCE = "uk_voa_band_mix"
PROPERTY_LOCALISATION_MODES: tuple[str, ...] = ("none", "price_paid")
PROPERTY_STREAM_EXPERIMENT = "uk-post-clone-property-localisation-v1"
METHOD_VERSION = "uk-property-localisation-v1"
#: Owner households at or above this within-region rank are the high-value
#: support stratum (microcosm#1240 fix 1, ruled top 5%).
SUPPORT_STRATUM_LOWER = 0.95
COUNCIL_TAX_BANDS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
KNOTS_KIND = "uk_high_value_home_value_knots"
BAND_MIX_KIND = "uk_voa_council_tax_band_mix_lsoa"
_OWNER_CATEGORIES = frozenset({UK_TENURE_OWNED_OUTRIGHT, UK_TENURE_OWNED_WITH_MORTGAGE})
#: The pool clones of one spine household share one draw: the key is the
#: spine path, never the geographic clone branch.
_TIE_BREAK = "tie_break"
_IDENTITY = "identity"
_SUPPORT = "support"


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


# ----------------------------------------------------------------------------
# Keys and uniforms


def spine_household_keys(
    household: pd.DataFrame, *, source: str, source_vintage: str
) -> np.ndarray:
    """One key per spine household, shared by all of its pool clones."""
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


def owner_uniforms(
    *,
    spine_key: np.ndarray,
    region: np.ndarray,
    tenure_type: np.ndarray,
    value: np.ndarray,
    weight: np.ndarray,
    stream: tuple,
    support_copy: np.ndarray | None = None,
    support_stratum_lower: float = 0.95,
) -> tuple[np.ndarray, dict[str, object]]:
    """The uniform each pool row maps its home value and band through.

    Owner rows with a positive stored value take their spine household's
    weighted mid-rank among the owner spine households of the same FRS
    region (clone weights summed per spine household; ties broken by a keyed
    uniform). Every other row takes an identity-keyed uniform. High-value
    support copies (microcosm#1240 fix 1) take a keyed uniform inside the top
    stratum ``[support_stratum_lower, 1)`` instead of their rank, so the copies
    of one tail household spread across the stratum rather than share one
    rank. The uniform is a function of the spine key alone, so it is identical
    for every pool clone of one spine household and never depends on K.
    """
    spine_key = np.asarray(spine_key, dtype=object)
    region = np.asarray(region).astype(str)
    value = np.asarray(value, dtype=float)
    weight = np.asarray(weight, dtype=float)
    owner = owner_mask(tenure_type)
    if not 0 < support_stratum_lower < 1:
        raise ValueError("support_stratum_lower must be in (0, 1).")
    if support_copy is None:
        support_copy = np.zeros(len(spine_key), dtype=bool)
    support_copy = np.asarray(support_copy, dtype=bool)
    if not (
        len(region)
        == len(value)
        == len(weight)
        == len(owner)
        == len(spine_key)
        == len(support_copy)
    ):
        raise ValueError("owner_uniforms inputs must align.")

    frame = pd.DataFrame(
        {
            "key": spine_key,
            "region": region,
            "owner": owner,
            "value": value,
            "weight": weight,
            "support": support_copy,
        }
    )
    grouped = frame.groupby("key", sort=True)
    spine = grouped.agg(
        region=("region", "first"),
        region_n=("region", "nunique"),
        owner=("owner", "first"),
        owner_n=("owner", "nunique"),
        value=("value", "first"),
        value_min=("value", "min"),
        value_max=("value", "max"),
        weight=("weight", "sum"),
        support=("support", "first"),
        support_n=("support", "nunique"),
    )
    if (
        (spine["region_n"] != 1).any()
        or (spine["owner_n"] != 1).any()
        or (spine["support_n"] != 1).any()
    ):
        raise ValueError(
            "Pool clones of one spine household disagree on region, tenure or support flag."
        )
    if (spine["value_max"] - spine["value_min"]).abs().max() > 1e-6:
        raise ValueError(
            "Pool clones of one spine household carry different stored home values."
        )
    keys = spine.index.to_numpy(dtype=object)
    tie_break = keyed_uniform(stream=stream, keys=[(str(k), _TIE_BREAK) for k in keys])
    identity = keyed_uniform(stream=stream, keys=[(str(k), _IDENTITY) for k in keys])
    support_draw = keyed_uniform(stream=stream, keys=[(str(k), _SUPPORT) for k in keys])
    u = identity.copy()
    ranked = (spine["owner"].to_numpy() & (spine["value"].to_numpy() > 0)).astype(bool)
    is_support = spine["support"].to_numpy().astype(bool)
    # The support copies are ranked beside their root (they tie on value and
    # split the tie-break), so the stratum keeps its mass in the rank: the
    # unselected owners' mid-ranks end where the stratum starts, and the
    # copies' own draws below then spread them inside it. Dropping the copies
    # from the rank would hand their 5/6 of the stratum's mass to the owners
    # below and nearly double the mass above ``support_stratum_lower``.
    rank_rows = ranked
    spine_region = spine["region"].to_numpy().astype(str)
    spine_value = spine["value"].to_numpy(dtype=float)
    spine_weight = spine["weight"].to_numpy(dtype=float)
    regions_ranked: dict[str, int] = {}
    for code in sorted(set(spine_region[rank_rows])):
        rows = np.flatnonzero(rank_rows & (spine_region == code))
        order = rows[np.lexsort((tie_break[rows], spine_value[rows]))]
        weights = spine_weight[order]
        total = float(weights.sum())
        if total <= 0:
            raise ValueError(f"Owner households in region {code} carry no weight.")
        cumulative = np.cumsum(weights)
        u[order] = (cumulative - weights / 2.0) / total
        regions_ranked[code] = int(len(rows))
    u[is_support] = (
        support_stratum_lower + (1.0 - support_stratum_lower) * support_draw[is_support]
    )
    per_row = (
        pd.Series(u, index=spine.index).reindex(frame["key"]).to_numpy(dtype=float)
    )
    if np.any(~np.isfinite(per_row)) or np.any(per_row < 0) or np.any(per_row >= 1):
        raise ValueError("Property uniforms must lie in [0, 1).")
    receipt = {
        "spine_households": int(len(spine)),
        "pool_rows": int(len(frame)),
        "ranked_owner_households_by_region": regions_ranked,
        "support_copies": int(is_support.sum()),
        "support_stratum_lower": support_stratum_lower,
        "rank_basis": "weighted_midpoint_within_frs_region",
    }
    return per_row, receipt


# ----------------------------------------------------------------------------
# Maps


def _england_wales(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes).astype(str)
    return np.char.startswith(codes, "E") | np.char.startswith(codes, "W")


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

    Returns the new values, the new property wealth (the old total moved by
    the value's change, so the wealth identity holds) and a receipt. Rows
    that are not owner-occupied, carry no value or sit outside England and
    Wales are untouched; a constituency without knots fails closed.
    """
    u = np.asarray(u, dtype=float)
    value = np.asarray(value, dtype=float)
    wealth = np.asarray(property_wealth, dtype=float)
    codes = np.asarray(constituency_code).astype(str)
    target = owner_mask(tenure_type) & (value > 0) & _england_wales(codes)
    new_value = value.copy()
    missing = sorted(set(codes[target]) - set(knots.quantiles))
    if missing:
        raise ValueError(f"No value knots for constituency {missing[:5]}.")
    for code in sorted(set(codes[target])):
        rows = np.flatnonzero(target & (codes == code))
        new_value[rows] = np.interp(u[rows], knots.grid, knots.quantiles[code])
    if np.any(~np.isfinite(new_value)) or np.any(new_value < 0):
        raise ValueError("Localised home values must be finite and non-negative.")
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
    mix: BandMix,
) -> tuple[np.ndarray, dict[str, object]]:
    """Draw each England-and-Wales row's band from its LSOA's cumulative shares at ``u``."""
    u = np.asarray(u, dtype=float)
    codes = np.asarray(lsoa_code).astype(str)
    current = np.asarray(band).astype(str)
    target = _england_wales(codes)
    missing = sorted(set(codes[target]) - set(mix.cumulative))
    if missing:
        raise ValueError(f"No VOA band mix for LSOA {missing[:5]}.")
    new_band = current.copy()
    for code in sorted(set(codes[target])):
        rows = np.flatnonzero(target & (codes == code))
        cumulative = mix.cumulative[code]
        index = np.searchsorted(cumulative, u[rows], side="right")
        index = np.minimum(index, len(COUNCIL_TAX_BANDS) - 1)
        new_band[rows] = np.asarray(COUNCIL_TAX_BANDS, dtype=object)[index]
    changed = target & (new_band != current)
    receipt = {
        "rows": int(len(current)),
        "rows_rewritten": int(target.sum()),
        "rows_changed": int(changed.sum()),
        "rows_untouched_outside_england_wales": int((~target).sum()),
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
    return new_band, receipt


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
