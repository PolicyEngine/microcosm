"""Loaders of the UK stock cell table and the land under homes (microcosm#1524).

``tools/build_uk_stock_cells.py`` writes five resources from public data; each
loader here reads one, refuses a malformed one, and returns arrays a consumer
can use without repeating the checks:

- :func:`load_uk_stock_cells_value`: cells of MSOA × council tax band with
  VOA's count, the private and social counts, homes sold, the owner share,
  the mean value and 13 knots, and each cell's split across constituencies;
- :func:`load_uk_stock_cells_structure`: per cell, floor area, retained
  share, unit cost at κ = 1 and the central case's land per home;
- :func:`load_uk_la_land_shares`: per authority and variant, land and stock
  value, with λ = land ÷ value derived here;
- :func:`load_uk_constituency_land_totals`: the same by constituency;
- :func:`load_uk_stock_cells_provenance`: inputs, the level and the checks.

The structure block, the rollups and the provenance never feed the solve
(``solve_scope.UK_POST_SOLVE_RESOURCES``); the value block does, since the
frame's home values will be read from it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources as importlib_resources

import numpy as np

from microcosm.build.derived_resource import payload_sha256
from microcosm.build.weighted_quantiles import check_knots

__all__ = [
    "AuthorityLand",
    "ConstituencyLand",
    "STOCK_CELL_RESOURCES",
    "StockCellStructure",
    "StockCellValues",
    "load_uk_constituency_land_totals",
    "load_uk_la_land_shares",
    "load_uk_stock_cells_provenance",
    "load_uk_stock_cells_structure",
    "load_uk_stock_cells_value",
    "resource_digest",
]

STOCK_CELL_RESOURCES: Mapping[str, str] = {
    "value": "stock_cells_value.json",
    "structure": "stock_cells_structure.json",
    "authorities": "la_land_shares.json",
    "constituencies": "constituency_land_totals.json",
    "provenance": "stock_cells.provenance.json",
}
_KINDS: Mapping[str, str] = {
    "value": "uk_stock_cells_value",
    "structure": "uk_stock_cells_structure",
    "authorities": "uk_la_land_shares",
    "constituencies": "uk_constituency_land_totals",
    "provenance": "uk_stock_cells_provenance",
}
_VALUE_COLUMNS = (
    "msoa",
    "band",
    "voa_count",
    "private_count",
    "social_count",
    "homes_sold",
    "owner_share_of_private",
    "mean_value",
    "means_level",
    "knots_level",
    "knots_exact_mean",
)
_STRUCTURE_COLUMNS = (
    "floor_area_mean",
    "retained_share_mean",
    "unit_cost_mean",
    "land_per_home_central",
)
_STATUSES = frozenset({"pass", "fail", "not_evaluated", "measured"})


def _read(name: str, payload: bytes | None) -> dict:
    if payload is None:
        payload = (
            importlib_resources.files("microcosm.build.uk")
            .joinpath(STOCK_CELL_RESOURCES[name])
            .read_bytes()
        )
    data = json.loads(payload.decode("utf-8"))
    if data.get("kind") != _KINDS[name] or data.get("schema_version") != 1:
        raise ValueError(
            f"{STOCK_CELL_RESOURCES[name]} is not a schema-1 {_KINDS[name]}."
        )
    return data


def resource_digest(data: Mapping) -> str:
    """The digest the other stock cell resources record for one: its content
    without the build date."""
    return payload_sha256(
        {key: value for key, value in data.items() if key != "built_on"}
    )


def _array(values, what: str) -> np.ndarray:
    out = np.array([np.nan if v is None else v for v in values], dtype=float)
    if np.any(np.isnan(out)):
        raise ValueError(f"{what} has a gap.")
    return out


@dataclass(frozen=True)
class StockCellValues:
    """The value block: one row per cell with homes, aligned arrays."""

    msoa: np.ndarray
    band: np.ndarray
    voa_count: np.ndarray
    private_count: np.ndarray
    social_count: np.ndarray
    homes_sold: np.ndarray
    owner_share: np.ndarray
    mean_value: np.ndarray
    knots: np.ndarray
    knot_grid: np.ndarray
    knot_cap: float
    seat_rows: np.ndarray
    seat_codes: np.ndarray
    seat_shares: np.ndarray
    social: Mapping[str, tuple[float, float]]
    msoa_authority: Mapping[str, str]
    msoa_region: Mapping[str, str]
    inputs_sha256: str
    digest: str = field(repr=False, default="")

    def authority_of_rows(self) -> np.ndarray:
        """Each cell's local authority."""
        return np.array([self.msoa_authority[code] for code in self.msoa])


def load_uk_stock_cells_value(payload: bytes | None = None) -> StockCellValues:
    data = _read("value", payload)
    if tuple(data.get("columns") or ()) != _VALUE_COLUMNS:
        raise ValueError("The value block's columns are not the declared ones.")
    rows = data["rows"]
    knots = np.array(data["knots"], dtype=float)
    if len(rows) != len(knots) or not rows:
        raise ValueError("The value block's rows and knots do not align.")
    grid = np.array(data["knot_grid"], dtype=float)
    cap = float(data["knot_cap"])
    for row_knots in knots:
        check_knots(grid, row_knots, cap=cap)
    bands = list(data["bands"])
    msoa = np.array([row[0] for row in rows])
    band = np.array([row[1] for row in rows])
    if not set(band) <= set(bands):
        raise ValueError("A cell has a band the block does not declare.")
    if len(set(zip(msoa.tolist(), band.tolist(), strict=True))) != len(rows):
        raise ValueError("The value block repeats a cell.")
    column = {name: i for i, name in enumerate(_VALUE_COLUMNS)}
    voa = _array([row[column["voa_count"]] for row in rows], "voa_count")
    private = _array([row[column["private_count"]] for row in rows], "private_count")
    social = _array([row[column["social_count"]] for row in rows], "social_count")
    if np.any(private < 0) or np.any(social < 0):
        raise ValueError("A cell has a negative count.")
    if not np.allclose(private + social, voa, rtol=0, atol=1e-3 + 1e-6 * voa):
        raise ValueError("A cell's private and social homes are not VOA's count.")
    owner = _array(
        [row[column["owner_share_of_private"]] for row in rows], "owner share"
    )
    if np.any(owner < 0) or np.any(owner > 1):
        raise ValueError("An owner share lies outside [0, 1].")
    mean_value = _array([row[column["mean_value"]] for row in rows], "mean_value")
    shares = data["constituency_shares"]
    if tuple(shares["columns"]) != ("row", "constituency", "share"):
        raise ValueError("The constituency shares' columns are not the declared ones.")
    seat_rows = np.array([r[0] for r in shares["rows"]], dtype=np.int64)
    seat_codes = np.array([r[1] for r in shares["rows"]])
    seat_shares = _array([r[2] for r in shares["rows"]], "a constituency share")
    if seat_rows.min() < 0 or seat_rows.max() >= len(rows):
        raise ValueError("A constituency share points outside the cells.")
    totals = np.bincount(seat_rows, weights=seat_shares, minlength=len(rows))
    if not np.allclose(totals, 1.0, atol=1e-5):
        raise ValueError("A cell's constituency shares do not add up to 1.")
    social_rows = {
        row[0]: (float(row[1]), float(row[2])) for row in data["social"]["rows"]
    }
    msoas = data["msoas"]
    if tuple(msoas["columns"]) != ("msoa", "local_authority", "region"):
        raise ValueError("The MSOA table's columns are not the declared ones.")
    authority = {row[0]: row[1] for row in msoas["rows"]}
    if len(authority) != len(msoas["rows"]) or not set(msoa) <= set(authority):
        raise ValueError(
            "A cell's MSOA is missing from, or repeated in, the MSOA table."
        )
    if set(authority.values()) != set(social_rows):
        raise ValueError(
            "The social homes' authorities are not the MSOAs' authorities."
        )
    return StockCellValues(
        msoa=msoa,
        band=band,
        voa_count=voa,
        private_count=private,
        social_count=social,
        homes_sold=np.array(
            [row[column["homes_sold"]] for row in rows], dtype=np.int64
        ),
        owner_share=owner,
        mean_value=mean_value,
        knots=knots,
        knot_grid=grid,
        knot_cap=cap,
        seat_rows=seat_rows,
        seat_codes=seat_codes,
        seat_shares=seat_shares,
        social=social_rows,
        msoa_authority=authority,
        msoa_region={row[0]: row[2] for row in msoas["rows"]},
        inputs_sha256=str(data.get("inputs_sha256") or ""),
        digest=resource_digest(data),
    )


@dataclass(frozen=True)
class StockCellStructure:
    """The structure block, aligned to the value block's rows."""

    floor_area: np.ndarray
    retained_share: np.ndarray
    unit_cost: np.ndarray
    land_per_home: np.ndarray
    social: Mapping[str, tuple[float, float, float, float]]
    kappa: float
    level_status: str
    value_block_sha256: str
    digest: str = field(repr=False, default="")


def load_uk_stock_cells_structure(
    payload: bytes | None = None, *, value: StockCellValues | None = None
) -> StockCellStructure:
    """The structure block; with ``value``, checked to be built on that block."""
    data = _read("structure", payload)
    if tuple(data.get("columns") or ()) != _STRUCTURE_COLUMNS:
        raise ValueError("The structure block's columns are not the declared ones.")
    rows = data["rows"]
    floor = _array([row[0] for row in rows], "floor area")
    retained = _array([row[1] for row in rows], "the retained share")
    unit = _array([row[2] for row in rows], "unit cost")
    land = _array([row[3] for row in rows], "land per home")
    if np.any(retained < 0) or np.any(retained > 1):
        raise ValueError("A retained share lies outside [0, 1].")
    if np.any(floor <= 0) or np.any(unit < 0) or np.any(land < 0):
        raise ValueError(
            "A cell has a nonpositive floor area or a negative cost or land."
        )
    if value is not None:
        if len(rows) != len(value.msoa):
            raise ValueError("The structure block does not align with the value block.")
        if data["value_block_sha256"] != value.digest:
            raise ValueError("The structure block was built on another value block.")
        if np.any(land > value.mean_value + 0.01):
            raise ValueError("A cell's land per home exceeds its mean value.")
    level = data.get("level") or {}
    kappa = float(level.get("kappa"))
    if not (math.isfinite(kappa) and kappa > 0):
        raise ValueError("The structure block's cost level is not positive.")
    social = {
        row[0]: (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        for row in data["social"]["rows"]
    }
    return StockCellStructure(
        floor_area=floor,
        retained_share=retained,
        unit_cost=unit,
        land_per_home=land,
        social=social,
        kappa=kappa,
        level_status=str(level.get("status") or ""),
        value_block_sha256=str(data["value_block_sha256"]),
        digest=resource_digest(data),
    )


@dataclass(frozen=True)
class AuthorityLand:
    """Land and stock value per authority and variant, with λ derived."""

    codes: tuple[str, ...]
    names: tuple[str, ...]
    regions: tuple[str, ...]
    dwellings: np.ndarray
    land: Mapping[str, np.ndarray]
    value: Mapping[str, np.ndarray]
    land_share: Mapping[str, np.ndarray]
    kappa: Mapping[str, float]
    mhclg_land: Mapping[str, np.ndarray]
    flags: Mapping[str, tuple[str, ...]]
    digests: Mapping[str, str]


def _variant_arrays(variants: Mapping, length: int, what: str):
    land, value = {}, {}
    central = variants.get("central")
    if central is None or "value" not in central:
        raise ValueError(f"{what} has no central land and value.")
    for name, entry in variants.items():
        land[name] = _array(entry["land"], f"{what} land, {name}")
        value[name] = _array(
            entry.get("value", central["value"]), f"{what} value, {name}"
        )
        if len(land[name]) != length or len(value[name]) != length:
            raise ValueError(f"{what}'s {name} arrays do not align with its rows.")
        if np.any(land[name] < 0) or np.any(land[name] > value[name] + 1.0):
            raise ValueError(f"{what}'s {name} land lies outside [0, value].")
    return land, value


def load_uk_la_land_shares(payload: bytes | None = None) -> AuthorityLand:
    data = _read("authorities", payload)
    rows = data["rows"]
    codes = tuple(str(row[0]) for row in rows)
    if len(set(codes)) != len(codes):
        raise ValueError("An authority appears twice.")
    land, value = _variant_arrays(data["variants"], len(rows), "la_land_shares.json")
    with np.errstate(divide="ignore", invalid="ignore"):
        land_share = {
            name: np.where(value[name] > 0, land[name] / value[name], 0.0)
            for name in land
        }
    mhclg = {}
    for name, entry in (data.get("mhclg_pattern") or {}).items():
        values = np.array(
            [np.nan if v is None else v for v in entry["land"]], dtype=float
        )
        if len(values) != len(rows):
            raise ValueError(f"MHCLG's {name} pattern does not align with the rows.")
        mhclg[name] = values
    return AuthorityLand(
        codes=codes,
        names=tuple(str(row[1]) for row in rows),
        regions=tuple(str(row[2]) for row in rows),
        dwellings=_array([row[3] for row in rows], "dwellings"),
        land=land,
        value=value,
        land_share=land_share,
        kappa={name: float(entry["kappa"]) for name, entry in data["variants"].items()},
        mhclg_land=mhclg,
        flags={str(row[0]): tuple(row[5]) for row in rows},
        digests=dict(data.get("digests") or {}),
    )


@dataclass(frozen=True)
class ConstituencyLand:
    codes: tuple[str, ...]
    land: Mapping[str, np.ndarray]
    value: Mapping[str, np.ndarray]
    digests: Mapping[str, str]


def load_uk_constituency_land_totals(payload: bytes | None = None) -> ConstituencyLand:
    data = _read("constituencies", payload)
    codes = tuple(str(code) for code in data["constituencies"])
    if len(set(codes)) != len(codes):
        raise ValueError("A constituency appears twice.")
    land, value = _variant_arrays(
        data["variants"], len(codes), "constituency_land_totals.json"
    )
    return ConstituencyLand(codes, land, value, dict(data.get("digests") or {}))


def load_uk_stock_cells_provenance(payload: bytes | None = None) -> dict:
    """The provenance, with every check and measurement carrying a status."""
    data = _read("provenance", payload)
    for entry in [*data.get("decision_rule", ()), *data.get("measurements", ())]:
        if entry.get("status") not in _STATUSES:
            raise ValueError(f"Check {entry.get('id')} has no valid status.")
    return data
