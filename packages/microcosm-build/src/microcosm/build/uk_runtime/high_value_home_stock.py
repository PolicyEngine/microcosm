"""Owner-occupied home stock above value thresholds from HM Land Registry data.

PolicyEngine/microcosm#1240, fix 2. Chronicle registers HM Land Registry's
Price Paid Data, the UK House Price Index full file and the ONS National
Statistics Postcode Lookup raw-only (PolicyEngine/chronicle#343): every
aggregate below is Microcosm's computation and none of its values is a
Chronicle fact. The functions here are pure so the builder tool
(``tools/build_uk_high_value_home_stock.py``) and its tests share them.

Method, in order:

1. Each yearly Price Paid file is reduced to category A transfers of
   detached, semi-detached, terraced and flat dwellings with a postcode the
   NSPL places in a 2021 output area. A dwelling is identified by its
   postcode, PAON and SAON (Tax Policy Associates' rule: the data carries no
   title number).
2. Transfers that share a completion date, price and postcode across more
   than one dwelling are bulk transfers; they are never used as a valuation.
3. The stock at the stock date is every dwelling whose first registered sale
   is not a new build completed after that date. Its valuation is its latest
   non-bulk sale.
4. Each valuation is moved to the frame's valuation basis (the calendar-2024
   average of the UK HPI for the dwelling's local authority and property
   type), with a declared fallback chain to the region and the nation.
5. Thresholds stated at April-2026 prices are deflated to the basis with the
   England all-dwellings index.
6. Dwellings are placed in constituencies and local authorities through the
   committed England-and-Wales atomic support (so boundary vintages match the
   frame), and counted above each threshold with a per-dwelling weight that
   carries the owner-occupied share and the stock-coverage correction.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "AreaKnots",
    "DwellingStock",
    "HPIIndex",
    "PRICE_PAID_COLUMN_POSITIONS",
    "RESIDENTIAL_PROPERTY_TYPES",
    "count_above_thresholds",
    "deflate_thresholds",
    "dwelling_stock",
    "latest_valuations",
    "load_hpi",
    "normalise_postcode",
    "read_nspl_postcode_output_areas",
    "read_price_paid_file",
    "uprate_to_basis",
    "value_knots",
]

#: Zero-based positions of the Price Paid fields the method reads (the file
#: has no header row; HM Land Registry's field order is the contract).
PRICE_PAID_COLUMN_POSITIONS: Mapping[str, int] = {
    "price": 1,
    "date": 2,
    "postcode": 3,
    "property_type": 4,
    "old_new": 5,
    "paon": 7,
    "saon": 8,
    "category": 14,
}
RESIDENTIAL_PROPERTY_TYPES: tuple[str, ...] = ("D", "S", "T", "F")
_TYPE_CODE = {code: index for index, code in enumerate(RESIDENTIAL_PROPERTY_TYPES)}
#: Property-type index columns of the UK HPI full file, in ``_TYPE_CODE`` order,
#: followed by the all-dwellings index used as the last fallback.
HPI_INDEX_COLUMNS: tuple[str, ...] = (
    "DetachedIndex",
    "SemiDetachedIndex",
    "TerracedIndex",
    "FlatIndex",
    "Index",
)
_ALL_TYPES = len(RESIDENTIAL_PROPERTY_TYPES)
HPI_FIRST_MONTH = (1995, 1)


def normalise_postcode(values: pd.Series) -> pd.Series:
    """Upper-case a postcode series and drop every internal space."""
    return values.astype("string").str.upper().str.replace(" ", "", regex=False)


def _month_number(year: int, month: int) -> int:
    first_year, first_month = HPI_FIRST_MONTH
    return (year - first_year) * 12 + (month - first_month)


def _yyyymm_to_month_number(yyyymm: np.ndarray) -> np.ndarray:
    year = yyyymm // 100
    month = yyyymm % 100
    first_year, first_month = HPI_FIRST_MONTH
    return (year - first_year) * 12 + (month - first_month)


def read_nspl_postcode_output_areas(stream) -> pd.DataFrame:
    """Read ``pcds`` and ``oa21cd`` from the NSPL data CSV (live and terminated).

    Returns a frame with ``postcode`` (normalised) and ``oa21cd`` columns, one
    row per postcode. Terminated postcodes are kept so old sales still place.
    """
    table = pd.read_csv(
        stream,
        usecols=["pcds", "oa21cd"],
        dtype={"pcds": "string", "oa21cd": "string"},
        engine="c",
    )
    table = table.dropna(subset=["pcds", "oa21cd"])
    out = pd.DataFrame(
        {
            "postcode": normalise_postcode(table["pcds"]),
            "oa21cd": table["oa21cd"].astype("string").str.strip(),
        }
    )
    out = out.drop_duplicates("postcode", keep="first").reset_index(drop=True)
    return out


def read_price_paid_file(
    path: Path | str,
    *,
    postcode_index: pd.Index,
) -> pd.DataFrame:
    """Reduce one yearly Price Paid file to residential category A transfers.

    ``postcode_index`` is the NSPL's normalised postcode index; the result
    carries the row position of each transfer's postcode in that index
    (``-1`` when the NSPL does not know it). Columns: ``key`` (uint64 dwelling
    identity), ``yyyymm`` (int32 completion month), ``price`` (int64),
    ``ptype`` (int8, ``_TYPE_CODE``), ``postcode_row`` (int32), ``newbuild``
    (bool) and ``batch`` (bool, bulk transfer).
    """
    names = list(PRICE_PAID_COLUMN_POSITIONS)
    positions = [PRICE_PAID_COLUMN_POSITIONS[name] for name in names]
    table = pd.read_csv(
        path,
        header=None,
        usecols=positions,
        names=names,
        dtype={name: "string" for name in names},
        engine="pyarrow",
    )
    table = table[
        (table["category"] == "A")
        & table["property_type"].isin(RESIDENTIAL_PROPERTY_TYPES)
        & table["postcode"].notna()
    ]
    if table.empty:
        return _empty_transfers()
    postcode = normalise_postcode(table["postcode"])
    paon = table["paon"].fillna("").str.strip().str.upper()
    saon = table["saon"].fillna("").str.strip().str.upper()
    identity = postcode.fillna("") + "|" + paon + "|" + saon
    key = pd.util.hash_pandas_object(identity, index=False).to_numpy(dtype=np.uint64)
    date = table["date"].str.slice(0, 10)
    yyyymm = date.str.slice(0, 4).astype(int) * 100 + date.str.slice(5, 7).astype(int)
    price = pd.to_numeric(table["price"], errors="raise").astype(np.int64)
    ptype = table["property_type"].map(_TYPE_CODE).astype(np.int8)
    newbuild = (table["old_new"] == "Y").to_numpy()
    postcode_row = postcode_index.get_indexer(postcode.fillna("")).astype(np.int32)
    frame = pd.DataFrame(
        {
            "key": key,
            "yyyymm": yyyymm.to_numpy(dtype=np.int32),
            "price": price.to_numpy(),
            "ptype": ptype.to_numpy(),
            "postcode_row": postcode_row,
            "newbuild": newbuild,
        }
    )
    batch_group = pd.DataFrame(
        {
            "date": date.to_numpy(),
            "price": frame["price"].to_numpy(),
            "pc": postcode.to_numpy(),
        }
    )
    distinct_keys = (
        pd.DataFrame({"key": frame["key"].to_numpy(), **batch_group})
        .groupby(["date", "price", "pc"], sort=False, dropna=False)["key"]
        .transform("nunique")
    )
    frame["batch"] = distinct_keys.to_numpy() > 1
    return frame.reset_index(drop=True)


def _empty_transfers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": np.zeros(0, dtype=np.uint64),
            "yyyymm": np.zeros(0, dtype=np.int32),
            "price": np.zeros(0, dtype=np.int64),
            "ptype": np.zeros(0, dtype=np.int8),
            "postcode_row": np.zeros(0, dtype=np.int32),
            "newbuild": np.zeros(0, dtype=bool),
            "batch": np.zeros(0, dtype=bool),
        }
    )


def latest_valuations(transfers: pd.DataFrame, *, stock_yyyymm: int) -> pd.DataFrame:
    """Collapse transfers to one row per dwelling.

    Columns: ``key``, ``first_yyyymm``, ``first_newbuild``, ``in_stock``
    (the first sale is not a new build after the stock date), ``yyyymm``,
    ``price``, ``ptype`` and ``postcode_row`` of the latest non-bulk sale,
    and ``valued`` (such a sale exists). Rows are ordered by ``key`` so the
    result is deterministic whatever the input order.
    """
    if transfers.empty:
        return pd.DataFrame(
            {
                "key": np.zeros(0, dtype=np.uint64),
                "first_yyyymm": np.zeros(0, dtype=np.int32),
                "first_newbuild": np.zeros(0, dtype=bool),
                "in_stock": np.zeros(0, dtype=bool),
                "yyyymm": np.zeros(0, dtype=np.int32),
                "price": np.zeros(0, dtype=np.int64),
                "ptype": np.zeros(0, dtype=np.int8),
                "postcode_row": np.zeros(0, dtype=np.int32),
                "valued": np.zeros(0, dtype=bool),
            }
        )
    ordered = transfers.sort_values(["key", "yyyymm"], kind="stable")
    first = ordered.drop_duplicates("key", keep="first")[["key", "yyyymm", "newbuild"]]
    first = first.rename(
        columns={"yyyymm": "first_yyyymm", "newbuild": "first_newbuild"}
    )
    latest = ordered[~ordered["batch"]].drop_duplicates("key", keep="last")[
        ["key", "yyyymm", "price", "ptype", "postcode_row"]
    ]
    merged = first.merge(latest, on="key", how="left")
    merged["valued"] = merged["yyyymm"].notna()
    merged["yyyymm"] = merged["yyyymm"].fillna(0).astype(np.int32)
    merged["price"] = merged["price"].fillna(0).astype(np.int64)
    merged["ptype"] = merged["ptype"].fillna(-1).astype(np.int8)
    merged["postcode_row"] = merged["postcode_row"].fillna(-1).astype(np.int32)
    merged["in_stock"] = ~(
        merged["first_newbuild"] & (merged["first_yyyymm"] > stock_yyyymm)
    )
    return merged.sort_values("key", kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class HPIIndex:
    """Monthly UK HPI index levels by area, month and property type.

    ``levels`` is shaped ``(areas, months, 5)``: the four property-type
    indices in ``RESIDENTIAL_PROPERTY_TYPES`` order, then the all-dwellings
    index; ``NaN`` marks a cell the publisher leaves blank. ``month_count``
    months run from :data:`HPI_FIRST_MONTH`.
    """

    area_codes: tuple[str, ...]
    area_names: Mapping[str, str]
    levels: np.ndarray
    last_month: tuple[int, int]

    @property
    def area_index(self) -> Mapping[str, int]:
        return {code: index for index, code in enumerate(self.area_codes)}

    def basis_levels(self, year: int) -> np.ndarray:
        """Calendar-year average level per area and type (NaN when any month is blank)."""
        start = _month_number(year, 1)
        window = self.levels[:, start : start + 12, :]
        if window.shape[1] != 12:
            raise ValueError(f"UK HPI does not cover every month of {year}.")
        return window.mean(axis=1)

    def month_levels(self, year: int, month: int) -> np.ndarray:
        return self.levels[:, _month_number(year, month), :]


def load_hpi(path: Path | str) -> HPIIndex:
    """Read the UK HPI full file into :class:`HPIIndex`."""
    table = pd.read_csv(
        path,
        usecols=["Date", "RegionName", "AreaCode", *HPI_INDEX_COLUMNS],
        dtype={"Date": "string", "RegionName": "string", "AreaCode": "string"},
    )
    parsed = pd.to_datetime(table["Date"], format="%d/%m/%Y")
    month_number = (
        (parsed.dt.year - HPI_FIRST_MONTH[0]) * 12
        + (parsed.dt.month - HPI_FIRST_MONTH[1])
    ).to_numpy()
    keep = month_number >= 0
    table = table[keep].reset_index(drop=True)
    month_number = month_number[keep]
    last = parsed[keep].max()
    area_codes = tuple(sorted(set(table["AreaCode"].astype(str))))
    area_index = {code: index for index, code in enumerate(area_codes)}
    month_count = int(month_number.max()) + 1
    levels = np.full((len(area_codes), month_count, len(HPI_INDEX_COLUMNS)), np.nan)
    rows = table["AreaCode"].astype(str).map(area_index).to_numpy()
    for column_index, column in enumerate(HPI_INDEX_COLUMNS):
        values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=float)
        levels[rows, month_number, column_index] = values
    names = (
        table.drop_duplicates("AreaCode")[["AreaCode", "RegionName"]]
        .astype(str)
        .set_index("AreaCode")["RegionName"]
        .to_dict()
    )
    return HPIIndex(
        area_codes=area_codes,
        area_names=names,
        levels=levels,
        last_month=(int(last.year), int(last.month)),
    )


def uprate_to_basis(
    hpi: HPIIndex,
    *,
    price: np.ndarray,
    yyyymm: np.ndarray,
    ptype: np.ndarray,
    area_chain: Sequence[np.ndarray],
    basis_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Move each sale price to the calendar-``basis_year`` average level.

    ``area_chain`` lists, per dwelling, the HPI area index to try first (its
    local authority), then each fallback (region, nation); ``-1`` skips a
    link. For each dwelling the first link with a finite sale-month and basis
    level for its property type is used; the all-dwellings index of the same
    link is tried before moving to the next link. Returns the basis values
    and, per dwelling, the chain position used (``link * 2 + used_all_types``,
    or ``-1`` when nothing resolved).
    """
    price = np.asarray(price, dtype=float)
    month_number = _yyyymm_to_month_number(np.asarray(yyyymm, dtype=np.int64))
    ptype = np.asarray(ptype, dtype=np.int64)
    basis = hpi.basis_levels(basis_year)
    out = np.full(price.shape, np.nan)
    used = np.full(price.shape, -1, dtype=np.int16)
    # A sale registered after the index's last month takes that month's level.
    valid_month = month_number >= 0
    month_clipped = np.clip(month_number, 0, hpi.levels.shape[1] - 1)
    for link, areas in enumerate(area_chain):
        areas = np.asarray(areas, dtype=np.int64)
        for type_column, flag in ((ptype, 0), (np.full(ptype.shape, _ALL_TYPES), 1)):
            pending = ~np.isfinite(out) & (areas >= 0) & valid_month & (ptype >= 0)
            if not pending.any():
                continue
            area_clipped = np.where(areas >= 0, areas, 0)
            sale_level = hpi.levels[area_clipped, month_clipped, type_column]
            basis_level = basis[area_clipped, type_column]
            resolvable = (
                pending
                & np.isfinite(sale_level)
                & np.isfinite(basis_level)
                & (sale_level > 0)
            )
            out[resolvable] = (
                price[resolvable] * basis_level[resolvable] / sale_level[resolvable]
            )
            used[resolvable] = link * 2 + flag
    return out, used


def deflate_thresholds(
    hpi: HPIIndex,
    *,
    thresholds: Sequence[float],
    nation_area_code: str,
    basis_year: int,
    reference_year: int,
    reference_month: int,
) -> tuple[dict[float, float], float]:
    """Deflate thresholds stated at a reference month to the basis-year average.

    Uses the nation's all-dwellings index. Returns ``{threshold: basis value}``
    and the factor applied.
    """
    area = hpi.area_index[nation_area_code]
    basis = hpi.basis_levels(basis_year)[area, _ALL_TYPES]
    reference = hpi.month_levels(reference_year, reference_month)[area, _ALL_TYPES]
    if not (math.isfinite(basis) and math.isfinite(reference) and reference > 0):
        raise ValueError("UK HPI lacks the nation's basis or reference level.")
    factor = float(basis / reference)
    return {float(t): float(t) * factor for t in thresholds}, factor


@dataclass(frozen=True)
class DwellingStock:
    """Valued dwellings with their areas and per-dwelling weights."""

    table: pd.DataFrame
    receipts: dict[str, object] = field(default_factory=dict)


def dwelling_stock(
    valuations: pd.DataFrame,
    *,
    postcode_oa: pd.Series,
    support: Mapping[str, np.ndarray],
    hpi: HPIIndex,
    hpi_area_for_local_authority: Mapping[str, str],
    region_hpi_code: Mapping[str, str],
    basis_year: int,
) -> DwellingStock:
    """Place and uprate the valued stock.

    ``postcode_oa`` maps NSPL row position to ``oa21cd``; ``support`` carries
    the E&W atomic support arrays (``area``, ``constituency_code``,
    ``local_authority_code``, ``lsoa_code``, ``region_code``);
    ``hpi_area_for_local_authority`` bridges roster local-authority codes to
    HPI area codes (recodes and the Isles of Scilly fallback);
    ``region_hpi_code`` maps support region codes to HPI region codes. Rows
    are the in-stock, valued dwellings whose postcode places in the support.
    """
    rows = valuations[
        valuations["in_stock"]
        & valuations["valued"]
        & (valuations["postcode_row"] >= 0)
    ]
    receipts: dict[str, object] = {
        "dwellings_ever_sold": int(len(valuations)),
        "dwellings_in_stock": int(valuations["in_stock"].sum()),
        "dwellings_not_valued_bulk_only": int(
            (valuations["in_stock"] & ~valuations["valued"]).sum()
        ),
        "dwellings_postcode_unmatched": int(
            (
                valuations["in_stock"]
                & valuations["valued"]
                & (valuations["postcode_row"] < 0)
            ).sum()
        ),
    }
    oa = postcode_oa.to_numpy(dtype=object)[rows["postcode_row"].to_numpy()]
    oa_index = pd.Index(support["area"])
    oa_row = oa_index.get_indexer(pd.Index(oa))
    placed = oa_row >= 0
    receipts["dwellings_output_area_outside_support"] = int((~placed).sum())
    rows = rows[placed]
    oa_row = oa_row[placed]
    local_authority = support["local_authority_code"][oa_row]
    region = support["region_code"][oa_row]
    nation = np.where(
        np.char.startswith(local_authority.astype(str), "W"), "W92000004", "E92000001"
    )
    hpi_index = hpi.area_index
    la_hpi = np.array(
        [
            hpi_index.get(hpi_area_for_local_authority.get(code, code), -1)
            for code in local_authority
        ],
        dtype=np.int64,
    )
    region_hpi = np.array(
        [hpi_index.get(region_hpi_code.get(code, code), -1) for code in region],
        dtype=np.int64,
    )
    nation_hpi = np.array([hpi_index.get(code, -1) for code in nation], dtype=np.int64)
    basis_value, used = uprate_to_basis(
        hpi,
        price=rows["price"].to_numpy(),
        yyyymm=rows["yyyymm"].to_numpy(),
        ptype=rows["ptype"].to_numpy(),
        area_chain=(la_hpi, region_hpi, nation_hpi),
        basis_year=basis_year,
    )
    resolved = np.isfinite(basis_value)
    receipts["dwellings_uprating_unresolved"] = int((~resolved).sum())
    receipts["uprating_chain_used"] = {
        label: int((used == code).sum())
        for code, label in (
            (0, "local_authority_type"),
            (1, "local_authority_all_types"),
            (2, "region_type"),
            (3, "region_all_types"),
            (4, "nation_type"),
            (5, "nation_all_types"),
        )
    }
    table = pd.DataFrame(
        {
            "key": rows["key"].to_numpy()[resolved],
            "constituency_code": support["constituency_code"][oa_row][resolved],
            "local_authority_code": local_authority[resolved],
            "lsoa_code": support["lsoa_code"][oa_row][resolved],
            "region_code": region[resolved],
            "nation_code": nation[resolved],
            "ptype": rows["ptype"].to_numpy()[resolved],
            "sale_yyyymm": rows["yyyymm"].to_numpy()[resolved],
            "price": rows["price"].to_numpy()[resolved],
            "basis_value": basis_value[resolved],
        }
    )
    receipts["dwellings_valued_in_stock"] = int(len(table))
    return DwellingStock(table.reset_index(drop=True), receipts)


def count_above_thresholds(
    stock: pd.DataFrame,
    *,
    thresholds_basis: Mapping[str, float],
    weight: np.ndarray,
    group_columns: Sequence[str],
) -> pd.DataFrame:
    """Sum ``weight`` over dwellings above each threshold, per group column value.

    Returns a long frame ``(level, code, threshold_id, value)`` where
    ``level`` is the group column name.
    """
    frames = []
    value = stock["basis_value"].to_numpy()
    weight = np.asarray(weight, dtype=float)
    for column in group_columns:
        codes = stock[column].to_numpy()
        for threshold_id, threshold in thresholds_basis.items():
            above = value > threshold
            sums = pd.Series(weight[above]).groupby(codes[above], sort=True).sum()
            all_codes = pd.Index(sorted(set(codes)))
            sums = sums.reindex(all_codes, fill_value=0.0)
            frames.append(
                pd.DataFrame(
                    {
                        "level": column,
                        "code": all_codes,
                        "threshold_id": threshold_id,
                        "value": sums.to_numpy(dtype=float),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


#: Quantile grid for the per-area value knots: coarse through the body of the
#: distribution, fine in the tail the surcharge bands sit in.
KNOT_GRID: tuple[float, ...] = tuple(
    sorted(
        {round(p, 3) for p in np.arange(0.0, 0.9001, 0.05)}
        | {round(p, 3) for p in np.arange(0.90, 0.9901, 0.01)}
        | {round(p, 3) for p in np.arange(0.990, 0.9991, 0.001)}
        | {1.0}
    )
)


@dataclass(frozen=True)
class AreaKnots:
    code: str
    count: int
    weight: float
    quantiles: tuple[float, ...]


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, grid: Sequence[float]
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights)
    total = cumulative[-1]
    midpoints = (cumulative - weights / 2.0) / total
    return np.interp(
        np.asarray(grid, dtype=float),
        midpoints,
        values,
        left=values[0],
        right=values[-1],
    )


def value_knots(
    stock: pd.DataFrame,
    *,
    weight: np.ndarray,
    group_column: str,
    cap: float,
    minimum_count: int,
    grid: Sequence[float] = KNOT_GRID,
) -> tuple[list[AreaKnots], list[str]]:
    """Weighted basis-value quantiles per area, winsorised at ``cap``.

    Areas with fewer than ``minimum_count`` dwellings are suppressed and
    returned by code in the second element.
    """
    values = np.minimum(stock["basis_value"].to_numpy(dtype=float), float(cap))
    weight = np.asarray(weight, dtype=float)
    codes = stock[group_column].to_numpy().astype(str)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    boundaries = np.flatnonzero(
        np.r_[True, sorted_codes[1:] != sorted_codes[:-1], True]
    )
    knots: list[AreaKnots] = []
    suppressed: list[str] = []
    for begin, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        code = sorted_codes[begin]
        rows = order[begin:stop]
        count = int(stop - begin)
        if count < minimum_count:
            suppressed.append(str(code))
            continue
        quantiles = _weighted_quantiles(values[rows], weight[rows], grid)
        knots.append(
            AreaKnots(
                code=str(code),
                count=count,
                weight=float(weight[rows].sum()),
                quantiles=tuple(float(round(q, 2)) for q in quantiles),
            )
        )
    return knots, suppressed


def sha256_of_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    """Digest of consumer-fact rows, independent of their order."""
    digest = hashlib.sha256()
    for line in sorted(
        json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        for row in rows
    ):
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def stock_yyyymm_from_date(value: str) -> int:
    parsed = _dt.date.fromisoformat(value)
    return parsed.year * 100 + parsed.month
