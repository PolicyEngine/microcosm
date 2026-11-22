"""UK firm rateable value: an assumed mapping of VOA property bands to firms.

policyengine-uk's ``firm_rateable_value`` is the total rateable value, in
pounds, of the non-domestic properties a firm occupies. It drives
``firm_business_rates``, which policyengine-uk models for England only
(``firm_in_england``; a firm elsewhere pays 0) and treats as one aggregate
property per firm (``firm_n_properties`` only decides small business rate
relief). The experimental UK firm generator draws no property, so this module
imputes the value. Its status is ``assumed``: nothing links a generated firm to
a property, and the result is not calibrated to business rates (NNDR) receipts
or to any other tax total.

Source: the Valuation Office Agency's *Non-domestic rating: stock of
properties, March 2026*, Table SOP2.2: hereditaments and total rateable value
in England by rateable-value band, on the **2023 rating list** as at 31 March
2026. The extracted CSV and its provenance sit in
``tests/fixtures/uk/firm_rateable_value_2024/``
(``tools/extract_uk_firm_rateable_value_tables.py`` re-creates it).

The mapping, every step of it an assumption beyond the source:

- hereditaments are properties, not firms. The module lines the two up by
  rank: firms in England are ordered by headcount, largest first, and the
  hereditaments by rateable value, largest first, and the firm at a given
  weighted rank takes the hereditaments at the same rank. A firm standing for
  ``firm_weight`` firms takes that many hereditaments and carries their mean
  rateable value, so the weighted total over the firms that get a property
  equals the table's rateable value for the hereditaments they cover;
- firms with the same headcount are ordered by a draw from a separate
  fixed-seed generator (``UK_FIRM_RATEABLE_VALUE_SEED``), which depends on
  row order and on nothing else in the build;
- once the hereditaments run out, the remaining (smallest) firms get 0: they
  are taken to trade from a home or from someone else's premises;
- one firm, one hereditament: a multi-site firm's properties are spread over
  several firms rather than summed on one, so the largest firms' values are
  too low and small business rate relief is overstated; empty properties and
  those of households, charities and public bodies outside the generator are
  handed to firms;
- inside a band, rateable value follows ``lower + (upper - lower) * q ** k``
  over the within-band quantile ``q``, with ``k`` set so the band's mean is
  the published one (total over count). The open top band, and any band whose
  published mean does not lie strictly inside its bounds, is flat at its
  mean;
- the all-sector band table is used. The table's sectors (retail, offices,
  industry, other) classify properties, not SIC sections, and the module
  does not map one to the other;
- a firm outside England gets 0. Wales is in the VOA release, but
  policyengine-uk does not model Welsh rates;
- the stock is as at 31 March 2026 and is used unchanged for the 2024-25
  build: the 2023 rating list was in force throughout, and
  ``voa_stock_by_year.csv`` in the fixture gives the all-property stock at
  31 March 2024 and 2025 for comparison.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: The extracted VOA band table's file name in a source directory.
UK_FIRM_RATEABLE_VALUE_FILE = "voa_rv_band_sector.csv"
#: Its columns, in order.
UK_FIRM_RATEABLE_VALUE_SOURCE_COLUMNS = (
    "geography",
    "rv_band",
    "band_lower_gbp",
    "band_upper_gbp",
    "sector",
    "hereditaments",
    "total_rv_gbp_thousand",
)
#: The rows the imputation reads.
UK_FIRM_RATEABLE_VALUE_GEOGRAPHY = "england"
UK_FIRM_RATEABLE_VALUE_SECTOR = "all"
UK_FIRM_RATEABLE_VALUE_TOTAL_BAND = "Total"
UK_FIRM_RATEABLE_VALUE_RATING_LIST = "2023"
UK_FIRM_RATEABLE_VALUE_STATUS = "assumed"
UK_FIRM_RATEABLE_VALUE_METHOD = (
    "Firms in England ranked by headcount (ties by a fixed-seed draw) take "
    "the VOA 2023-list hereditaments at the same weighted rank by rateable "
    "value; firms past the last hereditament, and firms outside England, "
    "get 0. Not calibrated to business rates receipts."
)
#: Seed of the tie-breaking generator, used for nothing else.
UK_FIRM_RATEABLE_VALUE_SEED = 20230401


@dataclass(frozen=True)
class UKFirmRateableValueSource:
    """England's hereditaments by rateable-value band, largest band first.

    ``lower_gbp`` and ``upper_gbp`` are the band bounds in pounds
    (``upper_gbp`` is ``inf`` for the open top band), ``hereditaments`` the
    counts and ``total_rv_gbp`` the bands' rateable value in pounds.
    ``published_*`` are the table's own ``Total`` row. ``sha256`` is the
    CSV's hash.
    """

    bands: tuple[str, ...]
    lower_gbp: tuple[float, ...]
    upper_gbp: tuple[float, ...]
    hereditaments: tuple[float, ...]
    total_rv_gbp: tuple[float, ...]
    published_hereditaments: float
    published_total_rv_gbp: float
    file: str
    sha256: str

    @property
    def n_hereditaments(self) -> float:
        return float(sum(self.hereditaments))

    @property
    def total_gbp(self) -> float:
        return float(sum(self.total_rv_gbp))


def read_uk_firm_rateable_value_source(path: str | Path) -> UKFirmRateableValueSource:
    """Read England's all-sector band rows from a directory or the CSV.

    Raises ``ValueError`` unless the file has exactly
    ``UK_FIRM_RATEABLE_VALUE_SOURCE_COLUMNS``, a ``Total`` row, and bands with
    positive counts and values whose bounds ascend without overlap and whose
    last band alone is open.
    """

    path = Path(path)
    if path.is_dir():
        path = path / UK_FIRM_RATEABLE_VALUE_FILE
    payload = path.read_bytes()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(frame.columns) != UK_FIRM_RATEABLE_VALUE_SOURCE_COLUMNS:
        raise ValueError(
            f"UK firm rateable value source columns {list(frame.columns)} are "
            f"not {list(UK_FIRM_RATEABLE_VALUE_SOURCE_COLUMNS)}."
        )
    rows = frame[
        (frame["geography"] == UK_FIRM_RATEABLE_VALUE_GEOGRAPHY)
        & (frame["sector"] == UK_FIRM_RATEABLE_VALUE_SECTOR)
    ]
    is_total = rows["rv_band"] == UK_FIRM_RATEABLE_VALUE_TOTAL_BAND
    if int(is_total.sum()) != 1:
        raise ValueError("UK firm rateable value source needs one England Total row.")
    total = rows[is_total].iloc[0]
    bands = rows[~is_total]
    if bands.empty:
        raise ValueError("UK firm rateable value source has no England bands.")
    try:
        lower = bands["band_lower_gbp"].astype(np.float64).to_numpy()
        upper = bands["band_upper_gbp"].replace("", "inf").astype(np.float64).to_numpy()
        counts = bands["hereditaments"].astype(np.float64).to_numpy()
        values = bands["total_rv_gbp_thousand"].astype(np.float64).to_numpy() * 1e3
        published_count = float(total["hereditaments"])
        published_value = float(total["total_rv_gbp_thousand"]) * 1e3
    except ValueError:
        raise ValueError("UK firm rateable value source has a non-number.") from None
    if (
        np.any(lower < 0)
        or np.any(upper <= lower)
        or np.any(lower[1:] <= upper[:-1])
        or not np.all(np.isfinite(upper[:-1]))
        or np.isfinite(upper[-1])
    ):
        raise ValueError(
            "UK firm rateable value bands must ascend without overlap and end "
            "with one open band."
        )
    if (
        not np.all(np.isfinite(counts))
        or not np.all(np.isfinite(values))
        or np.any(counts <= 0)
        or np.any(values <= 0)
    ):
        raise ValueError("UK firm rateable value bands need positive numbers.")
    order = slice(None, None, -1)
    return UKFirmRateableValueSource(
        bands=tuple(bands["rv_band"].tolist()[order]),
        lower_gbp=tuple(lower[order].tolist()),
        upper_gbp=tuple(upper[order].tolist()),
        hereditaments=tuple(counts[order].tolist()),
        total_rv_gbp=tuple(values[order].tolist()),
        published_hereditaments=published_count,
        published_total_rv_gbp=published_value,
        file=path.name,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def uk_firm_rateable_value_of_top(
    source: UKFirmRateableValueSource, rank: np.ndarray
) -> np.ndarray:
    """Rateable value (pounds) of England's ``rank`` most valuable hereditaments.

    ``rank`` is a count from the top, not necessarily whole; it is clipped to
    between 0 and ``source.n_hereditaments``. Within a band the value follows
    the module's assumed curve, which returns each band's published total at
    the band's end.
    """

    counts = np.asarray(source.hereditaments, dtype=np.float64)
    totals = np.asarray(source.total_rv_gbp, dtype=np.float64)
    lower = np.asarray(source.lower_gbp, dtype=np.float64)
    upper = np.asarray(source.upper_gbp, dtype=np.float64)
    mean = totals / counts
    curved = np.isfinite(upper) & (mean > lower) & (mean < upper)
    width = np.where(curved, upper - lower, 0.0)
    # Exponent plus one of the within-band curve: (upper - lower) / (mean - lower).
    power = np.where(curved, width / np.where(curved, mean - lower, 1.0), 1.0)
    edges = np.concatenate([[0.0], np.cumsum(counts)])
    before = np.concatenate([[0.0], np.cumsum(totals)])

    rank = np.clip(np.asarray(rank, dtype=np.float64), 0.0, edges[-1])
    band = np.clip(np.searchsorted(edges, rank, side="right") - 1, 0, len(counts) - 1)
    share = np.clip((rank - edges[band]) / counts[band], 0.0, 1.0)
    flat = share * totals[band]
    bent = counts[band] * (
        lower[band] * share
        + width[band] * (1.0 - (1.0 - share) ** power[band]) / power[band]
    )
    return before[band] + np.where(curved[band], bent, flat)


def _headcount(frame: pd.DataFrame) -> np.ndarray:
    for column in ("firm_employees", "employment"):
        if column in frame.columns:
            values = frame[column].to_numpy(dtype=np.float64)
            break
    else:
        raise ValueError("UK firm rateable value needs firm_employees or employment.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("UK firm employees must be non-negative numbers.")
    return values


def _weight(frame: pd.DataFrame) -> np.ndarray:
    if "firm_weight" not in frame.columns:
        return np.ones(len(frame), dtype=np.float64)
    weight = frame["firm_weight"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(weight)) or np.any(weight < 0):
        raise ValueError("UK firm firm_weight must be non-negative numbers.")
    return weight


def _in_england(frame: pd.DataFrame) -> np.ndarray:
    if "firm_in_england" not in frame.columns:
        raise ValueError("UK firm rateable value needs firm_in_england.")
    return frame["firm_in_england"].to_numpy(dtype=bool)


def assign_uk_firm_rateable_value(
    table_or_firms: pd.DataFrame,
    source: UKFirmRateableValueSource,
    *,
    seed: int = UK_FIRM_RATEABLE_VALUE_SEED,
) -> np.ndarray:
    """Assumed rateable value in pounds per row, in row order.

    ``table_or_firms`` needs ``firm_in_england`` and ``firm_employees`` (or
    the generator's ``employment``); ``firm_weight`` is 1 where absent. See
    the module docstring for the mapping. Rows outside England, rows with no
    weight and rows ranked past the last hereditament get 0.
    """

    headcount = _headcount(table_or_firms)
    weight = _weight(table_or_firms)
    england = np.flatnonzero(_in_england(table_or_firms) & (weight > 0))
    out = np.zeros(len(table_or_firms), dtype=np.float64)
    if england.size == 0:
        return out
    ties = np.random.default_rng(seed).random(england.size)
    order = england[np.lexsort((ties, -headcount[england]))]
    end = np.cumsum(weight[order])
    start = end - weight[order]
    covered = uk_firm_rateable_value_of_top(
        source, end
    ) - uk_firm_rateable_value_of_top(source, start)
    out[order] = np.maximum(covered, 0.0) / weight[order]
    return out


def uk_firm_rateable_value_summary(
    table_or_firms: pd.DataFrame,
    source: UKFirmRateableValueSource,
    *,
    seed: int = UK_FIRM_RATEABLE_VALUE_SEED,
) -> dict[str, float]:
    """Weighted totals of the assumed rateable value, for disclosure.

    ``weighted_firms_in_england`` and ``weighted_firms_with_property`` are
    sums of ``firm_weight``; ``weighted_rateable_value_gbp`` is the weighted
    total, and ``source_*`` are the band table's sums it can at most reach.
    """

    weight = _weight(table_or_firms)
    value = assign_uk_firm_rateable_value(table_or_firms, source, seed=seed)
    return {
        "weighted_firms_in_england": float(weight[_in_england(table_or_firms)].sum()),
        "weighted_firms_with_property": float(weight[value > 0].sum()),
        "weighted_rateable_value_gbp": float((weight * value).sum()),
        "source_hereditaments": source.n_hereditaments,
        "source_rateable_value_gbp": source.total_gbp,
    }


__all__ = [
    "UK_FIRM_RATEABLE_VALUE_FILE",
    "UK_FIRM_RATEABLE_VALUE_GEOGRAPHY",
    "UK_FIRM_RATEABLE_VALUE_METHOD",
    "UK_FIRM_RATEABLE_VALUE_RATING_LIST",
    "UK_FIRM_RATEABLE_VALUE_SECTOR",
    "UK_FIRM_RATEABLE_VALUE_SEED",
    "UK_FIRM_RATEABLE_VALUE_SOURCE_COLUMNS",
    "UK_FIRM_RATEABLE_VALUE_STATUS",
    "UK_FIRM_RATEABLE_VALUE_TOTAL_BAND",
    "UKFirmRateableValueSource",
    "assign_uk_firm_rateable_value",
    "read_uk_firm_rateable_value_source",
    "uk_firm_rateable_value_of_top",
    "uk_firm_rateable_value_summary",
]
