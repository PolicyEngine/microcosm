"""The release tables, the records behind them and their disclosure statuses (microcosm#1326 A6).

Rows repeat records: CGT clones, support copies and, in a dense file, area
clones all carry one source's values. So every count here is of distinct
sources, every interval resamples sources with their clones together, and the
disclosure rules read sources, never rows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from tools.uk_scotland_property_income.bands import (
    assign_bands,
    band_labels,
    coarse_index_of_fine,
)

#: The stored incomes that identify an SPI-channel record. Its tape record isn't
#: carried (microcosm#1287), so records with the same income vector count once.
SOURCE_INCOME_COLUMNS = (
    "property_income",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "savings_interest_income",
    "dividend_income",
)

MIN_SOURCES = 10
DOMINANCE_SHARE = 0.5

PUBLISHED = "published"
FEWER_THAN_FLOOR = "withheld: fewer than 10 records"
DOMINATED = "withheld: one record dominates"
SECONDARY = "withheld: protects another band"


def _text(values: pd.Series) -> pd.Series:
    return values.map(
        lambda value: value.decode() if isinstance(value, bytes) else str(value)
    )


def source_keys(person_table: pd.DataFrame) -> np.ndarray:
    """One key per independent record.

    FRS persons are keyed by ``person_source_id``. SPI-channel persons carry
    their FRS host's id, not their tape record's, so they are keyed by their
    stored income vector. Read it from the released file, before any
    back-cast.
    """

    channel = _text(person_table["person_support_channel"]).to_numpy()
    frs = "frs:" + person_table["person_source_id"].astype(np.int64).astype(str)
    incomes = (
        person_table[list(SOURCE_INCOME_COLUMNS)].astype(float).round(2).astype(str)
    )
    spi = "spi:" + incomes.agg("|".join, axis=1)
    return np.where(channel == "frs", frs.to_numpy(), spi.to_numpy())


def audit(
    cells: pd.DataFrame, edges: Sequence[int], amount_columns: Sequence[str]
) -> pd.DataFrame:
    """Per band: rows, sources, effective sample size by source, weighted totals
    and the largest source's share of the weight and of each amount.

    ``cells`` has one row per selected person, with ``w``, ``nsnd``, ``source``
    and the amount columns.
    """

    band = assign_bands(cells["nsnd"], edges)
    rows = []
    for index, label in enumerate(band_labels(edges)):
        members = cells[band == index]
        weighted = members[["source"]].assign(
            w=members["w"],
            **{name: members["w"] * members[name] for name in amount_columns},
        )
        by_source = weighted.groupby("source").sum()
        weight = by_source["w"]
        row = {
            "band": label,
            "rows": len(members),
            "sources": len(by_source),
            "source_ess": float(weight.sum() ** 2 / np.square(weight).sum())
            if len(weight)
            else 0.0,
            "count": float(weight.sum()),
            "top_source_weight_share": float(weight.max() / weight.sum())
            if len(weight)
            else 0.0,
        }
        for name in amount_columns:
            amount = by_source[name]
            row[name] = float(amount.sum())
            row[f"top_source_{name}_share"] = (
                float(amount.max() / amount.sum())
                if len(amount) and amount.sum() > 0
                else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _multinomial_draws(n_sources: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        n_sources, np.full(n_sources, 1.0 / n_sources), size=replicates
    )


def bootstrap(
    cells: pd.DataFrame,
    edges: Sequence[int],
    amount_columns: Sequence[str],
    *,
    replicates: int = 1_000,
    seed: int = 0,
    level: float = 0.9,
) -> pd.DataFrame:
    """Percentile intervals and coefficients of variation per band.

    Sources are resampled with replacement across all selected persons, so
    band membership varies between replicates, and a source's clones move
    together. Weights stay fixed, so calibration error is left out and the
    intervals are too narrow.
    """

    codes, sources = pd.factorize(cells["source"])
    draws = _multinomial_draws(len(sources), replicates, seed)
    band = assign_bands(cells["nsnd"], edges)
    tail = (1.0 - level) / 2.0
    weights = cells["w"].to_numpy(dtype=float)
    rows = []
    for index, label in enumerate(band_labels(edges)):
        members = band == index
        multiplier = draws[:, codes[members]]
        row = {"band": label}
        series = {"count": weights[members]}
        for name in amount_columns:
            series[name] = weights[members] * cells.loc[members, name].to_numpy(
                dtype=float
            )
        for name, values in series.items():
            replicate_totals = multiplier @ values
            estimate = values.sum()
            row[f"{name}_low"], row[f"{name}_high"] = np.quantile(
                replicate_totals, [tail, 1 - tail]
            )
            row[f"{name}_cv"] = (
                float(replicate_totals.std() / estimate) if estimate else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_total(
    values: object,
    sources: object,
    *,
    replicates: int = 1_000,
    seed: int = 0,
    level: float = 0.9,
) -> dict[str, float]:
    """The same source bootstrap for one weighted total (``values`` already weighted)."""

    values = np.asarray(values, dtype=float)
    codes, unique = pd.factorize(pd.Series(sources))
    draws = _multinomial_draws(len(unique), replicates, seed)
    by_source = np.bincount(codes, weights=values, minlength=len(unique))
    totals = draws @ by_source
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(totals, [tail, 1 - tail])
    estimate = float(values.sum())
    return {
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "cv": float(totals.std() / abs(estimate)) if estimate else float("nan"),
        "sources": int(len(unique)),
        "top_source_share": float(np.abs(by_source).max() / np.abs(by_source).sum())
        if len(unique)
        else 0.0,
    }


def _primary(
    table: pd.DataFrame,
    amount_columns: Sequence[str],
    min_sources: int,
    dominance: float,
) -> list[str]:
    statuses = []
    for _, row in table.iterrows():
        shares = [row["top_source_weight_share"]] + [
            row[f"top_source_{name}_share"] for name in amount_columns
        ]
        if row["sources"] < min_sources:
            statuses.append(FEWER_THAN_FLOOR)
        elif max(shares) > dominance:
            statuses.append(DOMINATED)
        else:
            statuses.append(PUBLISHED)
    return statuses


def _passes(
    cells: pd.DataFrame,
    mask: np.ndarray,
    amount_columns: Sequence[str],
    min_sources: int,
    dominance: float,
) -> bool:
    """Whether the persons under ``mask``, taken as one cell, pass the primary rules."""

    members = cells[mask]
    weighted = members[["source"]].assign(
        w=members["w"],
        **{name: members["w"] * members[name] for name in amount_columns},
    )
    by_source = weighted.groupby("source").sum()
    if len(by_source) < min_sources:
        return False
    for column in ("w", *amount_columns):
        total = by_source[column].sum()
        if total > 0 and by_source[column].max() / total > dominance:
            return False
    return True


def disclosure_statuses(
    cells: pd.DataFrame,
    fine_edges: Sequence[int],
    coarse_edges: Sequence[int],
    amount_columns: Sequence[str],
    *,
    total_published: bool = True,
    min_sources: int = MIN_SOURCES,
    dominance: float = DOMINANCE_SHARE,
) -> tuple[list[str], list[str]]:
    """Statuses for Release 1 (the ``fine`` bands) and Release 2 (the ``coarse``
    bins) together, read as one hierarchy: total, bins, bands.

    Primary: fewer than ``min_sources`` sources, or one source above
    ``dominance`` of the cell's weight or of any amount.

    Secondary: whatever a reader can rebuild by subtraction must itself pass
    the primary rules, so cells are withheld, fewest sources first, until:

    - the withheld bands inside each published bin pass, taken together;
    - no withheld bin can be rebuilt from its bands;
    - with ``total_published``, the withheld bins pass together, and so do the
      withheld bands inside them. The 1p costing is about one per cent of the
      total property slice, so it gives the total away, and it is published.
    """

    fine = audit(cells, fine_edges, amount_columns)
    coarse = audit(cells, coarse_edges, amount_columns)
    fine_status = _primary(fine, amount_columns, min_sources, dominance)
    coarse_status = _primary(coarse, amount_columns, min_sources, dominance)
    band = assign_bands(cells["nsnd"], fine_edges)
    holder = coarse_index_of_fine(fine_edges, coarse_edges)

    def passes(bands: Sequence[int]) -> bool:
        mask = np.isin(band, list(bands))
        return _passes(cells, mask, amount_columns, min_sources, dominance)

    def fewest(candidates: Sequence[int], table: pd.DataFrame) -> int:
        return min(
            candidates, key=lambda i: (table.loc[i, "sources"], table.loc[i, "count"])
        )

    def bands_in(bins: Sequence[int]) -> list[int]:
        return [i for i in range(len(fine_edges)) if holder[i] in bins]

    def withhold_more() -> bool:
        withheld_bins = [j for j, s in enumerate(coarse_status) if s != PUBLISHED]
        published_bins = [j for j, s in enumerate(coarse_status) if s == PUBLISHED]
        if (
            total_published
            and withheld_bins
            and published_bins
            and not passes(bands_in(withheld_bins))
        ):
            coarse_status[fewest(published_bins, coarse)] = SECONDARY
            return True
        for j in withheld_bins:
            members = bands_in([j])
            if all(fine_status[i] == PUBLISHED for i in members):
                fine_status[fewest(members, fine)] = SECONDARY
                return True
        groups = [bands_in([j]) for j in published_bins]
        if total_published and withheld_bins:
            groups.append(bands_in(withheld_bins))
        for members in groups:
            withheld = [i for i in members if fine_status[i] != PUBLISHED]
            published = [i for i in members if fine_status[i] == PUBLISHED]
            if withheld and published and not passes(withheld):
                fine_status[fewest(published, fine)] = SECONDARY
                return True
        return False

    while withhold_more():
        pass
    return fine_status, coarse_status


def _round(value: float, step: float) -> float:
    return float(np.round(value / step) * step)


def render(
    table: pd.DataFrame,
    intervals: pd.DataFrame,
    statuses: Sequence[str],
    amount_columns: Sequence[str],
    *,
    mode: str,
    count_step: float = 100.0,
    amount_step: float = 100_000.0,
    min_sources: int = MIN_SOURCES,
) -> pd.DataFrame:
    """A release table.

    ``restricted`` shows every band in full, with exact record counts and the
    status as a note; it is for recipients covered by the licence. ``public``
    blanks withheld bands and shows record counts under the floor as ``<10``.
    """

    if mode not in {"restricted", "public"}:
        raise ValueError(f"Unknown mode {mode!r}.")
    rows = []
    for index, row in table.iterrows():
        interval = intervals.loc[index]
        status = statuses[index]
        shown = mode == "restricted" or status == PUBLISHED
        out = {"band": row["band"]}
        if mode == "public" and row["sources"] < min_sources:
            out["distinct_records"] = f"<{min_sources}"
        else:
            out["distinct_records"] = str(int(row["sources"]))
        out["effective_records"] = round(float(row["source_ess"]), 1) if shown else None
        for name, step in [
            ("count", count_step),
            *[(n, amount_step) for n in amount_columns],
        ]:
            for suffix, value in (
                ("", row[name]),
                ("_low", interval[f"{name}_low"]),
                ("_high", interval[f"{name}_high"]),
            ):
                out[f"{name}{suffix}"] = _round(value, step) if shown else None
            out[f"{name}_cv"] = (
                round(float(interval[f"{name}_cv"]), 2) if shown else None
            )
        out["status"] = status
        rows.append(out)
    return pd.DataFrame(rows)
