"""The release tables, the records behind them and their disclosure statuses (microcosm#1326 A6).

Rows repeat records: CGT clones, support copies and, in a dense file, area
clones all carry one source's values. So every count here is of distinct
records, the disclosure rules read records, never rows, and the intervals
resample whole clusters: an FRS household, whose members often share a let
property, or a tape record.
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

#: The stored incomes that identify a tape record. Its id isn't carried
#: (microcosm#1287), so tape rows with the same income vector count once.
SOURCE_INCOME_COLUMNS = (
    "property_income",
    "employment_income",
    "self_employment_income",
    "private_pension_income",
    "savings_interest_income",
    "dividend_income",
)
#: The income stage's flag for rows whose incomes come from the tape
#: (``uk_runtime/spi_income.py``). Every other row, in either channel, keeps
#: its FRS person's values (``uk_runtime/property_components.py``).
TAPE_RECIPIENT_COLUMN = "person_is_spi_income_recipient"

MIN_SOURCES = 10
DOMINANCE_SHARE = 0.5
TOP_TWO_SHARE = 0.85

PUBLISHED = "published"
FEWER_THAN_FLOOR = "withheld: fewer than 10 records"
DOMINATED = "withheld: one or two records dominate"
SECONDARY = "withheld: protects another band"
WITHHELD_RECORDS = "withheld"


def as_text(values: pd.Series) -> pd.Series:
    """Strings, whether the store kept them as bytes or not."""

    return values.map(
        lambda value: value.decode() if isinstance(value, bytes) else str(value)
    )


def _flag(values: pd.Series) -> np.ndarray:
    if values.dtype == object:
        return as_text(values).isin({"True", "true", "1"}).to_numpy()
    return values.to_numpy(dtype=bool)


def _keeps_frs_values(person_table: pd.DataFrame) -> np.ndarray:
    if TAPE_RECIPIENT_COLUMN not in person_table:
        raise ValueError(f"The person table has no {TAPE_RECIPIENT_COLUMN!r}.")
    channel = as_text(person_table["person_support_channel"]).to_numpy()
    return (channel == "frs") | ~_flag(person_table[TAPE_RECIPIENT_COLUMN])


def _tape_keys(person_table: pd.DataFrame) -> np.ndarray:
    incomes = (
        person_table[list(SOURCE_INCOME_COLUMNS)].astype(float).round(2).astype(str)
    )
    return ("spi:" + incomes.agg("|".join, axis=1)).to_numpy()


def source_keys(person_table: pd.DataFrame) -> np.ndarray:
    """One key per independent record.

    A row that keeps FRS values is keyed by ``person_source_id``. That
    includes the SPI channel's dependants and twin copies with no taxable
    income, which carry their FRS person's values. A row with tape incomes
    carries its FRS host's id, not its tape record's, so it is keyed by its
    stored income vector. Read the table from the released file, before any
    back-cast.
    """

    frs = "frs:" + person_table["person_source_id"].astype(np.int64).astype(str)
    return np.where(
        _keeps_frs_values(person_table), frs.to_numpy(), _tape_keys(person_table)
    )


def cluster_keys(
    person_table: pd.DataFrame, source_household_keys: object
) -> np.ndarray:
    """The bootstrap's resampling unit for each person: the FRS household for
    rows with FRS values, so partners who share a let property move together,
    and the tape record for rows with tape incomes."""

    households = "frs-hh:" + pd.Series(source_household_keys).astype(str)
    return np.where(
        _keeps_frs_values(person_table), households.to_numpy(), _tape_keys(person_table)
    )


def _top_shares(values: pd.Series) -> tuple[float, float]:
    total = values.sum()
    if len(values) == 0 or total <= 0:
        return 0.0, 0.0
    largest = np.sort(values.to_numpy(dtype=float))[::-1]
    return float(largest[0] / total), float(largest[:2].sum() / total)


def _by_source(members: pd.DataFrame, amount_columns: Sequence[str]) -> pd.DataFrame:
    weighted = members[["source"]].assign(
        w=members["w"],
        **{name: members["w"] * members[name] for name in amount_columns},
    )
    return weighted.groupby("source").sum()


def audit(
    cells: pd.DataFrame, edges: Sequence[int], amount_columns: Sequence[str]
) -> pd.DataFrame:
    """Per band: rows, records, households, effective sample size by record,
    weighted totals, and the largest record's and the largest two records'
    shares of the weight and of each amount.

    ``cells`` has one row per selected person, with ``w``, ``nsnd``, ``source``,
    ``cluster`` and the amount columns.
    """

    band = assign_bands(cells["nsnd"], edges)
    rows = []
    for index, label in enumerate(band_labels(edges)):
        members = cells[band == index]
        by_source = _by_source(members, amount_columns)
        weight = by_source["w"]
        row = {
            "band": label,
            "rows": len(members),
            "sources": len(by_source),
            "households": members["cluster"].nunique(),
            "source_ess": float(weight.sum() ** 2 / np.square(weight).sum())
            if len(weight)
            else 0.0,
            "count": float(weight.sum()),
        }
        row["top_source_weight_share"], row["top_two_source_weight_share"] = (
            _top_shares(weight)
        )
        for name in amount_columns:
            amount = by_source[name]
            row[name] = float(amount.sum())
            row[f"top_source_{name}_share"], row[f"top_two_source_{name}_share"] = (
                _top_shares(amount)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _cluster_draws(n_clusters: int, replicates: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.multinomial(
        n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=replicates
    )


def _codes(clusters: object, universe: pd.Index) -> np.ndarray:
    codes = universe.get_indexer(pd.Index(clusters))
    if np.any(codes < 0):
        raise ValueError("Some clusters are outside the bootstrap's universe.")
    return codes


def bootstrap(
    cells: pd.DataFrame,
    edges: Sequence[int],
    amount_columns: Sequence[str],
    *,
    universe: object,
    replicates: int = 1_000,
    seed: int = 0,
    level: float = 0.9,
) -> pd.DataFrame:
    """Percentile intervals and coefficients of variation per band.

    Clusters are drawn with replacement from ``universe``, every cluster in
    the domain (all Scottish persons), so the number of landlords varies
    between replicates as well as which band they fall in. Weights stay
    fixed, so calibration error is left out and the intervals are too narrow.
    """

    universe = pd.Index(pd.unique(np.asarray(universe)))
    codes = _codes(cells["cluster"], universe)
    draws = _cluster_draws(len(universe), replicates, seed)
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
    clusters: object,
    *,
    replicates: int = 1_000,
    seed: int = 0,
    level: float = 0.9,
) -> dict[str, float]:
    """The same cluster bootstrap for one weighted total over the whole domain.

    ``values`` are already weighted, one per person, and ``clusters`` cover
    every person in the domain.
    """

    values = np.asarray(values, dtype=float)
    codes, unique = pd.factorize(pd.Series(clusters))
    draws = _cluster_draws(len(unique), replicates, seed)
    by_cluster = np.bincount(codes, weights=values, minlength=len(unique))
    totals = draws @ by_cluster
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(totals, [tail, 1 - tail])
    estimate = float(values.sum())
    return {
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "cv": float(totals.std() / abs(estimate)) if estimate else float("nan"),
    }


def record_shares(values: object, sources: object) -> dict[str, float]:
    """Distinct contributing records and the top one and two records' shares
    of a total, for the dominance check on a single published number."""

    by_source = (
        pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(sources)).sum()
    )
    by_source = by_source[by_source != 0].abs()
    top, top_two = _top_shares(by_source)
    return {
        "records": int(len(by_source)),
        "top_record_share": top,
        "top_two_record_share": top_two,
    }


def passes_primary(
    records: int,
    top_shares: Sequence[float],
    top_two_shares: Sequence[float],
    *,
    min_sources: int = MIN_SOURCES,
    dominance: float = DOMINANCE_SHARE,
    top_two: float = TOP_TWO_SHARE,
) -> str:
    if records < min_sources:
        return FEWER_THAN_FLOOR
    if (
        max(top_shares, default=0.0) > dominance
        or max(top_two_shares, default=0.0) > top_two
    ):
        return DOMINATED
    return PUBLISHED


def _primary(table: pd.DataFrame, amount_columns: Sequence[str], **rules) -> list[str]:
    columns = ["weight", *amount_columns]
    return [
        passes_primary(
            int(row["sources"]),
            [row[f"top_source_{name}_share"] for name in columns],
            [row[f"top_two_source_{name}_share"] for name in columns],
            **rules,
        )
        for _, row in table.iterrows()
    ]


def _passes(
    cells: pd.DataFrame, mask: np.ndarray, amount_columns: Sequence[str], **rules
) -> bool:
    """Whether the persons under ``mask``, taken as one cell, pass the primary rules."""

    by_source = _by_source(cells[mask], amount_columns)
    shares = [_top_shares(by_source[column]) for column in ("w", *amount_columns)]
    status = passes_primary(
        len(by_source), [s[0] for s in shares], [s[1] for s in shares], **rules
    )
    return status == PUBLISHED


def disclosure_statuses(
    cells: pd.DataFrame,
    fine_edges: Sequence[int],
    coarse_edges: Sequence[int],
    amount_columns: Sequence[str],
    *,
    total_published: bool = True,
    min_sources: int = MIN_SOURCES,
    dominance: float = DOMINANCE_SHARE,
    top_two: float = TOP_TWO_SHARE,
) -> tuple[list[str], list[str]]:
    """Statuses for Release 1 (the ``fine`` bands) and Release 2 (the ``coarse``
    bins) together, read as one hierarchy: total, bins, bands.

    Primary: fewer than ``min_sources`` records, one record above
    ``dominance`` of the cell's weight or of any amount, or the top two above
    ``top_two``.

    Secondary: whatever a reader can rebuild by subtraction must itself pass
    the primary rules, so cells are withheld, fewest records first, until:

    - the withheld bands inside each published bin pass, taken together;
    - no withheld bin can be rebuilt from its bands;
    - with ``total_published``, the withheld bins pass together, and so do the
      withheld bands inside them. The 1p costing is about one per cent of the
      total property slice, so it gives the total away, and it is published.

    The same holds for record counts, which ``render`` shows for published
    cells only.
    """

    rules = {"min_sources": min_sources, "dominance": dominance, "top_two": top_two}
    fine = audit(cells, fine_edges, amount_columns)
    coarse = audit(cells, coarse_edges, amount_columns)
    fine_status = _primary(fine, amount_columns, **rules)
    coarse_status = _primary(coarse, amount_columns, **rules)
    band = assign_bands(cells["nsnd"], fine_edges)
    holder = coarse_index_of_fine(fine_edges, coarse_edges)

    def passes(bands: Sequence[int]) -> bool:
        return _passes(cells, np.isin(band, list(bands)), amount_columns, **rules)

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
) -> pd.DataFrame:
    """A release table.

    ``restricted`` shows every band in full, with exact record counts and the
    status as a note; it is for recipients covered by the licence. ``public``
    shows a band's values and record count only when it is published. A
    withheld band's record count would otherwise give a thin band's count by
    subtraction from its bin.
    """

    if mode not in {"restricted", "public"}:
        raise ValueError(f"Unknown mode {mode!r}.")
    rows = []
    for index, row in table.iterrows():
        interval = intervals.loc[index]
        status = statuses[index]
        shown = mode == "restricted" or status == PUBLISHED
        out = {
            "band": row["band"],
            "distinct_records": str(int(row["sources"])) if shown else WITHHELD_RECORDS,
            "effective_records": round(float(row["source_ess"]), 1) if shown else None,
        }
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
