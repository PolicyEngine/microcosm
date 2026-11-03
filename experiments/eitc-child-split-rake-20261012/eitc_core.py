"""Shared pieces for the EITC child-split reconciliation analysis.

Parsers for SOI Publication 1304 Table 2.5 (``YYin25ic.xls``) and the SOI
mid-November filing-season Table 1 (``YYinweek47*.xls[x]``, dumped to TSV by
``dump_xls.py``), plus the uprating options. Everything here is a pure
function of the parsed tables.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

SPLITS = ("childless", "one", "two", "three_plus")
KINDS = ("amount", "returns")


# ---------------------------------------------------------------------------
# AGI bounds
# ---------------------------------------------------------------------------


def _money(token: str) -> float:
    return float(token.replace("$", "").replace(",", ""))


def label_bounds(label: str) -> tuple[float, float]:
    """AGI bounds of an SOI size-of-AGI label."""

    text = " ".join(label.split())
    low = text.lower()
    if low.startswith("total"):
        return (-math.inf, math.inf)
    if low.startswith("no adjusted gross income"):
        return (-math.inf, 1.0)
    m = re.match(r"\$([\d,]+) under \$([\d,]+)", text)
    if m:
        return (_money(m.group(1)), _money(m.group(2)))
    m = re.match(r"\$([\d,]+) (?:and over|or more)", text)
    if m:
        return (_money(m.group(1)), math.inf)
    raise ValueError(f"unrecognized AGI label {label!r}")


def _rows(tsv: Path) -> list[list[str]]:
    return [
        line.rstrip("\n").split("\t")[1:]
        for line in tsv.open()
        if not line.startswith("###")
    ]


def _num(cell: str) -> float:
    cell = cell.strip()
    if cell in ("", "*", "**", "d", "[2]"):
        return 0.0
    return float(cell)


# ---------------------------------------------------------------------------
# Table 2.5
# ---------------------------------------------------------------------------


def parse_t25(tsv: Path):
    """Return (bins, cells[kind] (4 x n_bins), totals[kind] (4,)).

    Cells are the "Total earned income credit" returns and amount (amounts in
    dollars) of each qualifying-children block.
    """

    rows = _rows(tsv)
    blocks = {}
    for j, cell in enumerate(rows[2]):
        low = cell.lower()
        if "no qualifying" in low:
            blocks["childless"] = j
        elif "one qualifying" in low:
            blocks["one"] = j
        elif "two qualifying" in low:
            blocks["two"] = j
        elif "three or more qualifying" in low:
            blocks["three_plus"] = j
    assert set(blocks) == set(SPLITS), blocks
    ordered = sorted(blocks.values())
    column = {}
    for split, start in blocks.items():
        end = min([o for o in ordered if o > start], default=len(rows[3]))
        hits = [
            j
            for j in range(start, end)
            if "total earned income credit" in rows[3][j].lower()
        ]
        assert len(hits) == 1, (tsv, split, hits)
        column[split] = hits[0]
        assert rows[5][hits[0]].strip() == "Number of returns", rows[5][hits[0]]
        assert rows[5][hits[0] + 1].strip() == "Amount", rows[5][hits[0] + 1]
    bins, cells, totals = [], {k: [] for k in KINDS}, {}
    for row in rows[6:]:
        if not row or not row[0].strip():
            continue
        label = row[0].strip()
        try:
            lo, hi = label_bounds(label)
        except ValueError:
            continue
        returns = [_num(row[column[s]]) for s in SPLITS]
        amount = [_num(row[column[s] + 1]) * 1_000 for s in SPLITS]
        if (lo, hi) == (-math.inf, math.inf):
            totals = {"returns": np.array(returns), "amount": np.array(amount)}
            continue
        bins.append((lo, hi))
        cells["returns"].append(returns)
        cells["amount"].append(amount)
    assert totals, tsv
    return (
        bins,
        {k: np.array(v).T for k, v in cells.items()},
        totals,
    )


# ---------------------------------------------------------------------------
# Filing-season Table 1
# ---------------------------------------------------------------------------


def parse_filing_season(tsv: Path):
    """Return (bins, values[kind] (n_bins,), totals[kind]) for EITC by AGI."""

    rows = _rows(tsv)
    header = next(
        r for r in rows if any("$1 under $5,000" in " ".join(c.split()) for c in r)
    )
    label_cols = [
        (j, label_bounds(c))
        for j, c in enumerate(header)
        if c.strip()
        and (c.strip().startswith("$") or c.strip().lower().startswith("no adjusted"))
    ]
    first_bin_col = label_cols[0][0]
    eic = next(
        i
        for i, r in enumerate(rows)
        if r
        and r[0].strip().lower().startswith("earned income credit")
        and "number of returns" in r[0].lower()
    )
    returns_row, amount_row = rows[eic], rows[eic + 1]
    assert amount_row[0].strip() == "Amount", amount_row[0]
    # The total sits in the numeric column just before the first AGI bin
    # (headed "Total returns", or unheaded in the TY2017 Table 1C).
    total_col = first_bin_col - 1
    bins = [b for _, b in label_cols]
    values = {
        "returns": np.array([_num(returns_row[j]) for j, _ in label_cols]),
        "amount": np.array([_num(amount_row[j]) * 1_000 for j, _ in label_cols]),
    }
    totals = {
        "returns": _num(returns_row[total_col]),
        "amount": _num(amount_row[total_col]) * 1_000,
    }
    return bins, values, totals


# ---------------------------------------------------------------------------
# Uprating groups, as ``_eitc_agi_uprating_group`` forms them
# ---------------------------------------------------------------------------


def group_of(bins, active_bounds):
    """Each source bin's uprating group: the smallest active row containing
    it, else the union of the active rows it contains (which must tile it)."""

    groups: list[tuple[float, float]] = []
    index = []
    for lo, hi in bins:
        containing = [(a, b) for a, b in active_bounds if a <= lo and hi <= b]
        if containing:
            g = min(containing, key=lambda ab: ab[1] - ab[0])
        else:
            contained = sorted((a, b) for a, b in active_bounds if lo <= a and b <= hi)
            cursor = lo
            for a, b in contained:
                assert a == cursor, ((lo, hi), contained)
                cursor = b
            assert cursor == hi, ((lo, hi), contained)
            g = (lo, hi)
        if g not in groups:
            groups.append(g)
        index.append(groups.index(g))
    return groups, np.array(index)


def group_margins(active_bounds, active_values, groups):
    return np.array(
        [
            sum(
                v
                for (a, b), v in zip(active_bounds, active_values, strict=True)
                if lo <= a and b <= hi
            )
            for lo, hi in groups
        ]
    )


def collapse(matrix: np.ndarray, index: np.ndarray, n_groups: int) -> np.ndarray:
    out = np.zeros((matrix.shape[0], n_groups))
    for b, g in enumerate(index):
        out[:, g] += matrix[:, b]
    return out


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def rake(seed, rows, cols, tol=1e-13, max_iter=100_000):
    """Classical IPF (Deming-Stephan) of a nonnegative matrix to margins."""

    x = seed.astype(float).copy()
    if abs(rows.sum() - cols.sum()) > 1e-9 * cols.sum():
        raise ValueError((rows.sum(), cols.sum()))
    err = math.inf
    for iteration in range(max_iter):
        r = x.sum(axis=1)
        x *= np.divide(rows, r, out=np.ones_like(rows, dtype=float), where=r != 0)[
            :, None
        ]
        c = x.sum(axis=0)
        x *= np.divide(cols, c, out=np.ones_like(cols, dtype=float), where=c != 0)[
            None, :
        ]
        err = np.max(np.abs(x.sum(axis=1) - rows) / np.maximum(rows, 1))
        if err < tol:
            return x, iteration + 1
    raise RuntimeError(f"IPF did not converge: {err}")


def options(source, source_totals, index, margins):
    """All four options for one kind.

    ``source`` is the source-year split x bin matrix, ``source_totals`` its
    total rows, ``index`` each bin's group, ``margins`` the active groups.
    """

    n_groups = len(margins)
    grouped = collapse(source, index, n_groups)
    by_group = grouped.sum(axis=0)
    factors = np.divide(
        margins, by_group, out=np.ones_like(margins, dtype=float), where=by_group != 0
    )
    bins_a = source * factors[index][None, :]
    total_factor = margins.sum() / source_totals.sum()
    totals_a = source_totals * total_factor
    rows = totals_a * (margins.sum() / totals_a.sum())
    raked, iterations = rake(grouped, rows, margins)
    ratio = np.divide(raked, grouped, out=np.zeros_like(raked), where=grouped != 0)
    bins_c = source * ratio[:, index]
    return {
        "a": {"bins": bins_a, "totals": totals_a},
        "b": {"bins": bins_a, "totals": bins_a.sum(axis=1)},
        "c": {
            "bins": bins_c,
            "totals": bins_c.sum(axis=1),
            "ipf_iterations": iterations,
        },
        "d": {"bins": bins_a, "totals": None},
        "group_factors": factors,
        "total_factor": total_factor,
    }
