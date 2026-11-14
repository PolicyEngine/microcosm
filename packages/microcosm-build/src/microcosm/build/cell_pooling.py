"""Pool the units of a stock into cells, with fallbacks for thin cells.

A cell table pools the units of a stock (homes, for example) into cells of a
small area × a level (a council tax band, for example). A publisher counts
the stock in each cell; a sample of the units (homes seen sold, say) carries
the values. Four steps need care, and each is a function here:

1. **Levels by rank** (:func:`assign_levels_by_rank`). When the sample does
   not record a unit's level, the unit takes one by its value rank within
   its group, against the group's quota at each level: the publisher's
   count there. Ties in value fall back to a declared key, so the result
   never depends on input order.
2. **Weights** (:func:`stratum_weights`). Each sampled unit stands for its
   stratum's quota divided by the stratum's sampled units, so the weights of
   a stratum add up to its quota.
3. **Pools and fallbacks** (:func:`pool`, :func:`apply_fallback`). A cell
   with too few sampled units reads a statistic from the first ancestor
   pool that has enough. The ancestor is always the cell's own parent, then
   that parent's parent: a fallback never crosses into another branch.
4. **Allocation** (:func:`allocation_shares`). Cells that do not nest in a
   reporting geography are split across it by their units' own shares.

The functions carry no country detail. A country declares its levels and
minimums in a :class:`PoolingDeclaration` and passes its codes as integer
indices. The level assignment ports the residual prototype's proxy bands
(PolicyEngine/uk-lvt-paper#16) with a key tie-break (microcosm#1524).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd

__all__ = [
    "PoolingDeclaration",
    "PooledCells",
    "allocation_shares",
    "apply_fallback",
    "assign_levels_by_rank",
    "cumulative_quota_shares",
    "pool",
    "stratum_weights",
]

#: Rows compared against the level boundaries at once; it bounds the memory
#: of the comparison at about 8 MiB per boundary.
_CHUNK_ROWS = 1 << 20


@dataclass(frozen=True)
class PoolingDeclaration:
    """How a cell table pools: its levels, finest first, and its minimums.

    ``levels`` names each pooling level from the cell itself up to the top,
    for example small area × band, authority × band, region × band and
    nation × band. ``minimums`` maps each statistic to the sampled units a
    pool needs before the statistic is read from it.
    """

    levels: tuple[str, ...]
    minimums: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("A pooling declaration needs distinct levels.")
        for name, value in self.minimums.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"The minimum for {name} must be a positive integer.")
        object.__setattr__(self, "minimums", MappingProxyType(dict(self.minimums)))

    def minimum(self, statistic: str) -> int:
        """The sampled units a pool needs for ``statistic``."""
        if statistic not in self.minimums:
            raise KeyError(f"The pooling declaration has no minimum for {statistic}.")
        return self.minimums[statistic]


def cumulative_quota_shares(quotas: np.ndarray) -> np.ndarray:
    """Each group's cumulative quota shares: the upper boundary of each level.

    Rows are groups and columns are levels. A level with no quota repeats the
    boundary below it exactly, so no rank can fall in it. The last level with
    a quota closes at exactly 1, and so does every level after it. A group
    with no quota at all has boundaries of 0.
    """
    quotas = np.asarray(quotas, dtype=float)
    if quotas.ndim != 2 or quotas.shape[1] < 1:
        raise ValueError("Quotas must be a groups × levels matrix.")
    if not np.all(np.isfinite(quotas)) or np.any(quotas < 0):
        raise ValueError("Quotas must be finite and nonnegative.")
    totals = quotas.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        shares = np.where(totals > 0, np.cumsum(quotas, axis=1) / totals, 0.0)
    shares = np.minimum(shares, 1.0)
    levels = quotas.shape[1]
    positive = quotas > 0
    last = levels - 1 - np.argmax(positive[:, ::-1], axis=1)
    closed = (np.arange(levels)[None, :] >= last[:, None]) & positive.any(axis=1)[
        :, None
    ]
    shares[closed] = 1.0
    return shares


def assign_levels_by_rank(
    values: np.ndarray,
    tie_break: np.ndarray,
    groups: np.ndarray,
    quotas: np.ndarray,
) -> np.ndarray:
    """Give each unit a level by its value rank within its group.

    Within a group, units are ordered by value, then by ``tie_break``. Unit
    ``i`` of ``n`` sits at the midpoint rank ``(i + 0.5) / n`` and takes the
    level whose span of the group's cumulative quota shares holds that rank.
    So a higher value never takes a lower level, and the result does not
    depend on input order. Only quota shares matter: scaling a group's
    quotas by a power of two changes nothing, and another scale can move
    only a unit whose rank sits exactly on a boundary, by floating-point
    rounding. A level with no quota never receives a unit. Every group that
    holds units must have a positive quota; resolving one that does not is
    the caller's call.
    """
    values = np.asarray(values, dtype=float)
    tie_break = np.asarray(tie_break)
    groups = np.asarray(groups)
    quotas = np.asarray(quotas, dtype=float)
    count = len(values)
    if values.ndim != 1 or tie_break.shape != (count,) or groups.shape != (count,):
        raise ValueError("Values, tie-breaks and groups must be matching 1-D arrays.")
    boundaries = cumulative_quota_shares(quotas)
    if count == 0:
        return np.zeros(0, dtype=np.int64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Values must be finite.")
    if groups.dtype.kind not in "iu":
        raise ValueError("Groups must be integer indices.")
    groups = groups.astype(np.int64)
    if groups.min() < 0 or groups.max() >= quotas.shape[0]:
        raise ValueError("A group index falls outside the quota matrix.")
    members = np.bincount(groups, minlength=quotas.shape[0])
    if np.any((members > 0) & (quotas.sum(axis=1) <= 0)):
        raise ValueError("A group that holds units has no quota.")
    order = np.lexsort((tie_break, values, groups))
    sorted_groups = groups[order]
    starts = np.concatenate(([0], np.cumsum(members)[:-1]))
    rank = np.arange(count) - starts[sorted_groups]
    position = (rank + 0.5) / members[sorted_groups]
    # A unit's level is the number of boundaries at or below its rank; the
    # last boundary is 1 and no rank reaches it.
    inner = boundaries[:, :-1]
    sorted_levels = np.empty(count, dtype=np.int64)
    for begin in range(0, count, _CHUNK_ROWS):
        stop = min(begin + _CHUNK_ROWS, count)
        sorted_levels[begin:stop] = (
            inner[sorted_groups[begin:stop]] <= position[begin:stop, None]
        ).sum(axis=1)
    levels = np.empty(count, dtype=np.int64)
    levels[order] = sorted_levels
    return levels


def stratum_weights(
    groups: np.ndarray, levels: np.ndarray, quotas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each unit's weight, and the quota no sampled unit represents.

    A stratum is a group × level. Each sampled unit in it weighs the
    stratum's quota over its sampled units, so a represented stratum's
    weights add up to its quota. The second element is the quota of each
    stratum that has no sampled unit; its pool reads from a fallback.
    """
    groups = np.asarray(groups, dtype=np.int64)
    levels = np.asarray(levels, dtype=np.int64)
    quotas = np.asarray(quotas, dtype=float)
    if groups.shape != levels.shape or quotas.ndim != 2:
        raise ValueError("Groups and levels must match, and quotas be a matrix.")
    n_groups, n_levels = quotas.shape
    if len(groups) and (
        groups.min() < 0
        or groups.max() >= n_groups
        or levels.min() < 0
        or levels.max() >= n_levels
    ):
        raise ValueError("A stratum falls outside the quota matrix.")
    flat = groups * n_levels + levels
    sampled = np.bincount(flat, minlength=n_groups * n_levels).reshape(quotas.shape)
    if np.any((sampled > 0) & (quotas <= 0)):
        raise ValueError("A sampled stratum has no quota.")
    weights = quotas.ravel()[flat] / sampled.ravel()[flat]
    unrepresented = np.where(sampled == 0, quotas, 0.0)
    return weights, unrepresented


@dataclass(frozen=True)
class PooledCells:
    """Per-pool sums: sampled units, their weight and weighted means."""

    sampled: np.ndarray
    weight: np.ndarray
    means: Mapping[str, np.ndarray]


def pool(
    pools: np.ndarray,
    weights: np.ndarray,
    columns: Mapping[str, np.ndarray],
    *,
    n_pools: int,
) -> PooledCells:
    """Sampled units, weight and weighted mean of each column, per pool.

    A pool with no weight has a mean of NaN, so a reader cannot take it for
    a value; :func:`apply_fallback` sends its cells to a parent instead.
    """
    pools = np.asarray(pools, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    if pools.shape != weights.shape:
        raise ValueError("Pools and weights must match.")
    if len(pools) and (pools.min() < 0 or pools.max() >= n_pools):
        raise ValueError("A pool index falls outside the pools.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative.")
    sampled = np.bincount(pools, minlength=n_pools)
    weight = np.bincount(pools, weights=weights, minlength=n_pools)
    means: dict[str, np.ndarray] = {}
    for name, column in columns.items():
        column = np.asarray(column, dtype=float)
        if column.shape != pools.shape:
            raise ValueError(f"Column {name} does not match the pools.")
        total = np.bincount(pools, weights=weights * column, minlength=n_pools)
        with np.errstate(divide="ignore", invalid="ignore"):
            means[name] = np.where(weight > 0, total / weight, np.nan)
    return PooledCells(sampled=sampled, weight=weight, means=MappingProxyType(means))


def apply_fallback(
    sampled_by_level: Sequence[np.ndarray],
    parents: Sequence[np.ndarray],
    *,
    minimum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where each finest cell reads a statistic that needs ``minimum`` units.

    ``sampled_by_level[k]`` is the sampled units of each pool at level ``k``
    (level 0 is the cell itself). ``parents[k]`` maps each level-``k`` pool
    to its level-``k + 1`` parent. A cell reads its own pool when it has
    enough units; otherwise it moves to its parent, and so on. The top level
    is the last resort, and a cell that reads it with too few units is
    flagged. Returns the level read, the pool index at that level and the
    flag, per cell.
    """
    if len(parents) != len(sampled_by_level) - 1 or not sampled_by_level:
        raise ValueError("Each level but the top needs a parent map.")
    sampled = [np.asarray(level, dtype=np.int64) for level in sampled_by_level]
    maps = [np.asarray(parent, dtype=np.int64) for parent in parents]
    for k, parent in enumerate(maps):
        if parent.shape != sampled[k].shape:
            raise ValueError(f"Level {k}'s parent map does not match its pools.")
        if len(parent) and (parent.min() < 0 or parent.max() >= len(sampled[k + 1])):
            raise ValueError(f"Level {k}'s parent map points outside level {k + 1}.")
    cells = len(sampled[0])
    level = np.zeros(cells, dtype=np.int64)
    index = np.arange(cells, dtype=np.int64)
    pending = sampled[0] < minimum
    for k in range(1, len(sampled)):
        # Only the cells still short move; the others keep a lower level's
        # index, which level k's counts cannot read.
        moving = np.flatnonzero(pending)
        index[moving] = maps[k - 1][index[moving]]
        level[moving] = k
        pending = np.zeros(cells, dtype=bool)
        pending[moving] = sampled[k][index[moving]] < minimum
    return level, index, pending


def allocation_shares(
    cells: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    n_cells: int,
    default: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Each cell's share in each target area, from its units' weights.

    Returns columns ``cell``, ``target`` and ``share``, sorted, with each
    cell's shares adding up to 1. A cell with no weight takes its rows from
    ``default`` (same columns); a cell with neither is refused.
    """
    cells = np.asarray(cells, dtype=np.int64)
    weights = np.asarray(weights, dtype=float)
    targets = np.asarray(targets)
    if not (cells.shape == weights.shape == targets.shape):
        raise ValueError("Cells, targets and weights must match.")
    if len(cells) and (cells.min() < 0 or cells.max() >= n_cells):
        raise ValueError("A cell index falls outside the cells.")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Weights must be finite and nonnegative.")
    frame = pd.DataFrame({"cell": cells, "target": targets, "weight": weights})
    sums = frame.groupby(["cell", "target"], sort=True)["weight"].sum().reset_index()
    totals = sums.groupby("cell")["weight"].transform("sum")
    sums = sums[totals > 0].copy()
    sums["share"] = sums["weight"] / totals[totals > 0]
    observed = sums[["cell", "target", "share"]]
    missing = np.setdiff1d(np.arange(n_cells), observed["cell"].to_numpy())
    if len(missing):
        if default is None:
            raise ValueError(f"{len(missing)} cells have no weight and no default.")
        filler = default[default["cell"].isin(missing)][["cell", "target", "share"]]
        absent = np.setdiff1d(missing, filler["cell"].to_numpy())
        if len(absent):
            raise ValueError(f"{len(absent)} cells have no weight and no default row.")
        filler_totals = filler.groupby("cell")["share"].transform("sum")
        if not np.allclose(filler_totals.to_numpy(), 1.0, rtol=0, atol=1e-9):
            raise ValueError("The default shares of a cell must add up to 1.")
        observed = pd.concat([observed, filler], ignore_index=True)
    return observed.sort_values(["cell", "target"], kind="stable").reset_index(
        drop=True
    )
