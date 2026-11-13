"""Land as the residual of value less the depreciated cost of buildings.

A property's land is what is left of its value once its building is paid
for: ``max(0, value − κ × unit cost)``. The unit cost is floor area × a
relative cost per square metre × a regional factor × the share of the
building that depreciation has left. One national cost level, κ, sets the
level of every building. The functions here solve for κ against a stated
target, take the residual in pooled cells or along the value rank, scale a
comparison pattern under a cap, and run the checks a decision rule reads.

Every function is country-neutral: the costs, factors, profiles and targets
are the caller's inputs (microcosm#1524).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from scipy import stats

__all__ = [
    "cell_residual",
    "convex_mean",
    "cost_level_for_ratio",
    "cost_level_for_total",
    "depreciation_scale",
    "land_per_unit",
    "log_ratio_flags",
    "rank_correlation",
    "residual_along_rank",
    "scale_cap_rescale",
    "scale_retained_share",
    "unit_cost",
]


def _finite_nonnegative(name: str, array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{name} must be finite and nonnegative.")
    return array


def unit_cost(
    floor_area: np.ndarray,
    cost_per_m2: np.ndarray,
    regional_factor: np.ndarray,
    retained_share: np.ndarray,
) -> np.ndarray:
    """Floor area × relative cost per m² × regional factor × retained share.

    It is the cost at κ = 1, the level of whoever set the relative costs.
    The arrays broadcast against each other.
    """
    retained = _finite_nonnegative("The retained share", retained_share)
    if np.any(retained > 1):
        raise ValueError("The retained share must lie in [0, 1].")
    return (
        _finite_nonnegative("Floor area", floor_area)
        * _finite_nonnegative("The cost per m²", cost_per_m2)
        * _finite_nonnegative("The regional factor", regional_factor)
        * retained
    )


def depreciation_scale(
    shape: np.ndarray, weights: np.ndarray, target_mean: float, *, cap: float
) -> float:
    """The θ at which the weighted mean of ``min(cap, θ × shape)`` is the target.

    ``shape`` is a depreciation profile (by construction age, say) and
    ``weights`` the undepreciated costs, so the target is one less the ratio
    of the net to the gross capital stock. The mean rises with θ and is
    linear between the points where a unit reaches the cap, so the solution
    is exact. A target above what the cap allows is refused.
    """
    shape = _finite_nonnegative("The depreciation shape", shape)
    weights = _finite_nonnegative("The weights", weights)
    if shape.shape != weights.shape or not len(shape):
        raise ValueError("The shape and weights must be matching, nonempty arrays.")
    if not 0 < cap <= 1:
        raise ValueError("The depreciation cap must lie in (0, 1].")
    total = weights.sum()
    if not total > 0:
        raise ValueError("The weights must have a positive total.")
    if not 0 <= target_mean <= 1:
        raise ValueError("The target mean depreciation must lie in [0, 1].")
    if target_mean == 0:
        return 0.0
    active = (shape > 0) & (weights > 0)
    reachable = cap * weights[active].sum() / total
    if target_mean > reachable:
        raise ValueError(
            "The target mean depreciation exceeds what the cap and profile allow."
        )
    with np.errstate(over="ignore"):
        breaks = cap / shape[active]
    order = np.argsort(breaks, kind="stable")
    breaks = breaks[order]
    mass = weights[active][order]
    slope_mass = (mass * shape[active][order])[::-1].cumsum()[::-1]
    capped = cap * np.concatenate(([0.0], np.cumsum(mass)[:-1]))
    # At breakpoint j, the units before j are capped and the rest are not. A
    # shape too small for its breakpoint to be finite never reaches the cap.
    with np.errstate(over="ignore", invalid="ignore"):
        mean_at_break = np.where(
            np.isfinite(breaks), (capped + breaks * slope_mass) / total, np.inf
        )
    j = int(np.searchsorted(mean_at_break, target_mean, side="left"))
    j = min(j, len(breaks) - 1)
    theta = float((target_mean * total - capped[j]) / slope_mass[j])
    if not math.isfinite(theta):
        raise ValueError("The depreciation profile is too flat to reach the target.")
    return theta


def scale_retained_share(
    shape: np.ndarray, weights: np.ndarray, target_mean: float, *, cap: float
) -> tuple[np.ndarray, float]:
    """The share of each building left after depreciation, and its scale θ.

    Depreciation is ``min(cap, θ × shape)`` with θ from
    :func:`depreciation_scale`; the retained share is one less it.
    """
    theta = depreciation_scale(shape, weights, target_mean, cap=cap)
    depreciation = np.minimum(cap, theta * np.asarray(shape, dtype=float))
    return 1.0 - depreciation, theta


def land_per_unit(
    value_mean: np.ndarray, cost_mean: np.ndarray, kappa: float
) -> np.ndarray:
    """``max(0, value − κ × cost)``: never below 0 and never above the value."""
    if not (math.isfinite(kappa) and kappa >= 0):
        raise ValueError("The cost level must be finite and nonnegative.")
    value = _finite_nonnegative("The mean value", value_mean)
    cost = _finite_nonnegative("The mean cost", cost_mean)
    return np.maximum(0.0, value - kappa * cost)


def cell_residual(
    value_mean: np.ndarray,
    cost_mean: np.ndarray,
    counts: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Each cell's land: its count × the residual of its mean value and cost.

    The residual is taken once per cell, over the cell's pooled means, so a
    building dearer than the cell's mean does not push the cell's land below
    zero home by home.
    """
    counts = _finite_nonnegative("The counts", counts)
    return counts * land_per_unit(value_mean, cost_mean, kappa)


def residual_along_rank(
    value_by_bin: np.ndarray,
    cost_by_bin: np.ndarray,
    bin_weights: np.ndarray,
    kappa: float,
) -> np.ndarray:
    """Land per unit read at each value rank, then averaged over the ranks.

    Each row is a cell and each column a rank bin (a value decile, say),
    holding the bin's mean value and mean cost. Land in a bin is
    ``max(0, value − κ × cost)``, and a cell's land per unit is the weighted
    mean over its bins. By convexity it is never below the pooled residual.
    """
    weights = _finite_nonnegative("The bin weights", bin_weights)
    land = land_per_unit(value_by_bin, cost_by_bin, kappa)
    if land.shape != weights.shape or land.ndim != 2:
        raise ValueError("Values, costs and bin weights must be cells × bins.")
    total = weights.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Every cell needs a positive bin weight.")
    return (land * weights).sum(axis=1) / total


def cost_level_for_total(target_total: float, cost_totals: np.ndarray) -> float:
    """κ at which the buildings add up to a target, such as a net capital stock."""
    total = _finite_nonnegative("The cost totals", cost_totals).sum()
    if not total > 0:
        raise ValueError("The cost totals must have a positive sum.")
    if not (math.isfinite(target_total) and target_total > 0):
        raise ValueError("The target total must be finite and positive.")
    return float(target_total / total)


def cost_level_for_ratio(
    value_mean: np.ndarray,
    cost_mean: np.ndarray,
    counts: np.ndarray,
    ratio: float,
) -> float:
    """κ at which land is a stated share of value, over all cells.

    Solves ``Σ N × max(0, v − κu) = ratio × Σ N × v``. The left side is
    convex, falls as κ rises and is linear between the cells' breakpoints
    ``v / u``, so the root is unique where the side falls and is found
    exactly. A ratio outside what some κ can reach is refused.
    """
    value = _finite_nonnegative("The mean value", value_mean)
    cost = _finite_nonnegative("The mean cost", cost_mean)
    counts = _finite_nonnegative("The counts", counts)
    if not (value.shape == cost.shape == counts.shape) or not len(value):
        raise ValueError("Values, costs and counts must be matching, nonempty arrays.")
    if not 0 < ratio < 1:
        raise ValueError("The land-to-value ratio must lie in (0, 1).")
    stock = float((counts * value).sum())
    if not stock > 0:
        raise ValueError("The stock value must be positive.")
    target = ratio * stock
    costly = (cost > 0) & (value > 0) & (counts > 0)
    fixed = float((counts * value)[(cost == 0) & (counts > 0)].sum())
    if target <= fixed or not costly.any():
        raise ValueError("No cost level brings land down to that ratio.")
    breaks = value[costly] / cost[costly]
    order = np.argsort(breaks, kind="stable")
    breaks = breaks[order]
    value_mass = (counts * value)[costly][order]
    cost_mass = (counts * cost)[costly][order]
    suffix_value = value_mass[::-1].cumsum()[::-1]
    suffix_cost = cost_mass[::-1].cumsum()[::-1]
    # Land at breakpoint j: the cells from j on are still above zero.
    land_at_break = suffix_value - breaks * suffix_cost + fixed
    # The last breakpoint leaves only the fixed land, which is below the
    # target; rounding cannot move the root past it.
    reached = land_at_break <= target
    j = int(np.argmax(reached)) if reached.any() else len(breaks) - 1
    return float((suffix_value[j] + fixed - target) / suffix_cost[j])


def scale_cap_rescale(
    raw: np.ndarray, cap: np.ndarray, total: float
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a pattern to a total, cap each entry, and rescale the rest.

    Each entry is scaled in proportion to ``raw`` so the entries add up to
    ``total``. An entry above its cap is held at the cap, and the uncapped
    entries are scaled again to restore the total, until none is above its
    cap. Returns the entries and which were capped. A total above the sum
    of the caps is refused.
    """
    raw = _finite_nonnegative("The pattern", raw)
    cap = _finite_nonnegative("The caps", cap)
    if raw.shape != cap.shape:
        raise ValueError("The pattern and caps must match.")
    if not (math.isfinite(total) and total >= 0):
        raise ValueError("The total must be finite and nonnegative.")
    if total > cap.sum() * (1 + 1e-12):
        raise ValueError("The total exceeds the sum of the caps.")
    capped = np.zeros(raw.shape, dtype=bool)
    tolerance = 1e-9 * max(total, 1.0)
    while True:
        free = ~capped
        remaining = total - cap[capped].sum()
        if remaining < -tolerance:
            raise ValueError("The capped entries add up to more than the total.")
        mass = raw[free].sum()
        out = np.where(capped, cap, 0.0)
        if mass <= 0:
            if remaining > tolerance:
                raise ValueError("No uncapped entry is left to carry the total.")
            break
        # Each free entry takes its share of the free mass, which is at most
        # one, so a tiny mass cannot overflow the scale and cap an entry that
        # should carry only part of the total.
        out[free] = raw[free] / mass * max(remaining, 0.0)
        over = free & (out > cap)
        if not over.any():
            break
        capped |= over
    return out, capped


def rank_correlation(first: np.ndarray, second: np.ndarray) -> float:
    """Spearman's rank correlation, with tied values given their mean rank."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise ValueError("Rank correlation needs two matching series of 2 or more.")
    if not (np.all(np.isfinite(first)) and np.all(np.isfinite(second))):
        raise ValueError("Rank correlation needs finite values.")
    ranks_first = stats.rankdata(first)
    ranks_second = stats.rankdata(second)
    if np.ptp(ranks_first) == 0 or np.ptp(ranks_second) == 0:
        raise ValueError("Rank correlation is undefined for a constant series.")
    return float(np.corrcoef(ranks_first, ranks_second)[0, 1])


def log_ratio_flags(
    values: Mapping[str, float],
    references: Mapping[str, Mapping[str, float]],
    *,
    threshold: float,
    minimum_references: int,
) -> dict[str, list[str]]:
    """Areas whose value differs from enough references by a log ratio.

    An area is flagged when ``|log(value / reference)|`` exceeds the
    threshold against at least ``minimum_references`` of the references.
    Returns each flagged area with the references it differs from. A zero
    or missing value is not compared.
    """
    flagged: dict[str, list[str]] = {}
    for code, value in sorted(values.items()):
        if not (value > 0 and math.isfinite(value)):
            continue
        breaches = [
            name
            for name, reference in sorted(references.items())
            if (other := reference.get(code)) is not None
            and other > 0
            and math.isfinite(other)
            and abs(math.log(value / other)) > threshold
        ]
        if len(breaches) >= minimum_references:
            flagged[code] = breaches
    return flagged


def convex_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """A weighted mean that is checked to lie between its smallest and largest value."""
    values = np.asarray(values, dtype=float)
    weights = _finite_nonnegative("The weights", weights)
    if values.shape != weights.shape or not np.all(np.isfinite(values)):
        raise ValueError("Values must be finite and match the weights.")
    total = weights.sum()
    if not (total > 0 and math.isfinite(total)):
        raise ValueError("The weights must have a finite, positive total.")
    # The weights are made shares first, so a tiny weight cannot underflow
    # its value × weight to zero.
    mean = float((values * (weights / total)).sum())
    carried = values[weights > 0]
    span = max(abs(carried.min()), abs(carried.max()), 1.0)
    if not (carried.min() - 1e-12 * span <= mean <= carried.max() + 1e-12 * span):
        raise ValueError("The weighted mean lies outside its values.")
    return min(max(mean, float(carried.min())), float(carried.max()))
