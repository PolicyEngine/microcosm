"""UK firm turnover against DBT BPE 2025: employment re-pairing and open tail.

The experimental UK firm generator draws turnover from ONS turnover bands and,
independently, employment from ONS employment bands. Two consequences show up
against the DBT *Business population estimates 2025* (BPE) turnover by size
band: turnover is spread evenly over employment bands (no firm-size gradient),
and the open ONS "5000+" turnover band, drawn log-uniform on [£5m, £50m),
cannot hold the turnover BPE attributes to large businesses. This module adds
three opt-in pieces; none runs by default. ``firm_generation`` runs them when
``UKFirmGenerationConfig.bpe_turnover`` is set (``build_uk_firm_dataset``'s
``bpe_turnover``).

A. Re-pairing (``repair_uk_firm_employment_turnover``). After employment is
   drawn, the drawn employment values are permuted among ONS-frame rows
   *within each SIC division* by a Gaussian copula on turnover rank: for row
   ``i`` of a division with ``n`` frame rows and turnover rank ``r_i``
   (0-based, ties broken by firm id), ``z_i = rho * Phi^-1((r_i + 0.5) / n) +
   sqrt(1 - rho^2) * eps_i``, where ``eps`` is a standard normal drawn from
   ``np.random.default_rng(SeedSequence([seed, tag]))`` in firm-id order. The
   row with the k-th smallest ``z`` takes the k-th smallest employment value of
   that division's frame rows. Each division's multiset of employment values
   and every row's turnover are kept exactly; rows outside the frame are not
   touched. ``rho = 0`` is a random permutation, ``rho = 1`` a monotone
   pairing. The permutation is unweighted: with stratified thinning (unequal
   base weights) it keeps each division's row multiset but not its
   base-weighted employment-band totals.

B. ``rho`` fit (``fit_uk_firm_employment_turnover_rho``). ``rho`` is the
   value on [0, 0.99] (a 0.05 grid, then a 0.005 grid around the best point)
   minimising the squared error between the base-weighted turnover shares of
   the seven ONS employment bands (0-4 ... 250+, on the generator's
   employment, i.e. ONS employment including working proprietors) and the BPE
   shares. The fit and the re-pairing share the same ``eps`` stream, so
   applying the fitted ``rho`` reproduces the fit table exactly. ``rho`` is
   never fixed in code.

C. Open turnover band tail (``uk_firm_open_band_tail_parameters`` and
   ``draw_uk_firm_open_band_turnover``). ONS "5000+" rows draw from a
   truncated Pareto on [£5m, C] instead of log-uniform [£5m, £50m). Rows are
   ranked by the generator's own uniforms (no new random draws) and the row
   of rank k among n takes the mean of the distribution over the quantile
   slice [k/n, (k+1)/n], so every value lies in [£5m, C] and the sample mean
   equals the distribution mean exactly for any n.

D. Diagnostics: ``uk_firm_turnover_by_employment_band`` (model against BPE,
   with any weights, so before or after the solve) and
   ``uk_firm_bpe_turnover_pins`` (every pin with its source).

Pins and sources:

- BPE turnover by size band: DBT, *Business population estimates 2025*,
  detailed tables,
  https://assets.publishing.service.gov.uk/media/68dbccc9c487360cc70c9f4e/BPE_2025_detailed_tables.xlsx
  (landing page https://www.gov.uk/government/statistics/business-population-estimates-2025),
  Table 2 whole economy turnover by employee band, less Table 1's "With no employees
  (unregistered)" private-sector turnover, taken off the 0-4 band (the ONS
  frame holds VAT- or PAYE-registered enterprises only). Read from
  ``tests/fixtures/uk/firm_bpe_2025/bpe_size_bands.csv``.
- Section K: BPE turnover already excludes SIC 2007 Section K (fixture
  provenance note), so nothing is subtracted on the BPE side; on the model
  side the Section K divisions 64-66 are left out of every turnover share and
  of the tail mean.
- Tail shape: ``P(turnover > £10m | open band)`` is HMRC VAT registered
  traders in the "Greater_than_£10m" band divided by ONS enterprises in the
  "5000+" turnover band (all SIC rows), both read from the generator's source
  tables; ``alpha = -ln(p) / ln(10m / 5m)``, the untruncated Pareto reading.
  This mixes VAT traders with ONS enterprises (assumed), and with the cap the
  model's truncated ``P(> £10m)`` lands slightly under ``p``
  (``UKFirmOpenBandTail.model_p_over_10m`` reports it).
- Tail cap: C is solved so the non-K open-band mean equals (BPE whole-economy
  turnover less BPE unregistered turnover less the drawn, weighted non-K
  closed-band turnover) divided by the weighted non-K open-band firm count.
  Section K open-band rows use the same shape (assumed).

Assumptions and caveats (status ``assumed``):

- BPE bands are by *employees*; the model's bands are ONS *employment*
  (employees plus working proprietors). The bands are matched across that
  difference ("Zero or 1" and "2 to 4" to 0-4, "100 to 199" and "200 to 249"
  to 100-249, 250 and above to 250+).
- BPE describes the start of 2025; the ONS frame is the generator's vintage.
- BPE turnover by band becomes an in-sample fact for the shares and the tail
  mean: a later comparison with that table is not an independent check.
- The tail's mean is exact within each call (or each ``strata`` group): the
  non-K and Section K open-band rows must be ranked separately, or the non-K
  mean is exact only in expectation.

Every number comes from the source tables passed in; none is written here
beyond band definitions (ONS and HMRC band edges, SIC 2007 Section K).
"""

from __future__ import annotations

import math
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import ndtri

from microcosm.build.uk_runtime.firm_headcount import UKFirmBPESource

UK_FIRM_BPE_TURNOVER_STATUS = "assumed"
UK_FIRM_BPE_TURNOVER_METHOD = (
    "ONS-frame employment permuted within SIC division by a Gaussian copula on "
    "turnover rank (rho fitted so base-weighted non-Section-K turnover shares "
    "by ONS employment band match DBT Business population estimates 2025 "
    "Table 2 whole-economy turnover by employee band less Table 1 "
    "unregistered turnover); ONS 5000+ turnover drawn as rank-slice means of a "
    "truncated Pareto on [5m, C], shape from HMRC traders above 10m over ONS "
    "5000+ enterprises, cap C matching the BPE registered turnover left after "
    "drawn closed-band turnover"
)

#: ONS employment bands, in the generator's ``EMPLOYMENT_BANDS`` order.
UK_FIRM_TURNOVER_EMPLOYMENT_BANDS: tuple[str, ...] = (
    "0-4",
    "5-9",
    "10-19",
    "20-49",
    "50-99",
    "100-249",
    "250+",
)
#: Upper edges of the closed ONS employment bands (the generator's
#: ``ONS_EMPLOYMENT_BANDS``), for ``searchsorted`` band lookup.
_EMPLOYMENT_BAND_UPPER_EDGES = np.asarray((4.0, 9.0, 19.0, 49.0, 99.0, 249.0))
#: BPE 2025 Table 2 employee bands summed into each ONS employment band.
UK_FIRM_BPE_BANDS_BY_ONS_BAND: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "0-4": ("Zero or 1 employee", "2 to 4"),
        "5-9": ("5 to 9",),
        "10-19": ("10 to 19",),
        "20-49": ("20 to 49",),
        "50-99": ("50 to 99",),
        "100-249": ("100 to 199", "200 to 249"),
        "250+": ("250 to 499", "500 to 999", "1000 or more"),
    }
)
#: SIC 2007 Section K (financial and insurance activities) divisions.
SIC_SECTION_K_DIVISIONS: tuple[int, ...] = (64, 65, 66)
#: The ONS open turnover band label and its lower edge in £ thousands.
ONS_OPEN_TURNOVER_BAND_LABEL = "5000+"
OPEN_BAND_LOWER_K = 5_000.0
#: Upper edge of the generator's legacy log-uniform open-band draw (£k).
OPEN_BAND_LOG_UNIFORM_UPPER_K = 50_000.0
#: HMRC's top VAT turnover band and its exclusive lower edge (£k).
HMRC_TOP_TURNOVER_BAND = "Greater_than_£10m"
HMRC_TOP_BAND_LOWER_K = 10_000.0
#: BPE rows the pins read.
_BPE_WHOLE_ECONOMY = ("Table 2", "whole_economy")
_BPE_PRIVATE_TABLE_1 = ("Table 1", "private_sector")
_BPE_ALL = "All"
_BPE_UNREGISTERED = "With no employees (unregistered)"
#: ``rho`` search: coarse grid, then a fine grid around the coarse best.
RHO_UPPER = 0.99
_RHO_COARSE_STEP = 0.05
_RHO_FINE_STEP = 0.005
#: Distinguishes the re-pairing's random stream from any other use of seed.
_REPAIR_SEED_TAG = zlib.crc32(b"microcosm.uk_firm_bpe_turnover.repair")


# -- shared helpers --------------------------------------------------------------


def _array(values: Any, name: str, n: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if n is not None and len(array) != n:
        raise ValueError(f"{name} must have one entry per firm ({n}).")
    return array


def _firm_ids(firm_ids: Any, n: int) -> np.ndarray:
    if firm_ids is None:
        return np.arange(n, dtype=np.int64)
    ids = _array(firm_ids, "firm_ids", n)
    if ids.dtype.kind not in "iu":
        raise ValueError("UK firm ids must be integers.")
    ordered = np.sort(ids.astype(np.int64))
    if ordered.size and np.any(np.diff(ordered) == 0):
        raise ValueError("UK firm ids repeat.")
    return ids.astype(np.int64)


def uk_firm_employment_band_index(employment: Any) -> np.ndarray:
    """ONS employment band index (0-6, ``UK_FIRM_TURNOVER_EMPLOYMENT_BANDS``)."""

    values = np.asarray(employment, dtype=np.float64)
    return np.searchsorted(_EMPLOYMENT_BAND_UPPER_EDGES, values, side="left").astype(
        np.int64
    )


def _non_k(sic: np.ndarray) -> np.ndarray:
    return ~np.isin(sic.astype(np.int64), SIC_SECTION_K_DIVISIONS)


def _bpe_row(source: UKFirmBPESource, key: tuple[str, str], band: str) -> float:
    frame = source.size_bands
    rows = frame[
        (frame["table"] == key[0])
        & (frame["sector"] == key[1])
        & (frame["size_band"] == band)
    ]
    if len(rows) != 1:
        raise ValueError(f"UK firm BPE {key} {band!r} found {len(rows)}x.")
    value = float(rows["turnover_million"].iloc[0])
    if not math.isfinite(value):
        raise ValueError(f"UK firm BPE {key} {band!r} turnover is missing.")
    return value


# -- BPE target ------------------------------------------------------------------


def uk_firm_bpe_turnover_by_band(source: UKFirmBPESource) -> pd.DataFrame:
    """BPE registered whole-economy turnover by ONS employment band.

    Columns ``band``, ``bpe_rows`` (the Table 2 employee bands summed),
    ``table_2_turnover_million``, ``unregistered_turnover_million`` (Table 1
    unregistered, on 0-4 only), ``bpe_turnover_million`` and ``bpe_share``.
    BPE turnover excludes Section K at source.
    """

    unregistered = _bpe_row(source, _BPE_PRIVATE_TABLE_1, _BPE_UNREGISTERED)
    rows = []
    for band in UK_FIRM_TURNOVER_EMPLOYMENT_BANDS:
        labels = UK_FIRM_BPE_BANDS_BY_ONS_BAND[band]
        table_2 = sum(_bpe_row(source, _BPE_WHOLE_ECONOMY, label) for label in labels)
        less = unregistered if band == UK_FIRM_TURNOVER_EMPLOYMENT_BANDS[0] else 0.0
        rows.append(
            {
                "band": band,
                "bpe_rows": " + ".join(labels),
                "table_2_turnover_million": table_2,
                "unregistered_turnover_million": less,
                "bpe_turnover_million": table_2 - less,
            }
        )
    table = pd.DataFrame(rows)
    if (table["bpe_turnover_million"] <= 0).any():
        raise ValueError("UK firm BPE registered turnover by band must be positive.")
    table["bpe_share"] = (
        table["bpe_turnover_million"] / table["bpe_turnover_million"].sum()
    )
    return table


def _target_shares(source: UKFirmBPESource | Sequence[float]) -> np.ndarray:
    if isinstance(source, UKFirmBPESource):
        return uk_firm_bpe_turnover_by_band(source)["bpe_share"].to_numpy(np.float64)
    shares = np.asarray(source, dtype=np.float64)
    if shares.shape != (len(UK_FIRM_TURNOVER_EMPLOYMENT_BANDS),) or not (
        np.all(shares >= 0) and shares.sum() > 0
    ):
        raise ValueError("Target shares must be seven non-negative band values.")
    return shares / shares.sum()


# -- A. re-pairing ---------------------------------------------------------------


class _Pairing:
    """Per-division turnover scores, noise and sorted employment, precomputed."""

    def __init__(
        self,
        sic: Any,
        turnover: Any,
        employment: Any,
        in_frame: Any,
        seed: int,
        firm_ids: Any,
    ) -> None:
        self.employment = _array(employment, "employment")
        n = len(self.employment)
        sic_codes = _array(sic, "sic", n).astype(np.int64)
        values = _array(turnover, "turnover", n).astype(np.float64)
        frame = _array(in_frame, "in_frame", n).astype(bool)
        ids = _firm_ids(firm_ids, n)
        rows = np.flatnonzero(frame)
        # Frame rows ordered by (division, firm id): fixes segments and ties.
        order = rows[np.lexsort((ids[rows], sic_codes[rows]))]
        division = sic_codes[order]
        noise = np.empty(n, dtype=np.float64)
        by_id = rows[np.argsort(ids[rows], kind="stable")]
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), _REPAIR_SEED_TAG])
        )
        noise[by_id] = rng.standard_normal(by_id.size)
        starts = np.searchsorted(division, division, side="left")
        sizes = np.searchsorted(division, division, side="right") - starts
        by_turnover = np.lexsort((ids[order], values[order], division))
        rank = np.empty(order.size, dtype=np.float64)
        rank[by_turnover] = np.arange(order.size) - starts[by_turnover]
        self.n = n
        self.order = order
        self.division = division
        self.ids = ids[order]
        self.score = ndtri((rank + 0.5) / np.maximum(sizes, 1))
        self.noise = noise[order]
        self.sorted_employment = self.employment[order][
            np.lexsort((self.employment[order], division))
        ]

    def repaired(self, rho: float) -> np.ndarray:
        rho = float(rho)
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"rho must lie in [0, 1], got {rho}.")
        z = rho * self.score + math.sqrt(1.0 - rho * rho) * self.noise
        ranked = self.order[np.lexsort((self.ids, z, self.division))]
        result = self.employment.copy()
        result[ranked] = self.sorted_employment
        return result


def repair_uk_firm_employment_turnover(
    sic: Any,
    turnover: Any,
    employment: Any,
    in_frame: Any,
    *,
    rho: float,
    seed: int,
    firm_ids: Any = None,
) -> np.ndarray:
    """Employment re-paired with turnover inside each SIC division.

    Returns a copy of ``employment`` (same dtype) in which the ONS-frame rows
    of each division hold a permutation of that division's frame values,
    ordered by the Gaussian copula score described in the module docstring;
    rows with ``in_frame`` False keep their value. ``firm_ids`` (integers,
    unique; row position when omitted) key the noise and break ties, so the
    result per firm does not depend on row order.
    """

    return _Pairing(sic, turnover, employment, in_frame, seed, firm_ids).repaired(rho)


# -- B. rho fit ------------------------------------------------------------------


def _band_turnover(
    employment: np.ndarray,
    turnover: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    bands = uk_firm_employment_band_index(employment[mask])
    return np.bincount(
        bands,
        weights=turnover[mask] * weights[mask],
        minlength=len(UK_FIRM_TURNOVER_EMPLOYMENT_BANDS),
    )


@dataclass(frozen=True)
class UKFirmRhoFit:
    """The fitted copula ``rho`` and how it was found.

    ``table`` is the per-band fit (model turnover and shares before
    re-pairing, at ``rho``, and the target); ``search`` every ``rho``
    evaluated with its squared error. ``at_bound`` is True when the optimum
    is an end of [0, ``RHO_UPPER``].
    """

    rho: float
    sse: float
    at_bound: bool
    table: pd.DataFrame
    search: pd.DataFrame


def fit_uk_firm_employment_turnover_rho(
    sic: Any,
    turnover: Any,
    employment: Any,
    in_frame: Any,
    base_weights: Any,
    source: UKFirmBPESource | Sequence[float],
    *,
    seed: int,
    firm_ids: Any = None,
) -> UKFirmRhoFit:
    """Fit ``rho`` to BPE turnover shares by ONS employment band.

    Shares are base-weighted turnover over ONS-frame rows outside Section K
    (divisions 64-66), by the band of the re-paired employment. ``source`` is
    the BPE tables (target from :func:`uk_firm_bpe_turnover_by_band`) or
    seven target values in band order. The search is a 0.05 grid on
    [0, 0.99] (0.99 included), then a 0.005 grid within 0.05 of the coarse
    best; ties go to the smaller ``rho``. ``seed`` and ``firm_ids`` must be
    those later passed to :func:`repair_uk_firm_employment_turnover`.
    """

    pairing = _Pairing(sic, turnover, employment, in_frame, seed, firm_ids)
    n = pairing.n
    values = _array(turnover, "turnover", n).astype(np.float64)
    weights = _array(base_weights, "base_weights", n).astype(np.float64)
    mask = _array(in_frame, "in_frame", n).astype(bool) & _non_k(_array(sic, "sic", n))
    if not mask.any():
        raise ValueError("rho fit needs ONS-frame rows outside Section K.")
    target = _target_shares(source)

    evaluated: dict[float, tuple[float, np.ndarray]] = {}

    def evaluate(rho: float) -> float:
        rho = round(float(rho), 6)
        if rho not in evaluated:
            totals = _band_turnover(pairing.repaired(rho), values, weights, mask)
            shares = totals / totals.sum()
            evaluated[rho] = (float(((shares - target) ** 2).sum()), totals)
        return evaluated[rho][0]

    coarse = np.append(np.arange(0.0, RHO_UPPER, _RHO_COARSE_STEP), RHO_UPPER)
    best = min(coarse, key=lambda rho: (evaluate(rho), rho))
    fine = np.arange(
        max(0.0, best - _RHO_COARSE_STEP),
        min(RHO_UPPER, best + _RHO_COARSE_STEP) + _RHO_FINE_STEP / 2,
        _RHO_FINE_STEP,
    )
    fine = np.clip(fine, 0.0, RHO_UPPER)
    best = min([*fine, best], key=lambda rho: (evaluate(rho), round(float(rho), 6)))
    rho = round(float(best), 6)
    sse, fitted = evaluated[rho]
    before = _band_turnover(pairing.employment, values, weights, mask)
    table = pd.DataFrame(
        {
            "band": UK_FIRM_TURNOVER_EMPLOYMENT_BANDS,
            "model_turnover_k_before": before,
            "model_share_before": before / before.sum(),
            "model_turnover_k_fitted": fitted,
            "model_share_fitted": fitted / fitted.sum(),
            "target_share": target,
        }
    )
    search = pd.DataFrame(
        sorted((r, s) for r, (s, _) in evaluated.items()), columns=["rho", "sse"]
    )
    return UKFirmRhoFit(
        rho=rho,
        sse=sse,
        at_bound=rho in (0.0, RHO_UPPER),
        table=table,
        search=search,
    )


# -- C. open-band tail -----------------------------------------------------------


def _truncated_pareto_mean(alpha: float, lower: float, cap: float) -> float:
    tail = (lower / cap) ** alpha
    if abs(alpha - 1.0) < 1e-9:
        return lower * math.log(cap / lower) / (1.0 - tail)
    return (
        alpha
        * lower**alpha
        * (cap ** (1.0 - alpha) - lower ** (1.0 - alpha))
        / ((1.0 - alpha) * (1.0 - tail))
    )


def truncated_pareto_exceedance(
    threshold: float, alpha: float, lower: float, cap: float
) -> float:
    """``P(X > threshold)`` for the Pareto(``alpha``) on [``lower``, ``cap``]."""

    if threshold <= lower:
        return 1.0
    if threshold >= cap:
        return 0.0
    tail = (lower / cap) ** alpha
    return ((lower / threshold) ** alpha - tail) / (1.0 - tail)


def _slice_means(n: int, alpha: float, lower: float, cap: float) -> np.ndarray:
    """Means of the truncated Pareto over the quantile slices [k/n, (k+1)/n]."""

    if n == 0:
        return np.empty(0, dtype=np.float64)
    span = 1.0 - (lower / cap) ** alpha
    exponent = 1.0 - 1.0 / alpha
    cumulative = np.arange(n + 1, dtype=np.float64) / n
    edges = 1.0 - cumulative * span
    if abs(exponent) < 1e-12:
        integral = np.log(edges) / span
    else:
        integral = edges**exponent / (exponent * span)
    means = lower * (integral[:-1] - integral[1:]) * n
    return np.clip(means, lower, cap)


def uk_firm_open_band_uniforms(
    turnover_k: Any,
    lower_k: float = OPEN_BAND_LOWER_K,
    upper_k: float = OPEN_BAND_LOG_UNIFORM_UPPER_K,
) -> np.ndarray:
    """The uniforms behind the generator's log-uniform open-band draws."""

    values = np.asarray(turnover_k, dtype=np.float64)
    unit = np.log(values / lower_k) / math.log(upper_k / lower_k)
    return np.clip(unit, 0.0, 1.0)


def draw_uk_firm_open_band_turnover(
    uniforms: Any,
    alpha: float,
    cap_k: float,
    lower_k: float = OPEN_BAND_LOWER_K,
    *,
    strata: Any = None,
) -> np.ndarray:
    """Open-band turnover (£k) from the generator's uniforms.

    Within each stratum (all rows when ``strata`` is None) the rows are
    ranked by ``uniforms`` (ties by position) and the row of rank k among n
    takes the mean of the truncated Pareto(``alpha``) on [``lower_k``,
    ``cap_k``] over the quantile slice [k/n, (k+1)/n]. Values lie in
    [``lower_k``, ``cap_k``], their mean is the distribution's mean, and no
    random number is drawn: only the ranks of ``uniforms`` matter.
    """

    unit = _array(uniforms, "uniforms").astype(np.float64)
    if not np.all(np.isfinite(unit)):
        raise ValueError("Open-band uniforms must be finite.")
    if not (alpha > 0 and cap_k > lower_k > 0):
        raise ValueError("Open-band tail needs alpha > 0 and cap > lower > 0.")
    groups = np.zeros(unit.size, dtype=np.int64)
    if strata is not None:
        groups = np.unique(_array(strata, "strata", unit.size), return_inverse=True)[
            1
        ].reshape(-1)
    result = np.empty(unit.size, dtype=np.float64)
    order = np.lexsort((np.arange(unit.size), unit, groups))
    sorted_groups = groups[order]
    bounds = np.flatnonzero(np.diff(sorted_groups)) + 1
    for rows in np.split(order, bounds):
        result[rows] = _slice_means(rows.size, float(alpha), float(lower_k), cap_k)
    return result


@dataclass(frozen=True)
class UKFirmOpenBandTail:
    """Open-band truncated Pareto parameters and the inputs behind them.

    Turnover in £ thousands, firm counts weighted. ``p_over_10m`` is HMRC's
    count over ONS's; ``model_p_over_10m`` the truncated Pareto's
    ``P(> £10m)`` at (``alpha``, ``cap_k``).
    """

    alpha: float
    cap_k: float
    lower_k: float
    target_mean_k: float
    p_over_10m: float
    model_p_over_10m: float
    hmrc_over_10m_traders: float
    ons_open_band_enterprises: float
    bpe_whole_economy_turnover_k: float
    bpe_unregistered_turnover_k: float
    closed_band_turnover_non_k_k: float
    open_band_firms_non_k: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


def _ons_open_band_enterprises(ons_turnover: pd.DataFrame) -> float:
    if ONS_OPEN_TURNOVER_BAND_LABEL not in ons_turnover.columns:
        raise ValueError("ONS turnover table has no 5000+ column.")
    sic = ons_turnover["SIC Code"]
    rows = ons_turnover[sic.notna() & ~sic.astype(str).isin(["", "Total"])]
    return float(rows[ONS_OPEN_TURNOVER_BAND_LABEL].fillna(0).sum())


def uk_firm_open_band_tail_parameters(
    ons_turnover: pd.DataFrame,
    hmrc_bands: Mapping[str, float],
    source: UKFirmBPESource,
    *,
    sic: Any,
    turnover_k: Any,
    in_frame: Any,
    open_band: Any,
    weights: Any = None,
) -> UKFirmOpenBandTail:
    """Shape, cap and target mean of the open-band truncated Pareto.

    ``ons_turnover`` is the generator's ONS SIC-by-turnover-band table and
    ``hmrc_bands`` its HMRC registered traders by band for the build's
    vintage. ``sic``, ``turnover_k``, ``in_frame``, ``open_band`` and
    ``weights`` (one when omitted) describe the drawn rows; the closed-band
    turnover is that of frame rows outside the open band and Section K.
    Raises ``ValueError`` when the target mean cannot be reached by a cap.
    """

    n = len(_array(turnover_k, "turnover_k"))
    sic_codes = _array(sic, "sic", n)
    values = _array(turnover_k, "turnover_k", n).astype(np.float64)
    frame = _array(in_frame, "in_frame", n).astype(bool)
    open_rows = _array(open_band, "open_band", n).astype(bool)
    mass = (
        np.ones(n, dtype=np.float64)
        if weights is None
        else _array(weights, "weights", n).astype(np.float64)
    )
    non_k = frame & _non_k(sic_codes)
    closed = float((values * mass)[non_k & ~open_rows].sum())
    open_firms = float(mass[non_k & open_rows].sum())
    if not open_firms > 0:
        raise ValueError("No weighted non-Section-K open-band rows.")

    hmrc_top = float(hmrc_bands[HMRC_TOP_TURNOVER_BAND])
    ons_open = _ons_open_band_enterprises(ons_turnover)
    p = hmrc_top / ons_open
    if not 0.0 < p < 1.0:
        raise ValueError(f"P(turnover > 10m | open band) = {p} is not in (0, 1).")
    lower = OPEN_BAND_LOWER_K
    alpha = -math.log(p) / math.log(HMRC_TOP_BAND_LOWER_K / lower)

    whole = _bpe_row(source, _BPE_WHOLE_ECONOMY, _BPE_ALL) * 1_000.0
    unregistered = _bpe_row(source, _BPE_PRIVATE_TABLE_1, _BPE_UNREGISTERED) * 1_000.0
    target = (whole - unregistered - closed) / open_firms
    if not target > lower:
        raise ValueError(f"Open-band target mean {target:,.1f}k is not above {lower}.")
    if alpha > 1.0 and target >= lower * alpha / (alpha - 1.0):
        raise ValueError(
            f"Open-band target mean {target:,.1f}k is beyond any cap at alpha "
            f"{alpha:.4f}."
        )
    log_cap = brentq(
        lambda x: _truncated_pareto_mean(alpha, lower, lower * math.exp(x)) - target,
        1e-9,
        200.0,
        xtol=1e-14,
        rtol=1e-15,
        maxiter=500,
    )
    cap = lower * math.exp(log_cap)
    return UKFirmOpenBandTail(
        alpha=alpha,
        cap_k=cap,
        lower_k=lower,
        target_mean_k=target,
        p_over_10m=p,
        model_p_over_10m=truncated_pareto_exceedance(
            HMRC_TOP_BAND_LOWER_K, alpha, lower, cap
        ),
        hmrc_over_10m_traders=hmrc_top,
        ons_open_band_enterprises=ons_open,
        bpe_whole_economy_turnover_k=whole,
        bpe_unregistered_turnover_k=unregistered,
        closed_band_turnover_non_k_k=closed,
        open_band_firms_non_k=open_firms,
    )


# -- D. diagnostics --------------------------------------------------------------


def uk_firm_turnover_by_employment_band(
    sic: Any,
    turnover_k: Any,
    employment: Any,
    in_frame: Any,
    weights: Any,
    source: UKFirmBPESource,
) -> pd.DataFrame:
    """Model turnover by ONS employment band against BPE, Section K excluded.

    ``weights`` may be base weights (before the solve) or calibrated weights.
    Columns ``band``, ``model_firms``, ``model_turnover_million``,
    ``model_share``, ``bpe_turnover_million``, ``bpe_share``, ``share_gap``.
    """

    n = len(_array(turnover_k, "turnover_k"))
    values = _array(turnover_k, "turnover_k", n).astype(np.float64)
    staff = _array(employment, "employment", n)
    mass = _array(weights, "weights", n).astype(np.float64)
    mask = _array(in_frame, "in_frame", n).astype(bool) & _non_k(_array(sic, "sic", n))
    totals = _band_turnover(staff, values, mass, mask) / 1_000.0
    firms = np.bincount(
        uk_firm_employment_band_index(staff[mask]),
        weights=mass[mask],
        minlength=len(UK_FIRM_TURNOVER_EMPLOYMENT_BANDS),
    )
    bpe = uk_firm_bpe_turnover_by_band(source)
    table = pd.DataFrame(
        {
            "band": UK_FIRM_TURNOVER_EMPLOYMENT_BANDS,
            "model_firms": firms,
            "model_turnover_million": totals,
            "model_share": totals / totals.sum() if totals.sum() > 0 else 0.0,
            "bpe_turnover_million": bpe["bpe_turnover_million"].to_numpy(),
            "bpe_share": bpe["bpe_share"].to_numpy(),
        }
    )
    table["share_gap"] = table["model_share"] - table["bpe_share"]
    return table


def uk_firm_bpe_turnover_pins(
    source: UKFirmBPESource,
    ons_turnover: pd.DataFrame,
    hmrc_bands: Mapping[str, float],
) -> dict[str, dict[str, Any]]:
    """Every source number this module reads, with its unit and source."""

    bpe = "DBT Business population estimates 2025, detailed tables"
    pins: dict[str, dict[str, Any]] = {
        "bpe_whole_economy_turnover": {
            "value": _bpe_row(source, _BPE_WHOLE_ECONOMY, _BPE_ALL),
            "unit": "£ million",
            "source": f"{bpe}, Table 2, whole economy, All (excludes Section K)",
        },
        "bpe_unregistered_turnover": {
            "value": _bpe_row(source, _BPE_PRIVATE_TABLE_1, _BPE_UNREGISTERED),
            "unit": "£ million",
            "source": f"{bpe}, Table 1, private sector, {_BPE_UNREGISTERED}",
        },
        "hmrc_over_10m_traders": {
            "value": float(hmrc_bands[HMRC_TOP_TURNOVER_BAND]),
            "unit": "VAT registered traders",
            "source": f"HMRC VAT population by turnover band, {HMRC_TOP_TURNOVER_BAND}",
        },
        "ons_open_band_enterprises": {
            "value": _ons_open_band_enterprises(ons_turnover),
            "unit": "enterprises",
            "source": "ONS UK business: activity, size and location, turnover "
            "band 5000+, sum over SIC divisions",
        },
    }
    for row in uk_firm_bpe_turnover_by_band(source).itertuples():
        pins[f"bpe_registered_turnover_{row.band}"] = {
            "value": float(row.bpe_turnover_million),
            "unit": "£ million",
            "source": f"{bpe}, Table 2 whole economy ({row.bpe_rows})"
            + (
                ", less Table 1 unregistered"
                if row.unregistered_turnover_million
                else ""
            ),
        }
    return pins
