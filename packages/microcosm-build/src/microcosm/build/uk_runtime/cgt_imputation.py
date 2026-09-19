"""Impute capital gains amounts from HMRC's published joint distribution.

The certified candidate carries a ``capital_gains`` column whose amounts were
drawn from a percentile table that stops near £1m, so the population holds
almost none of the gains above £2m that carry most of the tax. This stage
redraws the amounts of existing gainers from HMRC table 3 — the published
joint distribution of size of gain by taxable income — which reaches the top
bands. See PolicyEngine/microcosm#552 for the measured gap.

The stage owns amounts, not incidence and not weights. Who has gains comes
from the candidate; how much they have comes from the published distribution;
household weights pass through untouched, so the weight kind carries
through and the stage appends a mass-conservation receipt the terminal
family gate requires. Whether the published band facts also become calibration targets is a
separate adjudication (see the fence discussion on the issue), and nothing
here assumes it.

Three documented approximations, in order of consequence:

1. **Taxable income is an arithmetic proxy.** HMRC's rows condition on
   taxable income after reliefs and the Personal Allowance. The population
   carries raw inputs only, so the proxy sums the persisted components of the
   model's ``total_income`` concept (ITA 2007 s.23) and subtracts a tapered
   Personal Allowance, with the allowance parameters read from the
   policyengine-uk parameter tree rather than maintained here.
   ``state_pension_reported`` stands in for ``social_security_income``, whose
   other taxable benefits are not persisted; reliefs such as pension
   contributions and Gift Aid are not deducted. Both push the proxy up or
   down at the margins, which can move a person one income band.
2. **Allocation is rank-preserving within a conditioning cell.** Within
   each income band x age group x region group cell, gainers are ranked by
   their existing gains and the top of the ranking absorbs the cell's raked
   taxpayer mass, highest gain band first. A person is not split across
   bands, so band mass is matched to the granularity of one household
   weight. A cell short of support scales its targets down and releases the
   (gain band, income band) shortfall to the income band's pooled walk over
   its still-unassigned gainers (microcosm#725).
3. **Rounded published values are repaired, not trusted raw.** Counts round
   to the nearest thousand and amounts to the nearest million, and two
   cells of the 2024-25 table imply a mean below their own band. Implied
   means are clamped just inside the violated boundary, keeping the signal
   that the cell's mass sits near that edge; suppressed-count cells are
   allocated the count their own published gains imply at the band mean,
   and every income column is rescaled onto its published total, so the
   allocation reconciles to published numbers by construction.
   The bottom band's support is floored at the annual exempt amount, since
   every allocated person is a taxpayer with a liability.
4. **Gainers beyond the published taxpayer mass keep their existing amounts,
   capped at the annual exempt amount.** Table 3 covers only individuals
   with a CGT liability, so the candidate's remaining gainers are treated as
   sub-AEA gainers rather than being invented into the liability
   distribution or deleted.
5. **The joint is the published 2024-25 surface, not an aged one.** The
   2026 release publishes Table 3 for 2024-25, the tax year of the Table 1
   observations the calibration fits, and its rows are vendored verbatim
   from the pinned Chronicle feed. Table 2.1a's thirteen size bands folded
   onto Table 3's ten must agree with the joint's row totals within
   publication rounding; the summary reports the difference band by band
   and nothing is rescaled.
6. **Age and region margins are raked, not observed jointly.** No table
   publishes gains by age or region crossed with income, so the allocation
   targets are raked from the joint, the Table 6 age counts, the Table 5
   region counts on the individuals basis and the Table 6 gains by age,
   seeded by the frame's own weighted (age, region | income band) shares
   among gainers. The count margins are met by proportional scaling; the
   gains margin is one linear constraint per age group (raked people times
   Table 3 cell means) met by an exponential tilt of the group's cells that
   holds its count, so an age group carries its published gains as well as
   its taxpayers instead of inheriting the frame's band mix. The rake works
   at a coarser grain than the publications (seven age groups, four region
   groups); the summary reports achieved against published at the full
   Table 6 and Table 5 grain.

Only persons with positive existing gains are gainers. The certified
candidate also carries net losses (negative amounts) and zeros; both pass
through byte-identical — Table 3 describes taxpayers with a liability and
says nothing about losses, so the stage neither redraws nor zeroes them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_BUILD_PERIOD,
    HMRC_CGT_CONDITIONING_RECORD_SETS,
    HMRC_CGT_CONDITIONING_RESOURCE,
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRC_CGT_SOURCE_VINTAGE,
    HMRCCapitalGainsJointDistribution,
    HMRCCGTConditioningFacts,
    load_hmrc_cgt_conditioning_facts,
    load_hmrc_cgt_joint_distribution,
)
from microcosm.build.uk_runtime.national_frame import (
    UKNationalStage,
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.calibrate.geography_constants import UK_REGION_TIER_ENUM
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "UK_CGT_IMPUTATION_SEED",
    "UK_CGT_MASS_CONSERVATION_REASON",
    "UK_CGT_SPINE_MASS_CONSERVATION_REASON",
    "UK_CGT_IMPUTATION_STAGE_NAME",
    "UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS",
    "UKCGTImputationSummary",
    "UKCGTPolicyParameters",
    "UK_CGT_AGE_GROUP_LOWER_BOUNDS",
    "UK_CGT_CONDITIONING_DIMENSIONS",
    "UK_CGT_FALLBACK_POLICY",
    "UK_CGT_RAKE_ROUNDS",
    "UK_CGT_RAKE_TOLERANCE",
    "UK_CGT_REGION_GROUP_LABELS",
    "UK_CGT_REGION_GROUPS",
    "UKCGTAllocationReport",
    "impute_uk_capital_gains",
    "impute_uk_capital_gains_with_report",
    "summarize_uk_cgt_imputation",
    "uk_capital_gains_imputation_stage",
    "uk_cgt_spine_stage_transform",
    "uk_cgt_policy_parameters",
    "uk_cgt_taxable_income_proxy",
]

UK_CGT_IMPUTATION_STAGE_NAME = "hmrc_cgt_gains"

#: The reviewed mass-conservation receipt this stage records. The terminal
#: family gate requires a valid mass-conserving MassChangeRecord carrying
#: exactly this reason, so a build whose CGT stage silently moved household
#: mass — or never ran — fails by name.
UK_CGT_MASS_CONSERVATION_REASON = (
    "Amounts-only capital gains redraw: household weights pass through "
    "unchanged and total household mass is conserved."
)

#: Base seed for the stage's draws. Combined with the build period so two
#: periods draw differently while each build is reproducible.
UK_CGT_IMPUTATION_SEED = 552

#: The spine projection records the same conservation invariant under its
#: own reason so the terminal family validator can never satisfy the
#: certified and spine families with one shared record (adversarial-review
#: finding on the E8 PR: reason strings are the receipt identity).
UK_CGT_SPINE_MASS_CONSERVATION_REASON = (
    "Amounts-only capital gains redraw on the source spine: household "
    "weights pass through unchanged and total household mass is conserved."
)


#: Persisted components of the model's ``total_income`` concept (ITA 2007
#: s.23: taxable income after tax reliefs and before allowances).
#: ``state_pension_reported`` stands in for ``social_security_income``; the
#: other taxable benefits inside that concept are not persisted inputs.
UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS: tuple[str, ...] = (
    "employment_income",
    "self_employment_income",
    "state_pension_reported",
    "private_pension_income",
    "property_income",
    "savings_interest_income",
    "dividend_income",
    "miscellaneous_income",
)

#: A cell mean must sit strictly inside its band for a within-band
#: distribution to match it. Published values round to the nearest thousand
#: people and million pounds, and a rounded count of 1 against a rounded
#: amount can imply a mean outside the band — two cells of the 2024-25
#: table do exactly that — so implied means are repaired into the band by
#: this margin (a fraction of band width) rather than trusted raw.
_MEAN_MARGIN = 0.02

#: Below this many people of published mass, a repaired cell is allocated
#: nothing rather than a distribution being fitted to noise.
_MINIMUM_ALLOCATION_PEOPLE = 1.0

#: The draw families require a mean strictly inside the support. Plans are
#: repaired before they reach the solvers, so a violation here signals a
#: caller passing an unrepaired mean, not published data.
_MEAN_POSITION_TOLERANCE = 1e-9

#: Age groups the allocation conditions on (lower bounds, ascending). Every
#: gainer below the second bound falls in the first group; carriers are
#: adults, so the group is 16-24 in practice. Table 6's 16-24 and 25-34
#: bands are separate groups because pooled they left the 16-24 rows at
#: the bottom of every cell walk (two liable rows against 4,000 published
#: taxpayers); the bands from 35 pair up because a few hundred effective
#: carriers cannot populate 9 x 12 x 6 cells. The summary still reports at
#: the published grain.
UK_CGT_AGE_GROUP_LOWER_BOUNDS: tuple[int, ...] = (16, 25, 35, 45, 55, 65, 75)

#: Region groups the allocation conditions on: spine ``region`` enum name
#: -> group label. London and the South East plus East of England hold
#: over half of published gains; the remaining English regions and the
#: three devolved nations are pooled.
UK_CGT_REGION_GROUPS: Mapping[str, str] = MappingProxyType(
    {
        "LONDON": "london",
        "SOUTH_EAST": "south_east_and_east_of_england",
        "EAST_OF_ENGLAND": "south_east_and_east_of_england",
        "NORTH_EAST": "rest_of_england",
        "NORTH_WEST": "rest_of_england",
        "YORKSHIRE": "rest_of_england",
        "EAST_MIDLANDS": "rest_of_england",
        "WEST_MIDLANDS": "rest_of_england",
        "SOUTH_WEST": "rest_of_england",
        "WALES": "wales_scotland_northern_ireland",
        "SCOTLAND": "wales_scotland_northern_ireland",
        "NORTHERN_IRELAND": "wales_scotland_northern_ireland",
    }
)
if set(UK_CGT_REGION_GROUPS) != set(UK_REGION_TIER_ENUM.values()):
    raise RuntimeError("UK_CGT_REGION_GROUPS must cover exactly the region tier.")

#: Group labels in a fixed order, the index the allocation uses.
UK_CGT_REGION_GROUP_LABELS: tuple[str, ...] = tuple(
    dict.fromkeys(UK_CGT_REGION_GROUPS.values())
)

#: Rounds of the generalised rake (the three count margins by proportional
#: scaling, then the gains tilt per age group) and the relative tolerance
#: every attainable margin must meet for it to stop early; feasible margins
#: converge geometrically.
UK_CGT_RAKE_ROUNDS = 500
UK_CGT_RAKE_TOLERANCE = 1e-9

#: The tilt exponent is bounded (cell means are scaled onto [0, 1] by the
#: largest cell mean); a group whose target mean gain lies outside its
#: seeded cells' means saturates at the bound and is reported unattainable.
_TILT_LAMBDA_BOUND = 700.0
_TILT_BISECTION_ITERATIONS = 80

#: How a (gain band, income band) shortfall left by the conditioned cells is
#: met: the income band's still-unassigned gainers are pooled across age and
#: region groups and walked by rank, so the joint is met wherever the band
#: has pooled support instead of being split into allotments below one
#: carrier's weight.
UK_CGT_FALLBACK_POLICY = "pooled_income_band_rank_walk"

#: The dimensions the allocation conditions on, in nesting order.
UK_CGT_CONDITIONING_DIMENSIONS: tuple[str, ...] = (
    "income band",
    "age group",
    "region group",
)


@dataclass(frozen=True)
class UKCGTPolicyParameters:
    """Policy amounts the proxy and the sub-AEA cap depend on.

    Read from the policyengine-uk parameter tree at a stated instant by
    :func:`uk_cgt_policy_parameters`, or constructed directly in tests.
    """

    personal_allowance: float
    personal_allowance_taper_threshold: float
    personal_allowance_taper_rate: float
    annual_exempt_amount: float
    instant: str
    source: str


def uk_cgt_policy_parameters(build_period: int | str) -> UKCGTPolicyParameters:
    """Read the allowance parameters from the policyengine-uk tree.

    Values are read from the raw dated parameter files at 1 June of the tax
    year starting in ``build_period``, so they are the statutory values with
    their legislative references, not the model's fiscal-year snapshots.

    Requires the ``uk`` extra; the import is deferred so the base package
    does not import policyengine-uk at import time.
    """
    try:
        import policyengine_uk
        from policyengine_core.parameters import ParameterNode
    except ImportError as exc:
        raise ImportError(
            "uk_cgt_policy_parameters requires the microcosm-build 'uk' extra "
            "(policyengine-uk)."
        ) from exc

    parameters_dir = Path(policyengine_uk.__file__).parent / "parameters"
    parameters = ParameterNode(directory_path=str(parameters_dir))
    instant = f"{int(build_period)}-06-01"
    allowances = parameters.gov.hmrc.income_tax.allowances
    return UKCGTPolicyParameters(
        personal_allowance=float(allowances.personal_allowance.amount(instant)),
        personal_allowance_taper_threshold=float(
            allowances.personal_allowance.maximum_ANI(instant)
        ),
        personal_allowance_taper_rate=float(
            allowances.personal_allowance.reduction_rate(instant)
        ),
        annual_exempt_amount=float(
            parameters.gov.hmrc.cgt.annual_exempt_amount(instant)
        ),
        instant=instant,
        source="policyengine-uk parameters "
        f"{getattr(policyengine_uk, '__version__', 'unknown')}",
    )


def uk_cgt_taxable_income_proxy(
    person: pd.DataFrame, parameters: UKCGTPolicyParameters
) -> np.ndarray:
    """Approximate taxable income after reliefs and the Personal Allowance."""
    missing = [
        column
        for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS
        if column not in person.columns
    ]
    if missing:
        raise ValueError(
            f"Taxable income proxy components missing from person table: {missing}."
        )
    total = np.zeros(len(person))
    for column in UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS:
        total += pd.to_numeric(person[column], errors="raise").to_numpy(dtype=float)

    taper = parameters.personal_allowance_taper_rate * np.maximum(
        0.0, total - parameters.personal_allowance_taper_threshold
    )
    allowance = np.maximum(0.0, parameters.personal_allowance - taper)
    return np.maximum(0.0, total - allowance)


def _truncated_exponential_quantile(
    quantiles: np.ndarray, lower: float, upper: float, mean: float
) -> np.ndarray:
    """Map uniform quantiles onto [lower, upper) with the given mean.

    A truncated exponential is the one-parameter family on a bounded band
    whose mean can sit anywhere inside it; the rate is solved by bisection,
    and a mean at the midpoint degenerates to the uniform distribution.
    """
    width = upper - lower
    target = (mean - lower) / width
    if not _MEAN_POSITION_TOLERANCE < target < 1 - _MEAN_POSITION_TOLERANCE:
        raise ValueError(f"Band mean {mean} does not sit inside ({lower}, {upper}).")
    if abs(target - 0.5) < 1e-9:
        return lower + quantiles * width

    def normalized_mean(rate: float) -> float:
        # Mean of x ~ TruncExp(rate) on [0, 1]: 1/rate - 1/(exp(rate) - 1),
        # with a removable singularity at zero where the mean is one half.
        if abs(rate) < 1e-9:
            return 0.5
        return 1.0 / rate - 1.0 / np.expm1(rate)

    # normalized_mean is decreasing: rate -> +inf piles mass at 0 and
    # rate -> -inf piles it at 1.
    low_rate, high_rate = -700.0, 700.0
    for _ in range(200):
        mid_rate = (low_rate + high_rate) / 2.0
        if normalized_mean(mid_rate) > target:
            low_rate = mid_rate
        else:
            high_rate = mid_rate
    rate = (low_rate + high_rate) / 2.0
    # Inverse CDF of the truncated exponential on [0, 1].
    positions = -np.log1p(quantiles * (np.exp(-rate) - 1.0)) / rate
    return lower + np.clip(positions, 0.0, 1.0) * width


def _pareto_quantile(quantiles: np.ndarray, lower: float, mean: float) -> np.ndarray:
    """Map uniform quantiles onto [lower, inf) with the given mean.

    The open top band takes a Pareto whose shape is pinned by the published
    mean: alpha = mean / (mean - lower).
    """
    if mean <= lower:
        raise ValueError(
            f"Open-band mean {mean} must exceed the band lower bound {lower}."
        )
    alpha = mean / (mean - lower)
    return lower * np.power(1.0 - quantiles, -1.0 / alpha)


def _pareto_stratum_means(
    lower_quantiles: np.ndarray, upper_quantiles: np.ndarray, lower: float, mean: float
) -> np.ndarray:
    """Conditional mean of the open-band Pareto on each quantile stratum.

    ``E[X | U in [a, b)]`` for ``X = lower (1 - U)^(-1/alpha)`` integrates in
    closed form; the strata partition (0, 1), so the stratum means average
    back to the published mean exactly. A band that reaches only a few
    dozen carriers therefore carries its published mean instead of the
    typical iid draw, which for a Pareto with infinite variance sits well
    below it (most of the mean lives in the top two percent of the
    distribution).
    """
    if mean <= lower:
        raise ValueError(
            f"Open-band mean {mean} must exceed the band lower bound {lower}."
        )
    alpha = mean / (mean - lower)
    power = 1.0 - 1.0 / alpha
    width = upper_quantiles - lower_quantiles
    return (
        lower
        * (
            np.power(1.0 - lower_quantiles, power)
            - np.power(1.0 - upper_quantiles, power)
        )
        / (power * width)
    )


@dataclass(frozen=True)
class _CellPlan:
    """Allocation and draw parameters for one gain band within an income band."""

    gain_lower_bound: int
    gain_upper_bound: float
    effective_lower_bound: float
    allocation_people: float
    mean: float
    mean_is_band_fallback: bool
    mean_repaired: bool


def _repair_mean(
    candidate: float, *, effective_lower: float, upper: float
) -> tuple[float, bool]:
    """Force an implied mean strictly inside its band.

    Rounded published values can imply a mean outside the band (a rounded
    count of 1,000 against a rounded £463m gives £463,000 for the £500k-£1m
    band). The repair clamps toward the violated boundary, keeping the
    signal that the cell's mass sits near that edge.
    """
    if np.isinf(upper):
        floor = effective_lower * (1.0 + _MEAN_MARGIN)
        if candidate < floor:
            return floor, True
        return candidate, False
    width = upper - effective_lower
    low = effective_lower + _MEAN_MARGIN * width
    high = upper - _MEAN_MARGIN * width
    if candidate < low:
        return low, True
    if candidate > high:
        return high, True
    return candidate, False


def _band_plans(
    distribution: HMRCCapitalGainsJointDistribution,
    income_lower_bound: int,
    *,
    annual_exempt_amount: float,
) -> tuple[_CellPlan, ...]:
    """Build the per-gain-band plan for one income band, top band first.

    Suppressed-count cells split the residual between the published income
    column total and its unsuppressed cells, so the allocation reconciles to
    published totals instead of assuming a count. The bottom band's support
    is floored at the annual exempt amount where that fits inside the band,
    since every allocated person is a taxpayer with a liability.
    """
    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = dict(zip(bounds, (*bounds[1:], np.inf), strict=True))

    cells = {
        gain_lower_bound: distribution.cell(
            gain_lower_bound=gain_lower_bound, income_lower_bound=income_lower_bound
        )
        for gain_lower_bound in bounds
    }

    def band_mean(gain_lower_bound: int) -> float:
        band_total = distribution.band_total(gain_lower_bound)
        if band_total.individuals is None:
            raise ValueError(
                "Band total count is suppressed for gains from "
                f"{gain_lower_bound}, leaving no defined mean."
            )
        return band_total.gains / band_total.individuals

    # Raw allocation: published counts where published; for suppressed cells,
    # the count the cell's own published gains imply at the band mean — a
    # cell holding £1.3bn of gains at a £11.4m band mean holds people, and
    # a bare residual can zero it when rounding makes the column's
    # unsuppressed counts alone reach the column total.
    raw_allocation: dict[int, float] = {}
    for gain_lower_bound in bounds:
        cell = cells[gain_lower_bound]
        if cell.individuals is not None:
            raw_allocation[gain_lower_bound] = cell.individuals
        elif cell.gains is not None:
            raw_allocation[gain_lower_bound] = cell.gains / band_mean(gain_lower_bound)
        else:
            raw_allocation[gain_lower_bound] = 0.0

    # Published counts round to the nearest thousand, so a column's cells can
    # sum past its own published total. Rescale the column onto the published
    # total, which makes the allocation reconcile exactly by construction.
    column = distribution.income_total(income_lower_bound)
    if column.individuals is None:
        raise ValueError(
            "Income column total is suppressed for incomes from "
            f"{income_lower_bound}, leaving the allocation nothing to "
            "reconcile against."
        )
    raw_total = sum(raw_allocation.values())
    scale = column.individuals / raw_total if raw_total > 0 else 0.0

    plans = []
    for gain_lower_bound in reversed(bounds):
        cell = cells[gain_lower_bound]
        upper = uppers[gain_lower_bound]
        effective_lower = float(max(gain_lower_bound, 1.0))
        # Every allocated person is a taxpayer with a liability, and
        # liability requires gains strictly above the annual exempt amount,
        # so the bottom band's support starts one pound past it.
        liability_floor = annual_exempt_amount + 1.0
        if liability_floor > effective_lower and (
            np.isinf(upper) or liability_floor < 0.9 * upper
        ):
            effective_lower = float(liability_floor)

        if cell.individuals is not None and cell.gains is not None:
            candidate = cell.gains / cell.individuals
        else:
            candidate = band_mean(gain_lower_bound)
        mean, repaired = _repair_mean(
            candidate, effective_lower=effective_lower, upper=upper
        )

        allocation = raw_allocation[gain_lower_bound] * scale
        if allocation < _MINIMUM_ALLOCATION_PEOPLE:
            allocation = 0.0
        plans.append(
            _CellPlan(
                gain_lower_bound=gain_lower_bound,
                gain_upper_bound=upper,
                effective_lower_bound=effective_lower,
                allocation_people=allocation,
                mean=mean,
                mean_is_band_fallback=False,
                mean_repaired=repaired,
            )
        )
    return tuple(plans)


@dataclass(frozen=True)
class UKCGTAllocationReport:
    """What the rake asked for and what the walk delivered, cell by cell."""

    band_rows: tuple[dict[str, object], ...]
    joint_rows: tuple[dict[str, object], ...]
    rake: Mapping[str, object]
    fallback_released_mass: float
    fallback_share_by_band: Mapping[int, float]
    conditioning: Mapping[str, object]
    rounding_carry_out: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def evidence(self) -> dict[str, object]:
        return {
            "band_rows": [dict(row) for row in self.band_rows],
            "joint_rows": [dict(row) for row in self.joint_rows],
            "rake": dict(self.rake),
            "fallback_released_mass": self.fallback_released_mass,
            "fallback_share_by_band": {
                str(key): value for key, value in self.fallback_share_by_band.items()
            },
            "conditioning": dict(self.conditioning),
            "rounding_carry_out": {
                key: dict(value) for key, value in self.rounding_carry_out.items()
            },
        }


@dataclass(frozen=True)
class UKCGTImputationSummary:
    """Achieved allocation against the published surface, for reporting.

    ``rows`` compares the Table 3 gain bands with the 2024-25 Table 2.1a
    levels folded onto them; ``age_rows`` and ``region_rows`` compare the
    published Table 6 and Table 5 grain (the region rows scaled to the
    individuals basis); ``age_by_band_rows`` is the joint the #725 finding
    was measured on.
    """

    rows: pd.DataFrame
    taxpayer_mass: float
    published_taxpayer_mass: float
    remainder_mass: float
    age_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    region_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    age_by_band_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    allocation: UKCGTAllocationReport | None = None

    def evidence(self) -> dict[str, object]:
        evidence: dict[str, object] = {
            "stage": UK_CGT_IMPUTATION_STAGE_NAME,
            "rows": self.rows.to_dict(orient="records"),
            "taxpayer_mass": self.taxpayer_mass,
            "published_taxpayer_mass": self.published_taxpayer_mass,
            "remainder_mass": self.remainder_mass,
            "age_rows": self.age_rows.to_dict(orient="records"),
            "region_rows": self.region_rows.to_dict(orient="records"),
            "age_by_band_rows": self.age_by_band_rows.to_dict(orient="records"),
        }
        if self.allocation is not None:
            evidence["allocation"] = self.allocation.evidence()
        return evidence


def _person_conditioning_cells(
    person: pd.DataFrame, household: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Age, age group, region and region group for every person row.

    Region is a household column; it reaches persons through the household
    id, and a person without a region or with an unknown region name
    refuses the stage rather than falling into a silent group.
    """

    if "age" not in person.columns:
        raise ValueError("Person table has no age column to condition on.")
    if "region" not in household.columns:
        raise ValueError("Household table has no region column to condition on.")
    age = pd.to_numeric(person["age"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(age).all():
        raise ValueError("Person ages must be finite to condition the redraw.")
    region_by_household = pd.Series(
        household["region"].astype(str).to_numpy(), index=household["household_id"]
    )
    mapped = person["person_household_id"].map(region_by_household)
    if mapped.isna().any():
        raise ValueError(
            "CGT conditioning cannot map every person to a household region."
        )
    region = mapped.astype(str).to_numpy()
    unknown = sorted(set(region) - set(UK_CGT_REGION_GROUPS))
    if unknown:
        raise ValueError(f"Unknown region name(s) for CGT conditioning: {unknown}.")
    age_group = np.digitize(age, UK_CGT_AGE_GROUP_LOWER_BOUNDS[1:])
    label_index = {
        label: index for index, label in enumerate(UK_CGT_REGION_GROUP_LABELS)
    }
    region_group = np.asarray(
        [label_index[UK_CGT_REGION_GROUPS[name]] for name in region], dtype=int
    )
    return age, age_group, region, region_group


def _joint_plans(
    distribution: HMRCCapitalGainsJointDistribution,
    conditioning: HMRCCGTConditioningFacts,
    *,
    annual_exempt_amount: float,
) -> tuple[dict[tuple[int, int], _CellPlan], tuple[dict[str, object], ...]]:
    """Table 3's 2024-25 joint, reconciled column by column, with a receipt.

    Suppression, rounding repair, the liability floor and the column
    reconciliation onto the published All-row totals come from
    :func:`_band_plans` unchanged. The receipt compares each gain band's
    published row total with the Table 2.1a 2024-25 rows folded onto it;
    the two tables describe one universe at one vintage, and the vendored
    resource's loader refuses a pair that disagrees beyond publication
    rounding, so the difference here is a report, not a gate.
    """

    aggregated = conditioning.size_bands_aggregated(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS)
    incomes = HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    joint: dict[tuple[int, int], _CellPlan] = {}
    for income_lower in incomes:
        for plan in _band_plans(
            distribution, income_lower, annual_exempt_amount=annual_exempt_amount
        ):
            joint[(plan.gain_lower_bound, income_lower)] = plan
    band_rows: list[dict[str, object]] = []
    for gain_lower in HMRC_CGT_GAIN_BAND_LOWER_BOUNDS:
        total = distribution.band_total(gain_lower)
        folded_people, folded_gains = aggregated[gain_lower]
        people_difference = (
            None if total.individuals is None else total.individuals - folded_people
        )
        gains_difference = total.gains - folded_gains
        plans = [joint[(gain_lower, income_lower)] for income_lower in incomes]
        band_rows.append(
            {
                "gain_lower_bound": gain_lower,
                "published_people": total.individuals,
                "published_gains": total.gains,
                "table2_1a_people": folded_people,
                "table2_1a_gains": folded_gains,
                "people_difference": people_difference,
                "gains_difference": gains_difference,
                "target_people": sum(plan.allocation_people for plan in plans),
                "target_gains_at_cell_means": sum(
                    plan.allocation_people * plan.mean for plan in plans
                ),
                "means_repaired": sum(int(plan.mean_repaired) for plan in plans),
            }
        )
    return joint, tuple(band_rows)


def _fold_age(conditioning: HMRCCGTConditioningFacts, measure: str) -> np.ndarray:
    """Fold one Table 6 measure onto the age groups.

    Bands below the first group bound (the 0-15 band) are left out: carriers
    are adults, so the frame cannot hold them, and the normalisation onto
    the joint's total spreads their mass across the groups instead of
    folding a thousand children into the 16-24 group.
    """

    margin = np.zeros(len(UK_CGT_AGE_GROUP_LOWER_BOUNDS))
    for band in conditioning.age_bands:
        if band.lower_bound < UK_CGT_AGE_GROUP_LOWER_BOUNDS[0]:
            continue
        group = int(np.digitize(band.lower_bound, UK_CGT_AGE_GROUP_LOWER_BOUNDS[1:]))
        margin[group] += float(getattr(band, measure))
    return margin


def _fold_age_margin(conditioning: HMRCCGTConditioningFacts) -> np.ndarray:
    return _fold_age(conditioning, "taxpayers")


def _fold_age_gains_margin(conditioning: HMRCCGTConditioningFacts) -> np.ndarray:
    return _fold_age(conditioning, "gains")


def _fold_region_margin(conditioning: HMRCCGTConditioningFacts) -> np.ndarray:
    share = conditioning.table1.individuals_share("taxpayers")
    margin = np.zeros(len(UK_CGT_REGION_GROUP_LABELS))
    for row in conditioning.regions:
        group = UK_CGT_REGION_GROUP_LABELS.index(UK_CGT_REGION_GROUPS[row.region])
        margin[group] += row.taxpayers * share
    return margin


def _tilt_group(
    group: np.ndarray, scaled_means: np.ndarray, target_mean: float
) -> tuple[np.ndarray, float, bool]:
    """Tilt one age group's cells onto a target mean gain, holding its count.

    ``group`` is the group's mass over (gain band, income band, region
    group) and ``scaled_means`` the Table 3 cell means over (gain band,
    income band) scaled onto [0, 1]; ``target_mean`` is on the same scale.
    Each cell is multiplied by ``exp(lambda * scaled mean)`` with ``lambda``
    solved by bisection, the exponential-family step of generalised raking:
    the group's people-weighted mean gain is increasing in ``lambda`` and
    runs from the smallest to the largest seeded cell mean, so a target
    inside that range has exactly one solution, and one outside it takes
    the bound and is reported unattainable.
    """

    count = float(group.sum())
    means = np.broadcast_to(scaled_means[:, :, None], group.shape)
    positive = group > 0.0
    low = float(means[positive].min())
    high = float(means[positive].max())

    def tilted(lam: float) -> np.ndarray:
        # Anchor the exponent at the extreme the tilt favours so it never
        # exceeds zero: no overflow at the bound in either direction.
        anchor = high if lam >= 0.0 else low
        return group * np.exp(lam * (means - anchor))

    def tilted_mean(lam: float) -> float:
        weights = tilted(lam)
        return float((weights * means).sum() / weights.sum())

    if not low < target_mean < high:
        lam = -_TILT_LAMBDA_BOUND if target_mean <= low else _TILT_LAMBDA_BOUND
        attainable = False
    else:
        lower_bound, upper_bound = -_TILT_LAMBDA_BOUND, _TILT_LAMBDA_BOUND
        for _ in range(_TILT_BISECTION_ITERATIONS):
            mid = 0.5 * (lower_bound + upper_bound)
            if tilted_mean(mid) < target_mean:
                lower_bound = mid
            else:
                upper_bound = mid
        lam = 0.5 * (lower_bound + upper_bound)
        attainable = True
    weights = tilted(lam)
    return weights * (count / float(weights.sum())), lam, attainable


def _scaling_factor(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Proportional-fit factor: target over current where the current is
    positive, one where nothing can be scaled (a zero-seed category)."""

    factor = np.ones_like(current, dtype=float)
    mask = current > 0.0
    factor[mask] = target[mask] / current[mask]
    return factor


def _rake_error(
    targets: np.ndarray,
    *,
    seed: np.ndarray,
    joint_mass: np.ndarray,
    age_margin: np.ndarray,
    region_margin: np.ndarray,
    gains_margin: np.ndarray,
    cell_means: np.ndarray,
    attainable: np.ndarray,
) -> float:
    """Largest relative error over the seeded, attainable margin categories."""

    errors: list[float] = []

    def add(achieved: np.ndarray, wanted: np.ndarray, seeded: np.ndarray) -> None:
        for have, want, ok in zip(
            achieved.ravel(), wanted.ravel(), seeded.ravel(), strict=True
        ):
            if want > 0.0 and ok:
                errors.append(abs(float(have) - float(want)) / float(want))

    seeded_age = seed.sum(axis=(0, 1, 3)) > 0.0
    add(targets.sum(axis=(2, 3)), joint_mass, seed.sum(axis=(2, 3)) > 0.0)
    add(targets.sum(axis=(0, 1, 3)), age_margin, seeded_age)
    add(targets.sum(axis=(0, 1, 2)), region_margin, seed.sum(axis=(0, 1, 2)) > 0.0)
    add(
        (targets * cell_means[:, :, None, None]).sum(axis=(0, 1, 3)),
        gains_margin,
        seeded_age & attainable,
    )
    return max(errors) if errors else 0.0


def _rake_with_gains_tilt(
    seed: np.ndarray,
    *,
    joint_mass: np.ndarray,
    age_margin: np.ndarray,
    region_margin: np.ndarray,
    gains_margin: np.ndarray,
    cell_means: np.ndarray,
    rounds: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Rake the four-way seed to three count margins and one gains margin.

    Each round scales the age groups and the region groups onto their
    counts, tilts every age group onto its gains (:func:`_tilt_group`), and
    scales the (gain band, income band) joint last. The generic mean-raking helper
    scales columns independently and cannot hold a people-weighted gains
    constraint, which is why the fit lives here. Rounds stop early once
    every seeded, attainable margin holds to ``tolerance``.
    """

    targets = np.asarray(seed, dtype=float).copy()
    n_age = targets.shape[2]
    scale = float(cell_means.max()) if float(cell_means.max()) > 0.0 else 1.0
    scaled_means = cell_means / scale
    target_means = np.zeros(n_age)
    for a in range(n_age):
        if age_margin[a] > 0.0:
            target_means[a] = gains_margin[a] / age_margin[a] / scale
    # The tilt is applied incrementally to the current targets, so each
    # round's lambda shrinks toward zero as the fit converges; the reported
    # lambda is the cumulative exponent the group ended up with.
    lambdas = np.zeros(n_age)
    attainable = np.ones(n_age, dtype=bool)
    seeded_group = seed.sum(axis=(0, 1, 3)) > 0.0
    rounds_used = 0
    error = float("inf")
    for round_index in range(1, rounds + 1):
        rounds_used = round_index
        targets *= _scaling_factor(targets.sum(axis=(0, 1, 3)), age_margin)[
            None, None, :, None
        ]
        targets *= _scaling_factor(targets.sum(axis=(0, 1, 2)), region_margin)[
            None, None, None, :
        ]
        for a in range(n_age):
            if not seeded_group[a] or age_margin[a] <= 0.0 or gains_margin[a] <= 0.0:
                continue
            targets[:, :, a, :], lam, attainable[a] = _tilt_group(
                targets[:, :, a, :], scaled_means, float(target_means[a])
            )
            lambdas[a] += lam
        # The joint goes last: on a feasible system the order is immaterial
        # at convergence, and on a frame that cannot support every age or
        # region group (so the margins conflict) the Table 3 joint, which the
        # cell walks and the pooled fallback are built on, is the one that
        # holds exactly.
        targets *= _scaling_factor(targets.sum(axis=(2, 3)), joint_mass)[
            :, :, None, None
        ]
        error = _rake_error(
            targets,
            seed=seed,
            joint_mass=joint_mass,
            age_margin=age_margin,
            region_margin=region_margin,
            gains_margin=gains_margin,
            cell_means=cell_means,
            attainable=attainable,
        )
        if error < tolerance:
            break
    return targets, {
        "rounds_used": rounds_used,
        "max_error": error,
        "lambdas": lambdas,
        "attainable": attainable,
    }


def _rake_allocation_targets(
    joint: Mapping[tuple[int, int], _CellPlan],
    conditioning: HMRCCGTConditioningFacts,
    *,
    is_gainer: np.ndarray,
    person_weight: np.ndarray,
    income_band: np.ndarray,
    age_group: np.ndarray,
    region_group: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Rake taxpayer targets over gain band x income band x age x region.

    The seed is the joint cell mass spread by the frame's own weighted (age
    group, region group | income band) shares among gainers, so the frame's
    incidence structure is the prior; the margins are the joint itself, the
    Table 6 age counts and the Table 5 region counts on the individuals
    basis, every margin first normalised onto the joint's total so rounding
    cannot make them mutually infeasible. Cells the frame cannot support
    keep a zero seed and are reported, never invented.
    """

    gains = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    incomes = HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    n_gain, n_income = len(gains), len(incomes)
    n_age, n_region = (
        len(UK_CGT_AGE_GROUP_LOWER_BOUNDS),
        len(UK_CGT_REGION_GROUP_LABELS),
    )
    joint_mass = np.zeros((n_gain, n_income))
    for gi, gain_lower in enumerate(gains):
        for ii, income_lower in enumerate(incomes):
            joint_mass[gi, ii] = joint[(gain_lower, income_lower)].allocation_people
    total = float(joint_mass.sum())

    shares = np.zeros((n_income, n_age, n_region))
    for ii, income_lower in enumerate(incomes):
        in_band = is_gainer & (income_band == income_lower)
        if not in_band.any():
            continue
        np.add.at(
            shares[ii],
            (age_group[in_band], region_group[in_band]),
            person_weight[in_band],
        )
        band_mass = shares[ii].sum()
        if band_mass > 0:
            shares[ii] /= band_mass
    seed = joint_mass[:, :, None, None] * shares[None, :, :, :]

    age_margin_raw = _fold_age_margin(conditioning)
    region_margin_raw = _fold_region_margin(conditioning)
    age_margin = (
        age_margin_raw * (total / age_margin_raw.sum())
        if age_margin_raw.sum() > 0
        else age_margin_raw
    )
    region_margin = (
        region_margin_raw * (total / region_margin_raw.sum())
        if region_margin_raw.sum() > 0
        else region_margin_raw
    )

    cell_means = np.zeros((n_gain, n_income))
    for gi, gain_lower in enumerate(gains):
        for ii, income_lower in enumerate(incomes):
            cell_means[gi, ii] = joint[(gain_lower, income_lower)].mean
    gains_margin_raw = _fold_age_gains_margin(conditioning)
    joint_gains = float((joint_mass * cell_means).sum())
    gains_margin = (
        gains_margin_raw * (joint_gains / gains_margin_raw.sum())
        if gains_margin_raw.sum() > 0
        else gains_margin_raw
    )

    targets, fit = _rake_with_gains_tilt(
        seed,
        joint_mass=joint_mass,
        age_margin=age_margin,
        region_margin=region_margin,
        gains_margin=gains_margin,
        cell_means=cell_means,
        rounds=UK_CGT_RAKE_ROUNDS,
        tolerance=UK_CGT_RAKE_TOLERANCE,
    )
    lambdas = np.asarray(fit["lambdas"], dtype=float)
    attainable = np.asarray(fit["attainable"], dtype=bool)

    # A margin category the frame cannot support at all (no gainer in any
    # of its cells) is unattainable however the fit iterates; its mass is
    # reported apart from the fit error over the categories that have seed.
    # A gains margin whose target mean lies outside its seeded cells' means
    # is reported the same way.
    def margin_errors(
        achieved: np.ndarray, target: np.ndarray, seeded: np.ndarray
    ) -> tuple[list[float], float]:
        errors: list[float] = []
        unattainable = 0.0
        for have, want, has_seed in zip(
            achieved.ravel(), target.ravel(), seeded.ravel(), strict=True
        ):
            if want <= 0:
                continue
            if not has_seed:
                unattainable += float(want)
                continue
            errors.append(abs(float(have) - float(want)) / float(want))
        return errors, unattainable

    seeded_joint = seed.sum(axis=(2, 3)) > 0
    seeded_age = seed.sum(axis=(0, 1, 3)) > 0
    seeded_region = seed.sum(axis=(0, 1, 2)) > 0
    joint_errors, joint_unattainable = margin_errors(
        targets.sum(axis=(2, 3)), joint_mass, seeded_joint
    )
    age_errors, age_unattainable = margin_errors(
        targets.sum(axis=(0, 1, 3)), age_margin, seeded_age
    )
    region_errors, region_unattainable = margin_errors(
        targets.sum(axis=(0, 1, 2)), region_margin, seeded_region
    )
    raked_people = targets.sum(axis=(0, 1, 3))
    raked_gains = (targets * cell_means[:, :, None, None]).sum(axis=(0, 1, 3))
    gains_errors, gains_unattainable = margin_errors(
        raked_gains, gains_margin, seeded_age & attainable
    )
    errors = [*joint_errors, *age_errors, *region_errors]
    zero_seed_cells = int(
        (~seeded_joint & (joint_mass > 0)).sum()
        + (~seeded_age & (age_margin > 0)).sum()
        + (~seeded_region & (region_margin > 0)).sum()
    )
    report: dict[str, object] = {
        "rake_rounds": UK_CGT_RAKE_ROUNDS,
        "rake_rounds_used": int(fit["rounds_used"]),
        "rake_tolerance": UK_CGT_RAKE_TOLERANCE,
        "ipf_max_abs_margin_error": max(errors) if errors else 0.0,
        "gains_margin_max_abs_error": max(gains_errors) if gains_errors else 0.0,
        "ipf_unattainable_margin_mass": {
            "joint": joint_unattainable,
            "age": age_unattainable,
            "region": region_unattainable,
            "gains": gains_unattainable,
        },
        "ipf_zero_seed_cells": zero_seed_cells,
        "joint_total_people": total,
        "joint_total_gains_at_cell_means": joint_gains,
        "age_margin": {
            str(UK_CGT_AGE_GROUP_LOWER_BOUNDS[a]): {
                "published": float(age_margin_raw[a]),
                "normalised": float(age_margin[a]),
                "raked": float(raked_people[a]),
            }
            for a in range(n_age)
        },
        "gains_margin": {
            str(UK_CGT_AGE_GROUP_LOWER_BOUNDS[a]): {
                "published": float(gains_margin_raw[a]),
                "normalised": float(gains_margin[a]),
                "raked": float(raked_gains[a]),
                "target_mean": (
                    float(gains_margin[a] / age_margin[a]) if age_margin[a] > 0 else 0.0
                ),
                "raked_mean": (
                    float(raked_gains[a] / raked_people[a])
                    if raked_people[a] > 0
                    else 0.0
                ),
                "tilt_lambda": float(lambdas[a]),
                "attainable": bool(attainable[a]),
            }
            for a in range(n_age)
        },
        "region_margin": {
            UK_CGT_REGION_GROUP_LABELS[r]: {
                "published_individuals_basis": float(region_margin_raw[r]),
                "normalised": float(region_margin[r]),
                "raked": float(targets.sum(axis=(0, 1, 2))[r]),
            }
            for r in range(n_region)
        },
        "income_bands_without_gainers": [
            int(income_lower)
            for ii, income_lower in enumerate(incomes)
            if shares[ii].sum() <= 0
        ],
    }
    return targets, report


def _walk_bands(
    *,
    ranked: np.ndarray,
    weights: np.ndarray,
    boundaries: Sequence[tuple[int, float]],
    rng: np.random.Generator,
    members: dict[int, list[np.ndarray]],
) -> tuple[np.ndarray, dict[int, float]]:
    """Assign ranked gainers to bands, highest first, and draw their amounts.

    ``boundaries`` lists ``(gain band lower bound, people mass)`` from the
    top band down; a person joins the band whose cumulative boundary first
    covers ``(1 - offset)`` of their weight, with one seeded offset per walk,
    so no person splits across bands and the rounding is systematic: a
    band whose boundary is smaller than a person's weight still receives
    that person on the share of walks its boundary implies, rather than
    never (the midpoint rule's bias, which emptied the top bands inside
    the conditioning cells). The persons each band receives are appended to
    ``members`` (keyed by band lower bound); the amounts are drawn once every
    walk is done, so the quantiles can be stratified across a plan. Returns
    the mask of assigned positions in ``ranked`` and the mass each band
    received.
    """

    cumulative = np.cumsum(weights)
    offset = float(rng.random())
    boundary = 0.0
    assigned = np.zeros(len(ranked), dtype=bool)
    achieved: dict[int, float] = {}
    for gain_lower, mass in boundaries:
        boundary += mass
        in_cell = ~assigned & (cumulative - (1.0 - offset) * weights <= boundary)
        count = int(in_cell.sum())
        if count == 0:
            continue
        assigned |= in_cell
        members.setdefault(gain_lower, []).append(ranked[in_cell])
        achieved[gain_lower] = float(weights[in_cell].sum())
    return assigned, achieved


def _draw_plan_amounts(
    plan: _CellPlan,
    members: np.ndarray,
    *,
    person_id: np.ndarray,
    existing: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    new_gains: np.ndarray,
) -> None:
    """Draw one plan's amounts on weight-proportional quantile strata.

    The persons a (gain band, income band) plan received across every cell
    take, in ascending prior-gain order, the quantile strata that partition
    (0, 1) in proportion to their household weights, with one seeded jitter
    per plan inside the stratum. The published mean is a people-weighted
    mean, so the strata must be too: equal strata on unequal weights put the
    tail's mass on whichever rows happen to be light and the plan's weighted
    mean drifts (the GBP 5m+ band realised GBP 41bn of GBP 48bn on rows a
    third of whose weight sat in the bottom strata). The open band takes each
    stratum's conditional mean, so its weighted mean is the published mean
    exactly; a bounded band takes the quantile inside each stratum, which
    reproduces the mean to the width of the widest stratum.
    """

    if members.size == 0:
        return
    ordered = _ranked(members, person_id=person_id, existing=existing)[::-1]
    cumulative = np.cumsum(weights[ordered])
    total = float(cumulative[-1])
    if not total > 0.0:
        raise ValueError("A CGT plan received members with no positive weight.")
    upper_quantiles = cumulative / total
    lower_quantiles = np.concatenate(([0.0], upper_quantiles[:-1]))
    jitter = float(rng.random())
    lower = plan.effective_lower_bound
    if np.isinf(plan.gain_upper_bound):
        amounts = _pareto_stratum_means(
            lower_quantiles, upper_quantiles, lower, plan.mean
        )
    else:
        amounts = _truncated_exponential_quantile(
            lower_quantiles + jitter * (upper_quantiles - lower_quantiles),
            lower,
            float(plan.gain_upper_bound),
            plan.mean,
        )
    new_gains[ordered] = amounts


def _ranked(
    indices: np.ndarray, *, person_id: np.ndarray, existing: np.ndarray
) -> np.ndarray:
    # Rank by existing gains, largest first; person_id breaks ties so the
    # ordering, and with it every draw, is deterministic.
    order = np.lexsort((person_id[indices], -existing[indices]))
    return indices[order]


def impute_uk_capital_gains_with_report(
    frame: Frame,
    distribution: HMRCCapitalGainsJointDistribution,
    parameters: UKCGTPolicyParameters,
    *,
    conditioning: HMRCCGTConditioningFacts,
    seed: int = UK_CGT_IMPUTATION_SEED,
    mass_change_reason: str = UK_CGT_MASS_CONSERVATION_REASON,
) -> tuple[Frame, UKCGTAllocationReport]:
    """Redraw gainers' amounts, conditioned on income, age and region."""

    validate_uk_national_frame(frame)
    time_period = uk_time_period(frame)
    person = frame.table("person").reset_index(drop=True)
    if "capital_gains" not in person.columns:
        raise ValueError("Person table has no capital_gains column to redraw.")

    household = frame.table("household")
    weights_by_household = pd.Series(
        frame.weights_for("household").values,
        index=household["household_id"],
    )
    missing_households = set(person["person_household_id"]) - set(
        weights_by_household.index
    )
    if missing_households:
        raise ValueError(
            "Person rows reference households with no weight: "
            f"{sorted(missing_households)[:5]}."
        )
    person_weight = (
        person["person_household_id"].map(weights_by_household).to_numpy(dtype=float)
    )
    person_id = person["person_id"].to_numpy()

    existing = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(
        dtype=float
    )
    taxable_income = uk_cgt_taxable_income_proxy(person, parameters)
    income_band = np.asarray(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS)[
        np.digitize(taxable_income, HMRC_CGT_INCOME_BAND_LOWER_BOUNDS[1:])
    ]
    _, age_group, _, region_group = _person_conditioning_cells(person, household)

    rng = np.random.default_rng((seed, int(time_period)))
    new_gains = existing.copy()
    is_gainer = existing > 0
    assigned = np.zeros(len(person), dtype=bool)

    joint, band_rows = _joint_plans(
        distribution, conditioning, annual_exempt_amount=parameters.annual_exempt_amount
    )
    targets, rake_report = _rake_allocation_targets(
        joint,
        conditioning,
        is_gainer=is_gainer,
        person_weight=person_weight,
        income_band=income_band,
        age_group=age_group,
        region_group=region_group,
    )

    gains = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    incomes = HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
    achieved_pass1 = np.zeros(targets.shape)
    achieved_fallback = np.zeros((len(gains), len(incomes)))
    members: dict[tuple[int, int], list[np.ndarray]] = {}

    # Pass 1: every (income, age, region) cell walks its raked targets.
    # Each walk rounds to whole persons, so a cell's bands land up to one
    # person's weight off their allotments. The signed rounding error is
    # carried into the next cell's boundaries for the same gain band (error
    # diffusion), so each gain band's walked total within the income band
    # tracks its raked total within one weight instead of accumulating the
    # truncation of dozens of cell walks, which only ever compounds upward
    # once the pooled walk below fills shortfalls and cannot unassign an
    # overshoot. Only rounding
    # is carried: a cell short of support scales its boundaries down and
    # that structural shortfall goes to the pooled walk as before.
    rounding_carry_out = np.zeros((len(gains), len(incomes)))
    for ii, income_lower in enumerate(incomes):
        carry = np.zeros(len(gains))
        for a in range(len(UK_CGT_AGE_GROUP_LOWER_BOUNDS)):
            for r in range(len(UK_CGT_REGION_GROUP_LABELS)):
                in_cell = (
                    is_gainer
                    & (income_band == income_lower)
                    & (age_group == a)
                    & (region_group == r)
                )
                cell_targets = targets[:, ii, a, r]
                if not in_cell.any() or float(cell_targets.sum()) <= 0.0:
                    continue
                adjusted = cell_targets - carry
                walk_mass = np.maximum(adjusted, 0.0)
                # An overshoot larger than this cell's allotment for a band
                # walks nothing here and is deferred to the next cell.
                deferred = np.minimum(adjusted, 0.0)
                total_target = float(walk_mass.sum())
                if total_target <= 0.0:
                    carry = -deferred
                    continue
                ranked = _ranked(
                    np.flatnonzero(in_cell), person_id=person_id, existing=existing
                )
                weights = person_weight[ranked]
                # When the cell holds less gainer mass than its target,
                # allocate what exists in the raked proportions; the
                # shortfall goes to the income band's pooled walk.
                scale = min(1.0, float(weights.sum()) / total_target)
                boundary_mass = walk_mass * scale
                boundaries = [
                    (gain_lower, float(boundary_mass[gi]))
                    for gi, gain_lower in reversed(list(enumerate(gains)))
                ]
                cell_members: dict[int, list[np.ndarray]] = {}
                cell_assigned, achieved = _walk_bands(
                    ranked=ranked,
                    weights=weights,
                    boundaries=boundaries,
                    rng=rng,
                    members=cell_members,
                )
                for gain_lower, chunks in cell_members.items():
                    members.setdefault((gain_lower, income_lower), []).extend(chunks)
                assigned[ranked[cell_assigned]] = True
                achieved_cell = np.asarray(
                    [achieved.get(gain_lower, 0.0) for gain_lower in gains]
                )
                achieved_pass1[:, ii, a, r] = achieved_cell
                carry = (achieved_cell - boundary_mass) - deferred
        rounding_carry_out[:, ii] = carry

    # Pass 2: the joint's shortfall is walked over each income band's
    # pooled unassigned gainers, largest first. Every cell walk rounds to
    # whole persons, so some (band, cell) allotments overshoot by up to half
    # a weight while others fall short; the pooled walk fills only the
    # income band's net shortfall, apportioned to bands in proportion to
    # their positive shortfalls, so the rounding of a hundred cell walks is
    # not compounded into extra taxpayers.
    for ii, income_lower in enumerate(incomes):
        signed_shortfall = {
            gain_lower: joint[(gain_lower, income_lower)].allocation_people
            - float(achieved_pass1[gi, ii].sum())
            for gi, gain_lower in enumerate(gains)
        }
        net_shortfall = sum(signed_shortfall.values())
        positive = {
            gain_lower: max(0.0, value)
            for gain_lower, value in signed_shortfall.items()
        }
        positive_total = sum(positive.values())
        pool = is_gainer & (income_band == income_lower) & ~assigned
        if net_shortfall <= 0.0 or positive_total <= 0.0 or not pool.any():
            continue
        shortfall = {
            gain_lower: value * net_shortfall / positive_total
            for gain_lower, value in positive.items()
        }
        ranked = _ranked(np.flatnonzero(pool), person_id=person_id, existing=existing)
        weights = person_weight[ranked]
        scale = min(1.0, float(weights.sum()) / net_shortfall)
        boundaries = [
            (gain_lower, shortfall[gain_lower] * scale)
            for gain_lower in reversed(gains)
        ]
        pool_members: dict[int, list[np.ndarray]] = {}
        pool_assigned, achieved = _walk_bands(
            ranked=ranked,
            weights=weights,
            boundaries=boundaries,
            rng=rng,
            members=pool_members,
        )
        for gain_lower, chunks in pool_members.items():
            members.setdefault((gain_lower, income_lower), []).extend(chunks)
        assigned[ranked[pool_assigned]] = True
        for gi, gain_lower in enumerate(gains):
            achieved_fallback[gi, ii] = achieved.get(gain_lower, 0.0)

    # Amounts: one stratified draw per (gain band, income band) plan over
    # every person it received in either pass, income bands ascending and
    # gain bands highest first.
    for income_lower in incomes:
        for gain_lower in reversed(gains):
            chunks = members.get((gain_lower, income_lower), [])
            if not chunks:
                continue
            _draw_plan_amounts(
                joint[(gain_lower, income_lower)],
                np.concatenate(chunks),
                person_id=person_id,
                existing=existing,
                weights=person_weight,
                rng=rng,
                new_gains=new_gains,
            )

    # Below the published taxpayer mass: sub-AEA gainers keep their
    # existing amounts, capped at the annual exempt amount.
    remainder = is_gainer & ~assigned
    new_gains[remainder] = np.minimum(
        existing[remainder], parameters.annual_exempt_amount
    )

    if not np.isfinite(new_gains).all():
        raise ValueError("Imputed capital gains contain non-finite values.")
    # Only gainers are redrawn; loss-makers and zero-gain persons pass
    # through byte-identical. The certified candidate carries net losses
    # (negative amounts), and Table 3 says nothing about them — a blanket
    # non-negativity guard here would reject every real build.
    if (new_gains[is_gainer] < 0).any():
        raise ValueError("Redrawn capital gains contain negative values.")
    if (new_gains[~is_gainer] != existing[~is_gainer]).any():
        raise ValueError("Non-gainer capital gains were modified by the redraw.")

    joint_rows = []
    pass1_by_band = achieved_pass1.sum(axis=(2, 3))
    for gi, gain_lower in enumerate(gains):
        for ii, income_lower in enumerate(incomes):
            plan = joint[(gain_lower, income_lower)]
            joint_rows.append(
                {
                    "gain_lower_bound": gain_lower,
                    "income_lower_bound": income_lower,
                    "target_people": plan.allocation_people,
                    "achieved_pass1": float(pass1_by_band[gi, ii]),
                    "achieved_fallback": float(achieved_fallback[gi, ii]),
                    "residual": plan.allocation_people
                    - float(pass1_by_band[gi, ii])
                    - float(achieved_fallback[gi, ii]),
                    "mean": plan.mean,
                    "mean_repaired": plan.mean_repaired,
                }
            )
    band_mass = pass1_by_band.sum(axis=1) + achieved_fallback.sum(axis=1)
    fallback_share = {
        gain_lower: (
            float(achieved_fallback[gi].sum() / band_mass[gi])
            if band_mass[gi] > 0
            else 0.0
        )
        for gi, gain_lower in enumerate(gains)
    }
    report = UKCGTAllocationReport(
        band_rows=band_rows,
        joint_rows=tuple(joint_rows),
        rake=rake_report,
        fallback_released_mass=float(achieved_fallback.sum()),
        fallback_share_by_band=fallback_share,
        rounding_carry_out={
            str(income_lower): {
                str(gain_lower): float(rounding_carry_out[gi, ii])
                for gi, gain_lower in enumerate(gains)
            }
            for ii, income_lower in enumerate(incomes)
        },
        conditioning={
            "resource": conditioning.resource,
            "resource_sha256": conditioning.resource_sha256,
            "source_commit": conditioning.source_commit,
            "vintage_tax_year": conditioning.tax_year,
            "joint_resource": distribution.source.resource,
            "joint_resource_sha256": distribution.source.resource_sha256,
            "joint_record_set_prefix": distribution.source.record_set_prefix,
            "joint_source_file": distribution.source.source_file,
            "joint_source_sha256": distribution.source.source_sha256,
            "joint_vintage": distribution.source.source_vintage,
            "age_group_lower_bounds": list(UK_CGT_AGE_GROUP_LOWER_BOUNDS),
            "region_groups": dict(UK_CGT_REGION_GROUPS),
            "fallback_policy": UK_CGT_FALLBACK_POLICY,
        },
    )

    new_person = person.copy()
    new_person["capital_gains"] = new_gains
    # Person-only replacement: mass is untouched and the kind carries
    # through; the appended record is a conservation receipt, not a change —
    # the terminal family gate requires it, so a build whose CGT stage moved
    # mass or never ran fails by name.
    weights = frame.weights_for("household")
    household_mass = float(weights.total)
    receipt = MassChangeRecord(
        entity="household",
        old_total=household_mass,
        new_total=household_mass,
        declared_factor=1.0,
        reason=mass_change_reason,
    )
    result_frame = uk_national_frame(
        person=new_person,
        benunit=frame.table("benunit"),
        household=frame.table("household"),
        time_period=time_period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=weights.values,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result_frame)
    return result_frame, report


def impute_uk_capital_gains(
    frame: Frame,
    distribution: HMRCCapitalGainsJointDistribution,
    parameters: UKCGTPolicyParameters,
    *,
    conditioning: HMRCCGTConditioningFacts | None = None,
    seed: int = UK_CGT_IMPUTATION_SEED,
    mass_change_reason: str = UK_CGT_MASS_CONSERVATION_REASON,
) -> Frame:
    """Redraw gainers' amounts from the published joint distribution.

    ``conditioning`` defaults to the committed vendored 2024-25 resource.
    """

    result, _ = impute_uk_capital_gains_with_report(
        frame,
        distribution,
        parameters,
        conditioning=conditioning or load_hmrc_cgt_conditioning_facts(),
        seed=seed,
        mass_change_reason=mass_change_reason,
    )
    return result


def summarize_uk_cgt_imputation(
    before: Frame,
    after: Frame,
    distribution: HMRCCapitalGainsJointDistribution,
    parameters: UKCGTPolicyParameters,
    *,
    conditioning: HMRCCGTConditioningFacts | None = None,
    report: UKCGTAllocationReport | None = None,
) -> UKCGTImputationSummary:
    """Compare achieved totals with the published 2024-25 surface.

    Reporting, not a gate: where the population holds less gainer mass than
    HMRC's taxpayers the achieved totals sit below the published ones by
    construction, and holding levels to the published surface is the
    calibration adjudication's question. Bands compare with Table 2.1a
    folded onto Table 3's bands, ages with Table 6, regions with Table 5 on
    the individuals basis.
    """

    del distribution  # the 2024-25 levels come from the conditioning facts
    conditioning = conditioning or load_hmrc_cgt_conditioning_facts()
    person = after.table("person").reset_index(drop=True)
    household = after.table("household")
    weights_by_household = pd.Series(
        after.weights_for("household").values,
        index=household["household_id"],
    )
    weight = (
        person["person_household_id"].map(weights_by_household).to_numpy(dtype=float)
    )
    gains = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(dtype=float)
    liable = gains > parameters.annual_exempt_amount
    age, _, region, _ = _person_conditioning_cells(person, household)

    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = (*bounds[1:], np.inf)
    published = conditioning.size_bands_aggregated(bounds)
    rows = []
    for lower, upper in zip(bounds, uppers, strict=True):
        in_band = liable & (gains >= lower) & (gains < upper)
        rows.append(
            {
                "gain_lower_bound": lower,
                "achieved_people": float(weight[in_band].sum()),
                "published_people": published[lower][0],
                "achieved_gains": float((gains[in_band] * weight[in_band]).sum()),
                "published_gains": published[lower][1],
            }
        )

    age_rows = []
    age_by_band_rows = []
    for band in conditioning.age_bands:
        upper = np.inf if band.upper_bound is None else band.upper_bound
        in_age = liable & (age >= band.lower_bound) & (age < upper)
        age_rows.append(
            {
                "age_lower_bound": band.lower_bound,
                "achieved_people": float(weight[in_age].sum()),
                "published_people": band.taxpayers,
                "achieved_gains": float((gains[in_age] * weight[in_age]).sum()),
                "published_gains": band.gains,
            }
        )
        for lower, gain_upper in zip(bounds, uppers, strict=True):
            in_cell = in_age & (gains >= lower) & (gains < gain_upper)
            age_by_band_rows.append(
                {
                    "age_lower_bound": band.lower_bound,
                    "gain_lower_bound": lower,
                    "achieved_people": float(weight[in_cell].sum()),
                    "achieved_gains": float((gains[in_cell] * weight[in_cell]).sum()),
                }
            )

    taxpayer_share = conditioning.table1.individuals_share("taxpayers")
    gains_share = conditioning.table1.individuals_share("gains")
    region_rows = []
    for row in conditioning.regions:
        in_region = liable & (region == row.region)
        region_rows.append(
            {
                "region": row.region,
                "geography_id": row.geography_id,
                "achieved_people": float(weight[in_region].sum()),
                "published_people_individuals_basis": row.taxpayers * taxpayer_share,
                "achieved_gains": float((gains[in_region] * weight[in_region]).sum()),
                "published_gains_individuals_basis": row.gains * gains_share,
            }
        )

    return UKCGTImputationSummary(
        rows=pd.DataFrame(rows),
        taxpayer_mass=float(weight[liable].sum()),
        published_taxpayer_mass=float(conditioning.table1.individuals_taxpayers),
        remainder_mass=float(weight[(gains > 0) & ~liable].sum()),
        age_rows=pd.DataFrame(age_rows),
        region_rows=pd.DataFrame(region_rows),
        age_by_band_rows=pd.DataFrame(age_by_band_rows),
        allocation=report,
    )


def uk_capital_gains_imputation_stage(
    *,
    parameters: UKCGTPolicyParameters | None = None,
    distribution: HMRCCapitalGainsJointDistribution | None = None,
    conditioning: HMRCCGTConditioningFacts | None = None,
    seed: int = UK_CGT_IMPUTATION_SEED,
    mass_change_reason: str = UK_CGT_MASS_CONSERVATION_REASON,
) -> UKNationalStage:
    """Build the national stage that redraws capital gains amounts.

    The joint and the conditioning facts default to the committed vendored
    2024-25 resource, which is checked against the pinned Chronicle feed
    before a row is read. Parameters default to the policyengine-uk tree at
    the dataset's build period, resolved when the stage runs.
    """

    def transform(frame: Frame) -> Frame:
        joint = distribution or load_hmrc_cgt_joint_distribution()
        resolved = parameters or uk_cgt_policy_parameters(uk_time_period(frame))
        return impute_uk_capital_gains(
            frame,
            joint,
            resolved,
            conditioning=conditioning or load_hmrc_cgt_conditioning_facts(),
            seed=seed,
            mass_change_reason=mass_change_reason,
        )

    return UKNationalStage(name=UK_CGT_IMPUTATION_STAGE_NAME, transform=transform)


def uk_cgt_spine_stage_transform(
    stage: SourceStageSpec,
    *,
    distribution: HMRCCapitalGainsJointDistribution | None = None,
    parameters: UKCGTPolicyParameters | None = None,
    conditioning: HMRCCGTConditioningFacts | None = None,
):
    """Bind the spine manifest, then reuse the reviewed merged CGT runtime.

    The certified-H5 wrapper and its candidate verification remain untouched;
    this source-plan seam deliberately delegates only the amounts transform.
    """

    _assert_cgt_spine_stage_parameters(stage)
    return UKCGTSpineStageTransform(
        stage=stage,
        distribution=distribution,
        parameters=parameters,
        conditioning=conditioning,
    )


@dataclass(frozen=True)
class UKCGTSpineStageTransform:
    """Source-plan CGT amounts redraw with a stage-time summary receipt."""

    stage: SourceStageSpec
    distribution: HMRCCapitalGainsJointDistribution | None = None
    parameters: UKCGTPolicyParameters | None = None
    conditioning: HMRCCGTConditioningFacts | None = None
    last_result: UKCGTImputationSummary | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_cgt_spine_stage_parameters(self.stage)
        distribution = self.distribution
        if distribution is None:
            distribution = load_hmrc_cgt_joint_distribution()
        parameters = self.parameters
        if parameters is None:
            parameters = uk_cgt_policy_parameters(uk_time_period(frame))
        conditioning = self.conditioning
        if conditioning is None:
            conditioning = load_hmrc_cgt_conditioning_facts()
        result, report = impute_uk_capital_gains_with_report(
            frame,
            distribution,
            parameters,
            conditioning=conditioning,
            seed=UK_CGT_IMPUTATION_SEED,
            mass_change_reason=UK_CGT_SPINE_MASS_CONSERVATION_REASON,
        )
        summary = summarize_uk_cgt_imputation(
            frame,
            result,
            distribution,
            parameters,
            conditioning=conditioning,
            report=report,
        )
        object.__setattr__(self, "last_result", summary)
        return result

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        evidence = self.last_result.evidence()
        # The shared summary stamps the certified family's stage name; this
        # receipt belongs to the spine stage that produced it (the E8
        # distinct-receipts-per-family rule), and the stage-health gate
        # rightly refuses a receipt claiming another stage.
        evidence["stage"] = self.stage.stage
        return {"evidence": evidence}


def _assert_cgt_spine_stage_parameters(stage: SourceStageSpec) -> None:
    """Arm 1 of the #730/#684 two-arm rule for the spine projection."""

    expected_kinds = (
        "verify_vendored_fact_resource",
        "taxable_income_proxy",
        "rake_allocation_targets",
        "rank_preserving_allocation",
        "within_band_draws",
        "sub_aea_remainder",
        "record_mass_conservation_receipt",
        "classify_cgt_band_facts_with_reviewed_fence",
    )
    kinds = tuple(operation.kind for operation in stage.operations)
    if kinds != expected_kinds:
        raise ValueError(
            f"CGT spine operation order drifted: expected {expected_kinds}, got {kinds}."
        )
    operations = {
        operation.kind: dict(operation.parameters) for operation in stage.operations
    }
    # Closed-world reviewed mapping: every operation's FULL declared parameter
    # payload must equal the reviewed constants below (adversarial-review
    # finding on #740 — asserting a subset let lockstep manifest edits move
    # behavioral declarations without a matching reviewed code change; whole-
    # mapping equality also rejects extra keys).
    expected_operations = {
        "verify_vendored_fact_resource": {
            "artifact_role": "cgt_conditioning_facts",
            "resource": HMRC_CGT_CONDITIONING_RESOURCE,
            "feed_pin": "chronicle_feed.json",
            "record_sets": list(HMRC_CGT_CONDITIONING_RECORD_SETS),
            "source_vintage": HMRC_CGT_SOURCE_VINTAGE,
            "mapped_build_period": int(HMRC_CGT_BUILD_PERIOD),
            "period_mapping": "published_tax_year_equals_build_period",
            "require_before_source_read": True,
            "runtime_sha256_required": True,
            "fail_on_mismatch": True,
        },
        "taxable_income_proxy": {
            "components": list(UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS),
            "components_semantics": (
                "Persisted leaves of the model's total_income concept (ITA "
                "2007 s.23); state_pension_reported stands in for "
                "social_security_income, whose other taxable benefits are "
                "not persisted; reliefs such as pension contributions and "
                "Gift Aid are not deducted."
            ),
            "allowance": (
                "tapered Personal Allowance from the policy_parameters artifact"
            ),
            "fail_on_missing_component": True,
        },
        "rake_allocation_targets": {
            "resource": "hmrc_cgt_conditioning_facts.json",
            "joint_margin": (
                "Table 3 2024-25 reconciled cells (2026 release; individuals with "
                "a liability by size of gain and taxable income), every income "
                "column rescaled onto its published All-row taxpayer total; the "
                "same tax year as the Table 1 observations the calibration fits, "
                "so no vintage move is applied"
            ),
            "band_aggregation": (
                "Table 2.1a bands 0, 3,000 and 6,000 fold into Table 3 band 0; "
                "10,000 and 12,300 fold into 10,000; the remaining bands map one "
                "to one; the folded rows must agree with Table 3's row totals "
                "within publication rounding and the difference is reported per "
                "band"
            ),
            "age_margin": (
                "Table 6 2024-25 individual taxpayers by age band, folded to the "
                "age groups; the 0-15 band, which adult carriers cannot hold, is "
                "spread across the groups by the normalisation rather than folded "
                "into the youngest group"
            ),
            "gains_margin": (
                "Table 6 2024-25 individual gains by age band, folded to the age "
                "groups the same way, as one linear constraint per group, the "
                "raked people in the group's cells times their Table 3 cell means "
                "summing to the group's gains, so each age group carries its "
                "published gains as well as its count instead of the frame's band "
                "mix (the amounts-consistent tilt; microcosm#725)"
            ),
            "region_margin": (
                "Table 5 2024-25 taxpayers by country and region (all taxpayers) "
                "times the Table 1 2024-25 individuals/total taxpayer share, "
                "folded to the region groups"
            ),
            "margin_normalization": (
                "every count margin is rescaled onto the joint's total taxpayer "
                "mass and the gains margin onto the joint's gains at cell means "
                "before raking, so rounding cannot make the margins mutually "
                "infeasible"
            ),
            "seed_joint": (
                "joint cell mass times the frame's household-weighted (age "
                "group, region group | income band) shares among gainers"
            ),
            "age_group_lower_bounds": list(UK_CGT_AGE_GROUP_LOWER_BOUNDS),
            "region_groups": dict(UK_CGT_REGION_GROUPS),
            "gains_tilt": (
                "exponential tilt per age group; the group's cells are multiplied "
                "by exp(lambda x cell mean / largest cell mean) with lambda solved "
                "by bisection so the group's people-weighted mean gain meets its "
                "target while the group's count is held, alternated with the "
                "three count margins until every margin holds to the tolerance"
            ),
            "solver": (
                "microcosm.build.uk_runtime.cgt_imputation._rake_with_gains_tilt "
                "(proportional scaling of the count margins and the gains tilt on "
                "the four-way array; the generic mean-raking helper scales columns "
                "independently and cannot hold a people-weighted gains constraint)"
            ),
            "rake_rounds": UK_CGT_RAKE_ROUNDS,
            "rake_tolerance": UK_CGT_RAKE_TOLERANCE,
            "unattainable_policy": (
                "a group whose target mean gain lies outside the means of its "
                "seeded cells takes the nearest attainable tilt and is reported"
            ),
            "zero_seed_policy": (
                "a cell with no frame support keeps a zero target and is "
                "reported; its joint mass is met from the income band's pooled "
                "fallback where support exists"
            ),
            "fallback_policy": UK_CGT_FALLBACK_POLICY,
            "conditioning_dimensions": list(UK_CGT_CONDITIONING_DIMENSIONS),
            "reporting_grain": "Table 6 age bands and the twelve region-tier areas",
        },
        "rank_preserving_allocation": {
            "within": (
                "income band x age group x region group cell, then the income "
                "band's pooled unassigned gainers"
            ),
            "ordering": "existing gains descending, person_id ascending on ties",
            "band_order": "highest gain band first",
            "suppressed_cell_allocation": (
                "count implied by the cell's published gains at the band-total mean"
            ),
            "column_reconciliation": (
                "every income column rescales onto its published All-row taxpayer total"
            ),
            "shortfall_policy": (
                "proportional scale-down inside a cell that holds less gainer "
                "mass than its raked target; the income band's net shortfall "
                "after every cell walk, apportioned to gain bands in proportion "
                "to their positive shortfalls, is walked over the income band's "
                "pooled unassigned gainers, scaled down proportionally when the "
                "pool is short, so cell-walk rounding overshoot is netted rather "
                "than compounded"
            ),
            "minimum_allocation_people": int(_MINIMUM_ALLOCATION_PEOPLE),
            "weights": (
                "household_weight mapped to persons; no person splits across bands"
            ),
            "rounding": (
                "systematic with one seeded offset per walk: a person joins the "
                "band whose cumulative boundary first covers (1 - offset) of "
                "their weight, so a band boundary smaller than a person's weight "
                "still receives that person on the share of walks it implies "
                "and the rounding is unbiased across the conditioning cells; "
                "each cell walk's signed rounding error is carried into the next "
                "cell's boundaries for the same gain band (error diffusion), so "
                "each gain band's walked total within an income band stays "
                "within one person's weight of its raked total instead of "
                "accumulating the overshoot of many cell walks"
            ),
            "cell_order": (
                "income band ascending, age group ascending, region group "
                "ascending, gain bands highest first inside a cell; then the "
                "pooled walk per income band"
            ),
        },
        "within_band_draws": {
            "bounded_band_family": (
                "truncated exponential matched to the cell's published mean"
            ),
            "open_band_family": "Pareto with alpha = mean / (mean - lower bound)",
            "quantile_scheme": (
                "stratified by weight: the persons a (gain band, income band) "
                "plan receives across every cell and the pooled walk take the "
                "quantile strata that partition (0, 1) in proportion to their "
                "household weights, in ascending prior-gain order with one "
                "seeded jitter per plan, so the plan's weighted realised mean "
                "sits on its published mean rather than carrying n independent "
                "draws or the weight-rank mix of equal strata"
            ),
            "open_band_realization": (
                "each open-band person takes the conditional mean of their "
                "weight-proportional quantile stratum, which averages back to "
                "the published mean exactly under the household weights; an iid "
                "draw from a Pareto with infinite variance sits well below it on "
                "a few dozen carriers, and equal strata on unequal weights put "
                "the tail's mass on whichever rows happen to be light"
            ),
            "mean_repair_margin": _MEAN_MARGIN,
            "mean_repair_reason": (
                "Published counts round to the nearest thousand and amounts "
                "to the nearest million; two cells of the 2024-25 table (gains "
                "of GBP 1m to 2m at taxable incomes of GBP 37,700 to 49,999 and "
                "GBP 100,000 to 125,139) imply a mean below their own band, and "
                "repaired means clamp just inside the violated boundary."
            ),
            "bottom_band_floor": "annual exempt amount plus one pound",
            "seed_base": UK_CGT_IMPUTATION_SEED,
            "seed_mixing": (
                "seed combined with the build period; each walk draws its "
                "rounding offset in cell order then pooled-fallback order, then "
                "each plan draws its quantile jitter, income bands ascending and "
                "gain bands highest first"
            ),
            "deterministic": True,
            "cell_means": (
                "ratios of the published Table 3 2024-25 cell gains to cell "
                "taxpayers, both rounded by the publisher (counts to the nearest "
                "thousand, amounts to the nearest million), so derived rather "
                "than published numbers; suppressed-count cells take the "
                "band-total mean; every mean is repaired into its band"
            ),
        },
        "sub_aea_remainder": {
            "policy": (
                "gainers beyond the published taxpayer mass keep their "
                "existing amounts capped at the annual exempt amount"
            ),
            "rationale": (
                "Table 3 covers only individuals with a CGT liability; "
                "remaining gainers are treated as sub-AEA gainers rather "
                "than invented into the liability distribution or deleted."
            ),
        },
        "record_mass_conservation_receipt": {
            "entity": "household",
            "reason": UK_CGT_SPINE_MASS_CONSERVATION_REASON,
            "declared_factor": 1.0,
            "gate_coupling": (
                "The terminal family gate requires a valid mass-conserving "
                "MassChangeRecord carrying exactly this spine-specific reason."
            ),
        },
        "classify_cgt_band_facts_with_reviewed_fence": {
            "calibration_permitted": False,
            "fact_fence_id": "cgt_band_facts_policy_endogenous_proxy_conditioned",
            "fenced_fact_count": 76,
            "fenced_fact_composition": (
                "60 joint cells, 10 gain-band row totals, 6 income-column totals "
                "(Table 3 2024-25, 2026 release)"
            ),
            "classification_rationale": (
                "The taxpayer count is endogenous to policy, the income "
                "conditioning is an arithmetic proxy, and the published "
                "surface needs rounding and suppression reconciliation "
                "before any per-band fact is exact."
            ),
            "calibrated_facts_unchanged": (
                "The calibrated CGT facts are the FY2024-25 individual Table 1 "
                "references (hmrc.cgt.taxpayers_total, hmrc.cgt.gains_total, "
                "hmrc.cgt.liability_total; UK_CGT_TARGET_COVERAGE_REQUIREMENTS) "
                "and the Table 2.1a size-of-gain, Table 6 age-band and Table 5 "
                "region rows declared in uk_population_targets.json "
                "(microcosm#467, #725); the 76 Table 3 2024-25 joint and "
                "marginal cells stay fenced and condition the imputation only, "
                "and Table 3's all-gains, all-incomes pair restates the Table 1 "
                "individual observations already bound."
            ),
            "promotion_path": (
                "A separately reviewed target profile may lift specific band "
                "facts after the reconciliation and proxy adequacy are "
                "adjudicated; the Table 2.1a, 6 and 5 rows were promoted under "
                "microcosm#467 and #725 on the FY2024-25 individual "
                "observations, while the Table 3 joint cells remain fenced "
                "because the taxable-income proxy is still an arithmetic "
                "approximation."
            ),
            "adjudication": "https://github.com/PolicyEngine/microcosm/issues/552",
        },
    }
    for kind, expected_parameters in expected_operations.items():
        actual = operations[kind]
        if actual != expected_parameters:
            drifted = sorted(
                key
                for key in {*actual, *expected_parameters}
                if actual.get(key) != expected_parameters.get(key)
            )
            raise ValueError(
                f"CGT spine {kind} declaration drifted from the reviewed "
                f"mapping on parameter(s) {drifted}."
            )
