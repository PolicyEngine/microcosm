"""Universal Credit take-up by award band and family type (microcosm#1334).

``frs_take_up`` draws ``would_claim_uc`` early, before the SPI chain, and
``spi_benefit_coherence`` redraws the SPI-channel units on the same rule: a
reporter claims, and the rest of the take-up contract's one rate times the age
population is drawn uniformly over the non-reporting units of that population,
whatever their award. Real non-take-up is concentrated in small awards, so at
the spine's design weights the claimants overshot DWP's caseload for single
people without children, couples and the smallest awards and fell short for
lone parents, and the national solve met DWP's caseloads by reweighting
low-income families.

This stage redraws the flag once the SPI chain and ``uc_capital_coherence``
have set the incomes and capital the engine assesses, for both channels. One
temporary engine materialization, at the release's calibration year on the
frame stamped with its survey year (as ``pension_credit_take_up`` reads Pension
Credit), gives each benefit unit the award it would be paid if it claimed: the
engine's ``universal_credit`` on a copy where every unit claims and carries the
deduction attributes ``uc_deduction_attributes`` will assign it (they read only
the unit's identity and region), so after the benefit cap and deductions, the
award the registry's ``dwp/uc_payment_dist`` rows band. It also gives whether
the engine moves the unit off its legacy benefits that year
(``legacy_benefits_closed``) and the UC child status the calibration's family
type reads. A unit with an award falls into a cell of DWP's family type (the
calibration measure's composition) by DWP's GBP 100 monthly award band, banded
as the registry rows band an award, with one more band above GBP 2,500 a
month. Within a
family type, adjacent bands pool until the pool's drawable units rest on at
least 30 distinct source households. In each pool a reporter claims, and every
other entitled unit of the age population that the engine does not move off
its legacy benefits, and that reports none it keeps (policyengine-uk ends a
kept legacy benefit once the unit claims UC), claims at

    r = (T - F) / D,

clipped to [0, 1]: ``T`` is DWP's calendar-2025 mean count of the pool's
bands (Stat-Xplore Households on Universal Credit, April to December 2025,
the facts the registry rows average), ``D`` the weight of the units drawn, and
``F`` the expected claims of the pool's other entitled units: 1 for a
reporter, the claim probability ``uc_legacy_closure_take_up``'s rule gives a
unit the engine moves off its legacy benefits (its draw is left alone, so that
probability does not move), and its stored flag for any other. ``T``, ``F``
and ``D`` are Great Britain's, at the stage's household weights, which DWP's
counts cover. The drawn units claim in the order of their uniform until their
weight is nearest ``T - F``, so a pool realizes ``r`` to within one unit's
weight rather than only in expectation, and Northern Ireland's units claim
below the same uniform. A pool where
``F`` alone reaches ``T``, or where ``T`` exceeds every entitled unit, cannot
meet DWP's count; the receipt flags it. A unit with no award, or outside the
age population, keeps its stored flag, the single-rate draw. The uniform is
the ``would_claim_uc`` identity stream ``frs_take_up`` and
``spi_benefit_coherence`` use, so a unit whose pool's rate falls below the
early draw's keeps its claim only if it had one.

The age fit (María's calls of 2026-11-09, microcosm#1095). Y2 took full-time
students off Universal Credit, and the pools then refilled the single
caseload from every other entitled single, so claimants and partners aged
16 to 24 read well above DWP's count. Within every pool, the drawn units
with a claimant or partner aged 16 to 24 claim at one odds ratio against
the pool's other drawn units, the two rates together realizing the pool's
count. The ratio is solved over Great Britain so that those young claimants
and partners, in entitled units that claim, come as near DWP's People on
Universal Credit aged 16 to 24 (February, May, August and November 2025),
the facts the dwp.uc.people_aged_16_to_24 diagnostic reads, as the redraw
allows. A shortfall passed from one pool to another fills only the receiving
pool's drawn units with no young claimant or partner, so a family type whose
shortfall only young units could fill falls short of DWP's count by it, and
the receipt records what stays unplaced. Fixed claims (a reporter, a moved
unit, a unit kept on its legacy benefits) and a pool whose own count needs
more claims than its other drawn units can give keep young units claiming
whatever the ratio, so the young count can stay above DWP's; the ratio then
holds at the end of its range and the receipt says so.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.frs_take_up import (
    UK_UC_AGE_ELIGIBLE_AGGREGATE,
    UK_UC_AGE_ELIGIBLE_METHOD,
    UK_UC_AGE_ELIGIBLE_ROLE,
    UK_UC_AGE_ELIGIBLE_SOURCE,
    UK_UC_TAKE_UP_OUTPUT,
    UKTakeUpPopulationPolicy,
    uc_age_eligible_benunits,
    uk_take_up_population_policy,
)
from microcosm.build.uk_runtime.legacy_closure_take_up import (
    LEGACY_CLOSURE_RESOURCE,
    _reported_by_benunit,
    legacy_benefit_combinations,
    legacy_closure_claim_probabilities,
    load_move_to_uc_claim_rates,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_DISCLOSURE_FLOOR,
    UK_WITHHELD,
    distinct_count,
    extreme_count,
    rounded_count,
    small_cell_share,
    source_household_ids,
    support_band,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
    _benunit_households,
    _household_to_benunit_weights,
)
from microcosm.build.uk_runtime.uc_deduction_attributes import (
    assign_uc_deduction_attributes,
    load_uc_deduction_distributions,
)
from microcosm.build.uk_runtime.uc_relationships import (
    frs_uc_claimant_mask,
    uc_family_child_member,
)
from microcosm.frame import Frame
from microcosm.frame.adapters.policyengine_uk import ProjectingRulesEngine
from microcosm.frame.rules import assert_rules_engine_country

__all__ = [
    "UC_AWARD_BAND_LABELS",
    "UC_FAMILY_TYPES",
    "UC_TAKE_UP_AGE_RECORD_SETS",
    "UC_TAKE_UP_AGE_RESOURCE",
    "UC_TAKE_UP_ENTITLEMENT_VARIABLES",
    "UC_TAKE_UP_MINIMUM_POOL_SOURCES",
    "UC_TAKE_UP_OPERATION_KIND",
    "UC_TAKE_UP_POOLING",
    "UC_TAKE_UP_RECORD_SETS",
    "UC_TAKE_UP_RESOURCE",
    "UC_TAKE_UP_STAGE_NAME",
    "UKUCTakeUpResult",
    "UKUCTakeUpStageTransform",
    "assign_award_band_take_up",
    "load_uc_award_band_targets",
    "load_uc_young_people_target",
    "redraw_uc_take_up",
    "uc_award_band_targets_from_rows",
    "uc_award_bands",
    "uc_family_types",
    "uc_take_up_operation_parameters",
    "uc_young_people_target_from_rows",
    "working_age_housing_benefit_receipt",
]

UC_TAKE_UP_STAGE_NAME = "uc_take_up"
UC_TAKE_UP_OPERATION_KIND = "assign_award_band_take_up_residual"
UC_TAKE_UP_SEED = 0
UC_TAKE_UP_RESOURCE = "dwp_uc_award_band_facts.json"
#: The months DWP's Stat-Xplore counts cover, whose mean is the registry rows'
#: calendar-2025 value (their ``calendar_year_average``).
UC_TAKE_UP_MONTHS: tuple[str, ...] = tuple(
    f"2025-{month:02d}" for month in range(4, 13)
)
UC_TAKE_UP_RECORD_SETS: tuple[str, ...] = tuple(
    "dwp.statx.uc_benefit_units.award_band_family_type.month" + month.replace("-", "_")
    for month in UC_TAKE_UP_MONTHS
)
_CONCEPT = "dwp.uc_households"
_MEASURE = "benefit_units"
UC_TAKE_UP_ENTITLEMENT_VARIABLES = (
    "universal_credit",
    "legacy_benefits_closed",
    "is_child_or_qualifying_young_person_for_universal_credit",
)
#: Entitlement is read at the release's calibration year on the frame the
#: engine projects there from its survey year, as Pension Credit's is.
UC_TAKE_UP_ENTITLEMENT_YEAR_RULE = "calibration_year"
UC_TAKE_UP_DATASET_YEAR_RULE = "survey_year"
UC_TAKE_UP_SOLVE_SCOPE = "great_britain"
_OUTSIDE_SOLVE_REGION = "NORTHERN_IRELAND"
#: The legacy award a unit keeps off a Universal Credit claim: policyengine-uk
#: ends a claimant's or partner's income-related ESA once the family claims
#: Universal Credit (from 2.149.0; pe-uk#2563), so a unit on that award at the
#: interview claims none (microcosm#1095).
UC_TAKE_UP_KEPT_ESA_COLUMN = "esa_income_reported"
UC_TAKE_UP_KEPT_ESA_RULE = (
    "a unit whose claimant or partner reports income-related ESA "
    "(esa_income_reported), and that neither reports Universal Credit nor is "
    "moved off its legacy benefits, claims no Universal Credit, since "
    "policyengine-uk ends that award once the family claims it"
)
#: The calibration measure's family types with DWP's Stat-Xplore labels.
UC_FAMILY_TYPES: tuple[tuple[str, str], ...] = (
    ("SINGLE", "Single, no children"),
    ("LONE_PARENT", "Single, with children"),
    ("COUPLE_NO_CHILDREN", "Couple, no children"),
    ("COUPLE_WITH_CHILDREN", "Couple, with children"),
)
#: DWP's GBP 100 monthly bands as annual awards, (1,200 i, 1,200 (i + 1)] in
#: whole pence for i < 25 as the registry rows band them, and the band above.
_BAND_PENCE = 120_000
_BOUND_BANDS = 25
UC_AWARD_BAND_LABELS: tuple[str, ...] = (
    *(f"£{100 * index}.01 to £{100 * (index + 1)}.00" for index in range(_BOUND_BANDS)),
    f"£{100 * _BOUND_BANDS}.01 or over",
)
UC_TAKE_UP_MINIMUM_POOL_SOURCES = 30
UC_TAKE_UP_POOLING = (
    "adjacent bands within a family type, lowest first, until the drawn units "
    "rest on minimum_pool_sources distinct household_source_id values; a short "
    "last pool joins the one before it. Then, lowest first, a pool that cannot "
    "meet DWP's count (its undrawn entitled units' expected claims exceed the "
    "count, or the count exceeds them plus the weight of every drawn unit) is "
    "held at the nearer of those ends and passes its gap to the other pools, "
    "nearest first, the one with more room first at equal distance and the "
    "lower on a tie, each taking what its room allows; a gap no pool can take "
    "leaves the family type unreachable. With the age fit, a passed "
    "shortfall's room is the pool's undrawn claims plus the weight of its "
    "drawn units with no young claimant or partner, less its count (age_fit)"
)
UC_TAKE_UP_RESIDUAL_RATE = (
    "(T - F) / D, clipped to [0, 1]; T DWP's calendar-2025 mean count of the "
    "pool's bands after any gaps passed to or from it, D the weight of the pool's drawn units and F the expected "
    "claims of its other entitled units, over Great Britain at the stage's "
    "household weights; realized by the drawn units claiming in the order of "
    "their uniform until their weight is nearest T - F, and applied at that "
    "uniform to Northern Ireland's drawn units too"
)
#: The age fit (María's calls of 2026-11-09, Y4 of the November event-data
#: plan, microcosm#1095). After Y2 took full-time students off Universal
#: Credit, the award-band pools refilled the single caseload from every other
#: entitled single, and claimants and partners aged 16 to 24 read well above
#: DWP's count. The fit gives the drawn units with a young claimant or partner
#: one odds ratio against the pool's others, solved for DWP's count of people
#: on Universal Credit aged 16 to 24, and a shortfall passed between pools
#: fills only units with no young claimant or partner.
UC_TAKE_UP_AGE_RESOURCE = "dwp_uc_people_by_age_facts.json"
UC_TAKE_UP_AGE_MONTHS: tuple[str, ...] = ("2025-02", "2025-05", "2025-08", "2025-11")
UC_TAKE_UP_AGE_RECORD_SETS: tuple[str, ...] = tuple(
    "dwp.statx.uc_people.country_age_conditionality_regime.month"
    + month.replace("-", "_")
    + ".total"
    for month in UC_TAKE_UP_AGE_MONTHS
)
_AGE_CONCEPT = "dwp.people_on_universal_credit"
_AGE_MEASURE = "people"
_AGE_GEOGRAPHY = "K03000001"
_AGE_DIMENSION = "age_bands_and_single_year"
_AGE_REGIME_DIMENSION = "conditionality_regime"
UC_TAKE_UP_YOUNG_AGES: tuple[int, ...] = tuple(range(16, 25))
UC_TAKE_UP_YOUNG_ROLE = "is_claimant_or_partner"
UC_TAKE_UP_YOUNG_PEOPLE = (
    "claimants and partners (is_claimant_or_partner) aged 16 to 24, at the "
    "stored age, which policyengine-uk carries unchanged to the entitlement "
    "year; a unit counts each of them"
)
UC_TAKE_UP_AGE_TARGET = (
    "DWP Stat-Xplore People on Universal Credit in Great Britain aged 16 to 24, "
    "every conditionality regime: the single years summed in February, May, "
    "August and November 2025 and averaged, as the dwp.uc.people_aged_16_to_24 "
    "diagnostic reads them"
)
UC_TAKE_UP_AGE_FIT = (
    "one odds ratio theta for the drawn units with a young claimant or partner: "
    "in every pool they claim at theta times the odds of the pool's other drawn "
    "units, the two rates together realizing the pool's count, each realized in "
    "the order of the uniform as a pool's rate is; theta is solved so that the "
    "young claimants and partners of Great Britain's entitled units that claim, "
    "at the stage's household weights, are nearest DWP's count, held at the "
    "nearer end of exp(-12) to exp(12) where no theta reaches it; a shortfall "
    "passed to another pool fills only that pool's drawn units with no young "
    "claimant or partner, and what no pool can take that way stays unplaced, "
    "leaving the family type short of DWP's count by it (María's call of "
    "2026-11-09)"
)
#: The odds ratio's search interval, on its log.
_LOG_ODDS_BOUNDS = (-12.0, 12.0)
_ODDS_ITERATIONS = 60


def uc_award_bands(award: np.ndarray) -> np.ndarray:
    """Each unit's award band index, or -1 where it has no award.

    Band ``i`` below 25 is (GBP 1,200 i, 1,200 (i + 1)] a year compared in
    whole pence, as the registry's payment rows band the engine's award; band
    25 is everything above GBP 30,000, DWP's "GBP 2,500.01 or over".
    """

    pence = np.rint(np.asarray(award, dtype=np.float64) * 100.0)
    if not np.isfinite(pence).all():
        raise ValueError("Universal Credit awards must be finite.")
    band = np.full(pence.shape, -1, dtype=np.int64)
    positive = pence > 0
    band[positive] = np.minimum(
        np.ceil(pence[positive] / _BAND_PENCE).astype(np.int64) - 1, _BOUND_BANDS
    )
    return band


def uc_family_types(
    person: pd.DataFrame, benunit: pd.DataFrame, qualifying: np.ndarray
) -> np.ndarray:
    """DWP's family type per benefit unit, as the calibration measure reads it.

    Two FRS claimant or partner roles make a couple; a member counts as a
    child where ``uc_family_child_member`` says so (a UC child or qualifying
    young person, or anyone else under 20 who is not a claimant). The rule is
    ``measure_simulation``'s ``uc_calibration_family_type``.
    """

    claimant = frs_uc_claimant_mask(person, benunit)
    qualifying = np.asarray(qualifying, dtype=bool)
    if qualifying.shape != (len(person),):
        raise ValueError("UC child status must align with the person table.")
    members = person["person_benunit_id"].to_numpy()
    ids = benunit["benunit_id"].to_numpy()
    claimants = pd.Series(claimant).groupby(members).sum().reindex(ids).to_numpy()
    children = uc_family_child_member(
        claimant, qualifying, person["age"].to_numpy(dtype=float)
    )
    child_count = pd.Series(children).groupby(members).sum().reindex(ids).to_numpy()
    couple = claimants == 2
    has_children = child_count > 0
    return np.select(
        [couple & has_children, couple, has_children],
        ["COUPLE_WITH_CHILDREN", "COUPLE_NO_CHILDREN", "LONE_PARENT"],
        default="SINGLE",
    ).astype(object)


def uc_award_band_targets_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], float]:
    """DWP's calendar-2025 mean count by (family type, award band).

    The mean of the nine monthly Stat-Xplore counts, April to December 2025,
    as the registry rows' ``calendar_year_average`` reads them. Rows for the
    unknown family type and for households with no payment are not cells.
    """

    families = {label: name for name, label in UC_FAMILY_TYPES}
    bands = {label: index for index, label in enumerate(UC_AWARD_BAND_LABELS)}
    counts: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        if row.get("concept") != _CONCEPT or row.get("measure_id") != _MEASURE:
            continue
        dimensions = row.get("dimensions") or {}
        family = families.get(str(dimensions.get("family_type")))
        band = bands.get(str(dimensions.get("monthly_award_amount_bands")))
        if family is None or band is None:
            continue
        month = str((row.get("period") or {}).get("value"))
        if month not in UC_TAKE_UP_MONTHS:
            raise ValueError(f"UC award band facts carry an unexpected month {month}.")
        cell = counts.setdefault((family, band), {})
        if month in cell:
            raise ValueError(
                f"UC award band facts repeat {family} band {band} in {month}."
            )
        cell[month] = float(row["value"])
    expected = {
        (family, band)
        for family, _ in UC_FAMILY_TYPES
        for band in range(len(UC_AWARD_BAND_LABELS))
    }
    if set(counts) != expected or any(
        set(cell) != set(UC_TAKE_UP_MONTHS) for cell in counts.values()
    ):
        raise ValueError(
            "UC award band facts must give every family type and award band in "
            f"each month {UC_TAKE_UP_MONTHS[0]} to {UC_TAKE_UP_MONTHS[-1]}."
        )
    return {
        key: float(np.mean([cell[month] for month in UC_TAKE_UP_MONTHS]))
        for key, cell in sorted(counts.items())
    }


def load_uc_award_band_targets() -> dict[tuple[str, int], float]:
    """The targets from the committed vendored resource, checked against the pin."""

    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    return uc_award_band_targets_from_rows(vendored_rows(UC_TAKE_UP_RESOURCE))


def uc_young_people_target_from_rows(rows: Sequence[Mapping[str, Any]]) -> float:
    """DWP's people on Universal Credit aged 16 to 24 in Great Britain.

    Every conditionality regime's single years of age, summed in each of
    February, May, August and November 2025 and averaged: the value the
    ``dwp.uc.people_aged_16_to_24`` diagnostic compiles from the same facts.
    The age bands above 24 are not cells.
    """

    counts: dict[str, dict[int, float]] = {}
    for row in rows:
        if row.get("concept") != _AGE_CONCEPT or row.get("measure_id") != _AGE_MEASURE:
            continue
        if (row.get("geography") or {}).get("id") != _AGE_GEOGRAPHY:
            continue
        dimensions = row.get("dimensions") or {}
        if str(dimensions.get(_AGE_REGIME_DIMENSION)) != "all":
            continue
        age = dimensions.get(_AGE_DIMENSION)
        if isinstance(age, bool) or not isinstance(age, int):
            continue
        if age not in UC_TAKE_UP_YOUNG_AGES:
            continue
        month = str((row.get("period") or {}).get("value"))
        if month not in UC_TAKE_UP_AGE_MONTHS:
            raise ValueError(
                f"UC people by age facts carry an unexpected month {month}."
            )
        cell = counts.setdefault(month, {})
        if age in cell:
            raise ValueError(f"UC people by age facts repeat age {age} in {month}.")
        cell[age] = float(row["value"])
    if set(counts) != set(UC_TAKE_UP_AGE_MONTHS) or any(
        set(cell) != set(UC_TAKE_UP_YOUNG_AGES) for cell in counts.values()
    ):
        raise ValueError(
            "UC people by age facts must give Great Britain's every-regime count "
            "at each single age 16 to 24 in "
            f"{', '.join(UC_TAKE_UP_AGE_MONTHS)}."
        )
    return float(
        np.mean([sum(counts[month].values()) for month in UC_TAKE_UP_AGE_MONTHS])
    )


def load_uc_young_people_target() -> float:
    """The age fit's count from the committed vendored resource."""

    from microcosm.build.uk_runtime.ledger_fact_vendoring import vendored_rows

    return uc_young_people_target_from_rows(vendored_rows(UC_TAKE_UP_AGE_RESOURCE))


def _odds_scaled(rate: float, theta: float) -> float:
    """The rate whose odds are ``theta`` times ``rate``'s."""

    if rate <= 0.0:
        return 0.0
    if rate >= 1.0:
        return 1.0
    return theta * rate / (1.0 - rate + theta * rate)


def _split_rates(
    need: float, young: float, other: float, theta: float
) -> tuple[float, float]:
    """The young and other drawn units' rates that realize ``need`` together.

    ``young`` and ``other`` are the two groups' drawn weights. The young rate's
    odds are ``theta`` times the other's, so the weight claiming,
    ``young * r_y + other * r_o``, rises with ``r_o``; bisection finds the pair
    that meets ``need``, which lies strictly between nothing and every unit.
    """

    if need <= 0.0:
        return 0.0, 0.0
    if need >= young + other:
        return 1.0, 1.0
    if young <= 0.0:
        return 0.0, need / other
    if other <= 0.0:
        return need / young, 0.0
    low, high = 0.0, 1.0
    for _ in range(100):
        middle = 0.5 * (low + high)
        if young * _odds_scaled(middle, theta) + other * middle < need:
            low = middle
        else:
            high = middle
    other_rate = 0.5 * (low + high)
    return _odds_scaled(other_rate, theta), other_rate


def _pools(
    *,
    band: np.ndarray,
    in_family: np.ndarray,
    drawable_gb: np.ndarray,
    source_ids: np.ndarray,
    weights: np.ndarray,
    minimum_sources: int,
) -> list[list[int]]:
    """Adjacent bands, lowest first, pooled until the drawn units rest on
    ``minimum_sources`` distinct source households; a short last pool joins the
    one before it."""

    pools: list[list[int]] = []
    current: list[int] = []
    for index in range(len(UC_AWARD_BAND_LABELS)):
        current.append(index)
        mask = in_family & drawable_gb & np.isin(band, current)
        if distinct_count(source_ids, mask, weights) >= minimum_sources:
            pools.append(current)
            current = []
    if current:
        if pools:
            pools[-1].extend(current)
        else:
            pools.append(current)
    return pools


def _pass_unmet_gaps(
    sums: Sequence[tuple[float, float, float]],
    *,
    older: Sequence[float] | None = None,
) -> tuple[list[float], list[float]]:
    """Each pool's count after the unmet pools pass their gaps on.

    ``sums`` gives each pool's DWP count ``T``, the expected claims ``F`` of
    its undrawn entitled units and the weight ``D`` of its drawn units, lowest
    band first. A pool realizes between ``F`` and ``F + D`` claims; one whose
    count lies outside is held at the nearer end, and its gap is the claims it
    cannot add (a shortfall) or give up (an excess). A shortfall and an excess
    offset each other in the family type's total, so they cancel first, nearest
    pools first. What remains goes to the other pools, nearest first, the one
    with more room first at equal distance and the lower on a tie, each taking
    what its room allows (``F + D - T`` for a shortfall, ``T - F`` for an
    excess). Joining an unmet pool to its neighbour applied one rate to both,
    so the drawn units of an overshooting band started claiming and those of
    a short band stopped; held at its clip, each band realizes as near its own
    count as it can, and the family type's total is unchanged wherever its
    entitled units allow (microcosm#1428 review, finding 7). With ``older``,
    the weight of each pool's drawn units with no young claimant or partner
    (the age fit, María's call of 2026-11-09), a shortfall passed to another
    pool fills only those units: its room is ``F + older - count``, and what
    no pool can take that way stays unplaced. An excess passes as before.
    Returns each pool's count and the part of its gap no pool could take.
    """

    targets = [float(target) for target, _, _ in sums]
    lows = [float(undrawn) for _, undrawn, _ in sums]
    highs = [float(undrawn) + float(drawn) for _, undrawn, drawn in sums]
    counts = [
        min(max(t, lo), hi) for t, lo, hi in zip(targets, lows, highs, strict=True)
    ]
    gaps = [t - count for t, count in zip(targets, counts, strict=True)]
    for index, target in enumerate(targets):
        if abs(gaps[index]) <= 1e-9 * max(1.0, abs(target)):
            gaps[index] = 0.0

    def nearest(index: int) -> list[int]:
        return sorted(
            (other for other in range(len(sums)) if other != index),
            key=lambda other: (abs(other - index), other),
        )

    for index in range(len(sums)):
        for other in nearest(index):
            if gaps[index] == 0.0:
                break
            if gaps[index] * gaps[other] < 0.0:
                offset = min(abs(gaps[index]), abs(gaps[other]))
                gaps[index] -= np.sign(gaps[index]) * offset
                gaps[other] -= np.sign(gaps[other]) * offset
    for index, gap in enumerate(gaps):
        if gap == 0.0:
            continue
        short = gap > 0.0

        def room(other: int, *, short: bool = short) -> float:
            if not short:
                return max(counts[other] - lows[other], 0.0)
            ceiling = (
                highs[other] if older is None else lows[other] + float(older[other])
            )
            return max(ceiling - counts[other], 0.0)

        remaining = abs(gap)
        for other in sorted(
            nearest(index),
            key=lambda other: (abs(other - index), -room(other), other),
        ):
            take = min(remaining, room(other))
            counts[other] += take if short else -take
            remaining -= take
            if remaining <= 0.0:
                break
        gaps[index] = remaining if short else -remaining
    return counts, gaps


def _realized_rate(
    residual: float, *, need: float, draws: np.ndarray, weights: np.ndarray
) -> float:
    """The uniform below which a pool's drawn units claim.

    ``residual`` clipped to [0, 1] is the share of the drawn weight that
    claims. Inside (0, 1) the drawn units claim in the order of their uniform
    until their weight is nearest ``need``, so the pool realizes DWP's count
    to within one unit's weight rather than only in expectation; no claimant
    at all is a candidate too, when ``need`` sits nearer zero than the first
    unit's weight. The uniform just above the last claimant's (the lowest
    uniform, when none claims) is the rate the pool's Northern Ireland units,
    which carry no weight in the solve, are drawn at too.
    """

    if residual <= 0.0:
        return 0.0
    if residual >= 1.0:
        return 1.0
    order = np.argsort(draws, kind="stable")
    cumulative = np.cumsum(weights[order])
    last = min(int(np.searchsorted(cumulative, need)), len(order) - 1)
    if last > 0 and abs(cumulative[last - 1] - need) < abs(cumulative[last] - need):
        last -= 1
    if last == 0 and abs(need) < abs(cumulative[0] - need):
        # No claimant is nearer DWP's count than the first: a unit claims
        # below the rate, and no uniform lies below the lowest.
        return float(draws[order[0]])
    return float(np.nextafter(draws[order[last]], 2.0))


def assign_award_band_take_up(
    *,
    award: np.ndarray,
    family: np.ndarray,
    population: np.ndarray,
    reporter: np.ndarray,
    moved: np.ndarray,
    moved_claim: np.ndarray,
    previous: np.ndarray,
    weights: np.ndarray,
    draws: np.ndarray,
    great_britain: np.ndarray,
    source_ids: np.ndarray,
    targets: Mapping[tuple[str, int], float],
    minimum_pool_sources: int = UC_TAKE_UP_MINIMUM_POOL_SOURCES,
    legacy_kept: np.ndarray | None = None,
    young_people: np.ndarray | None = None,
    young_target: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Reporters claim; drawn units claim at their pool's residual rate.

    With ``young_people`` (each unit's claimants and partners aged 16 to 24)
    and ``young_target`` (DWP's count of them), the age fit splits each pool's
    rate: its drawn units with a young member claim at an odds ratio ``theta``
    against its other drawn units, the two rates realizing the pool's count
    together, and ``theta`` is solved for DWP's count over Great Britain
    (``UC_TAKE_UP_AGE_FIT``). Without them every pool draws at one rate.

    The drawn units are the entitled, non-reporting units of the age
    population that the engine does not move off their legacy benefits and
    that report none (``legacy_kept``: a unit reporting a legacy benefit the
    engine leaves in place keeps it, and policyengine-uk ends it once the unit
    claims UC, so the draw must not move it; Vahid's review of #1344, item 7).
    Every other unit keeps ``previous``, or claims if it reports. In a pool, ``F``
    counts a reporter at 1, a moved unit at ``moved_claim`` (its expected claim
    at the legacy closure) and any other undrawn entitled unit at ``previous``.
    A drawn unit claims where its uniform falls below the pool's realized rate
    (``_realized_rate``), so units claim in the order of their uniform and the
    early draw's claimants keep claiming wherever the rate rises.
    """

    award = np.asarray(award, dtype=np.float64)
    family = np.asarray(family, dtype=object)
    population = np.asarray(population, dtype=bool)
    reporter = np.asarray(reporter, dtype=bool)
    moved = np.asarray(moved, dtype=bool)
    moved_claim = np.asarray(moved_claim, dtype=np.float64)
    previous = np.asarray(previous, dtype=bool)
    legacy_kept = (
        np.zeros(previous.shape, dtype=bool)
        if legacy_kept is None
        else np.asarray(legacy_kept, dtype=bool)
    )
    weights = np.asarray(weights, dtype=np.float64)
    draws = np.asarray(draws, dtype=np.float64)
    gb = np.asarray(great_britain, dtype=bool)
    source_ids = np.asarray(source_ids)
    shape = award.shape
    for name, values in (
        ("family", family),
        ("population", population),
        ("reporter", reporter),
        ("moved", moved),
        ("moved_claim", moved_claim),
        ("previous", previous),
        ("legacy_kept", legacy_kept),
        ("weights", weights),
        ("draws", draws),
        ("great_britain", gb),
        ("source_ids", source_ids),
    ):
        if values.shape != shape:
            raise ValueError(f"UC take-up {name} must align to the benefit units.")
    if not ((moved_claim >= 0.0) & (moved_claim <= 1.0)).all():
        raise ValueError("Expected claims at the legacy closure must lie in [0, 1].")
    if (young_people is None) != (young_target is None):
        raise ValueError("The age fit needs both young_people and young_target.")
    if young_people is not None:
        young_people = np.asarray(young_people, dtype=np.float64)
        if young_people.shape != shape:
            raise ValueError("UC take-up young_people must align to the benefit units.")
        if not (np.isfinite(young_people).all() and (young_people >= 0.0).all()):
            raise ValueError("UC take-up young_people must be finite counts.")
        if not (np.isfinite(young_target) and young_target > 0.0):
            raise ValueError("The age fit's count must be positive and finite.")
    unknown = sorted(set(family) - {name for name, _ in UC_FAMILY_TYPES})
    if unknown:
        raise ValueError(f"UC take-up family types are unknown: {unknown}.")
    missing = sorted(
        (name, index)
        for name, _ in UC_FAMILY_TYPES
        for index in range(len(UC_AWARD_BAND_LABELS))
        if (name, index) not in targets
    )
    if missing:
        raise ValueError(f"UC take-up targets are missing cells: {missing}.")
    band = uc_award_bands(award)
    entitled = band >= 0
    drawable = entitled & population & ~reporter & ~moved & ~legacy_kept
    # Each undrawn entitled unit's expected claim.
    fixed = np.where(
        reporter, 1.0, np.where(moved, moved_claim, previous.astype(np.float64))
    )
    rates = np.zeros(shape, dtype=np.float64)

    def sources(mask: np.ndarray) -> int:
        return distinct_count(source_ids, mask, weights)

    def balance(
        name: str, in_family: np.ndarray, bands: list[int]
    ) -> tuple[float, float, float]:
        solve = in_family & np.isin(band, bands) & gb
        return (
            float(sum(targets[(name, index)] for index in bands)),
            float((weights * fixed)[solve & ~drawable].sum()),
            float(weights[solve & drawable].sum()),
        )

    young_unit = (
        np.zeros(shape, dtype=bool) if young_people is None else young_people > 0.0
    )
    pools: list[dict[str, object]] = []
    # Each pool's drawn units (with Northern Ireland's), its Great Britain
    # drawn units, the claims they must realize and their weight.
    needs: list[tuple[np.ndarray, np.ndarray, float, float]] = []
    # With the age fit, the shortfall each family type leaves unplaced.
    unplaced_shortfall: dict[str, float] = {}
    for name, _ in UC_FAMILY_TYPES:
        in_family = family == name
        family_pools = _pools(
            band=band,
            in_family=in_family,
            drawable_gb=drawable & gb,
            source_ids=source_ids,
            weights=weights,
            minimum_sources=minimum_pool_sources,
        )
        sums = [balance(name, in_family, bands) for bands in family_pools]
        # With the age fit, a passed shortfall fills only a pool's drawn units
        # with no young claimant or partner (María's call of 2026-11-09).
        older = (
            None
            if young_people is None
            else [
                float(
                    weights[
                        in_family & np.isin(band, bands) & gb & drawable & ~young_unit
                    ].sum()
                )
                for bands in family_pools
            ]
        )
        counts, unplaced = _pass_unmet_gaps(sums, older=older)
        unplaced_shortfall[name] = float(sum(gap for gap in unplaced if gap > 0.0))
        feasible = not any(unplaced)
        for bands, (target, undrawn, drawn), count in zip(
            family_pools, sums, counts, strict=True
        ):
            in_pool = in_family & np.isin(band, bands)
            solve = in_pool & gb
            needs.append((in_pool & drawable, solve & drawable, count - undrawn, drawn))
            # Where the undrawn claims reach DWP's count, or the count exceeds
            # every entitled unit, the pool cannot meet it alone: it is held
            # at its clip and its gap passed on.
            unreachable = (
                "undrawn_claims_reach_target"
                if target <= undrawn
                else "target_exceeds_entitled"
                if target >= undrawn + drawn
                else None
            )
            pools.append(
                {
                    "family_type": name,
                    "bands": bands,
                    "mask": solve,
                    "target": target,
                    "unreachable": unreachable,
                    "feasible_family": feasible,
                }
            )

    def realize(theta: float | None) -> np.ndarray:
        """Each unit's claim at the age fit's odds ratio, or at one rate."""

        rates[:] = 0.0
        for unit, solve_drawn, need, drawn in needs:
            if drawn <= 0.0:
                continue
            if theta is None:
                rates[unit] = _realized_rate(
                    need / drawn,
                    need=need,
                    draws=draws[solve_drawn],
                    weights=weights[solve_drawn],
                )
                continue
            young_solve = solve_drawn & young_unit
            other_solve = solve_drawn & ~young_unit
            young_rate, other_rate = _split_rates(
                need,
                float(weights[young_solve].sum()),
                float(weights[other_solve].sum()),
                theta,
            )
            for rate, group_solve, group in (
                (young_rate, young_solve, unit & young_unit),
                (other_rate, other_solve, unit & ~young_unit),
            ):
                if group_solve.any():
                    rates[group] = _realized_rate(
                        rate,
                        need=rate * float(weights[group_solve].sum()),
                        draws=draws[group_solve],
                        weights=weights[group_solve],
                    )
                else:
                    # Only Northern Ireland's units: drawn at the group's rate.
                    rates[group] = rate
        return np.where(drawable, draws < rates, previous | reporter)

    entitled_gb = entitled & gb
    young_weight = weights * (young_people if young_people is not None else 0.0)

    def young_claims(claims: np.ndarray) -> float:
        expected = np.where(drawable, claims.astype(np.float64), fixed)
        return float((young_weight * expected)[entitled_gb].sum())

    theta: float | None = None
    age_status: str | None = None
    if young_people is not None:
        low, high = _LOG_ODDS_BOUNDS
        if young_target <= young_claims(realize(float(np.exp(low)))):
            log_theta, age_status = low, "target_below_least_young_claims"
        elif young_target >= young_claims(realize(float(np.exp(high)))):
            log_theta, age_status = high, "target_above_most_young_claims"
        else:
            # Young claims rise with theta in every pool, so bisection on its
            # log brackets DWP's count; the nearer end of the bracket is kept.
            for _ in range(_ODDS_ITERATIONS):
                middle = 0.5 * (low + high)
                if young_claims(realize(float(np.exp(middle)))) < young_target:
                    low = middle
                else:
                    high = middle
            low_gap = abs(young_claims(realize(float(np.exp(low)))) - young_target)
            high_gap = abs(young_claims(realize(float(np.exp(high)))) - young_target)
            log_theta = low if low_gap <= high_gap else high
        theta = float(np.exp(log_theta))
    would_claim = realize(theta)
    expected_before = np.where(drawable, previous.astype(np.float64), fixed)
    expected_after = np.where(drawable, would_claim.astype(np.float64), fixed)

    pools_out: list[dict[str, object]] = []
    families_out: dict[str, dict[str, object]] = {}
    for name, _ in UC_FAMILY_TYPES:
        basis = (family == name) & entitled & gb
        reachable = np.zeros(shape, dtype=bool)
        reachable_target = 0.0
        for pool in pools:
            if pool["family_type"] != name:
                continue
            solve = pool["mask"]
            supported = sources(solve)
            pools_out.append(
                {
                    "family_type": name,
                    "bands": [UC_AWARD_BAND_LABELS[index] for index in pool["bands"]],
                    "dwp_caseload": pool["target"],
                    "support": support_band(supported),
                    "drawn_support": support_band(sources(solve & drawable)),
                    # One bit on which side of DWP's count the pool's undrawn
                    # claims lie, given only where three or more sources
                    # carry the pool (UKDS disclosure control).
                    "target_unreachable": (
                        pool["unreachable"]
                        if supported >= UK_DISCLOSURE_FLOOR
                        else UK_WITHHELD
                    ),
                }
            )
            # A family type whose gaps every pool could take realizes its
            # count over all its pools; one whose count exceeds what its
            # entitled units can realize has no reachable pool.
            if pool["feasible_family"]:
                reachable |= solve
                reachable_target += float(pool["target"])
        supported = sources(basis)
        families_out[name] = {
            "dwp_caseload": float(
                sum(
                    targets[(name, index)] for index in range(len(UC_AWARD_BAND_LABELS))
                )
            ),
            "entitled_weighted": small_cell_share(
                float(weights[basis].sum()), supported
            ),
            "caseload_before": small_cell_share(
                float((weights * expected_before)[basis].sum()), supported
            ),
            "caseload_after": small_cell_share(
                float((weights * expected_after)[basis].sum()), supported
            ),
            "reachable_pools_dwp_caseload": reachable_target,
            "reachable_pools_caseload_after": small_cell_share(
                float((weights * expected_after)[reachable].sum()),
                sources(reachable),
            ),
            "pools": sum(1 for pool in pools if pool["family_type"] == name),
        }
    changed = would_claim != previous
    receipt: dict[str, object] = {
        "families": families_out,
        "pools": pools_out,
        "minimum_pool_sources": int(minimum_pool_sources),
        # Zero exact, as the gate reads it; any other count in bands.
        "reporters_not_claiming": extreme_count(int((reporter & ~would_claim).sum())),
        "changed_outside_draw": extreme_count(
            int((changed & ~drawable & ~reporter).sum())
        ),
        "drawn_units": rounded_count(int(drawable.sum())),
        "drawn_units_northern_ireland": rounded_count(int((drawable & ~gb).sum())),
        "moved_units": rounded_count(int((entitled & moved & ~reporter).sum())),
        "legacy_kept_units": rounded_count(
            int((entitled & population & legacy_kept & ~reporter & ~moved).sum())
        ),
        "set_true": rounded_count(int((changed & would_claim).sum())),
        "set_false": rounded_count(int((changed & ~would_claim).sum())),
    }
    if young_people is not None:
        carried = sources(entitled_gb & young_unit)
        drawn_young = sources(drawable & gb & young_unit)
        before = float((young_weight * expected_before)[entitled_gb].sum())
        receipt["age_fit"] = {
            "rule": UC_TAKE_UP_AGE_FIT,
            "young_people": UC_TAKE_UP_YOUNG_PEOPLE,
            "target": UC_TAKE_UP_AGE_TARGET,
            "dwp_young_people": float(young_target),
            "young_people_before": small_cell_share(before, carried),
            "young_people_after": small_cell_share(young_claims(would_claim), carried),
            "support": support_band(carried),
            "drawn_support": support_band(drawn_young),
            # A ratio solved over the drawn young units' weights, given only
            # where three or more sources carry them (UKDS disclosure control).
            "odds_ratio": (
                round(float(theta), 6)
                if drawn_young >= UK_DISCLOSURE_FLOOR
                else UK_WITHHELD
            ),
            "target_unreachable": age_status,
            # The shortfall no pool could take with older units, by family
            # type, at the stage's household weights.
            "unplaced_shortfall": {
                name: small_cell_share(
                    unplaced_shortfall[name],
                    sources((family == name) & entitled & gb),
                )
                for name, _ in UC_FAMILY_TYPES
            },
        }
    return would_claim.astype(bool), receipt


@dataclass(frozen=True)
class UKUCTakeUpResult:
    """Output frame and executed-effect receipt for the Universal Credit redraw."""

    frame: Frame
    receipt: Mapping[str, object]
    entitlement_year: int

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UC_TAKE_UP_STAGE_NAME,
            "seed": UC_TAKE_UP_SEED,
            "residual_rate": UC_TAKE_UP_RESIDUAL_RATE,
            "solve_scope": UC_TAKE_UP_SOLVE_SCOPE,
            "entitlement_year": self.entitlement_year,
            **dict(self.receipt),
        }


def redraw_uc_take_up(
    frame: Frame,
    *,
    engine: ProjectingRulesEngine,
    population_policy: UKTakeUpPopulationPolicy,
    targets: Mapping[tuple[str, int], float],
    closure_rates: Mapping[str, float],
    deduction_resource: Mapping[str, Any] | None = None,
    minimum_pool_sources: int = UC_TAKE_UP_MINIMUM_POOL_SOURCES,
    young_target: float | None = None,
) -> UKUCTakeUpResult:
    """Redraw ``would_claim_uc`` by award band and family type.

    With ``young_target``, DWP's people on Universal Credit aged 16 to 24, the
    age fit runs on each unit's claimants and partners of those ages
    (``UC_TAKE_UP_YOUNG_PEOPLE``); the stage always passes it.
    """

    validate_uk_national_frame(frame)
    assert_rules_engine_country(engine, "uk")
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    for table, columns, label in (
        (
            person,
            (
                "person_benunit_id",
                "person_household_id",
                "universal_credit_reported",
                UK_UC_AGE_ELIGIBLE_SOURCE,
                UK_UC_AGE_ELIGIBLE_ROLE,
            ),
            "person",
        ),
        (benunit, ("benunit_id", UK_UC_TAKE_UP_OUTPUT), "benunit"),
        (household, ("household_id", "region"), "household"),
    ):
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise ValueError(f"UC take-up {label} columns missing: {missing}.")
    if young_target is not None and UC_TAKE_UP_YOUNG_ROLE not in person.columns:
        raise ValueError(
            f"UC take-up's age fit reads person.{UC_TAKE_UP_YOUNG_ROLE}, which "
            "the frame does not carry."
        )
    period = uk_time_period(frame)
    declared = resolve_uk_year_rule(UC_TAKE_UP_DATASET_YEAR_RULE)
    if int(period) != declared:
        raise ValueError(
            f"UC take-up reads the frame as {UC_TAKE_UP_DATASET_YEAR_RULE} "
            f"{declared} values, but the frame's period is {period}."
        )
    entitlement_year = resolve_uk_year_rule(UC_TAKE_UP_ENTITLEMENT_YEAR_RULE)
    # The award a unit would be paid if it claimed: every unit claims on a
    # temporary copy that carries the deduction attributes the later stage
    # assigns from each unit's identity and region, so the engine applies the
    # benefit cap and deductions as it does to a claimant on the released file.
    deductions = assign_uc_deduction_attributes(
        frame, resource=deduction_resource or load_uc_deduction_distributions()
    ).frame
    claiming = deductions.table("benunit").copy()
    claiming[UK_UC_TAKE_UP_OUTPUT] = True
    assessed = uk_national_frame(
        person=deductions.table("person"),
        benunit=claiming,
        household=deductions.table("household"),
        time_period=period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    # The frame holds survey-year values; stamped with its own period, the
    # engine projects them to the entitlement year as it does the released
    # file, so the award is the one a calibration-year run pays.
    materialized = engine.materialize(
        assessed,
        list(UC_TAKE_UP_ENTITLEMENT_VARIABLES),
        entitlement_year,
        dataset_period=period,
    )
    missing = sorted(set(UC_TAKE_UP_ENTITLEMENT_VARIABLES) - set(materialized))
    if missing:
        raise ValueError(f"UC take-up engine outputs missing: {missing}.")
    count = len(benunit)
    award = _aligned(
        materialized["universal_credit"], count, "universal_credit"
    ).astype(float)
    moved = _aligned(
        materialized["legacy_benefits_closed"], count, "legacy_benefits_closed"
    ).astype(bool)
    qualifying = _aligned(
        materialized["is_child_or_qualifying_young_person_for_universal_credit"],
        len(person),
        "is_child_or_qualifying_young_person_for_universal_credit",
    ).astype(bool)
    family = uc_family_types(person, benunit, qualifying)
    reported = (
        pd.to_numeric(person["universal_credit_reported"], errors="coerce").fillna(0.0)
        > 0.0
    )
    reporter = (
        benunit["benunit_id"]
        .isin(set(person.loc[reported, "person_benunit_id"]))
        .to_numpy(dtype=bool)
    )
    population = uc_age_eligible_benunits(person, benunit, population_policy)
    stored = benunit[UK_UC_TAKE_UP_OUTPUT].fillna(False).to_numpy(dtype=bool)
    # policyengine-uk ends a claimant's or partner's income-related ESA once
    # the family claims Universal Credit (esa_income_report_closed, from
    # 2.149.0; pe-uk#2563). A unit whose claimant or partner reports the award,
    # and that neither reports Universal Credit nor is moved off its legacy
    # benefits, was on that award at the interview, so it claims no Universal
    # Credit: a reported receipt is a fact (microcosm#1095). The closure draw
    # reads the flag this stage writes, so the closure probabilities below use
    # it too.
    esa_kept = (
        _reported_by_benunit(person, benunit, UC_TAKE_UP_KEPT_ESA_COLUMN)
        & ~moved
        & ~reporter
    )
    previous = stored & ~esa_kept
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=frame.weights_for("household").values,
    )
    unit_households = pd.Series(_benunit_households(benunit, person=person))
    regions = unit_households.map(household.set_index("household_id")["region"])
    if regions.isna().any():
        raise ValueError("Every benefit unit's household must carry a region.")
    great_britain = regions.astype(str).ne(_OUTSIDE_SOLVE_REGION).to_numpy(dtype=bool)
    source_ids = unit_households.map(
        pd.Series(
            source_household_ids(household, stage="The UC take-up stage"),
            index=household["household_id"].to_numpy(),
        )
    ).to_numpy()
    # A unit the engine moves off its legacy benefits claims at the closure
    # with the probability ``uc_legacy_closure_take_up`` gives it; this stage
    # leaves its flag alone, so the closure's share already claiming, and so
    # that probability, does not move.
    combination = legacy_benefit_combinations(person, benunit)
    closure_drawn = ~reporter & (combination != "")
    closure_probability, _ = legacy_closure_claim_probabilities(
        combination=combination,
        drawn=closure_drawn,
        claims_anyway=previous,
        weights=weights,
        rates=closure_rates,
        great_britain=great_britain,
        source_ids=source_ids,
    )
    moved_claim = np.where(
        previous | ~closure_drawn, previous.astype(float), closure_probability
    )
    draws = stable_identity_uniforms(
        benunit["benunit_id"].to_numpy(),
        seed=UC_TAKE_UP_SEED,
        salt=UK_UC_TAKE_UP_OUTPUT,
    )
    young_people = None
    if young_target is not None:
        # The ages the diagnostic reads: policyengine-uk carries the stored age
        # unchanged into every projected year, so the survey-year age is the
        # entitlement year's.
        age = pd.to_numeric(person[UK_UC_AGE_ELIGIBLE_SOURCE], errors="coerce")
        role = person[UC_TAKE_UP_YOUNG_ROLE].fillna(False).to_numpy(dtype=bool)
        young = (
            role
            & (age >= min(UC_TAKE_UP_YOUNG_AGES)).to_numpy()
            & (age <= max(UC_TAKE_UP_YOUNG_AGES)).to_numpy()
        )
        young_people = (
            pd.Series(young.astype(np.float64))
            .groupby(person["person_benunit_id"].to_numpy())
            .sum()
            .reindex(benunit["benunit_id"].to_numpy())
            .fillna(0.0)
            .to_numpy()
        )
    would_claim, receipt = assign_award_band_take_up(
        award=award,
        family=family,
        population=population,
        reporter=reporter,
        moved=moved,
        moved_claim=moved_claim,
        previous=previous,
        weights=weights,
        draws=draws,
        great_britain=great_britain,
        source_ids=source_ids,
        targets=targets,
        minimum_pool_sources=minimum_pool_sources,
        legacy_kept=(combination != "") & ~moved & ~reporter,
        young_people=young_people,
        young_target=young_target,
    )
    kept_off = esa_kept & stored & great_britain
    esa_kept_receipt = {
        "scope": "great_britain",
        "column": UC_TAKE_UP_KEPT_ESA_COLUMN,
        "rule": UC_TAKE_UP_KEPT_ESA_RULE,
        "units_kept_off_universal_credit": small_cell_share(
            float(weights[kept_off].sum()),
            distinct_count(source_ids, kept_off, weights),
        ),
        "kept_off_support": support_band(distinct_count(source_ids, kept_off, weights)),
    }
    working_age_hb = working_age_housing_benefit_receipt(
        person=person,
        benunit=benunit,
        population=population,
        moved=moved,
        uc_before=stored | reporter,
        uc_after=would_claim | reporter,
        weights=weights,
        great_britain=great_britain,
        source_ids=source_ids,
    )
    benunit[UK_UC_TAKE_UP_OUTPUT] = would_claim
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=frame.weights_for("household").values,
        mass_log=frame.mass_log,
    )
    validate_uk_national_frame(result)
    return UKUCTakeUpResult(
        frame=result,
        receipt={
            **receipt,
            "population_state_pension_age": int(population_policy.state_pension_age),
            "working_age_housing_benefit": working_age_hb,
            "income_related_esa_kept": esa_kept_receipt,
        },
        entitlement_year=entitlement_year,
    )


def working_age_housing_benefit_receipt(
    *,
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    population: np.ndarray,
    moved: np.ndarray,
    uc_before: np.ndarray,
    uc_after: np.ndarray,
    weights: np.ndarray,
    great_britain: np.ndarray,
    source_ids: np.ndarray,
) -> dict[str, object]:
    """Working-age Housing Benefit the redraw keeps, at the stage's weights.

    policyengine-uk continues an existing Housing Benefit award only while the
    unit neither claims Universal Credit nor is moved off its legacy benefits,
    so a working-age unit reporting Housing Benefit that the redraw sets to
    claim UC loses it. This counts, over Great Britain, the benefit units of
    the UC age population with a member reporting Housing Benefit that the
    engine does not move, and how many of them keep the award before and after
    the redraw, so the stage's effect on working-age Housing Benefit can be set
    beside DWP's 2025 caseload (Vahid's review of #1344, item 7). Weighted
    figures only where three or more source households carry them.
    """

    column = "housing_benefit_reported"
    reported = (
        pd.to_numeric(person[column], errors="coerce").fillna(0.0) > 0.0
        if column in person.columns
        else pd.Series(False, index=person.index)
    )
    reporting = (
        benunit["benunit_id"]
        .isin(set(person.loc[reported.to_numpy(dtype=bool), "person_benunit_id"]))
        .to_numpy(dtype=bool)
    )
    base = reporting & population & ~moved & great_britain
    kept_before = base & ~uc_before
    kept_after = base & ~uc_after
    drawn_onto_uc = kept_before & uc_after

    def weighted(mask: np.ndarray) -> object:
        return small_cell_share(
            float(weights[mask].sum()), distinct_count(source_ids, mask, weights)
        )

    return {
        "scope": "great_britain",
        "reporting_units": weighted(base),
        "keeping_housing_benefit_before": weighted(kept_before),
        "keeping_housing_benefit_after": weighted(kept_after),
        "drawn_onto_universal_credit": weighted(drawn_onto_uc),
        "drawn_support": support_band(
            distinct_count(source_ids, drawn_onto_uc, weights)
        ),
    }


def _aligned(values: object, expected: int, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (expected,):
        raise ValueError(f"{label} must align to its entity table.")
    if array.dtype != bool and not np.isfinite(array.astype(np.float64)).all():
        raise ValueError(f"{label} must be finite.")
    return array


def uc_take_up_operation_parameters() -> tuple[tuple[str, dict[str, object]], ...]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "verify_vendored_fact_resource",
            {
                "artifact_role": "uc_award_band_facts",
                "resource": UC_TAKE_UP_RESOURCE,
                "feed_pin": "chronicle_feed.json",
                "record_sets": list(UC_TAKE_UP_RECORD_SETS),
                "source_vintage": "April to December 2025",
                "mapped_build_period": 2025,
                "period_mapping": "calendar_year_average_of_monthly_counts",
                "require_before_source_read": True,
                "runtime_sha256_required": True,
                "fail_on_mismatch": True,
            },
        ),
        (
            "verify_vendored_fact_resource",
            {
                "artifact_role": "uc_people_by_age_facts",
                "resource": UC_TAKE_UP_AGE_RESOURCE,
                "feed_pin": "chronicle_feed.json",
                "record_sets": list(UC_TAKE_UP_AGE_RECORD_SETS),
                "source_vintage": "February, May, August and November 2025",
                "mapped_build_period": 2025,
                "period_mapping": "mean_of_four_quarterly_monthly_counts_in_2025",
                "require_before_source_read": True,
                "runtime_sha256_required": True,
                "fail_on_mismatch": True,
            },
        ),
        (
            "materialize_rules_engine_predictors",
            {
                "predictors": list(UC_TAKE_UP_ENTITLEMENT_VARIABLES),
                "consumed_only": True,
                "year_rule": UC_TAKE_UP_ENTITLEMENT_YEAR_RULE,
                "dataset_year_rule": UC_TAKE_UP_DATASET_YEAR_RULE,
            },
        ),
        (
            "aggregate_person_to_benunit",
            {
                "method": "any_positive",
                "consumed_only": True,
                "aggregates": {
                    "universal_credit_reported_anchor": "universal_credit_reported"
                },
            },
        ),
        (
            "aggregate_person_to_benunit",
            {
                "method": UK_UC_AGE_ELIGIBLE_METHOD,
                "consumed_only": True,
                "aggregates": {UK_UC_AGE_ELIGIBLE_AGGREGATE: UK_UC_AGE_ELIGIBLE_SOURCE},
                "role_column": UK_UC_AGE_ELIGIBLE_ROLE,
            },
        ),
        (
            UC_TAKE_UP_OPERATION_KIND,
            {
                "output": UK_UC_TAKE_UP_OUTPUT,
                "draw": UK_UC_TAKE_UP_OUTPUT,
                "seed": UC_TAKE_UP_SEED,
                "anchor": "universal_credit_reported_anchor",
                "population": UK_UC_AGE_ELIGIBLE_AGGREGATE,
                "award": (
                    "universal_credit on a temporary copy where every unit claims "
                    "and carries the deduction attributes uc_deduction_attributes "
                    "assigns from its identity and region"
                ),
                "family_type": (
                    "two FRS claimant or partner roles make a couple; children "
                    "are UC children or qualifying young people and other "
                    "non-claimants under 20 (the calibration measure's "
                    "uc_calibration_family_type)"
                ),
                "award_bands": (
                    "GBP 1,200 a year (DWP's GBP 100 a month), (1,200 i, "
                    "1,200 (i + 1)] in whole pence for i < 25, and above GBP 30,000"
                ),
                "targets": (
                    "DWP Stat-Xplore Households on Universal Credit by family type "
                    "and monthly award band, mean of April to December 2025"
                ),
                "pooling": UC_TAKE_UP_POOLING,
                "minimum_pool_sources": UC_TAKE_UP_MINIMUM_POOL_SOURCES,
                "drawn_units": (
                    "entitled, non-reporting units of the population that the "
                    "engine does not move off their legacy benefits "
                    "(legacy_benefits_closed) and that report none, since a "
                    "legacy benefit the engine leaves in place ends once the "
                    "unit claims UC"
                ),
                "undrawn_claims": (
                    "a reporter at 1, a moved unit at the claim probability "
                    "uc_legacy_closure_take_up's rule gives it "
                    f"({LEGACY_CLOSURE_RESOURCE}), a unit keeping income-related "
                    "ESA at 0, any other at its stored flag"
                ),
                "kept_legacy_award": UC_TAKE_UP_KEPT_ESA_RULE,
                "weight_mapping": "household_to_benunit",
                "residual_rate": UC_TAKE_UP_RESIDUAL_RATE,
                "solve_scope": UC_TAKE_UP_SOLVE_SCOPE,
                "not_drawn": (
                    "keeps its stored flag, except a unit keeping income-related "
                    "ESA, which claims none"
                ),
                "young_people": UC_TAKE_UP_YOUNG_PEOPLE,
                "age_targets": UC_TAKE_UP_AGE_TARGET,
                "age_fit": UC_TAKE_UP_AGE_FIT,
            },
        ),
    )


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    if stage.stage != UC_TAKE_UP_STAGE_NAME:
        raise ValueError(
            f"UC take-up received stage {stage.stage!r}, expected "
            f"{UC_TAKE_UP_STAGE_NAME!r}."
        )
    _assert_closed_world_operations(stage, uc_take_up_operation_parameters())
    if tuple(stage.outputs) or tuple(stage.rewrites) != (UK_UC_TAKE_UP_OUTPUT,):
        raise ValueError(
            f"{UC_TAKE_UP_STAGE_NAME} must declare no outputs and exactly the "
            f"{UK_UC_TAKE_UP_OUTPUT} rewrite."
        )
    if stage.grain != "benunit":
        raise ValueError(
            f"{UC_TAKE_UP_STAGE_NAME} is benunit grain, got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKUCTakeUpStageTransform:
    """Whole-stage callable for the post-SPI Universal Credit take-up redraw."""

    stage: SourceStageSpec
    engine: object
    population_policy: UKTakeUpPopulationPolicy | None = None
    #: DWP's counts and the Move to UC rates; read from the vendored facts
    #: when not given.
    targets: Mapping[tuple[str, int], float] | None = field(default=None, repr=False)
    closure_rates: Mapping[str, float] | None = field(default=None, repr=False)
    deduction_resource: Mapping[str, Any] | None = field(default=None, repr=False)
    #: DWP's people on Universal Credit aged 16 to 24; read from the vendored
    #: facts when not given.
    young_target: float | None = field(default=None, repr=False)
    last_result: UKUCTakeUpResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = redraw_uc_take_up(
            frame,
            engine=self.engine,
            population_policy=(
                self.population_policy
                or uk_take_up_population_policy(uk_time_period(frame))
            ),
            targets=(
                self.targets
                if self.targets is not None
                else load_uc_award_band_targets()
            ),
            closure_rates=(
                self.closure_rates
                if self.closure_rates is not None
                else load_move_to_uc_claim_rates()
            ),
            deduction_resource=self.deduction_resource,
            young_target=(
                self.young_target
                if self.young_target is not None
                else load_uc_young_people_target()
            ),
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return ()

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
