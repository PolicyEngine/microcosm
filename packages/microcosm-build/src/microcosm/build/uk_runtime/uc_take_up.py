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
    "UC_TAKE_UP_ENTITLEMENT_VARIABLES",
    "UC_TAKE_UP_MINIMUM_POOL_SOURCES",
    "UC_TAKE_UP_OPERATION_KIND",
    "UC_TAKE_UP_RECORD_SETS",
    "UC_TAKE_UP_RESOURCE",
    "UC_TAKE_UP_STAGE_NAME",
    "UKUCTakeUpResult",
    "UKUCTakeUpStageTransform",
    "assign_award_band_take_up",
    "load_uc_award_band_targets",
    "redraw_uc_take_up",
    "uc_award_band_targets_from_rows",
    "uc_award_bands",
    "uc_family_types",
    "uc_take_up_operation_parameters",
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
UC_TAKE_UP_RESIDUAL_RATE = (
    "(T - F) / D, clipped to [0, 1]; T DWP's calendar-2025 mean count of the "
    "pool's bands, D the weight of the pool's drawn units and F the expected "
    "claims of its other entitled units, over Great Britain at the stage's "
    "household weights; realized by the drawn units claiming in the order of "
    "their uniform until their weight is nearest T - F, and applied at that "
    "uniform to Northern Ireland's drawn units too"
)


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
) -> tuple[np.ndarray, dict[str, object]]:
    """Reporters claim; drawn units claim at their pool's residual rate.

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

    pools: list[dict[str, object]] = []
    for name, _ in UC_FAMILY_TYPES:
        in_family = family == name
        for bands in _pools(
            band=band,
            in_family=in_family,
            drawable_gb=drawable & gb,
            source_ids=source_ids,
            weights=weights,
            minimum_sources=minimum_pool_sources,
        ):
            in_pool = in_family & np.isin(band, bands)
            solve = in_pool & gb
            target = float(sum(targets[(name, index)] for index in bands))
            undrawn = float((weights * fixed)[solve & ~drawable].sum())
            drawn = float(weights[solve & drawable].sum())
            # Where the undrawn claims reach DWP's count, or the count exceeds
            # every entitled unit, the clipped rate cannot meet it.
            if drawn > 0.0:
                residual = (target - undrawn) / drawn
                rates[in_pool & drawable] = _realized_rate(
                    residual,
                    need=target - undrawn,
                    draws=draws[solve & drawable],
                    weights=weights[solve & drawable],
                )
            else:
                residual = 0.0 if undrawn >= target else 1.0
            unreachable = (
                "undrawn_claims_reach_target"
                if residual <= 0.0
                else "target_exceeds_entitled"
                if residual >= 1.0
                else None
            )
            pools.append(
                {
                    "family_type": name,
                    "bands": bands,
                    "mask": solve,
                    "target": target,
                    "unreachable": unreachable,
                }
            )
    would_claim = np.where(drawable, draws < rates, previous | reporter)
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
                    # claims lie, given only where ten or more sources carry
                    # the pool (UKDS disclosure control).
                    "target_unreachable": (
                        pool["unreachable"] if supported >= 10 else "<10"
                    ),
                }
            )
            if pool["unreachable"] is None:
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
) -> UKUCTakeUpResult:
    """Redraw ``would_claim_uc`` by award band and family type."""

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
    figures only where ten or more source households carry them.
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
                "pooling": (
                    "adjacent bands within a family type, lowest first, until the "
                    "drawn units rest on minimum_pool_sources distinct "
                    "household_source_id values; a short last pool joins the one "
                    "before it"
                ),
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
