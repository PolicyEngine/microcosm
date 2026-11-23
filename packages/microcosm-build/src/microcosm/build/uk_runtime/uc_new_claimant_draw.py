"""Universal Credit's new-claimant health element draw, by claim recency (microcosm#1095).

From 6 April 2026 the LCWRA element of a claim made on or after that date is
paid at the lower new-claimant rate (UC Regs 2013 reg. 36; Universal Credit
Act 2025 s. 2). policyengine-uk 2.179.0 (policyengine-uk#2877) reads a stored
uniform per benefit unit, ``uc_new_claimant_health_element_draw``: in each
rebalancing year a unit with the element gets the new-claimant rate when its
draw falls below ``gov.dwp.universal_credit.rebalancing.new_claimant_share``.
Without it the engine draws in benefit-unit order, so which claimants get the
lower rate is random with respect to everything the file records, though
young claimants are mostly recent claims.

This stage, the last of the spine, so that every writer of an input the award
reads has run, stores the draw so that the claims most likely to be new get
the lowest draws, and keeps it uniform:

* One engine materialization at the calibration year, on the frame as it
  stands, gives each benefit unit's Universal Credit award and LCWRA element.
  The units paid both are the draw's population, the units DWP's paid LCWRA
  households count.
* The population is ordered by claim recency, the completed months from the
  claim's start to the interview, fewest first: ``frs_uc_claim_months``,
  which ``frs_spine`` reads from DWP's claim start (BENEFITS.UCSTART) for a
  linked FRS claim. Ties go to the unit whose youngest claimant or partner is
  younger, then to a uniform keyed on the unit's identity.
* A unit with no linked claim start (an unlinked reporter, a take-up
  entrant, a legacy-closure mover, or any unit of an SPI-channel household)
  takes months drawn from the linked claims of the population whose youngest
  claimant or partner is in the same age band, at the band's weighted
  distribution, on a uniform keyed on its identity. Adjacent bands pool,
  youngest first, until the linked claims rest on 30 distinct source
  households.
* Each unit of the population takes the midpoint of its slice of the
  population's weight, in that order, over the population's weight. At the
  stage's weights the draw is then uniform on [0, 1) within the population:
  the share below any number is that number, to within one unit's weight.
  Every other unit takes a uniform keyed on its identity, which is uniform
  too, so the draw is uniform overall, and the engine keeps the parameter's
  share in every year while the units below it are the most recent claims.

The draw is stored as a survey-year input, which the engine carries into every
projected year, so each year's new claimants are the units below that year's
share, a set that grows with the share. Two limits: the calibration reweights
the units, so the share at the release's weights moves with how it reweights
recent claims; and a unit that has the element only in a later year, such as
one moved off income-related ESA from 2027-28, takes the identity uniform, so
its status is random, as the engine's own draw is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.frs_spine import (
    FRS_UC_CLAIM_MONTHS_COLUMN,
    FRS_UC_CLAIM_MONTHS_NOT_LINKED,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    record_source_households,
    rounded_count,
    small_cell_share,
    support_band,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
    _benunit_households,
    _household_to_benunit_weights,
)
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "UC_NEW_CLAIMANT_AGE_BANDS",
    "UC_NEW_CLAIMANT_DRAW_OUTPUT",
    "UC_NEW_CLAIMANT_DRAW_STAGE_NAME",
    "UC_NEW_CLAIMANT_ENGINE_VARIABLES",
    "UKUCNewClaimantDrawStageTransform",
    "assign_uc_new_claimant_draw",
    "claim_month_pools",
    "uc_new_claimant_draw_operation_parameters",
    "weighted_quantile_draws",
    "weighted_rank_midpoints",
]

UC_NEW_CLAIMANT_DRAW_STAGE_NAME = "uc_new_claimant_draw"
UC_NEW_CLAIMANT_DRAW_OPERATION_KIND = "assign_uc_new_claimant_draw"
UC_NEW_CLAIMANT_DRAW_OUTPUT = "uc_new_claimant_health_element_draw"
UC_NEW_CLAIMANT_DRAW_SEED = 0
#: The identity-keyed streams: the output's own for units outside the
#: population, one for ties in the order, one for the imputed claim months.
UC_NEW_CLAIMANT_ORDER_DRAW = "uc_new_claimant_order"
UC_NEW_CLAIMANT_CLAIM_MONTHS_DRAW = "uc_new_claimant_claim_months"
UC_NEW_CLAIMANT_AWARD = "universal_credit"
UC_NEW_CLAIMANT_ELEMENT = "uc_LCWRA_element"
#: What the stage asks the engine, for every benefit unit, on the frame as it
#: stands, at the calibration year.
UC_NEW_CLAIMANT_ENGINE_VARIABLES: tuple[str, ...] = (
    UC_NEW_CLAIMANT_AWARD,
    UC_NEW_CLAIMANT_ELEMENT,
)
UC_NEW_CLAIMANT_YEAR_RULE = "calibration_year"
#: The frame holds survey-year values, and the draw is stored as a survey-year
#: input, which the engine carries into every projected year.
UC_NEW_CLAIMANT_DATASET_YEAR_RULE = "survey_year"
UC_NEW_CLAIMANT_AGE_ROLE = "is_claimant_or_partner"
#: Every unit of an SPI-channel household is a copy with redrawn incomes and
#: benefit reports, so none has a linked claim start, whatever it copied.
UC_NEW_CLAIMANT_LINKED_SCOPE = "household_is_spi_synthetic"
#: The age bands of a unit's youngest claimant or partner that the imputed
#: claim months pool within, youngest first, as (lower, upper) whole years;
#: the last has no upper bound, and an age below the first band's lower bound
#: falls in it.
UC_NEW_CLAIMANT_AGE_BANDS: tuple[tuple[int, int | None], ...] = (
    (16, 24),
    (25, 34),
    (35, 44),
    (45, 54),
    (55, None),
)
UC_NEW_CLAIMANT_MINIMUM_POOL_SOURCES = 30
#: The receipt's probes of the draw: the share of the population, and of its
#: units whose youngest claimant or partner is in the first band, below each.
UC_NEW_CLAIMANT_PROBES: tuple[float, ...] = (0.1, 0.2, 0.3)

UC_NEW_CLAIMANT_POPULATION_RULE = (
    "benefit units the locked engine pays universal_credit with "
    "uc_LCWRA_element above zero at the calibration year, on the frame as it "
    "stands, its survey-year values projected by policyengine-uk"
)
UC_NEW_CLAIMANT_ORDER_RULE = (
    "claim recency first, the completed months from the claim's start to the "
    "interview, fewest first (frs_uc_claim_months on a linked claim, the "
    "imputed months otherwise); then the age of the unit's youngest claimant "
    "or partner (is_claimant_or_partner, at the stored age), youngest first; "
    "then the order draw, a uniform keyed on benunit_id"
)
UC_NEW_CLAIMANT_IMPUTATION_RULE = (
    "a unit of the population without a linked claim start "
    "(frs_uc_claim_months at not_linked, or a unit of an SPI-channel "
    "household, household_is_spi_synthetic) takes the months of the "
    "population's linked claims whose youngest claimant or partner is in its "
    "age band, at the household-weighted quantile of the claim months draw, a "
    "uniform keyed on benunit_id; adjacent bands pool, youngest first, until "
    "the linked claims rest on minimum_pool_sources distinct source "
    "households, and a short last pool joins the one before it"
)
UC_NEW_CLAIMANT_DRAW_RULE = (
    "each unit of the population takes the midpoint of its slice of the "
    "population's benefit-unit weight in that order, over the population's "
    "weight, so the weighted share of the population below any u in [0, 1) is "
    "u to within one unit's weight; every other unit takes a uniform keyed on "
    "benunit_id, the output's own stream, so the draw is uniform overall. "
    "policyengine-uk carries the stored survey-year draw into every projected "
    "year and gives a unit with the LCWRA element the new-claimant rate where "
    "it falls below that year's gov.dwp.universal_credit.rebalancing."
    "new_claimant_share"
)
UC_NEW_CLAIMANT_MASS_CHANGE_REASON = (
    "UC new-claimant health element draw stored on the source spine: household "
    "weights pass through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)


def _band_label(lower: int, upper: int | None) -> str:
    return f"{lower}+" if upper is None else f"{lower}-{upper}"


def uc_new_claimant_draw_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "materialize_rules_engine_predictors",
            {
                "predictors": list(UC_NEW_CLAIMANT_ENGINE_VARIABLES),
                "consumed_only": True,
                "year_rule": UC_NEW_CLAIMANT_YEAR_RULE,
                "dataset_year_rule": UC_NEW_CLAIMANT_DATASET_YEAR_RULE,
            },
        ),
        (
            UC_NEW_CLAIMANT_DRAW_OPERATION_KIND,
            {
                "output": UC_NEW_CLAIMANT_DRAW_OUTPUT,
                "draw": UC_NEW_CLAIMANT_DRAW_OUTPUT,
                "order_draw": UC_NEW_CLAIMANT_ORDER_DRAW,
                "claim_months_draw": UC_NEW_CLAIMANT_CLAIM_MONTHS_DRAW,
                "seed": UC_NEW_CLAIMANT_DRAW_SEED,
                "claim_months": FRS_UC_CLAIM_MONTHS_COLUMN,
                "not_linked": FRS_UC_CLAIM_MONTHS_NOT_LINKED,
                "linked_scope": UC_NEW_CLAIMANT_LINKED_SCOPE,
                "age_role": UC_NEW_CLAIMANT_AGE_ROLE,
                "age_bands": [
                    {"lower": lower}
                    if upper is None
                    else {"lower": lower, "upper": upper}
                    for lower, upper in UC_NEW_CLAIMANT_AGE_BANDS
                ],
                "minimum_pool_sources": UC_NEW_CLAIMANT_MINIMUM_POOL_SOURCES,
                "weight_mapping": "household_to_benunit",
                "population": UC_NEW_CLAIMANT_POPULATION_RULE,
                "order": UC_NEW_CLAIMANT_ORDER_RULE,
                "imputation": UC_NEW_CLAIMANT_IMPUTATION_RULE,
                "rule": UC_NEW_CLAIMANT_DRAW_RULE,
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": UC_NEW_CLAIMANT_MASS_CHANGE_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


def age_band_indices(ages: np.ndarray) -> np.ndarray:
    """Each age's band in ``UC_NEW_CLAIMANT_AGE_BANDS`` (the first below it)."""

    ages = np.asarray(ages, dtype=np.float64)
    indices = np.zeros(ages.shape, dtype=np.int64)
    for index, (lower, _) in enumerate(UC_NEW_CLAIMANT_AGE_BANDS):
        indices[ages >= lower] = index
    return indices


def claim_month_pools(
    bands: np.ndarray,
    donors: np.ndarray,
    *,
    sources: np.ndarray,
    weights: np.ndarray,
    minimum_sources: int = UC_NEW_CLAIMANT_MINIMUM_POOL_SOURCES,
) -> tuple[tuple[int, ...], ...]:
    """Pool adjacent age bands, youngest first, until the donors rest on enough.

    ``bands`` holds each unit's band index and ``donors`` marks the units whose
    months the pool's other units draw from. A pool closes once its donors rest
    on ``minimum_sources`` distinct source households; a short last pool joins
    the one before it, and with no full pool every band forms one.
    """

    bands = np.asarray(bands)
    donors = np.asarray(donors, dtype=bool)
    pools: list[tuple[int, ...]] = []
    current: list[int] = []
    for band in range(len(UC_NEW_CLAIMANT_AGE_BANDS)):
        current.append(band)
        members = donors & np.isin(bands, current)
        if distinct_count(sources, members, weights) >= minimum_sources:
            pools.append(tuple(current))
            current = []
    if current:
        if pools:
            pools[-1] = (*pools[-1], *current)
        else:
            pools.append(tuple(current))
    return tuple(pools)


def weighted_quantile_draws(
    values: np.ndarray, weights: np.ndarray, uniforms: np.ndarray
) -> np.ndarray:
    """The ``values`` at each uniform's quantile of their weighted distribution.

    The inverse of the weighted distribution function: a uniform u takes the
    smallest value whose cumulative weight share exceeds u. A value of zero
    weight is never taken unless every weight is zero, when each counts once.
    """

    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if values.size == 0:
        raise ValueError("A weighted quantile draw needs at least one value.")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    if not (sorted_weights > 0.0).any():
        sorted_weights = np.ones_like(sorted_weights)
    cumulative = np.cumsum(sorted_weights)
    targets = uniforms * cumulative[-1]
    positions = np.searchsorted(cumulative, targets, side="right")
    return sorted_values[np.minimum(positions, sorted_values.size - 1)]


def weighted_rank_midpoints(order: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Each unit's midpoint of its slice of the total weight, in ``order``.

    ``order`` lists the units' positions, first to last. The result, indexed as
    ``weights``, lies in [0, 1); with every weight zero the units share it
    evenly.
    """

    order = np.asarray(order)
    weights = np.asarray(weights, dtype=np.float64)
    ordered = weights[order]
    total = float(ordered.sum())
    if total > 0.0:
        start = np.cumsum(ordered) - ordered
        midpoints = (start + 0.5 * ordered) / total
    else:
        midpoints = (np.arange(ordered.size, dtype=np.float64) + 0.5) / max(
            ordered.size, 1
        )
    result = np.empty(weights.shape, dtype=np.float64)
    result[order] = np.minimum(midpoints, np.nextafter(1.0, 0.0))
    return result


def _aligned(values: object, count: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (count,):
        raise ValueError(
            f"The engine's {name} has shape {array.shape}, not ({count},)."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"The engine's {name} must be finite.")
    return array


@dataclass(frozen=True)
class UKUCNewClaimantDrawResult:
    """The frame with the stored draw, and the stage's receipt.

    ``claim_months`` holds the months each benefit unit of the population was
    ordered by (observed or imputed; infinite where neither exists) and NaN
    outside it. It stays in the build and is no part of the receipt.
    """

    frame: Frame
    receipt: Mapping[str, object]
    claim_months: np.ndarray

    def evidence(self) -> dict[str, object]:
        return {"stage": UC_NEW_CLAIMANT_DRAW_STAGE_NAME, **dict(self.receipt)}


def assign_uc_new_claimant_draw(
    frame: Frame,
    *,
    engine_answers: Mapping[str, object],
    entitlement_year: int | None = None,
) -> UKUCNewClaimantDrawResult:
    """Store the draw: ordered by claim recency in the population, uniform overall.

    ``engine_answers`` holds the engine's benefit-unit answers for
    ``UC_NEW_CLAIMANT_ENGINE_VARIABLES`` on ``frame`` as it stands;
    ``entitlement_year`` is the year they were read at, for the receipt.
    """

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    if UC_NEW_CLAIMANT_DRAW_OUTPUT in benunit.columns:
        raise ValueError(
            f"{UC_NEW_CLAIMANT_DRAW_OUTPUT!r} is already stored: this stage is its "
            "only writer."
        )
    for table, columns, label in (
        (benunit, ("benunit_id", FRS_UC_CLAIM_MONTHS_COLUMN), "benunit"),
        (
            person,
            (
                "person_benunit_id",
                "person_household_id",
                "age",
                UC_NEW_CLAIMANT_AGE_ROLE,
            ),
            "person",
        ),
        (household, ("household_id", UC_NEW_CLAIMANT_LINKED_SCOPE), "household"),
    ):
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise KeyError(f"The UC new-claimant draw needs {label} columns {missing}.")
    ids = benunit["benunit_id"].to_numpy()
    if pd.Series(ids).duplicated().any():
        raise ValueError(
            "benunit_id must be unique: the new-claimant draw is keyed on it."
        )
    count = len(benunit)
    award = _aligned(
        engine_answers[UC_NEW_CLAIMANT_AWARD], count, UC_NEW_CLAIMANT_AWARD
    )
    element = _aligned(
        engine_answers[UC_NEW_CLAIMANT_ELEMENT], count, UC_NEW_CLAIMANT_ELEMENT
    )
    population = (award > 0.0) & (element > 0.0)

    months = pd.to_numeric(
        benunit[FRS_UC_CLAIM_MONTHS_COLUMN], errors="raise"
    ).to_numpy(dtype=np.float64)
    valid = np.isfinite(months) & (
        (months == FRS_UC_CLAIM_MONTHS_NOT_LINKED)
        | ((months >= 0.0) & (months == np.floor(months)))
    )
    if not valid.all():
        raise ValueError(
            f"{FRS_UC_CLAIM_MONTHS_COLUMN} holds {int((~valid).sum())} value(s) that "
            f"are neither whole months nor {FRS_UC_CLAIM_MONTHS_NOT_LINKED}."
        )
    unit_households = pd.Series(_benunit_households(benunit, person=person))
    spi = (
        unit_households.map(
            household.set_index("household_id")[UC_NEW_CLAIMANT_LINKED_SCOPE]
        )
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    linked = (months >= 0.0) & ~spi
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=frame.weights_for("household").values,
    )
    sources = record_source_households(
        {"person": person, "benunit": benunit, "household": household},
        "benunit",
        stage=UC_NEW_CLAIMANT_DRAW_STAGE_NAME,
    )

    role = person[UC_NEW_CLAIMANT_AGE_ROLE].fillna(False).to_numpy(dtype=bool)
    age = pd.to_numeric(person["age"], errors="raise").to_numpy(dtype=np.float64)
    youngest = (
        pd.Series(np.where(role, age, np.inf))
        .groupby(person["person_benunit_id"].to_numpy())
        .min()
        .reindex(ids)
        .fillna(np.inf)
        .to_numpy(dtype=np.float64)
    )
    if np.isinf(youngest[population]).any():
        raise ValueError(
            "A benefit unit the engine pays Universal Credit has no claimant or "
            f"partner ({UC_NEW_CLAIMANT_AGE_ROLE}) to order the draw by."
        )
    bands = age_band_indices(np.where(np.isinf(youngest), 0.0, youngest))

    # Claim months for the units of the population without a linked start.
    donors = population & linked
    imputed = population & ~linked
    claim_months = np.where(linked, months, np.inf)
    pools = claim_month_pools(
        bands,
        donors,
        sources=sources,
        weights=weights,
        minimum_sources=UC_NEW_CLAIMANT_MINIMUM_POOL_SOURCES,
    )
    imputation_uniforms = stable_identity_uniforms(
        ids, seed=UC_NEW_CLAIMANT_DRAW_SEED, salt=UC_NEW_CLAIMANT_CLAIM_MONTHS_DRAW
    )
    pool_receipts: list[dict[str, object]] = []
    for pool in pools:
        in_pool = np.isin(bands, pool)
        pool_donors = donors & in_pool
        takers = imputed & in_pool
        if takers.any() and pool_donors.any():
            claim_months[takers] = weighted_quantile_draws(
                months[pool_donors], weights[pool_donors], imputation_uniforms[takers]
            )
        pool_receipts.append(
            {
                "age_bands": [
                    _band_label(*UC_NEW_CLAIMANT_AGE_BANDS[band]) for band in pool
                ],
                "linked_support": support_band(
                    distinct_count(sources, pool_donors, weights)
                ),
                "imputed_benefit_units": rounded_count(int(takers.sum())),
                "imputed_support": support_band(
                    distinct_count(sources, takers, weights)
                ),
            }
        )
    without_donors = imputed & np.isinf(claim_months)

    draws = stable_identity_uniforms(
        ids, seed=UC_NEW_CLAIMANT_DRAW_SEED, salt=UC_NEW_CLAIMANT_DRAW_OUTPUT
    ).astype(np.float64)
    rows = np.flatnonzero(population)
    if rows.size:
        ties = stable_identity_uniforms(
            ids[rows], seed=UC_NEW_CLAIMANT_DRAW_SEED, salt=UC_NEW_CLAIMANT_ORDER_DRAW
        )
        order = np.lexsort((ties, youngest[rows], claim_months[rows]))
        draws[rows] = weighted_rank_midpoints(order, weights[rows])
    benunit[UC_NEW_CLAIMANT_DRAW_OUTPUT] = draws

    population_support = distinct_count(sources, population, weights)
    population_weight = float(weights[population].sum())
    young = population & (bands == 0)
    young_support = distinct_count(sources, young, weights)
    young_weight = float(weights[young].sum())

    def share_below(mask: np.ndarray, total: float, support: int, probe: float):
        below = float(weights[mask & (draws < probe)].sum())
        return small_cell_share(below / total if total > 0.0 else 0.0, support)

    linked_support = distinct_count(sources, donors, weights)
    imputed_support = distinct_count(sources, imputed, weights)
    receipt: dict[str, object] = {
        "population_rule": UC_NEW_CLAIMANT_POPULATION_RULE,
        "order_rule": UC_NEW_CLAIMANT_ORDER_RULE,
        "imputation_rule": UC_NEW_CLAIMANT_IMPUTATION_RULE,
        "rule": UC_NEW_CLAIMANT_DRAW_RULE,
        "entitlement_year": entitlement_year,
        "population": {
            "benefit_units": rounded_count(int(population.sum())),
            "support": support_band(population_support),
            "weighted": small_cell_share(
                round(population_weight, -3), population_support
            ),
        },
        # A share that gives back the smaller of its two parts is withheld with
        # it (UKDS disclosure control).
        "linked_share": small_cell_share(
            float(weights[donors].sum()) / population_weight
            if population_weight > 0.0
            else 0.0,
            min(linked_support, imputed_support),
        ),
        "pools": pool_receipts,
        "imputed_without_linked_claims": rounded_count(int(without_donors.sum())),
        "probes": list(UC_NEW_CLAIMANT_PROBES),
        "population_share_below": [
            share_below(population, population_weight, population_support, probe)
            for probe in UC_NEW_CLAIMANT_PROBES
        ],
        "first_band": _band_label(*UC_NEW_CLAIMANT_AGE_BANDS[0]),
        "first_band_support": support_band(young_support),
        "first_band_share_below": [
            share_below(young, young_weight, young_support, probe)
            for probe in UC_NEW_CLAIMANT_PROBES
        ],
    }
    household_weights = frame.weights_for("household")
    mass = MassChangeRecord(
        entity="household",
        old_total=household_weights.total,
        new_total=household_weights.total,
        declared_factor=1.0,
        reason=UC_NEW_CLAIMANT_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=household_weights.values,
        mass_log=(*frame.mass_log, mass),
    )
    validate_uk_national_frame(result)
    return UKUCNewClaimantDrawResult(
        frame=result,
        receipt=receipt,
        claim_months=np.where(population, claim_months, np.nan),
    )


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind the stage's operations closed-world."""

    if stage.stage != UC_NEW_CLAIMANT_DRAW_STAGE_NAME:
        raise ValueError(
            f"UC new-claimant draw received stage {stage.stage!r}, expected "
            f"{UC_NEW_CLAIMANT_DRAW_STAGE_NAME!r}."
        )
    expected = uc_new_claimant_draw_operation_parameters()
    operations = tuple(stage.operations)
    if tuple(operation.kind for operation in operations) != tuple(
        kind for kind, _ in expected
    ):
        raise ValueError(
            "uc_new_claimant_draw operation order drifted: "
            f"{[operation.kind for operation in operations]!r}."
        )
    for operation, (kind, parameters) in zip(operations, expected, strict=True):
        if dict(operation.parameters) != dict(parameters):
            raise ValueError(
                f"uc_new_claimant_draw {kind} parameters drifted: "
                f"{dict(operation.parameters)!r} != {dict(parameters)!r}."
            )
    if tuple(stage.outputs) != (UC_NEW_CLAIMANT_DRAW_OUTPUT,):
        raise ValueError(
            f"{UC_NEW_CLAIMANT_DRAW_STAGE_NAME} must declare exactly the "
            f"{UC_NEW_CLAIMANT_DRAW_OUTPUT} output."
        )
    if stage.grain != "benunit":
        raise ValueError(
            f"{UC_NEW_CLAIMANT_DRAW_STAGE_NAME} is benunit grain, got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKUCNewClaimantDrawStageTransform:
    """Whole-stage callable storing the UC new-claimant health element draw."""

    stage: SourceStageSpec
    engine: object
    last_result: UKUCNewClaimantDrawResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        if UC_NEW_CLAIMANT_DRAW_OUTPUT in frame.table("benunit").columns:
            raise ValueError(
                f"{UC_NEW_CLAIMANT_DRAW_OUTPUT!r} is already stored: this stage is "
                "its only writer."
            )
        period = uk_time_period(frame)
        declared = resolve_uk_year_rule(UC_NEW_CLAIMANT_DATASET_YEAR_RULE)
        if int(period) != declared:
            raise ValueError(
                "The UC new-claimant draw reads the frame as "
                f"{UC_NEW_CLAIMANT_DATASET_YEAR_RULE} {declared} values, but the "
                f"frame's period is {period}."
            )
        entitlement_year = resolve_uk_year_rule(UC_NEW_CLAIMANT_YEAR_RULE)
        answers = self.engine.materialize(  # type: ignore[attr-defined]
            frame,
            list(UC_NEW_CLAIMANT_ENGINE_VARIABLES),
            entitlement_year,
            dataset_period=period,
        )
        result = assign_uc_new_claimant_draw(
            frame, engine_answers=answers, entitlement_year=entitlement_year
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (UC_NEW_CLAIMANT_DRAW_OUTPUT,)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
