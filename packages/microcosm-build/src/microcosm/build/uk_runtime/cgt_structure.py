"""UK capital-gains incidence cloning and the post-redraw incidence anchor
(microcosm#970).

The HMRC size-band donor stage that used to sit between them is retired by
microcosm#1045: row support at the top of the gains distribution now comes
from the mass-conserving support split in ``cgt_support`` that runs before
the clone, and the Table 3 redraw owns every gain amount.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.advani_summers import (
    ADVANI_SUMMERS_RESOURCE,
    CGT_QUANTILE_POINTS,
    advani_summers_band_index,
    advani_summers_knots,
    exempt_range_quantiles,
    load_advani_summers_distribution,
)
from microcosm.build.uk_runtime.advani_summers import (
    advani_summers_rows as _distribution_rows,
)
from microcosm.build.uk_runtime.advani_summers import (
    draw_banded_priors as _draw_banded_priors,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
    uk_cgt_policy_parameters,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.rowwise_geography import (
    clone_entity_frame,
    id_multiplier_for_values,
)
from microcosm.build.uk_runtime.spi_support import (
    _importance_weights_with_exact_total,
)
from microcosm.frame import Frame, MassChangeRecord, WeightKind

CGT_CLONE_MASS_SPLIT = 0.5
CGT_PRIOR_SEED = 0
CGT_PRIOR_SALT = "cgt_prior_amount"
CGT_ADULT_MINIMUM_AGE = 16
HOUSEHOLD_IS_CGT_CLONE = "household_is_capital_gains_clone"
CGT_CLONE_MASS_CHANGE_REASON = (
    "Capital-gains incidence clone splits every household's mass equally across "
    "original and clone records; total household mass is conserved."
)
CGT_INCIDENCE_ANCHOR_STAGE_NAME = "cgt_incidence_anchor"
#: A clone never gains mass at the anchor: its factor is capped at one.
CGT_ANCHOR_MAXIMUM_FACTOR = 1.0
#: The non-liable clone groups the anchor moves mass out of.
CGT_ANCHOR_GROUPS = ("sub_exempt", "loss")
#: What the anchor's groups, targets and receipt describe: the clone side of
#: the pairs. Originals are outside its reach, so the receipt certifies the
#: anchor's own arithmetic, not a population count.
CGT_ANCHOR_SCOPE = (
    "clone households paired to their originals; originals are outside the "
    "anchor, so the groups, targets and masses are clone-side quantities, not "
    "population counts"
)
CGT_ANCHOR_MASS_CHANGE_REASON = (
    "Capital-gains incidence anchor moves the mass of non-liable clone "
    "households back to their paired originals until the sub-exempt and "
    "loss-making clone mass match the Advani-Summers reporter composition at "
    "the redrawn liable mass; every pair's mass and the total household mass "
    "are conserved."
)
#: Bisection steps for the per-group scale; float64 brackets collapse long
#: before this, so the solve is exact to rounding and deterministic.
_ANCHOR_BISECTION_STEPS = 200


@dataclass(frozen=True)
class UKCGTIncidenceCloneResult:
    """Cloned frame and the executed-effect receipt for stage 19."""

    frame: Frame
    original_mass: float
    clone_mass: float
    carrier_count: int
    negative_prior_count: int

    def evidence(self) -> dict[str, object]:
        return {
            "stage": "cgt_incidence_clone",
            "mass_by_clone_flag": {
                "false": self.original_mass,
                "true": self.clone_mass,
            },
            "carrier_count": self.carrier_count,
            "negative_prior_count": self.negative_prior_count,
        }


@dataclass(frozen=True)
class UKCGTIncidenceCloneStageTransform:
    """Whole-stage transform for the incidence clone and prior draw."""

    stage: SourceStageSpec
    distribution: Mapping[str, Any] | None = None
    last_result: UKCGTIncidenceCloneResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        resource = self.distribution or load_advani_summers_distribution()
        _assert_cgt_incidence_stage_parameters(self.stage)
        result = clone_cgt_incidence(frame, distribution=resource)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (HOUSEHOLD_IS_CGT_CLONE, "capital_gains")

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


@dataclass(frozen=True)
class UKCGTIncidenceAnchorResult:
    """Anchored frame and the executed-effect receipt of the incidence anchor.

    Group masses are household weights over clone households; each clone
    household carries exactly one gainer, so they equal the weighted gainer
    persons of the group and are commensurate with ``liable_mass``. They are
    clone-side quantities: originals are outside the anchor,
    so ``after`` is what the anchor realised, not the population's sub-exempt
    or loss-making mass.
    """

    frame: Frame
    annual_exempt_amount: float
    liable_mass: float
    liable_persons: int
    zero_quantile: float
    exempt_quantile: float
    targets: Mapping[str, float]
    before: Mapping[str, float]
    after: Mapping[str, float]
    scale: Mapping[str, float | None]
    transferred_mass: float
    pair_count: int
    trimmed_households: int
    capped_households: int
    zero_gain_clone_households: int
    zero_gain_clone_mass: float
    max_pair_relative_error: float
    original_mass: float
    clone_mass: float
    by_income_band: tuple[Mapping[str, object], ...]

    def evidence(self) -> dict[str, object]:
        liable_share = 1.0 - self.exempt_quantile
        return {
            "stage": CGT_INCIDENCE_ANCHOR_STAGE_NAME,
            "annual_exempt_amount": self.annual_exempt_amount,
            "liable_mass": self.liable_mass,
            "liable_persons": self.liable_persons,
            "composition": {
                "scope": CGT_ANCHOR_SCOPE,
                "zero_quantile": self.zero_quantile,
                "exempt_quantile": self.exempt_quantile,
                "liable_share": liable_share,
                "implied_reporter_mass": self.liable_mass / liable_share,
            },
            "targets": dict(self.targets),
            "before": dict(self.before),
            "after": dict(self.after),
            "scale": dict(self.scale),
            "transferred_mass": self.transferred_mass,
            "pair_count": self.pair_count,
            "trimmed_households": self.trimmed_households,
            "capped_households": self.capped_households,
            "zero_gain_clone_households": self.zero_gain_clone_households,
            "zero_gain_clone_mass": self.zero_gain_clone_mass,
            "max_pair_relative_error": self.max_pair_relative_error,
            "mass_by_clone_flag": {
                "false": self.original_mass,
                "true": self.clone_mass,
            },
            "by_income_band": [dict(row) for row in self.by_income_band],
        }


@dataclass(frozen=True)
class UKCGTIncidenceAnchorStageTransform:
    """Whole-stage transform for the post-redraw incidence anchor.

    Weights only: no cell is written, no row is added or removed. The annual
    exempt amount is read from the policyengine-uk parameter tree at the
    frame's build period unless ``parameters`` is supplied.
    """

    stage: SourceStageSpec
    distribution: Mapping[str, Any] | None = None
    parameters: UKCGTPolicyParameters | None = None
    last_result: UKCGTIncidenceAnchorResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_cgt_incidence_anchor_stage_parameters(self.stage)
        distribution = self.distribution or load_advani_summers_distribution()
        parameters = self.parameters
        if parameters is None:
            parameters = uk_cgt_policy_parameters(uk_time_period(frame))
        result = anchor_cgt_incidence(
            frame, distribution=distribution, parameters=parameters
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


def clone_cgt_incidence(
    frame: Frame,
    *,
    distribution: Mapping[str, Any],
) -> UKCGTIncidenceCloneResult:
    """Clone every household at equal mass and assign A&S priors to clones."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    multiplier = id_multiplier_for_values(
        person["person_id"],
        person["person_household_id"],
        person["person_benunit_id"],
        benunit["benunit_id"],
        household["household_id"],
    )
    cloned_person = clone_entity_frame(
        person,
        id_columns=("person_id", "person_household_id", "person_benunit_id"),
        n_clones=2,
        id_multiplier=multiplier,
        clone_index_column=None,
    ).reset_index(drop=True)
    cloned_benunit = clone_entity_frame(
        benunit,
        id_columns=("benunit_id",),
        n_clones=2,
        id_multiplier=multiplier,
        clone_index_column=None,
    ).reset_index(drop=True)
    cloned_household = clone_entity_frame(
        household,
        id_columns=("household_id",),
        n_clones=2,
        id_multiplier=multiplier,
        clone_index_column=None,
    ).reset_index(drop=True)
    n_households = len(household)
    clone_flags = np.r_[
        np.zeros(n_households, dtype=bool),
        np.ones(n_households, dtype=bool),
    ]
    cloned_household[HOUSEHOLD_IS_CGT_CLONE] = clone_flags
    split = np.tile(
        frame.weights_for("household").values * CGT_CLONE_MASS_SPLIT,
        2,
    )
    exact_weights = _importance_weights_with_exact_total(
        split,
        frame.weights_for("household").total,
    )
    cloned_person["capital_gains"] = 0.0
    carrier_indices = _oldest_adult_indices(
        cloned_person,
        household_ids=set(cloned_household.loc[clone_flags, "household_id"].to_numpy()),
    )
    carrier_income = _component_sum_income(cloned_person.loc[carrier_indices])
    carrier_draws = stable_identity_uniforms(
        cloned_person.loc[carrier_indices, "person_id"].to_numpy(),
        seed=CGT_PRIOR_SEED,
        salt=CGT_PRIOR_SALT,
    )
    priors = _draw_banded_priors(
        carrier_income,
        carrier_draws,
        distribution=distribution,
    )
    cloned_person.loc[carrier_indices, "capital_gains"] = priors
    receipt = MassChangeRecord(
        entity="household",
        old_total=frame.weights_for("household").total,
        new_total=exact_weights.total,
        declared_factor=1.0,
        reason=CGT_CLONE_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=cloned_person,
        benunit=cloned_benunit,
        household=cloned_household,
        time_period=uk_time_period(frame),
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=exact_weights.values,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result)
    original_mass = float(exact_weights.values[~clone_flags].sum())
    clone_mass = float(exact_weights.values[clone_flags].sum())
    return UKCGTIncidenceCloneResult(
        frame=result,
        original_mass=original_mass,
        clone_mass=clone_mass,
        carrier_count=len(carrier_indices),
        negative_prior_count=int((priors < 0.0).sum()),
    )


def pair_clone_households(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Positions of every clone household and of its paired original.

    The clone stage offsets ids by ``id_multiplier_for_values`` over the rows
    it saw; those rows are exactly the households whose clone flag is false,
    the support split's copies included (they are pre-clone rows with their
    own clones, microcosm#1045). Recomputing the multiplier from them
    reproduces the executor's lineage rule, and the pairing must be a
    bijection: an unpaired clone or original fails closed.
    """

    if HOUSEHOLD_IS_CGT_CLONE not in household.columns:
        raise ValueError(
            f"CGT incidence anchor requires household.{HOUSEHOLD_IS_CGT_CLONE}."
        )
    is_clone = household[HOUSEHOLD_IS_CGT_CLONE].to_numpy(dtype=bool)
    original = ~is_clone
    clone = is_clone
    household_ids = (
        pd.to_numeric(household["household_id"], errors="raise")
        .astype("int64")
        .to_numpy()
    )
    if len(np.unique(household_ids)) != len(household_ids):
        raise ValueError("CGT incidence anchor requires unique household ids.")
    original_ids = set(household_ids[original].tolist())
    person_original = person["person_household_id"].isin(original_ids).to_numpy()
    benunit_original = (
        benunit["benunit_id"]
        .isin(set(person.loc[person_original, "person_benunit_id"].tolist()))
        .to_numpy()
    )
    multiplier = id_multiplier_for_values(
        person.loc[person_original, "person_id"],
        person.loc[person_original, "person_household_id"],
        person.loc[person_original, "person_benunit_id"],
        benunit.loc[benunit_original, "benunit_id"],
        household_ids[original],
    )
    position_by_id = {int(value): index for index, value in enumerate(household_ids)}
    clone_positions = np.flatnonzero(clone)
    original_positions = np.empty(len(clone_positions), dtype=np.int64)
    for slot, position in enumerate(clone_positions):
        source = position_by_id.get(int(household_ids[position]) - multiplier)
        if source is None or not original[source]:
            raise ValueError(
                "CGT incidence anchor found a clone household without a paired "
                f"original: household_id {int(household_ids[position])}."
            )
        original_positions[slot] = source
    if len(np.unique(original_positions)) != len(original_positions):
        raise ValueError(
            "CGT incidence anchor found two clone households paired to one original."
        )
    if len(clone_positions) != int(original.sum()):
        raise ValueError(
            "CGT incidence anchor requires exactly one clone per original "
            f"household; found {len(clone_positions)} clones for "
            f"{int(original.sum())} originals."
        )
    return clone_positions, original_positions, multiplier


def reporter_composition_quantiles(
    income: np.ndarray,
    *,
    distribution: Mapping[str, Any],
    annual_exempt_amount: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-carrier ``(zero_quantile, exempt_quantile)`` of their A&S band."""

    rows = _distribution_rows(distribution)
    indexes = advani_summers_band_index(rows, income)
    lower = np.empty(len(rows), dtype=float)
    upper = np.empty(len(rows), dtype=float)
    for index, row in enumerate(rows):
        lower[index], upper[index] = exempt_range_quantiles(
            advani_summers_knots(row), annual_exempt_amount
        )
    return lower[indexes], upper[indexes]


def _order_free_sum(values: np.ndarray) -> float:
    """The correctly rounded sum, which does not depend on the row order.

    Every sum the anchor's factors depend on goes through here: a pairwise
    float sum moves in its last bits when the rows are permuted, the bisection
    then settles on a neighbouring scale, and the anchored weights differ by a
    few ulps between two orderings of the same households (the E8 identity
    receipt found 1e-13 on the licensed spine; microcosm#1063).
    """

    return math.fsum(np.asarray(values, dtype=float).tolist())


def _solve_group_factors(
    weights: np.ndarray,
    propensity: np.ndarray,
    target: float,
) -> tuple[float, np.ndarray]:
    """The scale ``s`` with ``sum(w * min(1, s * p)) == target`` and its factors.

    The left side rises continuously from zero to the group's mass as ``s``
    grows, so for a target below that mass the bracket ``[0, 1 / min(p)]``
    holds exactly one solution; bisection to bracket collapse finds it to
    rounding, and the upper end of the collapsed bracket is kept so the
    achieved mass never falls short of the target by more than rounding.
    """

    lower, upper = 0.0, 1.0 / float(propensity.min())
    for _ in range(_ANCHOR_BISECTION_STEPS):
        middle = 0.5 * (lower + upper)
        if middle <= lower or middle >= upper:
            break
        achieved = _order_free_sum(
            weights * np.minimum(CGT_ANCHOR_MAXIMUM_FACTOR, middle * propensity)
        )
        if achieved < target:
            lower = middle
        else:
            upper = middle
    scale = upper
    return scale, np.minimum(CGT_ANCHOR_MAXIMUM_FACTOR, scale * propensity)


def anchor_cgt_incidence(
    frame: Frame,
    *,
    distribution: Mapping[str, Any],
    parameters: UKCGTPolicyParameters,
) -> UKCGTIncidenceAnchorResult:
    """Move non-liable clone mass back to the paired originals (microcosm#970).

    The equal-mass clone leaves half of all household mass on gainer
    households, far more than the reporter population the Table 3 redraw can
    place above the exempt amount, so after the redraw the sub-exempt and
    loss-making clones carry millions of weighted persons. HMRC publishes no
    count for them, so their level is derived: the liable mass the redraw
    realised, scaled by the Advani-Summers composition of reporters (the
    share of a band's reporters below zero and below the exempt amount,
    averaged over the clone carriers by weight times incidence). Each
    non-liable group is scaled down to its target with a factor rising in the
    band's incidence and capped at one; every unit of mass a clone loses goes
    to its paired original, which is identical in every cell but the gains,
    so pair mass, household mass and every non-CGT aggregate are conserved.
    Liable clones are untouched.
    """

    validate_uk_national_frame(frame)
    person = frame.table("person").copy().reset_index(drop=True)
    benunit = frame.table("benunit").copy().reset_index(drop=True)
    household = frame.table("household").copy().reset_index(drop=True)
    annual_exempt_amount = float(parameters.annual_exempt_amount)
    if not np.isfinite(annual_exempt_amount) or annual_exempt_amount <= 0.0:
        raise ValueError(
            "CGT incidence anchor requires a positive annual exempt amount."
        )
    if "capital_gains" not in person.columns:
        raise ValueError("CGT incidence anchor requires person.capital_gains.")
    clone_positions, original_positions, _ = pair_clone_households(
        person, benunit, household
    )
    if len(clone_positions) == 0:
        raise ValueError("CGT incidence anchor found no clone households.")
    old_weights = frame.weights_for("household")
    weights = np.asarray(old_weights.values, dtype=float).copy()
    gains = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(gains).all():
        raise ValueError("CGT incidence anchor requires finite capital gains.")
    household_ids = household["household_id"].to_numpy()
    position_by_id = pd.Series(np.arange(len(household)), index=household_ids)
    person_positions = position_by_id.reindex(
        person["person_household_id"].to_numpy()
    ).to_numpy()
    if np.isnan(person_positions.astype(float)).any():
        raise ValueError("CGT incidence anchor found a person without a household.")
    person_positions = person_positions.astype(np.int64)
    liable = gains > annual_exempt_amount
    liable_mass = _order_free_sum(weights[person_positions][liable])
    if not liable_mass > 0.0:
        raise ValueError(
            "CGT incidence anchor requires positive liable mass above the annual "
            "exempt amount after the redraw."
        )

    clone_ids = household_ids[clone_positions]
    carriers = _oldest_adult_indices(person, household_ids=set(clone_ids.tolist()))
    carrier_by_household = dict(
        zip(person.loc[carriers, "person_household_id"].tolist(), carriers, strict=True)
    )
    carrier_rows = np.asarray(
        [carrier_by_household[household_id] for household_id in clone_ids.tolist()],
        dtype=np.int64,
    )
    in_clone = person["person_household_id"].isin(set(clone_ids.tolist())).to_numpy()
    non_carrier = in_clone.copy()
    non_carrier[carrier_rows] = False
    if (gains[non_carrier] != 0.0).any():
        raise ValueError(
            "CGT incidence anchor requires the clone carrier to be the only "
            "person with capital gains in a clone household."
        )
    carrier_gains = gains[carrier_rows]
    income = _component_sum_income(person.loc[carrier_rows])
    propensity = _incidence_propensity(income, distribution=distribution)
    if not np.isfinite(propensity).all() or (propensity <= 0.0).any():
        raise ValueError(
            "CGT incidence anchor requires a finite, positive Advani-Summers "
            "incidence for every clone carrier."
        )
    zero_quantiles, exempt_quantiles = reporter_composition_quantiles(
        income, distribution=distribution, annual_exempt_amount=annual_exempt_amount
    )
    clone_weights = weights[clone_positions]
    mix = clone_weights * propensity
    mix_total = _order_free_sum(mix)
    if not mix_total > 0.0:
        raise ValueError("CGT incidence anchor found no clone mass to compose.")
    zero_quantile = _order_free_sum(mix * zero_quantiles) / mix_total
    exempt_quantile = _order_free_sum(mix * exempt_quantiles) / mix_total
    if not 0.0 <= zero_quantile < exempt_quantile < 1.0:
        raise ValueError(
            "CGT incidence anchor derived an unusable reporter composition: "
            f"zero quantile {zero_quantile}, exempt quantile {exempt_quantile}."
        )
    liable_share = 1.0 - exempt_quantile
    targets = {
        "sub_exempt": liable_mass * (exempt_quantile - zero_quantile) / liable_share,
        "loss": liable_mass * zero_quantile / liable_share,
    }
    if not all(value > 0.0 for value in targets.values()):
        raise ValueError("CGT incidence anchor derived a non-positive group target.")
    groups = {
        # A carrier with exactly zero gains is no reporter: it is neither in the
        # (0, AEA] slice the sub-exempt target derives from nor a loss-maker, so
        # it is left untouched and reported separately.
        "sub_exempt": (carrier_gains > 0.0) & (carrier_gains <= annual_exempt_amount),
        "loss": carrier_gains < 0.0,
    }
    liable_clone = carrier_gains > annual_exempt_amount
    before = {
        name: _order_free_sum(clone_weights[mask]) for name, mask in groups.items()
    }
    before["liable"] = float(clone_weights[liable_clone].sum())
    factors = np.ones(len(clone_positions), dtype=float)
    scale: dict[str, float | None] = {}
    capped = 0
    for name in CGT_ANCHOR_GROUPS:
        mask = groups[name]
        if not mask.any() or before[name] <= targets[name]:
            scale[name] = None
            continue
        group_scale, group_factors = _solve_group_factors(
            clone_weights[mask], propensity[mask], targets[name]
        )
        factors[mask] = group_factors
        scale[name] = group_scale
        capped += int((group_factors >= CGT_ANCHOR_MAXIMUM_FACTOR).sum())
    new_clone_weights = clone_weights * factors
    delta = clone_weights - new_clone_weights
    new_weights = weights.copy()
    new_weights[clone_positions] = new_clone_weights
    new_weights[original_positions] = weights[original_positions] + delta
    # Pairs are conserved arithmetically (three roundings per pair, so a few
    # ulps); no global exact-total correction is applied, because it would move
    # the summed rounding of every transfer onto one household, breaking that
    # household's pair (1.6e-13 relative on the licensed spine) and possibly
    # touching a liable clone. The total is recorded as realised.
    values = new_weights
    if (values < 0.0).any() or not np.array_equal(values > 0.0, weights > 0.0):
        raise ValueError(
            "CGT incidence anchor must keep every household weight non-negative "
            "and leave the zero-weight pattern unchanged."
        )
    pair_before = weights[clone_positions] + weights[original_positions]
    pair_after = values[clone_positions] + values[original_positions]
    pair_gap = np.abs(pair_after - pair_before)
    positive = pair_before > 0.0
    pair_error = np.where(
        positive, pair_gap / np.where(positive, pair_before, 1.0), pair_gap
    )
    after = {
        name: float(values[clone_positions][mask].sum())
        for name, mask in groups.items()
    }
    after["liable"] = float(values[clone_positions][liable_clone].sum())
    band_index = np.clip(
        np.searchsorted(
            np.asarray(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS, dtype=float),
            income,
            side="right",
        )
        - 1,
        0,
        len(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS) - 1,
    )
    by_income_band = []
    for index, lower_limit in enumerate(HMRC_CGT_INCOME_BAND_LOWER_BOUNDS):
        in_band = band_index == index
        by_income_band.append(
            {
                "lower_limit": int(lower_limit),
                "income_measure": "component_sum_without_allowance",
                "clone_households": int(in_band.sum()),
                "sub_exempt_before": float(
                    clone_weights[in_band & groups["sub_exempt"]].sum()
                ),
                "sub_exempt_after": float(
                    values[clone_positions][in_band & groups["sub_exempt"]].sum()
                ),
                "loss_before": float(clone_weights[in_band & groups["loss"]].sum()),
                "loss_after": float(
                    values[clone_positions][in_band & groups["loss"]].sum()
                ),
                "liable": float(clone_weights[in_band & liable_clone].sum()),
            }
        )
    receipt = MassChangeRecord(
        entity="household",
        old_total=old_weights.total,
        new_total=float(values.sum()),
        declared_factor=1.0,
        reason=CGT_ANCHOR_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=values,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result)
    return UKCGTIncidenceAnchorResult(
        frame=result,
        annual_exempt_amount=annual_exempt_amount,
        liable_mass=liable_mass,
        liable_persons=int(liable.sum()),
        zero_quantile=zero_quantile,
        exempt_quantile=exempt_quantile,
        targets=targets,
        before=before,
        after=after,
        scale=scale,
        transferred_mass=float(delta.sum()),
        pair_count=int(len(clone_positions)),
        trimmed_households=int((factors < CGT_ANCHOR_MAXIMUM_FACTOR).sum()),
        capped_households=capped,
        zero_gain_clone_households=int((carrier_gains == 0.0).sum()),
        zero_gain_clone_mass=float(clone_weights[carrier_gains == 0.0].sum()),
        max_pair_relative_error=float(pair_error.max()),
        original_mass=float(values[original_positions].sum()),
        clone_mass=float(values[clone_positions].sum()),
        by_income_band=tuple(by_income_band),
    )


def _oldest_adult_indices(
    person: pd.DataFrame,
    *,
    household_ids: set[object],
) -> np.ndarray:
    required = {"person_id", "person_household_id", "age"}
    missing = sorted(required - set(person.columns))
    if missing:
        raise ValueError(f"CGT carrier selection is missing person columns: {missing}.")
    candidates = person.loc[
        person.person_household_id.isin(household_ids)
        & (pd.to_numeric(person.age, errors="coerce") >= CGT_ADULT_MINIMUM_AGE)
    ].copy()
    missing_households = household_ids - set(candidates.person_household_id)
    if missing_households:
        raise ValueError(
            "Every CGT household requires an adult carrier; missing household "
            f"id(s): {sorted(missing_households)[:5]}."
        )
    candidates["_row"] = candidates.index
    candidates = candidates.sort_values(
        ["person_household_id", "age", "person_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    return (
        candidates.groupby("person_household_id", sort=False)["_row"].first().to_numpy()
    )


def _component_sum_income(person: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS) - set(person.columns))
    if missing:
        raise ValueError(f"CGT income proxy components missing: {missing}.")
    numeric = person.loc[:, UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS].apply(
        pd.to_numeric, errors="coerce"
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("CGT component-sum income contains non-finite values.")
    return numeric.sum(axis=1).to_numpy(dtype=float)


def _incidence_propensity(
    income: np.ndarray,
    *,
    distribution: Mapping[str, Any],
) -> np.ndarray:
    rows = _distribution_rows(distribution)
    indexes = advani_summers_band_index(rows, income)
    rates = np.asarray([row["percent_with_gains"] for row in rows], dtype=float)
    return rates[indexes]


def _assert_closed_world_operations(
    stage: SourceStageSpec,
    expected_operations: tuple[tuple[str, dict[str, object]], ...],
) -> None:
    """Exact operation order and full-mapping equality per operation.

    Whole-payload equality rejects value drift, missing keys, and extra keys
    alike (adversarial-review finding on the E8 PR: asserting a named subset
    let lockstep manifest edits move undeclared-but-load-bearing semantics).
    The expected sequence is ordered so repeated kinds are supported and an
    extra, missing, or reordered operation fails by position.
    """

    kinds = tuple(operation.kind for operation in stage.operations)
    expected_kinds = tuple(kind for kind, _ in expected_operations)
    if kinds != expected_kinds:
        raise ValueError(
            f"Stage {stage.stage!r} operation order drifted: expected "
            f"{expected_kinds}, got {kinds}."
        )
    for operation, (kind, expected) in zip(
        stage.operations, expected_operations, strict=True
    ):
        actual = dict(operation.parameters)
        if actual != expected:
            drifted = sorted(
                key
                for key in {*actual, *expected}
                if actual.get(key) != expected.get(key)
            )
            raise ValueError(
                f"Stage {stage.stage!r} {kind} declaration drifted "
                f"from the reviewed mapping on parameter(s) {drifted}."
            )


def _assert_cgt_incidence_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind every stage-19 manifest parameter to reviewed code constants.

    This is arm 1 of the #730/#684 two-arm rule documented in ``spi_spine``;
    :class:`UKCGTIncidenceCloneResult` supplies the executed-effect receipt.
    """

    _assert_closed_world_operations(
        stage,
        (
            (
                "clone_records",
                {
                    "entity": "household",
                    "copies": 2,
                    "flag_column": HOUSEHOLD_IS_CGT_CLONE,
                    "original_flag": False,
                    "clone_flag": True,
                    "mass_split": CGT_CLONE_MASS_SPLIT,
                    "weight_kind_out": WeightKind.IMPORTANCE.value,
                    "conservation": "exact_total",
                    "id_remapping": "id_multiplier_for_values",
                    "declared_factor": 1.0,
                    "reason": CGT_CLONE_MASS_CHANGE_REASON,
                },
            ),
            (
                "draw_capital_gains_prior_from_banded_quantiles",
                {
                    "resource": ADVANI_SUMMERS_RESOURCE,
                    "income_proxy_components": list(
                        UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS
                    ),
                    "allowance_subtraction": False,
                    "carrier": "oldest adult; person_id ascending breaks age ties",
                    "adult_minimum_age": CGT_ADULT_MINIMUM_AGE,
                    "quantile_points": list(CGT_QUANTILE_POINTS),
                    "spline_degree": 1,
                    "extrapolation": "ext=0",
                    "keep_negative_draws": True,
                    "seed": CGT_PRIOR_SEED,
                    "salt": CGT_PRIOR_SALT,
                },
            ),
        ),
    )


def cgt_incidence_anchor_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed anchor operations the manifests must carry verbatim."""

    return (
        (
            "pair_clone_households_to_originals",
            {
                "clone_flag_column": HOUSEHOLD_IS_CGT_CLONE,
                "id_remapping": "id_multiplier_for_values",
                "pairing": (
                    "clone household_id minus the clone stage's id multiplier, "
                    "recomputed from the pre-clone rows (clone flag false; the "
                    "support split's copies are pre-clone rows with their own "
                    "clones)"
                ),
                "requirement": (
                    "bijection between clone and original households; an "
                    "unpaired clone or original fails the build"
                ),
            },
        ),
        (
            "derive_reporter_composition_from_banded_quantiles",
            {
                "resource": ADVANI_SUMMERS_RESOURCE,
                "income_proxy_components": list(UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS),
                "allowance_subtraction": False,
                "carrier": "oldest adult; person_id ascending breaks age ties",
                "adult_minimum_age": CGT_ADULT_MINIMUM_AGE,
                "quantile_points": list(CGT_QUANTILE_POINTS),
                "spline_degree": 1,
                "extrapolation": "ext=0",
                "crossings": (
                    "each band's degree-1 quantile spline inverted at zero and at "
                    "the build period's annual exempt amount"
                ),
                "aggregation": (
                    "crossings averaged over clone carriers weighted by household "
                    "weight times the band's percent_with_gains"
                ),
                "liable_mass": (
                    "weighted persons with capital_gains above the annual exempt "
                    "amount on every household after the Table 3 redraw"
                ),
                "targets": (
                    "sub_exempt = liable_mass * (exempt_quantile - zero_quantile) "
                    "/ (1 - exempt_quantile); loss = liable_mass * zero_quantile "
                    "/ (1 - exempt_quantile)"
                ),
            },
        ),
        (
            "transfer_clone_mass_to_originals",
            {
                "population": CGT_ANCHOR_SCOPE,
                "sub_exempt_group": (
                    "clone households whose carrier's capital_gains lie in "
                    "(0, annual exempt amount]; a carrier with exactly zero gains is "
                    "no reporter and is left untouched"
                ),
                "loss_group": (
                    "clone households whose carrier's capital_gains are negative"
                ),
                "liable_clones": "untouched",
                "factor": (
                    "min(maximum_factor, scale * percent_with_gains) with scale "
                    "solved per group by bisection so the group's mass equals its "
                    "target; a group at or below its target is untouched"
                ),
                "maximum_factor": CGT_ANCHOR_MAXIMUM_FACTOR,
                "transfer": (
                    "each clone's removed mass is added to its paired original, "
                    "so every pair's mass is conserved to rounding; no global "
                    "exact-total correction is applied, so no household outside "
                    "a trimmed pair moves"
                ),
                "weight_kind_out": WeightKind.IMPORTANCE.value,
                "conservation": "household_total_to_rounding",
                "pair_conservation": "to_rounding",
                "declared_factor": 1.0,
                "reason": CGT_ANCHOR_MASS_CHANGE_REASON,
            },
        ),
    )


def _assert_cgt_incidence_anchor_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind every anchor manifest parameter to reviewed code constants."""

    _assert_closed_world_operations(stage, cgt_incidence_anchor_operation_parameters())
    if stage.stage != CGT_INCIDENCE_ANCHOR_STAGE_NAME:
        raise ValueError(
            f"Expected stage {CGT_INCIDENCE_ANCHOR_STAGE_NAME!r}, got {stage.stage!r}."
        )
    if tuple(stage.outputs) or tuple(stage.rewrites):
        raise ValueError(
            "The CGT incidence anchor writes no cell; the manifest must declare "
            f"no outputs or rewrites, got {stage.outputs} / {stage.rewrites}."
        )
    if stage.grain != "household":
        raise ValueError(
            f"The CGT incidence anchor moves household mass; grain must be "
            f"'household', got {stage.grain!r}."
        )
    artifacts = {str(artifact.get("role")): artifact for artifact in stage.artifacts}
    distribution = artifacts.get("capital_gains_incidence_and_quantiles")
    if (
        distribution is None
        or distribution.get("resource") != ADVANI_SUMMERS_RESOURCE
        or distribution.get("runtime_sha256_required") is not True
    ):
        raise ValueError(
            "The CGT incidence anchor must declare the Advani-Summers resource "
            "artifact with runtime_sha256_required: true."
        )
    policy = artifacts.get("policy_parameters")
    if policy is None or "gov.hmrc.cgt.annual_exempt_amount" not in tuple(
        policy.get("parameters", ())
    ):
        raise ValueError(
            "The CGT incidence anchor must declare the annual exempt amount "
            "policy-parameter artifact."
        )
