"""UK capital-gains support split: light copies of the wealthiest households
of each HMRC Table 3 income band, before the incidence clone (microcosm#1045).

The Table 3 redraw (``hmrc_cgt_gains_spine``) places whole rows into
published cells, and the bands of gains from GBP 250,000 hold a few thousand
taxpayers per income column, so a heavy row cannot be seated there and falls
to the sub-exempt remainder with its weight. This stage supplies row support
at the top of the gains distribution instead of stacking donor rows with
created mass: within each Table 3 taxable-income band it walks households in
descending investable wealth until their weight covers the band's support
mass (the published count of gainers at or above GBP 250,000 in that income
column, doubled for the clone split and doubled again for headroom) and
divides every selected household into ``ceil(weight / 60)`` identical copies
at equal weight. No value moves, no draw is made, and every household's mass
and the total household mass are conserved exactly; the clone stage then
gives each copy's clone its own identity-keyed prior, and the redraw's
wealth-ranked walk places the light rows where it was already placing heavy
ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_INVESTABLE_WEALTH_COLUMNS,
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
    UKCGTPolicyParameters,
    uk_cgt_policy_parameters,
    uk_cgt_taxable_income_proxy,
)
from microcosm.build.uk_runtime.cgt_structure import (
    CGT_ADULT_MINIMUM_AGE,
    HOUSEHOLD_IS_CGT_CLONE,
    _assert_closed_world_operations,
    _oldest_adult_indices,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_CONDITIONING_RESOURCE,
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS,
    HMRC_CGT_JOINT_RECORD_SET_PREFIX,
    HMRC_CGT_SOURCE_VINTAGE,
    HMRCCapitalGainsJointDistribution,
    load_hmrc_cgt_joint_distribution,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.rowwise_geography import id_multiplier_for_values
from microcosm.build.uk_runtime.spi_support import (
    _exact_total_correction,
    _importance_weights_with_exact_total,
)
from microcosm.frame import Frame, MassChangeRecord, WeightKind, Weights

CGT_SUPPORT_SPLIT_STAGE_NAME = "cgt_support_split"
CGT_SUPPORT_SPLIT_OPERATION_KIND = "split_top_wealth_support_households"
#: True only on the copies this stage creates (the clone-flag convention):
#: roots and unsplit households carry False.
HOUSEHOLD_IS_CGT_SUPPORT_COPY = "household_is_cgt_support_copy"
#: The family's copy count on every member of a split household, root
#: included; 1 on every household the stage left alone.
CGT_SUPPORT_COPIES_COLUMN = "cgt_support_copies"
#: The clone stage halves every household and only the clone twin gains.
CGT_SUPPORT_CLONE_SPLIT_FACTOR = 2
#: Covers clones whose prior draws at or below zero and the quarter of the
#: redraw's rank key the prior carries.
CGT_SUPPORT_HEADROOM = 2.0
#: A copy weighs at most this before the clone (half of it after), so the
#: GBP 5m+ band seats at least a hundred rows.
CGT_SUPPORT_MAXIMUM_COPY_WEIGHT = 60.0
#: The Table 3 bands of gains the support is sized on.
CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER = 250_000
#: The redraw's income bands, so classification matches its walk.
CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS: tuple[int, ...] = (
    HMRC_CGT_INCOME_BAND_LOWER_BOUNDS
)
#: The Chronicle concept the published cell counts carry.
CGT_SUPPORT_TAXPAYER_CONCEPT = "hmrc.cgt_taxpayers_individuals"
#: Reviewed pins on the 2024-25 vintage (HMRC Table 3, individuals with
#: gains at or above GBP 250,000 summed over the six income columns, and the
#: support mass that count implies): a re-vendored resource that moves them
#: fails the stage assert until reviewed here.
CGT_SUPPORT_PUBLISHED_TOP_BAND_TAXPAYERS = 58_000.0
CGT_SUPPORT_EXPECTED_MASS = 232_000.0
CGT_SUPPORT_MASS_CHANGE_REASON = (
    "Capital-gains support split divides the wealthiest households of each "
    "HMRC Table 3 income band into light copies at equal weight; every "
    "household's mass and the total household mass are conserved."
)
_HOUSEHOLD_SUPPORT_CHANNEL_COLUMN = "household_support_channel"
_PERSON_ID_COLUMNS = ("person_id", "person_household_id", "person_benunit_id")
_POLICY_PARAMETER_NAMES = (
    "gov.hmrc.income_tax.allowances.personal_allowance.amount",
    "gov.hmrc.income_tax.allowances.personal_allowance.maximum_ANI",
    "gov.hmrc.income_tax.allowances.personal_allowance.reduction_rate",
)


@dataclass(frozen=True)
class UKCGTSupportSplitResult:
    """Split frame and the executed-effect receipt of the support split.

    ``band_rows`` carry one mapping per Table 3 income band, in the order of
    :data:`CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS`; the totals, the mass pair,
    the three parameters and the support-channel split of the selected
    households make up the rest of :meth:`evidence`.
    """

    frame: Frame
    band_rows: tuple[Mapping[str, object], ...]
    published_top_band_taxpayers: float
    suppressed_cells: int
    support_mass: float
    households_selected: int
    copies_created: int
    selected_mass: float
    households_before: int
    households_after: int
    zero_weight_excluded: int
    old_total: float
    new_total: float
    frs_selected: int
    spi_selected: int
    clone_split_factor: int = CGT_SUPPORT_CLONE_SPLIT_FACTOR
    headroom: float = CGT_SUPPORT_HEADROOM
    maximum_copy_weight: float = CGT_SUPPORT_MAXIMUM_COPY_WEIGHT

    def evidence(self) -> dict[str, object]:
        return {
            "stage": CGT_SUPPORT_SPLIT_STAGE_NAME,
            "bands": [dict(row) for row in self.band_rows],
            "totals": {
                "published_top_band_taxpayers": float(
                    self.published_top_band_taxpayers
                ),
                "suppressed_cells": int(self.suppressed_cells),
                "support_mass": float(self.support_mass),
                "households_selected": int(self.households_selected),
                "copies_created": int(self.copies_created),
                "selected_mass": float(self.selected_mass),
                "households_before": int(self.households_before),
                "households_after": int(self.households_after),
                "zero_weight_excluded": int(self.zero_weight_excluded),
            },
            "mass": {
                "old_total": float(self.old_total),
                "new_total": float(self.new_total),
            },
            "parameters": {
                "clone_split_factor": int(self.clone_split_factor),
                "headroom": float(self.headroom),
                "maximum_copy_weight": float(self.maximum_copy_weight),
            },
            "support_channel_split": {
                "frs": int(self.frs_selected),
                "spi": int(self.spi_selected),
            },
        }


@dataclass(frozen=True)
class UKCGTSupportSplitStageTransform:
    """Whole-stage transform for the support split.

    ``distribution`` defaults to the vendored Table 3 joint; ``parameters``
    default to the policyengine-uk allowance parameters at the frame's build
    period, read at apply time so the base package never imports the engine
    at import time. Constructing the transform binds the manifest to the
    reviewed operation dictionary and the vendored pins.
    """

    stage: SourceStageSpec
    distribution: HMRCCapitalGainsJointDistribution | None = None
    parameters: UKCGTPolicyParameters | None = None
    last_result: UKCGTSupportSplitResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_cgt_support_split_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_cgt_support_split_stage_parameters(self.stage)
        distribution = self.distribution
        if distribution is None:
            distribution = load_hmrc_cgt_joint_distribution()
        parameters = self.parameters
        if parameters is None:
            parameters = uk_cgt_policy_parameters(uk_time_period(frame))
        result = split_cgt_support_households(
            frame, distribution=distribution, parameters=parameters
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (HOUSEHOLD_IS_CGT_SUPPORT_COPY, CGT_SUPPORT_COPIES_COLUMN)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def cgt_support_mass_by_income_band(
    distribution: HMRCCapitalGainsJointDistribution,
    *,
    clone_split_factor: int = CGT_SUPPORT_CLONE_SPLIT_FACTOR,
    headroom: float = CGT_SUPPORT_HEADROOM,
    minimum_gain_band_lower: int = CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
) -> tuple[dict[str, object], ...]:
    """The published top-band count and the support mass of each income band.

    One mapping per band of :data:`CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS`, in
    that order, with ``income_lower_bound``, ``published_top_band_taxpayers``
    (the joint's individuals summed over the bands of gains whose lower bound
    is at or above ``minimum_gain_band_lower``), ``suppressed_cells`` (cells
    whose count HMRC withheld; they count as zero) and ``support_mass``
    (``clone_split_factor * headroom * published_top_band_taxpayers``).
    """

    if isinstance(clone_split_factor, bool) or clone_split_factor < 1:
        raise ValueError(
            f"CGT support clone split factor must be at least 1, got "
            f"{clone_split_factor!r}."
        )
    if not np.isfinite(headroom) or headroom <= 0.0:
        raise ValueError(f"CGT support headroom must be positive, got {headroom!r}.")
    rows: list[dict[str, object]] = []
    for income_lower in CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS:
        published = 0.0
        suppressed = 0
        for cell in distribution.cells:
            if (
                cell.income_lower_bound != income_lower
                or cell.gain_lower_bound < minimum_gain_band_lower
            ):
                continue
            if cell.individuals is None:
                suppressed += 1
                continue
            published += float(cell.individuals)
        rows.append(
            {
                "income_lower_bound": int(income_lower),
                "published_top_band_taxpayers": published,
                "suppressed_cells": suppressed,
                "support_mass": float(clone_split_factor) * float(headroom) * published,
            }
        )
    return tuple(rows)


def cgt_support_income_band(
    person: pd.DataFrame,
    *,
    household_ids: Sequence[Any] | np.ndarray,
    parameters: UKCGTPolicyParameters,
) -> np.ndarray:
    """Each household's Table 3 income band, aligned to ``household_ids``.

    The carrier is the household's oldest adult (``person_id`` ascending
    breaks age ties, as the clone stage chooses it); its taxable-income proxy
    is digitised on the band lower bounds exactly as the Table 3 redraw
    classifies gainers. Returns one band lower bound per entry of
    ``household_ids``, in that order. Refuses a household without an adult
    or a person table without every proxy component.
    """

    ids = np.asarray(household_ids)
    carriers = _oldest_adult_indices(person, household_ids=set(ids.tolist()))
    carrier_rows = person.loc[carriers]
    taxable_income = uk_cgt_taxable_income_proxy(carrier_rows, parameters)
    band = np.asarray(CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS)[
        np.digitize(taxable_income, CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS[1:])
    ]
    by_household = pd.Series(band, index=carrier_rows["person_household_id"].to_numpy())
    return by_household.reindex(ids).to_numpy(dtype="int64")


def cgt_support_household_wealth(household: pd.DataFrame) -> np.ndarray:
    """Household investable wealth: the row sum of the c5 wealth columns.

    Refuses a household table without every declared column and any
    non-finite value, rather than ranking on a partial sum.
    """

    missing = [
        column
        for column in UK_CGT_INVESTABLE_WEALTH_COLUMNS
        if column not in household.columns
    ]
    if missing:
        raise ValueError(
            "The CGT support split ranks households on investable wealth; the "
            f"household table lacks {missing}."
        )
    matrix = (
        household.loc[:, list(UK_CGT_INVESTABLE_WEALTH_COLUMNS)]
        .apply(pd.to_numeric, errors="raise")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(matrix).all():
        raise ValueError(
            "Household investable wealth must be finite for every household."
        )
    return matrix.sum(axis=1)


def select_cgt_support_households(
    household_ids: Sequence[Any] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
    income_band: Sequence[int] | np.ndarray,
    wealth: Sequence[float] | np.ndarray,
    support_mass_by_band: Mapping[int, float],
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Select the wealthiest households of each income band, whole.

    Within each band of ``support_mass_by_band`` (income band lower bound to
    support mass, walked in mapping order) the positive-weight households of
    that band are ordered by wealth descending then household id ascending,
    and taken whole until the cumulative weight reaches the support mass; the
    household that crosses it is taken. A pool whose mass falls short is
    taken entirely and recorded as exhausted, never refused; zero-weight
    rows are outside every pool. Returns the selection mask aligned to the
    inputs and one receipt row per band.
    """

    ids = np.asarray(household_ids)
    values = np.asarray(weights, dtype=float)
    bands = np.asarray(income_band)
    wealth_values = np.asarray(wealth, dtype=float)
    n = len(ids)
    if not (len(values) == len(bands) == len(wealth_values) == n):
        raise ValueError(
            "CGT support selection inputs must have one entry per household."
        )
    if len(np.unique(ids)) != n:
        raise ValueError("CGT support selection requires unique household ids.")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError(
            "CGT support selection weights must be finite and non-negative."
        )
    if not np.isfinite(wealth_values).all():
        raise ValueError("CGT support selection wealth must be finite.")
    selected = np.zeros(n, dtype=bool)
    rows: list[dict[str, object]] = []
    for income_lower, support_mass in support_mass_by_band.items():
        target = float(support_mass)
        if not np.isfinite(target) or target < 0.0:
            raise ValueError(
                f"CGT support mass for income band {income_lower!r} must be finite "
                f"and non-negative, got {support_mass!r}."
            )
        pool = np.flatnonzero((bands == income_lower) & (values > 0.0))
        pool_mass = float(values[pool].sum())
        order = pool[np.lexsort((ids[pool], -wealth_values[pool]))]
        cumulative = np.cumsum(values[order])
        if target <= 0.0:
            take = 0
        else:
            reached = np.flatnonzero(cumulative >= target)
            take = int(reached[0]) + 1 if reached.size else len(order)
        chosen = order[:take]
        selected[chosen] = True
        rows.append(
            {
                "income_lower_bound": int(income_lower),
                "support_mass": target,
                "pool_households": int(pool.size),
                "pool_mass": pool_mass,
                "households_selected": int(take),
                "selected_mass": float(cumulative[take - 1]) if take else 0.0,
                "wealth_threshold": (
                    float(wealth_values[chosen[-1]]) if take else None
                ),
                "heaviest_selected_weight": (
                    float(values[chosen].max()) if take else 0.0
                ),
                "pool_exhausted": bool(pool_mass < target),
            }
        )
    return selected, tuple(rows)


def cgt_support_copy_counts(
    weights: Sequence[float] | np.ndarray,
    maximum_copy_weight: float = CGT_SUPPORT_MAXIMUM_COPY_WEIGHT,
) -> np.ndarray:
    """``ceil(weight / maximum_copy_weight)`` per household, at least one."""

    if not np.isfinite(maximum_copy_weight) or maximum_copy_weight <= 0.0:
        raise ValueError(
            f"CGT support maximum copy weight must be positive, got "
            f"{maximum_copy_weight!r}."
        )
    values = np.asarray(weights, dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("CGT support copy counts need finite, non-negative weights.")
    counts = np.ceil(values / float(maximum_copy_weight)).astype("int64")
    return np.maximum(counts, 1)


def split_cgt_support_households(
    frame: Frame,
    *,
    distribution: HMRCCapitalGainsJointDistribution,
    parameters: UKCGTPolicyParameters,
) -> UKCGTSupportSplitResult:
    """Split the wealthiest households of each income band into light copies.

    Deterministic and mass-conserving: no draw, no seed, no salt. Each
    selected household of weight ``w`` becomes ``n = ceil(w / 60)`` copies at
    ``w / n``; the root keeps its ids, copy ``k`` takes every entity id plus
    ``k`` times the frame's id multiplier and a full copy of the household's
    person and benefit-unit rows. The exact-total correction lands on the
    heaviest unselected incumbent so every copy weight stays bitwise
    ``w / n``. Refuses a frame the clone stage has already seen or one this
    stage has already split.
    """

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    if HOUSEHOLD_IS_CGT_CLONE in household.columns:
        raise ValueError(
            "The CGT support split runs before cgt_incidence_clone; the household "
            f"table already carries {HOUSEHOLD_IS_CGT_CLONE!r}."
        )
    for column in (HOUSEHOLD_IS_CGT_SUPPORT_COPY, CGT_SUPPORT_COPIES_COLUMN):
        if column in household.columns:
            raise ValueError(
                f"The CGT support split already ran: the household table carries "
                f"{column!r}."
            )
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    old_total = frame.weights_for("household").total
    household_ids = household["household_id"].to_numpy()
    wealth = cgt_support_household_wealth(household)
    income_band = cgt_support_income_band(
        person, household_ids=household_ids, parameters=parameters
    )
    mass_rows = cgt_support_mass_by_income_band(
        distribution,
        clone_split_factor=CGT_SUPPORT_CLONE_SPLIT_FACTOR,
        headroom=CGT_SUPPORT_HEADROOM,
        minimum_gain_band_lower=CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
    )
    support_by_band = {
        int(row["income_lower_bound"]): float(row["support_mass"]) for row in mass_rows
    }
    selected, selection_rows = select_cgt_support_households(
        household_ids, weights, income_band, wealth, support_by_band
    )
    copies = np.ones(len(household), dtype="int64")
    copies[selected] = cgt_support_copy_counts(
        weights[selected], CGT_SUPPORT_MAXIMUM_COPY_WEIGHT
    )
    multiplier = id_multiplier_for_values(
        person["person_id"],
        person["person_household_id"],
        person["person_benunit_id"],
        benunit["benunit_id"],
        household["household_id"],
    )
    household[HOUSEHOLD_IS_CGT_SUPPORT_COPY] = False
    household[CGT_SUPPORT_COPIES_COLUMN] = copies
    incumbent_weights = weights / copies
    copy_person: list[pd.DataFrame] = []
    copy_benunit: list[pd.DataFrame] = []
    copy_household: list[pd.DataFrame] = []
    copy_weights: list[np.ndarray] = []
    for k in range(1, int(copies.max())):
        family = copies > k
        family_ids = set(household_ids[family].tolist())
        members = person["person_household_id"].isin(family_ids).to_numpy()
        family_person = person.loc[members].copy()
        family_benunit = benunit.loc[
            benunit["benunit_id"].isin(set(family_person["person_benunit_id"]))
        ].copy()
        family_household = household.loc[family].copy()
        offset = k * multiplier
        for column in _PERSON_ID_COLUMNS:
            family_person[column] = family_person[column].astype("int64") + offset
        family_benunit["benunit_id"] = (
            family_benunit["benunit_id"].astype("int64") + offset
        )
        family_household["household_id"] = (
            family_household["household_id"].astype("int64") + offset
        )
        family_household[HOUSEHOLD_IS_CGT_SUPPORT_COPY] = True
        copy_person.append(family_person)
        copy_benunit.append(family_benunit)
        copy_household.append(family_household)
        copy_weights.append(incumbent_weights[family])
    final_person = pd.concat([person, *copy_person], ignore_index=True)
    final_benunit = pd.concat([benunit, *copy_benunit], ignore_index=True)
    final_household = pd.concat([household, *copy_household], ignore_index=True)
    final_weights = np.concatenate([incumbent_weights, *copy_weights])
    exact = _exact_support_weights(
        final_weights,
        target=old_total,
        unselected=np.flatnonzero(~selected & (incumbent_weights > 0.0)),
    )
    receipt = MassChangeRecord(
        entity="household",
        old_total=old_total,
        new_total=exact.total,
        declared_factor=1.0,
        reason=CGT_SUPPORT_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=final_person,
        benunit=final_benunit,
        household=final_household,
        time_period=uk_time_period(frame),
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=exact.values,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result)
    band_rows: list[dict[str, object]] = []
    for mass_row, selection_row in zip(mass_rows, selection_rows, strict=True):
        in_band = selected & (income_band == mass_row["income_lower_bound"])
        band_rows.append(
            {
                "income_lower_bound": int(mass_row["income_lower_bound"]),
                "published_top_band_taxpayers": float(
                    mass_row["published_top_band_taxpayers"]
                ),
                "suppressed_cells": int(mass_row["suppressed_cells"]),
                "support_mass": float(selection_row["support_mass"]),
                "pool_households": int(selection_row["pool_households"]),
                "pool_mass": float(selection_row["pool_mass"]),
                "households_selected": int(selection_row["households_selected"]),
                "copies_created": int((copies[in_band] - 1).sum()),
                "selected_mass": float(selection_row["selected_mass"]),
                "wealth_threshold": selection_row["wealth_threshold"],
                "heaviest_selected_weight": float(
                    selection_row["heaviest_selected_weight"]
                ),
                "heaviest_copy_weight": (
                    float(incumbent_weights[in_band].max()) if in_band.any() else 0.0
                ),
                "pool_exhausted": bool(selection_row["pool_exhausted"]),
            }
        )
    if _HOUSEHOLD_SUPPORT_CHANNEL_COLUMN in household.columns:
        channels = household.loc[selected, _HOUSEHOLD_SUPPORT_CHANNEL_COLUMN]
        frs_selected = int(channels.eq("frs").sum())
        spi_selected = int(channels.eq("spi").sum())
    else:
        frs_selected = spi_selected = 0
    return UKCGTSupportSplitResult(
        frame=result,
        band_rows=tuple(band_rows),
        published_top_band_taxpayers=float(
            sum(float(row["published_top_band_taxpayers"]) for row in mass_rows)
        ),
        suppressed_cells=int(sum(int(row["suppressed_cells"]) for row in mass_rows)),
        support_mass=float(sum(float(row["support_mass"]) for row in mass_rows)),
        households_selected=int(selected.sum()),
        copies_created=int((copies - 1).sum()),
        selected_mass=float(weights[selected].sum()),
        households_before=int(len(household)),
        households_after=int(len(final_household)),
        zero_weight_excluded=int((weights <= 0.0).sum()),
        old_total=float(old_total),
        new_total=float(exact.total),
        frs_selected=frs_selected,
        spi_selected=spi_selected,
    )


def _exact_support_weights(
    values: np.ndarray, *, target: float, unselected: np.ndarray
) -> Weights:
    """Importance weights summing bitwise to ``target``.

    The correction lands on the heaviest unselected incumbent so every split
    weight stays exactly ``w / n``; a frame with no unselected positive
    incumbent falls back to the shared multi-candidate correction.
    """

    if unselected.size:
        heaviest = int(unselected[np.argmax(values[unselected])])
        corrected = _exact_total_correction(
            values, target=target, correction_index=heaviest
        )
        if corrected is not None:
            return Weights(corrected, WeightKind.IMPORTANCE)
    return _importance_weights_with_exact_total(values, target)


def cgt_support_split_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operation the manifests must carry verbatim."""

    return (
        (
            CGT_SUPPORT_SPLIT_OPERATION_KIND,
            {
                "joint_resource": HMRC_CGT_CONDITIONING_RESOURCE,
                "joint_record_set_prefix": HMRC_CGT_JOINT_RECORD_SET_PREFIX,
                "joint_vintage": HMRC_CGT_SOURCE_VINTAGE,
                "taxpayer_concept": CGT_SUPPORT_TAXPAYER_CONCEPT,
                "income_band_lower_bounds": list(CGT_SUPPORT_INCOME_BAND_LOWER_BOUNDS),
                "minimum_gain_band_lower": CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
                "published_top_band_taxpayers": (
                    CGT_SUPPORT_PUBLISHED_TOP_BAND_TAXPAYERS
                ),
                "suppressed_cells": (
                    "published counts withheld as fewer than 1,000 count as zero "
                    "and are counted in the receipt"
                ),
                "clone_split_factor": CGT_SUPPORT_CLONE_SPLIT_FACTOR,
                "headroom": CGT_SUPPORT_HEADROOM,
                "expected_support_mass": CGT_SUPPORT_EXPECTED_MASS,
                "support_mass_rule": (
                    "clone_split_factor x headroom x published taxpayers with gains "
                    "at or above minimum_gain_band_lower in the income column"
                ),
                "income_band_measure": (
                    "taxable_income_proxy of the oldest adult (tapered Personal "
                    "Allowance from the policy_parameters artifact) digitised on "
                    "income_band_lower_bounds, as the Table 3 redraw classifies "
                    "gainers"
                ),
                "income_proxy_components": list(UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS),
                "allowance_subtraction": True,
                "adult_minimum_age": CGT_ADULT_MINIMUM_AGE,
                "carrier": "oldest adult; person_id ascending breaks age ties",
                "wealth_measure": (
                    "household sum of investable_wealth_columns; a missing or "
                    "non-finite value refuses the stage"
                ),
                "investable_wealth_columns": list(UK_CGT_INVESTABLE_WEALTH_COLUMNS),
                "selection": (
                    "within each income band walk households in investable wealth "
                    "descending, household_id ascending, accumulating positive "
                    "household weight until the cumulative weight reaches the "
                    "band's support mass; whole households; an exhausted pool is "
                    "recorded, not refused"
                ),
                "maximum_copy_weight": CGT_SUPPORT_MAXIMUM_COPY_WEIGHT,
                "copy_rule": (
                    "copies = ceil(household weight / maximum_copy_weight); every "
                    "copy carries weight / copies"
                ),
                "id_remapping": (
                    "id_multiplier_for_values; copy k takes every entity id + k x "
                    "multiplier"
                ),
                "flag_column": HOUSEHOLD_IS_CGT_SUPPORT_COPY,
                "copies_column": CGT_SUPPORT_COPIES_COLUMN,
                "values": "every column copied unchanged; no draw, no seed, no salt",
                "weight_kind_out": WeightKind.IMPORTANCE.value,
                "conservation": "exact_total",
                "declared_factor": 1.0,
                "reason": CGT_SUPPORT_MASS_CHANGE_REASON,
            },
        ),
    )


def _assert_cgt_support_split_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind every stage manifest parameter to reviewed code constants.

    Arm 1 of the two-arm rule: the manifest restates the code dictionary
    verbatim, the stage carries exactly the two output columns at household
    grain with no rewrites and the two reference artifacts, and the vintage
    pins are recomputed from the vendored joint so a re-vendored resource
    that moves them fails here until reviewed.
    """

    _assert_closed_world_operations(stage, cgt_support_split_operation_parameters())
    if stage.stage != CGT_SUPPORT_SPLIT_STAGE_NAME:
        raise ValueError(
            f"Expected stage {CGT_SUPPORT_SPLIT_STAGE_NAME!r}, got {stage.stage!r}."
        )
    expected_outputs = (HOUSEHOLD_IS_CGT_SUPPORT_COPY, CGT_SUPPORT_COPIES_COLUMN)
    if tuple(stage.outputs) != expected_outputs or tuple(stage.rewrites):
        raise ValueError(
            "The CGT support split writes exactly its flag and copy-count "
            f"columns; the manifest must declare outputs {expected_outputs} and "
            f"no rewrites, got {tuple(stage.outputs)} / {tuple(stage.rewrites)}."
        )
    if stage.grain != "household":
        raise ValueError(
            "The CGT support split moves household mass; grain must be "
            f"'household', got {stage.grain!r}."
        )
    artifacts = {str(artifact.get("role")): artifact for artifact in stage.artifacts}
    facts = artifacts.get("cgt_conditioning_facts")
    if (
        facts is None
        or facts.get("resource") != HMRC_CGT_CONDITIONING_RESOURCE
        or facts.get("runtime_sha256_required") is not True
    ):
        raise ValueError(
            "The CGT support split must declare the cgt_conditioning_facts "
            f"artifact {HMRC_CGT_CONDITIONING_RESOURCE!r} with "
            "runtime_sha256_required: true."
        )
    policy = artifacts.get("policy_parameters")
    declared = tuple(policy.get("parameters", ())) if policy is not None else ()
    if policy is None or any(name not in declared for name in _POLICY_PARAMETER_NAMES):
        raise ValueError(
            "The CGT support split must declare the Personal Allowance "
            f"policy-parameter artifact with {_POLICY_PARAMETER_NAMES}."
        )
    rows = cgt_support_mass_by_income_band(
        load_hmrc_cgt_joint_distribution(),
        clone_split_factor=CGT_SUPPORT_CLONE_SPLIT_FACTOR,
        headroom=CGT_SUPPORT_HEADROOM,
        minimum_gain_band_lower=CGT_SUPPORT_MINIMUM_GAIN_BAND_LOWER,
    )
    published = float(sum(float(row["published_top_band_taxpayers"]) for row in rows))
    support_mass = float(sum(float(row["support_mass"]) for row in rows))
    if abs(published - CGT_SUPPORT_PUBLISHED_TOP_BAND_TAXPAYERS) > 0.5 or (
        abs(support_mass - CGT_SUPPORT_EXPECTED_MASS) > 0.5
    ):
        raise ValueError(
            "HMRC Table 3 top-band taxpayers drifted from the reviewed "
            f"{HMRC_CGT_SOURCE_VINTAGE} pins: taxpayers {published} vs "
            f"{CGT_SUPPORT_PUBLISHED_TOP_BAND_TAXPAYERS}, support mass "
            f"{support_mass} vs {CGT_SUPPORT_EXPECTED_MASS}."
        )
