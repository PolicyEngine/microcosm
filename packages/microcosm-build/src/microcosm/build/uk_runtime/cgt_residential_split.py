"""UK capital-gains residential split: the residential-property flag carried as
weight, not drawn (microcosm#1063).

HMRC Table 8a counts the individuals who reported a residential property
disposal and their gains; the asset-type stage solved a logistic in log gains
to both totals and then realised the flag by a weighted systematic walk, one
whole flag per row. The walk's count error is bounded by one row's weight,
but its gains error by one row's stake (weight times gain), and at the top of
the gains distribution one stake is several times the band's whole expected
residential gains, so the realised gains sat either short of their expectation
by the lottery expectation of the giant stakes or over it by one of them.

This stage keeps the logistic and drops the lottery. Every household carrying
liable gainers is split into arms, identical in every cell except
``capital_gains_residential_property``: with one liable gainer of probability
``p`` the household's mass is divided into a non-residential arm at
``(1 - p) w`` (the household's own ids) and a residential arm at ``p w``
whose gainer's whole net gain is residential; with ``k`` liable gainers the
household becomes ``2 ** k`` arms at product weights, one per subset of
residential gainers (independent flags). At design weights the residential
count and gains are then the mass identities ``sum(p w)`` and
``sum(p w g)``, which the solve put on Table 8a, in every gain band and with
no draw, no offset and no seed; the calibration, which binds the Table 8a
totals, decides between the arms afterwards exactly as it decides between
the incidence clone's halves. Every household's mass and composition and the
total household mass are conserved; arm ``j`` takes every entity id plus
``j`` times the frame's id multiplier, the lineage rule the executor and the
geography identity kernel read for the support split and the clone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.cgt_asset_type import (
    CGT_RESIDENTIAL_GAINS_COLUMN,
    CGT_RESIDENTIAL_STOCK_SIGNAL,
    CGT_STOCK_LOG_ODDS,
    HMRC_CGT_ASSET_TYPE_RESOURCE,
    HMRCCGTAssetTypeFacts,
    _logistic,
    _stock_signals,
    _weighted_share,
    load_hmrc_cgt_asset_type_facts,
    solve_residential_logistic,
)
from microcosm.build.uk_runtime.cgt_imputation import (
    UKCGTPolicyParameters,
    uk_cgt_policy_parameters,
)
from microcosm.build.uk_runtime.cgt_structure import _assert_closed_world_operations
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
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

CGT_RESIDENTIAL_SPLIT_STAGE_NAME = "cgt_residential_split"
CGT_RESIDENTIAL_SPLIT_OPERATION_KIND = "split_liable_gainers_by_residential_probability"
#: True on every arm the split created; the household's own ids stay on the
#: arm with no residential gainer.
HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE = "household_is_cgt_residential_clone"
#: Arm index ``j`` (0 on the household's own arm and on unsplit households):
#: bit ``m`` of ``j`` says whether the household's ``m``-th liable gainer, in
#: person_id order, is residential on that arm.
CGT_RESIDENTIAL_CLONE_INDEX_COLUMN = "cgt_residential_clone_index"
#: The solved residential probability of each liable gainer (0 elsewhere),
#: carried on every arm so the arm weights can be reconstructed from ids.
CGT_RESIDENTIAL_PROBABILITY_COLUMN = "cgt_residential_probability"
#: A household with more liable gainers than this would need more than eight
#: arms; none exists on the FRS spine, so the stage refuses rather than grows.
CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS = 3
#: Concentration diagnostic: the share of the residential gains mass on the
#: largest residential arm stakes (arm weight times gain).
CGT_RESIDENTIAL_TOP_ARM_ROWS = 10
CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON = (
    "Capital-gains residential split divides every household carrying liable "
    "gainers into arms weighted by the solved residential probabilities, one "
    "arm per subset of residential gainers; every household's mass and the "
    "total household mass are conserved."
)
_PERSON_ID_COLUMNS = ("person_id", "person_household_id", "person_benunit_id")
_POLICY_PARAMETER_NAMES = ("gov.hmrc.cgt.annual_exempt_amount",)
_TABLE8_RECORD_SETS = (
    "hmrc.cgt_table8_2026.table8a.ty2024",
    "hmrc.cgt_table8_2026.table8b.ty2024",
)


@dataclass(frozen=True)
class UKCGTResidentialSplitResult:
    """Split frame and the executed-effect receipt of the residential split."""

    frame: Frame
    identities: Mapping[str, float]
    bands: tuple[Mapping[str, object], ...]
    logistic: Mapping[str, object]
    households_by_liable_gainers: Mapping[int, int]
    households_before: int
    households_after: int
    arms_created: int
    arm_weights: Mapping[str, object]
    concentration: Mapping[str, object]
    stock: Mapping[str, object]
    facts: Mapping[str, object]
    old_total: float
    new_total: float
    arm_weights_exact: bool

    def evidence(self) -> dict[str, object]:
        return {
            "stage": CGT_RESIDENTIAL_SPLIT_STAGE_NAME,
            "mass": {
                "old_total": float(self.old_total),
                "new_total": float(self.new_total),
            },
            "households_before": int(self.households_before),
            "households_after": int(self.households_after),
            "arms_created": int(self.arms_created),
            "households_by_liable_gainers": {
                str(k): int(v)
                for k, v in sorted(self.households_by_liable_gainers.items())
            },
            "maximum_liable_gainers_per_household": CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS,
            "arm_weights_exact": bool(self.arm_weights_exact),
            "identities": dict(self.identities),
            "bands": [dict(row) for row in self.bands],
            "logistic": dict(self.logistic),
            "arm_weights": dict(self.arm_weights),
            "concentration": dict(self.concentration),
            "stock": dict(self.stock),
            "facts": dict(self.facts),
        }


@dataclass(frozen=True)
class UKCGTResidentialSplitStageTransform:
    """Whole-stage transform for the residential split.

    ``facts`` default to the vendored Table 8 rows and ``parameters`` to the
    policyengine-uk annual exempt amount at the frame's build period, read at
    apply time so the base package never imports the engine at import time.
    """

    stage: SourceStageSpec
    facts: HMRCCGTAssetTypeFacts | None = None
    parameters: UKCGTPolicyParameters | None = None
    last_result: UKCGTResidentialSplitResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_cgt_residential_split_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_cgt_residential_split_stage_parameters(self.stage)
        facts = self.facts
        if facts is None:
            facts = load_hmrc_cgt_asset_type_facts()
        parameters = self.parameters
        if parameters is None:
            parameters = uk_cgt_policy_parameters(uk_time_period(frame))
        result = split_cgt_residential_households(
            frame, facts=facts, parameters=parameters
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (
            HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
            CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
            CGT_RESIDENTIAL_PROBABILITY_COLUMN,
            CGT_RESIDENTIAL_GAINS_COLUMN,
        )

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def residential_arm_factors(probabilities: np.ndarray) -> np.ndarray:
    """Weight factor of every arm of a household with these gainers.

    Row ``j`` of the result is the factor of arm ``j``: the product over the
    gainers of ``p`` where bit ``m`` of ``j`` is set and ``1 - p`` where it is
    not. Row 0 is the household's own arm; the rows sum to one.
    """

    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError("A household needs at least one liable gainer to split.")
    if not np.isfinite(p).all() or (p < 0.0).any() or (p > 1.0).any():
        raise ValueError("Residential probabilities must lie in [0, 1].")
    arms = np.arange(2**p.size)
    bits = ((arms[:, None] >> np.arange(p.size)[None, :]) & 1).astype(bool)
    return np.where(bits, p[None, :], 1.0 - p[None, :]).prod(axis=1)


def split_cgt_residential_households(
    frame: Frame,
    *,
    facts: HMRCCGTAssetTypeFacts,
    parameters: UKCGTPolicyParameters,
) -> UKCGTResidentialSplitResult:
    """Split every household carrying liable gainers into residential arms.

    Deterministic and mass-conserving: no draw, no seed, no salt. The logistic
    is solved exactly as the asset-type stage solved it (intercept to Table
    8a's count on the individuals basis, slope to its gains, with the stock
    shift); a household with ``k`` liable gainers becomes ``2 ** k`` arms at
    product weights, the arm with no residential gainer keeping the
    household's ids and every other arm ``j`` taking every entity id plus
    ``j`` times the frame's id multiplier. The exact-total correction lands on
    the heaviest unsplit incumbent so every arm weight stays bitwise its
    product. Refuses a frame the stage has already split, one the asset-type
    stage has already classified, or a household with more liable gainers
    than the declared maximum.
    """

    validate_uk_national_frame(frame)
    person = frame.table("person").copy().reset_index(drop=True)
    benunit = frame.table("benunit").copy().reset_index(drop=True)
    household = frame.table("household").copy().reset_index(drop=True)
    if "capital_gains" not in person.columns:
        raise ValueError("Person table has no capital_gains column to split on.")
    for column, table in (
        (HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE, household),
        (CGT_RESIDENTIAL_CLONE_INDEX_COLUMN, household),
        (CGT_RESIDENTIAL_PROBABILITY_COLUMN, person),
        (CGT_RESIDENTIAL_GAINS_COLUMN, person),
    ):
        if column in table.columns:
            raise ValueError(
                f"The CGT residential split already ran: the frame carries {column!r}."
            )
    weights = np.asarray(frame.weights_for("household").values, dtype=float)
    old_total = frame.weights_for("household").total
    household_ids = household["household_id"].to_numpy(dtype="int64")
    weight_by_household = pd.Series(weights, index=household_ids)
    person_weight = (
        person["person_household_id"].map(weight_by_household).to_numpy(dtype=float)
    )
    if not np.isfinite(person_weight).all():
        raise ValueError("Every person must map to a weighted household.")
    gains = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(gains).all():
        raise ValueError("capital_gains must be finite for every person.")
    liable = gains > parameters.annual_exempt_amount
    liable_index = np.flatnonzero(liable)
    liable_gains = gains[liable_index]
    liable_weights = person_weight[liable_index]
    stocks = _stock_signals(person, household)
    offset = CGT_STOCK_LOG_ODDS * stocks["residential"][liable_index].astype(float)
    count_target = float(facts.residential_taxpayers_individuals_basis)
    gains_target = float(facts.residential_gains_individuals_basis)
    a, b, centre = solve_residential_logistic(
        liable_gains,
        liable_weights,
        count_target=count_target,
        gains_target=gains_target,
        offset=offset,
    )
    probabilities = _logistic(np.log(liable_gains) - centre, a, b, offset)
    probability = np.zeros(len(person))
    probability[liable_index] = probabilities
    person[CGT_RESIDENTIAL_PROBABILITY_COLUMN] = probability
    person[CGT_RESIDENTIAL_GAINS_COLUMN] = 0.0
    household[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE] = False
    household[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN] = np.zeros(
        len(household), dtype="int64"
    )

    # Liable gainers by household, in person_id order within the household,
    # so bit m of an arm index always names the same gainer.
    liable_table = pd.DataFrame(
        {
            "position": liable_index,
            "household_id": person["person_household_id"].to_numpy(dtype="int64")[
                liable_index
            ],
            "person_id": person["person_id"].to_numpy(dtype="int64")[liable_index],
            "probability": probabilities,
        }
    ).sort_values(["household_id", "person_id"], kind="stable")
    liable_table["rank"] = liable_table.groupby("household_id").cumcount()
    gainers_per_household = liable_table.groupby("household_id").size()
    too_many = gainers_per_household[
        gainers_per_household > CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS
    ]
    if len(too_many):
        raise ValueError(
            f"{len(too_many)} household(s) carry more than "
            f"{CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS} liable gainers; the "
            "residential split refuses to enumerate their arms."
        )
    multiplier = id_multiplier_for_values(
        person["person_id"],
        person["person_household_id"],
        person["person_benunit_id"],
        benunit["benunit_id"],
        household["household_id"],
    )
    household_position = pd.Series(np.arange(len(household)), index=household_ids)
    incumbent_weights = weights.copy()
    arm_person: list[pd.DataFrame] = []
    arm_benunit: list[pd.DataFrame] = []
    arm_household: list[pd.DataFrame] = []
    arm_weights: list[np.ndarray] = []
    arm_exact = True
    households_by_k: dict[int, int] = {}
    for k, family_ids in gainers_per_household.groupby(gainers_per_household):
        k = int(k)
        family_ids = family_ids.index.to_numpy(dtype="int64")
        households_by_k[k] = int(family_ids.size)
        rows = liable_table.loc[liable_table["household_id"].isin(family_ids)]
        # Probability matrix: one row per family, one column per gainer rank.
        matrix = (
            rows.pivot(index="household_id", columns="rank", values="probability")
            .reindex(family_ids)
            .to_numpy(dtype=float)
        )
        factors = np.stack([residential_arm_factors(row) for row in matrix])
        if not np.allclose(factors.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            arm_exact = False
        positions = household_position.loc[family_ids].to_numpy()
        incumbent_weights[positions] = weights[positions] * factors[:, 0]
        family_set = set(family_ids.tolist())
        members = person["person_household_id"].isin(family_set).to_numpy()
        family_person = person.loc[members]
        family_benunit = benunit.loc[
            benunit["benunit_id"].isin(set(family_person["person_benunit_id"]))
        ]
        family_household = household.loc[positions]
        rank_by_position = pd.Series(
            rows["rank"].to_numpy(), index=rows["position"].to_numpy()
        )
        member_positions = np.flatnonzero(members)
        member_rank = rank_by_position.reindex(member_positions).to_numpy()
        for j in range(1, 2**k):
            id_offset = j * multiplier
            copy_person = family_person.copy()
            for column in _PERSON_ID_COLUMNS:
                copy_person[column] = copy_person[column].astype("int64") + id_offset
            residential = np.zeros(len(copy_person), dtype=bool)
            for m in range(k):
                if (j >> m) & 1:
                    residential |= member_rank == m
            copy_person[CGT_RESIDENTIAL_GAINS_COLUMN] = np.where(
                residential, gains[member_positions], 0.0
            )
            copy_benunit = family_benunit.copy()
            copy_benunit["benunit_id"] = (
                copy_benunit["benunit_id"].astype("int64") + id_offset
            )
            copy_household = family_household.copy()
            copy_household["household_id"] = (
                copy_household["household_id"].astype("int64") + id_offset
            )
            copy_household[HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE] = True
            copy_household[CGT_RESIDENTIAL_CLONE_INDEX_COLUMN] = np.full(
                len(copy_household), j, dtype="int64"
            )
            arm_person.append(copy_person)
            arm_benunit.append(copy_benunit)
            arm_household.append(copy_household)
            arm_weights.append(weights[positions] * factors[:, j])
    # Every arm id exceeds every incoming id (the multiplier is larger than
    # any id), so the arms follow the incumbents; within the arms the rows are
    # ordered by id so the group tables stay sorted whatever the household
    # order the arms were built in.
    arms_person = pd.concat(arm_person, ignore_index=True) if arm_person else None
    arms_benunit = pd.concat(arm_benunit, ignore_index=True) if arm_benunit else None
    arms_household = (
        pd.concat(arm_household, ignore_index=True) if arm_household else None
    )
    arms_weights = (
        np.concatenate(arm_weights) if arm_weights else np.zeros(0, dtype=float)
    )
    if arms_household is not None:
        household_order = np.argsort(
            arms_household["household_id"].to_numpy(dtype="int64"), kind="stable"
        )
        arms_household = arms_household.iloc[household_order].reset_index(drop=True)
        arms_weights = arms_weights[household_order]
        arms_benunit = arms_benunit.iloc[
            np.argsort(
                arms_benunit["benunit_id"].to_numpy(dtype="int64"), kind="stable"
            )
        ].reset_index(drop=True)
        arms_person = arms_person.iloc[
            np.argsort(arms_person["person_id"].to_numpy(dtype="int64"), kind="stable")
        ].reset_index(drop=True)
    final_person = pd.concat(
        [person, *([arms_person] if arms_person is not None else [])],
        ignore_index=True,
    )
    final_benunit = pd.concat(
        [benunit, *([arms_benunit] if arms_benunit is not None else [])],
        ignore_index=True,
    )
    final_household = pd.concat(
        [household, *([arms_household] if arms_household is not None else [])],
        ignore_index=True,
    )
    final_weights = np.concatenate([incumbent_weights, arms_weights])
    split_households = np.zeros(len(household), dtype=bool)
    split_households[household_position.loc[gainers_per_household.index].to_numpy()] = (
        True
    )
    exact = _exact_residential_weights(
        final_weights,
        target=old_total,
        unsplit=np.flatnonzero(~split_households & (incumbent_weights > 0.0)),
    )
    receipt = MassChangeRecord(
        entity="household",
        old_total=old_total,
        new_total=exact.total,
        declared_factor=1.0,
        reason=CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON,
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

    # Receipt: the identities at design weights, by gain band and in all.
    final_household_weight = pd.Series(
        exact.values, index=final_household["household_id"].to_numpy(dtype="int64")
    )
    final_person_weight = (
        final_person["person_household_id"].map(final_household_weight).to_numpy(float)
    )
    final_gains = final_person["capital_gains"].to_numpy(dtype=float)
    final_residential = final_person[CGT_RESIDENTIAL_GAINS_COLUMN].to_numpy(float)
    is_residential = final_residential > 0.0
    expected_count = float((liable_weights * probabilities).sum())
    expected_gains = float((liable_weights * liable_gains * probabilities).sum())
    achieved_count = float(final_person_weight[is_residential].sum())
    achieved_gains = float(
        (final_person_weight * final_residential)[is_residential].sum()
    )
    bounds = np.asarray(HMRC_CGT_GAIN_BAND_LOWER_BOUNDS, dtype=float)
    liable_band = np.searchsorted(bounds, liable_gains, side="right") - 1
    final_band = np.searchsorted(bounds, final_gains, side="right") - 1
    liable_stakes = liable_weights * liable_gains
    residential_stakes = (final_person_weight * final_residential)[is_residential]
    band_rows: list[dict[str, object]] = []
    for position, lower in enumerate(bounds):
        in_band = liable_band == position
        out_band = is_residential & (final_band == position)
        arms_in_band = final_person_weight[out_band]
        band_rows.append(
            {
                "gain_lower_bound": float(lower),
                "gain_upper_bound": (
                    float(bounds[position + 1]) if position + 1 < bounds.size else None
                ),
                "liable_rows": int(in_band.sum()),
                "liable_mass": float(liable_weights[in_band].sum()),
                "liable_gains": float(liable_stakes[in_band].sum()),
                "heaviest_liable_weight": (
                    float(liable_weights[in_band].max()) if in_band.any() else 0.0
                ),
                "largest_liable_stake": (
                    float(liable_stakes[in_band].max()) if in_band.any() else 0.0
                ),
                "expected_count": float(
                    (liable_weights[in_band] * probabilities[in_band]).sum()
                ),
                "expected_gains": float(
                    (liable_stakes[in_band] * probabilities[in_band]).sum()
                ),
                "achieved_count": float(arms_in_band.sum()),
                "achieved_gains": float(
                    (final_person_weight * final_residential)[out_band].sum()
                ),
                "residential_rows": int(out_band.sum()),
                "largest_residential_arm_weight": (
                    float(arms_in_band.max()) if out_band.any() else 0.0
                ),
            }
        )
    sorted_stakes = np.sort(residential_stakes)[::-1]
    top_stakes = float(sorted_stakes[:CGT_RESIDENTIAL_TOP_ARM_ROWS].sum())
    residential_arm_weights = exact.values[len(household) :]
    identities = {
        "count_target_individuals_basis": count_target,
        "gains_target_individuals_basis": gains_target,
        "expected_count": expected_count,
        "expected_gains": expected_gains,
        "achieved_count": achieved_count,
        "achieved_gains": achieved_gains,
        "count_solve_relative_error": abs(expected_count - count_target) / count_target,
        "gains_solve_relative_error": abs(expected_gains - gains_target) / gains_target,
        "count_identity_relative_error": (
            abs(achieved_count - expected_count) / expected_count
        ),
        "gains_identity_relative_error": (
            abs(achieved_gains - expected_gains) / expected_gains
        ),
        "liable_taxpayer_mass": float(liable_weights.sum()),
        "liable_gains_mass": float(liable_stakes.sum()),
    }
    return UKCGTResidentialSplitResult(
        frame=result,
        identities=identities,
        bands=tuple(band_rows),
        logistic={
            "intercept": float(a),
            "slope": float(b),
            "log_gain_centre": float(centre),
            "stock_log_odds": CGT_STOCK_LOG_ODDS,
            "stock_signal": CGT_RESIDENTIAL_STOCK_SIGNAL,
        },
        households_by_liable_gainers=households_by_k,
        households_before=int(len(household)),
        households_after=int(len(final_household)),
        arms_created=int(len(final_household) - len(household)),
        arm_weights={
            "residential_arms": int(residential_arm_weights.size),
            "minimum": (
                float(residential_arm_weights.min())
                if residential_arm_weights.size
                else 0.0
            ),
            "maximum": (
                float(residential_arm_weights.max())
                if residential_arm_weights.size
                else 0.0
            ),
            "below_one_household": int((residential_arm_weights < 1.0).sum()),
            "below_one_tenth": int((residential_arm_weights < 0.1).sum()),
        },
        concentration={
            "top_arm_rows": CGT_RESIDENTIAL_TOP_ARM_ROWS,
            "top_arms_share_of_achieved_gains": (
                top_stakes / achieved_gains if achieved_gains > 0.0 else 0.0
            ),
            "largest_residential_arm_stake": (
                float(sorted_stakes[0]) if sorted_stakes.size else 0.0
            ),
            "largest_liable_stake": (
                float(liable_stakes.max()) if liable_stakes.size else 0.0
            ),
            "largest_liable_stake_share_of_gains_target": (
                float(liable_stakes.max()) / gains_target
                if liable_stakes.size and gains_target > 0.0
                else 0.0
            ),
            "max_liable_weight": (
                float(liable_weights.max()) if liable_weights.size else 0.0
            ),
        },
        stock={
            "share_liable": _weighted_share(
                person_weight, liable, stocks["residential"]
            ),
            "expected_share_residential": (
                float(
                    (liable_weights * probabilities)[
                        stocks["residential"][liable_index]
                    ].sum()
                )
                / expected_count
                if expected_count > 0.0
                else None
            ),
        },
        facts={
            "table8a_taxpayers_total": float(facts.table8a_taxpayers_total),
            "table8a_gains_total": float(facts.table8a_gains_total),
            "table8b_individuals_taxpayer_share": float(
                facts.individuals_share("taxpayers")
            ),
            "table8b_individuals_gains_share": float(facts.individuals_share("gains")),
        },
        old_total=float(old_total),
        new_total=float(exact.total),
        arm_weights_exact=arm_exact,
    )


def _exact_residential_weights(
    values: np.ndarray, *, target: float, unsplit: np.ndarray
) -> Weights:
    """Importance weights summing bitwise to ``target``.

    The correction lands on the heaviest unsplit incumbent so every arm weight
    stays exactly its product; a frame with no unsplit positive incumbent
    falls back to the shared multi-candidate correction.
    """

    if unsplit.size:
        heaviest = int(unsplit[np.argmax(values[unsplit])])
        corrected = _exact_total_correction(
            values, target=target, correction_index=heaviest
        )
        if corrected is not None:
            return Weights(corrected, WeightKind.IMPORTANCE)
    return _importance_weights_with_exact_total(values, target)


def cgt_residential_split_operation_parameters() -> tuple[
    tuple[str, dict[str, Any]], ...
]:
    """The reviewed operations the manifests must carry verbatim."""

    return (
        (
            "verify_vendored_fact_resource",
            {
                "artifact_role": "cgt_asset_type_facts",
                "resource": HMRC_CGT_ASSET_TYPE_RESOURCE,
                "feed_pin": "chronicle_feed.json",
                "record_sets": list(_TABLE8_RECORD_SETS),
                "source_vintage": "2024-25",
                "mapped_build_period": 2024,
                "period_mapping": "published_tax_year_equals_build_period",
                "require_before_source_read": True,
                "runtime_sha256_required": True,
                "fail_on_mismatch": True,
            },
        ),
        (
            CGT_RESIDENTIAL_SPLIT_OPERATION_KIND,
            {
                "population": (
                    "persons with net capital gains above the annual exempt amount "
                    "(the national taxpayer proxy)"
                ),
                "model": (
                    "logistic probability in centred log gains with a stock shift, "
                    "p = 1 / (1 + exp(-(a + b (log g - c) + s h))) with c the weighted "
                    "mean log gain of the population, s the stock log-odds and h one "
                    "where the stock signal holds, zero otherwise"
                ),
                "stock_log_odds": CGT_STOCK_LOG_ODDS,
                "stock_signal": CGT_RESIDENTIAL_STOCK_SIGNAL,
                "parameter_solver": (
                    "nested bisection; the intercept matches the expected weighted "
                    "count at each trial slope, the slope matches the expected "
                    "weighted gains"
                ),
                "count_target": (
                    "Table 8a 2024-25 total taxpayers reporting residential property "
                    "disposals times the Table 8b 2024-25 individuals/all taxpayer "
                    "share on the UK Property service"
                ),
                "gains_target": (
                    "Table 8a 2024-25 total residential property gains times the "
                    "Table 8b 2024-25 individuals/all gains share on the UK Property "
                    "service"
                ),
                "basis_assumption": (
                    "trusts hold the same share of the Self Assessment component as "
                    "of the UK Property service, a residential gainer's whole net "
                    "gain is attributed to residential property, and the gainers of "
                    "one household are residential independently of each other"
                ),
                "realization": (
                    "probability carried as weight, no draw: a household carrying k "
                    "liable gainers becomes 2^k arms, one per subset of residential "
                    "gainers, at the household's weight times the product of p over "
                    "the residential gainers and 1 - p over the others; the arm with "
                    "no residential gainer keeps the household's ids; at design "
                    "weights the residential count and gains equal the solved "
                    "expectations sum(p w) and sum(p w g) in every gain band"
                ),
                "maximum_liable_gainers_per_household": (
                    CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS
                ),
                "gainer_order": "person_id ascending within the household",
                "id_remapping": (
                    "id_multiplier_for_values; arm j takes every entity id + j x "
                    "multiplier, bit m of j marking the household's m-th liable "
                    "gainer residential"
                ),
                "flag_column": HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE,
                "index_column": CGT_RESIDENTIAL_CLONE_INDEX_COLUMN,
                "probability_column": CGT_RESIDENTIAL_PROBABILITY_COLUMN,
                "output_column": CGT_RESIDENTIAL_GAINS_COLUMN,
                "output_semantics": (
                    "the person's capital_gains on an arm where the person is "
                    "residential, 0 otherwise"
                ),
                "values": "every other column copied unchanged; no draw, no seed, no salt",
                "weight_kind_out": WeightKind.IMPORTANCE.value,
                "conservation": "exact_total",
                "declared_factor": 1.0,
                "reason": CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON,
            },
        ),
    )


def _assert_cgt_residential_split_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind every stage manifest parameter to reviewed code constants."""

    _assert_closed_world_operations(stage, cgt_residential_split_operation_parameters())
    if stage.stage != CGT_RESIDENTIAL_SPLIT_STAGE_NAME:
        raise ValueError(
            f"Expected stage {CGT_RESIDENTIAL_SPLIT_STAGE_NAME!r}, got {stage.stage!r}."
        )
    expected_outputs = UKCGTResidentialSplitStageTransform.output_columns()
    if tuple(stage.outputs) != expected_outputs or tuple(stage.rewrites):
        raise ValueError(
            "The CGT residential split writes exactly its flag, arm-index, "
            "probability and residential-gains columns; the manifest must declare "
            f"outputs {expected_outputs} and no rewrites, got "
            f"{tuple(stage.outputs)} / {tuple(stage.rewrites)}."
        )
    if stage.grain != "household":
        raise ValueError(
            "The CGT residential split moves household mass; grain must be "
            f"'household', got {stage.grain!r}."
        )
    artifacts = {str(artifact.get("role")): artifact for artifact in stage.artifacts}
    facts = artifacts.get("cgt_asset_type_facts")
    if (
        facts is None
        or facts.get("resource") != HMRC_CGT_ASSET_TYPE_RESOURCE
        or facts.get("runtime_sha256_required") is not True
    ):
        raise ValueError(
            "The CGT residential split must declare the cgt_asset_type_facts "
            f"artifact {HMRC_CGT_ASSET_TYPE_RESOURCE!r} with "
            "runtime_sha256_required: true."
        )
    policy = artifacts.get("policy_parameters")
    declared = tuple(policy.get("parameters", ())) if policy is not None else ()
    if policy is None or any(name not in declared for name in _POLICY_PARAMETER_NAMES):
        raise ValueError(
            "The CGT residential split must declare the annual exempt amount "
            f"policy-parameter artifact with {_POLICY_PARAMETER_NAMES}."
        )


__all__ = [
    "CGT_RESIDENTIAL_CLONE_INDEX_COLUMN",
    "CGT_RESIDENTIAL_MAXIMUM_LIABLE_GAINERS",
    "CGT_RESIDENTIAL_PROBABILITY_COLUMN",
    "CGT_RESIDENTIAL_SPLIT_MASS_CHANGE_REASON",
    "CGT_RESIDENTIAL_SPLIT_OPERATION_KIND",
    "CGT_RESIDENTIAL_SPLIT_STAGE_NAME",
    "HOUSEHOLD_IS_CGT_RESIDENTIAL_CLONE",
    "UKCGTResidentialSplitResult",
    "UKCGTResidentialSplitStageTransform",
    "cgt_residential_split_operation_parameters",
    "residential_arm_factors",
    "split_cgt_residential_households",
]
