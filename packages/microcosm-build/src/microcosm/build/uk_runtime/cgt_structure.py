"""UK capital-gains incidence cloning and HMRC size-band support."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from microcosm.build.source_manifest import SourceOperationSpec, SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.cgt_imputation import (
    UK_CGT_TAXABLE_INCOME_PROXY_COMPONENTS,
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
CGT_QUANTILE_POINTS = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
CGT_PRIOR_PERCENTILE_COLUMNS = ("p05", "p10", "p25", "p50", "p75", "p90", "p95")
CGT_ADULT_MINIMUM_AGE = 16
DONORS_PER_BAND = 30
DONOR_BAND_COUNT = 9
DONOR_TOTAL = 270
MIN_DONOR_BAND_LOWER = 12_300
DONOR_SEED = 1
DONOR_NEVER_ZERO_WEIGHT = True
HOUSEHOLD_IS_CGT_CLONE = "household_is_capital_gains_clone"
HOUSEHOLD_IS_CGT_BAND_DONOR = "household_is_cgt_band_donor"
CGT_CLONE_MASS_CHANGE_REASON = (
    "Capital-gains incidence clone splits every household's mass equally across "
    "original and clone records; total household mass is conserved."
)
CGT_DONOR_MASS_CHANGE_REASON = (
    "Stack 30 positive-weight HMRC Table 2.1a support households per retained "
    "gain band; published donor mass is added explicitly."
)


def load_advani_summers_distribution() -> Mapping[str, Any]:
    """Load the committed Advani-Summers incidence and quantile surface."""

    return json.loads(
        files("microcosm.build.uk")
        .joinpath("advani_summers_capital_gains_distribution.json")
        .read_text(encoding="utf-8")
    )


def load_hmrc_cgt_size_bands() -> Mapping[str, Any]:
    """Load the committed HMRC Table 2.1a size-band surface."""

    return json.loads(
        files("microcosm.build.uk")
        .joinpath("hmrc_cgt_size_bands.json")
        .read_text(encoding="utf-8")
    )


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
class UKCGTBandDonorResult:
    """Band-donor frame and the executed-effect receipt for stage 20."""

    frame: Frame
    band_rows: tuple[Mapping[str, object], ...]
    frs_donors: int
    spi_donors: int

    def evidence(self) -> dict[str, object]:
        return {
            "stage": "cgt_band_donors",
            "bands": [dict(row) for row in self.band_rows],
            "support_channel_split": {
                "frs": self.frs_donors,
                "spi": self.spi_donors,
            },
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
class UKCGTBandDonorStageTransform:
    """Whole-stage transform for positive-weight HMRC size-band donors."""

    stage: SourceStageSpec
    size_bands: Mapping[str, Any] | None = None
    distribution: Mapping[str, Any] | None = None
    last_result: UKCGTBandDonorResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        bands = self.size_bands or load_hmrc_cgt_size_bands()
        distribution = self.distribution or load_advani_summers_distribution()
        _assert_cgt_donor_stage_parameters(self.stage, size_bands=bands)
        result = stack_cgt_band_donors(
            frame,
            size_bands=bands,
            distribution=distribution,
        )
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (HOUSEHOLD_IS_CGT_BAND_DONOR, "capital_gains")

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


def stack_cgt_band_donors(
    frame: Frame,
    *,
    size_bands: Mapping[str, Any],
    distribution: Mapping[str, Any],
) -> UKCGTBandDonorResult:
    """Add 30 households per retained HMRC size band at band-exact weights."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    household[HOUSEHOLD_IS_CGT_BAND_DONOR] = False
    bands = _retained_size_bands(size_bands)
    carriers = _oldest_adult_indices(person, household_ids=set(household.household_id))
    candidates = person.loc[carriers].copy()
    candidates["_income"] = _component_sum_income(candidates)
    candidates["_propensity"] = _incidence_propensity(
        candidates["_income"].to_numpy(dtype=float), distribution=distribution
    )
    candidates = candidates.sort_values("person_household_id", kind="stable")
    if len(candidates) < DONOR_TOTAL:
        raise ValueError(
            f"CGT donor stage requires at least {DONOR_TOTAL} candidate households; "
            f"found {len(candidates)}."
        )
    propensities = candidates["_propensity"].to_numpy(dtype=float)
    if not np.isfinite(propensities).all() or (propensities < 0).any():
        raise ValueError("CGT donor propensities must be finite and non-negative.")
    if propensities.sum() <= 0:
        raise ValueError("CGT donor propensities have no positive mass.")
    rng = np.random.default_rng(DONOR_SEED)
    selected = rng.choice(
        candidates["person_household_id"].to_numpy(),
        size=DONOR_TOTAL,
        replace=False,
        p=propensities / propensities.sum(),
    )
    selected_set = set(selected.tolist())
    donor_person = person.loc[person.person_household_id.isin(selected_set)].copy()
    donor_benunit_ids = set(donor_person.person_benunit_id)
    donor_benunit = benunit.loc[benunit.benunit_id.isin(donor_benunit_ids)].copy()
    donor_household = household.loc[household.household_id.isin(selected_set)].copy()
    multiplier = id_multiplier_for_values(
        person["person_id"],
        person["person_household_id"],
        person["person_benunit_id"],
        benunit["benunit_id"],
        household["household_id"],
    )
    for column in ("person_id", "person_household_id", "person_benunit_id"):
        donor_person[column] = donor_person[column].astype("int64") + multiplier
    donor_benunit["benunit_id"] = (
        donor_benunit["benunit_id"].astype("int64") + multiplier
    )
    donor_household["household_id"] = (
        donor_household["household_id"].astype("int64") + multiplier
    )
    position = {household_id: index for index, household_id in enumerate(selected)}
    donor_household["_band_position"] = (
        donor_household["household_id"].sub(multiplier).map(position)
    )
    if donor_household["_band_position"].isna().any():
        raise ValueError("CGT donor selection failed to map every donor household.")
    donor_household = donor_household.sort_values("_band_position", kind="stable")
    band_index = (
        donor_household["_band_position"].to_numpy(dtype=int) // DONORS_PER_BAND
    )
    taxpayers = np.asarray([row["taxpayers"] for row in bands], dtype=float)
    means = np.asarray([row["mean_gain"] for row in bands], dtype=float)
    donor_weights = taxpayers[band_index] / DONORS_PER_BAND
    if DONOR_NEVER_ZERO_WEIGHT and not (donor_weights > 0.0).all():
        raise ValueError("CGT band donors must all carry positive initial weight.")
    donor_household[HOUSEHOLD_IS_CGT_BAND_DONOR] = True
    gain_by_household = dict(
        zip(donor_household.household_id, means[band_index], strict=True)
    )
    evidence_band_index = band_index.copy()
    evidence_donor_weights = donor_weights.copy()
    donor_household["_donor_weight"] = donor_weights
    donor_household = donor_household.drop(columns=["_band_position"]).sort_values(
        "household_id", kind="stable"
    )
    donor_weights = donor_household.pop("_donor_weight").to_numpy(dtype=float)
    donor_person["capital_gains"] = 0.0
    donor_carriers = _oldest_adult_indices(
        donor_person,
        household_ids=set(donor_household.household_id),
    )
    donor_person.loc[donor_carriers, "capital_gains"] = donor_person.loc[
        donor_carriers, "person_household_id"
    ].map(gain_by_household)
    final_person = pd.concat([person, donor_person], ignore_index=True)
    final_benunit = pd.concat([benunit, donor_benunit], ignore_index=True)
    final_household = pd.concat([household, donor_household], ignore_index=True)
    final_weights = np.r_[frame.weights_for("household").values, donor_weights]
    old_total = frame.weights_for("household").total
    new_total = float(final_weights.sum())
    receipt = MassChangeRecord(
        entity="household",
        old_total=old_total,
        new_total=new_total,
        declared_factor=None,
        reason=CGT_DONOR_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=final_person,
        benunit=final_benunit,
        household=final_household,
        time_period=uk_time_period(frame),
        weight_kind=WeightKind.IMPORTANCE,
        household_weights=final_weights,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result)
    channels = household.set_index("household_id").get("household_support_channel")
    original_selected = donor_household.household_id.sub(multiplier)
    selected_channels = (
        original_selected.map(channels).fillna("unknown")
        if channels is not None
        else pd.Series("unknown", index=original_selected.index)
    )
    band_rows: list[Mapping[str, object]] = []
    for index, band in enumerate(bands):
        mask = evidence_band_index == index
        band_rows.append(
            {
                "lower_limit": band["lower_limit"],
                "donor_count": int(mask.sum()),
                "donor_weight": float(evidence_donor_weights[mask][0]),
                "weighted_taxpayers": float(evidence_donor_weights[mask].sum()),
                "mean_gain": band["mean_gain"],
            }
        )
    return UKCGTBandDonorResult(
        frame=result,
        band_rows=tuple(band_rows),
        frs_donors=int(selected_channels.eq("frs").sum()),
        spi_donors=int(selected_channels.eq("spi").sum()),
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


def _distribution_rows(resource: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = resource.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Advani-Summers resource must contain a non-empty rows list.")
    minimums = [float(row["minimum_total_income"]) for row in rows]
    if minimums != sorted(minimums) or minimums[0] != 0.0:
        raise ValueError("Advani-Summers income bands must be sorted and start at 0.")
    # Fail closed on non-monotone quantile rows: the prior draw interpolates
    # and extrapolates these values as a quantile function, so a malformed
    # row (the class the corrected percentile-69 dropped digit belonged to)
    # would silently fabricate loss-makers instead of failing the build.
    for row in rows:
        knots = [float(row[column]) for column in CGT_PRIOR_PERCENTILE_COLUMNS]
        if any(late < early for early, late in zip(knots, knots[1:], strict=False)):
            raise ValueError(
                "Advani-Summers quantile columns must be non-decreasing; "
                f"row with minimum_total_income {row['minimum_total_income']!r} "
                "is not a valid quantile function."
            )
    return rows


def _draw_banded_priors(
    income: np.ndarray,
    draws: np.ndarray,
    *,
    distribution: Mapping[str, Any],
) -> np.ndarray:
    rows = _distribution_rows(distribution)
    minimums = np.asarray([row["minimum_total_income"] for row in rows], dtype=float)
    indexes = np.clip(
        np.searchsorted(minimums, income, side="right") - 1, 0, len(rows) - 1
    )
    values = np.zeros(len(income), dtype=float)
    for index, row in enumerate(rows):
        mask = indexes == index
        if not mask.any():
            continue
        knots = np.asarray(
            [row[column] for column in CGT_PRIOR_PERCENTILE_COLUMNS], dtype=float
        )
        spline = UnivariateSpline(CGT_QUANTILE_POINTS, knots, k=1, s=0, ext=0)
        values[mask] = spline(draws[mask])
    return values


def _incidence_propensity(
    income: np.ndarray,
    *,
    distribution: Mapping[str, Any],
) -> np.ndarray:
    rows = _distribution_rows(distribution)
    minimums = np.asarray([row["minimum_total_income"] for row in rows], dtype=float)
    indexes = np.clip(
        np.searchsorted(minimums, income, side="right") - 1, 0, len(rows) - 1
    )
    rates = np.asarray([row["percent_with_gains"] for row in rows], dtype=float)
    return rates[indexes]


def _retained_size_bands(resource: Mapping[str, Any]) -> list[dict[str, float]]:
    rows = resource.get("rows")
    if not isinstance(rows, list):
        raise ValueError("HMRC CGT size-band resource must contain a rows list.")
    retained: list[dict[str, float]] = []
    for row in rows:
        lower = float(row["lower_limit"])
        if lower < MIN_DONOR_BAND_LOWER:
            continue
        taxpayers = float(row["taxpayers_thousands"]) * 1_000.0
        gains = float(row["gains_gbp_millions"]) * 1_000_000.0
        if taxpayers <= 0.0:
            raise ValueError(
                "Retained CGT size bands may not produce a zero initial weight."
            )
        retained.append(
            {
                "lower_limit": lower,
                "taxpayers": taxpayers,
                "gains": gains,
                "mean_gain": gains / taxpayers,
            }
        )
    return retained


def _operation(stage: SourceStageSpec, kind: str) -> SourceOperationSpec:
    matches = [operation for operation in stage.operations if operation.kind == kind]
    if len(matches) != 1:
        raise ValueError(
            f"Stage {stage.stage!r} must declare exactly one {kind!r} operation."
        )
    return matches[0]


def _assert_parameters(
    operation: SourceOperationSpec,
    expected: Mapping[str, object],
) -> None:
    for name, value in expected.items():
        actual = operation.parameters.get(name)
        if actual != value:
            raise ValueError(
                f"{operation.kind} manifest parameter {name!r} drifted: "
                f"expected {value!r}, got {actual!r}."
            )


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
                    "resource": "advani_summers_capital_gains_distribution.json",
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


def _assert_cgt_donor_stage_parameters(
    stage: SourceStageSpec,
    *,
    size_bands: Mapping[str, Any],
) -> None:
    """Bind stage-20 parameters and recompute the band/weight invariants."""

    _assert_closed_world_operations(
        stage,
        (
            (
                "stack_band_donor_households",
                {
                    "size_band_resource": "hmrc_cgt_size_bands.json",
                    "incidence_resource": (
                        "advani_summers_capital_gains_distribution.json"
                    ),
                    "minimum_band_lower": MIN_DONOR_BAND_LOWER,
                    "donors_per_band": DONORS_PER_BAND,
                    "expected_band_count": DONOR_BAND_COUNT,
                    "expected_donor_count": DONOR_TOTAL,
                    "candidate_order": "household_id ascending",
                    "draw": "weighted_without_replacement",
                    "propensity": (
                        "Advani-Summers percent_with_gains at oldest-adult "
                        "component-sum income"
                    ),
                    "seed": DONOR_SEED,
                    "flag_column": HOUSEHOLD_IS_CGT_BAND_DONOR,
                    "carrier": "oldest adult; person_id ascending breaks age ties",
                    "initial_weight": "published band taxpayers / donors_per_band",
                    "never_zero_weight": DONOR_NEVER_ZERO_WEIGHT,
                    "weight_kind_out": WeightKind.IMPORTANCE.value,
                    "reason": CGT_DONOR_MASS_CHANGE_REASON,
                },
            ),
        ),
    )
    bands = _retained_size_bands(size_bands)
    if len(bands) != DONOR_BAND_COUNT:
        raise ValueError(
            f"HMRC retained donor-band count drifted: expected {DONOR_BAND_COUNT}, "
            f"got {len(bands)}."
        )
    if DONORS_PER_BAND * len(bands) != DONOR_TOTAL:
        raise ValueError("CGT donor count no longer equals 30 times retained bands.")
    weights = np.asarray([band["taxpayers"] / DONORS_PER_BAND for band in bands])
    if DONOR_NEVER_ZERO_WEIGHT and not (weights > 0.0).all():
        raise ValueError("HMRC retained donor bands imply a zero initial weight.")
