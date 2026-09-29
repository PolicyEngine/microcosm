"""Assign a main asset type and a residential-property flag to CGT taxpayers.

The amounts redraw (``cgt_imputation``) places net gains on persons; nothing
says what was sold. Two HMRC surfaces do, and both are vendored verbatim
from the pinned Chronicle feed into ``hmrc_cgt_asset_type_facts.json``:

- **Table 8 (administrative, 2024-25).** Table 8a counts UK-resident
  taxpayers reporting residential property disposals, with their gains and
  tax, across the CGT on UK Property service and Self Assessment; Table 8b
  splits the service channel by taxpayer type (individuals, trusts). The
  national CGT targets are individuals only, so the Table 8a totals are
  restated on the individuals basis by the Table 8b individuals/all share
  for the same measure — the same declared assumption the Table 5 region
  targets make — and those two numbers are what the residential flag is
  solved to and what the residential targets bind (microcosm#725).
- **Table 7 (sample-based, 2023-24).** Disposals, proceeds and gains by asset
  type. It counts disposals rather than taxpayers, is a year older, and
  includes trusts and non-UK assets, so it is fenced from calibration; it
  seeds the composition of the non-residential remainder and the stage
  reports achieved against it as a diagnostic only.

Two declared mechanisms, in order:

1. **Residential flag.** Among liable gainers (net gains above the annual
   exempt amount, the national taxpayer proxy) the probability of holding
   a residential disposal is logistic in centred log gains, ``p = 1 / (1 +
   exp(-(a + b (log g - c))))`` with ``c`` the weighted mean log gain of the
   liable population. ``(a, b)`` are solved by nested bisection so the
   household-weighted expected count and gains of flagged persons equal the
   individuals-basis Table 8a totals: residential gainers are many and
   small relative to the whole (a fifth of the gains on a third of the
   taxpayers), which the negative slope carries. Flags are then realised by
   a weighted systematic walk in ascending gain order with one seeded
   offset: the walk flags a person whenever the expected weight owed so far
   reaches that person's weight, so the flagged weight tracks the expected
   weight within one person's weight everywhere along the gains axis, and
   both the realised count and the realised gains sit close to their
   expectations on frames whose liable rows are few and heavy. The
   Bernoulli sigma of each is reported as the envelope a gate can bound
   with. A flagged person's whole net
   gain is attributed to residential property (``capital_gains_residential_property``);
   a taxpayer with both residential and other disposals is not split.
2. **Main asset type.** Every liable gainer not flagged residential draws
   one of five Table 7 types from a size-tilted categorical: the type
   weight times a log-normal kernel in log gains centred on the type's
   Table 7 mean gain per disposal (listed shares near £5,500, unlisted
   shares near £115,000) with one declared log-scale width. The five type
   weights are fitted by multiplicative updates until the household-weighted
   gains shares by type reproduce Table 7's non-residential gains shares;
   one seeded uniform per person realises the draw. The composition by
   gain band is reported for the evidence note; nothing is fitted to it.

Non-gainers carry ``none``; positive gains at or below the annual exempt
amount carry ``sub_aea`` (Table 8 describes taxpayers with a liability and
the sub-AEA remainder is never a taxpayer). Household weights pass through
untouched and the stage appends a mass-conservation receipt the terminal
family gate requires.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.cgt_imputation import (
    UKCGTPolicyParameters,
    uk_cgt_policy_parameters,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_GAIN_BAND_LOWER_BOUNDS,
)
from microcosm.build.uk_runtime.ledger_fact_vendoring import (
    load_vendored_resource,
    verify_vendored_resource_feed_identity,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "CGT_ASSET_TYPE_COLUMN",
    "CGT_ASSET_TYPE_DOMAIN",
    "CGT_ASSET_TYPE_LOG_SIGMA",
    "CGT_ASSET_TYPE_NONE",
    "CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES",
    "CGT_ASSET_TYPE_RESIDENTIAL",
    "CGT_ASSET_TYPE_SEED",
    "CGT_ASSET_TYPE_SUB_AEA",
    "CGT_RESIDENTIAL_FLAG_SEED",
    "CGT_RESIDENTIAL_GAINS_COLUMN",
    "HMRC_CGT_ASSET_TYPE_RECORD_SETS",
    "HMRC_CGT_ASSET_TYPE_RESOURCE",
    "HMRC_CGT_TABLE7_SOURCE_SHA256",
    "HMRC_CGT_TABLE8_SOURCE_SHA256",
    "HMRCCGTAssetTypeFacts",
    "UKCGTAssetTypeStageTransform",
    "UKCGTAssetTypeSummary",
    "UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON",
    "UK_CGT_ASSET_TYPE_STAGE_NAME",
    "assign_uk_cgt_asset_types",
    "load_hmrc_cgt_asset_type_facts",
    "uk_cgt_asset_type_stage_transform",
]

UK_CGT_ASSET_TYPE_STAGE_NAME = "hmrc_cgt_asset_type_spine"
UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON = (
    "CGT asset-type assignment on the source spine: household weights pass "
    "through unchanged and total household mass is conserved."
)

#: The per-concern vendored resource (Tables 7, 8a and 8b). Regenerated
#: from the pinned Chronicle feed by ``tools/vendor_uk_ledger_facts.py``.
HMRC_CGT_ASSET_TYPE_RESOURCE = "hmrc_cgt_asset_type_facts.json"

#: Chronicle record sets the stage reads, as the manifest declares them.
_TABLE8A_RECORD_SET_ID = "hmrc.cgt_table8_2026.table8a.ty2024"
_TABLE8B_RECORD_SET_ID = "hmrc.cgt_table8_2026.table8b.ty2024"
_TABLE7_CATEGORY_RECORD_SET_ID = "hmrc.cgt_table7_2026.asset_category.ty2023"
_TABLE7_TOTAL_RECORD_SET_ID = "hmrc.cgt_table7_2026.asset_category_total.ty2023"
_TABLE7_FINANCIAL_RECORD_SET_ID = "hmrc.cgt_table7_2026.financial_asset_type.ty2023"
_TABLE7_NON_FINANCIAL_RECORD_SET_ID = (
    "hmrc.cgt_table7_2026.non_financial_asset_type.ty2023"
)
HMRC_CGT_ASSET_TYPE_RECORD_SETS: tuple[str, ...] = (
    _TABLE8A_RECORD_SET_ID,
    _TABLE8B_RECORD_SET_ID,
    _TABLE7_CATEGORY_RECORD_SET_ID,
    _TABLE7_TOTAL_RECORD_SET_ID,
    _TABLE7_FINANCIAL_RECORD_SET_ID,
    _TABLE7_NON_FINANCIAL_RECORD_SET_ID,
)

#: The publisher workbooks every vendored row must trace back to.
HMRC_CGT_TABLE8_SOURCE_SHA256 = (
    "fefe621cab478ca14b06aaeabe0ac2db95a4912b1b40ac029fbbd5ea3fa34952"
)
HMRC_CGT_TABLE7_SOURCE_SHA256 = (
    "a27ad79c3c67178e8b808c0561201dd6585b472d7919c0043fdc44142460b2bd"
)

CGT_ASSET_TYPE_COLUMN = "capital_gains_asset_type"
CGT_RESIDENTIAL_GAINS_COLUMN = "capital_gains_residential_property"

#: Value ids shared with Chronicle's Table 7 ``cgt_asset_type`` dimension,
#: plus the two states Table 7 does not describe.
CGT_ASSET_TYPE_NONE = "none"
CGT_ASSET_TYPE_SUB_AEA = "sub_aea"
CGT_ASSET_TYPE_RESIDENTIAL = "residential_land_buildings"
CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES: tuple[str, ...] = (
    "listed_shares",
    "unlisted_shares",
    "other_financial_assets",
    "agricultural_commercial_industrial_land_buildings",
    "other_non_financial_assets",
)
CGT_ASSET_TYPE_DOMAIN: tuple[str, ...] = (
    CGT_ASSET_TYPE_NONE,
    CGT_ASSET_TYPE_SUB_AEA,
    CGT_ASSET_TYPE_RESIDENTIAL,
    *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
)

#: Log-scale width of the size kernel around each type's mean gain per
#: disposal. Wide enough that every type reaches every band; the fitted
#: type weights, not the width, carry the composition.
CGT_ASSET_TYPE_LOG_SIGMA = 1.5

#: Seeds are combined with the build period, as the amounts stage does;
#: distinct from the amounts stage's 552 so the two stages never share a
#: stream.
CGT_RESIDENTIAL_FLAG_SEED = 553
CGT_ASSET_TYPE_SEED = 554

_BISECTION_ITERATIONS = 200
_SHARE_FIT_ITERATIONS = 200
_SHARE_FIT_TOLERANCE = 1e-6
_SOLVE_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# Vendored facts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HMRCCGTTable7Type:
    """One Table 7 asset-type row: disposals, proceeds and gains (2023-24)."""

    asset_type: str
    category: str
    disposals: float
    proceeds: float
    gains: float

    @property
    def mean_gain_per_disposal(self) -> float:
        return self.gains / self.disposals


@dataclass(frozen=True)
class HMRCCGTAssetTypeFacts:
    """The vendored Table 7, 8a and 8b rows, typed."""

    table8a_taxpayers_total: float
    table8a_gains_total: float
    table8a_disposals_total: float
    table8a_tax_total: float
    table8b_individuals_taxpayers: float
    table8b_individuals_gains: float
    table8b_all_taxpayers: float
    table8b_all_gains: float
    table7_types: tuple[HMRCCGTTable7Type, ...]
    table7_total_gains: float
    table7_total_disposals: float
    resource: str
    resource_sha256: str
    source_commit: str

    def individuals_share(self, measure: str) -> float:
        """Table 8b individuals ÷ all for ``taxpayers`` or ``gains``."""

        numerator = getattr(self, f"table8b_individuals_{measure}")
        denominator = getattr(self, f"table8b_all_{measure}")
        if denominator <= 0:
            raise ValueError(f"Table 8b all-taxpayer {measure} must be positive.")
        return numerator / denominator

    @property
    def residential_taxpayers_individuals_basis(self) -> float:
        return self.table8a_taxpayers_total * self.individuals_share("taxpayers")

    @property
    def residential_gains_individuals_basis(self) -> float:
        return self.table8a_gains_total * self.individuals_share("gains")

    def table7_type(self, asset_type: str) -> HMRCCGTTable7Type:
        for row in self.table7_types:
            if row.asset_type == asset_type:
                return row
        raise KeyError(f"No Table 7 row for asset type {asset_type!r}.")

    def non_residential_gains_shares(self) -> dict[str, float]:
        """Table 7 gains shares over the five non-residential types."""

        gains = {
            asset_type: self.table7_type(asset_type).gains
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        }
        total = sum(gains.values())
        return {asset_type: value / total for asset_type, value in gains.items()}


def _record_set_id(row: Mapping[str, Any]) -> str:
    return str((row.get("layout") or {}).get("record_set_id") or "")


def _value(row: Mapping[str, Any]) -> float:
    value = row.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(
            f"Vendored CGT asset-type row {row.get('aggregate_fact_key')!r} has no "
            "numeric value."
        )
    return float(value)


def _source_sha256(row: Mapping[str, Any]) -> str:
    return str((row.get("source") or {}).get("source_sha256") or "")


def load_hmrc_cgt_asset_type_facts(
    resource: str = HMRC_CGT_ASSET_TYPE_RESOURCE,
    *,
    verify_feed_identity: bool = True,
) -> HMRCCGTAssetTypeFacts:
    """Type the vendored rows; refuse a stale feed, source or roster."""

    payload = load_vendored_resource(resource)
    if verify_feed_identity:
        verify_vendored_resource_feed_identity(payload)
    digest = hashlib.sha256(
        files("microcosm.build.uk").joinpath(resource).read_bytes()
    ).hexdigest()
    rows = list(payload["rows"])

    table8a: dict[str, float] = {}
    table8b: dict[tuple[str, str], float] = {}
    table7: dict[tuple[str, str], dict[str, float]] = {}
    table7_total: dict[str, float] = {}
    for row in rows:
        record_set = _record_set_id(row)
        measure = str(row.get("measure_id") or "")
        dimensions = row.get("dimensions") or {}
        if record_set == _TABLE8A_RECORD_SET_ID:
            if _source_sha256(row) != HMRC_CGT_TABLE8_SOURCE_SHA256:
                raise ValueError(
                    "Table 8a rows trace to a workbook other than the pinned one."
                )
            table8a[measure] = _value(row)
        elif record_set == _TABLE8B_RECORD_SET_ID:
            if _source_sha256(row) != HMRC_CGT_TABLE8_SOURCE_SHA256:
                raise ValueError(
                    "Table 8b rows trace to a workbook other than the pinned one."
                )
            if dimensions.get("channel") != "uk_property_service":
                raise ValueError("Table 8b rows must describe the UK Property service.")
            taxpayer_type = str(dimensions.get("taxpayer_type") or "all")
            table8b[(taxpayer_type, measure)] = _value(row)
        elif record_set in (
            _TABLE7_FINANCIAL_RECORD_SET_ID,
            _TABLE7_NON_FINANCIAL_RECORD_SET_ID,
        ):
            if _source_sha256(row) != HMRC_CGT_TABLE7_SOURCE_SHA256:
                raise ValueError(
                    "Table 7 rows trace to a workbook other than the pinned one."
                )
            asset_type = dimensions.get("cgt_asset_type")
            if asset_type is None:
                continue  # the category subtotal restates the Table 7_1 rows
            category = str(dimensions.get("cgt_asset_category") or "")
            table7.setdefault((str(asset_type), category), {})[measure] = _value(row)
        elif record_set == _TABLE7_TOTAL_RECORD_SET_ID:
            if _source_sha256(row) != HMRC_CGT_TABLE7_SOURCE_SHA256:
                raise ValueError(
                    "Table 7 rows trace to a workbook other than the pinned one."
                )
            table7_total[measure] = _value(row)
        elif record_set == _TABLE7_CATEGORY_RECORD_SET_ID:
            continue
        else:
            raise ValueError(
                f"Vendored CGT asset-type resource carries an undeclared record set "
                f"{record_set!r}."
            )

    try:
        facts = HMRCCGTAssetTypeFacts(
            table8a_taxpayers_total=table8a["taxpayers_total"],
            table8a_gains_total=table8a["gains_total"],
            table8a_disposals_total=table8a["disposals_total"],
            table8a_tax_total=table8a["tax_total"],
            table8b_individuals_taxpayers=table8b[("individuals", "taxpayers")],
            table8b_individuals_gains=table8b[("individuals", "gains")],
            table8b_all_taxpayers=table8b[("all", "taxpayers")],
            table8b_all_gains=table8b[("all", "gains")],
            table7_types=tuple(
                HMRCCGTTable7Type(
                    asset_type=asset_type,
                    category=category,
                    disposals=cells["disposals"],
                    proceeds=cells["disposal_proceeds"],
                    gains=cells["gains"],
                )
                for (asset_type, category), cells in sorted(table7.items())
            ),
            table7_total_gains=table7_total["gains"],
            table7_total_disposals=table7_total["disposals"],
            resource=resource,
            resource_sha256=digest,
            source_commit=str(payload["source_fact_feed"]["source_commit"]),
        )
    except KeyError as missing:
        raise ValueError(f"Vendored CGT asset-type resource lacks {missing}.") from None
    expected_types = {CGT_ASSET_TYPE_RESIDENTIAL, *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES}
    actual_types = {row.asset_type for row in facts.table7_types}
    if actual_types != expected_types:
        raise ValueError(
            "Table 7 asset-type roster drifted: expected "
            f"{sorted(expected_types)}, got {sorted(actual_types)}."
        )
    for row in facts.table7_types:
        if row.disposals <= 0 or row.gains <= 0:
            raise ValueError(f"Table 7 {row.asset_type} must have positive rows.")
    if facts.table8b_individuals_taxpayers > facts.table8b_all_taxpayers:
        raise ValueError("Table 8b individuals cannot exceed all taxpayers.")
    if facts.table8b_individuals_gains > facts.table8b_all_gains:
        raise ValueError("Table 8b individual gains cannot exceed all gains.")
    return facts


# ---------------------------------------------------------------------------
# Residential flag
# ---------------------------------------------------------------------------


def _logistic(log_gains: np.ndarray, a: float, b: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(a + b * log_gains)))


def _centred_log_gains(
    gains: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    """Log gains less their weighted mean, so the intercept stays O(1)."""

    log_gains = np.log(gains)
    centre = float((weights * log_gains).sum() / weights.sum())
    return log_gains - centre, centre


def _solve_a(
    log_gains: np.ndarray, weights: np.ndarray, *, b: float, count_target: float
) -> float:
    """The intercept that puts the expected weighted count on target at slope b."""

    low, high = -400.0, 400.0
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        expected = float((weights * _logistic(log_gains, mid, b)).sum())
        if expected < count_target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def solve_residential_logistic(
    gains: np.ndarray,
    weights: np.ndarray,
    *,
    count_target: float,
    gains_target: float,
) -> tuple[float, float, float]:
    """Solve (a, b, c) so expected weighted count and gains hit both targets.

    ``c`` is the weighted mean log gain the logistic is centred on. For each
    trial slope the intercept is bisected onto the count; the slope is then
    bisected onto the gains, which rise monotonically in the slope at a
    fixed count. Refuses targets the frame cannot reach: more taxpayers or
    gains than the liable population holds, or a mean gain outside what a
    logistic tilt of these persons can produce.
    """

    if gains.size == 0:
        raise ValueError("No liable gainers to flag for residential property.")
    total_weight = float(weights.sum())
    total_gains = float((weights * gains).sum())
    if count_target >= total_weight:
        raise ValueError(
            f"Residential count target {count_target} is not below the liable "
            f"taxpayer mass {total_weight}."
        )
    if gains_target >= total_gains:
        raise ValueError(
            f"Residential gains target {gains_target} is not below the liable "
            f"gains mass {total_gains}."
        )
    log_gains, centre = _centred_log_gains(gains, weights)
    low, high = -12.0, 12.0

    def expected_gains(b: float) -> float:
        a = _solve_a(log_gains, weights, b=b, count_target=count_target)
        return float((weights * gains * _logistic(log_gains, a, b)).sum())

    if not expected_gains(low) <= gains_target <= expected_gains(high):
        raise ValueError(
            "Residential gains target lies outside the range a logistic tilt of "
            f"the liable gainers can reach at count {count_target}: "
            f"[{expected_gains(low)}, {expected_gains(high)}] versus {gains_target}."
        )
    for _ in range(_BISECTION_ITERATIONS):
        mid = 0.5 * (low + high)
        if expected_gains(mid) < gains_target:
            low = mid
        else:
            high = mid
        if high - low < _SOLVE_TOLERANCE:
            break
    b = 0.5 * (low + high)
    a = _solve_a(log_gains, weights, b=b, count_target=count_target)
    return a, b, centre


def _weighted_systematic_flags(
    probabilities: np.ndarray,
    weights: np.ndarray,
    order: np.ndarray,
    offset: float,
) -> np.ndarray:
    """Weighted systematic sampling along ``order``.

    Walking the persons in order, the expected weight ``w * p`` accrues to a
    running balance; a person is flagged when the balance reaches
    ``(1 - offset)`` of their own weight, and their whole weight is then
    drawn down. The balance therefore stays within one person's weight of
    zero at every step, so the flagged weight tracks the expected weight
    along the ordering rather than only in total.
    """

    flags = np.zeros(probabilities.shape, dtype=bool)
    owed = 0.0
    for index in order:
        weight = float(weights[index])
        owed += weight * float(probabilities[index])
        if owed >= (1.0 - offset) * weight:
            flags[index] = True
            owed -= weight
    return flags


# ---------------------------------------------------------------------------
# Main asset type
# ---------------------------------------------------------------------------


def _type_kernels(gains: np.ndarray, medians: Mapping[str, float]) -> np.ndarray:
    """Log-normal size kernels, one column per non-residential type."""

    log_gains = np.log(gains)[:, None]
    centres = np.log(
        np.asarray(
            [medians[asset_type] for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES]
        )
    )[None, :]
    return np.exp(-0.5 * ((log_gains - centres) / CGT_ASSET_TYPE_LOG_SIGMA) ** 2)


def fit_type_weights(
    gains: np.ndarray,
    weights: np.ndarray,
    *,
    medians: Mapping[str, float],
    share_targets: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit the type weights so expected gains shares match the targets.

    Returns the type weights, the per-person type probabilities and the
    number of iterations used. Multiplicative updates on the weights are a
    one-dimensional rake per type; with a wide kernel every type reaches
    every person, so the fixed point exists and the loop converges quickly.
    """

    kernels = _type_kernels(gains, medians)
    targets = np.asarray(
        [
            share_targets[asset_type]
            for asset_type in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        ]
    )
    type_weights = np.ones(len(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES))
    mass = weights * gains
    iterations = 0
    probabilities = kernels
    for step in range(1, _SHARE_FIT_ITERATIONS + 1):
        iterations = step
        scaled = kernels * type_weights[None, :]
        probabilities = scaled / scaled.sum(axis=1, keepdims=True)
        achieved = (mass[:, None] * probabilities).sum(axis=0) / mass.sum()
        if np.max(np.abs(achieved - targets)) < _SHARE_FIT_TOLERANCE:
            break
        type_weights = type_weights * (targets / np.maximum(achieved, 1e-300))
        type_weights = type_weights / type_weights.sum()
    return type_weights, probabilities, iterations


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UKCGTAssetTypeSummary:
    """What the stage was asked for and what it realised."""

    residential: Mapping[str, object]
    asset_type: Mapping[str, object]
    composition_by_band: tuple[dict[str, object], ...]
    value_counts: Mapping[str, int]
    facts: Mapping[str, object]
    seeds: Mapping[str, int]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": UK_CGT_ASSET_TYPE_STAGE_NAME,
            "residential": dict(self.residential),
            "asset_type": dict(self.asset_type),
            "composition_by_band": [dict(row) for row in self.composition_by_band],
            "value_counts": dict(self.value_counts),
            "facts": dict(self.facts),
            "seeds": dict(self.seeds),
        }


def assign_uk_cgt_asset_types(
    frame: Frame,
    facts: HMRCCGTAssetTypeFacts,
    parameters: UKCGTPolicyParameters,
    *,
    residential_seed: int = CGT_RESIDENTIAL_FLAG_SEED,
    asset_type_seed: int = CGT_ASSET_TYPE_SEED,
    mass_change_reason: str = UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON,
) -> tuple[Frame, UKCGTAssetTypeSummary]:
    """Write the asset-type column and the residential gains column."""

    validate_uk_national_frame(frame)
    time_period = uk_time_period(frame)
    person = frame.table("person").reset_index(drop=True)
    if "capital_gains" not in person.columns:
        raise ValueError("Person table has no capital_gains column to classify.")
    for column in (CGT_ASSET_TYPE_COLUMN, CGT_RESIDENTIAL_GAINS_COLUMN):
        if column in person.columns:
            raise ValueError(f"Person table already carries {column}.")
    household = frame.table("household")
    weights_by_household = pd.Series(
        frame.weights_for("household").values, index=household["household_id"]
    )
    person_weight = (
        person["person_household_id"].map(weights_by_household).to_numpy(dtype=float)
    )
    if not np.isfinite(person_weight).all():
        raise ValueError("Every person must map to a weighted household.")
    gains = pd.to_numeric(person["capital_gains"], errors="raise").to_numpy(dtype=float)
    person_id = person["person_id"].to_numpy()

    liable = gains > parameters.annual_exempt_amount
    positive = gains > 0
    asset_type = np.full(len(person), CGT_ASSET_TYPE_NONE, dtype=object)
    asset_type[positive & ~liable] = CGT_ASSET_TYPE_SUB_AEA

    # 1. Residential flag on the liable population.
    liable_index = np.flatnonzero(liable)
    liable_gains = gains[liable_index]
    liable_weights = person_weight[liable_index]
    count_target = facts.residential_taxpayers_individuals_basis
    gains_target = facts.residential_gains_individuals_basis
    a, b, centre = solve_residential_logistic(
        liable_gains,
        liable_weights,
        count_target=count_target,
        gains_target=gains_target,
    )
    probabilities = _logistic(np.log(liable_gains) - centre, a, b)
    rng_flag = np.random.default_rng((residential_seed, int(time_period)))
    order = np.lexsort((person_id[liable_index], liable_gains))
    flags = _weighted_systematic_flags(
        probabilities, liable_weights, order, float(rng_flag.random())
    )
    residential_rows = liable_index[flags]
    asset_type[residential_rows] = CGT_ASSET_TYPE_RESIDENTIAL
    residential_gains = np.zeros(len(person))
    residential_gains[residential_rows] = gains[residential_rows]

    # 2. Main asset type for the non-residential liable remainder.
    remainder_index = liable_index[~flags]
    medians = {
        asset_type_name: facts.table7_type(asset_type_name).mean_gain_per_disposal
        for asset_type_name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
    }
    share_targets = facts.non_residential_gains_shares()
    type_weights = np.full(len(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES), np.nan)
    fit_iterations = 0
    if remainder_index.size:
        type_weights, type_probabilities, fit_iterations = fit_type_weights(
            gains[remainder_index],
            person_weight[remainder_index],
            medians=medians,
            share_targets=share_targets,
        )
        rng_type = np.random.default_rng((asset_type_seed, int(time_period)))
        # Draws are consumed in person order so the stream is reproducible.
        remainder_order = np.argsort(person_id[remainder_index], kind="stable")
        uniforms = np.empty(remainder_index.size)
        uniforms[remainder_order] = rng_type.random(remainder_index.size)
        cumulative = np.cumsum(type_probabilities, axis=1)
        choice = (uniforms[:, None] > cumulative).sum(axis=1)
        choice = np.minimum(choice, len(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES) - 1)
        asset_type[remainder_index] = np.asarray(
            CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES, dtype=object
        )[choice]

    if (asset_type[liable] == CGT_ASSET_TYPE_NONE).any() or (
        asset_type[liable] == CGT_ASSET_TYPE_SUB_AEA
    ).any():
        raise ValueError("Every liable gainer must carry an asset type.")
    if set(asset_type) - set(CGT_ASSET_TYPE_DOMAIN):
        raise ValueError("Asset-type draw produced a value outside the domain.")

    # Reporting.
    expected_count = float((liable_weights * probabilities).sum())
    expected_gains = float((liable_weights * liable_gains * probabilities).sum())
    bernoulli = probabilities * (1.0 - probabilities)
    count_sigma = float(np.sqrt((liable_weights**2 * bernoulli).sum()))
    gains_sigma = float(
        np.sqrt(((liable_weights * liable_gains) ** 2 * bernoulli).sum())
    )
    achieved_count = float(person_weight[residential_rows].sum())
    achieved_gains = float(
        residential_gains[residential_rows].dot(person_weight[residential_rows])
    )
    residential = {
        "count_target_individuals_basis": count_target,
        "gains_target_individuals_basis": gains_target,
        "expected_count": expected_count,
        "expected_gains": expected_gains,
        "achieved_count": achieved_count,
        "achieved_gains": achieved_gains,
        "achieved_rows": int(residential_rows.size),
        "max_liable_weight": float(liable_weights.max()),
        "count_bernoulli_sigma": count_sigma,
        "gains_bernoulli_sigma": gains_sigma,
        "logistic_intercept": float(a),
        "logistic_slope": float(b),
        "log_gain_centre": float(centre),
        "liable_taxpayer_mass": float(liable_weights.sum()),
        "liable_gains_mass": float((liable_weights * liable_gains).sum()),
        "table8a_taxpayers_total": facts.table8a_taxpayers_total,
        "table8a_gains_total": facts.table8a_gains_total,
        "table8b_individuals_taxpayer_share": facts.individuals_share("taxpayers"),
        "table8b_individuals_gains_share": facts.individuals_share("gains"),
    }
    remainder_mass = float(
        (person_weight[remainder_index] * gains[remainder_index]).sum()
    )
    achieved_type_gains = {
        asset_type_name: float(
            (person_weight * gains)[asset_type == asset_type_name].sum()
        )
        for asset_type_name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
    }
    asset_type_report = {
        "log_sigma": CGT_ASSET_TYPE_LOG_SIGMA,
        "median_anchor_gbp": medians,
        "type_weights": {
            asset_type_name: float(type_weights[index])
            for index, asset_type_name in enumerate(
                CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
            )
        },
        "share_fit_iterations": int(fit_iterations),
        "target_gains_share": dict(share_targets),
        "achieved_gains_share": {
            asset_type_name: (value / remainder_mass if remainder_mass > 0 else 0.0)
            for asset_type_name, value in achieved_type_gains.items()
        },
        "achieved_people": {
            asset_type_name: float(person_weight[asset_type == asset_type_name].sum())
            for asset_type_name in CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES
        },
        "achieved_gains": achieved_type_gains,
        "non_residential_gains_mass": remainder_mass,
    }
    bounds = HMRC_CGT_GAIN_BAND_LOWER_BOUNDS
    uppers = (*bounds[1:], np.inf)
    composition = []
    for lower, upper in zip(bounds, uppers, strict=True):
        in_band = liable & (gains >= lower) & (gains < upper)
        row: dict[str, object] = {"gain_lower_bound": lower}
        for asset_type_name in (
            CGT_ASSET_TYPE_RESIDENTIAL,
            *CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES,
        ):
            mask = in_band & (asset_type == asset_type_name)
            row[f"{asset_type_name}_people"] = float(person_weight[mask].sum())
            row[f"{asset_type_name}_gains"] = float(
                (person_weight[mask] * gains[mask]).sum()
            )
        composition.append(row)
    value_counts = {
        value: int((asset_type == value).sum()) for value in CGT_ASSET_TYPE_DOMAIN
    }
    summary = UKCGTAssetTypeSummary(
        residential=residential,
        asset_type=asset_type_report,
        composition_by_band=tuple(composition),
        value_counts=value_counts,
        facts={
            "resource": facts.resource,
            "resource_sha256": facts.resource_sha256,
            "source_commit": facts.source_commit,
            "annual_exempt_amount": parameters.annual_exempt_amount,
        },
        seeds={"residential_flag": residential_seed, "asset_type": asset_type_seed},
    )

    new_person = person.copy()
    new_person[CGT_ASSET_TYPE_COLUMN] = asset_type
    new_person[CGT_RESIDENTIAL_GAINS_COLUMN] = residential_gains
    weights = frame.weights_for("household")
    household_mass = float(weights.total)
    receipt = MassChangeRecord(
        entity="household",
        old_total=household_mass,
        new_total=household_mass,
        declared_factor=1.0,
        reason=mass_change_reason,
    )
    result = uk_national_frame(
        person=new_person,
        benunit=frame.table("benunit"),
        household=household,
        time_period=time_period,
        weight_kind=uk_household_weight_kind(frame),
        household_weights=weights.values,
        mass_log=(*frame.mass_log, receipt),
    )
    validate_uk_national_frame(result)
    return result, summary


def uk_cgt_asset_type_stage_transform(
    stage: SourceStageSpec,
    *,
    facts: HMRCCGTAssetTypeFacts | None = None,
    parameters: UKCGTPolicyParameters | None = None,
):
    """Bind the spine manifest, then run the reviewed assignment."""

    _assert_cgt_asset_type_stage_parameters(stage)
    return UKCGTAssetTypeStageTransform(stage=stage, facts=facts, parameters=parameters)


@dataclass(frozen=True)
class UKCGTAssetTypeStageTransform:
    """Source-plan asset-type assignment with a stage-time summary receipt."""

    stage: SourceStageSpec
    facts: HMRCCGTAssetTypeFacts | None = None
    parameters: UKCGTPolicyParameters | None = None
    last_result: UKCGTAssetTypeSummary | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_cgt_asset_type_stage_parameters(self.stage)
        facts = self.facts
        if facts is None:
            facts = load_hmrc_cgt_asset_type_facts()
        parameters = self.parameters
        if parameters is None:
            parameters = uk_cgt_policy_parameters(uk_time_period(frame))
        result, summary = assign_uk_cgt_asset_types(frame, facts, parameters)
        object.__setattr__(self, "last_result", summary)
        return result

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (CGT_ASSET_TYPE_COLUMN, CGT_RESIDENTIAL_GAINS_COLUMN)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def cgt_asset_type_operation_parameters() -> dict[str, dict[str, Any]]:
    """The reviewed operation payloads the manifests must carry verbatim."""

    return {
        "verify_vendored_fact_resource": {
            "artifact_role": "cgt_asset_type_facts",
            "resource": HMRC_CGT_ASSET_TYPE_RESOURCE,
            "feed_pin": "chronicle_feed.json",
            "record_sets": list(HMRC_CGT_ASSET_TYPE_RECORD_SETS),
            "source_vintage": "2024-25",
            "mapped_build_period": 2024,
            "period_mapping": "published_tax_year_equals_build_period",
            "require_before_source_read": True,
            "runtime_sha256_required": True,
            "fail_on_mismatch": True,
        },
        "assign_residential_property_flag": {
            "population": (
                "persons with net capital gains above the annual exempt amount "
                "(the national taxpayer proxy)"
            ),
            "model": (
                "logistic probability in centred log gains, p = 1 / (1 + exp(-(a + "
                "b (log g - c)))) with c the weighted mean log gain of the "
                "population"
            ),
            "parameter_solver": (
                "nested bisection; the intercept matches the expected weighted "
                "count at each trial slope, the slope matches the expected weighted "
                "gains"
            ),
            "count_target": (
                "Table 8a 2024-25 total taxpayers reporting residential property "
                "disposals times the Table 8b 2024-25 individuals/all taxpayer share "
                "on the UK Property service"
            ),
            "gains_target": (
                "Table 8a 2024-25 total residential property gains times the Table "
                "8b 2024-25 individuals/all gains share on the UK Property service"
            ),
            "basis_assumption": (
                "trusts hold the same share of the Self Assessment component as of "
                "the UK Property service, and a flagged person's whole net gain is "
                "attributed to residential property"
            ),
            "realization": (
                "weighted systematic sampling in ascending gain order with one "
                "seeded offset: a person is flagged when the expected weight owed "
                "so far reaches their own weight, so the flagged weight tracks the "
                "expected weight within one person's weight along the gains axis "
                "and the realised count and gains sit close to their expectations; "
                "the Bernoulli sigma of each is reported as the envelope"
            ),
            "seed": CGT_RESIDENTIAL_FLAG_SEED,
            "seed_mixing": "seed combined with the build period",
            "output_column": CGT_RESIDENTIAL_GAINS_COLUMN,
            "output_semantics": (
                "the person's capital_gains where flagged residential, 0 otherwise"
            ),
        },
        "assign_main_asset_type": {
            "population": "liable gainers not flagged residential",
            "categories": list(CGT_ASSET_TYPE_NON_RESIDENTIAL_TYPES),
            "kernel": (
                "log-normal density in log gains centred on each type's Table 7 "
                "2023-24 mean gain per disposal, one common log-scale width"
            ),
            "log_sigma": CGT_ASSET_TYPE_LOG_SIGMA,
            "share_targets": (
                "Table 7 2023-24 gains by asset type excluding residential land "
                "and buildings, as shares"
            ),
            "weight_fitting": (
                "multiplicative updates of the type weights until the "
                "household-weighted expected gains shares by type match the share "
                "targets"
            ),
            "weight_fitting_iterations": _SHARE_FIT_ITERATIONS,
            "weight_fitting_tolerance": _SHARE_FIT_TOLERANCE,
            "realization": (
                "one seeded uniform per person against the cumulative type "
                "probabilities, consumed in person_id order"
            ),
            "seed": CGT_ASSET_TYPE_SEED,
            "seed_mixing": "seed combined with the build period",
            "output_column": CGT_ASSET_TYPE_COLUMN,
            "value_domain": list(CGT_ASSET_TYPE_DOMAIN),
            "none_semantics": "no positive net gains",
            "sub_aea_semantics": (
                "positive net gains at or below the annual exempt amount; never a "
                "taxpayer, so never classified"
            ),
            "diagnostic_only": True,
        },
        "record_mass_conservation_receipt": {
            "entity": "household",
            "reason": UK_CGT_ASSET_TYPE_MASS_CONSERVATION_REASON,
            "declared_factor": 1.0,
            "gate_coupling": (
                "The terminal family gate requires a valid mass-conserving "
                "MassChangeRecord carrying exactly this stage-specific reason."
            ),
        },
        "classify_cgt_asset_type_facts_with_reviewed_fence": {
            "calibration_permitted": False,
            "fact_fence_id": "cgt_asset_type_facts_sample_based_prior_vintage",
            "fenced_fact_count": 33,
            "fenced_fact_composition": (
                "Table 7 2023-24 rows: 6 asset-category rows, 3 category totals, "
                "12 financial asset-type rows, 12 non-financial asset-type rows"
            ),
            "classification_rationale": (
                "Table 7 is sample-based, published for 2023-24 only, counts "
                "disposals rather than taxpayers and includes trusts and non-UK "
                "assets; it seeds the asset-type draw and is reported against, "
                "never fitted."
            ),
            "calibrated_facts": (
                "The Table 8a 2024-25 total residential property taxpayers and "
                "gains restated on the individuals basis "
                "(hmrc.cgt.residential_property_taxpayers, "
                "hmrc.cgt.residential_property_gains in uk_population_targets.json)."
            ),
            "adjudication": "https://github.com/PolicyEngine/microcosm/issues/725",
        },
    }


def _assert_cgt_asset_type_stage_parameters(stage: SourceStageSpec) -> None:
    """Arm 1 of the #730/#684 two-arm rule: the manifest restates the code."""

    expected = cgt_asset_type_operation_parameters()
    kinds = tuple(operation.kind for operation in stage.operations)
    if kinds != tuple(expected):
        raise ValueError(
            "CGT asset-type stage operation order drifted: expected "
            f"{tuple(expected)}, got {kinds}."
        )
    for operation in stage.operations:
        actual = dict(operation.parameters)
        declared = expected[operation.kind]
        if actual != declared:
            drifted = sorted(
                key
                for key in {*actual, *declared}
                if actual.get(key) != declared.get(key)
            )
            raise ValueError(
                f"CGT asset-type stage {operation.kind} declaration drifted from "
                f"the reviewed mapping on parameter(s) {drifted}."
            )
    if tuple(stage.outputs) != UKCGTAssetTypeStageTransform.output_columns():
        raise ValueError(
            "CGT asset-type stage outputs drifted: expected "
            f"{UKCGTAssetTypeStageTransform.output_columns()}, got {stage.outputs}."
        )
