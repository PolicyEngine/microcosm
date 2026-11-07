"""The self-employed's trading allowance claim status (microcosm#1428, #1439).

policyengine-uk applies the trading allowance (ITTOIA 2005 Part 7A) against a
trader's gross receipts. Neither the FRS nor the SPI tape records turnover, so
from policyengine-uk 2.172.0 (policyengine-uk#2765) the build stores which
receipts lie behind each positive profit instead, the engine's input
``trading_allowance_claim_status``, and leaves ``self_employment_gross_receipts``
at the engine's zero. The statuses, as the engine reads them
(``utils/trading_allowance.py``):

* CLAIMANT: the profit is receipts less the 2024-25 allowance of GBP 1,000
  (partial relief, ITTOIA s. 783AI). A raise to GBP 3,000 relieves
  ``min(profit, GBP 2,000)`` more, with no drift as profits are uprated.
* NON_CLAIMANT: the profit is receipts less actual expenses above the
  allowance. With no supplied receipts the engine relieves them at no
  allowance; the costing books the traders who would switch as behaviour.
* UNKNOWN: anyone with no positive profit, whom the engine reads as UNKNOWN
  whatever is stored, and the full-relief group: FRS respondents whose receipts
  lie within the allowance, relieved in full without an election (s. 783AE).
  The engine's own documentation asks that such a person stay UNKNOWN.

The stage, the last of the spine, flags each trader with a
positive profit as a claimant with a probability by profit band, drawn on
their own ``person_id`` (seed 0, salt ``trading_allowance_claimant``), as
#1428 did. Of the FRS respondents (not the SPI channel's rows) with a profit of
GBP 1,000 or less who are not drawn claimants, it then places a share in the
full-relief group, drawn on ``person_id`` (salt
``trading_allowance_full_relief``). The share is María's ruling. The read of
6 November 2026 (M10 of the November event-data plan) found the frame's small
traders below SPI's declarers even before any full relief, so the share is
zero and the group is empty. SPI rows and their copies stay CLAIMANT or
NON_CLAIMANT: SPI counts the people HMRC holds a self-employment source for,
and receipts within the allowance need not be declared. Every other trader
with a positive profit is treated as filing, as #1428 did.

``trading_allowance_full_relief`` marks the group, so that HMRC's SPI rows of
self-employment, which count declarers (Table 3.10 by self-employment band,
Table 3.6 by total income), leave it out. The release keeps the column, so a
bound row re-measured on a built file reads the same population.

The claimant rates are #1428's, calibrated to HMRC's policy paper on the Budget
2026 raise (about 400,000 taxpayers with a lower bill, by about GBP 380 on
average) on the receipts encoding. ``tools/measure_uk_trading_allowance_gainers.py``
re-fits them to the statuses on a release before PR2's arm (María's answer of
6 November 2026, microcosm#1439), until HMRC publishes claimant counts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    rounded_count,
    small_cell_share,
    source_household_ids,
    support_band,
)
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "TRADING_ALLOWANCE_CLAIMANT_BANDS",
    "TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT",
    "TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME",
    "TRADING_ALLOWANCE_STATUS_OUTPUT",
    "UKTradingAllowanceReceiptsStageTransform",
    "assign_trading_allowance_claim_status",
    "claimant_rates",
    "store_trading_allowance_claim_status",
]

TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME = "trading_allowance_receipts"
TRADING_ALLOWANCE_STATUS_OUTPUT = "trading_allowance_claim_status"
TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT = "trading_allowance_full_relief"
TRADING_ALLOWANCE_PROFIT = "self_employment_income"
TRADING_ALLOWANCE_SPI_ROW = "household_is_spi_synthetic"
TRADING_ALLOWANCE_CLAIMANT_DRAW = "trading_allowance_claimant"
TRADING_ALLOWANCE_CLAIMANT_SEED = 0
#: policyengine-uk's TradingAllowanceClaimStatus members, by the trader they
#: describe.
CLAIMANT = "CLAIMANT"
NON_CLAIMANT = "NON_CLAIMANT"
UNKNOWN = "UNKNOWN"
TRADING_ALLOWANCE_STATUSES = {
    "claimant": CLAIMANT,
    "full_relief": UNKNOWN,
    "no_profit": UNKNOWN,
    "other_trader": NON_CLAIMANT,
}
#: The claimant probability by the person's profit, upper bounds inclusive,
#: over profits above zero; the last band has no upper bound.
TRADING_ALLOWANCE_CLAIMANT_BANDS: tuple[tuple[float | None, float], ...] = (
    (4_500.0, 0.72),
    (9_000.0, 0.14),
    (None, 0.0),
)
TRADING_ALLOWANCE_FULL_RELIEF_DRAW = "trading_allowance_full_relief"
#: The share of the group's population placed in it: María's ruling. M10 of the
#: November event-data plan found no FRS surplus over SPI's declarers, so it
#: is zero.
TRADING_ALLOWANCE_FULL_RELIEF_RATE = 0.0
#: The 2024-25 allowance: a profit above it cannot come from receipts within it.
TRADING_ALLOWANCE_FULL_RELIEF_PROFIT_LIMIT = 1_000.0
TRADING_ALLOWANCE_FULL_RELIEF_POPULATION = (
    "FRS respondents (household_is_spi_synthetic false) with a positive "
    "self_employment_income of at most profit_upper_inclusive who are not drawn "
    "claimants; SPI rows and their copies are never in it"
)
TRADING_ALLOWANCE_CALIBRATION = (
    "HMRC's policy paper on the Budget 2026 trading allowance raise: about "
    "400,000 taxpayers with a lower bill, by about GBP 380 on average, measured "
    "on policyengine-uk for the microcosm#1428 national from 9f70c71e7 at "
    "2028-29 prices on the receipts encoding; "
    "tools/measure_uk_trading_allowance_gainers.py re-fits them to the claim "
    "statuses on a release before each arm (microcosm#1439), until HMRC "
    "publishes claimant counts (María's choices of 2026-10-31, 2026-11-01 and "
    "2026-11-06)"
)
TRADING_ALLOWANCE_RECEIPTS_MASS_CHANGE_REASON = (
    "Trading allowance claim statuses stored on the source spine: household "
    "weights pass through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)
TRADING_ALLOWANCE_RECEIPTS_RULE = (
    "a person with a positive self_employment_income claims the allowance "
    "where their uniform falls below their profit band's rate and is CLAIMANT; "
    "of the full-relief population, those whose full-relief uniform falls below "
    "full_relief.rate are UNKNOWN and carry trading_allowance_full_relief; every "
    "other trader with a positive profit is NON_CLAIMANT, and a person with no "
    "positive profit is UNKNOWN; self_employment_gross_receipts is not written"
)


def _expected_operations() -> tuple[tuple[str, Mapping[str, object]], ...]:
    bands: list[dict[str, object]] = []
    for upper, rate in TRADING_ALLOWANCE_CLAIMANT_BANDS:
        band: dict[str, object] = {"rate": rate}
        if upper is not None:
            band = {"upper_inclusive": upper, **band}
        bands.append(band)
    return (
        (
            "assign_trading_allowance_receipts",
            {
                "output": TRADING_ALLOWANCE_STATUS_OUTPUT,
                "carrier": TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
                "profit": TRADING_ALLOWANCE_PROFIT,
                "draw": TRADING_ALLOWANCE_CLAIMANT_DRAW,
                "seed": TRADING_ALLOWANCE_CLAIMANT_SEED,
                "claimant_bands": bands,
                "statuses": dict(TRADING_ALLOWANCE_STATUSES),
                "full_relief": {
                    "draw": TRADING_ALLOWANCE_FULL_RELIEF_DRAW,
                    "rate": TRADING_ALLOWANCE_FULL_RELIEF_RATE,
                    "profit_upper_inclusive": TRADING_ALLOWANCE_FULL_RELIEF_PROFIT_LIMIT,
                    "population": TRADING_ALLOWANCE_FULL_RELIEF_POPULATION,
                },
                "calibration": TRADING_ALLOWANCE_CALIBRATION,
                "rule": TRADING_ALLOWANCE_RECEIPTS_RULE,
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": TRADING_ALLOWANCE_RECEIPTS_MASS_CHANGE_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


def claimant_rates(profit: np.ndarray) -> np.ndarray:
    """Each person's claimant probability by their profit band (zero at none)."""

    profit = np.asarray(profit, dtype=np.float64)
    rates = np.zeros(profit.shape, dtype=np.float64)
    lower = 0.0
    for upper, rate in TRADING_ALLOWANCE_CLAIMANT_BANDS:
        in_band = profit > lower
        if upper is not None:
            in_band &= profit <= upper
        rates[in_band] = rate
        if upper is None:
            break
        lower = upper
    return rates


@dataclass(frozen=True)
class UKTradingAllowanceStatuses:
    """Each person's claim status, and the draws behind it."""

    status: np.ndarray
    claimant: np.ndarray
    full_relief: np.ndarray


def assign_trading_allowance_claim_status(
    person_ids: np.ndarray,
    profit: np.ndarray,
    *,
    spi_row: np.ndarray,
    full_relief_rate: float = TRADING_ALLOWANCE_FULL_RELIEF_RATE,
    seed: int = TRADING_ALLOWANCE_CLAIMANT_SEED,
) -> UKTradingAllowanceStatuses:
    """Each person's claim status, claimant flag and full-relief flag."""

    profit = np.asarray(profit, dtype=np.float64)
    ids = np.asarray(person_ids)
    spi_row = np.asarray(spi_row, dtype=bool)
    if ids.shape != profit.shape or spi_row.shape != profit.shape:
        raise ValueError(
            "Trading allowance person ids and SPI flags must align to the profits."
        )
    if not np.isfinite(profit).all():
        raise ValueError("Trading allowance profits must be finite.")
    if not 0.0 <= float(full_relief_rate) <= 1.0:
        raise ValueError(
            f"The full-relief rate must lie in [0, 1]; got {full_relief_rate}."
        )
    trader = profit > 0.0
    claimant = np.zeros(profit.shape, dtype=bool)
    if trader.any():
        uniforms = stable_identity_uniforms(
            ids[trader], seed=seed, salt=TRADING_ALLOWANCE_CLAIMANT_DRAW
        )
        claimant[trader] = uniforms < claimant_rates(profit[trader])
    population = (
        trader
        & ~claimant
        & ~spi_row
        & (profit <= TRADING_ALLOWANCE_FULL_RELIEF_PROFIT_LIMIT)
    )
    full_relief = np.zeros(profit.shape, dtype=bool)
    if population.any() and full_relief_rate > 0.0:
        uniforms = stable_identity_uniforms(
            ids[population], seed=seed, salt=TRADING_ALLOWANCE_FULL_RELIEF_DRAW
        )
        full_relief[population] = uniforms < float(full_relief_rate)
    status = np.where(
        claimant,
        CLAIMANT,
        np.where(trader & ~full_relief, NON_CLAIMANT, UNKNOWN),
    ).astype(object)
    return UKTradingAllowanceStatuses(
        status=status, claimant=claimant, full_relief=full_relief
    )


@dataclass(frozen=True)
class UKTradingAllowanceReceiptsResult:
    """The frame with the stored claim statuses, and the stage's receipt."""

    frame: Frame
    traders: int
    claimants: int
    full_relief: int
    full_relief_support: str
    bands: tuple[Mapping[str, object], ...]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME,
            "seed": TRADING_ALLOWANCE_CLAIMANT_SEED,
            "draw": TRADING_ALLOWANCE_CLAIMANT_DRAW,
            # Record counts are rounded, never exact (UKDS CD137 cl. 8).
            "traders": rounded_count(self.traders),
            "claimants": rounded_count(self.claimants),
            "full_relief": {
                "draw": TRADING_ALLOWANCE_FULL_RELIEF_DRAW,
                "rate": TRADING_ALLOWANCE_FULL_RELIEF_RATE,
                "records": rounded_count(self.full_relief),
                "support": self.full_relief_support,
            },
            "bands": [dict(band) for band in self.bands],
        }


@dataclass(frozen=True)
class UKTradingAllowanceReceiptsStageTransform:
    """Whole-stage callable storing the self-employed's claim status."""

    stage: SourceStageSpec
    last_result: UKTradingAllowanceReceiptsResult | None = field(
        default=None, init=False
    )

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = store_trading_allowance_claim_status(frame)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (TRADING_ALLOWANCE_STATUS_OUTPUT, TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def store_trading_allowance_claim_status(
    frame: Frame,
) -> UKTradingAllowanceReceiptsResult:
    """Store each person's claim status and full-relief flag; nothing else moves."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    for column in (
        TRADING_ALLOWANCE_STATUS_OUTPUT,
        TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
    ):
        if column in person.columns:
            raise ValueError(
                f"{column!r} is already stored: this stage is its only writer."
            )
    if TRADING_ALLOWANCE_PROFIT not in person.columns:
        raise KeyError(
            f"person.{TRADING_ALLOWANCE_PROFIT} is missing; the statuses are "
            "drawn on the final self-employment profit."
        )
    if TRADING_ALLOWANCE_SPI_ROW not in household.columns:
        raise KeyError(
            f"household.{TRADING_ALLOWANCE_SPI_ROW} is missing; SPI rows never "
            "join the full-relief group."
        )
    if pd.Series(person["person_id"]).duplicated().any():
        raise ValueError(
            "person_id must be unique: the trading allowance draws are keyed on it."
        )
    profit = pd.to_numeric(person[TRADING_ALLOWANCE_PROFIT], errors="raise")
    spi_by_household = dict(
        zip(
            household["household_id"].to_numpy(),
            household[TRADING_ALLOWANCE_SPI_ROW].to_numpy(dtype=bool),
            strict=True,
        )
    )
    spi_row = person["person_household_id"].map(spi_by_household).to_numpy(bool)
    statuses = assign_trading_allowance_claim_status(
        person["person_id"].to_numpy(),
        profit.to_numpy(dtype=np.float64),
        spi_row=spi_row,
    )
    person[TRADING_ALLOWANCE_STATUS_OUTPUT] = statuses.status
    person[TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT] = statuses.full_relief

    household_weights = frame.weights_for("household")
    weight_by_household = dict(
        zip(
            household["household_id"].to_numpy(),
            np.asarray(household_weights.values, dtype=float),
            strict=True,
        )
    )
    source_by_household = dict(
        zip(
            household["household_id"].to_numpy(),
            source_household_ids(
                household, stage=TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME
            ),
            strict=True,
        )
    )
    weights = person["person_household_id"].map(weight_by_household).to_numpy(float)
    sources = person["person_household_id"].map(source_by_household).to_numpy()
    rates = claimant_rates(profit.to_numpy(dtype=np.float64))
    trader = profit.to_numpy(dtype=np.float64) > 0.0
    claimant = statuses.claimant
    bands: list[dict[str, object]] = []
    for rate in sorted(set(rates[trader].tolist()), reverse=True):
        in_band = trader & (rates == rate)
        supported = distinct_count(sources, in_band, weights)
        weighted = float(weights[in_band].sum())
        bands.append(
            {
                "rate": rate,
                "support": support_band(supported),
                # Weighted totals over the band's traders at the spine's weights.
                "weighted_traders": small_cell_share(weighted, supported),
                "weighted_claimant_share": small_cell_share(
                    float(weights[in_band & claimant].sum()) / weighted
                    if weighted > 0.0
                    else 0.0,
                    supported,
                ),
            }
        )
    mass = MassChangeRecord(
        entity="household",
        old_total=household_weights.total,
        new_total=household_weights.total,
        declared_factor=1.0,
        reason=TRADING_ALLOWANCE_RECEIPTS_MASS_CHANGE_REASON,
    )
    result = uk_national_frame(
        person=person,
        benunit=frame.table("benunit").copy(),
        household=household,
        time_period=uk_time_period(frame),
        weight_kind=uk_household_weight_kind(frame),
        household_weights=household_weights.values,
        mass_log=(*frame.mass_log, mass),
    )
    validate_uk_national_frame(result)
    return UKTradingAllowanceReceiptsResult(
        frame=result,
        traders=int(trader.sum()),
        claimants=int(claimant.sum()),
        full_relief=int(statuses.full_relief.sum()),
        full_relief_support=support_band(
            distinct_count(sources, statuses.full_relief, weights)
        ),
        bands=tuple(bands),
    )


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind the stage's operation closed-world."""

    if stage.stage != TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME:
        raise ValueError(
            f"Trading allowance receipts received stage {stage.stage!r}, expected "
            f"{TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME!r}."
        )
    expected = _expected_operations()
    operations = tuple(stage.operations)
    if tuple(operation.kind for operation in operations) != tuple(
        kind for kind, _ in expected
    ):
        raise ValueError(
            "trading_allowance_receipts operation order drifted: "
            f"{[operation.kind for operation in operations]!r}."
        )
    for operation, (kind, parameters) in zip(operations, expected, strict=True):
        if dict(operation.parameters) != dict(parameters):
            raise ValueError(
                f"trading_allowance_receipts {kind} parameters drifted: "
                f"{dict(operation.parameters)!r} != {dict(parameters)!r}."
            )
    if tuple(stage.outputs) != (
        TRADING_ALLOWANCE_STATUS_OUTPUT,
        TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
    ):
        raise ValueError(
            f"{TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME} must declare exactly the "
            f"{TRADING_ALLOWANCE_STATUS_OUTPUT} and "
            f"{TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT} outputs."
        )
    if stage.grain != "person":
        raise ValueError(
            f"{TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME} is person grain, got "
            f"{stage.grain!r}."
        )
