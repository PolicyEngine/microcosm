"""Self-employed gross receipts for the trading allowance (microcosm#1428).

policyengine-uk applies the trading allowance (ITTOIA 2005 Part 7A) against a
trader's gross receipts, ``self_employment_gross_receipts``. Neither the FRS
nor the SPI tape records turnover. Without receipts the engine treats a profit
within the allowance as receipts within it (full relief) and a larger profit
as already reflecting the better of expenses and the allowance (no relief). A
rise in the allowance then relieves only profits up to the new allowance, in
full, and misses its main base: traders who already claim the GBP 1,000
allowance against larger receipts and gain the larger deduction.

Under current law a trader who claims the allowance deducts GBP 1,000 from
receipts in place of expenses, so their receipts are their profit plus
GBP 1,000 exactly; a trader who deducts actual expenses has expenses above
GBP 1,000. This stage, the last of the spine, flags each trader with a
positive profit as a claimant with a probability by profit band, drawn on
their own ``person_id``, and stores their receipts:

* a claimant's are their profit plus GBP 1,000;
* any other trader's are their profit plus GBP 3,000, expenses at the raised
  allowance, so that the raise does not relieve them (the costing books the
  traders who would switch to the allowance as behaviour);
* a person with no positive profit keeps zero, which the engine reads as
  unknown.

The engine then gives a claimant nothing more at GBP 1,000 and up to
GBP 2,000 more at GBP 3,000, and any other trader nothing at either. Every
trader with a positive profit is treated as filing, as SPI Table 3.10's
counts, which bind the frame's traders, are of filers. So a profit of GBP 1,000
or less is now taxed where the engine used to relieve it in full.

Until HMRC publishes claimant counts, the rates are calibrated to HMRC's
policy paper on the Budget 2026 raise (about 400,000 taxpayers with a lower
bill, by about GBP 380 on average), measured on policyengine-uk for the
#1428 national from 9f70c71e7 at 2028-29 prices (María's choices of
2026-10-31 and 2026-11-01). The rates drift with each national, as its
small traders and their other incomes move, so microcosm#1439 re-measures
them after each build.

Two limits of encoding the claim in receipts (#1428 review, findings 2 and 8;
María's ruling of 2026-11-01 keeps them, documented, until policyengine-uk#2765
lets the stage store the claim status itself, microcosm#1439):

* The engine uprates receipts with profit, so a claimant's GBP 1,000 grows with
  the mixed income index while the allowance is nominal: the raise relieves
  GBP 1,932 more in 2027-28, falling to GBP 1,758 by 2031-32, 3.4% to 12.1%
  below GBP 2,000.
* The other traders' stored GBP 3,000 grows the same way, so an allowance above
  GBP 3,000 times that index relieves them from the stored expense, not from
  data. Allowances above GBP 3,000 are mis-scored on this encoding.
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
    "TRADING_ALLOWANCE_RECEIPTS_OUTPUT",
    "TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME",
    "UKTradingAllowanceReceiptsStageTransform",
    "assign_trading_allowance_receipts",
    "claimant_rates",
    "store_trading_allowance_receipts",
]

TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME = "trading_allowance_receipts"
TRADING_ALLOWANCE_RECEIPTS_OUTPUT = "self_employment_gross_receipts"
TRADING_ALLOWANCE_PROFIT = "self_employment_income"
TRADING_ALLOWANCE_CLAIMANT_DRAW = "trading_allowance_claimant"
TRADING_ALLOWANCE_CLAIMANT_SEED = 0
#: The allowance a claimant deducts in place of expenses (2024-25).
TRADING_ALLOWANCE_CLAIMANT_EXPENSES = 1_000.0
#: Any other trader's expenses: at the raised allowance, which then gives them
#: nothing.
TRADING_ALLOWANCE_OTHER_EXPENSES = 3_000.0
#: The claimant probability by the person's profit, upper bounds inclusive,
#: over profits above zero; the last band has no upper bound.
TRADING_ALLOWANCE_CLAIMANT_BANDS: tuple[tuple[float | None, float], ...] = (
    (4_500.0, 0.72),
    (9_000.0, 0.14),
    (None, 0.0),
)
TRADING_ALLOWANCE_CALIBRATION = (
    "HMRC's policy paper on the Budget 2026 trading allowance raise: about "
    "400,000 taxpayers with a lower bill, by about GBP 380 on average, measured "
    "on policyengine-uk for the microcosm#1428 national from 9f70c71e7 at "
    "2028-29 prices; the rates drift with each national (microcosm#1439), until "
    "HMRC publishes claimant counts (María's choices of 2026-10-31 and 2026-11-01)"
)
TRADING_ALLOWANCE_RECEIPTS_MASS_CHANGE_REASON = (
    "Self-employed gross receipts stored on the source spine: household weights "
    "pass through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)
TRADING_ALLOWANCE_RECEIPTS_RULE = (
    "a person with a positive self_employment_income claims the allowance "
    "where their uniform falls below their profit band's rate; a claimant's "
    "receipts are their profit plus claimant_expenses, any other trader's their "
    "profit plus other_expenses, and a person with no positive profit keeps zero"
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
                "output": TRADING_ALLOWANCE_RECEIPTS_OUTPUT,
                "profit": TRADING_ALLOWANCE_PROFIT,
                "draw": TRADING_ALLOWANCE_CLAIMANT_DRAW,
                "seed": TRADING_ALLOWANCE_CLAIMANT_SEED,
                "claimant_bands": bands,
                "claimant_expenses": TRADING_ALLOWANCE_CLAIMANT_EXPENSES,
                "other_expenses": TRADING_ALLOWANCE_OTHER_EXPENSES,
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


def assign_trading_allowance_receipts(
    person_ids: np.ndarray,
    profit: np.ndarray,
    *,
    seed: int = TRADING_ALLOWANCE_CLAIMANT_SEED,
    salt: str = TRADING_ALLOWANCE_CLAIMANT_DRAW,
) -> tuple[np.ndarray, np.ndarray]:
    """Each person's gross receipts and claimant flag."""

    profit = np.asarray(profit, dtype=np.float64)
    ids = np.asarray(person_ids)
    if ids.shape != profit.shape:
        raise ValueError("Trading allowance person ids must align to the profits.")
    if not np.isfinite(profit).all():
        raise ValueError("Trading allowance profits must be finite.")
    trader = profit > 0.0
    claimant = np.zeros(profit.shape, dtype=bool)
    if trader.any():
        uniforms = stable_identity_uniforms(ids[trader], seed=seed, salt=salt)
        claimant[trader] = uniforms < claimant_rates(profit[trader])
    receipts = np.where(
        claimant,
        profit + TRADING_ALLOWANCE_CLAIMANT_EXPENSES,
        np.where(trader, profit + TRADING_ALLOWANCE_OTHER_EXPENSES, 0.0),
    )
    return receipts, claimant


@dataclass(frozen=True)
class UKTradingAllowanceReceiptsResult:
    """The frame with the stored receipts, and the stage's receipt."""

    frame: Frame
    traders: int
    claimants: int
    bands: tuple[Mapping[str, object], ...]

    def evidence(self) -> dict[str, object]:
        return {
            "stage": TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME,
            "seed": TRADING_ALLOWANCE_CLAIMANT_SEED,
            "draw": TRADING_ALLOWANCE_CLAIMANT_DRAW,
            # Record counts are rounded, never exact (UKDS CD137 cl. 8).
            "traders": rounded_count(self.traders),
            "claimants": rounded_count(self.claimants),
            "bands": [dict(band) for band in self.bands],
        }


@dataclass(frozen=True)
class UKTradingAllowanceReceiptsStageTransform:
    """Whole-stage callable storing the self-employed's gross receipts."""

    stage: SourceStageSpec
    last_result: UKTradingAllowanceReceiptsResult | None = field(
        default=None, init=False
    )

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = store_trading_allowance_receipts(frame)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return (TRADING_ALLOWANCE_RECEIPTS_OUTPUT,)

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}


def store_trading_allowance_receipts(frame: Frame) -> UKTradingAllowanceReceiptsResult:
    """Store each person's gross receipts; nothing else moves."""

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    if TRADING_ALLOWANCE_RECEIPTS_OUTPUT in person.columns:
        raise ValueError(
            f"{TRADING_ALLOWANCE_RECEIPTS_OUTPUT!r} is already stored: this stage "
            "is its only writer."
        )
    if TRADING_ALLOWANCE_PROFIT not in person.columns:
        raise KeyError(
            f"person.{TRADING_ALLOWANCE_PROFIT} is missing; the receipts are "
            "drawn on the final self-employment profit."
        )
    if pd.Series(person["person_id"]).duplicated().any():
        raise ValueError(
            "person_id must be unique: the trading allowance draw is keyed on it."
        )
    profit = pd.to_numeric(person[TRADING_ALLOWANCE_PROFIT], errors="raise")
    receipts, claimant = assign_trading_allowance_receipts(
        person["person_id"].to_numpy(), profit.to_numpy(dtype=np.float64)
    )
    person[TRADING_ALLOWANCE_RECEIPTS_OUTPUT] = receipts

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
    if tuple(stage.outputs) != (TRADING_ALLOWANCE_RECEIPTS_OUTPUT,):
        raise ValueError(
            f"{TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME} must declare exactly the "
            f"{TRADING_ALLOWANCE_RECEIPTS_OUTPUT} output."
        )
    if stage.grain != "person":
        raise ValueError(
            f"{TRADING_ALLOWANCE_RECEIPTS_STAGE_NAME} is person grain, got "
            f"{stage.grain!r}."
        )
