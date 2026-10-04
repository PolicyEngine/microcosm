"""Pension Credit take-up by component on post-SPI incomes (microcosm#1069 R9).

``frs_take_up`` draws ``would_claim_pc`` early, on FRS incomes, so the engine
simulations of the intermediate stages see a value. This stage redraws it once
the SPI income chain has set the incomes the engine assesses. One temporary
engine materialization gives each benefit unit's pre-take-up Pension Credit
entitlement; the entitled units fall into two component bands, as DWP reports
take-up: Guarantee Credit (with or without Savings Credit) and Savings Credit
only. Within a band a reporter always claims and the other entitled units claim
at the residual rate

    r = (t * E - R) / (E - R),

with t the band's DWP caseload take-up rate, E the band's entitled units and R
the reporters among them, both weighted at the stage's household weights, so
reporters plus drawn claimants make up t of the band (or more, where reporters
alone exceed it; the receipt says so). A unit with no entitlement claims only
if it reports. The uniform draw is the ``would_claim_pc`` identity stream
``frs_take_up`` uses.
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
from microcosm.build.uk_runtime.take_up_contract import (
    UKTakeUpContract,
    load_uk_take_up_contract,
)
from microcosm.build.uk_runtime.uc_capital_coherence import (
    _household_to_benunit_weights,
)
from microcosm.frame import Frame
from microcosm.frame.rules import assert_rules_engine_country

PENSION_CREDIT_TAKE_UP_STAGE_NAME = "pension_credit_take_up"
PENSION_CREDIT_TAKE_UP_OUTPUT = "would_claim_pc"
PENSION_CREDIT_TAKE_UP_SEED = 0
PENSION_CREDIT_ENTITLEMENT_VARIABLES = (
    "guarantee_credit",
    "savings_credit",
    "is_pension_credit_eligible",
)
PENSION_CREDIT_REPORTED_ANCHOR = "pension_credit_reported_anchor"
PENSION_CREDIT_AGGREGATES = {PENSION_CREDIT_REPORTED_ANCHOR: "pension_credit_reported"}
#: The component bands, in the order DWP reports take-up, with their rules and
#: the take-up contract keys of their rates.
PENSION_CREDIT_TAKE_UP_BANDS = (
    {
        "name": "guarantee_credit",
        "rule": "is_pension_credit_eligible AND guarantee_credit > 0",
        "rate_key": "pension_credit_guarantee_credit",
    },
    {
        "name": "savings_credit_only",
        "rule": (
            "is_pension_credit_eligible AND guarantee_credit <= 0 "
            "AND savings_credit > 0"
        ),
        "rate_key": "pension_credit_savings_credit_only",
    },
)
PENSION_CREDIT_RESIDUAL_RATE = "(rate * entitled - reporters) / (entitled - reporters), weighted, clipped to [0, 1]"


@dataclass(frozen=True)
class UKPensionCreditTakeUpResult:
    """Output frame and executed-effect receipt for the Pension Credit redraw."""

    frame: Frame
    bands: tuple[Mapping[str, object], ...]
    reporters_without_entitlement: Mapping[str, float]
    changed_units: int

    def evidence(self) -> dict[str, object]:
        return {
            "stage": PENSION_CREDIT_TAKE_UP_STAGE_NAME,
            "seed": PENSION_CREDIT_TAKE_UP_SEED,
            "residual_rate": PENSION_CREDIT_RESIDUAL_RATE,
            "bands": [dict(band) for band in self.bands],
            "reporters_without_entitlement": dict(self.reporters_without_entitlement),
            "changed_units": self.changed_units,
        }


@dataclass(frozen=True)
class UKPensionCreditTakeUpStageTransform:
    """Whole-stage callable for the post-SPI Pension Credit take-up redraw."""

    stage: SourceStageSpec
    engine: object
    contract: UKTakeUpContract | None = None
    last_result: UKPensionCreditTakeUpResult | None = field(default=None, init=False)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        result = redraw_pension_credit_take_up(
            frame,
            engine=self.engine,
            contract=self.contract or load_uk_take_up_contract(),
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


def redraw_pension_credit_take_up(
    frame: Frame,
    *,
    engine: object,
    contract: UKTakeUpContract,
) -> UKPensionCreditTakeUpResult:
    """Redraw ``would_claim_pc`` by component band from engine entitlement."""

    validate_uk_national_frame(frame)
    assert_rules_engine_country(engine, "uk")
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    for table, columns, label in (
        (
            person,
            ("person_benunit_id", "person_household_id", "pension_credit_reported"),
            "person",
        ),
        (benunit, ("benunit_id", PENSION_CREDIT_TAKE_UP_OUTPUT), "benunit"),
    ):
        missing = sorted(set(columns) - set(table.columns))
        if missing:
            raise ValueError(
                f"Pension Credit take-up {label} columns missing: {missing}."
            )
    period = uk_time_period(frame)
    materialized = engine.materialize(
        frame, list(PENSION_CREDIT_ENTITLEMENT_VARIABLES), period
    )
    missing = sorted(set(PENSION_CREDIT_ENTITLEMENT_VARIABLES) - set(materialized))
    if missing:
        raise ValueError(f"Pension Credit take-up engine outputs missing: {missing}.")
    count = len(benunit)
    guarantee = _aligned(materialized["guarantee_credit"], count, "guarantee_credit")
    savings = _aligned(materialized["savings_credit"], count, "savings_credit")
    eligible = _aligned(
        materialized["is_pension_credit_eligible"], count, "is_pension_credit_eligible"
    ).astype(bool)
    reporter_ids = person.loc[
        pd.to_numeric(person["pension_credit_reported"], errors="coerce").fillna(0.0)
        > 0,
        "person_benunit_id",
    ]
    reporter = benunit["benunit_id"].isin(reporter_ids).to_numpy(dtype=bool)
    weights = _household_to_benunit_weights(
        benunit,
        person=person,
        household=household,
        household_weights=frame.weights_for("household").values,
    )
    draws = stable_identity_uniforms(
        benunit["benunit_id"].to_numpy(),
        seed=PENSION_CREDIT_TAKE_UP_SEED,
        salt=PENSION_CREDIT_TAKE_UP_OUTPUT,
    )
    would_claim, bands = assign_component_take_up(
        guarantee=guarantee,
        savings=savings,
        eligible=eligible,
        reporter=reporter,
        weights=weights,
        draws=draws,
        rates={
            band["rate_key"]: contract.rate(band["rate_key"])
            for band in PENSION_CREDIT_TAKE_UP_BANDS
        },
    )
    entitled = eligible & ((guarantee > 0.0) | (savings > 0.0))
    outside = reporter & ~entitled
    previous = benunit[PENSION_CREDIT_TAKE_UP_OUTPUT].fillna(False).to_numpy(dtype=bool)
    benunit[PENSION_CREDIT_TAKE_UP_OUTPUT] = would_claim
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
    return UKPensionCreditTakeUpResult(
        frame=result,
        bands=bands,
        reporters_without_entitlement={
            "units": int(outside.sum()),
            "weighted_units": float(weights[outside].sum()),
        },
        changed_units=int((previous != would_claim).sum()),
    )


def assign_component_take_up(
    *,
    guarantee: np.ndarray,
    savings: np.ndarray,
    eligible: np.ndarray,
    reporter: np.ndarray,
    weights: np.ndarray,
    draws: np.ndarray,
    rates: Mapping[str, float],
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Reporters claim; entitled non-reporters claim at their band's residual rate."""

    guarantee = np.asarray(guarantee, dtype=np.float64)
    savings = np.asarray(savings, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    reporter = np.asarray(reporter, dtype=bool)
    weights = np.asarray(weights, dtype=np.float64)
    draws = np.asarray(draws, dtype=np.float64)
    masks = {
        "guarantee_credit": eligible & (guarantee > 0.0),
        "savings_credit_only": eligible & (guarantee <= 0.0) & (savings > 0.0),
    }
    would_claim = reporter.copy()
    receipts = []
    for band in PENSION_CREDIT_TAKE_UP_BANDS:
        mask = masks[band["name"]]
        rate = float(rates[band["rate_key"]])
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                f"{band['rate_key']} must be a rate in [0, 1], got {rate}."
            )
        entitled = float(weights[mask].sum())
        reporting = float(weights[mask & reporter].sum())
        residual = 0.0
        if entitled > reporting:
            residual = float(
                np.clip(
                    (rate * entitled - reporting) / (entitled - reporting), 0.0, 1.0
                )
            )
        drawn = mask & ~reporter & (draws < residual)
        would_claim |= drawn
        realized = (
            float(weights[mask & would_claim].sum()) / entitled
            if entitled > 0.0
            else None
        )
        receipts.append(
            {
                "band": band["name"],
                "rule": band["rule"],
                "rate_key": band["rate_key"],
                "rate": rate,
                "entitled_units": int(mask.sum()),
                "entitled_weighted": entitled,
                "reporter_units": int((mask & reporter).sum()),
                "reporter_weighted": reporting,
                "residual_rate": residual,
                "drawn_units": int(drawn.sum()),
                "realized_take_up": realized,
                "reporters_exceed_rate": bool(
                    entitled > 0.0 and reporting > rate * entitled
                ),
            }
        )
    return would_claim, tuple(receipts)


def _aligned(values: object, expected: int, label: str) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (expected,):
        raise ValueError(f"{label} must align to the benefit-unit table.")
    if array.dtype != bool and not np.isfinite(array.astype(np.float64)).all():
        raise ValueError(f"{label} must be finite.")
    return array


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    if stage.stage != PENSION_CREDIT_TAKE_UP_STAGE_NAME:
        raise ValueError(
            f"Pension Credit take-up received stage {stage.stage!r}, expected "
            f"{PENSION_CREDIT_TAKE_UP_STAGE_NAME!r}."
        )
    kinds = [operation.kind for operation in stage.operations]
    expected_kinds = [
        "materialize_rules_engine_predictors",
        "aggregate_person_to_benunit",
        "assign_component_take_up_residual",
    ]
    if kinds != expected_kinds:
        raise ValueError(
            f"{PENSION_CREDIT_TAKE_UP_STAGE_NAME} operation order drifted: "
            f"expected {expected_kinds}, got {kinds}."
        )
    materialize, aggregate, assign = stage.operations
    expected = {
        "materialize": {
            "predictors": list(PENSION_CREDIT_ENTITLEMENT_VARIABLES),
            "consumed_only": True,
        },
        "aggregate": {
            "method": "any_positive",
            "consumed_only": True,
            "aggregates": PENSION_CREDIT_AGGREGATES,
        },
        "assign": {
            "output": PENSION_CREDIT_TAKE_UP_OUTPUT,
            "draw": PENSION_CREDIT_TAKE_UP_OUTPUT,
            "seed": PENSION_CREDIT_TAKE_UP_SEED,
            "anchor": PENSION_CREDIT_REPORTED_ANCHOR,
            "bands": [dict(band) for band in PENSION_CREDIT_TAKE_UP_BANDS],
            "weight_mapping": "household_to_benunit",
            "residual_rate": PENSION_CREDIT_RESIDUAL_RATE,
        },
    }
    actual = {
        "materialize": dict(materialize.parameters),
        "aggregate": dict(aggregate.parameters),
        "assign": dict(assign.parameters),
    }
    if actual != expected:
        raise ValueError(
            f"{PENSION_CREDIT_TAKE_UP_STAGE_NAME} parameters drifted: expected "
            f"{expected}, got {actual}."
        )
    if stage.outputs != () or stage.rewrites != (PENSION_CREDIT_TAKE_UP_OUTPUT,):
        raise ValueError(
            f"{PENSION_CREDIT_TAKE_UP_STAGE_NAME} must declare no new outputs and "
            "exactly the would_claim_pc rewrite."
        )


__all__ = [
    "PENSION_CREDIT_ENTITLEMENT_VARIABLES",
    "PENSION_CREDIT_TAKE_UP_BANDS",
    "PENSION_CREDIT_TAKE_UP_STAGE_NAME",
    "UKPensionCreditTakeUpResult",
    "UKPensionCreditTakeUpStageTransform",
    "assign_component_take_up",
    "redraw_pension_credit_take_up",
]
