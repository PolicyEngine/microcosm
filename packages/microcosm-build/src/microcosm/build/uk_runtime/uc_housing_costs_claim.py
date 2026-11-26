"""Universal Credit housing costs element from DWP's award statement (B1 of microcosm#1710).

policyengine-uk 2.180.0 pays the housing costs element
(``uc_housing_costs_element``) to every renting Universal Credit family from
its rent, tenure and the Local Housing Allowance, unless the family's
``would_claim_uc_housing_costs_element`` is false (policyengine-uk#2951). The
FRS 2024-25 links most UC claims to DWP's award statement for the assessment
period that contains the interview (DWP, *Family Resources Survey
transformation: integrating administrative data for benefits*, section
4.3.2), and BENEFITS.UCHOUSEL holds that statement's weekly housing element:
``frs_spine`` carries whether it is positive as
``frs_uc_housing_element_status`` on the UC row's person. On some linked
claims the statement carries no housing element where the engine pays one,
which overstates those families' Universal Credit and the benefit cap's
reach.

This stage stores the flag for every benefit unit, just before ``uc_take_up``
reads entitlement through the engine, so the take-up draw sees the awards the
flag leaves:

* every benefit unit starts true, the engine's default;
* a linked FRS claim (a non-SPI household whose UC row carries a status) is
  false where DWP's statement carries no housing element and the locked
  engine pays a positive element, and true otherwise. A false therefore
  carries only the zeros the engine can't explain, since a family with it
  false gets no element under any reform (Vahid's review of
  policyengine-uk#2951);
* every other benefit unit keeps the default. An unlinked reporter or a
  future take-up entrant has no statement of its own, and an SPI-channel
  unit's copied statement no longer belongs to it, since the SPI income chain
  redraws the copies' incomes and benefit reports. Nothing is imputed for
  them: drawing them at the linked claims' rate took the spine's share of
  Great Britain UC households paid the element from 62.7% to 58.1% at prior
  weights in 2025, against DWP's 63.3%, where the linked statements alone
  give 60.9%.

María signed these choices on 25 November 2026 (microcosm#1710), the day she
ruled that only linked claims set the flag.

One limit (Vahid's review of microcosm#1718): the statement covers one
assessment period, so its zero can be temporary, for example in a gap between
tenancies or before DWP has verified the housing costs, and the flag then
removes the element for the whole year. That is the one way the stage can
err, since it never adds an element. The calibration binds DWP's counts of UC
households paid the element by tenure, so it still fits those totals, but it
can't give the element back to a family the flag took it from.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.frs_spine import (
    FRS_UC_HOUSING_ELEMENT_CODES,
    FRS_UC_HOUSING_ELEMENT_COLUMN,
    FRS_UC_LCW_NOT_LINKED,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    UK_DISCLOSURE_FLOOR,
    distinct_count,
    rounded_count,
    source_household_ids,
    support_band,
)
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "UC_HOUSING_COSTS_CLAIM_OUTPUTS",
    "UC_HOUSING_COSTS_CLAIM_STAGE_NAME",
    "UKUCHousingCostsClaimStageTransform",
    "assign_uc_housing_costs_claim",
    "uc_housing_costs_claim_operation_parameters",
]

UC_HOUSING_COSTS_CLAIM_STAGE_NAME = "uc_housing_costs_claim"
UC_HOUSING_COSTS_CLAIM_OPERATION_KIND = "assign_uc_housing_claim_from_admin_status"
UC_HOUSING_COSTS_CLAIM = "would_claim_uc_housing_costs_element"
UC_HOUSING_COSTS_CLAIM_OUTPUTS = (UC_HOUSING_COSTS_CLAIM,)
#: The element the locked engine pays with the flag at its default.
UC_HOUSING_COSTS_ELEMENT = "uc_housing_costs_element"
#: What the stage asks the engine, for every benefit unit, on the frame as it
#: stands.
UC_HOUSING_COSTS_CLAIM_ENGINE_VARIABLES = (UC_HOUSING_COSTS_ELEMENT,)
_NONE, _PAID = FRS_UC_HOUSING_ELEMENT_CODES[0], FRS_UC_HOUSING_ELEMENT_CODES[1]
_STATUSES = (_NONE, _PAID)
#: The frame holds survey-year values, and the flag is stored as a survey-year
#: input, which the engine carries into every projected year.
UC_HOUSING_COSTS_CLAIM_YEAR_RULE = "survey_year"
UC_HOUSING_COSTS_CLAIM_LINKED_SCOPE = "household_is_spi_synthetic"

UC_HOUSING_COSTS_CLAIM_RULE = (
    "Every benefit unit takes the engine's default, true. A linked FRS claim "
    "(a non-SPI household whose UC row carries a status) is false where DWP's "
    "award statement carries no housing element (UCHOUSEL 0) and the locked "
    "engine pays a positive uc_housing_costs_element, and true otherwise, so "
    "a false carries only the zeros the engine can't explain."
)
UC_HOUSING_COSTS_CLAIM_SCOPE = (
    "Only linked FRS claims set the flag. Every other benefit unit keeps the "
    "default: an unlinked reporter or a future take-up entrant has no "
    "statement of its own, and an SPI-channel unit's copied statement no "
    "longer belongs to it, since the SPI income chain redraws the copies' "
    "incomes and benefit reports. Nothing is imputed for them."
)
UC_HOUSING_COSTS_CLAIM_MASS_CHANGE_REASON = (
    "UC housing costs element claim flag stored on the source spine: household "
    "weights pass through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)


def uc_housing_costs_claim_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "materialize_rules_engine_predictors",
            {
                "predictors": list(UC_HOUSING_COSTS_CLAIM_ENGINE_VARIABLES),
                "year_rule": UC_HOUSING_COSTS_CLAIM_YEAR_RULE,
            },
        ),
        (
            UC_HOUSING_COSTS_CLAIM_OPERATION_KIND,
            {
                "outputs": list(UC_HOUSING_COSTS_CLAIM_OUTPUTS),
                "carrier": FRS_UC_HOUSING_ELEMENT_COLUMN,
                "statuses": list(_STATUSES),
                "not_linked": FRS_UC_LCW_NOT_LINKED,
                "element": UC_HOUSING_COSTS_ELEMENT,
                "linked_scope": UC_HOUSING_COSTS_CLAIM_LINKED_SCOPE,
                "rule": UC_HOUSING_COSTS_CLAIM_RULE,
                "scope": UC_HOUSING_COSTS_CLAIM_SCOPE,
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": UC_HOUSING_COSTS_CLAIM_MASS_CHANGE_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


@dataclass(frozen=True)
class UKUCHousingCostsClaimResult:
    """The frame with the flag stored, and the stage's receipt."""

    frame: Frame
    receipt: Mapping[str, object]

    def evidence(self) -> dict[str, object]:
        return {"stage": UC_HOUSING_COSTS_CLAIM_STAGE_NAME, **dict(self.receipt)}


def _benunit_households(person: pd.DataFrame, benunit_ids: np.ndarray) -> np.ndarray:
    """Each benefit unit's household, from its people."""

    pairs = pd.DataFrame(
        {
            "benunit": person["person_benunit_id"].to_numpy(),
            "household": person["person_household_id"].to_numpy(),
        }
    )
    spread = pairs.groupby("benunit")["household"].nunique()
    if (spread > 1).any():
        raise ValueError(
            f"{int((spread > 1).sum())} benefit unit(s) have people in two households."
        )
    household = pairs.groupby("benunit")["household"].first().reindex(benunit_ids)
    if household.isna().any():
        raise ValueError(
            f"{int(household.isna().sum())} benefit unit(s) have no people on the frame."
        )
    return household.to_numpy()


def assign_uc_housing_costs_claim(
    frame: Frame, *, engine_answers: Mapping[str, object]
) -> UKUCHousingCostsClaimResult:
    """Store the flag: DWP's statement on linked claims, the default elsewhere.

    ``engine_answers`` holds the engine's benefit-unit answer for
    ``UC_HOUSING_COSTS_CLAIM_ENGINE_VARIABLES`` on ``frame`` as it stands,
    with the flag at its default.
    """

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    benunit = frame.table("benunit").copy()
    household = frame.table("household").copy()
    if UC_HOUSING_COSTS_CLAIM in benunit.columns:
        raise ValueError(
            f"{UC_HOUSING_COSTS_CLAIM!r} is already stored: this stage is its only writer."
        )
    if FRS_UC_HOUSING_ELEMENT_COLUMN not in person.columns:
        raise KeyError(
            f"person.{FRS_UC_HOUSING_ELEMENT_COLUMN} is missing; frs_spine carries "
            "DWP's housing element status on the UC row's person."
        )
    if UC_HOUSING_COSTS_CLAIM_LINKED_SCOPE not in household.columns:
        raise KeyError(
            f"household.{UC_HOUSING_COSTS_CLAIM_LINKED_SCOPE} is missing; only FRS "
            "households carry a linked claim."
        )
    count = len(benunit)
    element = np.asarray(engine_answers[UC_HOUSING_COSTS_ELEMENT], dtype=float)
    if element.shape != (count,):
        raise ValueError(
            f"The engine's {UC_HOUSING_COSTS_ELEMENT} has shape {element.shape}, "
            f"not ({count},)."
        )
    if not np.isfinite(element).all():
        raise ValueError(f"The engine's {UC_HOUSING_COSTS_ELEMENT} is not finite.")
    positive = element > 0

    status = (
        person[FRS_UC_HOUSING_ELEMENT_COLUMN]
        .fillna(FRS_UC_LCW_NOT_LINKED)
        .astype(str)
        .to_numpy(dtype=object)
    )
    unknown = sorted(set(status) - {FRS_UC_LCW_NOT_LINKED, *_STATUSES})
    if unknown:
        raise ValueError(
            f"{FRS_UC_HOUSING_ELEMENT_COLUMN} values outside the codes: {unknown}."
        )
    benunit_ids = benunit["benunit_id"].to_numpy()
    carriers = status != FRS_UC_LCW_NOT_LINKED
    carrier_units = person["person_benunit_id"].to_numpy()[carriers]
    if pd.Series(carrier_units).duplicated().any():
        raise ValueError(
            "A benefit unit carries DWP's housing element status on two people; "
            "UCHOUSEL is one claim's, on its single UC row."
        )
    unit_status = (
        pd.Series(status[carriers], index=carrier_units)
        .reindex(benunit_ids, fill_value=FRS_UC_LCW_NOT_LINKED)
        .to_numpy(dtype=object)
    )

    household_ids = household["household_id"].to_numpy()
    unit_household = _benunit_households(person, benunit_ids)
    by_household = pd.DataFrame(
        {
            "spi": household[UC_HOUSING_COSTS_CLAIM_LINKED_SCOPE]
            .fillna(False)
            .astype(bool)
            .to_numpy(),
            "weight": np.asarray(frame.weights_for("household").values, dtype=float),
            "source": source_household_ids(
                household, stage=UC_HOUSING_COSTS_CLAIM_STAGE_NAME
            ),
        },
        index=household_ids,
    ).reindex(unit_household)
    spi = by_household["spi"].to_numpy(dtype=bool)
    weights = by_household["weight"].to_numpy(dtype=float)
    sources = by_household["source"].to_numpy()

    linked = (unit_status != FRS_UC_LCW_NOT_LINKED) & ~spi
    statement_none = unit_status == _NONE
    linked_without = linked & statement_none & positive
    claim = np.ones(count, dtype=bool)
    claim[linked_without] = False
    benunit[UC_HOUSING_COSTS_CLAIM] = claim

    support = distinct_count(sources, linked_without, weights)
    receipt: dict[str, object] = {
        "rule": UC_HOUSING_COSTS_CLAIM_RULE,
        "scope": UC_HOUSING_COSTS_CLAIM_SCOPE,
        "linked_claims": rounded_count(int(linked.sum())),
        "linked_claims_support": support_band(distinct_count(sources, linked, weights)),
        "linked": {
            "no_element_over_a_positive_engine_element": rounded_count(
                int(linked_without.sum())
            ),
            "no_element_where_the_engine_pays_none": rounded_count(
                int((linked & statement_none & ~positive).sum())
            ),
            "element_paid": rounded_count(int((linked & ~statement_none).sum())),
        },
        "without_the_element": {
            "benefit_units": rounded_count(int(linked_without.sum())),
            "weighted": round(float(weights[linked_without].sum()), -3)
            if support >= UK_DISCLOSURE_FLOOR
            else "withheld",
            "support": support_band(support),
        },
        # The units the engine pays an element that keep the default: no
        # statement of their own, and nothing imputed for them.
        "kept_the_default_over_a_positive_engine_element": {
            "unlinked": rounded_count(int((~linked & ~spi & positive).sum())),
            "spi_channel": rounded_count(int((spi & positive).sum())),
        },
        "spi_channel_benefit_units": rounded_count(int(spi.sum())),
    }
    household_weights = frame.weights_for("household")
    mass = MassChangeRecord(
        entity="household",
        old_total=household_weights.total,
        new_total=household_weights.total,
        declared_factor=1.0,
        reason=UC_HOUSING_COSTS_CLAIM_MASS_CHANGE_REASON,
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
    return UKUCHousingCostsClaimResult(frame=result, receipt=receipt)


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind the stage's operations closed-world."""

    if stage.stage != UC_HOUSING_COSTS_CLAIM_STAGE_NAME:
        raise ValueError(
            f"UC housing costs claim received stage {stage.stage!r}, expected "
            f"{UC_HOUSING_COSTS_CLAIM_STAGE_NAME!r}."
        )
    expected = uc_housing_costs_claim_operation_parameters()
    operations = tuple(stage.operations)
    if tuple(operation.kind for operation in operations) != tuple(
        kind for kind, _ in expected
    ):
        raise ValueError(
            f"{UC_HOUSING_COSTS_CLAIM_STAGE_NAME} operation order drifted: "
            f"{[operation.kind for operation in operations]!r}."
        )
    for operation, (kind, parameters) in zip(operations, expected, strict=True):
        if dict(operation.parameters) != dict(parameters):
            raise ValueError(
                f"{UC_HOUSING_COSTS_CLAIM_STAGE_NAME} {kind} parameters drifted: "
                f"{dict(operation.parameters)!r} != {dict(parameters)!r}."
            )
    if tuple(stage.outputs) != UC_HOUSING_COSTS_CLAIM_OUTPUTS:
        raise ValueError(
            f"{UC_HOUSING_COSTS_CLAIM_STAGE_NAME} must declare exactly the flag."
        )
    if stage.grain != "benunit":
        raise ValueError(
            f"{UC_HOUSING_COSTS_CLAIM_STAGE_NAME} is benunit grain, got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKUCHousingCostsClaimStageTransform:
    """Whole-stage callable storing the UC housing costs element claim flag."""

    stage: SourceStageSpec
    engine: object
    last_result: UKUCHousingCostsClaimResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        if UC_HOUSING_COSTS_CLAIM in frame.table("benunit").columns:
            raise ValueError(
                f"{UC_HOUSING_COSTS_CLAIM!r} is already stored: this stage is its "
                "only writer."
            )
        period = uk_time_period(frame)
        declared = resolve_uk_year_rule(UC_HOUSING_COSTS_CLAIM_YEAR_RULE)
        if int(period) != declared:
            raise ValueError(
                "UC housing costs claim reads the frame as "
                f"{UC_HOUSING_COSTS_CLAIM_YEAR_RULE} {declared} values, but the "
                f"frame's period is {period}."
            )
        answers = self.engine.materialize(  # type: ignore[attr-defined]
            frame, list(UC_HOUSING_COSTS_CLAIM_ENGINE_VARIABLES), period
        )
        result = assign_uc_housing_costs_claim(frame, engine_answers=answers)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UC_HOUSING_COSTS_CLAIM_OUTPUTS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
