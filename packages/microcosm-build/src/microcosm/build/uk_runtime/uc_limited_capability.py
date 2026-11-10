"""Universal Credit limited capability from DWP's admin-linked status (Y1 of the November event-data plan, microcosm#1095).

policyengine-uk 2.177.4 decides a person's limited capability for work and
work-related activity (``uc_limited_capability_for_WRA``) from a proxy:
``is_disabled_for_benefits``, which the spine sets from any reported
disability benefit or ESA, unless the person's own ESA award lacks the support
component. Limited capability for work (``uc_limited_capability_for_work``) is
that or the same flag. The FRS 2024-25 links most UC claims to DWP's records,
and BENEFITS.UCLCW holds the claim's status: ``frs_spine`` carries it as
``frs_uc_lcw_status`` on the UC row's person.

On María's NS6 ruling of 9 November 2026, which revisits her ruling of 29
October to keep the proxy, this stage stores both flags for every person, just
before ``uc_take_up`` reads entitlement through the engine. Everyone takes the
engine's own proxy, from the locked engine, so nothing mirrors its formula
here. Linked FRS claims then take DWP's status:

* NONE: every claimant is false on both flags, whatever the proxy says;
* LCWRA: the claimants the proxy flags keep their flags. If it flags none, one
  claimant takes both flags: the UC row's person, or the partner where the UC
  row's person cares (``is_carer_for_benefits``) and the partner doesn't, so a
  carer keeps the carer element;
* LCW: the claimants the proxy gives limited capability for work keep it, and
  if it gives none, one claimant takes it by the same choice; nobody in the
  claim has WRA.

The FRS keeps one UC row per claim, couples included, so the status is the
claim's, and which partner holds it in a joint claim is this rule's
approximation. Unlinked reporters, the take-up redraw's entrants, the
legacy-closure movers and every SPI row keep the proxy, which their stored
value equals: no later stage writes the proxy's inputs. María signed these
choices on 9 November 2026.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.frs_spine import (
    FRS_UC_LCW_NOT_LINKED,
    FRS_UC_LCW_STATUS_CODES,
    FRS_UC_LCW_STATUS_COLUMN,
)
from microcosm.build.uk_runtime.national_frame import (
    uk_household_weight_kind,
    uk_national_frame,
    uk_time_period,
    validate_uk_national_frame,
)
from microcosm.build.uk_runtime.receipt_counts import (
    distinct_count,
    rounded_count,
    source_household_ids,
    support_band,
)
from microcosm.frame import Frame, MassChangeRecord

__all__ = [
    "UC_LIMITED_CAPABILITY_OUTPUTS",
    "UC_LIMITED_CAPABILITY_STAGE_NAME",
    "UKUCLimitedCapabilityStageTransform",
    "assign_uc_limited_capability",
    "uc_limited_capability_operation_parameters",
]

UC_LIMITED_CAPABILITY_STAGE_NAME = "uc_limited_capability"
UC_LIMITED_CAPABILITY_OPERATION_KIND = "assign_uc_limited_capability_from_admin_status"
UC_LCWRA = "uc_limited_capability_for_WRA"
UC_LCW = "uc_limited_capability_for_work"
UC_LIMITED_CAPABILITY_OUTPUTS: tuple[str, ...] = (UC_LCWRA, UC_LCW)
UC_LIMITED_CAPABILITY_CLAIMANT = "is_uc_assessed_claimant"
UC_LIMITED_CAPABILITY_CARER = "is_carer_for_benefits"
#: What the stage asks the engine, for every person, on the frame as it stands.
UC_LIMITED_CAPABILITY_ENGINE_VARIABLES: tuple[str, ...] = (
    UC_LCWRA,
    UC_LCW,
    UC_LIMITED_CAPABILITY_CLAIMANT,
    UC_LIMITED_CAPABILITY_CARER,
)
#: The frame holds survey-year values, and the flags are stored as survey-year
#: inputs, which the engine carries into every projected year.
UC_LIMITED_CAPABILITY_YEAR_RULE = "survey_year"
UC_LIMITED_CAPABILITY_LINKED_SCOPE = "household_is_spi_synthetic"
_STATUSES = tuple(FRS_UC_LCW_STATUS_CODES.values())

UC_LIMITED_CAPABILITY_RULE = (
    "Every person takes the locked engine's own proxy for both flags. A linked "
    "FRS claim (a non-SPI household whose UC row carries a status) takes DWP's: "
    "NONE sets every claimant false on both flags; LCWRA keeps the claimants "
    "the proxy flags, and if it flags none gives both flags to one claimant; "
    "LCW keeps the claimants the proxy gives limited capability for work, and "
    "if it gives none gives it to one claimant, with WRA false for every "
    "claimant."
)
UC_LIMITED_CAPABILITY_JOINT_CLAIM_RULE = (
    "The FRS keeps one UC row per claim, couples included, so the status is "
    "the claim's. Where a claimant must be chosen it is the UC row's person, "
    "or the partner where the UC row's person cares (is_carer_for_benefits) "
    "and the partner does not; the other claimant keeps the proxy."
)
UC_LIMITED_CAPABILITY_SCOPE = (
    "Unlinked reporters, the take-up redraw's entrants, the legacy-closure "
    "movers and every SPI row keep the proxy; no later stage writes the "
    "proxy's inputs, so their stored value equals the formula's."
)
UC_LIMITED_CAPABILITY_MASS_CHANGE_REASON = (
    "UC limited capability flags stored on the source spine: household weights "
    "pass through unchanged and total household mass is conserved."
)
_MASS_GATE_COUPLING = (
    "The terminal family gate requires a valid mass-conserving MassChangeRecord "
    "carrying exactly this stage-specific reason."
)
_BRANCHES = (
    "none_over_a_proxy_flag",
    "none_where_the_proxy_agrees",
    "lcwra_kept_from_the_proxy",
    "lcwra_given_to_a_single_claimant",
    "lcwra_given_to_the_uc_rows_person",
    "lcwra_given_to_the_partner_of_a_carer",
    "lcw_kept_from_the_proxy",
    "lcw_given_to_a_single_claimant",
    "lcw_given_to_the_uc_rows_person",
    "lcw_given_to_the_partner_of_a_carer",
    "linked_claim_with_no_claimant_kept_the_proxy",
)


def uc_limited_capability_operation_parameters() -> tuple[
    tuple[str, dict[str, object]], ...
]:
    """The reviewed operations the manifest must carry verbatim."""

    return (
        (
            "materialize_rules_engine_predictors",
            {
                "predictors": list(UC_LIMITED_CAPABILITY_ENGINE_VARIABLES),
                "year_rule": UC_LIMITED_CAPABILITY_YEAR_RULE,
            },
        ),
        (
            UC_LIMITED_CAPABILITY_OPERATION_KIND,
            {
                "outputs": list(UC_LIMITED_CAPABILITY_OUTPUTS),
                "carrier": FRS_UC_LCW_STATUS_COLUMN,
                "statuses": list(_STATUSES),
                "not_linked": FRS_UC_LCW_NOT_LINKED,
                "claimant": UC_LIMITED_CAPABILITY_CLAIMANT,
                "carer": UC_LIMITED_CAPABILITY_CARER,
                "linked_scope": UC_LIMITED_CAPABILITY_LINKED_SCOPE,
                "rule": UC_LIMITED_CAPABILITY_RULE,
                "joint_claim_rule": UC_LIMITED_CAPABILITY_JOINT_CLAIM_RULE,
                "scope": UC_LIMITED_CAPABILITY_SCOPE,
            },
        ),
        (
            "record_mass_conservation_receipt",
            {
                "entity": "household",
                "reason": UC_LIMITED_CAPABILITY_MASS_CHANGE_REASON,
                "declared_factor": 1.0,
                "gate_coupling": _MASS_GATE_COUPLING,
            },
        ),
    )


@dataclass(frozen=True)
class UKUCLimitedCapabilityResult:
    """The frame with both flags stored, and the stage's receipt."""

    frame: Frame
    receipt: Mapping[str, object]

    def evidence(self) -> dict[str, object]:
        return {"stage": UC_LIMITED_CAPABILITY_STAGE_NAME, **dict(self.receipt)}


def _bool(values: object, name: str, count: int) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (count,):
        raise ValueError(
            f"The engine's {name} has shape {array.shape}, not ({count},)."
        )
    return array.astype(bool)


def assign_uc_limited_capability(
    frame: Frame, *, engine_answers: Mapping[str, object]
) -> UKUCLimitedCapabilityResult:
    """Store both flags: the engine's proxy, with DWP's status on linked claims.

    ``engine_answers`` holds the engine's person-level answers for
    ``UC_LIMITED_CAPABILITY_ENGINE_VARIABLES`` on ``frame`` as it stands.
    """

    validate_uk_national_frame(frame)
    person = frame.table("person").copy()
    household = frame.table("household").copy()
    for output in UC_LIMITED_CAPABILITY_OUTPUTS:
        if output in person.columns:
            raise ValueError(
                f"{output!r} is already stored: this stage is its only writer."
            )
    if FRS_UC_LCW_STATUS_COLUMN not in person.columns:
        raise KeyError(
            f"person.{FRS_UC_LCW_STATUS_COLUMN} is missing; frs_spine carries "
            "DWP's status on the UC row's person."
        )
    if UC_LIMITED_CAPABILITY_LINKED_SCOPE not in household.columns:
        raise KeyError(
            f"household.{UC_LIMITED_CAPABILITY_LINKED_SCOPE} is missing; only "
            "FRS households carry a linked claim."
        )
    count = len(person)
    proxy_wra = _bool(engine_answers[UC_LCWRA], UC_LCWRA, count)
    proxy_lcw = _bool(engine_answers[UC_LCW], UC_LCW, count)
    claimant = _bool(
        engine_answers[UC_LIMITED_CAPABILITY_CLAIMANT],
        UC_LIMITED_CAPABILITY_CLAIMANT,
        count,
    )
    carer = _bool(
        engine_answers[UC_LIMITED_CAPABILITY_CARER], UC_LIMITED_CAPABILITY_CARER, count
    )
    status = (
        person[FRS_UC_LCW_STATUS_COLUMN]
        .fillna(FRS_UC_LCW_NOT_LINKED)
        .astype(str)
        .to_numpy(dtype=object)
    )
    unknown = sorted(set(status) - {FRS_UC_LCW_NOT_LINKED, *_STATUSES})
    if unknown:
        raise ValueError(
            f"{FRS_UC_LCW_STATUS_COLUMN} values outside the codes: {unknown}."
        )

    household_ids = household["household_id"].to_numpy()
    spi_by_household = dict(
        zip(
            household_ids,
            household[UC_LIMITED_CAPABILITY_LINKED_SCOPE]
            .fillna(False)
            .astype(bool)
            .to_numpy(),
            strict=True,
        )
    )
    weights_by_household = dict(
        zip(
            household_ids,
            np.asarray(frame.weights_for("household").values, dtype=float),
            strict=True,
        )
    )
    sources_by_household = dict(
        zip(
            household_ids,
            source_household_ids(household, stage=UC_LIMITED_CAPABILITY_STAGE_NAME),
            strict=True,
        )
    )
    p_household = person["person_household_id"]
    spi = p_household.map(spi_by_household).to_numpy(dtype=bool)
    weights = p_household.map(weights_by_household).to_numpy(dtype=float)
    sources = p_household.map(sources_by_household).to_numpy()
    benunits = person["person_benunit_id"].to_numpy()

    carriers = np.flatnonzero((status != FRS_UC_LCW_NOT_LINKED) & ~spi)
    if pd.Series(benunits[carriers]).duplicated().any():
        raise ValueError(
            "A benefit unit carries DWP's status on two people; UCLCW is one "
            "claim's status, on its single UC row."
        )
    members: dict[object, list[int]] = {}
    for row, benunit in enumerate(benunits):
        members.setdefault(benunit, []).append(row)

    wra = proxy_wra.copy()
    lcw = proxy_lcw.copy()
    claims: dict[str, list[int]] = {name: [] for name in _BRANCHES}
    agreement = {
        claim_status: {"proxy_flags": [], "proxy_does_not": []}
        for claim_status in _STATUSES
    }
    for carrier_row in carriers:
        claim_status = status[carrier_row]
        rows = [row for row in members[benunits[carrier_row]] if claimant[row]]
        flags = proxy_wra if claim_status in ("NONE", "LCWRA") else proxy_lcw
        flagged = [row for row in rows if flags[row]]
        agreement[claim_status]["proxy_flags" if flagged else "proxy_does_not"].append(
            carrier_row
        )
        if not rows:
            claims["linked_claim_with_no_claimant_kept_the_proxy"].append(carrier_row)
            continue
        if claim_status == "NONE":
            if any(proxy_wra[row] or proxy_lcw[row] for row in rows):
                claims["none_over_a_proxy_flag"].append(carrier_row)
            else:
                claims["none_where_the_proxy_agrees"].append(carrier_row)
            wra[rows] = False
            lcw[rows] = False
            continue
        prefix = claim_status.lower()
        if claim_status == "LCW":
            wra[rows] = False
        if flagged:
            claims[f"{prefix}_kept_from_the_proxy"].append(carrier_row)
            continue
        chosen = carrier_row if carrier_row in rows else rows[0]
        others = [row for row in rows if row != chosen]
        if len(rows) == 1:
            claims[f"{prefix}_given_to_a_single_claimant"].append(carrier_row)
        elif len(others) == 1 and carer[chosen] and not carer[others[0]]:
            chosen = others[0]
            claims[f"{prefix}_given_to_the_partner_of_a_carer"].append(carrier_row)
        else:
            claims[f"{prefix}_given_to_the_uc_rows_person"].append(carrier_row)
        lcw[chosen] = True
        wra[chosen] = claim_status == "LCWRA"

    person[UC_LCWRA] = wra
    person[UC_LCW] = lcw

    def cell(rows: list[int]) -> dict[str, object]:
        mask = np.zeros(count, dtype=bool)
        mask[rows] = True
        support = distinct_count(sources, mask, weights)
        return {
            "claims": rounded_count(len(rows)),
            "weighted": round(float(weights[mask].sum()), -3) if support >= 3 else None,
            "support": support_band(support),
        }

    cells = {
        claim_status: {side: cell(rows) for side, rows in sides.items()}
        for claim_status, sides in agreement.items()
    }
    # A cell under three source households is withheld, and so is its
    # status's other cell, so no total gives it back (UKDS disclosure control).
    for sides in cells.values():
        if any(entry["weighted"] is None for entry in sides.values()):
            for entry in sides.values():
                entry["weighted"] = "withheld"
    receipt: dict[str, object] = {
        "rule": UC_LIMITED_CAPABILITY_RULE,
        "joint_claim_rule": UC_LIMITED_CAPABILITY_JOINT_CLAIM_RULE,
        "scope": UC_LIMITED_CAPABILITY_SCOPE,
        "linked_claims": rounded_count(len(carriers)),
        "linked_claims_support": support_band(
            distinct_count(
                sources,
                np.isin(np.arange(count), carriers),
                weights,
            )
        ),
        "status_against_the_proxy": cells,
        "branches": {name: rounded_count(len(rows)) for name, rows in claims.items()},
        "people_changed": {
            UC_LCWRA: rounded_count(int((wra != proxy_wra).sum())),
            UC_LCW: rounded_count(int((lcw != proxy_lcw).sum())),
        },
    }
    household_weights = frame.weights_for("household")
    mass = MassChangeRecord(
        entity="household",
        old_total=household_weights.total,
        new_total=household_weights.total,
        declared_factor=1.0,
        reason=UC_LIMITED_CAPABILITY_MASS_CHANGE_REASON,
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
    return UKUCLimitedCapabilityResult(frame=result, receipt=receipt)


def _assert_stage_parameters(stage: SourceStageSpec) -> None:
    """Bind the stage's operations closed-world."""

    if stage.stage != UC_LIMITED_CAPABILITY_STAGE_NAME:
        raise ValueError(
            f"UC limited capability received stage {stage.stage!r}, expected "
            f"{UC_LIMITED_CAPABILITY_STAGE_NAME!r}."
        )
    expected = uc_limited_capability_operation_parameters()
    operations = tuple(stage.operations)
    if tuple(operation.kind for operation in operations) != tuple(
        kind for kind, _ in expected
    ):
        raise ValueError(
            "uc_limited_capability operation order drifted: "
            f"{[operation.kind for operation in operations]!r}."
        )
    for operation, (kind, parameters) in zip(operations, expected, strict=True):
        if dict(operation.parameters) != dict(parameters):
            raise ValueError(
                f"uc_limited_capability {kind} parameters drifted: "
                f"{dict(operation.parameters)!r} != {dict(parameters)!r}."
            )
    if tuple(stage.outputs) != UC_LIMITED_CAPABILITY_OUTPUTS:
        raise ValueError(
            f"{UC_LIMITED_CAPABILITY_STAGE_NAME} must declare exactly the two flags."
        )
    if stage.grain != "person":
        raise ValueError(
            f"{UC_LIMITED_CAPABILITY_STAGE_NAME} is person grain, got {stage.grain!r}."
        )


@dataclass(frozen=True)
class UKUCLimitedCapabilityStageTransform:
    """Whole-stage callable storing both UC limited capability flags."""

    stage: SourceStageSpec
    engine: object
    last_result: UKUCLimitedCapabilityResult | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _assert_stage_parameters(self.stage)

    def __call__(self, frame: Frame) -> Frame:
        _assert_stage_parameters(self.stage)
        for output in UC_LIMITED_CAPABILITY_OUTPUTS:
            if output in frame.table("person").columns:
                raise ValueError(
                    f"{output!r} is already stored: this stage is its only writer."
                )
        period = uk_time_period(frame)
        declared = resolve_uk_year_rule(UC_LIMITED_CAPABILITY_YEAR_RULE)
        if int(period) != declared:
            raise ValueError(
                f"UC limited capability reads the frame as {UC_LIMITED_CAPABILITY_YEAR_RULE} "
                f"{declared} values, but the frame's period is {period}."
            )
        answers = self.engine.materialize(  # type: ignore[attr-defined]
            frame, list(UC_LIMITED_CAPABILITY_ENGINE_VARIABLES), period
        )
        result = assign_uc_limited_capability(frame, engine_answers=answers)
        object.__setattr__(self, "last_result", result)
        return result.frame

    @staticmethod
    def output_columns() -> tuple[str, ...]:
        return UC_LIMITED_CAPABILITY_OUTPUTS

    def checkpoint_metadata(self) -> dict[str, object]:
        if self.last_result is None:
            raise RuntimeError("checkpoint metadata requires a completed stage run.")
        return {"evidence": self.last_result.evidence()}
