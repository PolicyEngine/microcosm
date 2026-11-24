"""Universal Credit limited capability from DWP's admin-linked status (Y1 of the November event-data plan, microcosm#1095).

policyengine-uk 2.180.0 decides a person's limited capability for work and
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
approximation. María signed these choices on 9 November 2026.

On María's call of 17 November 2026, every other claim takes an imputed
status instead of the proxy: the unlinked reporters, the take-up redraw's
entrants and any other benefit unit with a claimant, outside the SPI rows.
Before it, they kept the proxy, which gives limited capability to every
claimant on a disability benefit and to no one else, while on linked claims
DWP's status disagrees with the proxy in both directions. So the take-up
redraw's entrants carried the LCWRA element only where someone got PIP, DLA or
ESA, and young claimants above all: the Milburn review's benchmarks
(PolicyEngine/uk-milburn-review-dashboard#1). The imputation:

* a claim's cell is its youngest claimant's age band
  (``UC_LIMITED_CAPABILITY_AGE_BANDS``), whether the proxy flags any of its
  claimants and, on María's go of 18 November 2026, whether any claimant's
  long-standing illness limits their day-to-day activities a lot
  (``UC_LIMITED_CAPABILITY_CONDITION``, frs_spine's ADULT.CONDIT carrier). On
  linked FRS 2024-25 claims DWP's LCWRA status is far commoner where such a
  condition is reported, whether or not the proxy flags the claim, so without
  it a drawn claim with no disability benefit took the element at its whole
  group's rate, condition or not;
* the cell's rate is the share of linked claims in it with DWP's LCWRA status,
  at the stage's household weights; a cell with fewer than three distinct
  source households takes its proxy and condition group's rate over every
  age, then its proxy group's;
* a claim draws LCWRA when its identity-keyed uniform falls below the rate,
  and NONE otherwise, and the linked rules then apply with the claim's first
  claimant in place of the UC row's person. Linked claims have too few LCW
  statuses to give a rate, so none is imputed;
* a claim in which a claimant has their own ESA award keeps the proxy, since
  the ESA award carries DWP's determination, and so does every SPI row; no
  later stage writes the proxy's inputs, so their stored value equals the
  formula's.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from microcosm.build.source_manifest import SourceStageSpec
from microcosm.build.stochastic_assignment import stable_identity_uniforms
from microcosm.build.uk_runtime.frs_release import resolve_uk_year_rule
from microcosm.build.uk_runtime.frs_spine import (
    FRS_CONDITION_LIMITS_CODES,
    FRS_CONDITION_LIMITS_COLUMN,
    FRS_CONDITION_NONE,
    FRS_CONDITION_NOT_ASKED,
    FRS_UC_LCW_NOT_LINKED,
    FRS_UC_LCW_STATUS_CODES,
    FRS_UC_LCW_STATUS_COLUMN,
)
from microcosm.build.uk_runtime.frs_take_up import UK_UC_AGE_ELIGIBLE_SOURCE
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
from microcosm.build.uk_runtime.uc_take_up import UC_TAKE_UP_YOUNG_AGES
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
#: A claimant's own ESA award, current or ended, carries DWP's determination,
#: so a claim with one keeps the proxy.
UC_LIMITED_CAPABILITY_ESA = "has_own_esa_award"
#: What the stage asks the engine, for every person, on the frame as it stands.
UC_LIMITED_CAPABILITY_ENGINE_VARIABLES: tuple[str, ...] = (
    UC_LCWRA,
    UC_LCW,
    UC_LIMITED_CAPABILITY_CLAIMANT,
    UC_LIMITED_CAPABILITY_CARER,
    UC_LIMITED_CAPABILITY_ESA,
)
#: The stored survey-year age the cells read.
UC_LIMITED_CAPABILITY_AGE = UK_UC_AGE_ELIGIBLE_SOURCE
#: The imputation's age bands, by a claim's youngest claimant: the take-up
#: redraw's young ages (uc_take_up), then 25 to 49 and 50 and over.
UC_LIMITED_CAPABILITY_AGE_BANDS: tuple[tuple[str, int, int | None], ...] = (
    ("16_to_24", min(UC_TAKE_UP_YOUNG_AGES), max(UC_TAKE_UP_YOUNG_AGES)),
    ("25_to_49", max(UC_TAKE_UP_YOUNG_AGES) + 1, 49),
    ("50_and_over", 50, None),
)
#: The imputation's third cell dimension: whether any of a claim's claimants
#: reports a long-standing illness that limits their day-to-day activities a
#: lot (frs_spine's ADULT.CONDIT carrier, code 1; PolicyEngine/microcosm#1663).
UC_LIMITED_CAPABILITY_CONDITION = FRS_CONDITION_LIMITS_COLUMN
UC_LIMITED_CAPABILITY_CONDITION_CELL = FRS_CONDITION_LIMITS_CODES[1]
_CONDITION_LABELS = (
    *FRS_CONDITION_LIMITS_CODES.values(),
    FRS_CONDITION_NONE,
    FRS_CONDITION_NOT_ASKED,
)
_CONDITION_SIDES = ("limits_a_lot", "no_limit_a_lot")
UC_LIMITED_CAPABILITY_SEED = 0
UC_LIMITED_CAPABILITY_SALT = "frs_uc_lcw_status_imputed"
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
UC_LIMITED_CAPABILITY_IMPUTATION_RULE = (
    "Every other claim outside the SPI rows (an unlinked reporter, a take-up "
    "entrant or any other benefit unit with a claimant) takes an imputed "
    "status: LCWRA when its identity-keyed uniform falls below its cell's "
    "rate, NONE otherwise, and then the linked rules with the claim's first "
    "claimant in place of the UC row's person. A cell is the youngest "
    "claimant's age band, whether the proxy flags any claimant, and whether "
    "any claimant's long-standing illness limits their day-to-day activities "
    "a lot (frs_spine's ADULT.CONDIT carrier); its rate is the share of "
    "linked claims in it with DWP's LCWRA status, at the stage's household "
    "weights, counting only linked claims in which no claimant has their own "
    "ESA award: a claim with one keeps the proxy rather than drawing, so the "
    "rates are measured on the kind of claim that draws. A cell with fewer "
    "than three distinct source households takes its proxy and condition "
    "group's rate over every age, then its proxy group's over every age, and "
    "a group with fewer keeps the proxy."
)
UC_LIMITED_CAPABILITY_SCOPE = (
    "SPI rows, and claims in which a claimant has their own ESA award, keep "
    "the proxy; no later stage writes the proxy's inputs, so their stored "
    "value equals the formula's."
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
#: Imputed claims are LCWRA or NONE, and their lead is the first claimant.
_IMPUTED_BRANCHES = (
    "none_over_a_proxy_flag",
    "none_where_the_proxy_agrees",
    "lcwra_kept_from_the_proxy",
    "lcwra_given_to_a_single_claimant",
    "lcwra_given_to_the_first_claimant",
    "lcwra_given_to_the_partner_of_a_carer",
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
                "esa_award": UC_LIMITED_CAPABILITY_ESA,
                "linked_scope": UC_LIMITED_CAPABILITY_LINKED_SCOPE,
                "age": UC_LIMITED_CAPABILITY_AGE,
                "age_bands": [
                    {"band": name, "low": low, "high": high}
                    for name, low, high in UC_LIMITED_CAPABILITY_AGE_BANDS
                ],
                "condition": UC_LIMITED_CAPABILITY_CONDITION,
                "condition_cell": UC_LIMITED_CAPABILITY_CONDITION_CELL,
                "seed": UC_LIMITED_CAPABILITY_SEED,
                "salt": UC_LIMITED_CAPABILITY_SALT,
                "rule": UC_LIMITED_CAPABILITY_RULE,
                "joint_claim_rule": UC_LIMITED_CAPABILITY_JOINT_CLAIM_RULE,
                "imputation_rule": UC_LIMITED_CAPABILITY_IMPUTATION_RULE,
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

    if UC_LIMITED_CAPABILITY_AGE not in person.columns:
        raise KeyError(
            f"person.{UC_LIMITED_CAPABILITY_AGE} is missing; the imputation's "
            "cells read each claim's youngest claimant."
        )
    own_esa = _bool(
        engine_answers[UC_LIMITED_CAPABILITY_ESA], UC_LIMITED_CAPABILITY_ESA, count
    )
    age = pd.to_numeric(person[UC_LIMITED_CAPABILITY_AGE], errors="coerce").to_numpy(
        dtype=float
    )
    if np.isnan(age).any():
        raise ValueError(f"person.{UC_LIMITED_CAPABILITY_AGE} has missing values.")
    if UC_LIMITED_CAPABILITY_CONDITION not in person.columns:
        raise KeyError(
            f"person.{UC_LIMITED_CAPABILITY_CONDITION} is missing; frs_spine "
            "carries ADULT.CONDIT, which the imputation's cells read."
        )
    condition = (
        person[UC_LIMITED_CAPABILITY_CONDITION].astype(str).to_numpy(dtype=object)
    )
    unknown_conditions = sorted(set(condition) - set(_CONDITION_LABELS))
    if unknown_conditions:
        raise ValueError(
            f"{UC_LIMITED_CAPABILITY_CONDITION} values outside the labels: "
            f"{unknown_conditions}."
        )
    limits_a_lot = condition == UC_LIMITED_CAPABILITY_CONDITION_CELL

    wra = proxy_wra.copy()
    lcw = proxy_lcw.copy()

    def apply_status(
        claim_status: str,
        rows: list[int],
        lead: int,
        lead_label: str,
        log: dict[str, list[int]],
    ) -> None:
        """Give a claim's claimants the flags its status implies."""

        key = lead
        flags = proxy_wra if claim_status in ("NONE", "LCWRA") else proxy_lcw
        flagged = [row for row in rows if flags[row]]
        if claim_status == "NONE":
            if any(proxy_wra[row] or proxy_lcw[row] for row in rows):
                log["none_over_a_proxy_flag"].append(key)
            else:
                log["none_where_the_proxy_agrees"].append(key)
            wra[rows] = False
            lcw[rows] = False
            return
        prefix = claim_status.lower()
        if claim_status == "LCW":
            wra[rows] = False
        if flagged:
            log[f"{prefix}_kept_from_the_proxy"].append(key)
            return
        chosen = lead if lead in rows else rows[0]
        others = [row for row in rows if row != chosen]
        if len(rows) == 1:
            log[f"{prefix}_given_to_a_single_claimant"].append(key)
        elif len(others) == 1 and carer[chosen] and not carer[others[0]]:
            chosen = others[0]
            log[f"{prefix}_given_to_the_partner_of_a_carer"].append(key)
        else:
            log[f"{prefix}_given_to_the_{lead_label}"].append(key)
        lcw[chosen] = True
        wra[chosen] = claim_status == "LCWRA"

    claims: dict[str, list[int]] = {name: [] for name in _BRANCHES}
    agreement = {
        claim_status: {"proxy_flags": [], "proxy_does_not": []}
        for claim_status in _STATUSES
    }
    linked_claims: list[tuple[int, list[int]]] = []
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
        linked_claims.append((carrier_row, rows))
        apply_status(claim_status, rows, carrier_row, "uc_rows_person", claims)

    # The imputation's rates, from the linked claims at the stage's household
    # weights: each claim counts once, at its UC row's person, in the cell of
    # its youngest claimant's age band, of the proxy's view of its claimants and
    # of whether any claimant's condition limits their activities a lot.
    def band_of(rows: list[int]) -> str:
        youngest = float(age[rows].min())
        for name, _low, high in UC_LIMITED_CAPABILITY_AGE_BANDS:
            if high is None or youngest <= high:
                return name
        raise AssertionError("the last age band is open")

    def proxy_side(rows: list[int]) -> str:
        return (
            "proxy_flags" if any(proxy_wra[row] for row in rows) else "proxy_does_not"
        )

    def condition_side(rows: list[int]) -> str:
        return "limits_a_lot" if limits_a_lot[rows].any() else "no_limit_a_lot"

    sides = ("proxy_flags", "proxy_does_not")
    conditions = _CONDITION_SIDES
    bands = tuple(name for name, _low, _high in UC_LIMITED_CAPABILITY_AGE_BANDS)
    cell_rows: dict[tuple[str, str, str], list[int]] = {
        (band, side, cond): []
        for band in bands
        for side in sides
        for cond in conditions
    }
    # A linked claim in which a claimant has their own ESA award stays out of
    # the rates: a claim like it keeps the proxy instead of drawing, so a rate
    # measured on it would speak for claims that never draw from it. The
    # receipt gives these claims' own rate beside each cell's, to show the gap.
    own_esa_rows: dict[tuple[str, str, str], list[int]] = {
        (band, side, cond): []
        for band in bands
        for side in sides
        for cond in conditions
    }
    for carrier_row, rows in linked_claims:
        key = (band_of(rows), proxy_side(rows), condition_side(rows))
        if own_esa[rows].any():
            own_esa_rows[key].append(carrier_row)
        else:
            cell_rows[key].append(carrier_row)
    is_lcwra = status == "LCWRA"

    def rows_mask(rows: list[int]) -> np.ndarray:
        mask = np.zeros(count, dtype=bool)
        mask[rows] = True
        return mask

    def floor_holds(rows: list[int]) -> bool:
        # The rule a printed rate meets: its LCWRA claims and the rest each
        # rest on three source households or more.
        mask = rows_mask(rows)
        return (
            distinct_count(sources, mask & is_lcwra, weights) >= UK_DISCLOSURE_FLOOR
            and distinct_count(sources, mask & ~is_lcwra, weights)
            >= UK_DISCLOSURE_FLOOR
        )

    def measured_rate(rows: list[int]) -> tuple[float | None, dict[str, object]]:
        mask = rows_mask(rows)
        support = distinct_count(sources, mask, weights)
        total = float(weights[mask].sum())
        rate = float(weights[mask & is_lcwra].sum()) / total if total > 0 else None
        usable = support >= UK_DISCLOSURE_FLOOR and rate is not None
        # The rate prints only where its LCWRA claims and the rest each rest on
        # three source households or more, so neither part comes back from it.
        printable = usable and floor_holds(rows)
        entry: dict[str, object] = {
            "linked_claims": rounded_count(len(rows)),
            "support": support_band(support),
            "rate": round(rate, 3) if printable else "withheld",
        }
        return (rate if usable else None), entry

    def over_every_age(
        by_cell: dict[tuple[str, str, str], list[int]],
        side: str,
        cond: str | None = None,
    ) -> list[int]:
        return [
            row
            for band in bands
            for each in conditions
            if cond is None or each == cond
            for row in by_cell[(band, side, each)]
        ]

    pooled = {side: measured_rate(over_every_age(cell_rows, side)) for side in sides}
    for side in sides:
        pooled[side][1]["own_esa_award_claims"] = measured_rate(
            over_every_age(own_esa_rows, side)
        )[1]
    pooled_by_condition = {
        (side, cond): measured_rate(over_every_age(cell_rows, side, cond))
        for side in sides
        for cond in conditions
    }
    for (side, cond), (_rate, entry) in pooled_by_condition.items():
        entry["own_esa_award_claims"] = measured_rate(
            over_every_age(own_esa_rows, side, cond)
        )[1]
    rates: dict[tuple[str, str, str], float | None] = {}
    cells: dict[str, dict[str, dict[str, object]]] = {}
    for band in bands:
        cells[band] = {}
        for side in sides:
            cells[band][side] = {}
            for cond in conditions:
                rate, entry = measured_rate(cell_rows[(band, side, cond)])
                if rate is not None:
                    entry["source"] = "cell"
                else:
                    rate = pooled_by_condition[(side, cond)][0]
                    if rate is not None:
                        entry["source"] = "pooled_by_condition"
                    else:
                        rate = pooled[side][0]
                        entry["source"] = "pooled" if rate is not None else "proxy_kept"
                entry["own_esa_award_claims"] = measured_rate(
                    own_esa_rows[(band, side, cond)]
                )[1]
                rates[(band, side, cond)] = rate
                cells[band][side][cond] = entry

    # A group's rate printed beside the cells inside it gives back what the
    # withheld ones hold: with the other cells' rates, it tells which way a
    # one-household cell went (Vahid's review of PolicyEngine/microcosm#1663).
    # So a group prints only where the claims of its withheld cells, together,
    # pass the floor a cell's rate does, and each proxy group over every age
    # only where those of its withheld condition groups do. The draws keep
    # every rate; only the receipt withholds.
    def withhold_by_difference(
        by_cell: dict[tuple[str, str, str], list[int]],
        cell_entry,
        group_entry,
        side_entry,
    ) -> None:
        for side in sides:
            behind_side: list[int] = []
            for cond in conditions:
                hidden = [
                    row
                    for band in bands
                    if cell_entry(band, side, cond)["rate"] == "withheld"
                    for row in by_cell[(band, side, cond)]
                ]
                group = group_entry(side, cond)
                if hidden and not floor_holds(hidden):
                    group["rate"] = "withheld"
                if group["rate"] == "withheld":
                    behind_side.extend(hidden)
            if behind_side and not floor_holds(behind_side):
                side_entry(side)["rate"] = "withheld"

    withhold_by_difference(
        cell_rows,
        lambda band, side, cond: cells[band][side][cond],
        lambda side, cond: pooled_by_condition[(side, cond)][1],
        lambda side: pooled[side][1],
    )
    withhold_by_difference(
        own_esa_rows,
        lambda band, side, cond: cells[band][side][cond]["own_esa_award_claims"],
        lambda side, cond: pooled_by_condition[(side, cond)][1]["own_esa_award_claims"],
        lambda side: pooled[side][1]["own_esa_award_claims"],
    )

    # The imputed claims: every other benefit unit with a claimant outside the
    # SPI rows whose claimants hold no ESA award of their own, in benefit unit
    # order, each with its identity-keyed uniform.
    linked_units = {benunits[carrier_row] for carrier_row, _rows in linked_claims}
    linked_units |= {
        benunits[row] for row in claims["linked_claim_with_no_claimant_kept_the_proxy"]
    }
    imputed: dict[str, list[int]] = {name: [] for name in _IMPUTED_BRANCHES}
    kept: dict[str, list[int]] = {"own_esa_award": [], "spi": [], "no_rate": []}
    statuses: dict[str, list[int]] = {"LCWRA": [], "NONE": []}
    candidates = []
    for unit in sorted(members):
        if unit in linked_units:
            continue
        rows = [row for row in members[unit] if claimant[row]]
        if not rows:
            continue
        if spi[rows[0]]:
            kept["spi"].append(rows[0])
        elif own_esa[rows].any():
            kept["own_esa_award"].append(rows[0])
        else:
            candidates.append((unit, rows))
    uniforms = stable_identity_uniforms(
        [unit for unit, _rows in candidates],
        seed=UC_LIMITED_CAPABILITY_SEED,
        salt=UC_LIMITED_CAPABILITY_SALT,
    )
    for (_unit, rows), uniform in zip(candidates, uniforms, strict=True):
        rate = rates[(band_of(rows), proxy_side(rows), condition_side(rows))]
        if rate is None:
            kept["no_rate"].append(rows[0])
            continue
        claim_status = "LCWRA" if uniform < rate else "NONE"
        statuses[claim_status].append(rows[0])
        apply_status(claim_status, rows, rows[0], "first_claimant", imputed)

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

    status_cells = {
        claim_status: {side: cell(rows) for side, rows in by_side.items()}
        for claim_status, by_side in agreement.items()
    }
    # A cell under three source households is withheld, and so is its
    # status's other cell, so no total gives it back (UKDS disclosure control).
    for by_side in status_cells.values():
        if any(entry["weighted"] is None for entry in by_side.values()):
            for entry in by_side.values():
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
        "status_against_the_proxy": status_cells,
        "branches": {name: rounded_count(len(rows)) for name, rows in claims.items()},
        "imputation": {
            "rule": UC_LIMITED_CAPABILITY_IMPUTATION_RULE,
            "cells": cells,
            "pooled": {side: entry for side, (_rate, entry) in pooled.items()},
            "pooled_by_condition": {
                side: {
                    cond: pooled_by_condition[(side, cond)][1] for cond in conditions
                }
                for side in sides
            },
            "claims": rounded_count(sum(len(rows) for rows in statuses.values())),
            "claims_support": support_band(
                distinct_count(
                    sources,
                    rows_mask([row for rows in statuses.values() for row in rows]),
                    weights,
                )
            ),
            "statuses": {
                name: rounded_count(len(rows)) for name, rows in statuses.items()
            },
            "branches": {
                name: rounded_count(len(rows)) for name, rows in imputed.items()
            },
            "kept_the_proxy": {
                name: rounded_count(len(rows)) for name, rows in kept.items()
            },
        },
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
