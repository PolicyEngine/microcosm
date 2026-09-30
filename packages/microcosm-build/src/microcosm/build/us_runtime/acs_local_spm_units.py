"""Census SPM units for adult nonrelatives on ACS rows (the ACS local lane).

The ACS PUMS carries no SPM unit id, so :func:`microcosm.frame.units.
assign_us_unit_structure` takes its documented household fallback and every
ACS housing unit becomes exactly one SPM unit. That puts every adult roommate
and other nonrelative in the reference person's SPM unit: on release
``767312d60`` about 2.22M weighted people aged 15 or over with ``RELSHIPP`` 34
or 36, in 1.62M households (microcosm#1023). The Census SPM resource unit is
all related people at the address, any co-resident unrelated children cared
for by the family (such as foster children), and any cohabiting partners and
their children. Roommates and other unrelated adults are not in it.

This fresh-build stage runs on the ACS-only loader frame, before the native
mapping and every transfer, so each downstream SPM-unit input (tenure through
SPM membership, the SPM-unit QRF targets, the runtime take-up draws) is built
for the new units rather than patched onto them. The pool lane calls the
loader directly and never reaches this stage; only
``build_optional_acs_multispine(split_adult_nonrelative_spm_units=True)``,
which the ACS local-release staging builder passes, runs it.

The rule
--------

- **Roommates move.** A person with ``RELSHIPP`` 34 (roommate or housemate)
  aged 15 or over (``AGEP``) leaves the reference person's unit and forms an
  SPM unit of their own.
- **Other nonrelatives move only when no partner lives there.** A person
  with ``RELSHIPP`` 36 (other nonrelative) aged 15 or over forms an SPM unit
  of their own only when the household holds no unmarried partner
  (``RELSHIPP`` 22 or 24). The ACS records relationship to the householder
  only, so an unmarried partner's child or other relative is coded 36
  (Census ACS brief ACSBR-005), and the Census SPM unit includes cohabitors
  and their relatives. With a partner present, a 36 of any age therefore
  stays with the reference person and the partner (microcosm#1061 review).
- **Kept with the reference person.** Every relative (``RELSHIPP`` 21, 23,
  25-33), unmarried partners (22, 24; cohabitors are in the Census unit),
  foster children (35; cared for by the family), nonrelatives under 15, and
  other nonrelatives (36) in a household with an unmarried partner.
- **One unit per mover.** ``RELSHIPP`` relates each person to the reference
  person only, so a couple, or a parent and child, among the nonrelatives is
  not identifiable. Grouping every adult nonrelative of a household into one
  unit (the alternative the diagnostic also measured) would pool unrelated
  roommates, which the Census definition never does.
- **The remaining ambiguity is counted, not guessed.** A 36 moved out of a
  household that also holds a roommate (34) may be the roommate's own child
  or relative, which ``RELSHIPP`` cannot link; they keep a unit of their own.
  The receipt counts the 36s kept for a partner, the 36s moved, and the 36s
  moved from a household with a roommate, and carries a sensitivity block:
  the weighted persons per unit, single-person share and units per household
  under the rule, under "34 only" and under "34 and every 36", all on the
  same frame.
- **Nonrelatives under 15 stay.** A roommate's child is not identifiable
  either (the loader gives nonrelatives no parent pointer), and microunit
  claims a nonrelative under 15 as a dependent of the reference person's tax
  unit, so keeping the child there keeps the tax unit inside one SPM unit.
  Attaching the child to the household's only adult nonrelative, when there
  is exactly one, is the documented alternative.
- **Group quarters** are one person per record already and never move.

Tax units stay nested: a moved person is 15 or over with no parent pointer,
so microunit neither claims them nor lets them claim anyone, and their tax
unit is theirs alone. The stage verifies this and refuses otherwise. Families
(the household, since the ACS has no ``PF_SEQ``) and marital units (the
reference couple, everyone else alone) are unchanged.

Unit ids are re-densified over ``(household, SPORDER of a moved person or 0)``
in sorted order, so they are unique, content-determined, and identical to the
loader's household ids wherever no one moves.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_ADULT_NONRELATIVE_MIN_AGE",
    "ACS_ADULT_NONRELATIVE_RELATIONSHIPS",
    "ACS_LOCAL_SPM_UNIT_GATE_NAME",
    "ACS_LOCAL_SPM_UNIT_ISSUE",
    "ACS_LOCAL_SPM_UNIT_METHOD",
    "ACS_OTHER_NONRELATIVE_RELATIONSHIP",
    "ACS_ROOMMATE_RELATIONSHIP",
    "ACS_UNMARRIED_PARTNER_RELATIONSHIPS",
    "acs_adult_nonrelative_mask",
    "acs_local_spm_unit_signal_gate",
    "acs_spm_unit_mover_mask",
    "split_acs_adult_nonrelative_spm_units",
]

ACS_LOCAL_SPM_UNIT_ISSUE = "microcosm#1023"
ACS_LOCAL_SPM_UNIT_GATE_NAME = "acs_local_spm_unit_signal"
ACS_LOCAL_SPM_UNIT_METHOD = "roommates_and_unpartnered_other_nonrelatives_own_spm_unit"
#: ACS ``RELSHIPP`` codes that can leave the reference person's SPM unit at
#: 15 or over: 34 roommate or housemate, 36 other nonrelative.
ACS_ADULT_NONRELATIVE_RELATIONSHIPS: tuple[int, ...] = (34, 36)
ACS_ROOMMATE_RELATIONSHIP = 34
ACS_OTHER_NONRELATIVE_RELATIONSHIP = 36
#: Opposite-sex (22) and same-sex (24) unmarried partners. A 36 in a household
#: with one may be the partner's child or relative, whom the Census SPM unit
#: keeps with the cohabiting couple (microcosm#1061 review).
ACS_UNMARRIED_PARTNER_RELATIONSHIPS: tuple[int, ...] = (22, 24)
#: The Census SPM unit keeps co-resident unrelated children, and the ACS asks
#: its adult items from 15; below it a nonrelative stays with the family.
ACS_ADULT_NONRELATIVE_MIN_AGE = 15

_REFERENCE_PERSON = 20
_RELATIONSHIP = "RELSHIPP"
_AGE = "AGEP"
_LINE = "SPORDER"
_PERSON_ID = "person_id"
_HOUSEHOLD_MEMBERSHIP = "person_household_id"
_SPM_MEMBERSHIP = "person_spm_unit_id"
_TAX_MEMBERSHIP = "person_tax_unit_id"
_SPM_ID = "spm_unit_id"
_KEPT_WITH_REFERENCE = (
    "relatives (RELSHIPP 21, 23, 25-33), unmarried partners (22, 24), foster "
    "children (35), nonrelatives under 15, and other nonrelatives (36) in a "
    "household with an unmarried partner"
)
_MOVED = (
    "roommates and housemates (RELSHIPP 34) aged 15 or over, and other "
    "nonrelatives (36) aged 15 or over in a household with no unmarried "
    "partner (22, 24)"
)
_AMBIGUITY = (
    "A 36 moved from a household that also holds a roommate (34) may be the "
    "roommate's own child or relative; RELSHIPP cannot link them, so they keep "
    "a unit of their own and are counted here."
)


def acs_adult_nonrelative_mask(person: pd.DataFrame) -> np.ndarray:
    """Whether each row is an ACS roommate or other nonrelative aged 15+.

    These are the candidates for a unit of their own; not all of them move
    (:func:`acs_spm_unit_mover_mask` is the rule). Raises ``ValueError`` when
    ``RELSHIPP`` or ``AGEP`` is absent or blank on any row: the rule reads
    both on every ACS person.
    """

    relationship = _complete_integers(person, _RELATIONSHIP)
    age = _complete_integers(person, _AGE)
    return np.isin(relationship, ACS_ADULT_NONRELATIVE_RELATIONSHIPS) & (
        age >= ACS_ADULT_NONRELATIVE_MIN_AGE
    )


def acs_spm_unit_mover_mask(person: pd.DataFrame) -> np.ndarray:
    """Whether each ACS row leaves the reference person's SPM unit.

    True for a roommate (``RELSHIPP`` 34) aged 15 or over, and for an other
    nonrelative (36) aged 15 or over whose household holds no unmarried
    partner (22, 24). Reads ``RELSHIPP``, ``AGEP`` and the household
    membership; raises ``ValueError`` when any is absent or blank.
    """

    return _rule_masks(person)["moved"]


def _rule_masks(person: pd.DataFrame) -> dict[str, np.ndarray]:
    """The rule's person masks, from ``RELSHIPP``, ``AGEP`` and the household."""

    relationship = _complete_integers(person, _RELATIONSHIP)
    age = _complete_integers(person, _AGE)
    if _HOUSEHOLD_MEMBERSHIP not in person:
        raise ValueError(f"ACS column {_HOUSEHOLD_MEMBERSHIP!r} is absent.")
    household = person[_HOUSEHOLD_MEMBERSHIP].to_numpy()
    adult = age >= ACS_ADULT_NONRELATIVE_MIN_AGE
    roommate = adult & (relationship == ACS_ROOMMATE_RELATIONSHIP)
    other = adult & (relationship == ACS_OTHER_NONRELATIVE_RELATIONSHIP)
    with_partner = np.isin(
        household,
        household[np.isin(relationship, ACS_UNMARRIED_PARTNER_RELATIONSHIPS)],
    )
    with_roommate = np.isin(
        household, household[relationship == ACS_ROOMMATE_RELATIONSHIP]
    )
    other_moved = other & ~with_partner
    return {
        "relationship": relationship,
        "age": age,
        "household": household,
        "moved": roommate | other_moved,
        "roommate": roommate,
        "other": other,
        "other_kept_with_partner": other & with_partner,
        "other_moved": other_moved,
        "other_moved_with_roommate": other_moved & with_roommate,
        "roommates_and_every_other": roommate | other,
        "under_age_nonrelative": np.isin(
            relationship, ACS_ADULT_NONRELATIVE_RELATIONSHIPS
        )
        & ~adult,
    }


def _partition(
    household: np.ndarray, line: np.ndarray, moved: np.ndarray
) -> np.ndarray:
    """Dense sorted SPM ids over ``(household, SPORDER of a mover or 0)``."""

    member = np.where(moved, line, 0)
    codes, _ = pd.factorize(pd.MultiIndex.from_arrays([household, member]), sort=True)
    return (codes + 1).astype(np.int64)


def _rule_summary(
    unit: np.ndarray, weights: np.ndarray, household_weight: float
) -> dict[str, float]:
    summary = _unit_summary(unit, weights)
    return {
        "persons_per_spm_unit": summary["persons_per_unit"],
        "single_person_spm_unit_share": summary["single_person_share"],
        "spm_units_per_household": _ratio(summary["unit_weight"], household_weight),
    }


def split_acs_adult_nonrelative_spm_units(
    frame: Frame,
) -> tuple[Frame, dict[str, Any]]:
    """Give each ACS adult nonrelative an SPM unit of their own.

    ``frame`` must be the ACS loader frame: SPM units are still the
    household fallback and the ``spm_unit`` table carries only its ids, so no
    SPM-unit input exists yet that would need re-keying. Returns the frame
    with the new partition and a JSON-ready receipt.

    Raises:
        ValueError: If the frame is not the US schema, lacks a required
            column, is not on the household fallback, carries SPM-unit
            columns, or the split would leave a tax unit across two SPM
            units.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("The ACS SPM-unit split requires the US schema.")
    person = frame.table("person")
    required = (
        _RELATIONSHIP,
        _AGE,
        _LINE,
        _HOUSEHOLD_MEMBERSHIP,
        _SPM_MEMBERSHIP,
        _TAX_MEMBERSHIP,
    )
    missing = [column for column in required if column not in person]
    if missing:
        raise ValueError(f"The ACS SPM-unit split requires person column(s) {missing}.")
    spm_unit = frame.table("spm_unit")
    extra = sorted(set(spm_unit.columns) - {_SPM_ID})
    if extra:
        raise ValueError(
            "The ACS SPM-unit split runs on the loader frame, before any "
            f"SPM-unit input is mapped; the spm_unit table carries {extra}."
        )
    household = person[_HOUSEHOLD_MEMBERSHIP].to_numpy()
    before = person[_SPM_MEMBERSHIP].to_numpy()
    if not _one_to_one(household, before):
        raise ValueError(
            "The ACS SPM-unit split expects the loader's household fallback "
            "(one SPM unit per household); this frame's SPM units are "
            "already split."
        )

    masks = _rule_masks(person)
    relationship, moved = masks["relationship"], masks["moved"]
    line = _complete_integers(person, _LINE)
    if (line < 1).any():
        raise ValueError("ACS SPORDER must be a positive person number.")
    after = _partition(household, line, moved)

    tax_unit = person[_TAX_MEMBERSHIP].to_numpy()
    straddling = pd.Series(after).groupby(tax_unit, sort=True).nunique() > 1
    if straddling.any():
        examples = straddling.index[straddling.to_numpy()][:5].tolist()
        raise ValueError(
            f"{int(straddling.sum())} ACS tax unit(s) would span two SPM units "
            f"after the adult-nonrelative split: {examples}. A moved person "
            "must be alone in their tax unit."
        )

    tables = {entity: frame.table(entity) for entity in frame.entities}
    updated = person.copy()
    updated[_SPM_MEMBERSHIP] = after
    tables["person"] = updated
    tables["spm_unit"] = pd.DataFrame({_SPM_ID: np.unique(after)})
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )

    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)
    affected = np.unique(household[moved])
    households_before = _unit_summary(before, weights)
    units_after = _unit_summary(after, weights)
    household_weight = households_before["unit_weight"]
    other = {
        name: masks[f"other_{name}"]
        for name in ("kept_with_partner", "moved", "moved_with_roommate")
    }
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_SPM_UNIT_ISSUE,
        "method": ACS_LOCAL_SPM_UNIT_METHOD,
        "moved_relationship_codes": list(ACS_ADULT_NONRELATIVE_RELATIONSHIPS),
        "partner_relationship_codes": list(ACS_UNMARRIED_PARTNER_RELATIONSHIPS),
        "min_age": ACS_ADULT_NONRELATIVE_MIN_AGE,
        "moved_rule": _MOVED,
        "kept_with_reference_person": _KEPT_WITH_REFERENCE,
        "acs_households": int(len(np.unique(household))),
        "households_affected": int(len(affected)),
        "persons_moved": int(moved.sum()),
        "persons_moved_by_relationship": {
            str(code): int((moved & (relationship == code)).sum())
            for code in ACS_ADULT_NONRELATIVE_RELATIONSHIPS
        },
        "nonrelatives_under_min_age_kept": int(masks["under_age_nonrelative"].sum()),
        # microcosm#1061 review: the code-36 ambiguity, sized.
        "other_nonrelatives": {
            "kept_with_partner": int(other["kept_with_partner"].sum()),
            "moved": int(other["moved"].sum()),
            "moved_from_roommate_households": int(other["moved_with_roommate"].sum()),
            "weighted": {
                "kept_with_partner": float(weights[other["kept_with_partner"]].sum()),
                "moved": float(weights[other["moved"]].sum()),
                "moved_from_roommate_households": float(
                    weights[other["moved_with_roommate"]].sum()
                ),
            },
            "ambiguity": _AMBIGUITY,
        },
        "sensitivity": {
            "basis": (
                "weighted ACS persons per SPM unit, single-person unit share and "
                "SPM units per household on this frame under each rule"
            ),
            "rule": _rule_summary(after, weights, household_weight),
            "roommates_only": _rule_summary(
                _partition(household, line, masks["roommate"]),
                weights,
                household_weight,
            ),
            "roommates_and_every_other_nonrelative": _rule_summary(
                _partition(household, line, masks["roommates_and_every_other"]),
                weights,
                household_weight,
            ),
        },
        "units_created": int(len(np.unique(after)) - len(np.unique(before))),
        "spm_units_before": int(len(np.unique(before))),
        "spm_units_after": int(len(np.unique(after))),
        "weighted": {
            "households_affected": _first_weight_sum(
                household, weights, np.isin(household, affected)
            ),
            "persons_moved": float(weights[moved].sum()),
            "persons_per_spm_unit": {
                "before": households_before["persons_per_unit"],
                "after": units_after["persons_per_unit"],
            },
            "single_person_spm_unit_share": {
                "before": households_before["single_person_share"],
                "after": units_after["single_person_share"],
            },
            "spm_units_per_household": {
                "before": 1.0,
                "after": _ratio(
                    units_after["unit_weight"], households_before["unit_weight"]
                ),
            },
        },
    }
    return result, receipt


def acs_local_spm_unit_signal_gate(
    frame: Frame,
    *,
    receipt: Mapping[str, Any] | None,
) -> GateResult:
    """Require the Census adult-nonrelative SPM partition on the ACS rows.

    The expectation is the rule, read clause by clause from ``RELSHIPP``,
    ``AGEP`` and the household. Fails when an ACS SPM unit holds a mover (a
    roommate, 34, aged 15 or over, or an other nonrelative, 36, aged 15 or
    over with no unmarried partner, 22/24, in the household) together with
    the reference person; when a 36 aged 15 or over in a household with an
    unmarried partner is outside the reference person's unit; when the ACS
    partition is not exactly the rule (each mover alone, everyone else in the
    household's reference unit); when an ACS SPM unit holds a non-ACS row;
    when an ACS tax unit spans two SPM units; or when ``receipt`` (the
    staging receipt) is missing, from another method, or moved or kept a
    different number of people than the frame shows. Details report the
    weighted ACS persons per SPM unit, single-person share and units per
    household, and the code-36 counts.
    """

    person = frame.table("person")
    tag = spine_column("person")
    failures: list[str] = []
    details: dict[str, object] = {}
    if tag not in person or person[tag].isna().any():
        return _gate([f"Missing person origin tags: {tag}."], details)
    acs = person[tag].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    if not acs.any():
        return _gate([f"{ACS_2024_1YR_SPINE}: no person rows."], details)
    required = (
        _RELATIONSHIP,
        _AGE,
        _PERSON_ID,
        _HOUSEHOLD_MEMBERSHIP,
        _SPM_MEMBERSHIP,
        _TAX_MEMBERSHIP,
    )
    missing = [column for column in required if column not in person]
    if missing:
        return _gate(
            [f"{ACS_2024_1YR_SPINE}: person column(s) {missing} are absent."],
            details,
        )
    rows = person.loc[acs]
    try:
        masks = _rule_masks(rows)
    except ValueError as exc:
        return _gate([f"{ACS_2024_1YR_SPINE}: {exc}"], details)
    relationship, moved = masks["relationship"], masks["moved"]
    kept = masks["other_kept_with_partner"]

    spm = rows[_SPM_MEMBERSHIP].to_numpy()
    household = rows[_HOUSEHOLD_MEMBERSHIP].to_numpy()
    reference_units = pd.unique(spm[relationship == _REFERENCE_PERSON])
    shared = int((moved & np.isin(spm, reference_units)).sum())
    if shared:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: {shared} adult roommate(s) or unpartnered "
            "other nonrelative(s) (RELSHIPP 34, or 36 with no 22/24 in the "
            "household, 15 or over) share an SPM unit with the reference person."
        )
    split_from_partner = int((kept & ~np.isin(spm, reference_units)).sum())
    if split_from_partner:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: {split_from_partner} other nonrelative(s) "
            "(RELSHIPP 36, 15 or over) in a household with an unmarried partner "
            "(22/24) are outside the reference person's SPM unit; the Census "
            "unit keeps cohabitors' children and relatives."
        )
    member = np.where(moved, rows[_PERSON_ID].to_numpy(), -1)
    expected = len(pd.MultiIndex.from_arrays([household, member]).unique())
    actual = len(pd.unique(spm))
    pairings = len(pd.MultiIndex.from_arrays([household, member, spm]).unique())
    if not expected == actual == pairings:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: the SPM partition is not one unit per mover "
            f"plus one per household ({expected} expected unit(s), "
            f"{actual} SPM unit(s), {pairings} distinct pairing(s))."
        )
    foreign = int(np.isin(spm, person.loc[~acs, _SPM_MEMBERSHIP].to_numpy()).sum())
    if foreign:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: {foreign} person row(s) share an SPM unit "
            "with a row from another spine."
        )
    straddling = pd.Series(spm).groupby(rows[_TAX_MEMBERSHIP].to_numpy()).nunique() > 1
    if straddling.any():
        failures.append(
            f"{ACS_2024_1YR_SPINE}: {int(straddling.sum())} tax unit(s) span "
            "more than one SPM unit."
        )
    failures += _receipt_failures(
        receipt, persons_moved=int(moved.sum()), kept_with_partner=int(kept.sum())
    )

    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)[acs]
    units = _unit_summary(spm, weights)
    households = _unit_summary(household, weights)
    details.update(
        {
            "acs_person_rows": int(acs.sum()),
            "adult_nonrelatives": int(masks["roommates_and_every_other"].sum()),
            "movers": int(moved.sum()),
            "other_nonrelatives_kept_with_partner": int(kept.sum()),
            "other_nonrelatives_moved": int(masks["other_moved"].sum()),
            "other_nonrelatives_moved_from_roommate_households": int(
                masks["other_moved_with_roommate"].sum()
            ),
            "acs_spm_units": actual,
            "acs_households": int(len(pd.unique(household))),
            "weighted_persons_per_spm_unit": units["persons_per_unit"],
            "weighted_single_person_spm_unit_share": units["single_person_share"],
            "weighted_spm_units_per_household": _ratio(
                units["unit_weight"], households["unit_weight"]
            ),
        }
    )
    return _gate(failures, details)


def _receipt_failures(
    receipt: Mapping[str, Any] | None, *, persons_moved: int, kept_with_partner: int
) -> list[str]:
    if not isinstance(receipt, Mapping):
        return ["The staging summary carries no ACS SPM-unit receipt."]
    failures: list[str] = []
    if receipt.get("issue") != ACS_LOCAL_SPM_UNIT_ISSUE:
        failures.append(
            f"The ACS SPM-unit receipt is not {ACS_LOCAL_SPM_UNIT_ISSUE}'s."
        )
    if receipt.get("method") != ACS_LOCAL_SPM_UNIT_METHOD:
        failures.append(
            "The ACS SPM-unit receipt records method "
            f"{receipt.get('method')!r}, not {ACS_LOCAL_SPM_UNIT_METHOD!r}."
        )
    recorded = receipt.get("persons_moved")
    created = receipt.get("units_created")
    if type(recorded) is not int or type(created) is not int:
        failures.append("The ACS SPM-unit receipt's counts are not integers.")
    elif recorded != persons_moved or created != recorded:
        failures.append(
            f"The ACS SPM-unit receipt moved {recorded} person(s) into "
            f"{created} new unit(s); the frame holds {persons_moved} mover(s), "
            "each needing one."
        )
    other = receipt.get("other_nonrelatives")
    recorded_kept = (
        other.get("kept_with_partner") if isinstance(other, Mapping) else None
    )
    if type(recorded_kept) is not int or recorded_kept != kept_with_partner:
        failures.append(
            f"The ACS SPM-unit receipt keeps {recorded_kept!r} other "
            "nonrelative(s) with an unmarried partner; the frame holds "
            f"{kept_with_partner}."
        )
    return failures


def _gate(failures: list[str], details: dict[str, object]) -> GateResult:
    return GateResult(
        name=ACS_LOCAL_SPM_UNIT_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )


def _complete_integers(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"ACS column {column!r} is absent.")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    blank = int(np.isnan(values).sum())
    if blank:
        raise ValueError(f"ACS column {column!r} is blank on {blank} row(s).")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError(f"ACS column {column!r} carries non-integers.")
    return values.astype(np.int64)


def _one_to_one(first: np.ndarray, second: np.ndarray) -> bool:
    pairs = len(pd.MultiIndex.from_arrays([first, second]).unique())
    return pairs == len(pd.unique(first)) == len(pd.unique(second))


def _unit_summary(unit: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    """Weighted persons per unit and single-person share over a partition.

    A unit's weight is its members' (common) weight; its size is the member
    count, so persons per unit is the weighted mean size.
    """

    grouped = pd.DataFrame({"unit": unit, "weight": weights}).groupby(
        "unit", sort=True
    )["weight"]
    size = grouped.size().to_numpy(dtype=np.float64)
    unit_weight = grouped.first().to_numpy(dtype=np.float64)
    total = float(unit_weight.sum())
    return {
        "unit_weight": total,
        "persons_per_unit": _ratio(float((size * unit_weight).sum()), total),
        "single_person_share": _ratio(float(unit_weight[size == 1].sum()), total),
    }


def _first_weight_sum(
    unit: np.ndarray, weights: np.ndarray, selected: np.ndarray
) -> float:
    """Sum of one weight per selected unit (a household's weight, once)."""

    frame = pd.DataFrame({"unit": unit[selected], "weight": weights[selected]})
    return float(frame.groupby("unit", sort=True)["weight"].first().sum())


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0
