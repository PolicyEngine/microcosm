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

- **Moved.** A person with ``RELSHIPP`` 34 (roommate or housemate) or 36
  (other nonrelative) aged 15 or over (``AGEP``) leaves the reference person's
  unit and forms an SPM unit of their own.
- **Kept with the reference person.** Every relative (``RELSHIPP`` 21, 23,
  25-33), unmarried partners (22, 24; cohabitors are in the Census unit),
  foster children (35; cared for by the family) and nonrelatives under 15.
- **One unit per adult nonrelative.** ``RELSHIPP`` relates each person to the
  reference person only, so a couple, or a parent and child, among the
  nonrelatives is not identifiable. Grouping every adult nonrelative of a
  household into one unit (the alternative the diagnostic also measured)
  would pool unrelated roommates, which the Census definition never does.
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
    "acs_adult_nonrelative_mask",
    "acs_local_spm_unit_signal_gate",
    "split_acs_adult_nonrelative_spm_units",
]

ACS_LOCAL_SPM_UNIT_ISSUE = "microcosm#1023"
ACS_LOCAL_SPM_UNIT_GATE_NAME = "acs_local_spm_unit_signal"
ACS_LOCAL_SPM_UNIT_METHOD = "adult_nonrelative_own_spm_unit"
#: ACS ``RELSHIPP`` codes outside the Census SPM unit when 15 or over:
#: 34 roommate or housemate, 36 other nonrelative.
ACS_ADULT_NONRELATIVE_RELATIONSHIPS: tuple[int, ...] = (34, 36)
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
    "children (35) and nonrelatives under 15"
)


def acs_adult_nonrelative_mask(person: pd.DataFrame) -> np.ndarray:
    """Whether each row is an ACS roommate or other nonrelative aged 15+.

    Raises ``ValueError`` when ``RELSHIPP`` or ``AGEP`` is absent or blank on
    any row: the rule reads both on every ACS person.
    """

    relationship = _complete_integers(person, _RELATIONSHIP)
    age = _complete_integers(person, _AGE)
    return np.isin(relationship, ACS_ADULT_NONRELATIVE_RELATIONSHIPS) & (
        age >= ACS_ADULT_NONRELATIVE_MIN_AGE
    )


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

    relationship = _complete_integers(person, _RELATIONSHIP)
    age = _complete_integers(person, _AGE)
    moved = np.isin(relationship, ACS_ADULT_NONRELATIVE_RELATIONSHIPS) & (
        age >= ACS_ADULT_NONRELATIVE_MIN_AGE
    )
    line = _complete_integers(person, _LINE)
    if (line < 1).any():
        raise ValueError("ACS SPORDER must be a positive person number.")
    member = np.where(moved, line, 0)
    codes, _ = pd.factorize(pd.MultiIndex.from_arrays([household, member]), sort=True)
    after = (codes + 1).astype(np.int64)

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
    under_age_nonrelatives = np.isin(
        relationship, ACS_ADULT_NONRELATIVE_RELATIONSHIPS
    ) & (age < ACS_ADULT_NONRELATIVE_MIN_AGE)
    households_before = _unit_summary(before, weights)
    units_after = _unit_summary(after, weights)
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_SPM_UNIT_ISSUE,
        "method": ACS_LOCAL_SPM_UNIT_METHOD,
        "moved_relationship_codes": list(ACS_ADULT_NONRELATIVE_RELATIONSHIPS),
        "min_age": ACS_ADULT_NONRELATIVE_MIN_AGE,
        "kept_with_reference_person": _KEPT_WITH_REFERENCE,
        "acs_households": int(len(np.unique(household))),
        "households_affected": int(len(affected)),
        "persons_moved": int(moved.sum()),
        "persons_moved_by_relationship": {
            str(code): int((moved & (relationship == code)).sum())
            for code in ACS_ADULT_NONRELATIVE_RELATIONSHIPS
        },
        "nonrelatives_under_min_age_kept": int(under_age_nonrelatives.sum()),
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

    Fails when an ACS SPM unit holds an adult nonrelative (``RELSHIPP`` 34/36,
    15 or over) together with the reference person; when the ACS partition is
    not exactly the rule (each adult nonrelative alone, everyone else in the
    household's reference unit); when an ACS SPM unit holds a non-ACS row;
    when an ACS tax unit spans two SPM units; or when ``receipt`` (the
    staging receipt) is missing, from another method, or moved a different
    number of people than the frame shows. Details report the weighted ACS
    persons per SPM unit, single-person share and units per household.
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
        relationship = _complete_integers(rows, _RELATIONSHIP)
        moved = acs_adult_nonrelative_mask(rows)
    except ValueError as exc:
        return _gate([f"{ACS_2024_1YR_SPINE}: {exc}"], details)

    spm = rows[_SPM_MEMBERSHIP].to_numpy()
    household = rows[_HOUSEHOLD_MEMBERSHIP].to_numpy()
    reference_units = pd.unique(spm[relationship == _REFERENCE_PERSON])
    shared = int((moved & np.isin(spm, reference_units)).sum())
    if shared:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: {shared} adult roommate(s) or other "
            "nonrelative(s) (RELSHIPP 34/36, 15 or over) share an SPM unit "
            "with the reference person."
        )
    member = np.where(moved, rows[_PERSON_ID].to_numpy(), -1)
    expected = len(pd.MultiIndex.from_arrays([household, member]).unique())
    actual = len(pd.unique(spm))
    pairings = len(pd.MultiIndex.from_arrays([household, member, spm]).unique())
    if not expected == actual == pairings:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: the SPM partition is not one unit per adult "
            f"nonrelative plus one per household ({expected} expected unit(s), "
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
    failures += _receipt_failures(receipt, persons_moved=int(moved.sum()))

    weights = np.asarray(frame.resolve_weights("person").values, dtype=np.float64)[acs]
    units = _unit_summary(spm, weights)
    households = _unit_summary(household, weights)
    details.update(
        {
            "acs_person_rows": int(acs.sum()),
            "adult_nonrelatives": int(moved.sum()),
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
    receipt: Mapping[str, Any] | None, *, persons_moved: int
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
            f"{created} new unit(s); the frame holds {persons_moved} adult "
            "nonrelative(s), each needing one."
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
