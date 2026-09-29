"""Native ACS SNAP receipt as the reported-receipt anchor on ACS rows.

The ACS local lane's SNAP take-up stage
(:mod:`~microcosm.build.us_runtime.acs_local_take_up`, microcosm#1019) makes
every SPM unit with ``receives_snap`` take SNAP up. On ACS rows that leaf came
only from the shared QRF transfer, whose predictors (unit size, ages, sex,
state, income sums, headship, tenure) carry no receipt signal. The ACS asks
it directly: household ``FS``, whether anyone in the household received SNAP
in the past 12 months (1 yes, 2 no; asked of housing units, blank for vacant
units, and outside the item's universe in group quarters).

This fresh-build stage (microcosm#1022) runs on the ACS-only frame after the
shared transfer and before pooling. The declared transfer plan is unchanged,
so the transfer still fits and writes ``receives_snap``; this stage then
overrides it, never sees a donor row, and lands in the staging H5 that the
release tool's take-up stage reads.

The rule
--------

- ``FS == 2``: every SPM unit of the household is ``False``.
- ``FS == 1``: every SPM unit of the household is ``True``, including the
  one-person units microcosm#1023 gives adult roommates and other
  nonrelatives. The item names the housing unit, not the SNAP household, so
  it cannot say which unit received SNAP. Marking every unit is the donor's
  own convention: the CPS ASEC asks the same household question, Census
  prorates the household's SNAP amount to every SPM unit in it (Supplemental
  Poverty Measure technical documentation, section 4.3.1), and the donor's
  ``receives_snap`` is ``SPM_SNAPSUB > 0``.
- Group quarters (``TYPEHUGQ`` 2/3) are ``False``. FS is a housing-unit item,
  and a group-quarters record that carries a code anyway is ignored and
  counted in the receipt.
- A housing unit with a blank or unknown FS code is refused, never read as
  no.

A reporter only anchors take-up; the engine still applies SNAP eligibility
to each unit's own members and income.

TANF
----

``receives_tanf`` keeps its transferred value. ACS ``PAP`` (person public
assistance income, asked from age 15) covers TANF and general assistance
together. The donor gates ``receives_tanf`` on ``PAW_TYP`` because
``PAW_VAL > 0`` alone conflates the two (microcosm#591), and the ACS has no
type item. ``PAP`` is loaded, and the receipt records its recipients and
their overlap with the transferred ``receives_tanf``; it anchors nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import GateResult
from microcosm.build.us_runtime.acs_pums import ACS_2024_1YR_SPINE
from microcosm.build.us_runtime.base_pool import spine_column
from microcosm.frame import Frame
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "ACS_FS_NO",
    "ACS_FS_YES",
    "ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME",
    "ACS_LOCAL_RECEIPT_ANCHOR_ISSUE",
    "ACS_LOCAL_RECEIPT_ANCHOR_METHOD",
    "ACS_PUBLIC_ASSISTANCE_INCOME",
    "ACS_PUBLIC_ASSISTANCE_MIN_AGE",
    "ACS_SNAP_HOUSEHOLD_SHARE_BAND",
    "ACS_SNAP_HOUSEHOLD_SHARE_REFERENCE",
    "ACS_SNAP_RECIPIENCY",
    "acs_local_receipt_anchor_signal_gate",
    "require_acs_receipt_anchor_sources",
    "with_acs_local_snap_receipt_anchor",
]

ACS_LOCAL_RECEIPT_ANCHOR_ISSUE = "microcosm#1022"
ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME = "acs_local_receipt_anchor_signal"
ACS_LOCAL_RECEIPT_ANCHOR_METHOD = "acs_fs_household_receipt_on_every_spm_unit"
#: ACS household SNAP recipiency in the past 12 months: 1 yes, 2 no.
ACS_SNAP_RECIPIENCY = "FS"
ACS_FS_YES = 1
ACS_FS_NO = 2
#: ACS person public assistance income (TANF and general assistance
#: together; not SSI or SNAP), asked from age 15.
ACS_PUBLIC_ASSISTANCE_INCOME = "PAP"
ACS_PUBLIC_ASSISTANCE_MIN_AGE = 15
#: The 2023 ACS 1-year share of households receiving SNAP (table S2201).
ACS_SNAP_HOUSEHOLD_SHARE_REFERENCE = 0.122
_SHARE_REFERENCE_SOURCE = (
    "Glassman (2026), 'The Variation of SNAP Receipt by Survey', Census "
    "Bureau working paper SEHSD-WP2026-13: 12.2 percent of households "
    "received SNAP in the 2023 ACS (S2201); SNAP is underreported in every "
    "survey against administrative counts"
)
#: Informational only: the weighted FS == 1 share of ACS housing-unit
#: households is reported against this band, never graded.
ACS_SNAP_HOUSEHOLD_SHARE_BAND = (0.09, 0.15)

_SNAP = "receives_snap"
_TANF = "receives_tanf"
_HOUSEHOLD_KIND = "TYPEHUGQ"
_HOUSING_UNIT = 1
_HOUSEHOLD_KINDS = (1, 2, 3)
_AGE = "AGEP"
_RELATIONSHIP = "RELSHIPP"
_REFERENCE_PERSON = 20
_HOUSEHOLD_ID = "household_id"
_SPM_ID = "spm_unit_id"
_HOUSEHOLD_MEMBERSHIP = "person_household_id"
_SPM_MEMBERSHIP = "person_spm_unit_id"
_PERSON_COLUMNS = (
    _SPM_MEMBERSHIP,
    _HOUSEHOLD_MEMBERSHIP,
    _RELATIONSHIP,
    _AGE,
    ACS_PUBLIC_ASSISTANCE_INCOME,
)
_RULE = (
    "receives_snap is True on every SPM unit of an FS == 1 housing unit and "
    "False on every unit of an FS == 2 housing unit and in group quarters"
)
_TANF_DECISION = (
    "receives_tanf keeps its transferred value: PAP covers TANF and general "
    "assistance together, and PAP > 0 is not TANF receipt (microcosm#591); "
    "PAP is recorded, not applied"
)
#: Receipt counts the gate recomputes from the frame.
_GRADED_SNAP_COUNTS = (
    "acs_households",
    "housing_unit_households",
    "fs_yes_households",
    "acs_spm_units",
    "units_anchored",
)
_GRADED_TANF_COUNTS = ("pap_recipients", "pap_units")


@dataclass(frozen=True)
class _Anchor:
    """The FS rule on one set of ACS households, persons and SPM units."""

    housing_unit: np.ndarray  # per household
    fs_yes: np.ndarray  # per household: a housing unit with FS == 1
    group_quarters_coded: np.ndarray  # per household: GQ with an FS code
    unit_household: np.ndarray  # per unit: position in the household table
    unit_reference: np.ndarray  # per unit: holds the reference person
    unit_group_quarters: np.ndarray  # per unit
    unit_anchor: np.ndarray  # per unit: the rule's receives_snap


def _numeric(values: pd.Series) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _flags(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Boolean values with missing cells as ``False``, and the present mask."""
    present = values.notna().to_numpy(dtype=bool)
    flags = np.zeros(len(values), dtype=bool)
    flags[present] = values[present].astype(bool).to_numpy(dtype=bool)
    return flags, present


def _share(weights: np.ndarray, flags: np.ndarray) -> float:
    total = float(weights.sum())
    return float(weights[flags].sum()) / total if total > 0 else 0.0


def _require(table: pd.DataFrame, columns: tuple[str, ...], entity: str) -> None:
    absent = [column for column in columns if column not in table]
    if absent:
        raise ValueError(
            f"The ACS receipt anchor requires {entity} column(s) {absent}."
        )


def _household_codes(household: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """``TYPEHUGQ`` and ``FS`` per household; refuses an unreadable code."""
    _require(household, (_HOUSEHOLD_KIND, ACS_SNAP_RECIPIENCY), "household")
    kinds = _numeric(household[_HOUSEHOLD_KIND])
    unknown = int((~np.isin(kinds, _HOUSEHOLD_KINDS)).sum())
    if unknown:
        raise ValueError(
            f"{unknown} ACS household(s) have no {_HOUSEHOLD_KIND} 1/2/3, so "
            f"{ACS_SNAP_RECIPIENCY} cannot be scoped to housing units."
        )
    fs = _numeric(household[ACS_SNAP_RECIPIENCY])
    invalid = int(
        ((kinds == _HOUSING_UNIT) & ~np.isin(fs, (ACS_FS_YES, ACS_FS_NO))).sum()
    )
    if invalid:
        raise ValueError(
            f"{invalid} ACS housing-unit household(s) carry {ACS_SNAP_RECIPIENCY} "
            f"outside {ACS_FS_YES} (yes) / {ACS_FS_NO} (no); a blank is refused, "
            "not read as no."
        )
    return kinds, fs


def _public_assistance(person: pd.DataFrame) -> np.ndarray:
    """``PAP`` per person, blank under 15; refuses a blank or negative at 15+."""
    _require(person, (ACS_PUBLIC_ASSISTANCE_INCOME, _AGE), "person")
    amount = _numeric(person[ACS_PUBLIC_ASSISTANCE_INCOME])
    age = _numeric(person[_AGE])
    negative = int((amount < 0).sum())
    if negative:
        raise ValueError(
            f"ACS {ACS_PUBLIC_ASSISTANCE_INCOME} is negative on {negative} row(s)."
        )
    blank = int((np.isnan(amount) & (age >= ACS_PUBLIC_ASSISTANCE_MIN_AGE)).sum())
    if blank:
        raise ValueError(
            f"ACS {ACS_PUBLIC_ASSISTANCE_INCOME} is blank on {blank} person(s) "
            f"aged {ACS_PUBLIC_ASSISTANCE_MIN_AGE} or over; the ACS asks it from "
            f"{ACS_PUBLIC_ASSISTANCE_MIN_AGE}, so a blank there is refused, not "
            "read as none."
        )
    return amount


def require_acs_receipt_anchor_sources(frame: Frame) -> None:
    """Refuse, before any fit, an ACS frame whose FS or PAP the stage can't read.

    Checks household ``TYPEHUGQ``/``FS`` (a housing unit must carry FS 1 or 2)
    and person ``PAP`` (never negative, never blank at 15 or over), so a
    source without them fails before the transfer rather than after it.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("The ACS receipt anchor requires the US schema.")
    _household_codes(frame.table("household"))
    _public_assistance(frame.table("person"))


def _anchor(
    household: pd.DataFrame, person: pd.DataFrame, unit_ids: np.ndarray
) -> _Anchor:
    """Apply the FS rule to ACS rows. ``person`` holds exactly their members."""

    kinds, fs = _household_codes(household)
    _require(household, (_HOUSEHOLD_ID,), "household")
    _require(person, (_SPM_MEMBERSHIP, _HOUSEHOLD_MEMBERSHIP, _RELATIONSHIP), "person")
    housing_unit = kinds == _HOUSING_UNIT
    fs_yes = housing_unit & (fs == ACS_FS_YES)
    members = pd.DataFrame(
        {
            "unit": person[_SPM_MEMBERSHIP].to_numpy(),
            "household": person[_HOUSEHOLD_MEMBERSHIP].to_numpy(),
            "reference": _numeric(person[_RELATIONSHIP]) == _REFERENCE_PERSON,
        }
    ).groupby("unit", sort=True)
    spanning = int((members["household"].nunique() > 1).sum())
    if spanning:
        raise ValueError(
            f"{spanning} ACS SPM unit(s) span more than one household; the "
            "household FS item cannot be applied to them."
        )
    unit_household = members["household"].first().reindex(unit_ids)
    if unit_household.isna().any():
        raise ValueError(
            f"{int(unit_household.isna().sum())} ACS SPM unit(s) have no member."
        )
    positions = pd.Index(household[_HOUSEHOLD_ID].to_numpy()).get_indexer(
        unit_household.to_numpy()
    )
    if (positions < 0).any():
        raise ValueError(
            f"{int((positions < 0).sum())} ACS SPM unit(s) belong to a household "
            "outside the ACS household table."
        )
    reference = (
        members["reference"].any().reindex(unit_ids, fill_value=False).to_numpy(bool)
    )
    return _Anchor(
        housing_unit=housing_unit,
        fs_yes=fs_yes,
        group_quarters_coded=~housing_unit & ~np.isnan(fs),
        unit_household=positions,
        unit_reference=reference,
        unit_group_quarters=~housing_unit[positions],
        unit_anchor=fs_yes[positions],
    )


def _snap_counts(
    anchor: _Anchor, *, household_weights: np.ndarray, unit_weights: np.ndarray
) -> dict[str, Any]:
    housing_unit = anchor.housing_unit
    units_per_household = np.bincount(
        anchor.unit_household, minlength=len(housing_unit)
    )
    share = _share(household_weights[housing_unit], anchor.fs_yes[housing_unit])
    low, high = ACS_SNAP_HOUSEHOLD_SHARE_BAND
    return {
        "acs_households": int(len(housing_unit)),
        "housing_unit_households": int(housing_unit.sum()),
        "group_quarters_households": int((~housing_unit).sum()),
        "fs_yes_households": int(anchor.fs_yes.sum()),
        "fs_no_households": int((housing_unit & ~anchor.fs_yes).sum()),
        "group_quarters_fs_coded_ignored": int(anchor.group_quarters_coded.sum()),
        "fs_yes_households_with_several_units": int(
            (anchor.fs_yes & (units_per_household > 1)).sum()
        ),
        "acs_spm_units": int(len(anchor.unit_anchor)),
        "units_anchored": int(anchor.unit_anchor.sum()),
        "group_quarters_units": int(anchor.unit_group_quarters.sum()),
        # One-person units of adult nonrelatives (microcosm#1023) in FS == 1
        # housing units: the units this rule marks that the reference-unit
        # alternative would not.
        "non_reference_units_anchored": int(
            (anchor.unit_anchor & ~anchor.unit_reference).sum()
        ),
        "weighted": {
            "housing_unit_households": float(household_weights[housing_unit].sum()),
            "fs_yes_households": float(household_weights[anchor.fs_yes].sum()),
            "fs_yes_household_share": share,
            "anchored_unit_share": _share(unit_weights, anchor.unit_anchor),
        },
        "household_share_reference": {
            "value": ACS_SNAP_HOUSEHOLD_SHARE_REFERENCE,
            "source": _SHARE_REFERENCE_SOURCE,
            "band": [low, high],
            "within_band": bool(low <= share <= high),
            "graded": False,
        },
    }


def _tanf_counts(
    person: pd.DataFrame,
    amount: np.ndarray,
    unit_ids: np.ndarray,
    tanf: np.ndarray,
    unit_anchor: np.ndarray,
    *,
    unit_weights: np.ndarray,
    person_weights: np.ndarray,
) -> dict[str, Any]:
    recipient = np.nan_to_num(amount, nan=0.0) > 0
    unit_recipient = (
        pd.Series(recipient)
        .groupby(person[_SPM_MEMBERSHIP].to_numpy(), sort=True)
        .any()
        .reindex(unit_ids, fill_value=False)
        .to_numpy(bool)
    )
    return {
        "pap_recipients": int(recipient.sum()),
        "pap_blank_under_min_age": int(np.isnan(amount).sum()),
        "pap_units": int(unit_recipient.sum()),
        "transferred_tanf_units": int(tanf.sum()),
        "transferred_tanf_units_with_pap": int((tanf & unit_recipient).sum()),
        "transferred_tanf_units_without_pap": int((tanf & ~unit_recipient).sum()),
        "pap_units_without_transferred_tanf": int((unit_recipient & ~tanf).sum()),
        "transferred_tanf_units_not_snap_anchored": int((tanf & ~unit_anchor).sum()),
        "weighted": {
            "pap_recipients": float(person_weights[recipient].sum()),
            "pap_unit_share": _share(unit_weights, unit_recipient),
            "transferred_tanf_unit_share": _share(unit_weights, tanf),
        },
    }


def with_acs_local_snap_receipt_anchor(frame: Frame) -> tuple[Frame, dict[str, Any]]:
    """Replace the transferred ``receives_snap`` on ACS rows with the FS rule.

    ``frame`` is the ACS-only frame after the shared transfer, before
    pooling: every row is an ACS row, the household table carries
    ``TYPEHUGQ`` and ``FS``, the person table ``PAP``, ``AGEP`` and
    ``RELSHIPP``, and the spm_unit table the transferred ``receives_snap``
    and ``receives_tanf``. Only ``receives_snap`` changes. Returns the frame
    and a JSON-ready receipt: FS households (weighted and unweighted), units
    anchored, transferred values overridden in each direction, and the ``PAP``
    recipients against the transferred ``receives_tanf``.

    Raises:
        ValueError: If the frame is not US-schema, a required column is
            absent, a housing unit has no FS 1/2, ``PAP`` is blank at 15 or
            over, or an SPM unit spans households or has no member.
    """

    if frame.schema != US_SCHEMA:
        raise ValueError("The ACS receipt anchor requires the US schema.")
    spm_unit = frame.table("spm_unit")
    absent = [column for column in (_SNAP, _TANF) if column not in spm_unit]
    if absent:
        raise ValueError(
            "The ACS receipt anchor runs after the transfer, which writes "
            f"receives_snap and receives_tanf; the spm_unit table lacks {absent}."
        )
    person = frame.table("person")
    _require(person, _PERSON_COLUMNS, "person")
    household = frame.table("household")
    unit_ids = spm_unit[_SPM_ID].to_numpy()
    anchor = _anchor(household, person, unit_ids)
    amount = _public_assistance(person)
    transferred, transferred_present = _flags(spm_unit[_SNAP])
    tanf, _ = _flags(spm_unit[_TANF])
    household_weights = np.asarray(
        frame.resolve_weights("household").values, dtype=np.float64
    )
    unit_weights = np.asarray(
        frame.resolve_weights("spm_unit").values, dtype=np.float64
    )
    person_weights = np.asarray(
        frame.resolve_weights("person").values, dtype=np.float64
    )

    updated = spm_unit.copy()
    updated[_SNAP] = anchor.unit_anchor.copy()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["spm_unit"] = updated
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )

    to_true = anchor.unit_anchor & ~transferred
    to_false = ~anchor.unit_anchor & transferred
    snap = _snap_counts(
        anchor, household_weights=household_weights, unit_weights=unit_weights
    )
    snap["transferred"] = {
        "true_units": int(transferred.sum()),
        "missing_units": int((~transferred_present).sum()),
        "false_to_true": int(to_true.sum()),
        "true_to_false": int(to_false.sum()),
        "group_quarters_true_to_false": int(
            (anchor.unit_group_quarters & transferred).sum()
        ),
        "unchanged": int(
            (transferred_present & (anchor.unit_anchor == transferred)).sum()
        ),
        "weighted": {
            "true_unit_share": _share(unit_weights, transferred),
            "false_to_true_unit_share": _share(unit_weights, to_true),
            "true_to_false_unit_share": _share(unit_weights, to_false),
        },
    }
    receipt: dict[str, Any] = {
        "issue": ACS_LOCAL_RECEIPT_ANCHOR_ISSUE,
        "method": ACS_LOCAL_RECEIPT_ANCHOR_METHOD,
        "rule": _RULE,
        "source": (
            f"ACS {ACS_SNAP_RECIPIENCY}: anyone in the household received SNAP "
            "in the past 12 months (1 yes, 2 no)"
        ),
        "snap": snap,
        "tanf": {
            "decision": _TANF_DECISION,
            "source": (
                f"ACS {ACS_PUBLIC_ASSISTANCE_INCOME}: public assistance income in "
                "the past 12 months (TANF and general assistance), from age "
                f"{ACS_PUBLIC_ASSISTANCE_MIN_AGE}"
            ),
            **_tanf_counts(
                person,
                amount,
                unit_ids,
                tanf,
                anchor.unit_anchor,
                unit_weights=unit_weights,
                person_weights=person_weights,
            ),
        },
    }
    return result, receipt


def acs_local_receipt_anchor_signal_gate(
    frame: Frame,
    *,
    receipt: Mapping[str, Any] | None,
) -> GateResult:
    """Require the FS receipt anchor on every ACS SPM unit.

    Fails when origin tags or a required column are missing; when an ACS
    housing unit carries no FS 1/2 or an ACS ``PAP`` is blank at 15 or over;
    when an ACS SPM unit's ``receives_snap`` is missing or differs from the
    rule (``True`` exactly on the units of FS == 1 housing units); or when
    ``receipt`` (the staging receipt) is missing, from another method, or
    counts households, units, anchors or ``PAP`` recipients the frame does
    not hold. Details report the weighted FS == 1 share of ACS housing units
    against the ACS S2201 reference band (informational, never a failure),
    the ``PAP``/``receives_tanf`` overlap and the donor's ``receives_snap``
    share. Donor rows are reported, never graded: their receipt is ASEC's.
    """

    spm_unit = frame.table("spm_unit")
    household = frame.table("household")
    person = frame.table("person")
    tags = {
        entity: spine_column(entity) for entity in ("spm_unit", "household", "person")
    }
    details: dict[str, object] = {}
    for entity, table in (
        ("spm_unit", spm_unit),
        ("household", household),
        ("person", person),
    ):
        tag = tags[entity]
        if tag not in table or table[tag].isna().any():
            return _gate([f"Missing {entity} origin tags: {tag}."], details)
    acs_units = spm_unit[tags["spm_unit"]].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    acs_households = (
        household[tags["household"]].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    )
    acs_persons = person[tags["person"]].eq(ACS_2024_1YR_SPINE).to_numpy(dtype=bool)
    if not acs_units.any():
        return _gate([f"{ACS_2024_1YR_SPINE}: no spm_unit rows."], details)
    absent = [column for column in (_SNAP, _TANF) if column not in spm_unit]
    if absent:
        return _gate(
            [f"{ACS_2024_1YR_SPINE}: spm_unit column(s) {absent} are absent."], details
        )
    unit_ids = spm_unit.loc[acs_units, _SPM_ID].to_numpy()
    household_columns = (_HOUSEHOLD_ID, _HOUSEHOLD_KIND, ACS_SNAP_RECIPIENCY)
    try:
        _require(household, household_columns, "household")
        _require(person, _PERSON_COLUMNS, "person")
        acs_person = person.loc[acs_persons, list(_PERSON_COLUMNS)]
        anchor = _anchor(
            household.loc[acs_households, list(household_columns)],
            acs_person,
            unit_ids,
        )
        amount = _public_assistance(acs_person)
    except ValueError as exc:
        return _gate([f"{ACS_2024_1YR_SPINE}: {exc}"], details)

    failures: list[str] = []
    values, present = _flags(spm_unit.loc[acs_units, _SNAP])
    missing = int((~present).sum())
    if missing:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: receives_snap is missing on {missing} SPM "
            "unit(s); the engine default is no reported receipt."
        )
    unmeasured = int((present & values & ~anchor.unit_anchor).sum())
    if unmeasured:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: {unmeasured} SPM unit(s) report SNAP receipt "
            f"outside an {ACS_SNAP_RECIPIENCY} == {ACS_FS_YES} housing unit."
        )
    unanchored = int((present & ~values & anchor.unit_anchor).sum())
    if unanchored:
        failures.append(
            f"{ACS_2024_1YR_SPINE}: {unanchored} SPM unit(s) of an "
            f"{ACS_SNAP_RECIPIENCY} == {ACS_FS_YES} housing unit do not report "
            "SNAP receipt."
        )

    household_weights = np.asarray(
        frame.resolve_weights("household").values, dtype=np.float64
    )
    unit_weights = np.asarray(
        frame.resolve_weights("spm_unit").values, dtype=np.float64
    )
    person_weights = np.asarray(
        frame.resolve_weights("person").values, dtype=np.float64
    )
    snap = _snap_counts(
        anchor,
        household_weights=household_weights[acs_households],
        unit_weights=unit_weights[acs_units],
    )
    tanf = _tanf_counts(
        acs_person,
        amount,
        unit_ids,
        _flags(spm_unit.loc[acs_units, _TANF])[0],
        anchor.unit_anchor,
        unit_weights=unit_weights[acs_units],
        person_weights=person_weights[acs_persons],
    )
    failures += _receipt_failures(receipt, snap=snap, tanf=tanf)
    donor = ~acs_units
    details.update(
        {
            "rule": _RULE,
            "snap": snap,
            "tanf": tanf,
            "donor_spm_units": int(donor.sum()),
            "donor_receives_snap_unit_share": _share(
                unit_weights[donor], _flags(spm_unit.loc[donor, _SNAP])[0]
            ),
        }
    )
    return _gate(failures, details)


def _receipt_failures(
    receipt: Mapping[str, Any] | None,
    *,
    snap: Mapping[str, Any],
    tanf: Mapping[str, Any],
) -> list[str]:
    if not isinstance(receipt, Mapping):
        return ["The staging summary carries no ACS receipt-anchor receipt."]
    failures: list[str] = []
    if receipt.get("issue") != ACS_LOCAL_RECEIPT_ANCHOR_ISSUE:
        failures.append(
            f"The ACS receipt-anchor receipt is not {ACS_LOCAL_RECEIPT_ANCHOR_ISSUE}'s."
        )
    if receipt.get("method") != ACS_LOCAL_RECEIPT_ANCHOR_METHOD:
        failures.append(
            "The ACS receipt-anchor receipt records method "
            f"{receipt.get('method')!r}, not {ACS_LOCAL_RECEIPT_ANCHOR_METHOD!r}."
        )
    recorded_snap = receipt.get("snap")
    recorded_tanf = receipt.get("tanf")
    if not isinstance(recorded_snap, Mapping) or not isinstance(recorded_tanf, Mapping):
        return [*failures, "The ACS receipt-anchor receipt has no snap/tanf counts."]
    for section, recorded, expected, keys in (
        ("snap", recorded_snap, snap, _GRADED_SNAP_COUNTS),
        ("tanf", recorded_tanf, tanf, _GRADED_TANF_COUNTS),
    ):
        for key in keys:
            value = recorded.get(key)
            if type(value) is not int:
                failures.append(
                    f"The ACS receipt-anchor receipt's {section}.{key} is not an "
                    "integer."
                )
            elif value != expected[key]:
                failures.append(
                    f"The ACS receipt-anchor receipt records {section}.{key} = "
                    f"{value}; the frame holds {expected[key]}."
                )
    transferred = recorded_snap.get("transferred")
    moves = (
        [
            transferred.get(key)
            for key in ("true_units", "true_to_false", "false_to_true")
        ]
        if isinstance(transferred, Mapping)
        else []
    )
    anchored = recorded_snap.get("units_anchored")
    if len(moves) != 3 or any(type(value) is not int for value in moves):
        failures.append(
            "The ACS receipt-anchor receipt does not count the transferred "
            "receives_snap it overrode."
        )
    elif type(anchored) is int and moves[0] - moves[1] + moves[2] != anchored:
        failures.append(
            "The ACS receipt-anchor receipt's overrides do not reconcile: "
            f"{moves[0]} transferred True - {moves[1]} to False + {moves[2]} to "
            f"True != {anchored} anchored."
        )
    return failures


def _gate(failures: list[str], details: dict[str, object]) -> GateResult:
    return GateResult(
        name=ACS_LOCAL_RECEIPT_ANCHOR_GATE_NAME,
        passed=not failures,
        failures=tuple(failures),
        details=details,
    )
