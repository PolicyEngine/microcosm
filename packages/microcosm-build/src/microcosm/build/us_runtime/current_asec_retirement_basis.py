"""Pure retirement candidate ledger, with explicit measurement assumptions.

A descriptive table or weight Series grants no source authority. The host owns
complete original-source admission. Candidate bounds are neither fiscal amounts
nor identified ACS labels; aggregate differences are never allocated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import fsum
from types import MappingProxyType

import numpy as np
import pandas as pd

from . import current_asec_income_routing_source as routing
from . import current_asec_retirement_detail_source as detail

PROTOCOL = "microcosm.us.asec-retirement-candidate-basis.v1"
# These bridge route sets are decisions, distinct from the source code labels.
CANDIDATE_CODES = MappingProxyType(
    {
        "pension": frozenset(range(1, 7)),
        "disability": frozenset(range(2, 6)),
        "survivor": frozenset(range(1, 5)),
    }
)
RAILROAD_CODES = MappingProxyType({"pension": 7, "disability": 6, "survivor": 5})
FAMILIES = (
    ("pension", "PEN", "PNSN_VAL"),
    ("disability", "DIS", "DSAB_VAL"),
    ("survivor", "SUR", "SRVS_VAL"),
)
DISTRIBUTION_SLOTS = (
    ("slot1", "DST_VAL1", False),
    ("slot2", "DST_VAL2", False),
    ("slot1_young", "DST_VAL1_YNG", True),
    ("slot2_young", "DST_VAL2_YNG", True),
)
OTHER_RETIREMENT_CANDIDATES = frozenset((2, 13))
OTHER_UNSPECIFIED = 19


def _require(condition, reason):
    if not condition:
        raise ValueError("ASEC_RETIREMENT_BASIS_" + reason)


@dataclass(frozen=True)
class RetirementCandidateAssumptions:
    """Required choices; no constructor defaults or observed-treatment claims."""

    pension_annuity_regularity: str
    disability_pension_eligibility: str
    survivor_annuity_overlap: str
    withdrawal_regularity_netting: str
    aggregate_accounting: str

    def __post_init__(self):
        allowed = {
            "pension_annuity_regularity": ("unresolved", "assume_regular"),
            "disability_pension_eligibility": ("unresolved", "assume_qualifying"),
            "survivor_annuity_overlap": ("unresolved",),
            "withdrawal_regularity_netting": ("unresolved",),
            "aggregate_accounting": ("exact_visible_balance_only",),
        }
        _require(
            all(
                type(getattr(self, k)) is str and getattr(self, k) in v
                for k, v in allowed.items()
            ),
            "ASSUMPTIONS",
        )

    def to_bytes(self) -> bytes:
        self.__post_init__()
        return (
            json.dumps(
                {"protocol": PROTOCOL, **asdict(self)},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode()


@dataclass(frozen=True)
class AsecRetirementBasis:
    """Detached descriptive tables and immutable assumption bytes; no issuer."""

    person: pd.DataFrame
    slots: pd.DataFrame
    provenance: pd.DataFrame
    exclusions: pd.DataFrame
    summary: pd.DataFrame
    assumptions_payload: bytes


def _numeric(series, label):
    _require(
        pd.api.types.is_numeric_dtype(series.dtype)
        and not pd.api.types.is_bool_dtype(series.dtype)
        and not pd.api.types.is_complex_dtype(series.dtype),
        "NUMERIC:" + label,
    )
    values = series.to_numpy(dtype="float64", na_value=np.nan, copy=True)
    _require(not np.isinf(values).any(), "INFINITE:" + label)
    return values


def _text(frame, name):
    _require(name in frame, "COLUMN:" + name)
    values = frame[name].to_numpy(copy=True)
    _require(all(type(v) is str for v in values), "TEXT:" + name)
    return values


def _axis(frame):
    _require(
        type(frame) is pd.DataFrame
        and frame.columns.is_unique
        and frame.index.dtype == np.dtype("int64")
        and frame.index.name == "person_id"
        and frame.index.is_unique,
        "PERSON_AXIS",
    )
    _require({"native_person_id", "source_age"} <= set(frame), "IDENTITY_COLUMNS")
    _require(
        frame.native_person_id.dtype == np.dtype("int64")
        and frame.native_person_id.is_unique,
        "NATIVE_AXIS",
    )
    age = _numeric(frame.source_age, "AGE")
    _require(
        np.isfinite(age).all()
        and ((age >= 0) & (age <= 99) & (age == np.floor(age))).all(),
        "AGE",
    )
    return age


def _equal(left, right, label):
    _require(np.array_equal(left, right, equal_nan=True), "AGREEMENT:" + label)


def _codes(frame, prefix, domain):
    values = _numeric(frame[prefix + "_code"], prefix)
    status = _text(frame, prefix + "_literal_status")
    valid = status == "in_printed_range"
    _require(
        np.array_equal(valid, np.isfinite(values))
        and np.isin(values[valid], tuple(domain)).all(),
        "CODE:" + prefix,
    )
    return values, status


def _published(frame):
    values = {}
    for name, entry in detail.amount_entries().items():
        amount = _numeric(frame[name + "_published_amount"], name)
        status = _text(frame, name + "_literal_status")
        valid = status == "in_printed_range"
        _require(
            np.array_equal(valid, np.isfinite(amount))
            and (
                (amount[valid] >= entry.encoded_minimum)
                & (amount[valid] <= entry.encoded_maximum)
                & (amount[valid] == np.floor(amount[valid]))
            ).all(),
            "PUBLISHED:" + name,
        )
        values[name] = amount
    return values


def _known_routing(frame, prefix):
    status = _text(frame, prefix + "_reporting_status")
    amount = _numeric(frame[prefix + "_known_amount"], prefix)
    known = np.isin(status, routing.KNOWN_AMOUNT_STATUSES)
    _require(
        np.array_equal(known, np.isfinite(amount))
        and (amount[known] >= 0).all()
        and (
            amount[np.isin(status, ("known_nonreceipt", "known_recipient_zero"))] == 0
        ).all(),
        "KNOWN_ROUTING:" + prefix,
    )
    return amount, status


def _reference_agreement(d, r, published, ages):
    for family, field, receipt_prefix in (
        ("pension", "PNSN_VAL", "pension_annuity_pension_receipt"),
        ("annuity", "ANN_VAL", "pension_annuity_annuity_receipt"),
    ):
        prefix = "pension_annuity_" + family
        receipt, receipt_status = _codes(r, receipt_prefix, routing.RECEIPT_CODE_DOMAIN)
        if family == "pension":
            other, other_status = _codes(d, "PEN_YN", routing.RECEIPT_CODE_DOMAIN)
            _equal(receipt, other, "PEN_YN")
            _require(np.array_equal(receipt_status, other_status), "PEN_YN_STATUS")
        amount, statuses = _known_routing(r, prefix)
        source = published[field].copy()
        kinds = np.where(
            np.isnan(source),
            "missing",
            np.where(
                source == -1, "declared_niu", np.where(source == 0, "zero", "nonzero")
            ),
        )
        source[kinds == "declared_niu"] = np.nan
        _equal(source, _numeric(r[prefix + "_source_total"], prefix), field)
        expected = [
            routing.receipt_status(
                bool(age >= 15),
                (None if np.isnan(code) else int(code), literal),
                kind,
                net_measure=False,
                zero_is_dollars=routing._zero_is_dollars(field),
            )
            for age, code, literal, kind in zip(
                ages, receipt, receipt_status, kinds, strict=True
            )
        ]
        _require(np.array_equal(statuses, expected), "ROUTING_STATUS:" + field)
        _equal(
            amount,
            np.where(np.isin(expected, routing.KNOWN_AMOUNT_STATUSES), source, np.nan),
            "KNOWN:" + field,
        )
    for slot, field, _ in DISTRIBUTION_SLOTS:
        _equal(
            published[field],
            _numeric(r["retirement_distribution_" + slot + "_amount"], slot),
            field,
        )


def _route(family, code):
    if not np.isfinite(code):
        return "unresolved"
    if code == 0:
        return "unused_or_nonreceipt"
    if code in CANDIDATE_CODES[family]:
        return "candidate"
    if code == RAILROAD_CODES[family]:
        return "railroad"
    if family == "disability" and code in (1, 7, 8, 9):
        return "other_compensation"
    if family == "survivor" and code == 8:
        return "property"
    if family == "survivor" and code == 9:
        return "annuity_overlap"
    return "unresolved"


def _detail_slots(d, published, ages):
    result = {family: [] for family, *_ in FAMILIES}
    for family, prefix, _ in FAMILIES:
        receipt, receipt_literal = _codes(
            d, prefix + "_YN", routing.RECEIPT_CODE_DOMAIN
        )
        for slot in (1, 2):
            field = prefix + "_VAL" + str(slot)
            code, literal = _codes(
                d, prefix + "_SC" + str(slot), detail.SOURCE_CODES[prefix]
            )
            statuses = _text(d, field + "_reporting_status")
            amount = _numeric(d[field + "_amount"], field)
            known = d[field + "_amount_known"]
            _require(
                pd.api.types.is_bool_dtype(known.dtype) and known.notna().all(),
                "KNOWN_MASK:" + field,
            )
            expected = [
                detail._slot_status(
                    age,
                    (None if np.isnan(rec) else int(rec), recstat),
                    (None if np.isnan(c) else int(c), cstat),
                    (None if np.isnan(v) else int(v), vs),
                )
                for age, rec, recstat, c, cstat, v, vs in zip(
                    ages,
                    receipt,
                    receipt_literal,
                    code,
                    literal,
                    published[field],
                    _text(d, field + "_literal_status"),
                    strict=True,
                )
            ]
            _require(np.array_equal(statuses, expected), "SLOT_STATUS:" + field)
            expected_known = np.isin(statuses, routing.KNOWN_AMOUNT_STATUSES)
            _require(
                np.array_equal(known.to_numpy(dtype=bool), expected_known),
                "KNOWN_MASK:" + field,
            )
            _equal(
                amount,
                np.where(expected_known, published[field], np.nan),
                "SLOT_AMOUNT:" + field,
            )
            for i in range(len(d)):
                structural = (
                    statuses[i] == "unreported_source_slot"
                    and code[i] == 0
                    and published[field][i] == 0
                )
                result[family].append(
                    {
                        "position": i,
                        "family": family,
                        "slot": slot,
                        "amount_field": field,
                        "source_code": code[i],
                        "source_literal_status": literal[i],
                        "published_amount": published[field][i],
                        "reporting_status": statuses[i],
                        "source_known_amount": amount[i],
                        "source_amount_known": bool(expected_known[i]),
                        "route": _route(family, code[i]),
                        "applicable": bool(ages[i] >= 15),
                        "structural_zero_comparison": bool(structural),
                    }
                )
    return result


def _comparison(d, published, family, total, first, second):
    difference = published[total] - published[first] - published[second]
    suffix = "main_slots" if family == "distribution" else "visible_slots"
    _equal(
        difference,
        _numeric(d[family + "_total_minus_" + suffix], family),
        "COMPARISON:" + family,
    )
    return difference


def _family_row(family, receipt, total, difference, slots, age, assumptions):
    lower, upper = np.nan, np.nan
    route_amounts = {
        route: fsum(
            s["source_known_amount"]
            for s in slots
            if s["route"] == route and s["source_amount_known"]
        )
        for route in (
            "candidate",
            "railroad",
            "property",
            "annuity_overlap",
            "other_compensation",
            "unresolved",
        )
    }
    usable = all(
        s["source_amount_known"] or s["structural_zero_comparison"] for s in slots
    )
    no = (
        receipt == 2
        and total == 0
        and all(s["reporting_status"] == "known_nonreceipt" for s in slots)
    )
    contradiction = (
        any("contradictory" in s["reporting_status"] for s in slots)
        or (np.isfinite(difference) and difference < 0)
        or (receipt == 2 and np.isfinite(total) and total != 0)
    )
    if age < 15:
        status = (
            "contradictory_outside_reporting_universe"
            if contradiction or (np.isfinite(total) and total != 0)
            else "unresolved_outside_reporting_universe"
            if not np.isfinite(total)
            or any(s["reporting_status"] != "outside_reporting_universe" for s in slots)
            else "outside_reporting_universe"
        )
    elif contradiction:
        status = "contradictory_accounting"
    elif no:
        lower = upper = 0.0
        status = "known_nonreceipt"
    elif (
        not usable
        or not np.isfinite(total)
        or not np.isfinite(difference)
        or receipt != 1
        or total <= 0
    ):
        status = "unresolved_source_accounting"
    elif difference > 0:
        status = "additional_scope_unresolved"
    elif family == "survivor":
        status = "additional_survivor_scope_unresolved"
    else:
        excluded = route_amounts["railroad"] + route_amounts["other_compensation"]
        upper = total - excluded
        _require(
            upper >= 0
            and upper == route_amounts["candidate"] + route_amounts["unresolved"],
            "CANDIDATE_ACCOUNTING",
        )
        assumed = (
            assumptions.pension_annuity_regularity == "assume_regular"
            if family == "pension"
            else assumptions.disability_pension_eligibility == "assume_qualifying"
        )
        lower = route_amounts["candidate"] if assumed else 0.0
        status = (
            "route_outside_only"
            if upper == 0
            else "candidate_under_assumptions"
            if assumed
            else "candidate_regularity_or_scope_unresolved"
        )
    return {
        family + "_candidate_lower": lower,
        family + "_candidate_upper": upper,
        family + "_status": status,
        family + "_accounting_difference": difference,
        **{
            family + "_observed_" + route + "_subtotal": amount
            for route, amount in route_amounts.items()
        },
    }


def _distributions(r, published, ages, differences):
    total, status = _known_routing(r, "retirement_distribution")
    receipt, receipt_literal = _codes(
        r, "retirement_distribution_receipt", routing.RECEIPT_CODE_DOMAIN
    )
    codes = {
        slot: _codes(
            r, "retirement_distribution_" + slot + "_account", routing.ACCOUNT_CODES
        )
        for slot, *_ in DISTRIBUTION_SLOTS
    }
    raw_status = {
        slot: _text(r, "retirement_distribution_" + slot + "_slot_status")
        for slot, *_ in DISTRIBUTION_SLOTS
    }
    offroute_receipts = {
        name: _codes(
            r, "retirement_distribution_receipt_" + name, routing.RECEIPT_CODE_DOMAIN
        )
        for name in ("young", "58")
    }
    roster, rows = [], []
    for i, age in enumerate(ages):
        known_composition, account_amounts, sum_applicable = (
            True,
            {code: 0.0 for code in routing.ACCOUNT_CODES if code != 0},
            0.0,
        )
        offroute = False
        for slot_number, (slot, field, young) in enumerate(DISTRIBUTION_SLOTS, 1):
            applicable = bool((age < 58) == young)
            declared = r["retirement_distribution_" + slot + "_applicable"].iloc[i]
            _require(
                isinstance(declared, (bool, np.bool_)) and bool(declared) == applicable,
                "DISTRIBUTION_AGE_ROUTE",
            )
            code, literal = codes[slot][0][i], codes[slot][1][i]
            value = published[field][i]
            active = (
                applicable
                and literal == "in_printed_range"
                and code > 0
                and np.isfinite(value)
                and value > 0
                and raw_status[slot][i] == "known_slot"
            )
            niu = (
                applicable
                and code == 0
                and literal == "in_printed_range"
                and value == 0
                and raw_status[slot][i] == "niu_slot"
            )
            if applicable:
                known_composition &= bool(active or niu)
                sum_applicable += value
                if active:
                    account_amounts[int(code)] += value
            else:
                offroute |= not bool(
                    np.isfinite(value)
                    and value == 0
                    and literal == "in_printed_range"
                    and code == 0
                )
            roster.append(
                {
                    "position": i,
                    "family": "distribution",
                    "slot": slot_number,
                    "amount_field": field,
                    "source_code": code,
                    "source_literal_status": literal,
                    "published_amount": value,
                    "reporting_status": raw_status[slot][i],
                    "source_known_amount": value if active else np.nan,
                    "source_amount_known": bool(active),
                    "route": "withdrawal_regularity_netting_unresolved"
                    if active
                    else "unused_or_unresolved",
                    "applicable": applicable,
                    "structural_zero_comparison": bool(niu),
                }
            )
        other = "young" if age >= 58 else "58"
        other_receipt, other_literal = offroute_receipts[other]
        offroute |= not bool(
            other_literal[i] == "in_printed_range" and other_receipt[i] == 0
        )
        if np.isfinite(total[i]) and known_composition:
            _require(total[i] == sum_applicable, "DISTRIBUTION_TOTAL")
        eligible = (
            age >= 15
            and known_composition
            and not offroute
            and np.isfinite(total[i])
            and receipt_literal[i] == "in_printed_range"
            and receipt[i] in (1, 2)
        )
        if eligible:
            _require(
                (receipt[i] == 1 and total[i] > 0)
                or (receipt[i] == 2 and total[i] == 0),
                "DISTRIBUTION_RECEIPT",
            )
        candidate_status = (
            "known_nonreceipt"
            if eligible and total[i] == 0
            else "regularity_netting_unresolved"
            if eligible
            else "unresolved_source_composition"
        )
        if age >= 58 and (not np.isfinite(differences[i]) or differences[i] != 0):
            eligible = False
            candidate_status = (
                "unresolved_source_accounting"
                if not np.isfinite(differences[i])
                else "contradictory_accounting"
                if differences[i] < 0
                else "additional_scope_unresolved"
            )
        rows.append(
            {
                "distribution_candidate_lower": 0.0 if eligible else np.nan,
                "distribution_candidate_upper": total[i] if eligible else np.nan,
                "distribution_status": candidate_status,
                "distribution_source_known_amount": total[i],
                "distribution_source_reporting_status": status[i],
                "distribution_account_composition_known": bool(
                    known_composition and not offroute
                ),
                **{
                    "distribution_account_" + str(c) + "_amount": v
                    if known_composition and not offroute and age >= 15
                    else np.nan
                    for c, v in account_amounts.items()
                },
            }
        )
    return rows, roster


def _other_income(r, age):
    amount, status = _known_routing(r, "other_income")
    receipt, _ = _codes(r, "other_income_receipt", routing.RECEIPT_CODE_DOMAIN)
    category, _ = _codes(r, "other_income_category", routing.OTHER_INCOME_CATEGORIES)
    route = _text(r, "other_income_routing_status")
    no = (
        (receipt == 2)
        & (category == 0)
        & (amount == 0)
        & (status == "known_nonreceipt")
        & (route == "niu_category")
    )
    yes = (
        (receipt == 1)
        & (category > 0)
        & (amount > 0)
        & (status == "known_receipt")
        & (route == "reported_category")
    )
    clear = (age >= 15) & (
        no
        | (yes & ~np.isin(category, (*OTHER_RETIREMENT_CANDIDATES, OTHER_UNSPECIFIED)))
    )
    return (
        amount,
        clear,
        yes & np.isin(category, tuple(OTHER_RETIREMENT_CANDIDATES)),
        yes & (category == OTHER_UNSPECIFIED),
        yes & (category == 1),
        yes & (category == 8),
    )


def _summary(index, membership, weights, masks):
    _require(
        type(membership) is pd.Series
        and membership.index.equals(index)
        and membership.index.name == "person_id"
        and membership.dtype == np.dtype("int64"),
        "HOUSEHOLD_MEMBERSHIP",
    )
    _require(
        type(weights) is pd.Series
        and weights.index.dtype == np.dtype("int64")
        and weights.index.name == "household_id"
        and weights.index.is_unique
        and set(weights.index) == set(membership),
        "DESIGN_MEMBERSHIP",
    )
    values = _numeric(weights, "DESIGN_WEIGHTS")
    _require(np.isfinite(values).all() and (values >= 0).all(), "DESIGN_WEIGHTS")
    design = weights.loc[membership].to_numpy(dtype="float64")
    rows = []
    for name, mask in masks.items():
        households = membership.iloc[np.flatnonzero(mask)].unique()
        person_mass = fsum(design[mask])
        household_mass = fsum(weights.loc[households].to_numpy(dtype="float64"))
        _require(
            np.isfinite(person_mass) and np.isfinite(household_mass), "DESIGN_MASS"
        )
        rows.append(
            (name, int(mask.sum()), len(households), person_mass, household_mass)
        )
    return pd.DataFrame(
        rows,
        columns=(
            "selection",
            "person_count",
            "household_count",
            "design_weighted_person_mass",
            "union_household_design_mass",
        ),
    ).set_index("selection")


def build_asec_retirement_basis(
    *,
    retirement_detail: pd.DataFrame,
    income_routing: pd.DataFrame,
    original_household_membership: pd.Series,
    original_household_design_weights: pd.Series,
    assumptions: RetirementCandidateAssumptions,
) -> AsecRetirementBasis:
    """Describe aligned original ASEC source evidence; perform no I/O or fitting.

    Every bound is a conditional candidate accounting result, never a fiscal
    label. Positive survivor receipt and nonzero aggregate differences remain
    unresolved. DESIGN Series custody and complete-person admission stay external.
    """
    _require(type(assumptions) is RetirementCandidateAssumptions, "ASSUMPTIONS_TYPE")
    payload = assumptions.to_bytes()
    d, r = retirement_detail, income_routing
    age = _axis(d)
    other_age = _axis(r)
    _require(
        d.index.equals(r.index) and d.native_person_id.equals(r.native_person_id),
        "SOURCE_ALIGNMENT",
    )
    _equal(age, other_age, "AGE")
    _require(len(d) > 0, "EMPTY_CANDIDATE_SCOPE")
    published = _published(d)
    _reference_agreement(d, r, published, age)
    records = _detail_slots(d, published, age)
    by_person = {family: [[] for _ in range(len(d))] for family in records}
    for family, entries in records.items():
        for entry in entries:
            by_person[family][entry["position"]].append(entry)
    differences = {
        family: _comparison(d, published, family, total, first, second)
        for family, total, first, second in detail.COMPARISONS
    }
    distribution, distribution_records = _distributions(
        r, published, age, differences["distribution"]
    )
    annuity, annuity_status = _known_routing(r, "pension_annuity_annuity")
    other_amount, other_clear, oi_candidate, oi_unspecified, oi_ss, oi_property = (
        _other_income(r, age)
    )
    receipts = {
        family: _codes(d, prefix + "_YN", routing.RECEIPT_CODE_DOMAIN)[0]
        for family, prefix, _ in FAMILIES
    }
    rows = []
    for i in range(len(d)):
        row = {
            "native_person_id": int(d.native_person_id.iloc[i]),
            "source_age": int(age[i]),
        }
        for family, _, total in FAMILIES:
            slots = by_person[family][i]
            row.update(
                _family_row(
                    family,
                    receipts[family][i],
                    published[total][i],
                    differences[family][i],
                    slots,
                    age[i],
                    assumptions,
                )
            )
        known_annuity = age[i] >= 15 and np.isfinite(annuity[i])
        row.update(
            annuity_candidate_lower=(
                annuity[i]
                if assumptions.pension_annuity_regularity == "assume_regular"
                else 0.0
            )
            if known_annuity
            else np.nan,
            annuity_candidate_upper=annuity[i] if known_annuity else np.nan,
            annuity_status=annuity_status[i],
            other_income_reported_amount=other_amount[i],
            other_income_scope_clear=bool(other_clear[i]),
            distribution_accounting_difference=differences["distribution"][i],
            **distribution[i],
        )
        families = ("pension", "disability", "survivor", "annuity", "distribution")
        available = (
            age[i] >= 15
            and other_clear[i]
            and all(
                np.isfinite(row[f + "_candidate_lower"])
                and np.isfinite(row[f + "_candidate_upper"])
                for f in families
            )
        )
        lower = (
            fsum(row[f + "_candidate_lower"] for f in families) if available else np.nan
        )
        upper = (
            fsum(row[f + "_candidate_upper"] for f in families) if available else np.nan
        )
        row.update(
            retirement_candidate_lower=lower,
            retirement_candidate_upper=upper,
            candidate_interval_available=bool(available),
            candidate_point_under_assumptions=bool(available and lower == upper),
            candidate_basis_eligible=bool(available),
            point_identified=False,
            fiscal_outputs_produced=False,
        )
        rows.append(row)
    person = pd.DataFrame(rows, index=d.index.copy())
    all_records = [
        s.copy() for family in records.values() for s in family
    ] + distribution_records
    for item in all_records:
        i = item.pop("position")
        item["person_id"] = int(d.index[i])
        item["native_person_id"] = int(d.native_person_id.iloc[i])
        item["source_age"] = int(age[i])
    slots = (
        pd.DataFrame(all_records)
        .sort_values(["person_id", "family", "slot"], kind="stable")
        .reset_index(drop=True)
    )
    exclusions = pd.DataFrame(
        {
            "under15": age < 15,
            "survivor_additional_sources_unresolved": (age >= 15)
            & (receipts["survivor"] == 1),
            **{
                family + "_reported_" + route: person[
                    family + "_observed_" + route + "_subtotal"
                ].to_numpy()
                > 0
                for family in records
                for route in (
                    "railroad",
                    "property",
                    "annuity_overlap",
                    "other_compensation",
                    "unresolved",
                )
            },
            "other_income_scope_unresolved": ~other_clear,
            "other_income_retirement_candidate": oi_candidate,
            "other_income_unspecified": oi_unspecified,
            "other_income_ss_overlap": oi_ss,
            "other_income_property_overlap": oi_property,
            "survivor_annuity_overlap": (
                person.survivor_observed_annuity_overlap_subtotal.to_numpy() > 0
            )
            & (annuity > 0),
            **{
                family + "_candidate_unresolved": ~np.isfinite(
                    person[family + "_candidate_upper"].to_numpy()
                )
                for family in (
                    "pension",
                    "disability",
                    "survivor",
                    "annuity",
                    "distribution",
                )
            },
            **{
                family + "_aggregate_nonzero_or_unknown": ~np.isfinite(diff)
                | (diff != 0)
                for family, diff in differences.items()
            },
        },
        index=d.index.copy(),
    )
    provenance = pd.concat(
        [
            d.copy(deep=True).add_prefix("retirement_detail."),
            r.copy(deep=True).add_prefix("income_routing."),
        ],
        axis=1,
    )
    masks = {
        "all": np.ones(len(d), dtype=bool),
        **{
            name: person[name].to_numpy()
            for name in (
                "candidate_basis_eligible",
                "candidate_interval_available",
                "candidate_point_under_assumptions",
            )
        },
        **{"diagnostic:" + name: exclusions[name].to_numpy() for name in exclusions},
    }
    summary = _summary(
        d.index, original_household_membership, original_household_design_weights, masks
    )
    return AsecRetirementBasis(person, slots, provenance, exclusions, summary, payload)
