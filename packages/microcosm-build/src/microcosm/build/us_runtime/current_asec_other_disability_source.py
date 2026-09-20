"""Qualified ASEC other-disability income from the two published source slots.

This borrows the retirement-detail owner's already-qualified DIS literals and
the retired eCPS two-slot, non-workers-compensation arithmetic. It adds no raw
reader: the member capture and its requalification stay with that owner, which
the public entry point calls. No source authority is issued and nothing is
completed: NIU, missing, under-15 and contradictory slots stay unknown rather
than becoming an observed zero, and a non-ASEC arm carries no observation. Social Security
disability and workers' compensation are other leaves, not this one. A host
retains and requalifies the actual preparation and owns every completion,
model, tax treatment and release decision.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import FunctionType, MappingProxyType, ModuleType

import numpy as np
import pandas as pd

from . import asec_current_money_source as physical
from . import current_asec_income_routing_source as routing
from . import current_asec_retirement_detail_source as detail
from . import disability_benefits as legacy
from .support_provenance import (
    support_channel_column,
    support_clone_index_column,
    support_source_id_column,
)

PROTOCOL = "microcosm.us.current-asec-other-disability-source.v1"
# The archived derivation's own parameters decide this roster; a drift in the
# retired implementation fails here instead of being silently reinterpreted.
_ARCHIVED = legacy._EXPECTED_DIRECT_PARAMETERS
OUTPUT = _ARCHIVED["output"]
SLOTS = (
    (1, _ARCHIVED["first_amount_source"], _ARCHIVED["first_code_source"]),
    (2, _ARCHIVED["second_amount_source"], _ARCHIVED["second_code_source"]),
)
AMOUNT_FIELDS = tuple(amount for _, amount, _ in SLOTS)
SOURCE_CODE_FIELDS = tuple(code for _, _, code in SLOTS)
WORKERS_COMPENSATION_CODE = int(_ARCHIVED["workers_compensation_code"])
WORKERS_COMPENSATION_LABEL = "worker's compensation"
RECEIPT_FIELD = "DIS_YN"
RECEIPT_UNIVERSE = "All Persons aged 15+"
SOURCE_CODE_UNIVERSE = "DIS_YN = 1"
REPORTING_AGE = 15
# Published allocation flags for the income slots only. I_DISCS and I_DISHP
# flag the work-limitation answers the detail owner keeps beside these; those
# describe leaving or limiting work, not this leaf's dollars.
ALLOCATION_FIELDS = ("I_DISYN", "I_DISSC1", "I_DISSC2", "I_DISVL1", "I_DISVL2")
TOPCODE_FIELDS = ("TDISVAL1", "TDISVAL2")
PARENT_AXES = ("statuses", "validity", "zero_origin")

AMOUNT_COLUMN = "other_disability_amount"
KNOWN_COLUMN = "other_disability_known"
REASON_COLUMN = "other_disability_reason"
REPORT_PREFIX = "survey_other_disability_"
ASEC_CHANNEL = "asec"
# Arms this source never observes. An ACS row is unobserved, never a zero.
UNOBSERVED_CHANNELS = ("acs",)
UNOBSERVED_REASON = "acs_source_unobserved"

SLOT_KINDS = (
    "reported_source_slot",
    "excluded_workers_compensation",
    "unused_source_slot",
    "nonreceipt_slot",
    "niu_not_observed_zero",
    "outside_age_universe",
    "unresolved_slot_reporting",
)
REASONS = (
    "known_positive",
    "known_zero_nonreceipt",
    "known_zero_workers_compensation_only",
    "affirmed_receipt_without_reported_source",
    "niu_not_observed_zero",
    "outside_age_universe",
    "unresolved_slot_reporting",
    UNOBSERVED_REASON,
)
# Owner statuses whose slot reading this leaf can act on. Every other status
# leaves the slot, and therefore the person, unknown.
_ADMITTED_SLOT_STATUSES = MappingProxyType(
    {
        "known_receipt": "reported_source_slot",
        "unreported_source_slot": "unused_source_slot",
        "known_nonreceipt": "nonreceipt_slot",
    }
)
# A published workers' compensation code removes the slot from this leaf
# whatever it paid, but only from the two statuses a coherent slot can carry:
# a yes receipt, a readable code and a readable amount, whether that amount is
# positive or the printed none/niu zero. An unreadable or absent amount cell
# leaves the slot unresolved rather than a known zero, because a record that
# cannot be read here is not evidence that this code was read correctly.
_WORKERS_COMPENSATION_STATUSES = ("known_receipt", "ambiguous_recipient_zero")
_UNDER_AGE_STATUSES = (
    "outside_reporting_universe",
    "contradictory_outside_reporting_universe",
    "unresolved_outside_reporting_universe",
)
_ZERO_CONTRIBUTION_KINDS = (
    "excluded_workers_compensation",
    "unused_source_slot",
    "nonreceipt_slot",
)
_POPULATED_KINDS = ("reported_source_slot", "excluded_workers_compensation")
_ALLOCATION_STATUSES = (
    "unresolved_flag_universe",
    "outside_flag_universe",
    "allocation_flag_not_populated",
    "unresolved_allocation_literal",
    "publisher_allocated",
    "not_allocated_in_flag_universe",
)
# The owner's family reading is universe-blind by charter. This module does
# evaluate each printed universe, so it reports the two scopes separately
# rather than letting a flag read outside its own universe stand as the
# family's allocation answer.
ALLOCATION_FAMILY_STATUSES = (
    "publisher_allocated_in_flag_universe",
    "unresolved_allocation_provenance",
    "allocation_flag_not_populated",
    "not_allocated_in_flag_universe",
    "no_flag_in_its_universe",
)
_FLAG_LABEL_FAMILY_STATUS = MappingProxyType(
    {
        "publisher_allocated": "publisher_allocated_in_flag_universe",
        "unresolved_flag_universe": "unresolved_allocation_provenance",
        "unresolved_allocation_literal": "unresolved_allocation_provenance",
        "allocation_flag_not_populated": "allocation_flag_not_populated",
        "not_allocated_in_flag_universe": "not_allocated_in_flag_universe",
        "outside_flag_universe": "no_flag_in_its_universe",
    }
)


def _require(condition, reason):
    if not condition:
        raise ValueError("OTHER_DISABILITY_SOURCE_" + reason)


def _allocation_entries():
    entries = {entry[0]: entry for entry in detail.ALLOCATION_ENTRIES}
    return {name: entries[name] for name in ALLOCATION_FIELDS if name in entries}


def _check_roster():
    """Bind this leaf to the retired parameters and the live source owner."""
    _require(
        tuple(sorted((*AMOUNT_FIELDS, *SOURCE_CODE_FIELDS)))
        == tuple(sorted(legacy.US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS))
        and OUTPUT == legacy.US_DISABILITY_BENEFITS_OUTPUT_COLUMNS[0]
        and len(set(AMOUNT_FIELDS)) == len(set(SOURCE_CODE_FIELDS)) == len(SLOTS),
        "ARCHIVED_ROSTER",
    )
    _require(
        set(AMOUNT_FIELDS) <= set(detail.RETAINED_MONEY_FIELDS)
        and set(SOURCE_CODE_FIELDS) <= set(detail.SOURCE_ENTRIES)
        and RECEIPT_FIELD in detail.RECEIPT_ENTRIES
        and detail.SOURCE_CODES[RECEIPT_FIELD[:3]] is detail.DISABILITY_CODES,
        "OWNER_ROSTER",
    )
    # The exclusion targets a printed meaning, not a bare integer.
    _require(
        detail.DISABILITY_CODES.get(WORKERS_COMPENSATION_CODE)
        == WORKERS_COMPENSATION_LABEL
        and detail.RECEIPT_ENTRIES[RECEIPT_FIELD].universe_as_printed
        == RECEIPT_UNIVERSE
        and all(
            detail.SOURCE_ENTRIES[name][4] == SOURCE_CODE_UNIVERSE
            for name in SOURCE_CODE_FIELDS
        ),
        "PRINTED_SEMANTICS",
    )
    _require(
        set(_allocation_entries()) == set(ALLOCATION_FIELDS)
        and set(TOPCODE_FIELDS) <= {entry[0] for entry in detail.TOPCODE_ENTRIES},
        "PROVENANCE_ROSTER",
    )


def required_basis_columns():
    """Owner columns this adapter reads; nothing else is consulted."""
    columns = ["source_age"]
    for name in (RECEIPT_FIELD, *SOURCE_CODE_FIELDS):
        columns += [
            name + "_literal",
            name + "_code",
            name + "_literal_status",
            name + "_label",
        ]
    for name in AMOUNT_FIELDS:
        columns += [
            name + "_literal",
            name + "_literal_status",
            name + "_published_amount",
            name + "_reporting_status",
            name + "_amount_known",
            name + "_amount",
        ]
    for name in (*ALLOCATION_FIELDS, *TOPCODE_FIELDS):
        columns += [name + "_literal", name + "_code", name + "_literal_status"]
    return tuple(columns)


def parent_money_columns():
    """Optional retained-money axes the host's qualifier adds to the basis."""
    return tuple(
        name + "_parent_" + axis for name in AMOUNT_FIELDS for axis in PARENT_AXES
    )


def _slot_kind(age, status, receipt_code, source_code):
    """One slot's admissibility for the non-workers-compensation leaf."""
    if age < REPORTING_AGE or status in _UNDER_AGE_STATUSES:
        return "outside_age_universe"
    if (
        receipt_code == 1
        and source_code == WORKERS_COMPENSATION_CODE
        and status in _WORKERS_COMPENSATION_STATUSES
    ):
        return "excluded_workers_compensation"
    if status == "niu":
        return "niu_not_observed_zero"
    return _ADMITTED_SLOT_STATUSES.get(status, "unresolved_slot_reporting")


def _int_or_none(value):
    return None if pd.isna(value) else int(value)


def _float_or_nan(series):
    return series.astype("Float64").to_numpy(dtype="float64", na_value=np.nan)


def _flag_universe(basis, universe):
    """Evaluate one printed allocation universe from the owner's own literals."""
    field, separator, bound = universe.partition(" > ")
    _require(separator == " > " and bound == "0", "ALLOCATION_UNIVERSE:" + universe)
    if field in AMOUNT_FIELDS:
        values = basis[field + "_published_amount"]
    else:
        _require(
            field in (RECEIPT_FIELD, *SOURCE_CODE_FIELDS),
            "ALLOCATION_UNIVERSE_FIELD:" + universe,
        )
        values = basis[field + "_code"]
    return [pd.NA if pd.isna(value) else bool(value > 0) for value in values]


def _allocation_columns(basis, out):
    """Preserve conditional provenance; a flag never qualifies a receipt."""
    flags, flag_labels = {}, {}
    for name, entry in _allocation_entries().items():
        codes = [_int_or_none(value) for value in basis[name + "_code"]]
        statuses = list(basis[name + "_literal_status"])
        flags[name] = (codes, statuses)
        universe = _flag_universe(basis, entry[4])
        labels = []
        for active, code, status in zip(universe, codes, statuses, strict=True):
            if active is pd.NA:
                label = "unresolved_flag_universe"
            elif not active:
                label = "outside_flag_universe"
            elif status == "missing":
                label = "allocation_flag_not_populated"
            elif status != "in_printed_range":
                label = "unresolved_allocation_literal"
            else:
                label = (
                    "publisher_allocated" if code else "not_allocated_in_flag_universe"
                )
            labels.append(label)
        _require(set(labels) <= set(_ALLOCATION_STATUSES), "ALLOCATION_STATUS")
        flag_labels[name] = labels
        out[name + "_literal"] = basis[name + "_literal"].astype("string")
        out[name + "_code"] = basis[name + "_code"].astype("Int16")
        out[name + "_flag_universe"] = pd.array(universe, dtype="boolean")
        out[name + "_allocation_status"] = pd.array(labels, dtype="string")
    # The owner's reading, kept under a name that says what it is: it does not
    # evaluate the printed universes, so a nonzero flag outside its own
    # universe still reads as a publisher allocation here.
    out["other_disability_published_flag_origin"] = routing._allocation_origin(
        flags, unflagged=False
    )
    family = []
    for position in range(len(out)):
        seen = {
            _FLAG_LABEL_FAMILY_STATUS[flag_labels[name][position]]
            for name in flag_labels
        }
        family.append(
            next(status for status in ALLOCATION_FAMILY_STATUSES if status in seen)
        )
    out["other_disability_allocation_status"] = pd.array(family, dtype="string")
    out["other_disability_allocation_qualifies_receipt"] = pd.array(
        [False] * len(out), dtype="boolean"
    )
    return out


def _topcoded(kinds, codes, statuses):
    """Censoring of the dollars this leaf actually admits, or unknown.

    The reading is "some admitted slot is topcoded", so one readable flag of 1
    settles it even when another admitted slot's flag cannot be read. Only an
    unreadable flag with no censoring established elsewhere leaves it unknown.
    """
    unreadable = False
    for kind, code, status in zip(kinds, codes, statuses, strict=True):
        if kind != "reported_source_slot":
            continue
        if status != "in_printed_range":
            unreadable = True
        elif code == 1:
            return True
    return pd.NA if unreadable else False


def _legacy_arithmetic(basis):
    """Run the retired implementation itself wherever all four literals read."""
    evaluable = np.ones(len(basis), dtype=bool)
    columns = {}
    for name in (*AMOUNT_FIELDS, *SOURCE_CODE_FIELDS):
        suffix = "_published_amount" if name in AMOUNT_FIELDS else "_code"
        values = _float_or_nan(basis[name + suffix])
        evaluable &= np.isfinite(values)
        columns[name] = values
    amounts = np.full(len(basis), np.nan, dtype="float64")
    if evaluable.any():
        subset = pd.DataFrame(
            {name: values[evaluable] for name, values in columns.items()},
            index=basis.index[evaluable],
        )
        derived = legacy.derive_us_disability_benefits_from_asec(
            subset,
            first_amount_source=_ARCHIVED["first_amount_source"],
            first_code_source=_ARCHIVED["first_code_source"],
            second_amount_source=_ARCHIVED["second_amount_source"],
            second_code_source=_ARCHIVED["second_code_source"],
            workers_compensation_code=WORKERS_COMPENSATION_CODE,
            output_column=OUTPUT,
        )
        _require(derived.index.equals(subset.index), "ARCHIVED_AXIS")
        amounts[evaluable] = derived[OUTPUT].to_numpy(dtype="float64")
    return evaluable, amounts


def _parent_money_agreement(basis, out):
    """Cross-check the retained money owner's axes where the host supplied them."""
    present = set(parent_money_columns()) <= set(basis.columns)
    out["other_disability_money_owner_axes_present"] = pd.array(
        [present] * len(out), dtype="boolean"
    )
    if not present:
        return out
    for name in AMOUNT_FIELDS:
        validity = basis[name + "_parent_validity"].to_numpy()
        _require(
            np.array_equal(
                validity == 1, basis[name + "_published_amount"].notna().to_numpy()
            ),
            "PARENT_VALIDITY:" + name,
        )
        for axis in PARENT_AXES:
            out[name + "_parent_" + axis] = pd.array(
                basis[name + "_parent_" + axis].to_numpy(dtype="int16"),
                dtype="Int16",
            )
    return out


def project_other_disability(basis):
    """Reduce the owner's qualified DIS slots to one leaf, completing nothing."""
    _check_roster()
    _require(
        type(basis) is pd.DataFrame
        and basis.columns.is_unique
        and basis.index.is_unique
        and set(required_basis_columns()) <= set(basis.columns),
        "BASIS_COLUMNS",
    )
    ages = basis.source_age.to_numpy(dtype="int64")
    receipts = [_int_or_none(value) for value in basis[RECEIPT_FIELD + "_code"]]
    out = pd.DataFrame(index=basis.index.copy())
    # Every column is nullable, so one clone transport and one seal read the
    # same physical storage whether or not a row carries an observation.
    out["source_age"] = pd.array(ages, dtype="Int64")
    for name in (RECEIPT_FIELD, *SOURCE_CODE_FIELDS):
        out[name + "_literal"] = basis[name + "_literal"].astype("string")
        out[name + "_code"] = basis[name + "_code"].astype("Int16")
        out[name + "_literal_status"] = basis[name + "_literal_status"].astype("string")
        out[name + "_label"] = basis[name + "_label"].astype("string")
    kinds, contributions, topcodes = {}, {}, {}
    for _, amount_field, code_field in SLOTS:
        published = _float_or_nan(basis[amount_field + "_published_amount"])
        statuses = list(basis[amount_field + "_reporting_status"])
        known = basis[amount_field + "_amount_known"].to_numpy(dtype=bool)
        codes = [_int_or_none(value) for value in basis[code_field + "_code"]]
        slot_kinds, slot_values = [], np.full(len(basis), np.nan, dtype="float64")
        for i, status in enumerate(statuses):
            kind = _slot_kind(ages[i], status, receipts[i], codes[i])
            # The owner's own knownness must agree with the status it published.
            _require(
                known[i] == (status in routing.KNOWN_AMOUNT_STATUSES),
                "OWNER_KNOWNNESS:" + amount_field,
            )
            if kind == "reported_source_slot":
                _require(
                    known[i]
                    and np.isfinite(published[i])
                    and published[i] > 0
                    and codes[i] not in (None, 0, WORKERS_COMPENSATION_CODE),
                    "REPORTED_SLOT_EVIDENCE:" + amount_field,
                )
                slot_values[i] = published[i]
            elif kind in _ZERO_CONTRIBUTION_KINDS:
                slot_values[i] = 0.0
            slot_kinds.append(kind)
        topcode_field = TOPCODE_FIELDS[AMOUNT_FIELDS.index(amount_field)]
        kinds[amount_field] = slot_kinds
        contributions[amount_field] = slot_values
        topcodes[amount_field] = (
            [_int_or_none(value) for value in basis[topcode_field + "_code"]],
            list(basis[topcode_field + "_literal_status"]),
        )
        out[amount_field + "_literal"] = basis[amount_field + "_literal"].astype(
            "string"
        )
        out[amount_field + "_literal_status"] = basis[
            amount_field + "_literal_status"
        ].astype("string")
        out[amount_field + "_published_amount"] = basis[
            amount_field + "_published_amount"
        ].astype("Float64")
        out[amount_field + "_reporting_status"] = pd.array(statuses, dtype="string")
        out[amount_field + "_amount_known"] = pd.array(known, dtype="boolean")
        out[amount_field + "_slot_kind"] = pd.array(slot_kinds, dtype="string")
        out[amount_field + "_slot_contribution"] = pd.array(
            slot_values, dtype="Float64"
        )
        out[topcode_field + "_literal"] = basis[topcode_field + "_literal"].astype(
            "string"
        )
        out[topcode_field + "_code"] = basis[topcode_field + "_code"].astype("Int16")
    amounts = np.full(len(basis), np.nan, dtype="float64")
    known_leaf, reasons, censored = np.zeros(len(basis), dtype=bool), [], []
    for i in range(len(basis)):
        row = [kinds[name][i] for name in AMOUNT_FIELDS]
        values = [contributions[name][i] for name in AMOUNT_FIELDS]
        populated = sum(kind in _POPULATED_KINDS for kind in row)
        if not all(np.isfinite(value) for value in values):
            if "outside_age_universe" in row:
                reason = "outside_age_universe"
            elif "niu_not_observed_zero" in row:
                reason = "niu_not_observed_zero"
            else:
                reason = "unresolved_slot_reporting"
        elif receipts[i] == 1 and populated == 0:
            # A yes answer with no populated source is not a zero either.
            reason = "affirmed_receipt_without_reported_source"
        else:
            _require(receipts[i] in (1, 2), "RESOLVED_RECEIPT")
            total = float(sum(values))
            amounts[i] = total
            known_leaf[i] = True
            if total > 0:
                reason = "known_positive"
            elif receipts[i] == 2:
                reason = "known_zero_nonreceipt"
            else:
                _require(
                    "excluded_workers_compensation" in row, "RESOLVED_ZERO_EVIDENCE"
                )
                reason = "known_zero_workers_compensation_only"
        reasons.append(reason)
        censored.append(
            _topcoded(
                row,
                [topcodes[name][0][i] for name in AMOUNT_FIELDS],
                [topcodes[name][1][i] for name in AMOUNT_FIELDS],
            )
            if known_leaf[i]
            else pd.NA
        )
    _require(set(reasons) <= set(REASONS), "REASON_VOCABULARY")
    out[AMOUNT_COLUMN] = pd.array(amounts, dtype="Float64")
    out[KNOWN_COLUMN] = pd.array(known_leaf, dtype="boolean")
    out[REASON_COLUMN] = pd.array(reasons, dtype="string")
    out["other_disability_topcoded"] = pd.array(censored, dtype="boolean")
    out = _allocation_columns(basis, out)
    out = _parent_money_agreement(basis, out)
    evaluable, archived = _legacy_arithmetic(basis)
    out["other_disability_archived_arithmetic_evaluable"] = pd.array(
        evaluable, dtype="boolean"
    )
    out["other_disability_archived_arithmetic_amount"] = pd.array(
        archived, dtype="Float64"
    )
    out["other_disability_archived_arithmetic_agrees"] = pd.array(
        [
            (amounts[i] == archived[i]) if known_leaf[i] and evaluable[i] else pd.NA
            for i in range(len(basis))
        ],
        dtype="boolean",
    )
    # Explicit non-claims travel with every row, not only with the evidence.
    for name in (
        "other_disability_social_security_included",
        "other_disability_workers_compensation_included",
        "other_disability_completed_with_zero",
    ):
        out[name] = pd.array([False] * len(out), dtype="boolean")
    if "native_person_id" in basis:
        out["native_person_id"] = basis.native_person_id.to_numpy(copy=True)
    return out


@dataclass(frozen=True)
class CurrentAsecOtherDisabilityValues:
    person: pd.DataFrame
    evidence: dict


@dataclass(frozen=True)
class OtherDisabilityAttachment:
    columns: dict
    receipt: dict


def other_disability_values_seal(values):
    """Physical value seal, including nullable backing storage and exact bits."""
    _require(type(values) is CurrentAsecOtherDisabilityValues, "VALUES_TYPE")
    table = values.person
    _require(type(table) is pd.DataFrame and table.columns.is_unique, "TABLE_TYPE")
    digest = hashlib.sha256(PROTOCOL.encode())
    digest.update(
        physical._json(
            {
                "columns": list(table.columns),
                "columns_axis": physical.checkpoint._index_spec(
                    table.columns, label="other disability columns"
                ),
                "index": physical.checkpoint._index_spec(
                    table.index, label="other disability"
                ),
            }
        )
    )
    physical._series_digest(
        digest, pd.Series(table.index.to_numpy(copy=False), dtype=table.index.dtype)
    )
    for column in table:
        series = table[column]
        if isinstance(series.dtype, pd.Float64Dtype):
            # The checkpoint digest deliberately excludes Float64 extension
            # arrays. Keep the nullable tag and hash both physical arrays,
            # including the values that sit under null masks.
            digest.update(physical._json({"dtype": "Float64", "nullable": True}))
            physical._series_digest(digest, pd.Series(series.array._data))
            physical._series_digest(digest, pd.Series(series.array._mask))
        else:
            physical._series_digest(digest, series)
    digest.update(physical._json(values.evidence))
    return digest.hexdigest()


def _module_constants():
    """Every module-level constant, so none can be retuned after import.

    Collected rather than listed, so a constant added later is bound without
    anyone remembering to extend this seal.
    """
    # Detached, not aliased: a constant that is a live mapping owned by another
    # module would otherwise be snapshotted by reference, so mutating it in
    # place would mutate this seal with it and pass its own equality check.
    return {
        name: copy.deepcopy(
            dict(value) if isinstance(value, MappingProxyType) else value
        )
        for name, value in vars(sys.modules[__name__]).items()
        if name != "_LIVE"
        and name.isupper()
        and not isinstance(value, (FunctionType, ModuleType, type))
    }


def _live():
    return (
        {
            name: routing.source._function_seal(value)
            for name, value in vars(sys.modules[__name__]).items()
            if isinstance(value, FunctionType)
        },
        _module_constants(),
        CurrentAsecOtherDisabilityValues,
        OtherDisabilityAttachment,
        # The retired arithmetic and its archived parameters stay bound: this
        # leaf may not be reinterpreted by editing either owner in place.
        legacy,
        dict(legacy._EXPECTED_DIRECT_PARAMETERS),
        legacy.US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS,
        legacy.US_DISABILITY_BENEFITS_OUTPUT_COLUMNS,
        legacy.DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL,
        legacy.DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL,
        tuple(
            routing.source._function_seal(function)
            for function in (
                legacy.derive_us_disability_benefits_from_asec,
                legacy._strict_numeric_source,
            )
        ),
        # The qualified source owner, its printed semantics and its seal.
        detail,
        detail.PROTOCOL,
        dict(detail.DISABILITY_CODES),
        dict(detail.SOURCE_ENTRIES),
        {name: tuple(entry) for name, entry in detail.RECEIPT_ENTRIES.items()},
        detail.ALLOCATION_ENTRIES,
        detail.TOPCODE_ENTRIES,
        detail.RETAINED_MONEY_FIELDS,
        detail.CurrentAsecRetirementDetailValues,
        # The owner's qualifier supplies the basis this module interprets, and
        # its projection decides the status vocabulary the slot kinds map from.
        # Both are bound: swapping either could hand over doctored amounts or
        # relabel an unknown slot as an observation.
        tuple(
            routing.source._function_seal(function)
            for function in (
                detail.qualify_current_asec_retirement_detail,
                detail.project_retirement_detail_literals,
                detail.retirement_detail_values_seal,
                detail._slot_status,
                detail._amount_pair,
                # The owner cross-checks the retained DIS amounts against the
                # money owner, but the source codes, receipt and flag literals
                # this leaf reads come only from the capture, so the capture
                # and comparison path is bound here too.
                detail._capture_member,
                detail._compare_amount,
            )
        ),
        routing,
        routing.CURRENT_INCOME_YEAR,
        routing.DICTIONARY_URL,
        routing.DICTIONARY_SHA256,
        routing.KNOWN_AMOUNT_STATUSES,
        tuple(
            routing.source._function_seal(function)
            for function in (
                routing._allocation_origin,
                routing.receipt_status,
                routing.literal_code,
                routing._codes_frame,
                routing._read_capture,
                routing._file_sha,
                routing._sha,
                routing.coverage._capture,
                routing.coverage._identity,
                routing.source._function_seal,
                routing.source._pure_final,
                physical._series_digest,
            )
        ),
    )


def _evidence(person, detail_values, seal):
    allocation = _allocation_entries()
    return json.loads(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "canonical_leaf": OUTPUT,
                "concept": (
                    "annual income from the two published ASEC disability source "
                    "slots, excluding workers' compensation; Social Security "
                    "disability is a separate leaf and is not included"
                ),
                "archived_derivation_url": (
                    legacy.DISABILITY_BENEFITS_ARCHIVED_DERIVATION_URL
                ),
                "archived_source_columns_url": (
                    legacy.DISABILITY_BENEFITS_ARCHIVED_SOURCE_COLUMNS_URL
                ),
                "archived_parameters": dict(_ARCHIVED),
                "required_source_columns": (
                    legacy.US_DISABILITY_BENEFITS_REQUIRED_SOURCE_COLUMNS
                ),
                "workers_compensation_code": WORKERS_COMPENSATION_CODE,
                "workers_compensation_label": WORKERS_COMPENSATION_LABEL,
                "source_codes": dict(detail.DISABILITY_CODES),
                "receipt_entry": detail.RECEIPT_ENTRIES[RECEIPT_FIELD]._asdict(),
                "receipt_universe": RECEIPT_UNIVERSE,
                "source_code_entries": {
                    name: detail.SOURCE_ENTRIES[name] for name in SOURCE_CODE_FIELDS
                },
                "source_code_universe": SOURCE_CODE_UNIVERSE,
                "allocation_entries": {
                    name: entry[1:] for name, entry in allocation.items()
                },
                "allocation_excluded_work_limitation_flags": ["I_DISCS", "I_DISHP"],
                "allocation_family_statuses": ALLOCATION_FAMILY_STATUSES,
                "published_flag_origin_scope": (
                    "the owner's universe-blind reading of the published flags; "
                    "other_disability_allocation_status evaluates each printed "
                    "conditional universe instead"
                ),
                "topcode_entries": [
                    entry
                    for entry in detail.TOPCODE_ENTRIES
                    if entry[0] in TOPCODE_FIELDS
                ],
                "slot_kinds": SLOT_KINDS,
                "reasons": REASONS,
                "dictionary_url": routing.DICTIONARY_URL,
                "dictionary_sha256": routing.DICTIONARY_SHA256,
                "income_year": routing.CURRENT_INCOME_YEAR,
                "survey_year": routing.CURRENT_INCOME_YEAR + 1,
                "rows": len(person),
                "known_rows": int(person[KNOWN_COLUMN].sum()),
                "reason_counts": {
                    reason: int((person[REASON_COLUMN] == reason).sum())
                    for reason in REASONS
                    if reason != UNOBSERVED_REASON
                },
                "retirement_detail_protocol": detail.PROTOCOL,
                "retirement_detail_values_sha256": seal,
                "retirement_detail_evidence": detail_values.evidence,
                "implementation_sha256": _IMPLEMENTATION_SHA256,
                "implementation_sha256_scope": (
                    "this module's file as read at the end of its own import; a "
                    "later edit refuses rather than being reported, but the "
                    "window between the loader's read and this one is not "
                    "attested"
                ),
                "acs_completion_assigned": False,
                "under15_completed_with_zero": False,
                "niu_completed_with_zero": False,
                "unknown_completed_with_zero": False,
                "social_security_disability_included": False,
                "workers_compensation_included": False,
                "ssi_eligibility_assigned": False,
                "taxability_assigned": False,
                "topcode_corrected": False,
                "allocation_flags_qualify_receipt": False,
                "model_fitted": False,
                "clone_redraw_issued": False,
                "adds_raw_member_reader": False,
                "member_capture_owner": detail.PROTOCOL,
                "source_admission_issued": False,
                "release_eligible": False,
            }
        )
    )


def _implementation_unchanged():
    """The loaded functions and the file on disk must still be one thing."""
    return (
        _live() == _LIVE
        and hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        == _IMPLEMENTATION_SHA256
    )


def compose_other_disability(detail_values):
    """Borrow the qualified detail owner and return detached descriptive values."""
    _require(_implementation_unchanged(), "IMPLEMENTATION_CHANGED")
    _require(
        type(detail_values) is detail.CurrentAsecRetirementDetailValues,
        "DETAIL_OWNER_TYPE",
    )
    seal = detail.retirement_detail_values_seal(detail_values)
    evidence = detail_values.evidence
    _require(
        type(evidence) is dict
        and evidence.get("protocol") == detail.PROTOCOL
        and evidence.get("source_year") == routing.CURRENT_INCOME_YEAR
        and evidence.get("survey_year") == routing.CURRENT_INCOME_YEAR + 1
        and evidence.get("source_admission_issued") is False,
        "DETAIL_EVIDENCE",
    )
    _require("native_person_id" in detail_values.person, "DETAIL_NATIVE_AXIS")
    person = project_other_disability(detail_values.person)
    result = CurrentAsecOtherDisabilityValues(
        person, _evidence(person, detail_values, seal)
    )
    stamp = other_disability_values_seal(result)
    _require(
        detail.retirement_detail_values_seal(detail_values) == seal,
        "DETAIL_VALUES_CHANGED",
    )
    _require(_implementation_unchanged(), "IMPLEMENTATION_CHANGED")
    _require(other_disability_values_seal(result) == stamp, "FINAL_VALUES_CHANGED")
    return result


def qualify_current_asec_other_disability(preparation):
    """Requalify the actual retained preparation through its own detail owner."""
    source = routing.source
    _require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state, native = entry[2], entry[2].native[1]
    issued = source.asec_native._ISSUED.get(id(native))
    _require(issued is not None, "NATIVE_ISSUANCE")
    result = compose_other_disability(
        detail.qualify_current_asec_retirement_detail(preparation)
    )
    seal = other_disability_values_seal(result)
    source._pure_final(state)
    _require(
        preparation._checked() is entry
        and source._ISSUED.get(id(preparation)) is entry
        and source.asec_native._ISSUED.get(id(native)) is issued
        and native.payload == issued[1]
        and other_disability_values_seal(result) == seal,
        "FINAL_OWNER_OR_VALUES",
    )
    return result


def attached_name(name):
    """Canonical leaf name for the amount; a source report prefix otherwise."""
    if name == AMOUNT_COLUMN:
        return OUTPUT
    return REPORT_PREFIX + name.removeprefix("other_disability_")


def _entity_table(frame, name):
    return frame.table(name) if hasattr(frame, "table") else getattr(frame, name)


def attach_other_disability_columns(values, receiving):
    """Copy one qualified source row to every clone of that person; never redraw.

    The transport is an identity join on the pre-clone source id, so both
    clones of a source person carry bit-identical values and no draw is taken.
    Rows on an arm this source never observed stay unknown rather than zero.
    """
    _require(_implementation_unchanged(), "IMPLEMENTATION_CHANGED")
    _require(type(values) is CurrentAsecOtherDisabilityValues, "VALUES_TYPE")
    seal = other_disability_values_seal(values)
    basis = values.person
    _require(
        "native_person_id" in basis
        and basis.index.is_unique
        and basis.index.dtype == np.dtype("int64")
        and basis.native_person_id.dtype == np.dtype("int64"),
        "VALUES_AXIS",
    )
    people = _entity_table(receiving, "person")
    source_column = support_source_id_column("person")
    clone_column = support_clone_index_column("person")
    native_column = routing.spine_source_id_column("person")
    channel_column = support_channel_column("person")
    _require(
        type(people) is pd.DataFrame
        and people.columns.is_unique
        and {
            "person_id",
            source_column,
            clone_column,
            native_column,
            channel_column,
        }
        <= set(people.columns)
        and people.person_id.is_unique
        and all(
            people[column].dtype == np.dtype("int64")
            for column in ("person_id", source_column, clone_column, native_column)
        ),
        "RECEIVING_AXES",
    )
    channel = people[channel_column].to_numpy()
    _require(
        set(channel) <= {ASEC_CHANNEL, *UNOBSERVED_CHANNELS}, "RECEIVING_CHANNEL_ROSTER"
    )
    asec = channel == ASEC_CHANNEL
    sources = people[source_column].to_numpy()
    _require(
        set(sources[asec]) == set(basis.index)
        and not (set(sources[~asec]) & set(basis.index)),
        "SOURCE_ROSTER",
    )
    _require(
        np.array_equal(
            people[native_column].to_numpy()[asec],
            basis.native_person_id.to_numpy()[basis.index.get_indexer(sources[asec])],
        ),
        "RECEIVER_SOURCE_IDENTITY",
    )
    clones = people[clone_column].to_numpy()
    order = np.lexsort((clones, sources))
    ordered_sources, ordered_clones = sources[order], clones[order]
    _, starts, counts = np.unique(
        ordered_sources, return_index=True, return_counts=True
    )
    _require(len(set(counts.tolist())) == 1, "CLONE_COUNT")
    per_source = int(counts[0])
    _require(
        np.array_equal(
            ordered_clones,
            np.tile(np.arange(per_source, dtype="int64"), len(starts)),
        ),
        "CLONE_INDEX_SET",
    )
    heads = np.repeat(starts, counts)
    for name, vector in (
        (native_column, people[native_column].to_numpy()[order]),
        (channel_column, channel[order]),
    ):
        _require(
            np.array_equal(vector, vector[heads]), "CLONE_SOURCE_AGREEMENT:" + name
        )
    index = pd.Index(people.person_id.to_numpy(), name="person_id")
    columns = {}
    for name in basis.columns:
        if name == "native_person_id":
            # Already carried by the receiving spine-source identity column
            # this join just checked; do not restate it as a leaf report.
            continue
        attached = attached_name(name)
        series = pd.Series(
            basis[name].reindex(pd.Index(sources)).array, index=index, name=attached
        )
        if name == KNOWN_COLUMN:
            series[~asec] = False
        elif name == REASON_COLUMN:
            series[~asec] = UNOBSERVED_REASON
        ordered_series = series.iloc[order].reset_index(drop=True)
        _require(
            ordered_series.equals(series.iloc[order[heads]].reset_index(drop=True)),
            "CLONE_TRANSPORT:" + attached,
        )
        columns["person", attached] = series
    receipt = json.loads(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "values_sha256": seal,
                "canonical_leaf": OUTPUT,
                "clone_transport": (
                    "copied to every clone of the original source person by "
                    "identity join on the pre-clone source id"
                ),
                "clones_per_source": per_source,
                "redraw_issued": False,
                "draws_consumed": 0,
                "observed_rows": int(asec.sum()),
                "unobserved_rows": int((~asec).sum()),
                "unobserved_channels": sorted(set(channel[~asec].tolist())),
                "unobserved_reason": UNOBSERVED_REASON,
                "unknown_completed_with_zero": False,
                "attached_columns": sorted(name for _, name in columns),
            }
        )
    )
    # The receiver supplies its table through its own callable, which runs
    # after the entry check, so the implementation is rechecked before return.
    _require(_implementation_unchanged(), "IMPLEMENTATION_CHANGED")
    _require(other_disability_values_seal(values) == seal, "FINAL_VALUES_CHANGED")
    return OtherDisabilityAttachment(columns, receipt)


_IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_LIVE = _live()
