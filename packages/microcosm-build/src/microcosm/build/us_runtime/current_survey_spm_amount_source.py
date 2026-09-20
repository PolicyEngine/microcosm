"""Descriptive SPM-unit amounts borrowed from existing live source owners.

No CSV capture, new source authority, fit, canonical consumer input or release
qualification is created here. Consumers retain both original owners and must
requalify after their final relevant I/O. In particular, a frozen-storage zero
does not become an authenticated Census zero merely because replicas agree.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import FunctionType

import numpy as np
import pandas as pd

from . import asec_current_money as money
from . import current_asec_income_routing_source as routing
from . import current_survey_spm_source as spm

PROTOCOL = "microcosm.us.current-spm-amount-source.v1"
FIELDS = ("SPM_ENGVAL", "SPM_CHILDCAREXPNS")
STRING = pd.StringDtype(storage="python", na_value=pd.NA)
_AXES = ("status", "validity", "zero_origin")


def _require(condition, reason):
    if not condition:
        raise ValueError("SPM_AMOUNT_SOURCE_" + reason)


def _domains():
    payload = (
        resources.files(__package__).joinpath(routing.DOMAINS_RESOURCE).read_bytes()
    )
    _require(
        hashlib.sha256(payload).hexdigest() == money.RESOURCE_PINS[0], "DOMAINS_HASH"
    )
    found = [f for f in json.loads(payload)["fields"] if f["name"] in FIELDS]
    _require(
        len(found) == len(FIELDS) and {f["name"] for f in found} == set(FIELDS),
        "DOMAINS_ROSTER",
    )
    result = {}
    for field in found:
        name = field["name"]
        vintages = [
            v
            for v in field["vintages"]
            if v["income_year"] == routing.CURRENT_INCOME_YEAR
        ]
        _require(len(vintages) == 1, "DOMAIN_VINTAGE")
        vintage = vintages[0]
        _require(
            field["entity"] == "person"
            and field["grain"] == "spm_unit_repeated_on_person"
            and field["column"] == name
            and field["minimum"] == 0
            and type(field["maximum"]) is int
            and field["maximum"] > 0
            and field["declared_niu_codes"] == []
            and field["zero_semantics"] == "valid_zero_dollars_or_none_as_described"
            and vintage["pdf_sha256"] == routing.DICTIONARY_SHA256
            and vintage["source_url"] == routing.DICTIONARY_URL
            and vintage["survey_year"] == routing.CURRENT_INCOME_YEAR + 1,
            "DOMAIN_CONTRACT",
        )
        result[name] = field
    return result


@dataclass(frozen=True)
class CurrentSpmAmountValues:
    person: pd.DataFrame
    units: pd.DataFrame
    evidence: dict


def spm_amount_values_seal(values):
    """Hash a detached descriptive result; this does not grant source authority."""
    _require(type(values) is CurrentSpmAmountValues, "VALUES_TYPE")
    return hashlib.sha256(
        spm.source._encode(
            (
                PROTOCOL,
                spm.seals._table_seal(values.person),
                spm.seals._table_seal(values.units),
                values.evidence,
            )
        )
    ).hexdigest()


def _entity_table(frame, name):
    return frame.table(name) if hasattr(frame, "table") else getattr(frame, name)


def _field_arrays(field, name, count, maximum, positions):
    _require(type(field) is money.MoneyField and field.name == name, "FIELD_TYPE")
    _require(
        type(field.amount_bytes) is bytes
        and len(field.amount_bytes) == count * 8
        and all(
            type(v) is bytes and len(v) == count
            for v in (field.status_bytes, field.validity_bytes, field.zero_origin_bytes)
        ),
        "FIELD_LENGTH",
    )
    vectors = field.amounts, field.statuses, field.validity, field.zero_origin
    values, status, valid, zero = (v[positions] for v in vectors)
    present = valid == 1
    positive = present & (values > 0)
    zeros = present & (values == 0)
    _require(
        np.isin(valid, (0, 1)).all()
        and np.isfinite(values).all()
        and not np.signbit(values[values == 0]).any()
        and (values[~present] == 0).all()
        and (status[~present] == money.CodebookStatus.MISSING_NULL).all()
        and (zero[~present] == money.ZeroOrigin.NOT_ZERO).all()
        and np.isfinite(values[present]).all()
        and ((values[present] >= 0) & (values[present] <= maximum)).all()
        and (values[present] == np.floor(values[present])).all()
        and (status[positive] == money.CodebookStatus.AMOUNT_NONZERO).all()
        and (zero[positive] == money.ZeroOrigin.NOT_ZERO).all()
        and (status[zeros] == money.CodebookStatus.ZERO_DOLLARS_AS_CODED).all()
        and np.isin(
            zero[zeros],
            (
                money.ZeroOrigin.FROZEN_FILLNA_UNRESOLVED,
                money.ZeroOrigin.AUTHENTICATED_CENSUS_ENCODED,
            ),
        ).all(),
        "FIELD_EVIDENCE",
    )
    return vectors


def project_spm_amounts(
    source_frame, origins, asec_raw, unit_evidence, money_scope, fields
):
    """Reconcile supplied evidence at unit grain, without authenticating it."""
    _require(type(money_scope) is money.AsecMoneyScope, "MONEY_SCOPE")
    _require(type(fields) is dict and set(fields) == set(FIELDS), "FIELD_ROSTER")
    domains = _domains()
    positions = [
        i
        for i, year in enumerate(money_scope.person_years)
        if year == routing.CURRENT_INCOME_YEAR
    ]
    arrays = {
        name: _field_arrays(
            fields[name],
            name,
            len(money_scope.person_ids),
            domains[name]["maximum"],
            positions,
        )
        for name in FIELDS
    }
    for table in (origins, asec_raw, unit_evidence):
        _require(
            type(table) is pd.DataFrame
            and table.index.is_unique
            and table.columns.is_unique,
            "TABLE_AXES",
        )
    people = _entity_table(source_frame, "person")
    units = _entity_table(source_frame, "spm_unit")
    _require(
        type(people) is type(units) is pd.DataFrame
        and people.columns.is_unique
        and units.columns.is_unique
        and people.person_id.dtype == units.spm_unit_id.dtype == np.dtype("int64")
        and people.person_spm_unit_id.dtype == np.dtype("int64")
        and people.person_id.is_unique
        and units.spm_unit_id.is_unique
        and origins.index.dtype == np.dtype("int64")
        and np.array_equal(people.person_id.to_numpy(), origins.index.to_numpy())
        and unit_evidence.index.dtype == np.dtype("int64")
        and np.array_equal(units.spm_unit_id.to_numpy(), unit_evidence.index.to_numpy())
        and set(people.person_spm_unit_id) == set(units.spm_unit_id),
        "SOURCE_AXES",
    )
    _require(
        origins.source.isin(("acs", "asec")).all()
        and origins.native_person_id.dtype == np.dtype("int64")
        and origins.source_year.eq(routing.CURRENT_INCOME_YEAR).all()
        and origins.survey_year.eq(
            origins.source.map(
                {
                    "asec": routing.CURRENT_INCOME_YEAR + 1,
                    "acs": routing.CURRENT_INCOME_YEAR,
                }
            )
        ).all()
        and not origins[["source", "native_person_id"]].duplicated().any(),
        "ORIGIN_COORDINATES",
    )
    selected = origins.index[origins.source.eq("asec")]
    _require(
        asec_raw.index.dtype == np.dtype("int64")
        and set(asec_raw.index) == set(selected)
        and asec_raw.PERIDNUM.is_unique
        and asec_raw.PERIDNUM.str.fullmatch(r"[0-9]{22}").fillna(False).all()
        and asec_raw.SPM_ID.str.fullmatch(r"[0-9]+").fillna(False).all(),
        "RAW_COORDINATES",
    )
    asec_raw = asec_raw.loc[selected]
    by_key = {money_scope.person_native_keys[i]: i for i in positions}
    by_id = {money_scope.person_ids[i]: i for i in positions}
    person = origins[["source", "native_person_id"]].copy(deep=True)
    person["source"] = person.source.astype(STRING)
    person["spm_unit_id"] = people.person_spm_unit_id.to_numpy(copy=True)
    person["money_position"] = np.int64(-1)
    for pid in selected:
        position = by_key.get(asec_raw.at[pid, "PERIDNUM"])
        identity_position = by_id.get(int(origins.at[pid, "native_person_id"]))
        _require(position == identity_position, "NATIVE_KEY_ID_JOIN")
        if position is not None:
            person.at[pid, "money_position"] = position
    has_position = person.money_position.ge(0).to_numpy()
    chosen = person.money_position.to_numpy()[has_position]
    for name, vectors in arrays.items():
        for suffix, vector in zip(("amount", *_AXES), vectors, strict=True):
            result = np.full(
                len(person),
                np.nan if suffix == "amount" else -1,
                dtype="float64" if suffix == "amount" else "int16",
            )
            result[has_position] = vector[chosen]
            person[name + "_" + suffix] = result
    out = unit_evidence[["source", "source_spm_unit_id"]].copy(deep=True)
    out["scope_status"] = unit_evidence.status.astype(STRING)
    _require(
        out.scope_status.isin(spm.projection.UNIVERSE_STATUSES).all(), "SCOPE_STATUS"
    )
    for column in ("source", "source_spm_unit_id"):
        out[column] = out[column].astype(STRING)
    out["member_count"] = np.int64(0)
    out["missing_money_members"] = np.int64(0)
    for name in FIELDS:
        out[name + "_amount"] = np.nan
        for suffix in _AXES:
            out[name + "_" + suffix] = np.int16(-1)
        out[name + "_known"] = False
        out[name + "_reason"] = pd.Series(
            "survey_amount_unobserved", index=out.index, dtype=STRING
        )
    literal_units = set()
    for uid, group in person.groupby("spm_unit_id", sort=False):
        arm = out.at[uid, "source"]
        _require(group.source.eq(arm).all(), "UNIT_SOURCE")
        out.at[uid, "member_count"] = len(group)
        missing = int(group.money_position.lt(0).sum())
        out.at[uid, "missing_money_members"] = missing
        if arm == "acs":
            continue
        raw = asec_raw.loc[group.index]
        literal = out.at[uid, "source_spm_unit_id"]
        _require(
            literal not in literal_units
            and raw.SPM_ID.eq(literal).all()
            and raw.PH_SEQ.nunique(dropna=False) == 1
            and raw.SPM_NUMPER.str.fullmatch(r"[0-9]+").fillna(False).all()
            and raw.SPM_NUMPER.astype("int64").eq(len(group)).all(),
            "COMPLETE_LITERAL_UNIT",
        )
        literal_units.add(literal)
        present = group.loc[group.money_position.ge(0)]
        for name in FIELDS:
            for suffix in ("amount", *_AXES):
                values = present[name + "_" + suffix].to_numpy()
                compared = values.view("uint64") if suffix == "amount" else values
                _require(
                    len(compared) == 0 or (compared == compared[0]).all(),
                    "REPLICATED_UNIT_EVIDENCE",
                )
                if not missing:
                    out.at[uid, name + "_" + suffix] = values[0]
            if missing:
                reason = "missing_money_member"
            elif out.at[uid, name + "_validity"] != 1:
                reason = "missing_amount"
            elif out.at[uid, "scope_status"] != spm.INCLUDED:
                reason = "annual_scope_unresolved"
            elif out.at[uid, name + "_amount"] != 0:
                reason = "known_nonzero"
            elif (
                out.at[uid, name + "_zero_origin"]
                == money.ZeroOrigin.AUTHENTICATED_CENSUS_ENCODED
            ):
                reason = "known_zero"
            else:
                reason = "zero_origin_unresolved"
            out.at[uid, name + "_reason"] = reason
            out.at[uid, name + "_known"] = reason in ("known_nonzero", "known_zero")
    evidence = {
        "protocol": PROTOCOL,
        "income_year": routing.CURRENT_INCOME_YEAR,
        "source_admission_issued": False,
        "release_eligible": False,
        "zero_origin_upgraded": False,
        "childcare_measure": "reported_uncapped_expense",
        "pre_subsidy_amount_observed": False,
        "source_domains": domains,
        "energy_range_disagreement_resolved": False,
        "money_scope_sha256": hashlib.sha256(money_scope.identity).hexdigest(),
    }
    return CurrentSpmAmountValues(person, out, evidence)


def _live():
    return (
        {
            name: spm.source._function_seal(value)
            for name, value in vars(sys.modules[__name__]).items()
            if isinstance(value, FunctionType)
        },
        PROTOCOL,
        FIELDS,
        _AXES,
        repr(STRING),
        CurrentSpmAmountValues,
        # Owners authenticate their original dependencies. Also bind this
        # borrower's active aliases and consulted semantics: replacing our
        # money alias must not relabel frozen-origin zero as observed zero.
        money,
        money.MoneyField,
        money.MoneyDomain,
        money.AsecMoneyScope,
        tuple(money.RESOURCE_PINS),
        money.CodebookStatus,
        int(money.CodebookStatus.MISSING_NULL),
        int(money.CodebookStatus.AMOUNT_NONZERO),
        int(money.CodebookStatus.ZERO_DOLLARS_AS_CODED),
        money.ZeroOrigin,
        int(money.ZeroOrigin.NOT_ZERO),
        int(money.ZeroOrigin.FROZEN_FILLNA_UNRESOLVED),
        int(money.ZeroOrigin.AUTHENTICATED_CENSUS_ENCODED),
        routing,
        routing.CURRENT_INCOME_YEAR,
        routing.DOMAINS_RESOURCE,
        routing.DICTIONARY_SHA256,
        routing.DICTIONARY_URL,
        spm,
        spm.INCLUDED,
        spm.QualifiedNativeSpmInputs,
        spm.projection,
        tuple(spm.projection.UNIVERSE_STATUSES),
        spm.source,
        spm.source.AuthenticatedSurveyPopulationPreparation,
        spm.seals,
        tuple(
            spm.source._function_seal(function)
            for function in (
                spm.source._function_seal,
                spm.source._encode,
                spm.source._sha,
                spm.seals._table_seal,
            )
        ),
    )


def qualify_current_spm_amounts(preparation, spm_inputs):
    """Borrow matching live owners before/after projection; return no capability."""
    _require(_live() == _LIVE, "IMPLEMENTATION_CHANGED")
    _require(
        type(preparation) is spm.source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    _require(type(spm_inputs) is spm.QualifiedNativeSpmInputs, "SPM_OWNER_TYPE")
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    entry = preparation._checked()
    spm_inputs.validate()
    state, native = entry[2], entry[2].native[1]
    receipt = json.loads(spm_inputs.receipt)
    _require(
        spm_inputs.source_frame is state.frame
        and receipt["preparation_sha256"] == spm.source._sha(entry[1]),
        "MATCHING_OWNERS",
    )
    issued = spm.source.asec_native._ISSUED.get(id(native))
    _require(
        issued is not None and issued[0]() is native and issued[1] == native.payload,
        "NATIVE_ISSUANCE",
    )
    parent = issued[2].parent
    ready = parent.ready()
    header = json.loads(ready.header)
    _require(
        header["target_year"] == routing.CURRENT_INCOME_YEAR
        and header["semantic"] == "annual_current_money",
        "MONEY_PERIOD",
    )
    domains = _domains()
    for name, spec in domains.items():
        found = [d for d in ready.bindings.spec.fields if d.name == name]
        _require(
            len(found) == 1
            and found[0]
            == money.MoneyDomain(
                name,
                spec["entity"],
                spec["column"],
                spec["grain"],
                spec["minimum"],
                spec["maximum"],
                spec["zero_semantics"],
            ),
            "LIVE_DOMAIN",
        )
    result = project_spm_amounts(
        state.frame,
        spm_inputs.origins,
        spm_inputs.asec_raw,
        spm_inputs.unit_evidence,
        parent.scope,
        {name: ready.field(name) for name in FIELDS},
    )
    present = result.person.loc[
        result.person.source.eq("asec") & result.person.money_position.ge(0)
    ]
    positions = present.money_position.to_numpy()
    for raw_name, parent_name in (
        ("PH_SEQ", "source_household_id"),
        ("A_LINENO", "A_LINENO"),
        ("A_AGE", "A_AGE"),
    ):
        _require(
            np.array_equal(
                spm_inputs.asec_raw.loc[present.index, raw_name]
                .astype("int64")
                .to_numpy(),
                parent.frame.person.iloc[positions][parent_name].to_numpy(),
            ),
            "PARENT_COORDINATES",
        )
    result.evidence.update(
        preparation_sha256=spm.source._sha(entry[1]),
        spm_receipt_sha256=hashlib.sha256(spm_inputs.receipt).hexdigest(),
        money_header_sha256=hashlib.sha256(ready.header).hexdigest(),
        implementation_sha256=implementation,
    )
    seal = spm_amount_values_seal(result)
    _require(
        preparation._checked() is entry and parent.ready().header == ready.header,
        "SOURCE_CHANGED",
    )
    spm_inputs.validate()
    _require(
        _live() == _LIVE
        and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == implementation,
        "IMPLEMENTATION_CHANGED",
    )
    spm.source._pure_final(state)
    spm._check_retained_output(spm_inputs, spm._ISSUED.get(spm_inputs))
    _require(
        spm.source._ISSUED.get(id(preparation)) is entry
        and spm.source.asec_native._ISSUED.get(id(native)) is issued
        and issued[2].parent is parent
        and native.payload == issued[1]
        and spm_amount_values_seal(result) == seal,
        "FINAL_OWNER_OR_VALUES",
    )
    return result


_LIVE = _live()
