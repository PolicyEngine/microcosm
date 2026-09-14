"""Qualified ASEC pension, disability and survivor source details.

Published amounts and component comparisons are observations, not a complete
ACS retirement bridge. No regularity, taxability, overlap allocation or source
admission is inferred. A host retains and requalifies the actual preparation.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd

from . import asec_current_money_source as physical
from . import current_asec_dividend_source as dividend
from . import current_asec_income_routing_source as routing

PROTOCOL = "microcosm.us.current-asec-retirement-detail-source.v1"
REFERENCE_FIELDS = (
    "PNSN_VAL",
    "ANN_VAL",
    "DST_VAL1",
    "DST_VAL1_YNG",
    "DST_VAL2",
    "DST_VAL2_YNG",
)
# Exact dictionary entries not supplied by the routing owner's domain mapping.
# Name, width, position, physical PDF page, printed page, maximum, universe.
DETAIL_AMOUNT_ENTRIES = (
    ("PEN_VAL1", 6, 558, 47, "6C-26", 999999, "PEN_SC1 > 0"),
    ("PEN_VAL2", 6, 564, 47, "6C-26", 999999, "PEN_SC2 > 0"),
    ("DIS_VAL1", 6, 465, 44, "6C-23", 999999, "DIS_SC1 > 0"),
    ("DIS_VAL2", 6, 471, 44, "6C-23", 999999, "DIS_SC2 > 0"),
    ("SUR_VAL1", 6, 652, 50, "6C-29", 999999, "SUR_YN = 1"),
    ("SUR_VAL2", 6, 658, 50, "6C-29", 999999, "SUR_YN = 1"),
    ("DSAB_VAL", 6, 485, 45, "6C-24", 999999, "DIS_VAL1 > 0 OR DIS_VAL2 > 0"),
    ("DBTN_VAL", 7, 452, 44, "6C-23", 9999999, "DST_VAL1 > 0 OR DST_VAL2 > 0"),
    ("SRVS_VAL", 6, 628, 49, "6C-28", 999999, "SUR_YN = 1"),
)
AMOUNT_FIELDS = (*REFERENCE_FIELDS, *(entry[0] for entry in DETAIL_AMOUNT_ENTRIES))
RETAINED_MONEY_FIELDS = (*REFERENCE_FIELDS, "DIS_VAL1", "DIS_VAL2")
INDEPENDENT_LITERAL_FIELDS = tuple(
    n for n in AMOUNT_FIELDS if n not in RETAINED_MONEY_FIELDS
)
RECEIPT_ENTRIES = MappingProxyType(
    {
        "PEN_YN": routing.RECEIPT_ENTRIES["PEN_YN"],
        "SUR_YN": routing.ReceiptEntry(*dividend.RECEIPT_ENTRIES["SUR_YN"]),
        "DIS_YN": routing.ReceiptEntry(
            1, 477, 45, "6C-24", "All Persons aged 15+", "niu"
        ),
        "DIS_CS": routing.ReceiptEntry(
            1, 459, 44, "6C-23", "All Persons aged 15+", "niu"
        ),
        "DIS_HP": routing.ReceiptEntry(
            1, 460, 44, "6C-23", "All Persons aged 15+", "niu"
        ),
    }
)
PENSION_CODES = MappingProxyType(
    {
        0: "niu",
        1: "Company pension",
        2: "Union pension",
        3: "Federal government pension",
        4: "State government pension",
        5: "Local government pension",
        6: "US Military pension",
        7: "US Railroad Retirement",
        8: "Other",
    }
)
DISABILITY_CODES = MappingProxyType(
    {
        0: "NIU",
        1: "worker's compensation",
        2: "company or union disability",
        3: "federal government disability",
        4: "US military retirement disability",
        5: "state or local government employee disability",
        6: "US railroad retirement disability",
        7: "accident or disability insurance",
        8: "blacklung miners disability",
        9: "state temporary sickness",
        10: "other or don't know",
    }
)
# Source code width, position, physical page, printed page, printed universe.
SOURCE_ENTRIES = MappingProxyType(
    {
        "PEN_SC1": (1, 556, 47, "6C-26", "PEN_YN = 1"),
        "PEN_SC2": (1, 557, 47, "6C-26", "PEN_VAL2 > 0"),
        "DIS_SC1": (2, 461, 44, "6C-23", "DIS_YN = 1"),
        "DIS_SC2": (2, 463, 44, "6C-23", "DIS_YN = 1"),
        **dividend.SURVIVOR_ENTRIES,
    }
)
SOURCE_CODES = MappingProxyType(
    {"PEN": PENSION_CODES, "DIS": DISABILITY_CODES, "SUR": dividend.SURVIVOR_CODES}
)
# Position, physical page, printed page, universe, supported published values.
ALLOCATION_ENTRIES = (
    ("I_PENSC1", 850, 56, "6C-35", "PEN_SC1 > 0", routing.ALLOCATION_DSTSC_CODES),
    ("I_PENSC2", 851, 56, "6C-35", "PEN_SC2 > 0", routing.ALLOCATION_DSTSC_CODES),
    ("I_PENVAL1", 852, 57, "6C-36", "PEN_VAL1 > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_PENVAL2", 853, 57, "6C-36", "PEN_VAL2 > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_DISCS", 812, 54, "6C-33", "DIS_CS > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_DISHP", 813, 54, "6C-33", "DIS_HP > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_DISSC1", 814, 54, "6C-33", "DIS_SC1 > 0", routing.ALLOCATION_DSTSC_CODES),
    ("I_DISSC2", 815, 54, "6C-33", "DIS_SC2 > 0", routing.ALLOCATION_DSTSC_CODES),
    ("I_DISVL1", 816, 54, "6C-33", "DIS_VAL1 > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_DISVL2", 817, 54, "6C-33", "DIS_VAL2 > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_DISYN", 818, 54, "6C-33", "DIS_YN > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_SURSC1", 873, 58, "6C-37", "SUR_SC1 > 0", routing.ALLOCATION_DSTSC_CODES),
    ("I_SURSC2", 874, 58, "6C-37", "SUR_SC2 > 0", routing.ALLOCATION_DSTSC_CODES),
    ("I_SURVL1", 875, 58, "6C-37", "SUR_VAL1 > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_SURVL2", 876, 58, "6C-37", "SURV_VAL2 > 0", routing.ALLOCATION_ANNVAL_CODES),
    ("I_SURYN", 877, 58, "6C-37", "SUR_YN > 0", routing.ALLOCATION_ANNVAL_CODES),
)
TOPCODE_ENTRIES = (
    ("TPEN_VAL1", 913, 60, "6C-39", "PEN_VAL1 > 0"),
    ("TPEN_VAL2", 914, 60, "6C-39", "PEN_VAL2 > 0"),
    ("TDISVAL1", 903, 59, "6C-38", "DIS_VAL1 > 0"),
    ("TDISVAL2", 904, 59, "6C-38", "DIS_VAL2 > 0"),
    ("TSURVAL1", 670, 50, "6C-29", "SUR_VAL1 > 0"),
    ("TSURVAL2", 671, 50, "6C-29", "SUR_VAL2 > 0"),
)
READ_COLUMNS = (
    *routing.COORDINATE_COLUMNS,
    *AMOUNT_FIELDS,
    *RECEIPT_ENTRIES,
    *SOURCE_ENTRIES,
    *(e[0] for e in ALLOCATION_ENTRIES),
    *(e[0] for e in TOPCODE_ENTRIES),
)
COMPARISONS = (
    ("pension", "PNSN_VAL", "PEN_VAL1", "PEN_VAL2"),
    ("disability", "DSAB_VAL", "DIS_VAL1", "DIS_VAL2"),
    ("survivor", "SRVS_VAL", "SUR_VAL1", "SUR_VAL2"),
    ("distribution", "DBTN_VAL", "DST_VAL1", "DST_VAL2"),
)


def require(condition, reason):
    if not condition:
        raise ValueError("ASEC_RETIREMENT_DETAIL_" + reason)


@lru_cache(maxsize=1)
def _cached_amount_entries():
    reference = routing.printed_amount_entries()
    entries = [(name, reference[name]) for name in REFERENCE_FIELDS]
    for (
        name,
        width,
        position,
        page,
        printed,
        maximum,
        universe,
    ) in DETAIL_AMOUNT_ENTRIES:
        entries.append(
            (
                name,
                routing.PrintedAmountEntry(
                    name,
                    width,
                    position,
                    printed,
                    page,
                    0,
                    maximum,
                    universe,
                    "0 = none or niu; 1-" + str(maximum) + " = income amount",
                    False,
                    (),
                    "none_or_niu_not_distinguishable_from_amount_alone",
                ),
            )
        )
    require(len(entries) == len(set(n for n, _ in entries)), "DUPLICATE_AMOUNT")
    return tuple(entries)


def amount_entries():
    """Detached mapping over immutable, source-pinned amount entries."""
    return dict(_cached_amount_entries())


amount_entries.cache_clear = _cached_amount_entries.cache_clear


def _domain_agreement(ready):
    entries = amount_entries()
    routing._require_live_amount_domains(
        ready,
        {
            name: (
                entries[name].encoded_minimum,
                entries[name].encoded_maximum,
                entries[name].zero_semantics,
            )
            for name in RETAINED_MONEY_FIELDS
        },
    )
    # A future expanded money roster must add real retained-bit comparisons;
    # it may not silently leave those new fields on the independent path.
    require(
        not (
            set(INDEPENDENT_LITERAL_FIELDS)
            & {f.name for f in ready.bindings.spec.fields}
        ),
        "UNREVIEWED_RETAINED_DOMAIN",
    )


def _amount_pair(token, entry):
    require(type(token) is str and len(token) <= routing.TOKEN_MAX_CHARS, "TOKEN_BOUND")
    if token == "":
        return None, "missing"
    pattern = (
        ("-?" if entry.encoded_minimum < 0 else "")
        + r"[0-9]{1,"
        + str(entry.printed_length)
        + "}"
    )
    if re.fullmatch(pattern, token, re.ASCII) is None:
        return None, "malformed"
    value = int(token)
    if not entry.encoded_minimum <= value <= entry.encoded_maximum:
        return None, "outside_printed_range"
    return value, "in_printed_range"


def _slot_status(age, receipt, source_code, amount):
    value, status = amount
    if status not in ("missing", "in_printed_range"):
        return "invalid_amount_literal"
    kind = "missing" if value is None else "zero" if value == 0 else "nonzero"
    baseline = routing.receipt_status(
        bool(age >= 15), receipt, kind, net_measure=False, zero_is_dollars=False
    )
    if age < 15:
        if (
            receipt[1] != "in_printed_range"
            or source_code[1] != "in_printed_range"
            or value is None
        ):
            return "unresolved_outside_reporting_universe"
        return (
            "outside_reporting_universe"
            if receipt[0] == source_code[0] == 0 and value == 0
            else "contradictory_outside_reporting_universe"
        )
    if receipt[1] != "in_printed_range" or value is None:
        return baseline
    code, code_status = source_code
    if code_status != "in_printed_range":
        return "unresolved_source_slot"
    if receipt[0] == 1 and code == 0:
        return "unreported_source_slot" if value == 0 else "contradictory_source_slot"
    if receipt[0] in (0, 2) and code != 0:
        return "contradictory_source_slot"
    return baseline


def project_retirement_detail_literals(ordered):
    """Preserve literals/knownness; aggregate comparisons do not allocate dollars."""
    require(
        type(ordered) is pd.DataFrame
        and ordered.columns.is_unique
        and set(READ_COLUMNS) <= set(ordered),
        "COLUMNS",
    )
    age_pairs = [routing.literal_code(t, range(100), width=2) for t in ordered.A_AGE]
    require(all(s == "in_printed_range" for _, s in age_pairs), "AGE_LITERAL")
    ages = np.array([v for v, _ in age_pairs], dtype=np.int64)
    out = pd.DataFrame(index=range(len(ordered)))
    parsed, codes = {}, {}
    for name, entry in amount_entries().items():
        pairs = [_amount_pair(t, entry) for t in ordered[name]]
        parsed[name] = pairs
        out[name + "_literal"] = pd.array(ordered[name].tolist(), dtype="string")
        out[name + "_literal_status"] = pd.array([s for _, s in pairs], dtype="string")
        out[name + "_published_amount"] = pd.array(
            [v for v, _ in pairs], dtype="Float64"
        )
    for name, entry in RECEIPT_ENTRIES.items():
        frame, values, statuses = routing._codes_frame(
            name,
            ordered[name],
            routing.RECEIPT_CODE_DOMAIN,
            {0: entry.zero_label_as_printed, 1: "yes", 2: "no"},
            width=entry.printed_length,
        )
        out = pd.concat([out, frame], axis=1)
        codes[name] = list(zip(values, statuses, strict=True))
    for name, entry in SOURCE_ENTRIES.items():
        domain = SOURCE_CODES[name[:3]]
        frame, values, statuses = routing._codes_frame(
            name, ordered[name], domain, domain, width=entry[0]
        )
        out = pd.concat([out, frame], axis=1)
        codes[name] = list(zip(values, statuses, strict=True))
    for family in SOURCE_CODES:
        for slot in (1, 2):
            name = family + "_VAL" + str(slot)
            labels = [
                _slot_status(age, receipt, code, amount)
                for age, receipt, code, amount in zip(
                    ages,
                    codes[family + "_YN"],
                    codes[family + "_SC" + str(slot)],
                    parsed[name],
                    strict=True,
                )
            ]
            known = np.array(
                [s in routing.KNOWN_AMOUNT_STATUSES for s in labels], dtype=bool
            )
            out[name + "_reporting_status"] = pd.array(labels, dtype="string")
            out[name + "_amount_known"] = known
            out[name + "_amount"] = out[name + "_published_amount"].where(known)
    for prefix, total, first, second in COMPARISONS:
        suffix = "main_slots" if prefix == "distribution" else "visible_slots"
        # Only a difference of readable published numbers, including raw NIU
        # zeros. A zero difference is not analytic completeness or absence.
        out[prefix + "_total_minus_" + suffix] = (
            out[total + "_published_amount"]
            - out[first + "_published_amount"]
            - out[second + "_published_amount"]
        )
    for name, _, _, _, _, allowed in ALLOCATION_ENTRIES:
        frame, _, _ = routing._codes_frame(name, ordered[name], allowed, width=1)
        out = pd.concat([out, frame], axis=1)
    for name, *_ in TOPCODE_ENTRIES:
        frame, _, _ = routing._codes_frame(name, ordered[name], (0, 1), width=1)
        out = pd.concat([out, frame], axis=1)
    out["source_age"] = ages
    out["survivor_visible_slots_exhaustive"] = False
    out["taxability_assigned"] = False
    out["regularity_assigned"] = False
    out["acs_retirement_component_assigned"] = False
    return out


@dataclass(frozen=True)
class CurrentAsecRetirementDetailValues:
    person: pd.DataFrame
    asec_literals: pd.DataFrame
    evidence: dict


def retirement_detail_values_seal(values):
    """Physical value seal, including nullable backing storage and exact bits."""
    require(type(values) is CurrentAsecRetirementDetailValues, "VALUES_TYPE")
    digest = hashlib.sha256(PROTOCOL.encode())
    for table in (values.person, values.asec_literals):
        require(type(table) is pd.DataFrame and table.columns.is_unique, "TABLE_TYPE")
        digest.update(
            physical._json(
                {
                    "columns": list(table.columns),
                    "columns_axis": physical.checkpoint._index_spec(
                        table.columns, label="retirement detail columns"
                    ),
                    "index": physical.checkpoint._index_spec(
                        table.index, label="retirement detail"
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
                # The checkpoint digest supports NumPy floats, but deliberately
                # excludes Float64 extension arrays. Keep the nullable tag and
                # hash both physical arrays, including values under null masks.
                digest.update(physical._json({"dtype": "Float64", "nullable": True}))
                physical._series_digest(digest, pd.Series(series.array._data))
                physical._series_digest(digest, pd.Series(series.array._mask))
            else:
                physical._series_digest(digest, series)
    digest.update(physical._json(values.evidence))
    return digest.hexdigest()


def _capture_member(root, pin):
    _, member, _, digest, rows, size = pin
    with tempfile.TemporaryDirectory(prefix="microcosm-asec-retirement-detail-") as tmp:
        path = Path(tmp) / member
        identity = routing.coverage._capture(
            root / "asec" / member,
            path,
            size=size,
            digest=digest,
            budget=[routing.coverage._BODY_MAX],
        )
        raw = routing._read_capture(path, rows=rows, columns=READ_COLUMNS, patterns={})
        require(
            routing.coverage._identity(path.stat(follow_symlinks=False)) == identity
            and routing._file_sha(path) == digest,
            "CAPTURE_CHANGED",
        )
    return raw


def _compare_amount(ready, positions, name, literals):
    field = ready.field(name)
    entry = amount_entries()[name]
    pairs = [_amount_pair(t, entry) for t in literals]
    require(
        all(s in ("missing", "in_printed_range") for _, s in pairs), "AMOUNT_LITERAL"
    )
    valid = field.validity[positions] == 1
    require(np.array_equal(valid, [v is not None for v, _ in pairs]), "AMOUNT_VALIDITY")
    expected = np.array(
        [np.nan if v is None else v for v, _ in pairs], dtype=np.float64
    )
    if name in REFERENCE_FIELDS:
        routing.amount_observations(field, positions, literals)
    # ANN's published -1 is retained as +0 with DECLARED_NIU by the money owner.
    # Normalize only that declared non-dollar code for this comparison; the
    # separate literal/published projection below keeps the original -1.
    niu = valid & np.isin(expected, entry.nonmoney_codes)
    require(
        np.array_equal(
            field.statuses[positions] == routing.money.CodebookStatus.DECLARED_NIU,
            niu,
        ),
        "AMOUNT_NIU_STATUS",
    )
    expected[niu] = 0.0
    require(
        np.array_equal(
            field.amounts[positions][valid].view("uint64"),
            expected[valid].view("uint64"),
        ),
        "AMOUNT_BITS",
    )
    return field


def qualify_current_asec_retirement_detail(preparation):
    """Borrow the original preparation, capture once, and requalify before return."""
    source = routing.source
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state, native = entry[2], entry[2].native[1]
    issued = source.asec_native._ISSUED.get(id(native))
    require(
        issued is not None and issued[0]() is native and issued[1] == native.payload,
        "NATIVE_ISSUANCE",
    )
    parent = issued[2].parent
    ready = parent.ready()
    _domain_agreement(ready)
    header, native_document = json.loads(ready.header), json.loads(issued[1])
    require(
        header["target_year"] == routing.CURRENT_INCOME_YEAR
        and header["semantic"] == "annual_current_money"
        and native_document["source_year"]
        == native_document["income_year"]
        == routing.CURRENT_INCOME_YEAR
        and native_document["survey_year"] == (routing.CURRENT_INCOME_YEAR + 1),
        "PERIOD",
    )
    pins = [
        p for p in routing.coverage._MEMBER_PINS if p[0] == routing.CURRENT_INCOME_YEAR
    ]
    require(len(pins) == 1, "SOURCE_REGISTRY")
    year, member, archive, digest, rows, size = pins[0]
    retained = [
        s for s in issued[2].coverage.receipt["sources"] if s["source_year"] == year
    ]
    require(
        len(retained) == 1
        and all(
            retained[0][k] == v
            for k, v in (
                ("member", member),
                ("archive_sha256", archive),
                ("member_sha256", digest),
                ("rows", rows),
                ("member_bytes", size),
            )
        ),
        "MEMBER_BINDING",
    )
    raw = _capture_member(state.root, pins[0])
    positions = np.flatnonzero(
        np.asarray(parent.scope.person_years) == routing.CURRENT_INCOME_YEAR
    )
    keys = np.asarray(parent.scope.person_native_keys)[positions]
    require(
        len(keys) == rows and len(set(keys)) == rows and set(keys) == set(raw.index),
        "COMPLETE_SOURCE_JOIN",
    )
    ordered = raw.loc[keys]
    for raw_name, parent_name in (
        ("PH_SEQ", "source_household_id"),
        ("A_LINENO", "A_LINENO"),
        ("A_AGE", "A_AGE"),
    ):
        require(
            np.array_equal(
                ordered[raw_name].astype("int64").to_numpy(),
                parent.frame.person.iloc[positions][parent_name].to_numpy(),
            ),
            "PARENT_COORDINATE",
        )
    fields = {
        name: _compare_amount(ready, positions, name, ordered[name])
        for name in RETAINED_MONEY_FIELDS
    }
    basis = project_retirement_detail_literals(ordered)
    basis.index = pd.Index(
        np.asarray(parent.scope.person_ids)[positions], name="native_person_id"
    )
    for field_name, field in fields.items():
        for name in ("statuses", "validity", "zero_origin"):
            basis[field_name + "_parent_" + name] = getattr(field, name)[positions]
    selected = state.frame.person.loc[
        state.frame.person[routing.support_channel_column("person")].eq("asec")
    ]
    native_ids = selected[routing.spine_source_id_column("person")].to_numpy()
    require(
        len(set(native_ids)) == len(native_ids) and set(native_ids) <= set(basis.index),
        "SELECTED_NATIVE_JOIN",
    )
    out = basis.loc[native_ids].copy()
    out["native_person_id"] = native_ids
    out.index = pd.Index(selected.person_id.to_numpy(), name="person_id")
    literals = ordered.copy()
    literals.index = basis.index.copy()
    evidence = {
        "protocol": PROTOCOL,
        "dictionary_url": routing.DICTIONARY_URL,
        "dictionary_sha256": routing.DICTIONARY_SHA256,
        "amount_entries": {
            name: asdict(entry) for name, entry in amount_entries().items()
        },
        "receipt_entries": {
            name: entry._asdict() for name, entry in RECEIPT_ENTRIES.items()
        },
        "source_entries": dict(SOURCE_ENTRIES),
        "source_codes": {name: dict(values) for name, values in SOURCE_CODES.items()},
        "allocation_entries": ALLOCATION_ENTRIES,
        "allocation_entry_width": 1,
        "allocation_header_range": "0:9; supported value domains retained separately",
        "topcode_entries": TOPCODE_ENTRIES,
        "topcode_entry_width": 1,
        "retained_money_fields": RETAINED_MONEY_FIELDS,
        "independently_qualified_literal_fields": INDEPENDENT_LITERAL_FIELDS,
        "comparison_scope": "raw readable published amounts, including none/NIU zeros; not a decomposition",
        "survivor_total_scope": "SUR_VAL1/2 edited amounts plus unedited sources 3 and 4",
        "distribution_total_scope": "printed DBTN_VAL formula uses DST_VAL1 + DST_VAL2, not YNG slots",
        "annuity_distribution_reference_scope": "raw retained observations only; canonical receipt/route interpretation remains with income routing",
        "universe_notes": [
            "I_SURVL2 prints SURV_VAL2; preserve spelling rather than invent a compiler predicate",
            "DIS_CS describes leaving work for health; DIS_HP also includes limited work",
            "Regular IRA is an account type, not evidence of regular withdrawals",
        ],
        "preparation_sha256": routing._sha(entry[1]),
        "asec_native_sha256": routing._sha(issued[1]),
        "money_header_sha256": routing._sha(ready.header),
        "source_member_sha256": digest,
        "source_year": routing.CURRENT_INCOME_YEAR,
        "survey_year": (routing.CURRENT_INCOME_YEAR + 1),
        "complete_source_rows": rows,
        "selected_rows": len(out),
        "read_columns": READ_COLUMNS,
        "taxability_assigned": False,
        "regularity_assigned": False,
        "acs_retirement_component_assigned": False,
        "aggregate_discrepancy_allocated": False,
        "under15_completed_with_zero": False,
        "source_admission_issued": False,
        "release_eligible": False,
    }
    # Freeze detached JSON-compatible metadata before its seal; do not retain the
    # module's mutable constant dictionaries inside a returned descriptive view.
    result = CurrentAsecRetirementDetailValues(
        out, literals, json.loads(json.dumps(evidence))
    )
    seal = retirement_detail_values_seal(result)
    require(
        preparation._checked() is entry and parent.ready().header == ready.header,
        "SOURCE_CHANGED",
    )
    source._pure_final(state)
    require(
        source._ISSUED.get(id(preparation)) is entry
        and source.asec_native._ISSUED.get(id(native)) is issued
        and issued[2].parent is parent
        and native.payload == issued[1],
        "FINAL_OWNER",
    )
    require(retirement_detail_values_seal(result) == seal, "FINAL_VALUES_CHANGED")
    return result
