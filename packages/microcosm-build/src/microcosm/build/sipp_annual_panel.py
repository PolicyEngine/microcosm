"""Bounded source panel diagnostics; no fitting, recipient data or policy engines.

Two full sequential passes: identifiers only, then all rows of the chosen SSUIDs.
Only extract() opens a supplied source; importing this module reads code metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path

from . import sipp_december_source as production
from . import sipp_measurements as measurements

CONTRACT_PATH = Path(__file__).with_name("sipp_annual_panel_contract.json")
CONTRACT_SHA256 = "e7e1175ede334339578ae1095ab178a33159e7eb7110dfc2dcdb16f694c3a3dc"
CONTRACT_BYTES = CONTRACT_PATH.read_bytes()
if hashlib.sha256(CONTRACT_BYTES).hexdigest() != CONTRACT_SHA256:
    raise ValueError("PUBLIC_CONTRACT_HASH_MISMATCH")
CONTRACT = json.loads(CONTRACT_BYTES)
FIELDS = tuple(sorted(CONTRACT["fields"]))
PERSON_FIELDS = tuple(f for f in FIELDS if CONTRACT["fields"][f]["grain"] == "Person")
INDEX_FIELDS = (
    "SSUID",
    "PNUM",
    "MONTHCODE",
    "THHLDSTATUS",
    "RIN_UNIV",
    "ERESIDENCEID",
    "ARESIDENCEID",
)
SELECTION = (
    "first32_by_canonical_14_digit_lexical_SSUID_from_eligible_December_ID_index"
)
MAX_SOURCE_BYTES = 16 * 1024**3
MAX_ROWS = 10_000_000
MAX_INDEX = 1_000_000
MAX_PANEL_ROWS = 100_000
STATUS_CODES = {
    "0": "not_in_universe",
    "1": "in_universe_as_reported",
    "2": "statistical_hot_deck",
    "3": "logical_imputation",
    "4": "model_based_imputation",
    "5": "cold_deck",
    "6": "imputed_from_range",
    "7": "combination_of_1_and_2_3_5_6",
    "8": "combination_of_2_3_5_6",
    "9": "consult_component_flags_quality_unknown_here",
}
RANGES = {
    "PNUM": (101, 499),
    "MONTHCODE": (1, 12),
    "THHLDSTATUS": (1, 6),
    "RIN_UNIV": (1, 2),
    "ERESIDENCEID": (100001, 400999),
    "TAGE_EHC": (0, 90),
    "WPFINWGT": (0, Decimal("9999999.999999")),
    "RSSI_MNYN": (1, 2),
    "TPTOTINC": (-10_000_000_000, 10_000_000_000),
    "THTOTINC": (-10_000_000_000, 10_000_000_000),
    "RHNUMPER": (1, 20),
    "RHNUMPERWT2": (1, 20),
    "TMOVER": (1, 7),
    "TLMPAMT": (1, 10_000_000),
    "TROLLAMT": (1, 10_000_000),
}


class PanelError(ValueError):
    """An aggregate-safe refusal code, never including a source value or identity."""


def number(raw: str, field: str) -> Decimal | None:
    text = raw.strip()
    if not text:
        return None
    if len(text) > 32 or not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", text):
        raise PanelError("UNRECOGNIZED_LEXICAL_TOKEN")
    value = Decimal(text)
    if field in CONTRACT["fields"] and "flag_for" in CONTRACT["fields"][field]:
        low, high = 0, 9
    elif field.startswith("RREL_PNUM"):
        low, high = 60, 499
        if not (60 <= value <= 69 or 101 <= value <= 499):
            raise PanelError("INVALID_ROSTER_REFERENCE")
    elif field.startswith("RREL"):
        low, high = 1, 99
        if value not in (*range(1, 20), 99):
            raise PanelError("INVALID_RELATIONSHIP_CODE")
    elif field.startswith("ET2_LNO"):
        low = high = 59 + int(field.removeprefix("ET2_LNO"))
    else:
        low, high = RANGES[field]
    if not low <= value <= high or (
        field != "WPFINWGT" and value != value.to_integral_value()
    ):
        raise PanelError("FIELD_OUT_OF_DOMAIN")
    return value


def canonical_ssuid(raw: str) -> str:
    # Preserve all leading zeros; aliases are refused, not coerced to integers.
    if not re.fullmatch(r"[0-9]{14}", raw):
        raise PanelError("NONCANONICAL_SSUID")
    return raw


def usable(row: dict, field: str) -> tuple[Decimal | None, str]:
    value = number(row[field], field)
    flag_name = CONTRACT["fields"][field].get("flag")
    if flag_name:
        flag = number(row[flag_name], flag_name)
        if flag is None:
            return None, "missing_status"
        if flag == 0:
            return None, "structural_out_of_universe"
    return (value, "available") if value is not None else (None, "missing_value")


def key(row: dict) -> tuple[str, int, int]:
    ssuid = canonical_ssuid(row["SSUID"])
    pnum = number(row["PNUM"], "PNUM")
    month = number(row["MONTHCODE"], "MONTHCODE")
    if pnum is None or month is None:
        raise PanelError("MISSING_PERSON_MONTH_KEY")
    return ssuid, int(pnum), int(month)


def live(row: dict) -> bool:
    return number(row["THHLDSTATUS"], "THHLDSTATUS") in (1, 2, 3, 4)


def eligible_id(row: dict) -> bool:
    return (
        live(row)
        and number(row["RIN_UNIV"], "RIN_UNIV") == 1
        and usable(row, "ERESIDENCEID")[0] is not None
    )


class HashingStream:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.size = 0

    def readline(self, limit):
        data = self.stream.readline(limit)
        self.size += len(data)
        if self.size > MAX_SOURCE_BYTES:
            raise PanelError("SOURCE_BYTE_CEILING")
        self.digest.update(data)
        return data


def stream_rows(path: Path, columns: tuple[str, ...], audit: dict):
    """Use the frozen production physical-record reader, including width limits."""
    with os.fdopen(production._open_regular(path), "rb") as raw:
        before = production._identity(os.fstat(raw.fileno()))
        stream = HashingStream(raw)
        first = production._line(stream)
        if first is None:
            raise PanelError("EMPTY_SOURCE")
        header = production._parts(first)
        if (
            any(not f for f in header)
            or len(set(header)) != len(header)
            or not set(FIELDS) <= set(header)
        ):
            raise PanelError("INVALID_HEADER")
        positions = [header.index(f) for f in columns]
        records = 0
        while (line := production._line(stream)) is not None:
            tokens = production._parts(line)
            if len(tokens) != len(header):
                raise PanelError("ROW_WIDTH")
            records += 1
            if records > MAX_ROWS:
                raise PanelError("SOURCE_ROW_CEILING")
            yield dict(zip(columns, (tokens[p] for p in positions), strict=True))
        if before != production._identity(os.fstat(raw.fileno())):
            raise PanelError("SOURCE_CHANGED_DURING_PASS")
        audit.update(sha256=stream.digest.hexdigest(), bytes=stream.size, rows=records)


def check_digest(audit: dict, sha256: str, size: int):
    if audit["sha256"] != sha256 or audit["bytes"] != size:
        raise PanelError("SOURCE_PIN_MISMATCH")


@dataclass
class Panel:
    rows: dict[tuple[str, int, int], dict]
    selected: tuple[str, ...]
    index_audit: dict
    panel_audit: dict


def read_panel(path: Path, sha256: str, size: int) -> Panel:
    # Validate declarations before any stat/open of the source path.
    if (
        not re.fullmatch(r"[0-9a-f]{64}", sha256 or "")
        or type(size) is not int
        or not 0 < size <= MAX_SOURCE_BYTES
    ):
        raise PanelError("UNBOUND_OR_INVALID_SOURCE_PIN")
    if not path.is_absolute():
        raise PanelError("ABSOLUTE_SOURCE_PATH_REQUIRED")
    keys, seen = set(), set()
    index_audit = {}
    for row in stream_rows(path, INDEX_FIELDS, index_audit):
        identity = key(row)
        if identity[2] != 12:
            continue
        if identity in seen:
            raise PanelError("DUPLICATE_DECEMBER_PERSON")
        seen.add(identity)
        if len(seen) > MAX_INDEX:
            raise PanelError("DECEMBER_INDEX_CEILING")
        if eligible_id(row):
            keys.add((identity[0], int(usable(row, "ERESIDENCEID")[0])))
    check_digest(index_audit, sha256, size)
    selected = production._selected_ssuids(tuple(sorted(keys)), "first_32_ssuids")
    rows, panel_audit = {}, {}
    # Select before monetary columns are projected or interpreted. Physical bytes
    # necessarily traverse the hash/pipe tokenizer in both passes.
    for row in stream_rows(path, FIELDS, panel_audit):
        if canonical_ssuid(row["SSUID"]) not in selected:
            continue
        identity = key(row)
        if identity in rows:
            raise PanelError("DUPLICATE_PERSON_MONTH")
        if len(rows) >= MAX_PANEL_ROWS:
            raise PanelError("PANEL_ROW_CEILING")
        rows[identity] = row
    check_digest(panel_audit, sha256, size)
    index_audit["eligible_ssuids"] = len({s for s, _ in keys})
    index_audit["eligible_households"] = len(keys)
    index_audit["december_person_rows"] = len(seen)
    return Panel(rows, selected, index_audit, panel_audit)


def income_scope(row: dict) -> str:
    if not live(row):
        return "outside_monthly_membership"
    if number(row["RIN_UNIV"], "RIN_UNIV") != 1:
        return "outside_or_unknown_frame"
    age = number(row["TAGE_EHC"], "TAGE_EHC")
    if age is None:
        return "unknown_age"
    if age >= 15:
        return "in"
    receipt, _ = usable(row, "RSSI_MNYN")
    if receipt == 1:
        return "in"
    if receipt == 2:
        return "pre15_without_ssi"
    return "unknown_child_receipt"


def membership(rows: list[dict]) -> set[str]:
    reasons = set()
    collected = {int(number(r["PNUM"], "PNUM")) for r in rows}
    union = set()
    for row in rows:
        person = int(number(row["PNUM"], "PNUM"))
        refs = []
        self_count = 0
        for i in range(1, 17):
            ref, rel = (
                number(row[f"RREL_PNUM{i}"], f"RREL_PNUM{i}"),
                number(row[f"RREL{i}"], f"RREL{i}"),
            )
            if (ref is None) != (rel is None):
                reasons.add("unpaired_roster_slot")
            if ref is not None:
                refs.append(int(ref))
                self_count += int(ref == person and rel == 99)
                if (ref == person) != (rel == 99):
                    reasons.add("self_relationship_conflict")
        if self_count != 1:
            reasons.add("missing_or_duplicate_self")
        if len(refs) != len(set(refs)):
            reasons.add("duplicate_roster_reference")
        if row["RREL_PNUM16"].strip():
            reasons.add("suppressed_tail_unknown")
        native = {p for p in refs if p >= 101}
        union.update(native)
        if native != collected:
            reasons.add("row_roster_disagreement")
        for field, expected in [
            ("RHNUMPER", len(collected)),
            ("RHNUMPERWT2", len(set(refs))),
        ]:
            value, state = usable(row, field)
            if state != "available" or value != expected:
                reasons.add("household_count_disagreement_or_unknown")
    if union - collected:
        reasons.add("named_member_row_missing")
    if collected - union:
        reasons.add("collected_member_not_named")
    return reasons


def diagnostics(panel: Panel, sha256: str) -> dict:
    groups, people = defaultdict(list), defaultdict(dict)
    masks, native = Counter(), defaultdict(Counter)
    statuses = {
        f: Counter({str(i): 0 for i in range(10)})
        for f in FIELDS
        if "flag_for" in CONTRACT["fields"][f]
    }
    type2_rows = type2_slots = type2_roster_slots = 0
    for identity, row in sorted(panel.rows.items()):
        for field in FIELDS:
            if field != "SSUID":
                number(row[field], field)
        if number(row["THHLDSTATUS"], "THHLDSTATUS") is None:
            raise PanelError("MISSING_HOUSEHOLD_STATUS")
        person, month = identity[:2], identity[2]
        if people[person]:
            previous = next(iter(people[person].values()))
            if any(previous[f] != row[f] for f in PERSON_FIELDS):
                raise PanelError("PERSON_GRAIN_VALUE_CONFLICT")
        people[person][month] = row
        for field in statuses:
            if (
                CONTRACT["fields"][field]["grain"] != "Person"
                or len(people[person]) == 1
            ):
                code = number(row[field], field)
                statuses[field]["missing" if code is None else str(int(code))] += 1
        for field in ("THHLDSTATUS", "RIN_UNIV", "TMOVER"):
            value, state = usable(row, field)
            native[field][str(value) if value is not None else state] += 1
        scope = income_scope(row)
        masks[scope] += 1
        if scope == "pre15_without_ssi" and number(row["APTOTINC"], "APTOTINC") != 0:
            raise PanelError("PRE15_INCOME_UNIVERSE_CONFLICT")
        if not live(row):
            if (
                usable(row, "ERESIDENCEID")[0] is not None
                or usable(row, "THTOTINC")[0] is not None
            ):
                raise PanelError("INACTIVE_MONTHLY_UNIVERSE_CONFLICT")
        else:
            residence, state = usable(row, "ERESIDENCEID")
            if residence is None:
                masks["unknown_residence"] += 1
                if month == 12:
                    masks["unanchored_december_members"] += 1
            else:
                groups[(identity[0], int(residence), month)].append(row)
        refs = sum(
            number(row[f"ET2_LNO{i}"], f"ET2_LNO{i}") is not None for i in range(1, 11)
        )
        type2_rows += refs > 0
        type2_slots += refs
        type2_roster_slots += sum(
            (number(row[f"RREL_PNUM{i}"], f"RREL_PNUM{i}") or 100) < 70
            for i in range(1, 17)
        )
    monthly = {str(m): Counter() for m in range(1, 13)}
    membership_reasons = {}
    for household, rows in sorted(groups.items()):
        reasons = membership(rows)
        membership_reasons[household] = reasons
        counts = monthly[str(household[2])]
        counts["groups"] += 1
        for reason in reasons:
            counts[reason] += 1
        if reasons:
            counts["identity_skipped_membership"] += 1
            continue
        totals = {usable(r, "THTOTINC") for r in rows}
        if len(totals) != 1 or next(iter(totals))[0] is None:
            counts["identity_skipped_household_total_conflict_or_missing"] += 1
            continue
        terms, incomplete = [], False
        for row in rows:
            scope = income_scope(row)
            if scope == "pre15_without_ssi":
                counts["structural_noncontributors"] += 1
            elif scope == "in" and usable(row, "TPTOTINC")[0] is not None:
                terms.append(usable(row, "TPTOTINC")[0])
            else:
                incomplete = True
        if incomplete:
            counts["identity_skipped_income_universe"] += 1
        else:
            delta = sum(terms, Decimal(0)) - next(iter(totals))[0]
            counts["identity_tested"] += 1
            counts[
                "identity_equal"
                if abs(delta) <= len(terms) + 1
                else "identity_discrepant"
            ] += 1
    annual = Counter()
    sums, weighted = [], []
    excluded = Counter()
    contributor_count = child_count = 0
    source = measurements.SippSource(
        sha256, measurements.SIPP_2023_MASTER_SHA256, 2023, 2022
    )
    for household, rows in sorted(groups.items()):
        if household[2] != 12:
            continue
        annual["december_households"] += 1
        contributors = [
            r for r in rows if (number(r["TAGE_EHC"], "TAGE_EHC") or 0) >= 15
        ]
        contributor_count += len(contributors)
        child_count += sum(
            number(r["TAGE_EHC"], "TAGE_EHC") is not None
            and number(r["TAGE_EHC"], "TAGE_EHC") < 15
            for r in rows
        )
        reasons = {"december_" + r for r in membership_reasons[household]}
        if any(number(r["TAGE_EHC"], "TAGE_EHC") is None for r in rows):
            reasons.add("unknown_december_age")
        terms = []
        for row in contributors:
            person = key(row)[:2]
            if set(people[person]) != set(range(1, 13)):
                reasons.add("missing_contributor_months")
            for monthly_row in people[person].values():
                scope = income_scope(monthly_row)
                value, state = usable(monthly_row, "TPTOTINC")
                if scope != "in":
                    reasons.add(scope)
                if value is None:
                    reasons.add(state)
                if scope == "in" and value is not None:
                    terms.append(value)
        if not contributors:
            reasons.add("no_adult_contributors")
        # The accepted production implementation includes every live December
        # member in the mean, including children and zero weights; no rent gate.
        hh = measurements.december_households(
            [measurements.SippRow(r) for r in rows],
            source=source,
            complete_membership=True,
        )[0]
        weight = hh.weight.value
        if weight is None:
            annual["unknown_weight_households"] += 1
        elif weight == 0:
            annual["zero_weight_households"] += 1
        if reasons:
            annual["incomplete_households"] += 1
            excluded.update(reasons)
        else:
            value = sum(terms, Decimal(0))
            sums.append(value)
            annual["complete_households"] += 1
            annual[
                "negative" if value < 0 else "zero" if value == 0 else "positive"
            ] += 1
            if weight is not None:
                weighted.append(value * weight)
                annual["complete_households_with_known_weight"] += 1
    annual_once = {}
    for field in ("TLMPAMT", "TROLLAMT"):
        values, reasons = [], Counter()
        for months in people.values():
            value, state = usable(next(iter(months.values())), field)
            reasons[state] += 1
            if value is not None:
                values.append(value)
        annual_once[field] = {
            "observed_person_count": len(values),
            "observed_sum": str(sum(values, Decimal(0))) if values else None,
            "coverage": dict(reasons),
            "scope": "all_selected_persons_once; conditional_universe; not_a_full_panel_income_total",
        }
    rollover = 0
    for months in people.values():
        row = next(iter(months.values()))
        gross, _ = usable(row, "TLMPAMT")
        roll, _ = usable(row, "TROLLAMT")
        rollover += gross is not None and roll is not None and roll > gross
    return {
        "schema": "sipp-annual-panel-diagnostics/1",
        "money_basis": "nominal_USD_2022",
        "selection": {
            "rule": SELECTION,
            "limit": 32,
            "selected_ssuid_count": len(panel.selected),
            "amounts_used_for_selection": False,
        },
        "source": {"sha256": sha256, "passes": [panel.index_audit, panel.panel_audit]},
        "panel": {
            "person_month_rows": len(panel.rows),
            "persons": len(people),
            "masks": dict(masks),
            "native_codes": {k: dict(v) for k, v in native.items()},
        },
        "status_masks": {
            f: {
                "grain": CONTRACT["fields"][f]["grain"],
                "associated_field": CONTRACT["fields"][f]["flag_for"],
                "counts": dict(c),
            }
            for f, c in statuses.items()
        },
        "status_code_meanings": STATUS_CODES,
        "monthly": {k: dict(v) for k, v in monthly.items()},
        "monthly_identity": {
            "membership_scope": "unsaturated_live_vectors_and_reported_counts_on_published_file_only",
            "suppressed_tail_and_physical_household_completeness": "unknown",
            "tolerance_USD": "number_of_income_terms_plus_one; chosen_rounding_allowance_not_Census_validation",
        },
        "annual_december_adult_contributors": {
            "counts": dict(annual),
            "contributors": contributor_count,
            "under15_members_retained": child_count,
            "incomplete_causes": dict(excluded),
            "complete_household_sum": str(sum(sums, Decimal(0))) if sums else None,
            "complete_household_weighted_sum": str(sum(weighted, Decimal(0)))
            if weighted
            else None,
            "full_selected_household_sum": str(sum(sums, Decimal(0)))
            if sums
            and not annual["incomplete_households"]
            and not masks["unanchored_december_members"]
            else None,
            "weight_rule": "accepted_production_December_mean_all_live_members_including_zero_weights",
            "scope": "December_age15plus_persons_over_all_their_months_and_residences; under15_amounts_not_added",
        },
        "annual_person_fields": annual_once,
        "rollover_greater_than_gross_persons": rollover,
        "type2": {
            "respondent_month_rows_with_references": type2_rows,
            "respondent_month_slot_occurrences": type2_slots,
            "roster_slot_occurrences": type2_roster_slots,
            "unique_person_count": None,
            "annual_income_sum": None,
            "reason": "ownership_across_informants_unresolved; TT2INC_not_read",
        },
        "limitations": CONTRACT["decisions"],
        "suppressed_income_components": [
            "TSUR9AMT",
            "TSUR12AMT",
            "TDIS6AMT",
            "TDIS8AMT",
            "TDIS9AMT",
            "TVA5AMT",
        ],
        "recipient_totals": "HTOTVAL_and_HINCP_unchanged; no_recipient_reads",
    }


def extract(path: Path, sha256: str, size: int) -> dict:
    with localcontext(measurements._ARITHMETIC_CONTEXT):
        return diagnostics(read_panel(path, sha256, size), sha256)
