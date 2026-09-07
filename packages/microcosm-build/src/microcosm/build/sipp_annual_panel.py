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
SELECTION_LIMIT = 32
DIAGNOSTICS_SCHEMA = "sipp-annual-panel-diagnostics/1"
# Version 1 is the accepted module at commit 02e7b2933035ae609187d1c715531ca300ed50d7.
# Version 2 lifts the annual December calculation out of ``diagnostics`` into
# ``annual_december_households`` so that a second consumer cannot grow a second
# income definition; every aggregate value ``diagnostics`` returns is unchanged
# and ``DIAGNOSTICS_SCHEMA`` therefore does not move. The module bytes did move,
# so an artifact that binds these records must bind version 2, not version 1.
# The version 1 receipt already recorded under 02e7b293 remains valid for the
# bytes it names.
ANNUAL_PANEL_VERSION = "sipp-annual-panel/2"
SUPERSEDED_ANNUAL_PANEL_VERSIONS = ("sipp-annual-panel/1",)
VERSION_1_MODULE_SHA256 = (
    "65c47fed0bd54ff856f432a9c8797f128324849663dc842f75d734e9c8f0cc21"
)
ANNUAL_RECORDS_VERSION = "sipp-annual-december-households/1"
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


@dataclass(frozen=True)
class PanelState:
    """Every structure the aggregate reduction and the record builder both read.

    Built exactly once per panel by :func:`panel_state`, whose loop is the
    module's only place that validates a row and decides monthly membership.
    """

    people: dict[tuple[str, int], dict[int, dict]]
    groups: dict[tuple[str, int, int], list[dict]]
    membership_reasons: dict[tuple[str, int, int], set[str]]
    masks: Counter
    native: dict[str, Counter]
    statuses: dict[str, Counter]
    type2_rows: int
    type2_slots: int
    type2_roster_slots: int


def panel_state(panel: Panel) -> PanelState:
    """Validate every projected field and group the panel, once."""
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
    return PanelState(
        people,
        groups,
        {h: membership(rows) for h, rows in sorted(groups.items())},
        masks,
        native,
        statuses,
        type2_rows,
        type2_slots,
        type2_roster_slots,
    )


@dataclass(frozen=True)
class AnnualContributorMonth:
    """One adult contributor month, with the codes that decided its usability."""

    person: tuple[str, int]
    month: int
    household_status: int | None
    frame: int | None
    residence: int | None
    residence_state: str
    age: int | None
    scope: str
    income_status_code: int | None
    income_state: str
    amount: Decimal | None


@dataclass(frozen=True)
class AnnualDecemberHousehold:
    """One December residence's annual adult-contributor measurement.

    ``amount`` is present only when no reason blocked the measure. A blocked
    household keeps every reason and every month it read; nothing here is filled
    with a monetary zero, and no reason is a fit or release admission.
    """

    key: tuple[str, int]
    members: tuple[tuple[str, int], ...]
    december_person_months: tuple[tuple[str, int, int], ...]
    contributors: tuple[tuple[str, int], ...]
    children_under15: tuple[tuple[str, int], ...]
    unknown_age_members: tuple[tuple[str, int], ...]
    contributor_months: tuple[AnnualContributorMonth, ...]
    contributor_months_present: dict[tuple[str, int], tuple[int, ...]]
    membership_reasons: tuple[str, ...]
    reasons: tuple[str, ...]
    amount: Decimal | None
    weight: measurements.Measurement
    source: measurements.SippSource

    @property
    def weight_state(self) -> str:
        if self.weight.value is None:
            return "unknown"
        return "zero" if self.weight.value == 0 else "positive"


def annual_identity() -> dict:
    """The annual identity a downstream artifact must bind, recomputed from bytes."""
    return {
        "module_version": ANNUAL_PANEL_VERSION,
        "superseded_module_versions": list(SUPERSEDED_ANNUAL_PANEL_VERSIONS),
        "version_1_module_sha256": VERSION_1_MODULE_SHA256,
        "records_version": ANNUAL_RECORDS_VERSION,
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "contract_sha256": CONTRACT_SHA256,
        "diagnostics_schema": DIAGNOSTICS_SCHEMA,
        "selection": SELECTION,
        "selection_limit": SELECTION_LIMIT,
        "fields": list(FIELDS),
        "money_basis": CONTRACT["money_basis"],
        "survey_year": CONTRACT["survey_year"],
        "income_year": CONTRACT["income_year"],
    }


def annual_december_households(
    panel: Panel, source_sha256: str, *, state: PanelState | None = None
) -> tuple[AnnualDecemberHousehold, ...]:
    """The annual December-residence measurement, as reusable per-household records.

    This is the module's only implementation of that measure: ``diagnostics``
    reduces exactly these records, so an aggregate count and a per-household
    record cannot drift into different income definitions. Nothing is opened and
    ``source_sha256`` is a declaration the caller owns, not proof that this
    panel's rows came from those bytes.
    """
    if state is None:
        state = panel_state(panel)
    source = measurements.SippSource(
        source_sha256, measurements.SIPP_2023_MASTER_SHA256, 2023, 2022
    )
    with localcontext(measurements._ARITHMETIC_CONTEXT):
        return _annual_records(state, source)


def _annual_records(
    state: PanelState, source: measurements.SippSource
) -> tuple[AnnualDecemberHousehold, ...]:
    records = []
    for household, rows in sorted(state.groups.items()):
        if household[2] != 12:
            continue
        contributors = [
            r for r in rows if (number(r["TAGE_EHC"], "TAGE_EHC") or 0) >= 15
        ]
        children = [
            r
            for r in rows
            if number(r["TAGE_EHC"], "TAGE_EHC") is not None
            and number(r["TAGE_EHC"], "TAGE_EHC") < 15
        ]
        unknown_age = [r for r in rows if number(r["TAGE_EHC"], "TAGE_EHC") is None]
        reasons = {"december_" + r for r in state.membership_reasons[household]}
        if unknown_age:
            reasons.add("unknown_december_age")
        terms, months, present = [], [], {}
        for row in contributors:
            person = key(row)[:2]
            present[person] = tuple(state.people[person])
            if set(present[person]) != set(range(1, 13)):
                reasons.add("missing_contributor_months")
            for month, monthly_row in state.people[person].items():
                scope = income_scope(monthly_row)
                value, income_state = usable(monthly_row, "TPTOTINC")
                if scope != "in":
                    reasons.add(scope)
                if value is None:
                    reasons.add(income_state)
                if scope == "in" and value is not None:
                    terms.append(value)
                residence, residence_state = usable(monthly_row, "ERESIDENCEID")
                status = number(monthly_row["THHLDSTATUS"], "THHLDSTATUS")
                frame = number(monthly_row["RIN_UNIV"], "RIN_UNIV")
                age = number(monthly_row["TAGE_EHC"], "TAGE_EHC")
                code = number(monthly_row["APTOTINC"], "APTOTINC")
                months.append(
                    AnnualContributorMonth(
                        person,
                        month,
                        None if status is None else int(status),
                        None if frame is None else int(frame),
                        None if residence is None else int(residence),
                        residence_state,
                        None if age is None else int(age),
                        scope,
                        None if code is None else int(code),
                        income_state,
                        value,
                    )
                )
        if not contributors:
            reasons.add("no_adult_contributors")
        # The accepted production implementation includes every live December
        # member in the mean, including children and zero weights; no rent gate.
        hh = measurements.december_households(
            [measurements.SippRow(r) for r in rows],
            source=source,
            complete_membership=True,
        )[0]
        records.append(
            AnnualDecemberHousehold(
                (household[0], household[1]),
                tuple(key(r)[:2] for r in rows),
                tuple(key(r) for r in rows),
                tuple(key(r)[:2] for r in contributors),
                tuple(key(r)[:2] for r in children),
                tuple(key(r)[:2] for r in unknown_age),
                tuple(months),
                present,
                tuple(sorted(state.membership_reasons[household])),
                tuple(sorted(reasons)),
                None if reasons else sum(terms, Decimal(0)),
                hh.weight,
                source,
            )
        )
    return tuple(records)


def diagnostics(panel: Panel, sha256: str) -> dict:
    state = panel_state(panel)
    groups, people = state.groups, state.people
    masks, native, statuses = state.masks, state.native, state.statuses
    monthly = {str(m): Counter() for m in range(1, 13)}
    for household, rows in sorted(groups.items()):
        reasons = state.membership_reasons[household]
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
    # One shared calculation: the aggregate below is a reduction of exactly the
    # records a downstream artifact consumes, never a second income definition.
    annual = Counter()
    sums, weighted = [], []
    excluded = Counter()
    contributor_count = child_count = 0
    for record in annual_december_households(panel, sha256, state=state):
        annual["december_households"] += 1
        contributor_count += len(record.contributors)
        child_count += len(record.children_under15)
        weight = record.weight.value
        if weight is None:
            annual["unknown_weight_households"] += 1
        elif weight == 0:
            annual["zero_weight_households"] += 1
        if record.reasons:
            annual["incomplete_households"] += 1
            excluded.update(record.reasons)
        else:
            value = record.amount
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
            value, state_name = usable(next(iter(months.values())), field)
            reasons[state_name] += 1
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
        "schema": DIAGNOSTICS_SCHEMA,
        "money_basis": "nominal_USD_2022",
        "selection": {
            "rule": SELECTION,
            "limit": SELECTION_LIMIT,
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
            "respondent_month_rows_with_references": state.type2_rows,
            "respondent_month_slot_occurrences": state.type2_slots,
            "roster_slot_occurrences": state.type2_roster_slots,
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
