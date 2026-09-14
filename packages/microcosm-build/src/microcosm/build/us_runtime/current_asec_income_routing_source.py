"""Reporting/routing projection for five current ASEC original income families.

Pension/annuity, retirement distributions, net property income, farm operations
and other income are the original-channel leaves the current survey graph does
not yet supply. This module qualifies the exact retained 2025 ASEC person member
against the authenticated current-money owner and projects the source literals
those families publish: totals, receipt universes, routing codes and allocation
provenance.

Nothing here observes a taxable amount. The combined pension total is not split,
account identity is not a taxable fraction, the net property total is not
independently labelled rental, farm losses are not nonfarm self-employment, and
no residual other-income category rule is applied. Returned values are
descriptive: a consuming host requalifies the retained preparation and compares
these values before and after its own I/O. This module issues no source
admission and grants no release eligibility.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from . import asec_coverage_authentication as coverage
from . import asec_current_money as money
from . import source_csv_builtin
from . import survey_population_preparation as source
from .support_provenance import spine_source_id_column, support_channel_column

PROTOCOL = "microcosm.us.current-asec-income-routing-source.v1"
DICTIONARY_URL = (
    "https://www2.census.gov/programs-surveys/cps/datasets/2025/march/"
    "asec2025_ddl_pub_full.pdf"
)
DICTIONARY_SHA256 = "5cb80973326ef8b625fbaae70d80b0c641ce5d2b3911abd2fb4427abd5908a6f"

COORDINATE_COLUMNS = ("PERIDNUM", "PH_SEQ", "A_LINENO", "A_AGE")
COORDINATE_WIDTHS = {"PH_SEQ": 5, "A_LINENO": 2, "A_AGE": 2}
TOKEN_MAX_CHARS = 64

# The nine printed money entries are already attested in the packaged current
# money domains artifact, so they are read from it rather than re-typed here.
DOMAINS_RESOURCE = "asec_current_money_domains_v1.json"
DOMAINS_SHA256 = money.RESOURCE_PINS[0]
CURRENT_INCOME_YEAR = 2024
AMOUNT_FIELDS = (
    "PNSN_VAL",
    "ANN_VAL",
    "DST_VAL1",
    "DST_VAL1_YNG",
    "DST_VAL2",
    "DST_VAL2_YNG",
    "RNT_VAL",
    "FRSE_VAL",
    "OI_VAL",
)


@dataclass(frozen=True)
class PrintedAmountEntry:
    """One money entry exactly as the pinned domains artifact records it."""

    name: str
    printed_length: int
    printed_position: int
    printed_page: str
    pdf_page_1based: int
    encoded_minimum: int
    encoded_maximum: int
    universe_as_printed: str
    values_as_printed: str
    negative_dollars_permitted: bool
    nonmoney_codes: tuple[int, ...]
    zero_semantics: str


@lru_cache(maxsize=1)
def _cached_printed_amount_entries():
    """Read the pinned domains artifact once, on demand, never at import."""
    payload = resources.files(__package__).joinpath(DOMAINS_RESOURCE).read_bytes()
    require(_sha(payload) == DOMAINS_SHA256, "DOMAINS_RESOURCE_SHA256")
    data = json.loads(payload)
    entries = {}
    for field in data["fields"]:
        if field["name"] not in AMOUNT_FIELDS:
            continue
        require(field["name"] not in entries, "DOMAIN_FIELD_DUPLICATE:" + field["name"])
        domain = field["domain"]
        vintages = [
            v for v in field["vintages"] if v["income_year"] == CURRENT_INCOME_YEAR
        ]
        require(len(vintages) == 1, "DOMAIN_VINTAGE:" + field["name"])
        vintage = vintages[0]
        require(
            vintage["dictionary_spelling"] == field["name"]
            and vintage["pdf_sha256"] == DICTIONARY_SHA256
            and vintage["source_url"] == DICTIONARY_URL,
            "DOMAIN_DICTIONARY_PIN:" + field["name"],
        )
        entries[field["name"]] = PrintedAmountEntry(
            name=field["name"],
            printed_length=vintage["ascii_length_as_printed"],
            printed_position=vintage["ascii_position_as_printed"],
            printed_page=vintage["printed_page"],
            pdf_page_1based=vintage["pdf_page_1based"],
            encoded_minimum=domain["encoded_range_inclusive"]["minimum"],
            encoded_maximum=domain["encoded_range_inclusive"]["maximum"],
            universe_as_printed=vintage["universe_as_printed"],
            values_as_printed=vintage["values_description"],
            negative_dollars_permitted=domain["negative_dollars_permitted"],
            nonmoney_codes=tuple(domain["declared_negative_nonmoney_codes"]),
            zero_semantics=domain["zero_semantics"],
        )
    require(set(entries) == set(AMOUNT_FIELDS), "DOMAIN_FIELD_ROSTER")
    return tuple(entries.items())


def printed_amount_entries():
    """Return a detached mapping over privately cached immutable entries."""
    return dict(_cached_printed_amount_entries())


# Retain the existing test/revalidation cache-control surface.
printed_amount_entries.cache_clear = _cached_printed_amount_entries.cache_clear


def _require_nonnegative_amount_domain(
    field, vintage, *, zero_semantics, valid_minimum
):
    """Reject domains unsupported by the sibling unsigned-literal parsers.

    Amount bounds remain artifact-derived. The checks bind the exact semantics
    assumed by those parsers rather than silently adapting a later artifact.
    """
    name, domain = field["name"], field["domain"]
    maximum = domain["encoded_range_inclusive"]["maximum"]
    require(
        field["entity"] == field["grain"] == "person"
        and field["column"] == name
        and type(maximum) is int
        and maximum > 0
        and domain["encoded_range_inclusive"] == {"minimum": 0, "maximum": maximum}
        and field["minimum"] == 0
        and field["maximum"] == maximum
        and vintage["range_header"] == domain["encoded_range_inclusive"]
        and domain["negative_dollars_permitted"] is False
        and domain["declared_negative_nonmoney_codes"] == []
        and domain["declared_other_missing_codes"] == []
        and domain["declared_niu_codes"] == field["declared_niu_codes"] == [0]
        and domain["valid_dollar_range_excludes"] == []
        and domain["valid_dollar_range_inclusive"]
        == {"minimum": valid_minimum, "maximum": maximum}
        and domain["numeric_type"] == "integer_US_dollars_in_public_use_dictionary"
        and domain["zero_semantics"] == field["zero_semantics"] == zero_semantics,
        "UNSUPPORTED_AMOUNT_DOMAIN:" + name,
    )


def _require_live_amount_domains(ready, entries):
    """Bind source domain triples to a unique retained MoneyDomain roster."""
    fields = ready.bindings.spec.fields
    domains = {d.name: d for d in fields}
    require(len(domains) == len(fields), "MONEY_DOMAIN_DUPLICATE")
    for name, (minimum, maximum, zero_semantics) in entries.items():
        domain = domains.get(name)
        require(domain is not None, "MONEY_DOMAIN_MISSING:" + name)
        require(
            domain.entity == domain.grain == "person"
            and domain.column == name
            and domain.minimum == minimum
            and domain.maximum == maximum
            and domain.zero_semantics == zero_semantics,
            "MONEY_DOMAIN_DISAGREEMENT:" + name,
        )


# Printed yes/no entries. The zero label is not shared: OI_YN prints
# "none or niu" where the others print "niu", so a zero receipt literal does not
# carry the same meaning across families.
class ReceiptEntry(NamedTuple):
    printed_length: int
    printed_position: int
    pdf_page_1based: int
    printed_page: str
    universe_as_printed: str
    zero_label_as_printed: str


RECEIPT_ENTRIES = {
    "PEN_YN": ReceiptEntry(1, 570, 47, "6C-26", "All Persons aged 15+", "niu"),
    "ANN_YN": ReceiptEntry(1, 444, 44, "6C-23", "All Persons aged 15+", "niu"),
    "DST_YN": ReceiptEntry(
        1,
        519,
        46,
        "6C-25",
        "Persons aged 58 and over (a_age \u2265 58)",
        "niu",
    ),
    "DST_YN_YNG": ReceiptEntry(
        1, 520, 46, "6C-25", "Persons under age 58 (a_age < 58)", "niu"
    ),
    "RNT_YN": ReceiptEntry(1, 627, 49, "6C-28", "All Persons aged 15+", "niu"),
    "FRSE_YN": ReceiptEntry(1, 397, 43, "6C-22", "ERN_YN=1 or FRMOTR=1", "Niu"),
    "ERN_YN": ReceiptEntry(1, 381, 43, "6C-22", "WORKYN=1 OR WTEMP=1", "niu"),
    "FRMOTR": ReceiptEntry(1, 389, 43, "6C-22", "ERN_OTR = 1", "niu"),
    "OI_YN": ReceiptEntry(1, 555, 47, "6C-26", "All Persons aged 15+", "none or niu"),
}
RECEIPT_CODE_DOMAIN = (0, 1, 2)


def receipt_codes(field):
    """Printed yes/no labels for one entry; the zero label is not shared.

    OI_YN prints "none or niu" where the other entries print "niu"/"Niu", so a
    single shared label would attribute a "respondent reported none" reading to
    eight fields whose dictionary entry does not support it.
    """
    return {0: RECEIPT_ENTRIES[field].zero_label_as_printed, 1: "yes", 2: "no"}


# Retirement account identity. Code 4 names a regular IRA; it does not observe
# any taxable fraction of the distribution.
ACCOUNT_CODES = {
    0: "NIU",
    1: "401k account",
    2: "403b account",
    3: "Roth IRA",
    4: "Regular IRA",
    5: "KEOGH plan",
    6: "SEP plan (Simplified Employee Pension)",
    7: "Other type of retirement account",
}
REGULAR_IRA_CODE = 4


class AccountEntry(NamedTuple):
    printed_length: int
    printed_position: int
    pdf_page_1based: int
    printed_page: str
    universe_as_printed: str


ACCOUNT_ENTRIES = {
    "DST_SC1": AccountEntry(1, 491, 45, "6C-24", "DST_VAL1 > 0 and a_age \u2265 58"),
    "DST_SC1_YNG": AccountEntry(1, 492, 45, "6C-24", "DST_YN_YNG = 1 and a_age < 58"),
    "DST_SC2": AccountEntry(1, 493, 45, "6C-24", "DST_VAL2 > 0 and a_age \u2265 58"),
    "DST_SC2_YNG": AccountEntry(1, 494, 45, "6C-24", "DST_VAL_YNG > 0 and a_age < 58"),
}

# OI_OFF, verbatim. Code 20 is the reported alimony category. No residual rule
# maps any other code, including 19 "anything else", onto alimony.
OTHER_INCOME_CATEGORIES = {
    0: "niu",
    1: "social security",
    2: "private pensions",
    3: "afdc",
    4: "other public assistance",
    5: "interest",
    6: "dividends",
    7: "rents or royalties",
    8: "estates or trusts",
    9: "state disability payments (worker's comp)",
    10: "disability payments (own insurance)",
    11: "unemployment compensation",
    12: "strike benefits",
    13: "annuities or paid up insurance policies",
    14: "not income",
    15: "longest job",
    16: "wages or salary",
    17: "nonfarm self-employment",
    18: "farm self-employment",
    19: "anything else",
    20: "alimony",
}
ALIMONY_CATEGORY_CODE = 20
OTHER_INCOME_CATEGORY_ENTRY = AccountEntry(2, 547, 47, "6C-26", "OI_YN = 1")

# Published allocation flags. Values 0-9 follow I_ANNVAL; the DST composites
# follow I_INTYN (0, 10, 11); I_DSTSC prints its own 0/1/9 set. I_FRMYN prints
# an empty Values block, so only its (0:9) range header is published and the
# meaning of its codes is not; it is accepted on the printed range alone.
ALLOCATION_ANNVAL_CODES = tuple(range(10))
ALLOCATION_COMPOSITE_CODES = (0, 10, 11)
ALLOCATION_DSTSC_CODES = (0, 1, 9)
# I_FRMYN's printed Values block is empty; its (0:9) range header is all that is
# published, so no code meaning is claimed for it.
ALLOCATION_PRINTED_RANGE_CODES = tuple(range(10))
ALLOCATION_CODE_MEANINGS_UNPUBLISHED = frozenset({"I_FRMYN"})


class AllocationEntry(NamedTuple):
    printed_length: int
    printed_position: int
    pdf_page_1based: int
    printed_page: str
    universe_as_printed: str
    codes: tuple


ALLOCATION_ENTRIES = {
    "I_ANNVAL": AllocationEntry(
        1, 802, 53, "6C-32", "ANN_YN =1", ALLOCATION_ANNVAL_CODES
    ),
    "I_ANNYN": AllocationEntry(
        1, 803, 53, "6C-32", "ANN_YN > 0", ALLOCATION_ANNVAL_CODES
    ),
    "I_DSTSC": AllocationEntry(
        1, 821, 55, "6C-34", "DST_YN =1", ALLOCATION_DSTSC_CODES
    ),
    "I_DSTSCCOMP": AllocationEntry(
        1,
        822,
        55,
        "6C-34",
        "DST_YN = 1 or DST_YNG_YN = 1",
        ALLOCATION_ANNVAL_CODES,
    ),
    "I_DSTVAL1COMP": AllocationEntry(
        2, 823, 55, "6C-34", "", ALLOCATION_COMPOSITE_CODES
    ),
    "I_DSTVAL2COMP": AllocationEntry(
        2, 825, 55, "6C-34", "DST_VAL2> 0", ALLOCATION_COMPOSITE_CODES
    ),
    "I_DSTYNCOMP": AllocationEntry(
        2, 827, 55, "6C-34", "DST_YN > 0", ALLOCATION_COMPOSITE_CODES
    ),
    "I_ERNYN": AllocationEntry(
        1, 833, 55, "6C-34", "ERN_YN > 0", ALLOCATION_ANNVAL_CODES
    ),
    "I_FRMYN": AllocationEntry(
        1, 837, 55, "6C-34", "FRMOTR > 0", ALLOCATION_PRINTED_RANGE_CODES
    ),
    "I_OIVAL": AllocationEntry(
        1, 843, 56, "6C-35", "OI_VAL > 0", ALLOCATION_ANNVAL_CODES
    ),
    "I_PENYN": AllocationEntry(
        1, 854, 57, "6C-36", "PEN_YN > 0", ALLOCATION_ANNVAL_CODES
    ),
    "I_RNTVAL": AllocationEntry(
        1, 861, 57, "6C-36", "RNT_VAL > 0", ALLOCATION_ANNVAL_CODES
    ),
    "I_RNTYN": AllocationEntry(
        1, 862, 57, "6C-36", "RNT_YN > 0", ALLOCATION_ANNVAL_CODES
    ),
}

# Which printed field each published flag names, and the fields for which the
# 2025 dictionary publishes no flag at all. A family holding an unflagged field
# can never report a clean "no allocation": that evidence is simply not printed.
PUBLISHED_ALLOCATION_FLAG_BY_FIELD = {
    "ANN_VAL": "I_ANNVAL",
    "ANN_YN": "I_ANNYN",
    "PEN_YN": "I_PENYN",
    "RNT_VAL": "I_RNTVAL",
    "RNT_YN": "I_RNTYN",
    "OI_VAL": "I_OIVAL",
    "ERN_YN": "I_ERNYN",
    "FRMOTR": "I_FRMYN",
    "DST_SC1": "I_DSTSC",
    "DST_SC2": "I_DSTSC",
    "DST_VAL1": "I_DSTVAL1COMP",
    "DST_VAL2": "I_DSTVAL2COMP",
    "DST_YN": "I_DSTYNCOMP",
}
UNFLAGGED_FIELDS = (
    "PNSN_VAL",
    "FRSE_VAL",
    "FRSE_YN",
    "OI_OFF",
    "OI_YN",
    "DST_VAL1_YNG",
    "DST_VAL2_YNG",
    "DST_YN_YNG",
    "DST_SC1_YNG",
    "DST_SC2_YNG",
)
# The DST_SC(2) notation is read here as "the two DST_SC slots", which is an
# inference from the printed parenthesis rather than a printed statement.
AMBIGUOUS_FLAG_COVERAGE = {
    "I_DSTSC": "printed label names DST_SC(2); mapping it to both DST_SC1 and "
    "DST_SC2 reads the parenthesis as a slot count, which the dictionary does "
    "not state",
    "I_DSTSCCOMP": "printed label names DST_SC(2) while its printed universe "
    "names DST_YN = 1 or DST_YNG_YN = 1; whether it covers the under-58 source "
    "codes is unresolved",
    "I_FRMYN": "printed Values block is empty, so the meaning of its codes is "
    "not published; only the (0:9) range header is",
}

READ_COLUMNS = (
    *COORDINATE_COLUMNS,
    *AMOUNT_FIELDS,
    *RECEIPT_ENTRIES,
    *ACCOUNT_ENTRIES,
    "OI_OFF",
    *ALLOCATION_ENTRIES,
)

# Printed labels that stay separable from any modelled component.
PENSION_TOTAL_SCOPE = (
    "total combined amount of pension income received from all pension sources"
)
NET_PROPERTY_AMOUNT_SCOPE = "income from rent after expenses"
NET_PROPERTY_RECEIPT_SCOPE = (
    "own any land, property, rented to others, or receive income from royalties, "
    "roomers or boarders, or from estates or trusts"
)
FARM_AMOUNT_SCOPE = (
    "total amount of farm self-employment earnings (combined amounts in "
    "ERN_VAL, if ERN_SRCE=3, and FRM_VAL)"
)

FAMILIES = (
    "pension_annuity",
    "retirement_distribution",
    "net_property",
    "farm",
    "other_income",
)


def require(condition, reason):
    if not condition:
        raise ValueError("CURRENT_ASEC_INCOME_ROUTING_" + reason)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def amount_patterns():
    """Token shapes from the printed width and the printed encoded minimum."""
    entries = printed_amount_entries()
    return {
        name: re.compile(
            ("-?" if entry.encoded_minimum < 0 else "")
            + r"[0-9]{1,"
            + str(entry.printed_length)
            + r"}",
            re.ASCII,
        )
        for name, entry in entries.items()
    }


def literal_code(token, allowed, *, width):
    """Keep an unreadable literal unknown; never recode it to a printed value.

    Missing, malformed and out-of-range literals stay distinguishable from each
    other and from a printed code, so a consumer cannot mistake an unresolved
    routing answer for an observed one. A literal wider than the printed entry
    is malformed, not a zero-padded reading of a shorter code.
    """
    require(type(token) is str and len(token) <= TOKEN_MAX_CHARS, "TOKEN_BOUND")
    if token == "":
        return None, "missing"
    if re.fullmatch(r"[0-9]{1," + str(width) + r"}", token, re.ASCII) is None:
        return None, "malformed"
    value = int(token)
    if value not in allowed:
        return None, "outside_printed_range"
    return value, "in_printed_range"


def amount_state(value, status):
    """Read the dollar meaning of one cell from the parent's own status axis.

    The authenticated money owner normalizes ANN_VAL's printed -1 to a stored
    zero and records DECLARED_NIU. Re-deriving NIU from the stored number would
    read that cell as a zero dollar annuity, so the status axis decides.
    """
    status = int(status)
    if status == money.CodebookStatus.MISSING_NULL:
        require(np.isnan(value), "MISSING_STATUS_WITH_AMOUNT")
        return "missing", np.nan
    require(np.isfinite(value), "AMOUNT_NOT_FINITE")
    if status == money.CodebookStatus.DECLARED_NIU:
        return "declared_niu", np.nan
    return ("zero" if value == 0 else "nonzero"), float(value)


# Statuses whose dollar reading is established by the source itself. Every other
# status leaves the canonical amount unknown rather than completing it with zero.
# A recipient zero qualifies only where the printed entry says the zero is valid
# dollars; every entry here except ANN_VAL prints "0 = none or niu", which the
# pinned domains artifact records as not distinguishable from an amount alone.
DOLLAR_ZERO_SEMANTICS = "valid_zero_dollars"
KNOWN_AMOUNT_STATUSES = (
    "known_receipt",
    "known_recipient_zero",
    "known_nonreceipt",
)


def receipt_status(universe, receipt, amount_kind, *, net_measure, zero_is_dollars):
    """Classify one family row without inventing receipt, absence or dollars.

    `universe` is True, False, or None when the printed universe itself is not
    resolved by the retained literals.

    A zero amount under a yes answer establishes a dollar reading only when the
    printed entry says its zero is valid dollars. `net_measure` does not make a
    zero known: it only separates a signed net entry's recipient zero, which a
    consumer may want to treat differently, from a gross entry's. Both stay
    unknown, because both print "0 = none or niu".
    """
    code, status = receipt
    if universe is None:
        return "unresolved_reporting_universe"
    if universe is False:
        outside = code == 0 and amount_kind in ("zero", "declared_niu", "missing")
        return (
            "outside_reporting_universe"
            if outside
            else "contradictory_outside_reporting_universe"
        )
    if status == "missing":
        return "missing_receipt_literal"
    if status != "in_printed_range":
        return "unrecognized_receipt_literal"
    if amount_kind == "declared_niu":
        return "niu" if code == 0 else "contradictory_declared_niu_amount"
    if amount_kind == "missing":
        return "missing_amount"
    if code == 0:
        return "niu" if amount_kind == "zero" else "contradictory_niu_nonzero"
    if code == 1:
        if amount_kind == "nonzero":
            return "known_receipt"
        if zero_is_dollars:
            return "known_recipient_zero"
        return "receipt_with_net_zero" if net_measure else "ambiguous_recipient_zero"
    return "known_nonreceipt" if amount_kind == "zero" else "contradictory_no_nonzero"


def _classify(amounts, statuses, universes, receipts, *, net_measure, zero_is_dollars):
    """Row-wise family classification returning labels and canonical amounts."""
    labels, canonical = [], np.full(len(amounts), np.nan, dtype=np.float64)
    kinds, sources = [], np.full(len(amounts), np.nan, dtype=np.float64)
    for i, value in enumerate(amounts):
        kind, dollars = amount_state(value, statuses[i])
        kinds.append(kind)
        sources[i] = dollars
        label = receipt_status(
            universes[i],
            receipts[i],
            kind,
            net_measure=net_measure,
            zero_is_dollars=zero_is_dollars,
        )
        labels.append(label)
        if label in KNOWN_AMOUNT_STATUSES:
            canonical[i] = dollars if label == "known_receipt" else 0.0
    return labels, canonical, kinds, sources


def _zero_is_dollars(field):
    """Whether the printed entry says its zero is valid dollars, not none/niu."""
    return printed_amount_entries()[field].zero_semantics == DOLLAR_ZERO_SEMANTICS


def _codes_frame(prefix, tokens, allowed, labels=None, *, width=1):
    """Typed code/label/status columns for one printed routing literal."""
    pairs = [literal_code(t, allowed, width=width) for t in tokens]
    frame = pd.DataFrame(index=range(len(tokens)))
    frame[prefix + "_literal"] = pd.array(list(tokens), dtype="string")
    frame[prefix + "_code"] = pd.array([c for c, _ in pairs], dtype="Int16")
    frame[prefix + "_literal_status"] = pd.array([s for _, s in pairs], dtype="string")
    if labels is not None:
        frame[prefix + "_label"] = pd.array(
            [labels[c] if c is not None else pd.NA for c, _ in pairs], dtype="string"
        )
    return frame, [c for c, _ in pairs], [s for _, s in pairs]


def _age_universe(ages):
    return [bool(a >= 15) for a in ages]


def _allocation_origin(flags, unflagged):
    """Family allocation provenance from published flags only.

    Every published flag here prints a conditional universe (for example
    I_RNTVAL is printed for RNT_VAL > 0), and those universes are not evaluated
    here. An unpopulated literal is therefore not a defect, and an all-zero
    reading is not an assertion of non-allocation: on a non-recipient row the
    flag is outside its own printed universe. The zero readings are reported as
    exactly that, never as publisher-confirmed absence of allocation.
    A readable nonzero code whose meaning is unpublished cannot establish
    allocation; a documented allocation flag can still establish it.
    """
    origins = []
    count = len(next(iter(flags.values()))[0])
    for i in range(count):
        codes = {f: flags[f][0][i] for f in flags}
        statuses = [flags[f][1][i] for f in flags]
        if any(s not in ("missing", "in_printed_range") for s in statuses):
            origins.append("unresolved_allocation_provenance")
        elif any(
            c is not None and c != 0
            for f, c in codes.items()
            if f not in ALLOCATION_CODE_MEANINGS_UNPUBLISHED
        ):
            origins.append("publisher_allocated")
        elif any(
            c is not None and c != 0
            for f, c in codes.items()
            if f in ALLOCATION_CODE_MEANINGS_UNPUBLISHED
        ):
            origins.append("allocation_code_meaning_unpublished")
        elif any(s == "missing" for s in statuses):
            origins.append("allocation_flag_not_populated")
        elif unflagged:
            origins.append("published_flags_all_zero_with_unflagged_fields")
        else:
            origins.append("published_flags_all_zero")
    return pd.array(origins, dtype="string")


def _pension_annuity(raw, ages):
    """Separate pension and annuity totals; the combined total is never split."""
    out = pd.DataFrame(index=range(len(ages)))
    universe = _age_universe(ages)
    for prefix, amount_field, receipt_field in (
        ("pension", "PNSN_VAL", "PEN_YN"),
        ("annuity", "ANN_VAL", "ANN_YN"),
    ):
        printed = receipt_codes(receipt_field)
        frame, codes, statuses = _codes_frame(
            "pension_annuity_" + prefix + "_receipt",
            raw[receipt_field],
            printed,
            printed,
        )
        labels, canonical, kinds, sources = _classify(
            raw["amounts"][amount_field],
            raw["statuses"][amount_field],
            universe,
            list(zip(codes, statuses, strict=True)),
            net_measure=False,
            zero_is_dollars=_zero_is_dollars(amount_field),
        )
        out = pd.concat([out, frame], axis=1)
        out["pension_annuity_" + prefix + "_source_total"] = sources
        out["pension_annuity_" + prefix + "_amount_kind"] = pd.array(
            kinds, dtype="string"
        )
        out["pension_annuity_" + prefix + "_reporting_status"] = pd.array(
            labels, dtype="string"
        )
        out["pension_annuity_" + prefix + "_known_amount"] = canonical
    out["pension_annuity_source_reporting_universe"] = pd.array(
        universe, dtype="boolean"
    )
    out["pension_annuity_combined_total_scope"] = pd.array(
        [PENSION_TOTAL_SCOPE] * len(ages), dtype="string"
    )
    out["pension_annuity_private_share_applied"] = pd.array(
        [False] * len(ages), dtype="boolean"
    )
    out["pension_annuity_taxable_amount_known"] = pd.array(
        [False] * len(ages), dtype="boolean"
    )
    out["pension_annuity_allocation_origin"] = _allocation_origin(
        {f: raw["allocations"][f] for f in ("I_PENYN", "I_ANNVAL", "I_ANNYN")},
        unflagged=True,
    )
    out["pension_annuity_pension_total_has_published_flag"] = pd.array(
        [False] * len(ages), dtype="boolean"
    )
    return out


def _retirement_distribution(raw, ages):
    """Preserve every printed slot, the age route and the unresolved tax share."""
    rows = len(ages)
    out = pd.DataFrame(index=range(rows))
    route = ["age58_and_over" if a >= 58 else "under_age58" for a in ages]
    out["retirement_distribution_route"] = pd.array(route, dtype="string")
    slots = (
        ("slot1", "DST_SC1", "DST_VAL1", "age58_and_over"),
        ("slot2", "DST_SC2", "DST_VAL2", "age58_and_over"),
        ("slot1_young", "DST_SC1_YNG", "DST_VAL1_YNG", "under_age58"),
        ("slot2_young", "DST_SC2_YNG", "DST_VAL2_YNG", "under_age58"),
    )
    slot_codes, slot_amounts, slot_kinds, slot_status = {}, {}, {}, {}
    for name, code_field, amount_field, slot_route in slots:
        prefix = "retirement_distribution_" + name + "_account"
        frame, codes, statuses = _codes_frame(
            prefix, raw[code_field], ACCOUNT_CODES, ACCOUNT_CODES
        )
        out = pd.concat([out, frame], axis=1)
        values = raw["amounts"][amount_field]
        slot_statuses = raw["statuses"][amount_field]
        kinds = [amount_state(v, slot_statuses[i])[0] for i, v in enumerate(values)]
        out["retirement_distribution_" + name + "_amount"] = values
        out["retirement_distribution_" + name + "_applicable"] = pd.array(
            [r == slot_route for r in route], dtype="boolean"
        )
        statuses_out = []
        for i in range(rows):
            if route[i] != slot_route:
                statuses_out.append(
                    "off_route"
                    if kinds[i] in ("zero", "missing")
                    else "off_route_nonzero"
                )
            elif codes[i] is None:
                statuses_out.append("unresolved_slot_account")
            elif kinds[i] == "missing":
                statuses_out.append("missing_slot_amount")
            elif codes[i] == 0:
                statuses_out.append(
                    "niu_slot"
                    if kinds[i] == "zero"
                    else "contradictory_niu_slot_amount"
                )
            else:
                statuses_out.append(
                    "known_slot" if kinds[i] == "nonzero" else "ambiguous_slot_zero"
                )
        out["retirement_distribution_" + name + "_slot_status"] = pd.array(
            statuses_out, dtype="string"
        )
        slot_codes[name] = codes
        slot_amounts[name] = values
        slot_kinds[name] = kinds
        slot_status[name] = statuses_out
    applicable = {
        "age58_and_over": ("slot1", "slot2"),
        "under_age58": ("slot1_young", "slot2_young"),
    }
    totals = np.full(rows, np.nan, dtype=np.float64)
    total_status = np.full(rows, int(money.CodebookStatus.AMOUNT_NONZERO), dtype="u1")
    ira = np.full(rows, np.nan, dtype=np.float64)
    ira_slots, ambiguity, offroute = [], [], []
    for i in range(rows):
        names = applicable[route[i]]
        others = [n for n, *_ in slots if n not in names]
        offroute.append(any(slot_kinds[n][i] == "nonzero" for n in others))
        if all(slot_kinds[n][i] != "missing" for n in names):
            totals[i] = float(sum(slot_amounts[n][i] for n in names))
            if totals[i] == 0:
                total_status[i] = int(money.CodebookStatus.ZERO_NONE_OR_NIU)
        ambiguity.append(
            any(
                slot_codes[n][i] not in (None, 0) and slot_kinds[n][i] == "zero"
                for n in names
            )
        )
        # Both axes must be resolved: an unreadable account code could itself be
        # a regular IRA, and a declared account whose amount is a "none or niu"
        # zero does not observe a zero dollar distribution from that account.
        if all(slot_status[n][i] in ("niu_slot", "known_slot") for n in names):
            matched = [n for n in names if slot_codes[n][i] == REGULAR_IRA_CODE]
            ira[i] = float(sum(slot_amounts[n][i] for n in matched))
            ira_slots.append(len(matched))
        else:
            ira_slots.append(None)
    # Both printed recipiency literals are retained. The route selects which one
    # applies; the other stays inspectable, because an answered off-route
    # recipiency is a source contradiction rather than a missing answer.
    route_codes = {}
    for suffix, field in (("58", "DST_YN"), ("young", "DST_YN_YNG")):
        frame, codes, statuses = _codes_frame(
            "retirement_distribution_receipt_" + suffix,
            raw[field],
            receipt_codes(field),
            receipt_codes(field),
        )
        out = pd.concat([out, frame], axis=1)
        route_codes[field] = (codes, statuses)
    offroute_receipt = [
        (
            route_codes["DST_YN_YNG" if route[i] == "age58_and_over" else "DST_YN"][0][
                i
            ]
            not in (0, None)
        )
        for i in range(rows)
    ]
    out["retirement_distribution_offroute_receipt"] = pd.array(
        offroute_receipt, dtype="boolean"
    )
    receipt_tokens = [
        raw["DST_YN"][i] if route[i] == "age58_and_over" else raw["DST_YN_YNG"][i]
        for i in range(rows)
    ]
    frame, codes, statuses = _codes_frame(
        "retirement_distribution_receipt",
        receipt_tokens,
        receipt_codes("DST_YN"),
        receipt_codes("DST_YN"),
    )
    out = pd.concat([out, frame], axis=1)
    # The two printed DST universes name only the age-58 split. Unlike the four
    # age-universe families here (PEN_YN, ANN_YN, RNT_YN, OI_YN) they print no
    # 15+ floor, and unlike the farm family they are not gated on other
    # literals either, so whether a person under 15 is inside them is a source
    # question left unresolved.
    universes = [True if a >= 15 else None for a in ages]
    labels, canonical, _, _ = _classify(
        totals,
        np.where(np.isnan(totals), money.CodebookStatus.MISSING_NULL, total_status),
        universes,
        list(zip(codes, statuses, strict=True)),
        net_measure=False,
        zero_is_dollars=_zero_is_dollars("DST_VAL1"),
    )
    # An answered off-route recipiency or off-route dollars contradict the route
    # this row is on, and an applicable slot whose declared account carries a
    # "none or niu" zero leaves the composition unresolved. Neither may end in a
    # known amount, however the route's own receipt literal reads.
    for i in range(rows):
        if offroute[i] or offroute_receipt[i]:
            labels[i] = "contradictory_offroute_evidence"
            canonical[i] = np.nan
        elif ambiguity[i]:
            labels[i] = "unresolved_slot_composition"
            canonical[i] = np.nan
    out["retirement_distribution_source_total"] = totals
    out["retirement_distribution_reporting_status"] = pd.array(labels, dtype="string")
    out["retirement_distribution_known_amount"] = canonical
    out["retirement_distribution_regular_ira_amount"] = ira
    out["retirement_distribution_regular_ira_slots"] = pd.array(ira_slots, dtype="Int8")
    out["retirement_distribution_slot_zero_ambiguity"] = pd.array(
        ambiguity, dtype="boolean"
    )
    out["retirement_distribution_offroute_nonzero"] = pd.array(
        offroute, dtype="boolean"
    )
    out["retirement_distribution_source_reporting_universe"] = pd.array(
        universes, dtype="boolean"
    )
    out["retirement_distribution_taxable_amount_known"] = pd.array(
        [False] * rows, dtype="boolean"
    )
    out["retirement_distribution_allocation_origin"] = _allocation_origin(
        {
            f: raw["allocations"][f]
            for f in (
                "I_DSTSC",
                "I_DSTSCCOMP",
                "I_DSTVAL1COMP",
                "I_DSTVAL2COMP",
                "I_DSTYNCOMP",
            )
        },
        unflagged=True,
    )
    out["retirement_distribution_route_has_published_flag"] = pd.array(
        [r == "age58_and_over" for r in route], dtype="boolean"
    )
    return out


def _net_property(raw, ages):
    """Signed net property total; receipt scope is wider than the amount scope."""
    rows = len(ages)
    universe = _age_universe(ages)
    frame, codes, statuses = _codes_frame(
        "net_property_receipt",
        raw["RNT_YN"],
        receipt_codes("RNT_YN"),
        receipt_codes("RNT_YN"),
    )
    labels, canonical, kinds, sources = _classify(
        raw["amounts"]["RNT_VAL"],
        raw["statuses"]["RNT_VAL"],
        universe,
        list(zip(codes, statuses, strict=True)),
        net_measure=True,
        zero_is_dollars=_zero_is_dollars("RNT_VAL"),
    )
    out = frame
    out["net_property_source_total"] = sources
    out["net_property_amount_kind"] = pd.array(kinds, dtype="string")
    out["net_property_reporting_status"] = pd.array(labels, dtype="string")
    out["net_property_known_amount"] = canonical
    out["net_property_is_net_loss"] = pd.array(
        [bool(np.isfinite(v) and v < 0) for v in sources], dtype="boolean"
    )
    out["net_property_source_reporting_universe"] = pd.array(universe, dtype="boolean")
    out["net_property_amount_scope"] = pd.array(
        [NET_PROPERTY_AMOUNT_SCOPE] * rows, dtype="string"
    )
    out["net_property_receipt_scope"] = pd.array(
        [NET_PROPERTY_RECEIPT_SCOPE] * rows, dtype="string"
    )
    out["net_property_component_split_known"] = pd.array(
        [False] * rows, dtype="boolean"
    )
    out["net_property_allocation_origin"] = _allocation_origin(
        {f: raw["allocations"][f] for f in ("I_RNTVAL", "I_RNTYN")}, unflagged=False
    )
    return out


def _farm(raw, ages):
    """Farm universe comes from ERN_YN/FRMOTR evidence, never from age alone."""
    rows = len(ages)
    out = pd.DataFrame(index=range(rows))
    universe_codes = {}
    for field in ("ERN_YN", "FRMOTR"):
        frame, codes, _ = _codes_frame(
            "farm_" + field.lower(),
            raw[field],
            receipt_codes(field),
            receipt_codes(field),
        )
        out = pd.concat([out, frame], axis=1)
        universe_codes[field] = codes
    universe = []
    for i in range(rows):
        pair = (universe_codes["ERN_YN"][i], universe_codes["FRMOTR"][i])
        if 1 in pair:
            universe.append(True)
        elif all(c is not None for c in pair):
            universe.append(False)
        else:
            universe.append(None)
    frame, codes, statuses = _codes_frame(
        "farm_receipt",
        raw["FRSE_YN"],
        receipt_codes("FRSE_YN"),
        receipt_codes("FRSE_YN"),
    )
    out = pd.concat([out, frame], axis=1)
    labels, canonical, kinds, sources = _classify(
        raw["amounts"]["FRSE_VAL"],
        raw["statuses"]["FRSE_VAL"],
        universe,
        list(zip(codes, statuses, strict=True)),
        net_measure=True,
        zero_is_dollars=_zero_is_dollars("FRSE_VAL"),
    )
    out["farm_source_total"] = sources
    out["farm_amount_kind"] = pd.array(kinds, dtype="string")
    out["farm_reporting_status"] = pd.array(labels, dtype="string")
    out["farm_known_amount"] = canonical
    out["farm_is_net_loss"] = pd.array(
        [bool(np.isfinite(v) and v < 0) for v in sources], dtype="boolean"
    )
    out["farm_source_reporting_universe"] = pd.array(universe, dtype="boolean")
    out["farm_amount_scope"] = pd.array([FARM_AMOUNT_SCOPE] * rows, dtype="string")
    out["farm_is_nonfarm_self_employment"] = pd.array([False] * rows, dtype="boolean")
    out["farm_allocation_origin"] = _allocation_origin(
        {f: raw["allocations"][f] for f in ("I_ERNYN", "I_FRMYN")}, unflagged=True
    )
    out["farm_total_has_published_flag"] = pd.array([False] * rows, dtype="boolean")
    return out


def _other_income(raw, ages):
    """Reported category 20 is observed alimony; no residual rule creates one."""
    rows = len(ages)
    universe = _age_universe(ages)
    frame, codes, statuses = _codes_frame(
        "other_income_receipt",
        raw["OI_YN"],
        receipt_codes("OI_YN"),
        receipt_codes("OI_YN"),
    )
    category, category_codes, category_statuses = _codes_frame(
        "other_income_category",
        raw["OI_OFF"],
        OTHER_INCOME_CATEGORIES,
        OTHER_INCOME_CATEGORIES,
        width=OTHER_INCOME_CATEGORY_ENTRY.printed_length,
    )
    labels, canonical, kinds, sources = _classify(
        raw["amounts"]["OI_VAL"],
        raw["statuses"]["OI_VAL"],
        universe,
        list(zip(codes, statuses, strict=True)),
        net_measure=False,
        zero_is_dollars=_zero_is_dollars("OI_VAL"),
    )
    out = pd.concat([frame, category], axis=1)
    out["other_income_source_total"] = sources
    out["other_income_amount_kind"] = pd.array(kinds, dtype="string")
    out["other_income_reporting_status"] = pd.array(labels, dtype="string")
    out["other_income_known_amount"] = canonical
    out["other_income_source_reporting_universe"] = pd.array(universe, dtype="boolean")
    routing = []
    for i in range(rows):
        code, status = category_codes[i], category_statuses[i]
        # OI_OFF is printed for OI_YN = 1, and OI_YN for persons aged 15+, so a
        # row outside that universe carries no in-universe category either.
        if universe[i] is None:
            routing.append("unresolved_reporting_universe_routing")
        elif universe[i] is False:
            routing.append("outside_reporting_universe_routing")
        elif status == "missing":
            routing.append("missing_category_literal")
        elif code is None:
            routing.append("unrecognized_category_literal")
        elif codes[i] is None:
            # The category is printed but the receipt literal cannot be read, so
            # the pair is unresolved and the category is not a reported receipt.
            routing.append("unresolved_receipt_routing")
        elif codes[i] == 1 and code == 0:
            routing.append("receipt_without_category")
        elif codes[i] in (0, 2) and code != 0:
            routing.append("category_without_receipt")
        elif code == 0:
            routing.append("niu_category")
        else:
            routing.append("reported_category")
    out["other_income_routing_status"] = pd.array(routing, dtype="string")
    out["other_income_is_reported_alimony"] = pd.array(
        [
            bool(c == ALIMONY_CATEGORY_CODE and r == "reported_category")
            for c, r in zip(category_codes, routing, strict=True)
        ],
        dtype="boolean",
    )
    out["other_income_residual_rule_applied"] = pd.array(
        [False] * rows, dtype="boolean"
    )
    out["other_income_allocation_origin"] = _allocation_origin(
        {"I_OIVAL": raw["allocations"]["I_OIVAL"]}, unflagged=True
    )
    out["other_income_category_has_published_flag"] = pd.array(
        [False] * rows, dtype="boolean"
    )
    return out


def _read_capture(path, *, rows, columns=None, patterns=None):
    """Bounded literal reader; its path and row count establish no authority.

    Private source extensions may choose an explicit literal roster and amount
    grammars. The original routing qualifier retains its unchanged defaults.
    """
    columns = READ_COLUMNS if columns is None else tuple(columns)
    patterns = amount_patterns() if patterns is None else dict(patterns)
    require(
        len(set(columns)) == len(columns)
        and set(COORDINATE_COLUMNS) <= set(columns)
        and set(patterns) <= set(columns),
        "READ_COLUMN_CONTRACT",
    )
    reader = source_csv_builtin.capture_csv_reader(csv)
    require(reader is not None, "CSV_READER_CHANGED")
    records, keys, coordinates = [], set(), set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        stream = reader(handle, strict=True)
        header = next(stream, [])
        require(
            bool(header)
            and all(header)
            and len(header) == len(set(header))
            and set(columns) <= set(header),
            "HEADER",
        )
        positions = [header.index(c) for c in columns]
        for row in stream:
            require(len(row) == len(header) and len(records) < rows, "ROW_SHAPE")
            record = dict(zip(columns, (row[i] for i in positions), strict=True))
            key = record["PERIDNUM"]
            require(re.fullmatch(r"[0-9]{22}", key, re.ASCII) is not None, "PERSON_KEY")
            for name, width in COORDINATE_WIDTHS.items():
                require(
                    re.fullmatch(
                        r"[0-9]{1," + str(width) + r"}", record[name], re.ASCII
                    )
                    is not None,
                    "COORDINATE:" + name,
                )
            pair = (int(record["PH_SEQ"]), int(record["A_LINENO"]))
            require(
                min(pair) > 0 and pair not in coordinates and key not in keys,
                "DUPLICATE_OR_INVALID_COORDINATE",
            )
            for name, pattern in patterns.items():
                require(
                    record[name] == "" or pattern.fullmatch(record[name]) is not None,
                    "AMOUNT_TOKEN:" + name,
                )
            for name in columns:
                if name in COORDINATE_COLUMNS or name in patterns:
                    continue
                require(len(record[name]) <= TOKEN_MAX_CHARS, "TOKEN_BOUND:" + name)
            records.append(record)
            keys.add(key)
            coordinates.add(pair)
    require(len(records) == rows, "ROW_COUNT")
    return pd.DataFrame(records, columns=columns).set_index("PERIDNUM", drop=False)


def amount_observations(field, positions, literals):
    """Join literal validity; missing backing storage never becomes a zero.

    The retained parent requires complete ready money before issuing a
    preparation. This keeps that prerequisite separate from the literal join so
    a missing field would still be represented correctly if that scope grows.
    """
    require(type(field) is money.MoneyField, "MONEY_FIELD_TYPE")
    require(field.name in AMOUNT_FIELDS, "MONEY_FIELD_NAME")
    positions = np.asarray(positions)
    require(
        positions.dtype == np.dtype("int64") and positions.ndim == 1, "MONEY_POSITIONS"
    )
    tokens = tuple(literals)
    entry = printed_amount_entries()[field.name]
    pattern = amount_patterns()[field.name]
    require(
        len(tokens) == len(positions)
        and all(type(t) is str and (t == "" or pattern.fullmatch(t)) for t in tokens),
        "AMOUNT_TOKEN",
    )
    valid = field.validity[positions] == 1
    require(
        np.array_equal(valid, np.array([t != "" for t in tokens])),
        "CURRENT_AMOUNT_SOURCE_VALIDITY",
    )
    amounts = field.amounts[positions].copy()
    statuses = field.statuses[positions].copy()
    literal_values = np.asarray([float(t) if t else np.nan for t in tokens])
    minimum, maximum = entry.encoded_minimum, entry.encoded_maximum
    niu_codes = entry.nonmoney_codes
    niu = statuses == money.CodebookStatus.DECLARED_NIU
    require(not (niu & ~valid).any(), "NIU_WITHOUT_VALIDITY")
    # A printed NIU code is stored as a normalized zero under DECLARED_NIU. Its
    # literal must still be one of that entry's printed non-dollar codes.
    require(
        np.array_equal(amounts[valid & niu], np.zeros(int((valid & niu).sum())))
        and all(v in niu_codes for v in literal_values[valid & niu]),
        "CURRENT_AMOUNT_NIU_IDENTITY",
    )
    direct = valid & ~niu
    require(
        np.array_equal(amounts[direct], literal_values[direct]),
        "CURRENT_AMOUNT_SOURCE_IDENTITY",
    )
    amounts[~valid] = np.nan
    inside = amounts[direct]
    require(
        ((inside >= minimum) & (inside <= maximum)).all()
        and (inside == np.floor(inside)).all(),
        "AMOUNT_PRINTED_DOMAIN:" + field.name,
    )
    return amounts, statuses


def _domain_agreement(ready):
    """Bind printed entries to the live current-money domain contract."""
    _require_live_amount_domains(
        ready,
        {
            name: (entry.encoded_minimum, entry.encoded_maximum, entry.zero_semantics)
            for name, entry in printed_amount_entries().items()
        },
    )


@dataclass(frozen=True)
class CurrentAsecIncomeRoutingValues:
    """Descriptive transport. Constructing this grants no source authority."""

    person: pd.DataFrame
    asec_literals: pd.DataFrame
    evidence: dict


def project_income_routing(raw, ages):
    """Pure per-family projection over already-validated literal arrays."""
    ages = np.asarray(ages, dtype=np.float64)
    require(
        np.isfinite(ages).all()
        and ((ages >= 0) & (ages <= 99) & (ages == np.floor(ages))).all(),
        "AGE_DOMAIN",
    )
    require(
        set(raw["amounts"]) == set(AMOUNT_FIELDS)
        and set(raw["statuses"]) == set(AMOUNT_FIELDS),
        "AMOUNT_ARRAYS",
    )
    require(
        set(raw["allocations"]) == set(ALLOCATION_ENTRIES)
        and all(
            len(v) == 2 and len(v[0]) == len(v[1]) == len(ages)
            for v in raw["allocations"].values()
        ),
        "ALLOCATION_ARRAYS",
    )
    for name, values in raw["amounts"].items():
        values = np.asarray(values)
        statuses = np.asarray(raw["statuses"][name])
        require(
            values.dtype == np.dtype("float64")
            and values.ndim == 1
            and len(values) == len(ages)
            and not np.isinf(values).any()
            and statuses.dtype == np.dtype("u1")
            and len(statuses) == len(ages)
            and np.isin(statuses, [int(c) for c in money.CodebookStatus]).all(),
            "AMOUNT_ARRAY_CONTRACT:" + name,
        )
    for name in (*RECEIPT_ENTRIES, *ACCOUNT_ENTRIES, "OI_OFF"):
        tokens = raw.get(name)
        require(
            type(tokens) is list
            and len(tokens) == len(ages)
            and all(type(t) is str and len(t) <= TOKEN_MAX_CHARS for t in tokens),
            "ROUTING_TOKEN_CONTRACT:" + name,
        )
    parts = [
        _pension_annuity(raw, ages),
        _retirement_distribution(raw, ages),
        _net_property(raw, ages),
        _farm(raw, ages),
        _other_income(raw, ages),
    ]
    out = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    out.insert(0, "source_age", ages)
    require(len(set(out.columns)) == len(out.columns), "PROJECTION_COLUMN_COLLISION")
    return out


def _raw_arrays(ordered, ready, positions):
    allocations = {}
    for name, entry in ALLOCATION_ENTRIES.items():
        pairs = [
            literal_code(t, entry.codes, width=entry.printed_length)
            for t in ordered[name].tolist()
        ]
        allocations[name] = (
            [c for c, _ in pairs],
            [s for _, s in pairs],
        )
    observed = {
        name: amount_observations(ready.field(name), positions, ordered[name])
        for name in AMOUNT_FIELDS
    }
    raw = {
        "amounts": {k: v[0] for k, v in observed.items()},
        "statuses": {k: v[1] for k, v in observed.items()},
        "allocations": allocations,
    }
    for name in (*RECEIPT_ENTRIES, *ACCOUNT_ENTRIES, "OI_OFF"):
        raw[name] = ordered[name].tolist()
    return raw


def qualify_current_asec_income_routing(preparation):
    """Capture the exact current source member retained by the original owner."""
    require(
        type(preparation) is source.AuthenticatedSurveyPopulationPreparation,
        "PREPARATION_TYPE",
    )
    entry = preparation._checked()
    state = entry[2]
    native = state.native[1]
    issued = source.asec_native._ISSUED.get(id(native))
    require(
        issued is not None and issued[0]() is native and issued[1] == native.payload,
        "NATIVE_ISSUANCE",
    )
    parent = issued[2].parent
    ready = parent.ready()
    header = json.loads(ready.header)
    require(
        header["target_year"] == 2024 and header["semantic"] == "annual_current_money",
        "MONEY_PERIOD",
    )
    native_document = json.loads(issued[1])
    require(
        native_document["source_year"] == native_document["income_year"] == 2024
        and native_document["survey_year"] == 2025,
        "NATIVE_PERIOD",
    )
    _domain_agreement(ready)
    pins = [p for p in coverage._MEMBER_PINS if p[0] == 2024]
    require(len(pins) == 1, "SOURCE_REGISTRY")
    year, member, archive, digest, rows, size = pins[0]
    retained = [
        s for s in issued[2].coverage.receipt["sources"] if s["source_year"] == year
    ]
    require(
        len(retained) == 1
        and retained[0]["member"] == member
        and retained[0]["archive_sha256"] == archive
        and retained[0]["member_sha256"] == digest
        and retained[0]["rows"] == rows
        and retained[0]["member_bytes"] == size,
        "NATIVE_MEMBER_BINDING",
    )
    with tempfile.TemporaryDirectory(prefix="microcosm-current-income-routing-") as tmp:
        captured = Path(tmp) / member
        identity = coverage._capture(
            state.root / "asec" / member,
            captured,
            size=size,
            digest=digest,
            budget=[coverage._BODY_MAX],
        )
        raw_member = _read_capture(captured, rows=rows)
        require(
            coverage._identity(captured.stat(follow_symlinks=False)) == identity
            and _file_sha(captured) == digest,
            "CAPTURE_CHANGED",
        )
    positions = np.flatnonzero(np.asarray(parent.scope.person_years) == 2024)
    keys = np.asarray(parent.scope.person_native_keys)[positions]
    require(
        len(keys) == rows
        and len(set(keys)) == rows
        and set(keys) == set(raw_member.index),
        "COMPLETE_CURRENT_SOURCE_JOIN",
    )
    ordered = raw_member.loc[keys]
    parent_people = parent.frame.person.iloc[positions]
    for raw_name, parent_name in (
        ("PH_SEQ", "source_household_id"),
        ("A_LINENO", "A_LINENO"),
        ("A_AGE", "A_AGE"),
    ):
        require(
            np.array_equal(
                ordered[raw_name].astype("int64").to_numpy(),
                parent_people[parent_name].to_numpy(),
            ),
            "PARENT_COORDINATE_IDENTITY",
        )
    ages = ordered.A_AGE.to_numpy(dtype="float64")
    basis = project_income_routing(_raw_arrays(ordered, ready, positions), ages)
    basis.index = pd.Index(
        np.asarray(parent.scope.person_ids)[positions], name="native_person_id"
    )
    for name in AMOUNT_FIELDS:
        field = ready.field(name)
        basis["amount_status_" + name] = field.statuses[positions]
        basis["amount_validity_" + name] = field.validity[positions]
        basis["zero_origin_" + name] = field.zero_origin[positions]
    people = state.frame.person
    channel = people[support_channel_column("person")]
    selected = people.loc[channel.eq("asec")]
    native_ids = selected[spine_source_id_column("person")].to_numpy()
    require(
        len(set(native_ids)) == len(native_ids) and set(native_ids) <= set(basis.index),
        "SELECTED_NATIVE_JOIN",
    )
    out = basis.loc[native_ids].copy()
    out["native_person_id"] = native_ids
    out.index = pd.Index(selected.person_id.to_numpy(), name="person_id")
    literals = ordered.copy()
    literals.index = pd.Index(
        np.asarray(parent.scope.person_ids)[positions], name="native_person_id"
    )
    evidence = {
        "protocol": PROTOCOL,
        "dictionary": {
            "url": DICTIONARY_URL,
            "sha256": DICTIONARY_SHA256,
            "amount_entries_source": {
                "resource": DOMAINS_RESOURCE,
                "sha256": DOMAINS_SHA256,
                "income_year": CURRENT_INCOME_YEAR,
            },
            "amount_entries": {
                name: {
                    "printed_length": e.printed_length,
                    "printed_position": e.printed_position,
                    "printed_page": e.printed_page,
                    "pdf_page_1based": e.pdf_page_1based,
                    "encoded_range_inclusive": [e.encoded_minimum, e.encoded_maximum],
                    "universe_as_printed": e.universe_as_printed,
                    "values_as_printed": e.values_as_printed,
                    "negative_dollars_permitted": e.negative_dollars_permitted,
                    "declared_negative_nonmoney_codes": list(e.nonmoney_codes),
                    "zero_semantics": e.zero_semantics,
                }
                for name, e in printed_amount_entries().items()
            },
            "receipt_entries": {k: v._asdict() for k, v in RECEIPT_ENTRIES.items()},
            "account_entries": {k: v._asdict() for k, v in ACCOUNT_ENTRIES.items()},
            "other_income_category_entry": OTHER_INCOME_CATEGORY_ENTRY._asdict(),
            "allocation_entries": {
                k: {**v._asdict(), "codes": list(v.codes)}
                for k, v in ALLOCATION_ENTRIES.items()
            },
            "fields_without_published_allocation_flag": list(UNFLAGGED_FIELDS),
            "published_allocation_flag_by_field": dict(
                PUBLISHED_ALLOCATION_FLAG_BY_FIELD
            ),
            "ambiguous_allocation_flag_coverage": dict(AMBIGUOUS_FLAG_COVERAGE),
            "printed_universe_questions": {
                "DST_VAL1": "printed universe is DST_SC1 = 1 although the label names "
                "the source-1 distribution amount; retained verbatim",
                "DST_SC2_YNG": "printed universe names DST_VAL_YNG, which has no "
                "dictionary entry; retained verbatim",
                "I_DSTVAL1COMP": "printed universe line is empty",
                "DST_YN": "printed universes name only the a_age 58 split; they "
                "print no 15+ floor as the four age-universe families here do, "
                "and they are not gated on other literals as the farm family is, "
                "so coverage below age 15 is unresolved",
            },
            "printed_scope_and_code_questions": {
                "RNT_YN": "printed question scope: " + NET_PROPERTY_RECEIPT_SCOPE,
                "RNT_VAL": "printed question scope: " + NET_PROPERTY_AMOUNT_SCOPE,
                "FRSE_VAL": "printed label: " + FARM_AMOUNT_SCOPE,
                "PNSN_VAL": "printed label: " + PENSION_TOTAL_SCOPE,
                "OI_YN": "printed zero label is 'none or niu' where PEN_YN, "
                "ANN_YN, DST_YN and RNT_YN print 'niu'",
                "DST_SC1": "gated on DST_VAL1 > 0 and a_age \u2265 58 while "
                "DST_SC1_YNG is gated on DST_YN_YNG = 1 and a_age < 58; the two "
                "routes are not symmetric",
                "FRSE_YN": "printed universe is ERN_YN=1 or FRMOTR=1, so the farm "
                "family prints no age floor at all",
            },
        },
        "preparation_sha256": _sha(entry[1]),
        "asec_native_sha256": _sha(issued[1]),
        "money_header_sha256": _sha(ready.header),
        "source_member_sha256": digest,
        "read_columns": list(READ_COLUMNS),
        "complete_current_source_rows": rows,
        "declared_niu_normalized_to_zero": [
            name
            for name, entry in printed_amount_entries().items()
            if entry.nonmoney_codes
        ],
        "joined_person_years": [CURRENT_INCOME_YEAR],
        "restatement_note": (
            "the money owner restates non-2024 cohorts to the pinned price basis, "
            "so the literal identity join is only taken on the 2024 rows"
        ),
        "selected_rows": len(out),
        "acs_channel_rows": int(channel.eq("acs").sum()),
        "families": list(FAMILIES),
        "asec_income_year": 2024,
        "asec_interview_year": 2025,
        "policy": (
            "separate_source_totals_retained; ambiguous_NIU_missing_contradictions_"
            "retained_unknown; no_default_zero_completion"
        ),
        "pension_private_share_applied": False,
        "pension_taxable_amount_known": False,
        "retirement_distribution_taxable_amount_known": False,
        "net_property_component_split_known": False,
        "farm_mapped_to_nonfarm_self_employment": False,
        "other_income_residual_rule_applied": False,
        "acs_components_modeled": False,
        "observed_taxable_amount_claim": False,
        "unallocated_observation_claim": False,
        "source_admission_issued": False,
        "release_eligible": False,
        "projection_sha256": _sha(out.to_json(orient="table").encode()),
        "literals_sha256": _sha(literals.to_json(orient="table").encode()),
    }
    require(
        preparation._checked()[1] == entry[1] and parent.ready().header == ready.header,
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
    require(
        _sha(out.to_json(orient="table").encode()) == evidence["projection_sha256"]
        and _sha(literals.to_json(orient="table").encode())
        == evidence["literals_sha256"],
        "FINAL_PROJECTION_CHANGED",
    )
    return CurrentAsecIncomeRoutingValues(out, literals, evidence)
