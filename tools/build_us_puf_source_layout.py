#!/usr/bin/env python3
"""Generate the packaged PUF 2015 source layout and monetary projection document.

This reads the three retained, content-pinned IRS booklet text extractions and
the already-reviewed packaged raw source definition. It reads **no PUF
microdata**, opens no processed artifact, imports no country model and
computes no policy.

What it emits is one closed document with three parts.

*The full delivered layout.* Every one of the 222 delivered main columns and
the 7 delivered demographic columns is located in the September 2022 booklet
by its own publisher position number and name, its verbatim record-layout row
is retained, and its exact character span and span digest in the pinned text
are recorded. The February 2023 edition is matched again as corroboration; on
the rows that carry a left-margin section label its text extraction splits the
field code with one interior space, and the entries that needed the tolerant
pattern are listed rather than smoothed over.

*The enumerated monetary source projection.* A small, explicitly listed set of
income concepts the retained booklets actually support, each with its exact
units, sign evidence, missingness policy, return grain, source-period status
and source pin identity. Nothing is inferred from an ``E``/``P`` prefix: the
publisher's own "Misc Codes" block puts ``RECID``, ``S006``, ``S008``,
``S009``, ``WSAMP`` and ``TXRT`` inside the amount-field list, and ``TXRT``
carries an implied decimal point, so the prefix and the list membership both
prove nothing on their own.

*The refusals.* ``E22250`` is absent from the delivered layout and from both
booklets and is never aliased onto ``P22250``. An uncapped self-employment
earnings derivation from ``E30400``/``E30500`` is refused with the publisher
text that refuses it. Tax outcomes and every unselected amount field stay
source-only, and no column is admitted for target-year use.

Usage::

    python tools/build_us_puf_source_layout.py --write
    python tools/build_us_puf_source_layout.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "packages/microcosm-build/src/microcosm/build"
RAW_DEFINITION = PACKAGE / "us_runtime/puf_2015_raw_source_definition.json"
# The projection is a runtime source-adapter resource, not a country spec: it
# lives beside ``acs_2024_1yr_sources.json`` in ``us_runtime`` rather than in
# the ``us`` country package, whose manifest declares every file it ships.
OUTPUT = PACKAGE / "us_runtime/puf_2015_monetary_source_projection.json"

SCHEMA = "microcosm.us.puf_2015_monetary_source_projection"
SCHEMA_VERSION = 1

#: The retained booklets, by the identity the source-document receipt records.
BOOKLETS = {
    "main_2022": {
        "text": "main-2022.txt",
        "text_sha256": (
            "c2446b589f60592f337ed45f0d82406593fddafd16ec45d06440c07c35418765"
        ),
        "pdf_sha256": (
            "dfb2a6c85f98cba95624ece7c2bd239a0fd09457ca2498777c8242208ff7295b"
        ),
        "pdf_bytes": 23417059,
        "pdf_pages": 88,
        "edition": "September 2022",
        "title": "2015 Public Use Tax File General Description Booklet",
        "role": "primary_layout_authority",
    },
    "main_2023": {
        "text": "main-2023.txt",
        "text_sha256": (
            "c7f84a1ec266d341b713ed1251c7056b6636e6a8ac23ef1124f21fe367f34353"
        ),
        "pdf_sha256": (
            "93a1d863b5a4af21b624714fb2bb3f9d30509defdd8f2c052c2c5ca4d9cce35c"
        ),
        "pdf_bytes": 23413549,
        "pdf_pages": 88,
        "edition": "Updated February 2023",
        "title": "2015 Public Use Tax File General Description Booklet",
        "role": "corroborating_edition",
    },
    "demographic_2022": {
        "text": "demographic.txt",
        "text_sha256": (
            "d2074c3a2e2dabb6d25da4a9a62b8d9302e001cc5d86939edc0bdd09719d8c9a"
        ),
        "pdf_sha256": (
            "7bb623c7c7631e30eb620f0a522d74bff59487c1778d0b2023b2536cead5f38b"
        ),
        "pdf_bytes": 345247,
        "pdf_pages": 13,
        "edition": "September 2022",
        "title": "2015 Public Use Tax Demographic File",
        "role": "primary_layout_authority",
    },
}

#: Prose the projection cites, each located by a page-unique verbatim needle.
#: ``(booklet, pdf_page, needle, characters_of_context)``.
ANCHORS = {
    "amount_field_width": ("main_2022", 7, "12 digit amount fields", 0),
    "code_fields_heading": ("main_2022", 6, "CORE RECORD LAYOUT", 0),
    "amount_fields_heading": ("main_2022", 7, "Amount Fields and Large Codes", 0),
    "csv_order": (
        "main_2022",
        27,
        "Field names have been ordered first",
        120,
    ),
    "record_count_intro": (
        "main_2022",
        3,
        "The Internal Revenue Service 2015 Public Use Tax File, which contains "
        "207,696 records",
        0,
    ),
    "record_count_technical": (
        "main_2022",
        27,
        "This comma separated file contains 207,696",
        90,
    ),
    "national_estimates_only": (
        "main_2022",
        3,
        "2015 Public Use Tax File is only appropriate for national level estima tes.",
        0,
    ),
    "fiscal_year_conversion": (
        "main_2022",
        4,
        "First, fiscal year returns have been converted to reflect the most "
        "recent year-end Tax Year",
        140,
    ),
    "aggregate_records": (
        "main_2022",
        4,
        "Second, returns that contain one or more amount fields with values "
        "deeme d extremely",
        700,
    ),
    "aggregate_recid_footnote": (
        "main_2022",
        11,
        "[1] Records containing aggregated amounts are identified by "
        "RECID=999996, 999997, 999998, and 999999",
        0,
    ),
    "multivariate_blurring": (
        "main_2022",
        4,
        "Multivariate blurring has been applied to these returns with nonzero "
        "values in at least",
        260,
    ),
    "itemized_blurring_007": (
        "main_2022",
        5,
        "Multivariate blurring has been applied to itemized tax returns with "
        "nonzero values",
        240,
    ),
    "nonitemized_wage_noise": (
        "main_2022",
        5,
        "For nonitemized tax returns, up to 10 percent of random noise, drawn "
        "from a uniform",
        220,
    ),
    "rounding_grid": (
        "main_2022",
        5,
        "Ninth, for all records on the file, all amount fields have been rounded.",
        420,
    ),
    "rebalancing": (
        "main_2022",
        5,
        "Finally, all records in the file are rebalanced to ensure accounting accuracy",
        90,
    ),
    "not_one_taxpayer": (
        "main_2022",
        5,
        "Since individual records in this file may or may not contain data "
        "from just one tax return",
        330,
    ),
    "p22250_definition": (
        "main_2022",
        24,
        "P22250           SHORT-TERM GAINS LESS LOSSES NET OF CARRYOVER",
        190,
    ),
    "p23250_definition": (
        "main_2022",
        24,
        "P23250            LONG-TERM GAINS LESS LOSSES NET OF CARRYOVER",
        190,
    ),
    "e30400_definition": (
        "main_2022",
        24,
        "E30400 NET EARNINGS FROM SELF-EMPLOYMENT INCOME SUBJECT TO",
        330,
    ),
    "e30500_definition": (
        "main_2022",
        24,
        "E30500 NET EARNINGS FROM SELF-EMPLOYMENT INCOME SUBJECT TO",
        320,
    ),
    "s006_definition": ("main_2022", 25, "S006 DECIMAL WEIGHT", 260),
    "txrt_implied_decimal": (
        "main_2022",
        26,
        "that this field uses an implied decimal point (28.0 percent is "
        "represented by 280).",
        0,
    ),
    "flpdyr_printed_range": (
        "main_2022",
        14,
        "8. FLPDYR Filing (Accounting) Period, Year:",
        140,
    ),
    "demographic_join_section": (
        "demographic_2022",
        10,
        "This section describes the domain of the supplemental file and "
        "method for joining to the 2015",
        0,
    ),
    "demographic_strata_restriction": (
        "demographic_2022",
        10,
        "found on the 2015 PUF for records in strata 7 through 13, or those returns",
        0,
    ),
}


@dataclass(frozen=True)
class Projected:
    """One enumerated monetary column of the source projection."""

    field: str
    concept: str
    concept_group: str
    #: Extra publisher prose this column's semantics rest on, by anchor key.
    definition_anchors: tuple[str, ...] = ()
    #: Field-specific disclosure edits, by anchor key. The universal edits
    #: (rounding, rebalancing, aggregation, "not one taxpayer") are declared
    #: once at document level and apply to every column.
    edit_anchors: tuple[str, ...] = ()
    #: Named questions the retained booklets do not settle. A column with a
    #: non-empty list is still decoded and retained; it is refused as an input
    #: to any derived combination.
    unresolved: tuple[tuple[str, str], ...] = ()
    #: ``not_admitted_pending_basis_review`` or ``unresolved_retained``. Never a target year.
    fit_admission: str = "not_admitted_pending_basis_review"
    #: Set when the layout row prints no ``(+/-)`` marker but the publisher's
    #: own definition text documents a netted quantity that can be negative.
    negative_documented_by_definition: bool = False
    notes: tuple[str, ...] = ()


PROJECTION: tuple[Projected, ...] = (
    Projected(
        "E00200",
        "salaries_and_wages",
        "wages",
        edit_anchors=(
            "multivariate_blurring",
            "itemized_blurring_007",
            "nonitemized_wage_noise",
        ),
        notes=(
            "Wages carry two documented edits beyond the universal ones: "
            "multivariate blurring wherever at least two of wages, state and "
            "local income taxes and real estate taxes are nonzero, and up to "
            "10 percent uniform random noise on nonitemized returns sampled "
            "at 0.07 percent. A wage value is therefore an edited amount, "
            "not one taxpayer's reported wage.",
        ),
    ),
    Projected(
        "E00300",
        "taxable_interest_received",
        "interest",
        unresolved=(
            (
                "relationship_to_tax_exempt_interest",
                "Neither booklet states whether E00300 and E00400 partition "
                "interest or overlap. They are retained as two independent "
                "columns and never summed here.",
            ),
        ),
    ),
    Projected(
        "E00400",
        "tax_exempt_interest_income",
        "interest",
        unresolved=(
            (
                "relationship_to_taxable_interest",
                "See E00300. No publisher statement partitions the two.",
            ),
        ),
    ),
    Projected(
        "E00600",
        "dividends_included_in_agi",
        "dividends",
        unresolved=(
            (
                "qualified_dividend_containment",
                "Neither booklet states that E00650 is a part of E00600. The "
                "two are retained independently; no difference is computed.",
            ),
        ),
    ),
    Projected(
        "E00650",
        "qualified_dividends",
        "dividends",
        unresolved=(
            (
                "qualified_dividend_containment",
                "See E00600. Containment is not documented, so no ordinary "
                "dividend residual is derived.",
            ),
        ),
    ),
    Projected(
        "E00900",
        "schedule_c_net_profit_or_loss",
        "business_income",
        notes=(
            "The record layout prints the (+/-) marker, so negative values "
            "are documented. This is a net Schedule C result, not gross "
            "receipts, and it is not self-employment earnings subject to "
            "Social Security tax; see the refused E30400/E30500 derivation.",
        ),
    ),
    Projected(
        "E02100",
        "schedule_f_net_profit_or_loss",
        "farm_income",
        notes=(
            "Schedule F net profit or loss, marker-signed. Farm rent "
            "(E27200) is a separate publisher row and is not selected here.",
        ),
    ),
    Projected(
        "E01500",
        "total_pensions_and_annuities_received",
        "pensions",
        unresolved=(
            (
                "taxable_share_relationship",
                "The labels say 'received' and 'included in AGI' but neither "
                "booklet defines E01700 as a part of E01500. No taxable "
                "share is derived.",
            ),
        ),
    ),
    Projected(
        "E01700",
        "pensions_and_annuities_included_in_agi",
        "pensions",
        unresolved=(
            (
                "taxable_share_relationship",
                "See E01500.",
            ),
        ),
    ),
    Projected(
        "E01000",
        "net_capital_gain_or_loss",
        "capital_gains",
        unresolved=(
            (
                "composition_from_schedule_d_rows",
                "No publisher statement composes E01000 out of P22250 and "
                "P23250, and the statutory capital loss limitation is not "
                "described in either booklet. The three columns are retained "
                "independently and never reconciled here.",
            ),
        ),
        notes=("Marker-signed on the record layout row.",),
    ),
    Projected(
        "P22250",
        "short_term_gains_less_losses_net_of_carryover",
        "capital_gains",
        definition_anchors=("p22250_definition",),
        negative_documented_by_definition=True,
        unresolved=(
            (
                "composition_from_schedule_d_rows",
                "See E01000.",
            ),
        ),
        notes=(
            "The record-layout row prints no (+/-) marker, but the "
            "publisher's own definition is 'all short-term capital gains "
            "less losses ... except for the short-term capital loss "
            "carryover', which is a netted quantity. The marker's absence is "
            "recorded verbatim rather than treated as a nonnegative domain.",
            "E22250 does not exist in the delivered layout or in either "
            "booklet. It is never aliased onto this column.",
        ),
    ),
    Projected(
        "P23250",
        "long_term_gains_less_losses_net_of_carryover",
        "capital_gains",
        definition_anchors=("p23250_definition",),
        negative_documented_by_definition=True,
        unresolved=(
            (
                "composition_from_schedule_d_rows",
                "See E01000.",
            ),
        ),
        notes=("As P22250: netted by definition, unmarked on the layout row.",),
    ),
)

#: Amount-list members the publisher's own "Misc Codes" block shows are not
#: money. Their presence in the amount-field list is exactly why no E/P or
#: list-membership rule may be used to decide what an amount is.
MISC_CODES = ("RECID", "S006", "S008", "S009", "WSAMP", "TXRT")

#: Unselected columns worth naming explicitly, so their source-only status is
#: a recorded decision rather than an omission. Each is keyed by field name and
#: carries the reason it stays out of the projection.
NOTABLE_SOURCE_ONLY = {
    "E06500": (
        "Total income tax. A tax outcome of 2015 law applied to 2015 amounts. "
        "Aging it would not compute any later year's policy, and the archived "
        "SOI measure table binds this same code to both 'total income tax' and "
        "'income tax before credits', a conflict a successor must settle. It "
        "stays source-year evidence."
    ),
    "E05800": (
        "Income tax before credits: the publisher's own before-credits "
        "quantity, and the reason E06500 must not be read as one."
    ),
    "E08800": "Income tax after credits (SOI). A tax outcome; source-only.",
    "E10300": "Total tax liability (SOI). A tax outcome; source-only.",
    "E30400": (
        "The record-layout row says 'primary taxpayer', but the publisher's "
        "own definition on the same booklet's detailed pages says the combined "
        "primary and secondary net earnings from self-employment *subject to "
        "Social Security tax*. The label and the definition disagree, and the "
        "definition governs. See the refused uncapped-SE derivation."
    ),
    "E30500": (
        "The secondary taxpayer's capped amount. See E30400 and the refused "
        "uncapped-SE derivation."
    ),
    "E25920": (
        "Partnership total passive loss specifically, not a generic gross "
        "loss. The archived code path consumed this code without ever "
        "describing it; the label alone establishes no blank or negative "
        "decoder, so it stays source-only."
    ),
    "E01100": (
        "Capital gain distributions reported on Form 1040. A capital-gains "
        "component this increment does not project, because no publisher "
        "statement composes it with E01000, P22250 or P23250."
    ),
    "E22320": "Long term gain/loss from other forms on Schedule D. See E01100.",
    "E22370": "Schedule D capital gain distributions. See E01100.",
    "E02000": (
        "Schedule E net income or loss. Business-adjacent but a different "
        "publisher concept from Schedule C; not projected here."
    ),
    "E26270": (
        "Combined partnership and S corporation net income/loss. Not "
        "projected; the archived producer both sign-branched and AGI-fallback "
        "aged this code, which double-counted it."
    ),
    "E27200": (
        "Farm rent net income or loss. A separate publisher row from the "
        "Schedule F farm result; not projected here."
    ),
    "E01400": (
        "Taxable IRA distribution. Retirement-adjacent but a distinct "
        "publisher concept from pensions and annuities; not projected here."
    ),
    "E02400": ("Gross Social Security benefits. Not projected in this increment."),
    "E00100": (
        "Adjusted gross income (deficit). The reviewed growth slice already "
        "retains source-year AGI as provenance; projecting it here would "
        "create a second, competing source of the same quantity."
    ),
}

#: Field names this document refuses outright, with the reason.
REFUSED_FIELDS = {
    "E22250": {
        "state": "absent_from_delivered_layout_and_from_both_retained_booklets",
        "policy": "never_aliased_to_P22250",
        "reason": (
            "The archived producer's screened-field handling and several "
            "downstream discussions treat E22250 and P22250 as one column. "
            "The delivered header carries P22250 only, and neither booklet "
            "prints E22250 anywhere. Normalising one onto the other would be "
            "a guess, so a declaration or a delivered header naming E22250 "
            "is a refusal instead."
        ),
    },
}

#: Derivations this increment refuses to perform, with the publisher text.
REFUSED_DERIVATIONS = {
    "uncapped_self_employment_earnings": {
        "refused_inputs": ["E30400", "E30500"],
        "anchors": ["e30400_definition", "e30500_definition"],
        "reason": (
            "E30400 is the combined primary and secondary net earnings from "
            "self-employment *subject to Social Security tax* — the lesser "
            "of net Schedule SE earnings and the maximum amount subject to "
            "the tax — and E30500 is the secondary taxpayer's own capped "
            "amount. Subtracting one from the other yields a capped primary "
            "residual, not uncapped self-employment earnings, and grossing "
            "the difference up cannot recover what the cap removed."
        ),
    },
}


# --------------------------------------------------------------------------
# Booklet reading
# --------------------------------------------------------------------------


class GenerationError(RuntimeError):
    """The generator refuses rather than emitting an unauthenticated document."""


@dataclass(frozen=True)
class Booklet:
    """One content-pinned booklet text, split into its extracted pages."""

    key: str
    text: str
    pages: tuple[str, ...]
    #: Character offset of each page's first character in ``text``.
    offsets: tuple[int, ...]

    def page(self, pdf_page: int) -> tuple[str, int]:
        if not 1 <= pdf_page <= len(self.pages):
            raise GenerationError(f"{self.key}: no pdf page {pdf_page}")
        return self.pages[pdf_page - 1], self.offsets[pdf_page - 1]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_booklets(documents_dir: Path) -> dict[str, Booklet]:
    booklets: dict[str, Booklet] = {}
    for key, spec in BOOKLETS.items():
        path = documents_dir / spec["text"]
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != spec["text_sha256"]:
            raise GenerationError(f"{key}: text digest {digest} is not the pin")
        text = raw.decode("utf-8")
        pages = tuple(text.split("\f"))
        if len(pages) != spec["pdf_pages"]:
            raise GenerationError(
                f"{key}: {len(pages)} pages, pinned {spec['pdf_pages']}"
            )
        offsets: list[int] = []
        cursor = 0
        for page in pages:
            offsets.append(cursor)
            cursor += len(page) + 1
        booklets[key] = Booklet(key, text, pages, tuple(offsets))
    return booklets


def span(
    booklet: Booklet, pdf_page: int, start: int, end: int, *, label: str
) -> dict[str, object]:
    """One exact, self-verifying character span of a pinned booklet text."""

    page, base = booklet.page(pdf_page)
    if not 0 <= start < end <= len(page):
        raise GenerationError(f"{label}: span {start}:{end} outside page {pdf_page}")
    excerpt = page[start:end]
    return {
        "booklet": booklet.key,
        "pdf_page": pdf_page,
        "printed_page": pdf_page - 1,
        "text_start": base + start,
        "text_end": base + end,
        "page_start": start,
        "page_end": end,
        "text": excerpt,
        "text_sha256": sha256_text(excerpt),
    }


def anchor(booklets: dict[str, Booklet], key: str) -> dict[str, object]:
    """Locate one prose anchor, requiring exactly one hit on its own page."""

    booklet_key, pdf_page, needle, context = ANCHORS[key]
    booklet = booklets[booklet_key]
    page, _ = booklet.page(pdf_page)
    hits = page.count(needle)
    if hits != 1:
        raise GenerationError(f"anchor {key}: {hits} hits on page {pdf_page}")
    start = page.index(needle)
    end = min(len(page), start + len(needle) + context)
    located = span(booklet, pdf_page, start, end, label=f"anchor {key}")
    located["needle"] = needle
    located["needle_sha256"] = sha256_text(needle)
    return located


# --------------------------------------------------------------------------
# The delivered layout
# --------------------------------------------------------------------------

_FOOTNOTE = r"(?P<footnote>[0-9](?:,[0-9])*)?"


def _spaced(name: str) -> str:
    """``name``, tolerating exactly one interior space.

    The February 2023 extraction splits the field code on the rows that carry
    a left-margin section label (``P 04470``, ``RE CID``). This is a text
    extraction artifact of that PDF, not a different layout, so the
    corroborating pass names the tolerance explicitly instead of hiding it.
    """

    options = [re.escape(name)]
    options.extend(
        re.escape(name[:cut]) + " " + re.escape(name[cut:])
        for cut in range(1, len(name))
    )
    return "(?:" + "|".join(options) + ")"


def _locate(
    booklet: Booklet,
    pages: tuple[int, ...],
    pattern: re.Pattern[str],
    *,
    label: str,
) -> tuple[int, re.Match[str]]:
    hits = [
        (pdf_page, match)
        for pdf_page in pages
        for match in pattern.finditer(booklet.page(pdf_page)[0])
    ]
    if len(hits) != 1:
        raise GenerationError(f"{label}: {len(hits)} hits on pages {pages}")
    return hits[0]


def _row_bounds(page: str, match: re.Match[str]) -> tuple[int, int]:
    """The whole physical line the layout entry sits on, verbatim."""

    start = page.rfind("\n", 0, match.start()) + 1
    end = page.find("\n", match.end())
    return start, len(page) if end < 0 else end


def code_field_entry(
    booklets: dict[str, Booklet],
    booklet_key: str,
    layout_page: int,
    position: int,
    name: str,
) -> dict[str, object]:
    booklet = booklets[booklet_key]
    pattern = re.compile(
        rf"(?<![0-9]){position}\.[ \t]+{re.escape(name)}{_FOOTNOTE}(?![A-Za-z0-9])"
    )
    pdf_page, match = _locate(
        booklet, (layout_page,), pattern, label=f"{booklet_key} code {position} {name}"
    )
    page, _ = booklet.page(pdf_page)
    start, end = _row_bounds(page, match)
    entry = span(booklet, pdf_page, start, end, label=f"code {name}")
    entry["footnote_markers"] = match.group("footnote")
    return entry


def amount_field_entry(
    booklets: dict[str, Booklet],
    booklet_key: str,
    layout_pages: tuple[int, ...],
    position: int,
    name: str,
    *,
    tolerant: bool,
) -> tuple[dict[str, object], str, bool]:
    booklet = booklets[booklet_key]
    exact = re.compile(
        rf"(?<![0-9.]){position}[ \t]+{re.escape(name)}{_FOOTNOTE}"
        rf"(?![A-Za-z0-9])[ \t]*(?P<label>[^\n]*)"
    )
    used_exact = True
    try:
        pdf_page, match = _locate(
            booklet,
            layout_pages,
            exact,
            label=f"{booklet_key} amount {position} {name}",
        )
    except GenerationError:
        if not tolerant:
            raise
        used_exact = False
        loose = re.compile(
            rf"(?<![0-9.]){position}[ \t]+{_spaced(name)}{_FOOTNOTE}"
            rf"(?![A-Za-z0-9])[ \t]*(?P<label>[^\n]*)"
        )
        pdf_page, match = _locate(
            booklet,
            layout_pages,
            loose,
            label=f"{booklet_key} amount {position} {name} (tolerant)",
        )
    page, _ = booklet.page(pdf_page)
    start, end = _row_bounds(page, match)
    entry = span(booklet, pdf_page, start, end, label=f"amount {name}")
    entry["footnote_markers"] = match.group("footnote")
    return entry, match.group("label").strip(), used_exact


def code_definition_entry(
    booklets: dict[str, Booklet],
    booklet_key: str,
    pages: tuple[int, ...],
    position: int,
    name: str,
) -> tuple[dict[str, object], str]:
    booklet = booklets[booklet_key]
    pattern = re.compile(
        rf"(?<![0-9]){position}\.[ \t]+{re.escape(name)}"
        rf"(?![A-Za-z0-9])[ \t]*(?P<label>[^\n]*)"
    )
    pdf_page, match = _locate(
        booklet, pages, pattern, label=f"code definition {position} {name}"
    )
    page, _ = booklet.page(pdf_page)
    start, end = _row_bounds(page, match)
    return span(booklet, pdf_page, start, end, label=f"code def {name}"), match.group(
        "label"
    ).strip()


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------

MAIN_CODE_LAYOUT_PAGE = 6
MAIN_AMOUNT_LAYOUT_PAGES = (7, 8, 9, 10, 11)
MAIN_CODE_DEFINITION_PAGES = (12, 13, 14, 15, 16, 17, 18, 19)
DEMOGRAPHIC_LAYOUT_PAGE = 5
DEMOGRAPHIC_CODE_DEFINITION_PAGES = (6, 7, 8)

#: The closed set of reasons a delivered column is outside this increment.
NOT_SELECTED = {
    "code_field_not_an_amount": (
        "The publisher's CORE RECORD LAYOUT lists this name under 'Code "
        "Fields'. It is a code, a count or an indicator, never a dollar "
        "amount."
    ),
    "misc_code_not_an_amount": (
        "The publisher's own 'Misc Codes' block sits inside the amount-field "
        "list and holds a return identifier, a weight with implied decimals, "
        "two sample counts, a sample code and a marginal tax rate with an "
        "implied decimal point. Membership of the amount list is therefore "
        "not evidence that a column is money."
    ),
    "amount_field_outside_this_increment": (
        "A publisher amount field that this bounded increment does not "
        "project. It stays source-only: no unit, sign or period claim is "
        "made about it here, and nothing downstream may treat its absence "
        "from the projection as a statement about its value."
    ),
}


def _projection_index() -> dict[str, Projected]:
    index: dict[str, Projected] = {}
    for entry in PROJECTION:
        if entry.field in index:
            raise GenerationError(f"duplicate projection field {entry.field}")
        if entry.field in REFUSED_FIELDS:
            raise GenerationError(f"projection names a refused field {entry.field}")
        if entry.fit_admission not in (
            "not_admitted_pending_basis_review",
            "unresolved_retained",
        ):
            raise GenerationError(f"{entry.field}: bad fit_admission")
        index[entry.field] = entry
    return index


def build_main_layout(
    booklets: dict[str, Booklet], delivered: list[str], selected: dict[str, Projected]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    codes, amounts = delivered[:43], delivered[43:]
    if len(codes) != 43 or len(amounts) != 179:
        raise GenerationError(
            f"delivered main header splits {len(codes)}/{len(amounts)}, not 43/179"
        )
    columns: list[dict[str, object]] = []
    ordering: list[tuple[int, int]] = []
    tolerant_positions: list[dict[str, object]] = []

    for position, name in enumerate(codes, 1):
        entry = code_field_entry(
            booklets, "main_2022", MAIN_CODE_LAYOUT_PAGE, position, name
        )
        corroboration = code_field_entry(
            booklets, "main_2023", MAIN_CODE_LAYOUT_PAGE, position, name
        )
        definition, label = code_definition_entry(
            booklets, "main_2022", MAIN_CODE_DEFINITION_PAGES, position, name
        )
        ordering.append((entry["pdf_page"], entry["page_start"]))
        columns.append(
            {
                "name": name,
                "delivered_index": position - 1,
                "publisher_group": "code_field",
                "publisher_position": position,
                "publisher_label": label,
                "publisher_signed_marker": False,
                "layout_row": entry,
                "layout_row_corroboration": corroboration,
                "code_definition_row": definition,
                "admission": "not_selected",
                "not_selected_reason": "code_field_not_an_amount",
            }
        )

    for position, name in enumerate(amounts, 1):
        delivered_index = 43 + position - 1
        entry, label, _ = amount_field_entry(
            booklets,
            "main_2022",
            MAIN_AMOUNT_LAYOUT_PAGES,
            position,
            name,
            tolerant=False,
        )
        corroboration, _, exact = amount_field_entry(
            booklets,
            "main_2023",
            MAIN_AMOUNT_LAYOUT_PAGES,
            position,
            name,
            tolerant=True,
        )
        corroboration["code_token_spacing"] = (
            "exact" if exact else "one_interior_space_from_pdf_text_extraction"
        )
        if not exact:
            tolerant_positions.append({"position": position, "name": name})
        ordering.append((entry["pdf_page"], entry["page_start"]))
        column: dict[str, object] = {
            "name": name,
            "delivered_index": delivered_index,
            "publisher_group": "amount_or_large_code",
            "publisher_position": position,
            "publisher_label": label,
            "publisher_signed_marker": "(+/-)" in label,
            "layout_row": entry,
            "layout_row_corroboration": corroboration,
        }
        if name in selected:
            column["admission"] = "monetary_source_projection"
        else:
            column["admission"] = "not_selected"
            column["not_selected_reason"] = (
                "misc_code_not_an_amount"
                if name in MISC_CODES
                else "amount_field_outside_this_increment"
            )
        columns.append(column)

    if ordering != sorted(ordering):
        raise GenerationError(
            "the delivered header order is not the printed record-layout order"
        )
    facts = {
        "delivered_header_width": len(delivered),
        "code_fields": len(codes),
        "amount_or_large_code_fields": len(amounts),
        "printed_order_matches_delivered_order": True,
        "corroborating_edition_tolerant_positions": tolerant_positions,
    }
    return columns, facts


def build_demographic_layout(
    booklets: dict[str, Booklet], delivered: list[str]
) -> tuple[list[dict[str, object]], dict[str, object]]:
    columns: list[dict[str, object]] = []
    ordering: list[int] = []
    for position, name in enumerate(delivered, 1):
        entry = code_field_entry(
            booklets, "demographic_2022", DEMOGRAPHIC_LAYOUT_PAGE, position, name
        )
        ordering.append(entry["page_start"])
        column: dict[str, object] = {
            "name": name,
            "delivered_index": position - 1,
            "publisher_group": "code_field",
            "publisher_position": position,
            "publisher_signed_marker": False,
            "layout_row": entry,
            "admission": "not_selected",
            "not_selected_reason": "code_field_not_an_amount",
        }
        try:
            definition, label = code_definition_entry(
                booklets,
                "demographic_2022",
                DEMOGRAPHIC_CODE_DEFINITION_PAGES,
                position,
                name,
            )
        except GenerationError:
            column["publisher_label"] = None
            column["code_definition_row"] = None
            column["label_state"] = "no_code_definition_entry_retained_unresolved"
        else:
            column["publisher_label"] = label
            column["code_definition_row"] = definition
            column["label_state"] = "resolved_from_code_definitions"
        columns.append(column)
    if ordering != sorted(ordering):
        raise GenerationError("demographic delivered order is not the printed order")
    return columns, {
        "delivered_header_width": len(delivered),
        "code_fields": len(delivered),
        "printed_order_matches_delivered_order": True,
    }


def build_projection(
    booklets: dict[str, Booklet],
    main_columns: list[dict[str, object]],
    selected: dict[str, Projected],
) -> dict[str, object]:
    by_name = {column["name"]: column for column in main_columns}
    missing = sorted(set(selected) - set(by_name))
    if missing:
        raise GenerationError(f"projection names undelivered columns: {missing}")
    entries: list[dict[str, object]] = []
    # One order everywhere: the delivered file's, which is the publisher's
    # printed record-layout order. The concept grouping stays visible in
    # ``concept_group`` rather than in the column sequence.
    ordered = sorted(
        PROJECTION, key=lambda spec: by_name[spec.field]["delivered_index"]
    )
    for spec in ordered:
        column = by_name[spec.field]
        marker = bool(column["publisher_signed_marker"])
        if marker and spec.negative_documented_by_definition:
            raise GenerationError(
                f"{spec.field}: marker-signed columns need no definition claim"
            )
        if marker:
            negative_state = "documented_by_record_layout_marker"
        elif spec.negative_documented_by_definition:
            negative_state = "documented_by_publisher_definition_text"
        else:
            negative_state = "not_documented_unresolved"
        entries.append(
            {
                "field": spec.field,
                "concept": spec.concept,
                "concept_group": spec.concept_group,
                "delivered_index": column["delivered_index"],
                "publisher_position": column["publisher_position"],
                "publisher_label": column["publisher_label"],
                "layout_row": column["layout_row"],
                "dtype": "int64",
                "lexical_width": 13,
                "grammar": "ascii_decimal_signed_integer",
                "units": {
                    "currency": "USD",
                    "scale": "whole_dollars",
                    "implied_decimals": 0,
                    "field_width_digits": 12,
                    "basis_anchors": ["amount_field_width", "rounding_grid"],
                    "note": (
                        "Every amount field is rounded by the publisher before "
                        "release; nonzero amounts under $5 become $2 with the "
                        "sign retained. Whole dollars is the delivered scale. "
                        "A delivered zero is a value in the edited public file; "
                        "the rounding rule does not prove the original return "
                        "reported zero or rule out other disclosure edits."
                    ),
                },
                "sign": {
                    "accepted_grammar": "optional_leading_minus_then_decimal_digits",
                    "publisher_signed_marker": marker,
                    "negative_values": negative_state,
                    "definition_anchors": list(spec.definition_anchors),
                    "policy": (
                        "The parser accepts a signed token for every projected "
                        "column and records the observed negative and zero "
                        "counts. A missing (+/-) marker is preserved as a "
                        "missing marker; it is never enforced as a nonnegative "
                        "domain, and no absolute value is ever taken."
                    ),
                    "negative_zero": "refused",
                },
                "missingness": {
                    "empty_token": "refused",
                    "empty_token_basis": (
                        "Neither booklet declares a missing-value sentinel or a "
                        "blank convention for an amount field, so an empty "
                        "token is a refusal to be preserved and reviewed, not a "
                        "zero to be invented."
                    ),
                    "zero_meaning": "delivered_zero_in_edited_public_file",
                    "zero_basis_anchor": "rounding_grid",
                    "absent_demographic_row": (
                        "irrelevant_to_this_column; the projection is main-file "
                        "left and every delivered main record is retained, "
                        "matched or not"
                    ),
                },
                "return_grain": {
                    "grain": "source_return",
                    "statement": (
                        "One delivered main record. Not a person, not a "
                        "household, and not necessarily one taxpayer's return."
                    ),
                    "anchors": ["not_one_taxpayer", "national_estimates_only"],
                },
                "source_period_status": {
                    "status": "unresolved_raw_FLPDYR",
                    "anchors": ["flpdyr_printed_range", "fiscal_year_conversion"],
                    "statement": (
                        "The cover identifies Tax Year 2015, the code "
                        "definitions print FLPDYR 2011-2014, the disclosure "
                        "section says returns filed for years earlier than Tax "
                        "Year 2012 were removed, and the reviewed header audit "
                        "observed 2012-2015. The row period is therefore "
                        "unresolved and no growth, restatement or target-year "
                        "use is admitted."
                    ),
                },
                "edit_status": {
                    "universal_anchors": [
                        "rounding_grid",
                        "rebalancing",
                        "aggregate_records",
                        "not_one_taxpayer",
                    ],
                    "column_anchors": list(spec.edit_anchors),
                },
                "disclosure_aggregate_rows": (
                    "retained_and_reported_separately; never a person, never "
                    "disaggregated"
                ),
                "unresolved": [
                    {"code": code, "statement": statement}
                    for code, statement in spec.unresolved
                ],
                "derived_combinations": "refused",
                "fit_admission": spec.fit_admission,
                "target_year_use": "refused_pending_independent_growth_decision",
                "notes": list(spec.notes),
            }
        )
    return {
        "grain": "source_return",
        "row_policy": (
            "every delivered main record is projected, including the 88,021 "
            "with no demographic row and the four disclosure aggregates"
        ),
        "columns": entries,
        "column_count": len(entries),
        "concept_groups": sorted({spec.concept_group for spec in PROJECTION}),
        "target_year_use": "refused_pending_independent_growth_decision",
        "period_semantics": "unresolved_raw_FLPDYR",
        "amounts_decoded": True,
    }


def canonical_bytes(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def build_document(documents_dir: Path) -> dict[str, object]:
    booklets = read_booklets(documents_dir)
    raw_bytes = RAW_DEFINITION.read_bytes()
    raw = json.loads(raw_bytes.decode("utf-8"))
    if raw.get("schema") != "microcosm.us.puf_2015_raw_source_definition":
        raise GenerationError("the raw source definition is not the expected schema")
    raw_canonical = canonical_bytes(raw)
    selected = _projection_index()

    main_columns, main_facts = build_main_layout(
        booklets, list(raw["sources"]["main"]["delivered_header"]), selected
    )
    demographic_columns, demographic_facts = build_demographic_layout(
        booklets, list(raw["sources"]["demographic"]["delivered_header"])
    )
    projection = build_projection(booklets, main_columns, selected)

    anchors = {key: anchor(booklets, key) for key in sorted(ANCHORS)}
    referenced: set[str] = set()
    for column in projection["columns"]:
        referenced.update(column["units"]["basis_anchors"])
        referenced.update(column["sign"]["definition_anchors"])
        referenced.update(column["return_grain"]["anchors"])
        referenced.update(column["source_period_status"]["anchors"])
        referenced.update(column["edit_status"]["universal_anchors"])
        referenced.update(column["edit_status"]["column_anchors"])
    for entry in REFUSED_DERIVATIONS.values():
        referenced.update(entry["anchors"])
    unknown = sorted(referenced - set(anchors))
    if unknown:
        raise GenerationError(f"unknown anchor references: {unknown}")

    selected_names = set(selected)
    for name in NOTABLE_SOURCE_ONLY:
        column = next((c for c in main_columns if c["name"] == name), None)
        if column is None:
            raise GenerationError(f"{name} is annotated but not delivered")
        if name in selected_names:
            raise GenerationError(f"{name} is annotated source-only but projected")

    refused_fields = dict(REFUSED_FIELDS)
    delivered_names = {column["name"] for column in main_columns}
    for name in refused_fields:
        if name in delivered_names:
            raise GenerationError(f"{name} is refused but present in the layout")
        for booklet in booklets.values():
            if name in booklet.text:
                raise GenerationError(f"{name} is refused but printed in a booklet")

    document = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "route": "packaged",
        "authority": "irs_soi_2015_public_use_file_retained_booklets",
        "raw_source_definition": {
            "schema": raw["schema"],
            "schema_version": raw["schema_version"],
            "sha256": hashlib.sha256(raw_canonical).hexdigest(),
            "bytes": len(raw_canonical),
            "relationship": (
                "This document extends the reviewed raw source definition. It "
                "restates none of that document's pins as its own authority "
                "and never modifies it; the artifact binds both."
            ),
        },
        "sources": {
            name: {
                key: raw["sources"][name][key]
                for key in (
                    "source_name",
                    "codec",
                    "bytes",
                    "sha256",
                    "git_blob_sha1",
                    "header_record_canonical_sha256",
                    "delivered_header_width",
                    "data_records",
                )
            }
            for name in ("main", "demographic")
        },
        "booklets": {
            key: {
                "title": spec["title"],
                "edition": spec["edition"],
                "role": spec["role"],
                "pdf_sha256": spec["pdf_sha256"],
                "pdf_bytes": spec["pdf_bytes"],
                "pdf_pages": spec["pdf_pages"],
                "text_sha256": spec["text_sha256"],
                "text_characters": len(booklets[key].text),
                "text_extraction": (
                    "pypdf 6.11.0, strict, page texts joined with U+000C"
                ),
                "offset_unit": "unicode_codepoints_into_the_pinned_text",
                "span_digest": "sha256 of the span encoded as UTF-8",
            }
            for key, spec in BOOKLETS.items()
        },
        "layout": {
            "main": {**main_facts, "columns": main_columns},
            "demographic": {**demographic_facts, "columns": demographic_columns},
            "not_selected_reasons": NOT_SELECTED,
            "notable_source_only": NOTABLE_SOURCE_ONLY,
            "prefix_is_not_evidence": (
                "No column's role is inferred from an E, P, S or T prefix or "
                "from membership of the amount-field list. RECID, S006, S008, "
                "S009, WSAMP and TXRT are inside that list and none is money; "
                "TXRT carries an implied decimal point."
            ),
        },
        "projection": projection,
        "anchors": anchors,
        "source_conflicts": {
            "record_count_includes_header": {
                "status": "documentary_wording_unresolved",
                "anchors": ["record_count_intro", "record_count_technical"],
                "statement": (
                    "The technical description says 207,696 CSV records with "
                    "the first record containing field names, implying 207,695 "
                    "data records. The accepted raw-byte audit observed "
                    "207,696 data records plus a header. Preserve the measured "
                    "count without silently reconciling this wording conflict."
                ),
            },
            "rounding_five_dollar_tie": {
                "status": "documentary_rounding_tie_unresolved",
                "anchors": ["rounding_grid"],
                "statement": (
                    "The nearest-ten-dollar rule does not state how a five-"
                    "dollar tie is resolved. A delivered zero does not identify "
                    "reported zero, absence, nonapplicability or another edit."
                ),
            },
        },
        "refused_fields": refused_fields,
        "refused_derivations": REFUSED_DERIVATIONS,
        "non_claims": {
            "amount_growth_or_restatement": False,
            "target_year_frame_entry": False,
            "person_or_household_expansion": False,
            "aggregate_record_disaggregation": False,
            "tax_outcome_projection": False,
            "weight_promotion": False,
            "policyengine_import": False,
            "fit_or_calibration": False,
            "release_eligibility": False,
            "microdata_read_by_this_generator": False,
        },
        "generation": {
            "generator": "tools/build_us_puf_source_layout.py",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "reads": [
                "the three pinned booklet text extractions",
                "the packaged raw source definition",
            ],
            "reads_microdata": False,
        },
    }
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--documents-dir",
        type=Path,
        required=True,
        help="directory containing the three content-pinned booklet texts",
    )
    parser.add_argument(
        "--receipt",
        type=Path,
        help="optional local execution receipt; interpreter is outside document identity",
    )
    parser.add_argument("--write", action="store_true", help="write the document")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="regenerate and compare against the packaged bytes",
    )
    args = parser.parse_args(argv)
    document = build_document(args.documents_dir)
    payload = canonical_bytes(document) + b"\n"
    if args.write:
        OUTPUT.write_bytes(payload)
        print(f"wrote {OUTPUT} ({len(payload)} bytes)")
    if args.verify:
        current = OUTPUT.read_bytes() if OUTPUT.exists() else b""
        if current != payload:
            print("REGENERATED DOCUMENT DIFFERS FROM THE PACKAGED BYTES")
            return 1
        print(f"verified {OUTPUT} ({len(payload)} bytes)")
    if not args.write and not args.verify:
        print(json.dumps(document, indent=2)[:4000])
    if args.receipt:
        args.receipt.write_text(
            json.dumps(
                {
                    "interpreter_version": sys.version.split()[0],
                    "generator_sha256": document["generation"]["generator_sha256"],
                    "documents_dir": str(args.documents_dir.resolve()),
                    "booklet_text_sha256": {
                        name: spec["text_sha256"] for name, spec in BOOKLETS.items()
                    },
                    "document_sha256": hashlib.sha256(payload).hexdigest(),
                    "document_bytes": len(payload),
                    "verified_packaged_bytes": args.verify,
                    "microdata_read": False,
                },
                indent=2,
            )
            + "\n"
        )
    print(
        json.dumps(
            {
                "main_columns": len(document["layout"]["main"]["columns"]),
                "demographic_columns": len(
                    document["layout"]["demographic"]["columns"]
                ),
                "projected_columns": document["projection"]["column_count"],
                "anchors": len(document["anchors"]),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
