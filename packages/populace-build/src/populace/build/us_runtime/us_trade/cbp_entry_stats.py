"""CBP fiscal-year entry-summary statistics from the public trade stats page.

CBP publishes national entry-summary counts only as an HTML statistics page
(``CBP Trade Statistics``, cbp.gov/newsroom/stats/trade): total and informal
entry summaries, total import value, and total duty/taxes/fees by fiscal
year. There is no monthly, HTS, or country split in any machine-readable
CBP publication we could locate, so these fiscal-year totals are admitted
as the only public entry-count anchor and the import-entry generator's
entry-size distribution beyond them is an explicit documented assumption
(populace#615).

The ingest archives the page bytes verbatim (SHA-256 recorded) and parses
the "Imports and Revenue Collection" table from the archived bytes. Only
exact-integer cells become facts; CBP renders older fiscal years as rounded
display values ("50.08 million"), which are preserved in the parse result
but never admitted as exact facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "CBP_TRADE_STATS_URL",
    "CBP_ENTRY_MEASURES",
    "CbpEntryStats",
    "CbpFiscalYearCell",
    "fiscal_year_end",
    "parse_cbp_trade_stats",
]

CBP_TRADE_STATS_URL = "https://www.cbp.gov/newsroom/stats/trade"

#: Row label → (measure_id, unit) for the "Imports and Revenue Collection"
#: table. Labels are matched after tag-stripping and whitespace collapse.
CBP_ENTRY_MEASURES = {
    "Total Import Value for Goods": ("total_import_value", "usd"),
    "Total Entry Summaries": ("total_entry_summaries", "count"),
    "Total Informal Entry Summaries": ("informal_entry_summaries", "count"),
    "Total Duty, Taxes, and Fees Collected": ("duty_taxes_fees_collected", "usd"),
}

_TABLE_PATTERN = re.compile(r"<table[^>]*>.*?</table>", re.DOTALL)
_ROW_PATTERN = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL)
_CELL_PATTERN = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_EXACT_VALUE_PATTERN = re.compile(r"^\$?[0-9][0-9,]*$")
_FISCAL_YEAR_PATTERN = re.compile(r"FY\s*(\d{4})")
_AS_OF_PATTERN = re.compile(
    r"FY\s*\d{4}\s*(?:and|,)\s*FY\s*\d{4}\s*(?:is|are)\s*updated\s*as\s*of"
    r"\s*([A-Za-z]+)\s+(\d{1,2}),\s*(\d{4})",
)

_MONTH_NUMBERS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def fiscal_year_end(fiscal_year: int) -> str:
    """ISO date the federal fiscal year ends: September 30 of its name year."""
    return f"{fiscal_year:04d}-09-30"


@dataclass(frozen=True)
class CbpFiscalYearCell:
    """One published cell: exact integer when CBP printed exact digits."""

    measure_id: str
    unit: str
    fiscal_year: int
    text: str
    exact_value: int | None

    @property
    def is_exact(self) -> bool:
        return self.exact_value is not None


@dataclass(frozen=True)
class CbpEntryStats:
    """Parsed "Imports and Revenue Collection" table plus its as-of note.

    ``as_of_date`` is the publisher's update endpoint from the as-of note in
    ISO form (empty when the page carries no note). A fiscal year whose
    September 30 end postdates that endpoint is only covered *to* the
    endpoint — its cells are fiscal-year-to-date observations, not completed
    annual totals — and consumers must label them accordingly.
    """

    cells: tuple[CbpFiscalYearCell, ...]
    as_of_note: str
    as_of_date: str = ""

    def exact_cells(self) -> tuple[CbpFiscalYearCell, ...]:
        return tuple(cell for cell in self.cells if cell.is_exact)


def parse_cbp_trade_stats(raw_html: bytes) -> CbpEntryStats:
    """Parse the archived CBP trade statistics page bytes.

    Raises:
        ValueError: If the entry-summaries table or any expected row is
            missing — a silent page redesign must fail the ingest, not ship
            a partial anchor series.
    """
    html = raw_html.decode("utf-8", errors="replace")
    table_html = _entry_summaries_table(html)
    rows = _ROW_PATTERN.findall(table_html)
    if not rows:
        raise ValueError("CBP entry-summaries table has no rows.")
    header_cells = [_clean(cell) for cell in _CELL_PATTERN.findall(rows[0])]
    fiscal_years = _header_fiscal_years(header_cells)
    cells: list[CbpFiscalYearCell] = []
    seen_labels: set[str] = set()
    for row_html in rows[1:]:
        row_cells = [_clean(cell) for cell in _CELL_PATTERN.findall(row_html)]
        if not row_cells:
            continue
        label = _match_measure_label(row_cells[0])
        if label is None:
            continue
        seen_labels.add(label)
        measure_id, unit = CBP_ENTRY_MEASURES[label]
        value_cells = row_cells[1:]
        if len(value_cells) != len(fiscal_years):
            raise ValueError(
                f"CBP row {label!r} has {len(value_cells)} value cells for "
                f"{len(fiscal_years)} fiscal-year columns."
            )
        for fiscal_year, text in zip(fiscal_years, value_cells, strict=True):
            cells.append(
                CbpFiscalYearCell(
                    measure_id=measure_id,
                    unit=unit,
                    fiscal_year=fiscal_year,
                    text=text,
                    exact_value=_exact_value(text),
                )
            )
    missing = sorted(set(CBP_ENTRY_MEASURES) - seen_labels)
    if missing:
        raise ValueError(
            f"CBP entry-summaries table is missing expected row(s): {missing}."
        )
    note, note_date = _as_of_note(html)
    return CbpEntryStats(cells=tuple(cells), as_of_note=note, as_of_date=note_date)


def _entry_summaries_table(html: str) -> str:
    for table in _TABLE_PATTERN.findall(html):
        if "Total Entry Summaries" in table:
            return table
    raise ValueError(
        "CBP trade statistics page has no table containing "
        "'Total Entry Summaries'; the page layout changed and the parser "
        "must be re-verified against the new layout."
    )


def _header_fiscal_years(header_cells: list[str]) -> tuple[int, ...]:
    fiscal_years: list[int] = []
    for cell in header_cells[1:]:
        match = _FISCAL_YEAR_PATTERN.search(cell)
        if match is None:
            raise ValueError(
                f"CBP entry-summaries header cell {cell!r} is not a fiscal year column."
            )
        fiscal_years.append(int(match.group(1)))
    if not fiscal_years:
        raise ValueError("CBP entry-summaries table has no fiscal-year columns.")
    return tuple(fiscal_years)


def _match_measure_label(cell: str) -> str | None:
    for label in CBP_ENTRY_MEASURES:
        if cell.startswith(label):
            return label
    return None


def _exact_value(text: str) -> int | None:
    if _EXACT_VALUE_PATTERN.match(text.strip()):
        return int(text.strip().lstrip("$").replace(",", ""))
    return None


def _as_of_note(html: str) -> tuple[str, str]:
    """The verbatim as-of note and its date in ISO form (both may be empty)."""
    text = _TAG_PATTERN.sub(" ", html)
    text = re.sub(r"\s+", " ", text)
    match = _AS_OF_PATTERN.search(text)
    if match is None:
        return "", ""
    month_name, day, year = match.group(1), match.group(2), match.group(3)
    month_number = _MONTH_NUMBERS.get(month_name.lower())
    if month_number is None:
        raise ValueError(
            f"CBP as-of note carries an unrecognized month {month_name!r}."
        )
    return (
        match.group(0),
        f"{int(year):04d}-{month_number:02d}-{int(day):02d}",
    )


def _clean(cell_html: str) -> str:
    text = _TAG_PATTERN.sub(" ", cell_html)
    return re.sub(r"\s+", " ", text).strip()
