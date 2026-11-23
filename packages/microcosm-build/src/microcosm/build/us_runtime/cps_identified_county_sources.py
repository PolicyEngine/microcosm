"""Parse Census's list of counties the CPS ASEC public file identifies.

Each year's CPS ASEC technical documentation (``cpsmar{yy}.pdf``) carries, in
Appendix F, "List 4: FIPS County Codes": the counties a public-file record can
name through ``GTCO``, grouped under state headings, with the note "Counties
are only included on this list if the entire county is identified." A
trailing asterisk marks a county that is also a single-county micropolitan
area. :mod:`~microcosm.build.us_runtime.block_location` uses the list as one
of the two confirmations that rule a county out for a ``GTCO = 0`` record.

The parser reads the ``pdftotext -layout`` rendering of the PDF. Download and
text extraction live in ``tools/build_us_cps_identified_counties.py``;
everything here is pure and unit-testable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from microcosm.build.us_runtime.block_ladder_sources import US_STATES

LIST_HEADING = "List 4: FIPS County Codes"
#: The preamble sentence that makes List 4 a whole-county guarantee. ASEC
#: 2023-2025 carry it; the 2026 documentation drops it.
ENTIRE_COUNTY_GUARANTEE = (
    "only included on this list if the entire county is identified"
)
_END_MARKERS = ("* Counties marked", "List 5", "APPENDIX")
#: The one kind of non-county line inside the list after its first state
#: heading: the running page header ("SPECIFIC METROPOLITAN AREAS  F-18").
_PAGE_HEADER = re.compile(r"^SPECIFIC METROPOLITAN AREAS\b")
_COUNTY_LINE = re.compile(r"^(\d{3})\s{2,}(.+?)(\*?)$")
_STATE_BY_NAME = {name.replace("_", " "): fips for fips, _, name in US_STATES}


@dataclass(frozen=True)
class CpsIdentifiedCounty:
    """One county on the ASEC identified-county list."""

    state_fips: str
    county_code: str
    county_name: str
    single_county_micropolitan: bool


def list_carries_entire_county_guarantee(lines: Iterable[str]) -> bool:
    """Whether List 4's preamble promises whole-county identification.

    Reads the text between the list heading (its last occurrence) and the
    first county line, whitespace-normalized, for
    :data:`ENTIRE_COUNTY_GUARANTEE`.
    """

    rendered = [line.rstrip("\n") for line in lines]
    starts = [
        index for index, line in enumerate(rendered) if line.strip() == LIST_HEADING
    ]
    if not starts:
        raise ValueError(f"no {LIST_HEADING!r} heading in the rendered text.")
    preamble: list[str] = []
    for line in rendered[starts[-1] + 1 :]:
        if _COUNTY_LINE.match(line.strip()) or line.strip() in _STATE_BY_NAME:
            break
        preamble.append(line.strip())
    return ENTIRE_COUNTY_GUARANTEE in " ".join(" ".join(preamble).split())


def parse_cps_identified_county_list(
    lines: Iterable[str],
) -> list[CpsIdentifiedCounty]:
    """Parse List 4 from a ``pdftotext -layout`` rendering of an ASEC techdoc.

    The list starts at the last line reading exactly ``List 4: FIPS County
    Codes`` (earlier matches are the table of contents) and ends at its
    asterisk footnote, the next list, or the next appendix. State names head
    their counties. Before the first state heading, the preamble and column
    labels are skipped. After it, the only lines the list admits are state
    headings, county lines and the running page header, so any other
    non-blank line ends the list. The 2026 documentation has no footnote
    after List 4, and its next section would otherwise be read as counties.
    Raises on a county line before any state heading, a repeated county, or
    an empty list.
    """

    rendered = [line.rstrip("\n") for line in lines]
    starts = [
        index for index, line in enumerate(rendered) if line.strip() == LIST_HEADING
    ]
    if not starts:
        raise ValueError(f"no {LIST_HEADING!r} heading in the rendered text.")
    counties: list[CpsIdentifiedCounty] = []
    seen: set[tuple[str, str]] = set()
    state: str | None = None
    for line in rendered[starts[-1] + 1 :]:
        text = line.strip()
        if any(text.startswith(marker) for marker in _END_MARKERS):
            break
        if text in _STATE_BY_NAME:
            state = _STATE_BY_NAME[text]
            continue
        match = _COUNTY_LINE.match(text)
        if match is None:
            if state is not None and text and not _PAGE_HEADER.match(text):
                break
            continue
        if state is None:
            raise ValueError(f"county line {text!r} precedes every state heading.")
        key = (state, match.group(1))
        if key in seen:
            raise ValueError(f"county {state}{match.group(1)} is listed twice.")
        seen.add(key)
        counties.append(
            CpsIdentifiedCounty(
                state_fips=state,
                county_code=match.group(1),
                county_name=match.group(2).strip(),
                single_county_micropolitan=bool(match.group(3)),
            )
        )
    if not counties:
        raise ValueError(f"{LIST_HEADING!r} parsed to no counties.")
    return counties


__all__ = [
    "ENTIRE_COUNTY_GUARANTEE",
    "LIST_HEADING",
    "CpsIdentifiedCounty",
    "list_carries_entire_county_guarantee",
    "parse_cps_identified_county_list",
]
