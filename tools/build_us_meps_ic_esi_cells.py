"""Regenerate the packaged MEPS-IC employer-premium cell table (microcosm#454).

The ``meps_esi_premiums`` source stage assigns each ESI policyholder the
MEPS-IC employer share of the premium for their coverage tier, employer size
and state (or census division for State and local government employers).
Those cells are published only as PDF tables, so this tool pins
the official AHRQ PDFs by URL, byte length and SHA-256, extracts the cells the
stage reads with ``pdftotext -layout`` (poppler), and writes them, with their
provenance, to
``packages/microcosm-build/src/microcosm/build/us_runtime/data/
meps_ic_esi_premium_cells.json``.

Every value is copied as published. Suppressed cells (``--``) stay ``null``:
the fallback for them is stage logic, declared in the stage manifest, not
data. The premium tables' United States rows are checked against the AHRQ
national figures they must reproduce, and a parse that does not find exactly
one row per state, division or firm-size label refuses.

Three national blocks ride along: the share of enrollees whose coverage
required no employee contribution (Tables II.C/D/E.4.a), which turns the
average employee contribution into the contribution of those who pay one; the
private-sector enrollment rows (Tables II.B.1, II.B.2, II.B.2.b and
II.C/D/E.4) behind the stage's active-employee cross-check; and the firm-size
pretax-contribution offer rates.

Run (downloads the four PDFs, about 34 MB, into ``--pdf-dir``)::

    uv run python tools/build_us_meps_ic_esi_cells.py --pdf-dir /tmp/meps-ic

``--check`` regenerates in memory and fails if the committed file differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "us_runtime"
    / "data"
    / "meps_ic_esi_premium_cells.json"
)
_MEPS_ROOT = "https://meps.ahrq.gov/mepsweb/data_stats/summ_tables/insr/"


@dataclass(frozen=True)
class PinnedPdf:
    key: str
    url: str
    size_bytes: int
    sha256: str
    survey_year: int
    description: str

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


PDFS: tuple[PinnedPdf, ...] = (
    PinnedPdf(
        key="private_state_2025",
        url=_MEPS_ROOT + "state/series_2/2025/ic25_iia_f.pdf",
        size_bytes=12_530_005,
        sha256="ebebf083d132e2b5a72ac99d659ee6d46a382280a47d2ff2d20f6e9e25b69330",
        survey_year=2025,
        description=(
            "MEPS-IC 2025 Series II (private sector, by firm size and State): "
            "Tables II.C/D/E.1 (average total premium per enrolled employee), "
            "II.C/D/E.2 (average employee contribution), II.C/D/E.4.a (percent "
            "of enrollees whose coverage required no employee contribution), "
            "and the enrollment rows II.B.1, II.B.2, II.B.2.b and II.C/D/E.4"
        ),
    ),
    PinnedPdf(
        key="private_state_2024",
        url=_MEPS_ROOT + "state/series_2/2024/ic24_iia_f.pdf",
        size_bytes=10_687_163,
        sha256="4ae0dada70fac7c147508b946a2d00cf2e8930f5550f6beb9d8a1bf3de71d20b",
        survey_year=2024,
        description=(
            "MEPS-IC 2024 Series II (private sector, by firm size and State); "
            "the United States rows age the 2024 State and local government "
            "cells to 2025 and carry the 2024 private-sector enrollment"
        ),
    ),
    PinnedPdf(
        key="public_division_2024",
        url=_MEPS_ROOT + "national/series_3/2024/ic24_iiia_g.pdf",
        size_bytes=1_765_717,
        sha256="462afdd2b2629b32b0c4822a33d7e32260e4c4e196f6098f0d933a3c8ba0be79",
        survey_year=2024,
        description=(
            "MEPS-IC 2024 Series III (State and local government, by "
            "government type and size and census division): Tables "
            "III.C/D/E.1 and III.C/D/E.2. 2024 is the latest published year "
            "for this series (the 2025 series is not yet released)."
        ),
    ),
    PinnedPdf(
        key="private_national_2025",
        url=_MEPS_ROOT + "national/series_1/2025/ic25_ia_g.pdf",
        size_bytes=9_210_366,
        sha256="cceba5da68e41caf1b03b83f93d1c20fc1aa1b3ccd0eefc4a9dfc80754c950f3",
        survey_year=2025,
        description=(
            "MEPS-IC 2025 Series I (private sector, national): Table I.A.2 "
            "(percent of establishments that offer health insurance) and "
            "Table I.A.2.j (percent of establishments that offer pretax "
            "employee contributions to health insurance), by firm size"
        ),
    ),
)

TIERS = ("single", "employee_plus_one", "family")
_TIER_LETTER = {"single": "C", "employee_plus_one": "E", "family": "D"}
_SERIES_II_COLUMNS = (
    "total",
    "lt10",
    "10_24",
    "25_99",
    "100_999",
    "1000plus",
    "lt50",
    "50plus",
)
#: The columns the stage reads. The finer firm-size columns cannot be used:
#: CPS ASEC NOEMP bands (under 10, 10-49, 50-99, 100-499, 500-999, 1000+) do
#: not nest inside MEPS-IC's 10-24 / 25-99 / 100-999 bands.
_KEPT_SERIES_II_COLUMNS = ("total", "lt50", "50plus")
_SERIES_III_COLUMNS = (
    "all_state_and_local",
    "state",
    "local_lt250",
    "local_250_999",
    "local_1000_4999",
    "local_5000_9999",
    "local_10000plus",
)
_KEPT_SERIES_III_COLUMNS = ("all_state_and_local", "state")
DIVISIONS = (
    "New England",
    "Middle Atlantic",
    "East North Central",
    "West North Central",
    "South Atlantic",
    "East South Central",
    "West South Central",
    "Mountain",
    "Pacific",
)
STATE_FIPS = {
    "Alabama": 1,
    "Alaska": 2,
    "Arizona": 4,
    "Arkansas": 5,
    "California": 6,
    "Colorado": 8,
    "Connecticut": 9,
    "Delaware": 10,
    "District of Columbia": 11,
    "Florida": 12,
    "Georgia": 13,
    "Hawaii": 15,
    "Idaho": 16,
    "Illinois": 17,
    "Indiana": 18,
    "Iowa": 19,
    "Kansas": 20,
    "Kentucky": 21,
    "Louisiana": 22,
    "Maine": 23,
    "Maryland": 24,
    "Massachusetts": 25,
    "Michigan": 26,
    "Minnesota": 27,
    "Mississippi": 28,
    "Missouri": 29,
    "Montana": 30,
    "Nebraska": 31,
    "Nevada": 32,
    "New Hampshire": 33,
    "New Jersey": 34,
    "New Mexico": 35,
    "New York": 36,
    "North Carolina": 37,
    "North Dakota": 38,
    "Ohio": 39,
    "Oklahoma": 40,
    "Oregon": 41,
    "Pennsylvania": 42,
    "Rhode Island": 44,
    "South Carolina": 45,
    "South Dakota": 46,
    "Tennessee": 47,
    "Texas": 48,
    "Utah": 49,
    "Vermont": 50,
    "Virginia": 51,
    "Washington": 53,
    "West Virginia": 54,
    "Wisconsin": 55,
    "Wyoming": 56,
}

#: United States rows each extracted table must reproduce: the AHRQ national
#: figures (MEPS-IC Research Findings and the national dashboard). A parse
#: that drifts from them refuses.
EXPECTED_NATIONAL = {
    ("private_state_2025", "II.C.1"): 9_025.0,
    ("private_state_2025", "II.D.1"): 26_281.0,
    ("private_state_2024", "II.C.1"): 8_486.0,
    ("private_state_2024", "II.C.2"): 1_789.0,
    ("private_state_2024", "II.E.1"): 16_931.0,
    ("private_state_2024", "II.E.2"): 4_707.0,
    ("private_state_2024", "II.D.1"): 24_540.0,
    ("private_state_2024", "II.D.2"): 7_216.0,
}

_NUMBER = re.compile(r"(--|\d[\d,]*(?:\.\d+)?%?)(\s*\*)?")


class ExtractionError(ValueError):
    """The PDF text did not parse into the reviewed table shape."""


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(pin: PinnedPdf, pdf_dir: Path) -> Path:
    """Return the verified local copy of one pinned PDF, downloading it once."""

    pdf_dir.mkdir(parents=True, exist_ok=True)
    path = pdf_dir / pin.filename
    if not path.exists():
        request = urllib.request.Request(
            pin.url, headers={"User-Agent": "microcosm-build/meps-ic-esi-cells"}
        )
        temporary = path.with_suffix(".pdf.part")
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
            temporary.write_bytes(response.read())
        temporary.rename(path)
    size = path.stat().st_size
    digest = _sha256(path)
    if size != pin.size_bytes or digest != pin.sha256:
        raise ExtractionError(
            f"{pin.filename}: expected {pin.size_bytes} bytes / {pin.sha256}, "
            f"found {size} bytes / {digest}. AHRQ reissued the file; review "
            "the new tables before re-pinning."
        )
    return path


def pdf_text(path: Path) -> list[str]:
    completed = subprocess.run(
        ["pdftotext", "-layout", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.lstrip("\f") for line in completed.stdout.splitlines()]


def pdftotext_version() -> str:
    completed = subprocess.run(
        ["pdftotext", "-v"], check=True, capture_output=True, text=True
    )
    return (completed.stderr or completed.stdout).splitlines()[0].strip()


def _table_block(lines: list[str], table_id: str) -> tuple[int, list[str]]:
    head = re.compile(rf"^Table {re.escape(table_id)} (Average|Percent|Number)")
    starts = [
        index
        for index, line in enumerate(lines)
        if head.match(line) and "Standard errors" not in line
    ]
    if len(starts) != 1:
        raise ExtractionError(f"Table {table_id}: found {len(starts)} headers.")
    start = starts[0]
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].startswith("Table ")
        ),
        len(lines),
    )
    return start, lines[start:end]


def _row_values(block: list[str], label: str, width: int, table_id: str):
    pattern = re.compile(rf"^\s*{re.escape(label)}\s{{2,}}")
    rows = [line for line in block if pattern.match(line)]
    if len(rows) != 1:
        raise ExtractionError(f"Table {table_id}: {label!r} matched {len(rows)} rows.")
    tokens = _NUMBER.findall(rows[0].split(label, 1)[1])
    if len(tokens) != width:
        raise ExtractionError(
            f"Table {table_id}: {label!r} has {len(tokens)} values, not {width}: "
            f"{rows[0]!r}."
        )
    values: list[float | None] = []
    unreliable: list[bool] = []
    for token, star in tokens:
        values.append(
            None if token == "--" else float(token.replace(",", "").replace("%", ""))
        )
        unreliable.append(bool(star.strip()))
    return values, unreliable


def _table_year(block: list[str], table_id: str) -> int:
    match = re.search(r"United States, (\d{4})", " ".join(block[:3]))
    if match is None:
        raise ExtractionError(f"Table {table_id}: no survey year in the title.")
    return int(match.group(1))


def _state_divisions(block: list[str]) -> dict[str, str]:
    """State -> census division from the table's own division headings."""

    division = None
    membership: dict[str, str] = {}
    for line in block:
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1] in DIVISIONS:
            division = stripped[:-1]
            continue
        for state in STATE_FIPS:
            if re.match(rf"^{re.escape(state)}\s{{2,}}", stripped):
                if division is None:
                    raise ExtractionError(f"{state} precedes any division heading.")
                membership[state] = division
    if set(membership) != set(STATE_FIPS):
        raise ExtractionError(
            "Division headings did not place every state: missing "
            f"{sorted(set(STATE_FIPS) - set(membership))}."
        )
    return membership


def _series_ii_table(lines, pin: PinnedPdf, table_id: str) -> dict:
    start, block = _table_block(lines, table_id)
    if _table_year(block, table_id) != pin.survey_year:
        raise ExtractionError(f"Table {table_id} is not survey year {pin.survey_year}.")
    rows = {}
    for label in ("United States", *STATE_FIPS):
        # Labels anchor at the line start, so "Virginia" cannot match the
        # "West Virginia" row nor "Kansas" the "Arkansas" row.
        values, unreliable = _row_values(
            block, label, len(_SERIES_II_COLUMNS), table_id
        )
        by_column = dict(zip(_SERIES_II_COLUMNS, values, strict=True))
        flags = dict(zip(_SERIES_II_COLUMNS, unreliable, strict=True))
        key = "US" if label == "United States" else f"{STATE_FIPS[label]:02d}"
        rows[key] = {column: by_column[column] for column in _KEPT_SERIES_II_COLUMNS}
        starred = [column for column in _KEPT_SERIES_II_COLUMNS if flags[column]]
        if starred:
            rows[key]["unreliable"] = starred
    expected = EXPECTED_NATIONAL.get((pin.key, table_id))
    if expected is not None and rows["US"]["total"] != expected:
        raise ExtractionError(
            f"Table {table_id} ({pin.survey_year}) United States total is "
            f"{rows['US']['total']}, not the published {expected}."
        )
    return {"text_line": start + 1, "title": block[0].strip(), "rows": rows}


def _series_iii_table(lines, pin: PinnedPdf, table_id: str) -> dict:
    start, block = _table_block(lines, table_id)
    if _table_year(block, table_id) != pin.survey_year:
        raise ExtractionError(f"Table {table_id} is not survey year {pin.survey_year}.")
    rows = {}
    for label in ("United States", *DIVISIONS):
        values, _unreliable = _row_values(
            block, label, len(_SERIES_III_COLUMNS), table_id
        )
        by_column = dict(zip(_SERIES_III_COLUMNS, values, strict=True))
        kept = {column: by_column[column] for column in _KEPT_SERIES_III_COLUMNS}
        if any(value is None for value in kept.values()):
            raise ExtractionError(f"Table {table_id}: {label} has a suppressed cell.")
        rows["US" if label == "United States" else label] = kept
    return {"text_line": start + 1, "title": block[0].strip(), "rows": rows}


def _full_title(block: list[str]) -> str:
    """A table title through its ``United States, <year>`` line (it may wrap)."""

    title = ""
    for line in block[:4]:
        title = f"{title} {line.strip()}".strip()
        if re.search(r"United States, \d{4}$", title):
            return title
    raise ExtractionError(f"Table title does not end in a survey year: {block[0]!r}.")


def _national_row(lines, pin: PinnedPdf, table_id: str) -> dict:
    """The United States row of one Series II table, kept firm-size columns."""

    start, block = _table_block(lines, table_id)
    if _table_year(block, table_id) != pin.survey_year:
        raise ExtractionError(f"Table {table_id} is not survey year {pin.survey_year}.")
    values, unreliable = _row_values(
        block, "United States", len(_SERIES_II_COLUMNS), table_id
    )
    by_column = dict(zip(_SERIES_II_COLUMNS, values, strict=True))
    flags = dict(zip(_SERIES_II_COLUMNS, unreliable, strict=True))
    kept = {column: by_column[column] for column in _KEPT_SERIES_II_COLUMNS}
    bad = [
        column
        for column in _KEPT_SERIES_II_COLUMNS
        if kept[column] is None or flags[column]
    ]
    if bad:
        raise ExtractionError(
            f"Table {table_id}: United States {bad} suppressed or flagged unreliable."
        )
    return {"text_line": start + 1, "title": _full_title(block), **kept}


def _no_contribution_shares(lines, pin: PinnedPdf) -> dict:
    """Percent of enrollees, by tier, whose coverage required no contribution.

    National by firm size only: most State cells of these tables are flagged
    unreliable or suppressed.
    """

    return {
        tier: _national_row(lines, pin, f"II.{_TIER_LETTER[tier]}.4.a")
        for tier in TIERS
    }


def _private_enrollment(lines, pin: PinnedPdf) -> dict:
    """National private-sector enrollment rows for the active-employee check."""

    return {
        "employees": _national_row(lines, pin, "II.B.1"),
        "offer_percent": _national_row(lines, pin, "II.B.2"),
        "enrolled_percent": _national_row(lines, pin, "II.B.2.b"),
        "tier_share_percent": {
            tier: _national_row(lines, pin, f"II.{_TIER_LETTER[tier]}.4")
            for tier in TIERS
        },
    }


def _pretax_offer_shares(lines, pin: PinnedPdf) -> dict:
    """Firm-size pretax-contribution and health-insurance offer rates."""

    _start, offer = _table_block(lines, "I.A.2")
    offer_values, _ = _row_values(offer, "United States", 8, "I.A.2")
    offer_by_column = dict(zip(_SERIES_II_COLUMNS, offer_values, strict=True))
    start_j, pretax = _table_block(lines, "I.A.2.j")
    out = {}
    for key, label in (
        ("total", "United States"),
        ("lt50", "Less than 50 employees"),
        ("50plus", "50+ employees"),
    ):
        values, unreliable = _row_values(pretax, label, 3, "I.A.2.j")
        if unreliable[0]:
            raise ExtractionError(f"Table I.A.2.j {label} is flagged unreliable.")
        out[key] = {
            "pretax_contribution_offer_percent": values[0],
            "health_insurance_offer_percent": offer_by_column[key],
        }
    return {
        "text_line": start_j + 1,
        "title": pretax[0].strip(),
        "offer_title": offer[0].strip(),
        "rows": out,
    }


def build_payload(pdf_dir: Path) -> dict:
    texts = {pin.key: pdf_text(fetch(pin, pdf_dir)) for pin in PDFS}
    private_2025 = {
        tier: {
            "premium": _series_ii_table(
                texts["private_state_2025"],
                PDFS[0],
                f"II.{_TIER_LETTER[tier]}.1",
            ),
            "employee_contribution": _series_ii_table(
                texts["private_state_2025"],
                PDFS[0],
                f"II.{_TIER_LETTER[tier]}.2",
            ),
        }
        for tier in TIERS
    }
    private_2024_national = {}
    for tier in TIERS:
        private_2024_national[tier] = {}
        for measure, suffix in (("premium", "1"), ("employee_contribution", "2")):
            table = _series_ii_table(
                texts["private_state_2024"],
                PDFS[1],
                f"II.{_TIER_LETTER[tier]}.{suffix}",
            )
            private_2024_national[tier][measure] = {
                "text_line": table["text_line"],
                "total": table["rows"]["US"]["total"],
            }
    public_2024 = {
        tier: {
            "premium": _series_iii_table(
                texts["public_division_2024"],
                PDFS[2],
                f"III.{_TIER_LETTER[tier]}.1",
            ),
            "employee_contribution": _series_iii_table(
                texts["public_division_2024"],
                PDFS[2],
                f"III.{_TIER_LETTER[tier]}.2",
            ),
        }
        for tier in TIERS
    }
    _start, division_block = _table_block(texts["private_state_2025"], "II.C.1")
    divisions = _state_divisions(division_block)
    return {
        "schema_version": 2,
        "issue": "PolicyEngine/microcosm#454",
        "description": (
            "MEPS-IC average total premium and average employee contribution "
            "per enrolled employee, by coverage tier, for the "
            "meps_esi_premiums source stage: private-sector cells by State and "
            "firm size (2025), State and local government cells by census "
            "division (2024, the latest published year), the 2024 private "
            "national rows that age the government cells, the 2025 national "
            "share of enrollees whose coverage required no employee "
            "contribution, the 2024 and 2025 national private-sector "
            "enrollment rows, and the 2025 firm-size pretax-contribution "
            "offer rates. Values are copied as published; null marks a cell "
            "AHRQ suppressed ('--'). Dollar values are annual; percent values "
            "are percents; employee counts are persons."
        ),
        "generator": "tools/build_us_meps_ic_esi_cells.py",
        "extraction": {
            "method": "pdftotext -layout, one row per label, refusing any "
            "label that does not match exactly one line",
            "pdftotext": pdftotext_version(),
        },
        "sources": {
            pin.key: {
                "url": pin.url,
                "size_bytes": pin.size_bytes,
                "sha256": pin.sha256,
                "survey_year": pin.survey_year,
                "publisher": (
                    "Agency for Healthcare Research and Quality, Center for "
                    "Financing, Access and Cost Trends, Medical Expenditure "
                    "Panel Survey-Insurance Component"
                ),
                "description": pin.description,
            }
            for pin in PDFS
        },
        "tiers": list(TIERS),
        "state_census_division": {
            f"{STATE_FIPS[state]:02d}": divisions[state] for state in STATE_FIPS
        },
        "private_state_2025": private_2025,
        "private_national_2024": private_2024_national,
        "public_division_2024": public_2024,
        "no_contribution_share_2025": _no_contribution_shares(
            texts["private_state_2025"], PDFS[0]
        ),
        "private_enrollment_national": {
            "2024": _private_enrollment(texts["private_state_2024"], PDFS[1]),
            "2025": _private_enrollment(texts["private_state_2025"], PDFS[0]),
        },
        "pretax_contribution_2025": _pretax_offer_shares(
            texts["private_national_2025"], PDFS[3]
        ),
    }


def render(payload: dict) -> str:
    return json.dumps(payload, indent=1, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        required=True,
        help="Directory caching the pinned AHRQ PDFs (downloaded when absent).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if the committed cell table differs from a regeneration.",
    )
    args = parser.parse_args(argv)
    rendered = render(build_payload(args.pdf_dir))
    if args.check:
        if OUTPUT_PATH.read_text() != rendered:
            print(f"{OUTPUT_PATH} is stale; regenerate it.", file=sys.stderr)
            return 1
        print(f"{OUTPUT_PATH} is current.")
        return 0
    OUTPUT_PATH.write_text(rendered)
    print(f"Wrote {OUTPUT_PATH}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
