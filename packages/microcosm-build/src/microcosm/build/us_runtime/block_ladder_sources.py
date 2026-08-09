"""Parsers for the primary Census/OMB sources behind the US block ladder.

Each function parses one source format into plain mappings keyed by the
15-digit 2020 tabulation-block geoid (as ``int``), and
:func:`assemble_us_block_ladder` joins them into the array payload the
ladder artifact stores. Download orchestration lives in
``tools/build_us_block_ladder_artifact.py``; everything here is pure and
unit-testable.

Sources and their formats (verified against the published files):

- 119th Congressional District BEF (``cd119.zip`` → ``NationalCD119.txt``):
  comma-delimited ``GEOID,CDFP``; ``ZZ`` marks blocks assigned to no district
  (large water bodies), ``98`` marks non-voting delegate districts (DC).
- 2020 Block Assignment Files (``BlockAssign_ST{fips}_{usps}.zip``):
  pipe-delimited per-layer files — ``_SLDU``/``_SLDL`` carry
  ``BLOCKID|DISTRICT`` (``ZZZ`` = unassigned), ``_INCPLACE_CDP`` carries
  ``BLOCKID|PLACEFP`` (blank = in no place).
- 2020 P.L. 94-171 legacy geographic header (``{usps}geo2020.pl``):
  pipe-delimited, 97 fields; summary level 750 rows are blocks, with the
  15-digit geocode at field 9 and ``POP100`` at field 90 (validated per
  state against the summary-level 040 state row).
- OMB CBSA delineations (``list1_2023.xlsx``): county → CBSA rows keyed by
  the FIPS state + county code columns.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

#: The 50 states plus DC: (state FIPS, USPS abbreviation, P.L. 94-171
#: directory name). Puerto Rico and the island territories are outside the
#: US artifact's state spine.
US_STATES: tuple[tuple[str, str, str], ...] = (
    ("01", "AL", "Alabama"),
    ("02", "AK", "Alaska"),
    ("04", "AZ", "Arizona"),
    ("05", "AR", "Arkansas"),
    ("06", "CA", "California"),
    ("08", "CO", "Colorado"),
    ("09", "CT", "Connecticut"),
    ("10", "DE", "Delaware"),
    ("11", "DC", "District_of_Columbia"),
    ("12", "FL", "Florida"),
    ("13", "GA", "Georgia"),
    ("15", "HI", "Hawaii"),
    ("16", "ID", "Idaho"),
    ("17", "IL", "Illinois"),
    ("18", "IN", "Indiana"),
    ("19", "IA", "Iowa"),
    ("20", "KS", "Kansas"),
    ("21", "KY", "Kentucky"),
    ("22", "LA", "Louisiana"),
    ("23", "ME", "Maine"),
    ("24", "MD", "Maryland"),
    ("25", "MA", "Massachusetts"),
    ("26", "MI", "Michigan"),
    ("27", "MN", "Minnesota"),
    ("28", "MS", "Mississippi"),
    ("29", "MO", "Missouri"),
    ("30", "MT", "Montana"),
    ("31", "NE", "Nebraska"),
    ("32", "NV", "Nevada"),
    ("33", "NH", "New_Hampshire"),
    ("34", "NJ", "New_Jersey"),
    ("35", "NM", "New_Mexico"),
    ("36", "NY", "New_York"),
    ("37", "NC", "North_Carolina"),
    ("38", "ND", "North_Dakota"),
    ("39", "OH", "Ohio"),
    ("40", "OK", "Oklahoma"),
    ("41", "OR", "Oregon"),
    ("42", "PA", "Pennsylvania"),
    ("44", "RI", "Rhode_Island"),
    ("45", "SC", "South_Carolina"),
    ("46", "SD", "South_Dakota"),
    ("47", "TN", "Tennessee"),
    ("48", "TX", "Texas"),
    ("49", "UT", "Utah"),
    ("50", "VT", "Vermont"),
    ("51", "VA", "Virginia"),
    ("53", "WA", "Washington"),
    ("54", "WV", "West_Virginia"),
    ("55", "WI", "Wisconsin"),
    ("56", "WY", "Wyoming"),
)

_PL_GEO_SUMMARY_LEVEL_FIELD = 2
_PL_GEO_GEOCODE_FIELD = 9
_PL_GEO_POP100_FIELD = 90
_PL_BLOCK_SUMMARY_LEVEL = "750"
_PL_STATE_SUMMARY_LEVEL = "040"

#: Census "assigned to no district" markers, by source convention.
_CD_UNASSIGNED = "ZZ"
_SLD_UNASSIGNED_MARKERS = frozenset({"ZZZ", "ZZ"})
#: Non-voting delegate districts (DC) are the district-00 rung of the
#: Microcosm at-large convention ``state_fips * 100 + 00``.
_CD_DELEGATE = "98"


def parse_national_cd_bef(lines: Iterable[str]) -> dict[int, int]:
    """Parse the national CD BEF into block geoid → CD geoid (SSDD).

    Blocks marked ``ZZ`` (no district) are dropped; delegate districts
    (``98``) normalize to the at-large district ``00``.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source="national CD BEF")
    if [part.strip().upper() for part in header.split(",")] != ["GEOID", "CDFP"]:
        raise ValueError(
            f"National CD BEF header must be 'GEOID,CDFP', got {header!r}."
        )
    result: dict[int, int] = {}
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(",")
        if len(parts) != 2:
            raise ValueError(
                f"National CD BEF line {line_number} must have two fields, "
                f"got {stripped!r}."
            )
        geoid_raw, district_raw = parts[0].strip(), parts[1].strip()
        if district_raw == _CD_UNASSIGNED:
            continue
        block = _block_geoid(geoid_raw, source=f"national CD BEF line {line_number}")
        if district_raw == _CD_DELEGATE:
            district = 0
        elif district_raw.isdigit() and len(district_raw) == 2:
            district = int(district_raw)
        else:
            raise ValueError(
                f"National CD BEF line {line_number} has district {district_raw!r}; "
                "expected a two-digit code, '98', or 'ZZ'."
            )
        if block in result:
            raise ValueError(
                f"National CD BEF assigns block {geoid_raw} more than once."
            )
        result[block] = (block // 10**13) * 100 + district
    if not result:
        raise ValueError("National CD BEF contained no district assignments.")
    return result


def parse_baf_district_file(lines: Iterable[str], *, label: str) -> dict[int, str]:
    """Parse a BAF SLDU/SLDL file into block geoid → 3-character district.

    Census unassigned markers normalize to ``""``.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source=label)
    if [part.strip().upper() for part in header.split("|")] != [
        "BLOCKID",
        "DISTRICT",
    ]:
        raise ValueError(f"{label} header must be 'BLOCKID|DISTRICT', got {header!r}.")
    result: dict[int, str] = {}
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue
        parts = stripped.split("|")
        if len(parts) != 2:
            raise ValueError(
                f"{label} line {line_number} must have two fields, got {stripped!r}."
            )
        block = _block_geoid(parts[0].strip(), source=f"{label} line {line_number}")
        district = parts[1].strip()
        if district in _SLD_UNASSIGNED_MARKERS:
            district = ""
        if district and len(district) != 3:
            raise ValueError(
                f"{label} line {line_number} has district {district!r}; "
                "expected a 3-character code or an unassigned marker."
            )
        result[block] = district
    return result


def parse_baf_place_file(lines: Iterable[str], *, label: str) -> dict[int, int]:
    """Parse a BAF INCPLACE_CDP file into block geoid → place FIPS int.

    Blocks in no incorporated place or CDP carry a blank ``PLACEFP`` and map
    to ``0``.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source=label)
    if [part.strip().upper() for part in header.split("|")] != [
        "BLOCKID",
        "PLACEFP",
    ]:
        raise ValueError(f"{label} header must be 'BLOCKID|PLACEFP', got {header!r}.")
    result: dict[int, int] = {}
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue
        parts = stripped.split("|")
        if len(parts) != 2:
            raise ValueError(
                f"{label} line {line_number} must have two fields, got {stripped!r}."
            )
        block = _block_geoid(parts[0].strip(), source=f"{label} line {line_number}")
        place_raw = parts[1].strip()
        if not place_raw:
            result[block] = 0
            continue
        if not (place_raw.isdigit() and len(place_raw) == 5):
            raise ValueError(
                f"{label} line {line_number} has PLACEFP {place_raw!r}; "
                "expected 5 digits or blank."
            )
        result[block] = int(place_raw)
    return result


def parse_pl_geo_blocks(lines: Iterable[str], *, state_fips: str) -> dict[int, int]:
    """Parse a P.L. 94-171 geo header into populated block geoid → POP100.

    Zero-population blocks are excluded — they cannot host households.
    The per-state population is validated against the state's own
    summary-level 040 row, so a field-position drift in the fixed layout
    fails loudly instead of shipping nonsense weights.
    """

    blocks: dict[int, int] = {}
    state_population: int | None = None
    for line_number, line in enumerate(lines, start=1):
        fields = line.rstrip("\n").split("|")
        if len(fields) <= _PL_GEO_POP100_FIELD:
            continue
        summary_level = fields[_PL_GEO_SUMMARY_LEVEL_FIELD]
        if summary_level == _PL_STATE_SUMMARY_LEVEL:
            state_population = int(fields[_PL_GEO_POP100_FIELD])
            geocode = fields[_PL_GEO_GEOCODE_FIELD].strip()
            if geocode != state_fips:
                raise ValueError(
                    f"P.L. 94-171 geo state row geocode {geocode!r} does not "
                    f"match expected state {state_fips!r}."
                )
        elif summary_level == _PL_BLOCK_SUMMARY_LEVEL:
            geocode = fields[_PL_GEO_GEOCODE_FIELD].strip()
            block = _block_geoid(
                geocode,
                source=f"P.L. 94-171 geo line {line_number}",
            )
            if not geocode.startswith(state_fips):
                raise ValueError(
                    f"P.L. 94-171 block {geocode} is outside state {state_fips}."
                )
            if block in blocks:
                raise ValueError(f"P.L. 94-171 geo file repeats block {geocode}.")
            population = int(fields[_PL_GEO_POP100_FIELD])
            if population > 0:
                blocks[block] = population
    if state_population is None:
        raise ValueError(
            f"P.L. 94-171 geo file for state {state_fips} has no summary-"
            "level 040 state row to validate against."
        )
    total = sum(blocks.values())
    if total != state_population:
        raise ValueError(
            f"P.L. 94-171 block populations for state {state_fips} sum to "
            f"{total:,} but the state row records {state_population:,}."
        )
    if not blocks:
        raise ValueError(
            f"P.L. 94-171 geo file for state {state_fips} has no populated blocks."
        )
    return blocks


def parse_cbsa_delineations(
    rows: Iterable[tuple[Any, ...]],
) -> dict[str, int]:
    """Parse OMB delineation rows into county FIPS (5-digit) → CBSA code.

    ``rows`` are the spreadsheet's raw rows (header row included, in order),
    so the caller owns the xlsx mechanics and tests can pass plain tuples.
    """

    iterator = iter(rows)
    header: tuple[Any, ...] | None = None
    for row in iterator:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        if "CBSA Code" in cells and "FIPS State Code" in cells:
            header = tuple(cells)
            break
    if header is None:
        raise ValueError(
            "OMB delineation rows have no header row containing 'CBSA Code' "
            "and 'FIPS State Code'."
        )
    cbsa_index = header.index("CBSA Code")
    state_index = header.index("FIPS State Code")
    county_index = header.index("FIPS County Code")
    result: dict[str, int] = {}
    for row in iterator:
        cells = [str(cell).strip() if cell is not None else "" for cell in row]
        if len(cells) <= max(cbsa_index, state_index, county_index):
            continue
        cbsa_raw = cells[cbsa_index]
        state_raw = cells[state_index]
        county_raw = cells[county_index]
        if not (cbsa_raw and state_raw and county_raw):
            continue
        cbsa = _five_digit_code(cbsa_raw)
        if cbsa is None:
            # Footnote rows below the table are not data. (A numeric-typed
            # CBSA cell such as 35620.0 still parses.)
            continue
        try:
            county = f"{int(float(state_raw)):02d}{int(float(county_raw)):03d}"
        except ValueError as exc:
            raise ValueError(
                f"OMB delineation row with CBSA {cbsa_raw!r} has malformed "
                f"FIPS cell(s): state {state_raw!r}, county {county_raw!r}."
            ) from exc
        existing = result.get(county)
        if existing is not None and existing != cbsa:
            raise ValueError(
                f"OMB delineations assign county {county} to both CBSA "
                f"{existing} and {cbsa}."
            )
        result[county] = cbsa
    if not result:
        raise ValueError("OMB delineation rows contained no county→CBSA rows.")
    return result


def assemble_us_block_ladder(
    *,
    block_population: Mapping[int, int],
    cd_by_block: Mapping[int, int],
    sldu_by_block: Mapping[int, str],
    sldl_by_block: Mapping[int, str],
    place_by_block: Mapping[int, int],
    cbsa_by_county: Mapping[str, int],
    metadata: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """Join the parsed sources into the ladder artifact's NPZ payload.

    Every populated block must have a congressional district — a populated
    block the CD BEF does not cover is a source defect, not a skippable row.
    SLD and place maps may legitimately not cover a block (states without a
    layer); absent entries mean unassigned.
    """

    blocks = np.asarray(sorted(block_population), dtype=np.int64)
    missing_cd = [block for block in blocks.tolist() if block not in cd_by_block]
    if missing_cd:
        examples = [f"{block:015d}" for block in missing_cd[:5]]
        raise ValueError(
            f"{len(missing_cd)} populated block(s) have no congressional "
            f"district in the CD BEF; examples: {examples}."
        )
    population = np.asarray(
        [block_population[block] for block in blocks.tolist()], dtype=np.int64
    )
    cd = np.asarray([cd_by_block[block] for block in blocks.tolist()], dtype=np.int64)
    sldu = np.asarray(
        [sldu_by_block.get(block, "") for block in blocks.tolist()], dtype="U3"
    )
    sldl = np.asarray(
        [sldl_by_block.get(block, "") for block in blocks.tolist()], dtype="U3"
    )
    place = np.asarray(
        [place_by_block.get(block, 0) for block in blocks.tolist()], dtype=np.int32
    )
    cbsa = np.asarray(
        [cbsa_by_county.get(f"{block:015d}"[:5], 0) for block in blocks.tolist()],
        dtype=np.int32,
    )
    return {
        "block_geoid": blocks,
        "population": population,
        "congressional_district_geoid": cd,
        "sldu": sldu,
        "sldl": sldl,
        "place_fips": place,
        "cbsa_code": cbsa,
        "metadata_json": np.asarray(json.dumps(dict(metadata), sort_keys=True)),
    }


def _five_digit_code(value: str) -> int | None:
    """Normalize a spreadsheet cell to a 5-digit code; None if it is not one."""
    if value.endswith(".0"):
        value = value[:-2]
    if value.isdigit() and len(value) == 5:
        return int(value)
    return None


def _required_header(iterator: Iterable[str], *, source: str) -> str:
    for line in iterator:
        stripped = line.strip()
        if stripped:
            return stripped
    raise ValueError(f"{source} is empty.")


def _block_geoid(value: str, *, source: str) -> int:
    if not (value.isdigit() and len(value) == 15):
        raise ValueError(f"{source}: block geoid must be 15 digits, got {value!r}.")
    return int(value)


__all__ = [
    "US_STATES",
    "assemble_us_block_ladder",
    "parse_baf_district_file",
    "parse_baf_place_file",
    "parse_cbsa_delineations",
    "parse_national_cd_bef",
    "parse_pl_geo_blocks",
]
