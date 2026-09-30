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
- 2020 BAF county-subdivision layer (``_MCD``): pipe-delimited
  ``BLOCKID|COUNTYFP|COUSUBFP``; 2020 blocks nest in county subdivisions.
- Census Connecticut crosswalk (``ct_cou_to_cousub_crosswalk.txt``):
  pipe-delimited, a quoted multi-line header, one row per county subdivision
  giving its 2020 county and its 2022 planning region (the county-equivalent
  that replaced it), then a glossary after a blank line.

Connecticut is the one state whose county-equivalents differ between the
two vintages joined here. Its 2020 blocks carry the eight old counties
(09001-09015), while the OMB 2023 delineations list its nine 2022 planning
regions (09110-09190). A CT block reaches its CBSA through its town, because
planning regions are unions of whole towns.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from microcosm.build.us_runtime.geography_ladder import UsBlockLadder

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

#: Connecticut's state FIPS: the state whose OMB 2023 county-equivalents
#: (2022 planning regions) are not its 2020 block counties.
CT_STATE_FIPS = "09"
#: County-subdivision code for areas with no town ("County subdivisions not
#: defined": the coastal water of Fairfield, Middlesex, New Haven and New
#: London counties). Fairfield's is split between two planning regions, so the
#: code does not identify one; no populated 2020 block carries it.
_COUSUB_NOT_DEFINED = "00000"
_CT_CROSSWALK_COLUMNS = (
    "STATEFP",
    "OLD_COUNTYFP",
    "OLD_COUNTY_NAMELSAD",
    "NEW_COUNTYFP",
    "NEW_COUNTY_NAMELSAD",
    "COUSUBFP",
    "OLD_COUSUB_GEOID",
    "NEW_COUSUB_GEOID",
    "COUSUB_NAMELSAD",
    "COUSUBNS",
    "COUSUB_LSAD",
    "COUSUB_FUNCSTAT",
    "COUSUB_CLASSFP",
)


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


def parse_baf_county_subdivision_file(
    lines: Iterable[str], *, label: str
) -> dict[int, str]:
    """Parse a BAF MCD file into block geoid → 10-digit county-subdivision geoid.

    The geoid is state + 2020 county + county-subdivision FIPS, the key the
    Census Connecticut crosswalk uses for a town's 2020 identity. A row whose
    ``COUNTYFP`` disagrees with the block's own county prefix is refused.
    """

    iterator = iter(lines)
    header = _required_header(iterator, source=label)
    if [part.strip().upper() for part in header.split("|")] != [
        "BLOCKID",
        "COUNTYFP",
        "COUSUBFP",
    ]:
        raise ValueError(
            f"{label} header must be 'BLOCKID|COUNTYFP|COUSUBFP', got {header!r}."
        )
    result: dict[int, str] = {}
    for line_number, line in enumerate(iterator, start=2):
        stripped = line.rstrip("\n")
        if not stripped.strip():
            continue
        parts = stripped.split("|")
        if len(parts) != 3:
            raise ValueError(
                f"{label} line {line_number} must have three fields, got {stripped!r}."
            )
        block_raw, county_raw, cousub_raw = (part.strip() for part in parts)
        block = _block_geoid(block_raw, source=f"{label} line {line_number}")
        if not (county_raw.isdigit() and len(county_raw) == 3):
            raise ValueError(
                f"{label} line {line_number} has COUNTYFP {county_raw!r}; "
                "expected 3 digits."
            )
        if not (cousub_raw.isdigit() and len(cousub_raw) == 5):
            raise ValueError(
                f"{label} line {line_number} has COUSUBFP {cousub_raw!r}; "
                "expected 5 digits."
            )
        if block_raw[2:5] != county_raw:
            raise ValueError(
                f"{label} line {line_number} puts block {block_raw} in county "
                f"{county_raw}, but the block geoid's county is {block_raw[2:5]}."
            )
        if block in result:
            raise ValueError(f"{label} assigns block {block_raw} more than once.")
        result[block] = block_raw[:5] + cousub_raw
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


def parse_ct_planning_region_crosswalk(lines: Iterable[str]) -> dict[str, str]:
    """Parse the Census CT crosswalk into town → 2022 planning region.

    Keys are 2020 county-subdivision geoids (``09`` + old county + town,
    the value :func:`parse_baf_county_subdivision_file` yields); values are
    the 5-digit planning-region county-equivalent FIPS the OMB 2023
    delineations use. The table ends at the first blank row (a glossary
    follows). "County subdivisions not defined" rows are skipped: Fairfield
    County's is listed under two planning regions, so the code names no
    single region. A populated block there would fail the block join instead.
    """

    reader = csv.reader(_csv_lines(lines), delimiter="|")
    header: list[str] | None = None
    for row in reader:
        if any(cell.strip() for cell in row):
            # "STATEFP\n(INCITS38)" → "STATEFP": the parenthesized standard
            # is a second header line inside the quoted cell.
            header = [_crosswalk_column_name(cell) for cell in row]
            break
    if header is None:
        raise ValueError("CT planning-region crosswalk is empty.")
    if tuple(header) != _CT_CROSSWALK_COLUMNS:
        raise ValueError(
            "CT planning-region crosswalk header must be "
            f"{list(_CT_CROSSWALK_COLUMNS)}, got {header}."
        )
    column = {name: index for index, name in enumerate(header)}
    result: dict[str, str] = {}
    for row in reader:
        if not any(cell.strip() for cell in row):
            break
        where = f"CT planning-region crosswalk line {reader.line_num}"
        if len(row) != len(header):
            raise ValueError(
                f"{where} must have {len(header)} fields, got {len(row)}: {row!r}."
            )
        cells = {name: row[index].strip() for name, index in column.items()}
        state = cells["STATEFP"]
        old_county = cells["OLD_COUNTYFP"]
        new_county = cells["NEW_COUNTYFP"]
        cousub = cells["COUSUBFP"]
        if state != CT_STATE_FIPS:
            raise ValueError(
                f"{where} has STATEFP {state!r}; expected {CT_STATE_FIPS!r}."
            )
        for name, value, width in (
            ("OLD_COUNTYFP", old_county, 3),
            ("NEW_COUNTYFP", new_county, 3),
            ("COUSUBFP", cousub, 5),
        ):
            if not (value.isdigit() and len(value) == width):
                raise ValueError(
                    f"{where} has {name} {value!r}; expected {width} digits."
                )
        old_geoid = state + old_county + cousub
        new_geoid = state + new_county + cousub
        if (cells["OLD_COUSUB_GEOID"], cells["NEW_COUSUB_GEOID"]) != (
            old_geoid,
            new_geoid,
        ):
            raise ValueError(
                f"{where} geoids {cells['OLD_COUSUB_GEOID']!r}/"
                f"{cells['NEW_COUSUB_GEOID']!r} disagree with its code columns "
                f"({old_geoid}/{new_geoid})."
            )
        if cousub == _COUSUB_NOT_DEFINED:
            continue
        region = state + new_county
        existing = result.get(old_geoid)
        if existing is not None and existing != region:
            raise ValueError(
                f"CT planning-region crosswalk puts town {old_geoid} in both "
                f"{existing} and {region}."
            )
        result[old_geoid] = region
    if not result:
        raise ValueError("CT planning-region crosswalk contained no town rows.")
    return result


def ct_planning_region_by_block(
    blocks: Iterable[int],
    *,
    cousub_by_block: Mapping[int, str],
    planning_region_by_cousub: Mapping[str, str],
) -> dict[int, str]:
    """Map each Connecticut block to its 2022 planning region, via its town.

    ``blocks`` are the populated blocks to map; blocks outside Connecticut
    are ignored. 2020 blocks nest in towns and planning regions are unions of
    whole towns, so the mapping is exact. A CT block without a town, or in a
    town the crosswalk does not list, is a source defect and raises.
    """

    result: dict[int, str] = {}
    missing_town: list[int] = []
    unmapped_town: set[str] = set()
    ct_state = int(CT_STATE_FIPS)
    for block in blocks:
        if block // 10**13 != ct_state:
            continue
        cousub = cousub_by_block.get(block)
        if cousub is None:
            missing_town.append(block)
            continue
        region = planning_region_by_cousub.get(cousub)
        if region is None:
            unmapped_town.add(cousub)
            continue
        result[block] = region
    if missing_town:
        examples = [f"{block:015d}" for block in sorted(missing_town)[:5]]
        raise ValueError(
            f"{len(missing_town)} Connecticut block(s) have no county "
            f"subdivision in the BAF MCD layer; examples: {examples}."
        )
    if unmapped_town:
        raise ValueError(
            "Connecticut county subdivision(s) absent from the planning-region "
            f"crosswalk: {sorted(unmapped_town)[:10]}."
        )
    return result


def cbsa_delineated_states(
    cbsa_by_county: Mapping[str, int], states: Iterable[str]
) -> list[str]:
    """Return the ``states`` (2-digit FIPS) with any CBSA-delineated territory."""

    delineated = {county[:2] for county in cbsa_by_county}
    return sorted(state for state in set(states) if state in delineated)


def us_block_ladder_cbsa_coverage_failures(ladder: UsBlockLadder) -> list[str]:
    """Return one failure per CBSA-delineated state whose blocks all lack a CBSA.

    Every state with OMB-delineated territory has populated blocks inside a
    CBSA, so a state whose every block carries ``cbsa_code == 0`` is a join
    defect (Connecticut's planning regions against 2020 counties was one).
    The states come from the artifact's ``cbsa_delineated_states`` record.
    Artifacts built before that record fall back to every state they
    contain: the OMB 2023 delineations cover territory in all 50 states and
    DC. For a ladder this module's join just built the check cannot fail,
    since the join refuses an unreached delineated county-equivalent. Its
    job is to catch a ladder built elsewhere or before that refusal existed.
    """

    block_state = ladder.block_geoid // 10**13
    present = {f"{state:02d}" for state in np.unique(block_state).tolist()}
    recorded = ladder.metadata.get("cbsa_delineated_states")
    if recorded is None:
        delineated = sorted(present)
    elif isinstance(recorded, list) and all(
        isinstance(state, str) and state.isdigit() and len(state) == 2
        for state in recorded
    ):
        delineated = sorted(set(recorded) & present)
    else:
        return [
            "metadata cbsa_delineated_states must be a list of 2-digit state "
            f"FIPS strings, got {recorded!r}"
        ]
    failures: list[str] = []
    for state in delineated:
        in_state = block_state == int(state)
        if not (ladder.cbsa_code[in_state] > 0).any():
            failures.append(
                f"state {state}: all {int(in_state.sum()):,} populated blocks "
                "have cbsa_code 0, although the CBSA delineation covers "
                "territory there"
            )
    return failures


def assemble_us_block_ladder(
    *,
    block_population: Mapping[int, int],
    cd_by_block: Mapping[int, int],
    sldu_by_block: Mapping[int, str],
    sldl_by_block: Mapping[int, str],
    place_by_block: Mapping[int, int],
    cbsa_by_county: Mapping[str, int],
    metadata: Mapping[str, Any],
    cbsa_county_by_block: Mapping[int, str] | None = None,
) -> dict[str, np.ndarray]:
    """Join the parsed sources into the ladder artifact's NPZ payload.

    Every populated block must have a congressional district — a populated
    block the CD BEF does not cover is a source defect, not a skippable row.
    SLD and place maps may legitimately not cover a block (states without a
    layer); absent entries mean unassigned.

    A block's CBSA is looked up by its county-equivalent in the delineation
    vintage: its 2020 county, unless ``cbsa_county_by_block`` names another
    (Connecticut's planning regions, from :func:`ct_planning_region_by_block`).
    A state with any such entry must have one for every populated block.
    Every delineated county-equivalent in a built state must be reached by at
    least one populated block; one that is not means the delineation and the
    blocks disagree on county-equivalents, and the join refuses.
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
    counties, county_index = np.unique(
        _cbsa_county_equivalents(blocks, cbsa_county_by_block or {}),
        return_inverse=True,
    )
    county_codes = [f"{county:05d}" for county in counties.tolist()]
    reached = set(county_codes)
    built_states = {county[:2] for county in reached}
    unreached = sorted(
        county
        for county in cbsa_by_county
        if county[:2] in built_states and county not in reached
    )
    if unreached:
        raise ValueError(
            f"{len(unreached)} CBSA-delineated county-equivalent(s) in the "
            "built states match no populated block's county-equivalent; "
            f"examples: {unreached[:10]}. The delineation and the blocks use "
            "different county-equivalents (Connecticut's 2022 planning "
            "regions against 2020 counties is the known case: pass "
            "cbsa_county_by_block from ct_planning_region_by_block)."
        )
    cbsa = np.asarray(
        [cbsa_by_county.get(county, 0) for county in county_codes], dtype=np.int32
    )[county_index]
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


def _cbsa_county_equivalents(
    blocks: np.ndarray, cbsa_county_by_block: Mapping[int, str]
) -> np.ndarray:
    """Each block's delineation county-equivalent as a 5-digit FIPS int.

    ``blocks`` is sorted. Entries for blocks outside it (unpopulated blocks)
    are validated and otherwise ignored.
    """

    counties = blocks // 10**10
    if not cbsa_county_by_block:
        return counties
    override_block = np.fromiter(
        cbsa_county_by_block, dtype=np.int64, count=len(cbsa_county_by_block)
    )
    override_county = np.empty(len(override_block), dtype=np.int64)
    for index, (block, county) in enumerate(cbsa_county_by_block.items()):
        if not (isinstance(county, str) and county.isdigit() and len(county) == 5):
            raise ValueError(
                f"cbsa_county_by_block maps block {block:015d} to {county!r}; "
                "expected a 5-digit county-equivalent FIPS."
            )
        if int(county[:2]) != block // 10**13:
            raise ValueError(
                f"cbsa_county_by_block maps block {block:015d} to "
                f"county-equivalent {county} in another state."
            )
        override_county[index] = int(county)
    position = np.searchsorted(blocks, override_block)
    populated = position < len(blocks)
    populated[populated] = blocks[position[populated]] == override_block[populated]
    counties[position[populated]] = override_county[populated]
    remapped = np.zeros(len(blocks), dtype=bool)
    remapped[position[populated]] = True
    missing = ~remapped & np.isin(blocks // 10**13, np.unique(override_block // 10**13))
    if missing.any():
        examples = [f"{block:015d}" for block in blocks[missing][:5].tolist()]
        raise ValueError(
            f"{int(missing.sum())} populated block(s) lack a county-equivalent "
            "in cbsa_county_by_block although other blocks in their state have "
            f"one; examples: {examples}."
        )
    return counties


def _csv_lines(lines: Iterable[str]) -> Iterator[str]:
    """Lines with a leading byte-order mark removed; it would hide the quoting."""
    for index, line in enumerate(lines):
        yield line.removeprefix("\ufeff") if index == 0 else line


def _crosswalk_column_name(cell: str) -> str:
    """Header cell → column name, dropping a parenthesized standard suffix."""
    tokens = cell.split("(", 1)[0].split()
    return tokens[0] if tokens else ""


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
    "CT_STATE_FIPS",
    "US_STATES",
    "assemble_us_block_ladder",
    "cbsa_delineated_states",
    "ct_planning_region_by_block",
    "parse_baf_county_subdivision_file",
    "parse_baf_district_file",
    "parse_baf_place_file",
    "parse_cbsa_delineations",
    "parse_ct_planning_region_crosswalk",
    "parse_national_cd_bef",
    "parse_pl_geo_blocks",
    "us_block_ladder_cbsa_coverage_failures",
]
