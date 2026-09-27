"""Build the US block -> congressional-district plan registry from Census sources.

Downloads (with a local cache) the 2020 P.L. 94-171 geographic headers (the
populated-block universe and its populations), the 2020 Block Assignment
Files' ``CD`` layer (117th plan), and the Census 118th, 119th and 120th
Congressional District block equivalency files. It parses each plan onto
2020 blocks, applies any registered state substitutions, and writes one
national NPZ whose metadata records every source file's URL and SHA-256.

Before writing, the build cross-checks its sources against each other:

- the apportionment constant in ``cd_plan_registry`` against the Census
  apportionment table;
- each Census BEF's state files against that state's rows in its national
  file;
- the states whose blocks change between consecutive plans against the
  states Census ships a state file for (its list of redrawn states);
- each Census block-split note's tabulated district against the registry.

The artifact is then loaded back through ``load_us_cd_plan_registry`` (which
re-checks every invariant) and, with ``--block-ladder``, compared block by
block with the ladder's own 119th-Congress districts and populations.

The method, sources and invariants are documented in
``packages/microcosm-build/src/microcosm/build/us_runtime/US_CD_PLAN_REGISTRY.md``.

Example:
    uv run python tools/build_us_cd_plan_registry_artifact.py \\
        --out build/us/us_cd_plan_registry_2020.npz \\
        --block-ladder build/us/us_block_ladder_2020.npz \\
        --provenance-json packages/microcosm-build/src/microcosm/build/\\
us_runtime/us_cd_plan_registry.provenance.json

    # Smoke run over a few states (no ladder check, not publishable):
    uv run python tools/build_us_cd_plan_registry_artifact.py \\
        --out /tmp/cd_plans_smoke.npz --states 10,29,37
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from microcosm.build.us_runtime.block_ladder_sources import (
    US_STATES,
    parse_pl_geo_blocks,
)
from microcosm.build.us_runtime.cd_plan_registry import (
    PRIMARY_CD_PLAN,
    US_HOUSE_APPORTIONMENT,
    US_HOUSE_APPORTIONMENT_SOURCE,
    CdBlockAssignment,
    assemble_us_cd_plan_registry,
    assignment_agreement,
    check_cd_plan_registry_against_block_ladder,
    concat_cd_block_assignments,
    crosswalk_plan_to_2020_blocks,
    fill_unassigned_blocks,
    load_us_cd_plan_registry,
    parse_block_relationship_2010_2020,
    parse_cd_block_assignment,
    replace_state_assignments,
    summarize_cd_plan_registry,
)

ARTIFACT_NAME = "us_cd_plan_registry_2020.npz"

PL94171_URL_TEMPLATE = (
    "https://www2.census.gov/programs-surveys/decennial/2020/data/"
    "01-Redistricting_File--PL_94-171/{dirname}/{usps_lower}2020.pl.zip"
)
BAF2020_URL_TEMPLATE = (
    "https://www2.census.gov/geo/docs/maps-data/data/baf2020/"
    "BlockAssign_ST{fips}_{usps}.zip"
)
BAF2020_CD_MEMBER_TEMPLATE = "BlockAssign_ST{fips}_{usps}_CD.txt"
_MAPPING_FILES = "https://www2.census.gov/programs-surveys/decennial/rdo/mapping-files"

#: One entry per Census national BEF: the ZIP, its national member, the
#: pattern naming its state files (group 1 = state FIPS) and its split note.
CENSUS_BEFS: dict[str, dict[str, str]] = {
    "118th_congress": {
        "url": f"{_MAPPING_FILES}/2023/118-congressional-district-bef/cd118.zip",
        "member": "National_CD118.txt",
        "state_member_pattern": r"^(\d{2})_[A-Z]{2}_CD118\.txt$",
        "block_split_note": (
            f"{_MAPPING_FILES}/2023/118-congressional-district-bef/CD118_BlockSplits.pdf"
        ),
        "published": "2022-12-16",
        "source": "Census 118th Congressional District block equivalency file",
    },
    "119th_congress": {
        "url": f"{_MAPPING_FILES}/2025/119-congressional-district-befs/cd119.zip",
        "member": "NationalCD119.txt",
        "state_member_pattern": r"^(\d{2})_[A-Z]{2}_CD119\.txt$",
        "block_split_note": (
            f"{_MAPPING_FILES}/2025/119-congressional-district-befs/CD119_BlockSplits.pdf"
        ),
        "published": "2024-07-29",
        "source": "Census 119th Congressional District block equivalency file",
    },
    "120th_congress": {
        "url": f"{_MAPPING_FILES}/2027/120-congressional-district-befs/cd120.zip",
        "member": "NationalCD120.txt",
        "state_member_pattern": r"^CD120_(\d{2})\.txt$",
        "block_split_note": (
            f"{_MAPPING_FILES}/2027/120-congressional-district-befs/CD120_BlockSplits.pdf"
        ),
        "published": "2026-08-31",
        "source": "Census 120th Congressional District block equivalency file",
    },
}

#: Blocks the Census split notes list as divided by a district line. For
#: tabulation Census assigns each to one district "specified to the U.S.
#: Census Bureau by the state"; the build checks the registry agrees. The
#: 118th, 119th and 120th notes each list only this block.
CENSUS_BLOCK_SPLITS: dict[str, dict[str, dict[str, Any]]] = {
    plan: {
        "080010096072000": {
            "districts_within_block": ["07", "08"],
            "tabulated_with": "08",
        }
    }
    for plan in CENSUS_BEFS
}

PLAN_APPORTIONMENT = {
    "117th_congress": "2010_census",
    "118th_congress": "2020_census",
    "119th_congress": "2020_census",
    "120th_congress": "2020_census",
}

PLAN_NOTES = {
    "117th_congress": (
        "Districts of the 117th Congress (2021-2023), the geography of the IRS "
        "SOI congressional-district tables for tax years 2020-2022. Every state "
        "but North Carolina comes from the 2020 Block Assignment Files' CD "
        "layer, which carries the 116th-Congress plans on 2020 blocks; Census "
        "reports North Carolina as the only state whose districts changed "
        "between the 116th and 117th Congress (see known_deviations)."
    ),
    "118th_congress": (
        "Districts of the 118th Congress (2023-2025), the first plans drawn "
        "on 2020 census geography."
    ),
    "119th_congress": (
        "Districts of the 119th Congress (2025-2027); the block ladder's primary plan."
    ),
    "120th_congress": (
        "Districts for the November 3, 2026 election (120th Congress, "
        "2027-2029), including the 2025-26 mid-decade redraws: Census's 120th "
        "Congressional District block equivalency file, except any state whose "
        "map in force differs from Census's rows (see known_deviations)."
    ),
}

#: North Carolina is the one state whose 117th-Congress districts differ from
#: its 116th (Census, "Congressional Districts of the 117th Congress"): the
#: 2020 election used the 2019 remedial plan (HB 1029, S.L. 2019-249), which
#: Census never tabulated on 2020 blocks. The build carries the General
#: Assembly's block file for that plan (2010 blocks) onto 2020 blocks through
#: the Census 2010-2020 block relationship file, checks the method by running
#: it on the 2016 plan against Census's own 2020-block version of that plan,
#: and checks the result against the General Assembly's official 2020-census
#: district populations for the 2019 plan.
NC_FIPS = "37"
NC_117TH_SOURCES = {
    "ncga_2019_plan": {
        "url": (
            "https://www.ncleg.gov/Files/GIS/Plans_Main/Congress_2019/"
            "HB1029%203rd%20Edition%20-%20Blockfile.zip"
        ),
        "member": "C-Goodwin-A-1-TC.csv",
    },
    "ncga_2016_plan": {
        "url": "https://www.ncleg.gov/Files/GIS/Plans_Main/Congress_2016/baf.zip",
        "member": "2016 Contingent Congressional Plan - Corrected.csv",
    },
    "census_block_relationship_2010_2020_nc": {
        "url": (
            "https://www2.census.gov/geo/docs/maps-data/data/rel2020/t10t20/"
            "TAB2010_TAB2020_ST37.zip"
        ),
        "member": "tab2010_tab2020_st37_nc.txt",
    },
    "ncga_2020_district_populations_2019_plan": {
        "url": (
            "https://www.ncleg.gov/Files/GIS/Maps_Reports/Decennial_ReCalc/2020/"
            "DeviationReports/2020_Deviation_Rpt_Cong.xlsx"
        ),
    },
}
#: The method run on the 2016 plan must reproduce Census's 2020-block version
#: of that plan (the BAF CD layer) for at least this share of population.
NC_117TH_MIN_METHOD_AGREEMENT = 0.999
#: Each 2019-plan district's population must be within this share of the
#: General Assembly's official 2020 figure.
NC_117TH_MAX_DISTRICT_GAP = 0.005

#: Registered substitutions: a state's blocks in ``plan`` taken from another
#: registered plan instead of the plan's own source, with the reason and the
#: primary evidence. Each one is recorded in the plan's ``known_deviations``.
STATE_SUBSTITUTIONS: dict[str, dict[str, dict[str, Any]]] = {
    "120th_congress": {
        "29": {
            "from_plan": "119th_congress",
            "note": (
                "Census's 120th rows for Missouri encode the 2025 HB 1 map, and "
                "Census's own 120th landing page warns they may not be the "
                "districts in place for November 2026. The Missouri Supreme Court "
                "held on 2026-09-03 (von Glahn v. Hoskins, SC101805) that HB 1 "
                "cannot take effect before a referendum on the November 2026 "
                "ballot, and the U.S. Supreme Court held on 2026-09-25 (People "
                "Not Politicians v. Onder, 26A388, per curiam) that 'the 2022 "
                "map\u2014not the 2025 map\u2014must be used in the 2026 congressional "
                "election.' Missouri's rows here are its 2022 map, taken from the "
                "119th file (the build checks they are unchanged from the 118th)."
            ),
            "evidence": [
                "https://www.census.gov/geographies/mapping-files/2027/dec/rdo/"
                "120-congressional-district-bef.html",
                "https://www.supremecourt.gov/opinions/25pdf/26a388_q86b.pdf",
                "https://statecourtreport.org/case-tracker/von-glahn-v-hoskins",
            ],
            "as_of": "2026-09-27",
        }
    }
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the US block -> congressional-district plan registry."
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "populace-us-geography",
        help="Download cache; re-runs reuse cached files.",
    )
    parser.add_argument(
        "--states",
        help="Comma-separated state FIPS subset (smoke runs; not publishable).",
    )
    parser.add_argument(
        "--block-ladder",
        type=Path,
        help=(
            "Block ladder NPZ to compare against: same blocks, same "
            "populations, identical 119th-Congress districts."
        ),
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        help="Where to write the build receipt. Defaults beside --out.",
    )
    return parser.parse_args(argv)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / url.rsplit("/", 1)[-1]
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    _log(f"  downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "microcosm-build (cd plan registry)"}
    )
    with urllib.request.urlopen(request) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"Empty download from {url}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(destination)
    return destination


@contextmanager
def _zip_member_stream(
    archive_path: Path, member: str, *, encoding: str = "latin-1"
) -> Iterator[io.TextIOWrapper]:
    with zipfile.ZipFile(archive_path) as archive, archive.open(member) as raw:
        yield io.TextIOWrapper(raw, encoding=encoding)


def _source_entry(url: str, path: Path, member: str | None = None) -> dict[str, str]:
    entry = {"url": url, "sha256": _sha256(path)}
    if member is not None:
        entry["member"] = member
    return entry


def _check_apportionment(cache_dir: Path) -> None:
    """The packaged seat counts must equal the Census apportionment table."""

    path = _download(US_HOUSE_APPORTIONMENT_SOURCE["url"], cache_dir)
    digest = _sha256(path)
    if digest != US_HOUSE_APPORTIONMENT_SOURCE["sha256"]:
        raise SystemExit(
            f"Census apportionment table changed: sha256 {digest}, pinned "
            f"{US_HOUSE_APPORTIONMENT_SOURCE['sha256']}. Re-verify the constant."
        )
    name_to_fips = {dirname.replace("_", " "): fips for fips, _, dirname in US_STATES}
    table: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            year = row["Year"]
            if row["Geography Type"] != "State" or row["Name"] not in name_to_fips:
                continue
            seats = row["Number of Representatives"].strip()
            table.setdefault(f"{year}_census", {})[name_to_fips[row["Name"]]] = (
                int(seats) if seats else 0
            )
    for census, seats in US_HOUSE_APPORTIONMENT.items():
        if table.get(census) != dict(seats):
            raise SystemExit(
                f"US_HOUSE_APPORTIONMENT[{census!r}] disagrees with the Census table."
            )
    _log("  apportionment constant matches the Census table")


def _universe(
    states: list[tuple[str, str, str]], cache_dir: Path, sources: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    blocks: list[np.ndarray] = []
    populations: list[np.ndarray] = []
    for fips, usps, dirname in states:
        url = PL94171_URL_TEMPLATE.format(dirname=dirname, usps_lower=usps.lower())
        path = _download(url, cache_dir)
        member = f"{usps.lower()}geo2020.pl"
        sources[f"pl94171_{usps.lower()}"] = _source_entry(url, path, member)
        with _zip_member_stream(path, member) as stream:
            state_blocks = parse_pl_geo_blocks(stream, state_fips=fips)
        keys = np.fromiter(state_blocks.keys(), dtype=np.int64, count=len(state_blocks))
        values = np.fromiter(
            state_blocks.values(), dtype=np.int64, count=len(state_blocks)
        )
        blocks.append(keys)
        populations.append(values)
        _log(f"  {usps}: {len(keys):,} populated blocks, {int(values.sum()):,} people")
    return np.concatenate(blocks), np.concatenate(populations)


def _plan_117th(
    states: list[tuple[str, str, str]], cache_dir: Path
) -> tuple[CdBlockAssignment, dict[str, Any]]:
    parts: list[CdBlockAssignment] = []
    source_files: dict[str, Any] = {}
    for fips, usps, _ in states:
        url = BAF2020_URL_TEMPLATE.format(fips=fips, usps=usps)
        path = _download(url, cache_dir)
        member = BAF2020_CD_MEMBER_TEMPLATE.format(fips=fips, usps=usps)
        source_files[f"baf2020_{usps.lower()}"] = _source_entry(url, path, member)
        with _zip_member_stream(path, member) as stream:
            parts.append(parse_cd_block_assignment(stream, label=member))
    assignment = concat_cd_block_assignments(parts, label="2020 BAF CD layer")
    spec = {
        "source": "Census 2020 Block Assignment Files, CD layer",
        "url": BAF2020_URL_TEMPLATE,
        "source_files": source_files,
    }
    return assignment, spec


def _ncga_2020_district_populations(path: Path) -> dict[int, int]:
    """District -> 2020 population from the General Assembly's deviation report."""

    import openpyxl  # dev dependency, as for the block ladder's CBSA workbook

    workbook = openpyxl.load_workbook(path, read_only=True)
    populations: dict[int, int] = {}
    for row in workbook.worksheets[0].iter_rows(values_only=True):
        if row and isinstance(row[0], int) and isinstance(row[2], int):
            populations[row[0]] = row[2]
    if sorted(populations) != list(range(1, 14)):
        raise SystemExit(
            f"NCGA 2020 deviation report: expected districts 1-13, got {sorted(populations)}."
        )
    return populations


def _nc_117th(
    census_116th: CdBlockAssignment,
    block_geoid: np.ndarray,
    population: np.ndarray,
    cache_dir: Path,
) -> tuple[CdBlockAssignment, dict[str, Any], dict[str, Any]]:
    """North Carolina's 2019 plan on 2020 blocks, its deviation record and sources."""

    paths = {
        key: _download(spec["url"], cache_dir) for key, spec in NC_117TH_SOURCES.items()
    }
    source_files = {
        key: _source_entry(spec["url"], paths[key], spec.get("member"))
        for key, spec in NC_117TH_SOURCES.items()
    }
    in_state = block_geoid // 10**13 == int(NC_FIPS)
    nc_blocks, nc_population = block_geoid[in_state], population[in_state]

    relationship_spec = NC_117TH_SOURCES["census_block_relationship_2010_2020_nc"]
    with _zip_member_stream(
        paths["census_block_relationship_2010_2020_nc"],
        relationship_spec["member"],
        encoding="utf-8-sig",
    ) as stream:
        relationship = parse_block_relationship_2010_2020(
            stream, label=relationship_spec["member"]
        )

    def carry(key: str) -> tuple[CdBlockAssignment, dict[str, Any]]:
        spec = NC_117TH_SOURCES[key]
        with _zip_member_stream(
            paths[key], spec["member"], encoding="utf-8-sig"
        ) as stream:
            plan_2010 = parse_cd_block_assignment(stream, label=spec["member"])
        carried, diagnostics = crosswalk_plan_to_2020_blocks(
            plan_2010, relationship, state_fips=NC_FIPS
        )
        filled, fills = fill_unassigned_blocks(
            carried,
            diagnostics["unassigned"],
            block_geoid=nc_blocks,
            population=nc_population,
        )
        straddling = np.isin(nc_blocks, diagnostics["straddling"])
        return filled, {
            "straddling_blocks": int(len(diagnostics["straddling"])),
            "straddling_populated_blocks": int(straddling.sum()),
            "straddling_population": int(nc_population[straddling].sum()),
            "filled_blocks": fills,
        }

    plan_2019, diagnostics_2019 = carry("ncga_2019_plan")
    plan_2016, _ = carry("ncga_2016_plan")
    census_nc = census_116th.for_state(NC_FIPS)

    method_check = assignment_agreement(
        plan_2016, census_nc, block_geoid=nc_blocks, population=nc_population
    )
    if method_check["population_share_agreeing"] < NC_117TH_MIN_METHOD_AGREEMENT:
        raise SystemExit(
            "NC 117th: the crosswalk method reproduces Census's 2020-block 2016 "
            f"plan for only {method_check['population_share_agreeing']:.5f} of "
            f"the population (minimum {NC_117TH_MIN_METHOD_AGREEMENT})."
        )

    official = _ncga_2020_district_populations(
        paths["ncga_2020_district_populations_2019_plan"]
    )
    index = np.searchsorted(plan_2019.block_geoid, nc_blocks)
    districts = plan_2019.district_geoid[index] % 100
    if not np.array_equal(plan_2019.block_geoid[index], nc_blocks):
        raise SystemExit("NC 117th: the 2019 plan leaves populated blocks unassigned.")
    ours = {
        int(d): int(nc_population[districts == d].sum()) for d in np.unique(districts)
    }
    gaps = {f"{d:02d}": ours.get(d, 0) - official[d] for d in sorted(official)}
    worst = max(abs(gap) / official[int(d)] for d, gap in gaps.items())
    if worst > NC_117TH_MAX_DISTRICT_GAP:
        raise SystemExit(
            f"NC 117th: district populations differ from the General Assembly's "
            f"2020 figures by up to {worst:.4%}: {gaps}."
        )

    deviation = {
        "note": (
            "North Carolina is the one state whose 117th districts differ from "
            "its 116th. Its 117th districts are the 2019 remedial plan (HB 1029 "
            "3rd edition, S.L. 2019-249, 'C-Goodwin-A-1-TC'), which Census never "
            "tabulated on 2020 blocks, so this state's rows are not from the "
            "2020 BAF CD layer (the 2016 plan). The General Assembly's block "
            "file for the 2019 plan (2010 blocks) is carried onto 2020 blocks "
            "through the Census 2010-2020 tabulation block relationship file."
        ),
        "method": (
            "Each 2020 block takes the 2019-plan district whose 2010 blocks cover "
            "the most of its land area (then water area, then the lower district "
            "code). A 2020 block with no North Carolina 2010 counterpart takes the "
            "district holding most of its block group's population (else its "
            "tract's, else its county's)."
        ),
        **diagnostics_2019,
        "method_check": {
            "description": (
                "The same method run on the 2016 plan, compared with Census's own "
                "2020-block version of that plan (the 2020 BAF CD layer), over "
                "populated blocks."
            ),
            "minimum_population_share": NC_117TH_MIN_METHOD_AGREEMENT,
            **method_check,
        },
        "official_district_population_check": {
            "description": (
                "District populations of the carried 2019 plan minus the General "
                "Assembly's official 2020-census populations for that plan."
            ),
            "maximum_share": NC_117TH_MAX_DISTRICT_GAP,
            "differences": gaps,
            "largest_share": round(worst, 6),
        },
        "agreement_with_the_116th_plan": assignment_agreement(
            plan_2019, census_nc, block_geoid=nc_blocks, population=nc_population
        ),
        "source_files": source_files,
    }
    return plan_2019, deviation, source_files


def _plan_from_census_bef(
    plan: str,
    cache_dir: Path,
    selected: set[str],
) -> tuple[CdBlockAssignment, dict[str, Any], set[str]]:
    """Parse a Census national BEF and check its state files against it.

    Returns the assignment restricted to ``selected`` states, the plan's
    source spec, and the states Census shipped a state file for.
    """

    bef = CENSUS_BEFS[plan]
    path = _download(bef["url"], cache_dir)
    split_note = _download(bef["block_split_note"], cache_dir)
    with _zip_member_stream(path, bef["member"]) as stream:
        national = parse_cd_block_assignment(stream, label=bef["member"])
    national = concat_cd_block_assignments(
        [national.for_state(state) for state in sorted(selected & national.states())],
        label=bef["member"],
    )
    pattern = re.compile(bef["state_member_pattern"])
    with zipfile.ZipFile(path) as archive:
        state_members = {
            match.group(1): name
            for name in archive.namelist()
            if (match := pattern.match(name))
        }
    for state in sorted(set(state_members) & selected):
        with _zip_member_stream(path, state_members[state]) as stream:
            state_file = parse_cd_block_assignment(stream, label=state_members[state])
        national_rows = national.for_state(state)
        if not (
            np.array_equal(state_file.block_geoid, national_rows.block_geoid)
            and np.array_equal(state_file.district_geoid, national_rows.district_geoid)
        ):
            raise SystemExit(
                f"{plan}: state file {state_members[state]} disagrees with "
                f"{bef['member']} for state {state}."
            )
    spec = {
        "source": f"{bef['source']} ({bef['member']})",
        "url": bef["url"],
        "published": bef["published"],
        "source_files": {
            plan: _source_entry(bef["url"], path, bef["member"]),
            f"{plan}_block_split_note": _source_entry(
                bef["block_split_note"], split_note
            ),
        },
        "census_state_files": sorted(state_members),
        "block_splits": CENSUS_BLOCK_SPLITS[plan],
    }
    return national, spec, set(state_members)


def _changed_states(before: CdBlockAssignment, after: CdBlockAssignment) -> set[str]:
    """States with any block whose district differs between two plans."""

    common, before_index, after_index = np.intersect1d(
        before.block_geoid, after.block_geoid, assume_unique=True, return_indices=True
    )
    differs = before.district_geoid[before_index] != after.district_geoid[after_index]
    return {f"{state:02d}" for state in np.unique(common[differs] // 10**13).tolist()}


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    selected_fips = (
        {value.strip() for value in args.states.split(",")} if args.states else None
    )
    if selected_fips is not None:
        unknown = selected_fips - {fips for fips, _, _ in US_STATES}
        if unknown:
            raise SystemExit(f"Unknown state FIPS in --states: {sorted(unknown)}")
    states = [s for s in US_STATES if selected_fips is None or s[0] in selected_fips]
    selected = {fips for fips, _, _ in states}
    _log(f"Building the US CD plan registry for {len(states)} state(s)")

    _check_apportionment(args.cache_dir)

    population_sources: dict[str, Any] = {}
    block_geoid, population = _universe(states, args.cache_dir, population_sources)
    _log(f"  universe: {len(block_geoid):,} blocks, {int(population.sum()):,} people")

    assignments: dict[str, CdBlockAssignment] = {}
    specs: dict[str, dict[str, Any]] = {}
    assignments["117th_congress"], specs["117th_congress"] = _plan_117th(
        states, args.cache_dir
    )
    if NC_FIPS in selected:
        _log("  carrying North Carolina's 2019 plan onto 2020 blocks")
        nc_plan, nc_deviation, nc_sources = _nc_117th(
            assignments["117th_congress"], block_geoid, population, args.cache_dir
        )
        assignments["117th_congress"] = replace_state_assignments(
            assignments["117th_congress"], nc_plan, state_fips=NC_FIPS
        )
        specs["117th_congress"].setdefault("known_deviations", {})[NC_FIPS] = (
            nc_deviation
        )
        specs["117th_congress"]["source_files"].update(nc_sources)
        _log(
            "  NC 117th: method check "
            f"{nc_deviation['method_check']['population_share_agreeing']:.5f}, "
            "largest district gap "
            f"{nc_deviation['official_district_population_check']['largest_share']:.4%}"
        )
    census_state_files: dict[str, set[str]] = {}
    for plan in CENSUS_BEFS:
        _log(f"  parsing {plan}")
        assignments[plan], specs[plan], census_state_files[plan] = (
            _plan_from_census_bef(plan, args.cache_dir, selected)
        )

    # Census ships a state file exactly for the states it lists as redrawn;
    # the blocks must agree with that list.
    for previous, plan in (
        ("118th_congress", "119th_congress"),
        ("119th_congress", "120th_congress"),
    ):
        changed = _changed_states(assignments[previous], assignments[plan])
        expected = census_state_files[plan] & selected
        if changed != expected:
            raise SystemExit(
                f"{plan}: states changed from {previous} are {sorted(changed)}, "
                f"but Census ships state files for {sorted(expected)}."
            )

    for plan, substitutions in STATE_SUBSTITUTIONS.items():
        deviations = specs[plan].setdefault("known_deviations", {})
        for state, substitution in sorted(substitutions.items()):
            if state not in selected:
                continue
            donor = assignments[substitution["from_plan"]].for_state(state)
            assignments[plan] = replace_state_assignments(
                assignments[plan], donor, state_fips=state
            )
            deviations[state] = substitution
            _log(f"  {plan}: state {state} taken from {substitution['from_plan']}")

    plan_sources = {
        plan: {
            "vintage": plan,
            "apportionment": PLAN_APPORTIONMENT[plan],
            "notes": PLAN_NOTES[plan],
            **specs[plan],
        }
        for plan in sorted(assignments)
    }
    metadata = {
        "universe": (
            "2020 tabulation blocks with P.L. 94-171 POP100 > 0 in the 50 states and DC"
        ),
        "states": sorted(selected),
        "population_source": {
            "source": "2020 Census P.L. 94-171 geographic headers, POP100",
            "url": PL94171_URL_TEMPLATE,
            "source_files": population_sources,
        },
        "apportionment_source": dict(US_HOUSE_APPORTIONMENT_SOURCE),
    }
    payload = assemble_us_cd_plan_registry(
        block_geoid=block_geoid,
        population=population,
        plan_assignments=assignments,
        plan_sources=plan_sources,
        metadata=metadata,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)

    registry = load_us_cd_plan_registry(args.out)  # self-check: re-validates
    for plan in CENSUS_BEFS:
        for geoid, split in CENSUS_BLOCK_SPLITS[plan].items():
            block = int(geoid)
            if f"{block // 10**13:02d}" not in selected:
                continue
            index = int(np.searchsorted(registry.block_geoid, block))
            if index >= len(registry) or registry.block_geoid[index] != block:
                raise SystemExit(f"{plan}: split block {geoid} is not in the universe.")
            district = int(registry.plans[plan][index]) % 100
            if f"{district:02d}" != split["tabulated_with"]:
                raise SystemExit(
                    f"{plan}: split block {geoid} is in district {district:02d}, "
                    f"Census tabulates it with {split['tabulated_with']}."
                )

    summary = summarize_cd_plan_registry(registry)
    ladder_check = None
    if args.block_ladder is not None:
        with np.load(args.block_ladder, allow_pickle=False) as ladder:
            ladder_check = check_cd_plan_registry_against_block_ladder(
                registry,
                block_geoid=ladder["block_geoid"],
                population=ladder["population"],
                congressional_district_geoid=ladder["congressional_district_geoid"],
                plan=PRIMARY_CD_PLAN,
            )
        ladder_check["block_ladder_sha256"] = _sha256(args.block_ladder)
        _log(f"  block ladder agrees exactly ({ladder_check['blocks']:,} blocks)")

    receipt = {
        "artifact": ARTIFACT_NAME,
        "kind": registry.metadata["kind"],
        "schema_version": registry.metadata["schema_version"],
        "block_vintage": registry.metadata["block_vintage"],
        "output_sha256": registry.sha256,
        **summary,
        "block_ladder_check": ladder_check,
        "plan_sources": {
            plan: {
                key: value
                for key, value in spec.items()
                if key not in {"districts", "blocks_outside_universe"}
            }
            for plan, spec in registry.plan_sources.items()
        },
        "population_source": metadata["population_source"],
        "apportionment_source": metadata["apportionment_source"],
    }
    receipt_path = (
        args.provenance_json
        if args.provenance_json is not None
        else args.out.with_suffix(".provenance.json")
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    _log(f"  wrote {args.out} (sha256 {registry.sha256}) and {receipt_path}")
    print(
        json.dumps(
            {k: receipt[k] for k in ("output_sha256", "blocks", "population")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
