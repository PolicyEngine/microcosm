"""Build the compact 2020 CT tract → 2023 county map from cached Census sources.

Reuses the BAF/crosswalk parsers and source URLs used by the block ladder's
Connecticut CBSA join (microcosm#1072). This reads CT source files only and
never builds or publishes a ladder. For example::

    uv run python tools/build_us_county_vintage_crosswalk.py \
        --cache-dir ~/.cache/populace-us-geography \
        --out-dir packages/microcosm-build/src/microcosm/build/us_runtime/data
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from build_us_block_ladder_artifact import (
    BAF2020_URL_TEMPLATE,
    CT_PLANNING_REGION_CROSSWALK_URL,
    PL94171_URL_TEMPLATE,
)

from microcosm.build.us_runtime.block_ladder_sources import (
    ct_planning_region_by_block,
    parse_baf_county_subdivision_file,
    parse_ct_planning_region_crosswalk,
    parse_pl_geo_blocks,
)
from microcosm.build.us_runtime.county_vintage import (
    COUNTY_FIPS_2023_COLUMN,
    COUNTY_VINTAGE_PROVENANCE_RESOURCE,
    CT_TOWN_COUNTY_CROSSWALK_RESOURCE,
    CT_TRACT_COUNTY_CROSSWALK_RESOURCE,
)


def build_crosswalk(cache_dir: Path, out_dir: Path) -> dict:
    """Derive and write the crosswalk, source copy and provenance from CT inputs."""

    baf_path = cache_dir / "BlockAssign_ST09_CT.zip"
    town_path = cache_dir / CT_TOWN_COUNTY_CROSSWALK_RESOURCE
    pl_path = cache_dir / "ct2020.pl.zip"
    baf_member = "BlockAssign_ST09_CT_MCD.txt"
    with zipfile.ZipFile(baf_path) as archive:
        baf_bytes = archive.read(baf_member)
    town_bytes = town_path.read_bytes()
    towns = parse_ct_planning_region_crosswalk(
        town_bytes.decode("utf-8-sig").splitlines(keepends=True)
    )
    cousub_by_block = parse_baf_county_subdivision_file(
        baf_bytes.decode("latin-1").splitlines(), label=baf_member
    )
    with zipfile.ZipFile(pl_path) as archive:
        population = parse_pl_geo_blocks(
            archive.read("ctgeo2020.pl").decode("latin-1").splitlines(),
            state_fips="09",
        )
    # The maintained join refuses a populated block without a defined town.
    populated_regions = ct_planning_region_by_block(
        population,
        cousub_by_block=cousub_by_block,
        planning_region_by_cousub=towns,
    )
    defined_blocks = {
        block: town for block, town in cousub_by_block.items() if town in towns
    }
    regions = ct_planning_region_by_block(
        defined_blocks,
        cousub_by_block=defined_blocks,
        planning_region_by_cousub=towns,
    )
    by_tract: dict[str, set[str]] = defaultdict(set)
    for block, region in regions.items():
        by_tract[f"{block:015d}"[:11]].add(region)
    ambiguous = {
        tract: sorted(values) for tract, values in by_tract.items() if len(values) != 1
    }
    if ambiguous:
        raise ValueError(
            f"Connecticut tracts span multiple planning regions: {ambiguous}"
        )
    old_totals: dict[str, int] = defaultdict(int)
    new_totals: dict[str, int] = defaultdict(int)
    tract_population: dict[str, int] = defaultdict(int)
    for block, count in population.items():
        old_totals[f"{block:015d}"[:5]] += count
        new_totals[populated_regions[block]] += count
        tract_population[f"{block:015d}"[:11]] += count
    if sum(old_totals.values()) != sum(new_totals.values()):
        raise ValueError(
            "Connecticut county translation failed population conservation."
        )
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["tract_geoid", COUNTY_FIPS_2023_COLUMN, "population_2020"])
    writer.writerows(
        (tract, next(iter(values)), tract_population[tract])
        for tract, values in sorted(by_tract.items())
    )
    payload = output.getvalue().encode("utf-8")
    provenance = {
        "schema_version": 1,
        "artifact": CT_TRACT_COUNTY_CROSSWALK_RESOURCE,
        "output_sha256": hashlib.sha256(payload).hexdigest(),
        "source_vintage": "2020_tabulation_blocks",
        "target_vintage": "2023_county_equivalents",
        "method": (
            "Reuse microcosm#1072's exact 2020 BAF MCD block-to-town join to the "
            "Census CT 2022 planning-region crosswalk. Assert every town-defined "
            "2020 tract belongs to exactly one region, then retain that compact "
            "tract mapping. The 2023 county-equivalents use those nine regions; "
            "non-Connecticut blocks retain their original county prefix."
        ),
        "source_files": {
            "baf2020_ct": {
                "url": BAF2020_URL_TEMPLATE.format(fips="09", usps="CT"),
                "sha256": hashlib.sha256(baf_path.read_bytes()).hexdigest(),
                "member": baf_member,
                "member_sha256": hashlib.sha256(baf_bytes).hexdigest(),
            },
            "ct_planning_region_crosswalk": {
                "url": CT_PLANNING_REGION_CROSSWALK_URL,
                "sha256": hashlib.sha256(town_bytes).hexdigest(),
            },
            "pl94171_ct": {
                "url": PL94171_URL_TEMPLATE.format(
                    dirname="Connecticut", usps_lower="ct"
                ),
                "sha256": hashlib.sha256(pl_path.read_bytes()).hexdigest(),
            },
        },
        "coverage": {
            "towns": len(towns),
            "tracts": len(by_tract),
            "planning_regions": sorted(set(towns.values())),
            "town_defined_blocks": len(regions),
            "populated_blocks": len(population),
            "ambiguous_tracts": len(ambiguous),
        },
        "population_conservation": {
            "source_population": sum(old_totals.values()),
            "target_population": sum(new_totals.values()),
            "source_county_population": dict(sorted(old_totals.items())),
            "target_county_population": dict(sorted(new_totals.items())),
        },
        "excluded_zero_population_blocks": [
            f"{block:015d}"
            for block in sorted(set(cousub_by_block) - set(defined_blocks))
        ],
        "excluded_blocks_reason": (
            "County subdivisions not defined (coastal water). No populated "
            "2020 block has this town code; the block mapping refuses these "
            "blocks even if their tract contains town-defined blocks."
        ),
        "generator": "tools/build_us_county_vintage_crosswalk.py",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / CT_TRACT_COUNTY_CROSSWALK_RESOURCE).write_bytes(payload)
    (out_dir / CT_TOWN_COUNTY_CROSSWALK_RESOURCE).write_bytes(town_bytes)
    (out_dir / COUNTY_VINTAGE_PROVENANCE_RESOURCE).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    provenance = build_crosswalk(args.cache_dir, args.out_dir)
    print(json.dumps(provenance["coverage"], sort_keys=True))


if __name__ == "__main__":
    main()
