#!/usr/bin/env python3
"""Build the three UK conditioned atomic-area supports (microcosm#931).

Each extends the registered support of its system (``sources.yaml`` role
``uk_atomic_area_support``, built by ``build_uk_atomic_area_supports.py``)
with the columns the geography-first draw conditions on. Only that move binds
them; today's build binds the registered supports.

- ``imd_decile``: the area's deprivation decile at the 2019 vintage of each
  nation's index (IoD 2019, WIMD 2019, SIMD 2020v2, NIMDM 2017), the vintage
  the FRS 2024-25 deciles follow. The May 2026 NSPL, the last edition to carry
  those ranks, gives every live residential (small-user) postcode its small
  area and its rank. An area takes the decile most of its postcodes carry; a
  tie goes to the decile its parent area's postcodes favour, and an area
  without such a postcode takes its parent's (LSOA, data zone, super data
  zone). Deciles are the publishers' own; NISRA publishes none, so Northern
  Ireland's are its 890 SOA ranks in tenths of 89.
- ``households_band_a`` ... : the area's households by council-tax band, times
  10^6. England and Wales use the VOA CTSOP1.1 mix of the area's LSOA
  (``voa_council_tax_band_mix_lsoa.json``); Scotland uses the dwellings by
  band of the area's 2022 data zone, in which its 2022 output areas nest, from
  NRS's December 2024 sheet. Northern Ireland has domestic rates and no bands.

The publisher files are pinned here by sha256: chronicle#412's raw-only
registrations (PR chronicle#413), and NRS's dwelling estimates by 2022 data
zone, which Chronicle registers with only their total as facts (chronicle#419
asks for the band columns). The VOA band mix is a committed package resource.
The provenance register records every input's identity, each nation's decile
rule counts, the rounding, and every zero-mass draw cell.

Example (odfpy reads the Welsh Government's ODS; nothing else here needs it):
    uv run --no-sync --with odfpy python tools/build_uk_conditioned_area_supports.py \\
        --supports-dir build/uk/supports \\
        --source-manifest build/uk/conditioned_sources.json \\
        --out-dir build/uk/supports

``--source-manifest`` names exact local files (path/url/sha256/bytes per
source, schema_version 1, as for the registered supports); nothing is
downloaded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.atomic_support_conditioning import (
    apportion_weights,
    modal_area_codes,
    zero_mass_cells,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    BAND_WEIGHT_COLUMNS,
    BAND_WEIGHT_SCALE,
    CONDITIONED_PROVENANCE_RESOURCE,
    CONDITIONED_SOURCES,
    IMD_DECILE_COLUMN,
    IMD_DECILES,
    SYSTEMS,
    assemble_uk_conditioned_area_support,
    uk_conditioned_support_base,
)
from microcosm.build.uk_runtime.atomic_area_support import (
    SOURCES as SUPPORT_SOURCES,
)
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

_TOOLS = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _TOOLS.parent
_UK_PACKAGE = _REPOSITORY_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
_DEFAULT_PROVENANCE = _UK_PACKAGE / CONDITIONED_PROVENANCE_RESOURCE
VOA_BAND_MIX = "voa_council_tax_band_mix_lsoa.json"

PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_KIND = "uk_atomic_area_conditioned_supports_provenance"
EW, SCOT, NI = SYSTEMS

#: chronicle#412's raw-only registrations, as PR chronicle#413 pins them, and
#: NRS's dwellings by 2022 data zone, registered with its total as facts.
SOURCES: dict[str, dict[str, str | int]] = {
    "nspl": {
        "chronicle_source_id": "ons",
        "chronicle_package_id": "ons-nspl-may-2026",
        "chronicle_year": 2026,
        "filename": "NSPL_MAY_2026.zip",
        "url": "https://www.arcgis.com/sharing/rest/content/items/7668e0d35cab4f6db6f15f03be610fb0/data",
        "sha256": "e0ccd5b3d90fdb537c29afc85b33c6c7ff8647e89e390f2e9ae48530f85f42c1",
        "size_bytes": 187_209_017,
        "publisher": "ONS",
        "product": "National Statistics Postcode Lookup, May 2026",
    },
    "iod2019": {
        "chronicle_source_id": "mhclg",
        "chronicle_package_id": "mhclg-english-indices-of-deprivation-2019-file-1",
        "chronicle_year": 2019,
        "filename": "File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx",
        "url": "https://assets.publishing.service.gov.uk/media/5d8b3abded915d0373d3540f/File_1_-_IMD2019_Index_of_Multiple_Deprivation.xlsx",
        "sha256": "b95ca12952c3a8ebdb2e5c3bb03d5b0e56b76ede3fcf45205db26f280b912d9d",
        "size_bytes": 1_433_347,
        "publisher": "MHCLG",
        "product": "English indices of deprivation 2019, File 1: index of multiple deprivation",
    },
    "wimd2019": {
        "chronicle_source_id": "welshgov",
        "chronicle_package_id": "welshgov-wimd-2019-ranks-by-small-area",
        "chronicle_year": 2019,
        "filename": "welsh-index-multiple-deprivation-2019-index-and-domain-ranks-by-small-area.ods",
        "url": "https://www.gov.wales/sites/default/files/statistics-and-research/2022-02/welsh-index-multiple-deprivation-2019-index-and-domain-ranks-by-small-area.ods",
        "sha256": "2e7b65a3a276e31a10d4bf24a7d26e5cd9d2d95a796ab65e894fcd4e4b159430",
        "size_bytes": 232_940,
        "publisher": "Welsh Government",
        "product": "WIMD 2019 index and domain ranks by small area",
    },
    "simd2020v2": {
        "chronicle_source_id": "scotgov",
        "chronicle_package_id": "scotgov-simd-2020v2-data-zone-lookup",
        "chronicle_year": 2020,
        "filename": "SIMD_2020v2_datazone_lookup_updated_2025.xlsx",
        "url": "https://www.gov.scot/binaries/content/documents/govscot/publications/statistics/2020/01/scottish-index-of-multiple-deprivation-2020-data-zone-look-up-file/documents/scottish-index-of-multiple-deprivation-data-zone-look-up/scottish-index-of-multiple-deprivation-data-zone-look-up/govscot%3Adocument/SIMD%2B2020v2%2B-%2Bdatazone%2Blookup%2B-%2Bupdated%2B2025.xlsx",
        "sha256": "0a64937af60fb339a1be85cd2f65b6a3d410a38dc5b6c7f25a72ba2d7060a03e",
        "size_bytes": 1_339_239,
        "publisher": "Scottish Government",
        "product": "SIMD 2020v2 data zone lookup (updated 2025)",
    },
    "nimdm2017": {
        "chronicle_source_id": "nisra",
        "chronicle_package_id": "nisra-nimdm-2017-soa-results",
        "chronicle_year": 2017,
        "filename": "NIMDM17_SOAresults.xls",
        "url": "https://www.nisra.gov.uk/files/nisra/publications/NIMDM17_SOAresults.xls",
        "sha256": "3738edf2d264491301b41cc7b64967373d49b09a6f5719225a0e4f7d789af9d3",
        "size_bytes": 1_772_032,
        "publisher": "NISRA",
        "product": "NIMDM 2017 results by Super Output Area",
    },
    "nrs_dz_dwellings": {
        "chronicle_source_id": "nrs",
        "chronicle_package_id": "nrs-dwellings-data-zone-2025",
        "chronicle_year": 2025,
        "filename": "dwelling-estimates-by-2022-data-zones-24-25.xlsx",
        "url": "https://nrscotland.gov.uk/media/43tfx3zo/dwelling-estimates-by-2022-data-zones-24-25.xlsx",
        "sha256": "5d61db5faf7e81451d838e5d394b8e0682bbf501341c1220aea4a78082ceb033",
        "size_bytes": 1_223_284,
        "publisher": "NRS",
        "product": "Dwelling estimates by 2022 data zone, December 2024 and 2025",
    },
}
#: The Chronicle commit that registers these files: chronicle#413's merge, which
#: also carries the NRS registration. The supports were first built at #413's
#: head; the file digests did not move.
CHRONICLE = {
    "pull_request": "PolicyEngine/chronicle#413",
    "commit": "c51f40d2bdaa66315dd63f334ede5790a8891e27",
    "merged": True,
}
ISSUES = {
    "umbrella": "PolicyEngine/microcosm#931",
    "chronicle_registration": "PolicyEngine/chronicle#412",
    "chronicle_band_facts": "PolicyEngine/chronicle#419",
}
NSPL_MEMBER = "Data/NSPL_MAY_2026_UK.csv"
#: Small areas each index ranks, within its own nation only.
INDEX_AREAS = {"E": 32_844, "W": 1_909, "S": 6_976, "N": 890}
#: The source registrations behind each nation's ranks and deciles.
NATION_INDEX = {"E": "iod2019", "W": "wimd2019", "S": "simd2020v2", "N": "nimdm2017"}
NATION_PREFIXES = {EW: ("E", "W"), SCOT: ("S",), NI: ("N",)}
#: The vintage each system's decile column declares.
DECILE_VINTAGES = {EW: "2019_iod_wimd", SCOT: "2020v2_simd", NI: "2017_nimdm_soa"}
BAND_VINTAGES = {EW: "2025_ctsop", SCOT: "2024_nrs_dz_dwellings"}
#: Each area's parent for ties and areas without a residential postcode.
PARENT_COLUMNS = {EW: "lsoa_code", SCOT: "lsoa_code", NI: "msoa_code"}
#: NRS's December 2024 sheet: nearest the FRS 2024-25 year and VOA's stock at
#: 31 March 2025. Its header is row 5.
NRS_SHEET = "2024"
NRS_HEADER_ROW = 4
NRS_DATA_ZONES = 7_392
NRS_ZONE_COLUMN = "Data Zone code"
NRS_TOTAL_COLUMN = "Total number of dwellings"
NRS_BAND_COLUMNS = {band.lower(): f"Council Tax band:\n{band}" for band in "ABCDEFGH"}

NOTES = (
    "imd_decile is the 2019 vintage of each nation's index (IoD 2019, WIMD 2019, "
    "SIMD 2020v2, NIMDM 2017): the FRS 2024-25 variables IMDE, IMDW, IMDS and IMDN "
    "carry no change note from 2023-24, which predates the 2025 indices, and on an "
    "internal comparison England's FRS decile shares by region fit IoD 2019 better than "
    "IoD 2025, as the 2023-24 FRS does. The vintage is still to be confirmed with DWP.",
    "An area's decile is the one most of its live small-user postcodes carry in the May "
    "2026 NSPL (oa21cd: the 2021 output area in England and Wales, the 2022 output area "
    "in Scotland, the 2021 data zone in Northern Ireland). Ranks are within-nation only.",
    "Northern Ireland's SOAs (2001) do not nest in its 2021 data zones, so many data "
    "zones' postcodes span deciles; the modal decile is recorded with relation "
    "inferred_modal and the counts below.",
    "England and Wales band weights read each area's LSOA mix as committed (shares to "
    "6 dp ending at exactly 1), so households x share x 10^6 is exact. Scotland's read "
    "its 2022 data zone's dwellings by band, rounded half to even per band.",
    "Scotland's band mix is its 2022 data zone's, in which its 2022 output areas nest, "
    "from NRS's December 2024 sheet: nearest the FRS 2024-25 year and VOA's stock at "
    "31 March 2025. NRS counts dwellings on the valuation list, as VOA does in England "
    "and Wales. Chronicle registers the workbook (nrs-dwellings-data-zone-2025) with "
    "only its total as facts; until chronicle#419 adds the band columns, this tool "
    "reads them from the registered bytes.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--supports-dir", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--provenance-json", type=Path, default=_DEFAULT_PROVENANCE)
    parser.add_argument("--resource-dir", type=Path, default=_UK_PACKAGE)
    return parser.parse_args(argv)


def _pinned_sources(path: Path) -> dict[str, Path]:
    """Every source as an exact local file whose bytes match its registration."""
    manifest = json.loads(path.read_bytes())
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Pinned source manifest requires schema_version 1.")
    rows = manifest.get("sources")
    if not isinstance(rows, dict) or set(rows) != set(SOURCES):
        raise ValueError("Pinned source manifest must name exactly the tool's sources.")
    paths = {}
    for name, spec in SOURCES.items():
        row = rows[name]
        if not isinstance(row, dict) or set(row) != {"path", "url", "sha256", "bytes"}:
            raise ValueError(f"Pinned source {name} requires path/url/sha256/bytes.")
        local = Path(row["path"])
        if (
            row["url"] != spec["url"]
            or row["sha256"] != spec["sha256"]
            or row["bytes"] != spec["size_bytes"]
            or not local.is_absolute()
            or local.is_symlink()
            or not local.is_file()
        ):
            raise ValueError(f"Pinned source {name} differs from its registration.")
        if (
            local.stat().st_size != spec["size_bytes"]
            or _sha256(local) != spec["sha256"]
        ):
            raise ValueError(f"Pinned source identity mismatch for {name}.")
        paths[name] = local
    return paths


def _rank_deciles(rank: pd.Series, decile: pd.Series, *, nation: str) -> dict[int, str]:
    """A within-nation rank -> decile map; a tied rank must share one decile."""
    if len(rank) != INDEX_AREAS[nation]:
        raise ValueError(
            f"{nation}: {len(rank):,} ranked areas, expected {INDEX_AREAS[nation]:,}."
        )
    mapping: dict[int, str] = {}
    for r, d in zip(rank.astype(int), decile.astype(int), strict=True):
        code = str(d)
        if code not in IMD_DECILES or mapping.setdefault(int(r), code) != code:
            raise ValueError(f"{nation}: rank {r} has no single decile 1 to 10.")
    return mapping


def _england(path: Path) -> dict[int, str]:
    frame = pd.read_excel(path, sheet_name="IMD2019")
    return _rank_deciles(
        frame["Index of Multiple Deprivation (IMD) Rank"],
        frame["Index of Multiple Deprivation (IMD) Decile"],
        nation="E",
    )


def _wales(path: Path) -> dict[int, str]:
    raw = pd.read_excel(
        path, sheet_name="Deciles_quintiles_quartiles", header=None, engine="odf"
    )
    header = next(
        row for row in range(10) if str(raw.iloc[row, 0]).strip().lower() == "lsoa code"
    )
    frame = raw.iloc[header + 1 :].dropna(how="all")
    frame.columns = [str(column).strip() for column in raw.iloc[header]]
    return _rank_deciles(
        frame["WIMD 2019 overall rank"], frame["WIMD 2019 overall decile"], nation="W"
    )


def _scotland(path: Path) -> dict[int, str]:
    frame = pd.read_excel(path, sheet_name="SIMD 2020v2 DZ lookup data")
    return _rank_deciles(
        frame["SIMD2020v2_Rank"], frame["SIMD2020v2_Decile"], nation="S"
    )


def _northern_ireland() -> dict[int, str]:
    """NISRA publishes ranks only: tenths of the 890 SOAs, 89 each, rank 1 most deprived."""
    areas = INDEX_AREAS["N"]
    return {rank: str((rank - 1) * 10 // areas + 1) for rank in range(1, areas + 1)}


def _postcodes(path: Path) -> pd.DataFrame:
    """Live small-user postcodes with their small area and within-nation rank."""
    with zipfile.ZipFile(path) as archive, archive.open(NSPL_MEMBER) as handle:
        frame = pd.read_csv(
            handle,
            usecols=["doterm", "usrtypind", "oa21cd", "imd20ind"],
            dtype=str,
            keep_default_na=False,
        )
    live = (frame["doterm"] == "") & (frame["usrtypind"] == "0")
    return frame.loc[live, ["oa21cd", "imd20ind"]].reset_index(drop=True)


def _nrs_band_counts(path: Path) -> dict[str, dict[str, int]]:
    """Each 2022 data zone's dwellings by band a to h; bands must sum to its total."""
    frame = pd.read_excel(path, sheet_name=NRS_SHEET, header=NRS_HEADER_ROW)
    columns = [NRS_ZONE_COLUMN, NRS_TOTAL_COLUMN, *NRS_BAND_COLUMNS.values()]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"NRS sheet {NRS_SHEET} lacks columns {missing}.")
    zones = frame.loc[
        frame[NRS_ZONE_COLUMN].astype(str).str.fullmatch(r"S01\d{6}"), columns
    ]
    if len(zones) != NRS_DATA_ZONES or zones[NRS_ZONE_COLUMN].duplicated().any():
        raise ValueError(
            f"NRS sheet {NRS_SHEET}: {len(zones):,} data zones, "
            f"expected {NRS_DATA_ZONES:,}."
        )
    counts = zones[columns[1:]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.all(np.isfinite(counts) & (counts >= 0) & (counts == np.floor(counts))):
        raise ValueError(f"NRS sheet {NRS_SHEET} has a count that is not one count.")
    counts = counts.astype(np.int64)
    if not np.array_equal(counts[:, 1:].sum(axis=1), counts[:, 0]):
        raise ValueError(f"NRS sheet {NRS_SHEET}: a zone's bands miss its total.")
    return {
        str(zone): dict(zip(NRS_BAND_COLUMNS, map(int, row[1:]), strict=True))
        for zone, row in zip(zones[NRS_ZONE_COLUMN], counts, strict=True)
    }


def decile_column(
    support,
    postcodes: pd.DataFrame,
    deciles: Mapping[str, Mapping[int, str]],
    *,
    system: str,
):
    """The system's modal decile per area, and the postcode counts behind it."""
    areas = np.asarray(support.arrays["area"])
    mine = postcodes[postcodes["oa21cd"].isin(pd.Index(areas))]
    ranked = mine[~mine["imd20ind"].isin(["", "0"])]
    nation = ranked["oa21cd"].str[0].to_numpy()
    rank = ranked["imd20ind"].astype(int).to_numpy()
    codes = np.empty(len(ranked), dtype=object)
    for prefix in NATION_PREFIXES[system]:
        rows = nation == prefix
        lookup = deciles[prefix]
        missing = sorted({int(r) for r in rank[rows] if int(r) not in lookup})
        if missing:
            raise ValueError(
                f"{system}: NSPL ranks {missing[:5]} are not in the {prefix} index."
            )
        codes[rows] = [lookup[int(r)] for r in rank[rows]]
    modal = modal_area_codes(
        areas,
        np.asarray(support.arrays[PARENT_COLUMNS[system]]),
        ranked["oa21cd"].to_numpy(dtype="U"),
        np.asarray(codes.tolist(), dtype="U"),
        domain=IMD_DECILES,
    )
    receipt = {
        **modal.receipt(),
        "postcodes_in_areas": int(len(mine)),
        "postcodes_without_rank": int(len(mine) - len(ranked)),
    }
    return modal.codes, receipt


def _voa_numerators(support, mix: Mapping) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Each E&W area's band shares in millionths, from its LSOA's committed mix."""
    if mix.get("bands") != list("ABCDEFGHI"):
        raise ValueError("VOA band mix must list bands A to I.")
    parts = {}
    for lsoa, row in mix["lsoa"].items():
        cumulative = [round(value * BAND_WEIGHT_SCALE) for value in row["cumulative"]]
        steps = np.diff([0, *cumulative])
        if cumulative[-1] != BAND_WEIGHT_SCALE or np.any(steps < 0):
            raise ValueError(f"VOA band mix for {lsoa} is not a partition.")
        parts[lsoa] = steps
    lsoas = np.asarray(support.arrays["lsoa_code"])
    unknown = sorted(set(lsoas.tolist()) - set(parts))
    if unknown:
        raise ValueError(f"VOA band mix lacks LSOAs {unknown[:5]}.")
    table = np.stack([parts[code] for code in lsoas.tolist()])
    columns = BAND_WEIGHT_COLUMNS[EW]
    numerators = {
        column: table[:, index].astype(np.int64) for index, column in enumerate(columns)
    }
    return numerators, np.full(len(lsoas), BAND_WEIGHT_SCALE, dtype=np.int64)


def _scotland_numerators(
    support, counts: Mapping[str, Mapping[str, int]]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Each Scottish area's 2022 data zone dwellings by band A to H."""
    bands = {column: column.rsplit("_", 1)[1] for column in BAND_WEIGHT_COLUMNS[SCOT]}
    # The Scottish support's lsoa_code is the area's 2022 data zone.
    zones = np.asarray(support.arrays["lsoa_code"]).tolist()
    missing = sorted(set(zones) - set(counts))
    if missing or any(set(row) != set(bands.values()) for row in counts.values()):
        raise ValueError(
            f"Scottish band counts do not cover each data zone A to H: {missing[:5]}."
        )
    numerators = {
        column: np.asarray([counts[zone][band] for zone in zones], dtype=np.int64)
        for column, band in bands.items()
    }
    return numerators, sum(numerators.values())


def _source_tag(name: str) -> str:
    spec = SOURCES[name]
    return f"{spec['chronicle_package_id']}@sha256:{spec['sha256']}"


def _metadata(system: str, support, resources: Mapping[str, dict]) -> dict:
    indices = "+".join(_source_tag(NATION_INDEX[p]) for p in NATION_PREFIXES[system])
    metadata = {
        IMD_DECILE_COLUMN: {
            "kind": "code",
            "source": f"{_source_tag('nspl')}+{indices}#modal-live-small-user-postcode",
            "vintage": DECILE_VINTAGES[system],
            "relation": "inferred_modal",
        }
    }
    households = support.metadata["columns"]["households"]["basis"]
    for column in BAND_WEIGHT_COLUMNS[system]:
        band = column.rsplit("_", 1)[1]
        if system == EW:
            resource = resources[VOA_BAND_MIX]
            source = (
                f"{resource['publisher_package']}@sha256:{resource['publisher_sha256']}"
                f"#{resource['resource']}@sha256:{resource['sha256']}"
            )
            share = f"voa_ctsop1_1_2025_lsoa_band_{band}_share_6dp"
        else:
            source = f"{_source_tag('nrs_dz_dwellings')}#sheet-{NRS_SHEET}"
            share = f"nrs_dz2022_{NRS_SHEET}_band_{band}_share_round_half_even"
        metadata[column] = {
            "kind": "weight",
            "source": source,
            "basis": f"{households}_x_{share}_x1e6",
        }
    return metadata


def _resources(directory: Path) -> tuple[dict, dict[str, dict]]:
    voa_path = directory / VOA_BAND_MIX
    voa = json.loads(voa_path.read_text(encoding="utf-8"))
    identities = {
        VOA_BAND_MIX: {
            "resource": VOA_BAND_MIX,
            "sha256": _sha256(voa_path),
            "publisher_package": voa["source"]["chronicle_package_id"],
            "publisher_sha256": voa["source"]["sha256"],
            "publisher_file": voa["source"]["filename"],
            "as_at": voa["source"]["as_at"],
        },
    }
    return voa, identities


def condition_supports(
    base_payloads: Mapping[str, bytes],
    postcodes: pd.DataFrame,
    deciles: Mapping[str, Mapping[int, str]],
    *,
    voa_mix: Mapping,
    scotland_counts: Mapping[str, Mapping[str, int]],
    resources: Mapping[str, dict],
) -> tuple[dict[str, bytes], dict[str, dict]]:
    """The three conditioned supports and each one's receipts."""
    payloads, entries = {}, {}
    for system in SYSTEMS:
        base = base_payloads[system]
        support = decode_atomic_support(base)
        decile, decile_receipt = decile_column(
            support, postcodes, deciles, system=system
        )
        households = np.asarray(support.arrays["households"], dtype=np.int64)
        band_receipt = None
        weights: dict[str, np.ndarray] = {}
        if BAND_WEIGHT_COLUMNS[system]:
            numerators, denominators = (
                _voa_numerators(support, voa_mix)
                if system == EW
                else _scotland_numerators(support, scotland_counts)
            )
            weights, band_receipt = apportion_weights(
                households, numerators, denominators, scale=BAND_WEIGHT_SCALE
            )
        payload = assemble_uk_conditioned_area_support(
            base=base,
            imd_decile=decile,
            band_weights=weights,
            column_metadata=_metadata(system, support, resources),
        )
        if uk_conditioned_support_base(payload) != base:
            raise ValueError(f"{system}: conditioned support does not extend its base.")
        if len(payload) > RAW_BYTES_MAX_BYTES:
            raise ValueError(f"{system}: support exceeds the raw-bytes-v1 cap.")
        conditioned = decode_atomic_support(payload)
        by_decile = pd.Series(households).groupby(decile).sum()
        payloads[system] = payload
        entries[system] = {
            "imd_decile": decile_receipt,
            "households_by_imd_decile": {
                code: int(by_decile.get(code, 0)) for code in IMD_DECILES
            },
            "band_weights": band_receipt,
            "zero_mass_cells": zero_mass_cells(
                conditioned,
                domains={
                    "frs_region": sorted(
                        set(conditioned.arrays["frs_region"].tolist())
                    ),
                    IMD_DECILE_COLUMN: IMD_DECILES,
                },
                levels=(("frs_region", IMD_DECILE_COLUMN), ("frs_region",)),
                weights=("households", *BAND_WEIGHT_COLUMNS[system]),
            ),
            "column_metadata": {
                column: dict(conditioned.metadata["columns"][column])
                for column in (IMD_DECILE_COLUMN, *BAND_WEIGHT_COLUMNS[system])
            },
        }
    return payloads, entries


def _base_payloads(directory: Path) -> dict[str, bytes]:
    """The registered supports, refused unless their bytes are the registered ones."""
    from microcosm.build.uk_runtime.country_adapter import uk_atomic_support_register

    register = uk_atomic_support_register()
    payloads = {}
    for system in SYSTEMS:
        payload = (directory / f"{SUPPORT_SOURCES[system]}.npz").read_bytes()
        if (
            hashlib.sha256(payload).hexdigest() != register[system]["sha256"]
            or len(payload) != register[system]["size_bytes"]
        ):
            raise ValueError(f"{system}: support bytes are not the registered support.")
        payloads[system] = payload
    return payloads


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _log("Building UK conditioned atomic-area supports")
    paths = _pinned_sources(args.source_manifest)
    bases = _base_payloads(args.supports_dir)
    deciles = {
        "E": _england(paths["iod2019"]),
        "W": _wales(paths["wimd2019"]),
        "S": _scotland(paths["simd2020v2"]),
        "N": _northern_ireland(),
    }
    _log("  reading the NSPL")
    postcodes = _postcodes(paths["nspl"])
    voa, resources = _resources(args.resource_dir)
    payloads, entries = condition_supports(
        bases,
        postcodes,
        deciles,
        voa_mix=voa,
        scotland_counts=_nrs_band_counts(paths["nrs_dz_dwellings"]),
        resources=resources,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    supports = {}
    for system, payload in payloads.items():
        path = args.out_dir / f"{CONDITIONED_SOURCES[system]}.npz"
        path.write_bytes(payload)
        supports[system] = {
            "filename": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "base": {
                "source": SUPPORT_SOURCES[system],
                "sha256": hashlib.sha256(bases[system]).hexdigest(),
            },
            **entries[system],
        }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "kind": PROVENANCE_KIND,
        "built_on": dt.date.today().isoformat(),
        "tool": "tools/build_uk_conditioned_area_supports.py",
        "issues": ISSUES,
        "chronicle": CHRONICLE,
        "publisher_files": {name: dict(spec) for name, spec in SOURCES.items()},
        "resources": resources,
        "band_weight_scale": BAND_WEIGHT_SCALE,
        "supports": supports,
        "notes": list(NOTES),
    }
    args.provenance_json.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for system, entry in supports.items():
        _log(
            f"  {system}: {entry['size_bytes']:,} bytes, sha256 {entry['sha256'][:16]}"
        )
    _log(f"  provenance written to {args.provenance_json}")


if __name__ == "__main__":
    main()
