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
  band of the area's 2022 data zone, in which its 2022 output areas nest, at
  December 2024: NRS's estimates, which Chronicle emits as facts
  (chronicle#421). Northern Ireland has domestic rates and no bands.

The publisher files are chronicle#412's raw-only registrations (PR
chronicle#413), pinned here by sha256. The NRS facts are streamed from the
pinned UK Chronicle feed (``uk/chronicle_feed.json``), which is checked whole
against the pin. The VOA band mix is a committed package resource. The
provenance register records every input's identity, the fingerprint of the
facts read, each nation's decile rule counts, the rounding, and every
zero-mass draw cell.

Example (odfpy reads the Welsh Government's ODS; nothing else here needs it):
    uv run --no-sync --with odfpy python tools/build_uk_conditioned_area_supports.py \\
        --supports-dir build/uk/supports \\
        --source-manifest build/uk/conditioned_sources.json \\
        --ledger-facts .codex-work/uk-artifact/consumer_facts.jsonl \\
        --ledger-manifest .codex-work/uk-artifact/manifest.json \\
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
import math
import re
import sys
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.atomic_geography import decode_atomic_support
from microcosm.build.atomic_support_conditioning import (
    apportion_weights,
    modal_area_codes,
    zero_mass_cells,
)
from microcosm.build.ledger_artifact import consumer_fact_schema_versions_of
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
from microcosm.build.uk_runtime.chronicle_feed import (
    require_uk_chronicle_feed_subset_row_contracts,
)
from microcosm.graph.codecs import RAW_BYTES_MAX_BYTES

_TOOLS = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _TOOLS.parent
_UK_PACKAGE = _REPOSITORY_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
_DEFAULT_PROVENANCE = _UK_PACKAGE / CONDITIONED_PROVENANCE_RESOURCE
DEFAULT_FEED_PIN = _UK_PACKAGE / "chronicle_feed.json"
VOA_BAND_MIX = "voa_council_tax_band_mix_lsoa.json"

PROVENANCE_SCHEMA_VERSION = 1
PROVENANCE_KIND = "uk_atomic_area_conditioned_supports_provenance"
EW, SCOT, NI = SYSTEMS

#: chronicle#412's raw-only registrations, as PR chronicle#413 pins them.
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
}
#: The Chronicle commit the UK feed pins: chronicle#421's merge, which emits
#: NRS's band facts and registers these files as chronicle#413 did. The
#: supports were first built at #413's head; the file digests did not move.
CHRONICLE = {
    "pull_request": "PolicyEngine/chronicle#421",
    "commit": "80a3482905843d447f8811b500a56dbf3b5f47fa",
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
#: NRS's dwellings by council-tax band and 2022 data zone at December 2024
#: (chronicle#421): nearest the FRS 2024-25 year and VOA's stock at 31 March
#: 2025. Each zone carries its total and bands A to H.
NRS_RECORD_SET = "nrs.dwellings_data_zone_2025.2024"
NRS_PERIOD = ("calendar_year", "2024")
NRS_ZONE_VINTAGE = "dz_2022"
NRS_DATA_ZONES = 7_392
NRS_TOTAL_CONCEPT = "nrs.dwellings_total"
NRS_BAND_CONCEPTS = {f"nrs.dwellings_band_{band}": band for band in "abcdefgh"}

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
    "at December 2024: nearest the FRS 2024-25 year and VOA's stock at 31 March 2025. "
    "NRS counts dwellings on the valuation list, as VOA does in England and Wales. "
    "Chronicle emits the zones' totals and bands A to H as facts (record set "
    "nrs.dwellings_data_zone_2025.2024, chronicle#421); this tool reads them from the "
    "pinned UK feed and records their fingerprint.",
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
    parser.add_argument(
        "--ledger-facts",
        required=True,
        type=Path,
        help="consumer_facts.jsonl of the pinned feed",
    )
    parser.add_argument(
        "--ledger-manifest",
        required=True,
        type=Path,
        help="manifest.json of the pinned feed",
    )
    parser.add_argument("--feed-pin", type=Path, default=DEFAULT_FEED_PIN)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--provenance-json", type=Path, default=_DEFAULT_PROVENANCE)
    parser.add_argument("--resource-dir", type=Path, default=_UK_PACKAGE)
    return parser.parse_args(argv)


def _feed_identity(pin_path: Path, manifest_path: Path) -> dict[str, object]:
    """The pin, after the manifest's bytes and its facts digest are checked."""

    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != pin["manifest_sha256"]:
        raise ValueError(
            f"{manifest_path} is not the manifest uk/chronicle_feed.json pins."
        )
    if json.loads(manifest_bytes).get("facts_sha256") != pin["facts_sha256"]:
        raise ValueError("The feed manifest names other facts than the pin.")
    return {
        key: pin[key]
        for key in (
            "source_commit",
            "facts_sha256",
            "manifest_sha256",
            "fact_row_count",
        )
    }


def _pinned_row_contracts(pin_path: Path) -> list[str]:
    """The row contracts the feed pin lists, refusing a pin that lists none."""

    versions = json.loads(pin_path.read_text(encoding="utf-8")).get(
        "consumer_fact_schema_versions"
    )
    if not isinstance(versions, list) or not versions:
        raise ValueError(
            f"{pin_path} lists no consumer_fact_schema_versions for the feed's rows."
        )
    return versions


def _read_feed(
    path: Path, *, markers: tuple[bytes, ...], identity: Mapping[str, object]
) -> list[dict]:
    """Every row carrying a marker; the whole file's digest and rows are checked."""

    digest = hashlib.sha256()
    count = 0
    facts = []
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if line.strip():
                count += 1
            if any(marker in line for marker in markers):
                facts.append(json.loads(line))
    if digest.hexdigest() != identity["facts_sha256"]:
        raise ValueError(f"{path} is not the feed uk/chronicle_feed.json pins.")
    if count != identity["fact_row_count"]:
        raise ValueError(
            f"{path} carries {count} rows; the pin records {identity['fact_row_count']}."
        )
    return facts


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


def _nrs_band_counts(
    facts: Iterable[Mapping],
) -> tuple[dict[str, dict[str, int]], dict[str, object]]:
    """Each 2022 data zone's dwellings by band a to h, and the facts' receipt.

    Reads NRS's December 2024 record set only. Every row of it must be one
    data zone's total or band, each zone must carry all nine once, and its
    bands must sum to its total. The receipt fingerprints the counts read.
    """
    concepts = {NRS_TOTAL_CONCEPT: "total", **NRS_BAND_CONCEPTS}
    cells: dict[str, dict[str, int]] = {}
    files = set()
    rows = 0
    for fact in facts:
        if (fact.get("layout") or {}).get("record_set_id") != NRS_RECORD_SET:
            continue
        rows += 1
        geography = fact.get("geography") or {}
        period = fact.get("period") or {}
        zone = str(geography.get("id", ""))
        # A feed row names its concept as the source measure it observes.
        concept = (fact.get("observed_measure") or {}).get("source_concept")
        key = concepts.get(concept)
        if (
            key is None
            or geography.get("vintage") != NRS_ZONE_VINTAGE
            or re.fullmatch(r"S01\d{6}", zone) is None
            or (period.get("type"), str(period.get("value"))) != NRS_PERIOD
        ):
            raise ValueError(
                f"NRS record set {NRS_RECORD_SET} carries a row that is not a data "
                f"zone's December 2024 total or band: {zone} {concept}."
            )
        value = fact.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or value != int(value)
        ):
            raise ValueError(f"NRS zone {zone} has a {key} that is not one count.")
        if key in cells.setdefault(zone, {}):
            raise ValueError(f"NRS zone {zone} carries its {key} twice.")
        cells[zone][key] = int(value)
        source = fact.get("source") or {}
        files.add((source.get("source_file"), source.get("source_sha256")))
    if len(cells) != NRS_DATA_ZONES:
        raise ValueError(
            f"NRS record set {NRS_RECORD_SET}: {len(cells):,} data zones, "
            f"expected {NRS_DATA_ZONES:,}."
        )
    bands = tuple(NRS_BAND_CONCEPTS.values())
    if any(set(row) != {"total", *bands} for row in cells.values()):
        raise ValueError(f"NRS record set {NRS_RECORD_SET}: a zone lacks a band.")
    if any(sum(row[band] for band in bands) != row["total"] for row in cells.values()):
        raise ValueError(
            f"NRS record set {NRS_RECORD_SET}: a zone's bands miss its total."
        )
    if len(files) != 1:
        raise ValueError(
            f"NRS record set {NRS_RECORD_SET} names {len(files)} source files."
        )
    ((source_file, source_sha256),) = files
    canonical = json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()
    receipt = {
        "record_set_id": NRS_RECORD_SET,
        "period": dict(zip(("type", "value"), NRS_PERIOD, strict=True)),
        "rows": rows,
        "data_zones": len(cells),
        "source_file": source_file,
        "source_sha256": source_sha256,
        "fingerprint_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    counts = {zone: {band: row[band] for band in bands} for zone, row in cells.items()}
    return counts, receipt


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


def _metadata(
    system: str,
    support,
    resources: Mapping[str, dict],
    nrs_facts: Mapping[str, object],
) -> dict:
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
            source = (
                f"chronicle:{nrs_facts['record_set_id']}"
                f"@sha256:{nrs_facts['fingerprint_sha256']}"
            )
            share = f"nrs_dz2022_{NRS_PERIOD[1]}_band_{band}_share_round_half_even"
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
    nrs_facts: Mapping[str, object],
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
            column_metadata=_metadata(system, support, resources, nrs_facts),
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
    identity = _feed_identity(args.feed_pin, args.ledger_manifest)
    _log("  streaming the pinned feed")
    facts = _read_feed(
        args.ledger_facts, markers=(NRS_RECORD_SET.encode(),), identity=identity
    )
    require_uk_chronicle_feed_subset_row_contracts(
        consumer_fact_schema_versions_of(facts),
        pinned_versions=_pinned_row_contracts(args.feed_pin),
    )
    scotland_counts, nrs_facts = _nrs_band_counts(facts)
    payloads, entries = condition_supports(
        bases,
        postcodes,
        deciles,
        voa_mix=voa,
        scotland_counts=scotland_counts,
        resources=resources,
        nrs_facts=nrs_facts,
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
        "ledger_feed": identity,
        "feed_facts": {"nrs_dwellings_by_band": nrs_facts},
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
