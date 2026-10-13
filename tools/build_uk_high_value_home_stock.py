"""Build the UK high-value home stock resources (microcosm#1240, fix 2).

Reads the pinned HM Land Registry Price Paid yearly files, the UK HPI full
file and the ONS NSPL (all registered raw-only by PolicyEngine/chronicle#343
and verified here against their Chronicle manifests), the committed
England-and-Wales atomic support, the VOA CTSOP1.1 stock and the pinned
Chronicle consumer feed (census tenure by local authority, the OBR costing
note's non-owner-occupied share, the ONS HPSSA parity series), and writes
three committed resources under the UK country package:

- ``high_value_home_stock_facts.json``: owner-occupied and all-dwelling
  counts above each threshold by constituency, local authority, region and
  nation, in consumer-fact row shape under the Microcosm-owned
  ``microcosm.derived_fact.`` key namespace;
- ``high_value_home_value_knots.json``: weighted quantile knots of the
  basis value per constituency, region and nation, for the post-geography
  value map;
- ``high_value_home_stock.provenance.json``: inputs, parameters and receipts.

The deduplicated dwelling table is cached under ``--cache-dir`` keyed by the
Price Paid registration, so re-runs that only change parameters or facts skip
the 31.6-million-row pass. Every number the resources carry is an aggregate
of public administrative data; nothing here reads licensed microdata.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages/microcosm-calibrate/src"))

from microcosm.calibrate.geography_constants import (  # noqa: E402
    UK_GEOGRAPHY_ID_TO_LABEL,
)


def _load_stock_module():
    """Load the pure library module by path.

    Importing it through ``microcosm.build`` would load the UK country spec,
    which refuses a package whose declared resources are missing: the very
    resources this tool writes on a first run.
    """
    import importlib.util

    path = (
        REPO_ROOT
        / "packages/microcosm-build/src/microcosm/build/uk_runtime/high_value_home_stock.py"
    )
    spec = importlib.util.spec_from_file_location("high_value_home_stock", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hv = _load_stock_module()

UK_PACKAGE = REPO_ROOT / "packages/microcosm-build/src/microcosm/build/uk"
DEFAULT_CROSSWALK = UK_PACKAGE / "local_area_crosswalk.json"
DEFAULT_SUPPORT_PROVENANCE = UK_PACKAGE / "uk_atomic_area_supports.provenance.json"
DEFAULT_LA_NAMES = UK_PACKAGE / "local_authority_names.json"
DEFAULT_FACTS_OUT = UK_PACKAGE / "high_value_home_stock_facts.json"
DEFAULT_KNOTS_OUT = UK_PACKAGE / "high_value_home_value_knots.json"
DEFAULT_PROVENANCE_OUT = UK_PACKAGE / "high_value_home_stock.provenance.json"
DEFAULT_BOUNDS_OUT = UK_PACKAGE / "price_paid_property_support_bounds.json"
WAS_BOUNDS = UK_PACKAGE / "was_wealth_support_bounds.json"

METHOD_VERSION = "uk-high-value-home-stock-v1"
FACT_KEY_PREFIX = "microcosm.derived_fact.high_value_home_stock.v1:"
SEMANTIC_KEY_PREFIX = "microcosm.semantic_fact.high_value_home_stock.v1:"
ROW_SCHEMA_VERSION = "microcosm.derived_consumer_fact.v1"
RESOURCE_KIND = "uk_derived_consumer_facts"
KNOTS_KIND = "uk_high_value_home_value_knots"
PROVENANCE_KIND = "uk_high_value_home_stock_provenance"

NSPL_DATA_MEMBER = "Data/NSPL_AUG_2026_UK.csv"
NSPL_PCON_NAMES_MEMBER = "Documents/PCON Westminster Parliamentary Constituency names and codes UK as at 12_24.csv"
VOA_CSV_MEMBER = "CTSOP1.1/CTSOP1_1_2025_03_31.csv"

#: Threshold ids in April-2026 pounds.
THRESHOLDS: Mapping[str, int] = {
    "over_gbp_1m": 1_000_000,
    "over_gbp_1_5m": 1_500_000,
    "over_gbp_2m": 2_000_000,
}
THRESHOLD_LABELS: Mapping[str, str] = {
    "over_gbp_1m": "over £1 million",
    "over_gbp_1_5m": "over £1.5 million",
    "over_gbp_2m": "over £2 million",
}
MEASURES: Mapping[str, dict[str, str]] = {
    "owner_occupied_homes": {
        "label": "Owner-occupied dwellings",
        "concept": "microcosm.high_value_homes.owner_occupied_count",
        "tenure": "owner_occupied",
        "tenure_label": "Owner-occupied",
    },
    "private_sector_dwellings": {
        "label": "Private-sector dwellings",
        "concept": "microcosm.high_value_homes.private_sector_dwellings_count",
        "tenure": "private_sector",
        "tenure_label": "All tenures except social rented",
    },
}
#: Feed selectors (record set spec ids) the owner-occupied adjustment reads.
TENURE_OWNER_SPEC_IDS = (
    "uk.local_geography.tenure.owned_outright.v1",
    "uk.local_geography.tenure.owned_mortgage.v1",
)
TENURE_SPEC_PREFIX = "uk.local_geography.tenure."
TENURE_ALL_SPEC_ID = "uk.local_geography.tenure.all_households.v1"
TENURE_SOCIAL_SPEC_ID = "uk.local_geography.tenure.social_rent.v1"
OBR_SHARE_CONCEPT = "obr.hvcts_costing.non_owner_occupied_share"
HPSSA_SALES_CONCEPT = "ons.hpssa.residential_property_sales"
HPSSA_MEDIAN_CONCEPT = "ons.hpssa.median_price_paid"

LEVELS: Mapping[str, dict[str, str]] = {
    "constituency_code": {"level": "constituency", "vintage": "pcon_2024"},
    "local_authority_code": {"level": "local_authority", "vintage": "lad_2023"},
    "region_code": {"level": "region", "vintage": "rgn_2024"},
    "nation_code": {"level": "country", "vintage": "current"},
}
ISLES_OF_SCILLY = "E06000053"
CORNWALL = "E06000052"
COVERAGE_RATIO_BOUNDS = (0.5, 3.0)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ppd-dir", type=Path, required=True, help="directory holding pp-YYYY.csv"
    )
    parser.add_argument("--ppd-manifest", type=Path, required=True)
    parser.add_argument("--hpi-csv", type=Path, required=True)
    parser.add_argument("--hpi-manifest", type=Path, required=True)
    parser.add_argument("--nspl-zip", type=Path, required=True)
    parser.add_argument("--nspl-manifest", type=Path, required=True)
    parser.add_argument("--voa-zip", type=Path, required=True)
    parser.add_argument("--voa-manifest", type=Path, required=True)
    parser.add_argument("--ew-support", type=Path, required=True)
    parser.add_argument(
        "--support-provenance", type=Path, default=DEFAULT_SUPPORT_PROVENANCE
    )
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK)
    parser.add_argument("--la-names", type=Path, default=DEFAULT_LA_NAMES)
    parser.add_argument(
        "--ledger-facts", type=Path, help="consumer_facts.jsonl of the pinned feed"
    )
    parser.add_argument(
        "--ledger-manifest", type=Path, help="manifest.json of the pinned feed"
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("dwellings", "all"), default="all")
    parser.add_argument("--basis-year", type=int, default=2024)
    parser.add_argument("--reference-month", default="2026-04")
    parser.add_argument("--stock-date", default="2025-12-31")
    parser.add_argument("--cap", type=int, default=20_000_000)
    parser.add_argument("--knot-minimum", type=int, default=50)
    parser.add_argument(
        "--obr-non-owner-share",
        type=float,
        default=None,
        help=(
            "Fallback for the OBR costing note's non-owner-occupied share when the feed "
            "predates chronicle#349 (recorded as a parameter, not a fact)."
        ),
    )
    parser.add_argument("--built-on", default=_dt.date.today().isoformat())
    parser.add_argument("--facts-out", type=Path, default=DEFAULT_FACTS_OUT)
    parser.add_argument("--knots-out", type=Path, default=DEFAULT_KNOTS_OUT)
    parser.add_argument("--provenance-out", type=Path, default=DEFAULT_PROVENANCE_OUT)
    parser.add_argument("--bounds-out", type=Path, default=DEFAULT_BOUNDS_OUT)
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="development only: trust the local bytes without hashing them",
    )
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------
# Pins


def _manifest_files(manifest_path: Path) -> tuple[dict, dict[str, dict]]:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{manifest_path} declares no files.")
    return manifest, {str(key): dict(value) for key, value in files.items()}


def _verify_pin(
    path: Path, entry: Mapping[str, object], *, skip: bool
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path} must be a regular local file.")
    size = path.stat().st_size
    if size != int(entry["size_bytes"]):
        raise ValueError(
            f"{path}: size {size} differs from the registered {entry['size_bytes']}."
        )
    if not skip:
        digest = hv.sha256_of_file(path)
        if digest != str(entry["sha256"]):
            raise ValueError(f"{path}: sha256 differs from the registered pin.")
    return {
        "filename": str(entry["filename"]),
        "sha256": str(entry["sha256"]),
        "size_bytes": int(entry["size_bytes"]),
        "source_url": str(entry.get("source_url", "")),
        "r2_key": str(((entry.get("storage") or {}).get("r2") or {}).get("key", "")),
    }


def _registration(
    manifest: Mapping[str, object], files: Mapping[str, Mapping[str, object]]
) -> dict:
    return {
        "chronicle_source_id": manifest.get("source_id"),
        "chronicle_package_id": manifest.get("package_id"),
        "registration_kind": manifest.get("registration_kind", "committed"),
        "files": {key: files[key] for key in sorted(files)},
    }


# ----------------------------------------------------------------------------
# Stage A: the dwelling table


def _dwelling_cache_path(
    cache_dir: Path, ppd_files: Mapping[str, Mapping[str, object]], stock_yyyymm: int
) -> Path:
    digest = hashlib.sha256()
    for key in sorted(ppd_files):
        digest.update(f"{key}:{ppd_files[key]['sha256']}\n".encode())
    digest.update(f"stock:{stock_yyyymm}\nmethod:{METHOD_VERSION}\n".encode())
    return cache_dir / f"dwellings-{digest.hexdigest()[:16]}.parquet"


def _read_nspl(nspl_zip: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    with zipfile.ZipFile(nspl_zip) as archive:
        with archive.open(NSPL_DATA_MEMBER) as stream:
            postcodes = hv.read_nspl_postcode_output_areas(stream)
        with archive.open(NSPL_PCON_NAMES_MEMBER) as stream:
            text = io.TextIOWrapper(stream, encoding="utf-8-sig")
            names = {row["PCON24CD"]: row["PCON24NM"] for row in csv.DictReader(text)}
    return postcodes, names


def _build_dwellings(
    ppd_dir: Path,
    ppd_files: Mapping[str, Mapping[str, object]],
    postcode_index: pd.Index,
    *,
    stock_yyyymm: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frames = []
    receipts: dict[str, object] = {"transfers_by_year": {}}
    for key in sorted(ppd_files, key=int):
        path = ppd_dir / str(ppd_files[key]["filename"])
        frame = hv.read_price_paid_file(path, postcode_index=postcode_index)
        receipts["transfers_by_year"][key] = {
            "residential_category_a": int(len(frame)),
            "bulk_transfers": int(frame["batch"].sum()),
            "postcode_unmatched": int((frame["postcode_row"] < 0).sum()),
            "new_build_first_registrations": int(frame["newbuild"].sum()),
        }
        _log(f"  {path.name}: {len(frame):,} residential category A transfers")
        frames.append(frame)
    transfers = pd.concat(frames, ignore_index=True)
    del frames
    receipts["transfers_total"] = int(len(transfers))
    dwellings = hv.latest_valuations(transfers, stock_yyyymm=stock_yyyymm)
    return dwellings, receipts


# ----------------------------------------------------------------------------
# Stage B helpers


def _load_support(
    path: Path, provenance_path: Path, *, skip: bool
) -> dict[str, np.ndarray]:
    payload = path.read_bytes()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = provenance["supports"]["uk_ew_output_area_2021"]["sha256"]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected and not skip:
        raise ValueError(
            "The E&W support's sha256 differs from the committed provenance."
        )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: np.asarray(archive[name]) for name in metadata["columns"]}
    return arrays | {"__sha256__": digest}


def _read_voa(voa_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(voa_zip) as archive:
        with archive.open(VOA_CSV_MEMBER) as stream:
            table = pd.read_csv(stream, dtype=str, encoding="utf-8-sig")
    return table


def _voa_all_properties_by_local_authority(
    voa: pd.DataFrame, aliases: Mapping[str, str]
) -> dict[str, float]:
    rows = voa[voa["geography"] == "LAUA"]
    out: dict[str, float] = {}
    for code, value in zip(rows["ecode"], rows["all_properties"], strict=True):
        roster_code = aliases.get(str(code), str(code))
        if value in ("-", "..", ""):
            continue
        out[roster_code] = float(str(value).replace(",", ""))
    return out


def _read_facts(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _spec_id(fact: Mapping) -> str:
    return str((fact.get("layout") or {}).get("record_set_spec_id") or "")


def _tenure_shares(
    facts: Iterable[Mapping],
) -> tuple[dict[str, float], dict[str, float], dict[str, object]]:
    """Private-sector stock share and owner-occupied share per local authority.

    Price Paid Data records market transfers, so its dwellings are the
    private-sector stock: social rented dwellings are never in it. Both the
    coverage correction and the owner-occupied share therefore refer to the
    census households net of social renting (TS054 by local authority).
    Returns ``(private_share, owner_share_of_private, receipt)``.
    """
    totals: dict[str, float] = {}
    owner: dict[str, float] = {}
    social: dict[str, float] = {}
    spec_ids: set[str] = set()
    for fact in facts:
        spec = _spec_id(fact)
        if not spec.startswith(TENURE_SPEC_PREFIX):
            continue
        geography = fact.get("geography") or {}
        if geography.get("level") != "local_authority":
            continue
        code = str(geography.get("id"))
        spec_ids.add(spec)
        value = float(fact["value"])
        if spec == TENURE_ALL_SPEC_ID:
            totals[code] = value
        elif spec in TENURE_OWNER_SPEC_IDS:
            owner[code] = owner.get(code, 0.0) + value
        elif spec == TENURE_SOCIAL_SPEC_ID:
            social[code] = value
    private_share: dict[str, float] = {}
    owner_share: dict[str, float] = {}
    for code, total in totals.items():
        if total <= 0:
            continue
        private = total - social.get(code, 0.0)
        private_share[code] = private / total
        owner_share[code] = owner.get(code, 0.0) / private if private > 0 else 0.0
    receipt = {
        "tenure_record_set_spec_ids": sorted(spec_ids),
        "local_authorities_with_tenure": len(owner_share),
        "basis": (
            "Census 2021 TS054 by local authority: private-sector stock is all "
            "households less social rented; the owner-occupied share is owned "
            "outright plus owned with a mortgage over that private stock"
        ),
    }
    return private_share, owner_share, receipt


def _obr_non_owner_share(facts: Iterable[Mapping]) -> tuple[float, str]:
    for fact in facts:
        observed = fact.get("observed_measure") or {}
        if observed.get("source_concept") == OBR_SHARE_CONCEPT:
            value = float(fact["value"])
            unit = str(
                (fact.get("measure") or observed).get("unit")
                or observed.get("unit")
                or ""
            )
            share = value / 100.0 if unit == "percent" or value > 1.0 else value
            return share, str(fact.get("aggregate_fact_key"))
    raise ValueError(
        "The feed carries no OBR non-owner-occupied share fact (re-pin to chronicle#349)."
    )


def _hpssa_parity(
    facts: Iterable[Mapping], transfers_by_seat_fy: pd.DataFrame | None
) -> dict[str, object]:
    """Seat sales counts by fiscal year in the feed, for the parity receipt."""
    sales: dict[tuple[str, str], float] = {}
    for fact in facts:
        observed = fact.get("observed_measure") or {}
        if observed.get("source_concept") != HPSSA_SALES_CONCEPT:
            continue
        dims = fact.get("dimensions") or {}
        if dims.get("property_type") not in (None, "", "all"):
            continue
        geography = fact.get("geography") or {}
        if geography.get("level") != "constituency":
            continue
        period = str((fact.get("period") or {}).get("value"))
        sales[(str(geography.get("id")), period)] = float(fact["value"])
    return {"hpssa_seat_fiscal_year_cells": len(sales), "cells": sales}


# ----------------------------------------------------------------------------
# Rows


def _digest(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]


def _fact_row(
    *,
    level: str,
    code: str,
    name: str,
    vintage: str,
    measure_id: str,
    threshold_id: str,
    value: float,
    period_value: str,
    stock_date: str,
    basis_year: int,
    reference_month: str,
    inputs_sha256: str,
    ppd_vintage: str,
) -> dict[str, object]:
    measure = MEASURES[measure_id]
    record_set_spec_id = (
        f"uk.local_geography.high_value_homes.{measure_id}.{threshold_id}.v1"
    )
    identity = {
        "record_set_spec_id": record_set_spec_id,
        "level": level,
        "id": code,
        "period": period_value,
        "inputs_sha256": inputs_sha256,
        "method": METHOD_VERSION,
    }
    digest = _digest(identity)
    label = f"{measure['label']} valued {THRESHOLD_LABELS[threshold_id]} at April 2026 prices, {name}"
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "aggregate_fact_key": FACT_KEY_PREFIX + digest,
        "semantic_fact_key": SEMANTIC_KEY_PREFIX + digest,
        "value": float(round(value, 3)),
        "value_type": "float",
        "assertion": "observation",
        "provenance_class": "derived_administrative",
        "period": {"type": "calendar_year", "value": period_value},
        "period_coverage": {
            "start_date": f"{period_value}-01-01",
            "end_date": stock_date,
            "basis": "stock",
            "notes": (
                f"Dwelling stock at {stock_date}, valued at calendar-{basis_year} average "
                f"prices; thresholds are stated at {reference_month} prices and deflated "
                "with the England all-dwellings UK HPI."
            ),
        },
        "geography": {"level": level, "id": code, "vintage": vintage, "name": name},
        "entity": {"name": "dwelling", "role": measure["tenure"]},
        "aggregation": {"method": "sum"},
        "observed_measure": {
            "source_concept": measure["concept"],
            "source_measure_id": measure_id,
            "unit": "count",
        },
        "measure": {"unit": "count"},
        "concept_alignment": {
            "source_concept": measure["concept"],
            "canonical_concept": "uk.dwelling.count",
            "relation": "derived",
        },
        "dimensions": {
            "microcosm.value_threshold": threshold_id,
            "uk.dwelling.tenure": measure["tenure"],
        },
        "dimension_labels": {
            "microcosm.value_threshold": "Value threshold (April 2026 prices)",
            "uk.dwelling.tenure": "Tenure",
        },
        "dimension_value_labels": {
            "microcosm.value_threshold": {threshold_id: THRESHOLD_LABELS[threshold_id]},
            "uk.dwelling.tenure": {measure["tenure"]: measure["tenure_label"]},
        },
        "source": {
            "source_name": "microcosm",
            "source_table": "HM Land Registry Price Paid Data, UK HPI and ONS NSPL, Microcosm derivation",
            "source_file": "high_value_home_stock_facts.json",
            "vintage": ppd_vintage,
            "source_sha256": inputs_sha256,
            "url": "tools/build_uk_high_value_home_stock.py",
        },
        "lineage": {
            "source_record_id": f"microcosm.high_value_home_stock.{measure_id}.{threshold_id}.{level}.{code}"
        },
        "layout": {
            "record_set_id": f"uk.local_geography.high_value_homes.{measure_id}.{threshold_id}",
            "record_set_spec_id": record_set_spec_id,
            "measure_id": measure_id,
        },
        "label": label,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    path.write_text(rendered, encoding="utf-8")


# ----------------------------------------------------------------------------
# Main


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    stock_yyyymm = hv.stock_yyyymm_from_date(args.stock_date)
    reference_year, reference_month = (
        int(part) for part in args.reference_month.split("-")
    )

    _log("Verifying registered pins ...")
    ppd_manifest, ppd_files = _manifest_files(args.ppd_manifest)
    hpi_manifest, hpi_files = _manifest_files(args.hpi_manifest)
    nspl_manifest, nspl_files = _manifest_files(args.nspl_manifest)
    voa_manifest, voa_files = _manifest_files(args.voa_manifest)
    verified: dict[str, dict] = {}
    verified["price_paid"] = _registration(
        ppd_manifest,
        {
            key: _verify_pin(
                args.ppd_dir / str(entry["filename"]), entry, skip=args.skip_verify
            )
            for key, entry in ppd_files.items()
        },
    )
    ((hpi_key, hpi_entry),) = hpi_files.items()
    verified["uk_hpi"] = _registration(
        hpi_manifest,
        {hpi_key: _verify_pin(args.hpi_csv, hpi_entry, skip=args.skip_verify)},
    )
    ((nspl_key, nspl_entry),) = nspl_files.items()
    verified["nspl"] = _registration(
        nspl_manifest,
        {nspl_key: _verify_pin(args.nspl_zip, nspl_entry, skip=args.skip_verify)},
    )
    ((voa_key, voa_entry),) = voa_files.items()
    verified["voa_ctsop"] = _registration(
        voa_manifest,
        {voa_key: _verify_pin(args.voa_zip, voa_entry, skip=args.skip_verify)},
    )
    ppd_vintage = str(next(iter(ppd_files.values())).get("vintage", ""))

    _log("Reading the NSPL ...")
    postcodes, seat_names = _read_nspl(args.nspl_zip)
    postcode_index = pd.Index(postcodes["postcode"])

    cache_path = _dwelling_cache_path(args.cache_dir, ppd_files, stock_yyyymm)
    receipts_path = cache_path.with_suffix(".receipts.json")
    if cache_path.is_file() and receipts_path.is_file():
        _log(f"Reading cached dwellings from {cache_path.name} ...")
        dwellings = pd.read_parquet(cache_path)
        dwelling_receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    else:
        _log("Reducing the Price Paid yearly files ...")
        dwellings, dwelling_receipts = _build_dwellings(
            args.ppd_dir, ppd_files, postcode_index, stock_yyyymm=stock_yyyymm
        )
        dwellings.to_parquet(cache_path, index=False)
        receipts_path.write_text(
            json.dumps(dwelling_receipts, indent=2, sort_keys=True), encoding="utf-8"
        )
        _log(f"Cached {len(dwellings):,} dwellings at {cache_path}")
    if args.stage == "dwellings":
        _log(
            json.dumps(
                {
                    k: v
                    for k, v in dwelling_receipts.items()
                    if k != "transfers_by_year"
                },
                indent=2,
            )
        )
        return 0

    _log("Loading the support, the HPI and the VOA stock ...")
    support = _load_support(
        args.ew_support, args.support_provenance, skip=args.skip_verify
    )
    crosswalk = json.loads(args.crosswalk.read_text(encoding="utf-8"))
    la_aliases = {
        alias: roster
        for roster, entry in crosswalk["levels"]["local_authority"][
            "code_aliases"
        ].items()
        for alias in entry["alias_codes"]
    }
    hpi = hv.load_hpi(args.hpi_csv)
    hpi_area_for_la = {
        roster: aliases[0]
        for roster, aliases in (
            (roster, entry["alias_codes"])
            for roster, entry in crosswalk["levels"]["local_authority"][
                "code_aliases"
            ].items()
        )
    }
    hpi_area_for_la[ISLES_OF_SCILLY] = CORNWALL
    region_hpi = {"W99999999": "W92000004"}
    voa = _read_voa(args.voa_zip)
    voa_stock = _voa_all_properties_by_local_authority(voa, la_aliases)

    _log("Placing and uprating the stock ...")
    stock = hv.dwelling_stock(
        dwellings,
        postcode_oa=postcodes["oa21cd"],
        support=support,
        hpi=hpi,
        hpi_area_for_local_authority=hpi_area_for_la,
        region_hpi_code=region_hpi,
        basis_year=args.basis_year,
    )
    table = stock.table
    # The support carries the W99999999 sentinel for Wales's region; the rows
    # name the nation code the geography catalog knows.
    table["region_code"] = table["region_code"].replace({"W99999999": "W92000004"})
    receipts: dict[str, object] = {"dwellings": {**dwelling_receipts, **stock.receipts}}

    thresholds_basis, deflator = hv.deflate_thresholds(
        hpi,
        thresholds=list(THRESHOLDS.values()),
        nation_area_code="E92000001",
        basis_year=args.basis_year,
        reference_year=reference_year,
        reference_month=reference_month,
    )
    thresholds_by_id = {
        tid: thresholds_basis[float(value)] for tid, value in THRESHOLDS.items()
    }
    receipts["thresholds"] = {
        "reference_month": args.reference_month,
        "basis_year": args.basis_year,
        "england_all_dwellings_deflator": deflator,
        "basis_values": thresholds_by_id,
    }

    # Owner-occupied and private-stock shares from the feed's census tenure facts.
    if args.ledger_facts is None:
        raise SystemExit("--ledger-facts is required for the full build (stage all).")
    facts = _read_facts(args.ledger_facts)
    private_share, owner_share, tenure_receipt = _tenure_shares(facts)
    try:
        obr_share, obr_key = _obr_non_owner_share(facts)
    except ValueError:
        if args.obr_non_owner_share is None:
            raise
        obr_share, obr_key = (
            float(args.obr_non_owner_share),
            "parameter:--obr-non-owner-share",
        )

    # Coverage: the private-sector VOA stock over valued dwellings, per local authority.
    valued_by_la = table.groupby("local_authority_code").size()
    coverage: dict[str, float] = {}
    raw_ratios: dict[str, float] = {}
    for code, count in valued_by_la.items():
        stock_count = voa_stock.get(str(code))
        if stock_count is None:
            raise ValueError(f"VOA CTSOP has no all-properties row for {code}.")
        if str(code) not in private_share:
            raise ValueError(f"No census tenure facts for local authority {code}.")
        raw_ratios[str(code)] = stock_count * private_share[str(code)] / count
        coverage[str(code)] = float(
            np.clip(raw_ratios[str(code)], *COVERAGE_RATIO_BOUNDS)
        )
    receipts["coverage"] = {
        "basis": (
            "VOA CTSOP1.1 all properties at 31 March 2025, less the census social-rented "
            "share, over valued dwellings ever sold since 1995"
        ),
        "bounds": list(COVERAGE_RATIO_BOUNDS),
        "local_authorities": len(coverage),
        "ratio_quantiles": {
            q: float(np.quantile(list(raw_ratios.values()), float(q)))
            for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
        },
        "clipped_low": sorted(
            code for code, r in raw_ratios.items() if r < COVERAGE_RATIO_BOUNDS[0]
        ),
        "clipped_high": sorted(
            code for code, r in raw_ratios.items() if r > COVERAGE_RATIO_BOUNDS[1]
        ),
        "england_wales_stock_voa": float(sum(voa_stock[c] for c in coverage)),
        "england_wales_private_stock": float(
            sum(voa_stock[c] * private_share[c] for c in coverage)
        ),
        "england_wales_valued_dwellings": int(valued_by_la.sum()),
    }
    la_codes = table["local_authority_code"].to_numpy()
    c_weight = np.array([coverage[str(code)] for code in la_codes])

    # Owner-occupied share of the private stock, rescaled to the OBR control on £2m+.
    p_raw = np.array([owner_share[str(code)] for code in la_codes])
    above_2m_england = (
        table["basis_value"].to_numpy() > thresholds_by_id["over_gbp_2m"]
    ) & (table["nation_code"].to_numpy() == "E92000001")
    implied_share = float(
        (p_raw * c_weight)[above_2m_england].sum() / c_weight[above_2m_england].sum()
    )
    control = 1.0 - obr_share
    scale = control / implied_share
    p_adj = np.minimum(1.0, p_raw * scale)
    achieved = float(
        (p_adj * c_weight)[above_2m_england].sum() / c_weight[above_2m_england].sum()
    )
    receipts["owner_occupied"] = {
        **tenure_receipt,
        "obr_non_owner_occupied_share": obr_share,
        "obr_fact_key": obr_key,
        "control_owner_occupied_share_over_2m_england": control,
        "census_implied_share_over_2m_england": implied_share,
        "rescale_factor": float(scale),
        "achieved_share_over_2m_england": achieved,
        "local_authorities_capped_at_one": sorted(
            {
                str(code)
                for code, v in zip(la_codes, p_raw * scale, strict=True)
                if v > 1.0
            }
        ),
    }
    w_owner = p_adj * c_weight
    w_all = c_weight

    _log("Counting and writing the rows ...")
    group_columns = list(LEVELS)
    counts = {
        "owner_occupied_homes": hv.count_above_thresholds(
            table,
            thresholds_basis=thresholds_by_id,
            weight=w_owner,
            group_columns=group_columns,
        ),
        "private_sector_dwellings": hv.count_above_thresholds(
            table,
            thresholds_basis=thresholds_by_id,
            weight=w_all,
            group_columns=group_columns,
        ),
    }
    sensitivity = hv.count_above_thresholds(
        table,
        thresholds_basis=thresholds_by_id,
        weight=p_raw * c_weight,
        group_columns=["nation_code", "region_code"],
    )
    unscaled = hv.count_above_thresholds(
        table,
        thresholds_basis=thresholds_by_id,
        weight=p_adj,
        group_columns=["nation_code", "region_code"],
    )
    la_names = json.loads(args.la_names.read_text(encoding="utf-8"))["areas"]

    def name_for(level_column: str, code: str) -> str:
        if level_column == "constituency_code":
            return seat_names.get(code) or ""
        if level_column == "local_authority_code":
            return (la_names.get(code) or {}).get("name", "")
        return UK_GEOGRAPHY_ID_TO_LABEL.get(code, "")

    inputs_payload = {
        "registrations": verified,
        "support_sha256": support["__sha256__"],
        "method": METHOD_VERSION,
    }
    inputs_sha256 = hashlib.sha256(
        json.dumps(inputs_payload, sort_keys=True).encode()
    ).hexdigest()
    period_value = args.stock_date[:4]
    rows: list[dict[str, object]] = []
    for measure_id, frame in counts.items():
        for record in frame.itertuples(index=False):
            level = LEVELS[record.level]
            name = name_for(record.level, str(record.code))
            if not name:
                raise ValueError(f"No name for {record.level} {record.code}.")
            rows.append(
                _fact_row(
                    level=level["level"],
                    code=str(record.code),
                    name=name,
                    vintage=level["vintage"],
                    measure_id=measure_id,
                    threshold_id=str(record.threshold_id),
                    value=float(record.value),
                    period_value=period_value,
                    stock_date=args.stock_date,
                    basis_year=args.basis_year,
                    reference_month=args.reference_month,
                    inputs_sha256=inputs_sha256,
                    ppd_vintage=ppd_vintage,
                )
            )
    rows.sort(key=lambda row: row["aggregate_fact_key"])
    keys = {row["aggregate_fact_key"] for row in rows}
    if len(keys) != len(rows):
        raise ValueError("Derived fact keys collide.")

    knots_by_level: dict[str, dict[str, object]] = {}
    suppressed_by_level: dict[str, list[str]] = {}
    for column in ("constituency_code", "region_code", "nation_code"):
        knots, suppressed = hv.value_knots(
            table,
            weight=w_owner,
            group_column=column,
            cap=float(args.cap),
            minimum_count=args.knot_minimum,
        )
        knots_by_level[LEVELS[column]["level"]] = {
            k.code: {
                "count": k.count,
                "weight": round(k.weight, 3),
                "quantiles": list(k.quantiles),
            }
            for k in knots
        }
        suppressed_by_level[LEVELS[column]["level"]] = suppressed

    # Receipts: regional shares, parity, chain usage.
    owner_rows = counts["owner_occupied_homes"]
    region_2m = owner_rows[
        (owner_rows["level"] == "region_code")
        & (owner_rows["threshold_id"] == "over_gbp_2m")
    ]
    england_2m = float(region_2m[region_2m["code"].str.startswith("E")]["value"].sum())
    receipts["regional_over_2m_owner_occupied_share"] = {
        str(code): round(float(value) / england_2m, 4)
        for code, value in zip(region_2m["code"], region_2m["value"], strict=True)
        if str(code).startswith("E")
    }
    receipts["national_counts"] = {
        measure_id: {
            f"{record.code}:{record.threshold_id}": round(float(record.value), 1)
            for record in frame[frame["level"] == "nation_code"].itertuples(index=False)
        }
        for measure_id, frame in counts.items()
    }
    receipts["sensitivities"] = {
        "owner_share_without_obr_control": {
            f"{r.code}:{r.threshold_id}": round(float(r.value), 1)
            for r in sensitivity[sensitivity["level"] == "nation_code"].itertuples(
                index=False
            )
        },
        "no_coverage_correction": {
            f"{r.code}:{r.threshold_id}": round(float(r.value), 1)
            for r in unscaled[unscaled["level"] == "nation_code"].itertuples(
                index=False
            )
        },
    }
    hpssa = _hpssa_parity(facts, None)
    receipts["hpssa"] = {
        "seat_fiscal_year_cells_in_feed": hpssa["hpssa_seat_fiscal_year_cells"]
    }
    receipts["knots"] = {
        "cap": args.cap,
        "minimum_count": args.knot_minimum,
        "suppressed": suppressed_by_level,
        "grid": list(hv.KNOT_GRID),
    }
    receipts["hpi"] = {
        "areas": len(hpi.area_codes),
        "last_month": list(hpi.last_month),
        "scilly_fallback": CORNWALL,
    }

    # Benchmark for the release gates: VOA band H dwellings by constituency, LSOA
    # counts apportioned to seats by each output area's share of its LSOA's
    # census households (a public aggregate of two published tables).
    voa_lsoa = voa[voa["geography"] == "LSOA"]
    band_h_by_lsoa = {
        str(code): float(str(value).replace(",", ""))
        for code, value in zip(voa_lsoa["ecode"], voa_lsoa["band_h"], strict=True)
        if str(value) not in ("-", "..", "", "nan")
    }
    oa_lsoa = support["lsoa_code"].astype(str)
    oa_households = support["households"].astype(float)
    lsoa_households = (
        pd.Series(oa_households).groupby(oa_lsoa).transform("sum").to_numpy()
    )
    oa_share = np.where(
        lsoa_households > 0,
        oa_households / np.where(lsoa_households > 0, lsoa_households, 1.0),
        0.0,
    )
    oa_band_h = np.array([band_h_by_lsoa.get(code, 0.0) for code in oa_lsoa]) * oa_share
    band_h_by_seat = (
        pd.Series(oa_band_h).groupby(support["constituency_code"].astype(str)).sum()
    )
    benchmarks = {
        "voa_band_h_by_constituency": {
            str(code): round(float(value), 1)
            for code, value in band_h_by_seat.items()
            if str(code)[0] in "EW"
        },
        "voa_band_h_basis": (
            "VOA CTSOP1.1 band H at 31 March 2025 by 2021 LSOA, apportioned to July 2024 "
            "constituencies by output-area census household shares"
        ),
    }

    facts_payload = {
        "schema_version": 1,
        "kind": RESOURCE_KIND,
        "resource": args.facts_out.name,
        "description": (
            "Microcosm-derived counts of dwellings valued above thresholds, by area, from HM Land "
            "Registry Price Paid Data uprated with the UK HPI and placed through the NSPL and the "
            "committed England-and-Wales atomic support (microcosm#1240, fix 2). Not Chronicle facts."
        ),
        "generator": {
            "tool": "tools/build_uk_high_value_home_stock.py",
            "version": METHOD_VERSION,
        },
        "built_on": args.built_on,
        "inputs": inputs_payload,
        "inputs_sha256": inputs_sha256,
        "parameters": {
            "thresholds_gbp_at_reference_month": dict(THRESHOLDS),
            "reference_month": args.reference_month,
            "basis_year": args.basis_year,
            "frame_basis_thresholds": thresholds_by_id,
            "stock_date": args.stock_date,
            "owner_occupied_share_rule": (
                "Census 2021 TS054 owner-occupied share of the private-sector stock (all "
                "households less social rented) by local authority, rescaled by one factor so "
                "the England share of dwellings over £2m equals one minus the OBR costing's "
                "non-owner-occupied share"
            ),
            "coverage_rule": (
                "VOA CTSOP1.1 all properties over valued dwellings ever sold, per local authority, "
                f"clipped to {list(COVERAGE_RATIO_BOUNDS)}"
            ),
            "bulk_transfer_rule": "transfers sharing completion date, price and postcode across more than one dwelling are not valuations",
            "dwelling_identity": "postcode, PAON and SAON",
            "uprating": "UK HPI by local authority and property type to the calendar-year average, falling back to region then nation, Isles of Scilly to Cornwall",
        },
        "benchmarks": benchmarks,
        "row_count": len(rows),
        "rows_sha256": hv.rows_sha256(rows),
        "rows": rows,
    }
    _write_json(args.facts_out, facts_payload)
    knots_payload = {
        "schema_version": 1,
        "kind": KNOTS_KIND,
        "resource": args.knots_out.name,
        "built_on": args.built_on,
        "inputs_sha256": inputs_sha256,
        "method": METHOD_VERSION,
        "basis_year": args.basis_year,
        "cap": args.cap,
        "minimum_count": args.knot_minimum,
        "weight": "owner-occupied share times coverage correction",
        "grid": list(hv.KNOT_GRID),
        "levels": knots_by_level,
        "suppressed": suppressed_by_level,
    }
    _write_json(args.knots_out, knots_payload)
    provenance = {
        "schema_version": 1,
        "kind": PROVENANCE_KIND,
        "built_on": args.built_on,
        "issues": {
            "umbrella": "PolicyEngine/microcosm#1240",
            "chronicle_registration": "PolicyEngine/chronicle#343",
        },
        "tool": "tools/build_uk_high_value_home_stock.py",
        "method": METHOD_VERSION,
        "inputs": inputs_payload,
        "inputs_sha256": inputs_sha256,
        "ledger_feed": _feed_identity(args.ledger_manifest),
        "parameters": facts_payload["parameters"],
        "receipts": receipts,
    }
    _write_json(args.provenance_out, provenance)
    was_bounds = json.loads(WAS_BOUNDS.read_text(encoding="utf-8"))["bounds"]
    wealth_headroom = float(was_bounds["property_wealth"][1]) - float(
        was_bounds["main_residence_value"][1]
    )
    bounds_payload = {
        "version": 1,
        "country": "uk",
        "policy": (
            "Support bounds for the two household columns the post-geography value "
            "node rewrites (microcosm#1240): main_residence_value is winsorised at the "
            "value knots' declared cap, and property_wealth moves by the same amount, so "
            "its bound is the cap plus the WAS donor bound's headroom over the main "
            "residence. Listed after was_wealth_support_bounds.json in uk_support so "
            "these bounds win for the two columns. Public caps, never donor values."
        ),
        "source": {
            "knots_resource": args.knots_out.name,
            "knots_inputs_sha256": inputs_sha256,
            "cap": args.cap,
            "was_bounds_resource": WAS_BOUNDS.name,
        },
        "overrides": [WAS_BOUNDS.name],
        "bounds": {
            "main_residence_value": [0.0, float(args.cap)],
            "property_wealth": [0.0, float(args.cap) + wealth_headroom],
        },
    }
    _write_json(args.bounds_out, bounds_payload)
    _log(
        f"Wrote {len(rows)} rows, knots for {sum(len(v) for v in knots_by_level.values())} areas."
    )
    return 0


def _feed_identity(manifest_path: Path | None) -> dict[str, object]:
    if manifest_path is None or not manifest_path.is_file():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        key: manifest.get(key)
        for key in ("facts_sha256", "fact_row_count", "schema_version", "source_commit")
        if key in manifest
    }


if __name__ == "__main__":
    raise SystemExit(main())
