"""Build the UK stock cell table and the land under homes by area (microcosm#1524).

The stock of homes in England and Wales is pooled into cells of a small area
(MSOA) × a council tax band. Each cell holds VOA's count of homes there, what
they are worth, and what their buildings would cost to rebuild after
depreciation; land is value less building, never below zero, at one national
cost level κ. Authority and constituency figures are sums of cells.

Inputs, every one public and checked against its Chronicle registration:
HM Land Registry Price Paid (through the home-stock builder's dwelling table
and cache), the UK HPI, the ONS NSPL, VOA's CTSOP 1.1, 3.1 and 4.1, UCL's
Price Paid–EPC linked file and certificates, Tax Policy Associates' model
constants, the committed England-and-Wales atomic support, and facts of the
pinned Chronicle feed. It reads no survey microdata. Addresses stay in the
caches under ``--cache-dir``; the five resources hold aggregates only:

- ``stock_cells_value.json``: per cell, VOA's count, the private and social
  counts, homes seen sold, tenure shares, the mean value and 13 value knots,
  and each cell's split across constituencies;
- ``stock_cells_structure.json``: per cell, floor area, the retained share
  after depreciation, the unit cost and costs and values along the value
  rank, and the central case's land;
- ``la_land_shares.json``: per authority, the stock, value and land under
  every variant and under MHCLG's pattern;
- ``constituency_land_totals.json``: per constituency, the same sums;
- ``stock_cells.provenance.json``: inputs, parameters, the level, receipts,
  the decision rule and the measurements.

Stages: ``discover`` prints UCL's readme and every header against the
declared columns; ``dwellings`` builds or reuses the dwelling cache;
``attributes`` streams UCL's files into their caches; ``cells`` builds the
resources; ``all`` runs the last three. ``--check`` rebuilds and compares
with the committed files instead of writing.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import resource
import sys
import time
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD = REPO_ROOT / "packages/microcosm-build/src/microcosm/build"
UK_PACKAGE = BUILD / "uk"
UK_RUNTIME = BUILD / "uk_runtime"

from microcosm.build import (  # noqa: E402
    cell_pooling,
    derived_resource,
    pinned_inputs,
    residual_land,
    weighted_quantiles,
)


def _load(name: str):
    """Load a ``uk_runtime`` module by path.

    Importing it through ``microcosm.build.uk_runtime`` would load the UK
    country spec, which refuses a package whose declared resources are
    missing: the very resources this tool writes on a first run.
    """
    path = UK_RUNTIME / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_uk_stock_cells_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hv = _load("high_value_home_stock")
voa = _load("voa_stock_tables")
ucl = _load("ucl_price_paid_epc")
ref = _load("land_reference_facts")
declared = _load("stock_cell_declarations")

RESOURCES: Mapping[str, str] = {
    "value": "stock_cells_value.json",
    "structure": "stock_cells_structure.json",
    "authorities": "la_land_shares.json",
    "constituencies": "constituency_land_totals.json",
    "provenance": "stock_cells.provenance.json",
}
KINDS: Mapping[str, str] = {
    "value": "uk_stock_cells_value",
    "structure": "uk_stock_cells_structure",
    "authorities": "uk_la_land_shares",
    "constituencies": "uk_constituency_land_totals",
    "provenance": "uk_stock_cells_provenance",
}
NSPL_DATA_MEMBER = "Data/NSPL_AUG_2026_UK.csv"
SUPPORT_NAME = "uk_ew_output_area_2021"
ENGLAND = "E92000001"
WALES = "W92000004"
#: The support's sentinel for Wales's region; Wales has no regions.
WALES_REGION_SENTINEL = "W99999999"
#: Homes whose latest sale is a new build sold from this year are new
#: builds for the cost check and the like-for-like comparison.
NEW_BUILD_FROM_YEAR = 2015


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _peak_memory_gib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(peak if sys.platform == "darwin" else peak * 1024) / 2**30


@dataclass
class RunLog:
    """Wall time and peak memory per stage, kept beside the caches only."""

    stages: list[dict[str, object]] = field(default_factory=list)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        _log(f"[{name}] ...")
        yield
        self.stages.append(
            {
                "stage": name,
                "seconds": round(time.perf_counter() - start, 1),
                "peak_memory_gib": round(_peak_memory_gib(), 2),
            }
        )
        _log(f"[{name}] done in {self.stages[-1]['seconds']} s")


# ----------------------------------------------------------------------------
# Arguments


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--stage",
        choices=("discover", "dwellings", "attributes", "cells", "all"),
        default="all",
    )
    parser.add_argument(
        "--variants",
        choices=("central", "all"),
        default="central",
        help="'all' also rebuilds the cells for the value-block variants",
    )
    raw = parser.add_argument_group("raw inputs and their Chronicle manifests")
    raw.add_argument("--ppd-dir", type=Path, help="directory holding pp-YYYY.csv")
    raw.add_argument("--ppd-manifest", type=Path)
    raw.add_argument("--hpi-csv", type=Path)
    raw.add_argument("--hpi-manifest", type=Path)
    raw.add_argument("--nspl-zip", type=Path)
    raw.add_argument("--nspl-manifest", type=Path)
    for table in ("ctsop1", "ctsop3", "ctsop4"):
        raw.add_argument(f"--{table}-zip", type=Path)
        raw.add_argument(f"--{table}-manifest", type=Path)
    raw.add_argument("--ucl-bundle-1", type=Path)
    raw.add_argument("--ucl-bundle-2", type=Path)
    raw.add_argument("--ucl-manifest", type=Path)
    raw.add_argument("--tpa-config", type=Path)
    raw.add_argument("--tpa-cells", type=Path)
    raw.add_argument("--tpa-manifest", type=Path)
    committed = parser.add_argument_group("committed and feed inputs")
    committed.add_argument("--ew-support", type=Path)
    committed.add_argument(
        "--support-provenance",
        type=Path,
        default=UK_PACKAGE / "uk_atomic_area_supports.provenance.json",
    )
    committed.add_argument(
        "--crosswalk", type=Path, default=UK_PACKAGE / "local_area_crosswalk.json"
    )
    committed.add_argument(
        "--la-names", type=Path, default=UK_PACKAGE / "local_authority_names.json"
    )
    committed.add_argument(
        "--scotland-dwellings",
        type=Path,
        default=UK_PACKAGE / "scotgov_chargeable_dwellings_by_band.json",
    )
    committed.add_argument(
        "--home-stock-provenance",
        type=Path,
        default=UK_PACKAGE / "high_value_home_stock.provenance.json",
        help="the committed home-stock provenance, whose NSPL its cache indexes",
    )
    committed.add_argument("--ledger-facts", type=Path)
    committed.add_argument("--ledger-manifest", type=Path)
    level = parser.add_argument_group(
        "the level's inputs, used only while the feed lacks them (chronicle#438)"
    )
    for name, spec in declared.LEVEL_INPUTS.items():
        if spec.get("parameter"):
            level.add_argument(
                str(spec["parameter"]),
                dest=f"level_{name}",
                type=float,
                default=None,
                help=f"{name}, in pounds ({spec['request']})",
            )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=UK_PACKAGE)
    parser.add_argument(
        "--built-on",
        help="build date recorded in the resources (YYYY-MM-DD), passed explicitly",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="rebuild and fail if the committed resources differ",
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="threads for Arrow's CSV reader"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="development only: trust the local bytes without hashing them",
    )
    args = parser.parse_args(argv)
    if args.stage in ("cells", "all") and not args.built_on:
        parser.error("--built-on is required for the cells stage")
    return args


def _need(args: argparse.Namespace, *names: str) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        flags = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(f"Stage {args.stage} needs {flags}.")


# ----------------------------------------------------------------------------
# Pins


def _verify_package(
    manifest_path: Path, paths: Mapping[str, Path], *, skip: bool
) -> dict[str, object]:
    """Check each local file against its manifest entry; the record and licence."""
    manifest, files = pinned_inputs.manifest_files(manifest_path)
    unknown = sorted(set(paths) - set(files))
    if unknown:
        raise ValueError(f"{manifest_path} registers no file {unknown}.")
    verified = {
        key: pinned_inputs.verify_pinned_file(path, files[key], skip=skip)
        for key, path in paths.items()
    }
    return {
        "registration": pinned_inputs.registration(manifest, verified),
        "licence": pinned_inputs.licence_of(manifest),
    }


def _single(manifest_path: Path, path: Path, *, skip: bool) -> dict[str, object]:
    _, files = pinned_inputs.manifest_files(manifest_path)
    if len(files) != 1:
        raise ValueError(f"{manifest_path} registers {len(files)} files, not one.")
    return _verify_package(manifest_path, {next(iter(files)): path}, skip=skip)


def _price_paid_files(args: argparse.Namespace) -> tuple[dict, dict[str, dict]]:
    _, files = pinned_inputs.manifest_files(args.ppd_manifest)
    record = _verify_package(
        args.ppd_manifest,
        {key: args.ppd_dir / str(entry["filename"]) for key, entry in files.items()},
        skip=args.skip_verify,
    )
    return record, files


def _ucl_record(args: argparse.Namespace, bundles: Sequence[str]) -> dict[str, object]:
    paths = {"bundle_1": args.ucl_bundle_1, "bundle_2": args.ucl_bundle_2}
    return _verify_package(
        args.ucl_manifest, {key: paths[key] for key in bundles}, skip=args.skip_verify
    )


def _sha_of(record: Mapping[str, object], key: str | None = None) -> str:
    files = record["registration"]["files"]
    entry = files[key] if key is not None else next(iter(files.values()))
    return str(entry["sha256"])


# ----------------------------------------------------------------------------
# Readers with caches


def _nspl(nspl_zip: Path, sha256: str, cache_dir: Path) -> pd.DataFrame:
    """The NSPL's postcodes and output areas, cached by the zip's sha256."""
    path = cache_dir / f"nspl-{sha256[:16]}.parquet"
    if path.is_file():
        return pd.read_parquet(path)
    with zipfile.ZipFile(nspl_zip) as archive, archive.open(NSPL_DATA_MEMBER) as stream:
        postcodes = hv.read_nspl_postcode_output_areas(stream)
    postcodes.to_parquet(path, index=False)
    return postcodes


def _support(path: Path, provenance_path: Path, *, skip: bool) -> dict[str, np.ndarray]:
    payload = path.read_bytes()
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    expected = provenance["supports"][SUPPORT_NAME]["sha256"]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected and not skip:
        raise ValueError("The support's sha256 differs from the committed provenance.")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: np.asarray(archive[name]) for name in metadata["columns"]}
    return arrays | {"__sha256__": np.asarray(digest)}


def _aliases(crosswalk_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Recoded authorities: alias → roster code, and roster → its first alias."""
    crosswalk = json.loads(crosswalk_path.read_text(encoding="utf-8"))
    entries = crosswalk["levels"]["local_authority"]["code_aliases"]
    to_roster = {
        alias: roster
        for roster, entry in entries.items()
        for alias in entry["alias_codes"]
    }
    to_alias = {roster: entry["alias_codes"][0] for roster, entry in entries.items()}
    return to_roster, to_alias


def _cache_name(prefix: str, payload: Mapping[str, object]) -> str:
    return f"{prefix}-{derived_resource.payload_sha256(payload)[:16]}.parquet"


def _write_cache(path: Path, table: pd.DataFrame, receipts: Mapping) -> None:
    table.to_parquet(path, index=False)
    path.with_suffix(".receipts.json").write_text(
        json.dumps(receipts, indent=2, sort_keys=True), encoding="utf-8"
    )


def _read_cache(path: Path) -> tuple[pd.DataFrame, dict] | None:
    receipts = path.with_suffix(".receipts.json")
    if not (path.is_file() and receipts.is_file()):
        return None
    return pd.read_parquet(path), json.loads(receipts.read_text(encoding="utf-8"))


def _linked_columns():
    return ucl.LinkedColumns(**declared.UCL_LINKED_COLUMNS)


def _certificate_columns():
    return ucl.CertificateColumns(**declared.UCL_CERTIFICATE_COLUMNS)


def _attribute_cache_paths(args, ucl_record) -> tuple[Path, Path]:
    common = {
        "method": declared.METHOD_VERSION,
        "age_bands": [list(band) for band in declared.EPC_AGE_BANDS],
        "floor_area_bounds": list(declared.FLOOR_AREA_BOUNDS_M2),
        "pandas": pd.__version__,
    }
    linked = args.cache_dir / _cache_name(
        "ucl-linked",
        {
            **common,
            "bundle": _sha_of(ucl_record, "bundle_1"),
            "columns": declared.UCL_LINKED_COLUMNS,
        },
    )
    social = args.cache_dir / _cache_name(
        "ucl-social",
        {
            **common,
            "bundle": _sha_of(ucl_record, "bundle_2"),
            "columns": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in declared.UCL_CERTIFICATE_COLUMNS.items()
            },
            "authorities": _certificate_authorities(args),
            "rule": "the latest certificate per building id; one without an id counts once",
        },
    )
    return linked, social


def _certificate_authorities(args: argparse.Namespace) -> dict[str, str]:
    """A certificate's council on the roster: recoded ones and merged ones."""
    to_roster, _ = _aliases(args.crosswalk)
    return {**to_roster, **declared.AUTHORITY_PREDECESSORS}


# ----------------------------------------------------------------------------
# Stage: discover


def _discover(args: argparse.Namespace) -> dict[str, object]:
    """Print UCL's readme and every header against the declarations."""
    report: dict[str, object] = {}
    if args.ucl_bundle_1 is not None:
        name, text, digest = ucl.readme_text(args.ucl_bundle_1)
        report["readme"] = {"member": name, "sha256": digest}
        _log(f"UCL readme {name} (sha256 {digest}):")
        _log(text if text is not None else "  (not a text document)")
        report["bundle_1_members"] = ucl.bundle_members(args.ucl_bundle_1)
        with ucl.open_inner_zip(args.ucl_bundle_1, ucl.LINKED_MEMBER) as inner:
            for member in sorted(
                n for n in inner.namelist() if n.lower().endswith(".csv")
            ):
                header = ucl.csv_header(inner, member, encoding="utf8")
                declared_names = _linked_columns().names()
                report.setdefault("linked_headers", {})[member] = {
                    "header": header,
                    "declared_missing": [n for n in declared_names if n not in header],
                }
    if args.ucl_bundle_2 is not None:
        report["bundle_2_members"] = ucl.bundle_members(args.ucl_bundle_2)
        with ucl.open_inner_zip(args.ucl_bundle_2, ucl.EPC_MEMBER) as inner:
            members = sorted(n for n in inner.namelist() if n.lower().endswith(".csv"))
            for member in members[:3]:
                header = ucl.csv_header(inner, member, encoding="utf8")
                declared_names = _certificate_columns().names()
                report.setdefault("certificate_headers", {})[member] = {
                    "header": header,
                    "declared_missing": [n for n in declared_names if n not in header],
                }
            report["certificate_csv_members"] = len(members)
    tables = (
        ("ctsop1", voa.CTSOP1_1_MEMBER, None),
        ("ctsop3", voa.CTSOP3_1_MEMBER, voa.classify_type_columns),
        ("ctsop4", voa.CTSOP4_1_MEMBER, voa.classify_build_periods),
    )
    for name, member, classify in tables:
        path = getattr(args, f"{name}_zip")
        if path is None:
            continue
        header = voa.member_header(path, member)
        entry: dict[str, object] = {"header": header}
        if classify is not None:
            try:
                entry["classified"] = {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in classify(header).items()
                }
            except ValueError as error:
                entry["error"] = str(error)
        report[name] = entry
    _log(json.dumps(report, indent=2, sort_keys=True))
    args.cache_dir.joinpath("discover.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


# ----------------------------------------------------------------------------
# Stage: dwellings


def _committed_nspl_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    provenance = json.loads(path.read_text(encoding="utf-8"))
    nspl = ((provenance.get("inputs") or {}).get("registrations") or {}).get("nspl")
    files = (nspl or {}).get("files") or {}
    return str(next(iter(files.values()))["sha256"]) if len(files) == 1 else None


def _dwellings(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """The dwelling table, from the shared cache or a Price Paid pass."""
    _need(args, "ppd_dir", "ppd_manifest", "nspl_zip", "nspl_manifest")
    ppd_record, ppd_files = _price_paid_files(args)
    nspl_record = _single(args.nspl_manifest, args.nspl_zip, skip=args.skip_verify)
    nspl_sha = _sha_of(nspl_record)
    postcodes = _nspl(args.nspl_zip, nspl_sha, args.cache_dir)
    stock_yyyymm = hv.stock_yyyymm_from_date(declared.DWELLING_STOCK_DATE)
    cache, status = hv.resolve_dwelling_cache(
        args.cache_dir,
        ppd_files,
        stock_yyyymm=stock_yyyymm,
        method=declared.DWELLING_CACHE_METHOD,
        nspl_sha256=nspl_sha,
        legacy_nspl_sha256=_committed_nspl_sha256(args.home_stock_provenance),
    )
    if status == "absent":
        _log("Reducing the Price Paid yearly files ...")
        dwellings, receipts = hv.build_dwelling_table(
            args.ppd_dir,
            ppd_files,
            pd.Index(postcodes["postcode"]),
            stock_yyyymm=stock_yyyymm,
            log=_log,
        )
        _write_cache(cache, dwellings, receipts)
    else:
        _log(f"Reading the cached dwellings ({status}) from {cache.name} ...")
        dwellings, receipts = _read_cache(cache)
    records = {"price_paid": ppd_record, "nspl": nspl_record}
    # Whether the cache was built or reused depends on the machine, not the
    # inputs, so it is logged and never committed.
    _log(f"Dwelling cache: {status}")
    return dwellings, records, receipts, postcodes


# ----------------------------------------------------------------------------
# Stage: attributes


def _attributes(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """UCL's per-dwelling attributes and social certificate cells, cached."""
    _need(args, "ucl_bundle_1", "ucl_bundle_2", "ucl_manifest")
    record = _ucl_record(args, ("bundle_1", "bundle_2"))
    linked_path, social_path = _attribute_cache_paths(args, record)
    linked = _read_cache(linked_path)
    if linked is None:
        columns = _linked_columns()
        transaction_keys = None
        if not columns.has_address:
            _need(args, "ppd_dir", "ppd_manifest")
            _, ppd_files = _price_paid_files(args)
            transaction_keys = pd.concat(
                [
                    ucl.price_paid_transaction_keys(
                        args.ppd_dir / str(entry["filename"]),
                        key_function=hv.dwelling_key,
                    )
                    for _, entry in sorted(ppd_files.items())
                ],
                ignore_index=True,
            )
        _log("Streaming UCL's linked file ...")
        table, receipts = ucl.linked_attributes(
            args.ucl_bundle_1,
            columns,
            age_bands=declared.EPC_AGE_BANDS,
            floor_area_bounds=declared.FLOOR_AREA_BOUNDS_M2,
            key_function=hv.dwelling_key,
            transaction_keys=transaction_keys,
            log=_log,
        )
        _write_cache(linked_path, table, receipts)
        linked = (table, receipts)
    social = _read_cache(social_path)
    if social is None:
        _log("Streaming UCL's certificates for social rented homes ...")
        table, receipts = ucl.social_certificate_cells(
            args.ucl_bundle_2,
            _certificate_columns(),
            age_bands=declared.EPC_AGE_BANDS,
            floor_area_bounds=declared.FLOOR_AREA_BOUNDS_M2,
            authority_aliases=_certificate_authorities(args),
            log=_log,
        )
        _write_cache(social_path, table, receipts)
        social = (table, receipts)
    receipts = {"linked": linked[1], "social": social[1]}
    return linked[0], social[0], receipts, {"ucl": record}


# ----------------------------------------------------------------------------
# Geography


@dataclass(frozen=True)
class Geography:
    """The support's areas as integer indices, finest first."""

    msoa_codes: np.ndarray
    msoa_of_oa: np.ndarray
    la_codes: np.ndarray
    la_of_msoa: np.ndarray
    region_codes: np.ndarray
    region_of_la: np.ndarray
    nation_codes: np.ndarray
    nation_of_region: np.ndarray
    seat_codes: np.ndarray
    seat_of_oa: np.ndarray
    households: np.ndarray

    @property
    def bands(self) -> int:
        return len(declared.BANDS)


def _geography(support: Mapping[str, np.ndarray]) -> Geography:
    msoa = support["msoa_code"].astype(str)
    la = support["local_authority_code"].astype(str)
    region = np.where(
        np.char.startswith(la, "W"), WALES, support["region_code"].astype(str)
    )
    msoa_codes, msoa_of_oa = np.unique(msoa, return_inverse=True)
    pairs = pd.DataFrame({"m": msoa_of_oa, "la": la}).drop_duplicates()
    if pairs["m"].duplicated().any():
        raise ValueError("An MSOA spans two authorities in the support.")
    msoa_la = pairs.set_index("m")["la"].sort_index().to_numpy()
    la_codes, la_of_msoa = np.unique(msoa_la, return_inverse=True)
    regions = pd.DataFrame({"la": la, "region": region}).drop_duplicates()
    if regions["la"].duplicated().any():
        raise ValueError("An authority spans two regions in the support.")
    la_region = regions.set_index("la").loc[la_codes, "region"].to_numpy()
    region_codes, region_of_la = np.unique(la_region, return_inverse=True)
    nation_codes = np.array([ENGLAND, WALES])
    nation_of_region = np.where(np.char.startswith(region_codes.astype(str), "W"), 1, 0)
    seat_codes, seat_of_oa = np.unique(
        support["constituency_code"].astype(str), return_inverse=True
    )
    return Geography(
        msoa_codes=msoa_codes,
        msoa_of_oa=msoa_of_oa,
        la_codes=la_codes,
        la_of_msoa=la_of_msoa,
        region_codes=region_codes,
        region_of_la=region_of_la,
        nation_codes=nation_codes,
        nation_of_region=nation_of_region,
        seat_codes=seat_codes,
        seat_of_oa=seat_of_oa,
        households=support["households"].astype(float),
    )


def _align(codes: Sequence[str], wanted: np.ndarray) -> np.ndarray:
    return pd.Index(list(codes)).get_indexer(wanted.astype(str))


# ----------------------------------------------------------------------------
# The VOA stock and its tenure


@dataclass
class Stock:
    """VOA's counts by MSOA × band, its type and build-period mix, and tenure."""

    counts: np.ndarray  # msoa × band, as printed
    all_properties: np.ndarray  # msoa
    type_shares: np.ndarray  # msoa × band × type
    period_shares: np.ndarray  # msoa × band × period
    period_spans: tuple
    group_shares: np.ndarray  # msoa × band × declared build-period group
    la_all_properties: np.ndarray  # authority
    engwal_all_properties: float
    receipts: dict


def _stock(args, geo: Geography, to_roster, to_alias) -> Stock:
    suppressed = declared.VOA_SUPPRESSED_COUNT
    table1 = voa.read_member(args.ctsop1_zip, voa.CTSOP1_1_MEMBER)
    msoa = voa.band_counts(table1, geography="MSOA", suppressed_value=suppressed)
    laua = voa.band_counts(table1, geography="LAUA", suppressed_value=suppressed)
    engwal = voa.band_counts(table1, geography="ENGWAL", suppressed_value=suppressed)
    rows = _align(msoa.codes, geo.msoa_codes)
    counts = np.where(rows[:, None] >= 0, msoa.counts[np.maximum(rows, 0)], 0.0)
    total = np.where(rows >= 0, msoa.all_properties[np.maximum(rows, 0)], 0.0)
    la_rows = _align([to_roster.get(code, code) for code in laua.codes], geo.la_codes)
    la_all = np.where(la_rows >= 0, laua.all_properties[np.maximum(la_rows, 0)], np.nan)
    if np.isnan(la_all).any():
        missing = geo.la_codes[np.isnan(la_all)].tolist()
        raise ValueError(f"CTSOP 1.1 has no LAUA row for {missing}.")
    del table1
    parent_of = {
        str(code): to_alias.get(str(geo.la_codes[la]), str(geo.la_codes[la]))
        for code, la in zip(geo.msoa_codes, geo.la_of_msoa, strict=True)
    }
    table3 = voa.read_member(args.ctsop3_zip, voa.CTSOP3_1_MEMBER)
    types = voa.type_shares(
        table3, geography="MSOA", suppressed_value=suppressed, parent_of=parent_of
    )
    del table3
    trow = _align(types.codes, geo.msoa_codes)
    shares = np.where(trow[:, None, None] >= 0, types.shares[np.maximum(trow, 0)], 0.25)
    table4 = voa.read_member(args.ctsop4_zip, voa.CTSOP4_1_MEMBER)
    periods = voa.build_periods(table4, geography="MSOA", suppressed_value=suppressed)
    del table4
    prow = _align(periods.codes, geo.msoa_codes)
    period_counts = np.where(
        prow[:, None, None] >= 0, periods.counts[np.maximum(prow, 0)], 0.0
    )
    totals = period_counts.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        period_shares = np.where(totals > 0, period_counts / totals, 0.0)
    # The declared build-period groups, over the periods of known age.
    group_of_period = {
        name: g
        for g, (_, names, _) in enumerate(declared.BUILD_PERIOD_GROUPS)
        for name in names
    }
    ungrouped = [
        name
        for name, span in zip(periods.periods, periods.spans, strict=True)
        if name not in group_of_period and span != (None, None)
    ]
    if ungrouped:
        raise ValueError(f"CTSOP 4.1 periods in no declared group: {ungrouped}")
    groups = len(declared.BUILD_PERIOD_GROUPS)
    group_counts = np.zeros((*period_counts.shape[:2], groups))
    for k, name in enumerate(periods.periods):
        if name in group_of_period:
            group_counts[..., group_of_period[name]] += period_counts[..., k]
    group_totals = group_counts.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        group_shares = np.where(group_totals > 0, group_counts / group_totals, 0.0)
    receipts = {
        "msoas_in_support": int(len(geo.msoa_codes)),
        "msoas_without_ctsop1_1_row": int((rows < 0).sum()),
        "msoas_without_ctsop3_1_row": int((trow < 0).sum()),
        "msoas_without_ctsop4_1_row": int((prow < 0).sum()),
        "ctsop1_1_msoa_rows_outside_support": int(
            len(set(msoa.codes) - set(geo.msoa_codes.tolist()))
        ),
        "ctsop1_1_imputed_cells": msoa.suppressed_cells,
        "ctsop3_1": types.receipt,
        "ctsop4_1": periods.receipt,
        "stock_msoa_rows": float(total.sum()),
        "stock_band_cells": float(counts.sum()),
    }
    return Stock(
        counts=counts,
        all_properties=total,
        type_shares=shares,
        period_shares=period_shares,
        period_spans=periods.spans,
        group_shares=group_shares,
        la_all_properties=la_all,
        engwal_all_properties=float(engwal.all_properties.sum()),
        receipts=receipts,
    )


@dataclass
class Tenure:
    social_share_of_msoa: np.ndarray
    social_of_la: np.ndarray
    private_share_of_la: np.ndarray
    owner_share_of_la: np.ndarray
    status: str
    receipt: dict


def _tenure(facts, geo: Geography, stock: Stock, *, mode: str) -> Tenure:
    """Social homes per authority by the home-stock rule, spread over its MSOAs.

    An authority's social homes are VOA's count there × Census 2021's social
    rented share (TS054). They spread across its MSOAs in proportion to each
    MSOA's own census social share × its count when the feed carries TS054 by
    MSOA, else to its count alone (the authority's share, uniformly).
    Within an MSOA every band loses the same share.
    """
    private_la, owner_la, receipt = hv.tenure_shares(facts)
    missing = [code for code in geo.la_codes if code not in private_la]
    if missing:
        raise ValueError(f"The feed has no TS054 tenure for {missing}.")
    private = np.array([private_la[code] for code in geo.la_codes])
    owner = np.array([owner_la[code] for code in geo.la_codes])
    social_la = stock.la_all_properties * (1.0 - private)
    private_msoa, _, msoa_receipt = hv.tenure_shares(facts, level="msoa")
    status = "msoa" if (mode == "msoa" and private_msoa) else "uniform"
    if status == "msoa":
        msoa_private = np.array(
            [private_msoa.get(code, np.nan) for code in geo.msoa_codes]
        )
        fallback = private[geo.la_of_msoa]
        msoa_private = np.where(np.isnan(msoa_private), fallback, msoa_private)
        weight = stock.all_properties * (1.0 - msoa_private)
    else:
        weight = stock.all_properties * (1.0 - private[geo.la_of_msoa])
    la_weight = np.bincount(geo.la_of_msoa, weights=weight, minlength=len(geo.la_codes))
    with np.errstate(divide="ignore", invalid="ignore"):
        social_msoa = np.where(
            la_weight[geo.la_of_msoa] > 0,
            social_la[geo.la_of_msoa] * weight / la_weight[geo.la_of_msoa],
            0.0,
        )
        share = np.where(
            stock.all_properties > 0, social_msoa / stock.all_properties, 0.0
        )
    clipped = int((share > 1).sum())
    share = np.clip(share, 0.0, 1.0)
    receipt = {
        **receipt,
        "status": status,
        "requested": mode,
        "msoa_facts": bool(private_msoa),
        "msoa_facts_receipt": msoa_receipt if private_msoa else None,
        "msoas_with_social_share_clipped_at_one": clipped,
        "social_homes": float(social_la.sum()),
        "rule": (
            "social homes per authority = VOA all properties × Census 2021 TS054 "
            "social rented share; spread over its MSOAs by "
            + (
                "each MSOA's TS054 social share × its count"
                if status == "msoa"
                else "each MSOA's count (the authority's share, uniformly)"
            )
        ),
    }
    return Tenure(share, social_la, private, owner, status, receipt)


# ----------------------------------------------------------------------------
# Homes, bands and weights


@dataclass
class Homes:
    """The valued home table, one row per dwelling seen sold."""

    key: np.ndarray
    value: np.ndarray
    ptype: np.ndarray
    msoa: np.ndarray
    oa: np.ndarray
    floor_area: np.ndarray
    age_band: np.ndarray
    new_build: np.ndarray
    receipts: dict


def _homes(
    args,
    dwellings: pd.DataFrame,
    postcodes: pd.DataFrame,
    support,
    geo: Geography,
    linked: pd.DataFrame,
    hpi,
) -> Homes:
    to_roster, to_alias = _aliases(args.crosswalk)
    hpi_area = {roster: alias for roster, alias in to_alias.items()}
    hpi_area[declared.ISLES_OF_SCILLY] = declared.CORNWALL
    stock = hv.dwelling_stock(
        dwellings,
        postcode_oa=postcodes["oa21cd"],
        support=support,
        hpi=hpi,
        hpi_area_for_local_authority=hpi_area,
        region_hpi_code={WALES_REGION_SENTINEL: WALES},
        basis_year=declared.BASIS_YEAR,
        include_support_row=True,
    )
    table = stock.table
    keys = table["key"].to_numpy()
    oa = table["support_row"].to_numpy().astype(np.int64)
    linked_keys = linked["key"].to_numpy()
    position = np.searchsorted(linked_keys, keys)
    position = np.minimum(position, max(len(linked_keys) - 1, 0))
    matched = (
        (linked_keys[position] == keys)
        if len(linked_keys)
        else np.zeros(len(keys), bool)
    )
    floor = np.where(matched, linked["floor_area"].to_numpy()[position], np.nan)
    age = np.where(matched, linked["age_band"].to_numpy()[position], -1).astype(np.int8)
    # A home whose latest sale is its first, recorded as a new build.
    dwell_keys = dwellings["key"].to_numpy()
    where = np.searchsorted(dwell_keys, keys)
    first_new = dwellings["first_newbuild"].to_numpy()[where]
    first_month = dwellings["first_yyyymm"].to_numpy()[where]
    new_build = (
        first_new
        & (first_month == table["sale_yyyymm"].to_numpy())
        & (table["sale_yyyymm"].to_numpy() // 100 >= NEW_BUILD_FROM_YEAR)
    )
    receipts = {
        **stock.receipts,
        "homes_linked_to_ucl": int(matched.sum()),
        "homes_with_floor_area": int(np.isfinite(floor).sum()),
        "homes_with_age_band": int((age >= 0).sum()),
        "new_builds": int(new_build.sum()),
    }
    return Homes(
        key=keys,
        value=table["basis_value"].to_numpy(dtype=float),
        ptype=table["ptype"].to_numpy().astype(np.int64),
        msoa=geo.msoa_of_oa[oa],
        oa=oa,
        floor_area=floor.astype(float),
        age_band=age.astype(np.int64),
        new_build=new_build,
        receipts=receipts,
    )


def _group_of_age_band() -> np.ndarray:
    """Each EPC age band's declared build-period group."""
    labels = [label for label, _, _ in declared.EPC_AGE_BANDS]
    out = np.full(len(labels), -1, dtype=np.int64)
    for group, (_, _, bands) in enumerate(declared.BUILD_PERIOD_GROUPS):
        for label in bands:
            out[labels.index(label)] = group
    if (out < 0).any():
        missing = [labels[i] for i in np.flatnonzero(out < 0)]
        raise ValueError(f"EPC age bands in no build-period group: {missing}")
    return out


@dataclass
class Assignment:
    """Each home's band, weight and cell, under one stratification."""

    used: np.ndarray  # bool per home
    band: np.ndarray  # per used home
    weight: np.ndarray
    cell: np.ndarray
    strata: str
    receipt: dict


def _assign(homes: Homes, stock: Stock, tenure: Tenure, *, strata: str) -> Assignment:
    """Bands by value rank against the private quota, and stratum weights.

    The private quota of an MSOA × band × type is VOA's count × the type's
    share × the MSOA's private share. Within each MSOA × type (or MSOA alone,
    or MSOA × build period), homes take bands by value rank against it, and
    each home weighs its stratum's quota over the stratum's homes. A home in
    a stratum VOA counts no home in is left out and counted.
    """
    bands = len(declared.BANDS)
    private = stock.counts * (1.0 - tenure.social_share_of_msoa[:, None])
    msoas = len(private)
    if strata == "type":
        quota = (private[:, :, None] * stock.type_shares).transpose(0, 2, 1)
        groups = homes.msoa * 4 + homes.ptype
        quota = quota.reshape(msoas * 4, bands)
        eligible = np.ones(len(groups), dtype=bool)
    elif strata == "msoa":
        quota = private
        groups = homes.msoa
        eligible = np.ones(len(groups), dtype=bool)
    elif strata == "build_period":
        periods = stock.group_shares.shape[2]
        quota = (private[:, :, None] * stock.group_shares).transpose(0, 2, 1)
        quota = quota.reshape(msoas * periods, bands)
        lookup = _group_of_age_band()
        period = np.where(
            homes.age_band >= 0, lookup[np.maximum(homes.age_band, 0)], -1
        )
        eligible = period >= 0
        groups = homes.msoa * periods + np.maximum(period, 0)
    else:
        raise ValueError(f"Unknown strata {strata!r}.")
    has_quota = quota.sum(axis=1)[groups] > 0
    used = eligible & has_quota
    band = cell_pooling.assign_levels_by_rank(
        homes.value[used], homes.key[used], groups[used], quota
    )
    weight, unrepresented = cell_pooling.stratum_weights(groups[used], band, quota)
    receipt = {
        "strata": strata,
        "homes_used": int(used.sum()),
        "homes_without_stratum_quota": int((eligible & ~has_quota).sum()),
        "homes_without_build_period": int((~eligible).sum()),
        "private_homes": float(private.sum()),
        "private_homes_represented": float(weight.sum()),
        "private_homes_unrepresented": float(unrepresented.sum()),
    }
    return Assignment(
        used=used,
        band=band,
        weight=weight,
        cell=homes.msoa[used] * bands + band,
        strata=strata,
        receipt=receipt,
    )


# ----------------------------------------------------------------------------
# Pools


@dataclass
class Pools:
    """Each home's pool at every level, and where each cell reads from."""

    ids: list[np.ndarray]  # per level, per used home
    sizes: list[int]
    parents: list[np.ndarray]
    sampled: list[np.ndarray]
    means_level: np.ndarray
    means_index: np.ndarray
    knots_level: np.ndarray
    knots_index: np.ndarray
    knots_below: np.ndarray


def _pools(geo: Geography, assignment: Assignment) -> Pools:
    bands = len(declared.BANDS)
    msoas, las = len(geo.msoa_codes), len(geo.la_codes)
    regions, nations = len(geo.region_codes), len(geo.nation_codes)
    sizes = [msoas * bands, las * bands, regions * bands, nations * bands]
    band_of = np.tile(np.arange(bands), msoas)
    msoa_of = np.repeat(np.arange(msoas), bands)
    parents = [
        geo.la_of_msoa[msoa_of] * bands + band_of,
        geo.region_of_la[np.repeat(np.arange(las), bands)] * bands
        + np.tile(np.arange(bands), las),
        geo.nation_of_region[np.repeat(np.arange(regions), bands)] * bands
        + np.tile(np.arange(bands), regions),
    ]
    ids = [assignment.cell]
    for parent in parents:
        ids.append(parent[ids[-1]])
    sampled = [
        np.bincount(level, minlength=size)
        for level, size in zip(ids, sizes, strict=True)
    ]
    means_level, means_index, _ = cell_pooling.apply_fallback(
        sampled, parents, minimum=declared.POOLING.minimum("means")
    )
    knots_level, knots_index, knots_below = cell_pooling.apply_fallback(
        sampled, parents, minimum=declared.POOLING.minimum("knots")
    )
    return Pools(
        ids=ids,
        sizes=sizes,
        parents=parents,
        sampled=sampled,
        means_level=means_level,
        means_index=means_index,
        knots_level=knots_level,
        knots_index=knots_index,
        knots_below=knots_below,
    )


def _read_at(
    levels: Sequence[np.ndarray], level: np.ndarray, index: np.ndarray
) -> np.ndarray:
    """Each cell's value from the pool its fallback chose."""
    out = np.empty(len(level), dtype=float)
    for k, values in enumerate(levels):
        chosen = level == k
        out[chosen] = values[index[chosen]]
    return out


def _pooled_means(
    pools: Pools, weight: np.ndarray, column: np.ndarray
) -> list[np.ndarray]:
    return [
        cell_pooling.pool(ids, weight, {"x": column}, n_pools=size).means["x"]
        for ids, size in zip(pools.ids, pools.sizes, strict=True)
    ]


def _cell_mean(pools: Pools, weight: np.ndarray, column: np.ndarray) -> np.ndarray:
    return _read_at(
        _pooled_means(pools, weight, column), pools.means_level, pools.means_index
    )


def _attribute_mean(
    pools: Pools, weight: np.ndarray, column: np.ndarray, present: np.ndarray
) -> np.ndarray:
    """A cell's mean of an attribute some homes lack, with its own fallback."""
    carried = [
        np.bincount(ids[present], minlength=size)
        for ids, size in zip(pools.ids, pools.sizes, strict=True)
    ]
    level, index, _ = cell_pooling.apply_fallback(
        carried, pools.parents, minimum=declared.POOLING.minimum("means")
    )
    filled = np.where(present, column, 0.0)
    means = _pooled_means(pools, weight * present, filled)
    return _read_at(means, level, index)


def _knots(
    pools: Pools, assignment: Assignment, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Each cell's 13 knots and the exact mean of its capped values."""
    grid = np.asarray(declared.KNOT_GRID)
    cap = declared.KNOT_CAP
    capped = np.minimum(values, cap)
    knots = np.zeros((pools.sizes[0], len(grid)))
    exact = np.zeros(pools.sizes[0])
    for k, ids in enumerate(pools.ids):
        cells = np.flatnonzero(pools.knots_level == k)
        if not len(cells):
            continue
        needed = np.unique(pools.knots_index[cells])
        order = np.argsort(ids, kind="stable")
        sorted_ids = ids[order]
        starts = np.searchsorted(sorted_ids, needed, side="left")
        stops = np.searchsorted(sorted_ids, needed, side="right")
        table = {}
        for pool_id, start, stop in zip(needed, starts, stops, strict=True):
            rows = order[start:stop]
            if not len(rows):
                continue
            quantiles = weighted_quantiles.weighted_quantiles(
                capped[rows], assignment.weight[rows], grid
            )
            table[int(pool_id)] = (
                np.round(quantiles, 2),
                float(np.average(capped[rows], weights=assignment.weight[rows])),
            )
        for cell in cells:
            entry = table.get(int(pools.knots_index[cell]))
            if entry is not None:
                knots[cell], exact[cell] = entry
    return knots, exact


def _rank_bins(
    pools: Pools, assignment: Assignment, values: np.ndarray
) -> list[np.ndarray]:
    """Each home's value-rank bin within its pool, at every level."""
    bins = declared.RANK_BINS
    out = []
    for ids in pools.ids:
        order = np.lexsort((values, ids))
        sorted_ids = ids[order]
        weight = assignment.weight[order]
        cumulative = np.cumsum(weight)
        starts = np.searchsorted(sorted_ids, sorted_ids, side="left")
        before = np.where(starts > 0, cumulative[np.maximum(starts - 1, 0)], 0.0)
        totals = np.bincount(
            sorted_ids,
            weights=weight,
            minlength=int(sorted_ids.max()) + 1 if len(sorted_ids) else 1,
        )
        position = (cumulative - before - weight / 2.0) / totals[sorted_ids]
        sorted_bins = np.minimum((position * bins).astype(np.int64), bins - 1)
        # Bins run 0 to 9, so one byte per home keeps 15 million homes small.
        level_bins = np.empty(len(ids), dtype=np.int8)
        level_bins[order] = sorted_bins
        out.append(level_bins)
    return out


def _binned_means(
    pools: Pools, weight: np.ndarray, column: np.ndarray, bins: list[np.ndarray]
) -> np.ndarray:
    """Each cell's mean of ``column`` in every rank bin of the pool it reads."""
    count = declared.RANK_BINS
    out = np.zeros((pools.sizes[0], count))
    for k, (ids, size) in enumerate(zip(pools.ids, pools.sizes, strict=True)):
        flat = ids * count + bins[k]
        total = np.bincount(flat, weights=weight * column, minlength=size * count)
        mass = np.bincount(flat, weights=weight, minlength=size * count)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = np.where(mass > 0, total / mass, np.nan).reshape(size, count)
        chosen = pools.means_level == k
        out[chosen] = means[pools.means_index[chosen]]
    return out


# ----------------------------------------------------------------------------
# Costs


@dataclass
class Costs:
    cost_per_m2: dict[str, float]
    region_factor: dict[str, float]
    receipt: dict


def _costs(tpa_config: Path, geo: Geography, stock: Stock) -> Costs:
    """TPA's relative costs; Wales takes England's stock-weighted mean factor."""
    costs, factors = ref.parse_tpa_constants(tpa_config.read_text(encoding="utf-8"))
    region_stock = np.bincount(
        geo.region_of_la[geo.la_of_msoa],
        weights=stock.all_properties,
        minlength=len(geo.region_codes),
    )
    english = [i for i, code in enumerate(geo.region_codes) if code in factors]
    weights = region_stock[english]
    wales = float(
        np.average([factors[geo.region_codes[i]] for i in english], weights=weights)
        if weights.sum() > 0
        else np.mean(list(factors.values()))
    )
    region_factor = {
        str(code): float(factors.get(str(code), wales)) for code in geo.region_codes
    }
    return Costs(
        cost_per_m2=costs,
        region_factor=region_factor,
        receipt={
            "wales_factor_rule": declared.WALES_REGION_FACTOR_RULE,
            "wales_factor": round(wales, 6),
        },
    )


def _age_years(bands: np.ndarray) -> np.ndarray:
    middle = np.array([(first + last) / 2 for _, first, last in declared.EPC_AGE_BANDS])
    return np.where(
        bands >= 0, declared.BUILDING_AGE_AT_YEAR - middle[np.maximum(bands, 0)], np.nan
    )


def _shape(profile: Mapping[str, object], age: np.ndarray) -> np.ndarray:
    kind = profile["shape"]
    if kind == "straight_line":
        return np.maximum(age, 0.0)
    if kind == "front_loaded":
        return 1.0 - np.exp(-np.maximum(age, 0.0) / float(profile["scale_years"]))
    if kind == "none":
        return np.zeros_like(age)
    raise ValueError(f"Unknown depreciation shape {kind!r}.")


# ----------------------------------------------------------------------------
# The structure block under one variant


@dataclass
class Structure:
    floor_area: np.ndarray  # cell
    retained: np.ndarray
    unit_cost: np.ndarray
    cost_by_bin: np.ndarray
    value_by_bin: np.ndarray
    social_unit_cost: np.ndarray  # authority
    social_floor_area: np.ndarray
    social_retained: np.ndarray
    theta: float | None
    depreciation: str
    receipt: dict


def _structure(
    variant: Mapping[str, object],
    *,
    geo: Geography,
    homes: Homes,
    assignment: Assignment,
    pools: Pools,
    stock: Stock,
    tenure: Tenure,
    costs: Costs,
    social_cells: pd.DataFrame,
    epc_floor: Mapping[str, float],
    depreciation_target: float | None,
    bins: list[np.ndarray],
    private_count: np.ndarray,
) -> Structure:
    """Unit costs at κ = 1 per cell and per authority's social homes."""
    used = assignment.used
    weight = assignment.weight
    region = geo.region_of_la[geo.la_of_msoa[homes.msoa[used]]]
    regional = variant.get("regional", "tpa")
    factor_of_region = np.array([costs.region_factor[str(c)] for c in geo.region_codes])
    if regional == "flat":
        factor_of_region = np.ones_like(factor_of_region)
    elif regional == "doubled":
        factor_of_region = 1.0 + 2.0 * (factor_of_region - 1.0)
    type_cost = np.array([costs.cost_per_m2[t] for t in declared.PROPERTY_TYPES])
    # Floor area: each home's own, else its cell's mean; or the authority's
    # EPC D3 mean for every home.
    linked_floor = homes.floor_area[used]
    present = np.isfinite(linked_floor)
    cell_floor = _attribute_mean(pools, weight, np.nan_to_num(linked_floor), present)
    floor = np.where(present, linked_floor, cell_floor[assignment.cell])
    if variant.get("floor_area") == "epc_d3":
        la_floor = np.array([epc_floor.get(str(code), np.nan) for code in geo.la_codes])
        home_la = geo.la_of_msoa[homes.msoa[used]]
        floor = np.where(np.isfinite(la_floor[home_la]), la_floor[home_la], floor)
    gross = floor * type_cost[homes.ptype[used]] * factor_of_region[region]
    # Social homes: the certificates' floor areas and ages per authority.
    social = _social_costs(geo, social_cells, type_cost, factor_of_region)
    # Each certificate cell stands for its share of its authority's social homes.
    social_weights = (
        social["gross"]
        * social["count"]
        * tenure.social_of_la[social["la"]]
        / np.maximum(social["count_of_la"], 1.0)
    )
    # Depreciation.
    name = str(variant.get("depreciation", "central"))
    profile_name = "central" if name in ("central", "voa_build_period") else name
    profile = declared.DEPRECIATION_PROFILES[profile_name]
    status = "applied"
    if profile["shape"] == "none":
        status = "off"
    elif depreciation_target is None:
        status = "off: ONS's net and gross capital stock of dwellings are absent"
    theta = None
    if name == "voa_build_period":
        period_age = np.array(
            [
                declared.BUILDING_AGE_AT_YEAR
                - ((a + b) / 2 if a is not None else b - 25)
                if b is not None
                else np.nan
                for a, b in stock.period_spans
            ]
        )
        shares = stock.period_shares  # msoa × band × period
        known = np.isfinite(period_age)
        period_shape = _shape(profile, np.nan_to_num(period_age)) * known
        cell_mass = shares[..., known].sum(axis=-1)
    if status == "applied":
        if name == "voa_build_period":
            # Each cell's retained share is its VOA build-period mix's.
            cell_gross = np.bincount(
                assignment.cell, weights=weight * gross, minlength=pools.sizes[0]
            )
            mix = shares.reshape(pools.sizes[0], -1)[:, known]
            mass = cell_mass.reshape(-1)
            with np.errstate(divide="ignore", invalid="ignore"):
                mix = np.where(mass[:, None] > 0, mix / mass[:, None], 0.0)
            shape_weights = (cell_gross[:, None] * mix).ravel()
            shape_values = np.tile(period_shape[known], pools.sizes[0])
            all_shape = np.concatenate([shape_values, social["shape_input"](profile)])
            all_weights = np.concatenate([shape_weights, social_weights])
        else:
            age = _age_years(homes.age_band[used])
            has_age = np.isfinite(age)
            all_shape = np.concatenate(
                [
                    _shape(profile, np.nan_to_num(age))[has_age],
                    social["shape_input"](profile),
                ]
            )
            all_weights = np.concatenate([(weight * gross)[has_age], social_weights])
        theta = residual_land.depreciation_scale(
            all_shape, all_weights, depreciation_target, cap=float(profile["cap"])
        )
        cap = float(profile["cap"])
        if name == "voa_build_period":
            period_retained = 1.0 - np.minimum(cap, theta * period_shape)
            cell_retained = (mix * period_retained[known][None, :]).sum(axis=1)
            cell_retained = np.where(mix.sum(axis=1) > 0, cell_retained, 1.0)
            retained = cell_retained[assignment.cell]
        else:
            home_retained = np.where(
                has_age,
                1.0 - np.minimum(cap, theta * _shape(profile, np.nan_to_num(age))),
                np.nan,
            )
            cell_retained = _attribute_mean(
                pools, weight, np.nan_to_num(home_retained), has_age
            )
            retained = np.where(has_age, home_retained, cell_retained[assignment.cell])
        social_retained = social["retained"](profile, theta, cap)
    else:
        retained = np.ones(int(used.sum()))
        social_retained = np.ones(len(geo.la_codes))
    unit = gross * retained
    floor_cell = _cell_mean(pools, weight, floor)
    retained_cell = _cell_mean(pools, weight, retained)
    unit_cell = _cell_mean(pools, weight, unit)
    cost_by_bin = _binned_means(pools, weight, unit, bins)
    value_by_bin = _binned_means(pools, weight, homes.value[used], bins)
    social_unit = social["gross_of_la"] * social_retained
    return Structure(
        floor_area=floor_cell,
        retained=retained_cell,
        unit_cost=unit_cell,
        cost_by_bin=cost_by_bin,
        value_by_bin=value_by_bin,
        social_unit_cost=social_unit,
        social_floor_area=social["floor_of_la"],
        social_retained=social_retained,
        theta=theta,
        depreciation=status if name == "central" or status != "applied" else name,
        receipt={
            "regional": regional,
            "floor_area": variant.get("floor_area", "linked"),
            "homes_floor_area_imputed": int((~present).sum()),
            "depreciation": name,
            "depreciation_status": status,
            "theta": theta,
        },
    )


def _social_costs(
    geo: Geography,
    cells: pd.DataFrame,
    type_cost: np.ndarray,
    factor_of_region: np.ndarray,
) -> dict[str, object]:
    """Gross cost per social home by authority, from the certificate cells.

    A certificate cell with no floor area takes its authority's mean, then
    its region's. An authority with no certificate takes its region's cells.
    """
    la_index = pd.Index(geo.la_codes.astype(str)).get_indexer(
        cells["local_authority"].astype(str)
    )
    cells = cells.assign(la=la_index)[la_index >= 0].copy()
    cells["region"] = geo.region_of_la[cells["la"].to_numpy()]
    count = cells["certificates"].to_numpy(dtype=float)
    floor = cells["floor_area_mean"].to_numpy(dtype=float)
    for key, groups in (("la", len(geo.la_codes)), ("region", len(geo.region_codes))):
        group = cells[key].to_numpy()
        known = np.isfinite(floor)
        total = np.bincount(
            group[known], weights=(floor * count)[known], minlength=groups
        )
        mass = np.bincount(group[known], weights=count[known], minlength=groups)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(mass > 0, total / mass, np.nan)
        floor = np.where(known, floor, mean[group])
    if not np.isfinite(floor).all():
        raise ValueError("A social certificate cell has no floor area in its region.")
    gross = (
        floor
        * type_cost[cells["ptype"].to_numpy()]
        * factor_of_region[cells["region"].to_numpy()]
    )
    age = _age_years(cells["age_band"].to_numpy())
    las = len(geo.la_codes)
    la = cells["la"].to_numpy()
    region = cells["region"].to_numpy()
    certificates_of_la = np.bincount(la, weights=count, minlength=las)
    has = certificates_of_la > 0
    # An authority without certificates borrows its region's cells, weighted.
    region_of_la = geo.region_of_la

    def per_la(values: np.ndarray) -> np.ndarray:
        own = np.bincount(la, weights=count * values, minlength=las)
        regional = np.bincount(
            region, weights=count * values, minlength=len(geo.region_codes)
        )
        regional_count = np.bincount(
            region, weights=count, minlength=len(geo.region_codes)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_own = own / certificates_of_la
            mean_region = regional / regional_count
        return np.where(has, mean_own, mean_region[region_of_la])

    gross_of_la = per_la(gross)
    floor_of_la = per_la(floor)

    def shape_input(profile):
        return _shape(profile, np.nan_to_num(age))

    def retained(profile, theta, cap):
        values = 1.0 - np.minimum(cap, theta * _shape(profile, np.nan_to_num(age)))
        weighted = per_la(gross * values)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(gross_of_la > 0, weighted / gross_of_la, 1.0)

    return {
        "la": la,
        "gross": gross,
        "count": count,
        "count_of_la": certificates_of_la[la],
        "gross_of_la": np.nan_to_num(gross_of_la),
        "floor_of_la": np.nan_to_num(floor_of_la),
        "shape_input": shape_input,
        "retained": retained,
        "authorities_without_certificates": int((~has).sum()),
    }


# ----------------------------------------------------------------------------
# The level


@dataclass
class Level:
    kappa: float
    status: str
    target: str
    solutions: dict
    inputs: dict


def _level_inputs(args, facts) -> dict[str, dict]:
    out = {}
    for name, spec in declared.LEVEL_INPUTS.items():
        feed_value = feed_key = None
        concept = spec.get("concept")
        if concept:
            rows = [
                fact
                for fact in ref.select_facts(
                    facts, concept=str(concept), spec_prefix=spec.get("spec_prefix")
                )
                if str((fact.get("period") or {}).get("value")) == str(spec["period"])
                and not (fact.get("dimensions") or {})
            ]
            if len(rows) > 1:
                raise ValueError(f"The feed carries {len(rows)} facts for {name}.")
            if rows:
                feed_value = ref.fact_value(rows[0])
                feed_key = str(rows[0].get("aggregate_fact_key"))
        parameter = (
            getattr(args, f"level_{name}", None) if spec.get("parameter") else None
        )
        out[name] = ref.resolve_level_input(
            name, feed_value=feed_value, feed_key=feed_key, parameter=parameter
        )
    return out


def _uk_split(args, facts, stock: Stock) -> dict[str, object]:
    """England and Wales's share of UK dwellings."""
    scotland = json.loads(args.scotland_dwellings.read_text(encoding="utf-8"))
    scottish = sum(
        float(row["value"])
        for row in scotland["rows"]
        if str(row.get("concept", "")).startswith("scotgov.chargeable_dwellings_band_")
        and (row.get("geography") or {}).get("level") == "local_authority"
    )
    ni_rows = [
        fact
        for fact in ref.select_facts(facts, concept="lps.housing_stock")
        if (fact.get("geography") or {}).get("id") == "N92000002"
    ]
    if not ni_rows:
        return {"status": "absent", "share": None}
    latest = max(ni_rows, key=lambda fact: str((fact.get("period") or {}).get("value")))
    northern_ireland = ref.fact_value(latest)
    england_wales = stock.engwal_all_properties
    return {
        "status": "computed",
        "share": england_wales / (england_wales + scottish + northern_ireland),
        "england_and_wales": england_wales,
        "scotland": scottish,
        "northern_ireland": northern_ireland,
        "northern_ireland_period": str((latest.get("period") or {}).get("value")),
        "sources": declared.UK_DWELLING_SPLIT,
    }


def _solve_level(
    levels: Mapping[str, dict],
    split: Mapping[str, object],
    *,
    value: np.ndarray,
    cost: np.ndarray,
    count: np.ndarray,
) -> Level:
    """κ under each target, and the central one.

    ``value``, ``cost`` and ``count`` hold every cell, private and social:
    mean value, unit cost at κ = 1 and homes.
    """
    solutions: dict[str, dict] = {}
    dwellings = levels["dwellings_whole_economy"]["value"]
    share = split.get("share")
    total_cost = float((count * cost).sum())
    if dwellings is not None and share is not None:
        target = dwellings * share
        kappa = residual_land.cost_level_for_total(target, count * cost)
        solutions["a"] = {
            "kappa": kappa,
            "status": levels["dwellings_whole_economy"]["status"],
            "target_gbp": target,
        }
    else:
        solutions["a"] = {"kappa": None, "status": "absent"}
    land_2021 = levels["land_underlying_dwellings_2021"]["value"]
    dwellings_2021 = levels["dwellings_whole_economy_2021"]["value"]
    if land_2021 is not None and dwellings_2021 is not None:
        ratio = land_2021 / (land_2021 + dwellings_2021)
        kappa = residual_land.cost_level_for_ratio(value, cost, count, ratio)
        solutions["b"] = {"kappa": kappa, "status": "computed", "ratio": ratio}
    else:
        solutions["b"] = {"kappa": None, "status": "absent"}
    central = solutions[declared.CENTRAL_KAPPA_TARGET]
    if central["kappa"] is None:
        return Level(
            1.0,
            "absent: κ = 1, TPA's own level",
            declared.CENTRAL_KAPPA_TARGET,
            solutions,
            dict(levels),
        )
    return Level(
        float(central["kappa"]),
        str(central["status"]),
        declared.CENTRAL_KAPPA_TARGET,
        solutions,
        dict(levels) | {"total_cost_at_kappa_1": total_cost},
    )


# ----------------------------------------------------------------------------
# Social values


def _social_values(
    facts, geo: Geography, hpi, private_mean_of_la: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Each authority's social home value: its region's MVS social rented mean.

    The survey's July 2024 means move to the calendar-2024 basis with each
    region's all-dwellings index. Wales, which the survey does not cover,
    takes England's ratio of social to private-sector means × its own
    sales-based private mean, flagged.
    """
    regions = ref.market_value_means(facts, spec_prefix=declared.MVS_REGION_SPEC)
    england = ref.market_value_means(facts, spec_prefix=declared.MVS_ENGLAND_SPEC)[
        ENGLAND
    ]
    year, month = declared.MVS_MONTH
    basis = hpi.basis_levels(declared.BASIS_YEAR)
    at_month = hpi.month_levels(year, month)
    index = hpi.area_index

    def uprate(code: str) -> float:
        row = index.get(code)
        if row is None or not np.isfinite(at_month[row, 4]) or not at_month[row, 4] > 0:
            row = index[ENGLAND]
        return float(basis[row, 4] / at_month[row, 4])

    private_ratio = england[declared.MVS_SOCIAL_TENURE] / np.mean(
        [england[t] for t in declared.MVS_PRIVATE_TENURES]
    )
    values = np.zeros(len(geo.la_codes))
    for la in range(len(geo.la_codes)):
        region = str(geo.region_codes[geo.region_of_la[la]])
        if region in regions:
            values[la] = regions[region][declared.MVS_SOCIAL_TENURE] * uprate(region)
        else:
            values[la] = private_ratio * private_mean_of_la[la]
    return values, {
        "rule": (
            "MVS 2024 AT1.3 regional all-social-renters mean, July 2024, moved to "
            "the calendar-2024 basis by the region's UK HPI all-dwellings index; "
            "Wales: England's social ÷ private-sector mean ratio × its own "
            "sales-based private mean"
        ),
        "england_social_to_private_ratio": round(float(private_ratio), 6),
    }


# ----------------------------------------------------------------------------
# One build of the cells under a value stratification


@dataclass
class ValueBlock:
    assignment: Assignment
    pools: Pools
    bins: list[np.ndarray]
    mean_value: np.ndarray
    knots: np.ndarray
    knots_mean: np.ndarray
    private_count: np.ndarray
    cell_rows: np.ndarray  # cells with homes counted
    seat_shares: pd.DataFrame


def _value_block(
    geo, homes: Homes, stock: Stock, tenure: Tenure, *, strata: str
) -> ValueBlock:
    assignment = _assign(homes, stock, tenure, strata=strata)
    pools = _pools(geo, assignment)
    values = homes.value[assignment.used]
    mean_value = _cell_mean(pools, assignment.weight, values)
    knots, knots_mean = _knots(pools, assignment, values)
    bins = _rank_bins(pools, assignment, values)
    private = stock.counts * (1.0 - tenure.social_share_of_msoa[:, None])
    private_count = private.reshape(-1)
    cell_rows = np.flatnonzero(stock.counts.reshape(-1) > 0)
    # Each cell's split across constituencies, from its homes' output areas;
    # a cell with no home takes its MSOA's household shares.
    bands = len(declared.BANDS)
    oa_msoa = geo.msoa_of_oa
    default = pd.DataFrame(
        {"msoa": oa_msoa, "target": geo.seat_of_oa, "w": geo.households}
    )
    default = default.groupby(["msoa", "target"], sort=True)["w"].sum().reset_index()
    default["share"] = default["w"] / default.groupby("msoa")["w"].transform("sum")
    default = default.merge(pd.DataFrame({"band": np.arange(bands)}), how="cross")
    default["cell"] = default["msoa"] * bands + default["band"]
    seat_of_home = geo.seat_of_oa[homes.oa[assignment.used]]
    shares = cell_pooling.allocation_shares(
        assignment.cell,
        seat_of_home,
        assignment.weight,
        n_cells=pools.sizes[0],
        default=default[["cell", "target", "share"]],
    )
    shares = shares[shares["cell"].isin(cell_rows)].reset_index(drop=True)
    return ValueBlock(
        assignment=assignment,
        pools=pools,
        bins=bins,
        mean_value=mean_value,
        knots=knots,
        knots_mean=knots_mean,
        private_count=private_count,
        cell_rows=cell_rows,
        seat_shares=shares,
    )


@dataclass
class Rollup:
    la_land: np.ndarray
    la_value: np.ndarray
    seat_land: np.ndarray
    seat_value: np.ndarray
    cell_land_per_home: np.ndarray
    social_land_per_home: np.ndarray
    kappa: float


def _rollup(
    geo: Geography,
    block: ValueBlock,
    structure: Structure,
    tenure: Tenure,
    social_value: np.ndarray,
    kappa: float,
    *,
    residual: str = "pooled",
) -> Rollup:
    bands = len(declared.BANDS)
    cells = block.cell_rows
    value = np.nan_to_num(block.mean_value[cells])
    if residual == "rank":
        weights = np.isfinite(structure.value_by_bin[cells]).astype(float)
        land_per_home = residual_land.residual_along_rank(
            np.nan_to_num(structure.value_by_bin[cells]),
            np.nan_to_num(structure.cost_by_bin[cells]),
            np.where(weights.sum(axis=1, keepdims=True) > 0, weights, 1.0),
            kappa,
        )
    else:
        land_per_home = residual_land.land_per_unit(
            value, np.nan_to_num(structure.unit_cost[cells]), kappa
        )
    count = block.private_count[cells]
    la_of_cell = geo.la_of_msoa[cells // bands]
    las = len(geo.la_codes)
    social_land = residual_land.land_per_unit(
        social_value, structure.social_unit_cost, kappa
    )
    la_land = np.bincount(la_of_cell, weights=count * land_per_home, minlength=las)
    la_land += tenure.social_of_la * social_land
    la_value = np.bincount(la_of_cell, weights=count * value, minlength=las)
    la_value += tenure.social_of_la * social_value
    # Constituencies: private cells by their shares, social homes by the
    # authority's household shares.
    position = pd.Index(cells).get_indexer(block.seat_shares["cell"].to_numpy())
    seat_index = block.seat_shares["target"].to_numpy().astype(np.int64)
    share = block.seat_shares["share"].to_numpy()
    seats = len(geo.seat_codes)
    seat_land = np.bincount(
        seat_index, weights=share * (count * land_per_home)[position], minlength=seats
    )
    seat_value = np.bincount(
        seat_index, weights=share * (count * value)[position], minlength=seats
    )
    la_of_oa = geo.la_of_msoa[geo.msoa_of_oa]
    la_households = np.bincount(la_of_oa, weights=geo.households, minlength=las)
    with np.errstate(divide="ignore", invalid="ignore"):
        oa_share = np.where(
            la_households[la_of_oa] > 0, geo.households / la_households[la_of_oa], 0.0
        )
    seat_land += np.bincount(
        geo.seat_of_oa,
        weights=oa_share * (tenure.social_of_la * social_land)[la_of_oa],
        minlength=seats,
    )
    seat_value += np.bincount(
        geo.seat_of_oa,
        weights=oa_share * (tenure.social_of_la * social_value)[la_of_oa],
        minlength=seats,
    )
    full = np.zeros(block.pools.sizes[0])
    full[cells] = land_per_home
    return Rollup(la_land, la_value, seat_land, seat_value, full, social_land, kappa)


# ----------------------------------------------------------------------------
# MHCLG's pattern


def _mhclg_pattern(
    facts,
    geo: Geography,
    stock: Stock,
    central: Rollup,
    to_roster: Mapping[str, str],
) -> dict[str, dict]:
    """England's central land spread in MHCLG's proportions, per variant."""
    english = np.char.startswith(geo.la_codes.astype(str), "E")
    total = float(central.la_land[english].sum())
    caps = central.la_value[english]
    codes = geo.la_codes[english].astype(str)
    base = dict(declared.MHCLG_VARIANTS[0])
    out: dict[str, dict] = {}
    for entry in declared.MHCLG_VARIANTS:
        settings = {**base, **{k: v for k, v in entry.items() if k != "name"}}
        per_home = ref.land_per_new_home(
            facts,
            concept=declared.MHCLG_SHEETS[str(settings["sheet"])],
            density=str(settings["density"]),
            percentile=int(settings["percentile"]),
            density_factor=float(settings["density_factor"]),
        )
        per_home = {
            to_roster.get(code, code): value for code, value in per_home.items()
        }
        flags: dict[str, list[str]] = {"missing": [], "imputed": []}
        if (
            settings["scilly"] == "cornwall"
            and declared.ISLES_OF_SCILLY not in per_home
        ):
            if declared.CORNWALL in per_home:
                per_home[declared.ISLES_OF_SCILLY] = per_home[declared.CORNWALL]
                flags["imputed"].append(declared.ISLES_OF_SCILLY)
        raw = []
        for la, code in zip(np.flatnonzero(english), codes, strict=True):
            if code not in per_home:
                flags["missing"].append(code)
            raw.append(per_home.get(code, 0.0) * stock.la_all_properties[la])
        land, capped = residual_land.scale_cap_rescale(np.array(raw), caps, total)
        out[str(entry["name"])] = {
            "land": land,
            "per_new_home": [per_home.get(code) for code in codes],
            "capped": sorted(codes[capped].tolist()),
            "flags": flags,
        }
    return out


# ----------------------------------------------------------------------------
# Checks and measurements


def _lambda(land: np.ndarray, value: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(value > 0, land / value, 0.0)


def _status(passed: bool | None) -> str:
    return "not_evaluated" if passed is None else ("pass" if passed else "fail")


def _decision_rule(
    geo,
    rollups: Mapping[str, Rollup],
    level: Level,
    costs: Costs,
    homes: Homes,
    block: ValueBlock,
) -> list[dict]:
    english = np.char.startswith(geo.la_codes.astype(str), "E")
    central = rollups["central"]
    central_lambda = _lambda(central.la_land, central.la_value)[english]
    out = []
    for rule in declared.DECISION_RULE:
        if rule["id"] == "rank_stability":
            values = {}
            for name in rule["variants"]:
                other = rollups.get(name)
                if other is None:
                    values[name] = None
                    continue
                values[name] = round(
                    residual_land.rank_correlation(
                        central_lambda, _lambda(other.la_land, other.la_value)[english]
                    ),
                    6,
                )
            known = [v for v in values.values() if v is not None]
            passed = (
                (min(known) >= rule["at_least"]) if len(known) == len(values) else None
            )
            out.append(
                {
                    "id": rule["id"],
                    "value": values,
                    "threshold": rule["at_least"],
                    "status": _status(passed),
                }
            )
        elif rule["id"] == "kappa_band":
            low, high = rule["within"]
            solved = not level.status.startswith("absent")
            passed = (low <= level.kappa <= high) if solved else None
            out.append(
                {
                    "id": rule["id"],
                    "value": round(level.kappa, 6) if solved else None,
                    "threshold": [low, high],
                    "status": _status(passed),
                }
            )
        elif rule["id"] == "cost_below_new_build_lower_quartile":
            out.append(_cost_check(rule, geo, level, costs, homes))
    return out


def _cost_check(rule, geo, level: Level, costs: Costs, homes: Homes) -> dict:
    """Cost per m² at κ against the lower quartile of new-build price per m²."""
    new = homes.new_build & np.isfinite(homes.floor_area)
    region = geo.region_codes[geo.region_of_la[geo.la_of_msoa[homes.msoa]]]
    by_region = {}
    passed = []
    for code in geo.region_codes:
        code = str(code)
        if not code.startswith("E"):
            continue
        rows = new & (region == code)
        if not rows.any():
            by_region[code] = {"status": "not_evaluated", "new_builds": 0}
            continue
        price = homes.value[rows] / homes.floor_area[rows]
        quartile = float(np.quantile(price, 0.25))
        mix = np.bincount(homes.ptype[rows], minlength=4) / rows.sum()
        cost = (
            level.kappa
            * costs.region_factor[code]
            * float(
                sum(
                    mix[i] * costs.cost_per_m2[t]
                    for i, t in enumerate(declared.PROPERTY_TYPES)
                )
            )
        )
        ok = cost < quartile
        passed.append(ok)
        by_region[code] = {
            "cost_per_m2": round(cost, 2),
            "new_build_price_per_m2_lower_quartile": round(quartile, 2),
            "status": _status(ok),
        }
    overall = (
        all(passed)
        if passed
        and len(passed) == sum(str(c).startswith("E") for c in geo.region_codes)
        else None
    )
    if level.status.startswith("absent") and overall is not None:
        overall = None  # read at κ = 1 only; the rule needs the solved level
    return {
        "id": rule["id"],
        "value": by_region,
        "kappa": round(level.kappa, 6),
        "status": _status(overall),
    }


def _measurements(
    facts,
    geo: Geography,
    rollups: Mapping[str, Rollup],
    block: ValueBlock,
    stock: Stock,
    tenure: Tenure,
    mhclg: Mapping[str, dict],
    hpi,
    to_roster,
) -> list[dict]:
    out = []
    bands = len(declared.BANDS)
    cells = block.cell_rows
    msoa = cells // bands
    band = cells % bands
    la_of_cell = geo.la_of_msoa[msoa]
    count = block.private_count[cells]
    central = rollups["central"]
    band_d = ref.band_d_by_authority(
        facts,
        england_spec=declared.BAND_D_ENGLAND_SPEC,
        wales_spec=declared.BAND_D_WALES_SPEC,
    )
    band_d = {to_roster.get(code, code): value for code, value in band_d.items()}
    ratio = np.array([declared.BAND_D_RATIOS[b] for b in declared.BANDS])[band]
    la_band_d = np.array([band_d.get(str(code), np.nan) for code in geo.la_codes])
    tax = count * ratio * la_band_d[la_of_cell]
    has_tax = np.isfinite(tax) & (tax > 0)
    value = np.nan_to_num(block.mean_value[cells])
    english = np.char.startswith(geo.la_codes.astype(str), "E")
    patterns = {
        "cell_land": count * central.cell_land_per_home[cells],
        "authority_share": count
        * value
        * _lambda(central.la_land, central.la_value)[la_of_cell],
    }
    mhclg_lambda = np.zeros(len(geo.la_codes))
    mhclg_lambda[english] = _lambda(mhclg["central"]["land"], central.la_value[english])
    patterns["mhclg_pattern"] = count * value * mhclg_lambda[la_of_cell]
    if "land_at_rank" in rollups:
        patterns["land_at_rank"] = (
            count * rollups["land_at_rank"].cell_land_per_home[cells]
        )
    position = pd.Index(cells).get_indexer(block.seat_shares["cell"].to_numpy())
    seat_index = block.seat_shares["target"].to_numpy().astype(np.int64)
    share = block.seat_shares["share"].to_numpy()

    def redistribution(land: np.ndarray) -> tuple[float, np.ndarray]:
        rows = has_tax & (land >= 0)
        rate = tax[rows].sum() / land[rows].sum() if land[rows].sum() > 0 else 0.0
        moved = np.where(rows, np.abs(tax - rate * land), 0.0)
        national = 50.0 * moved.sum() / tax[rows].sum()
        seat_moved = np.bincount(
            seat_index, weights=share * moved[position], minlength=len(geo.seat_codes)
        )
        seat_tax = np.bincount(
            seat_index,
            weights=share * np.where(rows, tax, 0.0)[position],
            minlength=len(geo.seat_codes),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            seats = np.where(seat_tax > 0, 50.0 * seat_moved / seat_tax, np.nan)
        return float(national), seats

    grain = declared.MEASUREMENTS[0]
    results = {name: redistribution(land) for name, land in patterns.items()}
    national = {name: round(r[0], 4) for name, r in results.items()}
    gap = abs(national["cell_land"] - national["authority_share"])
    seat_gap = np.abs(results["cell_land"][1] - results["authority_share"][1])
    seat_gap = seat_gap[np.isfinite(seat_gap)]
    within = float((seat_gap < grain["seat_points"]).mean()) if len(seat_gap) else None
    enough = (gap < grain["national_points"]) and (
        within is not None and within >= grain["seat_share"]
    )
    out.append(
        {
            "id": "grain_test",
            "definition": (
                "for each pattern, the share of council tax (VOA private count × band "
                "ratio × the authority's Band D) that a land tax at the rate balancing "
                "the two would move between cells, in points; nationally and per seat"
            ),
            "national_points": national,
            "cell_vs_authority_national_gap": round(gap, 4),
            "seats_within_seat_points": round(within, 4)
            if within is not None
            else None,
            "thresholds": {
                k: grain[k] for k in ("national_points", "seat_points", "seat_share")
            },
            "reading": "authority shares are enough" if enough else "cells are needed",
            "status": "measured",
            "cells_without_band_d": int((~has_tax).sum()),
        }
    )
    # κ diagnostics: cells with no land by region, and seat ranks.
    diagnostics = declared.MEASUREMENTS[1]
    region_of_cell = geo.region_of_la[la_of_cell]
    zero = central.cell_land_per_home[cells] <= 0
    by_region = {
        str(code): round(float(zero[region_of_cell == r].mean()), 4)
        for r, code in enumerate(geo.region_codes)
        if (region_of_cell == r).any()
    }
    seat_lambda = _lambda(central.seat_land, central.seat_value)
    correlations = {}
    for name in (
        "regional_flat",
        "regional_doubled",
        "depreciation_off",
        "depreciation_steep",
        "depreciation_voa_build_period",
    ):
        other = rollups.get(name)
        if other is not None:
            correlations[name] = round(
                residual_land.rank_correlation(
                    seat_lambda, _lambda(other.seat_land, other.seat_value)
                ),
                6,
            )
    out.append(
        {
            "id": "kappa_diagnostics",
            "zero_land_cell_share_by_region": by_region,
            "seat_rank_correlations": correlations,
            "thresholds": {k: v for k, v in diagnostics.items() if k != "id"},
            "status": _status(
                all(
                    v < diagnostics["zero_land_cell_share_by_region"]
                    for v in by_region.values()
                )
                and all(
                    v >= diagnostics["seat_rank_correlation"]
                    for v in correlations.values()
                )
            ),
        }
    )
    out.append(_stock_value_check(facts, geo, block, stock, hpi, cells, count, value))
    return out


def _stock_value_check(facts, geo, block, stock, hpi, cells, count, value) -> dict:
    """The sales-based private-sector mean against the Market Value Survey."""
    spec = declared.MEASUREMENTS[2]
    regions = ref.market_value_means(facts, spec_prefix=declared.MVS_REGION_SPEC)
    year, month = declared.MVS_MONTH
    basis = hpi.basis_levels(declared.BASIS_YEAR)
    at_month = hpi.month_levels(year, month)
    bands = len(declared.BANDS)
    region_of_cell = geo.region_of_la[geo.la_of_msoa[cells // bands]]
    out: dict[str, object] = {}
    gaps = []
    for r, code in enumerate(geo.region_codes):
        code = str(code)
        if code not in regions:
            continue
        rows = region_of_cell == r
        ours = (
            float((count[rows] * value[rows]).sum() / count[rows].sum())
            if rows.any()
            else float("nan")
        )
        means = regions[code]
        survey = float(np.mean([means[t] for t in declared.MVS_PRIVATE_TENURES]))
        row = hpi.area_index.get(code, hpi.area_index[ENGLAND])
        survey *= float(basis[row, 4] / at_month[row, 4])
        gap = ours / survey - 1.0
        gaps.append(abs(gap))
        out[code] = {
            "ours": round(ours, 2),
            "survey_private_sector": round(survey, 2),
            "relative_gap": round(gap, 4),
        }
    english = np.char.startswith(geo.region_codes.astype(str)[region_of_cell], "E")
    ours = (
        float((count[english] * value[english]).sum() / count[english].sum())
        if english.any()
        else float("nan")
    )
    england = ref.market_value_means(facts, spec_prefix=declared.MVS_ENGLAND_SPEC)[
        ENGLAND
    ]
    survey = float(np.mean([england[t] for t in declared.MVS_PRIVATE_TENURES]))
    survey *= float(
        basis[hpi.area_index[ENGLAND], 4] / at_month[hpi.area_index[ENGLAND], 4]
    )
    national_gap = ours / survey - 1.0
    passed = abs(national_gap) <= spec["england_relative_gap"] and all(
        g <= spec["region_relative_gap"] for g in gaps
    )
    return {
        "id": "stock_value",
        "england": {
            "ours": round(ours, 2),
            "survey_private_sector": round(survey, 2),
            "relative_gap": round(national_gap, 4),
        },
        "regions": out,
        "survey_private_sector_rule": "the unweighted mean of the survey's owner-occupier and private-renter means, moved to the basis by the UK HPI",
        "thresholds": {
            "england": spec["england_relative_gap"],
            "region": spec["region_relative_gap"],
        },
        "status": _status(passed),
    }


def _also_recorded(
    facts,
    geo,
    rollups,
    mhclg,
    tpa_cells: Mapping[str, dict],
    levels,
    homes,
    structure_central: Structure,
    block: ValueBlock,
    level: Level,
) -> dict:
    english = np.char.startswith(geo.la_codes.astype(str), "E")
    codes = geo.la_codes[english].astype(str)
    central = rollups["central"]
    ours = _lambda(central.la_land, central.la_value)[english]
    tpa = np.array(
        [tpa_cells.get(code, {}).get("land_share", np.nan) for code in codes]
    )
    known = np.isfinite(tpa)
    out: dict[str, object] = {}
    out["tpa_rank_correlation"] = (
        round(residual_land.rank_correlation(ours[known], tpa[known]), 6)
        if known.sum() >= 2
        else None
    )
    mhclg_lambda = _lambda(mhclg["central"]["land"], central.la_value[english])
    flags = residual_land.log_ratio_flags(
        dict(zip(codes.tolist(), ours.tolist(), strict=True)),
        {
            "mhclg_pattern": dict(
                zip(codes.tolist(), mhclg_lambda.tolist(), strict=True)
            ),
            "tpa": {
                c: float(v)
                for c, v in zip(codes.tolist(), tpa.tolist(), strict=True)
                if np.isfinite(v)
            },
        },
        threshold=float(declared.LOG_RATIO_FLAG["threshold"]),
        minimum_references=int(declared.LOG_RATIO_FLAG["minimum_references"]),
    )
    out["log_ratio_flags"] = flags
    households = levels["households_land"]["value"]
    land_total = float(central.la_land.sum())
    out["household_land_against_e44n"] = {
        "england_and_wales_land": round(land_total, 0),
        "e44n_uk_households_land": households,
        "status": _status(None if households is None else land_total < households),
    }
    # Like for like: the residual per new home by authority against MHCLG's.
    used = block.assignment.used
    new = homes.new_build[used]
    la_of_home = geo.la_of_msoa[homes.msoa[used]]
    unit = structure_central.unit_cost[block.assignment.cell]
    residual = np.maximum(0.0, homes.value[used] - level.kappa * unit)
    totals = np.bincount(
        la_of_home[new], weights=residual[new], minlength=len(geo.la_codes)
    )
    counts = np.bincount(la_of_home[new], minlength=len(geo.la_codes))
    with np.errstate(divide="ignore", invalid="ignore"):
        per_new = np.where(counts > 0, totals / counts, np.nan)[english]
    mhclg_new = np.array(
        [np.nan if v is None else v for v in mhclg["central"]["per_new_home"]],
        dtype=float,
    )
    both = np.isfinite(per_new) & np.isfinite(mhclg_new)
    out["new_build_like_for_like"] = {
        "authorities_compared": int(both.sum()),
        "rank_correlation": round(
            residual_land.rank_correlation(per_new[both], mhclg_new[both]), 6
        )
        if both.sum() >= 2
        else None,
        "median_ratio": round(
            float(
                np.median(
                    per_new[both]
                    / np.where(mhclg_new[both] > 0, mhclg_new[both], np.nan)
                )
            ),
            6,
        )
        if both.any()
        else None,
    }
    out["stock_dates"] = (
        "VOA counts at 31 March 2026 against end-2024 buildings; not adjusted (no end-2024 count of England and Wales's dwellings is an input)"
    )
    return out


# ----------------------------------------------------------------------------
# Payloads


def _r(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return float(round(value, digits))


def _rounded(values: np.ndarray, digits: int) -> list:
    return [_r(float(v), digits) for v in values]


def _declarations_digest() -> str:
    public = {
        name: value
        for name, value in vars(declared).items()
        if name.isupper() and not name.startswith("_")
    }
    public["POOLING"] = {
        "levels": list(declared.POOLING.levels),
        "minimums": dict(declared.POOLING.minimums),
    }
    return derived_resource.payload_sha256(json.loads(json.dumps(public, default=list)))


def _feed_identity(path: Path | None) -> dict[str, object]:
    if path is None or not path.is_file():
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: manifest.get(key)
        for key in ("facts_sha256", "fact_row_count", "schema_version", "source_commit")
        if key in manifest
    }


def feed_inputs(facts) -> dict[str, object]:
    """Everything the builder reads from the feed, for its fingerprint.

    A test recomputes it on the pinned feed, so a re-pin that moves any of
    these values fails until the resources are rebuilt.
    """
    private, owner, receipt = hv.tenure_shares(facts)
    msoa_private, _, _ = hv.tenure_shares(facts, level="msoa")
    out: dict[str, object] = {
        "tenure": {"private": private, "owner": owner, "msoa_private": msoa_private},
        "band_d": ref.band_d_by_authority(
            facts,
            england_spec=declared.BAND_D_ENGLAND_SPEC,
            wales_spec=declared.BAND_D_WALES_SPEC,
        ),
        "mvs_regions": ref.market_value_means(
            facts, spec_prefix=declared.MVS_REGION_SPEC
        ),
        "mvs_england": ref.market_value_means(
            facts, spec_prefix=declared.MVS_ENGLAND_SPEC
        ),
        "epc_d3_floor_area": ref.epc_mean_floor_area(
            facts, spec_prefix=declared.EPC_D3_SPEC
        ),
    }
    for entry in declared.MHCLG_VARIANTS:
        settings = {**declared.MHCLG_VARIANTS[0], **entry}
        out[f"mhclg_{entry['name']}"] = ref.land_per_new_home(
            facts,
            concept=declared.MHCLG_SHEETS[str(settings["sheet"])],
            density=str(settings["density"]),
            percentile=int(settings["percentile"]),
            density_factor=float(settings["density_factor"]),
        )
    for name, spec in declared.LEVEL_INPUTS.items():
        concept = spec.get("concept")
        if concept:
            out[f"level_{name}"] = sorted(
                [str((f.get("period") or {}).get("value")), ref.fact_value(f)]
                for f in ref.select_facts(
                    facts, concept=str(concept), spec_prefix=spec.get("spec_prefix")
                )
            )
    out["ni_housing_stock"] = sorted(
        [str((f.get("period") or {}).get("value")), ref.fact_value(f)]
        for f in ref.select_facts(facts, concept="lps.housing_stock")
        if (f.get("geography") or {}).get("id") == "N92000002"
    )
    return out


def _value_payload(
    geo,
    block: ValueBlock,
    stock: Stock,
    tenure: Tenure,
    social_value: np.ndarray,
    common: dict,
) -> dict:
    bands = len(declared.BANDS)
    cells = block.cell_rows
    msoa = cells // bands
    band = cells % bands
    pools = block.pools
    owner = tenure.owner_share_of_la[geo.la_of_msoa[msoa]]
    social_count = stock.counts.reshape(-1)[cells] * tenure.social_share_of_msoa[msoa]
    rows = [
        [
            str(geo.msoa_codes[m]),
            declared.BANDS[b],
            _r(stock.counts[m, b], 1),
            _r(block.private_count[c], 6),
            _r(social_count[i], 6),
            int(pools.sampled[0][c]),
            _r(owner[i], 6),
            _r(block.mean_value[c], 2),
            int(pools.means_level[c]),
            int(pools.knots_level[c]),
            _r(block.knots_mean[c], 2),
        ]
        for i, (c, m, b) in enumerate(zip(cells, msoa, band, strict=True))
    ]
    knots = [[_r(float(v), 2) for v in block.knots[c]] for c in cells]
    position = pd.Index(cells).get_indexer(block.seat_shares["cell"].to_numpy())
    seat_rows = [
        [int(p), str(geo.seat_codes[int(t)]), _r(float(s), 6)]
        for p, t, s in zip(
            position,
            block.seat_shares["target"],
            block.seat_shares["share"],
            strict=True,
        )
    ]
    social_rows = [
        [str(code), _r(tenure.social_of_la[i], 6), _r(social_value[i], 2)]
        for i, code in enumerate(geo.la_codes)
    ]
    return {
        **common,
        "kind": KINDS["value"],
        "resource": RESOURCES["value"],
        "description": (
            "Homes in England and Wales by MSOA × council tax band: VOA's count at 31 March "
            "2026, the private and social counts, homes seen sold, the owner-occupied share "
            "of the private stock, the mean value and 13 value knots at calendar-2024 prices "
            "(microcosm#1524). Not Chronicle facts."
        ),
        "bands": list(declared.BANDS),
        "pooling_levels": list(declared.POOLING.levels),
        "minimums": dict(declared.POOLING.minimums),
        "knot_grid": list(declared.KNOT_GRID),
        "knot_cap": declared.KNOT_CAP,
        "columns": [
            "msoa",
            "band",
            "voa_count",
            "private_count",
            "social_count",
            "homes_sold",
            "owner_share_of_private",
            "mean_value",
            "means_level",
            "knots_level",
            "knots_exact_mean",
        ],
        "rows": rows,
        "knots": knots,
        "constituency_shares": {
            "columns": ["row", "constituency", "share"],
            "rows": seat_rows,
        },
        "social": {
            "columns": ["local_authority", "count", "mean_value"],
            "rows": social_rows,
        },
        # Each MSOA's authority and region, so the rollups can be checked
        # from the committed files alone.
        "msoas": {
            "columns": ["msoa", "local_authority", "region"],
            "rows": [
                [
                    str(code),
                    str(geo.la_codes[geo.la_of_msoa[m]]),
                    str(geo.region_codes[geo.region_of_la[geo.la_of_msoa[m]]]),
                ]
                for m, code in enumerate(geo.msoa_codes)
            ],
        },
    }


def _structure_payload(
    geo,
    block: ValueBlock,
    structure: Structure,
    central: Rollup,
    level: Level,
    costs: Costs,
    common: dict,
    value_sha: str,
) -> dict:
    cells = block.cell_rows
    rows = [
        [
            _r(structure.floor_area[c], 2),
            _r(structure.retained[c], 6),
            _r(structure.unit_cost[c], 2),
            _r(central.cell_land_per_home[c], 2),
        ]
        for c in cells
    ]
    social = [
        [
            str(code),
            _r(structure.social_floor_area[i], 2),
            _r(structure.social_retained[i], 6),
            _r(structure.social_unit_cost[i], 2),
            _r(central.social_land_per_home[i], 2),
        ]
        for i, code in enumerate(geo.la_codes)
    ]
    return {
        **common,
        "kind": KINDS["structure"],
        "resource": RESOURCES["structure"],
        "description": (
            "Building costs by stock cell, aligned to the value block's rows: floor area, "
            "the share of the building left after depreciation, the unit cost at TPA's own "
            "level (κ = 1), costs and values along the value rank, and the central case's "
            "land per home (microcosm#1524)."
        ),
        "value_block_sha256": value_sha,
        "columns": [
            "floor_area_mean",
            "retained_share_mean",
            "unit_cost_mean",
            "land_per_home_central",
        ],
        "rows": rows,
        "rank_bins": declared.RANK_BINS,
        "unit_cost_by_rank_bin": [_rounded(structure.cost_by_bin[c], 2) for c in cells],
        "value_by_rank_bin": [_rounded(structure.value_by_bin[c], 2) for c in cells],
        "social": {
            "columns": [
                "local_authority",
                "floor_area_mean",
                "retained_share_mean",
                "unit_cost_mean",
                "land_per_home_central",
            ],
            "rows": social,
        },
        "level": {
            "kappa": _r(level.kappa, 6),
            "status": level.status,
            "target": level.target,
            "theta": _r(structure.theta, 6) if structure.theta is not None else None,
            "depreciation": structure.depreciation,
        },
        "relative_costs": {
            "cost_per_m2": costs.cost_per_m2,
            "region_factor": costs.region_factor,
        },
    }


def _authority_payload(
    geo, names, stock, tenure, rollups, mhclg, tpa_cells, flags, common, digests
) -> dict:
    rows = []
    for i, code in enumerate(geo.la_codes):
        code = str(code)
        rows.append(
            [
                code,
                (names.get(code) or {}).get("name", ""),
                str(geo.region_codes[geo.region_of_la[i]]),
                _r(stock.la_all_properties[i], 1),
                _r(tenure.social_of_la[i], 6),
                sorted(flags.get(code, [])),
            ]
        )
    english = np.char.startswith(geo.la_codes.astype(str), "E")
    variants = {}
    for name, rollup in rollups.items():
        entry = {"land": _rounded(rollup.la_land, 0), "kappa": _r(rollup.kappa, 6)}
        if name == "central" or not np.allclose(
            rollup.la_value, rollups["central"].la_value
        ):
            entry["value"] = _rounded(rollup.la_value, 0)
        variants[name] = entry
    mhclg_out = {}
    for name, entry in mhclg.items():
        land = np.full(len(geo.la_codes), np.nan)
        land[english] = entry["land"]
        mhclg_out[name] = {
            "land": _rounded(land, 0),
            "capped": entry["capped"],
            "missing": entry["flags"]["missing"],
            "imputed": entry["flags"]["imputed"],
        }
    per_new = np.full(len(geo.la_codes), np.nan)
    per_new[english] = [
        np.nan if v is None else v for v in mhclg["central"]["per_new_home"]
    ]
    return {
        **common,
        "kind": KINDS["authorities"],
        "resource": RESOURCES["authorities"],
        "description": (
            "Land under homes by local authority in England and Wales: VOA's stock, the "
            "stock value and land under every variant, aligned to the rows; λ = land ÷ "
            "value is the loader's. MHCLG's pattern for English authorities (microcosm#1524)."
        ),
        "columns": [
            "local_authority",
            "name",
            "region",
            "dwellings",
            "social_homes",
            "flags",
        ],
        "rows": rows,
        "variants": variants,
        "mhclg_pattern": mhclg_out,
        "mhclg_land_per_new_home": _rounded(per_new, 2),
        "tpa_land_share": [
            _r(tpa_cells.get(str(code), {}).get("land_share", float("nan")), 6)
            for code in geo.la_codes
        ],
        "digests": digests,
    }


def _constituency_payload(geo, rollups, common, digests) -> dict:
    variants = {}
    for name, rollup in rollups.items():
        entry = {"land": _rounded(rollup.seat_land, 0)}
        if name == "central" or not np.allclose(
            rollup.seat_value, rollups["central"].seat_value
        ):
            entry["value"] = _rounded(rollup.seat_value, 0)
        variants[name] = entry
    return {
        **common,
        "kind": KINDS["constituencies"],
        "resource": RESOURCES["constituencies"],
        "description": (
            "Land under homes by July 2024 constituency in England and Wales, every "
            "variant: cells split by their homes' output areas, social homes by their "
            "authority's household shares (microcosm#1524)."
        ),
        "constituencies": [str(code) for code in geo.seat_codes],
        "variants": variants,
        "digests": digests,
    }


# ----------------------------------------------------------------------------
# Stage: cells


def _cells(args: argparse.Namespace, run: RunLog) -> dict[str, dict]:
    _need(
        args,
        "hpi_csv",
        "hpi_manifest",
        "ctsop1_zip",
        "ctsop1_manifest",
        "ctsop3_zip",
        "ctsop3_manifest",
        "ctsop4_zip",
        "ctsop4_manifest",
        "tpa_config",
        "tpa_cells",
        "tpa_manifest",
        "ew_support",
        "ledger_facts",
    )
    with run.stage("dwellings"):
        dwellings, records, dwelling_receipts, postcodes = _dwellings(args)
    with run.stage("attributes"):
        linked, social_cells, attribute_receipts, ucl_records = _attributes(args)
    records |= ucl_records
    with run.stage("inputs"):
        records["uk_hpi"] = _single(
            args.hpi_manifest, args.hpi_csv, skip=args.skip_verify
        )
        for table in ("ctsop1", "ctsop3", "ctsop4"):
            records[f"voa_{table}"] = _single(
                getattr(args, f"{table}_manifest"),
                getattr(args, f"{table}_zip"),
                skip=args.skip_verify,
            )
        _, tpa_files = pinned_inputs.manifest_files(args.tpa_manifest)
        tpa_keys = {
            key: path
            for key, entry in tpa_files.items()
            for path in (args.tpa_config, args.tpa_cells)
            if entry["filename"] == path.name
        }
        records["tpa"] = _verify_package(
            args.tpa_manifest, tpa_keys, skip=args.skip_verify
        )
        support = _support(
            args.ew_support, args.support_provenance, skip=args.skip_verify
        )
        geo = _geography(support)
        to_roster, to_alias = _aliases(args.crosswalk)
        hpi = hv.load_hpi(args.hpi_csv)
        facts = ref.read_feed_facts(args.ledger_facts)
        stock = _stock(args, geo, to_roster, to_alias)
        costs = _costs(args.tpa_config, geo, stock)
        outside = ~social_cells["local_authority"].isin(geo.la_codes.astype(str))
        attribute_receipts["social"] = {
            **attribute_receipts["social"],
            "certificates_outside_roster": int(
                social_cells.loc[outside, "certificates"].sum()
            ),
            "authorities_outside_roster": sorted(
                set(social_cells.loc[outside, "local_authority"].astype(str))
            ),
        }
        tpa_meta, tpa_cells = ref.parse_tpa_cells(args.tpa_cells.read_bytes())
        tpa_cells = {to_roster.get(code, code): row for code, row in tpa_cells.items()}
        levels = _level_inputs(args, facts)
        split = _uk_split(args, facts, stock)
        net = levels["dwellings_net_capital_stock"]["value"]
        gross = levels["dwellings_gross_capital_stock"]["value"]
        depreciation_target = (
            (1.0 - net / gross) if (net is not None and gross) else None
        )
        epc_floor = ref.epc_mean_floor_area(
            facts,
            spec_prefix=declared.EPC_D3_SPEC,
            code_aliases=to_roster,
            predecessors=declared.AUTHORITY_PREDECESSORS,
        )
    with run.stage("homes"):
        homes = _homes(args, dwellings, postcodes, support, geo, linked, hpi)
        del dwellings, linked
    tenure_mode = "msoa"
    with run.stage("value block"):
        tenure = _tenure(facts, geo, stock, mode=tenure_mode)
        block = _value_block(geo, homes, stock, tenure, strata="type")
    la_names = json.loads(args.la_names.read_text(encoding="utf-8"))["areas"]
    bands = len(declared.BANDS)

    def private_mean_of_la(blk: ValueBlock) -> np.ndarray:
        cells = blk.cell_rows
        la = geo.la_of_msoa[cells // bands]
        count = blk.private_count[cells]
        total = np.bincount(
            la,
            weights=count * np.nan_to_num(blk.mean_value[cells]),
            minlength=len(geo.la_codes),
        )
        mass = np.bincount(la, weights=count, minlength=len(geo.la_codes))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(mass > 0, total / mass, 0.0)

    social_value, social_receipt = _social_values(
        facts, geo, hpi, private_mean_of_la(block)
    )

    def structure_for(variant, blk: ValueBlock, ten: Tenure) -> Structure:
        return _structure(
            variant,
            geo=geo,
            homes=homes,
            assignment=blk.assignment,
            pools=blk.pools,
            stock=stock,
            tenure=ten,
            costs=costs,
            social_cells=social_cells,
            epc_floor=epc_floor,
            depreciation_target=depreciation_target,
            bins=blk.bins,
            private_count=blk.private_count,
        )

    def level_for(structure: Structure, blk: ValueBlock, ten: Tenure) -> Level:
        cells = blk.cell_rows
        value = np.concatenate([np.nan_to_num(blk.mean_value[cells]), social_value])
        cost = np.concatenate(
            [np.nan_to_num(structure.unit_cost[cells]), structure.social_unit_cost]
        )
        count = np.concatenate([blk.private_count[cells], ten.social_of_la])
        return _solve_level(levels, split, value=value, cost=cost, count=count)

    rollups: dict[str, Rollup] = {}
    levels_by_variant: dict[str, dict] = {}
    structures: dict[str, Structure] = {}
    with run.stage("structure variants"):
        for variant in declared.STRUCTURE_VARIANTS:
            name = str(variant["name"])
            structure = structure_for(variant, block, tenure)
            level = level_for(structure, block, tenure)
            kappa_rule = variant.get("kappa")
            if isinstance(kappa_rule, float):
                kappa = kappa_rule
            elif kappa_rule in ("a", "b"):
                kappa = level.solutions[kappa_rule]["kappa"]
                if kappa is None:
                    levels_by_variant[name] = {"status": "absent", "target": kappa_rule}
                    continue
            else:
                kappa = level.kappa
            structures[name] = structure
            rollups[name] = _rollup(
                geo,
                block,
                structure,
                tenure,
                social_value,
                float(kappa),
                residual="rank" if variant.get("residual") == "rank" else "pooled",
            )
            levels_by_variant[name] = {
                "kappa": _r(float(kappa), 6),
                "solutions": {
                    k: {
                        kk: _r(vv, 6) if isinstance(vv, float) else vv
                        for kk, vv in v.items()
                    }
                    for k, v in level.solutions.items()
                },
                **structure.receipt,
            }
            if name == "central":
                central_level = level
                _log(
                    f"κ (central, target {level.target}): {level.kappa:.6f} [{level.status}]"
                )
    value_variants = {}
    if args.variants == "all":
        with run.stage("value variants"):
            for variant in declared.VALUE_VARIANTS:
                name = str(variant["name"])
                ten = (
                    _tenure(facts, geo, stock, mode="uniform")
                    if variant.get("tenure") == "uniform"
                    else tenure
                )
                blk = _value_block(
                    geo, homes, stock, ten, strata=str(variant.get("strata", "type"))
                )
                structure = structure_for({"name": "central"}, blk, ten)
                level = level_for(structure, blk, ten)
                rollups[name] = _rollup(
                    geo, blk, structure, ten, social_value, level.kappa
                )
                value_variants[name] = {
                    "kappa": _r(level.kappa, 6),
                    "assignment": blk.assignment.receipt,
                    "tenure": ten.status,
                }
    with run.stage("comparisons"):
        mhclg = _mhclg_pattern(facts, geo, stock, rollups["central"], to_roster)
        decision = _decision_rule(geo, rollups, central_level, costs, homes, block)
        measurements = _measurements(
            facts, geo, rollups, block, stock, tenure, mhclg, hpi, to_roster
        )
        recorded = _also_recorded(
            facts,
            geo,
            rollups,
            mhclg,
            tpa_cells,
            levels,
            homes,
            structures["central"],
            block,
            central_level,
        )
    flags: dict[str, list[str]] = {}
    fallen = np.bincount(
        geo.la_of_msoa[block.cell_rows // bands],
        weights=(block.pools.means_level[block.cell_rows] > 0).astype(float),
        minlength=len(geo.la_codes),
    )
    for i, code in enumerate(geo.la_codes):
        code = str(code)
        if code == declared.ISLES_OF_SCILLY:
            flags.setdefault(code, []).append("scilly_imputed")
        if code in to_alias:
            flags.setdefault(code, []).append("code_alias")
        if fallen[i] > 0:
            flags.setdefault(code, []).append("cells_fell_back")
    for name, entry in mhclg.items():
        for code in entry["capped"]:
            flags.setdefault(code, []).append(f"mhclg_capped:{name}")
        for code in entry["flags"]["missing"]:
            flags.setdefault(code, []).append(f"mhclg_missing:{name}")

    feed = feed_inputs(facts)
    inputs_payload = {
        "registrations": {
            name: record["registration"] for name, record in sorted(records.items())
        },
        "support_sha256": str(support["__sha256__"]),
        "ledger_feed": _feed_identity(args.ledger_manifest),
        "feed_inputs_sha256": ref.feed_inputs_fingerprint(feed),
        "declarations_sha256": _declarations_digest(),
        "method": declared.METHOD_VERSION,
    }
    inputs_sha256 = derived_resource.payload_sha256(inputs_payload)
    common = {
        "schema_version": 1,
        "built_on": args.built_on,
        "method": declared.METHOD_VERSION,
        "inputs_sha256": inputs_sha256,
        "basis_year": declared.BASIS_YEAR,
        "stock_date": declared.STOCK_DATE,
    }
    value = _value_payload(geo, block, stock, tenure, social_value, common)
    value_sha = derived_resource.payload_sha256(
        {k: v for k, v in value.items() if k != "built_on"}
    )
    structure_payload = _structure_payload(
        geo,
        block,
        structures["central"],
        rollups["central"],
        central_level,
        costs,
        common,
        value_sha,
    )
    structure_sha = derived_resource.payload_sha256(
        {k: v for k, v in structure_payload.items() if k != "built_on"}
    )
    digests = {"value_block_sha256": value_sha, "structure_block_sha256": structure_sha}
    authorities = _authority_payload(
        geo, la_names, stock, tenure, rollups, mhclg, tpa_cells, flags, common, digests
    )
    constituencies = _constituency_payload(geo, rollups, common, digests)
    digests_all = {
        **digests,
        "authorities_sha256": derived_resource.payload_sha256(
            {k: v for k, v in authorities.items() if k != "built_on"}
        ),
        "constituencies_sha256": derived_resource.payload_sha256(
            {k: v for k, v in constituencies.items() if k != "built_on"}
        ),
    }
    licences = {name: record["licence"] for name, record in sorted(records.items())}
    provenance = {
        **common,
        "kind": KINDS["provenance"],
        "resource": RESOURCES["provenance"],
        "issues": {
            "builder": declared.ISSUE,
            "epic": "PolicyEngine/microcosm#1521",
            "level_inputs": "PolicyEngine/chronicle#438",
        },
        "tool": "tools/build_uk_stock_cells.py",
        "inputs": inputs_payload,
        "licences": licences,
        "parameters": {
            "pooling": {
                "levels": list(declared.POOLING.levels),
                "minimums": dict(declared.POOLING.minimums),
            },
            "knot_grid": list(declared.KNOT_GRID),
            "knot_cap": declared.KNOT_CAP,
            "variants": args.variants,
            "level_parameters": {
                name: entry
                for name, entry in levels.items()
                if entry["status"] == "parameter"
            },
            "floor_area_bounds_m2": list(declared.FLOOR_AREA_BOUNDS_M2),
            "ucl_columns": declared.UCL_LINKED_COLUMNS,
            "ucl_columns_verified_against": declared.UCL_COLUMNS_VERIFIED_AGAINST,
            "pandas": pd.__version__,
        },
        "level": {
            "central": {
                "kappa": _r(central_level.kappa, 6),
                "status": central_level.status,
                "target": central_level.target,
            },
            "inputs": levels,
            "uk_split": split,
            "depreciation_target": _r(depreciation_target, 6)
            if depreciation_target is not None
            else None,
            "by_variant": levels_by_variant,
            "value_variants": value_variants,
        },
        "receipts": {
            "dwellings": {
                k: v for k, v in dwelling_receipts.items() if k != "transfers_by_year"
            },
            "attributes": attribute_receipts,
            "homes": homes.receipts,
            "stock": stock.receipts,
            "tenure": tenure.receipt,
            "assignment": block.assignment.receipt,
            "costs": costs.receipt,
            "social_values": social_receipt,
            "pools": {
                "cells": int(len(block.cell_rows)),
                "means_level_counts": np.bincount(
                    block.pools.means_level[block.cell_rows], minlength=4
                ).tolist(),
                "knots_level_counts": np.bincount(
                    block.pools.knots_level[block.cell_rows], minlength=4
                ).tolist(),
                "knots_below_minimum_at_top": int(
                    block.pools.knots_below[block.cell_rows].sum()
                ),
            },
            "private_cells_below_homes_sold": int(
                (
                    block.private_count[block.cell_rows]
                    < block.pools.sampled[0][block.cell_rows]
                ).sum()
            ),
            "tpa_meta": tpa_meta,
        },
        "decision_rule": decision,
        "measurements": measurements,
        "also_recorded": recorded,
        "digests": digests_all,
    }
    return {
        "value": value,
        "structure": structure_payload,
        "authorities": authorities,
        "constituencies": constituencies,
        "provenance": provenance,
    }


# ----------------------------------------------------------------------------
# Main


def _set_threads(count: int) -> None:
    import pyarrow as pa

    pa.set_cpu_count(max(1, count))
    pa.set_io_thread_count(max(1, count))


def _write_or_check(args: argparse.Namespace, payloads: Mapping[str, dict]) -> int:
    if args.check:
        reasons = [
            reason
            for name, payload in payloads.items()
            for reason in derived_resource.stale_reasons(
                args.out_dir / RESOURCES[name], payload
            )
        ]
        if reasons:
            raise SystemExit("stale:\n  " + "\n  ".join(reasons))
        _log("current")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        (args.out_dir / RESOURCES[name]).write_text(
            derived_resource.render_json(payload), encoding="utf-8"
        )
    _log(f"Wrote {len(payloads)} resources to {args.out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _set_threads(args.threads)
    run = RunLog()
    try:
        if args.stage == "discover":
            _discover(args)
            return 0
        if args.stage == "dwellings":
            with run.stage("dwellings"):
                _, _, receipts, _ = _dwellings(args)
            _log(
                json.dumps(
                    {k: v for k, v in receipts.items() if k != "transfers_by_year"},
                    indent=2,
                )
            )
            return 0
        if args.stage == "attributes":
            with run.stage("attributes"):
                _, _, receipts, _ = _attributes(args)
            _log(json.dumps(receipts, indent=2, sort_keys=True))
            return 0
        payloads = _cells(args, run)
        return _write_or_check(args, payloads)
    finally:
        if run.stages:
            log = args.cache_dir / "stock-cells-run.json"
            log.write_text(
                json.dumps({"stage": args.stage, "stages": run.stages}, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    raise SystemExit(main())
