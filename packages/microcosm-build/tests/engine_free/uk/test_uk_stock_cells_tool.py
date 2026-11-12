"""The stock cell builder end to end, on a synthetic England and Wales (microcosm#1524).

Every input the tool reads is synthesised here, with Chronicle-style
manifests that pin its bytes: Price Paid yearly files, the UK HPI, the NSPL,
the atomic support, VOA's CTSOP 1.1, 3.1 and 4.1, UCL's two bundles, TPA's
files and a feed. Two builds from empty caches must write the same bytes,
``--check`` must pass on them and fail on an edit, and the resources must
hold the invariants the committed ones are tested for.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.weighted_quantiles import check_knots
from test_support.paths import paths_for

PATHS = paths_for("microcosm-build")
TOOL = PATHS.repository / "tools" / "build_uk_stock_cells.py"

AUTHORITIES = {
    "E06000001": ("E12000001", ["E02000001", "E02000002"]),
    "E06000002": ("E12000001", ["E02000003", "E02000004"]),
    "W06000001": ("W99999999", ["W02000001", "W02000002"]),
}
SEATS = {
    "E02000001": "E14001001",
    "E02000002": "E14001001",
    "E02000003": "E14001001",
    "E02000004": "E14001002",
    "W02000001": "W07000081",
    "W02000002": "W07000081",
}
BANDS = list("ABCDEFGHI")
TYPES = list("DSTF")


def _load_tool():
    # Registered before it runs: its dataclasses look their module up.
    if "build_uk_stock_cells" in sys.modules:
        return sys.modules["build_uk_stock_cells"]
    spec = importlib.util.spec_from_file_location("build_uk_stock_cells", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(path: Path, package: str, files: dict[str, Path]) -> Path:
    lines = [
        "source_id: synthetic",
        f"package_id: {package}",
        "license: Open Government Licence v3.0",
        "attribution: 'Source: a synthetic publisher.'",
        "registration_kind: raw_only",
        "files:",
    ]
    for key, file in files.items():
        payload = file.read_bytes()
        lines += [
            f"  {key}:",
            f"    filename: {file.name}",
            f"    sha256: {hashlib.sha256(payload).hexdigest()}",
            f"    size_bytes: {len(payload)}",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _zip(
    path: Path, members: dict[str, bytes], *, stored: set[str] = frozenset()
) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(
                zipfile.ZipInfo(name),
                payload,
                compress_type=zipfile.ZIP_STORED
                if name in stored
                else zipfile.ZIP_DEFLATED,
            )
    return path


def _csv(header: list[str], rows: list[list]) -> bytes:
    lines = [",".join(header)] + [",".join(f'"{v}"' for v in row) for row in rows]
    return ("\n".join(lines) + "\n").encode("utf-8")


def _rounded(value: float) -> str:
    """VOA's printing: the nearest ten, [c] for 1 to 4."""
    if value <= 0:
        return "0"
    if value < 5:
        return "[c]"
    return str(int(round(value / 10.0)) * 10)


def _fact(
    concept, code, value, *, level="local_authority", spec="", period=2026, **dims
):
    return {
        "aggregate_fact_key": f"synthetic:{concept}:{code}:{period}:{sorted(dims.items())}",
        "observed_measure": {"source_concept": concept},
        "geography": {"level": level, "id": code},
        "layout": {"record_set_spec_id": spec},
        "period": {"value": period},
        "dimensions": dims,
        "value": value,
    }


def _world(root: Path) -> list[str]:
    """Write every input and return the tool's arguments for them."""
    rng = np.random.RandomState(1524)
    root.mkdir(parents=True, exist_ok=True)
    # Geography: two output areas per MSOA, three postcodes per output area.
    oas, rows = [], []
    for la, (region, msoas) in AUTHORITIES.items():
        for msoa in msoas:
            for _ in range(2):
                oa = f"{msoa[0]}0{len(oas) + 1:07d}"
                oas.append(oa)
                rows.append((oa, msoa, la, region, SEATS[msoa]))
    support = {
        "area": np.array([r[0] for r in rows]),
        "msoa_code": np.array([r[1] for r in rows]),
        "lsoa_code": np.array([r[0].replace("00", "01", 1) for r in rows]),
        "local_authority_code": np.array([r[2] for r in rows]),
        "region_code": np.array([r[3] for r in rows]),
        "constituency_code": np.array([r[4] for r in rows]),
        "households": np.full(len(rows), 400.0),
    }
    buffer = io.BytesIO()
    metadata = {"columns": {name: {} for name in support}}
    np.savez(buffer, metadata_json=np.array(json.dumps(metadata)), **support)
    support_path = root / "uk_ew_output_area_2021_support.npz"
    support_path.write_bytes(buffer.getvalue())
    provenance = root / "supports.provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "supports": {
                    "uk_ew_output_area_2021": {
                        "sha256": hashlib.sha256(buffer.getvalue()).hexdigest()
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    postcodes = [(f"ZZ{i:02d} {k}AA", oa) for i, oa in enumerate(oas) for k in range(3)]
    nspl = _zip(
        root / "NSPL_AUG_2026.zip",
        {
            "Data/NSPL_AUG_2026_UK.csv": _csv(
                ["pcds", "oa21cd"], [list(p) for p in postcodes]
            )
        },
    )
    # Price Paid: dwellings with one or two sales over 2023 and 2024.
    oa_la = {r[0]: r[2] for r in rows}
    level = {"E06000001": 1.0, "E06000002": 0.8, "W06000001": 0.7}
    dwellings = []
    for n in range(1500):
        postcode, oa = postcodes[rng.randint(len(postcodes))]
        kind = TYPES[rng.choice(4, p=[0.25, 0.3, 0.25, 0.2])]
        base = {"D": 320_000, "S": 210_000, "T": 160_000, "F": 120_000}[kind]
        price = int(base * level[oa_la[oa]] * float(np.exp(rng.normal(0, 0.35))))
        dwellings.append((n, postcode, kind, price, oa))
    ppd_rows = {2023: [], 2024: []}
    for n, postcode, kind, price, _ in dwellings:
        sales = [(2024, rng.randint(1, 13))]
        if n % 3 == 0:
            sales.insert(0, (2023, rng.randint(1, 13)))
        for i, (year, month) in enumerate(sales):
            new = "Y" if (n % 17 == 0 and i == 0) else "N"
            ppd_rows[year].append(
                [
                    f"{{TX-{n:05d}-{i}}}",
                    str(int(price * (0.95 if year == 2023 else 1.0))),
                    f"{year}-{month:02d}-15 00:00",
                    postcode,
                    kind,
                    new,
                    "F",
                    str(n),
                    "",
                    "HIGH ST",
                    "",
                    "TOWN",
                    "DIST",
                    "CTY",
                    "A",
                    "A",
                ]
            )
    ppd_dir = root / "ppd"
    ppd_dir.mkdir()
    ppd_files = {}
    for year, lines in ppd_rows.items():
        path = ppd_dir / f"pp-{year}.csv"
        path.write_bytes(_csv([], lines).split(b"\n", 1)[1])
        ppd_files[str(year)] = path
    # UK HPI: every area, every month of 2023 and 2024.
    hpi_rows = []
    for year in (2023, 2024):
        for month in range(1, 13):
            level_now = 100 + (year - 2023) * 12 + month
            for code, name in (
                ("E06000001", "Hartlepool"),
                ("E06000002", "Middlesbrough"),
                ("W06000001", "Isle of Anglesey"),
                ("E12000001", "North East"),
                ("W92000004", "Wales"),
                ("E92000001", "England"),
            ):
                hpi_rows.append(
                    [f"01/{month:02d}/{year}", name, code] + [str(level_now)] * 5
                )
    hpi = root / "UK-HPI-full-file.csv"
    hpi.write_bytes(
        _csv(
            [
                "Date",
                "RegionName",
                "AreaCode",
                "DetachedIndex",
                "SemiDetachedIndex",
                "TerracedIndex",
                "FlatIndex",
                "Index",
            ],
            hpi_rows,
        ).replace(b'"', b"")
    )
    # VOA: more homes than were ever sold, in bands A to E (and I in Wales).
    sold = pd.DataFrame(dwellings, columns=["n", "postcode", "kind", "price", "oa"])
    oa_msoa = {r[0]: r[1] for r in rows}
    sold["msoa"] = sold["oa"].map(oa_msoa)
    band_mix = np.array([0.22, 0.3, 0.24, 0.14, 0.1, 0, 0, 0, 0])
    ctsop1, ctsop3, ctsop4 = [], [], []
    type_mix = {"D": 0.22, "S": 0.3, "T": 0.24, "F": 0.24}
    periods = [
        "bp_pre_1900",
        "bp_1900_1918",
        "bp_1919_1944",
        "bp_1945_1972",
        "bp_1973_1999",
        "bp_2000_2008",
        "bp_2009",
        "bp_unkw",
    ]
    period_mix = np.array([0.05, 0.1, 0.15, 0.3, 0.25, 0.1, 0.03, 0.02])
    totals_by_la: dict[str, float] = {}
    for la, (_, msoas) in AUTHORITIES.items():
        for msoa in msoas:
            stock = 1.5 * (sold["msoa"] == msoa).sum()
            mix = band_mix.copy()
            if la.startswith("W"):
                mix[8] = 0.01
            counts = [_rounded(stock * share) for share in mix]
            if not la.startswith("W"):
                counts[8] = "[z]"
            printed = [
                0 if c in ("[z]", "0") else (2.5 if c == "[c]" else float(c))
                for c in counts
            ]
            total = sum(printed)
            ctsop1.append(["MSOA", "", msoa, msoa, *counts, str(int(round(total)))])
            totals_by_la[la] = totals_by_la.get(la, 0) + total
            for label, band_total in [("All", total)] + list(
                zip(BANDS, printed, strict=True)
            ):
                if band_total <= 0:
                    continue
                columns, cells = [], []
                for kind, share in type_mix.items():
                    houses = band_total * share
                    if kind == "F":
                        cells += [_rounded(houses), _rounded(houses)]
                        columns += ["flat_mais_1", "flat_mais_total"]
                    else:
                        name = {
                            "D": "house_detached",
                            "S": "house_semi",
                            "T": "house_terraced",
                        }[kind]
                        cells += [_rounded(houses * 0.9), _rounded(houses * 0.9)]
                        columns += [f"{name}_1", f"{name}_total"]
                bungalows = _rounded(band_total * 0.07)
                other = _rounded(band_total * 0.02)
                cells += [bungalows, bungalows, "0", "0", other]
                columns += [
                    "bungalow_1",
                    "bungalow_total",
                    "annexe",
                    "caravan_houseboat_mobilehome",
                    "unknown",
                ]
                listed = [
                    0 if c == "0" else (2.5 if c == "[c]" else float(c))
                    for c, col in zip(cells, columns, strict=True)
                    if not col.endswith("_total")
                ]
                ctsop3.append(
                    [
                        "MSOA",
                        "",
                        msoa,
                        msoa,
                        label,
                        *cells,
                        str(int(round(sum(listed)))),
                    ]
                )
                if label != "All":
                    split = [_rounded(band_total * s) for s in period_mix]
                    listed = [
                        0 if c == "0" else (2.5 if c == "[c]" else float(c))
                        for c in split
                    ]
                    ctsop4.append(
                        [
                            "MSOA",
                            "",
                            msoa,
                            msoa,
                            label,
                            *split,
                            str(int(round(sum(listed)))),
                        ]
                    )
    for la, total in totals_by_la.items():
        code = la
        ctsop1.append(["LAUA", "", code, code, *["0"] * 9, str(int(round(total)))])
        ctsop3.append(
            [
                "LAUA",
                "",
                code,
                code,
                "All",
                "0",
                "0",
                "100",
                "100",
                "100",
                "100",
                "100",
                "100",
                "100",
                "100",
                "0",
                "0",
                "0",
                "0",
                "0",
                "400",
            ]
        )
    ctsop1.append(
        [
            "ENGWAL",
            "",
            "K04000001",
            "England and Wales",
            *["0"] * 9,
            str(int(round(sum(totals_by_la.values())))),
        ]
    )
    keys = ["geography", "ba_code", "ecode", "area_name"]
    type_header = keys + [
        "band",
        "flat_mais_1",
        "flat_mais_total",
        "house_detached_1",
        "house_detached_total",
        "house_semi_1",
        "house_semi_total",
        "house_terraced_1",
        "house_terraced_total",
        "bungalow_1",
        "bungalow_total",
        "annexe",
        "caravan_houseboat_mobilehome",
        "unknown",
        "all_properties",
    ]

    # Reorder each 3.1 row's cells to the header: the rows above list types in
    # D, S, T, F order with flats first in the header.
    def reorder(row):
        values = dict(
            zip(
                [
                    "house_detached_1",
                    "house_detached_total",
                    "house_semi_1",
                    "house_semi_total",
                    "house_terraced_1",
                    "house_terraced_total",
                    "flat_mais_1",
                    "flat_mais_total",
                    "bungalow_1",
                    "bungalow_total",
                    "annexe",
                    "caravan_houseboat_mobilehome",
                    "unknown",
                ],
                row[5:-1],
                strict=True,
            )
        )
        return row[:5] + [values[c] for c in type_header[5:-1]] + row[-1:]

    ctsop1_zip = _zip(
        root / "CTSOP_1_1.zip",
        {
            "CTSOP_1_1/CTSOP1_1_2026_03_31.csv": _csv(
                keys + [f"band_{b.lower()}" for b in BANDS] + ["all_properties"], ctsop1
            )
        },
    )
    ctsop3_rows = [r if r[0] == "LAUA" else reorder(r) for r in ctsop3]
    ctsop3_rows = [
        r
        if r[0] != "LAUA"
        else r[:5]
        + ["0", "0", "60", "60", "30", "30", "10", "10", "0", "0", "0", "0", "0", "100"]
        for r in ctsop3_rows
    ]
    ctsop3_zip = _zip(
        root / "CTSOP_3_1__1_.zip",
        {"CTSOP_3_1/CTSOP3_1_2026_03_31.csv": _csv(type_header, ctsop3_rows)},
    )
    ctsop4_zip = _zip(
        root / "CTSOP_4_1.zip",
        {
            "CTSOP_4_1/CTSOP4_1_2026_03_31.csv": _csv(
                keys + ["band", *periods, "all_properties"], ctsop4
            )
        },
    )
    # UCL: floor area and age band for four in five dwellings.
    ages = [
        "England and Wales: 1930-1949",
        "England and Wales: 1967-1975",
        "1991-1995",
        "England and Wales: 2012 onwards",
        "NO DATA!",
    ]
    linked = []
    for n, postcode, kind, price, _ in dwellings:
        if n % 5 == 4:
            continue
        floor = {"D": 120, "S": 90, "T": 80, "F": 60}[kind] * float(
            np.exp(rng.normal(0, 0.15))
        )
        linked.append(
            [
                f"{{TX-{n:05d}-0}}",
                str(price),
                "2023-06-15 00:00",
                postcode,
                str(n),
                "",
                f"{floor:.1f}",
                ages[n % len(ages)],
            ]
        )
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "tranall_link_02122023.csv",
            _csv(
                [
                    "transactionid",
                    "price",
                    "dateoftransfer",
                    "postcode",
                    "paon",
                    "saon",
                    "tfarea",
                    "CONSTRUCTION_AGE_BAND",
                ],
                linked,
            ),
        )
    bundle_1 = _zip(
        root / "856542_bundle_1.zip",
        {
            "tranall_link_02122023.zip": inner.getvalue(),
            "856542_readme.txt": b"Fields: transactionid, tfarea, CONSTRUCTION_AGE_BAND\n",
        },
        stored={"tranall_link_02122023.zip"},
    )
    certificates = []
    for la in AUTHORITIES:
        for i in range(40):
            kind, form = [
                ("Flat", "Mid-Terrace"),
                ("House", "Semi-Detached"),
                ("House", "Mid-Terrace"),
            ][i % 3]
            certificates.append(
                [
                    str(len(certificates)),
                    la,
                    "rental (social)",
                    kind,
                    form,
                    str(55 + i % 25),
                    ages[i % 4],
                    "2019-03-01",
                    f"{la}-{i}",
                ]
            )
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "certificates.csv",
            _csv(
                [
                    "LMK_KEY",
                    "LOCAL_AUTHORITY",
                    "TENURE",
                    "PROPERTY_TYPE",
                    "BUILT_FORM",
                    "TOTAL_FLOOR_AREA",
                    "CONSTRUCTION_AGE_BAND",
                    "LODGEMENT_DATE",
                    "UPRN",
                ],
                certificates,
            ),
        )
    bundle_2 = _zip(
        root / "856542_bundle_2.zip",
        {"epc20_id.zip": inner.getvalue()},
        stored={"epc20_id.zip"},
    )
    # TPA.
    regions = ", ".join(f'"E1200000{i}": 1.0' for i in range(1, 10))
    tpa_config = root / "config.py"
    tpa_config.write_text(
        'raise SystemExit("never run")\n'
        'BASE_COST_PER_M2 = {"D": 2100, "S": 1950, "T": 1900, "F": 2400}\n'
        f"REGION_FACTOR = {{{regions}}}\n",
        encoding="utf-8",
    )
    tpa_cells = root / "cells.json"
    tpa_cells.write_text(
        json.dumps(
            {
                "meta": {"kappa": 0.7054},
                "las": {
                    "E06000001": {
                        "bands": {
                            "A": {
                                "count": 10,
                                "value_deciles": [100, 200],
                                "land_deciles": [40, 90],
                            }
                        }
                    },
                    "E06000002": {
                        "bands": {
                            "A": {
                                "count": 10,
                                "value_deciles": [100, 200],
                                "land_deciles": [20, 60],
                            }
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    tpa_licence = root / "LICENSE"
    tpa_licence.write_text("MIT\n", encoding="utf-8")
    # The feed.
    facts = []
    for la in AUTHORITIES:
        for bucket, value in (
            ("all_households", 1000),
            ("social_rent", 180),
            ("owned_outright", 330),
            ("owned_mortgage", 300),
        ):
            facts.append(
                _fact(
                    "ons.census2021_households",
                    la,
                    value,
                    spec=f"uk.local_geography.tenure.{bucket}.v1",
                    period=2021,
                )
            )
        if la.startswith("E"):
            facts.append(
                _fact(
                    "mhclg.council_tax.average_band_d_billing_authority_area",
                    la,
                    2300.0,
                    spec="uk.local_geography.council_tax.levels_england.fy2026.v1",
                )
            )
            facts.append(
                _fact(
                    "mhclg.council_tax.council_tax_base",
                    la,
                    500.0,
                    spec="uk.local_geography.council_tax.levels_england.fy2026.v1",
                )
            )
            for sheet in (
                "mhclg.residential_land_value_floor_adjusted",
                "mhclg.residential_land_value_before_floor",
            ):
                for percentile, value in (
                    ("p25", 1.5e6),
                    ("p50", 2.0e6),
                    ("p75", 2.6e6),
                ):
                    facts.append(
                        _fact(
                            sheet,
                            la,
                            value * level[la],
                            period="2023-10",
                            new_build_price_percentile=percentile,
                            site_density_scenario="medium",
                        )
                    )
            facts.append(
                _fact(
                    "mhclg.residential_site_density",
                    la,
                    35.0,
                    period="2023-10",
                    site_density_scenario="medium",
                )
            )
        else:
            facts.append(
                _fact(
                    "welshgov.council_tax.average_band_d",
                    la,
                    2100.0,
                    spec="uk.local_geography.council_tax.levels_wales.band_d.v1",
                )
            )
        facts.append(
            _fact(
                "mhclg.epc_domestic_total_floor_area",
                la,
                9000.0,
                spec="mhclg.epc_d3_2026q3.local_authorities.2025_q4.v1",
                period="2025-Q4",
            )
        )
        facts.append(
            _fact(
                "mhclg.epc_domestic_lodgements",
                la,
                100,
                spec="mhclg.epc_d3_2026q3.local_authorities.2025_q4.v1",
                period="2025-Q4",
            )
        )
    for tenure, value in (
        ("owner_occupiers", 230_000),
        ("private_renters", 150_000),
        ("all_social_renters", 95_000),
    ):
        facts.append(
            _fact(
                "mhclg.mvs_market_value_mean",
                "E12000001",
                value,
                level="region",
                spec="mhclg.mvs2024.at1_3.region.p2024.full.v1",
                period="2024-07",
                ehs_tenure=tenure,
            )
        )
        facts.append(
            _fact(
                "mhclg.mvs_market_value_mean",
                "E92000001",
                value * 1.6,
                level="country",
                spec="mhclg.mvs2024.at1_1a.mean.p2024.full.v1",
                period="2024-07",
                ehs_tenure=tenure,
            )
        )
    facts.append(
        _fact(
            "lps.housing_stock",
            "N92000002",
            848_541,
            level="country",
            spec="lps.housing_stock_lgd.2026.v1",
            period=2026,
        )
    )
    facts.append(
        _fact(
            "ons.land_article_2022.land_underlying_dwellings",
            "K02000001",
            5.2e12,
            level="country",
            spec="ons.land_article_2022.figure_1.cy2021.v1",
            period=2021,
        )
    )
    facts.append(
        _fact(
            "ons.e44n",
            "K02000001",
            4.4e12,
            level="country",
            spec="ons.nbs2025.land.households.cy2024.v1",
            period=2024,
        )
    )
    feed = root / "consumer_facts.jsonl"
    feed.write_text("".join(json.dumps(f) + "\n" for f in facts), encoding="utf-8")
    feed_manifest = root / "manifest.json"
    feed_manifest.write_text(
        json.dumps(
            {
                "facts_sha256": hashlib.sha256(feed.read_bytes()).hexdigest(),
                "fact_row_count": len(facts),
            }
        ),
        encoding="utf-8",
    )
    return [
        "--ppd-dir",
        str(ppd_dir),
        "--ppd-manifest",
        str(_manifest(root / "ppd.yaml", "hmlr-price-paid-synthetic", ppd_files)),
        "--hpi-csv",
        str(hpi),
        "--hpi-manifest",
        str(_manifest(root / "hpi.yaml", "hmlr-uk-hpi-synthetic", {"2026": hpi})),
        "--nspl-zip",
        str(nspl),
        "--nspl-manifest",
        str(_manifest(root / "nspl.yaml", "ons-nspl-synthetic", {"2026": nspl})),
        "--ctsop1-zip",
        str(ctsop1_zip),
        "--ctsop1-manifest",
        str(
            _manifest(root / "c1.yaml", "voa-ctsop1-1-synthetic", {"2026": ctsop1_zip})
        ),
        "--ctsop3-zip",
        str(ctsop3_zip),
        "--ctsop3-manifest",
        str(
            _manifest(root / "c3.yaml", "voa-ctsop3-1-synthetic", {"2026": ctsop3_zip})
        ),
        "--ctsop4-zip",
        str(ctsop4_zip),
        "--ctsop4-manifest",
        str(
            _manifest(root / "c4.yaml", "voa-ctsop4-1-synthetic", {"2026": ctsop4_zip})
        ),
        "--ucl-bundle-1",
        str(bundle_1),
        "--ucl-bundle-2",
        str(bundle_2),
        "--ucl-manifest",
        str(
            _manifest(
                root / "ucl.yaml",
                "ucl-synthetic",
                {"bundle_1": bundle_1, "bundle_2": bundle_2},
            )
        ),
        "--tpa-config",
        str(tpa_config),
        "--tpa-cells",
        str(tpa_cells),
        "--tpa-manifest",
        str(
            _manifest(
                root / "tpa.yaml",
                "tpa-synthetic",
                {
                    "preprocessing_config": tpa_config,
                    "webapp_cells": tpa_cells,
                    "licence": tpa_licence,
                },
            )
        ),
        "--ew-support",
        str(support_path),
        "--support-provenance",
        str(provenance),
        "--ledger-facts",
        str(feed),
        "--ledger-manifest",
        str(feed_manifest),
    ]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tool = _load_tool()
    root = tmp_path_factory.mktemp("stock_cells")
    inputs = _world(root / "inputs")
    outs = []
    for run in ("first", "second"):
        out = root / run
        assert (
            tool.main(
                [
                    *inputs,
                    "--stage",
                    "all",
                    "--variants",
                    "all",
                    "--cache-dir",
                    str(root / f"cache-{run}"),
                    "--out-dir",
                    str(out),
                    "--built-on",
                    "2026-11-12",
                ]
            )
            == 0
        )
        outs.append(out)
    return tool, inputs, root, outs


def _payloads(out: Path) -> dict[str, dict]:
    return {
        name: json.loads((out / file).read_text(encoding="utf-8"))
        for name, file in _load_tool().RESOURCES.items()
    }


def test_two_builds_write_the_same_bytes_and_check_passes(built) -> None:
    tool, inputs, root, (first, second) = built
    for name in tool.RESOURCES.values():
        assert (first / name).read_bytes() == (second / name).read_bytes(), name
    common = [
        *inputs,
        "--stage",
        "all",
        "--variants",
        "all",
        "--cache-dir",
        str(root / "cache-first"),
    ]
    # A later build date alone is not staleness.
    check = [*common, "--out-dir", str(first), "--built-on", "2027-01-01", "--check"]
    assert tool.main(check) == 0
    edited = root / "edited"
    edited.mkdir()
    for name in tool.RESOURCES.values():
        (edited / name).write_bytes((first / name).read_bytes())
    payload = json.loads((edited / tool.RESOURCES["authorities"]).read_text())
    payload["variants"]["central"]["land"][0] += 1
    (edited / tool.RESOURCES["authorities"]).write_text(json.dumps(payload))
    with pytest.raises(SystemExit, match="la_land_shares.json: variants differs"):
        tool.main(
            [*common, "--out-dir", str(edited), "--built-on", "2026-11-12", "--check"]
        )


def test_the_resources_hold_the_invariants(built) -> None:
    tool, _, _, (first, _) = built
    p = _payloads(first)
    assert {p[name]["kind"] for name in p} == set(tool.KINDS.values())
    value, structure, authorities, seats = (
        p["value"],
        p["structure"],
        p["authorities"],
        p["constituencies"],
    )
    # The two blocks align and chain.
    assert len(value["rows"]) == len(structure["rows"]) == len(value["knots"])
    assert (
        structure["value_block_sha256"] == authorities["digests"]["value_block_sha256"]
    )
    # Knots satisfy the quantile reader's contract.
    for knots in value["knots"]:
        check_knots(value["knot_grid"], knots, cap=value["knot_cap"])
    # Each authority appears once; land lies in [0, value] for every variant.
    codes = [row[0] for row in authorities["rows"]]
    assert codes == sorted(AUTHORITIES) and len(set(codes)) == len(codes)
    central_value = np.array(authorities["variants"]["central"]["value"], dtype=float)
    for name, entry in authorities["variants"].items():
        land = np.array(entry["land"], dtype=float)
        values = np.array(entry.get("value", central_value), dtype=float)
        assert np.all(land >= 0) and np.all(land <= values + 1), name
    # Land never rises with κ.
    kappas = {
        n: authorities["variants"][n] for n in ("kappa_0_75", "kappa_1", "kappa_1_25")
    }
    assert np.all(
        np.array(kappas["kappa_0_75"]["land"]) >= np.array(kappas["kappa_1"]["land"])
    )
    assert np.all(
        np.array(kappas["kappa_1"]["land"]) >= np.array(kappas["kappa_1_25"]["land"])
    )
    # Cells sum to their authority, and authorities to the constituencies.
    columns = value["columns"]
    private = np.array([row[columns.index("private_count")] for row in value["rows"]])
    land_per_home = np.array([row[3] for row in structure["rows"]])
    social = {row[0]: row[1] for row in value["social"]["rows"]}
    social_land = {row[0]: row[4] for row in structure["social"]["rows"]}
    msoa_la = {m: la for la, (_, msoas) in AUTHORITIES.items() for m in msoas}
    sums = {la: social[la] * social_land[la] for la in AUTHORITIES}
    for row, count, land in zip(value["rows"], private, land_per_home, strict=True):
        sums[msoa_la[row[0]]] += count * land
    # Stored counts carry six decimals and land per home two.
    assert [sums[c] for c in codes] == pytest.approx(
        authorities["variants"]["central"]["land"], rel=1e-6, abs=2
    )
    assert sum(seats["variants"]["central"]["land"]) == pytest.approx(
        sum(authorities["variants"]["central"]["land"]), rel=1e-6
    )
    # Every check carries a status; the level is TPA's own while the feed lacks E46Y.
    provenance = p["provenance"]
    assert provenance["level"]["central"]["status"].startswith("absent")
    assert provenance["level"]["central"]["kappa"] == 1.0
    for entry in provenance["decision_rule"] + provenance["measurements"]:
        assert entry["status"] in ("pass", "fail", "not_evaluated", "measured"), entry
    # MHCLG's pattern holds England's central land total.
    english = [i for i, c in enumerate(codes) if c.startswith("E")]
    central_land = np.array(authorities["variants"]["central"]["land"], dtype=float)[
        english
    ].sum()
    for name, entry in authorities["mhclg_pattern"].items():
        assert np.nansum(np.array(entry["land"], dtype=float)) == pytest.approx(
            central_land, rel=1e-6, abs=2
        ), name
    # No address in any resource.
    for name in tool.RESOURCES.values():
        text = (first / name).read_text(encoding="utf-8")
        assert "ZZ0" not in text and "HIGH ST" not in text


def test_discover_prints_the_headers_against_the_declarations(built, tmp_path) -> None:
    tool, inputs, _, _ = built
    assert (
        tool.main([*inputs, "--stage", "discover", "--cache-dir", str(tmp_path)]) == 0
    )
    report = json.loads((tmp_path / "discover.json").read_text())
    (linked,) = report["linked_headers"].values()
    assert linked["declared_missing"] == []
    (certificates,) = report["certificate_headers"].values()
    assert certificates["declared_missing"] == []
    assert "error" not in report["ctsop3"] and "error" not in report["ctsop4"]
    assert report["readme"]["member"] == "856542_readme.txt"
