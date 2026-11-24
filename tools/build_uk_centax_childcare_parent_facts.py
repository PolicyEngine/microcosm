#!/usr/bin/env python
"""Build uk/centax_childcare_parent_facts.json from CenTax's registered workbook.

Advani, Flew, Pepin-Hall and Summers (CenTax, September 2026), *Removing the
£100,000 childcare cliff-edge*, publish their data for the charts as
``AdvaniFlewPepinhallSummers2026_ChildcareThreshold_data.xlsx``, which Chronicle
registers raw-only as ``centax-childcare-cliff-edge-2026``. This tool checks a
local copy of the workbook against that registration's pins
(``microcosm.build.pinned_inputs``), reads Figures 3.1, 3.6 and 3.7 from it and
writes five derived facts (see ``uk_runtime/centax_childcare_parents.py``):
three ANI band shares of the parents in (£50k, £200k], and two splits of the
parents in (£80k, £150k], each times ``parent_count``.

``parent_count`` is the baseline build's own number of parents in the aged
(£50k, £200k] at the calibration year, read from the JSON
``tools/diagnose_uk_centax_parents.py`` writes on the baseline H5; the resource
records that build and its H5's SHA-256. The uprating that ages the 2021-22
edges is the installed policyengine-uk's OBR average earnings index, which must
agree with the diagnostics'.

Never read into the resource: CenTax's model outputs (Tables 3.1, 3.2 and 4.1,
the frictions, elasticities, marginal bunchers and every cost) and the figures
kept as diagnostics (Table A.1's levels, Figures 2.1 and 3.12, the band below
£50k, the region groups outside London and Figure 3.3's non-employees), which the
diagnostic reads from the registered workbook itself.

    uv run --no-sync python tools/build_uk_centax_childcare_parent_facts.py \\
        --manifest <chronicle>/db/data/centax/childcare_cliff_edge_2026/manifest.yaml \\
        --workbook AdvaniFlewPepinhallSummers2026_ChildcareThreshold_data.xlsx \\
        --baseline-diagnostics centax_parents_<build>_2025.json --built-on YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import openpyxl

from microcosm.build import pinned_inputs
from microcosm.build.uk_runtime.centax_childcare_parents import (
    CENTAX_ANI_BANDS_2022,
    CENTAX_BOUND_REGION_GROUPS,
    CENTAX_CHRONICLE_PACKAGE_ID,
    CENTAX_CHRONICLE_SOURCE_ID,
    CENTAX_DATA_TAX_YEAR,
    CENTAX_DERIVED_FACT_KEY_PREFIX,
    CENTAX_FACTS_RESOURCE,
    CENTAX_REGION_GROUPS,
    CENTAX_REPORT_URL,
    CENTAX_SHARE_BASE_2022,
    CENTAX_SPLIT_BAND_2022,
    CENTAX_SPLITS,
    CENTAX_WORKBOOK_FILENAME,
    CENTAX_WORKBOOK_SHA256,
    RESOURCE_KIND,
    RESOURCE_SCHEMA_VERSION,
    ROW_SCHEMA_VERSION,
    aged_edges,
    centax_ani_uprating,
    centax_cell_ids,
    load_uk_centax_childcare_parent_facts,
    model_ani_edges,
    rows_sha256,
)

GENERATOR = {
    "tool": "tools/build_uk_centax_childcare_parent_facts.py",
    "version": "uk-centax-childcare-parents-v2",
}
#: The diagnostics schema whose ``parent_count`` this tool scales by.
DIAGNOSTICS_SCHEMA = "microcosm.uk.centax_parent_diagnostics.v2"
DEFAULT_OUT = (
    Path(__file__).resolve().parents[1]
    / "packages/microcosm-build/src/microcosm/build/uk"
    / CENTAX_FACTS_RESOURCE
)
#: Sheet, expected header row (from column B) and the bin step in pounds. The
#: bins are labelled by their upper edge: Figure 3.1's labels run from 500 to
#: 200,000, so its first bin is (0, £500] and its last ends at the figure's
#: top; the workbook does not say so in words. The report's notes give £1,000
#: bins for Figures 3.1 and 3.6, but their sheets hold £500 bins and say the
#: report's charts aggregate them. Every band edge here falls on a bin edge of
#: every figure, so a cell sums whole bins; read by lower edges, each cell
#: would move by one bin at each end.
SHEETS: Mapping[str, tuple[tuple[str, ...], int]] = {
    "Fig. 3.1": (("Income Bin (ANI)", "Number of Taxpayer in Income Bin"), 500),
    "Fig. 3.3": (
        (
            "Income Bin (ANI)",
            "Number of Taxpayers in Income Bin (Employees)",
            "Number of Taxpayers in Income Bin (Non-Employees)",
        ),
        1000,
    ),
    "Fig. 3.6": (
        (
            "Income Bin (ANI)",
            "Number of Taxpayers in Income Bin (Males)",
            "Number of Taxpayers in Income Bin (Females)",
        ),
        500,
    ),
    "Fig. 3.7": (
        (
            "Income Bin (ANI)",
            "Number of Taxpayers in Income Bin (London)",
            "Number of Taxpayers in Income Bin (North and Midlands)",
            "Number of Taxpayers in Income Bin (South excl. London)",
            "Number of Taxpayers in Income Bin (Scotland)",
        ),
        1000,
    ),
}
#: Figure 3.7's English columns, by region group.
FIG_3_7_COLUMN = {
    "london": "Number of Taxpayers in Income Bin (London)",
    "north_and_midlands": "Number of Taxpayers in Income Bin (North and Midlands)",
    "south_excl_london": "Number of Taxpayers in Income Bin (South excl. London)",
}
_CELL_LABELS = {
    "ani_50k_to_80k": "ANI over £50,000 to £80,000",
    "ani_80k_to_120k": "ANI over £80,000 to £120,000",
    "ani_120k_to_200k": "ANI over £120,000 to £200,000",
    "female_ani_80k_to_150k": "women, ANI over £80,000 to £150,000",
    "london_ani_80k_to_150k": "London, ANI over £80,000 to £150,000",
}
_CELL_FIGURES = {
    **{band: "Fig. 3.1" for band, _, _ in CENTAX_ANI_BANDS_2022},
    **{f"{split}_ani_80k_to_150k": figure for split, figure in CENTAX_SPLITS},
}
SOURCE_MEASURE_ID = "parents_of_eligible_3_4_year_olds"
SOURCE_CONCEPT = "microcosm.centax_childcare_parents.count"
#: The tax year the cells count, which policyengine-uk dates 2025 (the period
#: value the compile matches at the calibration year).
_TAX_YEAR_START = "{year}-04-06"
_TAX_YEAR_END = "{next_year}-04-05"


def open_registered_workbook(
    manifest_path: Path, workbook_path: Path
) -> tuple[Any, dict[str, Any], dict[str, str], dict[str, Any]]:
    """The workbook, checked against its Chronicle registration.

    Returns the open workbook, the registration record a resource stores
    (``pinned_inputs.registration``), the licence the manifest states and the
    manifest itself. Refuses a manifest of another package or kind, or a file
    whose bytes are not the registered ones.
    """

    manifest, files = pinned_inputs.manifest_files(manifest_path)
    if (
        manifest.get("package_id") != CENTAX_CHRONICLE_PACKAGE_ID
        or manifest.get("source_id") != CENTAX_CHRONICLE_SOURCE_ID
    ):
        raise ValueError(
            f"{manifest_path} is not the {CENTAX_CHRONICLE_PACKAGE_ID} registration."
        )
    if manifest.get("registration_kind") != "raw_only" or len(files) != 1:
        raise ValueError(f"{manifest_path} must register one raw-only workbook.")
    ((key, entry),) = files.items()
    verified = pinned_inputs.verify_pinned_file(workbook_path, entry)
    if (
        verified["sha256"] != CENTAX_WORKBOOK_SHA256
        or verified["filename"] != CENTAX_WORKBOOK_FILENAME
    ):
        raise ValueError(
            f"{manifest_path} does not register the workbook this module pins."
        )
    registration = pinned_inputs.registration(manifest, {key: verified})
    licence = pinned_inputs.licence_of(manifest)
    workbook = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    return workbook, registration, licence, manifest


def read_series(workbook: Any, sheet: str) -> dict[str, dict[int, float]]:
    """A figure's columns by header, each a map from bin upper edge to count.

    Refuses a sheet whose header differs from the one this tool was written
    against, a non-numeric cell inside the bins, or bins that are not one
    step apart from the first step to the last.
    """

    header, step = SHEETS[sheet]
    rows = list(workbook[sheet].iter_rows(values_only=True))
    try:
        start = next(
            index
            for index, row in enumerate(rows)
            if tuple(row[1 : 1 + len(header)]) == header
        )
    except StopIteration:
        raise ValueError(f"{sheet}: header {header!r} not found.") from None
    series: dict[str, dict[int, float]] = {column: {} for column in header[1:]}
    labels: list[int] = []
    for row in rows[start + 1 :]:
        label = row[1]
        if label is None:
            continue
        if not isinstance(label, (int, float)) or isinstance(label, bool):
            raise ValueError(f"{sheet}: non-numeric bin label {label!r}.")
        labels.append(int(label))
        for offset, column in enumerate(header[1:], start=2):
            value = row[offset]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{sheet}: bin {label} {column} is {value!r}.")
            series[column][int(label)] = float(value)
    if labels != list(range(step, labels[-1] + step, step)):
        raise ValueError(f"{sheet}: bins are not every £{step} from £{step}.")
    return series


def band_sum(series: Mapping[int, float], lower: float, upper: float) -> float:
    """Sum of the bins whose upper edge is in (lower, upper]."""

    return float(
        sum(value for label, value in series.items() if lower < label <= upper)
    )


def within_split_band(
    columns: Mapping[str, Mapping[int, float]], numerator: str
) -> float:
    """``numerator``'s share of the columns' parents in (£80k, £150k]."""

    lower, upper = CENTAX_SPLIT_BAND_2022
    whole = sum(band_sum(column, lower, upper) for column in columns.values())
    return band_sum(columns[numerator], lower, upper) / whole


def england_region_columns(workbook: Any) -> dict[str, dict[int, float]]:
    """Figure 3.7's English columns by region group (Scotland left out)."""

    fig37 = read_series(workbook, "Fig. 3.7")
    columns = {group: fig37[column] for group, column in FIG_3_7_COLUMN.items()}
    if set(columns) != set(CENTAX_REGION_GROUPS):
        raise ValueError("Fig. 3.7 groups differ from the module's region groups.")
    return columns


def centax_shares(workbook: Any) -> tuple[dict[str, float], float]:
    """Each bound cell's share of the parents in (£50k, £200k], and the split band's."""

    fig31 = read_series(workbook, "Fig. 3.1")["Number of Taxpayer in Income Bin"]
    fig36 = read_series(workbook, "Fig. 3.6")
    regions = england_region_columns(workbook)
    base_lower, base_upper = CENTAX_SHARE_BASE_2022
    if max(fig31) != base_upper:
        raise ValueError(f"Fig. 3.1 ends at £{max(fig31)}, not £{base_upper:,.0f}.")
    base = band_sum(fig31, base_lower, base_upper)
    shares = {
        band: band_sum(fig31, lower, upper) / base
        for band, lower, upper in CENTAX_ANI_BANDS_2022
    }
    split_share = band_sum(fig31, *CENTAX_SPLIT_BAND_2022) / base
    split_within = {
        "female": within_split_band(
            fig36, "Number of Taxpayers in Income Bin (Females)"
        ),
        **{
            group: within_split_band(regions, group)
            for group in CENTAX_BOUND_REGION_GROUPS
        },
    }
    for split, _ in CENTAX_SPLITS:
        shares[f"{split}_ani_80k_to_150k"] = split_share * split_within[split]
    return shares, split_share


def fact_row(
    *,
    cell: str,
    value: float,
    period: str,
    edges: Sequence[int],
    inputs_sha256: str,
    vintage: str,
) -> dict[str, Any]:
    record_set_spec_id = f"uk.centax_childcare_parents.{cell}.v1"
    identity = {
        "record_set_spec_id": record_set_spec_id,
        "period": period,
        "inputs_sha256": inputs_sha256,
        "method": GENERATOR["version"],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:24]
    label = (
        "Parents of age-eligible 3- and 4-year-olds, "
        f"{_CELL_LABELS[cell]} (2021-22 pounds), England"
    )
    tax_year = f"{period}-{(int(period) + 1) % 100:02d}"
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "aggregate_fact_key": CENTAX_DERIVED_FACT_KEY_PREFIX + digest,
        "semantic_fact_key": (
            "microcosm.semantic_fact.centax_childcare_parents.v1:" + digest
        ),
        "value": float(round(value, 3)),
        "value_type": "float",
        "assertion": "observation",
        "provenance_class": "derived_administrative",
        "period": {"type": "calendar_year", "value": period},
        "period_coverage": {
            "start_date": _TAX_YEAR_START.format(year=period),
            "end_date": _TAX_YEAR_END.format(next_year=int(period) + 1),
            "basis": "flow",
            "notes": (
                f"The {tax_year} tax year's ANI, which policyengine-uk dates "
                f"{period}, the period value. CenTax's share for tax year "
                f"{CENTAX_DATA_TAX_YEAR} times the baseline build's parents at "
                f"{period}; the ANI edges are aged to {period} pounds "
                f"({edges[0]:,} to {edges[1]:,})."
            ),
        },
        "geography": {
            "level": "country",
            "id": "E92000001",
            "vintage": "current",
            "name": "England",
        },
        "entity": {"name": "person", "role": "parent"},
        "aggregation": {"method": "sum"},
        "observed_measure": {
            "source_concept": SOURCE_CONCEPT,
            "source_measure_id": SOURCE_MEASURE_ID,
            "unit": "count",
        },
        "measure": {"unit": "count"},
        "concept_alignment": {
            "source_concept": SOURCE_CONCEPT,
            "canonical_concept": "uk.person.count",
            "relation": "derived",
        },
        "dimensions": {"microcosm.centax_cell": cell},
        "dimension_labels": {
            "microcosm.centax_cell": "CenTax parent cell (2021-22 ANI pounds)"
        },
        "dimension_value_labels": {"microcosm.centax_cell": {cell: _CELL_LABELS[cell]}},
        "source": {
            "source_name": "microcosm",
            "source_table": (
                f"CenTax, Removing the £100,000 childcare cliff-edge (2026), "
                f"{_CELL_FIGURES[cell]} (Chronicle {CENTAX_CHRONICLE_PACKAGE_ID}); "
                "Microcosm derivation"
            ),
            "source_file": CENTAX_FACTS_RESOURCE,
            "vintage": vintage,
            "source_sha256": inputs_sha256,
            "url": CENTAX_REPORT_URL,
        },
        "lineage": {"source_record_id": f"microcosm.centax_childcare_parents.{cell}"},
        "layout": {
            "record_set_id": f"uk.centax_childcare_parents.{cell}",
            "record_set_spec_id": record_set_spec_id,
            "measure_id": SOURCE_MEASURE_ID,
        },
        "label": label,
    }


def check_baseline(
    baseline: Mapping[str, Any], *, calibration_year: int, uprating: float
) -> None:
    """Refuse diagnostics that are not this engine's, at this year and range."""

    if baseline.get("schema") != DIAGNOSTICS_SCHEMA:
        raise ValueError(
            f"the baseline diagnostics are {baseline.get('schema')!r}, not "
            f"{DIAGNOSTICS_SCHEMA!r}."
        )
    if int(baseline["year"]) != calibration_year:
        raise ValueError(
            f"the baseline diagnostics are at {baseline['year']}, not {calibration_year}."
        )
    if baseline["model_ani_edges"] != model_ani_edges(uprating):
        raise ValueError("the baseline diagnostics were banded on other ANI edges.")
    if abs(float(baseline["ani_uprating"]) - uprating) > 1e-12:
        raise ValueError("the baseline diagnostics used another ANI uprating.")
    if baseline["parent_count_range"] != aged_edges(*CENTAX_SHARE_BASE_2022, uprating):
        raise ValueError(
            "the baseline diagnostics counted parents over another ANI range."
        )
    installed = importlib.metadata.version("policyengine-uk")
    measured = (baseline.get("resolver") or {}).get("policyengine_uk_version")
    if measured != installed:
        raise ValueError(
            f"the baseline diagnostics ran policyengine-uk {measured}, not the "
            f"installed {installed}."
        )


def build_resource(
    *,
    manifest_path: Path,
    workbook_path: Path,
    baseline: Mapping[str, Any],
    calibration_year: int,
    uprating: float,
    built_on: str,
) -> dict[str, Any]:
    workbook, registration, licence, manifest = open_registered_workbook(
        manifest_path, workbook_path
    )
    check_baseline(baseline, calibration_year=calibration_year, uprating=uprating)
    edges = model_ani_edges(uprating)
    shares, split_share = centax_shares(workbook)
    parent_count = float(baseline["parent_count"])
    vintage = _registered_vintage(manifest_path)
    inputs = {
        "workbook": {
            **registration,
            "source_page": manifest.get("source_page"),
            "licence": licence,
            "report_url": CENTAX_REPORT_URL,
            "figures": ["Fig. 3.1", "Fig. 3.6", "Fig. 3.7"],
        },
        "baseline": {
            "dataset": baseline["dataset"],
            "sha256": baseline["dataset_sha256"],
            "diagnostics_schema": baseline["schema"],
            "policyengine_uk_version": baseline["resolver"]["policyengine_uk_version"],
        },
        "method": GENERATOR["version"],
    }
    inputs_sha256 = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode()
    ).hexdigest()
    period = str(calibration_year)
    rows = [
        fact_row(
            cell=cell,
            value=shares[cell] * parent_count,
            period=period,
            edges=edges[cell],
            inputs_sha256=inputs_sha256,
            vintage=vintage,
        )
        for cell in centax_cell_ids()
    ]
    rows.sort(key=lambda row: row["aggregate_fact_key"])
    base_edges = aged_edges(*CENTAX_SHARE_BASE_2022, uprating)
    return {
        "kind": RESOURCE_KIND,
        "schema_version": RESOURCE_SCHEMA_VERSION,
        "resource": CENTAX_FACTS_RESOURCE,
        "description": (
            "Microcosm-derived counts of parents of age-eligible 3- and 4-year-olds in "
            "England by adjusted net income, from CenTax's HMRC tabulations for "
            f"{CENTAX_DATA_TAX_YEAR} (Advani, Flew, Pepin-Hall and Summers 2026, "
            "Figures 3.1, 3.3, 3.6 and 3.7, from the workbook Chronicle registers as "
            f"{CENTAX_CHRONICLE_PACKAGE_ID}): each published share of the parents "
            "in (£50k, £200k] times the baseline build's own parents in that range. "
            "No CenTax model output, and none of the figures kept as diagnostics."
        ),
        "built_on": built_on,
        "generator": GENERATOR,
        "licence": {"attribution": [str(licence.get("attribution", ""))]},
        "inputs": inputs,
        "inputs_sha256": inputs_sha256,
        "parameters": {
            "calibration_year": calibration_year,
            "ani_uprating": uprating,
            "ani_uprating_basis": (
                "policyengine-uk gov.economic_assumptions.indices.obr.average_earnings, "
                f"{calibration_year} over 2021"
            ),
            "share_base_2022": list(CENTAX_SHARE_BASE_2022),
            "ani_bands_2022": {
                band: [lower, upper] for band, lower, upper in CENTAX_ANI_BANDS_2022
            },
            "split_band_2022": list(CENTAX_SPLIT_BAND_2022),
            "model_ani_edges": edges,
            "parent_count": parent_count,
            "parent_count_basis": (
                "baseline parents of age-eligible 3- and 4-year-olds with annual ANI in "
                f"({base_edges[0]:,}, {base_edges[1]:,}] at {calibration_year}"
            ),
            "shares": shares,
            "split_band_share": split_share,
            "never_targets": [
                "Tables 3.1, 3.2 and 4.1 (bunching estimates)",
                "friction parameters, elasticities and marginal bunchers",
                "every fiscal cost",
            ],
        },
        "row_count": len(rows),
        "rows_sha256": rows_sha256(rows),
        "rows": rows,
    }


def _registered_vintage(manifest_path: Path) -> str:
    """The registered file's vintage (one file, checked by the opener)."""

    _, files = pinned_inputs.manifest_files(manifest_path)
    return str(next(iter(files.values())).get("vintage") or "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Chronicle's manifest.yaml for centax-childcare-cliff-edge-2026",
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help="a local copy of the registered workbook (checked against the manifest)",
    )
    parser.add_argument("--baseline-diagnostics", type=Path, required=True)
    parser.add_argument("--calibration-year", type=int, default=2025)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--built-on",
        required=True,
        help="build date recorded in the resource (YYYY-MM-DD), so a rerun reproduces it",
    )
    args = parser.parse_args(argv)
    baseline = json.loads(args.baseline_diagnostics.read_text(encoding="utf-8"))
    resource = build_resource(
        manifest_path=args.manifest,
        workbook_path=args.workbook,
        baseline=baseline,
        calibration_year=args.calibration_year,
        uprating=centax_ani_uprating(args.calibration_year),
        built_on=args.built_on,
    )
    rendered = json.dumps(resource, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    args.out.write_text(rendered, encoding="utf-8")
    facts = load_uk_centax_childcare_parent_facts(args.out)
    print(f"wrote {args.out} ({len(facts.rows)} rows, {facts.rows_sha256[:12]})")
    for cell in centax_cell_ids():
        print(
            f"{cell:38s} share {facts.shares[cell]:.5f} value {facts.value(cell):>11,.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
