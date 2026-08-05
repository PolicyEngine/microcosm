"""Ledger consumer-artifact emission for US import-entry margins.

Populace's ledger leg consumes sha-pinned consumer artifacts
(``manifest.json`` + ``consumer_facts.jsonl``,
:mod:`populace.build.ledger_artifact`) whose rows follow the
``ledger.consumer_fact.v1`` shape that :mod:`populace.build.ledger_targets`
compiles into calibration targets. This module mints the import-entry
margin series to that exact contract from the Census ingest and archived
CBP statistics.

Producer identity is explicit: these rows are **populace-minted** from
official source bytes — not an export of a PolicyEngine/ledger build — and
every row's ``source.extraction_method`` and the artifact manifest's
``generator`` block say so. Fact keys live in a populace namespace
(``populace_us_trade.*``) so they can never collide with or impersonate
ledger-built keys. The declarative ledger source-package harness
hand-enumerates selector rows per fact and cannot express a
10^5–10^6-cell API series; if that harness later grows a bulk contract,
these facts relocate by reproducing the same rows there.

Grain design: the JSONL feed carries margins at national, chapter, country,
and chapter × country grains (the P3 calibration axes); the full
HTS10 × country × month grain rides the parquet margins table emitted by
the ingest CLI, which is part of the same admitted artifact set. All grains
are exact sums of the same detail atoms, and the emitted rows are validated
against the parquet sums in the test battery.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from populace.build.us_runtime.us_trade.cbp_entry_stats import (
    CBP_TRADE_STATS_URL,
    CbpEntryStats,
)
from populace.build.us_runtime.us_trade.census_imports import CENSUS_IMPORTS_HS_ENDPOINT
from populace.build.us_runtime.us_trade.imdb_bulk import IMDB_URL_TEMPLATE

__all__ = [
    "CONSUMER_ARTIFACT_SCHEMA_VERSION",
    "IMDB_BULK_SOURCE_LEG",
    "IMPORT_ENTRY_FACT_GRAINS",
    "MEASURE_CATALOG",
    "FactSourceLeg",
    "build_cbp_entry_fact_rows",
    "build_district_entry_fact_rows",
    "build_import_entry_fact_rows",
    "default_generator_block",
    "write_consumer_artifact",
]


@dataclass(frozen=True)
class FactSourceLeg:
    """Retrieval-channel identity stamped into every emitted fact's source.

    The statistical series is the same official Census monthly import
    publication either way; the leg records *how the bytes were obtained*
    (monthly bulk IMDB archives vs the International Trade API), so fact
    identity (``record_set_id``, fact keys) never varies by channel while
    provenance stays honest.
    """

    source_name: str
    source_table: str
    url: str
    extraction_method: str


#: The primary retrieval channel: the monthly bulk database archives.
IMDB_BULK_SOURCE_LEG = FactSourceLeg(
    source_name="census_intltrade",
    source_table=(
        "US Imports of Merchandise monthly database (IMDB), IMP_DETL fixed-width detail"
    ),
    url=IMDB_URL_TEMPLATE,
    extraction_method=(
        "populace us_trade imdb_bulk ingest: Census monthly bulk IMDB "
        "archives parsed per the archives' own record layouts, detail "
        "summed to the {grain} grain and reconciled exactly against the "
        "publisher's in-archive control totals (IMP_CTY/IMP_COMM/IMP_DE); "
        "populace-minted to the ledger consumer contract (not a "
        "PolicyEngine/ledger build)."
    ),
)

#: The cross-check channel (kept for provenance parity with lane-G pulls).
CENSUS_API_SOURCE_LEG = FactSourceLeg(
    source_name="census_intltrade",
    source_table=(
        "US International Trade monthly imports, HS10 by country "
        "(timeseries/intltrade/imports/hs)"
    ),
    url=CENSUS_IMPORTS_HS_ENDPOINT,
    extraction_method=(
        "populace us_trade census_imports ingest: Census International "
        "Trade API monthly HS10 chapter pulls, DET country detail summed "
        "to the {grain} grain and reconciled exactly against the "
        "publisher's '-' totals; populace-minted to the ledger consumer "
        "contract (not a PolicyEngine/ledger build)."
    ),
)

CONSUMER_ARTIFACT_SCHEMA_VERSION = "policyengine_ledger.consumer_artifact.v1"
_FACT_SCHEMA_VERSION = "ledger.consumer_fact.v1"
_KEY_NAMESPACE = "populace_us_trade"

#: Feed grains. ``chapter_country`` is the P3 dashboard axis
#: (populace#615) and carries the duty-relevant measures only to bound the
#: feed size; coarser grains carry all four dollar measures.
IMPORT_ENTRY_FACT_GRAINS = ("national", "chapter", "country", "chapter_country")

_GRAIN_MEASURES = {
    "national": ("con_val_mo", "gen_val_mo", "cal_dut_mo", "dut_val_mo"),
    "chapter": ("con_val_mo", "gen_val_mo", "cal_dut_mo", "dut_val_mo"),
    "country": ("con_val_mo", "gen_val_mo", "cal_dut_mo", "dut_val_mo"),
    "chapter_country": ("con_val_mo", "cal_dut_mo"),
}

_COMPOSITION_MODULE = "us:policies/cbp/us-tariff-duty/composition"
_CUSTOMS_VALUE_DEFINITION_URL = (
    "https://www.census.gov/foreign-trade/reference/definitions/index.html"
)

#: Measure metadata, from the dataset's published variable catalog
#: (``variables.json``) and the Census import value definitions. Concept
#: alignment: imports-for-consumption customs value sums the entry-level
#:  ``customs_value`` input of the composed tariff spine exactly (customs
#: value = price paid excluding duties, freight, and insurance — the duty
#: base); calculated duty and dutiable value are the validation leg, related
#: to — never gating — engine-computed duty over the entries.
MEASURE_CATALOG: Mapping[str, Mapping[str, str]] = {
    "con_val_mo": {
        "label": "Imports for Consumption, Total Value",
        "source_measure_id": "CON_VAL_MO",
        "unit": "usd",
        "domain": "imports_for_consumption",
        "source_concept": "census_intltrade.imports_for_consumption_customs_value",
        "canonical_concept": f"{_COMPOSITION_MODULE}#input.customs_value",
        "concept_relation": "exact",
        "concept_evidence": (
            "Census import values are reported on a customs-value basis: "
            "'the price actually paid or payable for merchandise when sold "
            "for exportation to the United States, excluding U.S. import "
            "duties, freight, insurance, and other charges' — the duty base "
            "the composed tariff spine computes on. Imports for consumption "
            "measures merchandise cleared through Customs for consumption, "
            "i.e. the duty-paying entry universe."
        ),
    },
    "gen_val_mo": {
        "label": "General Imports, Total Value",
        "source_measure_id": "GEN_VAL_MO",
        "unit": "usd",
        "domain": "general_imports",
        "source_concept": "census_intltrade.general_imports_customs_value",
        "canonical_concept": "",
        "concept_relation": "",
        "concept_evidence": (
            "General imports measure total physical arrivals including "
            "bonded-warehouse and FTZ entries not yet cleared for "
            "consumption; context measure only, no engine concept."
        ),
    },
    "cal_dut_mo": {
        "label": "Imports for Consumption, Calculated Duty",
        "source_measure_id": "CAL_DUT_MO",
        "unit": "usd",
        "domain": "imports_for_consumption",
        "source_concept": "census_intltrade.imports_for_consumption_calculated_duty",
        "canonical_concept": f"{_COMPOSITION_MODULE}#us_tariff_total_ad_valorem_rate",
        "concept_relation": "related",
        "concept_evidence": (
            "Published calculated duty on consumption entries; the reality "
            "leg for engine-computed duty (rate × customs value) over the "
            "synthetic entries. Collected differs from statutory through "
            "exclusions, drawback, and compliance — divergence is signal, "
            "not error, and this series never gates (populace#615)."
        ),
    },
    "dut_val_mo": {
        "label": "Imports for Consumption, Dutiable Value",
        "source_measure_id": "DUT_VAL_MO",
        "unit": "usd",
        "domain": "imports_for_consumption",
        "source_concept": "census_intltrade.imports_for_consumption_dutiable_value",
        "canonical_concept": "",
        "concept_relation": "",
        "concept_evidence": (
            "Dutiable value is the customs value of the dutiable subset of "
            "consumption entries; validation context for duty coverage, no "
            "single engine concept."
        ),
    },
}

_CBP_MEASURE_CONCEPTS = {
    "total_entry_summaries": "cbp.total_entry_summaries",
    "informal_entry_summaries": "cbp.informal_entry_summaries",
    "total_import_value": "cbp.total_import_value",
    "duty_taxes_fees_collected": "cbp.duty_taxes_fees_collected",
}


def build_import_entry_fact_rows(
    margins: pd.DataFrame,
    *,
    retrieval_manifest: Iterable[Mapping[str, Any]],
    extracted_at: str,
    source_leg: FactSourceLeg = IMDB_BULK_SOURCE_LEG,
    grains: tuple[str, ...] = IMPORT_ENTRY_FACT_GRAINS,
) -> list[dict[str, Any]]:
    """Aggregate the tidy margins table into consumer fact rows.

    ``margins`` is the ingest's HTS10 × country × month table (one row per
    nonzero detail cell). Every emitted grain is an exact integer sum of
    those detail rows. ``source_leg`` names the retrieval channel; fact
    identity is channel-invariant (same record sets and keys from the bulk
    archives or the API), only the ``source`` provenance block varies.
    """
    unknown = sorted(set(grains) - set(IMPORT_ENTRY_FACT_GRAINS))
    if unknown:
        raise ValueError(f"Unknown import-entry fact grain(s): {unknown}.")
    if margins.empty:
        raise ValueError("Cannot emit import-entry facts from an empty margins table.")
    entries = [dict(entry) for entry in retrieval_manifest]
    manifest_by_month_chapter = _manifest_index(entries)
    month_identities = _month_file_identities(entries)
    rows: list[dict[str, Any]] = []
    for grain in grains:
        grouped = _aggregate(margins, grain)
        for record in grouped.itertuples(index=False):
            for measure in _GRAIN_MEASURES[grain]:
                rows.append(
                    _fact_row(
                        grain=grain,
                        record=record,
                        measure=measure,
                        value=int(getattr(record, measure)),
                        extracted_at=extracted_at,
                        source_leg=source_leg,
                        manifest_by_month_chapter=manifest_by_month_chapter,
                        month_identities=month_identities,
                    )
                )
    rows.sort(key=lambda row: row["lineage"]["source_record_id"])
    return rows


def build_district_entry_fact_rows(
    district_entry: pd.DataFrame,
    *,
    retrieval_manifest: Iterable[Mapping[str, Any]],
    extracted_at: str,
    source_leg: FactSourceLeg = IMDB_BULK_SOURCE_LEG,
) -> list[dict[str, Any]]:
    """Mint district-of-entry margin facts from the publisher's own table.

    ``district_entry`` is the per-month district control table (already
    reconciled exactly against the detail by the ingest); the emitted
    measures are the duty-relevant pair, mirroring the chapter × country
    feed grain. District facts carry their own record-set family
    (``census_intltrade.imports_district_entry``).
    """
    if district_entry.empty:
        raise ValueError(
            "Cannot emit district-entry facts from an empty district table."
        )
    entries = [dict(entry) for entry in retrieval_manifest]
    month_identities = _month_file_identities(entries)
    rows: list[dict[str, Any]] = []
    for record in district_entry.itertuples(index=False):
        month = str(record.period)
        if month not in month_identities:
            raise ValueError(
                f"No retrieval-manifest entry covers month {month}; refusing "
                "to mint district facts without raw-source provenance."
            )
        source_file, source_sha256, sha_list = month_identities[month]
        for measure in ("con_val_mo", "cal_dut_mo"):
            catalog = MEASURE_CATALOG[measure]
            record_set_id = (
                "census_intltrade.imports_district_entry."
                f"month_{month.replace('-', '_')}"
            )
            value_id = f"de{record.dist_entry}"
            source_record_id = f"{record_set_id}.{value_id}.{measure}"
            row: dict[str, Any] = {
                "schema_version": _FACT_SCHEMA_VERSION,
                "assertion": "observation",
                "aggregation": {"method": "sum"},
                "value": int(getattr(record, measure)),
                "value_type": "integer",
                "period": {"type": "month", "value": month},
                "geography": {
                    "id": "0100000US",
                    "level": "country",
                    "name": "United States",
                },
                "entity": {"name": "import_entry", "role": "customs_entry"},
                "dimensions": {"district_of_entry": str(record.dist_entry)},
                "universe_constraints": {"domain": catalog["domain"]},
                "provenance_class": "administrative",
                "observed_measure": {
                    "source_name": source_leg.source_name,
                    "source_table": source_leg.source_table,
                    "source_measure_id": catalog["source_measure_id"],
                    "source_concept": catalog["source_concept"],
                    "unit": catalog["unit"],
                },
                "layout": {
                    "record_set_id": record_set_id,
                    "groupby_dimension": "district_of_entry",
                    "groupby_value_id": value_id,
                    "measure_id": measure,
                    "measure_label": catalog["label"],
                },
                "lineage": {
                    "source_record_id": source_record_id,
                    "source_file_sha256s": sha_list,
                },
                "source": {
                    "source_name": source_leg.source_name,
                    "source_table": source_leg.source_table,
                    "source_file": source_file,
                    "source_sha256": source_sha256,
                    "url": source_leg.url,
                    "vintage": f"monthly_revision_as_retrieved_{extracted_at[:10]}",
                    "extracted_at": extracted_at,
                    "extraction_method": source_leg.extraction_method.format(
                        grain="district_entry"
                    ),
                },
                "label": (
                    f"United States {month} {catalog['label']} "
                    f"(district_of_entry={value_id} {record.dist_name}) "
                    f"[{source_leg.source_name}]"
                ),
            }
            if catalog["canonical_concept"]:
                row["concept_alignment"] = {
                    "authority": "populace-us-trade",
                    "canonical_concept": catalog["canonical_concept"],
                    "relation": catalog["concept_relation"],
                    "evidence_notes": catalog["concept_evidence"],
                    "evidence_url": _CUSTOMS_VALUE_DEFINITION_URL,
                    "legal_vintage": month,
                    "source_concept": catalog["source_concept"],
                }
            _stamp_keys(row)
            rows.append(row)
    rows.sort(key=lambda row: row["lineage"]["source_record_id"])
    return rows


def build_cbp_entry_fact_rows(
    stats: CbpEntryStats,
    *,
    page_sha256: str,
    retrieved_at: str,
) -> list[dict[str, Any]]:
    """Mint the CBP fiscal-year entry anchors (exact published cells only)."""
    rows: list[dict[str, Any]] = []
    for cell in stats.exact_cells():
        record_set_id = (
            f"cbp_trade_stats.imports_revenue_collection.fy{cell.fiscal_year}"
        )
        source_record_id = f"{record_set_id}.{cell.measure_id}"
        # The page's "as of" note is provenance, never identity: it changes
        # with every CBP refresh and must not rotate fact keys or become a
        # selector-demanded dimension filter.
        source = {
            "source_name": "cbp_trade_stats",
            "source_table": "CBP Trade Statistics: Imports and Revenue Collection",
            "source_file": "newsroom-stats-trade.html",
            "source_sha256": page_sha256,
            "url": CBP_TRADE_STATS_URL,
            "vintage": f"fiscal_year_{cell.fiscal_year}",
            "extracted_at": retrieved_at,
            "extraction_method": (
                "populace us_trade cbp_entry_stats ingest: archived public "
                "statistics page parsed from recorded bytes; populace-minted "
                "to the ledger consumer contract (not a PolicyEngine/ledger "
                "build)."
            ),
        }
        if stats.as_of_note:
            source["publisher_as_of_note"] = stats.as_of_note
        row = {
            "schema_version": _FACT_SCHEMA_VERSION,
            "assertion": "observation",
            "aggregation": {"method": "sum"},
            "value": cell.exact_value,
            "value_type": "integer",
            "period": {"type": "fiscal_year", "value": cell.fiscal_year},
            "geography": {
                "id": "0100000US",
                "level": "country",
                "name": "United States",
            },
            "entity": {"name": "import_entry", "role": "customs_entry"},
            "dimensions": {},
            "universe_constraints": {"domain": "all_import_entry_summaries"},
            "provenance_class": "administrative",
            "observed_measure": {
                "source_name": "cbp_trade_stats",
                "source_table": (
                    "CBP Trade Statistics: Imports and Revenue Collection"
                ),
                "source_measure_id": cell.measure_id,
                "source_concept": _CBP_MEASURE_CONCEPTS[cell.measure_id],
                "unit": cell.unit,
            },
            "layout": {
                "record_set_id": record_set_id,
                "groupby_dimension": "all",
                "groupby_value_id": "all",
                "measure_id": cell.measure_id,
            },
            "lineage": {"source_record_id": source_record_id},
            "source": source,
            "label": (
                f"United States FY{cell.fiscal_year} {cell.measure_id} "
                f"[cbp_trade_stats {cell.text}]"
            ),
        }
        _stamp_keys(row)
        rows.append(row)
    rows.sort(key=lambda row: row["lineage"]["source_record_id"])
    return rows


def write_consumer_artifact(
    out_dir: str | Path,
    fact_rows: list[dict[str, Any]],
    *,
    retrieval_manifest: Iterable[Mapping[str, Any]],
    generator: Mapping[str, Any],
) -> dict[str, Any]:
    """Write ``consumer_facts.jsonl`` + ``manifest.json`` and return the manifest.

    The directory loads through
    :func:`populace.build.ledger_artifact.load_ledger_consumer_artifact`,
    which re-hashes the fact file against ``manifest.facts_sha256``.
    """
    if not fact_rows:
        raise ValueError("Refusing to write an empty consumer artifact.")
    retrieval_entries = [dict(entry) for entry in retrieval_manifest]
    if not retrieval_entries:
        raise ValueError(
            "Refusing to write a consumer artifact with an empty retrieval "
            "manifest; every fact feed must carry its raw-source retrievals."
        )
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    facts_path = out_path / "consumer_facts.jsonl"
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in fact_rows
    ).encode("utf-8")
    facts_path.write_bytes(payload)
    retrievals = retrieval_entries
    manifest = {
        "schema_version": CONSUMER_ARTIFACT_SCHEMA_VERSION,
        "facts_sha256": hashlib.sha256(payload).hexdigest(),
        "fact_row_count": len(fact_rows),
        "profiles": {},
        "generator": dict(generator),
        "source_manifest": {
            "retrievals": retrievals,
            "set_digest": _retrieval_set_digest(retrievals),
        },
    }
    (out_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _aggregate(margins: pd.DataFrame, grain: str) -> pd.DataFrame:
    measures = list(_GRAIN_MEASURES[grain])
    if grain == "national":
        grouped = margins.groupby("period", as_index=False)[measures].sum()
    elif grain == "chapter":
        grouped = margins.groupby(["period", "chapter"], as_index=False)[measures].sum()
    elif grain == "country":
        grouped = margins.groupby(["period", "cty_code", "iso2"], as_index=False)[
            measures
        ].sum()
    else:
        grouped = margins.groupby(
            ["period", "chapter", "cty_code", "iso2"], as_index=False
        )[measures].sum()
    return grouped.sort_values(
        [column for column in grouped.columns if column not in measures],
        ignore_index=True,
    )


def _fact_row(
    *,
    grain: str,
    record: Any,
    measure: str,
    value: int,
    extracted_at: str,
    source_leg: FactSourceLeg,
    manifest_by_month_chapter: Mapping[tuple[str, str], list[Mapping[str, Any]]],
    month_identities: Mapping[str, tuple[str, str, list[str]]],
) -> dict[str, Any]:
    month = str(record.period)
    catalog = MEASURE_CATALOG[measure]
    dimensions: dict[str, str] = {}
    if grain in ("chapter", "chapter_country"):
        dimensions["hts_chapter"] = str(record.chapter)
    if grain in ("country", "chapter_country"):
        dimensions["country_of_origin"] = str(record.iso2)
        dimensions["census_country_code"] = str(record.cty_code)
    value_id = _value_id(grain, record)
    record_set_id = (
        f"census_intltrade.imports_hs10.month_{month.replace('-', '_')}.{grain}"
    )
    source_record_id = f"{record_set_id}.{value_id}.{measure}"
    lineage: dict[str, Any] = {"source_record_id": source_record_id}
    if month not in month_identities:
        raise ValueError(
            f"No retrieval-manifest entry covers month {month}; refusing to "
            "mint facts without raw-source provenance."
        )
    month_file, month_sha, month_shas = month_identities[month]
    source_sha256 = month_sha
    source_file = month_file
    if grain in ("chapter", "chapter_country"):
        # An API chapter served in one response has one file; a chapter the
        # API split into sub-prefixes has several, and the fact's source
        # hash is then the sorted-hash set digest over all of them. Bulk
        # months have no per-chapter files, so chapter facts carry the
        # month archive's identity.
        chapter_entries = manifest_by_month_chapter.get(
            (month, str(record.chapter)), ()
        )
        shas = sorted(str(entry["sha256"]) for entry in chapter_entries)
        files = [str(entry.get("filename") or "") for entry in chapter_entries]
        if len(shas) == 1:
            source_sha256 = shas[0]
            source_file = files[0]
        elif shas:
            source_sha256 = hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()
            source_file = f"{len(shas)} prefix files (set digest)"
        else:
            shas = month_shas
        lineage["source_file_sha256s"] = shas
    else:
        lineage["source_file_sha256s"] = month_shas
    source = {
        "source_name": source_leg.source_name,
        "source_table": source_leg.source_table,
        "source_file": source_file,
        "source_sha256": source_sha256,
        "url": source_leg.url,
        "vintage": f"monthly_revision_as_retrieved_{extracted_at[:10]}",
        "extracted_at": extracted_at,
        "extraction_method": source_leg.extraction_method.format(grain=grain),
    }
    row: dict[str, Any] = {
        "schema_version": _FACT_SCHEMA_VERSION,
        "assertion": "observation",
        "aggregation": {"method": "sum"},
        "value": value,
        "value_type": "integer",
        "period": {"type": "month", "value": month},
        "geography": {
            "id": "0100000US",
            "level": "country",
            "name": "United States",
        },
        "entity": {"name": "import_entry", "role": "customs_entry"},
        "dimensions": dimensions,
        "universe_constraints": {"domain": catalog["domain"]},
        "provenance_class": "administrative",
        "observed_measure": {
            "source_name": "census_intltrade",
            "source_table": ("US International Trade monthly imports, HS10 by country"),
            "source_measure_id": catalog["source_measure_id"],
            "source_concept": catalog["source_concept"],
            "unit": catalog["unit"],
        },
        "layout": {
            "record_set_id": record_set_id,
            "groupby_dimension": _GROUPBY_DIMENSION[grain],
            "groupby_value_id": value_id,
            "measure_id": measure,
            "measure_label": catalog["label"],
        },
        "lineage": lineage,
        "source": source,
        "label": (
            f"United States {month} {catalog['label']} "
            f"({_GROUPBY_DIMENSION[grain]}={value_id}) [census_intltrade]"
        ),
    }
    if catalog["canonical_concept"]:
        row["concept_alignment"] = {
            "authority": "populace-us-trade",
            "canonical_concept": catalog["canonical_concept"],
            "relation": catalog["concept_relation"],
            "evidence_notes": catalog["concept_evidence"],
            "evidence_url": _CUSTOMS_VALUE_DEFINITION_URL,
            "legal_vintage": month,
            "source_concept": catalog["source_concept"],
        }
    _stamp_keys(row)
    return row


_GROUPBY_DIMENSION = {
    "national": "all",
    "chapter": "hts_chapter",
    "country": "country_of_origin",
    "chapter_country": "hts_chapter_x_country_of_origin",
}


def _value_id(grain: str, record: Any) -> str:
    if grain == "national":
        return "all"
    if grain == "chapter":
        return f"ch{record.chapter}"
    if grain == "country":
        return str(record.cty_code)
    return f"ch{record.chapter}.{record.cty_code}"


def _stamp_keys(row: dict[str, Any]) -> None:
    """Deterministic populace-namespace fact keys from row identity.

    The aggregate key covers the full identity including period; the
    semantic key drops the period so ledger-target selectors can resolve
    "the same series, latest eligible period" across feed versions.
    """
    identity = {
        "record_set_id": row["layout"]["record_set_id"],
        "groupby_value_id": row["layout"]["groupby_value_id"],
        "measure_id": row["layout"]["measure_id"],
        "entity": row["entity"]["name"],
        "geography": row["geography"]["id"],
        "domain": row["universe_constraints"]["domain"],
        "dimensions": dict(row["dimensions"]),
        "period": str(row["period"]["value"]),
    }
    semantic_identity = {
        key: value
        for key, value in identity.items()
        if key not in ("period", "record_set_id")
    }
    semantic_identity["record_set_family"] = _period_invariant_record_set(
        row["layout"]["record_set_id"]
    )
    row["aggregate_fact_key"] = _key("aggregate_fact", identity)
    row["semantic_fact_key"] = _key("semantic_fact", semantic_identity)


def _period_invariant_record_set(record_set_id: str) -> str:
    """Drop period-carrying segments (monthly and fiscal-year alike)."""
    return ".".join(
        part
        for part in record_set_id.split(".")
        if not (part.startswith("month_") or re.fullmatch(r"fy\d{4}", part))
    )


def _key(kind: str, identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{_KEY_NAMESPACE}.{kind}.v1:{digest[:24]}"


def _manifest_index(
    retrieval_manifest: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    """All sha-bearing retrievals per (month, chapter), split files included."""
    index: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for entry in retrieval_manifest:
        month = str(entry.get("month") or "")
        chapter = str(entry.get("chapter") or "")
        if month and chapter and entry.get("sha256"):
            index.setdefault((month, chapter), []).append(entry)
    return index


def _month_file_identities(
    retrieval_manifest: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[str, str, list[str]]]:
    """Per-month source-file identity: ``(file, sha256, sha_list)``.

    A month retrieved as one archive (the bulk leg) is identified by that
    file and its hash directly. A month assembled from many responses (the
    API leg) is identified by the sorted-hash set digest; the per-row sha
    list is left empty there to keep the feed lean (chapter-grain rows
    carry their own chapter-level sha lists).
    """
    by_month: dict[str, list[tuple[str, str]]] = {}
    for entry in retrieval_manifest:
        month = str(entry.get("month") or "")
        sha = entry.get("sha256")
        if month and sha:
            by_month.setdefault(month, []).append(
                (str(entry.get("filename") or ""), str(sha))
            )
    identities: dict[str, tuple[str, str, list[str]]] = {}
    for month, pairs in by_month.items():
        if len(pairs) == 1:
            filename, sha = pairs[0]
            identities[month] = (filename, sha, [sha])
        else:
            shas = sorted(sha for _, sha in pairs)
            digest = hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()
            identities[month] = (
                f"{len(shas)} response files (set digest)",
                digest,
                [],
            )
    return identities


def _retrieval_set_digest(retrievals: list[dict[str, Any]]) -> str:
    shas = sorted(
        str(entry.get("sha256")) for entry in retrievals if entry.get("sha256")
    )
    return hashlib.sha256("\n".join(shas).encode("utf-8")).hexdigest()


def default_generator_block(
    *, months: tuple[str, ...], now: datetime | None = None
) -> dict[str, Any]:
    """Manifest ``generator`` block naming the populace producer."""
    moment = now or datetime.now(UTC)
    return {
        "producer": "populace.build.us_runtime.us_trade",
        "issue": "PolicyEngine/populace#615",
        "created_at": moment.isoformat(timespec="seconds"),
        "months": list(months),
        "note": (
            "Populace-minted ledger-contract feed for the import-entry unit "
            "family; official Census/CBP source bytes with recorded "
            "retrieval provenance. Not a PolicyEngine/ledger build export."
        ),
    }
