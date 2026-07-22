"""Census of the UK local calibration target surface (populace#495).

The UK dense/local arm calibrates cloned rowwise households against
constituency and local-authority targets. Before any family binds, three
questions must have executable answers: which household-side metrics the
runtime computes today (``local_targets.metric_names``), which official UK
local statistics could supply target values for them, and which reviewed
fences constrain binding. This module answers all three as one committed,
drift-gated JSON artifact.

Design rules:

* **Derived, never restated.** The metric inventory comes from
  ``metric_names()`` at build time, so the census cannot drift from code. A
  metric the classifier does not recognize fails the build closed rather than
  shipping unclassified.
* **Reviewed pointers, not scraped claims.** Source rows document official
  products verified by a human-reviewed fetch on ``verified_on``; they are
  ``documented_unpinned`` until a binding increment pins exact tables with
  hashes, mirroring the HMRC/SPI source-contract discipline.
* **Fences by reference.** The 208 banded HMRC facts stay fenced exactly as
  the national replay adjudicated them (``FULL_FRS_TI_BAND_FENCE_ID``,
  ``HMRC_SPI_TARGET_RECORD_COUNT`` are imported, never copied), and the
  census adds the two local-binding adjudications the in-code metrics imply.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from populace.build.uk_runtime.hmrc_income import HMRC_SPI_TARGET_RECORD_COUNT
from populace.build.uk_runtime.hmrc_replay import FULL_FRS_TI_BAND_FENCE_ID
from populace.build.uk_runtime.local_targets import AREA_TYPES, metric_names

__all__ = [
    "CENSUS_KIND",
    "CENSUS_RESOURCE",
    "CENSUS_SCHEMA_VERSION",
    "METRIC_STATUS_BOUND_IN_CODE",
    "SOURCE_STATUS_DOCUMENTED_UNPINNED",
    "assert_uk_local_target_census_current",
    "build_uk_local_target_census",
    "committed_uk_local_target_census_path",
    "load_uk_local_target_census",
    "write_uk_local_target_census",
]

CENSUS_SCHEMA_VERSION = 1
CENSUS_KIND = "uk_local_target_census"
CENSUS_RESOURCE = "uk_local_target_census.json"

METRIC_STATUS_BOUND_IN_CODE = "bound_in_code"
SOURCE_STATUS_DOCUMENTED_UNPINNED = "documented_unpinned"

#: Review date for every source row below: each URL was fetched and its
#: product description checked on this date (populace#495 scoping).
_SOURCES_VERIFIED_ON = "2026-07-22"

_SPI_FRAME_PROXY_FENCE_ID = "hmrc_spi_frame_model_proxy"
_FRS_CIRCULARITY_FENCE_ID = "frs_model_based_target_circularity"

#: Ordered metric-name prefix (or exact-name) rules mapping each in-code
#: metric to a census family. Fail-closed: an unmatched metric raises.
_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("hmrc/", "hmrc_income_by_area"),
    ("age/", "age_structure"),
    ("uc_households", "uc_households"),
    ("uc_hh_", "uc_households"),
    ("ons/equiv_", "equivalised_income"),
    ("tenure/", "tenure"),
    ("rent/private_rent", "private_rent"),
)

_FAMILIES: tuple[dict[str, Any], ...] = (
    {
        "family": "hmrc_income_by_area",
        "description": (
            "SPI-frame employment and self-employment income amounts and "
            "taxpayer counts by area."
        ),
        "sources": ["hmrc_personal_incomes_constituency_la"],
        "adjudications": [
            _SPI_FRAME_PROXY_FENCE_ID,
            FULL_FRS_TI_BAND_FENCE_ID,
        ],
    },
    {
        "family": "age_structure",
        "description": "Household members by ten-year age band.",
        "sources": [
            "ons_pcon_population_by_age",
            "nomis_lad_population_single_year_age",
        ],
        "adjudications": [],
    },
    {
        "family": "uc_households",
        "description": (
            "Households on Universal Credit, and (constituency only, in code "
            "today) the split by number of children."
        ),
        "sources": ["dwp_stat_xplore_uc"],
        "adjudications": [],
    },
    {
        "family": "equivalised_income",
        "description": (
            "Equivalised household disposable income before/after housing "
            "costs and implied housing costs (local-authority metrics)."
        ),
        "sources": ["ons_small_area_income_msoa"],
        "adjudications": [_FRS_CIRCULARITY_FENCE_ID],
    },
    {
        "family": "tenure",
        "description": "Households by tenure class (local-authority metrics).",
        "sources": ["census2021_ts054_tenure"],
        "adjudications": [],
    },
    {
        "family": "private_rent",
        "description": (
            "Private rent paid by privately renting households "
            "(local-authority metric)."
        ),
        "sources": ["ons_pipr_private_rents"],
        "adjudications": [],
    },
)

_SOURCES: tuple[dict[str, Any], ...] = (
    {
        "source_id": "hmrc_personal_incomes_constituency_la",
        "publisher": "HM Revenue & Customs",
        "product": (
            "Personal incomes statistics, tables 3.12 to 3.15a: income and "
            "tax by county and region, by borough/district/unitary "
            "authority, and by Parliamentary constituency (Survey of "
            "Personal Incomes)."
        ),
        "url": "https://www.gov.uk/government/collections/personal-incomes-statistics",
        "geographies": ["constituency", "la"],
        "latest_vintage": "tax year 2023 to 2024 (published 2026-04-29)",
        "status": SOURCE_STATUS_DOCUMENTED_UNPINNED,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "Unbanded area totals and taxpayer counts — a different surface "
            "from the fenced 208 banded Total Income facts. The constituency "
            "vintage used by each table edition must be pinned at binding "
            "time (2024 boundaries vs earlier)."
        ),
    },
    {
        "source_id": "ons_pcon_population_by_age",
        "publisher": "Office for National Statistics",
        "product": (
            "Parliamentary constituency mid-year population estimates "
            "(England and Wales), by year of age."
        ),
        "url": (
            "https://www.ons.gov.uk/peoplepopulationandcommunity/"
            "populationandmigration/populationestimates/datasets/"
            "parliamentaryconstituencymidyearpopulationestimates"
        ),
        "geographies": ["constituency"],
        "latest_vintage": (
            "mid-2022 on this page; estimates for 2012 onwards now released "
            "via Nomis (ONS notice dated 2024-11-25)"
        ),
        "status": SOURCE_STATUS_DOCUMENTED_UNPINNED,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "Binding must pin the Nomis release carrying 2024-constituency "
            "(PCON24) estimates to match the geography ladder's constituency "
            "vintage, plus the Scotland/Northern Ireland equivalents."
        ),
    },
    {
        "source_id": "nomis_lad_population_single_year_age",
        "publisher": "Office for National Statistics (via Nomis)",
        "product": (
            "Mid-year population estimates by single year of age, local "
            "authority and above (UK), rebased to Census 2021/2022."
        ),
        "url": "https://www.nomisweb.co.uk/datasets/pestsyoala",
        "geographies": ["la"],
        "latest_vintage": "mid-2024",
        "status": SOURCE_STATUS_DOCUMENTED_UNPINNED,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": "UK-wide coverage at local-authority grain.",
    },
    {
        "source_id": "dwp_stat_xplore_uc",
        "publisher": "Department for Work and Pensions",
        "product": (
            "Stat-Xplore Universal Credit statistics: households and people "
            "on UC, tabulated by Westminster parliamentary constituency or "
            "local authority, with family-type/child breakdowns built as "
            "custom tabulations."
        ),
        "url": "https://stat-xplore.dwp.gov.uk/webapi/jsf/dataCatalogueExplorer.xhtml",
        "geographies": ["constituency", "la"],
        "latest_vintage": (
            "UC statistics release cadence per DWP schedule (next noted "
            "release 2026-08-18 at verification)"
        ),
        "status": SOURCE_STATUS_DOCUMENTED_UNPINNED,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "Exact table definitions (households vs people, child-count "
            "bands, constituency vintage) are chosen inside the tool and "
            "must be pinned per tabulation at binding time."
        ),
    },
    {
        "source_id": "ons_small_area_income_msoa",
        "publisher": "Office for National Statistics",
        "product": (
            "Income estimates for small areas, England and Wales: mean "
            "household income at MSOA grain for four measures, including "
            "net equivalised disposable income before and after housing "
            "costs."
        ),
        "url": (
            "https://www.ons.gov.uk/peoplepopulationandcommunity/"
            "personalandhouseholdfinances/incomeandwealth/bulletins/"
            "smallareamodelbasedincomeestimates/financialyearending2023"
        ),
        "geographies": ["msoa"],
        "latest_vintage": "financial year ending 2023",
        "status": SOURCE_STATUS_DOCUMENTED_UNPINNED,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "Model-based estimates built from the Family Resources Survey — "
            "see the frs_model_based_target_circularity adjudication. "
            "MSOA-to-local-authority aggregation method must be declared at "
            "binding time; England and Wales only."
        ),
    },
    {
        "source_id": "census2021_ts054_tenure",
        "publisher": "Office for National Statistics (via Nomis)",
        "product": (
            "Census 2021 table TS054 (tenure of household), England and "
            "Wales, OA best-fit to higher geographies."
        ),
        "url": "https://www.nomisweb.co.uk/datasets/c2021ts054",
        "geographies": ["constituency", "la"],
        "latest_vintage": "Census Day 2021-03-21",
        "status": SOURCE_STATUS_DOCUMENTED_UNPINNED,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "England and Wales only; Scotland's Census 2022 and Northern "
            "Ireland's Census 2021 tenure tables need separate pinning for "
            "full-UK coverage. Census tenure classes must be crosswalked to "
            "the runtime's four tenure metrics explicitly."
        ),
    },
    {
        "source_id": "ons_pipr_private_rents",
        "publisher": "Office for National Statistics",
        "product": (
            "Price Index of Private Rents (PIPR), UK: monthly price "
            "statistics — rent indices, annual change, and price levels."
        ),
        "url": (
            "https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/"
            "priceindexofprivaterentsukmonthlypricestatistics"
        ),
        "geographies": ["la"],
        "latest_vintage": "release of 2026-07-22 at verification",
        "status": SOURCE_STATUS_DOCUMENTED_UNPINNED,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "The dataset landing page was verified; the local-authority "
            "granularity of the price-level tables must be confirmed when "
            "the exact table is pinned."
        ),
    },
)

_BINDING_FENCES: tuple[dict[str, Any], ...] = (
    {
        "fence_id": FULL_FRS_TI_BAND_FENCE_ID,
        "fenced_fact_count": HMRC_SPI_TARGET_RECORD_COUNT,
        "rule": (
            "No local target family may bind facts banded by SPI Total "
            "Income: the FRS instrument cannot materialize complete Total "
            "Income, so band membership is unassignable. The national "
            "adjudication that holds all "
            f"{HMRC_SPI_TARGET_RECORD_COUNT} published banded facts as "
            "fenced exclusions carries to every local surface unchanged; "
            "minting banded local variants is forbidden."
        ),
        "authority": (
            "UK_COVERAGE_PROGRESS.md real-donor HMRC replay (2026-07-13); "
            "canonical fences in populace.build.uk_runtime.hmrc_replay."
        ),
    },
    {
        "fence_id": _SPI_FRAME_PROXY_FENCE_ID,
        "fenced_fact_count": None,
        "rule": (
            "The in-code HMRC area metrics gate frame membership on modeled "
            "income_tax > 0 — a model-output proxy for 'taxpayer in the SPI "
            "frame', not a source-faithful frame. Binding any HMRC area "
            "family requires an explicit adjudication accepting or replacing "
            "that proxy before targets enter a solve."
        ),
        "authority": (
            "populace.build.uk_runtime.local_targets.compute_household_metrics "
            "(in_spi_frame definition); populace#495 scoping."
        ),
    },
    {
        "fence_id": _FRS_CIRCULARITY_FENCE_ID,
        "fenced_fact_count": None,
        "rule": (
            "ONS income estimates for small areas are model-based outputs "
            "built from the Family Resources Survey. Calibrating FRS-derived "
            "microdata to them feeds the instrument back into itself; "
            "binding requires an explicit adjudication of that circularity "
            "(accept with documented rationale, or reject the family)."
        ),
        "authority": (
            "ONS income estimates for small areas QMI (FRS-based method); "
            "populace#495 scoping."
        ),
    },
)

_STATUS_DEFINITIONS: dict[str, str] = {
    METRIC_STATUS_BOUND_IN_CODE: (
        "The household-side metric is computed by "
        "local_targets.compute_household_metrics today."
    ),
    SOURCE_STATUS_DOCUMENTED_UNPINNED: (
        "The official product exists and was verified at the recorded URL "
        "on verified_on, but no exact table, vintage, or hash is pinned; "
        "binding work pins it per family."
    ),
}


def build_uk_local_target_census() -> dict[str, Any]:
    """Build the census from the live metric surface plus reviewed registers."""

    metric_rows: dict[str, dict[str, Any]] = {}
    for area_type in AREA_TYPES:
        for name in metric_names(area_type):
            row = metric_rows.setdefault(
                name,
                {
                    "name": name,
                    "family": _family_for_metric(name),
                    "area_types": [],
                    "status": METRIC_STATUS_BOUND_IN_CODE,
                },
            )
            if area_type not in row["area_types"]:
                row["area_types"].append(area_type)

    families = [dict(row) for row in _FAMILIES]
    used_families = {row["family"] for row in metric_rows.values()}
    declared_families = {row["family"] for row in families}
    unused = sorted(declared_families - used_families)
    if unused:
        raise ValueError(f"census declares family(ies) with no metrics: {unused}.")

    return {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "census_kind": CENSUS_KIND,
        "area_types": list(AREA_TYPES),
        "metrics": [metric_rows[name] for name in sorted(metric_rows)],
        "families": families,
        "sources": [dict(row) for row in _SOURCES],
        "binding_fences": [dict(row) for row in _BINDING_FENCES],
        "status_definitions": dict(_STATUS_DEFINITIONS),
    }


def committed_uk_local_target_census_path() -> Path:
    """Path of the committed census artifact inside ``populace.build.uk``."""

    return Path(str(files("populace.build.uk").joinpath(CENSUS_RESOURCE)))


def load_uk_local_target_census(path: str | Path | None = None) -> dict[str, Any]:
    """Load a census JSON, defaulting to the committed artifact."""

    census_path = (
        committed_uk_local_target_census_path() if path is None else Path(path)
    )
    payload = json.loads(census_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{census_path}: census must be a JSON object.")
    return payload


def write_uk_local_target_census(
    path: str | Path | None = None,
    census: dict[str, Any] | None = None,
) -> Path:
    """Write the census JSON, defaulting to the committed artifact path."""

    census_path = (
        committed_uk_local_target_census_path() if path is None else Path(path)
    )
    payload = build_uk_local_target_census() if census is None else census
    census_path.parent.mkdir(parents=True, exist_ok=True)
    census_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return census_path


def assert_uk_local_target_census_current(path: str | Path | None = None) -> None:
    """Fail if the committed census no longer matches the live surface."""

    committed = load_uk_local_target_census(path)
    live = build_uk_local_target_census()
    if committed != live:
        drifted = _drifted_keys(committed, live)
        raise ValueError(
            "UK local-target census is stale: committed artifact does not "
            f"match the live metric surface (drift in {drifted}). Regenerate "
            "with `uv run python tools/census_uk_local_targets.py`."
        )


def _family_for_metric(name: str) -> str:
    for prefix, family in _FAMILY_RULES:
        if name == prefix or name.startswith(prefix):
            return family
    raise ValueError(
        f"UK local metric {name!r} has no census family classification; add "
        "a rule to local_target_census._FAMILY_RULES with its official "
        "source(s) before shipping it."
    )


def _drifted_keys(committed: dict[str, Any], live: dict[str, Any]) -> list[str]:
    keys = sorted(set(committed) | set(live))
    return [key for key in keys if committed.get(key) != live.get(key)]
