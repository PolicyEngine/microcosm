"""Census of the UK local calibration target surface (microcosm#495).

The UK dense/local arm calibrates cloned rowwise households against
constituency and local-authority targets. Before any family binds, three
questions must have executable answers: which household-side metrics the
runtime computes today (``local_targets.metric_names``), which official UK
local statistics could supply target values for them, and which reviewed
fences constrain binding. This module answers all three as one committed,
drift-gated JSON artifact.

Design rules:

* **Derived, never restated.** The metric inventory comes from
  ``metric_names()`` at build time — including each area type's exact metric
  order, so a reorder or duplicate drifts the artifact. A metric the
  classifier does not recognize (exact names match exactly; prefixes must
  match a declared family prefix) fails the build closed rather than shipping
  unclassified.
* **Scoped honestly.** The census covers the default in-code metric surface.
  Ledger target-profile-driven surfaces are governed by their profile
  contract and are explicitly out of scope here.
* **Reviewed pointers, not scraped claims.** Source rows document official
  products verified by a human-reviewed fetch on ``verified_on``. They start
  ``documented_unpinned`` and move to a pinned status once a sha-pinned build
  artifact or Ledger consumer fact feed owns the facts, mirroring the
  HMRC/SPI source-contract discipline. Rows that remain unsuitable for this
  surface carry a signed deferral reason.
* **Fences by reference, enforced at binding.** The banded HMRC facts stay
  fenced exactly as the national replay adjudicated them
  (``FULL_FRS_TI_BAND_FENCE_ID``, ``HMRC_SPI_TARGET_RECORD_COUNT`` are
  imported, never copied), and every census fence declares its enforcement
  status: the rowwise doctrine solve
  (``local_rowwise.require_adjudicated_uk_local_binding``) refuses binding a
  family whose fences lack an in-force entry in
  ``local_binding_adjudications.json``.
"""

from __future__ import annotations

import copy
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.hmrc_income import HMRC_SPI_TARGET_RECORD_COUNT
from microcosm.build.uk_runtime.hmrc_replay import FULL_FRS_TI_BAND_FENCE_ID
from microcosm.build.uk_runtime.local_targets import AREA_TYPES, metric_names

__all__ = [
    "CENSUS_KIND",
    "CENSUS_RESOURCE",
    "CENSUS_SCHEMA_VERSION",
    "family_for_metric",
    "METRIC_STATUS_BOUND_IN_CODE",
    "SOURCE_STATUS_DOCUMENTED_UNPINNED",
    "SOURCE_STATUS_PINNED_IN_LEDGER_FACTS",
    "SOURCE_STATUS_PINNED_IN_LADDER",
    "SOURCE_STATUS_SIGNED_DEFERRED",
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
SOURCE_STATUS_PINNED_IN_LEDGER_FACTS = "pinned_in_ledger_facts"
SOURCE_STATUS_PINNED_IN_LADDER = "pinned_in_ladder"
SOURCE_STATUS_SIGNED_DEFERRED = "signed_deferred"
FENCE_ENFORCEMENT_REVIEW = "review_required_before_binding"

#: Review date for every source row below: each URL was fetched and its
#: product description checked on this date (microcosm#495 scoping).
_SOURCES_VERIFIED_ON = "2026-07-22"

_LEDGER_FACT_FEED_PIN: dict[str, str] = {
    "artifact": ".codex-work/consumer_facts_uk.jsonl",
    "manifest": ".codex-work/consumer_facts_uk_manifest.json",
    "facts_sha256": (
        "6ae49d7d7ab297df25a0b9bfe2d6776827c672d284fbb360957fe8337089549f"
    ),
    "manifest_sha256": (
        "dcda51d6496aea67f768a284e7955c7520e7c8b91e2bed3569f247567b7153f0"
    ),
    "source_repo": "PolicyEngine/chronicle",
    "source_commit": "6fb700e",
    "build": "build-bundle --suite uk -> build-consumer-artifact",
}

_SPI_FRAME_PROXY_FENCE_ID = "hmrc_spi_frame_model_proxy"
_FRS_CIRCULARITY_FENCE_ID = "frs_model_based_target_circularity"
_BHC_AHC_FENCE_ID = "ons_bhc_ahc_noncomparable"
_UC_GRAIN_FENCE_ID = "uc_unit_vs_household_grain"
_POPULATION_UNIVERSE_FENCE_ID = "population_universe_private_households"
_CENSUS_DISCLOSURE_FENCE_ID = "census_disclosure_control_noise"
_COUNCIL_TAX_UNIVERSE_FENCE_ID = "voa_dwellings_vs_household_frame"

#: Exact metric names mapping to a census family. Matching is equality only,
#: so a near-miss such as ``uc_householdsX`` fails closed instead of
#: inheriting the family.
_EXACT_FAMILY_RULES: dict[str, str] = {
    "uc_households": "uc_households",
    "rent/private_rent": "private_rent",
}

#: Metric-name prefixes mapping to a census family. Every prefix ends at a
#: separator so it cannot swallow near-miss sibling names.
#: Exact names continued: the census household-count family is bound via
#: the ladder artifact rather than an in-code engine metric alone.
_EXACT_FAMILY_RULES["households"] = "census_households"

_PREFIX_FAMILY_RULES: tuple[tuple[str, str], ...] = (
    ("hmrc/", "hmrc_income_by_area"),
    ("age/", "age_structure"),
    ("uc_hh_", "uc_households"),
    ("ons/equiv_", "equivalised_income"),
    ("tenure/", "tenure"),
    ("council_tax/", "council_tax"),
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
        "adjudications": [_POPULATION_UNIVERSE_FENCE_ID],
    },
    {
        "family": "census_households",
        "description": (
            "Weighted household counts by area, bound to census occupied-"
            "household totals summed from the sha-pinned UK OA ladder "
            "artifact (constituency_household_targets / "
            "local_authority_household_targets). Universe-compatible with "
            "the FRS instrument: census occupied households match the "
            "survey's own household frame, so the person-universe "
            "adjudication does not bind here."
        ),
        "sources": [
            "nomis_ts041_ew_oa_households",
            "nrs_census_2022_index",
            "nisra_dz21_households",
        ],
        "adjudications": [_CENSUS_DISCLOSURE_FENCE_ID],
    },
    {
        "family": "uc_households",
        "description": (
            "Universal Credit benefit units summed onto household rows, and "
            "(constituency only) their split by benefit-unit num_children."
        ),
        "sources": ["dwp_stat_xplore_uc"],
        "adjudications": [_UC_GRAIN_FENCE_ID],
    },
    {
        "family": "equivalised_income",
        "description": (
            "Equivalised household disposable income before/after housing "
            "costs and implied housing costs (local-authority metrics)."
        ),
        "sources": ["ons_small_area_income_msoa"],
        "adjudications": [_FRS_CIRCULARITY_FENCE_ID, _BHC_AHC_FENCE_ID],
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
    {
        "family": "council_tax",
        "description": (
            "Chargeable dwelling stock by council-tax band A-H at local-"
            "authority grain, represented on the household frame."
        ),
        "sources": ["voa_council_tax_stock_la"],
        "adjudications": [_COUNCIL_TAX_UNIVERSE_FENCE_ID],
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
        "status": SOURCE_STATUS_PINNED_IN_LEDGER_FACTS,
        "ledger_fact_pin": _LEDGER_FACT_FEED_PIN,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "Unbanded area statistics in the pinned Ledger fact feed supply "
            "the local HMRC count targets directly and the amount targets via "
            "the signed count_x_mean construction. The feed publishes SPI "
            "target count/mean measures for 650/650 constituencies except "
            "self_employment_income_mean at E14001416 (so that one amount "
            "row is signed deferred), and for 359/361 crosswalk local "
            "authorities; E06000027 has only median SPI measures and "
            "E06000053 has no SPI local-authority target-measure rows. This "
            "remains a different surface from the "
            f"{HMRC_SPI_TARGET_RECORD_COUNT} banded Total Income facts held "
            "fenced by the national adjudication."
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
        "status": SOURCE_STATUS_PINNED_IN_LEDGER_FACTS,
        "ledger_fact_pin": _LEDGER_FACT_FEED_PIN,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "The pinned Ledger fact feed carries the PCON24 constituency "
            "population age specs for all 650 constituencies across the "
            "ONS, NRS, and NISRA publisher legs. Mid-year estimates cover "
            "the total usual-resident population — see the "
            "population_universe_private_households adjudication."
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
        "status": SOURCE_STATUS_PINNED_IN_LEDGER_FACTS,
        "ledger_fact_pin": _LEDGER_FACT_FEED_PIN,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "The pinned Ledger fact feed carries local-authority population "
            "age specs for all 361 crosswalk local authorities. Covers the "
            "total usual-resident population — see the "
            "population_universe_private_households adjudication."
        ),
    },
    {
        "source_id": "nomis_ts041_ew_oa_households",
        "publisher": "Office for National Statistics (via Nomis)",
        "product": (
            "Census 2021 table TS041 (number of households), England and "
            "Wales, output-area grain — the E&W leg of the ladder's "
            "household counts."
        ),
        "url": "https://www.nomisweb.co.uk/output/census/2021/census2021-ts041.zip",
        "geographies": ["constituency", "la"],
        "latest_vintage": "Census Day 2021-03-21",
        "status": SOURCE_STATUS_PINNED_IN_LADDER,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "Sha-pinned per build by tools/build_uk_oa_ladder_artifact.py "
            "(recorded in the artifact's source_files map)."
        ),
    },
    {
        "source_id": "nrs_census_2022_index",
        "publisher": "National Records of Scotland",
        "product": (
            "Census 2022 index zip: Postcode_To_OA.csv census occupied "
            "household counts (cell-key perturbed), summed by OA2022 — the "
            "Scotland leg of the ladder's household counts."
        ),
        "url": "https://www.nrscotland.gov.uk/media/utrbt5ze/census_2022_index.zip",
        "geographies": ["constituency", "la"],
        "latest_vintage": "Census Day 2022-03-20",
        "status": SOURCE_STATUS_PINNED_IN_LADDER,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "Sha-pinned per build by tools/build_uk_oa_ladder_artifact.py; "
            "the in-zip specification defines HouseholdCount as the 2022 "
            "Census occupied household count."
        ),
    },
    {
        "source_id": "nisra_dz21_households",
        "publisher": "Northern Ireland Statistics and Research Agency",
        "product": (
            "Census 2021 table-builder HOUSEHOLD dataset at DZ21 grain — "
            "the NI leg of the ladder's household counts."
        ),
        "url": "https://build.nisra.gov.uk/en/custom/table.csv?d=HOUSEHOLD&v=DZ21",
        "geographies": ["constituency", "la"],
        "latest_vintage": "Census Day 2021-03-21",
        "status": SOURCE_STATUS_PINNED_IN_LADDER,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": ("Sha-pinned per build by tools/build_uk_oa_ladder_artifact.py."),
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
        "status": SOURCE_STATUS_PINNED_IN_LEDGER_FACTS,
        "ledger_fact_pin": _LEDGER_FACT_FEED_PIN,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "The pinned Ledger fact feed carries DWP Stat-Xplore UC "
            "household facts for 632/650 constituencies and 350/361 local "
            "authorities, plus the four constituency child-bucket specs. "
            "DWP UC official statistics in this feed cover Great Britain; "
            "the 18 Northern Ireland constituencies and 11 Northern Ireland "
            "local authorities are signed deferred. DWP counts UC claim "
            "households — see the uc_unit_vs_household_grain adjudication."
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
        "status": SOURCE_STATUS_SIGNED_DEFERRED,
        "ledger_fact_pin": _LEDGER_FACT_FEED_PIN,
        "signed_reason_id": "msoa_mean_to_la_deferred",
        "signed_rationale": (
            "The pinned Ledger fact feed carries ONS equivalised-income facts "
            "at MSOA grain, not local-authority grain; the local-authority "
            "mean aggregation design is deferred."
        ),
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "Model-based estimates built from the Family Resources Survey — "
            "see the frs_model_based_target_circularity adjudication. The "
            "BHC and AHC measures are independently modelled and not "
            "directly comparable — see the ons_bhc_ahc_noncomparable "
            "adjudication. MSOA-to-local-authority aggregation is signed "
            "deferred for this run; England and Wales only."
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
        "status": SOURCE_STATUS_PINNED_IN_LEDGER_FACTS,
        "ledger_fact_pin": _LEDGER_FACT_FEED_PIN,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "The pinned Ledger fact feed carries tenure specs for all 361 "
            "crosswalk local authorities across ONS Census 2021, NRS Census "
            "2022, and NISRA Census 2021 publisher legs. Private-rent and "
            "social-rent tenure rows are additive over publisher-native "
            "subclasses and compile through the signed sum operation."
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
        "status": SOURCE_STATUS_SIGNED_DEFERRED,
        "ledger_fact_pin": _LEDGER_FACT_FEED_PIN,
        "signed_reason_id": "private_rent_pipr_partial_coverage_2025",
        "signed_rationale": (
            "The pinned Ledger fact feed carries one PIPR period, 2026-06, "
            "with 348 facts: 314 crosswalk England/Wales LA rows, two English "
            "LA ids outside the crosswalk, 18 Scottish BRMA rows, nine region "
            "rows, and five country rows. None is at or before the 2025 target "
            "period; four English crosswalk authorities are absent, Scotland "
            "has no LA rows, and Northern Ireland has no rows at any grain."
        ),
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "PIPR publishes average rent price levels, not additive rent "
            "totals. All 361 local-authority cells remain signed deferred for "
            "the 2025 compile: 314 are after-period, four English cells have "
            "no matching LA id, 32 Scottish cells are represented only at "
            "BRMA grain without a signed translation, and 11 Northern Ireland "
            "cells are absent."
        ),
    },
    {
        "source_id": "voa_council_tax_stock_la",
        "publisher": "Valuation Office Agency",
        "product": (
            "Council Tax stock of properties, 2025: local-authority counts "
            "by valuation band."
        ),
        "url": (
            "https://www.gov.uk/government/statistics/council-tax-stock-of-"
            "properties-2025"
        ),
        "geographies": ["la"],
        "latest_vintage": "2025",
        "status": SOURCE_STATUS_PINNED_IN_LEDGER_FACTS,
        "ledger_fact_pin": _LEDGER_FACT_FEED_PIN,
        "verified_on": _SOURCES_VERIFIED_ON,
        "notes": (
            "The pinned feed record-set spec "
            "uk.local_geography.council_tax_stock.by_local_authority.v1 "
            "supplies 2,541 active band cells: bands B-G cover all 318 "
            "England/Wales crosswalk authorities, band A covers 317 because "
            "E09000001 is suppressed, and band H covers 316 because "
            "W06000019 and W06000024 are absent. Scotland's 32 authorities "
            "have no VOA band-count rows and Northern Ireland's 11 LGDs use "
            "domestic rates; all 347 absent cells are signed deferrals. The "
            "feed has no comparable 2025 LA net series across the roster, so "
            "council_tax/net is not declared."
        ),
    },
)

_BINDING_FENCES: tuple[dict[str, Any], ...] = (
    {
        "fence_id": FULL_FRS_TI_BAND_FENCE_ID,
        "fenced_fact_count": HMRC_SPI_TARGET_RECORD_COUNT,
        "enforcement": FENCE_ENFORCEMENT_REVIEW,
        "rule": (
            "No local target family may bind facts banded by SPI Total "
            "Income: the FRS instrument cannot materialize complete Total "
            "Income, so band membership is unassignable. This is the census's "
            "application of the national adjudication that holds all "
            f"{HMRC_SPI_TARGET_RECORD_COUNT} published banded facts as "
            "fenced exclusions (the canonical fence text lives with the "
            "national constants); minting banded local variants is "
            "forbidden."
        ),
        "authority": (
            "UK_COVERAGE_PROGRESS.md real-donor HMRC replay (2026-07-13); "
            "canonical fences in microcosm.build.uk_runtime.hmrc_replay."
        ),
    },
    {
        "fence_id": _SPI_FRAME_PROXY_FENCE_ID,
        "fenced_fact_count": None,
        "enforcement": FENCE_ENFORCEMENT_REVIEW,
        "rule": (
            "The in-code HMRC area metrics gate frame membership on modeled "
            "income_tax > 0 — a model-output proxy for 'taxpayer in the SPI "
            "frame', not a source-faithful frame. Binding any HMRC area "
            "family requires an explicit adjudication accepting or replacing "
            "that proxy before targets enter a solve."
        ),
        "authority": (
            "microcosm.build.uk_runtime.local_targets.compute_household_metrics "
            "(in_spi_frame definition); microcosm#495 scoping."
        ),
    },
    {
        "fence_id": _FRS_CIRCULARITY_FENCE_ID,
        "fenced_fact_count": None,
        "enforcement": FENCE_ENFORCEMENT_REVIEW,
        "rule": (
            "ONS income estimates for small areas are model-based outputs "
            "built from the Family Resources Survey. Calibrating FRS-derived "
            "microdata to them feeds the instrument back into itself; "
            "binding requires an explicit adjudication of that circularity "
            "(accept with documented rationale, or reject the family)."
        ),
        "authority": (
            "ONS income estimates for small areas QMI (FRS-based method); "
            "microcosm#495 scoping."
        ),
    },
    {
        "fence_id": _BHC_AHC_FENCE_ID,
        "fenced_fact_count": None,
        "enforcement": FENCE_ENFORCEMENT_REVIEW,
        "rule": (
            "The runtime's ons/equiv_housing_costs metric has no declared "
            "supplier: ONS states its small-area BHC and AHC income "
            "estimates are independently modelled and should not be "
            "directly compared, so their difference is not a publishable "
            "housing-costs target. Binding the equivalised-income family "
            "must target the published BHC and AHC measures separately or "
            "document a different housing-costs source."
        ),
        "authority": (
            "ONS income estimates for small areas, England and Wales, "
            "financial year ending 2023 (comparability guidance); "
            "microcosm#495 scoping (cross-family review finding)."
        ),
    },
    {
        "fence_id": _UC_GRAIN_FENCE_ID,
        "fenced_fact_count": None,
        "enforcement": FENCE_ENFORCEMENT_REVIEW,
        "rule": (
            "The runtime maps benefit-unit-level UC receipt and each UC "
            "benefit unit's num_children band onto household rows by sum: a "
            "physical household containing two UC benefit units contributes "
            "2 to uc_households and one contribution to each unit's child "
            "band. This aligns the runtime with DWP UC claim households, "
            "which correspond to benefit units rather than physical FRS "
            "households; binding must preserve this reviewed grain crosswalk."
        ),
        "authority": (
            "DWP Universal Credit official statistics Stat-Xplore user "
            "guide (household definition); "
            "microcosm.build.uk_runtime.local_targets.compute_household_metrics "
            "(benunit-to-household mapping); microcosm#495 scoping "
            "(cross-family review finding)."
        ),
    },
    {
        "fence_id": _CENSUS_DISCLOSURE_FENCE_ID,
        "fenced_fact_count": None,
        "enforcement": FENCE_ENFORCEMENT_REVIEW,
        "rule": (
            "All three census household-count legs are disclosure-controlled "
            "(ONS/NRS cell-key perturbation; NISRA flexible-table-builder "
            "controls), so area counts carry small deliberate noise and do "
            "not add exactly across grains (measured constituency-sum vs "
            "national-total deltas at review: E&W +105, Scotland -554, NI "
            "+3). Binding treats the published counts as the target values "
            "with that noise documented — never as exact controls — and any "
            "cross-grain reconciliation applies the standing cross-grain rule "
            "declared in uk_runtime.ledger_targets: country wins, so a bound "
            "national same-concept control rescales the constituency values "
            "before the solve."
        ),
        "authority": (
            "ONS/NRS/NISRA statistical disclosure control documentation; "
            "microcosm#495 scoping (cross-family review measurement)."
        ),
    },
    {
        "fence_id": _POPULATION_UNIVERSE_FENCE_ID,
        "fenced_fact_count": None,
        "enforcement": FENCE_ENFORCEMENT_REVIEW,
        "rule": (
            "The FRS covers private households, while ONS mid-year "
            "population estimates cover the total usual-resident population "
            "including communal establishments (care homes, student halls, "
            "barracks, prisons). Binding age-structure targets must declare "
            "the universe treatment (communal-establishment adjustment or "
            "documented acceptance) per area type."
        ),
        "authority": (
            "FRS background and methodology (private-household universe); "
            "ONS population estimates methodology (communal-establishment "
            "population); microcosm#495 scoping (cross-family review "
            "finding)."
        ),
    },
    {
        "fence_id": _COUNCIL_TAX_UNIVERSE_FENCE_ID,
        "fenced_fact_count": None,
        "enforcement": FENCE_ENFORCEMENT_REVIEW,
        "rule": (
            "VOA council-tax stock counts chargeable dwellings, including "
            "empty properties, second homes, and other dwellings that need "
            "not correspond one-for-one with occupied private households in "
            "the FRS frame. Binding the band-count family requires an explicit "
            "adjudication accepting the household proxy or a source-faithful "
            "dwelling representation."
        ),
        "authority": (
            "uk-data targets/sources/la_council_tax.py lineage doctrine and "
            "datasets/local_areas/local_authorities/loss.py; uk-data#371; "
            "microcosm#147 adjudication A3."
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
    SOURCE_STATUS_PINNED_IN_LEDGER_FACTS: (
        "The official product's target facts are present in the sha-pinned "
        "Ledger consumer fact feed recorded on the source row."
    ),
    SOURCE_STATUS_PINNED_IN_LADDER: (
        "The product is downloaded and sha-pinned per build by the UK OA "
        "ladder artifact tool, which records every source hash in the "
        "artifact metadata; target values derive from the artifact's own "
        "sums."
    ),
    SOURCE_STATUS_SIGNED_DEFERRED: (
        "The official product is present or documented, but this target "
        "surface does not bind it in the current compile; the source row "
        "carries signed_reason_id and signed_rationale."
    ),
    FENCE_ENFORCEMENT_REVIEW: (
        "The fence is a reviewed adjudication requirement enforced by the "
        "rowwise doctrine solve "
        "(local_rowwise.require_adjudicated_uk_local_binding): binding a "
        "family whose fences lack an in-force entry in "
        "local_binding_adjudications.json is refused before any solve runs."
    ),
}

_SCOPE_NOTE = (
    "This census covers the default in-code metric surface returned by "
    "local_targets.metric_names(area_type) with no target profile. "
    "Ledger target-profile-driven metric surfaces are governed by their "
    "profile contract and are out of census scope."
)

_DOCTRINE: dict[str, Any] = {
    "masked_missing": {
        "rule": (
            "A missing local target is represented at compile time by an "
            "AreaSignedDeferral carrying a signed reason and explicit area ids. "
            "Unsigned absence raises, and a deferral becomes stale and raises "
            "as soon as the pinned feed can compile that cell."
        ),
        "enforcement_point": (
            "Both refusals fire inside the reference compiler, which runs only "
            "when the surface is regenerated against the pinned consumer feed "
            "(a licensed, untracked input passed explicitly to "
            "tools/generate_uk_local_target_references.py). CI verifies the "
            "committed compile through the census and membership drift gates; "
            "it cannot re-run the compiler, so a feed that later carries new "
            "facts is caught at the next regeneration, not continuously."
        ),
        "solve_semantics": (
            "The solve surface remains dense and finite by design; "
            "local_rowwise.py refuses non-finite targets rather than treating "
            "NaN as an implicit loss mask."
        ),
        "incumbent_comparison": (
            "This supersedes uk-data's per-cell NaN mask with a stronger, "
            "reviewable compile-time accounting contract."
        ),
    },
    "zero_targets": {
        "rule": "A published zero-valued local target is legal and intentional.",
        "enforcement": (
            "Zero is preserved as a finite target; only an unreachable nonzero "
            "target is refused by the matrix builder."
        ),
        "test": "test_matrix_builder_fails_closed_on_unreachable_nonzero_targets",
    },
    "never_fabricate": {
        "rule": (
            "Never create a local target by allocating a national or country "
            "total across areas by population share or another smooth proxy."
        ),
        "cautionary_example": (
            "uk-data datasets/local_areas/constituencies/devolved_housing.py "
            "allocated hardcoded Wales and Scotland rent totals across "
            "constituencies in proportion to age-target population. Those "
            "anchors are deliberately not ported."
        ),
        "replacement": (
            "Record signed absence, a reviewed exclusion, or a declared "
            "cross-grain bridge backed by a pinned crosswalk; never mint cells."
        ),
    },
    "acceptance_criteria_to_mechanism": {
        "every_target_accounted": (
            "The local target compiler requires a compiled reference or an "
            "AreaSignedDeferral for every metric-by-area cell; unsigned and "
            "stale deferrals raise."
        ),
        "missing_cells_masked_deliberately": (
            "AreaSignedDeferral is the deliberate compile-time mask; the solve "
            "accepts only finite rows."
        ),
        "intentional_zeros_preserved": (
            "Finite zero facts compile unchanged and are distinguished from "
            "missing or unreachable nonzero targets."
        ),
        "no_silent_target_fabrication": (
            "The compiler reads only the sha-pinned Ledger feed, and the "
            "devolved_housing population-share allocation is a reviewed "
            "non-port."
        ),
        "binding_requires_review": (
            "require_adjudicated_uk_local_binding refuses a family whose "
            "declared fences lack an in-force local_binding_adjudications row."
        ),
    },
}


def build_uk_local_target_census() -> dict[str, Any]:
    """Build the census from the live metric surface plus reviewed registers."""

    area_metric_order: dict[str, list[str]] = {}
    metric_rows: dict[str, dict[str, Any]] = {}
    for area_type in AREA_TYPES:
        names = [str(name) for name in metric_names(area_type)]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"UK local metric surface for {area_type!r} declares "
                f"duplicate metric name(s): {duplicates}."
            )
        area_metric_order[area_type] = names
        for name in names:
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
    _require_unique_ids(families, key="family", label="family")
    sources = [dict(row) for row in _SOURCES]
    _require_unique_ids(sources, key="source_id", label="source")
    fences = [dict(row) for row in _BINDING_FENCES]
    _require_unique_ids(fences, key="fence_id", label="binding fence")

    source_ids = {row["source_id"] for row in sources}
    fence_ids = {row["fence_id"] for row in fences}
    used_families = {row["family"] for row in metric_rows.values()}
    declared_families = {row["family"] for row in families}
    unused = sorted(declared_families - used_families)
    if unused:
        raise ValueError(f"census declares family(ies) with no metrics: {unused}.")
    unknown_families = sorted(used_families - declared_families)
    if unknown_families:
        raise ValueError(
            f"census classified metric(s) into undeclared family(ies): "
            f"{unknown_families}."
        )
    for family in families:
        unknown_sources = sorted(set(family["sources"]) - source_ids)
        if unknown_sources:
            raise ValueError(
                f"family {family['family']!r} references unknown source(s): "
                f"{unknown_sources}."
            )
        unknown_fences = sorted(set(family.get("adjudications", [])) - fence_ids)
        if unknown_fences:
            raise ValueError(
                f"family {family['family']!r} references unknown fence(s): "
                f"{unknown_fences}."
            )

    payload = {
        "schema_version": CENSUS_SCHEMA_VERSION,
        "census_kind": CENSUS_KIND,
        "scope": _SCOPE_NOTE,
        "doctrine": copy.deepcopy(_DOCTRINE),
        "area_types": list(AREA_TYPES),
        "area_metric_order": area_metric_order,
        "metrics": [metric_rows[name] for name in sorted(metric_rows)],
        "families": families,
        "sources": sources,
        "binding_fences": fences,
        "status_definitions": dict(_STATUS_DEFINITIONS),
    }
    return copy.deepcopy(payload)


def committed_uk_local_target_census_path() -> Path:
    """Path of the committed census artifact inside ``microcosm.build.uk``."""

    return Path(str(files("microcosm.build.uk").joinpath(CENSUS_RESOURCE)))


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
    """Fail if the committed census no longer matches the live surface.

    Comparison is by canonical JSON serialization, not Python equality, so
    JSON-visible type drift (``true`` vs ``1``, ``1.0`` vs ``1``) is stale.
    """

    committed = load_uk_local_target_census(path)
    live = build_uk_local_target_census()
    committed_text = json.dumps(committed, sort_keys=True)
    live_text = json.dumps(live, sort_keys=True)
    if committed_text != live_text:
        drifted = _drifted_keys(committed, live)
        raise ValueError(
            "UK local-target census is stale: committed artifact does not "
            f"match the live metric surface (drift in {drifted}). Regenerate "
            "with `uv run python tools/census_uk_local_targets.py`."
        )


def _family_for_metric(name: str) -> str:
    exact = _EXACT_FAMILY_RULES.get(name)
    if exact is not None:
        return exact
    for prefix, family in _PREFIX_FAMILY_RULES:
        if name.startswith(prefix):
            return family
    raise ValueError(
        f"UK local metric {name!r} has no census family classification; add "
        "a rule to local_target_census._EXACT_FAMILY_RULES or "
        "_PREFIX_FAMILY_RULES with its official source(s) before shipping "
        "it."
    )


def family_for_metric(name: str) -> str:
    """Public fail-closed UK local metric-to-family classifier."""

    return _family_for_metric(name)


def _require_unique_ids(
    rows: list[dict[str, Any]],
    *,
    key: str,
    label: str,
) -> None:
    ids = [str(row[key]) for row in rows]
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"census declares duplicate {label} id(s): {duplicates}.")


def _drifted_keys(committed: dict[str, Any], live: dict[str, Any]) -> list[str]:
    keys = sorted(set(committed) | set(live))
    return [
        key
        for key in keys
        if json.dumps(committed.get(key), sort_keys=True)
        != json.dumps(live.get(key), sort_keys=True)
    ]
