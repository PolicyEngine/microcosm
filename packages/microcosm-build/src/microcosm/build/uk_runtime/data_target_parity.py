"""Late uk-data target, anchor, and credibility-check parity register."""

from __future__ import annotations

import copy
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from microcosm.build.uk_runtime.local_targets import load_uk_population_contract

UK_DATA_TARGET_PARITY_RESOURCE = "uk_data_target_parity.json"
UK_DATA_TARGET_PARITY_SCHEMA_VERSION = 1

VALID_PARITY_STATUSES = frozenset(
    {
        "ported_national",
        "ported_local_declared",
        "ported_local_bound",
        "routed",
        "blocked_source",
        "reviewed_exclusion",
    }
)
PORTED_PARITY_STATUSES = frozenset(
    {"ported_national", "ported_local_declared", "ported_local_bound"}
)

_REFERENCE = {
    "incumbent_tree": ".codex-work/uk-data-8629dbb",
    "incumbent_commit": "8629dbb",
    "verified_on": "2026-08-31",
    "governing_rule": (
        "Every late uk-data target family, dataset-level anchor, and credibility "
        "check has a Microcosm home or an explicit Chesterton fence."
    ),
    "issue_citation_rule": "Incumbent issues are cited as uk-data#NNN.",
}


def _fence(origin: str, purpose: str, verdict_basis: str) -> dict[str, str]:
    return {
        "origin": origin,
        "purpose": purpose,
        "verdict_basis": verdict_basis,
    }


_CONCERNS: tuple[dict[str, Any], ...] = (
    {
        "concern_id": "national_obr_receipts",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/obr.py: OBR receipt rows; Microcosm uk/uk_population_targets.json: obr.income_tax, obr.vat, obr.fuel_duty, obr.capital_gains_tax, and obr.stamp_duty_land_tax.",
    },
    {
        "concern_id": "national_obr_national_insurance",
        "status": "ported_national",
        "classification": "red_line_national_family",
        "evidence": "uk-data targets/sources/obr.py:348; Microcosm target ids obr.ni, obr.ni_employee, obr.ni_employer, and obr.ni_self_employed.",
    },
    {
        "concern_id": "national_obr_national_insurance_class_3",
        "status": "reviewed_exclusion",
        "classification": "not_modeled",
        "reason": "The incumbent deliberately skipped Class 3 and no Microcosm dataset path exists.",
        "evidence": "uk-data targets/sources/obr.py:402-409 and uk-data#88.",
        "fence": _fence(
            "uk-data skipped the Class 3 OBR row pending uk-data#88.",
            "Avoid pretending voluntary Class 3 contributions are represented by another NIC variable.",
            "Keep the omission explicit until the country model exposes a source-faithful Class 3 path.",
        ),
    },
    {
        "concern_id": "national_obr_council_tax",
        "status": "ported_national",
        "classification": "red_line_national_family",
        "evidence": "uk-data targets/sources/obr.py:284; Microcosm target ids obr.council_tax and country legs obr.council_tax_{england,scotland,wales}.",
    },
    {
        "concern_id": "national_obr_welfare",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/obr.py:458 welfare rows; Microcosm uk_population_targets.json OBR benefit-spending targets.",
    },
    {
        "concern_id": "national_tv_licence_and_policy_statics",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/obr.py:617-728; Microcosm uk_population_targets.json tv_licence, private-school, and salary-sacrifice target rows.",
    },
    {
        "concern_id": "national_hmrc_spi_income_bands",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/hmrc_spi.py:230; Microcosm HMRC SPI amount/count-by-total-income-band targets and replay fences.",
    },
    {
        "concern_id": "national_hmrc_cgt_size_bands",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/hmrc_cgt.py:195; Microcosm hmrc_cgt_size_bands.json and uk_population_targets.json CGT rows.",
    },
    {
        "concern_id": "national_hmrc_salary_sacrifice",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/hmrc_salary_sacrifice.py; Microcosm hmrc.salary_sacrifice target rows.",
    },
    {
        "concern_id": "national_dwp_benefits_and_uc_caseloads",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/dwp.py, including post-ebf733c uk-data#458 UC claimant-by-children, family-type, and two-child-limit caseload rows; Microcosm DWP target rows in uk_population_targets.json.",
    },
    {
        "concern_id": "national_ons_demographics",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/ons_demographics.py; Microcosm ONS population targets in uk_population_targets.json.",
    },
    {
        "concern_id": "national_ons_household_composition",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/ons_households.py; Microcosm ten-row ons.household_composition partition and UK_CROSS_GRAIN_BRIDGES declaration.",
    },
    {
        "concern_id": "national_ons_england_tenure",
        "status": "routed",
        "classification": "national_grain_follow_up",
        "reason": "The incumbent England-national tenure controls are not part of the current national contract; local tenure is declared separately.",
        "evidence": "uk-data targets/sources/ons_tenure.py:69; Microcosm local ons.tenure.* contract rows.",
        "fence": _fence(
            "uk-data calibrated five England tenure controls.",
            "Preserve an England-level representation check independently of the local Census tenure surface.",
            "Route to the national target workstream; do not synthesize an England total from local cells.",
        ),
    },
    {
        "concern_id": "national_ons_savings_interest",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/ons_savings.py; Microcosm savings-interest target translation in uk_runtime/ledger_targets.py.",
    },
    {
        "concern_id": "national_ons_public_sector_employment",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/ons_public_sector_employment.py; Microcosm target id ons.public_sector_employment.",
    },
    {
        "concern_id": "local_ons_public_sector_employment",
        "status": "routed",
        "classification": "awaiting_facts",
        "reason": "No pinned local-grain PSE facts are available.",
        "evidence": "Microcosm local_validation_levels.json regional PSE placeholder; Chronicle fact request required.",
        "fence": _fence(
            "uk-data carried only the national PSE control and a credibility test.",
            "Do not infer regional employment from the national total.",
            "Retain a report-only placeholder until Chronicle supplies regional facts.",
        ),
    },
    {
        "concern_id": "national_ons_land_values",
        "status": "ported_national",
        "classification": "red_line_national_family",
        "evidence": "uk-data targets/sources/ons_land_values.py and _land.py; Microcosm target ids ons.land.{land_value,household_land_value,corporate_land_value}.",
    },
    {
        "concern_id": "regional_household_land_values",
        "status": "routed",
        "classification": "adjudication_pending",
        "reason": "The incumbent uses a uniform national land-to-property ratio; uk-data#357 per-region ratios never merged and Chronicle#205 WAS R8 facts are pending.",
        "evidence": "uk-data targets/sources/mhclg_regional_land.py and storage/regional_land_values.csv; adjudication A5.",
        "fence": _fence(
            "uk-data calibrated 11 GB regional household-land rows using a uniform ratio.",
            "Guard regional ordering without presenting a national scalar as region-specific evidence.",
            "Await A5 and Chronicle#205; property_wealth_intensity is currently national and the target alone cannot move regional intensity.",
        ),
    },
    {
        "concern_id": "national_council_tax_stock",
        "status": "ported_national",
        "classification": "red_line_national_family",
        "evidence": "uk-data targets/sources/voa_council_tax.py:235,275; Microcosm voa.council_tax_stock.* and scotgov.council_tax_stock.* targets.",
    },
    {
        "concern_id": "national_housing_rate_headcount_products",
        "status": "blocked_source",
        "classification": "source_unpinned",
        "reason": "The mortgage/rent product source is not pinned.",
        "evidence": "uk-data targets/sources/housing.py; housing/total_mortgage cites uk-data#453.",
        "fence": _fence(
            "uk-data multiplied three housing rates by headcounts.",
            "Prevent an unpinned derived product from becoming a calibration control.",
            "Block until the underlying rate and population sources are pinned and their universes reconciled.",
        ),
    },
    {
        "concern_id": "national_nts_vehicle_counts",
        "status": "routed",
        "classification": "national_grain_follow_up",
        "reason": "The three NTS vehicle controls have no current national contract analog.",
        "evidence": "uk-data targets/sources/nts_vehicles.py; route to the transport workstream.",
        "fence": _fence(
            "uk-data calibrated household vehicle-count controls from NTS.",
            "Preserve the vehicle-distribution anchor.",
            "Route to the national transport surface; no local allocation is authorized.",
        ),
    },
    {
        "concern_id": "national_student_loans",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/slc.py and slc_repayments.py; Microcosm slc.* target rows and slc_liable_stocks.json.",
    },
    {
        "concern_id": "national_scottish_child_payment",
        "status": "ported_national",
        "classification": "national_registry_family",
        "evidence": "uk-data targets/sources/scottish_government.py; Microcosm Scottish Child Payment contract row.",
    },
    {
        "concern_id": "local_council_tax_band_counts",
        "status": "ported_local_declared",
        "classification": "local_registry_family",
        "evidence": "uk-data targets/sources/la_council_tax.py and datasets/local_areas/local_authorities/loss.py:251-300; Microcosm council_tax/band_a..h contract rows compile 2,541 pinned-feed cells with 347 signed deferrals.",
    },
    {
        "concern_id": "local_private_rent_pipr",
        "status": "ported_local_declared",
        "classification": "local_registry_family",
        "evidence": "Microcosm ons.rent.private_rent is declared, but the pinned feed has only 2026-06 PIPR facts: 314 crosswalk E&W rows are after the 2025 target period, four English LAs are absent, 32 Scottish LAs have BRMA-only facts, and 11 NI LAs are absent.",
    },
    {
        "concern_id": "cross_grain_private_rent_scotland_brma",
        "status": "routed",
        "classification": "cross_grain_grain_gap",
        "reason": "CrossGrainBridge cannot express an overlapping BRMA-to-LA geography translation and no signed crosswalk is available.",
        "evidence": "Pinned feed has 18 Scottish statistical_scope BRMA rows at 2026-06 and zero Scottish LA rows; the local crosswalk has 32 Scottish LAs.",
        "fence": _fence(
            "PIPR publishes Scottish rent levels at BRMA grain.",
            "Preserve source geography without allocating BRMA averages over local authorities.",
            "Keep all 32 Scottish LA cells signed deferred until a source-backed BRMA-to-LA bridge with an explicit aggregation rule exists.",
        ),
    },
    {
        "concern_id": "local_council_tax_net",
        "status": "blocked_source",
        "classification": "awaiting_facts",
        "reason": "The pinned feed has no comparable 2025 local-authority council-tax net series across the 361-area roster.",
        "evidence": "Pinned feed contains country totals and partial/local later-vintage budget series, but no uk.local_geography 2025 LA net record-set spec; needs Chronicle.",
        "fence": _fence(
            "uk-data wired housing/council_tax_net for 318 local authorities.",
            "Anchor net council-tax liability after reductions at LA grain.",
            "Do not combine unlike country totals, 2026 Welsh budgets, or gross dwelling stock; await a comparable pinned LA series.",
        ),
    },
    {
        "concern_id": "cross_grain_council_tax_stock_country_over_la",
        "status": "ported_local_declared",
        "classification": "cross_grain_rule",
        "evidence": "Local voa.council_tax_stock.by_area.band_a..h measurement signatures exactly match the national VOA/scotgov band-count signatures; apply_uk_cross_grain_reconciliation uses UK_CROSS_GRAIN_RULE so a bound country control rescales LA values.",
    },
    {
        "concern_id": "cross_grain_council_tax_stock_england_region",
        "status": "routed",
        "classification": "cross_grain_grain_gap",
        "reason": "The incumbent English VOA controls are region-grain, while the #802 rule declares only country > constituency > LA and cannot accept region rows.",
        "evidence": "Microcosm voa.council_tax_stock.band_* geography_levels include region; CrossGrainRule.grain_precedence is country,constituency,la.",
        "fence": _fence(
            "The national contract materializes English VOA stock by region and Wales by country.",
            "Prevent the LA solve from implicitly choosing between conflicting regional and local stock totals.",
            "Use exact-signature country reconciliation where a country leg exists; record the England regional leg here until #802 machinery gains an adjudicated region bridge.",
        ),
    },
    {
        "concern_id": "local_council_tax_band_d_rate",
        "status": "reviewed_exclusion",
        "classification": "non_linear",
        "reason": "Band D currency amounts are per-rate, not additive household controls.",
        "evidence": "uk-data targets/sources/la_council_tax.py module doctrine; no Microcosm contract row.",
        "fence": _fence(
            "uk-data published Band D rates but did not wire them into the additive LA loss matrix.",
            "Avoid treating a rate as an additive target.",
            "Keep excluded until a rate-aware non-linear objective exists.",
        ),
    },
    {
        "concern_id": "local_devolved_constituency_rent_anchors",
        "status": "reviewed_exclusion",
        "classification": "fabricated_allocation",
        "reason": "The incumbent allocated country totals across constituencies by population share; Microcosm doctrine forbids fabricated local targets.",
        "evidence": "uk-data datasets/local_areas/constituencies/devolved_housing.py; uk-data#316 and uk-data#206.",
        "fence": _fence(
            "uk-data created Wales and Scotland constituency rent anchors from hardcoded country totals.",
            "Provide a smooth constraint where no constituency facts existed.",
            "Do not rebuild: compile-time signed absence is preferred to population-share fabrication.",
        ),
    },
    {
        "concern_id": "dataset_anchor_rail_subsidy",
        "status": "routed",
        "classification": "imputation_anchor",
        "reason": "The £21.6bn rail anchor belongs to WS-E/F (#148), not the local target solve.",
        "evidence": "uk-data datasets/imputations/services/services.py:18; Chronicle transport facts landed in #200/#202.",
        "fence": _fence(
            "uk-data rescaled rail_usage to a national subsidy total.",
            "Anchor the imputed service-consumption mass.",
            "Route to WS-E/F (#148); do not recast it as a local calibration target.",
        ),
    },
    {
        "concern_id": "dataset_anchor_bus_subsidy_and_fares",
        "status": "routed",
        "classification": "imputation_anchor",
        "reason": "BUS05 subsidy/fare totals and the NTS0705 quintile IPF belong to WS-E/F (#148).",
        "evidence": "uk-data datasets/imputations/services/services.py:33 and datasets/imputations/consumption.py:790,830; Chronicle#200/#202.",
        "fence": _fence(
            "uk-data post-imputation-rescaled bus subsidy and fares, including London share and quintile rates.",
            "Preserve service-consumption totals and distribution.",
            "Route to WS-E/F (#148), using landed DfT facts; never allocate the national total over local areas.",
        ),
    },
    {
        "concern_id": "dataset_anchor_dfe_education_spending",
        "status": "reviewed_exclusion",
        "classification": "untargeted_both_sides",
        "reason": "Education spending was imputed but its aggregate test was commented out; corrected post-ebf733c DfE entitlement data from uk-data#474 remains input-side.",
        "evidence": "uk-data datasets/imputations/frs_only.py and policyengine_uk_data/tests/test_aggregates.py:5; uk-data#199 and uk-data#474.",
        "fence": _fence(
            "uk-data imputed DfE education spending without an active calibration target.",
            "Retain the imputed input while avoiding an unsupported aggregate claim.",
            "No target fence to rebuild; keep the corrected DfE entitlement data on the input side until an official aggregate is declared.",
        ),
    },
    {
        "concern_id": "dataset_anchor_vat_band_shares",
        "status": "routed",
        "classification": "imputation_side",
        "reason": "VAT band shares are imputation-side; only the OBR VAT receipts total is a calibration target.",
        "evidence": "uk-data datasets/imputations/vat.py and uk-data#241; Microcosm target id obr.vat.",
        "fence": _fence(
            "uk-data imputed VAT categories from the ETB and calibrated only aggregate receipts.",
            "Preserve category composition without presenting it as administrative calibration evidence.",
            "Keep the band shares in the imputation lane and the compiled OBR receipts total in calibration.",
        ),
    },
    {
        "concern_id": "credibility_population_fidelity",
        "status": "routed",
        "classification": "validation_check",
        "reason": "Population fidelity is a release validation concern, not another calibration target.",
        "evidence": "uk-data population fidelity test (69.5m ±4%); Microcosm release/gate validation lane.",
        "fence": _fence(
            "uk-data aggregate test.",
            "Detect gross population-scale drift.",
            "Retain as a validation check in the gate workstream, outside this target-activation PR.",
        ),
    },
    {
        "concern_id": "credibility_aggregate_smokes",
        "status": "routed",
        "classification": "validation_check",
        "reason": "NHS/rail/bus ±70% smokes belong to report-only validation.",
        "evidence": "uk-data aggregate smoke tests; DfE and bus-fare checks were commented out.",
        "fence": _fence(
            "uk-data aggregate tests.",
            "Catch order-of-magnitude imputation failures.",
            "Route to the credibility-gate workstream; do not bind the validation totals.",
        ),
    },
    {
        "concern_id": "credibility_bus_fare_distribution",
        "status": "routed",
        "classification": "validation_check",
        "reason": "Bus total and London-share checks follow the WS-E/F transport anchors.",
        "evidence": "uk-data bus fare distribution test; Chronicle#200/#202.",
        "fence": _fence(
            "uk-data consumption test.",
            "Protect the bus-fare total and London share.",
            "Route with the transport imputation anchors to #148.",
        ),
    },
    {
        "concern_id": "credibility_nic_signal",
        "status": "routed",
        "classification": "validation_check",
        "reason": "Non-zero variation per active NIC target is a validation invariant over the ported national family.",
        "evidence": "uk-data OBR NIC signal test; Microcosm obr.ni* target rows.",
        "fence": _fence(
            "uk-data target credibility test.",
            "Catch inert NIC target columns.",
            "Route to national target-materialization validation; the targets themselves are ported.",
        ),
    },
    {
        "concern_id": "credibility_public_sector_employment",
        "status": "routed",
        "classification": "validation_check",
        "reason": "The incumbent's 50% simulated / 20% target sanity check validates the ported PSE target.",
        "evidence": "uk-data public-sector-employment aggregate test; Microcosm ons.public_sector_employment.",
        "fence": _fence(
            "uk-data aggregate test documented the FRS overcount.",
            "Expose PSE universe/model mismatch.",
            "Route to post-solve validation; do not alter the pinned target value.",
        ),
    },
    {
        "concern_id": "credibility_regional_land_ordering",
        "status": "routed",
        "classification": "validation_check",
        "reason": "Regional sum/order checks depend on the unresolved regional-land A5 decision.",
        "evidence": "uk-data regional-land tests; uk-data#357 and Chronicle#205.",
        "fence": _fence(
            "uk-data regional land validation tests.",
            "Guard the London-highest and London/NE relationship.",
            "Route with A5; do not validate a regional surface that is not yet evidenced.",
        ),
    },
    {
        "concern_id": "credibility_local_council_tax_outliers",
        "status": "routed",
        "classification": "validation_check",
        "reason": "The Scilly 2000× leak guard belongs to post-compile local validation.",
        "evidence": "uk-data LA council-tax outlier test and uk-data#371; Microcosm council-tax LA surface.",
        "fence": _fence(
            "uk-data LA target test.",
            "Catch implausible band-count/rate joins.",
            "Route to local validation once council-tax references compile; do not modify target values to satisfy the check.",
        ),
    },
    {
        "concern_id": "credibility_local_missing_source_mask",
        "status": "ported_local_declared",
        "classification": "compile_contract",
        "evidence": "uk-data finite/NaN missing-source test; Microcosm AreaSignedDeferral compiler, stale-deferral refusal, and finite-target solve surface.",
    },
    {
        "concern_id": "credibility_reform_impact_regression",
        "status": "routed",
        "classification": "validation_check",
        "reason": "The £0.1bn reform-impact suite belongs to microcosm#365, not the target contract.",
        "evidence": "uk-data reform-impact regression tests; microcosm#365.",
        "fence": _fence(
            "uk-data regression suite.",
            "Detect economically material end-to-end drift.",
            "Route to microcosm#365; target activation must not tune inputs to this outcome test.",
        ),
    },
    {
        "concern_id": "credibility_registry_and_database_contracts",
        "status": "routed",
        "classification": "validation_check",
        "reason": "Registry/database and local-H5 publication contracts are superseded by Microcosm spec and artifact gates.",
        "evidence": "uk-data registry/db tests and publish_local_h5s.py::validate_local_h5s; Microcosm country-package and release gates.",
        "fence": _fence(
            "uk-data packaging tests.",
            "Prevent malformed target registries and local artifacts.",
            "Use Microcosm country-spec/load and release gates rather than porting the retired H5 publisher.",
        ),
    },
)


def _local_contract_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target in load_uk_population_contract().get("targets", ()):
        levels = tuple(target.get("geography_levels", ()))
        if not {"constituency", "local_authority"}.intersection(levels):
            continue
        target_id = str(target["target_id"])
        rows.append(
            {
                "concern_id": f"local_contract.{target_id}",
                "status": "ported_local_declared",
                "classification": "red_line_local_contract_target",
                "evidence": (
                    "Microcosm uk/uk_population_targets.json target "
                    f"{target_id}; incumbent local matrices enumerated in "
                    ".codex-work/REFERENCE.md."
                ),
            }
        )
    return rows


def build_uk_data_target_parity() -> dict[str, Any]:
    """Build the canonical parity register from declarations and live targets."""

    rows = [copy.deepcopy(row) for row in _CONCERNS]
    rows.extend(_local_contract_rows())
    rows.sort(key=lambda row: str(row["concern_id"]))
    _validate_rows(rows)
    return {
        "schema_version": UK_DATA_TARGET_PARITY_SCHEMA_VERSION,
        "register_kind": "uk_data_target_parity",
        "reference": copy.deepcopy(_REFERENCE),
        "status_definitions": {
            "ported_national": "Represented on the Microcosm national target surface.",
            "ported_local_declared": "Declared and compile-accounted on the local contract; solve binding still requires adjudication.",
            "ported_local_bound": "Declared, compiled, and admitted to the local solve.",
            "routed": "Owned by another named workstream or report-only validation lane.",
            "blocked_source": "Cannot proceed without pinned source evidence.",
            "reviewed_exclusion": "Deliberately not ported after review, with a complete fence.",
        },
        "concerns": rows,
    }


def _validate_rows(rows: list[dict[str, Any]]) -> None:
    ids = [str(row.get("concern_id", "")) for row in rows]
    if any(not concern_id for concern_id in ids):
        raise ValueError("UK data-target parity rows require concern_id.")
    duplicates = sorted({value for value in ids if ids.count(value) > 1})
    if duplicates:
        raise ValueError(f"duplicate UK data-target parity concern(s): {duplicates}.")
    for row in rows:
        status = str(row.get("status", ""))
        if status not in VALID_PARITY_STATUSES:
            raise ValueError(
                f"parity concern {row['concern_id']!r} has invalid status {status!r}."
            )
        for key in ("classification", "evidence"):
            if not str(row.get(key, "")):
                raise ValueError(
                    f"parity concern {row['concern_id']!r} requires {key}."
                )
        if status not in PORTED_PARITY_STATUSES:
            if not str(row.get("reason", "")):
                raise ValueError(
                    f"non-ported parity concern {row['concern_id']!r} requires reason."
                )
            fence = row.get("fence")
            if not isinstance(fence, dict) or any(
                not str(fence.get(key, ""))
                for key in ("origin", "purpose", "verdict_basis")
            ):
                raise ValueError(
                    f"non-ported parity concern {row['concern_id']!r} requires a complete fence."
                )


def committed_uk_data_target_parity_path() -> Path:
    return Path(
        str(files("microcosm.build.uk").joinpath(UK_DATA_TARGET_PARITY_RESOURCE))
    )


def write_uk_data_target_parity(path: str | Path | None = None) -> Path:
    output = committed_uk_data_target_parity_path() if path is None else Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_uk_data_target_parity(), indent=2, sort_keys=True) + "\n"
    )
    return output


def assert_uk_data_target_parity_current(path: str | Path | None = None) -> None:
    source = committed_uk_data_target_parity_path() if path is None else Path(path)
    committed = json.loads(source.read_text())
    live = build_uk_data_target_parity()
    if json.dumps(committed, sort_keys=True) != json.dumps(live, sort_keys=True):
        raise ValueError(
            "UK data-target parity register is stale; regenerate with "
            "`uv run --no-sync python tools/census_uk_data_target_parity.py`."
        )
