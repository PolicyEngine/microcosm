#!/usr/bin/env python
"""Generate UK local-area Ledger target references from the local contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.build.target_reference_authoring import (
    AreaSignedDeferral,
    AreaTargetReferenceAuthoringConfig,
    author_area_target_references,
    target_references_resource,
)

DESCRIPTION = (
    "UK local-area Ledger target references for constituency and local-authority "
    "calibration. Rows are generated from the local rows in uk_population_targets.json and "
    "local_area_crosswalk.json: name is target_id@geography_id, ledger_selector "
    "is the contract selector plus geography_level/geography_id pins, entity "
    "and measure come from the policyengine binding, and observed values stay "
    "in Ledger facts. Deferred area absences are recorded in the membership "
    "report."
)
LOCAL_GEOGRAPHY_LEVELS = frozenset({"constituency", "local_authority"})

POLICYENGINE_BINDING_KEYS = frozenset(
    {
        "filters",
        "from_entity",
        "map_to",
        "metric_name",
        "value_expression",
        "value_variable",
    }
)
UK_PACKAGE_ROOT = Path("packages/microcosm-build/src/microcosm/build/uk")


def main() -> None:
    args = _parser().parse_args()
    contract = _filter_contract_by_geography_levels(
        json.loads(args.contract.read_text(encoding="utf-8")),
        allowed_levels=LOCAL_GEOGRAPHY_LEVELS,
    )
    crosswalk = json.loads(args.crosswalk.read_text(encoding="utf-8"))
    config = AreaTargetReferenceAuthoringConfig(
        target_period=args.period,
        areas_by_geography_level=_areas_by_geography_level(crosswalk),
        area_signed_deferrals=tuple(_area_signed_deferrals(contract, crosswalk)),
        value_operation_by_target_id=_value_operation_by_target_id(contract),
        reference_metadata_by_target_id={
            "ons.rent.private_rent": {
                "fact_aggregation": "time_mean",
                "period_basis_note": (
                    "Calendar-year average of the available 2025 PIPR monthly "
                    "area price levels; approved by María on 2026-09-05."
                ),
            }
        },
        binding_vocabulary=POLICYENGINE_BINDING_KEYS,
        source_fact_feed=args.source_fact_feed or str(args.ledger_facts),
    )
    authored = author_area_target_references(
        contract,
        _read_jsonl(args.ledger_facts),
        config,
    )
    resource = target_references_resource(
        country="uk",
        description=DESCRIPTION,
        authored=authored,
    )
    args.output.write_text(json.dumps(resource, indent=2) + "\n", encoding="utf-8")
    args.membership_report.write_text(
        json.dumps(authored.membership_report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(authored.membership_report["status_counts"], sort_keys=True))
    print(
        f"active_reference_count={authored.membership_report['active_reference_count']}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path, default=UK_PACKAGE_ROOT / "uk_population_targets.json"
    )
    # The pinned consumer feed is licensed and untracked: it must be named
    # explicitly so a regeneration never silently binds to a stale local copy.
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument(
        "--source-fact-feed",
        help="Stable display name recorded in the generated membership report.",
    )
    parser.add_argument(
        "--crosswalk", type=Path, default=UK_PACKAGE_ROOT / "local_area_crosswalk.json"
    )
    parser.add_argument("--period", type=int, default=2025)
    parser.add_argument(
        "--output", type=Path, default=UK_PACKAGE_ROOT / "local_target_references.json"
    )
    parser.add_argument(
        "--membership-report",
        type=Path,
        default=UK_PACKAGE_ROOT / "local_target_reference_membership.json",
    )
    return parser


def _filter_contract_by_geography_levels(
    contract: Mapping[str, Any],
    *,
    allowed_levels: frozenset[str],
) -> dict[str, Any]:
    filtered = dict(contract)
    filtered["targets"] = [
        target
        for target in contract.get("targets", ())
        if set(target.get("geography_levels") or ()) & allowed_levels
    ]
    return filtered


def _areas_by_geography_level(
    crosswalk: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    levels = crosswalk.get("levels")
    if not isinstance(levels, Mapping):
        raise ValueError("local area crosswalk must expose a levels object.")
    result: dict[str, tuple[str, ...]] = {}
    for level, payload in levels.items():
        if not isinstance(payload, Mapping):
            raise ValueError(f"local area crosswalk level {level!r} must be an object.")
        area_ids = payload.get("area_ids")
        if not isinstance(area_ids, list) or not area_ids:
            raise ValueError(
                f"local area crosswalk level {level!r} must expose area_ids."
            )
        result[str(level)] = tuple(str(area_id) for area_id in area_ids)
    return result


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _value_operation_by_target_id(contract: Mapping[str, Any]) -> dict[str, str]:
    operations: dict[str, str] = {}
    for target in contract.get("targets", ()):
        target_id = str(target["target_id"])
        if target_id.startswith("ons.age."):
            operations[target_id] = "sum"
        if target_id in {
            "hmrc.self_employment_income.amount",
            "hmrc.employment_income.amount",
        }:
            operations[target_id] = "count_x_mean"
        if target_id == "dwp.uc.households_by_area_children_3plus":
            operations[target_id] = "sum"
        if target_id in {
            "ons.tenure.private_rent",
            "ons.tenure.social_rent",
        }:
            operations[target_id] = "sum"
        if target_id == "ons.rent.private_rent":
            operations[target_id] = "calendar_year_average"
    return operations


def _area_signed_deferrals(
    contract: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
) -> list[AreaSignedDeferral]:
    target_ids = {str(target["target_id"]) for target in contract.get("targets", ())}
    areas = _areas_by_geography_level(crosswalk)
    constituency_ids = areas.get("constituency", ())
    local_authority_ids = areas.get("local_authority", ())
    ni_constituencies = tuple(
        area_id for area_id in constituency_ids if area_id[:1] == "N"
    )
    ni_local_authorities = tuple(
        area_id for area_id in local_authority_ids if area_id[:1] == "N"
    )
    scottish_local_authorities = tuple(
        area_id for area_id in local_authority_ids if area_id[:1] == "S"
    )
    if len(scottish_local_authorities) != 32:
        raise ValueError(
            "Scottish council-tax deferral mask expected 32 crosswalk local "
            f"authorities; measured {len(scottish_local_authorities)}."
        )
    if len(ni_local_authorities) != 11:
        raise ValueError(
            "Northern Ireland council-tax deferral mask expected 11 crosswalk "
            f"local authorities; measured {len(ni_local_authorities)}."
        )
    council_tax_ni_area_ids = ni_local_authorities
    council_tax_scotland_area_ids = scottish_local_authorities
    council_tax_city_band_a_area_ids = ("E09000001",)
    council_tax_wales_band_h_area_ids = ("W06000019", "W06000024")
    pipr_lad_absent_area_ids = (
        "E06000053",
        "E08000016",
        "E08000019",
        "E09000001",
    )
    spi_la_measure_gap_area_ids = ("E06000027", "E06000053")
    deferrals: list[AreaSignedDeferral] = []

    def add(
        *,
        target_id: str,
        geography_level: str,
        reason_id: str,
        rationale: str,
        area_ids: tuple[str, ...],
        allow_empty: bool = False,
    ) -> None:
        if target_id not in target_ids:
            raise ValueError(
                f"signed deferral target_id {target_id!r} is absent from the "
                "local contract."
            )
        roster = set(areas.get(geography_level, ()))
        matched_area_ids = tuple(area_id for area_id in area_ids if area_id in roster)
        if not matched_area_ids:
            if allow_empty:
                return
            unmatched_area_ids = sorted(
                area_id for area_id in area_ids if area_id not in roster
            )
            raise ValueError(
                f"signed deferral {reason_id!r} declared area ids match no "
                f"{geography_level} crosswalk area; unmatched area id(s): "
                f"{unmatched_area_ids}."
            )
        deferrals.append(
            AreaSignedDeferral(
                target_id=target_id,
                geography_level=geography_level,
                reason_id=reason_id,
                rationale=rationale,
                area_ids=matched_area_ids,
            )
        )

    add(
        target_id="dwp.uc.households_by_area",
        geography_level="constituency",
        reason_id="uc_gb_only_ni_absent",
        rationale="DWP Stat-Xplore Universal Credit local-area facts in the pinned feed cover Great Britain only: 632/650 PCON24 constituencies compile and the 18 Northern Ireland constituencies have no UC household facts.",
        area_ids=ni_constituencies,
    )
    add(
        target_id="dwp.uc.households_by_area",
        geography_level="local_authority",
        reason_id="uc_gb_only_ni_absent",
        rationale="DWP Stat-Xplore Universal Credit local-authority facts in the pinned feed cover Great Britain only: 350/361 local authorities compile and the 11 Northern Ireland local authorities have no UC household facts.",
        area_ids=ni_local_authorities,
    )
    for target_id in (
        "dwp.uc.households_by_area_children_0",
        "dwp.uc.households_by_area_children_1",
        "dwp.uc.households_by_area_children_2",
        "dwp.uc.households_by_area_children_3plus",
    ):
        add(
            target_id=target_id,
            geography_level="constituency",
            reason_id="uc_children_gb_only_ni_absent",
            rationale="DWP Stat-Xplore Universal Credit child-bucket facts in the pinned feed cover Great Britain constituencies only: 632/650 compile and the 18 Northern Ireland constituencies have no UC child-bucket household facts.",
            area_ids=ni_constituencies,
        )
    for target_id in (
        "hmrc.self_employment_income.amount",
        "hmrc.self_employment_income.count",
        "hmrc.employment_income.amount",
        "hmrc.employment_income.count",
    ):
        add(
            target_id=target_id,
            geography_level="local_authority",
            reason_id="spi_la_target_measure_coverage_absent",
            rationale="HMRC SPI local-authority facts in the pinned feed publish the target count/mean measures for 359/361 crosswalk local authorities; E06000027 has only median SPI measures and E06000053 has no SPI local-authority target-measure rows.",
            area_ids=spi_la_measure_gap_area_ids,
        )
    add(
        target_id="hmrc.self_employment_income.amount",
        geography_level="constituency",
        reason_id="spi_pcon_self_employment_mean_absent",
        rationale="HMRC SPI constituency facts in the pinned feed publish self_employment_income_count for 650/650 constituencies but self_employment_income_mean for 649/650; E14001416 has the count fact but no mean fact, so count_x_mean cannot form the amount.",
        area_ids=("E14001416",),
    )
    for target_id in (
        "ons.equiv_net_income_bhc",
        "ons.equiv_net_income_ahc",
        "ons.equiv_housing_costs",
    ):
        add(
            target_id=target_id,
            geography_level="local_authority",
            reason_id="msoa_mean_to_la_deferred",
            rationale="ONS equivalised-income facts in the pinned feed are MSOA-grain mean-valued targets, with no local-authority rows; local-authority aggregation is deferred pending the signed mean-aggregation design.",
            area_ids=local_authority_ids,
        )
    for band in "abcdefgh":
        target_id = f"voa.council_tax_stock.by_area.band_{band}"
        add(
            target_id=target_id,
            geography_level="local_authority",
            reason_id="council_tax_voa_scotland_absent",
            rationale=(
                "The pinned 2025 VOA local-authority council-tax stock feed "
                "contains no Scottish band-count rows; all 32 Scottish "
                "crosswalk authorities are absent for bands A-H. Scottish "
                "Band D equivalents and rates are different measures and "
                "cannot fill these cells."
            ),
            area_ids=council_tax_scotland_area_ids,
        )
        add(
            target_id=target_id,
            geography_level="local_authority",
            reason_id="council_tax_ni_domestic_rates",
            rationale=(
                "Northern Ireland uses domestic rates rather than council "
                "tax; the pinned feed has no council-tax band-count rows for "
                "the 11 Northern Ireland local-government districts."
            ),
            area_ids=council_tax_ni_area_ids,
        )
    add(
        target_id="voa.council_tax_stock.by_area.band_a",
        geography_level="local_authority",
        reason_id="council_tax_city_of_london_band_a_suppressed",
        rationale=(
            "The pinned 2025 VOA local-authority record set publishes bands "
            "B-H and all-properties for E09000001 but suppresses its Band A "
            "cell; 317/318 England/Wales Band A cells compile."
        ),
        area_ids=council_tax_city_band_a_area_ids,
    )
    add(
        target_id="voa.council_tax_stock.by_area.band_h",
        geography_level="local_authority",
        reason_id="council_tax_wales_band_h_absent",
        rationale=(
            "The pinned 2025 VOA local-authority record set publishes no "
            "Band H cell for W06000019 or W06000024; 316/318 England/Wales "
            "Band H cells compile."
        ),
        area_ids=council_tax_wales_band_h_area_ids,
    )
    add(
        target_id="ons.rent.private_rent",
        geography_level="local_authority",
        reason_id="private_rent_pipr_english_lad_absent",
        rationale="The pinned PIPR monthly series carries 294 English LA ids, of which 292 overlap the crosswalk and two (E08000038, E08000039) do not. It omits four English crosswalk authorities: E06000053, E08000016, E08000019, and E09000001.",
        area_ids=pipr_lad_absent_area_ids,
    )
    add(
        target_id="ons.rent.private_rent",
        geography_level="local_authority",
        reason_id="private_rent_pipr_scotland_brma_grain",
        rationale="The pinned PIPR monthly series carries 18 Scottish BRMA rows at statistical_scope grain and no Scottish LA rows for the 32-authority crosswalk. CrossGrainBridge declares target identity but cannot translate overlapping BRMA geographies to LAs, and no signed BRMA-to-LA crosswalk is present, so allocation is forbidden and all 32 cells remain deferred.",
        area_ids=scottish_local_authorities,
    )
    add(
        target_id="ons.rent.private_rent",
        geography_level="local_authority",
        reason_id="private_rent_pipr_ni_absent",
        rationale="The pinned PIPR monthly series contains zero Northern Ireland rows at any geography level, so all 11 Northern Ireland local-authority cells remain signed absent.",
        area_ids=ni_local_authorities,
    )
    return deferrals


if __name__ == "__main__":
    main()
