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
    "calibration. Rows are generated from uk_local_geography_targets.json and "
    "local_area_crosswalk.json: name is target_id@geography_id, ledger_selector "
    "is the contract selector plus geography_level/geography_id pins, entity "
    "and measure come from the policyengine binding, and observed values stay "
    "in Ledger facts. Deferred area absences are recorded in the membership "
    "report."
)

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


def main() -> None:
    args = _parser().parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    crosswalk = json.loads(args.crosswalk.read_text(encoding="utf-8"))
    facts = [
        json.loads(line)
        for line in args.ledger_facts.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = AreaTargetReferenceAuthoringConfig(
        target_period=args.period,
        areas_by_geography_level=_areas_by_geography_level(crosswalk),
        area_signed_deferrals=tuple(_area_signed_deferrals(contract, crosswalk)),
        value_operation_by_target_id=_value_operation_by_target_id(contract),
        binding_vocabulary=POLICYENGINE_BINDING_KEYS,
        source_fact_feed=str(args.ledger_facts),
    )
    authored = author_area_target_references(contract, facts, config)
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
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--period", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membership-report", type=Path, required=True)
    return parser


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
        if target_id == "dwp.universal_credit.households.3plus_children":
            operations[target_id] = "sum"
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
    non_english_local_authorities = tuple(
        area_id for area_id in local_authority_ids if area_id[:1] != "E"
    )
    deferrals: list[AreaSignedDeferral] = []

    def add(
        *,
        target_id: str,
        geography_level: str,
        reason_id: str,
        rationale: str,
        area_ids: tuple[str, ...],
    ) -> None:
        if target_id in target_ids and area_ids:
            deferrals.append(
                AreaSignedDeferral(
                    target_id=target_id,
                    geography_level=geography_level,
                    reason_id=reason_id,
                    rationale=rationale,
                    area_ids=area_ids,
                )
            )

    add(
        target_id="dwp.universal_credit.households",
        geography_level="constituency",
        reason_id="uc_gb_only_ni_absent",
        rationale="Universal Credit local-area facts are GB-only; Northern Ireland constituencies are signed absent.",
        area_ids=ni_constituencies,
    )
    add(
        target_id="dwp.universal_credit.households",
        geography_level="local_authority",
        reason_id="uc_gb_only_ni_absent",
        rationale="Universal Credit local-authority facts are GB-only; Northern Ireland local authorities are signed absent.",
        area_ids=ni_local_authorities,
    )
    for target_id in (
        "dwp.universal_credit.households.0_children",
        "dwp.universal_credit.households.1_child",
        "dwp.universal_credit.households.2_children",
        "dwp.universal_credit.households.3plus_children",
    ):
        add(
            target_id=target_id,
            geography_level="constituency",
            reason_id="uc_children_gb_only_ni_absent",
            rationale="Universal Credit child-bucket local facts are GB-only; Northern Ireland constituencies are signed absent.",
            area_ids=ni_constituencies,
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
            rationale="Equivalised-income facts are MSOA-grain mean-valued targets; local-authority aggregation is deferred pending the signed mean-aggregation design.",
            area_ids=local_authority_ids,
        )
    add(
        target_id="ons.rent.private_rent",
        geography_level="local_authority",
        reason_id="private_rent_non_english_or_masked_absent",
        rationale="Private-rent local facts are England LA rows only in the active feed; Scotland, Wales, and Northern Ireland local authorities are signed absent pending publisher-compatible area facts.",
        area_ids=non_english_local_authorities,
    )
    return deferrals


if __name__ == "__main__":
    main()
