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
from microcosm.build.uk_runtime.weighted_integrity import (
    load_uk_local_area_support_exclusion_register,
)

DESCRIPTION = (
    "UK local-area Ledger target references for constituency and local-authority "
    "calibration. Rows are generated from the Chronicle-backed local rows in "
    "uk_population_targets.json and local_area_crosswalk.json: name is "
    "target_id@geography_id, ledger_selector is the contract selector plus "
    "geography_level/geography_id pins, entity and measure come from the "
    "policyengine binding, and observed values stay in Ledger facts. Targets "
    "without matching Chronicle facts fail unless their exact area absences are "
    "explicitly reviewed. Deferred area absences are recorded in the membership "
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
        area_scope_by_target_id=_area_scope_by_target_id(contract, crosswalk),
        area_id_aliases=_area_id_aliases(crosswalk),
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
        # Keep the explicit display-name option and default to an artifact-relative
        # name, never the operator's absolute local path.
        source_fact_feed=args.source_fact_feed
        or f"{args.ledger_facts.resolve().parent.name}/{args.ledger_facts.name}",
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
        hierarchy=contract["hierarchy"],
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


def _area_id_aliases(
    crosswalk: Mapping[str, Any],
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Publisher recodings declared on the crosswalk (``code_aliases``)."""

    aliases: dict[str, dict[str, tuple[str, ...]]] = {}
    for level, payload in (crosswalk.get("levels") or {}).items():
        declared = payload.get("code_aliases") or {}
        for area_id, alias in declared.items():
            codes = tuple(str(code) for code in alias.get("alias_codes", ()))
            if codes:
                aliases.setdefault(str(level), {})[str(area_id)] = codes
    return aliases


def _area_scope_by_target_id(
    contract: Mapping[str, Any],
    crosswalk: Mapping[str, Any],
) -> dict[str, dict[str, frozenset[str]]]:
    """Nation-scoped rosters for targets whose publication covers one nation.

    ``area_scope: {"<level>": {"gss_prefixes": ["E"]}}`` on a contract target
    keeps the target's cells to the roster areas whose GSS code starts with
    one of the prefixes (the MHCLG, StatsWales and CTAXBASE council-tax
    families each cover one nation, microcosm#929). The other nations' areas
    are not candidates of that target at all, so they are neither absences to
    sign nor cells to compile.
    """

    areas = _areas_by_geography_level(crosswalk)
    scopes: dict[str, dict[str, frozenset[str]]] = {}
    for target in contract.get("targets", ()):
        declaration = target.get("area_scope")
        if not declaration:
            continue
        target_id = str(target["target_id"])
        for level, rule in declaration.items():
            prefixes = tuple(str(prefix) for prefix in rule.get("gss_prefixes", ()))
            if not prefixes:
                raise ValueError(
                    f"{target_id}: area_scope for {level!r} declares no gss_prefixes."
                )
            roster = areas.get(str(level))
            if roster is None:
                raise ValueError(
                    f"{target_id}: area_scope names level {level!r}, which the "
                    "crosswalk does not carry."
                )
            scopes.setdefault(target_id, {})[str(level)] = frozenset(
                area_id for area_id in roster if area_id.startswith(prefixes)
            )
    return scopes


def _value_operation_by_target_id(contract: Mapping[str, Any]) -> dict[str, str]:
    operations: dict[str, str] = {}
    for target in contract.get("targets", ()):
        target_id = str(target["target_id"])
        declared = target.get("value_operation")
        if declared is not None:
            operations[target_id] = str(declared)
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
    english_local_authorities = tuple(
        area_id for area_id in local_authority_ids if area_id[:1] == "E"
    )
    welsh_local_authorities = tuple(
        area_id for area_id in local_authority_ids if area_id[:1] == "W"
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
    if len(welsh_local_authorities) != 22:
        raise ValueError(
            "Welsh council-tax deferral mask expected 22 crosswalk local "
            f"authorities; measured {len(welsh_local_authorities)}."
        )
    if len(english_local_authorities) != 296:
        raise ValueError(
            "English council-tax Band H deferral mask expected 296 crosswalk "
            f"local authorities; measured {len(english_local_authorities)}."
        )
    support_floor_excluded_area_ids, support_floor_binding_families = (
        _support_floor_register_scope()
    )
    # Barnsley (E08000016) and Sheffield (E08000019) left this mask on
    # microcosm#929: PIPR files them under the April 2025 codes E08000038 and
    # E08000039, which the crosswalk now declares as aliases of the roster
    # codes, so their cells select the published rows.
    pipr_lad_absent_area_ids = (
        "E06000053",
        "E09000001",
    )
    spi_la_measure_gap_area_ids = ("E06000027", "E06000053")
    deferrals: list[AreaSignedDeferral] = []
    declared_area_keys: set[tuple[str, str, str]] = set()

    def add(
        *,
        target_id: str,
        geography_level: str,
        reason_id: str,
        rationale: str,
        area_ids: tuple[str, ...],
        allow_empty: bool = False,
        defer_if_compiles: bool = False,
        skip_declared: bool = False,
    ) -> None:
        if target_id not in target_ids:
            raise ValueError(
                f"signed deferral target_id {target_id!r} is absent from the "
                "local contract."
            )
        roster = set(areas.get(geography_level, ()))
        matched_area_ids = tuple(
            area_id
            for area_id in area_ids
            if area_id in roster
            and (
                not skip_declared
                or (target_id, geography_level, area_id) not in declared_area_keys
            )
        )
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
                defer_if_compiles=defer_if_compiles,
            )
        )
        declared_area_keys.update(
            (target_id, geography_level, area_id) for area_id in matched_area_ids
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
    add(
        target_id="mhclg.council_tax_stock.by_area.band_h",
        geography_level="local_authority",
        reason_id="council_tax_band_h_spine_support_absent",
        rationale=(
            "Spine-m carries 170 band-H households from 49 raw FRS households "
            "(London 13, Wales 11, South East 8, Scotland 6, West Midlands 4, "
            "South West 3, East of England 3, East Midlands 1). At the ruled "
            "K=15, 76 of the 296 authorities draw no band-H household and the "
            "median of the rest draws five (84 of 296 at K=10), so the family "
            "cannot bind at local-authority grain until a spine vintage "
            "carries broader band-H support (microcosm#762 A14; carried onto "
            "the MHCLG taxbase basis by microcosm#929)."
        ),
        area_ids=english_local_authorities,
        defer_if_compiles=True,
    )
    add(
        target_id="scotgov.council_tax_stock.by_area.band_h",
        geography_level="local_authority",
        reason_id="council_tax_band_h_spine_support_absent",
        rationale=(
            "Scotland's six raw FRS band-H households leave Shetland "
            "(S12000027) with no band-H household at the ruled K=15: the "
            "rowwise driver's support check refuses the cell (nonzero target, "
            "zero household support) while the other 31 council cells draw at "
            "least one band-H clone and bind (microcosm#929; the English "
            "authorities stay under microcosm#762 A14)."
        ),
        area_ids=("S12000027",),
        defer_if_compiles=True,
    )
    add(
        target_id="ons.rent.private_rent",
        geography_level="local_authority",
        reason_id="private_rent_pipr_english_lad_absent",
        rationale="The pinned PIPR monthly series carries 294 English LA ids: 292 overlap the crosswalk directly and two, E08000038 and E08000039, are the April 2025 codes of Barnsley and Sheffield, which the crosswalk declares as aliases of E08000016 and E08000019 (microcosm#929), so those cells bind. It omits two English crosswalk authorities: E06000053 and E09000001.",
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
    for target in contract.get("targets", ()):
        if "local_authority" not in target.get("geography_levels", ()):
            continue
        if str(target.get("family")) in support_floor_binding_families:
            continue
        add(
            target_id=str(target["target_id"]),
            geography_level="local_authority",
            reason_id="local_authority_support_floor_excluded",
            rationale=(
                "E06000053 and E09000001 are reviewed exclusions of the "
                "local-authority support floor (uk/local_area_support_"
                "exclusions.json, microcosm#762 A4): respectively 7 and 14 "
                "positive-weight rows at K=4, and 17 and 49 at K=10. Their own "
                "local-authority cells are signed-deferred on the same "
                "evidence; their rows stay in the solve through the "
                "constituency families and the national rows (A14 amendment "
                "to A4)."
            ),
            area_ids=support_floor_excluded_area_ids,
            allow_empty=True,
            defer_if_compiles=True,
            skip_declared=True,
        )
    return deferrals


def _support_floor_register_scope() -> tuple[tuple[str, ...], frozenset[str]]:
    """Return generator masks from the signed local-area support register."""

    register = load_uk_local_area_support_exclusion_register(None)
    area_ids: list[str] = []
    for key in register["exclusions"]:
        level, separator, area_id = key.partition("/")
        if separator != "/" or level != "local_authority" or not area_id:
            raise ValueError(
                "local-area support exclusion keys used by the generator must "
                f"have local_authority/<area_id> shape, got {key!r}."
            )
        area_ids.append(area_id)
    return (
        tuple(sorted(area_ids)),
        frozenset(register["bound_despite_support_floor"]),
    )


if __name__ == "__main__":
    main()
