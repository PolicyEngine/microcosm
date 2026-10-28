#!/usr/bin/env python
"""Generate UK local-area Ledger target references from the local contract."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import replace
from importlib import resources as importlib_resources
from pathlib import Path
from typing import Any

from microcosm.build.target_reference_authoring import (
    AreaSignedDeferral,
    AreaTargetReferenceAuthoringConfig,
    author_area_target_references,
    target_references_resource,
)
from microcosm.build.uk_runtime.desnz_constituency_energy import (
    FACTS_RESOURCE as DESNZ_CONSTITUENCY_ENERGY_RESOURCE,
)
from microcosm.build.uk_runtime.desnz_constituency_energy import (
    load_uk_desnz_constituency_energy_facts,
)
from microcosm.build.uk_runtime.high_value_homes import (
    load_uk_high_value_home_stock_facts,
)
from microcosm.build.uk_runtime.ledger_targets import (
    UK_UPRATING_APPLIERS,
    uk_small_cell_rule_is_signed,
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
#: ``statistical_scope`` is Scotland's BRMA grain, where PIPR publishes its
#: Scottish rents (microcosm#1090, #1262).
LOCAL_GEOGRAPHY_LEVELS = frozenset(
    {"constituency", "local_authority", "statistical_scope"}
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
UK_PACKAGE_ROOT = Path("packages/microcosm-build/src/microcosm/build/uk")
# The local authorities whose own local-authority cells are signed-deferred
# because they miss the support floor at every K measured: the Isles of Scilly
# and the City of London (microcosm#762 A4, as amended by A14). These are
# solve-scoped and live here, not in uk/local_area_support_exclusions.json,
# which only the area-support gate reads since microcosm#1301, so signing an
# area gate-only never changes the compiled problem. Lockstep tests hold them
# inside that register's local_authority/* exclusion keys.
SUPPORT_FLOOR_DEFERRED_AREA_IDS: tuple[str, ...] = ("E06000053", "E09000001")
# The families that bind in those areas despite the floor (microcosm#887),
# lockstep-tested equal to the register's bound_despite_support_floor block.
SUPPORT_FLOOR_BOUND_FAMILIES: frozenset[str] = frozenset({"census_households"})
# The high-value homes area targets whose constituency level is deferred whole
# (microcosm#1301); their region rows bind in target_references.json instead.
HIGH_VALUE_HOMES_LOCAL_TARGET_IDS: tuple[str, ...] = (
    "microcosm.high_value_homes.over_1m",
    "microcosm.high_value_homes.over_1_5m",
    "microcosm.high_value_homes.over_2m",
)


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
            target_id: {
                "fact_aggregation": "time_mean",
                "period_basis_note": (
                    "Calendar-year average of the available 2025 PIPR monthly "
                    "area price levels; approved by María on 2026-09-05."
                ),
            }
            for target_id in ("ons.rent.private_rent", "ons.rent.private_rent_brma")
        },
        binding_vocabulary=POLICYENGINE_BINDING_KEYS,
        uprating_appliers=UK_UPRATING_APPLIERS,
        # Keep the explicit display-name option and default to an artifact-relative
        # name, never the operator's absolute local path.
        source_fact_feed=args.source_fact_feed
        or f"{args.ledger_facts.resolve().parent.name}/{args.ledger_facts.name}",
    )
    facts = list(_read_jsonl(args.ledger_facts))
    derived_resources: list[dict[str, Any]] = []
    if not args.no_derived_facts:
        # microcosm#1240: the high-value homes family selects Microcosm-derived
        # facts (HM Land Registry stock counts) carried in the committed resource;
        # microcosm#1095: so do the domestic energy family's constituency cells
        # (DESNZ's authority totals split by its small-area figures).
        for resource, derived in (
            (
                "high_value_home_stock_facts.json",
                load_uk_high_value_home_stock_facts(args.derived_facts),
            ),
            (
                DESNZ_CONSTITUENCY_ENERGY_RESOURCE,
                load_uk_desnz_constituency_energy_facts(),
            ),
        ):
            facts.extend(dict(row) for row in derived.rows)
            derived_resources.append(
                {
                    "resource": resource,
                    "rows": len(derived.rows),
                    "rows_sha256": derived.rows_sha256,
                    "inputs_sha256": derived.inputs_sha256,
                }
            )
    authored = author_area_target_references(contract, facts, config)
    # microcosm#1123 item 7: a second pass defers, by rule, the cells the
    # first pass found below the small-cell floor.
    small_cells = _small_cell_deferrals(authored.membership_report)
    if small_cells:
        config = replace(
            config,
            area_signed_deferrals=(*config.area_signed_deferrals, *small_cells),
        )
        authored = author_area_target_references(contract, facts, config)
    authored.membership_report["derived_fact_resources"] = derived_resources
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


def _small_cell_deferrals(
    membership_report: Mapping[str, Any],
    rule_path: Path | None = None,
) -> tuple[AreaSignedDeferral, ...]:
    """Deferrals for every active cell a rule's floor puts out of reach.

    Read from the first authoring pass, so the rule names the cells by their
    published values, never by hand; each carries the rule's expiry.
    """

    register = json.loads(
        rule_path.read_text(encoding="utf-8")
        if rule_path is not None
        else importlib_resources.files("microcosm.build.uk")
        .joinpath("local_small_cell_rule.json")
        .read_text(encoding="utf-8")
    )
    if register.get("schema_version") != 1:
        raise ValueError("UK local small-cell rule schema_version must be 1.")
    deferrals: list[AreaSignedDeferral] = []
    for rule in register["rules"]:
        if not uk_small_cell_rule_is_signed(rule):
            # A proposed rule defers nothing: its cells stay bound until the
            # floor is signed (approved_by, approved_on, no longer proposed).
            continue
        floor = float(rule["absolute_floor"])
        prefixes = tuple(str(prefix) for prefix in rule["target_id_prefixes"])
        for target_id, target in sorted(membership_report["targets"].items()):
            if not target_id.startswith(prefixes):
                continue
            for level, payload in sorted(target["geography_levels"].items()):
                areas = sorted(
                    str(candidate["geography_id"])
                    for candidate in payload["candidates"]
                    if candidate.get("status") == "active"
                    and abs(float(candidate["resolved_value"])) < floor
                )
                if not areas:
                    continue
                deferrals.append(
                    AreaSignedDeferral(
                        target_id=target_id,
                        geography_level=level,
                        reason_id=str(rule["rule_id"]),
                        rationale=(
                            f"{rule['reason']} Rule {rule['rule_id']} in "
                            "uk/local_small_cell_rule.json: published value "
                            f"below {floor:g}; approved by "
                            f"{rule['approved_by']} on {rule['approved_on']}."
                        ),
                        area_ids=tuple(areas),
                        defer_if_compiles=True,
                        expires_on=rule.get("expires_on"),
                    )
                )
    return tuple(deferrals)


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
        "--derived-facts",
        type=Path,
        default=None,
        help=(
            "Microcosm-derived facts resource merged beside the feed (default: the "
            "committed uk/high_value_home_stock_facts.json)."
        ),
    )
    parser.add_argument(
        "--no-derived-facts",
        action="store_true",
        help="Diagnostics only: author the surface from the Chronicle feed alone.",
    )
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
            "hmrc.pension_income.amount",
        }:
            operations[target_id] = "count_x_mean"
        if target_id == "dwp.uc.households_by_area_children_3plus" and declared is None:
            # The May-snapshot spelling summed the three published child
            # counts; the calendar-2025 window declares its own monthly sum
            # (microcosm#1123).
            operations[target_id] = "sum"
        if target_id in {
            "ons.tenure.private_rent",
            "ons.tenure.social_rent",
        }:
            operations[target_id] = "sum"
        if target_id in {"ons.rent.private_rent", "ons.rent.private_rent_brma"}:
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
    england_and_wales_constituencies = tuple(
        area_id for area_id in constituency_ids if area_id[:1] in {"E", "W"}
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
    if len(england_and_wales_constituencies) != 575:
        raise ValueError(
            "England and Wales high-value homes constituency deferral mask "
            "expected 575 crosswalk constituencies; measured "
            f"{len(england_and_wales_constituencies)}."
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
        declared_value: float | None = None,
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
                declared_value=declared_value,
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
        "hmrc.higher_rate_taxpayers.count",
        "hmrc.pension_income.amount",
        "hmrc.pension_income.count",
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
            "Spine-m carries 170 band-H households from 49 raw FRS households. "
            "At the ruled K=15, 76 of the 296 authorities draw no band-H "
            "household and the median of the rest draws five (84 of 296 at "
            "K=10), so the family cannot bind at local-authority grain until a "
            "spine vintage carries broader band-H support (microcosm#762 A14; "
            "carried onto the MHCLG taxbase basis by microcosm#929)."
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
    # microcosm#1301: the high-value homes family is controlled by its region
    # rows on the local line and binds its local-authority cells down to the
    # signed hvcts_la_household_floor rule, so its constituency level is
    # deferred whole, as the English band-H level is. The rule's second pass
    # then finds no active constituency cell to defer.
    for target_id in HIGH_VALUE_HOMES_LOCAL_TARGET_IDS:
        add(
            target_id=target_id,
            geography_level="constituency",
            reason_id="high_value_homes_constituency_grain_deferred",
            rationale=(
                "The derived high-value homes stock binds at the nine English "
                "regions and Wales (microcosm.high_value_homes.over_*_by_region, "
                "local line only) and at local-authority grain down to the "
                "hvcts_la_household_floor rule (uk/local_small_cell_rule.json). "
                "At constituency grain most cells hold fewer than 400 homes "
                "(289 of the 575 seats over £1m, 448 over £1.5m, 512 over £2m), "
                "about two household weights at the ruled 150,000 households, so "
                "their fit would be noise. The level is deferred whole rather "
                "than cell by cell, and the region rows control the stock the "
                "seats share. The seat rank correlations of the "
                "uk_high_value_homes_benchmarks diagnostic read the "
                "constituency stock out of sample (microcosm#1301)."
            ),
            area_ids=england_and_wales_constituencies,
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
        rationale="PIPR publishes Scotland's rents for its 18 Broad Rental Market Areas only, and they bind at that grain (ons.rent.private_rent_brma, microcosm#1090, #1262). The BRMAs overlap the councils (uk/scotland_brma_la_overlap.json: 63 intersections, 18 councils in more than one BRMA), so a council has no published rent of its own, and a household-weighted mean of BRMA rents would give every council inside a BRMA that BRMA's average. All 32 council cells remain deferred.",
        area_ids=scottish_local_authorities,
    )
    add(
        target_id="ons.rent.private_rent",
        geography_level="local_authority",
        reason_id="private_rent_pipr_ni_absent",
        rationale="PIPR publishes Northern Ireland only as a country mean (every month of 2025, beside the UK, Great Britain, nation and region means), with no district rows, so all 11 Northern Ireland local-authority cells remain signed absent. A nation or region mean is no higher control for the district cells: a mean is not additive, and the rents' level comes from the FRS (microcosm#1262).",
        area_ids=ni_local_authorities,
    )
    # microcosm#1207: Orkney and Shetland have no mains gas network, so DESNZ
    # publishes no domestic gas for them; neither cell is bound at zero.
    gas_network_absent_area_ids = ("S12000023", "S12000027")
    add(
        target_id="desnz.subnational.gas_kwh",
        geography_level="local_authority",
        reason_id="desnz_gas_no_mains_network",
        rationale=(
            "DESNZ's sub-national gas statistics carry no domestic consumption "
            "for Orkney (S12000023) or Shetland (S12000027), which have no "
            "mains gas network; the cells are signed absent rather than bound "
            "at zero (never_fabricate; microcosm#1207). Reconciliation counts "
            "them at a declared zero, the gas they hold, so their region's "
            "other cells keep their own share (María's ruling of 2026-10-25, "
            "microcosm#1095)."
        ),
        area_ids=gas_network_absent_area_ids,
        declared_value=0.0,
    )
    add(
        target_id="desnz.subnational.gas_connected_households",
        geography_level="local_authority",
        reason_id="desnz_gas_no_mains_network",
        rationale=(
            "Orkney (S12000023) and Shetland (S12000027) have no mains gas "
            "network and DESNZ publishes no gas meters for them, so no meter "
            "share puts their census households on the gas network. The "
            "census base compiles there, so the cells are deferred by this "
            "adjudication (defer_if_compiles) rather than bound at zero "
            "(microcosm#1207). Reconciliation counts them at a declared zero "
            "gas-connected households, not their census base (María's ruling "
            "of 2026-10-25, microcosm#1095)."
        ),
        area_ids=gas_network_absent_area_ids,
        defer_if_compiles=True,
        declared_value=0.0,
    )
    # microcosm#1095: the Orkney and Shetland seat is the two councils, so it
    # has no mains gas network either and DESNZ's split gives it no gas row.
    gas_network_absent_seat_ids = ("S14000051",)
    add(
        target_id="desnz.subnational.gas_kwh",
        geography_level="constituency",
        reason_id="desnz_gas_no_mains_network",
        rationale=(
            "Orkney and Shetland (S14000051) is the two councils DESNZ publishes "
            "no domestic gas for, which have no mains gas network, so the split "
            "of the authorities' gas onto constituencies gives the seat no row; "
            "the cell is signed absent rather than bound at zero "
            "(never_fabricate; microcosm#1095). Reconciliation counts it at a "
            "declared zero, as the councils' cells, so the two grains agree "
            "within Scotland (María's ruling of 2026-10-25)."
        ),
        area_ids=gas_network_absent_seat_ids,
        declared_value=0.0,
    )
    add(
        target_id="desnz.subnational.gas_connected_households",
        geography_level="constituency",
        reason_id="desnz_gas_no_mains_network",
        rationale=(
            "Orkney and Shetland (S14000051) has no mains gas network and no "
            "gas meters in DESNZ's split, so no meter share puts its census "
            "households on the gas network. The census base compiles there, so "
            "the cell is deferred by this adjudication (defer_if_compiles) "
            "rather than bound at zero (microcosm#1095). Reconciliation counts "
            "it at a declared zero gas-connected households, not its census "
            "base (María's ruling of 2026-10-25)."
        ),
        area_ids=gas_network_absent_seat_ids,
        defer_if_compiles=True,
        declared_value=0.0,
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
                "exclusions.json, microcosm#762 A4): each misses the floor at "
                "every K measured. Their own "
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
    """Return the support-floor deferral masks: the areas and bound families.

    Both are this module's constants, never read from the gate-only
    local-area support register (microcosm#1301).
    """

    return SUPPORT_FLOOR_DEFERRED_AREA_IDS, SUPPORT_FLOOR_BOUND_FAMILIES


if __name__ == "__main__":
    main()
