#!/usr/bin/env python
"""Generate UK Ledger target references from the national contract.

This is an offline authoring tool. It consumes an already-exported Ledger
consumer fact JSONL feed; it does not fetch data or contact Chronicle.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from microcosm.build.ledger_targets import MONTHLY_WINDOW_OPERATIONS
from microcosm.build.target_reference_authoring import (
    TargetReferenceAuthoringConfig,
    author_target_references,
    target_references_resource,
)
from microcosm.build.uk_runtime.ledger_targets import UK_UPRATING_APPLIERS
from microcosm.build.uk_runtime.uc_source_periods import uc_source_month_metadata
from microcosm.calibrate.geography_constants import (
    UK_REGION_TIER,
    UK_REGION_TIER_ENUM,
)

UK_GEOGRAPHY_IDS = {
    "uk": "K02000001",
    "great_britain": "K03000001",
    "england": "E92000001",
    "scotland": "S92000003",
    "wales": "W92000004",
    "northern_ireland": "N92000002",
}

POLICYENGINE_BINDING_KEYS = frozenset(
    {
        "affected_flag_variable",
        "band",
        "band_filter_dimension",
        "band_period_factor",
        "band_upper_bound",
        "band_upper_bound_inclusive",
        "count_of",
        "filters",
        "folded_into",
        "from_entity",
        "gate_comparison",
        "gate_parameter",
        "gated_variable",
        "groupby_variable",
        "household_conditions",
        "kind",
        "map_to",
        "metric_name",
        "measurement_period",
        "notes",
        "output_delta",
        "output_variable",
        "reduce",
        "require_matching_fact_period",
        "source_lines",
        "threshold_price_base_year",
        "value_expression",
        "value_reduction",
        "value_variable",
        "zeroed_input",
    }
)

_UK_DATA_REPO = "policyengine-" + "uk-data"

DESCRIPTION = (
    "UK active-subset Ledger target references for the FRS 2024-25 line. "
    "Rows are generated from the national rows in uk_population_targets.json: "
    "name is the contract "
    "target_id or an incumbent-compatible fan-out row name; ledger_selector "
    "is the contract selector plus geography pins; entity is from_entity, "
    "then map_to, then household; measure is the prepared-column metric name "
    "or fan-out row name; family is the contract family; period is the model "
    "and calibration year 2025, distinct from the FRS 2024-25 base period 2024. "
    "Observation windows are declared separately; metadata.measurement_period "
    "records observed-year exceptions. Observed values stay in Ledger facts "
    "and resolve through each reference's declared value operation. Deferred "
    "classes and geography-pin decisions are recorded in "
    "uk/target_reference_membership.json. metadata.measure_kind records that "
    "measures are prepared columns produced from the contract binding payload "
    "referenced by metadata.contract_target_id. Provider and category ownership "
    "come from the normalized hierarchy in the target contract."
)
NATIONAL_GEOGRAPHY_LEVELS = frozenset({"country", "region"})


def main() -> None:
    args = _parser().parse_args()
    contract = _filter_contract_by_geography_levels(
        json.loads(args.contract.read_text()),
        allowed_levels=NATIONAL_GEOGRAPHY_LEVELS,
    )
    _validate_amount_entity_pins(contract)
    facts = [
        json.loads(line)
        for line in args.ledger_facts.read_text().splitlines()
        if line.strip()
    ]
    inverse_mapping = _registry_inverse(contract)
    config = TargetReferenceAuthoringConfig(
        target_period=args.period,
        geography_pins=_geography_pins(contract),
        geography_fanout_by_target_id=_geography_fanout(contract),
        geography_fanout_metadata=_geography_fanout_metadata,
        geography_composition_by_target_id=_geography_composition(contract),
        geography_composition_aliases=_geography_composition_aliases(),
        fanout_name=lambda target, fact: _fanout_name(
            target,
            fact,
            inverse_mapping,
        ),
        sum_target_ids=_sum_target_ids(contract),
        value_operation_by_target_id=_value_operation_by_target_id(contract),
        selector_pins_by_target_id=_selector_pins(contract),
        signed_exclusions_by_target_id=_signed_exclusions(contract),
        signed_row_exclusions_by_target_id=_signed_row_exclusions(contract),
        reference_metadata_by_target_id=_reference_metadata(contract),
        binding_vocabulary=POLICYENGINE_BINDING_KEYS,
        source_fact_feed=args.source_fact_feed or str(args.ledger_facts),
        uprating_appliers=UK_UPRATING_APPLIERS,
    )
    authored = author_target_references(contract, facts, config)
    _add_uk_membership_accounting(authored.membership_report, authored.references)
    resource = target_references_resource(
        country="uk",
        description=DESCRIPTION,
        authored=authored,
        hierarchy=contract["hierarchy"],
    )
    args.output.write_text(json.dumps(resource, indent=2) + "\n")
    args.membership_report.write_text(
        json.dumps(authored.membership_report, indent=2) + "\n"
    )
    print(json.dumps(authored.membership_report["status_counts"], sort_keys=True))
    print(
        f"active_reference_count={authored.membership_report['active_reference_count']}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--ledger-facts", type=Path, required=True)
    parser.add_argument(
        "--source-fact-feed",
        help="Stable display name recorded in the generated membership report.",
    )
    parser.add_argument("--period", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membership-report", type=Path, required=True)
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
        if set(target.get("geography_levels") or ()) <= allowed_levels
    ]
    return filtered


def _validate_amount_entity_pins(contract: Mapping[str, Any]) -> None:
    missing = [
        str(target["target_id"])
        for target in contract.get("targets", ())
        if str(target.get("measurement", {}).get("concept", "")).endswith(".amount")
        and not str(target.get("ledger_selector", {}).get("entity_name", ""))
    ]
    if missing:
        raise ValueError(
            "National amount targets must pin ledger_selector.entity_name: "
            f"{missing!r}."
        )


def _registry_inverse(contract: Mapping[str, Any]) -> dict[str, list[str]]:
    inverse: dict[str, list[str]] = {}
    for ancestor, target_id in contract["registry_parity"]["mapped"].items():
        inverse.setdefault(str(target_id), []).append(str(ancestor))
    return {key: sorted(value) for key, value in inverse.items()}


def _geography_pins(contract: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Pin every national contract target to a Ledger geography.

    A target whose only declared level is ``region`` (the DfT London bus
    rows) pins at region level with its E12 code, so it can match the
    region-stamped publisher fact instead of a country row that never
    exists. Every other national target pins at country level.
    """

    fanned_out = _geography_fanout(contract)
    pins: dict[str, dict[str, str]] = {}
    for target in contract.get("targets", ()):
        target_id = str(target["target_id"])
        if target_id in fanned_out:
            # A two-level target takes its geographies from the region tier
            # roster (microcosm#905); a single pin would collapse it back to
            # the country row the roster replaces.
            continue
        levels = tuple(str(level) for level in target.get("geography_levels") or ())
        level = "region" if levels == ("region",) else "country"
        pins[target_id] = {
            "geography_level": level,
            "geography_id": _geography_id_for_target(target),
        }
    return pins


#: The declared level set that fans out over the region tier instead of
#: pinning one country: a target that is published both nationally and by
#: region (ONS mid-year population by age band, VOA council-tax stock by band).
REGION_TIER_FANOUT_LEVELS = frozenset({"country", "region"})


def _geography_fanout(
    contract: Mapping[str, Any],
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Region-tier cells for every two-level (country + region) target.

    The nine English regions are always in; the three nations join only when
    the binding does not already pin a country itself (the VOA stock bindings
    filter ``country == ENGLAND`` because the Welsh, Scottish and Northern
    Irish stocks are separate publications and contract families), so a
    nation cell is never a row the binding would measure as zero.
    """

    fanout: dict[str, tuple[tuple[str, str], ...]] = {}
    for target in contract.get("targets", ()):
        levels = {str(level) for level in target.get("geography_levels") or ()}
        if levels != REGION_TIER_FANOUT_LEVELS:
            continue
        cells = [(level, code) for level, code in UK_REGION_TIER if level == "region"]
        if not _binding_pins_country(target["bindings"]["policyengine"]):
            cells.extend(
                (level, code) for level, code in UK_REGION_TIER if level == "country"
            )
        fanout[str(target["target_id"])] = tuple(cells)
    return fanout


def _binding_pins_country(binding: Mapping[str, Any]) -> bool:
    predicates = [
        *binding.get("filters", ()),
        *binding.get("household_conditions", ()),
    ]
    return any(
        str(predicate.get("variable") or predicate.get("concept") or "") == "country"
        for predicate in predicates
        if isinstance(predicate, Mapping)
    )


def _geography_fanout_metadata(
    target: Mapping[str, Any],
    geography_level: str,
    geography_id: str,
    entity: str,
) -> dict[str, str]:
    """Scope one region-tier row to its area on the spine.

    The predicate compares the household's ``region`` enum (FRS ``gvtregno``
    through ``REGION_MAP``) with the tier code's enum name and projects the
    household match to the binding's own entity, the way the incumbent's
    ``compute_regional_age`` masks persons by their household's region. The
    ``cross_grain_grain`` stamp places every tier row, the three nation rows
    included, at the ``region`` grain of the cross-grain rule; Chronicle's
    ``country`` stamp on those three stays on the selector and the ledger
    metadata untouched.
    """

    del target, geography_level
    predicate = {
        "entity": "household",
        "variable": "region",
        "operator": "==",
        "value": UK_REGION_TIER_ENUM[geography_id],
        "reduce": "any",
        "map_to": entity,
    }
    return {
        "geography_predicate": json.dumps(
            predicate, sort_keys=True, separators=(",", ":")
        ),
        "cross_grain_grain": "region",
    }


def _local_area_crosswalk() -> dict[str, Any]:
    return json.loads(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(
            "packages/microcosm-build/src/microcosm/build/uk/local_area_crosswalk.json"
        )
        .read_text()
    )


def _geography_composition_aliases() -> dict[str, dict[str, tuple[str, ...]]]:
    aliases: dict[str, dict[str, tuple[str, ...]]] = {}
    for level, payload in _local_area_crosswalk()["levels"].items():
        for area_id, alias in (payload.get("code_aliases") or {}).items():
            codes = tuple(str(code) for code in alias.get("alias_codes", ()))
            if codes:
                aliases.setdefault(str(level), {})[str(area_id)] = codes
    return aliases


def _geography_composition(
    contract: Mapping[str, Any],
) -> dict[str, dict[str, tuple[str, tuple[str, ...]]]]:
    """Region cells composed from the authorities the crosswalk places in them.

    A two-level target that declares ``region_composition`` binds no region
    fact of its own: the publisher prints the England row and the billing
    authorities (MHCLG Council Taxbase), so each English region cell is the
    signed combination of its authorities' rows, the consumer rollup Chronicle's
    doctrine leaves to the consumer (microcosm#929). Membership comes from the
    sha-pinned OA ladder through ``region_code_by_area`` in the local-area
    crosswalk, the same map the cross-grain legs read, so the cell's members
    are exactly the authorities the cross-grain rule reconciles beneath it.
    """

    composition: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {}
    crosswalk: Mapping[str, Any] | None = None
    for target in contract.get("targets", ()):
        declaration = target.get("region_composition")
        if declaration is None:
            continue
        levels = {str(level) for level in target.get("geography_levels") or ()}
        if levels != REGION_TIER_FANOUT_LEVELS:
            raise ValueError(
                f"{target['target_id']}: region_composition needs the two-level "
                "(country + region) declaration."
            )
        from_level = str(declaration.get("from_level") or "")
        if from_level != "local_authority":
            raise ValueError(
                f"{target['target_id']}: region_composition.from_level must be "
                f"'local_authority', got {from_level!r}."
            )
        if crosswalk is None:
            crosswalk = _local_area_crosswalk()
        level_payload = crosswalk["levels"][from_level]
        region_by_area = level_payload["region_code_by_area"]
        code_aliases = level_payload.get("code_aliases") or {}
        members: dict[str, list[str]] = {}
        for area_id, region_code in sorted(region_by_area.items()):
            if str(region_code).startswith("E12"):
                codes = members.setdefault(str(region_code), [])
                codes.append(str(area_id))
                # A recoded authority's facts carry the alias code; the member
                # count stays one per authority (the resolver takes the latest
                # period, where only one spelling has a row).
                codes.extend(
                    str(code)
                    for code in (code_aliases.get(str(area_id)) or {}).get(
                        "alias_codes", ()
                    )
                )
        cells = {
            code: (from_level, tuple(members[code]))
            for level, code in UK_REGION_TIER
            if level == "region"
        }
        missing = [code for code, (_, ids) in cells.items() if not ids]
        if missing:
            raise ValueError(
                f"{target['target_id']}: no {from_level} member for region(s) "
                f"{missing!r} in the crosswalk."
            )
        composition[str(target["target_id"])] = cells
    return composition


MHCLG_COUNCIL_TAX_STOCK_PREFIX = "mhclg.council_tax_stock."
WELSHGOV_COUNCIL_TAX_STOCK_PREFIX = "welshgov.council_tax_stock."
SCOTGOV_COUNCIL_TAX_STOCK_PREFIX = "scotgov.council_tax_stock."

# Target-id prefixes whose geography pins cannot come from the nation-substring
# rule below. Each entry is justified by the layer that fixes the geography:
# the Chronicle fact stamp, the contract binding, or the model variable's own
# country gate.
TARGET_PREFIX_GEOGRAPHY_PINS: tuple[tuple[str, str], ...] = (
    # Scottish Government statistics are Scotland-scoped and Chronicle stamps
    # them S92000003 (CTAXBASE chargeable dwellings, Scottish Budget
    # social-security lines); the substring rule sees no "scotland" in
    # "scotgov" or "scottish_child_payment" and would fall through to the UK
    # pin, which never matches a Scotland-stamped fact.
    ("scotgov.", "scotland"),
    # MHCLG's council taxbase return is England-only (facts stamped E92000001
    # and the 296 English billing authorities); the substring rule sees no
    # nation in "mhclg".
    ("mhclg.", "england"),
    # The SLC borrower-plan forecasts Chronicle carries are England-scoped
    # (facts stamped E92000001) and the contract bindings already filter
    # country == ENGLAND explicitly, so the GB default could never match.
    ("slc.borrowers.", "england"),
    # The SLC student-support publication is England-scoped (facts stamped
    # E92000001) and the bound model variables are England-gated by
    # construction (maintenance_loan_in_england_system,
    # parents_learning_allowance_eligible and adult_dependants_grant_eligible
    # all require country == ENGLAND). slc.repayments.* stays with the
    # substring rule: its england_* ids name their nation, and devolved_total
    # needs a per-nation redesign before it can activate.
    ("slc.support.", "england"),
    ("dfe.", "england"),
    # ORR rail industry finance is Great Britain (K03000001): NI Railways sits
    # outside the ORR series, and Chronicle stamps every ORR fact GB.
    ("orr.", "great_britain"),
    # Devolved bus publishers stamp their nation and name it in neither the
    # target id nor the concept.
    ("welshgov.", "wales"),
    ("nithc.", "northern_ireland"),
    ("dfi_ni.", "northern_ireland"),
)
# DfT BUS05i rows name their area in the selector; the geography follows the
# declared area, never a prefix, so a London or UK row can never be stamped
# England if it is activated. Sub-national DfT areas carry DfT's own ids.
DFT_BUS_AREA_GEOGRAPHY_IDS = {
    "england": UK_GEOGRAPHY_IDS["england"],
    "london": "E12000007",
    "england_outside_london": "dft:england_outside_london",
    "uk": UK_GEOGRAPHY_IDS["uk"],
}


def _geography_id_for_target(target: Mapping[str, Any]) -> str:
    target_id = str(target["target_id"]).lower()
    if target_id.startswith("dft."):
        area = str(
            (target.get("ledger_selector") or {}).get("layout_groupby_value_id", "")
        )
        if area not in DFT_BUS_AREA_GEOGRAPHY_IDS:
            raise ValueError(
                f"{target_id}: DfT rows must declare layout_groupby_value_id in "
                f"{sorted(DFT_BUS_AREA_GEOGRAPHY_IDS)}, got {area!r}."
            )
        return DFT_BUS_AREA_GEOGRAPHY_IDS[area]
    for prefix, geography_key in TARGET_PREFIX_GEOGRAPHY_PINS:
        if target_id.startswith(prefix):
            return UK_GEOGRAPHY_IDS[geography_key]
    selector = target.get("ledger_selector") or {}
    concept = str(selector.get("source_concept", "")).lower()
    measure = str(selector.get("source_measure_id", "")).lower()
    haystack = " ".join((target_id, concept, measure))
    if "northern" in haystack or "domestic_rates" in haystack:
        return UK_GEOGRAPHY_IDS["northern_ireland"]
    if "scotland" in haystack:
        return UK_GEOGRAPHY_IDS["scotland"]
    if "wales" in haystack:
        return UK_GEOGRAPHY_IDS["wales"]
    if "england" in haystack or target_id.startswith("voa."):
        return UK_GEOGRAPHY_IDS["england"]
    if target_id.startswith("dwp.") or target_id.startswith("slc."):
        return UK_GEOGRAPHY_IDS["great_britain"]
    return UK_GEOGRAPHY_IDS["uk"]


def _sum_target_ids(contract: Mapping[str, Any]) -> frozenset[str]:
    target_ids: set[str] = set()
    for target in contract.get("targets", ()):
        selector = target["ledger_selector"]
        binding = target["bindings"]["policyengine"]
        if "value_expression" in binding:
            target_ids.add(str(target["target_id"]))
        if any(
            key not in {"any_of", "dimensions"} and isinstance(value, list)
            for key, value in selector.items()
        ):
            target_ids.add(str(target["target_id"]))
        dimension_values = selector.get("dimension_values")
        if isinstance(dimension_values, Mapping) and any(
            isinstance(value, list) for value in dimension_values.values()
        ):
            target_ids.add(str(target["target_id"]))
    return frozenset(target_ids)


def _value_operation_by_target_id(contract: Mapping[str, Any]) -> dict[str, str]:
    operations = {target_id: "sum" for target_id in _sum_target_ids(contract)}
    for target in contract.get("targets", ()):
        target_id = str(target["target_id"])
        declared = target.get("value_operation")
        if declared is not None:
            operations[target_id] = str(declared)
        if (
            target.get("family") == "dwp_universal_credit"
            and declared not in MONTHLY_WINDOW_OPERATIONS
        ):
            operations[target_id] = "calendar_year_average"
    return operations


def _selector_pins(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    pins: dict[str, dict[str, Any]] = {}
    for target in contract.get("targets", ()):
        selector = target["ledger_selector"]
        dimension_values = selector.get("dimension_values")
        if selector.get(
            "source_concept"
        ) == "ons.mid_year_population_estimate" and isinstance(
            dimension_values, Mapping
        ):
            pins[str(target["target_id"])] = {
                "dimensions": sorted(str(key) for key in dimension_values)
            }
    return pins


def _signed_exclusions(contract: Mapping[str, Any]) -> dict[str, str]:
    target_ids = {str(target["target_id"]) for target in contract.get("targets", ())}
    resource = json.loads(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(
            "packages/microcosm-build/src/microcosm/build/uk/"
            "target_reference_signed_exclusions.json"
        )
        .read_text()
    )
    exclusions = {
        str(entry["target_id"]): str(entry["rationale"])
        for entry in resource["exclusions"]
        if "row" not in entry
    }
    return {
        target_id: rationale
        for target_id, rationale in exclusions.items()
        if target_id in target_ids
    }


def _signed_row_exclusions(
    contract: Mapping[str, Any],
) -> dict[str, dict[tuple[str, str], str]]:
    """Row-level sign-outs from the same register: one fan-out row of a target.

    An entry carrying ``row: {dimension, value}`` signs out the fan-out row
    whose selector pins that dimension value (the HMRC CGT age band 0-15,
    which the frame cannot carry; the size-of-gain band below the annual
    exempt amount) and leaves the target's other rows active.
    """

    target_ids = {str(target["target_id"]) for target in contract.get("targets", ())}
    resource = _signed_exclusion_register()
    rows: dict[str, dict[tuple[str, str], str]] = {}
    for entry in resource["exclusions"]:
        row = entry.get("row")
        if row is None:
            continue
        target_id = str(entry["target_id"])
        if target_id not in target_ids:
            continue
        if not isinstance(row, Mapping) or set(row) != {"dimension", "value"}:
            raise ValueError(
                f"Signed exclusion for {target_id!r} declares a malformed row "
                f"{row!r}; expected exactly dimension and value."
            )
        key = (str(row["dimension"]), json.dumps(row["value"], sort_keys=True))
        rows.setdefault(target_id, {})[key] = str(entry["rationale"])
    return rows


def _signed_exclusion_register() -> dict[str, Any]:
    return json.loads(
        Path(__file__)
        .resolve()
        .parents[1]
        .joinpath(
            "packages/microcosm-build/src/microcosm/build/uk/"
            "target_reference_signed_exclusions.json"
        )
        .read_text()
    )


def _reference_metadata(contract: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    result = {}
    for target in contract.get("targets", ()):
        measurement = target.get("measurement", {})
        metadata = {}
        if measurement.get("observation_basis") is not None:
            metadata["observation_basis"] = str(measurement["observation_basis"])
        binding = target.get("bindings", {}).get("policyengine", {})
        if binding.get("measurement_period") is not None:
            metadata["measurement_period"] = str(binding["measurement_period"])
        if binding.get("require_matching_fact_period"):
            metadata["source_period_policy"] = "exact_observation"
        if "source_months" in measurement:
            if target.get("family") != "dwp_universal_credit":
                raise ValueError("source_months is currently a UK UC-only declaration.")
            metadata.update(
                uc_source_month_metadata(
                    measurement["source_months"],
                    value_operation=str(
                        target.get("value_operation", "calendar_year_average")
                    ),
                )
            )
        if metadata:
            result[str(target["target_id"])] = metadata
    return result


#: Contract-level naming rule for fan-out rows whose publisher band carries
#: no incumbent registry name: the row takes ``<metric_name>_<lower edge>``,
#: the same form as the incumbent banded names, so a published band the
#: incumbent never carried is authored rather than silently dropped.
FANOUT_ROW_NAMING_METRIC_BAND_LOWER = "metric_name_band_lower"

_CGT_GAIN_BAND_VALUE = re.compile(r"^gain_(\d+)_(?:to_\d+|plus)$")


def _cgt_gain_band_lower(fact: Mapping[str, Any]) -> int | None:
    """Lower edge of an HMRC CGT size-of-gain band value id, if the fact has one."""

    dimensions = fact.get("dimensions") or {}
    if not isinstance(dimensions, Mapping):
        return None
    value = dimensions.get("cgt_gain_band")
    if not isinstance(value, str):
        return None
    match = _CGT_GAIN_BAND_VALUE.match(value)
    if match is None:
        raise ValueError(
            f"Unrecognised HMRC CGT gain band value {value!r}; expected "
            "gain_<lower>_to_<upper> or gain_<lower>_plus."
        )
    return int(match.group(1))


def _fanout_name(
    target: Mapping[str, Any],
    fact: Mapping[str, Any],
    inverse_mapping: Mapping[str, list[str]],
) -> str | None:
    target_id = str(target["target_id"])
    value_id = str(fact.get("layout", {}).get("groupby_value_id") or "")
    preferred_tokens, fallback_tokens = _dimension_tokens(fact)
    band_lower = _cgt_gain_band_lower(fact)
    candidates = inverse_mapping.get(target_id, ())
    if band_lower is not None:
        # Incumbent banded names end in the band's lower edge; a substring
        # match would let ``_band_50000`` claim ``_band_500000``.
        suffix = f"_band_{band_lower}"
        for candidate in candidates:
            if candidate.endswith(suffix):
                return candidate
        naming = target.get("fanout_row_naming")
        if naming == FANOUT_ROW_NAMING_METRIC_BAND_LOWER:
            metric_name = str(target["bindings"]["policyengine"]["metric_name"])
            return f"{metric_name}_{band_lower}"
        if naming is not None:
            raise ValueError(
                f"Unsupported fanout_row_naming {naming!r} on {target_id!r}."
            )
        return None if candidates else f"{target_id}.{value_id or 'detail'}"
    if target.get("fanout_row_naming") is not None:
        raise ValueError(
            f"{target_id!r} declares fanout_row_naming but its fact carries no "
            "recognised band dimension; the row would otherwise be dropped."
        )
    for candidate in candidates:
        if value_id and value_id not in _GEOGRAPHY_VALUE_IDS and value_id in candidate:
            return candidate
        if any(token in candidate for token in preferred_tokens):
            return candidate
    for candidate in candidates:
        if any(token in candidate for token in fallback_tokens):
            return candidate
    if len(candidates) == 1:
        return None
    if candidates:
        return None
    safe_value = value_id.replace("/", "_") or "detail"
    return f"{target_id}.{safe_value}"


def _dimension_tokens(fact: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    preferred: list[str] = []
    fallback: list[str] = []
    dimensions = fact.get("dimensions") or {}
    if not isinstance(dimensions, Mapping):
        return preferred, fallback
    for value in dimensions.values():
        if isinstance(value, int):
            token = _int_token(value)
            preferred.append(f"income_band_{token}_to")
            fallback.append(f"_{token}")
        elif isinstance(value, str) and value.isdigit():
            token = _int_token(int(value))
            preferred.append(f"income_band_{token}_to")
            fallback.append(f"_{token}")
        elif isinstance(value, str):
            annual_band = _annual_uc_award_band_token(value)
            if annual_band:
                preferred.append(annual_band)
                fallback.append(annual_band.removeprefix("annual_payment_"))
    return preferred, fallback


def _int_token(value: int) -> str:
    return f"{value:,}".replace(",", "_")


def _annual_uc_award_band_token(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "no payment" or "or over" in normalized:
        return ""
    if " to " not in normalized:
        return ""
    numbers = [
        float(number.replace(",", ""))
        for number in re.findall(r"[0-9][0-9,]*(?:\.[0-9]+)?", value)
    ]
    if len(numbers) != 2:
        return ""
    lower = int((numbers[0] // 100) * 100 * 12)
    upper = int(numbers[1] * 12)
    return f"annual_payment_{_int_token(lower)}_to_{_int_token(upper)}"


def _add_uk_membership_accounting(
    report: dict[str, Any],
    references: tuple[dict[str, Any], ...],
) -> None:
    fanout_counts = Counter(
        reference["family"]
        for reference in references
        if reference["name"] != reference["metadata"]["contract_target_id"]
    )
    council_tax_count = sum(
        1
        for reference in references
        if reference["metadata"]["contract_target_id"].startswith(
            (
                MHCLG_COUNCIL_TAX_STOCK_PREFIX,
                WELSHGOV_COUNCIL_TAX_STOCK_PREFIX,
                SCOTGOV_COUNCIL_TAX_STOCK_PREFIX,
            )
        )
    )
    report["fanout_family_outcomes"] = [
        {
            "family": "hmrc_spi",
            "status": "active_with_signed_property_amount_exclusion",
            "active_reference_count": fanout_counts.get("hmrc_spi", 0),
            "signed_rationale": (
                "SPI income-band targets fan out by strict total-income-band "
                "dimension pins, except the HMRC property-income amount "
                "surface. Those 13 rows are signed out because Ledger carries "
                "the official SPI Table 3.7 net property-income amounts, "
                "while the incumbent target applies the populace-side x1.9 "
                "property-income undercount adjustment traced to "
                f"{_UK_DATA_REPO} PR #311 / issue #230 and HMRC "
                "Property Rental Income Statistics."
            ),
        },
        {
            "family": "dwp_universal_credit",
            "status": "active_with_unmapped_vintage_residue_skipped",
            "active_reference_count": fanout_counts.get("dwp_universal_credit", 0),
            "skipped_unmapped_fact_count": 12,
            "signed_rationale": (
                "UC payment-distribution targets fan out over family_type and "
                "monthly_award_amount_bands into incumbent-compatible "
                "annual-payment rows. The four source-only 'No payment' facts "
                "and eight overlapping source-only 'or over' facts are left "
                "out; no active reference may use the legacy nan_to_nan band "
                "name."
            ),
        },
        {
            "family": "council_tax_stock",
            "status": "active_declared_rows",
            "active_reference_count": council_tax_count,
            "signed_rationale": (
                "MHCLG council taxbase (England), Welsh Government CT1 "
                "(Wales) and Scottish Government CTAXBASE (Scotland) "
                "council-tax stock bands are declared as explicit target rows "
                "per band plus total (England and Scotland A-H, Wales A-I). "
                "The Welsh and Scottish rows resolve with their country-level "
                "geography and band pin; the English rows fan out over the "
                "nine English regions of the region tier (microcosm#905), each "
                "cell composed as the linear combination of the billing-"
                "authority facts the crosswalk places in that region "
                "(microcosm#929)."
            ),
        },
        {
            "family": "hmrc_cgt",
            "status": "active_with_row_level_signed_exclusions",
            "active_reference_count": fanout_counts.get("hmrc_cgt", 0),
            "signed_rationale": (
                "The FY2024-25 individual CGT observations fan out three ways "
                "(microcosm#725, #467): Table 6 age bands as dimension rows "
                "(the 0-15 band and the all-ages total are signed out row by "
                "row), Table 5 country/region cells over the twelve-area "
                "region tier restated on the individuals basis by the Table 1 "
                "share through the scaled_by_ratio operation, and Table 2.1a "
                "size-of-gain bands under the incumbent banded names (the "
                "0-2,999 band below the 2024 annual exempt amount is signed "
                "out). Table 2.1a publishes no tax column, so liability binds "
                "nationally and by age band only."
            ),
        },
        {
            "family": "ons_population",
            "status": "active_region_tier_fanout",
            "active_reference_count": sum(
                1
                for reference in references
                if reference["metadata"]["contract_target_id"].startswith(
                    "ons.population."
                )
                and reference["metadata"]["contract_target_id"].endswith("_by_region")
            ),
            "signed_rationale": (
                "The nine ONS population-by-age-band targets fan out over the "
                "twelve-area region tier (nine English regions at Chronicle's "
                "region level, Wales, Scotland and Northern Ireland at country "
                "level), one reference per area, each scoped on the spine by "
                "a household-region predicate and placed at the region grain "
                "of the cross-grain rule (microcosm#905). The former single "
                "UK-wide row per band is retired: the tier sums to it within "
                "the same publication."
            ),
        },
    ]
    report["signed_exclusion_rationales"] = [
        {
            "family": "hmrc_spi",
            "target_id": "hmrc.spi.property_income.amount_by_total_income_band",
            "status": "signed_excluded",
            "signed_rationale": report["targets"][
                "hmrc.spi.property_income.amount_by_total_income_band"
            ]["candidates"][0]["signed_rationale"],
        },
        {
            "family": "ons_population",
            "target_id": "ons.population.scotland_households_3plus_children",
            "status": "signed_excluded",
            "signed_rationale": report["targets"][
                "ons.population.scotland_households_3plus_children"
            ]["candidates"][0]["signed_rationale"],
        },
    ]
    for target_id, entry in sorted(report["targets"].items()):
        if not str(target_id).startswith("hmrc.cgt."):
            continue
        for candidate in entry["candidates"]:
            if candidate.get("status") != "signed_excluded":
                continue
            report["signed_exclusion_rationales"].append(
                {
                    "family": "hmrc_cgt",
                    "target_id": target_id,
                    "row": candidate["name"],
                    "status": "signed_excluded",
                    "signed_rationale": candidate["signed_rationale"],
                }
            )
    report["multi_fact_rationales"] = []


_GEOGRAPHY_VALUE_IDS = frozenset(
    {
        "england",
        "great_britain",
        "northern_ireland",
        "scotland",
        "uk",
        "wales",
    }
)


if __name__ == "__main__":
    main()
