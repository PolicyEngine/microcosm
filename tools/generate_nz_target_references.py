"""Generate the reviewed NZ G8-r1b selector inventory from a pinned artifact.

Run with ``NZ_CHRONICLE_ARTIFACT`` or ``--artifact``. Observations stay in
Chronicle: references contain selectors only; the separate inventory resource
records source discrepancies and independently computed bridge diagnostics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from collections import Counter
from pathlib import Path

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.frame.concepts import TENURE_DOMAIN

SPEC = (
    Path(__file__).resolve().parents[1]
    / "packages/microcosm-build/src/microcosm/build/nz"
)
REGIONS = (
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
)
AGE_BANDS = (
    ("0_14", 0, 15),
    ("15_39", 15, 40),
    ("40_64", 40, 65),
    ("65_plus", 65, None),
)
RECEIPT_AGES = (
    ("18_24", "18-24 years", 18, 25),
    ("25_39", "25-39 years", 25, 40),
    ("40_54", "40-54 years", 40, 55),
    ("55_64", "55-64 years", 55, 65),
)
PROGRAMS = ("jobseeker_support", "sole_parent_support", "supported_living_payment")
TENURE = {
    "owned": ["owned_outright", "owned_with_mortgage"],
    "not_owned": [
        "rented_public_authority",
        "rented_nonprofit_social",
        "rented_private",
        "rent_free",
    ],
}
RESIDUAL_REASON = "Requires the shared reference-profile resolver to apply d1886 Census not-stated residual allocation before the single ERP bridge; raw size and tenure rows cannot compile."
RESIDUAL_DISCLOSURE = (
    "d1886 modelling convention, 23 October 2026. Microcosm allocates Census "
    "counts with unstated household size across stated size categories in "
    "proportion to their stated counts, assuming missing at random. Each region's "
    "computed residual is its independently published household total minus its "
    "stated leaves. The national size calculation is 61,725 households; regional "
    "calculations sum to 61,713. Tenure uses the same proportional convention: "
    "the national difference is 426 and regional differences sum to 429. These "
    "regional differences and allocated leaves are Microcosm calculations, not "
    "Stats NZ published residual cells or allocated estimates. The sensitivity "
    "allocates size residuals only to one- and two-person households in their "
    "stated proportions. Fractional targets are retained. Allocation precedes "
    "one application of B2 ERP growth. Region 99 remains separate and uses the "
    "disclosed national-growth proxy."
)
FAMILY_REASON = "Stats NZ Census-family nucleus standard and deterministic roster producer are not encoded; MC7 benefit units and family labels cannot substitute for Census multiplicities."
IRD_REASON = "B4: reviewed IRD-definition taxable-income measurement, annual component closure and record universe remain required. Ageing resolution does not establish that measurement."
IRD_MEASUREMENT_BINDING = "ird_taxable_income_ty2027_v1"
# Reviewed national detail-row map: nil row 8 through open tail row 244.
# Published national total row 246 is diagnostic and is deliberately absent.
IRD_NATIONAL_ROWS = tuple(f"row_{row}" for row in range(8, 245))
RECEIPT_REASON = "G8-w: age-stratified receipt assignment is not supported by the current total-only receipt kernel; preserve the source snapshot and age filters."
NZS_REASON = "NZ Super receipt binding and prepared receipt flag are not implemented."
VP_REASON = "Veterans Pension is unsupported: no veteran facts or producer; a missing flag must refuse, never default to false."


def _selector(fact, *, pin_key=True):
    measure, geography = fact["observed_measure"], fact["geography"]
    result = {
        "source_name": measure["source_name"],
        "source_measure_id": measure["source_measure_id"],
        "source_concept": measure["source_concept"],
        "entity_name": fact["entity"]["name"],
        "period_type": fact["period"]["type"],
        "period_value": fact["period"]["value"],
        "geography_level": geography["level"],
        "geography_id": geography["id"],
        "geography_vintage": geography["vintage"],
        "record_set_id": fact["layout"]["record_set_id"],
        "dimensions": list(fact["dimensions"]),
    }
    if fact["dimensions"]:
        result["dimension_values"] = dict(fact["dimensions"])
    if pin_key:
        result["aggregate_fact_key"] = fact["aggregate_fact_key"]
    return result


class Inventory:
    def __init__(self, facts):
        self.by_record_set = {}
        for fact in facts:
            self.by_record_set.setdefault(
                fact.get("layout", {}).get("record_set_id"), []
            ).append(fact)

    def cells(self, record_set, *, measure=None, geography=None, dimensions=None):
        result = self.by_record_set.get(record_set, [])
        if measure is not None:
            result = [
                f
                for f in result
                if f["observed_measure"]["source_measure_id"] == measure
            ]
        if geography is not None:
            result = [f for f in result if f["geography"]["id"] == geography]
        if dimensions is not None:
            result = [f for f in result if _typed_equal(f["dimensions"], dimensions)]
        return result

    def one(self, *args, **kwargs):
        matches = self.cells(*args, **kwargs)
        if len(matches) != 1:
            raise ValueError(
                f"Expected one source cell for {args!r} {kwargs!r}; found {len(matches)}"
            )
        return matches[0]


def _typed_equal(actual, expected):
    return actual.keys() == expected.keys() and all(
        type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items()
    )


def _reference(
    fact,
    *,
    name,
    contract,
    entity,
    measure,
    family,
    reference_set="calibration",
    status="active",
    reason=None,
    filter_column=None,
    period=None,
    method_card="MC5",
    label=None,
):
    provider = fact["observed_measure"]["source_name"]
    provider_label = {
        "stats_nz": "Stats NZ",
        "ird": "Inland Revenue",
        "msd": "Ministry of Social Development",
    }[provider]
    result = {
        "name": name,
        "ledger_selector": _selector(fact),
        "entity": entity,
        "measure": measure,
        "period": fact["period"]["value"] if period is None else period,
        "family": family,
        "assertion_policy": "observed_only",
        "metadata": {
            "contract_target_id": contract,
            "activation_status": status,
            "criticality": "release_blocking",
            "measure_kind": "prepared_column",
            "publisher": provider_label,
            "target_role": "calibration"
            if reference_set == "calibration"
            else "pre_calibration",
            "reference_set": reference_set,
            "method_card_row": method_card,
        },
        "hierarchy": {
            "provider": {"id": provider, "label": provider_label},
            "category": {
                "id": family,
                "label": family.replace("_", " ").title(),
                "provider_id": provider,
            },
            "target_label": label or fact["label"],
        },
    }
    legacy_contracts = {
        "stats_nz_population_by_age_region": "stats_nz_population_by_age_sex_region",
        "stats_nz_census_households_by_size_region": "stats_nz_census_households_by_composition",
        "stats_nz_census_families_by_type_region": "stats_nz_census_families_by_type",
        "msd_nz_super_recipients": "msd_nz_super_veterans_pension_recipients",
        "msd_veterans_pension_recipients": "msd_nz_super_veterans_pension_recipients",
    }
    if contract in legacy_contracts:
        result["metadata"]["legacy_contract_target_id"] = legacy_contracts[contract]
    if filter_column is not None:
        result["filter"] = filter_column
    if reason:
        result["metadata"]["parking_reason"] = reason
        result["notes"] = reason
    return result


def _bridge(reference, region, erp, *, members=None):
    growth_region = "NZ" if region == "99" else region
    base = {"role": "base"}
    if members is not None:
        base["expected_member_count"] = len(members)
        selector = reference["ledger_selector"]
        selector.pop("aggregate_fact_key")
        dimension = selector["dimensions"][0]
        selector["dimension_values"][dimension] = members
    reference.update(
        {
            "value_operation": "rolled_forward_by_ratio",
            "value_operands": [
                base,
                {"role": "numerator", **_selector(erp[growth_region][2025])},
                {"role": "denominator", **_selector(erp[growth_region][2023])},
            ],
            "period_match_policy": "latest_not_after",
            "period": 2025,
        }
    )
    reference["metadata"].update(
        {
            "source_snapshot": "2023-03-07 Census",
            "growth_denominator_snapshot": "2023-06-30 ERP",
            "target_level_snapshot": "2025-06-30 ERP",
            "model_rules_period": "TY2026-27",
            "bridge": "national_growth_proxy"
            if region == "99"
            else "regional_erp_density_preserving",
            "bridge_status": "national-growth proxy"
            if region == "99"
            else "regional ERP growth",
            "held_households_per_erp_person": "true",
            "held_census_families_per_erp_person": "true",
            "projection_from_june2025_to_model_period": "none",
            "method_decision": "d1598; g8r-hub-rulings 2026-10-22",
        }
    )


def calibration_bindings():
    """Return observation-free column declarations consumed by the neutral helper."""
    regions = [*REGIONS, "99"]
    columns = [
        {"entity": "person", "name": "people", "constant": 1},
        {"entity": "household", "name": "households", "constant": 1},
    ]
    for region in REGIONS:
        for band, low, high in AGE_BANDS:
            predicates = [
                {"column": "region_code", "operator": "eq", "value": region},
                {"column": "age", "operator": "ge", "value": low},
            ]
            if high is not None:
                predicates.append({"column": "age", "operator": "lt", "value": high})
            columns.append(
                {
                    "entity": "person",
                    "name": f"population_region_{region}_age_{band}_indicator",
                    "predicates": predicates,
                }
            )
    for region in regions:
        for size in range(1, 9):
            columns.append(
                {
                    "entity": "household",
                    "name": f"household_region_{region}_size_{size}_indicator",
                    "predicates": [
                        {
                            "column": "household_region_code",
                            "operator": "eq",
                            "value": region,
                        },
                        {
                            "column": "household_size",
                            "operator": "ge" if size == 8 else "eq",
                            "value": size,
                        },
                    ],
                }
            )
        for tenure, values in TENURE.items():
            columns.append(
                {
                    "entity": "household",
                    "name": f"household_region_{region}_{tenure}_indicator",
                    "predicates": [
                        {
                            "column": "household_region_code",
                            "operator": "eq",
                            "value": region,
                        },
                        {"column": "tenure", "operator": "in", "value": values},
                    ],
                }
            )
    for code, _, low, high in RECEIPT_AGES:
        columns.append(
            {
                "entity": "person",
                "name": f"age_{code}_indicator",
                "predicates": [
                    {"column": "age", "operator": "ge", "value": low},
                    {"column": "age", "operator": "lt", "value": high},
                ],
            }
        )
    return {
        "schema_version": 1,
        "country": "nz",
        "required_inputs": {
            "person": ["age", "region_code", "person_id", "person_household_id"],
            "household": ["household_region_code", "tenure", "household_id"],
        },
        "roster_counts": [
            {
                "entity": "household",
                "name": "household_size",
                "member_entity": "person",
                "entity_id": "household_id",
                "member_id": "person_id",
                "membership": "person_household_id",
                "representation": "Model roster represents usual-resident household membership.",
            }
        ],
        "input_domains": {
            "person": {
                "person_id": {"kind": "identity"},
                "person_household_id": {"kind": "identity"},
                "age": {"kind": "nonnegative_integer"},
                "region_code": {"values": regions},
            },
            "household": {
                "household_id": {"kind": "identity"},
                "household_region_code": {"values": regions},
                "tenure": {"values": list(TENURE_DOMAIN)},
            },
        },
        "columns": columns,
        "citations": [
            "packages/microcosm-frame/src/microcosm/frame/concepts.py TENURE_DOMAIN",
            "d1598",
            "g8r-hub-rulings:2026-10-22",
            "design-g8r:Prepared-column requirements",
        ],
        "parked_columns": [
            {
                "name": "census_families_type_<integer>",
                "kind": "household_multiplicity_measure",
                "activation_status": "requires_census_family_nucleus_standard",
                "reason": FAMILY_REASON,
            },
            {
                "name": "receives_<program>",
                "activation_status": "requires_stratified_receipt_assignment",
                "reason": RECEIPT_REASON,
            },
            {
                "name": "receives_nz_super",
                "activation_status": "requires_receipt_binding",
                "reason": NZS_REASON,
            },
            {
                "name": "receives_veterans_pension",
                "activation_status": "requires_unsupported_veterans_pension",
                "reason": VP_REASON,
            },
        ],
        "geography_inputs": "Geography stage supplies person.region_code and broadcasts the household location to household.household_region_code.",
        "region_99_support": {
            "calibration": "national-growth proxy",
            "geography": "99 remains separate from every region and from TA 999",
            "support": "No ERP series for region 99 and no TA 999 ERP; geography-stage assignment support remains an explicit G8-w prerequisite.",
        },
    }


def build_reference_documents(artifact, *, package_dir=SPEC):
    """Return generated file-name/document pairs; verify the full feed identity first."""
    from microcosm.build.ledger_artifact import validate_chronicle_feed_pin

    package_dir = Path(package_dir)
    feed_pin = json.loads((package_dir / "chronicle_feed.json").read_text())
    validate_chronicle_feed_pin(artifact, feed_pin)
    inventory = Inventory(artifact.facts)
    erp = {
        region: {
            year: inventory.one(
                f"stats_nz.cy{year}.subnational_population.all_ages",
                geography=region,
                dimensions={},
            )
            for year in (2023, 2025)
        }
        for region in (*REGIONS, "NZ")
    }
    targets, precal = [], []
    for region in REGIONS:
        for band, _, _ in AGE_BANDS:
            fact = inventory.one(
                f"stats_nz.cy2025.subnational_population.age_{band}",
                geography=region,
                dimensions={"person.age_band": band},
            )
            targets.append(
                _reference(
                    fact,
                    name=f"stats_nz_population_by_age_region__r{region}__a{band}",
                    contract="stats_nz_population_by_age_region",
                    entity="person",
                    measure="people",
                    family="demography",
                    filter_column=f"population_region_{region}_age_{band}_indicator",
                )
            )
    for table, count, contract, family, status, reason in (
        (
            "household_size",
            8,
            "stats_nz_census_households_by_size_region",
            "household_structure",
            "requires_reference_profile_resolution",
            RESIDUAL_REASON,
        ),
        (
            "family_type",
            3,
            "stats_nz_census_families_by_type_region",
            "household_structure",
            "requires_census_family_nucleus_standard",
            FAMILY_REASON,
        ),
    ):
        dimension = f"nz_census_2023.{table}_source_label"
        for region in (*REGIONS, "99"):
            for code in range(1, count + 1):
                fact = inventory.one(
                    f"stats_nz.census_2023.rc_{region}.{table}",
                    dimensions={dimension: code},
                )
                is_family = table == "family_type"
                row = _reference(
                    fact,
                    name=f"{contract}__r{region}__{'t' if is_family else 'n'}{code}",
                    contract=contract,
                    entity="household",
                    measure=f"census_families_type_{code}"
                    if is_family
                    else "households",
                    family=family,
                    status=status,
                    reason=reason,
                    filter_column=f"region_{region}_indicator"
                    if is_family
                    else f"household_region_{region}_size_{code}_indicator",
                    label=f"{fact['geography']['name']} Census {'family type' if is_family else 'household size'} {code}, ERP-density bridge to June 2025",
                )
                _bridge(row, region, erp)
                if not is_family:
                    row["metadata"]["required_reference_profile"] = (
                        "census_residual_allocation"
                    )
                    row["hierarchy"]["target_label"] += (
                        "; d1886 profile-resolved allocation"
                    )
                    if region == "99":
                        row["metadata"]["growth_geography_proxy"] = (
                            "national-growth proxy"
                        )
                        row["hierarchy"]["target_label"] += " (national-growth proxy)"
                targets.append(row)
    for region in (*REGIONS, "99"):
        for tenure, codes in (("owned", [1, 3]), ("not_owned", [2])):
            dimension = "nz_census_2023.household_tenure_source_label"
            fact = inventory.one(
                f"stats_nz.census_2023.rc_{region}.tenure",
                dimensions={dimension: codes[0]},
            )
            for code in codes:
                inventory.one(
                    f"stats_nz.census_2023.rc_{region}.tenure",
                    dimensions={dimension: code},
                )
            tenure_label = (
                "Owned including family trust" if tenure == "owned" else "Not owned"
            )
            label = f"{fact['geography']['name']} households: {tenure_label}, June 2025 ERP-density bridge"
            if region == "99":
                label += " (national-growth proxy)"
            row = _reference(
                fact,
                name=f"stats_nz_census_households_by_tenure_region__r{region}__{tenure}",
                contract="stats_nz_census_households_by_tenure_region",
                entity="household",
                measure="households",
                family="housing_tenure",
                status="requires_reference_profile_resolution",
                reason=RESIDUAL_REASON,
                filter_column=f"household_region_{region}_{tenure}_indicator",
                label=label,
            )
            _bridge(row, region, erp, members=codes if tenure == "owned" else None)
            row["metadata"]["required_reference_profile"] = "census_residual_allocation"
            row["hierarchy"]["target_label"] += "; d1886 profile-resolved allocation"
            if region == "99":
                row["metadata"]["growth_geography_proxy"] = "national-growth proxy"
            targets.append(row)
    for measure, contract, column in (
        ("individuals", "ird_taxable_income_people_by_band", "people"),
        (
            "taxable_income_nzd",
            "ird_taxable_income_total_by_band",
            "taxable_income_nzd",
        ),
    ):
        facts = inventory.cells(
            "ird_taxable_income_distribution.ty2024.income_bands", measure=measure
        )
        details = sorted(
            (f for f in facts if f["layout"]["table_record_kind"] == "detail"),
            key=lambda f: int(f["layout"]["source_row_id"].removeprefix("row_")),
        )
        if tuple(f["layout"]["source_row_id"] for f in details) != IRD_NATIONAL_ROWS:
            raise ValueError(
                "Expected the reviewed 237 taxable-income detail rows 8–244"
            )
        for fact in details:
            row_id = fact["layout"]["source_row_id"]
            targets.append(
                _reference(
                    fact,
                    name=f"{contract}__{row_id}",
                    contract=contract,
                    entity="person",
                    measure=column,
                    family="income",
                    period=2027,
                    status="requires_prerequisites",
                    reason=IRD_REASON,
                    filter_column=f"taxable_income_band_{row_id}_indicator",
                    label=f"TY2026-27 {contract.replace('_', ' ')} {fact['dimensions']['taxable_income_band_source_label']} (parked)",
                )
            )
            targets[-1]["metadata"].update(
                {
                    "reference_prerequisites": json.dumps(
                        [
                            "reference_profile:aging_index",
                            f"measurement:{IRD_MEASUREMENT_BINDING}",
                        ],
                        separators=(",", ":"),
                    ),
                    "required_reference_profile": "aging_index",
                    "required_measurement_binding": IRD_MEASUREMENT_BINDING,
                    "reference_value_kind": "count" if column == "people" else "money",
                }
            )
    wage_facts = inventory.cells(
        "ird_wage_salary_distribution.ty2026.income_bands", measure="individuals"
    )
    wage_details = sorted(
        (f for f in wage_facts if f["layout"]["table_record_kind"] == "detail"),
        key=lambda f: int(f["layout"]["source_row_id"].removeprefix("row_")),
    )
    if len(wage_details) != 235:
        raise ValueError("Expected 235 wage/salary detail bands")
    for fact in wage_details:
        precal.append(
            _reference(
                fact,
                name=f"ird_wage_salary_people_by_band__{fact['layout']['source_row_id']}",
                contract="ird_wage_salary_people_by_band",
                entity="person",
                measure="people",
                family="earnings",
                reference_set="precal",
                period=2027,
                method_card="MC4",
                status="requires_g8r2_source_period_wage_bridge_and_intervals",
                reason="G8-r2: source TY2026 requires its own TY2026-to-TY2027 wage bridge and reviewed numerical interval map; TY2024 factors are inapplicable.",
                label=f"TY2026-27 wage/salary band {fact['dimensions']['wage_salary_band_source_label']} (parked)",
            )
        )
    old_precal = json.loads((package_dir / "precal_references.json").read_text())
    for contract, status, reason in (
        (
            "mbie_bond_rents_by_territorial_authority",
            "requires_rent_band_frequencies",
            "MBIE medians, quartiles and bond counts do not supply rent-band frequencies by territorial authority, bedrooms and dwelling type.",
        ),
        (
            "stats_nz_household_financial_net_worth_distribution",
            "requires_all_household_liquid_asset_histogram",
            "Nonzero-holder summaries do not supply an all-household liquid-financial-asset histogram.",
        ),
    ):
        row = copy.deepcopy(
            next(r for r in old_precal["target_references"] if r["name"] == contract)
        )
        row["metadata"].update(
            {
                "contract_target_id": contract,
                "activation_status": status,
                "parking_reason": reason,
            }
        )
        row["notes"] = reason
        precal.append(row)
    ta_facts = sorted(
        inventory.cells(
            "stats_nz.cy2025.subnational_population_ta.all_ages", dimensions={}
        ),
        key=lambda f: f["geography"]["id"],
    )
    if len(ta_facts) != 67 or any(
        f["geography"]["id"] == "nz-ta-999" for f in ta_facts
    ):
        raise ValueError("Expected 67 published TA ERP support facts, excluding TA 999")
    for fact in ta_facts:
        row = _reference(
            fact,
            name=f"stats_nz_population_by_territorial_authority__ta{fact['geography']['id'].removeprefix('nz-ta-')}",
            contract="stats_nz_population_by_territorial_authority",
            entity="person",
            measure="people",
            family="demography",
            reference_set="precal",
            method_card="MC9",
        )
        row["metadata"]["precal_use"] = "atomic_geography_support"
        precal.append(row)
    for reference_set, destination in (("calibration", targets), ("precal", precal)):
        for program in PROGRAMS:
            for code, age_label, _, _ in RECEIPT_AGES:
                fact = inventory.one(
                    "msd_benefit_statistics.month2026_06.working_age_by_benefit_and_age",
                    measure=f"{program}_recipients",
                    dimensions={
                        "msd_main_benefit_age_group_source_label": age_label,
                        "msd_main_benefit_age_status": "working_age",
                        "msd_main_benefit_type": program,
                    },
                )
                row = _reference(
                    fact,
                    name=f"msd_{program}_recipients__age_{code}",
                    contract="msd_main_benefit_recipients_by_program_age",
                    entity="person",
                    measure=f"receives_{program}",
                    family="main_benefits",
                    reference_set=reference_set,
                    status="requires_stratified_receipt_assignment",
                    reason=RECEIPT_REASON,
                    filter_column=f"age_{code}_indicator",
                    method_card="MC5" if reference_set == "calibration" else "MC6",
                )
                row["metadata"]["measurement_basis"] = (
                    "June 2026 recipient snapshot stock; not annual recipients or full entitlement"
                )
                destination.append(row)
        for source_program, program, status, reason in (
            ("nzs", "nz_super", "requires_receipt_binding", NZS_REASON),
            (
                "vp",
                "veterans_pension",
                "requires_unsupported_veterans_pension",
                VP_REASON,
            ),
        ):
            fact = inventory.one(
                "msd_nzs_vp.june2026.recipient_total",
                measure=f"{source_program}_recipients",
                dimensions={
                    "msd_nzs_vp.benefit_type": source_program,
                    "msd_nzs_vp.recipient_total": "all",
                },
            )
            row = _reference(
                fact,
                name=f"msd_{program}_recipients",
                contract=f"msd_{program}_recipients",
                entity="person",
                measure=f"receives_{program}",
                family="superannuation",
                reference_set=reference_set,
                status=status,
                reason=reason,
                method_card="MC5" if reference_set == "calibration" else "MC6",
            )
            row["metadata"]["measurement_basis"] = (
                "June 2026 recipient snapshot stock; NZS and VP remain distinct"
            )
            destination.append(row)
    documents = {}
    for reference_set, rows, filename in (
        ("calibration", targets, "target_references.json"),
        ("precal", precal, "precal_references.json"),
    ):
        documents[filename] = {
            "country": "nz",
            "reference_set": reference_set,
            "description": "Generated by tools/generate_nz_target_references.py from the pinned Chronicle feed. Size and tenure require d1886 profile resolution; demography uses prepared columns and TA ERP is support input. Named parked rows refuse direct compilation. No observation is carried by a reference.",
            "allowed_value_operations": [
                "identity",
                "rolled_forward_by_ratio",
                "allocated_residual_by_ratio",
                "linear_combination",
            ]
            if reference_set == "calibration"
            else ["identity"],
            "target_references": rows,
            "target_profile": {"schema_version": 1, "hierarchy_reconciliations": []},
        }
    documents["calibration_bindings.json"] = calibration_bindings()
    documents["reference_profiles.json"] = _reference_profiles(inventory, targets)
    documents["reference_inventory.json"] = _receipts(
        inventory, erp, documents, feed_pin, ta_facts
    )
    return documents


def _allocation_operands(inventory, row, recipients):
    selector = row["ledger_selector"]
    region = selector["geography_id"]
    size = "households_by_size" in row["name"]
    table = "household_size" if size else "tenure"
    dimension = f"nz_census_2023.{'household_size' if size else 'household_tenure'}_source_label"
    leaves = [
        inventory.one(
            f"stats_nz.census_2023.rc_{region}.{table}",
            dimensions={dimension: code},
        )
        for code in range(1, 9 if size else 4)
    ]
    codes = selector["dimension_values"][dimension]
    return [
        {
            "role": "base",
            "expected_member_count": len(codes) if isinstance(codes, list) else 1,
        },
        {
            "role": "total",
            **_selector(
                inventory.one(
                    f"stats_nz.census_2023.rc_{region}.households", dimensions={}
                )
            ),
        },
        {
            "role": "partition",
            "member_selectors": [_selector(fact) for fact in leaves],
            "allocation_member_keys": [
                fact["aggregate_fact_key"]
                for fact in leaves
                if fact["dimensions"][dimension] in recipients
            ],
        },
        *copy.deepcopy(row["value_operands"][1:]),
    ]


def _reference_profiles(inventory, rows):
    profiles = {}
    for profile in ("mar", "small_households"):
        transformations = []
        for row in rows:
            if (
                row["metadata"]["activation_status"]
                != "requires_reference_profile_resolution"
            ):
                continue
            size = "households_by_size" in row["name"]
            recipients = (
                [1, 2]
                if size and profile == "small_households"
                else list(range(1, 9 if size else 4))
            )
            transformations.append(
                {
                    "reference_names": [row["name"]],
                    "operation": "allocated_residual_by_ratio",
                    "value_operands": _allocation_operands(inventory, row, recipients),
                    "metadata": {
                        "allocation_ruling": "d1886",
                        "census_residual_allocation": profile,
                    },
                    "notes": f"d1886 modelling convention: {'missing at random proportional size and tenure allocation' if profile == 'mar' else 'size residual only to one- and two-person households; tenure remains proportional'}. Computed regional residuals and fractional targets; allocation precedes one B2 ERP bridge.",
                    "target_label": row["hierarchy"]["target_label"] + f" ({profile})",
                }
            )
        profiles[profile] = {"transformations": transformations}
    return {
        "schema_version": 1,
        "factor_context_resource": "reference_aging.json",
        "dimensions": {
            "census_residual_allocation": {"default": "mar", "profiles": profiles},
            "aging_index": {
                "default": "central",
                "profiles": {
                    profile: {
                        "transformations": [
                            {
                                "operation": "scale_money_and_intervals",
                                "factor_id": f"ird_ty2024_to_ty2027_{profile}",
                                "reference_names": [
                                    f"ird_taxable_income_total_by_band__{row_id}"
                                    for row_id in IRD_NATIONAL_ROWS
                                ],
                                "count_reference_names": [
                                    f"ird_taxable_income_people_by_band__{row_id}"
                                    for row_id in IRD_NATIONAL_ROWS
                                ],
                                "interval_table_ids": [],
                                "required_measurement_binding": IRD_MEASUREMENT_BINDING,
                            }
                        ]
                    }
                    for profile in ("low", "central", "high")
                },
            },
        },
    }


def _residual_receipts(inventory, erp):
    receipts = []
    for region in (*REGIONS, "99"):
        total = inventory.one(
            f"stats_nz.census_2023.rc_{region}.households", dimensions={}
        )
        growth = erp["NZ" if region == "99" else region]
        ratio = growth[2025]["value"] / growth[2023]["value"]
        for table, dimension, codes in (
            ("household_size", "household_size", range(1, 9)),
            ("tenure", "household_tenure", range(1, 4)),
        ):
            leaves = [
                inventory.one(
                    f"stats_nz.census_2023.rc_{region}.{table}",
                    dimensions={f"nz_census_2023.{dimension}_source_label": code},
                )
                for code in codes
            ]
            leaf_sum = math.fsum(fact["value"] for fact in leaves)
            residual = total["value"] - leaf_sum
            if residual < 0:
                raise ValueError(f"Negative {table} residual for {region}")
            for profile in ("mar", "small_households"):
                recipients = (
                    [1, 2]
                    if table == "household_size" and profile == "small_households"
                    else list(codes)
                )
                mass = math.fsum(
                    fact["value"]
                    for code, fact in zip(codes, leaves, strict=True)
                    if code in recipients
                )
                allocated = [
                    fact["value"]
                    + (
                        residual * fact["value"] / mass
                        if code in recipients and residual
                        else 0
                    )
                    for code, fact in zip(codes, leaves, strict=True)
                ]
                closure = math.fsum(allocated)
                final_closure = math.fsum(count * ratio for count in allocated)
                if not math.isclose(
                    closure, total["value"], rel_tol=1e-12, abs_tol=1e-7
                ) or not math.isclose(
                    final_closure, total["value"] * ratio, rel_tol=1e-12, abs_tol=1e-7
                ):
                    raise ValueError(
                        f"Allocation does not close for {region} {table} {profile}"
                    )
                receipts.append(
                    {
                        "geography_id": region,
                        "table": table,
                        "profile": profile,
                        "classification": "Microcosm computed regional residual and allocated outputs; not published Stats NZ cells",
                        "allocation_ruling": "d1886",
                        "total_selector": _selector(total),
                        "leaf_selectors": [_selector(fact) for fact in leaves],
                        "published_total": total["value"],
                        "raw_leaf_sum": leaf_sum,
                        "computed_residual": residual,
                        "recipient_codes": recipients,
                        "allocation_policy": "proportional to stated recipient counts",
                        "allocated_counts": allocated,
                        "allocated_closure": closure,
                        "numerator_selector": _selector(growth[2025]),
                        "denominator_selector": _selector(growth[2023]),
                        "growth_ratio": ratio,
                        "growth_applications": 1,
                        "bridge": "national-growth proxy"
                        if region == "99"
                        else "regional ERP growth",
                        "final_closure": final_closure,
                        "expected_final_closure": total["value"] * ratio,
                        "tolerance": {"rel_tol": 1e-12, "abs_tol": 1e-7},
                    }
                )
    return receipts


def _receipts(inventory, erp, documents, feed_pin, ta_facts):
    discrepancies, diagnostics = [], []
    for table, dimension, codes, total_table in (
        ("tenure", "household_tenure", range(1, 4), "households"),
        ("household_size", "household_size", range(1, 9), "households"),
        ("family_type", "family_type", range(1, 4), "families"),
    ):
        total = inventory.one(f"stats_nz.census_2023.nz.{total_table}", dimensions={})
        national_leaves = [
            inventory.one(
                f"stats_nz.census_2023.nz.{table}",
                dimensions={f"nz_census_2023.{dimension}_source_label": code},
            )
            for code in codes
        ]
        source_sum = math.fsum(f["value"] for f in national_leaves)
        regional_scaled, regional_totals_scaled, regional_total_selectors = [], [], []
        for region in (*REGIONS, "99"):
            growth = erp["NZ" if region == "99" else region]
            ratio = growth[2025]["value"] / growth[2023]["value"]
            regional_total = inventory.one(
                f"stats_nz.census_2023.rc_{region}.{total_table}", dimensions={}
            )
            regional_totals_scaled.append(regional_total["value"] * ratio)
            regional_total_selectors.append(_selector(regional_total))
            for code in codes:
                fact = inventory.one(
                    f"stats_nz.census_2023.rc_{region}.{table}",
                    dimensions={f"nz_census_2023.{dimension}_source_label": code},
                )
                regional_scaled.append(fact["value"] * ratio)
        nz_ratio = erp["NZ"][2025]["value"] / erp["NZ"][2023]["value"]
        discrepancies.append(
            {
                "table": table,
                "published_national_total_selector": _selector(total),
                "national_leaf_selectors": [_selector(f) for f in national_leaves],
                "published_national_total": total["value"],
                "published_national_leaf_sum": source_sum,
                "deficit": total["value"] - source_sum,
                "handling": "Preserve raw published discrepancies and allocate each region's independently computed residual under d1886; national calculations remain separate diagnostics."
                if table != "family_type"
                else "Preserve the published discrepancy; no residual category or hidden reconciliation.",
                "activation": "parked: " + FAMILY_REASON
                if table == "family_type"
                else "Requires d1886 reference-profile resolution; total is an allocation operand and never an additional fitted target.",
            }
        )
        diagnostics.append(
            {
                "table": table,
                "regional_leaf_sum_scaled": math.fsum(regional_scaled),
                "regional_published_total_sum_scaled": math.fsum(
                    regional_totals_scaled
                ),
                "regional_published_total_selectors": regional_total_selectors,
                "independently_scaled_published_national_total": total["value"]
                * nz_ratio,
                "independently_scaled_published_national_leaves": source_sum * nz_ratio,
                "national_growth_operand_geography": "NZ",
                "region_99": "included separately using national-growth proxy",
                "use": "diagnostic only; never selected with the fitted leaves; aggregate_admin does not synthesize this roll-up",
            }
        )
    counts = {}
    for filename in ("target_references.json", "precal_references.json"):
        rows = documents[filename]["target_references"]
        counts[filename] = {
            "total": len(rows),
            "by_contract": dict(
                Counter(row["metadata"]["contract_target_id"] for row in rows)
            ),
            "by_activation_status": dict(
                Counter(row["metadata"]["activation_status"] for row in rows)
            ),
        }
    excluded = [
        (
            "regional_all_age_erp",
            "Bridge operands only; selecting totals together with age leaves duplicates the fit partition.",
        ),
        (
            "national_erp",
            "Bridge operand or diagnostic only; national and regional ERP totals are independently published.",
        ),
        (
            "sex_specific_regional_erp",
            "Chosen source surface is age-by-region, all sexes; no sex-specific split is emitted.",
        ),
        (
            "census_national_and_regional_totals",
            "Allocation operands or diagnostic receipts only; do not select totals as fitted rows with leaves.",
        ),
        (
            "ird_taxable_income_total_row_246",
            "Detail bands are the selected partition; published Total remains diagnostic.",
        ),
        (
            "ird_wage_salary_total_row_242",
            "Detail bands are the quantile input partition; published Total remains diagnostic.",
        ),
        (
            "other_main_benefits",
            "Unsupported aggregate of other programs; outside the three declared receipt programs.",
        ),
        ("all_main_benefits", "Aggregate overlaps selected program-age leaves."),
        (
            "combined_youth_and_young_parent",
            "Combined source category has no separate reviewed program binding.",
        ),
        (
            "msd_program_totals",
            "Do not add program totals to the program-age fit partition.",
        ),
        (
            "combined_nzs_vp",
            "No combined published total is emitted; NZS and unsupported VP remain distinct.",
        ),
        (
            "ta_999_erp",
            "No ERP fact exists for TA 999; its crosswalk Census population is 33; it is not region 99.",
        ),
        (
            "region_99_erp",
            "No regional ERP series exists; Census region 99 keeps its local total and computed residual, then uses the explicit national-growth proxy.",
        ),
        (
            "qes_prefu_and_b5_ageing",
            "QES/PREFU feed rows support the declared factor calculations in reference_aging.json; they never become direct calibration references or carried target values.",
        ),
        (
            "wff_tripwires",
            "G8-w; held-out WFF references remain explicitly parked and selected parked rows must refuse.",
        ),
    ]
    return {
        "schema_version": 1,
        "country": "nz",
        "generated_by": "tools/generate_nz_target_references.py",
        "feed_identity": feed_pin,
        "reference_counts": counts,
        "regions": list(REGIONS),
        "census_regions": [*REGIONS, "99"],
        "age_bands": [band for band, _, _ in AGE_BANDS],
        "erp_bridges": [
            {
                "geography_id": region,
                "numerator_selector": _selector(values[2025]),
                "denominator_selector": _selector(values[2023]),
                "numerator": values[2025]["value"],
                "denominator": values[2023]["value"],
                "ratio": values[2025]["value"] / values[2023]["value"],
            }
            for region, values in erp.items()
        ],
        "source_discrepancies": discrepancies,
        "modelling_convention": {
            "ruling": "d1886",
            "date": "2026-10-23",
            "disclosure": RESIDUAL_DISCLOSURE,
        },
        "regional_profile_allocations": _residual_receipts(inventory, erp),
        "national_bridge_diagnostics": diagnostics,
        "ta_support": {
            "count": len(ta_facts),
            "population_sum": math.fsum(f["value"] for f in ta_facts),
            "ta_999": "absent ERP; crosswalk Census population 33 remains explicit",
        },
        "exclusions": [{"id": name, "reason": reason} for name, reason in excluded],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", default=os.environ.get("NZ_CHRONICLE_ARTIFACT"))
    parser.add_argument("--out", type=Path, default=SPEC)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.artifact:
        parser.error("--artifact or NZ_CHRONICLE_ARTIFACT is required")
    artifact = load_ledger_consumer_artifact(args.artifact)
    documents = build_reference_documents(artifact)
    for filename, document in documents.items():
        encoded = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        destination = args.out / filename
        if args.check:
            if destination.read_text() != encoded:
                raise SystemExit(f"Stale generated resource: {destination}")
        else:
            destination.write_text(encoded)


if __name__ == "__main__":
    main()
