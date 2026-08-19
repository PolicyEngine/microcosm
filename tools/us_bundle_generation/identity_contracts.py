"""One-shot extraction of US pipeline and stochastic-owner identities.

This module is migration tooling only.  It reads the generation-0 runtime
registries once so the generated bundle can become the single authored source;
the production spec-engine does not import any of these constants.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from microcosm.build.spec_engine.seeds import LEGACY_V1_PROTOCOL
from microcosm.build.us_runtime.h5_io import US_STACKED_POOL_OPERATOR_ORDER
from microcosm.build.us_runtime.multispine_pool import (
    POOL_DERIVE_OPERATOR_ORDER,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
)
from microcosm.build.us_runtime.qbi_inputs import (
    us_qbi_reconciliation_contract_identity,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_TRANSFER_GROUPS,
)
from tools.build_us_multispine_pool import (
    stacked_checkpoint_artifact_protocol_identity,
)

__all__ = ["build_pipeline_contract", "build_seed_site_bindings"]


_AUXILIARY_OPERATIONS = (
    "assign_us_puma_ladder",
    "calibrate",
    "select_exact_k",
)


def build_pipeline_contract() -> dict[str, object]:
    """Extract the complete static pipeline/checkpoint identity surface."""

    return {
        "artifact_protocol": stacked_checkpoint_artifact_protocol_identity(),
        "stacked_operator_order": list(US_STACKED_POOL_OPERATOR_ORDER),
        "pre_clone_source_operator_order": list(POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER),
        "post_clone_source_operator_order": list(POOL_POST_CLONE_SOURCE_OPERATOR_ORDER),
        "derive_operator_order": list(POOL_DERIVE_OPERATOR_ORDER),
        "auxiliary_operations": list(_AUXILIARY_OPERATIONS),
        "qbi_reconciliation": us_qbi_reconciliation_contract_identity(),
        "simulation_household_batch_size": {
            "value": POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
            # No cross-partition byte-invariance proof exists at generation 0,
            # so this cannot be classified as an execution-profile knob.  Keep
            # the conservative normative identity fence until such a proof lands.
            "classification": "normative_legacy_identity_fence",
            "current_identity_effect": ("bound_into_generation_0_checkpoint_identity"),
        },
    }


def _owner(kind: str, owner_id: str) -> dict[str, str]:
    return {"kind": kind, "id": owner_id}


def _source_stage_ids(source_document: Mapping[str, Any]) -> frozenset[str]:
    stages = source_document.get("stages")
    if not isinstance(stages, list):
        raise RuntimeError("sources document has no stage array")
    result: set[str] = set()
    for index, row in enumerate(stages):
        if not isinstance(row, Mapping) or not isinstance(row.get("stage"), str):
            raise RuntimeError(f"sources stage {index} has no id")
        stage_id = str(row["stage"])
        if stage_id in result:
            raise RuntimeError(f"duplicate sources stage {stage_id!r}")
        result.add(stage_id)
    return frozenset(result)


def _stages_with_operation(
    source_document: Mapping[str, Any], operation: str
) -> tuple[str, ...]:
    stages = source_document.get("stages")
    assert isinstance(stages, list)
    matches: list[str] = []
    for row in stages:
        assert isinstance(row, Mapping)
        operations = row.get("operations", [])
        if not isinstance(operations, list):
            continue
        if any(
            isinstance(item, Mapping) and item.get("kind") == operation
            for item in operations
        ):
            matches.append(str(row["stage"]))
    if not matches:
        raise RuntimeError(f"no source stages own operation {operation!r}")
    return tuple(matches)


def build_seed_site_bindings(
    source_document: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Bind all 72 legacy-v1 sites to their concrete execution owners."""

    source_ids = _source_stage_ids(source_document)
    transfer_nodes = tuple(group.name for group in CANONICAL_US_LATE_TRANSFER_GROUPS)
    producer_ids = frozenset({"primary_puf_qrf", *transfer_nodes})
    pipeline_ids = frozenset(
        {
            *US_STACKED_POOL_OPERATOR_ORDER,
            *POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
            *POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
            *POOL_DERIVE_OPERATOR_ORDER,
            *_AUXILIARY_OPERATIONS,
        }
    )

    source = lambda stage: (_owner("source_stage", stage),)  # noqa: E731
    producer = lambda node: (_owner("producer_node", node),)  # noqa: E731
    pipeline = lambda operation: (  # noqa: E731
        _owner("pipeline_operation", operation),
    )

    owners: dict[str, tuple[dict[str, str], ...]] = {
        "survey_sample_asec": pipeline("assemble_stacked_spine"),
        "survey_sample_acs": pipeline("assemble_stacked_spine"),
        "puf_clone_attachment": pipeline("prepare_multispine_source_inputs_for_clone"),
        "puf_archived_aggregate_disaggregation": source("puf_tax_detail"),
        "puf_live_aggregate_disaggregation": source("puf_tax_detail"),
        "ssi_weighted_replacement_training": source("ssi_disability_criteria"),
        "ssi_archived_qrf_model": source("ssi_disability_criteria"),
        "sipp_vehicle_training_cap": source("vehicle_assets"),
        "sipp_vehicle_qrf_model": source("vehicle_assets"),
        "sipp_vehicle_count_random_forest_model": source("vehicle_assets"),
        "sipp_financial_asset_training_cap": source("scf_wealth"),
        "sipp_financial_asset_qrf_models": source("scf_wealth"),
        "acs_rent_archived_training_cap": source("acs_rent"),
        "sipp_tip_training_cap": source("sipp_tips"),
        "scf_household_source_selector": source("scf_wealth"),
        "scf_financial_asset_qrf_model": source("scf_wealth"),
        "scf_net_worth_qrf_model": source("scf_wealth"),
        "scf_auto_loan_qrf_model": source("scf_wealth"),
        "acs_transfer_family_seed": (
            *pipeline("gap_fill_stacked_spine"),
            *tuple(_owner("producer_node", node) for node in transfer_nodes),
        ),
        "acs_transfer_pattern_seed": (
            *pipeline("gap_fill_stacked_spine"),
            *tuple(_owner("producer_node", node) for node in transfer_nodes),
        ),
        "primary_qrf_fit_draw": producer("primary_puf_qrf"),
        "acs_qrf_fit_draw": (
            *pipeline("gap_fill_stacked_spine"),
            *tuple(_owner("producer_node", node) for node in transfer_nodes),
        ),
        "child_support_puf_qrf_model": source("child_support_inputs"),
        "childcare_puf_qrf_model": source("childcare_inputs"),
        "disability_benefits_puf_qrf_model": source("disability_benefits_input"),
        "energy_subsidy_puf_qrf_model": source("energy_subsidy"),
        "acs_rent_qrf_model": source("acs_rent"),
        "housing_assistance_puf_qrf_model": source("acs_rent"),
        "org_wages_qrf_model": source("org_wages"),
        "other_health_insurance_puf_qrf_model": source(
            "other_health_insurance_premiums"
        ),
        "prior_year_income_puf_qrf_model": source("prior_year_income"),
        "primary_puf_monolithic_qrf_model": producer("primary_puf_qrf"),
        "retirement_contributions_puf_qrf_model": source("retirement_contributions"),
        "retirement_distributions_puf_qrf_model": source("retirement_distributions"),
        "sipp_head_start_qrf_model": source("sipp_head_start"),
        "sipp_tip_qrf_model": source("sipp_tips"),
        "voluntary_filing_qrf_model": source("voluntary_filing_input"),
        "weeks_unemployed_puf_qrf_model": source("weeks_unemployed_input"),
        "workers_compensation_puf_qrf_model": source("workers_compensation_input"),
        "org_union_hash_lottery": source("org_wages"),
        "source_aca_assignment": source("aca_marketplace_inputs"),
        "source_count_calibration": tuple(
            _owner("source_stage", stage)
            for stage in _stages_with_operation(
                source_document, "calibrate_binary_assignment"
            )
        ),
        "source_joint_count_calibration": tuple(
            _owner("source_stage", stage)
            for stage in _stages_with_operation(
                source_document, "calibrate_binary_assignment_joint_targets"
            )
        ),
        "snap_take_up_assignment": source("snap_take_up"),
        "pregnancy_assignment": source("pregnancy"),
        "wic_claim_assignment": source("wic_claim_input"),
        "snap_discretionary_exemption_assignment": source(
            "snap_abawd_discretionary_exemption"
        ),
        "immigration_ead_workers_assignment": source("immigration_status"),
        "immigration_ead_students_assignment": source("immigration_status"),
        "ssi_take_up_assignment": source("ssi_take_up"),
        "medicaid_take_up_assignment": source("medicaid_take_up"),
        "snap_state_take_up_assignment": source("snap_state_take_up"),
        "tanf_take_up_assignment": pipeline("seed_multispine_pool_inputs"),
        "eitc_take_up_assignment": pipeline("seed_multispine_pool_inputs"),
        "adult_care_weighted_prefix_assignment": source("adult_care_inputs"),
        "capital_gains_tail_random_rank": pipeline("prepare_stacked_tail_derivation"),
        "torch_calibration_reseed": pipeline("calibrate"),
        "exact_k_pcg64_selection": pipeline("select_exact_k"),
        "prior_year_income_training_cap": source("prior_year_income"),
        "childcare_training_cap": source("childcare_inputs"),
        "retirement_contributions_training_cap": source("retirement_contributions"),
        "disability_benefits_training_cap": source("disability_benefits_input"),
        "housing_inputs_training_cap": source("acs_rent"),
        "workers_compensation_training_cap": source("workers_compensation_input"),
        "retirement_distributions_training_cap": source("retirement_distributions"),
        "child_support_training_cap": source("child_support_inputs"),
        "energy_subsidy_training_cap": source("energy_subsidy"),
        "other_health_insurance_training_cap": source(
            "other_health_insurance_premiums"
        ),
        "weeks_unemployed_training_cap": source("weeks_unemployed_input"),
        "legacy_geography_ladder": pipeline("assign_us_puma_ladder"),
        "legacy_puma_ladder": pipeline("assign_us_puma_ladder"),
        "legacy_congressional_district_assignment": pipeline("assign_us_puma_ladder"),
    }

    protocol_site_ids = tuple(site.id for site in LEGACY_V1_PROTOCOL.sites)
    if len(protocol_site_ids) != 72 or set(owners) != set(protocol_site_ids):
        raise RuntimeError(
            "legacy-v1 seed owner ledger must cover exactly 72 protocol sites; "
            f"missing={sorted(set(protocol_site_ids) - owners.keys())!r}, "
            f"extra={sorted(owners.keys() - set(protocol_site_ids))!r}"
        )
    valid_owner_ids = {
        "producer_node": producer_ids,
        "source_stage": source_ids,
        "pipeline_operation": pipeline_ids,
    }
    rows: list[dict[str, object]] = []
    for site in protocol_site_ids:
        site_owners = owners[site]
        for owner in site_owners:
            if owner["id"] not in valid_owner_ids[owner["kind"]]:
                raise RuntimeError(
                    f"seed site {site!r} has dangling {owner['kind']} "
                    f"owner {owner['id']!r}"
                )
        rows.append(
            {
                "site": site,
                "owners": sorted(site_owners, key=lambda row: (row["kind"], row["id"])),
            }
        )
    return rows
