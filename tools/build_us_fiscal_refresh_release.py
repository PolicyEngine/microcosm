"""Build a contract-valid US fiscal refresh release from a Populace H5.

This is a narrow release builder for the Issue #40 fiscal target surface. It
starts from an existing Populace US H5, materializes the current structured
fiscal target rows, recalibrates only the household weights, writes a fresh
PolicyEngine-US H5, and emits the release contract files required by
``populace-publish-release``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.gates import (
    GateResult,
    default_valued_columns_gate,
    input_mass_parity_gate,
    nonconstant_columns_gate,
    parity_gate,
    target_profile_coverage_gate,
)
from populace.build.ledger_artifact import load_ledger_consumer_artifact
from populace.build.source_runtime import SourceRuntimeConfig, run_source_stage
from populace.build.staging import StagingTelemetry
from populace.build.us_runtime import (
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    ORG_2024_DONOR_CONTENT_SHA256,
    SIPP_2023_TIP_DONOR_SHA256,
    SOI_VARIABLE_MAP,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_SUPPORT_EXCLUSIONS,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    US_MEDICAID_ENROLLMENT_TARGET_TABLE,
    US_SOURCE_MANIFEST,
    apply_us_medicaid_enrollment_substitutions,
    assert_release_input_coverage_manifest_current,
    assert_take_up_contract_current,
    assert_take_up_treatments_consistent,
    assert_validation_leaf_registry_current,
    compile_us_fiscal_target_registry,
    fetch_org_2024_donor,
    fetch_scf_2022_full_extract,
    fetch_scf_2022_summary_extract,
    fetch_sipp_2023_tip_donor,
    hard_target_package_aliases,
    load_congressional_district_vintage_crosswalk,
    load_org_2024_donor,
    load_scf_2022_auto_loan_donor,
    load_scf_2022_financial_asset_donor,
    load_sipp_2023_tip_donor,
    us_education_inputs_signal_gate,
    us_eligibility_inputs_signal_gate,
    us_hours_worked_signal_gate,
    us_immigration_composition_gate,
    us_medicaid_take_up_gate,
    us_org_wages_signal_gate,
    us_pregnancy_signal_gate,
    us_reform_coverage_smoke_gate,
    us_register_consistency_gate,
    us_release_input_coverage_gate,
    us_retirement_contributions_signal_gate,
    us_scf_auto_loans_signal_gate,
    us_scf_wealth_signal_gate,
    us_sipp_tips_signal_gate,
    us_snap_discretionary_exemption_signal_gate,
    us_snap_take_up_signal_gate,
    us_source_coverage_diagnostics,
    us_source_operation_handlers,
    us_take_up_participation_diagnostics,
    us_take_up_signal_gate,
    us_validation_input_coverage_gate,
    with_us_education_inputs,
    with_us_eligibility_inputs,
    with_us_hours_worked_inputs,
    with_us_immigration_inputs,
    with_us_medicaid_take_up,
    with_us_org_wages_inputs,
    with_us_pregnancy_inputs,
    with_us_retirement_contribution_inputs,
    with_us_scf_auto_loan_inputs,
    with_us_scf_wealth_inputs,
    with_us_sipp_tip_inputs,
    with_us_snap_discretionary_exemption_inputs,
    with_us_snap_take_up_inputs,
    with_us_take_up_inputs,
    write_us_medicaid_take_up_diagnostics,
    write_us_source_coverage_diagnostics,
    write_us_take_up_participation_diagnostics,
)
from populace.build.us_runtime.demographics import (
    CENSUS_NATIONAL_AGE_BENCHMARK,
    demographics_payload,
    geography_coverage_payload,
    population_by_age_from_sim,
    write_demographics,
)
from populace.build.us_runtime.input_mass import us_input_mass_totals
from populace.build.us_runtime.l0_refit_export import (
    attach_l0_refit_entity_weights,
    load_us_frame,
)
from populace.build.us_runtime.nonzero_shares import us_nonzero_shares
from populace.build.us_runtime.parity_reference import (
    EcpsParityReference,
    ParityKnownGap,
    load_ecps_parity_known_gaps,
    load_ecps_parity_reference,
)
from populace.build.us_runtime.reform_validation import (
    default_baseline_level_specs,
    default_simulate_factory,
    load_default_reform_specs,
    reform_validation_payload,
    write_reform_validation,
)
from populace.build.us_runtime.warm_start_selection import (
    DEFAULT_SELECTION_JOIN_KEY,
    SELECTION_MODES,
    load_selection_source_from_h5,
    load_selection_source_from_manifest,
    select_frozen_support,
)
from populace.calibrate import TargetRegistry, calibrate, calibrate_l0_refit
from populace.calibrate.diagnostics import (
    diagnostics_payload,
    write_calibration_diagnostics,
)
from populace.frame import Frame, MassChange, WeightKind, Weights
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine
from populace.frame.units import US_SCHEMA

PERIOD = 2024
REPO_ID = "policyengine/populace-us"
DATASET_FILENAME = "populace_us_2024.h5"
CALIBRATION_FILENAME = "populace_us_2024_calibration.npz"
POST_EXPORT_ABSOLUTE_TOLERANCE = 1_000_000.0
POST_EXPORT_RELATIVE_TOLERANCE = 5e-4
US_FISCAL_TARGET_LOSS_WEIGHTING = (
    "sqrt_value_concept_budget_weighted_mape_50_50_amount_count_target_scale_cap_100pct"
)
US_FISCAL_TARGET_VALUE_WEIGHT_POWER = 0.5
US_FISCAL_TARGET_LOSS_CAP = 1.0
# Bumped 1 -> 2 for #217: the per-reform income-tax cache key now depends only on
# the inputs that actually determine per-household reform estimates and no longer
# includes build_commit / seed / target_registry_version. Old (v1) coarse-key
# entries live under different filenames, so a mixed cache dir never collides.
TARGET_MATERIALIZATION_CACHE_SCHEMA_VERSION = 2
# The subset of the materialization cache context that determines per-household JCT
# reform income-tax vectors (#217). Anything outside this set — build commit, RNG
# seed, target registry version, calibration settings — cannot change a reform's
# per-household estimate, so it is deliberately excluded from the reform-vector
# cache key. That lets a restart at a newer build commit (or after a registry-only
# change) reuse already-materialized reforms instead of recomputing all of them,
# while a change to any of these keys still invalidates the entry (no stale reuse).
REFORM_VECTOR_CACHE_CONTEXT_KEYS: tuple[str, ...] = (
    "base_dataset_sha256",
    "policyengine_us_version",
    "target_period",
    "congressional_district_vintage_crosswalk_sha256",
)
TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION = 1
# 2: the medicaid_take_up stage (populace #331) changed base_frame's
# takes_up_medicaid_if_eligible before target-frame materialization, so
# medicaid_enrolled target columns differ from version-1 checkpoints; the
# checkpoint identity hashes the on-disk base dataset, not the staged frame,
# and would otherwise silently reuse pre-stage frames.
TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION = 2
DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE = 5_000
DEFAULT_L0_REFIT_LAMBDA_SHARE = 0.8
DEFAULT_US_FISCAL_CALIBRATION_EPOCHS = 1_500


def _collect_batch_garbage() -> None:
    """Keep batch loops tidy without traversing the full object graph."""

    gc.collect(0)


@contextmanager
def _automatic_gc_suspended():
    """Avoid surprise full-graph cyclic GC inside memory-heavy PE batches."""

    was_enabled = gc.isenabled()
    if was_enabled:
        gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


US_BASE_PERSON_POPULATION_BENCHMARK = float(sum(CENSUS_NATIONAL_AGE_BENCHMARK.values()))
US_BASE_PERSON_POPULATION_MAX_ABS_RELATIVE_ERROR = 0.25
US_BASE_PERSON_POPULATION_REPAIR_REASON = (
    "US fiscal refresh rescaled base household weights to the Census 2024 "
    "national person-population benchmark before mass='conserve' calibration."
)
US_SOCIAL_SECURITY_COMPONENT_REPAIR_REASON = (
    "US fiscal refresh rescaled Social Security component leaf inputs to SSA "
    "component payment targets from the active fiscal target registry before "
    "mass='conserve' calibration."
)
US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS = frozenset(
    {
        "congressional_district_geoid",
        "geography_scope",
        "hierarchy_child_ids",
        "hierarchy_child_sum_raw",
        "hierarchy_coverage_ratio",
        "hierarchy_expected_child_count",
        "hierarchy_observed_child_count",
        "hierarchy_parent_geography_id",
        "hierarchy_parent_geography_level",
        "hierarchy_parent_key",
        "hierarchy_parent_target_name",
        "hierarchy_parent_target_period",
        "hierarchy_parent_value",
        "hierarchy_raw_value",
        "hierarchy_reconciliation_factor",
        "hierarchy_reconciliation_method",
        "hierarchy_reconciliation_rule",
        "ledger_aggregate_fact_key",
        "ledger_dimension_set_key",
        "ledger_fact_key",
        "ledger_geography_id",
        "ledger_geography_level",
        "ledger_geography_name",
        "ledger_geography_vintage",
        "ledger_legacy_fact_key",
        "ledger_layout_groupby_dimension",
        "ledger_layout_groupby_value_id",
        "ledger_layout_record_set_id",
        "ledger_observed_measure_key",
        "ledger_semantic_fact_key",
        "ledger_source_record_id",
        "state_fips",
    }
)
US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR = 0.25
US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR = 0.15
US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES = {
    "ssa_retirement_total": "social_security_retirement",
    "ssa_disability_total": "social_security_disability",
    "ssa_dependents_total": "social_security_dependents",
    "ssa_survivors_total": "social_security_survivors",
}
US_CRITICAL_TARGET_FIT_REQUIREMENTS = (
    {
        "name": (
            "irs_soi.ty2022.historic_table_2.us.all."
            f"income_tax_liability_amount@{PERIOD}"
        ),
        "label": "federal income tax liability amount",
        "max_abs_relative_error": 0.05,
    },
    {
        "name": (
            "irs_soi.ty2022.historic_table_2.us.all."
            f"income_tax_liability_returns@{PERIOD}"
        ),
        "label": "income tax liability returns",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (
            "ssa_supplement.cy2024.oasdi_ssi_payments."
            f"social_security_benefits.payment_amount@{PERIOD}"
        ),
        "label": "Social Security benefits",
        "max_abs_relative_error": 0.05,
    },
    {
        "name": (f"irs_soi.ty2022.historic_table_2.us.all.ctc_amount@{PERIOD}"),
        "label": "Child Tax Credit amount",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (f"irs_soi.ty2022.historic_table_2.us.all.ctc_claims@{PERIOD}"),
        "label": "Child Tax Credit claims",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (f"irs_soi.ty2022.historic_table_2.us.all.actc_amount@{PERIOD}"),
        "label": "Additional Child Tax Credit amount",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (f"irs_soi.ty2022.historic_table_2.us.all.actc_claims@{PERIOD}"),
        "label": "Additional Child Tax Credit claims",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            f"earned_income_credit.total_earned_income_credit_amount@{PERIOD}"
        ),
        "label": "Earned Income Tax Credit amount",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (
            "irs_soi.ty2024.filing_season_week47.eitc_all_returns."
            f"earned_income_credit.total_earned_income_credit_returns@{PERIOD}"
        ),
        "label": "Earned Income Tax Credit claims",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (
            f"irs_soi.ty2022.historic_table_2.us.all.premium_tax_credit_amount@{PERIOD}"
        ),
        "label": "Premium Tax Credit amount",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (
            "irs_soi.ty2022.historic_table_2.us.all."
            f"premium_tax_credit_returns@{PERIOD}"
        ),
        "label": "Premium Tax Credit returns",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (
            "irs_soi.ty2022.historic_table_2.us.all."
            f"taxable_social_security_amount@{PERIOD}"
        ),
        "label": "taxable Social Security amount",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
    {
        "name": (
            "irs_soi.ty2022.historic_table_2.us.all."
            f"taxable_social_security_returns@{PERIOD}"
        ),
        "label": "taxable Social Security returns",
        "max_abs_relative_error": US_CRITICAL_CREDIT_MAX_ABS_RELATIVE_ERROR,
    },
)

DIRECT_ACTIVE_ALIASES = (
    "census-pep-2024-national-age-sex",
    "census-pep-2024-state-age-sex",
    "cms-aca-oep-state-level",
    "cms-medicaid-chip-monthly-enrollment-december-2024",
    "cms-medicare-trustees-report-2025-part-b-premium-income",
    "census-stc-individual-income-tax",
    "hhs-acf-tanf-caseload-2024",
    "hhs-acf-tanf-financial-2024",
    "jct-tax-expenditures-2024",
    "soi-table-1-1",
    "soi-table-1-2",
    "soi-table-1-4",
    "soi-table-2-1",
    "soi-table-2-5",
    "soi-table-2-5-eitc-agi-children-2023",
    "soi-filing-season-week47-2024-eitc-total",
    "soi-table-4-3",
    "soi-state-2022",
    "soi-historic-table-2",
    "soi-historic-table-2-state-agi-2022",
    "soi-historic-table-2-state-broad-2022",
    "soi-historic-table-2-state-eitc-2022",
    "soi-w2-statistics-2020",
    "ssa-annual-statistical-supplement-2025",
    "usda-snap-fy69-to-current",
)

REVIEWED_EXCLUDED_ALIASES = (
    "bea-nipa-pension-contributions",
    "bea-nipa-personal-income-components",
    "bea-nipa-personal-income-disposition",
    "bea-nipa-total-wages-salaries",
    "census-acs-s0101-congressional-district-age-2024",
    "census-acs-s0101-national-age-2024",
    "census-acs-s0101-state-age-2024",
    "cms-aca-effectuated-enrollment-2022",
    "cms-aca-oep-state-level-2022",
    "cms-aca-oep-state-level-2025",
    "cms-medicaid-chip-monthly-enrollment-dataset",
    "cms-nhe-historical-service-source",
    "hhs-acf-liheap-fy2023-national-profile",
    "hhs-acf-liheap-fy2024-national-profile",
    "soi-congressional-district-2022",
    "ssa-ssi-table-7b1-2024",
)

SUPPORTED_LEDGER_FILTER_METADATA_KEYS = frozenset(
    {
        "ledger_filter_amount_basis",
        "ledger_filter_eitc_child_count",
        "ledger_filter_filing_status",
        "ledger_filter_income_range",
        "ledger_filter_medicare.financing_component",
        "ledger_filter_medicare.part",
        "ledger_filter_tax_expenditure",
        "ledger_filter_us_social_security_and_ssi.program_payment_type",
    }
)

FISCAL_TARGET_SOURCE_KEYS = {
    "cbo": "Congressional Budget Office revenue projections",
    "cms_aca": "CMS ACA marketplace enrollment public use files",
    "cms_medicaid": "CMS Medicaid enrollment and expenditure sources",
    "cms_medicare": "CMS Medicare Trustees Report Part B premium income",
    "hhs_acf_tanf": "HHS ACF TANF administrative data",
    "irs_soi": "IRS Statistics of Income public tables",
    "jct": "Joint Committee on Taxation tax expenditure estimates",
    "ssa": "Social Security Administration statistical supplement",
    "state_income_tax": "Census State Tax Collections individual income tax",
    "usda_snap": "USDA SNAP administrative data",
}

US_HEALTH_INPUT_NONCONSTANT_COLUMNS = (
    "takes_up_aca_if_eligible",
    "selected_marketplace_plan_benchmark_ratio",
)
# Persisted input columns known to be constant at the PolicyEngine-US default
# in current bases. Each is accepted with the issue tracking its fix; a new
# degenerate column, or one of these becoming non-degenerate, fails the
# default-valued-columns gate so this list cannot rot.
US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS = {
    "takes_up_ssi_if_eligible": (
        "SSI take-up imputation backlog; constant True forces 100% take-up."
    ),
    "takes_up_medicare_if_eligible": (
        "Medicare take-up imputation backlog (PolicyEngine/populace#98)."
    ),
    "takes_up_dc_ptc": ("DC PTC take-up imputation backlog; constant True."),
    "second_home_mortgage_balance": (
        "Second-home mortgage decomposition not imputed; constant at the"
        " engine default (PolicyEngine/populace#38)."
    ),
    "second_home_mortgage_interest": (
        "Second-home mortgage decomposition not imputed; constant at the"
        " engine default (PolicyEngine/populace#38)."
    ),
    "second_home_mortgage_origination_year": (
        "Second-home mortgage decomposition not imputed; constant at the"
        " engine default (PolicyEngine/populace#38)."
    ),
    "takes_up_head_start_if_eligible": (
        "Head Start take-up imputation backlog; constant True."
    ),
    "takes_up_early_head_start_if_eligible": (
        "Early Head Start take-up imputation backlog; constant True."
    ),
    # ssn_card_type and immigration_status_str are intentionally NOT excluded:
    # PR #266 imputes them from CPS ASEC citizenship, so a base where they are
    # still constant at CITIZEN skipped that stage and should fail this gate.
    "spm_unit_tenure_type": (
        "SPM tenure input not yet carried through (PolicyEngine/populace#32); "
        "constant RENTER misstates SNAP shelter deductions and SPM housing."
    ),
    "is_wic_at_nutritional_risk": (
        "WIC inputs not yet carried through (PolicyEngine/populace#32)."
    ),
    "would_claim_wic": (
        "WIC take-up inputs not yet carried through (PolicyEngine/populace#32)."
    ),
    "s_corp_income": (
        "Combined partnership/S-corp income is carried in partnership_income "
        "in pre-PUF-support bases; the S-corp leaf is constant zero there."
    ),
    "estate_income_would_be_qualified": (
        "QBI qualification flags default True pending formula-constrained "
        "leaf imputation (PolicyEngine/populace#186)."
    ),
    "farm_operations_income_would_be_qualified": (
        "QBI qualification flags default True pending formula-constrained "
        "leaf imputation (PolicyEngine/populace#186)."
    ),
    "farm_rent_income_would_be_qualified": (
        "QBI qualification flags default True pending formula-constrained "
        "leaf imputation (PolicyEngine/populace#186)."
    ),
    "partnership_s_corp_income_would_be_qualified": (
        "QBI qualification flags default True pending formula-constrained "
        "leaf imputation (PolicyEngine/populace#186)."
    ),
    "rental_income_would_be_qualified": (
        "QBI qualification flags default True pending formula-constrained "
        "leaf imputation (PolicyEngine/populace#186)."
    ),
    "self_employment_income_would_be_qualified": (
        "QBI qualification flags default True pending formula-constrained "
        "leaf imputation (PolicyEngine/populace#186)."
    ),
}

#: Person inputs SNAP work-requirement rules read that have NO CPS ASEC
#: source and are not seeded: they default to False in the engine, so the
#: compliance/exemption channels they drive never fire (populace #351,
#: #249 for the work-program family). They are not
#: persisted columns, so the degenerate-input gate cannot see them; this
#: register makes the assumption visible in every release manifest instead.
#: is_pregnant is NOT here: the pregnancy stage seeds it.
#: Scope note (per #340): this register is for the NO-SURVEY-SOURCE class —
#: structurally unfixable from ASEC. The #340 column families (tips,
#: overtime, education credits, ...) have a source and were merely never
#: persisted; those belong to stage/persistence work, not this register.
#: If a not-persisted register is added later, the two should surface as
#: labeled sub-blocks of one "inputs not reaching the engine" diagnostic.
US_DOCUMENTED_ABSENT_INPUTS = {
    "is_homeless": (
        "No ASEC source: the CPS samples the housed population, so the "
        "pre-HR1 SNAP ABAWD homeless exemption cannot fire "
        "(PolicyEngine/populace#351)."
    ),
    "was_in_foster_care": (
        "No ASEC item for foster-care history, so the pre-HR1 former-"
        "foster-youth ABAWD exemption (7 CFR 273.24(c)(9)) cannot fire "
        "(PolicyEngine/populace#351)."
    ),
    "is_incapable_of_self_care": (
        "No direct ASEC item; the incapacity/caregiving work-registration "
        "exemptions rely on the disability battery only "
        "(PolicyEngine/populace#351)."
    ),
    "is_snap_work_program_participant": (
        "No ASEC item measures SNAP E&T or qualifying work-program "
        "participation, so compliance via program participation never "
        "fires; USDA reports E&T reaches a small minority of "
        "participants, so the always-False default understates "
        "compliance only modestly (PolicyEngine/populace#249)."
    ),
    "weekly_snap_work_program_hours": (
        "No ASEC item measures qualifying work-program hours, so combined "
        "work-plus-program hours toward the 20-hour ABAWD test omit "
        "program hours (PolicyEngine/populace#249)."
    ),
    "is_snap_workfare_participant": (
        "No ASEC item measures workfare participation under 7 CFR "
        "273.7(m), which satisfies the ABAWD requirement regardless of "
        "hours; defaults False (PolicyEngine/populace#249)."
    ),
}
US_ACA_MARKETPLACE_STAGE = "aca_marketplace_inputs"
US_ACA_SOURCE_OUTPUT_COLUMNS = US_HEALTH_INPUT_NONCONSTANT_COLUMNS
US_ACA_REPORTED_SUBSIDIZED_ANCHOR = (
    "reported_has_subsidized_marketplace_health_coverage_at_interview"
)
US_ACA_REPORTED_MARKETPLACE_COVERAGE = "has_marketplace_health_coverage_at_interview"
US_ACA_APTC_TARGET_TABLE = "cms_aca_aptc_recipients_by_state"
US_ACA_TARGET_ROLE_TABLES = {
    "aca_ptc_recipients": US_ACA_APTC_TARGET_TABLE,
    "aca_bronze_aptc_consumers": "cms_aca_bronze_aptc_consumers_by_state",
    "aca_bronze_ptc_consumers": "cms_aca_bronze_aptc_consumers_by_state",
    "aca_below_benchmark_ptc_consumers": "cms_aca_bronze_aptc_consumers_by_state",
    "aca_spending": "irs_soi_premium_tax_credit_amount_by_state",
    "aca_ptc_returns": "irs_soi_premium_tax_credit_returns_by_state",
}
US_ACA_PERSON_COUNT_TARGET_TABLES = frozenset(
    {
        US_ACA_APTC_TARGET_TABLE,
        "cms_aca_bronze_aptc_consumers_by_state",
    }
)
US_MEDICAID_ENROLLMENT_TARGET_ROLE = "medicaid_enrollment"

FILING_STATUS_MAP = {
    "All": None,
    "Head of Household": "HEAD_OF_HOUSEHOLD",
    "Married Filing Jointly/Surviving Spouse": {
        "JOINT",
        "SURVIVING_SPOUSE",
    },
    "Married Filing Separately": "SEPARATE",
    "Single": "SINGLE",
}

SUPPORTED_SOI_LEDGER_FILTERS = frozenset(
    {
        "ledger_filter_income_range",
        "ledger_filter_filing_status",
        "ledger_filter_eitc_child_count",
    }
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-h5",
        type=Path,
        help="Existing Populace US H5 to recalibrate. Defaults to HF latest.",
    )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help=(
            "PolicyEngine Ledger consumer artifact directory (manifest.json "
            "+ consumer_facts.jsonl, hash-verified) or a bare "
            "consumer_facts.jsonl file, used to resolve every fiscal target "
            "value. Populace package resources declare target references "
            "only. The artifact identity is recorded in the build and "
            "release manifests."
        ),
    )
    parser.add_argument(
        "--ledger-facts-sha256",
        help=(
            "Pin: expected SHA-256 of consumer_facts.jsonl. The build "
            "refuses to start if the feed does not match."
        ),
    )
    parser.add_argument(
        "--ledger-manifest-sha256",
        help=(
            "Pin: expected SHA-256 of the Ledger consumer artifact "
            "manifest.json. Requires an artifact directory feed."
        ),
    )
    parser.add_argument(
        "--allow-unaged-dollar-targets",
        action="store_true",
        help=(
            "Waive the period contract: compile observation dollar levels at "
            "a build period other than their fact period without aging "
            "(PolicyEngine/ledger#71; populace#212). Every waived target "
            "carries period_contract_waiver metadata in diagnostics. Without "
            "this flag such targets fail the build unless --age-targets "
            "transforms them."
        ),
    )
    parser.add_argument(
        "--zero-support-exclusions",
        type=Path,
        help=(
            "Optional JSON mapping source_record_id -> reason of per-run, "
            "per-artifact support-expressibility exclusions that augment the "
            "standing US_FISCAL_TARGET_SUPPORT_EXCLUSIONS registry for THIS "
            "build only (PolicyEngine/populace#299 Build G). A sparse "
            "artifact's frozen support cannot populate narrow state/tail cells "
            "the dense parent can; declare those cells here so they do not "
            "fail the zero-support gate, with each reason recorded in the "
            "release manifest. The module registry is never mutated."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--release-id")
    parser.add_argument(
        "--incumbent-diagnostics",
        type=Path,
        help=(
            "Optional calibration_diagnostics.json for the current published "
            "release. Critical targets outside their absolute tolerance can "
            "still pass if they improve on this incumbent row by row."
        ),
    )
    parser.add_argument(
        "--input-mass-reference-h5",
        type=Path,
        help=(
            "Optional certified release H5 whose persisted PolicyEngine input "
            "mass the RAW BASE frame must carry. Catches a rebuilt base "
            "pipeline that silently drops input bases the incumbent populates "
            "(issue #278: IRA/HSA/pension-contribution/childcare inputs). NOTE: "
            "this compares the PRE-calibration base to the reference, so a "
            "certified (calibrated) reference will over-fire on PUF-imputed "
            "income columns the raw base structurally under-reports; use it "
            "only when the reference is itself an uncalibrated base of the same "
            "lineage. For the calibrated-export comparison, use "
            "--export-input-mass-reference-h5 (populace#327)."
        ),
    )
    parser.add_argument(
        "--export-input-mass-reference-h5",
        type=Path,
        help=(
            "Optional certified release H5 whose persisted PolicyEngine input "
            "mass the CALIBRATED EXPORT frame is compared against, instead of "
            "the raw pre-calibration base (populace#327). Calibration correctly "
            "scales PUF-imputed income up toward SOI/CBO targets, so comparing "
            "the export to the raw base flags those correct gains; a certified "
            "reference puts them in-band while a genuine #278 zeroing still "
            "fails. Distinct from --input-mass-reference-h5, which gates the "
            "raw base and must not use a calibrated reference."
        ),
    )
    parser.add_argument(
        "--input-mass-relative-tolerance",
        type=float,
        default=0.5,
        help=(
            "Maximum allowed relative drift of a persisted input column's "
            "weighted total in the input-mass parity gates."
        ),
    )
    parser.add_argument(
        "--input-mass-minimum-reference-total",
        type=float,
        default=1e9,
        help=(
            "Reference-mass floor below which input-mass parity is not "
            "checked; relative drift on near-zero totals is meaningless."
        ),
    )
    parser.add_argument(
        "--allow-input-mass-drift",
        action="store_true",
        help=(
            "Diagnostic escape hatch: record input-mass parity gate results "
            "without failing the build."
        ),
    )
    parser.add_argument(
        "--allow-ecps-parity-gaps",
        action="store_true",
        help=(
            "Diagnostic escape hatch: record the eCPS parity gate result "
            "(incumbent-populated layers the candidate leaves empty and "
            "unexempted) without failing the build."
        ),
    )
    parser.add_argument(
        "--allow-input-coverage-gaps",
        action="store_true",
        help=(
            "Diagnostic escape hatch (populace#368): record the release "
            "input-column coverage gate result — required eCPS input columns "
            "the export drops or leaves degenerate — without failing the "
            "build. Release builds must leave this unset; the gate is a hard "
            "certification blocker."
        ),
    )
    parser.add_argument(
        "--skip-reform-coverage-smoke",
        action="store_true",
        help=(
            "Do not run the reform-coverage smoke gate (populace#368): the "
            "pinned bound-reform probes (SSI $10k/$20k asset limits) that must "
            "score nonzero on the written release. Skipping loses the "
            "end-to-end $0-reform backstop; release builds should leave it on."
        ),
    )
    parser.add_argument(
        "--allow-reform-coverage-smoke-failures",
        action="store_true",
        help=(
            "Diagnostic escape hatch (populace#368): record the reform-coverage "
            "smoke gate result without failing the build when a bound reform "
            "probe scores ~$0. Release builds must leave this unset."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_US_FISCAL_CALIBRATION_EPOCHS,
        help=(
            "Optimization epochs for each calibration stage. The default "
            "matches the current L0+refit release run: L0 selection for 1,500 "
            "epochs followed by a 1,500-epoch ordinary refit."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-weight-ratio", type=float, default=5.0)
    parser.add_argument(
        "--l0-refit-lambda-share",
        type=float,
        default=DEFAULT_L0_REFIT_LAMBDA_SHARE,
        help=(
            "Default national dataset sparsity control. The builder divides "
            "this value by the candidate household count and uses the result "
            "as the fixed L0 penalty before refitting ordinary calibration on "
            "the selected support. The default 0.8 reproduces the current "
            "57k-household US fiscal surface run from the 337k support."
        ),
    )
    parser.add_argument(
        "--l2-lambda",
        type=float,
        default=0.0,
        help=(
            "Soft L2 concentration penalty on the mean squared "
            "calibrated-to-initial weight ratio. The default 0.0 preserves "
            "the unpenalized objective; positive values trade target fit for "
            "a higher effective sample size (max-weight-ratio stays the hard "
            "cap). Under the default L0+refit path the same penalty applies "
            "to both stages unless --refit-l2-lambda overrides the refit."
        ),
    )
    parser.add_argument(
        "--refit-l2-lambda",
        type=float,
        default=None,
        help=(
            "Override the L2 concentration penalty for the post-L0 refit "
            "stage only — the stage that produces the shipped weights. "
            "Defaults to --l2-lambda. Requires the sparse L0+refit default "
            "dataset (incompatible with --dense-default-dataset, which has "
            "no refit stage)."
        ),
    )
    parser.add_argument(
        "--dense-default-dataset",
        action="store_true",
        help=(
            "Diagnostic only: publish the dense no-L0 calibrated frame as the "
            "default national dataset. Release builds should leave this unset "
            "so populace_us_2024.h5 is the sparse L0+refit dataset that runs "
            "on standard machines."
        ),
    )
    parser.add_argument(
        "--warm-start-calibration-npz",
        type=Path,
        help=(
            "Optional populace_us_2024_calibration.npz artifact from an earlier "
            "run on the same base frame. The builder validates that the stored "
            "initial_household_weight vector matches the current calibration "
            "frame before using household_weight as the optimizer start."
        ),
    )
    parser.add_argument(
        "--selection-source-h5",
        type=Path,
        help=(
            "Optional populace US H5 whose record set defines a frozen support "
            "to recover onto the base pool (populace#328). The base frame is "
            "reduced to exactly the source's records — matched by stable source "
            "identity (see --selection-join-key), not row order — before target "
            "materialization and calibration. This reconstructs the certified "
            "default's informed-L0/warm-start step as committed machinery. "
            "Mutually exclusive with --selection-source-manifest."
        ),
    )
    parser.add_argument(
        "--selection-source-manifest",
        type=Path,
        help=(
            "Optional committed selection-source manifest JSON (produced by "
            "tools/build_us_selection_source_manifest.py) naming the frozen "
            "support's record identities directly, so the sparse default is "
            "reproducible on a laptop without downloading the source H5. "
            "Mutually exclusive with --selection-source-h5."
        ),
    )
    parser.add_argument(
        "--selection-join-key",
        default=",".join(DEFAULT_SELECTION_JOIN_KEY),
        help=(
            "Comma-separated stable-identity columns used to match the selection "
            "source's records onto the base pool. Defaults to "
            f"{','.join(DEFAULT_SELECTION_JOIN_KEY)!r}. Only stable identity "
            "columns are allowed; assigned row ids are rejected because they are "
            "order-dependent across a base rebuild."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=SELECTION_MODES,
        default="frozen_support",
        help=(
            "How the selection source is applied. 'frozen_support' (default) "
            "reduces the base to exactly the named records and refuses any "
            "unmapped identity. 'informed_init' is reserved for a future "
            "drifting rebuild and is not yet wired into calibration."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scf-summary-extract",
        dest="scf_summary_extract",
        default=None,
        help=(
            "Path to the Federal Reserve SCF 2022 public summary extract "
            "(rscfp2022.dta) that feeds the SSI countable-resource asset "
            "imputation (scf_wealth stage, populace#356/#368). When omitted the "
            "fixed-vintage extract is fetched and cached from the Federal "
            "Reserve."
        ),
    )
    parser.add_argument(
        "--scf-full-extract",
        type=Path,
        help=(
            "Optional path to the Federal Reserve SCF 2022 full public Stata "
            "extract (p22i6.dta) used for household auto-loan balance and "
            "interest. When omitted, scf2022s.zip is fetched and cached."
        ),
    )
    parser.add_argument(
        "--sipp-tip-donor",
        type=Path,
        help=(
            "Optional local path to the sha-pinned SIPP 2023 slim CSV that "
            "feeds tip_income and Treasury tipped-occupation coverage. When "
            "omitted the immutable donor revision is fetched and verified."
        ),
    )
    parser.add_argument(
        "--org-wages-donor",
        type=Path,
        help=(
            "Optional local path to the canonical transformed 2024 CPS ORG "
            "donor cache. When omitted, the twelve official 2024 CPS "
            "basic-month ORG files are fetched, transformed, and verified "
            "against the pinned canonical donor-content SHA-256."
        ),
    )
    parser.add_argument(
        "--maximum-microsim-batch-size",
        "--maximum-microsimulation-batch-size",
        dest="maximum_microsim_batch_size",
        type=int,
        default=DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
        help=(
            "Maximum households per PolicyEngine microsimulation batch. "
            "Use 0 to run each requested microsimulation on the full dataset "
            "at once."
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help=(
            "Root directory for durable, restart-surviving checkpoints "
            "(the per-reform target-materialization cache and the full "
            "target-frame checkpoint). When set, these default to "
            "<checkpoint-root>/target_materialization_cache and "
            "<checkpoint-root>/target_frame_checkpoint.h5 instead of under "
            "<out>/artifacts, so a run that restarts into a fresh --out (for "
            "example a preempted Modal worker) still finds already-completed "
            "reform materializations. Point this at a persistent volume path "
            "that is stable across worker restarts. The explicit "
            "--target-materialization-cache-dir / --target-frame-checkpoint "
            "overrides still take precedence."
        ),
    )
    parser.add_argument(
        "--target-materialization-cache-dir",
        type=Path,
        help=(
            "Override the standard target-materialization checkpoint "
            "directory. Defaults to <checkpoint-root>/"
            "target_materialization_cache when --checkpoint-root is set, "
            "otherwise <out>/artifacts/target_materialization_cache. The cache "
            "stores expensive per-household target materialization artifacts "
            "such as JCT reform income-tax vectors and is content-addressed by "
            "base H5, policyengine-us version, period, geography crosswalk, and "
            "reform (see #217)."
        ),
    )
    parser.add_argument(
        "--no-target-materialization-cache",
        action="store_true",
        help=(
            "Diagnostic only: disable the standard target-materialization "
            "checkpoint. Release builds should leave caching enabled."
        ),
    )
    parser.add_argument(
        "--target-frame-checkpoint",
        type=Path,
        help=(
            "Override the standard materialized target-frame checkpoint path. "
            "Defaults to <checkpoint-root>/target_frame_checkpoint.h5 when "
            "--checkpoint-root is set, otherwise "
            "<out>/artifacts/target_frame_checkpoint.h5. This checkpoint stores "
            "the full post-source, post-target-materialized Frame used by "
            "calibration, so warm-start-only reruns can skip PolicyEngine "
            "materialization entirely when the input identity matches."
        ),
    )
    parser.add_argument(
        "--no-target-frame-checkpoint",
        action="store_true",
        help=(
            "Diagnostic only: disable the full target-frame checkpoint. "
            "This does not affect the lower-level per-reform target cache."
        ),
    )
    parser.add_argument(
        "--audit-export-targets",
        action="store_true",
        help=(
            "After writing the H5, reload it and rematerialize the full fiscal "
            "target surface. This is a slow audit pass; default release builds "
            "rely on the writer's H5 round-trip verification and calibration "
            "diagnostics instead."
        ),
    )
    parser.add_argument(
        "--skip-reform-validation",
        action="store_true",
        help="Do not emit reform_validation.json for this release.",
    )
    parser.add_argument(
        "--skip-out-of-sample-reforms",
        action="store_true",
        help=(
            "Emit reform_validation.json with the in-sample JCT tax-expenditure "
            "rows only (from the calibration fit), skipping the out-of-sample "
            "OBBBA simulations. Faster; useful when policyengine-us microsim runs "
            "are not wanted in the build."
        ),
    )
    parser.add_argument(
        "--diagnostic-skip-tax-expenditure-targets",
        action="store_true",
        help=(
            "Diagnostic only: drop JCT tax-expenditure calibration targets so "
            "local target materialization can skip reform simulations. Do not "
            "use for publishable releases."
        ),
    )
    parser.add_argument(
        "--include-congressional-district-targets",
        action="store_true",
        help=(
            "Opt into SOI congressional-district target rows. Requires the "
            "support frame to contain household congressional_district_geoid."
        ),
    )
    parser.add_argument(
        "--congressional-district-vintage-crosswalk",
        type=Path,
        help=(
            "Source-to-current congressional-district crosswalk "
            "artifact with source_geography_id, target_geography_id, and "
            "weight columns. Required when congressional-district targets "
            "are requested."
        ),
    )
    parser.add_argument(
        "--gate-congressional-district-targets",
        action="store_true",
        help=(
            "Treat congressional-district targets as hard release gates. "
            "The default small-dataset path keeps them diagnostic-only because "
            "sparse CD support can make zero-support rows expected rather than "
            "release blockers."
        ),
    )
    parser.add_argument(
        "--skip-demographics",
        action="store_true",
        help="Do not emit demographics.json (weighted population by age) for this release.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help=(
            "Optional local directory for staging telemetry artifacts. Defaults "
            "to <out>/staging/runs/<run_id> when --staging-repo-id is set."
        ),
    )
    parser.add_argument(
        "--staging-repo-id",
        default=os.environ.get(
            "POPULACE_STAGING_REPO_ID", "policyengine/populace-us-staging"
        ),
        help=(
            "Hugging Face dataset repo to upload staging telemetry to while "
            "the build runs. On by default (uploads are best-effort and never "
            "fail the build); override with POPULACE_STAGING_REPO_ID or "
            "disable with --no-staging."
        ),
    )
    parser.add_argument(
        "--age-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compile-time period aging of dollar-amount targets whose source "
            "period differs from the build period (PolicyEngine/populace#116, "
            "#212), on by default under the named cbo_growth_factor_aging "
            "model: dollar amounts are scaled by CBO revenue-projection "
            "growth ratios drawn from the Ledger facts feed (matching "
            "income-source series where available, CBO AGI growth "
            "otherwise); counts stay raw. Each target records basis/"
            "source_period/aged_to/aging_factor/aging_factor_source/"
            "alignment_model_id/alignment_model_version diagnostics. "
            "--no-age-targets disables the transform, in which case "
            "cross-period observation dollars fail the period contract "
            "unless --allow-unaged-dollar-targets waives it."
        ),
    )
    parser.add_argument(
        "--no-staging",
        action="store_true",
        help="Disable staging telemetry (local staging dir and uploads) for this build.",
    )
    parser.add_argument(
        "--staging-prefix",
        default=os.environ.get("POPULACE_STAGING_PREFIX", "runs"),
        help=(
            "Repo prefix for staging run artifacts. Defaults to "
            "POPULACE_STAGING_PREFIX or runs."
        ),
    )
    parser.add_argument(
        "--staging-run-id",
        help="Override the staging run id. Defaults to the candidate release id.",
    )
    parser.add_argument(
        "--staging-upload-interval-seconds",
        type=float,
        default=30.0,
        help="Minimum seconds between progress uploads to the staging repo.",
    )
    args = parser.parse_args()
    if (
        args.include_congressional_district_targets
        and args.congressional_district_vintage_crosswalk is None
    ):
        parser.error(
            "--congressional-district-vintage-crosswalk is required when "
            "--include-congressional-district-targets is set."
        )
    if not args.dense_default_dataset and not (
        math.isfinite(args.l0_refit_lambda_share) and args.l0_refit_lambda_share > 0.0
    ):
        parser.error(
            "--l0-refit-lambda-share must be positive unless "
            "--dense-default-dataset is set."
        )
    if args.refit_l2_lambda is not None and args.dense_default_dataset:
        parser.error(
            "--refit-l2-lambda requires the sparse L0+refit default dataset; "
            "--dense-default-dataset has no refit stage (use --l2-lambda)."
        )
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_versions() -> dict[str, str]:
    packages = (
        "populace-build",
        "populace-calibrate",
        "populace-data",
        "populace-frame",
        "policyengine-core",
        "policyengine-us",
        "numpy",
        "pandas",
        "torch",
    )
    versions = {"python": platform.python_version()}
    for package in packages:
        versions[package] = _package_or_workspace_version(package)
    return versions


def _package_or_workspace_version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return _local_workspace_package_version(package)


def _strict_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _strict_json_text(value: object, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=None if indent is not None else (",", ":"),
        indent=indent,
        allow_nan=False,
    )


def _reform_vector_cache_context(context: Mapping[str, object]) -> dict[str, object]:
    """Project the full build context onto the #217 reform-vector key subset.

    Only keys that change a reform's per-household income-tax estimate are kept
    (see ``REFORM_VECTOR_CACHE_CONTEXT_KEYS``). Keys present in ``context`` are
    carried through (including ``None`` values, so a geography-independent build
    that omits the crosswalk sha is distinct from one that sets it); absent keys
    are simply not part of the identity.
    """
    return {
        key: context[key] for key in REFORM_VECTOR_CACHE_CONTEXT_KEYS if key in context
    }


def _target_materialization_cache_identity(
    *,
    context: Mapping[str, object],
    reform_spec,
    n_households: int,
) -> dict[str, object]:
    return {
        "schema_version": TARGET_MATERIALIZATION_CACHE_SCHEMA_VERSION,
        "kind": "jct_reform_income_tax_by_household",
        "country": "us",
        "period": PERIOD,
        "n_households": int(n_households),
        "reform_measure": str(reform_spec.measure),
        "neutralized_variable": str(reform_spec.neutralized_variable),
        # #217: reform-vector identity uses only the inputs that determine the
        # per-household estimate. Build commit / seed / registry version are
        # intentionally excluded so calibration-only or commit-only reruns reuse
        # the cache; base H5 / PE-US version / period / geography still invalidate.
        "context": dict(sorted(_reform_vector_cache_context(context).items())),
    }


def _target_materialization_cache_digest(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(_strict_json_bytes(identity)).hexdigest()


def _cache_safe_name(value: str) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in value)
    safe = "-".join(part for part in safe.split("-") if part)
    return safe[:80] or "target"


def _target_materialization_cache_paths(
    cache_dir: Path,
    identity: Mapping[str, object],
) -> tuple[str, Path, Path]:
    digest = _target_materialization_cache_digest(identity)
    measure = _cache_safe_name(str(identity["reform_measure"]))
    stem = f"{measure}-{digest[:16]}"
    return digest, cache_dir / f"{stem}.json", cache_dir / f"{stem}.npy"


def _read_reform_income_tax_cache(
    cache_dir: Path,
    identity: Mapping[str, object],
    *,
    n_households: int,
) -> tuple[np.ndarray, str, Path] | None:
    digest, metadata_path, values_path = _target_materialization_cache_paths(
        cache_dir,
        identity,
    )
    if not metadata_path.exists() or not values_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("identity") != identity:
        raise RuntimeError(
            "Target materialization cache metadata identity mismatch for "
            f"{metadata_path}."
        )
    metadata_values = metadata.get("values")
    if not isinstance(metadata_values, Mapping):
        raise RuntimeError(
            "Target materialization cache metadata is missing values entry for "
            f"{metadata_path}."
        )
    if metadata_values.get("file") != values_path.name:
        raise RuntimeError(
            "Target materialization cache values filename mismatch for "
            f"{metadata_path}: got {metadata_values.get('file')!r}, expected "
            f"{values_path.name!r}."
        )
    expected_sha256 = metadata_values.get("sha256")
    actual_sha256 = _sha256(values_path)
    if expected_sha256 != actual_sha256:
        raise RuntimeError(
            "Target materialization cache values hash mismatch for "
            f"{values_path}: got {actual_sha256}, expected {expected_sha256}."
        )
    values = np.load(values_path, allow_pickle=False)
    if values.shape != (n_households,):
        raise RuntimeError(
            "Target materialization cache shape mismatch for "
            f"{values_path}: got {values.shape}, expected {(n_households,)}."
        )
    return values.astype(np.float64, copy=False), digest, values_path


def _write_reform_income_tax_cache(
    cache_dir: Path,
    identity: Mapping[str, object],
    values: np.ndarray,
) -> tuple[str, Path]:
    digest, metadata_path, values_path = _target_materialization_cache_paths(
        cache_dir,
        identity,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_values_path = values_path.with_name(f"{values_path.name}.tmp")
    tmp_metadata_path = metadata_path.with_name(f"{metadata_path.name}.tmp")
    values_array = np.asarray(values, dtype=np.float64)
    with tmp_values_path.open("wb") as stream:
        np.save(stream, values_array, allow_pickle=False)
    metadata = {
        "identity": identity,
        "values": {
            "file": values_path.name,
            "dtype": "float64",
            "shape": [int(values_array.shape[0])],
            "sha256": _sha256(tmp_values_path),
        },
    }
    tmp_metadata_path.write_text(
        _strict_json_text(metadata, indent=1) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_values_path, values_path)
    os.replace(tmp_metadata_path, metadata_path)
    return digest, values_path


def _target_frame_checkpoint_identity(
    *,
    base_dataset_sha256: str,
    policyengine_us_version: str,
    seed: int,
    target_period: int,
    target_registry_version: str,
    congressional_district_vintage_crosswalk_sha256: object,
) -> dict[str, object]:
    return {
        "schema_version": TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION,
        "materializer_version": TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION,
        "kind": "us_fiscal_refresh_target_frame",
        "country": "us",
        "base_dataset_sha256": str(base_dataset_sha256),
        "policyengine_us_version": str(policyengine_us_version),
        "seed": int(seed),
        "target_period": int(target_period),
        "target_registry_version": str(target_registry_version),
        "congressional_district_vintage_crosswalk_sha256": (
            None
            if congressional_district_vintage_crosswalk_sha256 is None
            else str(congressional_district_vintage_crosswalk_sha256)
        ),
    }


def _target_frame_checkpoint_digest(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(_strict_json_bytes(identity)).hexdigest()


def _write_target_frame_checkpoint(
    path: Path,
    *,
    frame: Frame,
    identity: Mapping[str, object],
    compilation: Mapping[str, object],
) -> dict[str, object]:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["schema_version"] = TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION
        h5.attrs["identity_json"] = _strict_json_text(identity)
        h5.attrs["identity_sha256"] = _target_frame_checkpoint_digest(identity)
        h5.attrs["compilation_json"] = _strict_json_text(compilation)
        tables_group = h5.create_group("tables")
        for entity in frame.entities:
            _write_checkpoint_dataframe(
                tables_group.create_group(entity), frame.table(entity)
            )
        weights_group = h5.create_group("weights")
        weights_group.attrs["entities_json"] = _strict_json_text(
            frame.weighted_entities
        )
        for entity in frame.weighted_entities:
            weights = frame.weights_for(entity)
            entity_group = weights_group.create_group(entity)
            entity_group.attrs["kind"] = weights.kind.value
            entity_group.create_dataset(
                "values",
                data=np.asarray(weights.values, dtype=np.float64),
                compression="gzip",
            )
        _write_checkpoint_series(h5.create_group("strata"), frame.strata)
    os.replace(tmp_path, path)
    return {
        "enabled": True,
        "status": "miss_written",
        "path": str(path),
        "identity_sha256": _target_frame_checkpoint_digest(identity),
        "schema_version": TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION,
    }


def _read_target_frame_checkpoint(
    path: Path,
    *,
    identity: Mapping[str, object],
    target_specs: tuple,
    gate_congressional_district_targets: bool = True,
) -> tuple[Frame, TargetRegistry, dict[str, object]] | None:
    if not path.exists():
        return None
    import h5py

    with h5py.File(path, "r") as h5:
        schema_version = int(h5.attrs.get("schema_version", -1))
        if schema_version != TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                "Target-frame checkpoint schema mismatch for "
                f"{path}: got {schema_version}, expected "
                f"{TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION}."
            )
        stored_identity = json.loads(str(h5.attrs["identity_json"]))
        if stored_identity != dict(identity):
            return None
        stored_digest = str(h5.attrs.get("identity_sha256", ""))
        expected_digest = _target_frame_checkpoint_digest(identity)
        if stored_digest != expected_digest:
            raise RuntimeError(
                "Target-frame checkpoint identity hash mismatch for "
                f"{path}: got {stored_digest}, expected {expected_digest}."
            )
        tables_group = h5["tables"]
        tables = {
            entity: _read_checkpoint_dataframe(tables_group[entity])
            for entity in US_SCHEMA.entities
        }
        weights_group = h5["weights"]
        weighted_entities = json.loads(str(weights_group.attrs["entities_json"]))
        weights = {}
        for entity in weighted_entities:
            entity_group = weights_group[entity]
            kind = WeightKind(str(entity_group.attrs["kind"]))
            values = np.asarray(entity_group["values"], dtype=np.float64)
            weights[str(entity)] = Weights(values, kind)
        strata = _read_checkpoint_series(h5["strata"])
        stored_compilation = json.loads(str(h5.attrs.get("compilation_json", "{}")))
    frame = Frame(tables, US_SCHEMA, weights, strata)
    registry, compilation = _compile_materialized_target_registry(
        frame,
        target_specs,
        gate_congressional_district_targets=gate_congressional_district_targets,
    )
    compilation = {
        **compilation,
        "target_frame_checkpoint": {
            "enabled": True,
            "status": "hit",
            "path": str(path),
            "identity_sha256": _target_frame_checkpoint_digest(identity),
            "schema_version": TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION,
            "stored_compilation": stored_compilation,
        },
        "target_materialization_cache": {
            "enabled": False,
            "status": "skipped_target_frame_checkpoint_hit",
        },
    }
    return frame, registry, compilation


def _write_checkpoint_dataframe(group, frame: pd.DataFrame) -> None:
    group.attrs["columns_json"] = _strict_json_text(
        [str(column) for column in frame.columns]
    )
    columns_group = group.create_group("columns")
    for index, column in enumerate(frame.columns):
        column_group = columns_group.create_group(f"{index:05d}")
        column_group.attrs["name"] = str(column)
        _write_checkpoint_column(column_group, frame[column])


def _read_checkpoint_dataframe(group) -> pd.DataFrame:
    columns = json.loads(str(group.attrs["columns_json"]))
    data: dict[str, np.ndarray] = {}
    columns_group = group["columns"]
    for index, column in enumerate(columns):
        column_group = columns_group[f"{index:05d}"]
        stored_name = str(column_group.attrs["name"])
        if stored_name != column:
            raise RuntimeError(
                "Target-frame checkpoint column order mismatch: "
                f"slot {index} is {stored_name!r}, expected {column!r}."
            )
        data[column] = _read_checkpoint_column(column_group)
    return pd.DataFrame(data, columns=columns)


def _write_checkpoint_series(group, series: pd.Series) -> None:
    group.attrs["name"] = "" if series.name is None else str(series.name)
    _write_checkpoint_column(group, series)


def _read_checkpoint_series(group) -> pd.Series:
    return pd.Series(
        _read_checkpoint_column(group), name=str(group.attrs.get("name", "")) or None
    )


def _write_checkpoint_column(group, series: pd.Series) -> None:
    import h5py

    dtype = series.dtype
    group.attrs["pandas_dtype"] = str(dtype)
    if pd.api.types.is_bool_dtype(dtype):
        group.attrs["storage_kind"] = "bool"
        values = series.to_numpy(dtype=np.bool_)
        group.create_dataset("values", data=values, compression="gzip")
    elif pd.api.types.is_integer_dtype(dtype):
        if bool(series.isna().any()):
            raise ValueError(
                f"Target-frame checkpoint cannot serialize missing integer column "
                f"{series.name!r}."
            )
        group.attrs["storage_kind"] = "int64"
        values = series.to_numpy(dtype=np.int64)
        group.create_dataset("values", data=values, compression="gzip")
    elif pd.api.types.is_float_dtype(dtype):
        group.attrs["storage_kind"] = "float64"
        values = series.to_numpy(dtype=np.float64)
        group.create_dataset("values", data=values, compression="gzip")
    else:
        group.attrs["storage_kind"] = "string"
        string_dtype = h5py.string_dtype(encoding="utf-8")
        values = np.asarray(
            [
                "" if pd.isna(value) else str(value)
                for value in series.to_numpy(dtype=object)
            ],
            dtype=object,
        )
        group.create_dataset(
            "values", data=values, dtype=string_dtype, compression="gzip"
        )


def _read_checkpoint_column(group) -> np.ndarray:
    storage_kind = str(group.attrs["storage_kind"])
    dataset = group["values"]
    if storage_kind == "string":
        return np.asarray(dataset.asstr()[()], dtype=object)
    if storage_kind == "bool":
        return np.asarray(dataset[()], dtype=np.bool_)
    if storage_kind == "int64":
        return np.asarray(dataset[()], dtype=np.int64)
    if storage_kind == "float64":
        return np.asarray(dataset[()], dtype=np.float64)
    raise RuntimeError(f"Unknown checkpoint storage kind {storage_kind!r}.")


def _resolve_checkpoint_paths(
    args: argparse.Namespace,
    *,
    artifact_root: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    """Resolve the durable checkpoint locations from parsed CLI arguments.

    Returns ``(checkpoint_root, target_materialization_cache_dir,
    target_frame_checkpoint_path)``.

    Both checkpoint locations DEFAULT under ``artifact_root`` (i.e.
    ``<out>/artifacts``), which is ephemeral: a preempted worker that restarts
    into a fresh ``--out`` would orphan the completed reform materializations
    written under the dead out-dir (the root cause of Build B losing its cache).
    When ``--checkpoint-root`` is set, the defaults instead resolve under that
    root — point it at a persistent volume that is stable across worker
    restarts so a fresh ``--out`` still finds already-completed reforms.

    Precedence for each location: an explicit ``--target-materialization-cache-dir``
    / ``--target-frame-checkpoint`` override always wins; otherwise the default
    is taken under ``checkpoint_root`` when provided, else under ``artifact_root``.
    The diagnostic ``--no-*`` flags still force the corresponding location to
    ``None`` (caching/checkpointing disabled) regardless of the root.
    """
    checkpoint_root = getattr(args, "checkpoint_root", None)
    default_root = checkpoint_root if checkpoint_root is not None else artifact_root

    target_materialization_cache_dir = (
        None
        if args.no_target_materialization_cache
        else (
            args.target_materialization_cache_dir
            or default_root / "target_materialization_cache"
        )
    )
    target_frame_checkpoint_path = (
        None
        if args.no_target_frame_checkpoint
        else (
            args.target_frame_checkpoint or default_root / "target_frame_checkpoint.h5"
        )
    )
    return (
        checkpoint_root,
        target_materialization_cache_dir,
        target_frame_checkpoint_path,
    )


def _local_workspace_package_version(package: str) -> str:
    pyproject = Path("packages") / package / "pyproject.toml"
    if not pyproject.is_file():
        return "not-installed"
    with pyproject.open("rb") as file:
        project = tomllib.load(file).get("project")
    if not isinstance(project, Mapping):
        return "not-installed"
    version = project.get("version")
    return version if isinstance(version, str) and version else "not-installed"


def _git_output(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()


def _git_dirty() -> bool:
    return bool(_git_output("status", "--porcelain"))


def _download_base_h5() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=DATASET_FILENAME,
            repo_type="dataset",
        )
    )


def _load_frame(path: Path) -> Frame:
    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(path))
    tables = {
        "person": dataset.person.copy(),
        "household": dataset.household.copy(),
        "tax_unit": dataset.tax_unit.copy(),
        "spm_unit": dataset.spm_unit.copy(),
        "family": dataset.family.copy(),
        "marital_unit": dataset.marital_unit.copy(),
    }
    weights = tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )


def _resolve_selection_source(args):
    """Build the frozen-support selection source from CLI args, or None.

    Reconstructs the certified informed-L0/warm-start selection (populace#328) as
    a committed input: either a source H5 whose record set defines the support, or
    a committed selection-source manifest naming the identities directly. Returns
    ``(source, join_key)`` or ``(None, join_key)`` when no selection is requested.
    """
    join_key = tuple(
        part.strip() for part in str(args.selection_join_key).split(",") if part.strip()
    )
    if not join_key:
        join_key = DEFAULT_SELECTION_JOIN_KEY
    if (
        args.selection_source_h5 is not None
        and args.selection_source_manifest is not None
    ):
        raise ValueError(
            "Pass at most one of --selection-source-h5 / --selection-source-manifest."
        )
    if args.selection_source_manifest is not None:
        source = load_selection_source_from_manifest(args.selection_source_manifest)
        if tuple(source.join_key) != join_key and (
            args.selection_join_key != ",".join(DEFAULT_SELECTION_JOIN_KEY)
        ):
            raise ValueError(
                "--selection-join-key "
                f"{join_key} conflicts with the manifest's own join key "
                f"{tuple(source.join_key)}; omit the flag to use the manifest's."
            )
        return source, tuple(source.join_key)
    if args.selection_source_h5 is not None:
        source_frame = _load_frame(args.selection_source_h5)
        provenance = {
            "kind": "h5",
            "path": str(args.selection_source_h5),
            "sha256": _sha256(args.selection_source_h5),
        }
        source = load_selection_source_from_h5(
            source_frame, join_key=join_key, provenance=provenance
        )
        return source, join_key
    return None, join_key


def _assert_cd_vintage_support_matches(
    h5_path: Path,
    crosswalk_metadata: Mapping[str, object] | None,
) -> None:
    if crosswalk_metadata is None:
        return
    expected_sha256 = str(crosswalk_metadata.get("sha256") or "")
    support_provenance = _read_cd_vintage_support_provenance(h5_path)
    actual_sha256 = support_provenance.get(
        CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR
    )
    actual_target = support_provenance.get(CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR)
    failures: list[str] = []
    if actual_sha256 != expected_sha256:
        failures.append(
            f"crosswalk sha256 {actual_sha256!r} != expected {expected_sha256!r}"
        )
    if actual_target != CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE:
        failures.append(
            "target vintage "
            f"{actual_target!r} != expected {CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE!r}"
        )
    cd_lookup = support_provenance.get("household_congressional_district_geoid")
    if not isinstance(cd_lookup, Mapping) or not cd_lookup.get("exists"):
        failures.append("missing household congressional_district_geoid lookup column")
    elif int(cd_lookup.get("positive_unique_count") or 0) <= 0:
        failures.append("household congressional_district_geoid has no positive values")
    if failures:
        raise ValueError(
            "Congressional-district support crosswalk provenance mismatch in "
            f"{h5_path}: " + "; ".join(failures)
        )


def _read_cd_vintage_support_provenance(h5_path: Path) -> dict[str, object]:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Reading congressional-district support provenance requires h5py. "
            "Run the fiscal refresh builder with the US extra, for example "
            "`uv run --python 3.13 --package populace-build --extra us --group "
            "dev python tools/build_us_fiscal_refresh_release.py ...`. This "
            "preflight is intentionally before calibration or donor imputation."
        ) from exc

    with h5py.File(h5_path, "r") as h5:
        return {
            CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: _h5_attr_text(
                h5.attrs, CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR
            ),
            CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: _h5_attr_text(
                h5.attrs, CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR
            ),
            "household_congressional_district_geoid": (
                _h5_table_column_status(
                    h5,
                    table="household",
                    column="congressional_district_geoid",
                )
            ),
        }


def _h5_table_column_status(h5, *, table: str, column: str) -> dict[str, object]:
    table_path = f"{table}/table"
    if table_path not in h5:
        return {"exists": False, "table": table, "column": column}
    dataset = h5[table_path]
    names = getattr(dataset.dtype, "names", None) or ()
    if column not in names:
        return {"exists": False, "table": table, "column": column}
    values = np.asarray(dataset[column])
    return {
        "exists": True,
        "table": table,
        "column": column,
        "rows": int(values.shape[0]),
        "positive_unique_count": _positive_numeric_unique_count(values),
    }


def _positive_numeric_unique_count(values: np.ndarray) -> int:
    positive: set[float] = set()
    for value in np.asarray(values).ravel():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric_value) and numeric_value > 0:
            positive.add(numeric_value)
    return len(positive)


def _h5_attr_text(attrs: Mapping[str, object], key: str) -> str | None:
    value = attrs.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.bytes_):
        return value.decode()
    return str(value)


def _load_incumbent_diagnostics_payload(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} is not a Populace calibration_diagnostics.json file: "
            "expected a JSON object."
        )
    return payload


def _diagnostics_by_target_name(
    payload: Mapping[str, object],
    *,
    path: Path | None = None,
) -> dict[str, Mapping[str, object]]:
    targets = payload.get("targets")
    if not isinstance(targets, list):
        label = str(path) if path is not None else "diagnostics payload"
        raise ValueError(
            f"{label} is not a Populace calibration_diagnostics.json file: "
            "missing targets list."
        )
    diagnostics: dict[str, Mapping[str, object]] = {}
    for target in targets:
        if not isinstance(target, Mapping):
            continue
        name = target.get("name")
        if isinstance(name, str) and name:
            diagnostics[name] = target
    return diagnostics


def _load_incumbent_diagnostics(
    path: Path | None,
) -> dict[str, Mapping[str, object]]:
    if path is None:
        return {}
    return _diagnostics_by_target_name(
        _load_incumbent_diagnostics_payload(path),
        path=path,
    )


def _assert_incumbent_target_surface_matches(
    current_target_surface: Mapping[str, object],
    incumbent_payload: Mapping[str, object],
    *,
    path: Path,
) -> None:
    incumbent_surface = incumbent_payload.get("target_surface")
    if not isinstance(incumbent_surface, Mapping):
        raise ValueError(
            f"{path} cannot be used as incumbent diagnostics because it has no "
            "target_surface fingerprint."
        )
    current_sha = current_target_surface.get("sha256")
    incumbent_sha = incumbent_surface.get("sha256")
    if not isinstance(current_sha, str) or not isinstance(incumbent_sha, str):
        raise ValueError(
            f"{path} cannot be used as incumbent diagnostics because its "
            "target_surface SHA is missing or invalid."
        )
    if current_sha != incumbent_sha:
        raise RuntimeError(
            "Incumbent diagnostics target surface mismatch: "
            f"current sha256={current_sha} "
            f"n_targets={current_target_surface.get('n_targets')}; "
            f"incumbent sha256={incumbent_sha} "
            f"n_targets={incumbent_surface.get('n_targets')} "
            f"path={path}. Score the incumbent on the current target surface "
            "before using it for release gates."
        )


def _state_fips_text(values: Iterable[object]) -> list[str]:
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                result.append(stripped.zfill(2))
                continue
        if isinstance(value, bytes):
            stripped = value.decode().strip()
            if stripped:
                result.append(stripped.zfill(2))
                continue
        if pd.isna(value):
            raise ValueError("state_fips contains missing values.")
        result.append(f"{int(value):02d}")
    return result


def _integer_geography_codes(values: Iterable[object], *, column: str) -> np.ndarray:
    result: list[int] = []
    for value in values:
        if isinstance(value, bytes):
            stripped = value.decode().strip()
            if stripped:
                result.append(int(stripped))
                continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                result.append(int(stripped))
                continue
        if pd.isna(value):
            raise ValueError(f"{column} contains missing values.")
        result.append(int(value))
    return np.asarray(result, dtype=np.int64)


def _aca_source_target_tables(target_specs: tuple) -> dict[str, pd.DataFrame]:
    rows_by_table: dict[str, list[dict[str, object]]] = {}
    for spec in target_specs:
        if spec.metadata.get("ledger_geography_level") != "state":
            continue
        groupby_dimension = spec.metadata.get("ledger_layout_groupby_dimension")
        if (
            isinstance(groupby_dimension, str)
            and "congressional_district" in groupby_dimension
        ):
            continue
        state_fips = spec.metadata.get("state_fips")
        if not state_fips:
            continue
        table_name = US_ACA_TARGET_ROLE_TABLES.get(spec.metadata.get("target_role"))
        if table_name is None:
            continue
        rows_by_table.setdefault(table_name, []).append(
            {
                "state_fips": str(state_fips).zfill(2),
                "target": float(spec.value),
                "source_record_id": spec.name,
            }
        )

    return {
        table_name: pd.DataFrame(rows)
        for table_name, rows in sorted(rows_by_table.items())
    }


def _with_state_take_up_rates(
    tax_unit: pd.DataFrame,
    target_tables: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    result = tax_unit.copy(deep=True)
    source = target_tables.get(US_ACA_APTC_TARGET_TABLE)
    if source is None or source.empty:
        result["aca_take_up_rate"] = 0.0
        return result

    eligible = result["is_aca_ptc_eligible"].fillna(False).astype(bool)
    weighted_eligible = (
        result.loc[eligible]
        .groupby("state_fips")["tax_unit_weight"]
        .sum()
        .astype(float)
    )
    targets = source.groupby("state_fips")["target"].sum().astype(float)
    rates = (targets / weighted_eligible).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["aca_take_up_rate"] = (
        result["state_fips"].map(rates).fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    return result


def _tax_unit_person_count_weights(
    frame: Frame,
    *,
    person_filter: np.ndarray | None = None,
) -> np.ndarray:
    person = frame.table("person")
    household_ids = frame.table("household")["household_id"].to_numpy()
    tax_unit_ids = frame.table("tax_unit")["tax_unit_id"].to_numpy()

    household_positions = pd.Series(
        np.arange(len(household_ids), dtype=np.int64),
        index=household_ids,
    )
    person_household_positions = household_positions.reindex(
        person["person_household_id"].to_numpy()
    ).to_numpy()
    if np.isnan(person_household_positions).any():
        raise ValueError(
            "Person rows reference household ids not present in household."
        )

    tax_unit_positions = pd.Series(
        np.arange(len(tax_unit_ids), dtype=np.int64),
        index=tax_unit_ids,
    ).reindex(person["person_tax_unit_id"].to_numpy())
    if np.isnan(tax_unit_positions).any():
        raise ValueError("Person rows reference tax_unit ids not present in tax_unit.")

    person_weights = frame.weights_for("household").values[
        person_household_positions.astype(np.int64)
    ]
    if person_filter is not None:
        if len(person_filter) != len(person):
            raise ValueError(
                "Person filter length does not match the person table: "
                f"{len(person_filter)} != {len(person)}."
            )
        person_weights = np.where(
            np.asarray(person_filter, dtype=bool), person_weights, 0.0
        )

    out = np.zeros(len(tax_unit_ids), dtype=np.float64)
    np.add.at(out, tax_unit_positions.astype(np.int64), person_weights)
    return out


def _aca_source_person_table(frame: Frame) -> pd.DataFrame:
    person = frame.table("person").copy()
    if "person_tax_unit_id" not in person:
        raise RuntimeError(
            "ACA source runtime requires person_tax_unit_id in the person table."
        )
    if US_ACA_REPORTED_MARKETPLACE_COVERAGE not in person:
        if "has_marketplace_health_coverage" not in person:
            raise RuntimeError(
                "ACA source runtime requires observed Marketplace coverage in "
                f"{US_ACA_REPORTED_MARKETPLACE_COVERAGE!r} or "
                "'has_marketplace_health_coverage'."
            )
        person[US_ACA_REPORTED_MARKETPLACE_COVERAGE] = person[
            "has_marketplace_health_coverage"
        ]

    person["tax_unit_id"] = person["person_tax_unit_id"]
    if US_ACA_REPORTED_SUBSIDIZED_ANCHOR not in person:
        # Observed Marketplace coverage is broader than subsidized coverage.
        # When the narrower anchor is unavailable, leave no preserved APTC
        # anchor rather than freezing unsubsidized observations as recipients.
        person[US_ACA_REPORTED_SUBSIDIZED_ANCHOR] = False
    return person


def _aca_source_tax_unit_table_from_simulation(
    frame: Frame,
    target_tables: Mapping[str, pd.DataFrame],
    *,
    simulation,
) -> pd.DataFrame:
    tax_unit = frame.table("tax_unit").copy()
    household = frame.table("household")
    positions = _tax_unit_to_household_positions(frame)
    state_fips = np.asarray(household["state_fips"].to_numpy())[positions]
    household_weights = frame.weights_for("household").values[positions]
    has_person_count_targets = any(
        table_name in target_tables for table_name in US_ACA_PERSON_COUNT_TARGET_TABLES
    )

    tax_unit["state_fips"] = _state_fips_text(state_fips)
    is_aca_ptc_eligible = (
        _calculate_array(
            simulation,
            "is_aca_ptc_eligible",
            map_to="tax_unit",
        )
        > 0
    )
    # Source-runtime support must ignore pre-refresh take-up flags:
    # assigned_aca_ptc is aca_ptc multiplied by takes_up_aca_if_eligible.
    potential_aca_ptc = _calculate_array(
        simulation,
        "aca_ptc",
        map_to="tax_unit",
    )
    has_potential_ptc = potential_aca_ptc > 0
    if has_person_count_targets:
        eligible_people = (
            _calculate_array(
                simulation,
                "is_aca_ptc_eligible",
                map_to="person",
            )
            > 0
        )
        tax_unit["tax_unit_weight"] = _tax_unit_person_count_weights(
            frame,
            person_filter=eligible_people,
        )
        tax_unit["is_aca_ptc_eligible"] = (
            tax_unit["tax_unit_weight"] > 0
        ) & has_potential_ptc
    else:
        tax_unit["tax_unit_weight"] = household_weights
        tax_unit["is_aca_ptc_eligible"] = is_aca_ptc_eligible & has_potential_ptc
    tax_unit["household_weight"] = household_weights
    tax_unit["health_insurance_premiums_without_medicare_part_b"] = _calculate_array(
        simulation,
        "health_insurance_premiums_without_medicare_part_b",
        map_to="tax_unit",
    )
    tax_unit["assigned_aca_ptc"] = potential_aca_ptc
    tax_unit["weighted_assigned_aca_ptc"] = potential_aca_ptc * household_weights
    tax_unit["slcsp"] = _calculate_array(simulation, "slcsp", map_to="tax_unit")
    return _with_state_take_up_rates(tax_unit, target_tables)


def _aca_source_tax_unit_table_batched(
    frame: Frame,
    target_tables: Mapping[str, pd.DataFrame],
    *,
    microsimulation_cls,
    maximum_microsim_batch_size: int | None,
) -> pd.DataFrame:
    _assert_no_formula_owned_columns(frame)
    tax_unit = frame.table("tax_unit").copy()
    household = frame.table("household")
    positions = _tax_unit_to_household_positions(frame)
    tax_unit["state_fips"] = _state_fips_text(
        np.asarray(household["state_fips"].to_numpy())[positions]
    )

    fill_columns = (
        "tax_unit_weight",
        "household_weight",
        "is_aca_ptc_eligible",
        "health_insurance_premiums_without_medicare_part_b",
        "assigned_aca_ptc",
        "weighted_assigned_aca_ptc",
        "slcsp",
    )
    tax_unit["tax_unit_weight"] = 0.0
    tax_unit["household_weight"] = frame.weights_for("household").values[positions]
    tax_unit["is_aca_ptc_eligible"] = False
    tax_unit["health_insurance_premiums_without_medicare_part_b"] = 0.0
    tax_unit["assigned_aca_ptc"] = 0.0
    tax_unit["weighted_assigned_aca_ptc"] = 0.0
    tax_unit["slcsp"] = 0.0

    tax_unit_positions = pd.Series(
        np.arange(len(tax_unit), dtype=np.int64),
        index=tax_unit["tax_unit_id"].to_numpy(),
    )
    n_households = frame.n("household")
    batches = tuple(
        _household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if len(batches) > 1:
        print(
            "Materializing ACA source inputs in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )

    for household_positions in batches:
        with _automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else _select_households_by_position(frame, household_positions)
            )
            batch_simulation = microsimulation_cls(
                dataset=_dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                )
            )
            batch_tax_unit = _aca_source_tax_unit_table_from_simulation(
                batch_frame,
                target_tables,
                simulation=batch_simulation,
            )
            full_positions = tax_unit_positions.reindex(
                batch_tax_unit["tax_unit_id"].to_numpy()
            ).to_numpy()
            if np.isnan(full_positions).any():
                raise RuntimeError(
                    "ACA source batch produced tax_unit_id values not present in "
                    "the full tax_unit table."
                )
            full_positions = full_positions.astype(np.int64)
            for column in fill_columns:
                tax_unit.iloc[
                    full_positions,
                    tax_unit.columns.get_loc(column),
                ] = batch_tax_unit[column].to_numpy()
            batch_simulation._invalidate_all_caches()
            del batch_tax_unit, batch_simulation, batch_frame
        _collect_batch_garbage()
    return _with_state_take_up_rates(tax_unit, target_tables)


def _aca_source_tax_unit_table(
    frame: Frame,
    target_tables: Mapping[str, pd.DataFrame],
    *,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> pd.DataFrame:
    if simulation is not None:
        return _aca_source_tax_unit_table_from_simulation(
            frame,
            target_tables,
            simulation=simulation,
        )
    from policyengine_us import Microsimulation

    return _aca_source_tax_unit_table_batched(
        frame,
        target_tables,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )


def _with_aca_marketplace_source_outputs(
    frame: Frame,
    target_specs: tuple,
    *,
    seed: int,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> Frame:
    target_tables = _aca_source_target_tables(target_specs)
    if US_ACA_APTC_TARGET_TABLE not in target_tables:
        raise RuntimeError(
            "ACA Marketplace source refresh requires an APTC-recipient target. "
            "The Marketplace enrollment target is observed person coverage and "
            "must not be used as a simulated PTC take-up fallback."
        )
    stage = US_SOURCE_MANIFEST.stage_map()[US_ACA_MARKETPLACE_STAGE]
    stop_after = (
        None
        if "cms_aca_bronze_aptc_consumers_by_state" in target_tables
        else "support_clip"
    )
    tables = {
        "cps_person": _aca_source_person_table(frame),
        "tax_unit": _aca_source_tax_unit_table(
            frame,
            target_tables,
            simulation=simulation,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
        ),
        **target_tables,
    }
    source_output = run_source_stage(
        stage,
        tables=tables,
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(seed=seed, target_year=PERIOD),
        stop_after=stop_after,
    )

    if "tax_unit_id" not in source_output:
        raise RuntimeError("ACA source runtime output is missing tax_unit_id.")
    missing_outputs = [
        column
        for column in US_ACA_SOURCE_OUTPUT_COLUMNS
        if column not in source_output.columns
    ]
    if missing_outputs:
        raise RuntimeError(
            "ACA source runtime output is missing declared column(s): "
            f"{missing_outputs}."
        )

    tax_unit = frame.table("tax_unit").copy()
    output_by_id = source_output.set_index("tax_unit_id")
    aligned = output_by_id.reindex(tax_unit["tax_unit_id"])
    if aligned[list(US_ACA_SOURCE_OUTPUT_COLUMNS)].isna().any().any():
        raise RuntimeError(
            "ACA source runtime output does not cover every tax_unit_id in the "
            "release frame."
        )
    for column in US_ACA_SOURCE_OUTPUT_COLUMNS:
        tax_unit[column] = aligned[column].to_numpy()

    tables_out = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables_out["tax_unit"] = tax_unit
    return Frame(
        tables_out,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def _medicaid_source_target_table(target_specs: tuple) -> pd.DataFrame:
    """CMS state Medicaid enrollment counts as the take-up calibration table.

    Mirrors :func:`_aca_source_target_tables` for the ``medicaid_enrollment``
    target role: month-tagged December 2024 state snapshot facts
    (``cms_medicaid.month2024_12.state_enrollment.*``), point-in-time
    semantics per populace #332.
    """
    rows: list[dict[str, object]] = []
    for spec in target_specs:
        if spec.metadata.get("ledger_geography_level") != "state":
            continue
        groupby_dimension = spec.metadata.get("ledger_layout_groupby_dimension")
        if (
            isinstance(groupby_dimension, str)
            and "congressional_district" in groupby_dimension
        ):
            continue
        state_fips = spec.metadata.get("state_fips")
        if not state_fips:
            continue
        if spec.metadata.get("target_role") != US_MEDICAID_ENROLLMENT_TARGET_ROLE:
            continue
        rows.append(
            {
                "state_fips": str(state_fips).zfill(2),
                "target": float(spec.value),
                "source_record_id": spec.name,
            }
        )
    table = pd.DataFrame(rows, columns=["state_fips", "target", "source_record_id"])
    duplicated = table["state_fips"][table["state_fips"].duplicated()].unique()
    if len(duplicated):
        # The calibrate operation applies duplicate state rows sequentially
        # (last row wins) while the rate prior, diagnostics, and gate SUM
        # them — divergent semantics that must never reach the stage.
        raise RuntimeError(
            "Medicaid enrollment targets carry duplicate state rows for "
            f"state_fips {sorted(duplicated)}; the ledger feed must supply "
            "exactly one medicaid_enrollment count per state."
        )
    return table


def _person_state_fips(frame: Frame) -> np.ndarray:
    """Person-aligned state FIPS text codes via the frame's linkage."""
    # Frame.broadcast is the validated membership-mapping path (linkage is
    # asserted at Frame construction), so no manual reindex/NaN handling.
    return np.asarray(_state_fips_text(frame.broadcast("state_fips").to_numpy()))


def _medicaid_person_eligibility(
    frame: Frame,
    *,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> np.ndarray:
    """Engine-computed person-level ``is_medicaid_eligible``, batched like ACA."""
    if simulation is not None:
        return _calculate_array(simulation, "is_medicaid_eligible", map_to="person") > 0
    from policyengine_us import Microsimulation

    person_ids = frame.table("person")["person_id"].to_numpy()
    eligibility = np.zeros(len(person_ids), dtype=bool)
    person_positions = pd.Series(
        np.arange(len(person_ids), dtype=np.int64), index=person_ids
    )
    n_households = frame.n("household")
    batches = tuple(
        _household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if len(batches) > 1:
        print(
            "Materializing Medicaid eligibility in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )
    for household_positions in batches:
        with _automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else _select_households_by_position(frame, household_positions)
            )
            batch_simulation = Microsimulation(
                dataset=_dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                )
            )
            batch_eligible = (
                _calculate_array(
                    batch_simulation, "is_medicaid_eligible", map_to="person"
                )
                > 0
            )
            positions = person_positions.reindex(
                batch_frame.table("person")["person_id"].to_numpy()
            ).to_numpy()
            if np.isnan(positions).any():
                raise RuntimeError(
                    "Medicaid eligibility batch produced person_id values not "
                    "present in the full person table."
                )
            eligibility[positions.astype(np.int64)] = batch_eligible
            batch_simulation._invalidate_all_caches()
            del batch_frame, batch_simulation
        _collect_batch_garbage()
    return eligibility


def _with_medicaid_take_up_outputs(
    frame: Frame,
    target_specs: tuple,
    *,
    seed: int,
    substitutions: Sequence[dict[str, object]] = (),
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> tuple[Frame, dict[str, object]]:
    """Assign Medicaid take-up (populace #331) and return frame + diagnostics.

    ``substitutions`` are the reviewed CMS enrollment substitution records
    (populace#386) produced by :func:`apply_us_medicaid_enrollment_substitutions`;
    they ride the take-up diagnostics so the gate can rot-check a backfilled
    substitution and the ``us_medicaid_take_up.json`` release artifact records
    exactly which states shipped a substituted point-in-time count.
    """
    target_table = _medicaid_source_target_table(target_specs)
    if target_table.empty:
        raise RuntimeError(
            "Medicaid take-up requires CMS state enrollment targets "
            f"({US_MEDICAID_ENROLLMENT_TARGET_TABLE}); none were compiled from "
            "the ledger facts."
        )
    eligibility = _medicaid_person_eligibility(
        frame,
        simulation=simulation,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )
    return with_us_medicaid_take_up(
        frame,
        is_medicaid_eligible=eligibility,
        state_fips=_person_state_fips(frame),
        state_targets=target_table,
        seed=seed,
        substitutions=substitutions,
    )


def _assert_no_formula_owned_columns(frame: Frame) -> None:
    adapter = PolicyEngineUSEngine()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    formula_owned = adapter._engine_computed_columns(tables, period=PERIOD)
    if formula_owned:
        raise ValueError(
            "Formula-owned PolicyEngine columns are present before export: "
            f"{sorted(formula_owned)}. Source stages must emit leaf inputs."
        )


def _dataset_from_frame(
    frame: Frame,
    *,
    zero_variables: Iterable[str] = (),
    system=None,
    assert_no_formula_owned_columns: bool = True,
):
    if assert_no_formula_owned_columns:
        _assert_no_formula_owned_columns(frame)

    from policyengine_us.data import USSingleYearDataset

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for variable_name in zero_variables:
        if system is None:
            raise ValueError("system is required when zero_variables are provided.")
        entity = _variable_entity(system, variable_name)
        if entity is not None and variable_name in tables[entity]:
            tables[entity][variable_name] = 0
    tables["household"]["household_weight"] = frame.weights_for("household").values
    return USSingleYearDataset(
        person=tables["person"],
        household=tables["household"],
        tax_unit=tables["tax_unit"],
        spm_unit=tables["spm_unit"],
        family=tables["family"],
        marital_unit=tables["marital_unit"],
        time_period=PERIOD,
    )


def _calculate_array(
    simulation, variable: str, *, map_to: str | None = None
) -> np.ndarray:
    kwargs: dict[str, Any] = {"period": PERIOD}
    if map_to is not None:
        kwargs["map_to"] = map_to
    return np.asarray(simulation.calculate(variable, **kwargs))


def _tax_unit_to_household_positions(frame: Frame) -> np.ndarray:
    return _group_to_household_positions(frame, "tax_unit")


def _group_to_household_positions(frame: Frame, group_entity: str) -> np.ndarray:
    person = frame.table("person")
    household_ids = frame.table("household")["household_id"].to_numpy()
    group_ids = frame.table(group_entity)[f"{group_entity}_id"].to_numpy()
    person_group_column = f"person_{group_entity}_id"

    membership = person[[person_group_column, "person_household_id"]].drop_duplicates()
    counts = membership.groupby(person_group_column)["person_household_id"].nunique()
    ambiguous = counts[counts != 1]
    if not ambiguous.empty:
        raise ValueError(
            f"{group_entity} units must be nested in households; ambiguous ids "
            f"examples: {ambiguous.index[:5].tolist()}."
        )
    group_to_household = (
        membership.drop_duplicates(person_group_column)
        .set_index(person_group_column)["person_household_id"]
        .reindex(group_ids)
    )
    if group_to_household.isna().any():
        missing = group_ids[group_to_household.isna().to_numpy()][:5].tolist()
        raise ValueError(f"{group_entity} units with no person membership: {missing}.")
    household_positions = pd.Series(
        np.arange(len(household_ids), dtype=np.int64), index=household_ids
    )
    positions = household_positions.reindex(group_to_household.to_numpy()).to_numpy()
    if np.isnan(positions).any():
        raise ValueError(
            f"{group_entity} household ids are not present in household table."
        )
    return positions.astype(np.int64)


def _collapse_tax_unit(
    values: np.ndarray, positions: np.ndarray, n_households: int
) -> np.ndarray:
    return _collapse_group(values, positions, n_households)


def _collapse_group(
    values: np.ndarray, positions: np.ndarray, n_households: int
) -> np.ndarray:
    out = np.zeros(n_households, dtype=np.float64)
    np.add.at(out, positions, np.asarray(values, dtype=np.float64))
    return out


def _collapse_person(frame: Frame, values: np.ndarray) -> np.ndarray:
    household_ids = frame.table("household")["household_id"].to_numpy()
    person_households = frame.table("person")["person_household_id"].to_numpy()
    household_positions = pd.Series(
        np.arange(len(household_ids), dtype=np.int64), index=household_ids
    )
    positions = (
        household_positions.reindex(person_households).to_numpy().astype(np.int64)
    )
    out = np.zeros(len(household_ids), dtype=np.float64)
    np.add.at(out, positions, np.asarray(values, dtype=np.float64))
    return out


def _household_position_batches(
    n_households: int, batch_size: int | None
) -> Iterable[np.ndarray]:
    if n_households <= 0:
        return
    if batch_size is None or batch_size <= 0 or batch_size >= n_households:
        yield np.arange(n_households, dtype=np.int64)
        return
    for start in range(0, n_households, batch_size):
        stop = min(start + batch_size, n_households)
        yield np.arange(start, stop, dtype=np.int64)


def _select_households_by_position(frame: Frame, positions: np.ndarray) -> Frame:
    household_ids = frame.table("household")["household_id"].to_numpy()[positions]
    person_mask = frame.table("person")["person_household_id"].isin(household_ids)
    return frame.select(person_mask)


class _BatchedScalarTotal:
    def __init__(self, value: float):
        self.value = float(value)

    def sum(self) -> float:
        return self.value


class _BatchedReformValidationSimulation:
    def __init__(
        self,
        frame: Frame,
        *,
        reform,
        maximum_microsim_batch_size: int | None,
        microsimulation_cls,
        dataset_from_frame,
    ):
        self._frame = frame
        self._reform = reform
        self._maximum_microsim_batch_size = maximum_microsim_batch_size
        self._microsimulation_cls = microsimulation_cls
        self._dataset_from_frame = dataset_from_frame
        self._cache: dict[tuple[str, int], float] = {}

    def calculate(self, measure: str, period: int) -> _BatchedScalarTotal:
        key = (str(measure), int(period))
        if key not in self._cache:
            self._cache[key] = self._calculate_total(str(measure), int(period))
        return _BatchedScalarTotal(self._cache[key])

    def _calculate_total(self, measure: str, period: int) -> float:
        n_households = self._frame.n("household")
        batches = tuple(
            _household_position_batches(
                n_households,
                self._maximum_microsim_batch_size,
            )
        )
        if len(batches) > 1:
            print(
                "Scoring reform validation measure "
                f"{measure} in {len(batches)} batches of up to "
                f"{self._maximum_microsim_batch_size:,} households.",
                flush=True,
            )
        total = 0.0
        for household_positions in batches:
            with _automatic_gc_suspended():
                full_batch = len(household_positions) == n_households
                batch_frame = (
                    self._frame
                    if full_batch
                    else _select_households_by_position(
                        self._frame, household_positions
                    )
                )
                dataset = self._dataset_from_frame(batch_frame)
                simulation = (
                    self._microsimulation_cls(dataset=dataset)
                    if self._reform is None
                    else self._microsimulation_cls(dataset=dataset, reform=self._reform)
                )
                total += float(simulation.calculate(measure, period).sum())
                if hasattr(simulation, "_invalidate_all_caches"):
                    simulation._invalidate_all_caches()
                del simulation, dataset, batch_frame
            _collect_batch_garbage()
        return total


def _batched_reform_validation_simulate_factory_from_frame(
    frame: Frame,
    *,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    microsimulation_cls=None,
    dataset_from_frame=None,
):
    if microsimulation_cls is None:
        from policyengine_us import Microsimulation

        microsimulation_cls = Microsimulation
    if dataset_from_frame is None:

        def dataset_from_frame(batch_frame: Frame):
            return _dataset_from_frame(
                batch_frame,
                assert_no_formula_owned_columns=False,
            )

    def simulate(reform):
        return _BatchedReformValidationSimulation(
            frame,
            reform=reform,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
            microsimulation_cls=microsimulation_cls,
            dataset_from_frame=dataset_from_frame,
        )

    return simulate


def _reform_household_income_tax(
    *,
    base_frame: Frame,
    reform_spec,
    system,
    microsimulation_cls,
    n_households: int,
    batch_size: int | None,
) -> np.ndarray:
    _assert_no_formula_owned_columns(base_frame)
    reform_income_tax = np.zeros(n_households, dtype=np.float64)
    reform = _make_zero_variable_reform(system, reform_spec.neutralized_variable)
    batches = tuple(_household_position_batches(n_households, batch_size))
    if len(batches) > 1:
        print(
            "Materializing reform target "
            f"{reform_spec.measure} in {len(batches)} batches "
            f"of up to {batch_size:,} households.",
            flush=True,
        )
    for household_positions in batches:
        with _automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                base_frame
                if full_batch
                else _select_households_by_position(base_frame, household_positions)
            )
            batch_tax_unit_positions = _tax_unit_to_household_positions(batch_frame)
            reformed_dataset = _dataset_from_frame(
                batch_frame,
                zero_variables=(reform_spec.neutralized_variable,),
                system=system,
                assert_no_formula_owned_columns=False,
            )
            reformed = microsimulation_cls(dataset=reformed_dataset, reform=reform)
            batch_income_tax = _collapse_tax_unit(
                _calculate_array(reformed, "income_tax"),
                batch_tax_unit_positions,
                batch_frame.n("household"),
            )
            reform_income_tax[household_positions] = batch_income_tax
            reformed._invalidate_all_caches()
            del batch_income_tax, reformed, reformed_dataset, batch_frame
        _collect_batch_garbage()
    return reform_income_tax


def _variable_entity(system, name: str) -> str | None:
    variable = system.variables.get(name)
    if variable is None:
        return None
    return variable.entity.key


def _household_values(
    *,
    frame: Frame,
    simulation,
    system,
    variable: str,
    tax_unit_positions: np.ndarray,
    positive_indicator: bool = False,
    map_to: str | None = None,
    filter_variable: str | None = None,
    less_than: float | None = None,
) -> np.ndarray:
    entity = _variable_entity(system, variable)
    if entity is None:
        raise KeyError(variable)
    if map_to is not None:
        entity = map_to
    if entity == "household":
        values = _calculate_array(simulation, variable, map_to=map_to)
        filter_values = None
        if filter_variable is not None:
            filter_values = _calculate_array(simulation, filter_variable, map_to=map_to)
        if less_than is not None:
            indicator = values < less_than
            if filter_values is not None:
                indicator &= filter_values > 0
            return indicator.astype(np.float64)
        if filter_values is not None:
            values = np.where(filter_values > 0, values, 0)
        return (values > 0).astype(np.float64) if positive_indicator else values
    raw = _calculate_array(simulation, variable, map_to=map_to)
    filter_values = None
    if filter_variable is not None:
        filter_values = _calculate_array(simulation, filter_variable, map_to=map_to)
    if less_than is not None:
        indicator = raw < less_than
        if filter_values is not None:
            indicator &= filter_values > 0
        raw = indicator.astype(np.float64)
    elif positive_indicator:
        if filter_values is not None:
            raw = np.where(filter_values > 0, raw, 0)
        raw = (raw > 0).astype(np.float64)
    elif filter_values is not None:
        raw = np.where(filter_values > 0, raw, 0)
    if entity == "tax_unit":
        return _collapse_tax_unit(
            raw,
            tax_unit_positions,
            frame.n("household"),
        )
    if entity == "person":
        return _collapse_person(frame, raw)
    if entity in {"spm_unit", "family", "marital_unit"}:
        return _collapse_group(
            raw,
            _group_to_household_positions(frame, entity),
            frame.n("household"),
        )
    raise ValueError(f"Unsupported variable entity {entity!r} for {variable!r}.")


def _base_variables_from_metadata(metadata: Mapping[str, str]) -> tuple[str, ...]:
    combined = metadata.get("base_variables")
    if combined:
        variables = tuple(
            variable.strip() for variable in combined.split(",") if variable.strip()
        )
        if not variables:
            raise ValueError("base_variables metadata must name at least one variable.")
        return variables
    return (metadata["base_variable"],)


def _less_than_from_metadata(metadata: Mapping[str, str]) -> float | None:
    mode = metadata.get("measure_mode", "sum")
    threshold = metadata.get("indicator_less_than")
    if mode != "less_than_indicator_sum":
        return None
    if threshold is None:
        raise ValueError(
            "less-than indicator-sum targets must set indicator_less_than metadata."
        )
    return float(threshold)


def _combined_household_values(
    *,
    frame: Frame,
    simulation,
    system,
    variables: tuple[str, ...],
    tax_unit_positions: np.ndarray,
    positive_indicator: bool = False,
    map_to: str | None = None,
    filter_variable: str | None = None,
    less_than: float | None = None,
) -> np.ndarray:
    if len(variables) == 1:
        return _household_values(
            frame=frame,
            simulation=simulation,
            system=system,
            variable=variables[0],
            tax_unit_positions=tax_unit_positions,
            positive_indicator=positive_indicator,
            map_to=map_to,
            filter_variable=filter_variable,
            less_than=less_than,
        )

    entities = tuple(_variable_entity(system, variable) for variable in variables)
    missing = tuple(
        variable
        for variable, entity in zip(variables, entities, strict=True)
        if entity is None
    )
    if missing:
        raise KeyError(", ".join(missing))
    if map_to is None and len(set(entities)) != 1:
        raise ValueError(
            f"Cannot combine variables from different entities: "
            f"{dict(zip(variables, entities, strict=True))}."
        )

    raw_arrays = tuple(
        np.asarray(
            _calculate_array(simulation, variable, map_to=map_to), dtype=np.float64
        )
        for variable in variables
    )
    if positive_indicator:
        raw = np.logical_or.reduce([values > 0 for values in raw_arrays]).astype(
            np.float64
        )
    else:
        raw = np.sum(raw_arrays, axis=0, dtype=np.float64)

    if less_than is not None:
        raw = raw < less_than
        if filter_variable is not None:
            filter_values = _calculate_array(simulation, filter_variable, map_to=map_to)
            raw &= filter_values > 0
        raw = raw.astype(np.float64)
    elif filter_variable is not None:
        filter_values = _calculate_array(simulation, filter_variable, map_to=map_to)
        raw = np.where(filter_values > 0, raw, 0)

    entity = map_to or entities[0]
    if entity == "household":
        return raw
    if entity == "tax_unit":
        return _collapse_tax_unit(raw, tax_unit_positions, frame.n("household"))
    if entity == "person":
        return _collapse_person(frame, raw)
    if entity in {"spm_unit", "family", "marital_unit"}:
        return _collapse_group(
            raw,
            _group_to_household_positions(frame, entity),
            frame.n("household"),
        )
    raise ValueError(f"Unsupported variable entity {entity!r} for {variables!r}.")


def _filing_status_names(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [getattr(value, "name", str(value)) for value in values], dtype=object
    )


def _soi_eitc_child_count_filter(metadata: Mapping[str, str]) -> str | None:
    explicit = metadata.get("ledger_filter_eitc_child_count")
    if explicit and explicit.strip().lower() != "all":
        return explicit
    record_set = metadata.get("ledger_layout_record_set_id", "")
    if record_set.endswith(".eitc_by_agi_children.no_qualifying_children"):
        return "0"
    if record_set.endswith(".eitc_by_agi_children.one_qualifying_child"):
        return "1"
    if record_set.endswith(".eitc_by_agi_children.two_qualifying_children"):
        return "2"
    if record_set.endswith(".eitc_by_agi_children.three_or_more_qualifying_children"):
        return "3plus"
    measure = metadata.get("source_measure_id", "")
    if measure.startswith("eitc_no_children_"):
        return "0"
    if measure.startswith("eitc_one_child_"):
        return "1"
    if measure.startswith("eitc_two_children_"):
        return "2"
    if measure.startswith("eitc_three_or_more_children_"):
        return "3plus"
    return None


def _soi_requires_positive_eitc_filter(metadata: Mapping[str, str]) -> bool:
    return (
        metadata.get("ledger_domain")
        == "individual_income_tax_returns_with_earned_income_credit"
    )


def _is_noop_ledger_filter_value(value: str) -> bool:
    return value.strip().lower().replace("_", " ") in {"", "all"}


def _unsupported_soi_ledger_filters(metadata: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            key
            for key, value in metadata.items()
            if key.startswith("ledger_filter_")
            and key not in SUPPORTED_SOI_LEDGER_FILTERS
            and not _is_noop_ledger_filter_value(str(value))
        )
    )


def _eitc_child_count_mask(values: np.ndarray, filter_value: str) -> np.ndarray:
    counts = np.asarray(values, dtype=np.float64)
    normalized = str(filter_value).strip().lower().replace("_", " ")
    if normalized in {"0", "none", "no children", "no qualifying children"}:
        return counts == 0
    if normalized in {"1", "one", "one child", "one qualifying child"}:
        return counts == 1
    if normalized in {"2", "two", "two children", "two qualifying children"}:
        return counts == 2
    if normalized in {
        "3",
        "3+",
        "3plus",
        "three",
        "three plus",
        "three or more",
        "three or more children",
        "three or more qualifying children",
    }:
        return counts >= 3
    raise ValueError(f"Unsupported EITC child-count filter {filter_value!r}.")


def _as_bound(value: str) -> float:
    if value == "-inf":
        return -math.inf
    if value == "inf":
        return math.inf
    return float(value)


def _population_age_household_values(
    *,
    frame: Frame,
    household: pd.DataFrame,
    age: np.ndarray,
    metadata: Mapping[str, str],
    household_congressional_district_geoid: np.ndarray | None = None,
) -> np.ndarray:
    lower = _as_bound(metadata.get("age_lower_bound", "-inf"))
    upper = _as_bound(metadata.get("age_upper_bound", "inf"))
    person_mask = (age >= lower) & (age < upper)
    values = _collapse_person(frame, person_mask.astype(np.float64))
    state_fips = metadata.get("state_fips")
    if state_fips:
        values = np.where(
            household["state_fips"].to_numpy() == int(state_fips),
            values,
            0.0,
        )
    congressional_district_geoid = metadata.get("congressional_district_geoid")
    if congressional_district_geoid:
        if household_congressional_district_geoid is None:
            household_congressional_district_geoid = _integer_geography_codes(
                household["congressional_district_geoid"].to_numpy(),
                column="congressional_district_geoid",
            )
        values = np.where(
            household_congressional_district_geoid == int(congressional_district_geoid),
            values,
            0.0,
        )
    return values


def _unsupported_ledger_filter_metadata(
    target_specs: Iterable[object],
) -> dict[str, tuple[str, ...]]:
    unsupported: dict[str, tuple[str, ...]] = {}
    for spec in target_specs:
        metadata = getattr(spec, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        keys = tuple(
            sorted(
                str(key)
                for key, value in metadata.items()
                if str(key).startswith("ledger_filter")
                and str(key) not in SUPPORTED_LEDGER_FILTER_METADATA_KEYS
                and not _is_noop_ledger_filter_value(str(value))
            )
        )
        if keys:
            unsupported[str(getattr(spec, "name", "<unnamed target>"))] = keys
    return unsupported


def _assert_supported_ledger_filter_metadata(
    target_specs: Iterable[object],
) -> None:
    unsupported = _unsupported_ledger_filter_metadata(target_specs)
    if not unsupported:
        return
    details = "; ".join(
        f"{name}: {', '.join(keys)}" for name, keys in sorted(unsupported.items())
    )
    raise RuntimeError(
        "Unsupported Ledger target filter metadata would be ignored by the "
        f"US fiscal materializer: {details}."
    )


def _signed_component(values: np.ndarray, source_name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if source_name in {"adjusted_gross_income", "rent_and_royalty_net_income"}:
        return values
    if source_name == "capital_gains_losses":
        return np.maximum(values, 0.0)
    if source_name.endswith("_losses") or source_name in {
        "business_net_losses",
        "capital_gains_losses",
        "estate_losses",
        "partnership_and_s_corp_losses",
        "rent_and_royalty_net_losses",
    }:
        return np.minimum(values, 0.0) * -1.0
    return np.maximum(values, 0.0)


def _soi_component_row(
    values: np.ndarray, source_name: str, *, indicator: bool
) -> np.ndarray:
    component = _signed_component(values, source_name)
    if indicator:
        return (component > 0).astype(np.float64)
    return component


def _person_variable_to_tax_unit(*, frame: Frame, values: np.ndarray) -> np.ndarray:
    person = frame.table("person")
    tax_unit_ids = frame.table("tax_unit")["tax_unit_id"].to_numpy()
    tax_positions = (
        pd.Series(np.arange(len(tax_unit_ids), dtype=np.int64), index=tax_unit_ids)
        .reindex(person["person_tax_unit_id"].to_numpy())
        .to_numpy()
    )
    out = np.zeros(len(tax_unit_ids), dtype=np.float64)
    np.add.at(out, tax_positions.astype(np.int64), np.asarray(values, dtype=np.float64))
    return out


def _make_zero_variable_reform(system, variable_name: str):
    from policyengine_us.model_api import Reform, Variable

    original = system.variables[variable_name]

    class NeutralizedVariable(Variable):
        value_type = original.value_type
        entity = original.entity
        label = f"Neutralized {variable_name}"
        definition_period = original.definition_period
        unit = getattr(original, "unit", None)
        adds = None
        subtracts = None
        uprating = None

        def formula(self, period, parameters):  # pragma: no cover - exercised in PE
            return 0

    NeutralizedVariable.__name__ = variable_name

    class NeutralizeVariableReform(Reform):
        def apply(self):
            self.replace_variable(NeutralizedVariable)

    NeutralizeVariableReform.__name__ = f"neutralize_{variable_name}"
    return NeutralizeVariableReform


def _compile_materialized_target_registry(
    target_frame: Frame,
    target_specs: tuple,
    *,
    gate_congressional_district_targets: bool = True,
) -> tuple[TargetRegistry, dict[str, object]]:
    household = target_frame.table("household")
    compileable_specs = [
        spec for spec in target_specs if _target_spec_is_materialized(spec, household)
    ]
    registry = TargetRegistry(compileable_specs, country="us")
    dropped_specs = tuple(
        spec for spec in target_specs if spec not in compileable_specs
    )
    dropped = sorted(spec.name for spec in dropped_specs)
    diagnostic_only_dropped = sorted(
        spec.name for spec in dropped_specs if _target_is_congressional_district(spec)
    )
    return registry, {
        "declared_targets": len(target_specs),
        "compiled_candidate_targets": len(compileable_specs),
        "dropped_target_names": dropped,
        "gate_congressional_district_targets": (gate_congressional_district_targets),
        "diagnostic_only_dropped_target_names": diagnostic_only_dropped,
    }


def _load_or_materialize_target_frame(
    base_frame: Frame,
    target_specs: tuple,
    *,
    target_frame_checkpoint_path: Path | None = None,
    target_frame_checkpoint_identity: Mapping[str, object] | None = None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    target_materialization_cache_dir: Path | None = None,
    target_materialization_cache_context: Mapping[str, object] | None = None,
    gate_congressional_district_targets: bool = True,
) -> tuple[Frame, TargetRegistry, dict[str, object]]:
    if (
        target_frame_checkpoint_path is not None
        and target_frame_checkpoint_identity is None
    ):
        raise ValueError(
            "target_frame_checkpoint_identity is required when "
            "target_frame_checkpoint_path is set."
        )
    if (
        target_frame_checkpoint_path is not None
        and target_frame_checkpoint_identity is not None
    ):
        loaded = _read_target_frame_checkpoint(
            target_frame_checkpoint_path,
            identity=target_frame_checkpoint_identity,
            target_specs=target_specs,
            gate_congressional_district_targets=gate_congressional_district_targets,
        )
        if loaded is not None:
            return loaded

    target_frame, registry, compilation = _materialize_target_frame(
        base_frame,
        target_specs,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
        target_materialization_cache_dir=target_materialization_cache_dir,
        target_materialization_cache_context=target_materialization_cache_context,
        gate_congressional_district_targets=gate_congressional_district_targets,
    )
    if (
        target_frame_checkpoint_path is not None
        and target_frame_checkpoint_identity is not None
    ):
        checkpoint_payload = _write_target_frame_checkpoint(
            target_frame_checkpoint_path,
            frame=target_frame,
            identity=target_frame_checkpoint_identity,
            compilation=compilation,
        )
    else:
        checkpoint_payload = {
            "enabled": False,
            "status": "disabled",
        }
    compilation = {
        **dict(compilation),
        "target_frame_checkpoint": checkpoint_payload,
    }
    return target_frame, registry, compilation


def _materialize_target_frame(
    base_frame: Frame,
    target_specs: tuple,
    *,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    target_materialization_cache_dir: Path | None = None,
    target_materialization_cache_context: Mapping[str, object] | None = None,
    gate_congressional_district_targets: bool = False,
) -> tuple[Frame, TargetRegistry, dict[str, object]]:
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    if (
        target_materialization_cache_dir is not None
        and target_materialization_cache_context is None
    ):
        raise ValueError(
            "target_materialization_cache_context is required when "
            "target_materialization_cache_dir is set."
        )
    _assert_supported_ledger_filter_metadata(target_specs)
    _assert_no_formula_owned_columns(base_frame)
    dataset = _dataset_from_frame(
        base_frame,
        assert_no_formula_owned_columns=False,
    )
    simulation = Microsimulation(dataset=dataset)
    system = CountryTaxBenefitSystem()
    household = base_frame.table("household")
    tax_unit_positions = _tax_unit_to_household_positions(base_frame)
    n_households = base_frame.n("household")

    materialized = {
        entity: base_frame.table(entity).copy() for entity in base_frame.entities
    }
    hh = materialized["household"]

    income_tax_tax_unit = _calculate_array(simulation, "income_tax")
    taxable_income_tax_unit = _calculate_array(simulation, "taxable_income")
    agi_tax_unit = _calculate_array(simulation, "adjusted_gross_income")
    filing_status = _filing_status_names(_calculate_array(simulation, "filing_status"))
    eitc_child_count = (
        np.asarray(_calculate_array(simulation, "eitc_child_count"), dtype=np.float64)
        if "eitc_child_count" in system.variables
        else None
    )
    needs_itemizer_mask = any(
        spec.family == "irs_soi" and spec.metadata.get("itemized_only") == "true"
        for spec in target_specs
    )
    tax_unit_itemizes = (
        np.asarray(_calculate_array(simulation, "tax_unit_itemizes"), dtype=bool)
        if needs_itemizer_mask and "tax_unit_itemizes" in system.variables
        else None
    )
    tax_unit_state_fips = household["state_fips"].to_numpy()[tax_unit_positions]
    needs_congressional_district_mask = any(
        spec.metadata.get("congressional_district_geoid") for spec in target_specs
    )
    if needs_congressional_district_mask:
        if "congressional_district_geoid" not in household.columns:
            raise RuntimeError(
                "Congressional-district target rows require a household "
                "congressional_district_geoid column in the support frame."
            )
        household_congressional_district_geoid = _integer_geography_codes(
            household["congressional_district_geoid"].to_numpy(),
            column="congressional_district_geoid",
        )
        tax_unit_congressional_district_geoid = household_congressional_district_geoid[
            tax_unit_positions
        ]
    else:
        household_congressional_district_geoid = None
        tax_unit_congressional_district_geoid = None

    hh["income_tax"] = _collapse_tax_unit(
        income_tax_tax_unit, tax_unit_positions, n_households
    )
    hh["state_income_tax"] = _household_values(
        frame=base_frame,
        simulation=simulation,
        system=system,
        variable="state_income_tax",
        tax_unit_positions=tax_unit_positions,
    )
    population_age_target_specs = [
        spec
        for spec in target_specs
        if spec.metadata.get("materializer") == "population_age"
    ]
    if population_age_target_specs and "age" in system.variables:
        person_age = np.asarray(_calculate_array(simulation, "age"), dtype=np.float64)
        for spec in population_age_target_specs:
            hh[spec.measure] = _population_age_household_values(
                frame=base_frame,
                household=household,
                age=person_age,
                metadata=spec.metadata,
                household_congressional_district_geoid=(
                    household_congressional_district_geoid
                ),
            )
    direct_target_specs = [
        spec
        for spec in target_specs
        if spec.metadata.get("materializer") == "policyengine_variable"
    ]
    direct_value_cache: dict[
        tuple[tuple[str, ...], str, str, str, str], np.ndarray
    ] = {}
    for spec in direct_target_specs:
        base_variables = _base_variables_from_metadata(spec.metadata)
        mode = spec.metadata.get("measure_mode", "sum")
        map_to = spec.metadata.get("indicator_map_to")
        filter_variable = spec.metadata.get("indicator_filter_variable")
        less_than = _less_than_from_metadata(spec.metadata)
        cache_key = (
            base_variables,
            mode,
            map_to or "",
            filter_variable or "",
            "" if less_than is None else str(less_than),
        )
        if cache_key not in direct_value_cache:
            variables_to_check = (
                *base_variables,
                *(() if filter_variable is None else (filter_variable,)),
            )
            if any(variable not in system.variables for variable in variables_to_check):
                continue
            direct_value_cache[cache_key] = _combined_household_values(
                frame=base_frame,
                simulation=simulation,
                system=system,
                variables=base_variables,
                tax_unit_positions=tax_unit_positions,
                positive_indicator=mode == "indicator_sum",
                map_to=map_to,
                filter_variable=filter_variable,
                less_than=less_than,
            )
        values = direct_value_cache[cache_key]
        state_fips = spec.metadata.get("state_fips")
        if state_fips:
            values = np.where(
                household["state_fips"].to_numpy() == int(state_fips),
                values,
                0.0,
            )
        congressional_district_geoid = spec.metadata.get("congressional_district_geoid")
        if congressional_district_geoid:
            if household_congressional_district_geoid is None:
                raise RuntimeError(
                    "Congressional-district target rows require a household "
                    "congressional_district_geoid column in the support frame."
                )
            values = np.where(
                household_congressional_district_geoid
                == int(congressional_district_geoid),
                values,
                0.0,
            )
        hh[spec.measure] = values

    direct_measures = {
        spec.measure
        for spec in target_specs
        if spec.measure
        and spec.family not in {"irs_soi", "jct", "state_income_tax"}
        and spec.metadata.get("materializer") != "policyengine_variable"
        and spec.measure not in hh.columns
    }
    for measure in sorted(direct_measures):
        if measure not in system.variables:
            continue
        hh[measure] = _household_values(
            frame=base_frame,
            simulation=simulation,
            system=system,
            variable=measure,
            tax_unit_positions=tax_unit_positions,
        )

    for spec in target_specs:
        if spec.family == "state_income_tax":
            state_fips = int(spec.metadata["state_fips"])
            hh[spec.measure] = hh["state_income_tax"].where(
                household["state_fips"].to_numpy() == state_fips,
                0.0,
            )

    variable_cache: dict[str, np.ndarray] = {}
    for source_name, pe_name in SOI_VARIABLE_MAP.items():
        if source_name == "ctc":
            if pe_name not in system.variables:
                continue
            if "ctc_limiting_tax_liability" not in system.variables:
                continue
            total_ctc = _calculate_array(simulation, pe_name)
            limiting_tax = _calculate_array(simulation, "ctc_limiting_tax_liability")
            variable_cache[source_name] = np.maximum(
                np.minimum(total_ctc, limiting_tax),
                0.0,
            ).astype(np.float64)
            continue
        if pe_name == "rent_and_royalty_net_income":
            rental_income = _calculate_array(simulation, "rental_income")
            farm_rent_income = _calculate_array(simulation, "farm_rent_income")
            variable_cache[source_name] = _person_variable_to_tax_unit(
                frame=base_frame,
                values=rental_income + farm_rent_income,
            )
            continue
        if pe_name not in system.variables:
            continue
        entity = _variable_entity(system, pe_name)
        raw = _calculate_array(simulation, pe_name)
        if entity == "tax_unit":
            variable_cache[source_name] = raw.astype(np.float64)
        elif entity == "person":
            variable_cache[source_name] = _person_variable_to_tax_unit(
                frame=base_frame,
                values=raw,
            )
        else:
            raise ValueError(
                f"SOI variable {pe_name!r} has unsupported entity {entity!r}."
            )

    for spec in target_specs:
        if spec.family != "irs_soi":
            continue
        if _unsupported_soi_ledger_filters(spec.metadata):
            continue
        source_name = spec.metadata.get("source_variable", spec.metadata["variable"])
        lower = _as_bound(spec.metadata["agi_lower_bound"])
        upper = _as_bound(spec.metadata["agi_upper_bound"])
        mask = (agi_tax_unit >= lower) & (agi_tax_unit < upper)
        if spec.metadata.get("taxable_only") == "true":
            mask &= (income_tax_tax_unit > 0) | (taxable_income_tax_unit > 0)
        status = FILING_STATUS_MAP[spec.metadata["filing_status"]]
        if isinstance(status, str):
            mask &= filing_status == status
        elif isinstance(status, set):
            mask &= np.isin(filing_status, sorted(status))
        child_filter = _soi_eitc_child_count_filter(spec.metadata)
        if child_filter is not None:
            if eitc_child_count is None:
                continue
            mask &= _eitc_child_count_mask(eitc_child_count, child_filter)
        if _soi_requires_positive_eitc_filter(spec.metadata):
            if "eitc" not in variable_cache:
                continue
            mask &= variable_cache["eitc"] > 0
        if spec.metadata.get("itemized_only") == "true":
            if tax_unit_itemizes is None:
                continue
            mask &= tax_unit_itemizes
        if "state_fips" in spec.metadata:
            mask &= tax_unit_state_fips == int(spec.metadata["state_fips"])
        if "congressional_district_geoid" in spec.metadata:
            if tax_unit_congressional_district_geoid is None:
                raise RuntimeError(
                    "Congressional-district target rows require a household "
                    "congressional_district_geoid column in the support frame."
                )
            mask &= tax_unit_congressional_district_geoid == int(
                spec.metadata["congressional_district_geoid"]
            )
        indicator_sum = spec.metadata.get("measure_mode") == "indicator_sum"
        if source_name == "count":
            values = mask.astype(np.float64)
        else:
            if source_name not in variable_cache:
                continue
            values = (
                _soi_component_row(
                    variable_cache[source_name],
                    source_name,
                    indicator=indicator_sum,
                )
                * mask
            )
        hh[spec.measure] = _collapse_tax_unit(values, tax_unit_positions, n_households)

    base_income_tax_household = hh["income_tax"].to_numpy(dtype=np.float64)
    del (
        direct_value_cache,
        variable_cache,
        income_tax_tax_unit,
        taxable_income_tax_unit,
        agi_tax_unit,
        filing_status,
        eitc_child_count,
        tax_unit_itemizes,
    )
    simulation._invalidate_all_caches()
    del simulation, dataset
    _collect_batch_garbage()
    requested_reform_measures = {spec.measure for spec in target_specs}
    cache_context = (
        dict(target_materialization_cache_context)
        if target_materialization_cache_context is not None
        else None
    )
    cache_stats: dict[str, object] = {
        "enabled": target_materialization_cache_dir is not None,
        "cache_dir": (
            None
            if target_materialization_cache_dir is None
            else str(target_materialization_cache_dir)
        ),
        "schema_version": TARGET_MATERIALIZATION_CACHE_SCHEMA_VERSION,
        "hits": 0,
        "misses": 0,
        "writes": 0,
        "entries": [],
    }
    for reform_spec in US_JCT_TAX_EXPENDITURE_REFORMS:
        if reform_spec.measure not in requested_reform_measures:
            continue
        reform_income_tax = None
        cache_entry: dict[str, object] | None = None
        if target_materialization_cache_dir is not None and cache_context is not None:
            identity = _target_materialization_cache_identity(
                context=cache_context,
                reform_spec=reform_spec,
                n_households=n_households,
            )
            cached = _read_reform_income_tax_cache(
                target_materialization_cache_dir,
                identity,
                n_households=n_households,
            )
            if cached is not None:
                reform_income_tax, cache_digest, cache_path = cached
                cache_stats["hits"] = int(cache_stats["hits"]) + 1
                cache_entry = {
                    "measure": reform_spec.measure,
                    "neutralized_variable": reform_spec.neutralized_variable,
                    "status": "hit",
                    "identity_sha256": cache_digest,
                    "path": str(cache_path),
                }
            else:
                cache_stats["misses"] = int(cache_stats["misses"]) + 1
                cache_digest = _target_materialization_cache_digest(identity)
                cache_entry = {
                    "measure": reform_spec.measure,
                    "neutralized_variable": reform_spec.neutralized_variable,
                    "status": "miss",
                    "identity_sha256": cache_digest,
                }
        if reform_income_tax is None:
            reform_income_tax = _reform_household_income_tax(
                base_frame=base_frame,
                reform_spec=reform_spec,
                system=system,
                microsimulation_cls=Microsimulation,
                n_households=n_households,
                batch_size=maximum_microsim_batch_size,
            )
            if (
                target_materialization_cache_dir is not None
                and cache_context is not None
            ):
                identity = _target_materialization_cache_identity(
                    context=cache_context,
                    reform_spec=reform_spec,
                    n_households=n_households,
                )
                cache_digest, cache_path = _write_reform_income_tax_cache(
                    target_materialization_cache_dir,
                    identity,
                    reform_income_tax,
                )
                cache_stats["writes"] = int(cache_stats["writes"]) + 1
                if cache_entry is None:
                    cache_entry = {
                        "measure": reform_spec.measure,
                        "neutralized_variable": reform_spec.neutralized_variable,
                        "status": "write",
                        "identity_sha256": cache_digest,
                    }
                cache_entry["status"] = "miss_written"
                cache_entry["identity_sha256"] = cache_digest
                cache_entry["path"] = str(cache_path)
        if cache_entry is not None:
            entries = cache_stats["entries"]
            assert isinstance(entries, list)
            entries.append(cache_entry)
        hh[reform_spec.measure] = reform_income_tax - base_income_tax_household
        del reform_income_tax
        _collect_batch_garbage()

    target_frame = Frame(
        materialized,
        US_SCHEMA,
        {"household": base_frame.weights_for("household")},
        base_frame.strata,
    )
    registry, compilation = _compile_materialized_target_registry(
        target_frame,
        target_specs,
        gate_congressional_district_targets=gate_congressional_district_targets,
    )
    compilation = {
        **compilation,
        "target_materialization_cache": cache_stats,
    }
    return (
        target_frame,
        registry,
        compilation,
    )


def _target_spec_is_materialized(spec, household_table: pd.DataFrame) -> bool:
    measure_ready = spec.measure is None or spec.measure in household_table.columns
    filter_ready = spec.filter is None or spec.filter in household_table.columns
    return measure_ready and filter_ready


def _with_calibrated_weights(
    base_frame: Frame, calibrated_weights: np.ndarray
) -> Frame:
    _assert_no_formula_owned_columns(base_frame)
    return base_frame.with_weights(
        "household",
        Weights(calibrated_weights, WeightKind.CALIBRATED),
        mass=MassChange(
            factor=calibrated_weights.sum() / base_frame.weights_for("household").total,
            reason="US fiscal target refresh calibration",
        ),
    )


def _with_l0_refit_weights(base_frame: Frame, result) -> Frame:
    """Attach post-L0 refit weights to the clean selected base-frame support."""
    _assert_no_formula_owned_columns(base_frame)
    return attach_l0_refit_entity_weights(
        base_frame,
        weight_entity=result.weight_entity,
        selected_entity_ids=np.asarray(result.selected_entity_ids),
        selected_weights=np.asarray(result.weights),
        reason="US fiscal target refresh L0/refit calibration",
    )


def _selected_plan_ratio_bucket(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "neutral_count": 0,
            "below_benchmark_count": 0,
            "above_benchmark_count": 0,
            "below_support_count": 0,
            "above_support_count": 0,
        }
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "neutral_count": int(np.isclose(finite, 1.0).sum()),
        "below_benchmark_count": int((finite < 1.0).sum()),
        "above_benchmark_count": int((finite > 1.0).sum()),
        "below_support_count": int((finite < 0.5).sum()),
        "above_support_count": int((finite > 1.5).sum()),
    }


def _selected_plan_ratio_diagnostics(tax_unit: pd.DataFrame) -> dict[str, object]:
    column = "selected_marketplace_plan_benchmark_ratio"
    if column not in tax_unit.columns:
        return {}
    values = pd.to_numeric(tax_unit[column], errors="coerce").to_numpy(dtype=np.float64)
    diagnostics = {
        "support": {"lower": 0.5, "upper": 1.5},
        "all_tax_units": _selected_plan_ratio_bucket(values),
    }
    if "takes_up_aca_if_eligible" in tax_unit.columns:
        takes_up = tax_unit["takes_up_aca_if_eligible"].fillna(False).astype(bool)
        diagnostics["marketplace_takers"] = _selected_plan_ratio_bucket(
            values[takes_up.to_numpy()]
        )
    return diagnostics


def _structural_frame_columns() -> set[str]:
    structural = {US_SCHEMA.person_id_column, "household_weight"}
    for group in US_SCHEMA.group_entities:
        structural.add(US_SCHEMA.id_column(group))
        structural.add(US_SCHEMA.membership_column(group))
    return structural


def _degenerate_input_signal_gate(
    frame: Frame,
    engine: PolicyEngineUSEngine,
) -> GateResult:
    """Sweep every persisted input column for values stuck at the engine default.

    Unlike the health-input gate's named allowlist, this covers the whole
    export surface: any PolicyEngine input column whose values all equal the
    engine default fails unless it carries a reviewed exclusion naming the
    tracking issue.
    """
    structural = _structural_frame_columns()
    column_values: dict[str, object] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        for column in table.columns:
            if column in structural:
                continue
            column_values[column] = table[column].to_numpy()
    defaults = engine.default_values(sorted(column_values))
    gate = default_valued_columns_gate(
        column_values,
        defaults,
        reviewed_exclusions=US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS,
    )
    return GateResult(
        name="degenerate_input_signal",
        passed=gate.passed,
        failures=gate.failures,
        details=gate.details,
    )


def _health_input_signal_gate(frame: Frame) -> GateResult:
    tax_unit = frame.table("tax_unit")
    gate = nonconstant_columns_gate(
        {
            column: tax_unit[column].to_numpy()
            for column in US_HEALTH_INPUT_NONCONSTANT_COLUMNS
            if column in tax_unit.columns
        },
        US_HEALTH_INPUT_NONCONSTANT_COLUMNS,
    )
    details = dict(gate.details)
    selected_plan_diagnostics = _selected_plan_ratio_diagnostics(tax_unit)
    if selected_plan_diagnostics:
        details["selected_marketplace_plan_benchmark_ratio"] = selected_plan_diagnostics
    return GateResult(
        name="health_input_signal",
        passed=gate.passed,
        failures=gate.failures,
        details=details,
    )


def _engine_input_variables() -> tuple[str, ...]:
    """Persistable PolicyEngine input variables (formula-owned excluded)."""
    return tuple(PolicyEngineUSEngine().variables())


def _input_mass_reference_gate(
    base_frame: Frame,
    *,
    reference_h5: Path | None,
    relative_tolerance: float,
    minimum_reference_total: float,
) -> GateResult | None:
    """Gate the base frame's persisted input mass against a certified release.

    A rebuilt base pipeline can drop input bases the incumbent release
    carries (issue #278: IRA/HSA/pension-contribution/childcare inputs) while
    every calibration target still fits. Comparing weighted totals of the
    engine's input variables against the reference release catches that
    before calibration burns hours on a base that cannot score those reforms.
    """
    if reference_h5 is None:
        return None
    input_variables = _engine_input_variables()
    return input_mass_parity_gate(
        us_input_mass_totals(base_frame, columns=input_variables),
        us_input_mass_totals(load_us_frame(reference_h5), columns=input_variables),
        candidate_name="base_frame",
        reference_name=reference_h5.name,
        relative_tolerance=relative_tolerance,
        minimum_reference_total=minimum_reference_total,
    )


def _ecps_parity_gate(
    base_frame: Frame,
    *,
    reference: EcpsParityReference | None = None,
    known_gaps: tuple[ParityKnownGap, ...] | None = None,
) -> GateResult:
    """Gate the candidate frame's populated layers against the incumbent eCPS.

    The launch contract for replacing the enhanced-CPS: every layer the
    incumbent populates, the candidate must populate or exempt by documented
    name (populace #313). Unlike :func:`_input_mass_reference_gate`, whose
    reference mass is recomputed live from a reference H5, the parity reference
    is a *pinned* per-variable nonzero-share file computed once from the
    sha-verified incumbent artifact — a candidate cannot move the bar by
    changing which reference it is compared against. The candidate's own shares
    are measured live over the engine's input-variable surface (formula-owned
    excluded, so a formula-owned output the engine computes is never a parity
    layer). Exemptions come from the checked-in register, each carrying a reason
    and a tracking issue; those reasons are recorded in the gate details so the
    release manifest states the remaining distance from full eCPS parity.
    """
    reference = reference if reference is not None else load_ecps_parity_reference()
    known_gaps = known_gaps if known_gaps is not None else load_ecps_parity_known_gaps()
    input_variables = _engine_input_variables()
    candidate_shares = us_nonzero_shares(base_frame, columns=input_variables)
    gate = parity_gate(
        candidate_shares,
        reference.nonzero_shares,
        known_gaps=tuple(gap.variable for gap in known_gaps),
    )
    details = dict(gate.details)
    details["reference"] = {
        "repo_id": reference.source.repo_id,
        "repo_type": reference.source.repo_type,
        "filename": reference.source.filename,
        "revision": reference.source.revision,
        "sha256": reference.source.sha256,
        "vintage": reference.source.vintage,
        "period": reference.source.period,
    }
    details["candidate_populated_layers"] = sum(
        1 for share in candidate_shares.values() if share > 0.0
    )
    # The reasoned register: names alone say a layer is exempt; the manifest
    # must also carry WHY and which issue owns closing it (the debt ledger).
    details["known_gaps"] = {
        gap.variable: {"reason": gap.reason, "issue": gap.issue} for gap in known_gaps
    }
    return GateResult(
        name=gate.name,
        passed=gate.passed,
        failures=gate.failures,
        details=details,
    )


# Build H (populace#299): export-input-mass reviewed exclusions.
#
# The export-mass gate compares each calibrated export column against the
# live-default 57k reference with a +/-50% band. It never re-references to a
# calibration target — so for a column that Build H now *identifies* with a
# real SOI target, the gate still measures drift against the incumbent's
# incidental value on that column. An exclusion is defensible ONLY where the
# SOI-true level provably cannot sit inside the reference band (the
# incumbent's value on the column is off-source, an artifact of its own
# weight solve); there the exclusion is justified BY the target itself.
# Band math (reference = live-default 57k c2065b64, +/-50%, CBO-aged
# TY2023 -> 2024 factor 1.0872):
#   estate_income          ref $98.434B  band [$49.217B, $147.651B]
#                          SOI net target ~$46.74B  -> BELOW the lower edge;
#                          cannot pass truthfully -> excluded.
#   non_sch_d_capital_gains ref $75.747B band [$37.874B, $113.621B]
#                          SOI target ~$10.16B -> far below the lower edge;
#                          cannot pass truthfully -> excluded.
# Deliberately NOT excluded (parity checks stay live; the run adjudicates):
#   miscellaneous_income   ref $47.401B  band [$23.700B, $71.101B]; SOI net
#                          target ~$52.84B is IN-band -> genuine pass expected.
#   home_mortgage_interest ref $311.126B band [$155.563B, $466.689B]; the new
#                          itemizer-masked Table 2.1 target (~$186.3B aged)
#                          pins the itemizer share and pulls the full column
#                          down toward the band from $474-526B.
#   first_home_mortgage_interest follows home_mortgage_interest (second-home
#                          leg un-imputed / 0 per populace#38).
US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS: dict[str, str] = {
    "estate_income": (
        "Identified by SOI Table 1.4 estate/trust net income (income leg "
        "$47.892B + loss leg $4.899B, TY2023; CBO-aged net ~$46.74B@2024). "
        "The live-default reference carries $98.434B on this column — ~2.1x "
        "the SOI net level — so the +/-50% band's lower edge ($49.217B) sits "
        "ABOVE the true SOI level: a correctly calibrated column cannot pass "
        "this parity check. The reference value is an incidental artifact of "
        "the incumbent weight solve (nothing pinned this column before Build "
        "H). estate_income does not feed AGI in PolicyEngine-US (loss-cap "
        "input only), so pinning it identifies the export dimension without "
        "moving revenue (PolicyEngine/populace#299 Build H)."
    ),
    "non_sch_d_capital_gains": (
        "Identified by SOI Table 1.4 capital gain distributions reported on "
        "Form 1040 ($9.341B TY2023, CBO-aged ~$10.16B@2024) — concept-"
        "confirmed against PolicyEngine-US non_sch_d_capital_gains (PUF "
        "E01100, Form 1040 line 7 when Schedule D is not required). The "
        "live-default reference carries $75.747B — ~8x the SOI value — an "
        "incidental within-net-capital-gains split (the aggregate is "
        "constrained, the component split was not). The band's lower edge "
        "($37.874B) sits far above the true SOI level, so a correctly "
        "calibrated column cannot pass this parity check "
        "(PolicyEngine/populace#299 Build H)."
    ),
}


def _export_input_mass_gate(
    export_frame: Frame,
    base_frame: Frame,
    *,
    relative_tolerance: float,
    minimum_reference_total: float,
    reference_frame: Frame | None = None,
    reference_name: str = "base_frame",
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> GateResult:
    """Gate the export support's persisted input mass against a reference frame.

    L0 selection and reweighting optimize the target surface; an untargeted
    input column can lose its mass without moving any residual. The export
    must keep every material input base the candidate frame carries.

    The reference defaults to ``base_frame`` (the raw pre-calibration base),
    which preserves the historical behaviour. But for a dense parent built from
    a raw pooled-ASEC base, calibration is *supposed* to scale PUF-imputed
    income columns up toward their SOI/CBO fiscal targets, and comparing the
    calibrated export against the raw base flags those correct, target-aligned
    gains as failures (populace #327). Passing a certified-release
    ``reference_frame`` (the live default) puts calibration-driven upward
    alignment of under-reported PUF income in-band while a genuine sparse
    zeroing (candidate == 0 or candidate << reference — the #278 signature)
    still fails. ``reviewed_exclusions`` documents any column allowed to drift.
    """
    input_variables = _engine_input_variables()
    reference = reference_frame if reference_frame is not None else base_frame
    return input_mass_parity_gate(
        us_input_mass_totals(export_frame, columns=input_variables),
        us_input_mass_totals(reference, columns=input_variables),
        candidate_name="export_frame",
        reference_name=reference_name,
        relative_tolerance=relative_tolerance,
        minimum_reference_total=minimum_reference_total,
        reviewed_exclusions=reviewed_exclusions,
    )


def _person_population(frame: Frame) -> float:
    return float(frame.resolve_weights("person").values.sum())


def _base_population_relative_error(population: float) -> float | None:
    benchmark = US_BASE_PERSON_POPULATION_BENCHMARK
    if not math.isfinite(population) or population <= 0 or not benchmark:
        return None
    return (population - benchmark) / benchmark


def _mass_change_record_payload(record) -> dict[str, object]:
    return {
        "entity": record.entity,
        "old_total": record.old_total,
        "new_total": record.new_total,
        "declared_factor": record.declared_factor,
        "reason": record.reason,
    }


def _with_base_population_mass_repair(
    frame: Frame,
) -> tuple[Frame, dict[str, object]]:
    initial_population = _person_population(frame)
    initial_relative_error = _base_population_relative_error(initial_population)
    benchmark = US_BASE_PERSON_POPULATION_BENCHMARK
    if initial_relative_error is None:
        raise RuntimeError(
            "Base population mass repair requires a positive, finite weighted "
            f"person population; got {initial_population!r}."
        )

    factor = benchmark / initial_population
    applied = not math.isclose(factor, 1.0, rel_tol=1e-12, abs_tol=0.0)
    repaired = frame
    if applied:
        weights = frame.weights_for("household")
        repaired = frame.with_weights(
            "household",
            weights.with_values(weights.values * factor, weights.kind),
            mass=MassChange(
                factor=factor,
                reason=US_BASE_PERSON_POPULATION_REPAIR_REASON,
            ),
        )

    repaired_population = _person_population(repaired)
    repaired_relative_error = _base_population_relative_error(repaired_population)
    payload: dict[str, object] = {
        "method": "rescale_household_weights_to_census_person_population",
        "applied": applied,
        "reason": US_BASE_PERSON_POPULATION_REPAIR_REASON,
        "initial_population": initial_population,
        "benchmark": benchmark,
        "factor": factor,
        "initial_relative_error": initial_relative_error,
        "repaired_population": repaired_population,
        "repaired_relative_error": repaired_relative_error,
    }
    if applied:
        payload["mass_change"] = _mass_change_record_payload(repaired.mass_log[-1])
    return repaired, payload


def _with_social_security_component_value_repair(
    frame: Frame,
    target_specs: Iterable[object],
) -> tuple[Frame, dict[str, object]]:
    targets_by_column: dict[str, float] = {}
    for spec in target_specs:
        role = spec.metadata.get("target_role")
        column = US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES.get(role)
        if column is not None:
            targets_by_column[column] = float(spec.value)

    missing_targets = sorted(
        set(US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES.values()) - set(targets_by_column)
    )
    if missing_targets:
        raise RuntimeError(
            "Social Security component repair requires target(s) for "
            f"{missing_targets}."
        )

    person = frame.table("person").copy()
    person_weights = pd.Series(
        frame.resolve_weights("person").values, index=person.index
    )
    component_payload: dict[str, object] = {}
    applied = False
    for column, target in targets_by_column.items():
        if column not in person.columns:
            raise RuntimeError(
                f"Social Security component repair requires person column {column!r}."
            )
        values = pd.to_numeric(person[column], errors="coerce").fillna(0.0)
        initial = float((values * person_weights).sum())
        if not math.isfinite(initial) or initial <= 0.0:
            raise RuntimeError(
                "Social Security component repair requires positive finite "
                f"support for {column!r}; got {initial!r}."
            )
        factor = target / initial
        if not math.isclose(factor, 1.0, rel_tol=1e-12, abs_tol=0.0):
            person[column] = values.to_numpy(dtype=np.float64) * factor
            applied = True
        component_payload[column] = {
            "target": target,
            "initial_estimate": initial,
            "factor": factor,
            "repaired_estimate": initial * factor,
        }

    tables_out = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables_out["person"] = person
    repaired = Frame(
        tables_out,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )
    return repaired, {
        "method": "rescale_social_security_component_leaves_to_ssa_targets",
        "applied": applied,
        "reason": US_SOCIAL_SECURITY_COMPONENT_REPAIR_REASON,
        "components": component_payload,
    }


def _base_population_scale_gate(
    frame: Frame,
    *,
    mass_repair: Mapping[str, object] | None = None,
) -> GateResult:
    population = _person_population(frame)
    benchmark = US_BASE_PERSON_POPULATION_BENCHMARK
    relative_error = _base_population_relative_error(population)
    max_abs = US_BASE_PERSON_POPULATION_MAX_ABS_RELATIVE_ERROR
    passed = relative_error is not None and abs(relative_error) <= max_abs
    details = {
        "measure": "person_weight",
        "population": population if math.isfinite(population) else None,
        "benchmark": benchmark,
        "relative_error": relative_error,
        "max_abs_relative_error": max_abs,
        "calibration_mass_policy": "conserve",
    }
    if mass_repair is not None:
        details["mass_repair"] = dict(mass_repair)
    if passed:
        return GateResult(
            name="base_population_scale",
            passed=True,
            details=details,
        )
    if relative_error is None:
        failure = "weighted person population is non-finite."
    else:
        failure = (
            f"weighted person population {population:,.0f} differs from Census "
            f"benchmark {benchmark:,.0f} by {relative_error:.1%}; release "
            "calibration uses mass='conserve', so the base H5 must already be "
            "national scale."
        )
    return GateResult(
        name="base_population_scale",
        passed=False,
        failures=(failure,),
        details=details,
    )


def _write_npz(path: Path, *, result, registry: TargetRegistry) -> None:
    np.savez_compressed(
        path,
        household_weight=result.weights,
        initial_household_weight=result.initial_weights,
        target_names=np.asarray([d.name for d in result.diagnostics], dtype=object),
        target_values=np.asarray(
            [d.target for d in result.diagnostics], dtype=np.float64
        ),
        initial_estimates=np.asarray(
            [d.initial_estimate for d in result.diagnostics], dtype=np.float64
        ),
        final_estimates=np.asarray(
            [d.final_estimate for d in result.diagnostics], dtype=np.float64
        ),
        relative_errors=np.asarray(
            [d.relative_error for d in result.diagnostics], dtype=np.float64
        ),
        registry_version=np.asarray(registry.version),
    )


def _load_warm_start_calibration_npz(
    path: Path,
    *,
    expected_initial_weights: np.ndarray,
) -> tuple[np.ndarray, Mapping[str, object]]:
    expected_initial = np.asarray(expected_initial_weights, dtype=np.float64)
    if expected_initial.ndim != 1:
        raise ValueError(
            "expected_initial_weights must be a one-dimensional household vector."
        )
    if not path.exists():
        raise FileNotFoundError(f"Warm-start calibration NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required = {"household_weight", "initial_household_weight"}
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(
                f"{path} is missing warm-start calibration arrays: {missing}."
            )
        weights = np.asarray(data["household_weight"], dtype=np.float64)
        stored_initial = np.asarray(data["initial_household_weight"], dtype=np.float64)

    if weights.shape != expected_initial.shape:
        raise ValueError(
            "warm-start household_weight shape does not match current calibration "
            f"frame: got {weights.shape}, expected {expected_initial.shape}."
        )
    if stored_initial.shape != expected_initial.shape:
        raise ValueError(
            "warm-start initial_household_weight shape does not match current "
            f"calibration frame: got {stored_initial.shape}, expected "
            f"{expected_initial.shape}."
        )
    if not np.isfinite(weights).all():
        raise ValueError("warm-start household_weight values must be finite.")
    if (weights <= 0.0).any():
        raise ValueError(
            "warm-start household_weight values must be strictly positive."
        )
    if not np.isfinite(stored_initial).all():
        raise ValueError("warm-start initial_household_weight values must be finite.")
    if (stored_initial <= 0.0).any():
        raise ValueError(
            "warm-start initial_household_weight values must be strictly positive."
        )
    delta = np.abs(stored_initial - expected_initial)
    if not np.allclose(stored_initial, expected_initial, rtol=1e-9, atol=1e-6):
        raise ValueError(
            "warm-start calibration was built from different initial household "
            "weights; refusing to continue a run on a different support frame "
            "or record order."
        )

    payload = {
        "enabled": True,
        "path": str(path),
        "sha256": _sha256(path),
        "n_households": int(weights.shape[0]),
        "household_weight_sum": float(weights.sum()),
        "initial_household_weight_sum": float(stored_initial.sum()),
        "max_abs_initial_household_weight_delta": float(delta.max(initial=0.0)),
    }
    return weights, payload


def _fiscal_target_loss_weights(registry: TargetRegistry) -> np.ndarray:
    weights = _fiscal_target_concept_budget_weights(registry)
    bases = np.asarray(
        [_fiscal_target_value_basis(spec) for spec in registry.specs],
        dtype=object,
    )
    unique_bases = sorted(set(bases.tolist()))
    if not unique_bases:
        return weights
    basis_total = len(weights) / len(unique_bases)
    for basis in unique_bases:
        mask = bases == basis
        current_total = weights[mask].sum()
        if current_total > 0:
            weights[mask] *= basis_total / current_total
    return weights / weights.mean()


def _fiscal_target_concept_budget_weights(registry: TargetRegistry) -> np.ndarray:
    weights = _fiscal_target_value_basis_weights(registry)
    group_indices: dict[tuple[object, ...], list[int]] = {}
    for index, spec in enumerate(registry.specs):
        group_indices.setdefault(_fiscal_target_concept_budget_key(spec), []).append(
            index
        )
    for indices in group_indices.values():
        group_weights = weights[indices]
        group_total = float(group_weights.sum())
        group_budget = float(group_weights.max(initial=0.0))
        if group_total > 0 and group_budget > 0:
            weights[indices] *= group_budget / group_total
    return weights


def _fiscal_target_concept_budget_key(spec) -> tuple[object, ...]:
    metadata = spec.metadata
    if metadata.get("ledger_geography_level") != "congressional_district":
        return (
            _fiscal_target_value_basis(spec),
            spec.entity,
            spec.period,
            spec.family,
            spec.name,
        )
    semantic_metadata = tuple(
        sorted(
            (key, value)
            for key, value in metadata.items()
            if key not in US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS
        )
    )
    return (
        _fiscal_target_value_basis(spec),
        spec.entity,
        spec.period,
        spec.family,
        spec.filter or "",
        metadata.get("state_fips", ""),
        semantic_metadata,
    )


def _fiscal_target_value_basis_weights(registry: TargetRegistry) -> np.ndarray:
    weights = np.ones(len(registry.specs), dtype=np.float64)
    bases = np.asarray(
        [_fiscal_target_value_basis(spec) for spec in registry.specs],
        dtype=object,
    )
    values = np.asarray(
        [max(abs(float(spec.value)), 1.0) for spec in registry.specs],
        dtype=np.float64,
    )
    raw_weights = values**US_FISCAL_TARGET_VALUE_WEIGHT_POWER
    for basis in sorted(set(bases.tolist())):
        mask = bases == basis
        mean_value = raw_weights[mask].mean()
        if mean_value > 0:
            weights[mask] = raw_weights[mask] / mean_value
    return weights


def _fiscal_target_value_basis(spec) -> str:
    metadata = spec.metadata
    measure_mode = metadata.get("measure_mode", "")
    source_measure_id = metadata.get("source_measure_id", "")
    if measure_mode in {
        "indicator_sum",
        "less_than_indicator_sum",
    }:
        return "count"
    if "enrollment" in source_measure_id or "recipients" in source_measure_id:
        return "count"
    if "return" in source_measure_id and "count" in source_measure_id:
        return "count"
    return "amount"


def _target_is_congressional_district(target: object) -> bool:
    metadata = getattr(target, "metadata", {}) or {}
    return (
        metadata.get("ledger_geography_level") == "congressional_district"
        or metadata.get("geography_scope") == "congressional_district"
        or bool(metadata.get("congressional_district_geoid"))
    )


def _target_row_name(target: object) -> str:
    row_name = getattr(target, "row_name", None)
    if row_name is not None:
        return str(row_name)
    name = getattr(target, "name", "")
    period = getattr(target, "period", None)
    return str(name) if period is None else f"{name}@{period}"


def _diagnostic_targets_by_name(result) -> dict[str, object]:
    problem = getattr(result, "problem", None)
    if problem is None:
        return {}
    targets = tuple(getattr(problem, "targets", ()) or ())
    if not targets:
        return {}
    names = tuple(getattr(problem, "names", ()) or ())
    if len(names) == len(targets):
        return {str(name): target for name, target in zip(names, targets, strict=True)}
    return {_target_row_name(target): target for target in targets}


def _congressional_district_release_gates_enabled(
    compilation: Mapping[str, object],
) -> bool:
    return bool(compilation.get("gate_congressional_district_targets", True))


def _release_gate_failures(
    result,
    compilation: Mapping[str, object],
    target_profile_gate: GateResult | None = None,
    health_input_gate: GateResult | None = None,
    base_population_gate: GateResult | None = None,
    incumbent_diagnostics: Mapping[str, Mapping[str, object]] | None = None,
    immigration_gate: GateResult | None = None,
    input_mass_reference_gate: GateResult | None = None,
    degenerate_input_gate: GateResult | None = None,
    ecps_parity_gate: GateResult | None = None,
    hours_worked_gate: GateResult | None = None,
    snap_take_up_gate: GateResult | None = None,
    eligibility_inputs_gate: GateResult | None = None,
    pregnancy_gate: GateResult | None = None,
    snap_discretionary_exemption_gate: GateResult | None = None,
) -> list[str]:
    failures: list[str] = []
    if target_profile_gate is not None and not target_profile_gate.passed:
        failures.extend(
            f"Target profile coverage failed: {failure}"
            for failure in target_profile_gate.failures
        )
    if base_population_gate is not None and not base_population_gate.passed:
        failures.extend(
            f"Base population scale failed: {failure}"
            for failure in base_population_gate.failures
        )
    if health_input_gate is not None and not health_input_gate.passed:
        failures.extend(
            f"Health input signal failed: {failure}"
            for failure in health_input_gate.failures
        )
    if immigration_gate is not None and not immigration_gate.passed:
        failures.extend(
            f"Immigration composition failed: {failure}"
            for failure in immigration_gate.failures
        )
    if hours_worked_gate is not None and not hours_worked_gate.passed:
        failures.extend(
            f"Hours-worked signal failed: {failure}"
            for failure in hours_worked_gate.failures
        )
    if snap_take_up_gate is not None and not snap_take_up_gate.passed:
        failures.extend(
            f"SNAP take-up signal failed: {failure}"
            for failure in snap_take_up_gate.failures
        )
    if eligibility_inputs_gate is not None and not eligibility_inputs_gate.passed:
        failures.extend(
            f"Eligibility-inputs signal failed: {failure}"
            for failure in eligibility_inputs_gate.failures
        )
    if pregnancy_gate is not None and not pregnancy_gate.passed:
        failures.extend(
            f"Pregnancy signal failed: {failure}" for failure in pregnancy_gate.failures
        )
    if (
        snap_discretionary_exemption_gate is not None
        and not snap_discretionary_exemption_gate.passed
    ):
        failures.extend(
            f"SNAP discretionary-exemption signal failed: {failure}"
            for failure in snap_discretionary_exemption_gate.failures
        )
    if input_mass_reference_gate is not None and not input_mass_reference_gate.passed:
        failures.extend(
            f"Input mass parity failed: {failure}"
            for failure in input_mass_reference_gate.failures
        )
    if degenerate_input_gate is not None and not degenerate_input_gate.passed:
        failures.extend(
            f"Degenerate input signal failed: {failure}"
            for failure in degenerate_input_gate.failures
        )
    if ecps_parity_gate is not None and not ecps_parity_gate.passed:
        failures.extend(
            f"eCPS parity failed: {failure}" for failure in ecps_parity_gate.failures
        )
    gate_congressional_district_targets = _congressional_district_release_gates_enabled(
        compilation
    )
    diagnostic_only_dropped_target_names = set(
        compilation.get("diagnostic_only_dropped_target_names") or ()
    )
    dropped = compilation.get("dropped_target_names") or []
    if not gate_congressional_district_targets:
        dropped = [
            name for name in dropped if name not in diagnostic_only_dropped_target_names
        ]
    if dropped:
        failures.append(f"{len(dropped)} fiscal targets were not materialized.")
    skipped = tuple(getattr(result, "skipped", ()) or ())
    if not gate_congressional_district_targets:
        skipped = tuple(
            skipped_target
            for skipped_target in skipped
            if not _target_is_congressional_district(
                getattr(skipped_target, "target", None)
            )
        )
    if skipped:
        failures.append(f"{len(skipped)} fiscal targets were skipped by calibration.")
    if not result.diagnostics:
        failures.append("No fiscal targets were compiled.")
    diagnostic_targets = _diagnostic_targets_by_name(result)
    zero_support = [
        diagnostic.name
        for diagnostic in result.diagnostics
        if float(getattr(diagnostic, "target", 0.0)) > 0.0
        and abs(float(getattr(diagnostic, "initial_estimate", 0.0))) <= 1e-9
        and abs(float(getattr(diagnostic, "final_estimate", 0.0))) <= 1e-9
        and (
            gate_congressional_district_targets
            or not _target_is_congressional_district(
                diagnostic_targets.get(diagnostic.name)
            )
        )
    ]
    if zero_support:
        examples = ", ".join(zero_support[:5])
        suffix = "" if len(zero_support) <= 5 else ", ..."
        failures.append(
            f"{len(zero_support)} positive fiscal targets have zero "
            f"materialized support (examples: {examples}{suffix})."
        )
    failures.extend(
        _critical_target_fit_failures(
            result,
            incumbent_diagnostics=incumbent_diagnostics,
        )
    )
    if not math.isfinite(result.initial_loss) or not math.isfinite(result.final_loss):
        failures.append("Calibration loss is non-finite.")
    elif result.final_loss > result.initial_loss:
        failures.append(
            "Calibration final loss is worse than the initial loss "
            f"({result.final_loss} > {result.initial_loss})."
        )
    return failures


def _critical_target_fit_failures(
    result,
    *,
    incumbent_diagnostics: Mapping[str, Mapping[str, object]] | None = None,
) -> list[str]:
    incumbent_diagnostics = incumbent_diagnostics or {}
    diagnostics_by_name = {
        getattr(diagnostic, "name", None): diagnostic
        for diagnostic in getattr(result, "diagnostics", ())
    }
    failures: list[str] = []
    for requirement in US_CRITICAL_TARGET_FIT_REQUIREMENTS:
        diagnostic = diagnostics_by_name.get(requirement["name"])
        if diagnostic is None:
            failures.append(
                "Critical fiscal target "
                f"{requirement['name']!r} ({requirement['label']}) is missing "
                "from calibration diagnostics."
            )
            continue
        relative_error = getattr(diagnostic, "relative_error", None)
        computed_relative_error = _diagnostic_relative_error(diagnostic, failures)
        if computed_relative_error is None:
            continue
        if not isinstance(relative_error, int | float):
            failures.append(
                "Critical fiscal target "
                f"{requirement['name']!r} ({requirement['label']}) has "
                f"non-numeric relative_error {relative_error!r}."
            )
        elif not math.isclose(
            float(relative_error),
            computed_relative_error,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            failures.append(
                "Critical fiscal target "
                f"{requirement['name']!r} ({requirement['label']}) has "
                f"stale relative_error {relative_error!r}; computed "
                f"{computed_relative_error:.6g} from target and final_estimate."
            )
        max_abs = float(requirement["max_abs_relative_error"])
        if abs(computed_relative_error) > max_abs:
            incumbent_relative_error = _incumbent_relative_error(
                incumbent_diagnostics.get(requirement["name"]),
                current_target=float(getattr(diagnostic, "target", 0.0)),
            )
            improved_over_incumbent = incumbent_relative_error is not None and abs(
                computed_relative_error
            ) < abs(incumbent_relative_error)
            if (
                improved_over_incumbent
                and abs(computed_relative_error)
                <= US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR
            ):
                continue
            failures.append(
                "Critical fiscal target "
                f"{requirement['name']!r} ({requirement['label']}) has "
                f"relative_error={computed_relative_error:.6g}, exceeding "
                f"{max_abs:.6g}; target={getattr(diagnostic, 'target', None)!r}, "
                "final_estimate="
                f"{getattr(diagnostic, 'final_estimate', None)!r}"
                + (
                    "."
                    if incumbent_relative_error is None
                    else (
                        f"; incumbent_relative_error={incumbent_relative_error:.6g}; "
                        "improvement_hard_stop="
                        f"{US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR:.6g}."
                    )
                )
            )
    return failures


def _incumbent_relative_error(
    row: Mapping[str, object] | None,
    *,
    current_target: float | None = None,
) -> float | None:
    if row is None:
        return None
    target_value = row.get("target")
    final_estimate = row.get("final_estimate")
    if not isinstance(target_value, int | float) or not isinstance(
        final_estimate, int | float
    ):
        return None
    target_value = float(target_value)
    final_estimate = float(final_estimate)
    if not math.isfinite(target_value) or not math.isfinite(final_estimate):
        return None
    if current_target is not None and not math.isclose(
        target_value,
        current_target,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        return None
    if target_value == 0.0:
        return final_estimate - target_value
    return (final_estimate - target_value) / target_value


def _incumbent_critical_target_payload(
    incumbent_diagnostics: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for requirement in US_CRITICAL_TARGET_FIT_REQUIREMENTS:
        name = requirement["name"]
        row = incumbent_diagnostics.get(name)
        if row is None:
            continue
        target_value = row.get("target")
        final_estimate = row.get("final_estimate")
        relative_error = _incumbent_relative_error(row)
        if (
            isinstance(target_value, int | float)
            and isinstance(final_estimate, int | float)
            and relative_error is not None
        ):
            payload[name] = {
                "target": float(target_value),
                "final_estimate": float(final_estimate),
                "relative_error": float(relative_error),
            }
    return payload


def _diagnostic_relative_error(diagnostic, failures: list[str]) -> float | None:
    target_value = getattr(diagnostic, "target", None)
    final_estimate = getattr(diagnostic, "final_estimate", None)
    if not isinstance(target_value, int | float) or not isinstance(
        final_estimate, int | float
    ):
        failures.append(
            "Critical fiscal target "
            f"{getattr(diagnostic, 'name', None)!r} has non-numeric "
            f"target/final_estimate: target={target_value!r}, "
            f"final_estimate={final_estimate!r}."
        )
        return None
    target_value = float(target_value)
    final_estimate = float(final_estimate)
    if not math.isfinite(target_value) or not math.isfinite(final_estimate):
        failures.append(
            "Critical fiscal target "
            f"{getattr(diagnostic, 'name', None)!r} has non-finite "
            f"target/final_estimate: target={target_value!r}, "
            f"final_estimate={final_estimate!r}."
        )
        return None
    if target_value == 0.0:
        return final_estimate - target_value
    return (final_estimate - target_value) / target_value


def _assert_release_gates(
    result,
    compilation: Mapping[str, object],
    target_profile_gate: GateResult | None = None,
    health_input_gate: GateResult | None = None,
    base_population_gate: GateResult | None = None,
    incumbent_diagnostics: Mapping[str, Mapping[str, object]] | None = None,
    immigration_gate: GateResult | None = None,
    degenerate_input_gate: GateResult | None = None,
    ecps_parity_gate: GateResult | None = None,
) -> None:
    failures = _release_gate_failures(
        result,
        compilation,
        target_profile_gate,
        health_input_gate,
        base_population_gate,
        incumbent_diagnostics,
        immigration_gate,
        degenerate_input_gate=degenerate_input_gate,
        ecps_parity_gate=ecps_parity_gate,
    )
    if failures:
        raise RuntimeError("Release gates failed: " + "; ".join(failures))


def _write_release_calibration_diagnostics(
    *,
    result,
    release_dir: Path,
    registry: TargetRegistry,
    base_h5: Path,
    compilation: Mapping[str, object],
    target_profile_gate: GateResult,
    health_input_gate: GateResult | None,
    base_population_gate: GateResult | None,
    support_value_repairs: Mapping[str, object] | None,
    audit_export_targets: bool,
    immigration_gate: GateResult | None = None,
    input_mass_reference_gate: GateResult | None = None,
    hours_worked_gate: GateResult | None = None,
    snap_take_up_gate: GateResult | None = None,
    eligibility_inputs_gate: GateResult | None = None,
    pregnancy_gate: GateResult | None = None,
    snap_discretionary_exemption_gate: GateResult | None = None,
    gate_failures: Iterable[str],
    timing: Mapping[str, object] | None = None,
    warm_start_calibration: Mapping[str, object] | None = None,
    selection_source: Mapping[str, object] | None = None,
    default_dataset: Mapping[str, object] | None = None,
    incumbent_diagnostics_path: Path | None = None,
    incumbent_diagnostics: Mapping[str, Mapping[str, object]] | None = None,
    degenerate_input_gate: GateResult | None = None,
    ecps_parity_gate: GateResult | None = None,
    validation_input_coverage_gate: GateResult | None = None,
) -> None:
    """Write calibration diagnostics even when hard release gates fail."""
    failures = list(gate_failures)
    incumbent_rows = incumbent_diagnostics or {}
    incumbent_payload = (
        {
            "path": str(incumbent_diagnostics_path),
            "sha256": _sha256(incumbent_diagnostics_path),
            "critical_targets": _incumbent_critical_target_payload(incumbent_rows),
        }
        if incumbent_diagnostics_path is not None
        else None
    )
    write_calibration_diagnostics(
        result,
        release_dir / "calibration_diagnostics.json",
        target_registry=registry,
        build={
            "base_dataset_sha256": _sha256(base_h5),
            "target_compilation": compilation,
            "target_loss_weighting": US_FISCAL_TARGET_LOSS_WEIGHTING,
            "target_loss_cap": US_FISCAL_TARGET_LOSS_CAP,
            "target_profile_coverage": {
                "passed": target_profile_gate.passed,
                "failures": list(target_profile_gate.failures),
                "details": dict(target_profile_gate.details),
            },
            "health_input_signal": (
                {
                    "passed": health_input_gate.passed,
                    "failures": list(health_input_gate.failures),
                    "details": dict(health_input_gate.details),
                }
                if health_input_gate is not None
                else None
            ),
            "degenerate_input_signal": (
                {
                    "passed": degenerate_input_gate.passed,
                    "failures": list(degenerate_input_gate.failures),
                    "details": dict(degenerate_input_gate.details),
                }
                if degenerate_input_gate is not None
                else None
            ),
            "base_population_scale": (
                {
                    "passed": base_population_gate.passed,
                    "failures": list(base_population_gate.failures),
                    "details": dict(base_population_gate.details),
                }
                if base_population_gate is not None
                else None
            ),
            "immigration_composition": (
                {
                    "passed": immigration_gate.passed,
                    "failures": list(immigration_gate.failures),
                    "details": dict(immigration_gate.details),
                }
                if immigration_gate is not None
                else None
            ),
            "hours_worked_signal": (
                {
                    "passed": hours_worked_gate.passed,
                    "failures": list(hours_worked_gate.failures),
                    "details": dict(hours_worked_gate.details),
                }
                if hours_worked_gate is not None
                else None
            ),
            "snap_take_up_signal": (
                {
                    "passed": snap_take_up_gate.passed,
                    "failures": list(snap_take_up_gate.failures),
                    "details": dict(snap_take_up_gate.details),
                }
                if snap_take_up_gate is not None
                else None
            ),
            "eligibility_inputs_signal": (
                {
                    "passed": eligibility_inputs_gate.passed,
                    "failures": list(eligibility_inputs_gate.failures),
                    "details": dict(eligibility_inputs_gate.details),
                }
                if eligibility_inputs_gate is not None
                else None
            ),
            "pregnancy_signal": (
                {
                    "passed": pregnancy_gate.passed,
                    "failures": list(pregnancy_gate.failures),
                    "details": dict(pregnancy_gate.details),
                }
                if pregnancy_gate is not None
                else None
            ),
            "snap_discretionary_exemption_signal": (
                {
                    "passed": snap_discretionary_exemption_gate.passed,
                    "failures": list(snap_discretionary_exemption_gate.failures),
                    "details": dict(snap_discretionary_exemption_gate.details),
                }
                if snap_discretionary_exemption_gate is not None
                else None
            ),
            "documented_absent_inputs": dict(US_DOCUMENTED_ABSENT_INPUTS),
            "input_mass_reference": (
                {
                    "passed": input_mass_reference_gate.passed,
                    "failures": list(input_mass_reference_gate.failures),
                    "details": dict(input_mass_reference_gate.details),
                }
                if input_mass_reference_gate is not None
                else None
            ),
            "ecps_parity": (
                {
                    "passed": ecps_parity_gate.passed,
                    "failures": list(ecps_parity_gate.failures),
                    "details": dict(ecps_parity_gate.details),
                }
                if ecps_parity_gate is not None
                else None
            ),
            "validation_input_coverage": (
                {
                    "passed": validation_input_coverage_gate.passed,
                    "failures": list(validation_input_coverage_gate.failures),
                    "details": dict(validation_input_coverage_gate.details),
                }
                if validation_input_coverage_gate is not None
                else None
            ),
            "support_value_repairs": support_value_repairs,
            "warm_start_calibration": (
                dict(warm_start_calibration)
                if warm_start_calibration is not None
                else {"enabled": False}
            ),
            "selection_source": (
                dict(selection_source)
                if selection_source is not None
                else {"enabled": False}
            ),
            "default_dataset": (
                dict(default_dataset) if default_dataset is not None else None
            ),
            "timing": dict(timing or {}),
            "release_gates": {
                "passed": not failures,
                "failures": failures,
            },
            "incumbent_diagnostics": incumbent_payload,
            "post_export_target_audit": bool(audit_export_targets),
        },
    )


def _target_final_estimate(result, target_name: str) -> float:
    for diagnostic in result.diagnostics:
        if diagnostic.name == f"{target_name}@{PERIOD}":
            return float(diagnostic.final_estimate)
    raise KeyError(target_name)


def _state_income_tax_target_sum(result) -> float:
    return float(
        sum(
            diagnostic.final_estimate
            for diagnostic in result.diagnostics
            if diagnostic.name.startswith("state/")
            and diagnostic.name.endswith(f"/state_income_tax@{PERIOD}")
        )
    )


def _assert_export_matches_calibration(
    dataset_path: Path,
    result,
    target_specs: tuple,
    *,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> None:
    target_frame, registry, compilation = _materialize_target_frame(
        _load_frame(dataset_path),
        target_specs,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )
    dropped = compilation.get("dropped_target_names") or []
    if dropped:
        raise RuntimeError(
            "Post-export sanity failed: "
            f"{len(dropped)} fiscal targets were not materialized after export."
        )
    diagnostics_by_name = {
        diagnostic.name: diagnostic for diagnostic in result.diagnostics
    }
    failures = []
    for target in registry.to_target_set():
        diagnostic = diagnostics_by_name.get(target.row_name)
        if diagnostic is None:
            failures.append(f"{target.row_name} missing from calibration diagnostics")
            continue
        expected = float(diagnostic.final_estimate)
        observed = target.achieved_value(
            target_frame, target_frame.weights_for(target.entity).values
        )
        tolerance = max(
            abs(expected) * POST_EXPORT_RELATIVE_TOLERANCE,
            POST_EXPORT_ABSOLUTE_TOLERANCE,
        )
        if abs(observed - expected) > tolerance:
            failures.append(
                f"{target.row_name} exported value {observed:.6g} differs from "
                f"calibration final estimate {expected:.6g} by more than "
                f"{tolerance:.6g}"
            )
    if failures:
        raise RuntimeError("Post-export sanity failed: " + "; ".join(failures))


def _artifact_entry(path: str, sha: str, *, kind: str, revision: str) -> dict[str, str]:
    return {
        "kind": kind,
        "path": path,
        "repo_id": REPO_ID,
        "revision": revision,
        "sha256": sha,
    }


def _in_sample_estimates(result) -> dict[str, float]:
    """Calibrated final estimate per JCT target, keyed by target name.

    The in-sample reform validation rows reuse the calibration's own fit (the
    JCT tax-expenditure targets *are* calibration targets), so no extra
    simulation is run for them.
    """
    estimates: dict[str, float] = {}
    for diagnostic, target in zip(
        result.diagnostics, result.problem.targets, strict=True
    ):
        value = diagnostic.final_estimate
        if value is not None and math.isfinite(float(value)):
            estimates[target.name] = float(value)
    return estimates


def _in_sample_targets(result) -> dict[str, float]:
    """Calibration target value (the JCT figure) per target, keyed by name.

    In-sample reforms are JCT tax-expenditure calibration targets, so their JCT
    figure is the target's own value the calibration fit against.
    """
    targets: dict[str, float] = {}
    for diagnostic, target in zip(
        result.diagnostics, result.problem.targets, strict=True
    ):
        value = diagnostic.target
        if value is not None and math.isfinite(float(value)):
            targets[target.name] = float(value)
    return targets


def _write_reform_validation(
    *,
    release_dir: Path,
    dataset_path: Path,
    result,
    registry: TargetRegistry,
    release_id: str,
    simulate_out_of_sample: bool,
) -> None:
    """Emit reform_validation.json: populace budget effects vs JCT scores.

    In-sample JCT tax-expenditure reforms come straight from the calibration
    fit; out-of-sample OBBBA provisions are simulated on the freshly written
    release H5 (skipped if ``simulate_out_of_sample`` is False, e.g. for a fast
    diagnostics-only build).
    """
    specs = load_default_reform_specs(period=PERIOD)
    if not simulate_out_of_sample:
        print(
            "\n".join(
                (
                    "",
                    "!" * 72,
                    "WARNING: --skip-out-of-sample-reforms is set.",
                    "reform_validation.json will publish the in-sample JCT rows only;",
                    "every out-of-sample (OBBBA / tax-expenditure) row will have a null",
                    "budget effect and the dashboard will show no fidelity test for them.",
                    "Do NOT use this for a publishable release.",
                    "!" * 72,
                    "",
                )
            ),
            file=sys.stderr,
        )
    simulate = (
        default_simulate_factory(dataset_path) if simulate_out_of_sample else None
    )
    payload = reform_validation_payload(
        specs,
        period=PERIOD,
        simulate=simulate,
        in_sample_estimates=_in_sample_estimates(result),
        in_sample_targets=_in_sample_targets(result),
        baseline_levels=default_baseline_level_specs(),
        release_id=release_id,
    )
    write_reform_validation(payload, release_dir / "reform_validation.json")


def _write_demographics(
    *,
    release_dir: Path,
    dataset_path: Path,
    release_id: str,
) -> None:
    """Emit demographics.json: the dataset's weighted population by age band.

    The fiscal-refresh release calibrates source-backed Census PEP age targets;
    this file remains a compact summary diagnostic for release consumers.
    """
    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    sim = Microsimulation(dataset=USSingleYearDataset(file_path=str(dataset_path)))
    ages, weights = population_by_age_from_sim(sim, PERIOD)
    payload = demographics_payload(ages, weights, period=PERIOD, release_id=release_id)
    # Household-record counts by state and congressional district: the
    # release's sub-national resolution floor, surfaced on the dashboard.
    payload["geography_coverage"] = geography_coverage_payload(dataset_path)
    write_demographics(payload, release_dir / "demographics.json")


def _build_manifests(
    *,
    release_id: str,
    release_dir: Path,
    artifact_root: Path,
    result,
    registry: TargetRegistry,
    dropped: Mapping[str, object],
    target_profile_gate: GateResult,
    health_input_gate: GateResult | None = None,
    base_population_gate: GateResult | None = None,
    incumbent_diagnostics: Mapping[str, Mapping[str, object]] | None = None,
    immigration_gate: GateResult | None = None,
    input_mass_reference_gate: GateResult | None = None,
    degenerate_input_gate: GateResult | None = None,
    ecps_parity_gate: GateResult | None = None,
    hours_worked_gate: GateResult | None = None,
    snap_take_up_gate: GateResult | None = None,
    eligibility_inputs_gate: GateResult | None = None,
    pregnancy_gate: GateResult | None = None,
    snap_discretionary_exemption_gate: GateResult | None = None,
    timing: Mapping[str, object] | None = None,
    warm_start_calibration: Mapping[str, object] | None = None,
    selection_source: Mapping[str, object] | None = None,
    default_dataset: Mapping[str, object] | None = None,
    medicaid_enrollment_substitutions: Sequence[Mapping[str, object]] = (),
    staging: Mapping[str, object] | None = None,
    ledger_artifact: Mapping[str, object] | None = None,
) -> None:
    dataset_path = artifact_root / DATASET_FILENAME
    calibration_path = artifact_root / CALIBRATION_FILENAME
    diagnostics_path = release_dir / "calibration_diagnostics.json"
    coverage_path = release_dir / "us_source_coverage.json"
    dataset_sha = _sha256(dataset_path)
    calibration_sha = _sha256(calibration_path)
    diagnostics_sha = _sha256(diagnostics_path)
    coverage_sha = _sha256(coverage_path)
    diag = diagnostics_payload(result, target_registry=registry)
    gate_failures = _release_gate_failures(
        result,
        dropped,
        target_profile_gate,
        health_input_gate,
        base_population_gate,
        incumbent_diagnostics,
        immigration_gate,
        input_mass_reference_gate,
        degenerate_input_gate,
        ecps_parity_gate=ecps_parity_gate,
        hours_worked_gate=hours_worked_gate,
        snap_take_up_gate=snap_take_up_gate,
        eligibility_inputs_gate=eligibility_inputs_gate,
        pregnancy_gate=pregnancy_gate,
        snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
    )

    commit = _git_output("rev-parse", "HEAD")
    built_at = datetime.now(UTC).isoformat()
    runtime = _runtime_versions()
    timing_payload = dict(timing or {})
    warm_start_payload = (
        dict(warm_start_calibration)
        if warm_start_calibration is not None
        else {"enabled": False}
    )
    selection_source_payload = (
        dict(selection_source) if selection_source is not None else {"enabled": False}
    )
    default_dataset_payload = (
        dict(default_dataset) if default_dataset is not None else None
    )
    manifest = {
        "build_id": release_id,
        "build_sha": commit[:7],
        "created_at": built_at,
        # Staging telemetry provenance: which staging run (if any) this build
        # published while running, so a release without one is auditable.
        "staging": dict(staging) if staging else None,
        "code": {
            "repository": "PolicyEngine/populace",
            "git_commit": commit,
            "git_dirty": False,
        },
        "runtime": runtime,
        "timing": timing_payload,
        "ledger_artifact": dict(ledger_artifact) if ledger_artifact else None,
        "dataset": {
            "filename": DATASET_FILENAME,
            "sha256": dataset_sha,
            "default": default_dataset_payload,
        },
        "calibration": {
            "filename": CALIBRATION_FILENAME,
            "sha256": calibration_sha,
            "warm_start": warm_start_payload,
            "selection_source": selection_source_payload,
            "target_surface": {
                "sha256": diag["target_surface"]["sha256"],
                "n_targets": diag["target_surface"]["n_targets"],
            },
            "target_registry": {
                "version": registry.version,
                "n_specs": len(registry),
            },
            # Reviewed CMS Medicaid enrollment substitutions (populace#386):
            # the per-state records the register applied to the compiled target
            # surface, so a release artifact shows exactly which point-in-time
            # counts were substituted (and why) alongside the registry version
            # that carries the injected spec's provenance metadata.
            "medicaid_enrollment_substitutions": [
                dict(record) for record in medicaid_enrollment_substitutions
            ],
        },
        "gates": {
            "calibration": {
                "passed": not gate_failures,
                "failures": gate_failures,
                "initial_loss": diag["initial_loss"],
                "final_loss": diag["final_loss"],
                "fraction_within_10pct": diag["fraction_within_10pct"],
            },
            "target_compilation": dropped,
            "target_profile_coverage": {
                "passed": target_profile_gate.passed,
                "failures": list(target_profile_gate.failures),
                "details": dict(target_profile_gate.details),
            },
            **(
                {
                    "base_population_scale": {
                        "passed": base_population_gate.passed,
                        "failures": list(base_population_gate.failures),
                        "details": dict(base_population_gate.details),
                    }
                }
                if base_population_gate is not None
                else {}
            ),
            **(
                {
                    "health_input_signal": {
                        "passed": health_input_gate.passed,
                        "failures": list(health_input_gate.failures),
                        "details": dict(health_input_gate.details),
                    }
                }
                if health_input_gate is not None
                else {}
            ),
            **(
                {
                    "immigration_composition": {
                        "passed": immigration_gate.passed,
                        "failures": list(immigration_gate.failures),
                        "details": dict(immigration_gate.details),
                    }
                }
                if immigration_gate is not None
                else {}
            ),
            **(
                {
                    "degenerate_input_signal": {
                        "passed": degenerate_input_gate.passed,
                        "failures": list(degenerate_input_gate.failures),
                        "details": dict(degenerate_input_gate.details),
                    }
                }
                if degenerate_input_gate is not None
                else {}
            ),
            **(
                {
                    "ecps_parity": {
                        "passed": ecps_parity_gate.passed,
                        "failures": list(ecps_parity_gate.failures),
                        "details": dict(ecps_parity_gate.details),
                    }
                }
                if ecps_parity_gate is not None
                else {}
            ),
            **(
                {
                    "hours_worked_signal": {
                        "passed": hours_worked_gate.passed,
                        "failures": list(hours_worked_gate.failures),
                        "details": dict(hours_worked_gate.details),
                    }
                }
                if hours_worked_gate is not None
                else {}
            ),
            **(
                {
                    "snap_take_up_signal": {
                        "passed": snap_take_up_gate.passed,
                        "failures": list(snap_take_up_gate.failures),
                        "details": dict(snap_take_up_gate.details),
                    }
                }
                if snap_take_up_gate is not None
                else {}
            ),
            **(
                {
                    "eligibility_inputs_signal": {
                        "passed": eligibility_inputs_gate.passed,
                        "failures": list(eligibility_inputs_gate.failures),
                        "details": dict(eligibility_inputs_gate.details),
                    }
                }
                if eligibility_inputs_gate is not None
                else {}
            ),
            **(
                {
                    "pregnancy_signal": {
                        "passed": pregnancy_gate.passed,
                        "failures": list(pregnancy_gate.failures),
                        "details": dict(pregnancy_gate.details),
                    }
                }
                if pregnancy_gate is not None
                else {}
            ),
            **(
                {
                    "snap_discretionary_exemption_signal": {
                        "passed": snap_discretionary_exemption_gate.passed,
                        "failures": list(snap_discretionary_exemption_gate.failures),
                        "details": dict(snap_discretionary_exemption_gate.details),
                    }
                }
                if snap_discretionary_exemption_gate is not None
                else {}
            ),
            "documented_absent_inputs": dict(US_DOCUMENTED_ABSENT_INPUTS),
        },
    }
    (release_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=1, allow_nan=False)
    )

    release_manifest = {
        "schema_version": 1,
        "data_package": {
            "name": "populace-data",
            "version": runtime["populace-data"],
        },
        "default_datasets": {"national": "populace_us_2024"},
        "build": {
            "build_id": release_id,
            "built_at": built_at,
            "built_with_core_package": {
                "name": "policyengine-core",
                "version": runtime["policyengine-core"],
            },
            "built_with_model_package": {
                "name": "policyengine-us",
                "version": runtime["policyengine-us"],
            },
            "timing": timing_payload,
            "ledger_artifact": dict(ledger_artifact) if ledger_artifact else None,
            "warm_start_calibration": warm_start_payload,
            "selection_source": selection_source_payload,
            "default_dataset": default_dataset_payload,
            **(
                {
                    "base_population_scale": {
                        "passed": base_population_gate.passed,
                        "details": dict(base_population_gate.details),
                    }
                }
                if base_population_gate is not None
                else {}
            ),
            **(
                {
                    "immigration_composition": {
                        "passed": immigration_gate.passed,
                        "details": dict(immigration_gate.details),
                    }
                }
                if immigration_gate is not None
                else {}
            ),
            **(
                {
                    "ecps_parity": {
                        "passed": ecps_parity_gate.passed,
                        "details": dict(ecps_parity_gate.details),
                    }
                }
                if ecps_parity_gate is not None
                else {}
            ),
            **(
                {
                    "hours_worked_signal": {
                        "passed": hours_worked_gate.passed,
                        "details": dict(hours_worked_gate.details),
                    }
                }
                if hours_worked_gate is not None
                else {}
            ),
            **(
                {
                    "snap_take_up_signal": {
                        "passed": snap_take_up_gate.passed,
                        "details": dict(snap_take_up_gate.details),
                    }
                }
                if snap_take_up_gate is not None
                else {}
            ),
            **(
                {
                    "eligibility_inputs_signal": {
                        "passed": eligibility_inputs_gate.passed,
                        "details": dict(eligibility_inputs_gate.details),
                    }
                }
                if eligibility_inputs_gate is not None
                else {}
            ),
            **(
                {
                    "pregnancy_signal": {
                        "passed": pregnancy_gate.passed,
                        "details": dict(pregnancy_gate.details),
                    }
                }
                if pregnancy_gate is not None
                else {}
            ),
            **(
                {
                    "snap_discretionary_exemption_signal": {
                        "passed": snap_discretionary_exemption_gate.passed,
                        "details": dict(snap_discretionary_exemption_gate.details),
                    }
                }
                if snap_discretionary_exemption_gate is not None
                else {}
            ),
            "documented_absent_inputs": dict(US_DOCUMENTED_ABSENT_INPUTS),
        },
        "compatible_core_packages": [
            {
                "name": "policyengine-core",
                "specifier": f"=={runtime['policyengine-core']}",
            }
        ],
        "compatible_model_packages": [
            {
                "name": "policyengine-us",
                "specifier": f"=={runtime['policyengine-us']}",
            }
        ],
        "artifacts": {
            "populace_us_2024": _artifact_entry(
                DATASET_FILENAME,
                dataset_sha,
                kind="microdata",
                revision=release_id,
            ),
            "populace_us_2024_calibration": _artifact_entry(
                CALIBRATION_FILENAME,
                calibration_sha,
                kind="calibration",
                revision=release_id,
            ),
            "calibration_diagnostics": _artifact_entry(
                "calibration_diagnostics.json",
                diagnostics_sha,
                kind="diagnostics",
                revision=release_id,
            ),
            "us_source_coverage": _artifact_entry(
                "us_source_coverage.json",
                coverage_sha,
                kind="diagnostics",
                revision=release_id,
            ),
            **(
                {
                    "reform_validation": _artifact_entry(
                        "reform_validation.json",
                        _sha256(release_dir / "reform_validation.json"),
                        kind="diagnostics",
                        revision=release_id,
                    )
                }
                if (release_dir / "reform_validation.json").exists()
                else {}
            ),
            **(
                {
                    "demographics": _artifact_entry(
                        "demographics.json",
                        _sha256(release_dir / "demographics.json"),
                        kind="diagnostics",
                        revision=release_id,
                    )
                }
                if (release_dir / "demographics.json").exists()
                else {}
            ),
        },
    }
    (release_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=1, allow_nan=False)
    )


def _load_zero_support_exclusions(path: Path | None) -> dict[str, str]:
    """Load a per-run, per-artifact zero-support exclusion mapping.

    The file is a JSON object of ``source_record_id -> reason``. Each reason
    must be a non-empty string documenting why the sparse artifact's frozen
    support cannot express that cell (PolicyEngine/populace#299 Build G). These
    augment the standing :data:`US_FISCAL_TARGET_SUPPORT_EXCLUSIONS` for a single
    build; the module constant is never mutated. Returns an empty mapping when
    no path is given.
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"Zero-support exclusions file {path} must be a JSON object of "
            "source_record_id -> reason."
        )
    exclusions: dict[str, str] = {}
    for source_record_id, reason in payload.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "Every zero-support exclusion needs a non-empty reason; "
                f"{source_record_id!r} in {path} has {reason!r}."
            )
        exclusions[str(source_record_id)] = reason
    return exclusions


def _reviewed_exclusions(active_aliases: Iterable[str]) -> dict[str, str]:
    active = set(active_aliases)
    hard = set(hard_target_package_aliases())
    unknown_active = sorted(active - hard)
    if unknown_active:
        raise RuntimeError(
            "Fiscal-refresh active aliases are not hard-target aliases: "
            + ", ".join(unknown_active)
        )
    excluded = hard - active
    reviewed = set(REVIEWED_EXCLUDED_ALIASES) - active
    if excluded != reviewed:
        missing = sorted(excluded - reviewed)
        extra = sorted(reviewed - excluded)
        raise RuntimeError(
            "Reviewed hard-target exclusion list is stale "
            f"(missing={missing}, extra={extra})."
        )
    return {
        alias: (
            "Reviewed fiscal-refresh exclusion: this release recalibrates the "
            "Issue #40 fiscal target surface only. This source family is not "
            "certified by this release and should be validated in a dedicated "
            "non-fiscal refresh before being treated as active calibration "
            "coverage."
        )
        for alias in REVIEWED_EXCLUDED_ALIASES
        if alias in reviewed
    }


def _fiscal_target_source_provenance(
    target_specs: Iterable[object],
) -> dict[str, dict[str, object]]:
    provenance: dict[str, dict[str, object]] = {}
    for spec in target_specs:
        entry = provenance.setdefault(
            spec.family,
            {
                "label": FISCAL_TARGET_SOURCE_KEYS.get(spec.family, spec.family),
                "target_count": 0,
                "sources": [],
                "reference_urls": [],
            },
        )
        entry["target_count"] = int(entry["target_count"]) + 1
        sources = entry["sources"]
        if isinstance(sources, list) and spec.source not in sources:
            sources.append(spec.source)
        url = spec.metadata.get("reference_url")
        urls = entry["reference_urls"]
        if isinstance(urls, list) and url and url not in urls:
            urls.append(url)
    return {
        family: {
            "label": payload["label"],
            "target_count": payload["target_count"],
            "sources": sorted(payload["sources"]),
            "reference_urls": sorted(payload["reference_urls"]),
        }
        for family, payload in sorted(provenance.items())
    }


def _assert_us_release_id(release_id: str) -> None:
    if not release_id.startswith("populace-us-"):
        raise ValueError(
            "US fiscal refresh release ids must start with 'populace-us-' so "
            "the US release contract requires source coverage diagnostics."
        )


def _staging_telemetry(
    args: argparse.Namespace,
    *,
    release_root: Path,
    release_id: str,
) -> StagingTelemetry | None:
    if args.no_staging:
        return None
    if not args.staging_dir and not args.staging_repo_id:
        return None
    run_id = args.staging_run_id or release_id
    run_dir = args.staging_dir or release_root / "staging" / "runs" / run_id
    return StagingTelemetry(
        run_id=run_id,
        candidate_release_id=release_id,
        run_dir=run_dir,
        repo_id=args.staging_repo_id,
        path_prefix=args.staging_prefix,
        upload_interval_seconds=args.staging_upload_interval_seconds,
    )


def main() -> None:
    args = _parse_args()
    if _git_dirty():
        raise SystemExit("Refusing to build a release from a dirty git worktree.")
    build_started = time.perf_counter()
    timing: dict[str, float] = {}

    base_h5 = args.base_h5 or _download_base_h5()
    base_dataset_sha256 = _sha256(base_h5)
    digest = base_dataset_sha256[:7]
    build_timestamp = datetime.now(UTC)
    full_commit = _git_output("rev-parse", "HEAD")
    commit = _git_output("rev-parse", "--short=12", "HEAD")
    release_id = (
        args.release_id
        or f"populace-us-2024-{digest}-{commit}-{build_timestamp:%Y%m%dT%H%M%SZ}"
    )
    _assert_us_release_id(release_id)
    congressional_district_vintage_crosswalk = (
        load_congressional_district_vintage_crosswalk(
            args.congressional_district_vintage_crosswalk
        )
        if args.congressional_district_vintage_crosswalk is not None
        else None
    )
    congressional_district_vintage_crosswalk_metadata = (
        {
            "path": str(args.congressional_district_vintage_crosswalk.resolve()),
            "sha256": _sha256(args.congressional_district_vintage_crosswalk),
        }
        if args.congressional_district_vintage_crosswalk is not None
        else None
    )
    _assert_cd_vintage_support_matches(
        base_h5, congressional_district_vintage_crosswalk_metadata
    )
    # Preflight (before the expensive calibration): every provision-critical
    # input leaf the reform-validation configs depend on must be produced by a
    # source stage or be an allowlisted known gap. This catches the
    # structurally-missing-input class (PolicyEngine/populace#252, #253) before a
    # validation row can ship a silent structural zero. The registry is first
    # checked against the live PolicyEngine-US graph so a stale entry cannot mask
    # a real gap.
    assert_validation_leaf_registry_current()
    validation_input_coverage_gate = us_validation_input_coverage_gate()
    if not validation_input_coverage_gate.passed:
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Validation input coverage failed: {failure}"
                for failure in validation_input_coverage_gate.failures
            )
        )
    # Preflight (populace#368): the release input-column coverage manifest must
    # still equal the pinned reference eCPS input surface, keep the SSI
    # countable-resource assets as hard requirements, and declare only live
    # PolicyEngine-US input leaves. A drifted manifest fails here before the
    # expensive calibration so a stale contract cannot silently narrow coverage.
    assert_release_input_coverage_manifest_current()
    # Preflight (populace#377): no column may be required-to-signal by one
    # register (seeded/count-calibrated take-up, health nonconstant, coverage
    # 'required') and excused as absent/degenerate by another (degenerate
    # reviewed exclusions, coverage reviewed exclusions, parity known gaps,
    # documented-absent). Such a pincer aborts every build AFTER the expensive
    # source stages — the stale TANF exclusion did exactly that on Build G and
    # Build I. Fail it here, in seconds.
    register_consistency_gate = us_register_consistency_gate(
        degenerate_reviewed_exclusions=US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS,
        documented_absent_inputs=US_DOCUMENTED_ABSENT_INPUTS,
        nonconstant_required_columns=US_HEALTH_INPUT_NONCONSTANT_COLUMNS,
    )
    if not register_consistency_gate.passed:
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Register consistency failed: {failure}"
                for failure in register_consistency_gate.failures
            )
        )
    # Preflight (populace#381): the checked-in take-up contract must still match
    # the installed policyengine-us (entity/default/engine_class of every
    # ``takes_up_*`` flag) and its curated treatments must not contradict the
    # engine class. These are cheap engine-metadata checks; running them here,
    # before the expensive source stages, means an engine bump that changes a
    # take-up default or flips a seeded flag to a formula fails in seconds
    # instead of shipping a mechanical universal-take-up landmine. The seeding
    # stages below read this same contract, so a stale contract must abort the
    # build, not merely a test.
    assert_take_up_contract_current()
    assert_take_up_treatments_consistent()
    ledger_artifact = load_ledger_consumer_artifact(
        args.ledger_facts,
        expected_facts_sha256=args.ledger_facts_sha256,
        expected_manifest_sha256=args.ledger_manifest_sha256,
    )
    extra_support_exclusions = _load_zero_support_exclusions(
        args.zero_support_exclusions
    )
    target_registry = compile_us_fiscal_target_registry(
        ledger_artifact.facts,
        target_period=PERIOD,
        include_congressional_district_targets=(
            args.include_congressional_district_targets
        ),
        congressional_district_vintage_crosswalk=(
            congressional_district_vintage_crosswalk
        ),
        age_targets=args.age_targets,
        allow_unaged_dollar_targets=args.allow_unaged_dollar_targets,
        extra_support_exclusions=extra_support_exclusions,
    )
    # Reviewed CMS Medicaid enrollment substitutions (populace#386): a state
    # whose point-in-time snapshot is unreported at source ships its cited
    # nearest-prior-month count instead of failing the take-up gate closed.
    # The records ride the take-up diagnostics; the gate fails a stale entry
    # (CMS backfilled the substituted-for month) so the register cannot rot.
    # Applied once here, before the dense/sparse split, so the injected spec
    # flows through `target_specs` into the materialized calibration registry
    # that BOTH the dense (`calibrate`) and sparse (`calibrate_l0_refit`) arms
    # consume, and into the take-up target table and build manifest.
    target_registry, medicaid_enrollment_substitutions = (
        apply_us_medicaid_enrollment_substitutions(target_registry)
    )
    target_specs = target_registry.specs
    if args.diagnostic_skip_tax_expenditure_targets:
        tax_expenditure_measures = {
            reform_spec.measure for reform_spec in US_JCT_TAX_EXPENDITURE_REFORMS
        }
        target_specs = tuple(
            spec
            for spec in target_specs
            if spec.measure not in tax_expenditure_measures
        )
    active_target_registry = TargetRegistry(target_specs, country="us")
    target_profile_gate = target_profile_coverage_gate(
        target_specs,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    if (
        not target_profile_gate.passed
        and not args.diagnostic_skip_tax_expenditure_targets
    ):
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Target profile coverage failed: {failure}"
                for failure in target_profile_gate.failures
            )
        )
    release_root = args.out.resolve()
    artifact_root = release_root / "artifacts"
    release_dir = release_root / "releases" / release_id
    checkpoint_root, target_materialization_cache_dir, target_frame_checkpoint_path = (
        _resolve_checkpoint_paths(args, artifact_root=artifact_root)
    )
    if checkpoint_root is not None:
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    release_dir.mkdir(parents=True, exist_ok=True)
    telemetry = _staging_telemetry(
        args,
        release_root=release_root,
        release_id=release_id,
    )
    if telemetry is not None:
        telemetry.stage(
            "target_registry",
            message="Compiled fiscal target registry.",
            n_targets=len(target_specs),
            target_profile_gate_passed=target_profile_gate.passed,
            force_upload=True,
        )

    if telemetry is not None:
        telemetry.stage("load_base_frame", message="Loading base population H5.")
    base_frame = _load_frame(base_h5)

    # Frozen-support recovery (populace#328): if a selection source is supplied,
    # reduce the base pool to exactly that support by stable source identity,
    # BEFORE the population mass repair — matching the certified sequence, where
    # the 340.1M-person frozen support is rescaled to the 334.2M Census benchmark
    # (not the full pool). Reducing here also materializes PolicyEngine over the
    # ~57k support instead of the ~338k pool.
    selection_source, _selection_join_key = _resolve_selection_source(args)
    selection_source_payload: dict | None = None
    if selection_source is not None:
        if args.selection_mode != "frozen_support":
            raise ValueError(
                f"--selection-mode {args.selection_mode!r} is not yet wired into "
                "calibration; only 'frozen_support' is implemented (populace#328)."
            )
        if telemetry is not None:
            telemetry.stage(
                "frozen_support_selection",
                message="Recovering frozen support by stable source identity.",
                n_source=selection_source.n_identities,
                join_key=list(selection_source.join_key),
            )
        base_frame, selection_report = select_frozen_support(
            base_frame, selection_source
        )
        selection_source_payload = selection_report.as_manifest()
        if telemetry is not None:
            telemetry.stage(
                "frozen_support_selection_done",
                message="Reduced base pool to the frozen support.",
                n_selected=selection_report.n_selected,
                n_base_candidates=selection_report.n_base_candidates,
                n_unmapped=selection_report.n_unmapped,
            )

    base_frame, base_population_repair = _with_base_population_mass_repair(base_frame)
    base_frame, social_security_component_repair = (
        _with_social_security_component_value_repair(base_frame, target_specs)
    )
    if telemetry is not None:
        telemetry.stage(
            "base_population_repair",
            message="Repaired base population mass for conserved calibration.",
            applied=base_population_repair.get("applied"),
            factor=base_population_repair.get("factor"),
            initial_population=base_population_repair.get("initial_population"),
            repaired_population=base_population_repair.get("repaired_population"),
        )
        telemetry.stage(
            "social_security_component_repair",
            message="Repaired Social Security component value support.",
            applied=social_security_component_repair.get("applied"),
            components=social_security_component_repair.get("components"),
        )
    base_population_gate = _base_population_scale_gate(
        base_frame,
        mass_repair=base_population_repair,
    )
    if not base_population_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "base_population_gate",
                status="failed",
                message="Base population scale gate failed.",
                failures=list(base_population_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Base population scale failed: {failure}"
                for failure in base_population_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "retirement_contribution_inputs",
            message=("Verifying ASEC-sourced desired retirement-contribution inputs."),
        )
    base_frame = with_us_retirement_contribution_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    retirement_contributions_gate = us_retirement_contributions_signal_gate(base_frame)
    if not retirement_contributions_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "retirement_contribution_inputs_gate",
                status="failed",
                message="Retirement-contribution signal gate failed.",
                failures=list(retirement_contributions_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Retirement-contribution signal failed: " + failure
                for failure in retirement_contributions_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "immigration_inputs",
            message="Deriving SSN card type and immigration status inputs.",
        )
    base_frame = with_us_immigration_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    immigration_gate = us_immigration_composition_gate(base_frame)
    if not immigration_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "immigration_gate",
                status="failed",
                message="Immigration composition gate failed.",
                failures=list(immigration_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Immigration composition failed: {failure}"
                for failure in immigration_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "take_up_inputs",
            message="Seeding TANF and EITC take-up from administrative rates.",
        )
    base_frame = with_us_take_up_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    take_up_gate = us_take_up_signal_gate(base_frame)
    if not take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "take_up_gate",
                status="failed",
                message="Take-up signal gate failed.",
                failures=list(take_up_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Take-up signal failed: {failure}" for failure in take_up_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "hours_worked_inputs",
            message="Deriving hours-worked inputs from ASEC reported hours.",
        )
    base_frame = with_us_hours_worked_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    hours_worked_gate = us_hours_worked_signal_gate(base_frame)
    if not hours_worked_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "hours_worked_gate",
                status="failed",
                message="Hours-worked signal gate failed.",
                failures=list(hours_worked_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Hours-worked signal failed: {failure}"
                for failure in hours_worked_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "snap_take_up_inputs",
            message="Assigning SNAP take-up from reported receipt.",
        )
    base_frame = with_us_snap_take_up_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    snap_take_up_gate = us_snap_take_up_signal_gate(base_frame)
    if not snap_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "snap_take_up_gate",
                status="failed",
                message="SNAP take-up signal gate failed.",
                failures=list(snap_take_up_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SNAP take-up signal failed: {failure}"
                for failure in snap_take_up_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "eligibility_inputs",
            message="Deriving SNAP eligibility and exemption inputs from ASEC.",
        )
    base_frame = with_us_eligibility_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    eligibility_inputs_gate = us_eligibility_inputs_signal_gate(base_frame)
    if not eligibility_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "eligibility_inputs_gate",
                status="failed",
                message="Eligibility-inputs signal gate failed.",
                failures=list(eligibility_inputs_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Eligibility-inputs signal failed: {failure}"
                for failure in eligibility_inputs_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "education_inputs",
            message=(
                "Carrying ASEC educational assistance and deriving AOTC "
                "factual inputs from PUF qualified tuition."
            ),
        )
    base_frame = with_us_education_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    education_inputs_gate = us_education_inputs_signal_gate(base_frame)
    if not education_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "education_inputs_gate",
                status="failed",
                message="Education-input signal gate failed.",
                failures=list(education_inputs_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Education-input signal failed: {failure}"
                for failure in education_inputs_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "pregnancy_inputs",
            message="Seeding pregnancy among women 15-44 at the national rate.",
        )
    base_frame = with_us_pregnancy_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    pregnancy_gate = us_pregnancy_signal_gate(base_frame)
    if not pregnancy_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "pregnancy_gate",
                status="failed",
                message="Pregnancy signal gate failed.",
                failures=list(pregnancy_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Pregnancy signal failed: {failure}"
                for failure in pregnancy_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "snap_discretionary_exemption_inputs",
            message="Seeding ABAWD discretionary exemptions at the statutory cap.",
        )
    base_frame = with_us_snap_discretionary_exemption_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
    )
    snap_discretionary_exemption_gate = us_snap_discretionary_exemption_signal_gate(
        base_frame
    )
    if not snap_discretionary_exemption_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "snap_discretionary_exemption_gate",
                status="failed",
                message="SNAP discretionary-exemption signal gate failed.",
                failures=list(snap_discretionary_exemption_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SNAP discretionary-exemption signal failed: {failure}"
                for failure in snap_discretionary_exemption_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "scf_wealth_inputs",
            message=(
                "Imputing SSI countable-resource assets (bank/stock/bond) from "
                "the Federal Reserve SCF 2022 summary extract."
            ),
        )
    # populace#356/#368 Deliverable 2: restore the three SSI countable-resource
    # asset inputs (bank_account_assets / stock_assets / bond_assets). Without
    # them ssi_countable_resources is 0 for every record and every SSI
    # resource-limit reform silently scores $0 — the failure the #368 column and
    # reform-coverage gates are RED on. A CLI-supplied extract path is used when
    # given; otherwise the fixed-vintage public extract is fetched and cached.
    scf_summary_extract_path = (
        Path(args.scf_summary_extract)
        if args.scf_summary_extract is not None
        else fetch_scf_2022_summary_extract()
    )
    scf_wealth_donor = load_scf_2022_financial_asset_donor(scf_summary_extract_path)
    base_frame = with_us_scf_wealth_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        scf_donor=scf_wealth_donor,
    )
    scf_wealth_gate = us_scf_wealth_signal_gate(base_frame)
    if not scf_wealth_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "scf_wealth_gate",
                status="failed",
                message="SCF-wealth signal gate failed.",
                failures=list(scf_wealth_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SCF-wealth signal failed: {failure}"
                for failure in scf_wealth_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "scf_auto_loan_inputs",
            message=(
                "Imputing household auto-loan balance and interest from the "
                "full Federal Reserve SCF 2022 extract, then deriving the "
                "OBBBA qualifying-interest proxy."
            ),
        )
    scf_full_extract_path = (
        Path(args.scf_full_extract)
        if args.scf_full_extract is not None
        else fetch_scf_2022_full_extract()
    )
    scf_auto_loan_donor = load_scf_2022_auto_loan_donor(
        scf_summary_extract_path,
        scf_full_extract_path,
    )
    base_frame = with_us_scf_auto_loan_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        scf_auto_loan_donor=scf_auto_loan_donor,
    )
    scf_auto_loan_gate = us_scf_auto_loans_signal_gate(base_frame)
    if not scf_auto_loan_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "scf_auto_loan_gate",
                status="failed",
                message="SCF auto-loan signal gate failed.",
                failures=list(scf_auto_loan_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SCF auto-loan signal failed: {failure}"
                for failure in scf_auto_loan_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "sipp_tip_inputs",
            message=(
                "Imputing tip income from the sha-pinned SIPP donor and "
                "carrying Treasury tipped-occupation codes from ASEC."
            ),
        )
    sipp_tip_donor_path = (
        Path(args.sipp_tip_donor)
        if args.sipp_tip_donor is not None
        else fetch_sipp_2023_tip_donor()
    )
    sipp_tip_donor = load_sipp_2023_tip_donor(
        sipp_tip_donor_path,
        expected_sha256=SIPP_2023_TIP_DONOR_SHA256,
    )
    base_frame = with_us_sipp_tip_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        sipp_donor=sipp_tip_donor,
    )
    sipp_tips_gate = us_sipp_tips_signal_gate(base_frame)
    if not sipp_tips_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "sipp_tips_gate",
                status="failed",
                message="SIPP-tip signal gate failed.",
                failures=list(sipp_tips_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SIPP-tip signal failed: {failure}"
                for failure in sipp_tips_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "org_wages_inputs",
            message=(
                "Imputing CPS ORG hourly-pay inputs, carrying ASEC occupation "
                "groups, assigning BLS union coverage, and deriving the FLSA "
                "overtime premium."
            ),
        )
    org_wages_donor_path = (
        Path(args.org_wages_donor)
        if args.org_wages_donor is not None
        else fetch_org_2024_donor()
    )
    org_wages_donor = load_org_2024_donor(
        org_wages_donor_path,
        expected_content_sha256=ORG_2024_DONOR_CONTENT_SHA256,
    )
    base_frame = with_us_org_wages_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        org_donor=org_wages_donor,
    )
    org_wages_gate = us_org_wages_signal_gate(base_frame)
    if not org_wages_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "org_wages_gate",
                status="failed",
                message="CPS ORG/FLSA signal gate failed.",
                failures=list(org_wages_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"CPS ORG/FLSA signal failed: {failure}"
                for failure in org_wages_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "source_inputs",
            message="Materializing ACA marketplace source outputs.",
        )
    base_frame = _with_aca_marketplace_source_outputs(
        base_frame,
        target_specs,
        seed=args.seed,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
    )
    health_input_gate = _health_input_signal_gate(base_frame)
    if not health_input_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "health_input_gate",
                status="failed",
                message="Health input signal gate failed.",
                failures=list(health_input_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Health input signal failed: {failure}"
                for failure in health_input_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "medicaid_take_up",
            message=(
                "Assigning Medicaid take-up from reported coverage and CMS "
                "state enrollment snapshots."
            ),
        )
    base_frame, medicaid_take_up_diagnostics = _with_medicaid_take_up_outputs(
        base_frame,
        target_specs,
        seed=args.seed,
        substitutions=medicaid_enrollment_substitutions,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
    )
    medicaid_take_up_gate = us_medicaid_take_up_gate(medicaid_take_up_diagnostics)
    if not medicaid_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "medicaid_take_up_gate",
                status="failed",
                message="Medicaid take-up gate failed.",
                failures=list(medicaid_take_up_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Medicaid take-up failed: {failure}"
                for failure in medicaid_take_up_gate.failures
            )
        )
    if telemetry is not None and args.input_mass_reference_h5 is not None:
        telemetry.stage(
            "input_mass_reference_gate",
            message="Gating base-frame input mass against the reference release.",
        )
    input_mass_reference_gate = _input_mass_reference_gate(
        base_frame,
        reference_h5=args.input_mass_reference_h5,
        relative_tolerance=args.input_mass_relative_tolerance,
        minimum_reference_total=args.input_mass_minimum_reference_total,
    )
    if (
        input_mass_reference_gate is not None
        and not input_mass_reference_gate.passed
        and not args.allow_input_mass_drift
    ):
        if telemetry is not None:
            telemetry.stage(
                "input_mass_reference_gate",
                status="failed",
                message="Base-frame input mass parity gate failed.",
                failures=list(input_mass_reference_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Input mass parity failed: {failure}"
                for failure in input_mass_reference_gate.failures
            )
        )
    degenerate_input_gate = _degenerate_input_signal_gate(
        base_frame, PolicyEngineUSEngine()
    )
    if not degenerate_input_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "degenerate_input_gate",
                status="failed",
                message="Degenerate input signal gate failed.",
                failures=list(degenerate_input_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Degenerate input signal failed: {failure}"
                for failure in degenerate_input_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "ecps_parity_gate",
            message="Gating candidate layers against the incumbent eCPS reference.",
        )
    ecps_parity_gate = _ecps_parity_gate(base_frame)
    if not ecps_parity_gate.passed and not args.allow_ecps_parity_gaps:
        if telemetry is not None:
            telemetry.stage(
                "ecps_parity_gate",
                status="failed",
                message="eCPS parity gate failed.",
                failures=list(ecps_parity_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"eCPS parity failed: {failure}"
                for failure in ecps_parity_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage("target_compilation", message="Materializing target frame.")
    target_compilation_started = time.perf_counter()
    policyengine_us_version = _package_or_workspace_version("policyengine-us")
    target_frame_checkpoint_identity = _target_frame_checkpoint_identity(
        base_dataset_sha256=base_dataset_sha256,
        policyengine_us_version=policyengine_us_version,
        seed=args.seed,
        target_period=PERIOD,
        target_registry_version=active_target_registry.version,
        congressional_district_vintage_crosswalk_sha256=(
            congressional_district_vintage_crosswalk_metadata or {}
        ).get("sha256"),
    )
    target_frame, registry, compilation = _load_or_materialize_target_frame(
        base_frame,
        target_specs,
        target_frame_checkpoint_path=target_frame_checkpoint_path,
        target_frame_checkpoint_identity=target_frame_checkpoint_identity,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
        target_materialization_cache_dir=target_materialization_cache_dir,
        target_materialization_cache_context={
            "base_dataset_sha256": base_dataset_sha256,
            "build_commit": full_commit,
            "policyengine_us_version": policyengine_us_version,
            "seed": args.seed,
            "target_period": PERIOD,
            "target_registry_version": active_target_registry.version,
            "congressional_district_vintage_crosswalk_sha256": (
                congressional_district_vintage_crosswalk_metadata or {}
            ).get("sha256"),
        },
        gate_congressional_district_targets=args.gate_congressional_district_targets,
    )
    if congressional_district_vintage_crosswalk_metadata is not None:
        compilation = {
            **dict(compilation),
            "congressional_district_vintage_crosswalk": (
                congressional_district_vintage_crosswalk_metadata
            ),
        }
    timing["target_compilation_seconds"] = (
        time.perf_counter() - target_compilation_started
    )
    warm_start_weights: np.ndarray | None = None
    warm_start_calibration: Mapping[str, object] | None = None
    if args.warm_start_calibration_npz is not None:
        warm_start_weights, warm_start_calibration = _load_warm_start_calibration_npz(
            args.warm_start_calibration_npz,
            expected_initial_weights=target_frame.resolve_weights("household").values,
        )
    candidate_households = int(target_frame.n("household"))
    l0_refit_lambda = (
        None
        if args.dense_default_dataset
        else args.l0_refit_lambda_share / float(candidate_households)
    )
    target_loss_weights = _fiscal_target_loss_weights(registry)
    if telemetry is not None:
        telemetry.stage(
            "calibrating",
            message=(
                "Calibrating dense household weights."
                if args.dense_default_dataset
                else "Selecting sparse L0 support and refitting household weights."
            ),
            default_dataset_method=(
                "dense_no_l0" if args.dense_default_dataset else "l0_refit"
            ),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_weight_ratio=args.max_weight_ratio,
            n_targets=len(registry),
            n_candidate_households=candidate_households,
            l0_refit_lambda_share=(
                None
                if args.dense_default_dataset
                else float(args.l0_refit_lambda_share)
            ),
            l0_lambda=l0_refit_lambda,
            l2_lambda=float(args.l2_lambda),
            refit_l2_lambda=(
                None
                if args.dense_default_dataset
                else float(
                    args.l2_lambda
                    if args.refit_l2_lambda is None
                    else args.refit_l2_lambda
                )
            ),
            warm_start_calibration=(
                dict(warm_start_calibration)
                if warm_start_calibration is not None
                else {"enabled": False}
            ),
            target_compilation_seconds=timing["target_compilation_seconds"],
        )
    calibration_started = time.perf_counter()
    if args.dense_default_dataset:
        result = calibrate(
            target_frame,
            registry.to_target_set(),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_weight_ratio=args.max_weight_ratio,
            seed=args.seed,
            mass="conserve",
            l2_lambda=args.l2_lambda,
            target_loss_weights=target_loss_weights,
            target_loss_cap=US_FISCAL_TARGET_LOSS_CAP,
            warm_start_weights=warm_start_weights,
            progress_callback=(
                telemetry.calibration_progress if telemetry is not None else None
            ),
        )
        default_dataset = {
            "method": "dense_no_l0",
            "sparse": False,
            "n_candidate_households": candidate_households,
            "n_exported_households": int(target_frame.n("household")),
            "epochs": int(args.epochs),
            "l2_lambda": float(args.l2_lambda),
            "final_loss": float(result.final_loss),
        }
    else:
        result = calibrate_l0_refit(
            target_frame,
            registry.to_target_set(),
            epochs=args.epochs,
            refit_epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_weight_ratio=args.max_weight_ratio,
            seed=args.seed,
            mass="conserve",
            l0_lambda=float(l0_refit_lambda),
            l2_lambda=args.l2_lambda,
            refit_l2_lambda=args.refit_l2_lambda,
            target_loss_weights=target_loss_weights,
            target_loss_cap=US_FISCAL_TARGET_LOSS_CAP,
            warm_start_weights=warm_start_weights,
            progress_callback=(
                telemetry.calibration_progress if telemetry is not None else None
            ),
        )
        default_dataset = {
            "method": "l0_refit",
            "sparse": True,
            "n_candidate_households": candidate_households,
            "n_selected_households": int(result.selection.n_nonzero),
            "n_exported_households": int(result.frame.n("household")),
            "l0_lambda_share": float(args.l0_refit_lambda_share),
            "l0_lambda": float(result.l0_lambda),
            "selection_epochs": int(args.epochs),
            "refit_epochs": int(args.epochs),
            "selection_l2_lambda": float(args.l2_lambda),
            "refit_l2_lambda": float(
                args.l2_lambda if args.refit_l2_lambda is None else args.refit_l2_lambda
            ),
            "selection_final_loss": float(result.selection.final_loss),
            "refit_initial_loss": float(result.initial_loss),
            "refit_final_loss": float(result.final_loss),
        }
    timing["calibration_seconds"] = time.perf_counter() - calibration_started
    timing["elapsed_through_calibration_seconds"] = time.perf_counter() - build_started
    if telemetry is not None:
        telemetry.stage(
            "release_gates",
            message="Evaluating release gates.",
            final_loss=result.final_loss,
            n_nonzero=result.n_nonzero,
            default_dataset=default_dataset,
            calibration_seconds=timing["calibration_seconds"],
            elapsed_through_calibration_seconds=timing[
                "elapsed_through_calibration_seconds"
            ],
        )
    incumbent_payload = _load_incumbent_diagnostics_payload(args.incumbent_diagnostics)
    if args.incumbent_diagnostics is not None:
        current_target_surface = diagnostics_payload(
            result,
            target_registry=registry,
        )["target_surface"]
        _assert_incumbent_target_surface_matches(
            current_target_surface,
            incumbent_payload,
            path=args.incumbent_diagnostics,
        )
    incumbent_diagnostics = (
        _diagnostics_by_target_name(
            incumbent_payload,
            path=args.incumbent_diagnostics,
        )
        if args.incumbent_diagnostics is not None
        else {}
    )
    enforced_input_mass_reference_gate = (
        None if args.allow_input_mass_drift else input_mass_reference_gate
    )
    enforced_ecps_parity_gate = (
        None if args.allow_ecps_parity_gaps else ecps_parity_gate
    )
    gate_failures = _release_gate_failures(
        result,
        compilation,
        target_profile_gate,
        health_input_gate,
        base_population_gate,
        incumbent_diagnostics,
        immigration_gate,
        enforced_input_mass_reference_gate,
        degenerate_input_gate,
        ecps_parity_gate=enforced_ecps_parity_gate,
        hours_worked_gate=hours_worked_gate,
        snap_take_up_gate=snap_take_up_gate,
        eligibility_inputs_gate=eligibility_inputs_gate,
        pregnancy_gate=pregnancy_gate,
        snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
    )
    _write_release_calibration_diagnostics(
        result=result,
        release_dir=release_dir,
        registry=registry,
        base_h5=base_h5,
        compilation=compilation,
        target_profile_gate=target_profile_gate,
        health_input_gate=health_input_gate,
        base_population_gate=base_population_gate,
        immigration_gate=immigration_gate,
        input_mass_reference_gate=input_mass_reference_gate,
        hours_worked_gate=hours_worked_gate,
        snap_take_up_gate=snap_take_up_gate,
        eligibility_inputs_gate=eligibility_inputs_gate,
        pregnancy_gate=pregnancy_gate,
        snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
        support_value_repairs={
            "social_security_components": social_security_component_repair
        },
        warm_start_calibration=warm_start_calibration,
        selection_source=selection_source_payload,
        audit_export_targets=bool(args.audit_export_targets),
        gate_failures=gate_failures,
        timing=timing,
        incumbent_diagnostics_path=args.incumbent_diagnostics,
        incumbent_diagnostics=incumbent_diagnostics,
        default_dataset=default_dataset,
        degenerate_input_gate=degenerate_input_gate,
        ecps_parity_gate=ecps_parity_gate,
        validation_input_coverage_gate=validation_input_coverage_gate,
    )
    if telemetry is not None:
        telemetry.attach_artifact(
            "calibration_diagnostics",
            release_dir / "calibration_diagnostics.json",
        )
    if gate_failures:
        if telemetry is not None:
            telemetry.stage(
                "release_gates",
                status="failed",
                message="Release gates failed.",
                failures=gate_failures,
                force_upload=True,
            )
        raise RuntimeError("Release gates failed: " + "; ".join(gate_failures))

    if telemetry is not None:
        telemetry.stage("export_dataset", message="Writing PolicyEngine-US H5.")
    export_frame = (
        _with_calibrated_weights(base_frame, result.weights)
        if args.dense_default_dataset
        else _with_l0_refit_weights(base_frame, result)
    )
    release_engine = PolicyEngineUSEngine()
    # populace#368: full eCPS input-column coverage as a HARD release gate.
    # Every input column the reference eCPS exports must be persisted by the
    # export as a key with non-default signal, or carry a reviewed exclusion.
    # This generalizes assert_required_us_release_source_columns (5 SNAP/ACA/
    # immigration columns) to the whole eCPS input surface and closes the hole
    # export-mass parity leaves open: an input the base never imputed (the SSI
    # countable-resource assets, tips, education credits, ...) is absent from the
    # export entirely, so every reform binding through it silently scores ~$0
    # while mass and parity gates still pass. Runs on the calibrated export for
    # BOTH the dense and sparse default paths (both reach this line). The SSI
    # asset inputs ship as hard requirements with no exclusion, so this gate is
    # RED on today's asset-less artifacts by design (Deliverable 2 turns it
    # green); that is the gate doing its job, not a bug.
    input_coverage_gate = us_release_input_coverage_gate(export_frame, release_engine)
    input_coverage_path = release_dir / "input_coverage.json"
    input_coverage_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enforced": not args.allow_input_coverage_gaps,
                "input_coverage": {
                    "passed": input_coverage_gate.passed,
                    "failures": list(input_coverage_gate.failures),
                    "details": dict(input_coverage_gate.details),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if telemetry is not None:
        telemetry.attach_artifact("input_coverage", input_coverage_path)
    if not input_coverage_gate.passed and not args.allow_input_coverage_gaps:
        if telemetry is not None:
            telemetry.stage(
                "export_dataset",
                status="failed",
                message="Release input-column coverage gate failed.",
                failures=list(input_coverage_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Input coverage failed: {failure}"
                for failure in input_coverage_gate.failures
            )
        )
    # #327: the export gate compares the calibrated export against a reference.
    # By default that reference is the raw pre-calibration base — but for a dense
    # parent built from a raw pooled-ASEC base, calibration correctly scales
    # PUF-imputed income up toward SOI/CBO targets, and the raw-base yardstick
    # flags those correct gains. When --export-input-mass-reference-h5 is given
    # (the live default, per #327's reference decision), compare against it so
    # calibration-driven upward alignment is in-band while a genuine #278 zeroing
    # still fails. This is a DISTINCT flag from --input-mass-reference-h5: the
    # base-vs-reference gate compares the *pre-calibration* base and would
    # over-fire against a calibrated reference on the same PUF columns.
    export_reference_frame = (
        load_us_frame(args.export_input_mass_reference_h5)
        if args.export_input_mass_reference_h5 is not None
        else None
    )
    export_input_mass_gate = _export_input_mass_gate(
        export_frame,
        base_frame,
        relative_tolerance=args.input_mass_relative_tolerance,
        minimum_reference_total=args.input_mass_minimum_reference_total,
        reference_frame=export_reference_frame,
        reference_name=(
            args.export_input_mass_reference_h5.name
            if args.export_input_mass_reference_h5 is not None
            else "base_frame"
        ),
        # Build H (populace#299): the two SOI-identified columns whose true
        # target level provably cannot sit inside the live-default reference
        # band (estate_income, non_sch_d_capital_gains). miscellaneous_income
        # and both mortgage columns are deliberately NOT excluded so the run
        # adjudicates their actual gate outcome. See the constant's band-math
        # rationale above.
        reviewed_exclusions=US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
    )
    input_mass_parity_path = release_dir / "input_mass_parity.json"
    input_mass_parity_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enforced": not args.allow_input_mass_drift,
                "base_frame_vs_reference": (
                    {
                        "passed": input_mass_reference_gate.passed,
                        "failures": list(input_mass_reference_gate.failures),
                        "details": dict(input_mass_reference_gate.details),
                    }
                    if input_mass_reference_gate is not None
                    else None
                ),
                "export_vs_base_frame": {
                    "passed": export_input_mass_gate.passed,
                    "failures": list(export_input_mass_gate.failures),
                    "details": dict(export_input_mass_gate.details),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if telemetry is not None:
        telemetry.attach_artifact("input_mass_parity", input_mass_parity_path)
    if not export_input_mass_gate.passed and not args.allow_input_mass_drift:
        if telemetry is not None:
            telemetry.stage(
                "export_dataset",
                status="failed",
                message="Export input mass parity gate failed.",
                failures=list(export_input_mass_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Input mass parity failed: {failure}"
                for failure in export_input_mass_gate.failures
            )
        )
    dataset_path = artifact_root / DATASET_FILENAME
    release_engine.write_dataset(export_frame, dataset_path, period=PERIOD)
    # populace#368: reform-coverage smoke on the WRITTEN release H5. The column
    # gate above proves the required keys exist and carry signal; this is the
    # end-to-end backstop: each pinned probe (first: SSI asset limits at
    # $10k/$20k) mechanically binds through named input leaves and must move
    # its budget measure. A ~$0 score on a bound reform means those leaves are
    # absent or degenerate — the release aborts before any manifest or
    # contract file certifies the artifact. Runs on both dense and sparse
    # default paths (both reach this line). Until the asset stage is restored
    # (Deliverable 2) the SSI probe fails by design; that is the gate working.
    if not args.skip_reform_coverage_smoke:
        if telemetry is not None:
            telemetry.stage(
                "reform_coverage_smoke",
                message="Scoring pinned bound-reform probes on the written H5.",
            )
        reform_coverage_smoke_gate = us_reform_coverage_smoke_gate(
            simulate=default_simulate_factory(dataset_path),
            period=PERIOD,
        )
        reform_coverage_smoke_path = release_dir / "reform_coverage_smoke.json"
        reform_coverage_smoke_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "enforced": not args.allow_reform_coverage_smoke_failures,
                    "reform_coverage_smoke": {
                        "passed": reform_coverage_smoke_gate.passed,
                        "failures": list(reform_coverage_smoke_gate.failures),
                        "details": dict(reform_coverage_smoke_gate.details),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        if telemetry is not None:
            telemetry.attach_artifact(
                "reform_coverage_smoke", reform_coverage_smoke_path
            )
        if (
            not reform_coverage_smoke_gate.passed
            and not args.allow_reform_coverage_smoke_failures
        ):
            if telemetry is not None:
                telemetry.stage(
                    "reform_coverage_smoke",
                    status="failed",
                    message="Reform-coverage smoke gate failed.",
                    failures=list(reform_coverage_smoke_gate.failures),
                    force_upload=True,
                )
            raise RuntimeError(
                "Release gates failed: "
                + "; ".join(
                    f"Reform coverage smoke failed: {failure}"
                    for failure in reform_coverage_smoke_gate.failures
                )
            )
    if args.audit_export_targets:
        if telemetry is not None:
            telemetry.stage(
                "post_export_audit",
                message="Auditing exported H5 against calibration targets.",
            )
        _assert_export_matches_calibration(
            dataset_path,
            result,
            target_specs,
            maximum_microsim_batch_size=args.maximum_microsim_batch_size,
        )

    if telemetry is not None:
        telemetry.stage("write_calibration_npz", message="Writing calibration NPZ.")
    calibration_path = artifact_root / CALIBRATION_FILENAME
    _write_npz(calibration_path, result=result, registry=registry)

    if not args.skip_reform_validation:
        if telemetry is not None:
            telemetry.stage(
                "reform_validation",
                message="Writing reform validation diagnostics.",
            )
        _write_reform_validation(
            release_dir=release_dir,
            dataset_path=dataset_path,
            result=result,
            registry=registry,
            release_id=release_id,
            simulate_out_of_sample=not args.skip_out_of_sample_reforms,
        )
        if telemetry is not None:
            telemetry.attach_artifact(
                "reform_validation",
                release_dir / "reform_validation.json",
            )

    if not args.skip_demographics:
        if telemetry is not None:
            telemetry.stage("demographics", message="Writing demographics diagnostics.")
        _write_demographics(
            release_dir=release_dir,
            dataset_path=dataset_path,
            release_id=release_id,
        )
        if telemetry is not None:
            telemetry.attach_artifact("demographics", release_dir / "demographics.json")

    if telemetry is not None:
        telemetry.stage(
            "source_coverage", message="Writing source coverage diagnostics."
        )
    active_aliases = DIRECT_ACTIVE_ALIASES + (
        (
            "census-acs-s0101-congressional-district-age-2024",
            "soi-congressional-district-2022",
        )
        if args.include_congressional_district_targets
        else ()
    )
    coverage = us_source_coverage_diagnostics(
        active_target_aliases=active_aliases,
        reviewed_exclusions=_reviewed_exclusions(active_aliases),
    )
    coverage["fiscal_target_sources"] = _fiscal_target_source_provenance(target_specs)
    if congressional_district_vintage_crosswalk_metadata is not None:
        coverage["congressional_district_vintage_crosswalk"] = (
            congressional_district_vintage_crosswalk_metadata
        )
    coverage["fiscal_target_support_exclusions"] = [
        {"source_record_id": source_record_id, "reason": reason}
        for source_record_id, reason in sorted(
            US_FISCAL_TARGET_SUPPORT_EXCLUSIONS.items()
        )
    ]
    # Per-run, per-artifact support-expressibility exclusions (populace#299
    # Build G): recorded separately from the standing global registry so the
    # manifest documents exactly which cells this artifact declared un-
    # expressible on its support, without mutating the module constant.
    coverage["fiscal_target_support_exclusions_per_run"] = {
        "source": (
            str(args.zero_support_exclusions)
            if args.zero_support_exclusions is not None
            else None
        ),
        "reason": (
            "Sparse frozen-support cells the artifact's support cannot express; "
            "augments US_FISCAL_TARGET_SUPPORT_EXCLUSIONS for this build only."
        ),
        "exclusions": [
            {"source_record_id": source_record_id, "reason": reason}
            for source_record_id, reason in sorted(extra_support_exclusions.items())
        ],
    }
    write_us_source_coverage_diagnostics(
        coverage, release_dir / "us_source_coverage.json"
    )
    if telemetry is not None:
        telemetry.attach_artifact(
            "us_source_coverage",
            release_dir / "us_source_coverage.json",
        )
        telemetry.stage(
            "take_up_participation",
            message="Writing take-up participation diagnostics.",
        )
    take_up_participation = us_take_up_participation_diagnostics(export_frame)
    write_us_take_up_participation_diagnostics(
        take_up_participation,
        release_dir / "us_take_up_participation.json",
    )
    write_us_medicaid_take_up_diagnostics(
        medicaid_take_up_diagnostics,
        release_dir / "us_medicaid_take_up.json",
    )
    # The stage gate ran on the stage output; this re-checks the EXPORT frame
    # so a downstream transform that drops or flattens a count-calibrated
    # column cannot ship the engine-default landmine with only an
    # observational JSON field recording it.
    stale_count_calibrated = [
        str(row["variable"])
        for row in take_up_participation["programs"]
        if row.get("populace_treatment") == "count_calibrated"
        and row.get("ships_at_engine_default")
    ]
    if stale_count_calibrated:
        raise RuntimeError(
            "Release gates failed: count-calibrated take-up column(s) "
            f"{stale_count_calibrated} ship at the engine default on the "
            "export frame despite the stage having run."
        )
    if telemetry is not None:
        telemetry.attach_artifact(
            "us_take_up_participation",
            release_dir / "us_take_up_participation.json",
        )
        telemetry.attach_artifact(
            "us_medicaid_take_up",
            release_dir / "us_medicaid_take_up.json",
        )
        telemetry.stage("manifests", message="Writing release manifests.")
    timing["total_build_seconds"] = time.perf_counter() - build_started
    _build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=registry,
        dropped=compilation,
        target_profile_gate=target_profile_gate,
        health_input_gate=health_input_gate,
        base_population_gate=base_population_gate,
        incumbent_diagnostics=incumbent_diagnostics,
        immigration_gate=immigration_gate,
        input_mass_reference_gate=enforced_input_mass_reference_gate,
        degenerate_input_gate=degenerate_input_gate,
        ecps_parity_gate=enforced_ecps_parity_gate,
        hours_worked_gate=hours_worked_gate,
        snap_take_up_gate=snap_take_up_gate,
        eligibility_inputs_gate=eligibility_inputs_gate,
        pregnancy_gate=pregnancy_gate,
        snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
        timing=timing,
        warm_start_calibration=warm_start_calibration,
        selection_source=selection_source_payload,
        ledger_artifact=ledger_artifact.provenance(),
        default_dataset=default_dataset,
        medicaid_enrollment_substitutions=medicaid_enrollment_substitutions,
        staging=(
            {"run_id": telemetry.run_id, "repo_id": args.staging_repo_id}
            if telemetry is not None
            else None
        ),
    )
    if telemetry is not None:
        telemetry.attach_artifact("build_manifest", release_dir / "build_manifest.json")
        telemetry.attach_artifact(
            "release_manifest",
            release_dir / "release_manifest.json",
        )
        telemetry.complete()

    # Keep a copy of the exact base artifact beside diagnostics for local audit.
    shutil.copy2(base_h5, release_root / f"base_{base_h5.name}")
    print(
        json.dumps(
            {
                "release_id": release_id,
                "release_dir": str(release_dir),
                "artifact_root": str(artifact_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
