"""Build a contract-valid US fiscal refresh release from a Microcosm H5.

This is a narrow release builder for the Issue #40 fiscal target surface. It
starts from an existing Microcosm US H5, materializes the current structured
fiscal target rows, recalibrates only the household weights, writes a fresh
PolicyEngine-US H5, and emits the release contract files required by
``microcosm-publish-release``.
"""

from __future__ import annotations

import os

# microcosm#456 (#447 ops note): bound the default BLAS/OpenMP/joblib thread
# pools instead of inheriting machine geometry — per-thread scratch buffers
# scale the resident set with core count (measured 125 GB anon-RSS at 16 vCPU
# vs 249 GB at 32 vCPU for the same build). OpenBLAS and torch read these at
# library load, so this must precede the numpy/torch imports below; operator-
# set values always win (setdefault only). E402 is ignored for this file in
# pyproject for exactly this block.
_THREAD_POOL_DEFAULT = str(min(os.cpu_count() or 1, 16))
for _thread_pool_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "LOKY_MAX_CPU_COUNT",
):
    os.environ.setdefault(_thread_pool_variable, _THREAD_POOL_DEFAULT)

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.gates import (
    GateResult,
    TargetFitRequirement,
    default_valued_columns_gate,
    input_mass_parity_gate,
    nonconstant_columns_gate,
    parity_gate,
    tail_concentration_gate,
    target_fit_gate,
    target_profile_coverage_gate,
)
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage
from microcosm.build.staging import DEFAULT_STAGING_PREFIX, StagingTelemetry
from microcosm.build.us_runtime import (
    ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256,
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    ORG_2024_DONOR_CONTENT_SHA256,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
    SIPP_2023_HEAD_START_DONOR_SHA256,
    SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
    SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
    SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
    SIPP_2023_TIP_DONOR_SHA256,
    SIPP_2023_VEHICLE_DONOR_SHA256,
    SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
    SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    SOI_VARIABLE_MAP,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_SUPPORT_EXCLUSIONS,
    US_JCT_TAX_EXPENDITURE_REFORMS,
    US_MEDICAID_ENROLLMENT_TARGET_TABLE,
    US_MEDICAID_TAKE_UP_VARIABLE,
    US_SOURCE_MANIFEST,
    apply_us_medicaid_enrollment_substitutions,
    assert_release_input_coverage_manifest_current,
    assert_take_up_contract_current,
    assert_take_up_treatments_consistent,
    assert_target_parity_manifest_current,
    assert_validation_leaf_registry_current,
    compile_us_fiscal_target_registry,
    default_congressional_district_vintage_crosswalk_path,
    fetch_asec_2023_weeks_unemployed_source,
    fetch_org_2024_donor,
    fetch_scf_2022_full_extract,
    fetch_scf_2022_summary_extract,
    fetch_sipp_2023_financial_asset_donor,
    fetch_sipp_2023_tip_donor,
    hard_target_package_aliases,
    load_asec_2023_weeks_unemployed_source,
    load_congressional_district_vintage_crosswalk,
    load_org_2024_donor,
    load_scf_2022_auto_loan_donor,
    load_scf_2022_financial_asset_donor,
    load_sipp_2023_financial_asset_donor,
    load_sipp_2023_head_start_donor,
    load_sipp_2023_ssi_disability_donor,
    load_sipp_2023_tip_donor,
    load_sipp_2023_vehicle_donor,
    load_sipp_2023_voluntary_filing_donor,
    ssi_take_up_prior_basis_from_artifact,
    ssi_take_up_prior_basis_from_diagnostics,
    us_alimony_signal_gate,
    us_capital_gain_details_signal_gate,
    us_casualty_loss_signal_gate,
    us_child_support_signal_gate,
    us_childcare_signal_gate,
    us_disability_benefits_signal_gate,
    us_domestic_production_ald_signal_gate,
    us_education_inputs_signal_gate,
    us_educator_expense_signal_gate,
    us_eligibility_inputs_signal_gate,
    us_energy_subsidy_signal_gate,
    us_farm_business_income_signal_gate,
    us_form_4952_election_signal_gate,
    us_hours_worked_signal_gate,
    us_housing_inputs_signal_gate,
    us_immigration_composition_gate,
    us_medicaid_source_person_table,
    us_medicaid_take_up_diagnostics,
    us_medicaid_take_up_gate,
    us_medicare_take_up_signal_gate,
    us_misc_itemized_signal_gate,
    us_org_wages_signal_gate,
    us_other_health_insurance_signal_gate,
    us_pregnancy_signal_gate,
    us_prior_year_income_signal_gate,
    us_qbi_inputs_signal_gate,
    us_reform_coverage_smoke_gate,
    us_register_consistency_gate,
    us_relationship_inputs_signal_gate,
    us_release_input_coverage_gate,
    us_release_target_parity_gate,
    us_reported_coverage_vintage_signal_gate,
    us_retirement_contributions_signal_gate,
    us_retirement_distributions_signal_gate,
    us_salt_refund_income_signal_gate,
    us_scf_auto_loans_signal_gate,
    us_scf_wealth_signal_gate,
    us_sipp_head_start_signal_gate,
    us_sipp_tips_signal_gate,
    us_sipp_vehicles_signal_gate,
    us_snap_discretionary_exemption_signal_gate,
    us_snap_state_take_up_gate,
    us_snap_take_up_signal_gate,
    us_source_coverage_diagnostics,
    us_source_operation_handlers,
    us_ssi_disability_criteria_signal_gate,
    us_ssi_take_up_delivery_gate,
    us_ssi_take_up_diagnostics,
    us_ssi_take_up_gate,
    us_ssi_take_up_reporter_source_ids,
    us_take_up_participation_diagnostics,
    us_take_up_signal_gate,
    us_validation_input_coverage_gate,
    us_voluntary_filing_signal_gate,
    us_weeks_unemployed_signal_gate,
    us_wic_claim_signal_gate,
    us_workers_compensation_signal_gate,
    with_us_childcare_inputs,
    with_us_education_inputs,
    with_us_eligibility_inputs,
    with_us_energy_subsidy_input,
    with_us_hours_worked_inputs,
    with_us_immigration_inputs,
    with_us_medicaid_take_up,
    with_us_medicare_take_up_input,
    with_us_org_wages_inputs,
    with_us_other_health_insurance_inputs,
    with_us_pregnancy_inputs,
    with_us_qbi_input_reconciliation,
    with_us_relationship_inputs,
    with_us_retirement_contribution_inputs,
    with_us_retirement_distribution_inputs,
    with_us_scf_auto_loan_inputs,
    with_us_scf_wealth_inputs,
    with_us_sipp_head_start_input,
    with_us_sipp_tip_inputs,
    with_us_sipp_vehicle_inputs,
    with_us_snap_discretionary_exemption_inputs,
    with_us_snap_state_take_up,
    with_us_snap_take_up_inputs,
    with_us_ssi_disability_criteria,
    with_us_ssi_take_up,
    with_us_take_up_inputs,
    with_us_voluntary_filing_input,
    with_us_weeks_unemployed,
    with_us_wic_claim_input,
    write_us_medicaid_take_up_diagnostics,
    write_us_snap_state_take_up_diagnostics,
    write_us_source_coverage_diagnostics,
    write_us_ssi_take_up_diagnostics,
    write_us_take_up_participation_diagnostics,
)
from microcosm.build.us_runtime.demographics import (
    CENSUS_NATIONAL_AGE_BENCHMARK,
    demographics_payload,
    geography_coverage_payload,
    population_by_age_from_sim,
    write_demographics,
)
from microcosm.build.us_runtime.engine_lifecycle import release_engine_simulation
from microcosm.build.us_runtime.exact_k_ladder import (
    ExactKLadderCalibration,
    assert_exact_k_realized_count,
    calibrate_exact_k_ladder,
    exact_k_ladder_manifest_payload,
)
from microcosm.build.us_runtime.fiscal_targets import (
    SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE,
)
from microcosm.build.us_runtime.h5_io import (
    AuthenticatedPoolH5,
    identify_us_multispine_pool_manifest,
    load_authenticated_us_multispine_pool_for_release,
    load_simulation_ready_us_multispine_pool,
    require_authenticated_us_multispine_pool_h5,
    us_multispine_pool_release_receipt,
)
from microcosm.build.us_runtime.input_mass import us_input_mass_totals
from microcosm.build.us_runtime.l0_refit_export import (
    attach_l0_refit_entity_weights,
    load_us_frame,
)
from microcosm.build.us_runtime.nonzero_shares import us_nonzero_shares
from microcosm.build.us_runtime.parity_reference import (
    EcpsParityReference,
    ParityKnownGap,
    load_ecps_parity_known_gaps,
    load_ecps_parity_reference,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    assert_puf_capital_gains_tail_survives_selection,
)
from microcosm.build.us_runtime.reform_validation import (
    default_baseline_level_specs,
    default_simulate_factory,
    load_default_reform_specs,
    reform_validation_payload,
    write_reform_validation,
)
from microcosm.build.us_runtime.ssi_take_up import (
    US_SSI_TAKE_UP_AGE_TARGETS,
    US_SSI_TAKE_UP_ENFORCED_BAND_KEYS,
    US_SSI_TAKE_UP_OUTPUT_COLUMNS,
    SSITakeUpPriorBasis,
)
from microcosm.build.us_runtime.warm_start_selection import (
    DEFAULT_SELECTION_JOIN_KEY,
    SELECTION_MODES,
    load_selection_source_from_h5,
    load_selection_source_from_manifest,
    select_frozen_support,
)
from microcosm.calibrate import (
    TargetRegistry,
    TargetSpec,
    calibrate,
    calibrate_l0_refit,
    relative_error_loss,
)
from microcosm.calibrate.diagnostics import (
    diagnostics_payload,
    write_calibration_diagnostics,
)
from microcosm.data.contract import (
    EVIDENCE_RELEASE_ID_SEGMENT,
    EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION,
)
from microcosm.data.us_critical_targets import (
    US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR,
    US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS,
    is_congressional_district_target,
)
from microcosm.data.us_critical_targets import (
    US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT as SHARED_US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT,
)
from microcosm.frame import Frame, MassChange, WeightKind, Weights, read_frame_table
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
)
from microcosm.frame.units import US_SCHEMA

PERIOD = 2024
REPO_ID = "policyengine/populace-us"
STAGING_REPO_ID = "policyengine/populace-us-staging"
DATASET_FILENAME = "populace_us_2024.h5"
CALIBRATION_FILENAME = "populace_us_2024_calibration.npz"
FINAL_HOUSEHOLD_WEIGHTS_FILENAME = "final_household_weights.npy"
FINAL_HOUSEHOLD_WEIGHTS_METADATA_FILENAME = "final_household_weights.json"
FINAL_HOUSEHOLD_WEIGHT_IDS_FILENAME = "final_household_weight_ids.npy"
FINAL_HOUSEHOLD_WEIGHTS_SCHEMA_VERSION = 1
POST_EXPORT_ABSOLUTE_TOLERANCE = 1_000_000.0
POST_EXPORT_RELATIVE_TOLERANCE = 5e-4
# microcosm#566/#567 dense-arm adjudication: microcosm#508 delivered-weight
# recomputes have not landed the dense frame's adult band pair in the
# envelope on either observed frame. P2 is the clean one-retry record
# (current-frame attempt then its one permitted recompute: 18-64
# +5.8%/65+ +24.8% -> +8.2%/+20.0%). P3's attempts were already anchored
# on delivered bases (65+ +34.6% with 18-64 in-band, then +8.3%/+19.8%
# after recomputing again — the recompute moved 18-64 OUT of band while
# improving 65+); that chain shape is now refused outright by the
# chain-depth guard in ssi_take_up_prior_basis_from_artifact. The dense
# diagnostic arm therefore FENCES its adult bands — the under-18 pattern
# extended: the miss ships in the scorecard as a known boundary, never as
# an enforced contract and never as saturation-as-success. The sparse
# certified default passes no fences and keeps hard enforcement.
# RE-ADJUDICATES when microcosm#566's damped fixed-point protocol lands.
_US_DENSE_SSI_FENCE_ADJUDICATION = (
    "Fenced for the dense diagnostic arm (microcosm#566/#567): "
    "microcosm#508 delivered-weight recomputes have not landed the adult "
    "band pair in the envelope on either observed frame (P2, current-"
    "frame attempt then its one permitted recompute: +5.8%/+24.8% -> "
    "+8.2%/+20.0%; P3, attempts already anchored on delivered bases: "
    "65+ +34.6%, then +8.3%/+19.8% after recomputing again — a chain "
    "the microcosm#508 loader now refuses). Further recomputes are the "
    "deleted microcosm#463-class loop. This band's miss ships in the "
    "scorecard as a known boundary — never as an enforced contract. "
    "Re-adjudicates when the microcosm#566 damped fixed-point protocol "
    "lands. The sparse certified default keeps hard enforcement."
)
US_DENSE_SSI_TAKE_UP_ENFORCEMENT_FENCES: dict[str, str] = {
    "18_64": _US_DENSE_SSI_FENCE_ADJUDICATION,
    "65_plus": _US_DENSE_SSI_FENCE_ADJUDICATION,
}
assert set(US_DENSE_SSI_TAKE_UP_ENFORCEMENT_FENCES) == set(
    US_SSI_TAKE_UP_ENFORCED_BAND_KEYS
), (
    "The dense-arm fence adjudication must cover exactly the "
    "normally-enforced SSI bands. A new enforced band needs a dense-arm "
    "adjudication first: fence it here with its documented reason, or "
    "amend this assertion as the record of the decision to enforce it "
    "on the dense arm too."
)

US_FISCAL_TARGET_LOSS_WEIGHTING = (
    "sqrt_value_concept_budget_weighted_mape_50_50_amount_count_target_scale_cap_100pct"
)
US_FISCAL_TARGET_VALUE_WEIGHT_POWER = 0.5
US_FISCAL_TARGET_LOSS_CAP = 1.0
RATIFIED_EXACT_K_COUNTS = frozenset({57_240, 20_000})


class IncumbentLossBasisMismatchError(RuntimeError):
    """The pinned incumbent was scored on a different fiscal-loss basis."""


class PoolReleaseIdentityMismatchError(ValueError):
    """The configured pool identity differs from its authenticated manifest."""


# Bumped 1 -> 2 for #217: the per-reform income-tax cache key now depends only on
# the inputs that actually determine per-household reform estimates and no longer
# includes build_commit / seed / target_registry_version. Old (v1) coarse-key
# entries live under different filenames, so a mixed cache dir never collides.
# Bumped 2 -> 3 for #557: absolute reform vectors now bind to the target-frame
# materializer identity. Pre-#557 vectors can reflect release-refitted retirement
# leaves and must not mix with a preserved-surface baseline.
TARGET_MATERIALIZATION_CACHE_SCHEMA_VERSION = 3
# The subset of the materialization cache context that determines per-household JCT
# reform income-tax vectors (#217). The raw build commit and calibration settings
# stay excluded. The materializer-identity digest transitively binds staged-frame
# semantics, seed, registry, and support selection, so any such change invalidates
# the vectors even when the on-disk base hash is stable.
REFORM_VECTOR_CACHE_CONTEXT_KEYS: tuple[str, ...] = (
    "base_dataset_sha256",
    "weeks_unemployed_source_sha256",
    "policyengine_us_version",
    "target_period",
    "congressional_district_vintage_crosswalk_sha256",
    # The frozen SSI take-up assignment is a base-frame input to every
    # materialized vector. Whether any JCT reform income-tax estimate can
    # actually move with takes_up_ssi_if_eligible is an engine-graph
    # question this build must not answer by assumption, so the digest
    # invalidates reform vectors too — correctness over cache warmth
    # (microcosm#507/#508 sol review round 2, finding 2).
    "ssi_take_up_assignment_sha256",
    # The selected identities determine which household rows the vectors
    # describe. Same-length supports can share positional SSI flag bytes, so
    # this digest remains independent of the assignment digest.
    "selection_identities_sha256",
    # Absolute reform-tax vectors are later subtracted from the freshly
    # materialized baseline. Bind them to the complete target-frame identity
    # so pre-#557 release-refitted surfaces cannot mix with preserved surfaces.
    "target_frame_materializer_identity_sha256",
)
TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION = 2
# 2: the medicaid_take_up stage (microcosm #331) changed base_frame's
# takes_up_medicaid_if_eligible before target-frame materialization, so
# medicaid_enrolled target columns differ from version-1 checkpoints; the
# checkpoint identity hashes the on-disk base dataset, not the staged frame,
# and would otherwise silently reuse pre-stage frames.
# 3: the post-base SIPP SSI-disability stage restores
# meets_ssi_disability_criteria after SCF assets, changing SSI eligibility and
# target vectors without changing that on-disk base hash.
# 4: reporter-anchored SSI take-up now replaces the engine-default universal
# flag after the disability stage, changing SSI and its target vectors while
# the on-disk base hash remains unchanged.
# 5: the measured-SIPP Head Start stage now replaces the engine-default
# universal take-up flag before target materialization and must be present on
# every restored checkpoint even though the on-disk base hash is unchanged.
# 6: the official-ASEC sidecar restores 2022 LKWEEKS before target
# materialization. The source is external to the on-disk base hash, so old
# checkpoints must not survive the new measured input.
# 7: SSI take-up became a one-shot seeded Bernoulli at registry band priors
# (microcosm#469) — checkpoints materialized from count-matched flags must
# not survive, or the solve would run on old SSI rows while the frame
# carries the new assignment (PR #477 review finding 2).
# 8: the ORG full-year-equivalence stage (#539) rewrites the staged org-wage
# inputs before target materialization while the on-disk base hash is
# unchanged; pre-#539 checkpoints carry the old ORG rows and must not be
# reused (microcosm#543, post-merge audit).
# 9: #374 SIPP+SCF financial-asset blend changes the pre-materialization
# frame; warm SCF-only checkpoints must not calibrate the blended frame.
# 10: #557 preserves the staged retirement-distribution surface through
# release materialization; pre-#557 checkpoints can carry QRF-refitted leaves
# and must not serve the preserved-surface baseline.
# 11: target-frame checkpoint columns now preserve nullable booleans as
# canonical bool values plus an explicit uint8 null mask. Older checkpoints
# cannot attest this lossless physical representation.
TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION = 11
DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE = 5_000
DEFAULT_L0_REFIT_LAMBDA_SHARE = 0.8
DEFAULT_US_FISCAL_CALIBRATION_EPOCHS = 1_500


def _collect_batch_garbage() -> None:
    """Keep batch loops tidy without traversing the full object graph."""

    gc.collect(0)


def _collect_family_garbage() -> None:
    """Full collection at target-family boundaries (microcosm#456).

    ``release_engine_simulation`` frees each batch's array mass by refcount,
    but the residual simulation skeletons are cyclic, and anything promoted
    past generation 0 is invisible to ``_collect_batch_garbage`` forever —
    CPython's own full collections are throttled against the build's multi-GB
    long-lived heap, so without an explicit sweep the skeletons accumulate
    for the run's lifetime. One full collection per materialized family is
    cheap next to the family's engine work.
    """

    gc.collect()


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
US_SOCIAL_SECURITY_COMPONENT_TARGET_ROLES = {
    "ssa_retirement_total": "social_security_retirement",
    "ssa_disability_total": "social_security_disability",
    "ssa_dependents_total": "social_security_dependents",
    "ssa_survivors_total": "social_security_survivors",
}
US_CRITICAL_TARGET_FIT_REQUIREMENTS = US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS

#: Blanket within-tolerance blocking for every national SOI Pub 1304 Table 1.4
#: dollar row (microcosm#462). The exact-name register above only blocks rows
#: someone enumerated; Build M shipped the Table 1.4 capital-gain-distributions
#: dollar row at +634.8% relative error (and net capital gains at -25.6%) with
#: both recorded in its own calibration_diagnostics.json, because no register
#: named them. 0.25 is the established broad-fit bound (the per-family hard
#: threshold and the incumbent-improvement hard stop): on the live Build M
#: surface it fails exactly the two defect rows and passes the other nine
#: Table 1.4 dollar rows (worst passer: taxable_social_security_amount at
#: -10.9%). Deliberately no incumbent-improvement escape — a national dollar
#: row beyond broad fit is never certifiable.
US_SOI_TABLE_1_4_NATIONAL_DOLLAR_MAX_ABS_RELATIVE_ERROR = (
    SHARED_US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT.max_abs_relative_error
)
US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT = TargetFitRequirement(
    requirement_id=(
        SHARED_US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT.requirement_id
    ),
    label=SHARED_US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT.label,
    accepted_name_substrings=(
        SHARED_US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT.name_substrings
    ),
    accepted_name_suffixes=(
        SHARED_US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT.name_suffixes
    ),
    max_abs_relative_error=US_SOI_TABLE_1_4_NATIONAL_DOLLAR_MAX_ABS_RELATIVE_ERROR,
    notes=(
        "microcosm#462: the Build M live default shipped non_sch_d_capital_gains "
        "at $74.6B against its $10.2B SOI target with no blocking tolerance on "
        "the dollar row."
    ),
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

#: Series-identity qualifiers, distinct from the domain filters above: each
#: identifies WHICH published series the registry selected (a NIPA table line,
#: a LIHEAP program count), was applied at registry fact-selection time, and
#: restricts nothing in the microdata — there is no household-level "series
#: code" to filter on. The materializer must treat them as inert provenance;
#: listing a key here asserts a reviewer verified it is identity-only. Unknown
#: ledger_filter_* keys remain fatal so a genuine domain filter can never be
#: silently ignored (the Build M sparse stop that motivated this class).
IDENTITY_LEDGER_FILTER_METADATA_KEYS = frozenset(
    {
        "ledger_filter_bea_nipa.series_code",
        "ledger_filter_administering_entity",
        "ledger_filter_program",
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
    "takes_up_dc_ptc": ("DC PTC take-up imputation backlog; constant True."),
    "second_home_mortgage_balance": (
        "Second-home mortgage decomposition not imputed; constant at the"
        " engine default (PolicyEngine/microcosm#38)."
    ),
    "second_home_mortgage_interest": (
        "Second-home mortgage decomposition not imputed; constant at the"
        " engine default (PolicyEngine/microcosm#38)."
    ),
    "second_home_mortgage_origination_year": (
        "Second-home mortgage decomposition not imputed; constant at the"
        " engine default (PolicyEngine/microcosm#38)."
    ),
    "takes_up_early_head_start_if_eligible": (
        "Early Head Start person enrollment is absent from every locked source; "
        "see the archived-derivation and source-domain evidence in the parity "
        "gap register (PolicyEngine/microcosm#312)."
    ),
    # ssn_card_type and immigration_status_str are intentionally NOT excluded:
    # PR #266 imputes them from CPS ASEC citizenship, so a base where they are
    # still constant at CITIZEN skipped that stage and should fail this gate.
    "is_wic_at_nutritional_risk": (
        "Person-level nutritional-risk assessments are absent from all locked "
        "sources; see the archived-derivation evidence in the parity gap "
        "register (PolicyEngine/microcosm#312)."
    ),
    "s_corp_income": (
        "Combined partnership/S-corp income is carried in partnership_income "
        "in pre-PUF-support bases; the S-corp leaf is constant zero there."
    ),
}

#: Person inputs SNAP work-requirement rules read that have NO CPS ASEC
#: source and are not seeded: they default to False in the engine, so the
#: compliance/exemption channels they drive never fire (microcosm #351,
#: #249 for the work-program family). They are not
#: persisted columns, so the degenerate-input gate cannot see them; this
#: register makes the assumption visible in every release manifest instead.
#: is_pregnant is NOT here: the pregnancy stage seeds it. Likewise
#: is_incapable_of_self_care left this register with microcosm#451 item 1:
#: the adult_care_inputs base stage seeds it from the measured ASEC
#: PEDISDRS self-care difficulty item, which is the direct instrument
#: operationalization the original entry believed absent.
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
        "(PolicyEngine/microcosm#351)."
    ),
    "was_in_foster_care": (
        "No ASEC item for foster-care history, so the pre-HR1 former-"
        "foster-youth ABAWD exemption (7 CFR 273.24(c)(9)) cannot fire "
        "(PolicyEngine/microcosm#351)."
    ),
    "is_snap_work_program_participant": (
        "No ASEC item measures SNAP E&T or qualifying work-program "
        "participation, so compliance via program participation never "
        "fires; USDA reports E&T reaches a small minority of "
        "participants, so the always-False default understates "
        "compliance only modestly (PolicyEngine/microcosm#249)."
    ),
    "weekly_snap_work_program_hours": (
        "No ASEC item measures qualifying work-program hours, so combined "
        "work-plus-program hours toward the 20-hour ABAWD test omit "
        "program hours (PolicyEngine/microcosm#249)."
    ),
    "is_snap_workfare_participant": (
        "No ASEC item measures workfare participation under 7 CFR "
        "273.7(m), which satisfies the ABAWD requirement regardless of "
        "hours; defaults False (PolicyEngine/microcosm#249)."
    ),
}


def _env_default(name: str, default: str) -> str:
    """Return a trimmed environment override, treating blank as unset.

    ``os.environ.get(name, default)`` falls back only when the variable is
    absent. An exported empty string otherwise silently defeats the staging
    default. For staging, only ``--no-staging`` should turn telemetry off.
    """

    return os.environ.get(name, "").strip() or default


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
US_SNAP_HOUSEHOLDS_TARGET_ROLE = "snap_households"

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


def _parse_ratified_exact_k(value: str) -> str | int:
    if value == "N":
        return value
    if value in {str(k) for k in RATIFIED_EXACT_K_COUNTS}:
        return int(value)
    raise argparse.ArgumentTypeError(
        "ExactKCharterError: --exact-k must be exactly N, 57240, or 20000; "
        f"got {value!r}."
    )


def _assert_pool_release_identity(
    configured_release_id: str,
    pool_manifest: Mapping[str, object],
) -> str:
    return _assert_pool_release_id_value(
        configured_release_id,
        pool_manifest.get("publication_run_id"),
    )


def _assert_pool_release_id_value(
    configured_release_id: str,
    publication_run_id: object,
) -> str:
    if (
        not isinstance(publication_run_id, str)
        or not publication_run_id
        or configured_release_id != publication_run_id
    ):
        raise PoolReleaseIdentityMismatchError(
            "PoolReleaseIdentityMismatchError: configured pool release id "
            f"{configured_release_id!r} does not match authenticated manifest "
            f"publication_run_id {publication_run_id!r}."
        )
    return publication_run_id


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-h5",
        type=Path,
        help="Existing Microcosm US H5 to recalibrate. Defaults to HF latest.",
    )
    parser.add_argument(
        "--pool-manifest",
        type=Path,
        help=(
            "Simulation-ready build_us_multispine_pool.py manifest. The "
            "manifest, rather than a bare H5, is the readiness authority. "
            "Mutually exclusive with --base-h5."
        ),
    )
    parser.add_argument(
        "--allow-gate-failed-base-pool",
        action="store_true",
        help=(
            "Explicitly allow --base-h5 to consume an authenticated current "
            "stacked pool with status=gate_failed and simulation_ready=false. "
            "The red agreement-gate verdict is carried into the release "
            "manifest for a separate human publication decision. Without "
            "this flag an identified pool H5 remains fail-closed. The exact-k "
            "--pool-manifest arm is always simulation-ready-only."
        ),
    )
    parser.add_argument(
        "--pool-manifest-sha256",
        help=(
            "Expected SHA-256 of --pool-manifest. Required for an exact-k "
            "ladder release so the artifact-store envelope is pinned."
        ),
    )
    parser.add_argument(
        "--pool-release-id",
        help=("Authenticated publication_run_id of the pool artifact envelope."),
    )
    parser.add_argument(
        "--exact-k",
        type=_parse_ratified_exact_k,
        help=(
            "Ratified ladder point: N, 57240, or 20000 households. N resolves "
            "to the authenticated pool size and uses identity support with an "
            "ordinary full-pool refit."
        ),
    )
    parser.add_argument(
        "--exact-k-pi-hi",
        type=float,
        help="Certainty-unit threshold for --exact-k selection.",
    )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help=(
            "PolicyEngine Ledger consumer artifact directory (manifest.json "
            "+ consumer_facts.jsonl, hash-verified) or a bare "
            "consumer_facts.jsonl file, used to resolve every fiscal target "
            "value. Microcosm package resources declare target references "
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
            "(PolicyEngine/ledger#71; microcosm#212). Every waived target "
            "carries period_contract_waiver metadata in diagnostics. Without "
            "this flag such targets fail the build unless --age-targets "
            "transforms them."
        ),
    )
    parser.add_argument(
        "--qrf-tail-concentration-exclusions",
        type=Path,
        help=(
            "Optional JSON object of export column -> reason for sparse "
            "QRF-imputed columns allowed past the tail-concentration "
            "top-share threshold (microcosm#464 gate). Stale entries fail the "
            "gate; the file sha and entries are recorded in the release "
            "diagnostics."
        ),
    )
    parser.add_argument(
        "--selection-mass-protection",
        action="append",
        default=[],
        metavar="COLUMN",
        help=(
            "Input column whose locked-source mass (measured on the base "
            "pool at base weights, never hardcoded) is injected as a "
            "synthetic national calibration target, so the refit cannot "
            "crush the carriers a protect-swap placed in the frozen "
            "selection (PolicyEngine/microcosm#445; #434). Repeatable. The "
            "protection lifts when the concept gains a real Ledger fact."
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
        "--incumbent-diagnostics-sha256",
        help=(
            "Expected SHA-256 of --incumbent-diagnostics. Required for an "
            "exact-k ladder release."
        ),
    )
    parser.add_argument(
        "--frozen-target-surface-sha256",
        help=(
            "Expected target-surface SHA-256 embedded in the pinned "
            "incumbent diagnostics. Required for an exact-k ladder release."
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
            "--export-input-mass-reference-h5 (microcosm#327)."
        ),
    )
    parser.add_argument(
        "--export-input-mass-reference-h5",
        type=Path,
        help=(
            "Optional certified release H5 whose persisted PolicyEngine input "
            "mass the CALIBRATED EXPORT frame is compared against, instead of "
            "the raw pre-calibration base (microcosm#327). Calibration correctly "
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
            "Diagnostic escape hatch (microcosm#368): record the release "
            "input-column coverage gate result — required eCPS input columns "
            "the export drops or leaves degenerate — without failing the "
            "build. Release builds must leave this unset; the gate is a hard "
            "certification blocker."
        ),
    )
    parser.add_argument(
        "--allow-qrf-tail-concentration",
        action="store_true",
        help=(
            "Diagnostic escape hatch (microcosm#462): record the QRF "
            "tail-concentration gate result — sparse QRF-imputed dollar "
            "columns whose top-k weighted records carry an implausible share "
            "of the weighted mass (the non_sch_d_capital_gains donor-ceiling "
            "point mass) — without failing the build. Release builds must "
            "leave this unset."
        ),
    )
    parser.add_argument(
        "--skip-reform-coverage-smoke",
        action="store_true",
        help=(
            "Do not run the reform-coverage smoke gate (microcosm#368): the "
            "pinned bound-reform probes (SSI $10k/$20k asset limits) that must "
            "score nonzero on the written release. Skipping loses the "
            "end-to-end $0-reform backstop; release builds should leave it on."
        ),
    )
    parser.add_argument(
        "--allow-reform-coverage-smoke-failures",
        action="store_true",
        help=(
            "Diagnostic escape hatch (microcosm#368): record the reform-coverage "
            "smoke gate result without failing the build when a bound reform "
            "probe scores ~$0. Release builds must leave this unset."
        ),
    )
    parser.add_argument(
        "--evidence-release",
        action="store_true",
        help=(
            "Build an EVIDENCE-tier release (microcosm#506): on terminal-gate "
            "failure, continue to write the H5 and full manifests with the "
            "recorded failures carried verbatim in the release manifest's "
            "known_failures block (each with an owner issue), instead of "
            "refusing to export. NOT a gate bypass: the release id carries an "
            "'-evidence-' segment, the release manifest declares the evidence "
            "schema marker the certified contract structurally rejects, and a "
            "run whose terminal gates all pass is refused under this flag "
            "(rerun without it). Preflight and mid-build source-stage gates, "
            "artifact-integrity assertions (e.g. --audit-export-targets), and "
            "the dirty-worktree refusal still abort: the evidence tier "
            "relaxes terminal gate verdicts, never artifact auditability. "
            "Incompatible with --exact-k (the ladder candidate lane has its "
            "own tag-only publication contract)."
        ),
    )
    parser.add_argument(
        "--evidence-failure-owners",
        type=Path,
        help=(
            "JSON file mapping failure-substring patterns to owner issue refs "
            '(e.g. {"Input coverage failed:": "PolicyEngine/microcosm#368"}) '
            "for --evidence-release. Checked ahead of the standing "
            "US_EVIDENCE_FAILURE_OWNERS register; a recorded failure matching "
            "neither refuses the evidence export — every shipped failure "
            "needs an owner."
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
        "--target-family-loss-multiplier",
        action="append",
        default=[],
        metavar="FAMILY=MULTIPLIER",
        help=(
            "Multiply the compiled loss weight of every target in FAMILY by "
            "MULTIPLIER before calibration (repeatable, e.g. usda_snap=8). "
            "Applied on top of the standard loss weighting, after which the "
            "weight vector is renormalized to mean 1 so the overall loss "
            "scale is unchanged. The build fails if FAMILY matches no "
            "compiled target. Multipliers are recorded in the calibration "
            "diagnostics."
        ),
    )
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
        "--ssi-take-up-prior-weight-basis",
        type=Path,
        help=(
            "Optional us_ssi_take_up.json diagnostics artifact from a prior "
            "attempt's final release-weight measurement (current schema 4, or "
            "legacy schema 2/3 capacity-floor seeds such as the certified "
            "predecessor release's). The SSI take-up Bernoulli thresholds are "
            "then computed against that attempt's delivered per-band candidate "
            "capacities instead of this run's pre-calibration weights — the "
            "microcosm#508 fix for the microcosm#507 aged-band collapse: "
            "thresholds truthful against release-kind weights, still drawn "
            "exactly once, with no reconcile loop. The artifact must carry the "
            "same SSA band target contract this build compiles; the enforced-"
            "band delivery gate verifies the landed counts either way, and on "
            "failure writes this run's us_ssi_take_up.json as the basis for the "
            "retry. Requires --ssi-take-up-prior-weight-basis-sha256."
        ),
    )
    parser.add_argument(
        "--ssi-take-up-prior-weight-basis-sha256",
        help=(
            "Required companion to --ssi-take-up-prior-weight-basis: the "
            "expected sha256 of that artifact, read from the producing "
            "release's manifest (or the failed attempt's error message). "
            "Pinning the hash in the launch command makes the basis choice "
            "an auditable receipt instead of whatever bytes sit at the "
            "path (microcosm#507/#508)."
        ),
    )
    parser.add_argument(
        "--selection-source-h5",
        type=Path,
        help=(
            "Optional microcosm US H5 whose record set defines a frozen support "
            "to recover onto the base pool (microcosm#328). The base frame is "
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
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--asec-2023-weeks-unemployed-source",
        type=Path,
        help=(
            "Optional local path to the SHA-pinned official 2023 ASEC CSV ZIP "
            "used to restore income-year-2022 LKWEEKS. When omitted the "
            "official Census archive is fetched and verified."
        ),
    )
    parser.add_argument(
        "--scf-summary-extract",
        dest="scf_summary_extract",
        default=None,
        help=(
            "Path to the Federal Reserve SCF 2022 public summary extract "
            "(rscfp2022.dta) that feeds the signed household net-worth and SSI "
            "countable-resource asset imputations (scf_wealth stage, "
            "microcosm#49/#356/#368). When omitted the fixed-vintage extract is "
            "fetched and cached from the Federal Reserve."
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
        "--sipp-vehicle-donor",
        type=Path,
        help=(
            "Optional local path to the sha-pinned full SIPP 2023 public-use "
            "file that feeds financial assets, SSI disability criteria, "
            "household vehicle count/value, and measured voluntary tax filing. "
            "When omitted the immutable donor revision is fetched and verified."
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
            "the target-frame materializer identity and reform (see #217/#557)."
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
        "--congressional-district-vintage-crosswalk",
        type=Path,
        help=(
            "Source-to-current congressional-district crosswalk "
            "artifact with source_geography_id, target_geography_id, and "
            "weight columns. Defaults to the packaged Census-built crosswalk "
            "(microcosm.build.us_runtime.data; see "
            "CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK.md); pass a path to "
            "override the default."
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
        default=_env_default("POPULACE_STAGING_REPO_ID", STAGING_REPO_ID),
        help=(
            "Hugging Face dataset repo to upload staging telemetry to while "
            "the build runs. On by default (uploads are best-effort and never "
            "fail the build); override with POPULACE_STAGING_REPO_ID or "
            "disable with --no-staging. An empty POPULACE_STAGING_REPO_ID is "
            "ignored rather than read as off."
        ),
    )
    parser.add_argument(
        "--age-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compile-time period aging of dollar-amount targets whose source "
            "period differs from the build period (PolicyEngine/microcosm#116, "
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
        default=_env_default("POPULACE_STAGING_PREFIX", DEFAULT_STAGING_PREFIX),
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
    args = parser.parse_args(argv)
    if args.congressional_district_vintage_crosswalk is None:
        # Every build compiles the same national + state + CD target surface,
        # translated through the canonical packaged vintage crosswalk unless
        # the caller supplies an explicit replacement.
        args.congressional_district_vintage_crosswalk = (
            default_congressional_district_vintage_crosswalk_path()
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
    if not args.no_staging and not args.staging_dir and not args.staging_repo_id:
        # Staging with nowhere to write is a configuration error, not a quiet
        # skip. Turning staging off is an explicit decision, never a side
        # effect of a blank repo id.
        parser.error(
            "--staging-repo-id is empty and no --staging-dir is set, so staging "
            "telemetry would silently do nothing. Pass --no-staging to skip "
            "staging deliberately, or --staging-dir for a local-only run."
        )
    if args.evidence_failure_owners is not None and not args.evidence_release:
        parser.error("--evidence-failure-owners requires --evidence-release.")
    if args.allow_gate_failed_base_pool and args.base_h5 is None:
        parser.error("--allow-gate-failed-base-pool requires --base-h5.")
    if args.evidence_release and args.exact_k is not None:
        parser.error(
            "--evidence-release is incompatible with --exact-k: ladder "
            "candidates publish tag-only under their own contract; the "
            "evidence tier (microcosm#506) covers the national artifact."
        )
    ladder_values = (
        args.exact_k,
        args.exact_k_pi_hi,
        args.pool_manifest,
        args.pool_manifest_sha256,
        args.pool_release_id,
    )
    if any(value is not None for value in ladder_values):
        if any(value is None for value in ladder_values):
            parser.error(
                "--exact-k, --exact-k-pi-hi, --pool-manifest, "
                "--pool-manifest-sha256, and --pool-release-id must be "
                "provided together."
            )
        if args.base_h5 is not None:
            parser.error("--pool-manifest is mutually exclusive with --base-h5.")
        if args.seed is None or args.seed < 0:
            parser.error(
                "ExactKExplicitSeedError: --exact-k requires an explicit "
                "non-negative --seed."
            )
        if not args.no_staging:
            parser.error(
                "ExactKPointerSuppressionError: --exact-k requires --no-staging."
            )
        if not math.isfinite(args.exact_k_pi_hi) or not (
            0.0 <= args.exact_k_pi_hi <= 1.0
        ):
            parser.error("--exact-k-pi-hi must be finite and in [0, 1].")
        if len(args.pool_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in args.pool_manifest_sha256
        ):
            parser.error(
                "--pool-manifest-sha256 must be exactly 64 lowercase "
                "hexadecimal characters."
            )
        if not args.pool_release_id.strip():
            parser.error("--pool-release-id must be non-empty.")
        if args.incumbent_diagnostics is None:
            parser.error(
                "--exact-k requires --incumbent-diagnostics so every ladder "
                "point is judged against the incumbent on the frozen target "
                "register."
            )
        for flag, value in (
            (
                "--incumbent-diagnostics-sha256",
                args.incumbent_diagnostics_sha256,
            ),
            (
                "--frozen-target-surface-sha256",
                args.frozen_target_surface_sha256,
            ),
        ):
            if (
                value is None
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                parser.error(
                    f"{flag} must be exactly 64 lowercase hexadecimal characters."
                )
        if args.ledger_facts_sha256 is None or args.ledger_manifest_sha256 is None:
            parser.error(
                "--exact-k requires both --ledger-facts-sha256 and "
                "--ledger-manifest-sha256 to pin the frozen target register."
            )
        if args.dense_default_dataset:
            parser.error(
                "--exact-k owns the full-pool identity arm; do not combine it "
                "with --dense-default-dataset."
            )
        if (
            args.selection_source_h5 is not None
            or args.selection_source_manifest is not None
        ):
            parser.error(
                "--exact-k operates on the complete multispine pool and cannot "
                "be combined with a frozen selection source."
            )
    elif args.seed is None:
        # Preserve the pre-exact-k default for every legacy lane.
        args.seed = 0
    multipliers: dict[str, float] = {}
    for entry in args.target_family_loss_multiplier:
        family, separator, raw_value = entry.partition("=")
        try:
            value = float(raw_value)
        except ValueError:
            value = math.nan
        if not separator or not family or not math.isfinite(value) or value <= 0.0:
            parser.error(
                "--target-family-loss-multiplier expects FAMILY=MULTIPLIER "
                f"with a positive finite multiplier, got {entry!r}."
            )
        if family in multipliers:
            parser.error(f"--target-family-loss-multiplier repeats family {family!r}.")
        multipliers[family] = value
    args.target_family_loss_multipliers = multipliers
    return args


def _finite_or_none(value: float) -> float | None:
    """A loss for the diagnostics payload, scrubbed the way JSON needs it.

    Mirrors ``microcosm.calibrate.diagnostics._finite``: the artifact
    serializes strict JSON (``allow_nan=False``), and a non-finite loss is
    an EXPECTED batched gate failure — ``_release_gate_failures`` records it
    and the run continues to the terminal batch. Smuggling the raw value
    into the payload makes the failure destroy the artifact that reports it
    (microcosm#547).
    """

    value = float(value)
    return value if math.isfinite(value) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _legacy_base_h5_sha256(path: Path) -> str:
    """Hash only the non-pool base path selected by main's legacy branch."""

    return _sha256(path)


def _load_base_pool_if_identified(
    path: Path,
    *,
    allow_gate_failed_base_pool: bool,
) -> tuple[Frame | None, dict[str, object] | None, AuthenticatedPoolH5 | None]:
    """Authenticate a pool supplied through legacy ``--base-h5`` if present."""

    manifest_path = identify_us_multispine_pool_manifest(path)
    if manifest_path is None:
        if allow_gate_failed_base_pool:
            raise ValueError(
                "--allow-gate-failed-base-pool was set, but --base-h5 does not "
                "identify as a US multispine pool."
            )
        return None, None, None

    frame, manifest, authenticated_pool_h5 = (
        load_authenticated_us_multispine_pool_for_release(
            manifest_path,
            allow_terminal_gate_failure=allow_gate_failed_base_pool,
        )
    )
    require_authenticated_us_multispine_pool_h5(
        path,
        authenticated_pool_h5,
        consumer="US fiscal refresh release builder --base-h5",
    )
    receipt = us_multispine_pool_release_receipt(
        manifest,
        authenticated_pool_h5,
        allow_gate_failed_base_pool=allow_gate_failed_base_pool,
    )
    return frame, receipt, authenticated_pool_h5


def _copy_base_h5_for_local_audit(
    source: Path,
    destination: Path,
    *,
    authenticated_pool_h5: AuthenticatedPoolH5 | None,
) -> Path:
    if authenticated_pool_h5 is not None:
        return authenticated_pool_h5.copy_verified_to(
            destination,
            consumer="builder final local-audit copy",
        )
    shutil.copy2(source, destination)
    return destination


def _runtime_versions() -> dict[str, str]:
    packages = (
        "microcosm-build",
        "microcosm-calibrate",
        "microcosm-data",
        "microcosm-frame",
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
    # The materializer identity is the anti-mixing key (PR #557): a producer
    # that cannot state which target-frame semantics built its vectors must
    # not share the cache. Presence is required; an explicit None is a valid
    # declaration for non-release producers (the scorers) and is
    # identity-distinct from every release digest.
    if "target_frame_materializer_identity_sha256" not in context:
        raise ValueError(
            "Reform-vector cache context must declare "
            "target_frame_materializer_identity_sha256 (explicit None for "
            "non-release producers); silently omitting it would let vectors "
            "from different target-frame semantics mix (PR #557 review)."
        )
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
        # #217/#557: build commit remains intentionally excluded, while the
        # target-frame materializer digest binds seed, registry, staged-frame
        # semantics, and support selection to the absolute reform vector.
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


def _refuse_certified_release_dir_reuse(release_dir: Path) -> None:
    """Fail loud when --out/--release-id points at a certified release.

    microcosm#568 round 3: a failed retry into a directory that already
    carries a certified release would write failed-attempt weight evidence
    beside the prior run's manifest and H5 — mixing attempts the manifest
    knows nothing about. Release ids are immutable once certified; reruns
    pick a new id (every launcher stamps a fresh UTC timestamp) or remove
    the directory deliberately.
    """

    manifest_path = Path(release_dir) / "release_manifest.json"
    if manifest_path.exists():
        raise RuntimeError(
            f"Release directory {release_dir} already carries a certified "
            "release (release_manifest.json present). Choose a new "
            "--release-id or deliberately remove the stale directory before "
            "rerunning."
        )


def _write_final_household_weight_evidence(
    release_dir: Path,
    export_frame: Frame,
    *,
    identity: Mapping[str, object],
) -> dict[str, object]:
    """Atomically persist the final release-grain household weight vector.

    Written on the gate-failure path only (microcosm#568 review): a failed
    pre-export run minted no H5, so this evidence pair — the weight vector
    plus the ORDERED household ids it aligns to, bound to the target-frame
    identity — is the only way to reattach weights to records for
    record-level diagnosis. Green runs never write it: the certified H5
    carries the weights itself.
    """

    weights = export_frame.weights_for("household")
    if weights.kind is not WeightKind.CALIBRATED:
        raise ValueError(
            "Final household weight evidence requires calibrated weights, got "
            f"{weights.kind.value!r}."
        )
    values = np.asarray(weights.values, dtype=np.float64)
    if values.ndim != 1 or len(values) != export_frame.n("household"):
        raise ValueError(
            "Final household weights must align one-for-one with exported households."
        )

    household_ids = np.asarray(
        export_frame.table("household")["household_id"].to_numpy(),
        dtype=np.int64,
    )
    if household_ids.shape != values.shape:
        raise ValueError(
            "Final household weight evidence ids must align one-for-one with "
            "the weight vector."
        )

    evidence_dir = Path(release_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    values_path = evidence_dir / FINAL_HOUSEHOLD_WEIGHTS_FILENAME
    ids_path = evidence_dir / FINAL_HOUSEHOLD_WEIGHT_IDS_FILENAME
    metadata_path = evidence_dir / FINAL_HOUSEHOLD_WEIGHTS_METADATA_FILENAME
    temporary_values = values_path.with_name(f".{values_path.name}.tmp")
    temporary_ids = ids_path.with_name(f".{ids_path.name}.tmp")
    temporary_metadata = metadata_path.with_name(f".{metadata_path.name}.tmp")
    temporary_values.unlink(missing_ok=True)
    temporary_ids.unlink(missing_ok=True)
    temporary_metadata.unlink(missing_ok=True)
    try:
        with temporary_values.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
        with temporary_ids.open("wb") as stream:
            np.save(stream, household_ids, allow_pickle=False)
        metadata: dict[str, object] = {
            "artifact_kind": "populace_final_household_weight_evidence",
            "schema_version": FINAL_HOUSEHOLD_WEIGHTS_SCHEMA_VERSION,
            "measurement_phase": "release_final",
            "entity": "household",
            "weight_kind": weights.kind.value,
            "identity": dict(identity),
            "values": {
                "file": values_path.name,
                "dtype": "float64",
                "shape": [int(len(values))],
                "sha256": _sha256(temporary_values),
            },
            "household_ids": {
                "file": ids_path.name,
                "dtype": "int64",
                "shape": [int(len(household_ids))],
                "sha256": _sha256(temporary_ids),
                "ordering_sha256": hashlib.sha256(household_ids.tobytes()).hexdigest(),
            },
            "summary": {
                "n_households": int(len(values)),
                "household_weight_sum": float(values.sum()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "nonzero_count": int(np.count_nonzero(values)),
                "zero_count": int((values == 0.0).sum()),
            },
        }
        temporary_metadata.write_text(
            _strict_json_text(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_values, values_path)
        os.replace(temporary_ids, ids_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        temporary_values.unlink(missing_ok=True)
        temporary_ids.unlink(missing_ok=True)
        temporary_metadata.unlink(missing_ok=True)
    return metadata


def _selection_mass_protection_specs(
    base_frame: Frame,
    columns: Sequence[str],
) -> tuple[TargetSpec, ...]:
    """Synthetic national targets pinning locked-source input masses (#445).

    A protect-swap (#434) can carry a restored thin input's carriers into the
    frozen selection, but nothing stops the refit from crushing their weights
    when the concept is untargeted: Build M attempt 15 exported $6.1M of the
    ASEC source's $148.97M ``keogh_distributions``. Each protected column
    becomes an ordinary calibration target whose value is the base pool's own
    locked-source mass at base weights — measured here at build time — so the
    solve must preserve the mass the swap put in-selection. The target rides
    the standard ``policyengine_variable`` materializer (the column must be an
    engine variable) and its provenance rides the compilation payload. The
    protection lifts when the concept gains a real Ledger fact (#445).
    """
    specs: list[TargetSpec] = []
    for column in columns:
        owner = None
        for entity in base_frame.entities:
            if column in base_frame.table(entity).columns:
                owner = entity
                break
        if owner is None:
            raise RuntimeError(
                "Selection-mass protection column "
                f"{column!r} is absent from every entity table in the base "
                "pool."
            )
        values = np.asarray(base_frame.table(owner)[column], dtype=np.float64)
        weights = np.asarray(base_frame.resolve_weights(owner).values, dtype=np.float64)
        carriers = int((values != 0.0).sum())
        if carriers == 0:
            raise RuntimeError(
                f"Selection-mass protection column {column!r} has no nonzero "
                "carriers in the base pool; protecting it is a configuration "
                "error."
            )
        mass = float(values @ weights)
        if mass == 0.0:
            raise RuntimeError(
                f"Selection-mass protection column {column!r} nets to zero "
                "at base weights (signed cancellation); a sum-mode mass "
                "protection cannot pin it."
            )
        specs.append(
            TargetSpec(
                name=f"selection_mass_protection.{column}",
                entity="household",
                value=mass,
                measure=f"selection_mass_protection.{column}",
                source=(
                    "Locked-source mass measured on the base pool at base "
                    "weights (PolicyEngine/microcosm#445; #434 protect-swap)"
                ),
                signed=mass < 0,
                metadata={
                    "materializer": "policyengine_variable",
                    "measure_mode": "sum",
                    "base_variable": column,
                    "target_role": "selection_mass_protection",
                    "protected_column": column,
                    "protected_entity": owner,
                    "base_pool_carriers": str(carriers),
                    "issue": "PolicyEngine/microcosm#445",
                },
            )
        )
    return tuple(specs)


def _target_frame_checkpoint_identity(
    *,
    base_dataset_sha256: str,
    policyengine_us_version: str,
    seed: int,
    target_period: int,
    target_registry_version: str,
    weeks_unemployed_source_sha256: str,
    congressional_district_vintage_crosswalk_sha256: object,
    ssi_take_up_assignment_sha256: str,
    selection_identities_sha256: str | None,
    selection_mass_protections: tuple[str, ...] = (),
    ssi_take_up_prior_weight_basis_sha256: object = None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION,
        "materializer_version": TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION,
        "kind": "us_fiscal_refresh_target_frame",
        "country": "us",
        "base_dataset_sha256": str(base_dataset_sha256),
        "weeks_unemployed_source_sha256": str(weeks_unemployed_source_sha256),
        "policyengine_us_version": str(policyengine_us_version),
        "seed": int(seed),
        "target_period": int(target_period),
        "target_registry_version": str(target_registry_version),
        # The frozen SSI take-up decisions (flags + priors + basis
        # provenance) feed the materialized ssi target columns; a retry
        # under a different prior basis must invalidate the checkpoint
        # (microcosm#507/#508). Always present, so every pre-#507 checkpoint
        # — built on the collapsed Build-N-class flags — also misses once.
        "ssi_take_up_assignment_sha256": str(ssi_take_up_assignment_sha256),
        # The frozen-support selection prunes the base pool before assignment
        # and materialization. Its identity-set digest is independent of the
        # positional SSI assignment digest: different same-length supports can
        # carry identical flag bytes (microcosm#507).
        "selection_identities_sha256": (
            None
            if selection_identities_sha256 is None
            else str(selection_identities_sha256)
        ),
        "congressional_district_vintage_crosswalk_sha256": (
            None
            if congressional_district_vintage_crosswalk_sha256 is None
            else str(congressional_district_vintage_crosswalk_sha256)
        ),
        # The SSI prior-weight basis (#524) changes the take-up flags the
        # materialized target frame is built on, but arrives via a flag the
        # base hash cannot see: O attempt 3 warm-hit attempt 2's checkpoint
        # and solved on the other basis's rows (microcosm#543, instance 2).
        # Unconditional None-able key — v8 starts a fresh checkpoint world,
        # so no legacy-identity preservation applies.
        "ssi_take_up_prior_weight_basis_sha256": (
            None
            if ssi_take_up_prior_weight_basis_sha256 is None
            else str(ssi_take_up_prior_weight_basis_sha256)
        ),
    }
    if selection_mass_protections:
        # Key present only when protections are configured, so unprotected
        # runs (the dense arm, every pre-#445 sparse run) keep their legacy
        # digests and warm checkpoints. A protected run must MISS a
        # column-less checkpoint: _compile_materialized_target_registry
        # silently drops specs whose measures are absent from the
        # materialized household table.
        identity["selection_mass_protections"] = sorted(
            str(column) for column in selection_mass_protections
        )
    return identity


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

    from microcosm.frame import nullable_boolean_values_and_mask

    dtype = series.dtype
    group.attrs["pandas_dtype"] = str(dtype)
    if pd.api.types.is_bool_dtype(dtype):
        group.attrs["storage_kind"] = "bool"
        if isinstance(dtype, pd.BooleanDtype):
            values, null_mask = nullable_boolean_values_and_mask(series)
            group.attrs["nullable"] = True
            group.attrs["has_null_mask"] = bool(null_mask.any())
            if null_mask.any():
                group.create_dataset(
                    "null_mask",
                    data=null_mask.astype(np.uint8, copy=False),
                    compression="gzip",
                )
        else:
            values = series.to_numpy(dtype=np.bool_, copy=False)
            group.attrs["nullable"] = False
            group.attrs["has_null_mask"] = False
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
        values = np.asarray(dataset[()])
        if values.ndim != 1 or values.dtype != np.dtype(np.bool_):
            raise RuntimeError(
                "Target-frame checkpoint boolean values must be a "
                "one-dimensional bool array."
            )
        nullable = group.attrs.get("nullable")
        has_null_mask = group.attrs.get("has_null_mask")
        if not isinstance(nullable, (bool, np.bool_)) or not isinstance(
            has_null_mask, (bool, np.bool_)
        ):
            raise RuntimeError(
                "Target-frame checkpoint boolean metadata must declare "
                "nullable and has_null_mask booleans."
            )
        nullable = bool(nullable)
        has_null_mask = bool(has_null_mask)
        if not nullable and has_null_mask:
            raise RuntimeError(
                "Native target-frame checkpoint boolean cannot carry a null mask."
            )
        if has_null_mask:
            if "null_mask" not in group:
                raise RuntimeError(
                    "Nullable target-frame checkpoint boolean is missing its "
                    "declared null mask."
                )
            null_mask = np.asarray(group["null_mask"])
            if (
                null_mask.ndim != 1
                or null_mask.dtype != np.dtype(np.uint8)
                or len(null_mask) != len(values)
                or ((null_mask != 0) & (null_mask != 1)).any()
                or not null_mask.any()
            ):
                raise RuntimeError(
                    "Target-frame checkpoint nullable boolean null mask must "
                    "be an aligned, nonempty uint8 0/1 array."
                )
            mask = null_mask.astype(np.bool_, copy=False)
            if values[mask].any():
                raise RuntimeError(
                    "Target-frame checkpoint nullable boolean null mask covers "
                    "noncanonical true value bits."
                )
        else:
            if "null_mask" in group:
                raise RuntimeError(
                    "Target-frame checkpoint boolean has an unexpected null mask."
                )
            mask = np.zeros(len(values), dtype=np.bool_)
        if nullable:
            return pd.arrays.BooleanArray(values, mask, copy=False)
        return values
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

    Reconstructs the certified informed-L0/warm-start selection (microcosm#328) as
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
    *,
    authenticated_pool_h5: AuthenticatedPoolH5 | None = None,
) -> None:
    if crosswalk_metadata is None:
        return
    expected_sha256 = str(crosswalk_metadata.get("sha256") or "")
    if authenticated_pool_h5 is not None:
        authenticated_pool_h5.verified_digest(
            consumer="congressional-district support preflight before H5 read"
        )
    try:
        support_provenance = _read_cd_vintage_support_provenance(h5_path)
    except Exception:
        if authenticated_pool_h5 is not None:
            authenticated_pool_h5.verified_digest(
                consumer="congressional-district support preflight failed H5 read"
            )
        raise
    if authenticated_pool_h5 is not None:
        authenticated_pool_h5.verified_digest(
            consumer="congressional-district support preflight after H5 read"
        )
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
        with pd.HDFStore(h5_path, mode="r") as store:
            root_attributes = store.get_node("/")._v_attrs
            crosswalk_sha256 = _h5_attr_text(
                root_attributes,
                CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
            )
            target_vintage = _h5_attr_text(
                root_attributes,
                CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
            )
            cd_lookup = _hdf_store_frame_column_status(
                store,
                table="household",
                column="congressional_district_geoid",
            )
    except ImportError as exc:
        raise RuntimeError(
            "Reading congressional-district support provenance requires PyTables. "
            "Run the fiscal refresh builder with the US extra, for example "
            "`uv run --python 3.13 --package microcosm-build --extra us --group "
            "dev python tools/build_us_fiscal_refresh_release.py ...`. This "
            "preflight is intentionally before calibration or donor imputation."
        ) from exc
    return {
        CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: crosswalk_sha256,
        CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: target_vintage,
        "household_congressional_district_geoid": cd_lookup,
    }


def _hdf_frame_column_status(
    h5_path: Path,
    *,
    table: str,
    column: str,
) -> dict[str, object]:
    with pd.HDFStore(h5_path, mode="r") as store:
        return _hdf_store_frame_column_status(store, table=table, column=column)


def _hdf_store_frame_column_status(
    store: pd.HDFStore,
    *,
    table: str,
    column: str,
) -> dict[str, object]:
    try:
        frame_table = read_frame_table(store, table)
    except KeyError:
        return {"exists": False, "table": table, "column": column}
    if column not in frame_table.columns:
        return {"exists": False, "table": table, "column": column}
    values = frame_table[column].to_numpy(copy=False)
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


def _h5_attr_text(attrs: object, key: str) -> str | None:
    try:
        if isinstance(attrs, Mapping):
            value = attrs.get(key)
        else:
            value = attrs[key]  # type: ignore[index]
    except KeyError:
        return None
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
            f"{path} is not a Microcosm calibration_diagnostics.json file: "
            "expected a JSON object."
        )
    return payload


def _load_verified_incumbent_diagnostics_payload(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, object], str]:
    """Load one incumbent from the exact bytes authenticated for scoring."""

    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError(
            "Incumbent diagnostics SHA-256 mismatch for "
            f"{path}: got {observed_sha256}, expected {expected_sha256}."
        )
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} is not a Microcosm calibration_diagnostics.json file: "
            "expected a JSON object."
        )
    return payload, observed_sha256


def _diagnostics_by_target_name(
    payload: Mapping[str, object],
    *,
    path: Path | None = None,
) -> dict[str, Mapping[str, object]]:
    targets = payload.get("targets")
    if not isinstance(targets, list):
        label = str(path) if path is not None else "diagnostics payload"
        raise ValueError(
            f"{label} is not a Microcosm calibration_diagnostics.json file: "
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
            release_engine_simulation(batch_simulation)
            del batch_tax_unit, batch_simulation, batch_frame
        _collect_batch_garbage()
    _collect_family_garbage()
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
    semantics per microcosm #332.
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
            release_engine_simulation(batch_simulation)
            del batch_frame, batch_simulation
        _collect_batch_garbage()
    _collect_family_garbage()
    return eligibility


def _ssi_person_uncapped_amount(
    frame: Frame,
    *,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> np.ndarray:
    """December person-level potential federal SSI, batched like Medicaid.

    SSA's age-band recipient counts are December 2024 point-in-time stocks.
    ``uncapped_ssi > 0`` is the PolicyEngine-US 1.819.0 current-benefit
    candidate mask and does not depend on the take-up input being assigned.
    """

    period = f"{PERIOD}-12"

    def calculate(active_simulation) -> np.ndarray:
        values = np.asarray(
            active_simulation.calculate(
                "uncapped_ssi",
                period=period,
                map_to="person",
            ),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise RuntimeError(
                "SSI take-up materialization produced nonfinite uncapped_ssi values."
            )
        return values

    if simulation is not None:
        return calculate(simulation)

    from policyengine_us import Microsimulation

    person_ids = frame.table("person")["person_id"].to_numpy()
    uncapped = np.zeros(len(person_ids), dtype=np.float64)
    person_positions = pd.Series(
        np.arange(len(person_ids), dtype=np.int64), index=person_ids
    )
    n_households = frame.n("household")
    batches = tuple(
        _household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if len(batches) > 1:
        print(
            "Materializing December SSI candidate amounts in "
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
            batch_uncapped = calculate(batch_simulation)
            positions = person_positions.reindex(
                batch_frame.table("person")["person_id"].to_numpy()
            ).to_numpy()
            if np.isnan(positions).any():
                raise RuntimeError(
                    "SSI candidate batch produced person_id values not present "
                    "in the full person table."
                )
            uncapped[positions.astype(np.int64)] = batch_uncapped
            release_engine_simulation(batch_simulation)
            del batch_frame, batch_simulation
        _collect_batch_garbage()
    _collect_family_garbage()
    return uncapped


def _with_medicaid_take_up_outputs(
    frame: Frame,
    target_specs: tuple,
    *,
    seed: int,
    substitutions: Sequence[dict[str, object]] = (),
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> tuple[Frame, dict[str, object]]:
    """Assign Medicaid take-up (microcosm #331) and return frame + diagnostics.

    ``substitutions`` are the reviewed CMS enrollment substitution records
    (microcosm#386) produced by :func:`apply_us_medicaid_enrollment_substitutions`;
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


def _medicaid_diagnostics_for_existing_output(
    frame: Frame,
    target_specs: tuple,
    *,
    seed: int,
    substitutions: Sequence[dict[str, object]] = (),
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> dict[str, object]:
    """Diagnose persisted Medicaid flags on actual release weights."""

    target_table = _medicaid_source_target_table(target_specs)
    if target_table.empty:
        raise RuntimeError(
            "Final Medicaid take-up diagnostics require CMS state targets."
        )
    eligibility = _medicaid_person_eligibility(
        frame,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )
    assigned = us_medicaid_source_person_table(
        frame,
        is_medicaid_eligible=eligibility,
        state_fips=_person_state_fips(frame),
        seed=seed,
    )
    person = frame.table("person")
    if US_MEDICAID_TAKE_UP_VARIABLE not in person:
        raise RuntimeError(
            f"Final release is missing person.{US_MEDICAID_TAKE_UP_VARIABLE}."
        )
    takes_up = person[US_MEDICAID_TAKE_UP_VARIABLE]
    if not pd.api.types.is_bool_dtype(takes_up.dtype) or takes_up.isna().any():
        raise RuntimeError("Final Medicaid take-up output must be complete boolean.")
    assigned[US_MEDICAID_TAKE_UP_VARIABLE] = takes_up.to_numpy(dtype=bool)
    return us_medicaid_take_up_diagnostics(
        assigned,
        target_table,
        substitutions=substitutions,
        weights_basis="final_release_weights",
    )


def _snap_state_target_table(target_specs: tuple) -> pd.DataFrame:
    """FNS state household caseload counts as the take-up calibration table.

    Mirrors :func:`_medicaid_source_target_table` for the ``snap_households``
    target role: FY2024 state average-monthly household facts
    (``usda_snap.fy2024.state_average_monthly_households.*``), fiscal-year
    average-monthly stock semantics — the same rows the ``snap_households``
    weight-calibration targets compile from, so the take-up seed and the
    calibration objective agree.
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
        if spec.metadata.get("target_role") != US_SNAP_HOUSEHOLDS_TARGET_ROLE:
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
            "SNAP household caseload targets carry duplicate state rows for "
            f"state_fips {sorted(duplicated)}; the ledger feed must supply "
            "exactly one snap_households count per state."
        )
    return table


def _spm_unit_state_fips(frame: Frame) -> np.ndarray:
    """SPM-unit-aligned state FIPS text codes via the frame's linkage.

    ``Frame.broadcast`` only targets the person entity, so route household
    state through persons and collapse per SPM unit. Every person in an SPM
    unit shares its household's state; the fail-closed check keeps that
    invariant honest rather than assuming it.
    """
    person = frame.table("person")
    person_fips = _state_fips_text(
        frame.broadcast("state_fips", to="person").to_numpy()
    )
    per_unit = (
        pd.Series(person_fips, index=person["person_spm_unit_id"].to_numpy())
        .groupby(level=0)
        .agg(["first", "nunique"])
    )
    if per_unit["nunique"].gt(1).any():
        bad = per_unit.index[per_unit["nunique"].gt(1)].tolist()
        raise ValueError(
            f"SPM unit(s) span multiple state FIPS codes; invalid unit(s) {bad[:5]}."
        )
    spm_ids = frame.table("spm_unit")["spm_unit_id"].to_numpy()
    aligned = per_unit["first"].reindex(spm_ids)
    if aligned.isna().any():
        bad = list(aligned.index[aligned.isna()])[:5]
        raise ValueError(f"SPM unit(s) carry no person rows for state FIPS: {bad}.")
    return np.asarray(aligned.to_numpy())


def _snap_spm_unit_eligibility(
    frame: Frame,
    *,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> np.ndarray:
    """Engine-computed SPM-unit ``is_snap_eligible``, batched like Medicaid."""
    if simulation is not None:
        return _calculate_array(simulation, "is_snap_eligible", map_to="spm_unit") > 0
    from policyengine_us import Microsimulation

    spm_unit_ids = frame.table("spm_unit")["spm_unit_id"].to_numpy()
    eligibility = np.zeros(len(spm_unit_ids), dtype=bool)
    unit_positions = pd.Series(
        np.arange(len(spm_unit_ids), dtype=np.int64), index=spm_unit_ids
    )
    n_households = frame.n("household")
    batches = tuple(
        _household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if len(batches) > 1:
        print(
            "Materializing SNAP eligibility in "
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
                    batch_simulation, "is_snap_eligible", map_to="spm_unit"
                )
                > 0
            )
            positions = unit_positions.reindex(
                batch_frame.table("spm_unit")["spm_unit_id"].to_numpy()
            ).to_numpy()
            if np.isnan(positions).any():
                raise RuntimeError(
                    "SNAP eligibility batch produced spm_unit_id values not "
                    "present in the full spm_unit table."
                )
            eligibility[positions.astype(np.int64)] = batch_eligible
            release_engine_simulation(batch_simulation)
            del batch_frame, batch_simulation
        _collect_batch_garbage()
    _collect_family_garbage()
    return eligibility


def _with_snap_state_take_up_outputs(
    frame: Frame,
    target_specs: tuple,
    *,
    seed: int,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> tuple[Frame, dict[str, object]]:
    """Assign state-calibrated SNAP take-up (microcosm #372): frame + diagnostics."""
    target_table = _snap_state_target_table(target_specs)
    if target_table.empty:
        raise RuntimeError(
            "SNAP state take-up requires FNS state household caseload targets "
            "(snap_households); none were compiled from the ledger facts."
        )
    eligibility = _snap_spm_unit_eligibility(
        frame,
        simulation=simulation,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )
    return with_us_snap_state_take_up(
        frame,
        is_snap_eligible=eligibility,
        state_fips=_spm_unit_state_fips(frame),
        state_targets=target_table,
        seed=seed,
    )


_FORMULA_OWNED_GATE_ADAPTER: PolicyEngineUSVariableMetadataIndex | None = None


def _formula_owned_gate_adapter() -> PolicyEngineUSVariableMetadataIndex:
    """One import-free source metadata index for every gate call.

    The export gate needs variable ownership only. Parsing the installed
    variable declarations avoids importing ``policyengine_us`` (which creates
    a module-global tax-benefit system) or constructing a second adapter system.
    """
    global _FORMULA_OWNED_GATE_ADAPTER
    if _FORMULA_OWNED_GATE_ADAPTER is None:
        _FORMULA_OWNED_GATE_ADAPTER = PolicyEngineUSVariableMetadataIndex()
    return _FORMULA_OWNED_GATE_ADAPTER


def _assert_no_formula_owned_columns(frame: Frame) -> None:
    adapter = _formula_owned_gate_adapter()
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
        self._reform_system = None

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
                if self._reform is None:
                    simulation = self._microsimulation_cls(dataset=dataset)
                else:
                    # microcosm#456: one reform system per scored reform, not
                    # one per batch (each engine build permanently leaks
                    # ~5,600 sys.modules entries).
                    if self._reform_system is None:
                        self._reform_system = (
                            self._microsimulation_cls.default_tax_benefit_system(
                                reform=self._reform
                            )
                        )
                    simulation = self._microsimulation_cls(
                        tax_benefit_system=self._reform_system,
                        dataset=dataset,
                        reform=self._reform,
                    )
                total += float(simulation.calculate(measure, period).sum())
                release_engine_simulation(simulation)
                del simulation, dataset, batch_frame
            _collect_batch_garbage()
        _collect_family_garbage()
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
    # microcosm#456: a reform simulation cannot reuse the engine's shared
    # class-level system instance, so letting each batch construct its own
    # ``Microsimulation(reform=...)`` rebuilt the full tax-benefit system per
    # batch — measured at ~0.45 GB and ~5,600 permanently-leaked sys.modules
    # entries per build (policyengine-core registers every variable module
    # under a unique per-system name and never evicts). Build the reform
    # system once per target family instead — the same
    # ``default_tax_benefit_system(reform=...)`` construction the engine ran
    # per batch — and hand it to every batch simulation explicitly.
    reform_system = microsimulation_cls.default_tax_benefit_system(reform=reform)
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
            reformed = microsimulation_cls(
                tax_benefit_system=reform_system,
                dataset=reformed_dataset,
                reform=reform,
            )
            batch_income_tax = _collapse_tax_unit(
                _calculate_array(reformed, "income_tax"),
                batch_tax_unit_positions,
                batch_frame.n("household"),
            )
            reform_income_tax[household_positions] = batch_income_tax
            release_engine_simulation(reformed)
            del batch_income_tax, reformed, reformed_dataset, batch_frame
        _collect_batch_garbage()
    del reform_system
    _collect_family_garbage()
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
                and str(key) not in IDENTITY_LEDGER_FILTER_METADATA_KEYS
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
    person_age_for_bands: np.ndarray | None = None
    for spec in direct_target_specs:
        base_variables = _base_variables_from_metadata(spec.metadata)
        mode = spec.metadata.get("measure_mode", "sum")
        age_lower = spec.metadata.get("age_lower_bound")
        age_upper = spec.metadata.get("age_upper_bound")
        if age_lower is not None or age_upper is not None:
            # Age-banded person-variable targets (microcosm#470, the SSA SSI
            # recipients-by-age counts): mask the person-entity base variable
            # by the fact's age constraints BEFORE the household collapse —
            # the state/CD masks below act on household values and cannot
            # express person-age slices.
            if any(variable not in system.variables for variable in base_variables):
                continue
            for variable in base_variables:
                if _variable_entity(system, variable) != "person":
                    raise ValueError(
                        "Age-banded target "
                        f"{spec.name!r} requires person-entity base "
                        f"variables; {variable!r} is "
                        f"{_variable_entity(system, variable)!r}."
                    )
            if person_age_for_bands is None:
                person_age_for_bands = np.asarray(
                    _calculate_array(simulation, "age"), dtype=np.float64
                )
            person_values = np.zeros_like(person_age_for_bands)
            for variable in base_variables:
                person_values = person_values + np.asarray(
                    _calculate_array(simulation, variable), dtype=np.float64
                )
            if mode == "indicator_sum":
                person_values = (person_values > 0.0).astype(np.float64)
            band_lower = _as_bound(str(age_lower if age_lower is not None else "-inf"))
            band_upper = _as_bound(str(age_upper if age_upper is not None else "inf"))
            band_mask = (person_age_for_bands >= band_lower) & (
                person_age_for_bands < band_upper
            )
            hh[spec.measure] = _collapse_person(
                base_frame, person_values * band_mask.astype(np.float64)
            )
            continue
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
    # microcosm#456: the base simulation is pinned past its ``del`` by the
    # shared system instance's ``simulation`` backref — for the full pool that
    # is tens of GB held across the entire reform phase. Release it properly.
    release_engine_simulation(simulation)
    del simulation, dataset
    _collect_family_garbage()
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


US_REPORTED_COVERAGE_VINTAGE_GATE_RECEIPT = (
    "reported_coverage_vintage_gate_failure.json"
)


def _write_reported_coverage_vintage_gate_receipt(
    release_dir: Path, gate: GateResult
) -> Path:
    """Persist a RED reported-coverage vintage gate before the early raise.

    The base-frame gate raises before the collector, diagnostics, and
    manifests run, so without this receipt a failure would leave only
    optional staging telemetry and an exception (microcosm #720 review).
    The receipt never marks a directory certified: certification is keyed
    on ``release_manifest.json`` (#568), so a rerun under the same id is
    not blocked by it.
    """

    release_dir.mkdir(parents=True, exist_ok=True)
    path = release_dir / US_REPORTED_COVERAGE_VINTAGE_GATE_RECEIPT
    path.write_text(
        json.dumps(
            {
                "gate": gate.name,
                "passed": gate.passed,
                "failures": list(gate.failures),
                "details": dict(gate.details),
            },
            indent=1,
            allow_nan=False,
        )
    )
    return path


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
    name (microcosm #313). Unlike :func:`_input_mass_reference_gate`, whose
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


# Build H (microcosm#299): export-input-mass reviewed exclusions.
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
# Build M additions (same doctrine, measured on the Build M frame; the gate
# flagged all three on sparse attempt 10, 5482d38-20260715T114657Z):
#   rental_income          ref $432.870B band [$216.435B, $649.305B]
#                          The registry now identifies the concept: SOI ht2
#                          rental_royalty_income_amount is an ACTIVE national
#                          target at $95.95B@2024 (plus 51 state rows) — far
#                          BELOW the band's lower edge; a correctly calibrated
#                          column cannot pass -> excluded.
#   charitable_non_cash_donations ref $52.840B band [$26.420B, $79.260B]
#                          SOI Table 2.1 TY2023 noncash contributions =
#                          $116.417B (CBO-aged ~$126.57B@2024) — ABOVE the
#                          band's upper edge; cannot pass truthfully ->
#                          excluded (non_sch_d class: total pinned, split not).
#   partnership_self_employment_net_earnings ref $61.740B
#                          band [$30.870B, $92.610B]; EXPORT-side defect
#                          (misc/#393 class), remedy tracked in microcosm#432;
#                          exclusion lifts with the base rebuild.
#   farm_income            ref $62.387B  band [$31.194B, $93.581B]
#                          UNPINNED free dimension (attempt 12, 6584dfa, the
#                          run's only failing group): 353 one-signed pool
#                          carriers ($10.24B at base weights, identical
#                          pre/post-#435), 109 in the frozen selection
#                          ($4.10B); zero farm facts in the feed, zero farm
#                          specs compiled; exports wander J -25.6% ->
#                          att11 -43.2% -> att12 -62.4% on an identical
#                          pool. Excluded per microcosm#441; lifts with the
#                          SOI Table 1.4 Schedule F identification.
# Deliberately NOT excluded (parity checks stay live; the run adjudicates):
#   miscellaneous_income   ref $47.401B  band [$23.700B, $71.101B]; SOI net
#                          target ~$52.84B is IN-band -> genuine pass expected.
#   home_mortgage_interest ref $311.126B band [$155.563B, $466.689B]; the new
#                          itemizer-masked Table 2.1 target (~$186.3B aged)
#                          pins the itemizer share and pulls the full column
#                          down toward the band from $474-526B.
#   first_home_mortgage_interest follows home_mortgage_interest (second-home
#                          leg un-imputed / 0 per microcosm#38).
#   taxable_interest_income ref $320.159B band [$160.079B, $480.238B]; the
#                          microcosm#489 adjudication VINDICATED this
#                          reference: SOI Pub 1304 Table 1.4 puts taxable
#                          interest at $313.813B TY2023 (23in14ar.xls; a
#                          x2.349 realized explosion over TY2022's
#                          $133.597B that the CBO-AGI aging default missed),
#                          so the reference sits at 102% of the same-year
#                          official actual. The stale HT2/CD-lineage rows
#                          that demanded $134.6-149.1B now rebase onto the
#                          live Table 4.3 control (~$340.4B@2024, +6.3% vs
#                          the reference — comfortably in-band); a solve
#                          satisfying the corrected family passes this
#                          parity check truthfully. The #492 experiment
#                          arms' failures on this column were the WRONG
#                          side of the target self-contradiction winning
#                          (rational arm: ht2-all -4.3% but Table 4.3
#                          -58.1% and Table 2.1 -63.6%), not a reference
#                          defect. No exclusion, by adjudication.
US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS: dict[str, str] = {
    "rental_income": (
        "Identified by the ACTIVE registry: irs_soi ht2 "
        "rental_royalty_income_amount is a compiled national target at "
        "$95.95B@2024 (with 51 state rows), so the solve pins the rental "
        "concept to SOI while the live-default reference carries $432.87B "
        "on this column — ~4.5x the SOI level, an incidental artifact of "
        "the incumbent solve (nothing pinned rental before the Build H SOI "
        "identification). The +/-50% band's lower edge ($216.44B) sits far "
        "ABOVE the SOI-true level, so a correctly calibrated column cannot "
        "pass this parity check. Corroboration: partnerships' net rental "
        "real estate income is NEGATIVE (-$66.9B TY2022, SOI Partnership "
        "Returns bulletin Fig. K) — net rental concepts run an order of "
        "magnitude below the reference's gross-like mass. Build M sparse "
        "exports $90.70B, at the pinned level (attempt 10, 5482d38)."
    ),
    "charitable_non_cash_donations": (
        "Identified by SOI Table 2.1 TY2023 (the same vendored ledger bytes "
        "the active $230.46B@2024 total-charitable target compiles from, "
        "23in21id.xls): noncash ('other than cash') contributions = "
        "$116.417B TY2023, CBO-aged ~$126.57B@2024 — ABOVE the reference "
        "band's upper edge ($79.26B; ref $52.84B on this column). The "
        "registry constrains the charitable TOTAL; the cash/noncash split "
        "was never pinned, so the reference's split is an incidental "
        "artifact of the incumbent solve (the non_sch_d_capital_gains "
        "class: aggregate constrained, component split not). A correctly "
        "calibrated column cannot pass. Build M sparse exports $92.67B, "
        "between the reference and the SOI level (attempt 10, 5482d38)."
    ),
    "partnership_self_employment_net_earnings": (
        "EXPORT-side defect, excluded with the remedy tracked (the "
        "miscellaneous_income/#393 pattern, NOT a reference error): the "
        "base-m pool collapses this signed, sparse column (4,467 nonzero "
        "of 865k persons) to near-cancellation — positive leg +$47.71B "
        "vs negative leg -$48.06B, net -$0.35B at base weights, against "
        "base-j's +$12.21B and the PUF-direct-era reference's +$61.74B. "
        "The unpinned solve then amplifies unpredictably (Build J landed "
        "-37.6% in-band; Build M lands -$22.72B, -136.8%). Reachability "
        "is not the barrier (ratio-5 on the positive leg alone reaches "
        "$238B); the pool's leg cancellation is. Remedy = diagnose and "
        "fix the QRF chain for this column in the staged base builder and "
        "rebuild (single-chain, checkpointed), then identify the concept "
        "with an SOI Schedule SE target so the solve stops treating it as "
        "a free dimension (microcosm#432). This exclusion lifts with that "
        "rebuild."
    ),
    "farm_income": (
        "Unpinned free dimension, adjudicated on Build M sparse attempt 12 "
        "(6584dfa-20260716T105513Z; the run's ONLY failing gate group "
        "under the #437 batched report): the v8 Ledger feed contains zero "
        "farm facts and the compiled 2024 registry zero farm specs, so "
        "nothing pins this column and the live-default reference's "
        "$62.39B is an incidental artifact of the incumbent solve (the "
        "estate_income class). The pool carries 353 one-signed nonzero "
        "persons ($10.24B at base weights, byte-identical pre/post-#435 "
        "— this is NOT the farm_operations_income signed leaf), 109 of "
        "them in the frozen rmloss100+keogh-swap selection ($4.10B at "
        "base weights); reaching the band floor ($31.19B) would require "
        "stretching those 109 records 7.6x toward the incidental value. "
        "Unpinned exports wander exactly as microcosm#432 describes: "
        "Build J -25.6% in-band, attempt 11 -43.2% in-band, attempt 12 "
        "-62.4% ($23.45B) out — on an identical pool. Excluded per "
        "microcosm#441; the exclusion lifts when SOI Table 1.4 farm net "
        "income (Schedule F) is identified as a Ledger target."
    ),
    "miscellaneous_income": (
        "Source concept mismatch, established by microcosm#393's remedy "
        "experiments: the PUF pipeline maps miscellaneous_income = E01200, "
        "but E01200 is Form 4797 / 1040 line 14 (business-property gains/"
        "losses), not the SOI Table 1.4 line-21 concept the reference "
        "carries. The pool holds ~4.6x SOI's loss-return prevalence, so at "
        "design weights misc is net -$8.15B and the ratio-5 ceiling caps "
        "the dense solve at ~$21.3-22.8B against the $23.70B band floor - "
        "mathematically unreachable (loss-leg multipliers inert through "
        "10x; income-leg plateaus at -52% through 20x). The sparse arm "
        "holds the band via selection and stays gated. Remedy = remap "
        "E01200 to other_net_gain and rebuild the processed PUF "
        "(microcosm#393 final determination); this exclusion lifts with "
        "that rebuild."
    ),
    "short_term_capital_gains": (
        "Untargeted signed dimension measured against an incidental reference "
        "(the microcosm#432/#433 rental_income class, called in advance by the "
        "release-gate preflight: 'confirm the calibration surface targets this "
        "column — an untargeted one fails the export-mass parity gate'). The "
        "compiled register carries NO short-term-specific target (verified on "
        "Build P3 dense diagnostics: zero short_term rows among 5,695 compiled "
        "targets), so nothing pins the signed ST dimension; the combined ST+LT "
        "surface is pinned instead and lands (net_capital_gains_amount -2.85% on "
        "the same run). The reference h5 carries $118.072B of signed ST mass — "
        "an artifact of the incumbent solve, not an identified level — while the "
        "Build P3 dense export delivers $58.465B, 0.97% past the +/-50% band "
        "floor ($59.036B). This exclusion RE-ADJUDICATES when SOI Publication "
        "1304 Table 1.4A (the Schedule-D sales-of-capital-assets table) short- "
        "term gain/loss legs land as Ledger targets — a new target pins the "
        "dimension but does not itself validate the incumbent parity reference, "
        "so the entry is re-decided on that build's receipts, the farm-entry "
        "pattern."
    ),
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
        "moving revenue (PolicyEngine/microcosm#299 Build H)."
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
        "(PolicyEngine/microcosm#299 Build H)."
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
    gains as failures (microcosm #327). Passing a certified-release
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


#: QRF tail-concentration gate parameters (microcosm#462). top_k=100 and the
#: 0.75 share threshold are calibrated to the incident: the Build M
#: non_sch_d_capital_gains column carried 89% of its weighted mass in its top
#: 100 records (a repeated $594,484 donor-ceiling value) across 2,295
#: carriers. Columns are checked only when sparse (nonzero on at most 5% of
#: their entity's records — the regime where a conditional QRF without a
#: participation margin tail-broadcasts) and wide enough (at least 500
#: weighted carriers) for a top-100 share to be evidence rather than
#: arithmetic.
US_QRF_TAIL_CONCENTRATION_TOP_K = 100
US_QRF_TAIL_CONCENTRATION_MAX_TOP_SHARE = 0.75
US_QRF_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS = 500
US_QRF_SPARSE_NONZERO_SHARE_MAX = 0.05


def _qrf_imputed_source_outputs() -> frozenset[str]:
    """Variables produced by a ``fit_weighted_qrf`` source-stage operation.

    Derived from the declarative stage manifest (``us/source_stages.json``)
    rather than a hand list, so a new QRF-imputed stage output is covered by
    the tail-concentration gate the day the manifest declares it.
    """
    return frozenset(
        output
        for stage in US_SOURCE_MANIFEST.stages
        if any(operation.kind == "fit_weighted_qrf" for operation in stage.operations)
        for output in stage.outputs
    )


def _qrf_tail_concentration_gate(
    export_frame: Frame,
    *,
    reviewed_exclusions: Mapping[str, str] | None = None,
) -> tuple[GateResult, dict[str, object]]:
    """Tail-concentration gate over the sparse QRF-imputed export columns.

    Runs :func:`microcosm.build.gates.tail_concentration_gate` on every
    QRF-imputed source-stage output the export persists that is sparse
    (nonzero share at most :data:`US_QRF_SPARSE_NONZERO_SHARE_MAX` of its
    entity's records), at the export's calibrated weights. Returns the gate
    result plus the surface metadata (which QRF outputs were checked, dense,
    absent, or non-numeric) for the release artifact.
    """
    qrf_outputs = sorted(_qrf_imputed_source_outputs())
    values: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    absent: list[str] = []
    dense: list[str] = []
    non_numeric: list[str] = []
    entity_weights: dict[str, np.ndarray] = {}
    for column in qrf_outputs:
        try:
            entity = export_frame.column_entity(column)
        except ValueError:
            absent.append(column)
            continue
        series = export_frame.table(entity)[column]
        if pd.api.types.is_bool_dtype(series):
            non_numeric.append(column)
            continue
        column_values = pd.to_numeric(series, errors="coerce").fillna(0.0)
        array = column_values.to_numpy(dtype=np.float64)
        nonzero_share = float((array != 0.0).mean()) if array.size else 0.0
        if nonzero_share > US_QRF_SPARSE_NONZERO_SHARE_MAX:
            dense.append(column)
            continue
        if entity not in entity_weights:
            entity_weights[entity] = np.asarray(
                export_frame.resolve_weights(entity).values, dtype=np.float64
            )
        values[column] = array
        weights[column] = entity_weights[entity]
    gate = tail_concentration_gate(
        values,
        weights,
        top_k=US_QRF_TAIL_CONCENTRATION_TOP_K,
        max_top_share=US_QRF_TAIL_CONCENTRATION_MAX_TOP_SHARE,
        min_nonzero_records=US_QRF_TAIL_CONCENTRATION_MIN_NONZERO_RECORDS,
        reviewed_exclusions=reviewed_exclusions,
    )
    surface: dict[str, object] = {
        "qrf_imputed_outputs": len(qrf_outputs),
        "checked_sparse_columns": sorted(values),
        "dense_columns": dense,
        "absent_columns": absent,
        "non_numeric_columns": non_numeric,
        "sparse_nonzero_share_max": US_QRF_SPARSE_NONZERO_SHARE_MAX,
    }
    return gate, surface


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


US_NON_SCH_D_CGD_REPAIR_REASON = (
    "The PUF E01100-lineage donor carries $24.31B across 4.67M weighted "
    "carriers (weighted mean $5,206) against the SOI Pub 1304 Table 1.4 "
    "TY2023 direct-route concept of $10.16B across 3.21M returns (mean "
    "$3,165) - 2.39x on mass, measured on the sha-pinned puf_2024.h5 donor "
    "via puf_tax_unit_donor_from_arrays. The eCPS-era pipeline produced "
    "$13.69B from the same lineage, so the current 2024-level uprating "
    "overstates a mean-reverting distribution series. Until the donor "
    "uprating is variable-specific (root issue filed on the #462 thread), "
    "the level is pinned to the ledger-fed Table 1.4 dollar fact (aging "
    "provenance in target_aged_to) - the "
    "same repair class as the Social Security component rescale above; the "
    "returns-count row is an indicator and is unaffected."
)


def _with_non_sch_d_cgd_value_repair(
    frame: Frame,
    target_specs: Iterable[object],
) -> tuple[Frame, dict[str, object]]:
    """Rescale non_sch_d_capital_gains to the aged SOI Table 1.4 dollar fact."""

    column = "non_sch_d_capital_gains"
    # TargetSpec names are the unsuffixed ledger source_record_ids (the
    # @period suffix exists only on diagnostic names); the national row is
    # the .all. filing-status segment with no state fips (PR #486 review
    # finding 1 — the suffixed matcher matched nothing on a real compile).
    matching = [
        spec
        for spec in target_specs
        if str(getattr(spec, "name", "")).startswith("irs_soi.")
        and ".table_1_4.all." in str(getattr(spec, "name", ""))
        and str(getattr(spec, "name", "")).endswith("capital_gain_distributions_amount")
        and not spec.metadata.get("state_fips")
    ]
    if len(matching) != 1:
        raise RuntimeError(
            "non_sch_d capital-gain-distributions repair requires exactly one "
            f"aged Table 1.4 dollar target; found {len(matching)}."
        )
    target = float(matching[0].value)
    if not math.isfinite(target) or target <= 0.0:
        raise RuntimeError(
            "non_sch_d capital-gain-distributions repair target must be "
            f"finite and positive; got {target!r}."
        )

    person = frame.table("person").copy()
    if column not in person.columns:
        raise RuntimeError(
            f"non_sch_d capital-gain-distributions repair requires person "
            f"column {column!r}."
        )
    person_weights = pd.Series(
        frame.resolve_weights("person").values, index=person.index
    )
    values = pd.to_numeric(person[column], errors="coerce").fillna(0.0)
    initial = float((values * person_weights).sum())
    if not math.isfinite(initial) or initial <= 0.0:
        raise RuntimeError(
            "non_sch_d capital-gain-distributions repair requires positive "
            f"finite support; got {initial!r}."
        )
    factor = target / initial
    applied = not math.isclose(factor, 1.0, rel_tol=1e-12, abs_tol=0.0)
    if applied:
        person[column] = values.to_numpy(dtype=np.float64) * factor

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
        "method": "rescale_non_sch_d_capital_gains_to_soi_table_1_4_fact",
        "applied": applied,
        "reason": US_NON_SCH_D_CGD_REPAIR_REASON,
        "target": target,
        "target_aged_to": matching[0].metadata.get("aged_to"),
        "initial_estimate": initial,
        "factor": factor,
        "repaired_estimate": initial * factor,
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


def _fiscal_target_loss_weights(
    registry: TargetRegistry,
    family_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
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
    weights = weights / weights.mean()
    if not family_multipliers:
        return weights
    families = np.asarray(
        [spec.family for spec in registry.specs],
        dtype=object,
    )
    for family, multiplier in sorted(family_multipliers.items()):
        mask = families == family
        if not mask.any():
            raise ValueError(
                f"--target-family-loss-multiplier family {family!r} matches "
                "no compiled target."
            )
        weights[mask] *= multiplier
    return weights / weights.mean()


def _fiscal_target_loss_basis(
    registry: TargetRegistry,
    target_loss_weights: np.ndarray,
    family_multipliers: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Content-address the complete loss basis without changing target surface."""

    weights = np.asarray(target_loss_weights, dtype=np.float64)
    if weights.shape != (len(registry.specs),):
        raise ValueError(
            "Fiscal target loss basis vector shape does not match the compiled "
            f"registry: got {weights.shape}, expected {(len(registry.specs),)}."
        )
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError(
            "Fiscal target loss basis weights must be finite and positive."
        )
    loss_vector = [
        {
            "row_name": _target_row_name(spec),
            "weight_hex": float(weight).hex(),
        }
        for spec, weight in zip(registry.specs, weights, strict=True)
    ]
    loss_vector_sha256 = hashlib.sha256(
        json.dumps(
            loss_vector,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "target_loss_weighting": US_FISCAL_TARGET_LOSS_WEIGHTING,
        "target_loss_family_multipliers": {
            family: float(multiplier)
            for family, multiplier in sorted((family_multipliers or {}).items())
        },
        "target_loss_cap": US_FISCAL_TARGET_LOSS_CAP,
        "n_targets": len(loss_vector),
        "loss_vector_sha256": loss_vector_sha256,
    }


def _incumbent_target_loss_basis(
    payload: Mapping[str, object],
) -> Mapping[str, object] | None:
    build = payload.get("build")
    if not isinstance(build, Mapping):
        return None
    basis = build.get("target_loss_basis")
    return basis if isinstance(basis, Mapping) else None


def _assert_incumbent_loss_basis_matches(
    configured: Mapping[str, object],
    incumbent: Mapping[str, object] | None,
) -> None:
    if incumbent is None:
        raise IncumbentLossBasisMismatchError(
            "pinned incumbent diagnostics have no build.target_loss_basis; "
            "rescore the incumbent on the frozen register before release."
        )
    if dict(incumbent) != dict(configured):
        raise IncumbentLossBasisMismatchError(
            "pinned incumbent target-loss basis differs from the configured "
            f"basis: configured={dict(configured)!r}, incumbent={dict(incumbent)!r}."
        )


def _ssi_take_up_band_targets_from_registry(target_specs: tuple) -> dict[str, float]:
    """SSA age-band recipient counts as compiled into the calibration registry.

    The SSI take-up stage derives its Bernoulli priors from the same
    ledger-fed band targets the weight solve enforces (role
    :data:`SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE`, microcosm#469/#470): one
    official measure, bound once, never hardcoded in the module. Bands are
    matched on the facts' first-class age constraints. Fails closed when the
    feed does not carry all three SSA age bands.
    """

    expected_bounds: dict[tuple[float, float], str] = {}
    for band in US_SSI_TAKE_UP_AGE_TARGETS:
        lower = float("-inf") if band.minimum_age is None else float(band.minimum_age)
        upper = (
            float("inf") if band.maximum_age is None else float(band.maximum_age) + 1.0
        )
        expected_bounds[(lower, upper)] = band.key
    band_targets: dict[str, float] = {}
    for spec in target_specs:
        if spec.metadata.get("target_role") != SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE:
            continue
        raw_bounds = (
            spec.metadata.get("age_lower_bound"),
            spec.metadata.get("age_upper_bound"),
        )
        key = None
        try:
            lower_value = float(raw_bounds[0])
            upper_value = float(raw_bounds[1])
        except (TypeError, ValueError):
            lower_value = upper_value = float("nan")
        else:
            # Ages are nonnegative, so an explicit "age >= 0" floor is the
            # same stratum as an unbounded lower edge — the real feed's
            # under-18 fact carries one (PR #477 review finding 1).
            if lower_value <= 0:
                lower_value = float("-inf")
            key = expected_bounds.get((lower_value, upper_value))
        if key is None:
            raise RuntimeError(
                "SSI take-up found a registry "
                f"{SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE!r} target with "
                f"unrecognized age bounds {raw_bounds!r}; expected one of "
                f"{sorted(expected_bounds)}."
            )
        if key in band_targets:
            raise RuntimeError(
                "SSI take-up found duplicate registry targets for SSA age "
                f"band {key!r}."
            )
        value = float(spec.value)
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(
                f"SSI take-up registry target for age band {key!r} must be "
                f"finite and positive; got {value!r}."
            )
        band_targets[key] = value
    missing = [
        band.key for band in US_SSI_TAKE_UP_AGE_TARGETS if band.key not in band_targets
    ]
    if missing:
        raise RuntimeError(
            "SSI take-up requires the ledger-fed SSA age-band recipient "
            f"targets (role {SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE!r}) in "
            f"the fiscal registry; missing band(s) {missing}. The consumer "
            "facts feed must carry the ssa ssi_federal_payment_recipients "
            "by_age rows (microcosm#470)."
        )
    return band_targets


def _load_ssi_take_up_prior_weight_basis(
    path: Path | None,
    *,
    targets: Mapping[str, float],
    expected_sha256: str | None,
) -> SSITakeUpPriorBasis | None:
    """Load and strictly validate --ssi-take-up-prior-weight-basis.

    Runs right after the target registry compiles (fail-fast: a bad artifact
    must die before the imputation stages, not hours in). The artifact is a
    prior attempt's final ``us_ssi_take_up.json``; its per-band
    ``candidate_capacity`` / ``reporter_candidate_floor`` were measured on
    the weights that attempt delivered, and the module-side validator
    enforces the release-final phase and full diagnostics-gate pass for the
    current schema, while schemas 2/3 contribute legacy capacity/floor seeds
    only; same-target-contract and enforced-band-feasibility rules apply to
    every accepted version (microcosm#507/#508). The caller must pin the
    artifact's sha256 in the launch command so the basis is an auditable
    receipt, never whatever bytes happen to sit at the path.
    """

    if path is None:
        if expected_sha256 is not None:
            raise RuntimeError(
                "--ssi-take-up-prior-weight-basis-sha256 requires "
                "--ssi-take-up-prior-weight-basis."
            )
        return None
    if expected_sha256 is None or not str(expected_sha256).strip():
        raise RuntimeError(
            "--ssi-take-up-prior-weight-basis requires the companion "
            "--ssi-take-up-prior-weight-basis-sha256 pin (read it from the "
            "producing release's manifest or the failed attempt's error)."
        )
    resolved = Path(path)
    if not resolved.is_file():
        raise RuntimeError(
            "--ssi-take-up-prior-weight-basis does not exist or is not a "
            f"file: {resolved}"
        )
    raw = resolved.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != str(expected_sha256).strip().lower():
        raise RuntimeError(
            f"--ssi-take-up-prior-weight-basis {resolved} has sha256 "
            f"{actual_sha256}, not the pinned "
            f"{str(expected_sha256).strip().lower()}."
        )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"--ssi-take-up-prior-weight-basis {resolved} is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"--ssi-take-up-prior-weight-basis {resolved} must contain a "
            "JSON object of us_ssi_take_up diagnostics."
        )
    try:
        return ssi_take_up_prior_basis_from_artifact(
            payload,
            targets=targets,
            source_sha256=actual_sha256,
        )
    except ValueError as error:
        raise RuntimeError(
            f"--ssi-take-up-prior-weight-basis {resolved} was rejected: {error}"
        ) from error


def _final_medicaid_diagnostics_or_quarantine(
    *,
    ssi_law_degraded: bool,
    degraded: bool,
    evaluate: Callable[[], dict],
) -> tuple[dict, list[str]]:
    """Final Medicaid diagnostics under the #547 degraded-mode contract.

    A Bernoulli-law violation upstream corrupts the frozen SSI decisions
    that pe-us Medicaid eligibility consumes (takes_up_ssi_if_eligible ->
    ssi -> is_ssi_recipient_for_medicaid -> medicaid_category ->
    is_medicaid_eligible), so on a law failure the evaluation is
    quarantined — recorded as not evaluated, never mis-measured.
    Delivery-only degradation still evaluates, but an evaluation crash in
    degraded mode records a line instead of masking the earlier failures
    and destroying the diagnostics artifact (microcosm#547). On the green
    path a crash raises exactly as before.
    """

    if ssi_law_degraded:
        return {}, [
            "Medicaid final diagnostics not evaluated: SSI decision "
            "integrity failed upstream (Bernoulli-law violation) and "
            "Medicaid eligibility consumes the frozen SSI decisions; "
            "quarantined instead of mis-measured (microcosm#547)."
        ]
    try:
        return evaluate(), []
    except Exception as error:
        if not degraded:
            raise
        return {}, [
            "Medicaid final diagnostics evaluation crashed in degraded "
            f"mode; recorded instead of masking earlier failures: {error}"
        ]


def _ssi_take_up_assignment_digest(
    frame: Frame,
    *,
    assignment_priors: Mapping[str, float],
    prior_basis: SSITakeUpPriorBasis,
) -> str:
    """Digest of the frozen SSI assignment for checkpoint/cache identity.

    Covers the persisted flag vector plus the priors and basis provenance
    that generated it, so a retry whose thresholds differ — the
    --ssi-take-up-prior-weight-basis path — can never reuse a target-frame
    checkpoint or materialized-column cache built on the previous flags
    (microcosm#507 sol review finding 2).
    """

    flags = frame.table("person")[US_SSI_TAKE_UP_OUTPUT_COLUMNS[0]]
    return hashlib.sha256(
        flags.to_numpy(dtype=np.uint8).tobytes()
        + json.dumps(
            {
                "assignment_priors": {
                    str(key): float(value) for key, value in assignment_priors.items()
                },
                "prior_weight_basis": dict(prior_basis.provenance()),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _enforce_ssi_take_up_delivery(
    diagnostics: Mapping[str, object],
    *,
    targets: Mapping[str, float],
    release_dir: Path,
    telemetry: StagingTelemetry | None,
    enforcement_fences: Mapping[str, str] | None = None,
) -> tuple[list[str], GateResult]:
    """Fail the release on an enforced-band delivery miss, via the batch.

    microcosm#507/#508: a miss beyond tolerance on release weights fails the
    build instead of shipping in the scorecard. ``enforcement_fences``
    (microcosm#566/#567) fences normally-enforced bands for the dense
    diagnostic arm, where delivered-weight recomputes have not landed
    the adult pair in band on either observed frame (P2's clean
    one-retry record; P3's refused delivered-basis chain) — fenced
    misses ship in the scorecard with their adjudication text instead
    of failing the release. The delivered-weight
    diagnostics are written before returning failures — that artifact IS the
    remedy: the retry passes it via ``--ssi-take-up-prior-weight-basis`` so
    the thresholds are recomputed exactly once from measured delivery, never
    iterated in-process (the microcosm#463-class loop stays deleted,
    microcosm#477). Failures return to the caller and join the #437 batched
    terminal gates rather than raising here: an early raise destroyed the
    failed run's calibration diagnostics and skipped every other gate group
    (microcosm#547 — the 2026-07-25 sparsecd retest left no target-surface
    evidence). Enforcement is unchanged: any returned failure still aborts
    the release at the terminal batch and certification manifests are never
    written.
    """

    delivery_gate = us_ssi_take_up_delivery_gate(
        diagnostics, targets=targets, enforcement_fences=enforcement_fences
    )
    if delivery_gate.passed:
        return [], delivery_gate
    # The gate failures are secured FIRST: the retry-artifact write and the
    # telemetry are reporting conveniences for an already-failed gate, and
    # neither may destroy the evidence chain by raising. Concretely: a
    # nonfinite delivered weight both fails the gate AND makes the strict
    # JSON writer (allow_nan=False) raise — the reporting crash would have
    # masked the gate failure and skipped the diagnostics artifact
    # (microcosm#547, confirm round 2 finding 1).
    failures = [
        f"SSI take-up delivery failed: {failure}" for failure in delivery_gate.failures
    ]
    try:
        failed_basis_path = write_us_ssi_take_up_diagnostics(
            diagnostics,
            release_dir / "us_ssi_take_up.json",
        )
        # The written artifact IS the retry's basis; its sha256 is the
        # required --ssi-take-up-prior-weight-basis-sha256 pin, so the
        # failure itself hands the operator both halves of the remedy (a
        # failed attempt never reaches the release manifest that would
        # otherwise carry the hash).
        failed_basis_sha256 = hashlib.sha256(failed_basis_path.read_bytes()).hexdigest()
        failures.append(
            "SSI take-up delivered-weight prior basis written to "
            f"{failed_basis_path} (sha256 {failed_basis_sha256}) for the "
            "--ssi-take-up-prior-weight-basis retry."
        )
    except Exception as error:
        failures.append(
            "SSI take-up delivered-weight prior basis could NOT be written "
            f"(the retry must recompute delivery itself): {error}"
        )
    try:
        if telemetry is not None:
            telemetry.stage(
                "ssi_take_up_delivery_gate",
                status="failed",
                message="SSI take-up enforced-band delivery gate failed.",
                failures=list(delivery_gate.failures),
                force_upload=True,
            )
    except Exception as error:
        failures.append(
            "SSI delivery-gate failure telemetry crashed; recorded instead "
            f"of masking the failure: {error}"
        )
    return failures, delivery_gate


def _ssi_assignment_priors_from_diagnostics(
    diagnostics: Mapping[str, object],
) -> dict[str, float]:
    """The per-band Bernoulli priors the gated assignment stage documented.

    The final release-weight measurement republishes these verbatim and
    re-verifies every frozen flag against the seeded law they define
    (microcosm#469) — recomputing priors from release weights would
    misdocument the one-shot assignment.
    """

    bands = diagnostics.get("age_bands")
    if not isinstance(bands, list) or not bands:
        raise RuntimeError(
            "SSI take-up stage diagnostics carry no age-band rows to read "
            "assignment priors from."
        )
    priors: dict[str, float] = {}
    for row in bands:
        if not isinstance(row, Mapping):
            raise RuntimeError("SSI take-up stage diagnostics band row is invalid.")
        key = str(row.get("age_band"))
        prior = float(row.get("assignment_prior", np.nan))
        if not np.isfinite(prior) or not 0.0 <= prior <= 1.0:
            raise RuntimeError(
                f"SSI take-up stage diagnostics band {key!r} carries an "
                f"invalid assignment prior {prior!r}."
            )
        priors[key] = prior
    return priors


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


def _target_metadata(target: object | None) -> Mapping[str, object]:
    if isinstance(target, Mapping):
        metadata = target.get("metadata")
    else:
        metadata = getattr(target, "metadata", None)
    return metadata if isinstance(metadata, Mapping) else {}


def _target_family(target: object | None) -> str:
    family = getattr(target, "family", None)
    if family is not None:
        return str(family)
    if isinstance(target, Mapping):
        registry = target.get("registry")
        if isinstance(registry, Mapping) and registry.get("family") is not None:
            return str(registry["family"])
    return ""


def _target_is_congressional_district(target: object | None) -> bool:
    return is_congressional_district_target(
        _target_row_name(target) if target is not None else "",
        _target_metadata(target),
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


def _critical_requirement_matches_target(
    requirement,
    *,
    row_name: str,
    target: object | None,
) -> bool:
    if _target_is_congressional_district(target):
        return False
    metadata = _target_metadata(target)
    return requirement.matches(
        name=row_name,
        family=_target_family(target),
        target_role=str(metadata.get("target_role") or ""),
    )


def _critical_target_specs_by_row_name(
    target_registry: TargetRegistry | None,
) -> dict[str, TargetSpec]:
    if target_registry is None:
        return {}
    return {_target_row_name(spec): spec for spec in target_registry.specs}


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
    reported_coverage_vintage_gate: GateResult | None = None,
    snap_discretionary_exemption_gate: GateResult | None = None,
    target_registry: TargetRegistry | None = None,
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
        reported_coverage_vintage_gate is not None
        and not reported_coverage_vintage_gate.passed
    ):
        failures.extend(
            f"Reported-coverage vintage signal failed: {failure}"
            for failure in reported_coverage_vintage_gate.failures
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
    diagnostic_targets = {
        **_diagnostic_targets_by_name(result),
        **_critical_target_specs_by_row_name(target_registry),
    }
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
            target_registry=target_registry,
        )
    )
    # microcosm#462: every national SOI Pub 1304 Table 1.4 dollar row is
    # within-tolerance-blocking, by name pattern rather than enumeration, so
    # rows the exact-name register above never listed (the +634.8%
    # capital-gain-distributions defect) cannot certify. No incumbent-
    # improvement escape: a national dollar row beyond broad fit never ships.
    soi_table_1_4_gate = target_fit_gate(
        tuple(
            diagnostic
            for diagnostic in (getattr(result, "diagnostics", ()) or ())
            if not _target_is_congressional_district(
                diagnostic_targets.get(str(getattr(diagnostic, "name", "")))
            )
        ),
        (US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT,),
        name="soi_table_1_4_national_dollar_fit",
    )
    if not soi_table_1_4_gate.passed:
        failures.extend(
            f"SOI Table 1.4 national dollar fit failed: {failure}"
            for failure in soi_table_1_4_gate.failures
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
    target_registry: TargetRegistry | None = None,
) -> list[str]:
    incumbent_diagnostics = incumbent_diagnostics or {}
    diagnostics_by_name = {
        getattr(diagnostic, "name", None): diagnostic
        for diagnostic in getattr(result, "diagnostics", ())
    }
    problem_targets = _diagnostic_targets_by_name(result)
    specs_by_name = _critical_target_specs_by_row_name(target_registry)
    failures: list[str] = []
    for requirement in US_CRITICAL_TARGET_FIT_REQUIREMENTS:
        matches = [
            diagnostic
            for row_name, diagnostic in diagnostics_by_name.items()
            if isinstance(row_name, str)
            and _critical_requirement_matches_target(
                requirement,
                row_name=row_name,
                target=specs_by_name.get(row_name, problem_targets.get(row_name)),
            )
        ]
        if not matches:
            missing_identity = (
                repr(requirement.names[0])
                if len(requirement.names) == 1
                else f"requirement {requirement.requirement_id!r}"
            )
            failures.append(
                f"Critical fiscal target {missing_identity} "
                f"({requirement.label}) is missing "
                "from calibration diagnostics."
            )
            continue
        for diagnostic in matches:
            row_name = str(getattr(diagnostic, "name", ""))
            relative_error = getattr(diagnostic, "relative_error", None)
            computed_relative_error = _diagnostic_relative_error(diagnostic, failures)
            if computed_relative_error is None:
                continue
            if not isinstance(relative_error, int | float):
                failures.append(
                    "Critical fiscal target "
                    f"{row_name!r} ({requirement.label}) has "
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
                    f"{row_name!r} ({requirement.label}) has "
                    f"stale relative_error {relative_error!r}; computed "
                    f"{computed_relative_error:.6g} from target and final_estimate."
                )
            max_abs = float(requirement.max_abs_relative_error)
            if abs(computed_relative_error) > max_abs:
                incumbent_relative_error = _incumbent_relative_error(
                    incumbent_diagnostics.get(row_name),
                    current_target=float(getattr(diagnostic, "target", 0.0)),
                )
                improved_over_incumbent = incumbent_relative_error is not None and abs(
                    computed_relative_error
                ) < abs(incumbent_relative_error)
                if (
                    requirement.allow_incumbent_improvement
                    and improved_over_incumbent
                    and abs(computed_relative_error)
                    <= US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR
                ):
                    continue
                failures.append(
                    "Critical fiscal target "
                    f"{row_name!r} ({requirement.label}) has "
                    f"relative_error={computed_relative_error:.6g}, exceeding "
                    f"{max_abs:.6g}; target="
                    f"{getattr(diagnostic, 'target', None)!r}, final_estimate="
                    f"{getattr(diagnostic, 'final_estimate', None)!r}"
                    + (
                        "."
                        if incumbent_relative_error is None
                        else (
                            "; incumbent_relative_error="
                            f"{incumbent_relative_error:.6g}; "
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


def _exact_k_frozen_register_fit_gate(
    result,
    incumbent_diagnostics: Mapping[str, Mapping[str, object]],
    *,
    target_registry: TargetRegistry,
    target_loss_weights: np.ndarray,
    configured_loss_basis: Mapping[str, object],
    incumbent_loss_basis: Mapping[str, object] | None,
) -> GateResult:
    """Require an exact-k candidate to beat the incumbent on one register.

    The comparison re-scores both artifacts from their per-target rows with
    the same capped, concept-budget-weighted loss used by the release solve.
    Target-surface fingerprint equality is checked before this gate is called;
    exact row-set, value, and loss-vector checks here make that binding
    executable rather than trusting summary scalars from the incumbent file.
    """

    diagnostics = tuple(getattr(result, "diagnostics", ()) or ())
    names = [str(getattr(row, "name", "")) for row in diagnostics]
    failures: list[str] = []
    try:
        _assert_incumbent_loss_basis_matches(
            configured_loss_basis,
            incumbent_loss_basis,
        )
    except IncumbentLossBasisMismatchError as error:
        failures.append(f"{type(error).__name__}: {error}")
    if not names or any(not name for name in names):
        failures.append(
            "Exact-k frozen-register comparison has no complete candidate target rows."
        )
    if len(names) != len(set(names)):
        failures.append(
            "Exact-k frozen-register comparison found duplicate candidate target names."
        )

    incumbent_names = set(incumbent_diagnostics)
    candidate_names = set(names)
    missing = sorted(candidate_names - incumbent_names)
    extra = sorted(incumbent_names - candidate_names)
    if missing or extra:
        failures.append(
            "Exact-k incumbent target rows do not equal the frozen candidate "
            f"register (missing={missing[:5]}, extra={extra[:5]})."
        )

    weights_by_name = {
        _target_row_name(spec): float(weight)
        for spec, weight in zip(
            target_registry.specs,
            np.asarray(target_loss_weights, dtype=np.float64),
            strict=True,
        )
    }
    missing_weights = sorted(candidate_names - set(weights_by_name))
    if missing_weights:
        failures.append(
            "Exact-k frozen-register comparison has no loss weight for target "
            f"row(s) {missing_weights[:5]}."
        )

    candidate_targets: list[float] = []
    candidate_estimates: list[float] = []
    incumbent_estimates: list[float] = []
    aligned_weights: list[float] = []
    for diagnostic in diagnostics:
        name = str(getattr(diagnostic, "name", ""))
        incumbent = incumbent_diagnostics.get(name)
        target = getattr(diagnostic, "target", None)
        estimate = getattr(diagnostic, "final_estimate", None)
        incumbent_target = None if incumbent is None else incumbent.get("target")
        incumbent_estimate = (
            None if incumbent is None else incumbent.get("final_estimate")
        )
        values = (target, estimate, incumbent_target, incumbent_estimate)
        if not all(
            isinstance(value, int | float) and math.isfinite(float(value))
            for value in values
        ):
            failures.append(
                "Exact-k frozen-register comparison has non-finite target or "
                f"estimate data for {name!r}."
            )
            continue
        if not math.isclose(
            float(target),
            float(incumbent_target),
            rel_tol=1e-9,
            abs_tol=1e-6,
        ):
            failures.append(
                "Exact-k incumbent target value changed for "
                f"{name!r}: candidate={target!r}, incumbent={incumbent_target!r}."
            )
            continue
        weight = weights_by_name.get(name)
        if weight is None:
            continue
        candidate_targets.append(float(target))
        candidate_estimates.append(float(estimate))
        incumbent_estimates.append(float(incumbent_estimate))
        aligned_weights.append(weight)

    candidate_loss: float | None = None
    incumbent_loss: float | None = None
    if not failures:
        target_vector = np.asarray(candidate_targets, dtype=np.float64)
        weights = np.asarray(aligned_weights, dtype=np.float64)
        candidate_loss = relative_error_loss(
            np.asarray(candidate_estimates, dtype=np.float64),
            target_vector,
            target_loss_weights=weights,
            target_loss_cap=US_FISCAL_TARGET_LOSS_CAP,
        )
        incumbent_loss = relative_error_loss(
            np.asarray(incumbent_estimates, dtype=np.float64),
            target_vector,
            target_loss_weights=weights,
            target_loss_cap=US_FISCAL_TARGET_LOSS_CAP,
        )
        reported_loss = float(getattr(result, "final_loss", math.nan))
        if not math.isclose(
            candidate_loss,
            reported_loss,
            rel_tol=1e-6,
            abs_tol=1e-12,
        ):
            failures.append(
                "Exact-k frozen-register candidate re-score does not match the "
                f"solver loss: rescored={candidate_loss}, reported={reported_loss}."
            )
        elif not candidate_loss < incumbent_loss:
            failures.append(
                "Exact-k candidate did not beat the incumbent on the frozen "
                f"target register: candidate_loss={candidate_loss}, "
                f"incumbent_loss={incumbent_loss}."
            )

    return GateResult(
        name="exact_k_frozen_register_fit",
        passed=not failures,
        failures=tuple(failures),
        details={
            "metric": "capped_concept_budget_weighted_mean_absolute_relative_error",
            "target_loss_cap": US_FISCAL_TARGET_LOSS_CAP,
            "configured_loss_basis": dict(configured_loss_basis),
            "incumbent_loss_basis": (
                dict(incumbent_loss_basis) if incumbent_loss_basis is not None else None
            ),
            "n_targets": len(names),
            "candidate_loss": candidate_loss,
            "incumbent_loss": incumbent_loss,
            "strict_improvement_required": True,
            "improvement": (
                None
                if candidate_loss is None or incumbent_loss is None
                else incumbent_loss - candidate_loss
            ),
        },
    )


def _incumbent_critical_target_payload(
    incumbent_diagnostics: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    payload: dict[str, dict[str, float]] = {}
    for name, row in incumbent_diagnostics.items():
        if not any(
            _critical_requirement_matches_target(
                requirement,
                row_name=name,
                target=row,
            )
            for requirement in US_CRITICAL_TARGET_FIT_REQUIREMENTS
        ):
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
    target_registry: TargetRegistry | None = None,
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
        target_registry=target_registry,
    )
    if failures:
        raise RuntimeError("Release gates failed: " + "; ".join(failures))


def _write_release_calibration_diagnostics(
    *,
    result,
    release_dir: Path,
    registry: TargetRegistry,
    base_dataset_sha256: str,
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
    reported_coverage_vintage_gate: GateResult | None = None,
    snap_discretionary_exemption_gate: GateResult | None = None,
    gate_failures: Iterable[str],
    timing: Mapping[str, object] | None = None,
    warm_start_calibration: Mapping[str, object] | None = None,
    selection_source: Mapping[str, object] | None = None,
    default_dataset: Mapping[str, object] | None = None,
    incumbent_diagnostics_path: Path | None = None,
    incumbent_diagnostics_sha256: str | None = None,
    incumbent_diagnostics: Mapping[str, Mapping[str, object]] | None = None,
    degenerate_input_gate: GateResult | None = None,
    ecps_parity_gate: GateResult | None = None,
    validation_input_coverage_gate: GateResult | None = None,
    target_loss_family_multipliers: Mapping[str, float] | None = None,
    target_loss_basis: Mapping[str, object] | None = None,
    exact_k_ladder: Mapping[str, object] | None = None,
) -> None:
    """Write calibration diagnostics even when hard release gates fail."""
    failures = list(gate_failures)
    incumbent_rows = incumbent_diagnostics or {}
    incumbent_payload = (
        {
            "path": str(incumbent_diagnostics_path),
            "sha256": (
                incumbent_diagnostics_sha256
                if incumbent_diagnostics_sha256 is not None
                else _sha256(incumbent_diagnostics_path)
            ),
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
            "base_dataset_sha256": base_dataset_sha256,
            "target_compilation": compilation,
            "target_loss_weighting": US_FISCAL_TARGET_LOSS_WEIGHTING,
            "target_loss_family_multipliers": (
                dict(target_loss_family_multipliers)
                if target_loss_family_multipliers
                else None
            ),
            "target_loss_cap": US_FISCAL_TARGET_LOSS_CAP,
            **(
                {"target_loss_basis": dict(target_loss_basis)}
                if target_loss_basis is not None
                else {}
            ),
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
            "reported_coverage_vintage_signal": (
                {
                    "passed": reported_coverage_vintage_gate.passed,
                    "failures": list(reported_coverage_vintage_gate.failures),
                    "details": dict(reported_coverage_vintage_gate.details),
                }
                if reported_coverage_vintage_gate is not None
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
            **(
                {"exact_k_ladder": dict(exact_k_ladder)}
                if exact_k_ladder is not None
                else {}
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
    """Emit reform_validation.json: microcosm budget effects vs JCT scores.

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
    ssi_take_up_delivery_gate_result: GateResult | None = None,
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
    reported_coverage_vintage_gate: GateResult | None = None,
    snap_discretionary_exemption_gate: GateResult | None = None,
    timing: Mapping[str, object] | None = None,
    warm_start_calibration: Mapping[str, object] | None = None,
    selection_source: Mapping[str, object] | None = None,
    default_dataset: Mapping[str, object] | None = None,
    medicaid_enrollment_substitutions: Sequence[Mapping[str, object]] = (),
    staging: Mapping[str, object] | None = None,
    ledger_artifact: Mapping[str, object] | None = None,
    dataset_key: str = "populace_us_2024",
    dataset_filename: str = DATASET_FILENAME,
    calibration_key: str = "populace_us_2024_calibration",
    calibration_filename: str = CALIBRATION_FILENAME,
    exact_k_ladder: Mapping[str, object] | None = None,
    base_pool: Mapping[str, object] | None = None,
    evidence_known_failures: Sequence[Mapping[str, str]] | None = None,
) -> None:
    dataset_path = artifact_root / dataset_filename
    calibration_path = artifact_root / calibration_filename
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
        reported_coverage_vintage_gate=reported_coverage_vintage_gate,
        snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
        target_registry=registry,
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
            "repository": "PolicyEngine/microcosm",
            "git_commit": commit,
            "git_dirty": False,
        },
        "runtime": runtime,
        "timing": timing_payload,
        "ledger_artifact": dict(ledger_artifact) if ledger_artifact else None,
        **(
            {"exact_k_ladder": dict(exact_k_ladder)}
            if exact_k_ladder is not None
            else {}
        ),
        **({"base_pool": dict(base_pool)} if base_pool is not None else {}),
        "dataset": {
            "filename": dataset_filename,
            "sha256": dataset_sha,
            "default": default_dataset_payload,
        },
        "calibration": {
            "filename": calibration_filename,
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
            # Reviewed CMS Medicaid enrollment substitutions (microcosm#386):
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
            **(
                {
                    "exact_k_frozen_register_fit": dict(
                        exact_k_ladder["frozen_target_register"]["incumbent_fit"]
                    ),
                    "exact_k_puf_capital_gains_tail": dict(
                        exact_k_ladder["invariant_battery"]["puf_capital_gains_tail"]
                    ),
                }
                if exact_k_ladder is not None
                else {}
            ),
            "target_compilation": dropped,
            "target_profile_coverage": {
                "passed": target_profile_gate.passed,
                "failures": list(target_profile_gate.failures),
                "details": dict(target_profile_gate.details),
            },
            **(
                {
                    # The delivery gate result is the release's enforcement
                    # receipt: under the microcosm#566/#567 dense-arm fences a
                    # green release no longer implies every adult band was
                    # ENFORCED, so the manifest must say which bands were
                    # (enforced_band_keys) and which were fenced with their
                    # adjudication text (fenced_bands).
                    "ssi_take_up_delivery": {
                        "passed": ssi_take_up_delivery_gate_result.passed,
                        "failures": list(ssi_take_up_delivery_gate_result.failures),
                        "details": dict(ssi_take_up_delivery_gate_result.details),
                    }
                }
                if ssi_take_up_delivery_gate_result is not None
                else {}
            ),
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
                    "reported_coverage_vintage_signal": {
                        "passed": reported_coverage_vintage_gate.passed,
                        "failures": list(reported_coverage_vintage_gate.failures),
                        "details": dict(reported_coverage_vintage_gate.details),
                    }
                }
                if reported_coverage_vintage_gate is not None
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
            "name": "microcosm-data",
            "version": runtime["microcosm-data"],
        },
        "default_datasets": {"national": dataset_key},
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
            **(
                {"exact_k_ladder": dict(exact_k_ladder)}
                if exact_k_ladder is not None
                else {}
            ),
            **({"base_pool": dict(base_pool)} if base_pool is not None else {}),
            "warm_start_calibration": warm_start_payload,
            "selection_source": selection_source_payload,
            "default_dataset": default_dataset_payload,
            **(
                {
                    # microcosm#566/#567: release_manifest.json alone must
                    # distinguish fenced from enforced SSI delivery — the
                    # effective enforced set and the fenced rows (with
                    # their adjudication text) ride here as well as in
                    # build_manifest.json's gates block.
                    "ssi_take_up_delivery": {
                        "passed": ssi_take_up_delivery_gate_result.passed,
                        "details": dict(ssi_take_up_delivery_gate_result.details),
                    }
                }
                if ssi_take_up_delivery_gate_result is not None
                else {}
            ),
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
                    "reported_coverage_vintage_signal": {
                        "passed": reported_coverage_vintage_gate.passed,
                        "details": dict(reported_coverage_vintage_gate.details),
                    }
                }
                if reported_coverage_vintage_gate is not None
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
            dataset_key: _artifact_entry(
                dataset_filename,
                dataset_sha,
                kind="microdata",
                revision=release_id,
            ),
            calibration_key: _artifact_entry(
                calibration_filename,
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
            "us_ssi_take_up": _artifact_entry(
                "us_ssi_take_up.json",
                _sha256(release_dir / "us_ssi_take_up.json"),
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
    if evidence_known_failures is not None:
        # Evidence tier (microcosm#506): replace the certified schema marker
        # with the evidence one and carry the owned failure record. The
        # certified branch (evidence_known_failures None) writes the exact
        # dict above, byte-identical to a build without the flag.
        release_manifest.update(
            _evidence_release_manifest_fields(evidence_known_failures)
        )
    (release_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=1, allow_nan=False)
    )


def _load_qrf_tail_concentration_exclusions(path: Path | None) -> dict[str, str]:
    """Load a per-run QRF tail-concentration exclusion mapping.

    JSON object of ``column -> reason``: sparse QRF-imputed export columns
    allowed to stay concentrated past the #464 top-share threshold, each with
    a non-empty reason naming the tracked defect (the #481 weighted-leaf-draw
    root fix) or the genuinely concentrated instrument. The gate itself
    reports dormant entries and FAILS stale ones (a column now under the
    threshold), so the register cannot rot. Returns an empty mapping when no
    path is given.
    """
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(
            f"QRF tail-concentration exclusions file {path} must be a JSON "
            "object of column -> reason."
        )
    exclusions: dict[str, str] = {}
    for column, reason in payload.items():
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(
                "Every QRF tail-concentration exclusion needs a non-empty "
                f"reason; {column!r} in {path} has {reason!r}."
            )
        exclusions[str(column)] = reason
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


#: Standing failure-pattern -> owner register for --evidence-release
#: (microcosm#506). Only owner-adjudicated families belong here: the two dense
#: blockers named in the #506 brief — the SOI Table 1.4 dollar blanket (the
#: −30.2% capital-gain-distributions row, PUF donor uprating, #487) and the
#: dense QRF tail-concentration set (bootstrap seed-lottery tail draws, #481,
#: with the #487 uprating interaction). Any other recorded failure needs a
#: per-run adjudication via --evidence-failure-owners; an unowned failure
#: refuses the evidence export.
US_EVIDENCE_FAILURE_OWNERS: tuple[tuple[str, str], ...] = (
    (
        "SOI Table 1.4 national dollar fit failed:",
        "PolicyEngine/microcosm#487",
    ),
    (
        "QRF tail concentration failed:",
        "PolicyEngine/microcosm#481, PolicyEngine/microcosm#487",
    ),
)

# Kept in lockstep with microcosm.data.contract._ISSUE_REF_RE; the evidence
# publish contract re-validates every owner ref, so drift here fails at
# publish rather than silently.
_EVIDENCE_ISSUE_REF_RE = re.compile(r"#\d+|github\.com/\S+/(?:issues|pull)/\d+")


def _load_evidence_failure_owner_patterns(
    path: Path | None,
) -> tuple[tuple[str, str], ...]:
    """Failure-pattern -> owner pairs: per-run adjudications first, then the
    standing register, so a run-specific entry can sharpen a standing owner."""
    per_run: list[tuple[str, str]] = []
    if path is not None:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(
                f"--evidence-failure-owners {path} must be a JSON object of "
                "failure-substring pattern -> owner issue ref."
            )
        for pattern, owner in payload.items():
            if not isinstance(pattern, str) or not pattern.strip():
                raise ValueError(
                    f"--evidence-failure-owners {path} has an empty pattern key."
                )
            if not isinstance(owner, str) or not _EVIDENCE_ISSUE_REF_RE.search(owner):
                raise ValueError(
                    f"--evidence-failure-owners {path} owner for pattern "
                    f"{pattern!r} must carry an issue reference (e.g. "
                    f"'PolicyEngine/microcosm#487'), got {owner!r}."
                )
            per_run.append((pattern, owner))
    return (*per_run, *US_EVIDENCE_FAILURE_OWNERS)


def _evidence_known_failures(
    failures: Sequence[str],
    owner_patterns: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    """Recorded gate failures -> owned known_failures entries, verbatim.

    Every failure string must match an owner pattern (substring, first match
    wins); an unowned failure refuses the evidence export — the tier ships
    failures only when somewhere is accountable for fixing them.
    """
    entries: list[dict[str, str]] = []
    unowned: list[str] = []
    for failure in failures:
        owner = next(
            (owner for pattern, owner in owner_patterns if pattern in failure),
            None,
        )
        if owner is None:
            unowned.append(failure)
        else:
            entries.append({"failure": failure, "owner": owner})
    if unowned:
        raise RuntimeError(
            "Evidence release refused: recorded gate failure(s) match no "
            "owner in the standing US_EVIDENCE_FAILURE_OWNERS register or "
            "the --evidence-failure-owners file. Adjudicate an owner issue "
            "for each before shipping (microcosm#506): " + "; ".join(unowned)
        )
    return entries


def _evidence_release_manifest_fields(
    evidence_known_failures: Sequence[Mapping[str, str]],
) -> dict[str, object]:
    """The release-manifest fields that mark the evidence tier.

    The schema marker is the structural guarantee (microcosm#506): the
    certified contract rejects any manifest carrying it, so no flag-plumbing
    bug can mint a certified-shape manifest from an evidence build. An empty
    failure record is refused here too — the tier exists to carry failures,
    and an all-green run belongs on the certified path.
    """
    if not evidence_known_failures:
        raise ValueError(
            "an evidence release manifest requires at least one known "
            "failure; an all-green artifact belongs on the certified path."
        )
    return {
        "schema_version": EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION,
        "tier": "evidence",
        "known_failures": [dict(entry) for entry in evidence_known_failures],
    }


def _default_release_id(
    args: argparse.Namespace,
    *,
    digest: str,
    commit: str,
    build_timestamp: datetime,
) -> str:
    """The auto-generated release id for a run without --release-id.

    Exact-k candidates carry the k segment; evidence-tier builds carry the
    ``evidence`` segment (microcosm#506) so the tier is visible in every tag
    and download path; certified national builds keep the historical shape.
    """
    if args.exact_k is not None:
        return (
            f"populace-us-2024-k{args.exact_k}-{digest}-{commit}-"
            f"{build_timestamp:%Y%m%dT%H%M%SZ}"
        )
    if args.evidence_release:
        return (
            f"populace-us-2024-evidence-{digest}-{commit}-"
            f"{build_timestamp:%Y%m%dT%H%M%SZ}"
        )
    return f"populace-us-2024-{digest}-{commit}-{build_timestamp:%Y%m%dT%H%M%SZ}"


def _assert_us_release_id(release_id: str, *, evidence_release: bool = False) -> None:
    if not release_id.startswith("populace-us-"):
        raise ValueError(
            "US fiscal refresh release ids must start with 'populace-us-' so "
            "the US release contract requires source coverage diagnostics."
        )
    if evidence_release and EVIDENCE_RELEASE_ID_SEGMENT not in release_id:
        raise ValueError(
            "--evidence-release ids must carry the "
            f"{EVIDENCE_RELEASE_ID_SEGMENT!r} segment (e.g. "
            "populace-us-2024-evidence-<sha>-<date>) so the tier is visible "
            "in every tag and download path."
        )
    if not evidence_release and EVIDENCE_RELEASE_ID_SEGMENT in release_id:
        raise ValueError(
            f"release ids containing {EVIDENCE_RELEASE_ID_SEGMENT!r} are "
            "reserved for --evidence-release builds; a certified build must "
            "not squat the evidence namespace."
        )


def _assert_exact_k_release_id(release_id: str, k: int) -> None:
    expected_prefix = f"populace-us-{PERIOD}-k{k}-"
    if not release_id.startswith(expected_prefix):
        raise ValueError(
            "US exact-k ladder release ids must start with "
            f"{expected_prefix!r}; got {release_id!r}."
        )
    if any(not (character.isalnum() or character == "-") for character in release_id):
        raise ValueError(
            "US exact-k ladder release ids may contain only ASCII letters, "
            f"digits, and hyphens; got {release_id!r}."
        )


def _exact_k_ladder_manifest_payload(
    *,
    args: argparse.Namespace,
    outcome: ExactKLadderCalibration,
    pool_manifest: Mapping[str, object],
    authenticated_pool_h5: AuthenticatedPoolH5,
    ledger_artifact: Mapping[str, object],
    target_surface: Mapping[str, object],
    target_loss_basis: Mapping[str, object],
    incumbent_diagnostics_sha256: str,
    incumbent_fit_gate: GateResult,
    puf_tail_gate: GateResult,
) -> dict[str, object]:
    """Build the one receipt block shared by diagnostics and both manifests."""

    agreement_diagnostics = pool_manifest.get("agreement_diagnostics")
    agreement_gate = pool_manifest.get("agreement_gate")
    if not all(
        isinstance(value, Mapping) for value in (agreement_diagnostics, agreement_gate)
    ):
        raise RuntimeError("Validated pool manifest lost a required receipt block.")
    if agreement_gate.get("passed") is not True:
        raise RuntimeError("Validated pool manifest lost its passing agreement gate.")
    pool_release_id = _assert_pool_release_id_value(
        args.pool_release_id,
        authenticated_pool_h5.publication_run_id,
    )
    payload = exact_k_ladder_manifest_payload(
        outcome,
        k=int(args.exact_k),
        seed=int(args.seed),
        pool={
            "release_id": pool_release_id,
            "release_id_source": "pool_manifest.publication_run_id",
            "manifest_sha256": authenticated_pool_h5.manifest_sha256,
            "publication_run_id": authenticated_pool_h5.publication_run_id,
            "pool_h5_sha256": authenticated_pool_h5.sha256,
            "pool_h5_size_bytes": authenticated_pool_h5.size_bytes,
            "agreement_diagnostics_sha256": agreement_diagnostics.get("sha256"),
        },
        agreement_gate_reference={
            "passed": True,
            "publication_run_id": authenticated_pool_h5.publication_run_id,
            "diagnostics_sha256": agreement_diagnostics.get("sha256"),
            "verdict": dict(agreement_gate),
        },
        frozen_target_register={
            "ledger_artifact": dict(ledger_artifact),
            "target_surface_sha256": target_surface.get("sha256"),
            "target_loss_basis": dict(target_loss_basis),
            "incumbent_diagnostics_sha256": incumbent_diagnostics_sha256,
            "incumbent_fit": {
                "passed": incumbent_fit_gate.passed,
                "failures": list(incumbent_fit_gate.failures),
                "details": dict(incumbent_fit_gate.details),
            },
        },
    )
    payload["invariant_battery"] = {
        "puf_capital_gains_tail": {
            "passed": puf_tail_gate.passed,
            "failures": list(puf_tail_gate.failures),
            "details": dict(puf_tail_gate.details),
        }
    }
    return payload


def _exact_k_original_support_frame(
    frame: Frame,
    support: np.ndarray,
) -> Frame:
    """Subset an original frame by household positions without changing weights."""

    household_ids = frame.table("household")[
        frame.schema.id_column("household")
    ].to_numpy()[np.asarray(support, dtype=np.int64)]
    person_households = frame.table("person")[
        frame.schema.membership_column("household")
    ]
    return frame.select(person_households.isin(household_ids).to_numpy())


def _exact_k_puf_tail_support_gate(
    frame: Frame,
    support: np.ndarray,
) -> GateResult:
    """Batch a post-selection PUF-tail miss without discarding solve evidence."""

    try:
        receipt = assert_puf_capital_gains_tail_survives_selection(
            frame,
            _exact_k_original_support_frame(frame, support),
            require_present=True,
        )
    except ValueError as error:
        return GateResult(
            name="exact_k_puf_capital_gains_tail",
            passed=False,
            failures=(str(error),),
            details={
                "status": "failed",
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            },
        )
    return GateResult(
        name="exact_k_puf_capital_gains_tail",
        passed=True,
        details={key: value for key, value in receipt.items() if key != "passed"},
    )


def _assert_exact_k_original_pool_alignment(
    frame: Frame,
    *,
    household_ids: np.ndarray,
    household_weights: np.ndarray,
) -> None:
    """Fail if downstream preparation changed the pool rows or design weights."""

    observed_weights = frame.weights_for("household")
    observed_ids = frame.table("household")["household_id"].to_numpy()
    if observed_weights.kind is not WeightKind.IMPORTANCE:
        raise RuntimeError(
            "Exact-k target preparation changed the original pool weight kind: "
            f"got {observed_weights.kind.value!r}, expected 'importance'."
        )
    if not np.array_equal(observed_ids, household_ids):
        raise RuntimeError(
            "Exact-k target preparation changed or reordered the original pool "
            "household support."
        )
    if not np.array_equal(observed_weights.values, household_weights):
        raise RuntimeError(
            "Exact-k target preparation changed the original pool weights before "
            "selection; the HT-with-q baseline would no longer be frame-original."
        )


#: The staging run for the build in flight, so the entry point can mark it
#: failed on the way out. The build body hands its telemetry object down a
#: large call stack; a module-level handle avoids threading a second copy back
#: up purely for the failure path.
_ACTIVE_TELEMETRY: StagingTelemetry | None = None


def _staging_manifest_block(telemetry: StagingTelemetry | None) -> dict[str, object]:
    """Record what staging did and distinguish an opt-out from non-delivery.

    Uploads are best-effort and self-disable after repeated failures, so a
    configured destination is not evidence that anything reached it.
    """

    if telemetry is None:
        return {"enabled": False, "reason": "--no-staging"}
    return {
        "enabled": True,
        "run_id": telemetry.run_id,
        "repo_id": telemetry.repo_id,
        "uploads_succeeded": telemetry.uploads_succeeded,
    }


def _staging_telemetry(
    args: argparse.Namespace,
    *,
    release_root: Path,
    release_id: str,
) -> StagingTelemetry | None:
    global _ACTIVE_TELEMETRY
    # Each call establishes the current run, so a handle from a previous one
    # can never be marked failed in place of this build's.
    _ACTIVE_TELEMETRY = None
    if args.no_staging:
        return None
    if not args.staging_dir and not args.staging_repo_id:
        # The parser rejects this combination, so reaching it means a caller
        # built the namespace directly. Returning None here would reinstate
        # the silent skip the parser guard exists to prevent.
        raise ValueError(
            "staging is enabled but has no destination: staging_repo_id is "
            "empty and staging_dir is unset. Set no_staging to skip staging, "
            "or give a staging_dir for a local-only run."
        )
    run_id = args.staging_run_id or release_id
    run_dir = args.staging_dir or release_root / "staging" / "runs" / run_id
    _ACTIVE_TELEMETRY = StagingTelemetry(
        run_id=run_id,
        candidate_release_id=release_id,
        run_dir=run_dir,
        repo_id=args.staging_repo_id,
        path_prefix=args.staging_prefix,
        upload_interval_seconds=args.staging_upload_interval_seconds,
    )
    return _ACTIVE_TELEMETRY


class _TerminalBatchTelemetry:
    """Turn terminal-batch telemetry crashes into release-gate failures.

    The proxy is deliberately scoped to the post-diagnostics terminal batch.
    A telemetry crash on an otherwise-green run now fails the release as a
    recorded batch line rather than causing an opaque abort, and every later
    gate group still gets a chance to contribute its evidence.
    """

    def __init__(
        self,
        telemetry: StagingTelemetry | None,
        terminal_gate_failures: list[str],
    ) -> None:
        self._telemetry = telemetry
        self._terminal_gate_failures = terminal_gate_failures

    def _call(
        self,
        method_name: str,
        label: str,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            getattr(self._telemetry, method_name)(label, *args, **kwargs)
        except Exception as error:
            self._terminal_gate_failures.append(
                "Terminal-batch telemetry "
                f"{method_name}({label!r}) crashed; recorded as a release "
                "failure instead of interrupting the remaining gate "
                f"evaluations: {type(error).__name__}: {error}"
            )

    def stage(self, stage: str, **details: Any) -> None:
        self._call("stage", stage, **details)

    def attach_artifact(
        self,
        name: str,
        path: Path | str,
        **details: Any,
    ) -> None:
        self._call("attach_artifact", name, path, **details)


def _print_build_result(
    *,
    release_id: str,
    release_dir: Path,
    artifact_root: Path,
) -> None:
    """Print the legacy three-key builder result without widening its API."""

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


def main(argv: Sequence[str] | None = None) -> None:
    """Run the build and mark its staging run failed if it does not finish.

    An uncaught build error previously left the dashboard status at ``running``
    forever. SIGKILL remains outside the reach of an in-process handler; this
    closes the ordinary exception and termination half of the gap.
    """

    try:
        _main(argv)
    except BaseException as error:
        if _ACTIVE_TELEMETRY is not None:
            try:
                _ACTIVE_TELEMETRY.fail(error)
            except Exception as telemetry_error:  # pragma: no cover - defensive
                # A failing failure-report must not replace the real traceback.
                print(
                    "warning: could not record the staging run as failed: "
                    f"{type(telemetry_error).__name__}: {telemetry_error}",
                    file=sys.stderr,
                )
        raise


def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if _git_dirty():
        raise SystemExit("Refusing to build a release from a dirty git worktree.")
    # Loaded (and validated) up front so a malformed owners file dies in
    # seconds, not after the multi-hour build. The flag-combination rules
    # (--evidence-failure-owners requires the tier flag; --exact-k is
    # incompatible with it) live in _parse_args with the other combos.
    evidence_failure_owner_patterns = (
        _load_evidence_failure_owner_patterns(args.evidence_failure_owners)
        if args.evidence_release
        else ()
    )
    build_started = time.perf_counter()
    timing: dict[str, float] = {}

    if args.release_id:
        # microcosm#568 round 4: when the id is known up front (every launcher
        # passes one), refuse certified-dir reuse before ANY side effect —
        # including the base download and cache writes below. The
        # auto-generated-id path derives its id from the base digest, so its
        # refusal necessarily runs later, but still before any output-dir
        # creation.
        _refuse_certified_release_dir_reuse(
            args.out.resolve() / "releases" / args.release_id
        )
    pinned_incumbent_payload: dict[str, object] | None = None
    pinned_incumbent_sha256: str | None = None
    if args.exact_k is not None:
        pinned_incumbent_payload, pinned_incumbent_sha256 = (
            _load_verified_incumbent_diagnostics_payload(
                args.incumbent_diagnostics,
                expected_sha256=args.incumbent_diagnostics_sha256,
            )
        )
    pool_frame: Frame | None = None
    pool_original_household_ids: np.ndarray | None = None
    pool_original_household_weights: np.ndarray | None = None
    pool_manifest_payload: dict[str, object] | None = None
    base_pool_receipt: dict[str, object] | None = None
    authenticated_pool_h5: AuthenticatedPoolH5 | None = None
    if args.pool_manifest is not None:
        pool_frame, pool_manifest_payload, authenticated_pool_h5 = (
            load_simulation_ready_us_multispine_pool(
                args.pool_manifest,
                expected_manifest_sha256=args.pool_manifest_sha256,
            )
        )
        base_dataset_sha256 = authenticated_pool_h5.verified_digest(
            consumer="builder base dataset identity"
        )
        _assert_pool_release_id_value(
            args.pool_release_id,
            authenticated_pool_h5.publication_run_id,
        )
        if args.exact_k == "N":
            args.exact_k = int(pool_frame.n("household"))
        pool_original_household_ids = pool_frame.table("household")[
            "household_id"
        ].to_numpy(copy=True)
        pool_original_household_weights = pool_frame.weights_for(
            "household"
        ).values.copy()
        if args.exact_k > pool_frame.n("household"):
            raise ValueError(
                f"k={args.exact_k} exceeds the pool size "
                f"{pool_frame.n('household')}; ladder selection never clamps "
                "the requested cardinality."
            )
        base_h5 = authenticated_pool_h5.path
    else:
        base_h5 = args.base_h5 or _download_base_h5()
        pool_frame, base_pool_receipt, authenticated_pool_h5 = (
            _load_base_pool_if_identified(
                base_h5,
                allow_gate_failed_base_pool=args.allow_gate_failed_base_pool,
            )
        )
        if authenticated_pool_h5 is None:
            base_dataset_sha256 = _legacy_base_h5_sha256(base_h5)
        else:
            base_dataset_sha256 = authenticated_pool_h5.verified_digest(
                consumer="builder base dataset identity"
            )
            base_h5 = authenticated_pool_h5.path
    digest = base_dataset_sha256[:7]
    build_timestamp = datetime.now(UTC)
    full_commit = _git_output("rev-parse", "HEAD")
    commit = _git_output("rev-parse", "--short=12", "HEAD")
    release_id = args.release_id or _default_release_id(
        args,
        digest=digest,
        commit=commit,
        build_timestamp=build_timestamp,
    )
    _assert_us_release_id(release_id, evidence_release=args.evidence_release)
    if args.exact_k is not None:
        _assert_exact_k_release_id(release_id, args.exact_k)
        # The immutable release id is the dataset's exact-count name. Keep the
        # files at the registry's canonical paths so a later *manual* pointer
        # flip remains loadable by microcosm-data.
        dataset_key = "populace_us_2024"
        dataset_filename = DATASET_FILENAME
        calibration_key = "populace_us_2024_calibration"
        calibration_filename = CALIBRATION_FILENAME
    else:
        dataset_key = "populace_us_2024"
        dataset_filename = DATASET_FILENAME
        calibration_key = "populace_us_2024_calibration"
        calibration_filename = CALIBRATION_FILENAME
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
        base_h5,
        congressional_district_vintage_crosswalk_metadata,
        authenticated_pool_h5=authenticated_pool_h5,
    )
    # Preflight (before the expensive calibration): every provision-critical
    # input leaf the reform-validation configs depend on must be produced by a
    # source stage or be an allowlisted known gap. This catches the
    # structurally-missing-input class (PolicyEngine/microcosm#252, #253) before a
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
    # Preflight (microcosm#368): the release input-column coverage manifest must
    # still equal the pinned reference eCPS input surface, keep the SSI
    # countable-resource assets as hard requirements, and declare only live
    # PolicyEngine-US input leaves. A drifted manifest fails here before the
    # expensive calibration so a stale contract cannot silently narrow coverage.
    assert_release_input_coverage_manifest_current()
    # Preflight (microcosm#377): no column may be required-to-signal by one
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
    # Preflight (microcosm#381): the checked-in take-up contract must still match
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
    target_registry = compile_us_fiscal_target_registry(
        ledger_artifact.facts,
        target_period=PERIOD,
        congressional_district_vintage_crosswalk=(
            congressional_district_vintage_crosswalk
        ),
        age_targets=args.age_targets,
        allow_unaged_dollar_targets=args.allow_unaged_dollar_targets,
    )
    # Reviewed CMS Medicaid enrollment substitutions (microcosm#386): a state
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
    # Target-parity contract (launch gate): every administrative target family
    # the retired us-data/eCPS pipeline calibrated to must be compiled into the
    # registry or carry a reviewed exclusion (target_parity_manifest.json). Runs
    # on the full compiled + substituted registry — before the optional
    # diagnostic JCT skip — so the gate sees the true family surface, and
    # hard-fails the build on a silently dropped family or a rotted manifest,
    # exactly like the release input-coverage gate on the export frame.
    assert_target_parity_manifest_current(registry=target_registry)
    target_parity_gate = us_release_target_parity_gate(target_registry)
    if not target_parity_gate.passed:
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Target parity coverage failed: {failure}"
                for failure in target_parity_gate.failures
            )
        )
    target_specs = target_registry.specs
    active_target_registry = TargetRegistry(target_specs, country="us")
    # SSI take-up wiring resolves as soon as the registry exists (fail-fast,
    # microcosm#507/#508): the band targets come from the same ledger-fed
    # registry rows the solve enforces, and an invalid delivered-weight
    # basis artifact must fail here, not after the imputation stages.
    ssi_band_targets = _ssi_take_up_band_targets_from_registry(target_specs)
    ssi_take_up_prior_basis = _load_ssi_take_up_prior_weight_basis(
        args.ssi_take_up_prior_weight_basis,
        targets=ssi_band_targets,
        expected_sha256=args.ssi_take_up_prior_weight_basis_sha256,
    )
    target_profile_gate = target_profile_coverage_gate(
        target_specs,
        US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    )
    if not target_profile_gate.passed:
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
    # Unconditional refusal BEFORE any output-directory creation: a hostile
    # --checkpoint-root beneath releases/<id> must not mutate a certified
    # directory before the raise (microcosm#568 round 4).
    _refuse_certified_release_dir_reuse(release_dir)
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
    if pool_frame is None:
        base_frame = _load_frame(base_h5)
    else:
        base_frame = pool_frame
    capital_gains_tail_presence = assert_puf_capital_gains_tail_survives_selection(
        base_frame,
        base_frame,
        require_present=True,
    )
    if telemetry is not None:
        telemetry.stage(
            "capital_gains_tail_presence",
            message="Verified the materialized PUF capital-gains own-tail.",
            **capital_gains_tail_presence,
        )
    if pool_frame is None:
        weeks_unemployed_source_path = (
            args.asec_2023_weeks_unemployed_source
            if args.asec_2023_weeks_unemployed_source is not None
            else fetch_asec_2023_weeks_unemployed_source()
        )
        weeks_unemployed_source = load_asec_2023_weeks_unemployed_source(
            weeks_unemployed_source_path
        )
        base_frame = with_us_weeks_unemployed(
            base_frame,
            seed=args.seed,
            time_period=PERIOD,
            asec_2023_source=weeks_unemployed_source,
        )
        weeks_unemployed_source_receipt = {
            "source_path": str(Path(weeks_unemployed_source_path).resolve()),
            "source_sha256": ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256,
            "source_rows": len(weeks_unemployed_source),
        }
        weeks_unemployed_message = (
            "Restored measured ASEC LKWEEKS before frozen-support selection "
            "and target materialization."
        )
    else:
        weeks_unemployed_source_receipt = {
            "source": "validated_multispine_pool",
            "pool_publication_run_id": authenticated_pool_h5.publication_run_id,
        }
        weeks_unemployed_message = (
            "Verified measured ASEC LKWEEKS before selection and target "
            "materialization."
        )
    weeks_unemployed_gate = us_weeks_unemployed_signal_gate(base_frame)
    if not weeks_unemployed_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "weeks_unemployed_input_gate",
                status="failed",
                message="Weeks-unemployed input signal gate failed.",
                failures=list(weeks_unemployed_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Weeks-unemployed input signal failed: " + failure
                for failure in weeks_unemployed_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "weeks_unemployed_input",
            message=weeks_unemployed_message,
            **weeks_unemployed_source_receipt,
        )
    # Capture direct ASEC reporter lineage on the FULL clone-aware support.
    # Frozen-support recovery may retain only a PUF clone; deriving anchors
    # after that prune would erase the underlying measured ASEC reporter.
    ssi_reporter_source_ids = us_ssi_take_up_reporter_source_ids(base_frame)

    # Frozen-support recovery (microcosm#328): if a selection source is supplied,
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
                "calibration; only 'frozen_support' is implemented (microcosm#328)."
            )
        if telemetry is not None:
            telemetry.stage(
                "frozen_support_selection",
                message="Recovering frozen support by stable source identity.",
                n_source=selection_source.n_identities,
                join_key=list(selection_source.join_key),
            )
        selection_candidate_frame = base_frame
        base_frame, selection_report = select_frozen_support(
            selection_candidate_frame,
            selection_source,
        )
        selection_source_payload = selection_report.as_manifest()
        selection_source_payload["puf_capital_gains_tail_retention"] = (
            assert_puf_capital_gains_tail_survives_selection(
                selection_candidate_frame,
                base_frame,
                require_present=True,
            )
        )
        if telemetry is not None:
            telemetry.stage(
                "frozen_support_selection_done",
                message="Reduced base pool to the frozen support.",
                n_selected=selection_report.n_selected,
                n_base_candidates=selection_report.n_base_candidates,
                n_unmapped=selection_report.n_unmapped,
            )
        post_selection_weeks_unemployed_gate = us_weeks_unemployed_signal_gate(
            base_frame
        )
        if not post_selection_weeks_unemployed_gate.passed:
            if telemetry is not None:
                telemetry.stage(
                    "post_selection_weeks_unemployed_input_gate",
                    status="failed",
                    message=(
                        "Frozen-support selection collapsed weeks-unemployed "
                        "input signal."
                    ),
                    failures=list(post_selection_weeks_unemployed_gate.failures),
                    force_upload=True,
                )
            raise RuntimeError(
                "Release gates failed: "
                + "; ".join(
                    "Post-selection weeks-unemployed input signal failed: " + failure
                    for failure in post_selection_weeks_unemployed_gate.failures
                )
            )

    if pool_frame is None:
        base_frame, base_population_repair = _with_base_population_mass_repair(
            base_frame
        )
    else:
        pool_population = _person_population(base_frame)
        base_population_repair = {
            "method": "preserve_validated_multispine_pool_weights",
            "applied": False,
            "reason": (
                "The authenticated multispine input retains the published "
                "pool artifact's original importance-weight baseline; "
                "pool-owned preparation stages are not replayed."
            ),
            "initial_population": pool_population,
            "benchmark": US_BASE_PERSON_POPULATION_BENCHMARK,
            "factor": 1.0,
            "initial_relative_error": _base_population_relative_error(pool_population),
            "repaired_population": pool_population,
            "repaired_relative_error": _base_population_relative_error(pool_population),
        }
    base_frame, social_security_component_repair = (
        _with_social_security_component_value_repair(base_frame, target_specs)
    )
    base_frame, non_sch_d_cgd_repair = _with_non_sch_d_cgd_value_repair(
        base_frame, target_specs
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
        telemetry.stage(
            "non_sch_d_cgd_repair",
            message=(
                "Pinned non_sch_d_capital_gains to the registry's SOI Table "
                "1.4 dollar fact (microcosm#462 donor-uprating interim "
                "repair; aging recorded in target_aged_to)."
            ),
            applied=non_sch_d_cgd_repair.get("applied"),
            target_aged_to=non_sch_d_cgd_repair.get("target_aged_to"),
            factor=non_sch_d_cgd_repair.get("factor"),
            target=non_sch_d_cgd_repair.get("target"),
            initial_estimate=non_sch_d_cgd_repair.get("initial_estimate"),
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
    if pool_frame is None:
        base_frame = with_us_qbi_input_reconciliation(base_frame)
    qbi_inputs_gate = us_qbi_inputs_signal_gate(base_frame)
    if not qbi_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "qbi_input_gate",
                status="failed",
                message="QBI-input signal gate failed.",
                failures=list(qbi_inputs_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "QBI-input signal failed: " + failure
                for failure in qbi_inputs_gate.failures
            )
        )
    farm_business_income_gate = us_farm_business_income_signal_gate(base_frame)
    if not farm_business_income_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "farm_business_income_gate",
                status="failed",
                message="Farm-business-income signal gate failed.",
                failures=list(farm_business_income_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Farm-business-income signal failed: " + failure
                for failure in farm_business_income_gate.failures
            )
        )
    domestic_production_ald_gate = us_domestic_production_ald_signal_gate(base_frame)
    if not domestic_production_ald_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "domestic_production_ald_gate",
                status="failed",
                message="Domestic-production-ALD signal gate failed.",
                failures=list(domestic_production_ald_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Domestic-production-ALD signal failed: " + failure
                for failure in domestic_production_ald_gate.failures
            )
        )
    child_support_gate = us_child_support_signal_gate(base_frame)
    if not child_support_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "child_support_input_gate",
                status="failed",
                message="Child-support input signal gate failed.",
                failures=list(child_support_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Child-support input signal failed: " + failure
                for failure in child_support_gate.failures
            )
        )
    disability_benefits_gate = us_disability_benefits_signal_gate(base_frame)
    if not disability_benefits_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "disability_benefits_input_gate",
                status="failed",
                message="Disability-benefits input signal gate failed.",
                failures=list(disability_benefits_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Disability-benefits input signal failed: " + failure
                for failure in disability_benefits_gate.failures
            )
        )
    workers_compensation_gate = us_workers_compensation_signal_gate(base_frame)
    if not workers_compensation_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "workers_compensation_input_gate",
                status="failed",
                message="Workers-compensation input signal gate failed.",
                failures=list(workers_compensation_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Workers-compensation input signal failed: " + failure
                for failure in workers_compensation_gate.failures
            )
        )
    educator_expense_gate = us_educator_expense_signal_gate(base_frame)
    if not educator_expense_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "educator_expense_input_gate",
                status="failed",
                message="Educator-expense input signal gate failed.",
                failures=list(educator_expense_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Educator-expense input signal failed: " + failure
                for failure in educator_expense_gate.failures
            )
        )
    form_4952_election_gate = us_form_4952_election_signal_gate(base_frame)
    if not form_4952_election_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "form_4952_election_input_gate",
                status="failed",
                message="Form 4952 election input signal gate failed.",
                failures=list(form_4952_election_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Form 4952 election signal failed: " + failure
                for failure in form_4952_election_gate.failures
            )
        )
    salt_refund_income_gate = us_salt_refund_income_signal_gate(base_frame)
    if not salt_refund_income_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "salt_refund_income_input_gate",
                status="failed",
                message="SALT-refund-income signal gate failed.",
                failures=list(salt_refund_income_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "SALT-refund-income signal failed: " + failure
                for failure in salt_refund_income_gate.failures
            )
        )
    capital_gain_details_gate = us_capital_gain_details_signal_gate(base_frame)
    if not capital_gain_details_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "capital_gain_details_input_gate",
                status="failed",
                message="Capital-gain details input signal gate failed.",
                failures=list(capital_gain_details_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Capital-gain details signal failed: " + failure
                for failure in capital_gain_details_gate.failures
            )
        )
    if pool_frame is None:
        base_frame = with_us_childcare_inputs(
            base_frame,
            seed=args.seed,
            time_period=PERIOD,
            allow_existing_without_source=True,
        )
    childcare_gate = us_childcare_signal_gate(base_frame)
    if not childcare_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "childcare_input_gate",
                status="failed",
                message="Childcare-input signal gate failed.",
                failures=list(childcare_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Childcare-input signal failed: " + failure
                for failure in childcare_gate.failures
            )
        )
    if pool_frame is None:
        base_frame = with_us_energy_subsidy_input(
            base_frame,
            seed=args.seed,
            time_period=PERIOD,
            allow_existing_without_source=True,
        )
    energy_subsidy_gate = us_energy_subsidy_signal_gate(base_frame)
    if not energy_subsidy_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "energy_subsidy_gate",
                status="failed",
                message="Energy-subsidy signal gate failed.",
                failures=list(energy_subsidy_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Energy-subsidy signal failed: " + failure
                for failure in energy_subsidy_gate.failures
            )
        )
    alimony_gate = us_alimony_signal_gate(base_frame)
    if not alimony_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "alimony_input_gate",
                status="failed",
                message="Alimony-input signal gate failed.",
                failures=list(alimony_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Alimony-input signal failed: " + failure
                for failure in alimony_gate.failures
            )
        )
    casualty_loss_gate = us_casualty_loss_signal_gate(base_frame)
    if not casualty_loss_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "casualty_loss_input_gate",
                status="failed",
                message="Casualty-loss signal gate failed.",
                failures=list(casualty_loss_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Casualty-loss signal failed: " + failure
                for failure in casualty_loss_gate.failures
            )
        )
    misc_itemized_gate = us_misc_itemized_signal_gate(base_frame)
    if not misc_itemized_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "misc_itemized_input_gate",
                status="failed",
                message="Miscellaneous-itemized signal gate failed.",
                failures=list(misc_itemized_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Miscellaneous-itemized signal failed: " + failure
                for failure in misc_itemized_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "retirement_contribution_inputs",
            message=("Verifying ASEC-sourced desired retirement-contribution inputs."),
        )
    if pool_frame is None:
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
    if pool_frame is None:
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
    if pool_frame is None:
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
    if pool_frame is None:
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
            "relationship_inputs",
            message=("Deriving household-head and marital-status inputs from ASEC."),
        )
    if pool_frame is None:
        base_frame = with_us_relationship_inputs(
            base_frame,
            seed=args.seed,
            time_period=PERIOD,
        )
    relationship_inputs_gate = us_relationship_inputs_signal_gate(base_frame)
    if not relationship_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "relationship_inputs_gate",
                status="failed",
                message="Relationship-input signal gate failed.",
                failures=list(relationship_inputs_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Relationship-input signal failed: {failure}"
                for failure in relationship_inputs_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "medicare_take_up_input",
            message="Deriving measured Medicare enrollment from ASEC MCARE.",
        )
    if pool_frame is None:
        base_frame = with_us_medicare_take_up_input(
            base_frame,
            seed=args.seed,
            time_period=PERIOD,
        )
    medicare_take_up_gate = us_medicare_take_up_signal_gate(base_frame)
    if not medicare_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "medicare_take_up_input_gate",
                status="failed",
                message="Medicare take-up input signal gate failed.",
                failures=list(medicare_take_up_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Medicare take-up input signal failed: {failure}"
                for failure in medicare_take_up_gate.failures
            )
        )
    prior_year_income_gate = us_prior_year_income_signal_gate(base_frame)
    if not prior_year_income_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "prior_year_income_gate",
                status="failed",
                message="Prior-year-income signal gate failed.",
                failures=list(prior_year_income_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Prior-year-income signal failed: {failure}"
                for failure in prior_year_income_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "prior_year_income_gate",
            message="Verified restored adjacent-ASEC prior-year income inputs.",
            details=dict(prior_year_income_gate.details),
        )
    housing_inputs_gate = us_housing_inputs_signal_gate(base_frame)
    if not housing_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "housing_inputs_gate",
                status="failed",
                message="Housing/tenure input signal gate failed.",
                failures=list(housing_inputs_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Housing/tenure input signal failed: {failure}"
                for failure in housing_inputs_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "housing_inputs_gate",
            message="Verified restored CPS/ACS housing and tenure inputs.",
            details=dict(housing_inputs_gate.details),
        )
    if telemetry is not None:
        telemetry.stage(
            "retirement_distribution_inputs",
            message=(
                "Carrying measured ASEC retirement distributions by account type."
            ),
        )
    if pool_frame is None:
        base_frame = with_us_retirement_distribution_inputs(
            base_frame,
            seed=args.seed,
            time_period=PERIOD,
        )
    retirement_distributions_gate = us_retirement_distributions_signal_gate(base_frame)
    early_terminal_gate_failures: list[str] = []
    if not retirement_distributions_gate.passed:
        # A selected support is consume-only at this boundary. Preserve the
        # gate's batched missing/degenerate leaf evidence and carry it to the
        # #548 terminal accumulator instead of refitting or raising early.
        early_terminal_gate_failures.extend(
            "Retirement-distribution signal failed: " + failure
            for failure in retirement_distributions_gate.failures
        )
        try:
            if telemetry is not None:
                telemetry.stage(
                    "retirement_distribution_inputs_gate",
                    status="failed",
                    message="Retirement-distribution signal gate failed.",
                    failures=list(retirement_distributions_gate.failures),
                    force_upload=True,
                )
        except Exception as error:
            early_terminal_gate_failures.append(
                "Retirement-distribution gate failure telemetry crashed; "
                f"recorded instead of masking the failure: {error}"
            )
    if telemetry is not None:
        telemetry.stage(
            "eligibility_inputs",
            message="Deriving SNAP eligibility and exemption inputs from ASEC.",
        )
    if pool_frame is None:
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
    if pool_frame is None:
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
    if pool_frame is None:
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
    reported_coverage_vintage_gate = us_reported_coverage_vintage_signal_gate(
        base_frame
    )
    if not reported_coverage_vintage_gate.passed:
        receipt_path = _write_reported_coverage_vintage_gate_receipt(
            release_dir, reported_coverage_vintage_gate
        )
        if telemetry is not None:
            telemetry.stage(
                "reported_coverage_vintage_gate",
                status="failed",
                message="Reported-coverage vintage signal gate failed.",
                failures=list(reported_coverage_vintage_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Reported-coverage vintage signal failed: {failure}"
                for failure in reported_coverage_vintage_gate.failures
            )
            + f" (receipt: {receipt_path})"
        )
    if telemetry is not None:
        telemetry.stage(
            "wic_claim_input",
            message="Assigning WIC claims from USDA FNS category coverage rates.",
        )
    if pool_frame is None:
        base_frame = with_us_wic_claim_input(
            base_frame,
            seed=args.seed,
            time_period=PERIOD,
        )
    wic_claim_gate = us_wic_claim_signal_gate(base_frame)
    if not wic_claim_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "wic_claim_input_gate",
                status="failed",
                message="WIC-claim input signal gate failed.",
                failures=list(wic_claim_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"WIC-claim input signal failed: {failure}"
                for failure in wic_claim_gate.failures
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
                "Imputing signed household net worth and SSI countable-resource "
                "assets (bank/stock/bond) from the seeded SIPP 2023 / SCF 2022 "
                "household blend."
            ),
        )
    # microcosm#49/#356/#368/#374: restore signed household net_worth plus the three
    # SSI countable-resource asset inputs (bank_account_assets / stock_assets /
    # bond_assets). One seeded household source draw selects all three leaves
    # from either the SCF or SIPP donor; SCF still anchors signed net worth.
    # Without the leaves, ssi_countable_resources is 0 for every record and SSI
    # resource-limit reforms silently score $0.
    scf_summary_extract_path = (
        Path(args.scf_summary_extract)
        if args.scf_summary_extract is not None
        else fetch_scf_2022_summary_extract()
    )
    # The asset blend, SSI criterion, Head Start, vehicle, and filing families
    # share one immutable full SIPP artifact. Resolve it once; the CLI option's
    # historical name remains stable for existing release invocations.
    sipp_full_donor_path = (
        Path(args.sipp_vehicle_donor)
        if args.sipp_vehicle_donor is not None
        else fetch_sipp_2023_financial_asset_donor()
    )
    scf_wealth_donor = load_scf_2022_financial_asset_donor(scf_summary_extract_path)
    sipp_financial_asset_donor = load_sipp_2023_financial_asset_donor(
        sipp_full_donor_path,
        expected_sha256=SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
    )
    base_frame = with_us_scf_wealth_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        scf_donor=scf_wealth_donor,
        sipp_donor=sipp_financial_asset_donor,
    )
    scf_wealth_gate = us_scf_wealth_signal_gate(
        base_frame,
        require_sipp_blend=True,
    )
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
    sipp_vehicle_donor_path = sipp_full_donor_path
    if telemetry is not None:
        telemetry.stage(
            "ssi_disability_criteria",
            message=(
                "Imputing SSI-specific disability criteria from the "
                "sha-pinned full SIPP 2023 donor after SCF assets."
            ),
        )
    ssi_disability_donor = load_sipp_2023_ssi_disability_donor(
        sipp_vehicle_donor_path,
        expected_sha256=SIPP_2023_SSI_DISABILITY_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_SSI_DISABILITY_DONOR_SIZE_BYTES,
        time_period=PERIOD,
    )
    base_frame = with_us_ssi_disability_criteria(
        base_frame,
        # The retired weighted bootstrap and MicroImpute forest are fixed.
        seed=42,
        time_period=PERIOD,
        sipp_donor=ssi_disability_donor,
    )
    ssi_disability_gate = us_ssi_disability_criteria_signal_gate(base_frame)
    if not ssi_disability_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "ssi_disability_criteria_gate",
                status="failed",
                message="SSI disability-criteria signal gate failed.",
                failures=list(ssi_disability_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SSI disability-criteria signal failed: {failure}"
                for failure in ssi_disability_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "sipp_head_start",
            message=(
                "Imputing age-3--5 Head Start take-up from direct and strict "
                "structural December responses in the sha-pinned full SIPP "
                "2023 donor."
            ),
        )
    head_start_donor = load_sipp_2023_head_start_donor(
        sipp_vehicle_donor_path,
        expected_sha256=SIPP_2023_HEAD_START_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_HEAD_START_DONOR_SIZE_BYTES,
    )
    base_frame = with_us_sipp_head_start_input(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        sipp_donor=head_start_donor,
    )
    head_start_gate = us_sipp_head_start_signal_gate(base_frame)
    if not head_start_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "sipp_head_start_gate",
                status="failed",
                message="SIPP Head Start signal gate failed.",
                failures=list(head_start_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SIPP Head Start signal failed: {failure}"
                for failure in head_start_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "ssi_take_up",
            message=(
                "Assigning SSI take-up once (microcosm#469): ASEC reporter "
                "anchors plus a seeded Bernoulli draw at the registry's SSA "
                "age-band priors."
            ),
            prior_weight_basis=(
                "current_frame"
                if ssi_take_up_prior_basis is None
                else dict(ssi_take_up_prior_basis.provenance())
            ),
        )
    # One-shot assignment before any target materialization (microcosm#469).
    # The priors derive from the same ledger-fed SSA band counts the weight
    # solve enforces as ordinary targets (microcosm#470), over either this
    # frame's capacities or the delivered-weight basis resolved at startup
    # (microcosm#507/#508); the flags are frozen from here and the
    # enforced-band delivery is gated on release weights.
    ssi_uncapped_amount = _ssi_person_uncapped_amount(
        base_frame,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
    )
    base_frame, ssi_take_up_stage_diagnostics = with_us_ssi_take_up(
        base_frame,
        uncapped_ssi=ssi_uncapped_amount,
        seed=args.seed,
        targets=ssi_band_targets,
        reporter_source_ids=ssi_reporter_source_ids,
        prior_basis=ssi_take_up_prior_basis,
    )
    ssi_take_up_gate = us_ssi_take_up_gate(
        ssi_take_up_stage_diagnostics, targets=ssi_band_targets
    )
    if not ssi_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "ssi_take_up_gate",
                status="failed",
                message="SSI take-up assignment gate failed.",
                failures=list(ssi_take_up_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SSI take-up failed: {failure}"
                for failure in ssi_take_up_gate.failures
            )
        )
    ssi_assignment_priors = _ssi_assignment_priors_from_diagnostics(
        ssi_take_up_stage_diagnostics
    )
    # Reconstruct the basis that actually generated the frozen flags from the
    # stage's own diagnostics (not from the CLI value), so the final
    # release-weight artifact documents the assignment as it happened.
    ssi_assignment_prior_basis = ssi_take_up_prior_basis_from_diagnostics(
        ssi_take_up_stage_diagnostics
    )
    # microcosm#507 sol review finding 2: the frozen SSI decisions and their
    # basis are materialization inputs. A retry with different flags MUST
    # miss the previous attempt's target-frame checkpoint and target
    # materialization cache — otherwise the solve runs against the stale
    # SSI rows while the export carries the fresh ones (split-brain
    # certification).
    ssi_take_up_assignment_sha256 = _ssi_take_up_assignment_digest(
        base_frame,
        assignment_priors=ssi_assignment_priors,
        prior_basis=ssi_assignment_prior_basis,
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
            "sipp_vehicle_inputs",
            message=(
                "Imputing household vehicle count and value from the "
                "sha-pinned full SIPP 2023 donor."
            ),
        )
    sipp_vehicle_donor = load_sipp_2023_vehicle_donor(
        sipp_vehicle_donor_path,
        expected_sha256=SIPP_2023_VEHICLE_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_VEHICLE_DONOR_SIZE_BYTES,
    )
    base_frame = with_us_sipp_vehicle_inputs(
        base_frame,
        # The retired MicroImpute model pins both forests to seed 42.
        seed=42,
        time_period=PERIOD,
        sipp_donor=sipp_vehicle_donor,
    )
    sipp_vehicles_gate = us_sipp_vehicles_signal_gate(base_frame)
    if not sipp_vehicles_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "sipp_vehicles_gate",
                status="failed",
                message="SIPP-vehicle signal gate failed.",
                failures=list(sipp_vehicles_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SIPP-vehicle signal failed: {failure}"
                for failure in sipp_vehicles_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "voluntary_filing_input",
            message=(
                "Imputing measured voluntary tax-filing responses from the "
                "same sha-pinned full SIPP 2023 donor."
            ),
        )
    voluntary_filing_donor = load_sipp_2023_voluntary_filing_donor(
        sipp_vehicle_donor_path,
        expected_sha256=SIPP_2023_VOLUNTARY_FILING_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_VOLUNTARY_FILING_DONOR_SIZE_BYTES,
    )
    base_frame = with_us_voluntary_filing_input(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        sipp_donor=voluntary_filing_donor,
    )
    voluntary_filing_gate = us_voluntary_filing_signal_gate(base_frame)
    if not voluntary_filing_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "voluntary_filing_gate",
                status="failed",
                message="Voluntary-filing signal gate failed.",
                failures=list(voluntary_filing_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Voluntary-filing signal failed: {failure}"
                for failure in voluntary_filing_gate.failures
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
    if telemetry is not None:
        telemetry.stage(
            "snap_state_take_up",
            message=("Recalibrating SNAP take-up to FNS state household caseloads."),
        )
    base_frame, snap_state_take_up_diagnostics = _with_snap_state_take_up_outputs(
        base_frame,
        target_specs,
        seed=args.seed,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
    )
    snap_state_take_up_gate = us_snap_state_take_up_gate(snap_state_take_up_diagnostics)
    if not snap_state_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "snap_state_take_up_gate",
                status="failed",
                message="SNAP state take-up gate failed.",
                failures=list(snap_state_take_up_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SNAP state take-up failed: {failure}"
                for failure in snap_state_take_up_gate.failures
            )
        )
    if telemetry is not None:
        telemetry.stage(
            "other_health_insurance_inputs",
            message=(
                "Deriving the ASEC non-Medicare premium residual after ACA, "
                "CHIP, and Medicaid premiums, then imputing the PUF support."
            ),
        )
    base_frame = with_us_other_health_insurance_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
    )
    other_health_insurance_gate = us_other_health_insurance_signal_gate(base_frame)
    if not other_health_insurance_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "other_health_insurance_input_gate",
                status="failed",
                message="Other-health-insurance premium signal gate failed.",
                failures=list(other_health_insurance_gate.failures),
                force_upload=True,
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Other-health-insurance premium signal failed: " + failure
                for failure in other_health_insurance_gate.failures
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
        # Degraded pre-solve (PR #557 rounds 2-3): when the retirement
        # boundary already failed, this raise would supersede the specific
        # missing-leaf diagnosis. The gate object already rides
        # _release_gate_failures into the single terminal batch, so the
        # degraded branch simply does NOT raise — no duplicate append — and
        # its telemetry is guarded so a reporting crash cannot mask the
        # pending diagnosis (the #547 secure-before-report pattern). A green
        # run keeps today's fail-fast raise.
        if early_terminal_gate_failures:
            try:
                if telemetry is not None:
                    telemetry.stage(
                        "input_mass_reference_gate",
                        status="failed",
                        message="Base-frame input mass parity gate failed.",
                        failures=list(input_mass_reference_gate.failures),
                        force_upload=True,
                    )
            except Exception as error:
                early_terminal_gate_failures.append(
                    "Input-mass gate failure telemetry crashed in degraded "
                    f"mode; recorded instead of masking the diagnosis: {error}"
                )
        else:
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
        # Same degraded contract as the input-mass gate above: the gate
        # object rides _release_gate_failures to the batch; no raise, no
        # duplicate append, guarded telemetry.
        if early_terminal_gate_failures:
            try:
                if telemetry is not None:
                    telemetry.stage(
                        "degenerate_input_gate",
                        status="failed",
                        message="Degenerate input signal gate failed.",
                        failures=list(degenerate_input_gate.failures),
                        force_upload=True,
                    )
            except Exception as error:
                early_terminal_gate_failures.append(
                    "Degenerate-input gate failure telemetry crashed in "
                    f"degraded mode; recorded instead of masking the "
                    f"diagnosis: {error}"
                )
        else:
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
    # No combined pre-solve raise: a degraded run continues through the
    # solve so calibration_diagnostics.json and the single terminal batch
    # exist (the #547/#548 evidence contract — compute is cheaper than a
    # destroyed failure record). The two gates above keep fail-fast raises
    # on otherwise-green runs only.
    if telemetry is not None:
        telemetry.stage(
            "ecps_parity_gate",
            message="Gating candidate layers against the incumbent eCPS reference.",
        )
    ecps_parity_gate = _ecps_parity_gate(base_frame)
    if not ecps_parity_gate.passed and not args.allow_ecps_parity_gaps:
        # Same degraded contract as the input-mass/degenerate gates above
        # (PR #557 round 3 finding 1): the pinned parity reference requires
        # the retirement leaves, so a missing-leaf frame fails HERE too and
        # an unconditional raise would supersede the retirement diagnosis
        # before the solve. The gate object rides _release_gate_failures
        # (as enforced_ecps_parity_gate) into the terminal batch; degraded
        # runs continue, green runs keep the fail-fast raise.
        if early_terminal_gate_failures:
            try:
                if telemetry is not None:
                    telemetry.stage(
                        "ecps_parity_gate",
                        status="failed",
                        message="eCPS parity gate failed.",
                        failures=list(ecps_parity_gate.failures),
                        force_upload=True,
                    )
            except Exception as error:
                early_terminal_gate_failures.append(
                    "eCPS parity gate failure telemetry crashed in degraded "
                    f"mode; recorded instead of masking the diagnosis: {error}"
                )
        else:
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
    selection_mass_protections = tuple(
        dict.fromkeys(args.selection_mass_protection or ())
    )
    selection_mass_protection_specs: tuple[TargetSpec, ...] = ()
    if selection_mass_protections:
        # #445: injected AFTER the target-parity contract ran on the compiled
        # feed registry (these are run-scoped builder targets, not feed
        # families) and BEFORE the target frame materializes, so both the
        # fresh and checkpoint-reload paths compile them. The identity below
        # carries the protection list, so a column-less checkpoint misses.
        selection_mass_protection_specs = _selection_mass_protection_specs(
            base_frame, selection_mass_protections
        )
        target_specs = (*target_specs, *selection_mass_protection_specs)
    target_frame_checkpoint_identity = _target_frame_checkpoint_identity(
        base_dataset_sha256=base_dataset_sha256,
        policyengine_us_version=policyengine_us_version,
        seed=args.seed,
        target_period=PERIOD,
        target_registry_version=active_target_registry.version,
        weeks_unemployed_source_sha256=(ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256),
        congressional_district_vintage_crosswalk_sha256=(
            congressional_district_vintage_crosswalk_metadata or {}
        ).get("sha256"),
        ssi_take_up_assignment_sha256=ssi_take_up_assignment_sha256,
        selection_identities_sha256=(
            None if selection_source is None else selection_source.identities_sha256
        ),
        selection_mass_protections=selection_mass_protections,
        ssi_take_up_prior_weight_basis_sha256=(
            None
            if ssi_take_up_prior_basis is None
            else ssi_take_up_prior_basis.source_sha256
        ),
    )
    target_frame_materializer_identity_sha256 = _target_frame_checkpoint_digest(
        target_frame_checkpoint_identity
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
            "weeks_unemployed_source_sha256": (
                ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256
            ),
            "build_commit": full_commit,
            "policyengine_us_version": policyengine_us_version,
            "seed": args.seed,
            "target_period": PERIOD,
            "target_registry_version": active_target_registry.version,
            "congressional_district_vintage_crosswalk_sha256": (
                congressional_district_vintage_crosswalk_metadata or {}
            ).get("sha256"),
            # Conservative on purpose: a changed SSI assignment invalidates
            # every cached materialized column, not only the SSI-dependent
            # ones — correctness over cache warmth (microcosm#507/#508).
            "ssi_take_up_assignment_sha256": ssi_take_up_assignment_sha256,
            # Reform vectors are computed over the selected support. Thread
            # the source identity-set digest directly; the report manifest
            # intentionally does not carry this cache identity.
            "selection_identities_sha256": (
                None if selection_source is None else selection_source.identities_sha256
            ),
            # Reform caches store absolute income-tax vectors, which are
            # subtracted from a freshly materialized baseline. Bind both sides
            # to one complete materializer identity (microcosm#557).
            "target_frame_materializer_identity_sha256": (
                target_frame_materializer_identity_sha256
            ),
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
    if args.exact_k is not None and args.exact_k > candidate_households:
        raise ValueError(
            f"k={args.exact_k} exceeds the pool size {candidate_households}; "
            "ladder selection never clamps the requested cardinality."
        )
    full_pool_calibration = bool(
        args.dense_default_dataset
        or (args.exact_k is not None and args.exact_k == candidate_households)
    )
    l0_refit_lambda = (
        None
        if full_pool_calibration
        else args.l0_refit_lambda_share / float(candidate_households)
    )
    target_loss_weights = _fiscal_target_loss_weights(
        registry, args.target_family_loss_multipliers
    )
    target_loss_basis = (
        _fiscal_target_loss_basis(
            registry,
            target_loss_weights,
            args.target_family_loss_multipliers,
        )
        if args.exact_k is not None
        else None
    )
    if telemetry is not None:
        telemetry.stage(
            "calibrating",
            message=(
                "Calibrating dense household weights."
                if full_pool_calibration
                else (
                    "Selecting an exact-k Sampford support and refitting "
                    "household weights."
                    if args.exact_k is not None
                    else (
                        "Selecting sparse L0 support and refitting household weights."
                    )
                )
            ),
            default_dataset_method=(
                ("full_pool_refit" if args.exact_k is not None else "dense_no_l0")
                if full_pool_calibration
                else (
                    "exact_k_sampford_refit" if args.exact_k is not None else "l0_refit"
                )
            ),
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_weight_ratio=args.max_weight_ratio,
            target_family_loss_multipliers=(
                dict(args.target_family_loss_multipliers) or None
            ),
            n_targets=len(registry),
            n_candidate_households=candidate_households,
            l0_refit_lambda_share=(
                None if full_pool_calibration else float(args.l0_refit_lambda_share)
            ),
            l0_lambda=l0_refit_lambda,
            l2_lambda=float(args.l2_lambda),
            refit_l2_lambda=(
                None
                if full_pool_calibration
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
    ladder_outcome = None
    exact_k_puf_tail_gate: GateResult | None = None
    if args.exact_k is not None:
        if (
            pool_original_household_ids is None
            or pool_original_household_weights is None
        ):
            raise RuntimeError("Exact-k calibration lost its original pool baseline.")
        _assert_exact_k_original_pool_alignment(
            target_frame,
            household_ids=pool_original_household_ids,
            household_weights=pool_original_household_weights,
        )
        if l0_refit_lambda is None:
            # The full-pool branch does not use L0, but the shared function's
            # ignored value stays finite for a single, explicit call shape.
            ladder_l0_lambda = float(args.l0_refit_lambda_share) / float(
                candidate_households
            )
        else:
            ladder_l0_lambda = float(l0_refit_lambda)
        ladder_outcome = calibrate_exact_k_ladder(
            target_frame,
            registry.to_target_set(),
            k=args.exact_k,
            pi_hi=args.exact_k_pi_hi,
            seed=args.seed,
            epochs=args.epochs,
            refit_epochs=args.epochs,
            learning_rate=args.learning_rate,
            max_weight_ratio=args.max_weight_ratio,
            mass="conserve",
            l0_lambda=ladder_l0_lambda,
            l2_lambda=args.l2_lambda,
            refit_l2_lambda=args.refit_l2_lambda,
            target_loss_weights=target_loss_weights,
            target_loss_cap=US_FISCAL_TARGET_LOSS_CAP,
            warm_start_weights=warm_start_weights,
            progress_callback=(
                telemetry.calibration_progress if telemetry is not None else None
            ),
        )
        assert_exact_k_realized_count(ladder_outcome, args.exact_k)
        result = ladder_outcome.result
        exact_k_puf_tail_gate = _exact_k_puf_tail_support_gate(
            target_frame,
            ladder_outcome.support,
        )
        early_terminal_gate_failures.extend(
            f"Exact-k PUF capital-gains tail failed: {failure}"
            for failure in exact_k_puf_tail_gate.failures
        )
        default_dataset = {
            "method": (
                "full_pool_refit"
                if args.exact_k == candidate_households
                else "exact_k_sampford_refit"
            ),
            "sparse": args.exact_k < candidate_households,
            "n_candidate_households": candidate_households,
            "n_selected_households": int(args.exact_k),
            "n_exported_households": int(result.frame.n("household")),
            "l0_lambda_share": (
                None
                if args.exact_k == candidate_households
                else float(args.l0_refit_lambda_share)
            ),
            "l0_lambda": (
                None
                if args.exact_k == candidate_households
                else float(result.l0_lambda)
            ),
            "selection_epochs": (
                0 if args.exact_k == candidate_households else int(args.epochs)
            ),
            "refit_epochs": int(args.epochs),
            "selection_l2_lambda": (
                None if args.exact_k == candidate_households else float(args.l2_lambda)
            ),
            "refit_l2_lambda": float(
                args.l2_lambda if args.refit_l2_lambda is None else args.refit_l2_lambda
            ),
            "selection_final_loss": (
                None
                if args.exact_k == candidate_households
                else _finite_or_none(result.selection.final_loss)
            ),
            "refit_initial_loss": _finite_or_none(result.initial_loss),
            "refit_final_loss": _finite_or_none(result.final_loss),
            "puf_capital_gains_tail_retention": {
                "passed": exact_k_puf_tail_gate.passed,
                "failures": list(exact_k_puf_tail_gate.failures),
                "details": dict(exact_k_puf_tail_gate.details),
            },
        }
    elif args.dense_default_dataset:
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
            # Same scrub as default_dataset["final_loss"] below: these losses
            # ride the diagnostics build payload, which serializes strict JSON
            # (allow_nan=False), and a non-finite loss is a BATCHED gate
            # failure the artifact must survive to report (microcosm#547).
            "selection_final_loss": _finite_or_none(result.selection.final_loss),
            "refit_initial_loss": _finite_or_none(result.initial_loss),
            "refit_final_loss": _finite_or_none(result.final_loss),
        }
    if telemetry is not None:
        telemetry.stage(
            "take_up_final_diagnostics",
            message=(
                "Applying release weights and measuring the frozen take-up "
                "assignments (report-only; microcosm#469)."
            ),
        )
    # SSI take-up was assigned once before target materialization
    # (microcosm#469): apply the release weights to the same support, measure
    # the frozen flags for the published diagnostics, and let the gap to the
    # SSA band counts ship in the scorecard as calibration's residual on the
    # #470 registry targets — like every other program's take-up miss.
    if full_pool_calibration:
        export_frame = _with_calibrated_weights(
            base_frame,
            np.asarray(result.weights, dtype=np.float64),
        )
    else:
        export_frame = _with_l0_refit_weights(base_frame, result)
    compilation = dict(compilation)
    final_uncapped_ssi = _ssi_person_uncapped_amount(
        export_frame,
        maximum_microsim_batch_size=args.maximum_microsim_batch_size,
    )
    ssi_take_up_diagnostics = dict(
        us_ssi_take_up_diagnostics(
            export_frame,
            uncapped_ssi=final_uncapped_ssi,
            seed=args.seed,
            targets=ssi_band_targets,
            assignment_priors=ssi_assignment_priors,
            prior_basis=ssi_assignment_prior_basis,
            reporter_source_ids=ssi_reporter_source_ids,
        )
    )
    # The delivered-weight measurement is written the moment it exists —
    # BEFORE any gate can raise — because this artifact is the retry's
    # prior basis and the forensic record. A simultaneous integrity and
    # delivery failure must still leave it on disk (microcosm#507 sol
    # review finding 3).
    write_us_ssi_take_up_diagnostics(
        ssi_take_up_diagnostics,
        release_dir / "us_ssi_take_up.json",
    )
    # Gate the persisted flags on the export frame, not just the stage
    # output: the Bernoulli-law recheck and anchor/envelope laws are
    # weight-safe, so any downstream transform that corrupted the frozen
    # decisions fails the build here instead of shipping (PR #477 review
    # finding 3). The SSA-count miss itself stays scorecard-only.
    final_ssi_take_up_gate = us_ssi_take_up_gate(
        ssi_take_up_diagnostics, targets=ssi_band_targets
    )
    # SSI gate failures join the #437 batched terminal gates instead of
    # raising here: an early raise destroyed the failed run's calibration
    # diagnostics and skipped every other gate group (microcosm#547). A law
    # violation additionally quarantines SSI-dependent evaluations below.
    ssi_law_degraded = not final_ssi_take_up_gate.passed
    if not final_ssi_take_up_gate.passed:
        # Failures enter the list BEFORE any reporting: the telemetry stage
        # performs local writes and must not be able to mask the gate
        # failure by raising (microcosm#547, confirm round 2 finding 1).
        early_terminal_gate_failures.extend(
            f"SSI take-up final measurement failed: {failure}"
            for failure in final_ssi_take_up_gate.failures
        )
        try:
            if telemetry is not None:
                telemetry.stage(
                    "ssi_take_up_final_gate",
                    status="failed",
                    message="SSI take-up final export-frame gate failed.",
                    failures=list(final_ssi_take_up_gate.failures),
                    force_upload=True,
                )
        except Exception as error:
            early_terminal_gate_failures.append(
                "SSI final-gate failure telemetry crashed; recorded instead "
                f"of masking the failure: {error}"
            )
    ssi_delivery_failures, ssi_take_up_delivery_gate_result = (
        _enforce_ssi_take_up_delivery(
            ssi_take_up_diagnostics,
            targets=ssi_band_targets,
            release_dir=release_dir,
            telemetry=telemetry,
            # The dense diagnostic arm fences its adult bands per the
            # microcosm#566/#567 fence adjudication; the sparse certified
            # arm passes no fences and keeps hard enforcement.
            enforcement_fences=(
                US_DENSE_SSI_TAKE_UP_ENFORCEMENT_FENCES
                if args.dense_default_dataset
                else None
            ),
        )
    )
    early_terminal_gate_failures.extend(ssi_delivery_failures)
    medicaid_take_up_diagnostics, medicaid_guard_failures = (
        _final_medicaid_diagnostics_or_quarantine(
            ssi_law_degraded=ssi_law_degraded,
            degraded=bool(early_terminal_gate_failures),
            evaluate=lambda: dict(
                _medicaid_diagnostics_for_existing_output(
                    export_frame,
                    target_specs,
                    seed=args.seed,
                    substitutions=medicaid_enrollment_substitutions,
                    maximum_microsim_batch_size=args.maximum_microsim_batch_size,
                )
            ),
        )
    )
    early_terminal_gate_failures.extend(medicaid_guard_failures)
    # Signal gates re-check the exported support: sparse selection can drop
    # rows, and a column nonconstant on the candidate base can flatten on the
    # selected support.
    try:
        health_input_gate = _health_input_signal_gate(export_frame)
    except Exception as error:
        # Degraded-mode guard (microcosm#547): record instead of masking; the
        # downstream gate-failure evaluation is itself guarded, so a None
        # gate cannot re-destroy the evidence. Green path raises as before.
        if not early_terminal_gate_failures:
            raise
        health_input_gate = None
        early_terminal_gate_failures.append(
            "Health-input signal evaluation crashed in degraded mode; "
            f"recorded instead of masking earlier failures: {error}"
        )
    try:
        reported_coverage_vintage_gate = us_reported_coverage_vintage_signal_gate(
            export_frame
        )
    except Exception as error:
        if not early_terminal_gate_failures:
            raise
        reported_coverage_vintage_gate = None
        early_terminal_gate_failures.append(
            "Reported-coverage vintage signal evaluation crashed in degraded "
            f"mode; recorded instead of masking earlier failures: {error}"
        )
    try:
        other_health_insurance_gate = us_other_health_insurance_signal_gate(
            export_frame
        )
    except Exception as error:
        if not early_terminal_gate_failures:
            raise
        other_health_insurance_gate = None
        early_terminal_gate_failures.append(
            "Other-health signal evaluation crashed in degraded mode; "
            f"recorded instead of masking earlier failures: {error}"
        )
    if (
        other_health_insurance_gate is not None
        and not other_health_insurance_gate.passed
    ):
        # Batched, not raised: an in-place raise here masked co-occurring
        # SSI failures and destroyed the diagnostics artifact (microcosm#547).
        early_terminal_gate_failures.extend(
            f"Other health insurance signal failed on the export frame: {failure}"
            for failure in other_health_insurance_gate.failures
        )
    if congressional_district_vintage_crosswalk_metadata is not None:
        compilation = {
            **compilation,
            "congressional_district_vintage_crosswalk": (
                congressional_district_vintage_crosswalk_metadata
            ),
        }
    default_dataset = {
        **default_dataset,
        "final_loss": _finite_or_none(result.final_loss),
    }
    timing["calibration_seconds"] = time.perf_counter() - calibration_started
    timing["elapsed_through_calibration_seconds"] = time.perf_counter() - build_started
    try:
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
    except Exception as error:
        # Degraded-mode guard (microcosm#547): telemetry writes locally and
        # sits in the corridor between SSI collection and the diagnostics
        # write. Green path raises as before.
        if not early_terminal_gate_failures:
            raise
        early_terminal_gate_failures.append(
            "Release-gates telemetry crashed in degraded mode; recorded "
            f"instead of masking earlier failures: {error}"
        )
    # The path travels separately so the fallback can null it: the
    # diagnostics writer re-hashes any non-None incumbent path, which would
    # replay the exact I/O failure the guard just caught (microcosm#547,
    # confirm round 2 finding 2).
    current_target_surface: Mapping[str, object] | None = None
    incumbent_diagnostics_path: Path | None = args.incumbent_diagnostics
    incumbent_loss_basis: Mapping[str, object] | None = None
    try:
        incumbent_payload = (
            pinned_incumbent_payload
            if pinned_incumbent_payload is not None
            else _load_incumbent_diagnostics_payload(args.incumbent_diagnostics)
        )
        if args.incumbent_diagnostics is not None:
            current_target_surface = diagnostics_payload(
                result,
                target_registry=registry,
            )["target_surface"]
            if (
                args.exact_k is not None
                and current_target_surface.get("sha256")
                != args.frozen_target_surface_sha256
            ):
                raise ValueError(
                    "Exact-k target surface does not match the frozen register: "
                    f"got {current_target_surface.get('sha256')}, expected "
                    f"{args.frozen_target_surface_sha256}."
                )
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
        incumbent_loss_basis = _incumbent_target_loss_basis(incumbent_payload)
    except Exception as error:
        # Degraded-mode guard (microcosm#547): with earlier terminal failures
        # pending, an incumbent load/validation crash must record a line and
        # fall back to the no-incumbent gate shape, not mask the failures
        # and destroy the diagnostics artifact. Green path raises as before.
        if not early_terminal_gate_failures:
            raise
        incumbent_diagnostics = {}
        incumbent_diagnostics_path = None
        incumbent_loss_basis = None
        early_terminal_gate_failures.append(
            "Incumbent diagnostics could not be loaded/validated in "
            f"degraded mode; recorded instead of masking earlier failures: "
            f"{error}"
        )
    exact_k_ladder_provenance: Mapping[str, object] | None = None
    exact_k_incumbent_fit_gate: GateResult | None = None
    if args.exact_k is not None:
        if (
            ladder_outcome is None
            or pool_manifest_payload is None
            or authenticated_pool_h5 is None
            or exact_k_puf_tail_gate is None
        ):
            raise RuntimeError(
                "Exact-k calibration lost its pool or selection receipt."
            )
        if current_target_surface is None:
            current_target_surface = diagnostics_payload(
                result,
                target_registry=registry,
            )["target_surface"]
        exact_k_incumbent_fit_gate = _exact_k_frozen_register_fit_gate(
            result,
            incumbent_diagnostics,
            target_registry=registry,
            target_loss_weights=target_loss_weights,
            configured_loss_basis=target_loss_basis,
            incumbent_loss_basis=incumbent_loss_basis,
        )
        exact_k_ladder_provenance = _exact_k_ladder_manifest_payload(
            args=args,
            outcome=ladder_outcome,
            pool_manifest=pool_manifest_payload,
            authenticated_pool_h5=authenticated_pool_h5,
            ledger_artifact=ledger_artifact.provenance(),
            target_surface=current_target_surface,
            target_loss_basis=target_loss_basis,
            incumbent_diagnostics_sha256=pinned_incumbent_sha256,
            incumbent_fit_gate=exact_k_incumbent_fit_gate,
            puf_tail_gate=exact_k_puf_tail_gate,
        )
    enforced_input_mass_reference_gate = (
        None if args.allow_input_mass_drift else input_mass_reference_gate
    )
    enforced_ecps_parity_gate = (
        None if args.allow_ecps_parity_gaps else ecps_parity_gate
    )
    try:
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
            reported_coverage_vintage_gate=reported_coverage_vintage_gate,
            snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
            target_registry=registry,
        )
    except Exception as error:
        # Degraded-mode guard (microcosm#547): the batch must still form and
        # the diagnostics artifact must still be written when earlier
        # terminal failures are pending. Green path raises as before.
        if not early_terminal_gate_failures:
            raise
        gate_failures = []
        early_terminal_gate_failures.append(
            "Release gate evaluation crashed in degraded mode; recorded "
            f"instead of masking earlier failures: {error}"
        )
    # The early terminal failures (SSI gates, other-health signal, degraded-
    # mode guard lines) ride the same list as every other gate group, so the
    # diagnostics artifact records them and the terminal batch aborts on
    # them (microcosm#547).
    exact_k_fit_failures = (
        []
        if exact_k_incumbent_fit_gate is None
        else [
            f"Exact-k frozen-register fit failed: {failure}"
            for failure in exact_k_incumbent_fit_gate.failures
        ]
    )
    gate_failures = [
        *early_terminal_gate_failures,
        *exact_k_fit_failures,
        *gate_failures,
    ]
    _write_release_calibration_diagnostics(
        result=result,
        release_dir=release_dir,
        registry=registry,
        base_dataset_sha256=base_dataset_sha256,
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
        reported_coverage_vintage_gate=reported_coverage_vintage_gate,
        snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
        support_value_repairs={
            "social_security_components": social_security_component_repair,
            "non_sch_d_capital_gains": non_sch_d_cgd_repair,
        },
        warm_start_calibration=warm_start_calibration,
        selection_source=selection_source_payload,
        audit_export_targets=bool(args.audit_export_targets),
        gate_failures=gate_failures,
        timing=timing,
        incumbent_diagnostics_path=incumbent_diagnostics_path,
        incumbent_diagnostics_sha256=pinned_incumbent_sha256,
        incumbent_diagnostics=incumbent_diagnostics,
        default_dataset=default_dataset,
        degenerate_input_gate=degenerate_input_gate,
        ecps_parity_gate=ecps_parity_gate,
        validation_input_coverage_gate=validation_input_coverage_gate,
        target_loss_family_multipliers=args.target_family_loss_multipliers,
        target_loss_basis=target_loss_basis,
        exact_k_ladder=exact_k_ladder_provenance,
    )
    # Terminal-gate batching: evaluate EVERY terminal gate
    # group and raise once with the full failure list, instead of aborting at
    # the first failing group. Build M attempts 9/10/11 each burned a full
    # release run to surface one group (zero-support, then export parity,
    # then reform smoke) that a single run evaluates end to end. Green-path
    # behavior is unchanged; on failure, later groups still run (guarded so
    # an evaluation crash in degraded mode records a line rather than masking
    # the earlier failures) and certification manifests are never written.
    terminal_gate_failures: list[str] = list(gate_failures)
    terminal_batch_telemetry = _TerminalBatchTelemetry(
        telemetry,
        terminal_gate_failures,
    )
    terminal_batch_telemetry.attach_artifact(
        "calibration_diagnostics",
        release_dir / "calibration_diagnostics.json",
    )
    if gate_failures:
        terminal_batch_telemetry.stage(
            "release_gates",
            status="failed",
            message="Release gates failed; continuing to evaluate the "
            "remaining terminal gate groups before aborting.",
            failures=gate_failures,
            force_upload=True,
        )

    terminal_batch_telemetry.stage(
        "export_dataset",
        message="Writing PolicyEngine-US H5.",
    )
    release_engine = PolicyEngineUSEngine()
    # microcosm#368: full eCPS input-column coverage as a HARD release gate.
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
    try:
        input_coverage_gate = us_release_input_coverage_gate(
            export_frame, release_engine
        )
    except Exception as exc:
        # Degraded mode only: with earlier pre-export failures on record, a
        # coverage-evaluation crash becomes one more failure line instead of
        # masking them. On a clean run the exception propagates as before.
        if not terminal_gate_failures:
            raise
        terminal_gate_failures.append(
            "Input coverage failed: evaluation error under earlier gate "
            f"failures: {type(exc).__name__}: {exc}"
        )
        input_coverage_gate = None
    if input_coverage_gate is not None:
        input_coverage_failed = (
            not input_coverage_gate.passed and not args.allow_input_coverage_gaps
        )
        if input_coverage_failed:
            terminal_gate_failures.extend(
                f"Input coverage failed: {failure}"
                for failure in input_coverage_gate.failures
            )
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
        terminal_batch_telemetry.attach_artifact(
            "input_coverage",
            input_coverage_path,
        )
        if input_coverage_failed:
            terminal_batch_telemetry.stage(
                "export_dataset",
                status="failed",
                message="Release input-column coverage gate failed.",
                failures=list(input_coverage_gate.failures),
                force_upload=True,
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
    try:
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
            # Build H (microcosm#299): the two SOI-identified columns whose true
            # target level provably cannot sit inside the live-default reference
            # band (estate_income, non_sch_d_capital_gains). miscellaneous_income
            # and both mortgage columns are deliberately NOT excluded so the run
            # adjudicates their actual gate outcome. See the constant's band-math
            # rationale above.
            reviewed_exclusions=US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
        )
    except Exception as exc:
        # Same degraded-mode contract as the coverage gate above.
        if not terminal_gate_failures:
            raise
        terminal_gate_failures.append(
            "Input mass parity failed: evaluation error under earlier gate "
            f"failures: {type(exc).__name__}: {exc}"
        )
        export_input_mass_gate = None
    if export_input_mass_gate is not None:
        input_mass_parity_failed = (
            not export_input_mass_gate.passed and not args.allow_input_mass_drift
        )
        if input_mass_parity_failed:
            terminal_gate_failures.extend(
                f"Input mass parity failed: {failure}"
                for failure in export_input_mass_gate.failures
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
        terminal_batch_telemetry.attach_artifact(
            "input_mass_parity",
            input_mass_parity_path,
        )
        if input_mass_parity_failed:
            terminal_batch_telemetry.stage(
                "export_dataset",
                status="failed",
                message="Export input mass parity gate failed.",
                failures=list(export_input_mass_gate.failures),
                force_upload=True,
            )
    # microcosm#462: tail-concentration gate over the sparse QRF-imputed dollar
    # columns at the export's calibrated weights. The Build M defect — 89% of
    # the shipped non_sch_d_capital_gains mass in 100 records via a repeated
    # $594,484 donor-ceiling value — is invisible to support clipping (every
    # draw inside donor range), count targets (carrier count exact), and mass
    # parity (column excluded from the reference band), but is unmistakable as
    # top-k weighted-mass share.
    try:
        qrf_tail_exclusions = _load_qrf_tail_concentration_exclusions(
            args.qrf_tail_concentration_exclusions
        )
        qrf_tail_gate, qrf_tail_surface = _qrf_tail_concentration_gate(
            export_frame,
            reviewed_exclusions=qrf_tail_exclusions,
        )
        register_dormant = sorted(
            set(qrf_tail_exclusions)
            - set(qrf_tail_gate.details.get("reviewed_exclusions", ()))
        )
        if register_dormant:
            raise RuntimeError(
                "QRF tail-concentration exclusion register carries entries "
                "the checked surface did not use (column dense, thin, "
                f"absent, or below threshold): {register_dormant}. The "
                "per-run register must exactly match the concentrated "
                "columns — remove the stale entries."
            )
        qrf_tail_surface = {
            **qrf_tail_surface,
            "reviewed_exclusions_file": (
                str(args.qrf_tail_concentration_exclusions)
                if args.qrf_tail_concentration_exclusions is not None
                else None
            ),
            "reviewed_exclusions_sha256": (
                _sha256(args.qrf_tail_concentration_exclusions)
                if args.qrf_tail_concentration_exclusions is not None
                else None
            ),
            "reviewed_exclusions": dict(qrf_tail_exclusions),
        }
    except Exception as exc:
        # Same degraded-mode contract as the coverage gate above.
        if not terminal_gate_failures:
            raise
        terminal_gate_failures.append(
            "QRF tail concentration failed: evaluation error under earlier "
            f"gate failures: {type(exc).__name__}: {exc}"
        )
        qrf_tail_gate = None
        qrf_tail_surface = None
    if qrf_tail_gate is not None:
        qrf_tail_failed = (
            not qrf_tail_gate.passed and not args.allow_qrf_tail_concentration
        )
        if qrf_tail_failed:
            terminal_gate_failures.extend(
                f"QRF tail concentration failed: {failure}"
                for failure in qrf_tail_gate.failures
            )
        qrf_tail_path = release_dir / "qrf_tail_concentration.json"
        qrf_tail_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "enforced": not args.allow_qrf_tail_concentration,
                    "surface": qrf_tail_surface,
                    "tail_concentration": {
                        "passed": qrf_tail_gate.passed,
                        "failures": list(qrf_tail_gate.failures),
                        "details": dict(qrf_tail_gate.details),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        terminal_batch_telemetry.attach_artifact(
            "qrf_tail_concentration",
            qrf_tail_path,
        )
        if qrf_tail_failed:
            terminal_batch_telemetry.stage(
                "export_dataset",
                status="failed",
                message="QRF tail-concentration gate failed.",
                failures=list(qrf_tail_gate.failures),
                force_upload=True,
            )
    # Batched pre-export raise: the calibration battery, input coverage,
    # export-mass parity, and QRF tail concentration have ALL been evaluated
    # at this point, so one failed
    # run reports every failing pre-export group at once (Build M attempts 9
    # and 10 each burned a ~2h run to surface one of these groups serially).
    # The reform-coverage smoke and take-up contract keep their own raises
    # below: each already evaluates and reports its full failure set
    # internally, and both require the written H5 / export artifacts that a
    # gate-failed run must not produce.
    if terminal_gate_failures:
        # Evidence tier (microcosm#506): owners are resolved NOW so an
        # unowned failure refuses the export before the H5 and manifest work
        # below. A refused evidence attempt then falls through to the SAME
        # failed-run path as a certified gate failure — #568 weight-evidence
        # sidecar included — so no failed run ever loses its record-level
        # weight evidence.
        evidence_refusal: RuntimeError | None = None
        if args.evidence_release:
            try:
                _evidence_known_failures(
                    terminal_gate_failures, evidence_failure_owner_patterns
                )
            except RuntimeError as error:
                evidence_refusal = error
        if not args.evidence_release or evidence_refusal is not None:
            # Gate-failure path ONLY (microcosm#568 review): a batched
            # pre-export failure mints no H5, so the exact calibrated weight
            # vector — with the ordered household ids it aligns to, bound to
            # the target-frame identity — is persisted here as the run's only
            # record-level weight evidence. Green runs never write these files
            # (the certified H5 carries the weights); late gates (reform
            # smoke, take-up contract) raise after the H5 write, so their
            # failed runs retain weights in the written dataset itself.
            _write_final_household_weight_evidence(
                release_dir,
                export_frame,
                identity=target_frame_checkpoint_identity,
            )
            terminal_batch_telemetry.stage(
                "release_gates",
                status="failed",
                message="Release gates failed (batched pre-export report).",
                failures=terminal_gate_failures,
                force_upload=True,
            )
            if evidence_refusal is not None:
                raise evidence_refusal
            raise RuntimeError(
                "Release gates failed: " + "; ".join(terminal_gate_failures)
            )
        # The owned failures ride into the release manifest's known_failures
        # block instead of aborting the export; the H5 written below carries
        # the calibrated weights, so the sidecar is not written on this path.
        # Failures appended AFTER this point (a _TerminalBatchTelemetry crash
        # line, the smoke/take-up/coverage recordings) are owner-checked at
        # their own append sites and again before the manifest write; if one
        # is unowned the run dies post-H5 — weights retained in the written
        # dataset, per the #568 doctrine for late-gate failures — and no
        # manifest is minted.
        terminal_batch_telemetry.stage(
            "release_gates",
            status="failed",
            message=(
                "Release gates failed; --evidence-release continues to "
                "export with the failures recorded (microcosm#506)."
            ),
            failures=terminal_gate_failures,
            force_upload=True,
        )
    # A green run must not inherit a prior failed attempt's weight evidence
    # (microcosm#568 round 2): with --out/--release-id reuse, stale evidence
    # files would coexist with a certified release whose manifest knows
    # nothing about them. The batched gates have passed (or --evidence-release
    # is recording their failures), so any evidence present here belongs to a
    # superseded attempt — remove it before the release artifacts are written.
    for stale_evidence in (
        release_dir / FINAL_HOUSEHOLD_WEIGHTS_FILENAME,
        release_dir / FINAL_HOUSEHOLD_WEIGHT_IDS_FILENAME,
        release_dir / FINAL_HOUSEHOLD_WEIGHTS_METADATA_FILENAME,
    ):
        stale_evidence.unlink(missing_ok=True)
    dataset_path = artifact_root / dataset_filename
    # The export H5 write: everything below (reform smoke, take-up contract,
    # release manifest sha) reads THIS file, and it must be written only after
    # the batched pre-export raise so a gate-failed run never produces it.
    # microcosm#443: #437 dropped this call while inserting the batched raise,
    # so attempts 13/14 smoke-scored a stale artifact from a prior run.
    release_engine.write_dataset(export_frame, dataset_path, period=PERIOD)
    # microcosm#368: reform-coverage smoke on the WRITTEN release H5. The column
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
            smoke_failures = [
                f"Reform coverage smoke failed: {failure}"
                for failure in reform_coverage_smoke_gate.failures
            ]
            if not args.evidence_release:
                raise RuntimeError("Release gates failed: " + "; ".join(smoke_failures))
            # Evidence tier: the smoke verdict joins the recorded terminal
            # set (owner-checked immediately, so an unowned failure aborts
            # before further export work).
            terminal_gate_failures.extend(smoke_failures)
            _evidence_known_failures(
                terminal_gate_failures, evidence_failure_owner_patterns
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
    calibration_path = artifact_root / calibration_filename
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
        "census-acs-s0101-congressional-district-age-2024",
        "soi-congressional-district-2022",
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
    write_us_source_coverage_diagnostics(
        coverage, release_dir / "us_source_coverage.json"
    )
    if args.evidence_release:
        # The certified path surfaces a failed source-coverage gate at publish
        # (the contract requires gate.passed); the evidence contract relaxes
        # that verdict but BINDS gate.failures into known_failures — so an
        # evidence build must record them here, owner-checked immediately, or
        # its manifest would be unpublishable by construction.
        coverage_gate = coverage.get("gate") or {}
        coverage_gate_failures = [
            f"Source coverage failed: {failure}"
            for failure in (coverage_gate.get("failures") or ())
        ]
        if coverage_gate_failures:
            terminal_gate_failures.extend(coverage_gate_failures)
            _evidence_known_failures(
                terminal_gate_failures, evidence_failure_owner_patterns
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
    # us_ssi_take_up.json was already written the moment the final
    # measurement existed, ahead of the integrity and delivery gates
    # (microcosm#507/#508) — nothing mutates the dict in between.
    write_us_snap_state_take_up_diagnostics(
        snap_state_take_up_diagnostics,
        release_dir / "us_snap_state_take_up.json",
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
        stale_count_calibrated_failure = (
            "count-calibrated take-up column(s) "
            f"{stale_count_calibrated} ship at the engine default on the "
            "export frame despite the stage having run."
        )
        if not args.evidence_release:
            raise RuntimeError(
                "Release gates failed: " + stale_count_calibrated_failure
            )
        # Evidence tier: recorded like every other terminal verdict, with
        # the same immediate owner check.
        terminal_gate_failures.append(stale_count_calibrated_failure)
        _evidence_known_failures(
            terminal_gate_failures, evidence_failure_owner_patterns
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
        telemetry.attach_artifact(
            "us_ssi_take_up",
            release_dir / "us_ssi_take_up.json",
        )
        telemetry.attach_artifact(
            "us_snap_state_take_up",
            release_dir / "us_snap_state_take_up.json",
        )
        telemetry.stage("manifests", message="Writing release manifests.")
    timing["total_build_seconds"] = time.perf_counter() - build_started
    evidence_known_failures = None
    if args.evidence_release:
        if not terminal_gate_failures:
            raise RuntimeError(
                "Evidence release refused: every terminal gate passed. This "
                "artifact qualifies for the certified path — rerun without "
                "--evidence-release (the flag cannot mint a certified-shape "
                "manifest, and an evidence manifest with no known failures "
                "is invalid by contract)."
            )
        evidence_known_failures = _evidence_known_failures(
            terminal_gate_failures, evidence_failure_owner_patterns
        )
    _build_manifests(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        result=result,
        registry=registry,
        dropped=compilation,
        target_profile_gate=target_profile_gate,
        ssi_take_up_delivery_gate_result=ssi_take_up_delivery_gate_result,
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
        reported_coverage_vintage_gate=reported_coverage_vintage_gate,
        snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
        timing=timing,
        warm_start_calibration=warm_start_calibration,
        selection_source=selection_source_payload,
        ledger_artifact=ledger_artifact.provenance(),
        default_dataset=default_dataset,
        medicaid_enrollment_substitutions=medicaid_enrollment_substitutions,
        staging=_staging_manifest_block(telemetry),
        dataset_key=dataset_key,
        dataset_filename=dataset_filename,
        calibration_key=calibration_key,
        calibration_filename=calibration_filename,
        exact_k_ladder=exact_k_ladder_provenance,
        base_pool=base_pool_receipt,
        evidence_known_failures=evidence_known_failures,
    )
    if telemetry is not None:
        telemetry.attach_artifact("build_manifest", release_dir / "build_manifest.json")
        telemetry.attach_artifact(
            "release_manifest",
            release_dir / "release_manifest.json",
        )
        telemetry.complete()

    # Keep a copy of the exact base artifact beside diagnostics for local audit.
    _copy_base_h5_for_local_audit(
        base_h5,
        release_root / f"base_{base_h5.name}",
        authenticated_pool_h5=authenticated_pool_h5,
    )
    _print_build_result(
        release_id=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
    )


if __name__ == "__main__":
    main()
