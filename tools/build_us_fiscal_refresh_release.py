"""Build a contract-valid US fiscal refresh release from a Microcosm H5.

This is a narrow release builder for the Issue #40 fiscal target surface. It
starts from an existing Microcosm US H5, materializes the current structured
fiscal target rows, recalibrates only the household weights, writes a fresh
PolicyEngine-US H5, and emits the release contract files.

Each run is one build of the US national dataset (microcosm#1570), and a
build has no tier (microcosm#1585 item 11): it writes its H5 and calibration
weights to ``<out>/artifacts/`` and its own files (the diagnostics, the gate
evidence) to ``<out>/build/``, whatever release certification later places
it in.

The run builds and stops (microcosm#1585 item 11, PR 4a): once the H5, the
calibration weights, the diagnostics and the build's gate evidence are
written, it writes ``<out>/build_record.json`` and prints the commands that
certify the build, so the build can be evaluated first. Nothing it writes
is publishable. ``tools/certify_us_national_build.py`` certifies it and
chooses its tier: certified, into the national release
``microcosm-us-2024-national``, or, with ``--evidence-release``, evidence,
into the evidence release ``microcosm-us-2024-evidence``. It runs the
release's post-export checks on the written H5, places the build's files in
``<out>/releases/<release>/``, then writes both manifests, which name the
release and the build (the ``populace-us-2024-…`` build ID) in the
build-folder layout, and the ``microcosm-publish-release-build`` commands
that publish the build into its own folder of the release and then make it
the release's current build.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# microcosm#456 (#447 ops note): bound the default BLAS/OpenMP/joblib thread
# pools instead of inheriting machine geometry — per-thread scratch buffers
# scale the resident set with core count (measured 125 GB anon-RSS at 16 vCPU
# vs 249 GB at 32 vCPU for the same build). OpenBLAS and torch read these at
# library load, so this must precede the numpy/torch imports below. The one
# thread setting of the US build commands, MICROCOSM_THREADS, replaces the
# default of min(os.cpu_count(), 16), and variables already set keep their
# values (tools/us_build_thread_environment.py). This also runs on import, as
# it always has. The helper is loaded from this file's folder, which an import
# does not always have on its path. E402 is ignored for this file in pyproject
# for exactly this block.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from us_build_thread_environment import set_build_thread_environment

del sys.path[0]
set_build_thread_environment(default_count=min(os.cpu_count() or 1, 16))

import argparse
import dataclasses
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import pickle
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import tomllib
import traceback
import uuid
import weakref
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import nullcontext
from datetime import UTC, datetime
from functools import partial
from importlib.resources import files as package_files
from pathlib import PurePosixPath
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from microcosm_emitter_interface.telemetry import OuterProgress

from microcosm.build.calibration_epoch_telemetry import (
    CALIBRATION_EPOCH_EVERY,
    calibration_epoch_stride,
    forwards_calibration_epoch,
)
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.gates import (
    GateResult,
    TargetFitRequirement,
    default_valued_columns_gate,
    nonconstant_columns_gate,
    observed_value_counts,
    parity_gate,
    single_record_concentration,
    tail_concentration_gate,
    target_fit_gate,
    target_profile_coverage_gate,
)
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.run_outcome import classify_failure
from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage
from microcosm.build.staging_storage import confirm_and_report_final_commit
from microcosm.build.staging_v2 import (
    EMITTER_MAX_FILE_BYTES,
    MAX_MESSAGE_CHARACTERS,
    EventStatus,
    StagingContentPolicy,
    StagingRunBundleWriterV2,
    disabled_staging_delivery,
)
from microcosm.build.telemetry_emitter import (
    LocalTelemetryEmitter,
    start_local_telemetry_emitter_service,
)
from microcosm.build.termination import BuildTerminatedError, raise_on_sigterm
from microcosm.build.us_runtime import (
    ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256,
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    ORG_2024_DONOR_CONTENT_SHA256,
    SCF_2025_FULL_EXTRACT,
    SCF_2025_SUMMARY_EXTRACT,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
    SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
    SIPP_2023_TIP_DONOR_SHA256,
    SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
    SIPP_HEAD_START_SOURCE_COLUMNS,
    SIPP_SSI_DISABILITY_SOURCE_COLUMNS,
    SIPP_VEHICLE_SOURCE_COLUMNS,
    SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS,
    SOI_VARIABLE_MAP,
    US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
    US_FISCAL_TARGET_SOURCE_COLUMN_EXCLUSIONS,
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
    build_sipp_2023_financial_asset_donor,
    build_sipp_2023_head_start_donor,
    build_sipp_2023_ssi_disability_donor,
    build_sipp_2023_vehicle_donor,
    build_sipp_2023_voluntary_filing_donor,
    compile_us_fiscal_target_registry,
    default_congressional_district_vintage_crosswalk_path,
    fetch_asec_2023_weeks_unemployed_source,
    fetch_org_2024_donor,
    fetch_scf_2022_full_extract,
    fetch_scf_2022_summary_extract,
    fetch_scf_public_file,
    fetch_sipp_2023_financial_asset_donor,
    fetch_sipp_2023_tip_donor,
    hard_target_package_aliases,
    load_asec_2023_weeks_unemployed_source,
    load_congressional_district_vintage_crosswalk,
    load_org_2024_donor,
    load_scf_2022_auto_loan_donor,
    load_scf_2022_financial_asset_donor,
    load_scf_2025_education_debt_donor,
    load_sipp_2023_tip_donor,
    read_sipp_december_person_rows,
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
    us_esi_premiums_anchor_feed_mismatches,
    us_esi_premiums_anchor_gate,
    us_esi_premiums_signal_gate,
    us_farm_business_income_signal_gate,
    us_fiscal_target_exclusion_receipt,
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
    us_scf_education_debt_fit_table,
    us_scf_education_debt_signal_gate,
    us_scf_wealth_signal_gate,
    us_sipp_head_start_signal_gate,
    us_sipp_tips_signal_gate,
    us_sipp_vehicles_signal_gate,
    us_snap_discretionary_exemption_signal_gate,
    us_snap_state_take_up_gate,
    us_snap_take_up_signal_gate,
    us_source_coverage_diagnostics,
    us_source_operation_handlers,
    us_spm_independence_role_signal_gate,
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
    with_us_scf_education_debt_inputs,
    with_us_scf_wealth_inputs,
    with_us_sipp_head_start_input,
    with_us_sipp_tip_inputs,
    with_us_sipp_vehicle_inputs,
    with_us_snap_discretionary_exemption_inputs,
    with_us_snap_state_take_up,
    with_us_snap_take_up_inputs,
    with_us_spm_independence_role,
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
from microcosm.build.us_runtime.acs_release_predictors import (
    join_acs_release_predictors,
)
from microcosm.build.us_runtime.battery_bindings import (
    us_flag_overrides,
    us_gate_registry,
    us_gate_specs,
)
from microcosm.build.us_runtime.chronicle_feed import (
    USChronicleFeed,
    load_us_chronicle_feed,
    require_us_chronicle_feed_row_contracts,
)
from microcosm.build.us_runtime.demographics import (
    CENSUS_NATIONAL_AGE_BENCHMARK,
    demographics_payload,
    geography_coverage_payload,
    population_by_age_from_sim,
    write_demographics,
)
from microcosm.build.us_runtime.engine_lifecycle import (
    EngineVariableModuleScope,
    release_engine_simulation,
)
from microcosm.build.us_runtime.exact_k_ladder import (
    ExactKLadderCalibration,
    assert_exact_k_realized_count,
    calibrate_exact_k_ladder,
    exact_k_ladder_manifest_payload,
)
from microcosm.build.us_runtime.fiscal_targets import (
    AGE_BOUND_STAMP_FROM_CONSTRAINT_ROWS,
    AGE_BOUND_STAMP_SOURCE_KEY,
    SSA_SSI_AGE_BAND_RECIPIENTS_TARGET_ROLE,
)
from microcosm.build.us_runtime.fsa_student_loan_targets import (
    FSA_BALANCE_LOWER_BOUND_KEY,
    FSA_BALANCE_UPPER_BOUND_KEY,
    FSA_BOUND_TABLES,
    FSA_MATERIALIZER,
    FSA_SOURCE_NAME,
    FSA_TARGET_FAMILY,
    FsaTargetPlan,
    fsa_student_loan_household_values,
    fsa_student_loan_target_receipt,
    plan_fsa_student_loan_targets,
)
from microcosm.build.us_runtime.gate_scopes import us_scoped_gate_manifest
from microcosm.build.us_runtime.gate_shadow import (
    GATE_BATTERY_SHADOW_DIRECTORY,
    GateBatteryShadow,
    LegacyVerdict,
    arm_gate_shadow,
    disarm_gate_shadow,
    shadow_failures,
    shadow_file_paths,
    shadow_gate,
    shadow_not_run,
    shadow_value,
    tool_gate_shadows_enabled,
)
from microcosm.build.us_runtime.h5_io import (
    AuthenticatedPoolH5,
    assert_h5_unchanged,
    identify_us_multispine_pool_manifest,
    load_authenticated_us_multispine_pool_for_release,
    load_simulation_ready_us_multispine_pool,
    refuse_denied_frame,
    refuse_denied_pool_h5,
    refuse_denied_pool_h5_digest,
    require_authenticated_us_multispine_pool_h5,
    us_multispine_pool_release_receipt,
)
from microcosm.build.us_runtime.household_batches import (
    INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
    assert_group_entities_nest_in_households,
    assert_medicaid_claiming_tax_units_local,
    automatic_gc_suspended,
    checked_batch_engine,
    collect_batch_garbage,
    collect_pass_garbage,
    group_to_household_positions,
    household_batch_plan,
    household_position_batches,
    record_engine_branches,
    refuse_batch_population_aggregates,
    select_households_by_position,
)
from microcosm.build.us_runtime.input_mass import (
    us_input_mass_parity_gate,
    us_input_mass_totals,
    us_line_32_carriers,
    us_line_32_on_reference_terms,
)
from microcosm.build.us_runtime.l0_refit_export import (
    attach_l0_refit_entity_weights,
    load_us_frame,
)
from microcosm.build.us_runtime.medicare_irmaa_holdout import (
    medicare_irmaa_holdout_from_sim,
    medicare_irmaa_holdout_unavailable,
    write_medicare_irmaa_holdout,
)
from microcosm.build.us_runtime.nonzero_shares import us_nonzero_shares
from microcosm.build.us_runtime.parity_reference import (
    EcpsParityReference,
    ParityKnownGap,
    load_ecps_parity_known_gaps,
    load_ecps_parity_reference,
)
from microcosm.build.us_runtime.population_structure_holdout import (
    population_structure_holdout_from_h5,
    population_structure_holdout_unavailable,
    write_population_structure_holdout,
)
from microcosm.build.us_runtime.post_export_requests import (
    RECORDS_REQUESTS_ONLY,
    PostExportRequest,
    check_post_export_reform,
)
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    assert_puf_capital_gains_tail_survives_selection,
)
from microcosm.build.us_runtime.reform_validation import (
    US_RELEASE_SPM_SELECTION,
    default_baseline_level_specs,
    load_default_reform_specs,
    reform_validation_payload,
    write_reform_validation,
)
from microcosm.build.us_runtime.release_gate_dry_run import (
    DEFAULT_L0_TAIL_SHARE_MARGIN,
    DEFAULT_MASS_DRIFT_MARGIN,
    DEFAULT_SUPPORT_CARRIER_RETENTION,
    DEFAULT_SUPPORT_NONZERO_SHARE_MARGIN,
    DEFAULT_TAIL_SHARE_FALL_MARGIN,
    DEFAULT_TAIL_SHARE_RISE_MARGIN,
    DryRunMargins,
    ReleaseDryRunReport,
    apply_evidence_ownership,
    bound_zero_support_on_l0,
    certain_lines_check,
    degenerate_input_register_check,
    ecps_parity_register_check,
    export_input_mass_check,
    export_signal_regrades_check,
    input_coverage_register_check,
    not_previewable_check,
    post_stop_error_report,
    pre_solve_refusal_report,
    qrf_tail_register_check,
    qrf_tail_register_unloadable_check,
)
from microcosm.build.us_runtime.release_gate_preflight import (  # noqa: TC001
    CheckResult,
    check_zero_support_preview,
)
from microcosm.build.us_runtime.release_input_coverage import (
    REFERENCE_ECPS_LAYER_RENAMES,
    clear_installed_list_parameter_baseline_cache,
    project_ecps_parity_known_gap_names,
    us_release_input_coverage_signal_counts,
)
from microcosm.build.us_runtime.ssi_social_security_holdout import (
    ssi_social_security_holdout_from_sim,
    ssi_social_security_holdout_unavailable,
    write_ssi_social_security_holdout,
)
from microcosm.build.us_runtime.ssi_take_up import (
    US_SSI_TAKE_UP_AGE_TARGETS,
    US_SSI_TAKE_UP_ENFORCED_BAND_KEYS,
    US_SSI_TAKE_UP_OUTPUT_COLUMNS,
    SSITakeUpPriorBasis,
)

# The target-loss weights live in one shared module, which the ACS local
# release calls too. These names stay importable from this tool for the
# scorers and the bundle contract that read them off it.
from microcosm.build.us_runtime.target_loss_weights import (  # noqa: F401
    US_FISCAL_TARGET_CONCEPT_METADATA_EXCLUSIONS,
    US_FISCAL_TARGET_LOSS_WEIGHTING,
    US_FISCAL_TARGET_VALUE_WEIGHT_POWER,
    parse_target_family_loss_multipliers,
)
from microcosm.build.us_runtime.target_loss_weights import (
    fiscal_target_loss_weights as _fiscal_target_loss_weights,
)
from microcosm.build.us_runtime.target_loss_weights import (  # noqa: F401
    fiscal_target_value_basis as _fiscal_target_value_basis,
)
from microcosm.build.us_runtime.warm_start_selection import (
    DEFAULT_SELECTION_JOIN_KEY,
    SELECTION_MODES,
    load_selection_source_from_h5,
    load_selection_source_from_manifest,
    select_frozen_support,
)
from microcosm.calibrate import (
    CalibrationResult,
    L0RefitResult,
    TargetRegistry,
    TargetSpec,
    calibrate,
    calibrate_l0_refit,
    relative_error_loss,
    without_constraint_matrix,
)
from microcosm.calibrate.diagnostics import (
    target_surface_payload,
    write_calibration_diagnostics,
)
from microcosm.data import compatibility_recertification as recert
from microcosm.data.build_folder_manifests import (
    publication_commands,
    rewrite_release_dir,
)
from microcosm.data.contract import (
    EVIDENCE_RELEASE_ID_SEGMENT,
    EVIDENCE_RELEASE_MANIFEST_SCHEMA_VERSION,
    US_EVIDENCE_RELEASE_ID,
    US_FISCAL_STORED_INPUTS_STAGE,
    US_NATIONAL_RELEASE_ID,
    US_RELEASES,
    US_STORED_INPUTS_EVIDENCE_KEY,
    US_STORED_INPUTS_EVIDENCE_SCHEMA_VERSION,
    stored_input_evidence_manifest_block,
)
from microcosm.data.stored_inputs import (
    US_STORED_NON_VARIABLE_COLUMNS,
    StoredTableLayoutError,
    h5_stored_tables,
    installed_us_engine,
    is_model_named,
    register_sha256,
    stored_input_failures,
    undefined_stored_inputs,
)
from microcosm.data.us_critical_targets import (
    US_CRITICAL_TARGET_IMPROVEMENT_MAX_ABS_RELATIVE_ERROR,
    US_EXACT_CRITICAL_TARGET_FIT_REQUIREMENTS,
    is_congressional_district_target,
)
from microcosm.data.us_critical_targets import (
    US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT as SHARED_US_SOI_TABLE_1_4_NATIONAL_DOLLAR_FIT_REQUIREMENT,
)
from microcosm.diagnostics import (
    CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
    DiagnosticsWriteFailure,
    DiagnosticsWriteOutcome,
    DiagnosticsWriteSuccess,
)
from microcosm.frame import Frame, MassChange, WeightKind, Weights, read_frame_table
from microcosm.frame.adapters.policyengine_us import (
    PolicyEngineUSEngine,
    PolicyEngineUSVariableMetadataIndex,
    us_dataset_through,
)
from microcosm.frame.units import US_SCHEMA

PERIOD = 2024
REPO_ID = "policyengine/populace-us"
STAGING_REPO_ID = "policyengine/populace-us-staging"
#: The dry-run flag (see ``_ReleaseDryRun``), and each margin override with its
#: default and meaning, paired with the ``DryRunMargins`` field it sets.
_DRY_RUN_REPORT_FLAG = "--dry-run-gates-report"
_DRY_RUN_MARGIN_FLAGS: tuple[tuple[str, float, str], ...] = (
    (
        "--dry-run-tail-share-rise-margin",
        DEFAULT_TAIL_SHARE_RISE_MARGIN,
        "how far calibration may raise a QRF column's top-k share",
    ),
    (
        "--dry-run-tail-share-fall-margin",
        DEFAULT_TAIL_SHARE_FALL_MARGIN,
        "how far calibration may lower a QRF column's top-k share",
    ),
    (
        "--dry-run-mass-drift-margin",
        DEFAULT_MASS_DRIFT_MARGIN,
        "distance from the export input-mass band edge inside which an "
        "in-band column is flagged",
    ),
    (
        "--dry-run-support-nonzero-share-margin",
        DEFAULT_SUPPORT_NONZERO_SHARE_MARGIN,
        "L0 path: how far a column's record nonzero share may move",
    ),
    (
        "--dry-run-support-carrier-retention",
        DEFAULT_SUPPORT_CARRIER_RETENTION,
        "L0 path: the smallest kept fraction of the records carrying one signal",
    ),
    (
        "--dry-run-l0-tail-share-rise-margin",
        DEFAULT_L0_TAIL_SHARE_MARGIN,
        "L0 path: how far an L0 selection and refit may raise a top-k share "
        "(unmeasured; the default admits any share)",
    ),
    (
        "--dry-run-l0-tail-share-fall-margin",
        DEFAULT_L0_TAIL_SHARE_MARGIN,
        "L0 path: how far an L0 selection and refit may lower a top-k share "
        "(unmeasured; the default admits any share)",
    ),
)
_DRY_RUN_MARGIN_FIELDS = (
    "tail_share_rise",
    "tail_share_fall",
    "mass_drift",
    "support_nonzero_share",
    "support_carrier_retention",
    "l0_tail_share_rise",
    "l0_tail_share_fall",
)
#: The release-manifest keys and file names of the dataset and calibration
#: weights a run writes: the names new builds of the national and evidence
#: releases declare (microcosm#1570).
DATASET_KEY = "microcosm_us_2024"
DATASET_FILENAME = "microcosm_us_2024.h5"
CALIBRATION_KEY = "microcosm_us_2024_calibration"
CALIBRATION_FILENAME = "microcosm_us_2024_calibration.npz"
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
# arm (--dense-default-dataset) therefore FENCES its adult bands — the
# under-18 pattern extended: the miss ships in the scorecard as a known
# boundary, never as an enforced contract and never as saturation-as-success.
# The L0+refit and --exact-k arms pass no fences and keep hard enforcement.
# RE-ADJUDICATES when microcosm#566's damped fixed-point protocol lands.
_US_DENSE_SSI_FENCE_ADJUDICATION = (
    "Fenced for the --dense-default-dataset arm (microcosm#566/#567): "
    "microcosm#508 delivered-weight recomputes have not landed the adult "
    "band pair in the envelope on either observed full-pool frame (P2, current-"
    "frame attempt then its one permitted recompute: +5.8%/+24.8% -> "
    "+8.2%/+20.0%; P3, attempts already anchored on delivered bases: "
    "65+ +34.6%, then +8.3%/+19.8% after recomputing again — a chain "
    "the microcosm#508 loader now refuses). Further recomputes are the "
    "deleted microcosm#463-class loop. This band's miss ships in the "
    "scorecard as a known boundary — never as an enforced contract. "
    "Re-adjudicates when the microcosm#566 damped fixed-point protocol "
    "lands. The L0+refit and --exact-k arms pass no fences and keep hard "
    "enforcement."
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
# 12: the identity now carries staged_frame_sha256, a digest of the staged
# frame handed to the materializer (microcosm#1018), so a staging change that
# forgets this bump can no longer reuse stale target columns. Checkpoints
# written without the digest cannot attest their staged frame.
# 13: the base simulation runs over household batches and its compilation
# carries a batching receipt (microcosm#956), so checkpoints written by the
# whole-pool base pass miss too.
TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION = 13
DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE = 5_000
DEFAULT_L0_REFIT_LAMBDA_SHARE = 0.8
DEFAULT_US_FISCAL_CALIBRATION_EPOCHS = 1_500


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
    "soi-historic-table-2-state-agi-2023",
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

#: Restated constraints, a third class distinct from both sets above: a
#: labelled feed re-expresses a fact's universe constraint in the Ledger's own
#: concept vocabulary (``us:statutes/26/62#adjusted_gross_income_lower_bound``)
#: alongside the compiled metadata the materializer actually slices on
#: (``agi_lower_bound``). These keys are NOT inert — they restrict the
#: microdata — so they can never join the supported set outright: a future
#: feed whose labelled bound disagreed with the compiled one would then be
#: silently ignored, materializing a wider population than the published cell
#: covers, which is the failure the guard exists to stop. Each key here is
#: accepted only per spec, and only when that spec also carries the filter the
#: materializer applies and the two select the same population; a restatement
#: that disagrees, or that has no compiled counterpart, stays fatal and the
#: refusal names the values. Values are the restated-constraint rule that
#: decides the comparison (see ``_restated_ledger_filter_refusal``).
RESTATED_LEDGER_FILTER_CONCEPTS = {
    "us:statutes/26/62#adjusted_gross_income": "agi_band",
    "us.tax.earned_income_credit_qualifying_children": "eitc_child_count",
    "age": "age_band",
    "federal_student_loan_balance": "student_loan_balance_band",
}

#: Bound-side suffixes a restated constraint key may carry. A key with no
#: suffix restates an exact value of the concept.
RESTATED_LEDGER_FILTER_BOUND_SIDES = (
    ("_lower_bound", "lower"),
    ("_upper_bound", "upper"),
)

#: Compiled metadata the materializer slices the AGI band on, by bound
#: side: the ``irs_soi`` loop in :func:`_materialize_target_frame` reads
#: both through :func:`_as_bound` into one half-open mask. A restated AGI
#: bound is accepted only against its own side; the sibling side is judged
#: by its own key.
RESTATED_AGI_BAND_COMPILED_KEYS = {
    "lower": "agi_lower_bound",
    "upper": "agi_upper_bound",
}

#: Qualifying-child counts probed when deciding whether a restated EITC
#: child-count bound selects the same returns as the compiled filter.
#: :func:`_eitc_child_count_mask` resolves every compiled value to ``== 0``,
#: ``== 1``, ``== 2`` or ``>= 3``, and a compared restatement is ``>= n`` or
#: ``== n`` (an upper bound is refused before any comparison; see
#: :data:`RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES`); evaluating both over
#: 0..16 separates every such pair, so equal masks over the probe mean equal
#: populations. Counts above the probe are refused rather than compared.
RESTATED_EITC_CHILD_COUNT_PROBE_MAX = 16

#: Bound sides on which a restated EITC qualifying-child key is refused
#: outright, whatever its value and whatever the spec compiles. What an upper
#: bound selects turns on the Ledger's operator — ``< 1`` is the childless
#: returns, ``<= 1`` adds the one-child returns — and that operator is
#: unconfirmed for count constraints. The metadata key does not settle it:
#: :func:`microcosm.build.ledger_targets._constraint_bound_filters` stamps a
#: ``<`` row as ``_upper_bound`` (``<=`` becomes ``_upper_bound_inclusive``,
#: which no restated concept names, so it is refused by bare key), but the
#: dimension stamp in ``_ledger_metadata`` runs first and wins through
#: ``setdefault``, and can carry the same key with no operator at all. So no
#: reading is guessed until the Ledger confirms one (Max, 2026-09-22). Lower
#: and exact restatements stay under the agreement rule.
RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES = frozenset({"upper"})

#: Compiled metadata the materializer slices a person-age band on, by bound
#: side. Both age paths in :func:`_materialize_target_frame` read these
#: through :func:`_as_bound` into one half-open mask ``lower <= age < upper``
#: — the ``population_age`` loop in :func:`_population_age_household_values`
#: and the age-banded ``policyengine_variable`` branch (the SSA SSI
#: recipients-by-age counts) — so a restated age bound is judged, like an AGI
#: bound, against its own side only.
#:
#: Numeric equality of the edges is equality of populations only for a
#: restated key whose operator is the mask's: ``age < 10`` and ``age <= 10``
#: differ by everyone aged exactly ten, so the operator decides.
#: :func:`microcosm.build.ledger_targets._constraint_bound_filters` writes
#: ``age_lower_bound`` only for a ``>=`` row and ``age_upper_bound`` only for
#: a ``<`` row, the two operators of the materializer's mask. A ``>`` or
#: ``<=`` row is stamped ``age_lower_bound_exclusive`` /
#: ``age_upper_bound_inclusive``, which no restated concept names, so it stays
#: refused by bare key — as it must, because the compiled bound comes from
#: ``us_runtime.fiscal_targets._age_bounds``, which drops the operator: a
#: ``<= 4`` row compiles to ``age_upper_bound=4`` and the materializer's
#: ``age < 4`` would leave out the four-year-olds the published cell counts.
#: But the same ``ledger_filter_age_{lower,upper}_bound`` key can also be a
#: dimension's, which carries no operator (the ambiguity that keeps restated
#: qualifying-child upper bounds refused), so the key name alone does not
#: settle it; the compile's attestation does
#: (:data:`microcosm.build.us_runtime.fiscal_targets.AGE_BOUND_STAMP_SOURCE_KEY`).
RESTATED_AGE_BAND_COMPILED_KEYS = {
    "lower": "age_lower_bound",
    "upper": "age_upper_bound",
}

#: Materializers that apply the compiled age band, and so the only ones for
#: which ignoring an agreeing age restatement changes nothing. Both are read
#: in :func:`_materialize_target_frame`: ``population_age`` through
#: :func:`_population_age_household_values`, and ``policyengine_variable``
#: through its age-banded branch, entered whenever either compiled age key is
#: present. Every other materializer (the ``irs_soi`` slice, the direct
#: household-variable measures) never reads an age bound, so a restated age
#: bound there would be silently ignored however well it agreed with metadata
#: the materializer also ignores; it is refused instead.
RESTATED_AGE_BAND_MATERIALIZERS = frozenset(
    {"population_age", "policyengine_variable", FSA_MATERIALIZER}
)

#: Compiled edges of a Federal Student Aid debt-size band, by bound side, and
#: the one materializer that slices on them
#: (:func:`microcosm.build.us_runtime.fsa_student_loan_targets.fsa_student_loan_household_values`,
#: ``lower <= balance < upper`` on the person's own plus Parent PLUS balance,
#: microcosm#1074). A restated ``federal_student_loan_balance`` bound on any
#: other materializer would be ignored, so it refuses.
RESTATED_STUDENT_LOAN_BALANCE_BAND_COMPILED_KEYS = {
    "lower": FSA_BALANCE_LOWER_BOUND_KEY,
    "upper": FSA_BALANCE_UPPER_BOUND_KEY,
}

#: The exact-value age key the dimension stamp writes when ``age`` is itself
#: a dimension of the fact.
#: :func:`microcosm.build.ledger_targets._constraint_bound_filters` skips
#: every constraint row whose variable is already a dimension key, so on
#: such a fact an ``age_{lower,upper}_bound`` restatement cannot have come
#: from an operator-checked ``>=``/``<`` row — only from a dimension of that
#: name, which carries no operator at all. A restated age bound on a spec
#: carrying this key is refused, whatever its value, including ``all``.
#: ``_ledger_metadata`` stamps a dimension only when its value is not
#: ``None`` and then drops empty values, so an ``age`` dimension valued
#: ``None`` or ``""`` leaves no key here; the compile's attestation
#: (:data:`AGE_BOUND_STAMP_SOURCE_KEY`, which checks dimension keys whatever
#: their value) is what refuses that case. The pinned feeds have no age
#: dimension of any kind.
RESTATED_AGE_DIMENSION_KEY = "ledger_filter_age"

FISCAL_TARGET_SOURCE_KEYS = {
    "cbo": "Congressional Budget Office revenue projections",
    "cms_aca": "CMS ACA marketplace enrollment public use files",
    "cms_medicaid": "CMS Medicaid enrollment and expenditure sources",
    "cms_medicare": "CMS Medicare Trustees Report Part B premium income",
    "ed_fsa": "Federal Student Aid federal student loan portfolio",
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
    "strike_benefits": (
        "No respondent in the pinned public ASEC files reports strike benefits: "
        "OI_OFF code 12 ('strike benefits', ASEC 2024 public use data "
        "dictionary) has zero person rows in pppub23, pppub24 and pppub25 "
        "(income years 2022-2024), and the PUF half carries no strike-benefit "
        "field, so the OI_OFF == 12 split in alimony.py is identically zero. "
        "The mapping is correct; a vintage with a code-12 reporter makes this "
        "entry stale and fails the gate."
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
#: The SCF education-debt fit table (microcosm#1073), in the release
#: directory of a ``--scf-education-debt`` build: at the staged frame's
#: pre-calibration weights, and at the release's calibrated weights before any
#: post-export scoring (microcosm#1074).
US_SCF_EDUCATION_DEBT_FIT_FILENAME = "us_scf_education_debt_fit.json"
US_SCF_EDUCATION_DEBT_CALIBRATED_FIT_FILENAME = (
    "us_scf_education_debt_calibrated_fit.json"
)
#: Both fit tables, keyed by their release-manifest artifact key. Only a
#: ``--scf-education-debt`` build writes them, so only its manifest lists
#: them, and ``microcosm-publish-release-build`` publishes them into the
#: build's folder (``releases/<release>/<build>/``) with the other
#: diagnostics, without ``--extra-file``.
US_SCF_EDUCATION_DEBT_FIT_ARTIFACTS: dict[str, str] = {
    "us_scf_education_debt_fit": US_SCF_EDUCATION_DEBT_FIT_FILENAME,
    "us_scf_education_debt_calibrated_fit": (
        US_SCF_EDUCATION_DEBT_CALIBRATED_FIT_FILENAME
    ),
}
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
        "--allow-unpinned-feed",
        action="store_true",
        help=(
            "Build from a Chronicle feed whose facts (and, for an artifact "
            "pin, manifest) digest differs from the committed US pin in "
            "us/chronicle_feed.json. Also lets through a feed whose BEA "
            "B4923C facts differ from the ESI premium anchor (microcosm#454); "
            "the difference is recorded in esi_premiums.json, or in the "
            "dry-run report. Only for an explicitly reviewed diagnostic run; "
            "a release build must use the pinned feed."
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
            "gate; the file path, sha256, entries and any mismatch are "
            "recorded in qrf_tail_concentration.json and bound into both "
            "manifests as qrf_tail_register, with that file a release "
            "artifact."
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
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help=(
            "Output root: the H5 and calibration weights go in <out>/artifacts, "
            "the build's diagnostics and gate evidence in <out>/build and its "
            "record in <out>/build_record.json. A build has no tier "
            "(microcosm#1585 item 11): certification places it in "
            "<out>/releases/<release>, the release of the tier it chooses "
            "(microcosm#1570). A finished build's directory is never reused, "
            "so each build needs its own --out."
        ),
    )
    parser.add_argument(
        "--release-id",
        help=(
            "The build ID (populace-us-2024-...), the name of the build's own "
            "folder in its release. It carries no tier: certified into "
            "microcosm-us-2024-national, the build keeps it; certified as "
            "evidence into microcosm-us-2024-evidence, it gains an "
            "'-evidence-' segment (microcosm#1585 item 11). Defaults to an ID "
            "made from the base dataset's digest, the commit and the time."
        ),
    )
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
        "--allow-esi-premium-gaps",
        action="store_true",
        help=(
            "Diagnostic escape hatch (microcosm#454): record the ESI premium "
            "gates — employer_sponsored_insurance_premiums absent, zero-mass, "
            "missing its raw ASEC coverage columns, or off the BEA NIPA 7.8 "
            "line 17 (B4923C) anchor, or a feed whose B4923C facts differ "
            "from that anchor — without failing the build (for example on a "
            "base built before the meps_esi_premiums stage). The failures are "
            "still written to esi_premiums.json. Release builds must leave "
            "this unset."
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
            "end-to-end $0-reform backstop; release builds should leave it on. "
            "Both manifests' gate_evidence block records the smoke as skipped."
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
            "With --dry-run-gates-report only: preview an evidence-tier "
            "certification (microcosm#506), grading a certain failure whose "
            "every release line an owner pattern matches as AT-RISK (OWNED). "
            "A build has no tier (microcosm#1585 item 11): it records every "
            "release-gate failure and writes its H5, and certification "
            "chooses the tier after it (tools/certify_us_national_build.py "
            "--evidence-release), so a build refuses this flag."
        ),
    )
    parser.add_argument(
        "--evidence-failure-owners",
        type=Path,
        help=(
            "With --dry-run-gates-report --evidence-release only: the per-run "
            "JSON file mapping failure-substring patterns to owner issue refs "
            '(e.g. {"Input coverage failed:": "PolicyEngine/microcosm#368"}) '
            "the evidence certification would take, checked ahead of the "
            "standing register (the US evidence_failure_owners.json)."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_US_FISCAL_CALIBRATION_EPOCHS,
        help=(
            "Optimization epochs for each calibration stage (default "
            f"{DEFAULT_US_FISCAL_CALIBRATION_EPOCHS:,}). "
            "--dense-default-dataset and the full-pool --exact-k point run "
            "one stage; the L0+refit arm and smaller --exact-k points run an "
            "L0 selection stage and then a refit stage, and each stage runs "
            "this many epochs. As of 2026-11-15, the US national calibrations "
            "that PolicyEngine had tagged in policyengine/populace-us since "
            "2026-07-01 ran the dense arm for 1,500 to 6,000 epochs; Build P, "
            "the parent of populace-us-2024-spm-20260915, ran 6,000."
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
            f"L0 penalty share (default {DEFAULT_L0_REFIT_LAMBDA_SHARE}) for "
            "the L0 selection stage of the L0+refit arm and of each --exact-k "
            "point below the full pool. The L0+refit arm runs when the caller "
            "sets neither --dense-default-dataset nor --exact-k. The builder "
            "divides the share by the candidate household count and uses the "
            "result as a fixed L0 penalty; the L0+refit arm then refits "
            "ordinary calibration on the households its L0 gates keep. "
            "--dense-default-dataset ignores it. As of 2026-11-15, none of "
            "the US national calibrations that PolicyEngine had tagged in "
            "policyengine/populace-us since 2026-07-01 ran an L0 stage; "
            "docs/us-release-build-rule.md lists the households each one "
            "calibrated."
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
            "Override the L2 concentration penalty for the refit stage only. "
            "The L0+refit arm and every --exact-k point write the refit "
            "stage's weights, and at the full pool, --exact-k runs that stage "
            "alone. Defaults to --l2-lambda. The builder refuses to combine "
            "this flag with --dense-default-dataset, which has no refit "
            "stage."
        ),
    )
    parser.add_argument(
        "--dense-default-dataset",
        action="store_true",
        help=(
            "Calibrate every candidate household in one solve with no L0 term "
            "and write that frame as the default national dataset, "
            "microcosm_us_2024.h5; the build manifest records "
            "dataset.default.method: dense_no_l0. This arm also fences the "
            "SSI take-up bands for ages 18-64 and 65+ (microcosm#566/#567), "
            "so a delivery miss there ships in the scorecard. The other arms "
            "count a miss of more than 5%% as a release-gate failure, which "
            "fails the build; only an evidence-tier certification, with an "
            "owner for the miss, can ship it. As of 2026-11-15, every US "
            "national calibration that PolicyEngine had tagged in "
            "policyengine/populace-us since 2026-07-01 came from this arm, "
            "and docs/us-release-build-rule.md lists them. A selection source "
            "(--selection-source-manifest or --selection-source-h5) first "
            "cuts the candidates to the households it names; without one, the "
            "solve covers every household the builder loads. The builder "
            "refuses to combine this flag with --exact-k, and the L0+refit "
            "arm runs when the caller sets neither."
        ),
    )
    parser.add_argument(
        "--warm-start-calibration-npz",
        type=Path,
        help=(
            "Optional calibration weights artifact from an earlier run on the "
            "same base frame (microcosm_us_2024_calibration.npz, or "
            "populace_us_2024_calibration.npz from a build before "
            "microcosm#1570). The builder validates that the stored "
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
        "--acs-person-zip",
        type=Path,
        help=(
            "Local canonical 2024 one-year ACS PUMS person CSV zip used to "
            "populate release-model predictors on physical ACS rows. Requires "
            "--acs-person-sha256 and both household archive options."
        ),
    )
    parser.add_argument(
        "--acs-person-sha256",
        help="Expected SHA-256 of --acs-person-zip.",
    )
    parser.add_argument(
        "--acs-household-zip",
        type=Path,
        help=(
            "Local canonical 2024 one-year ACS PUMS household CSV zip used by "
            "the release predictor join. Requires the other three ACS archive "
            "options."
        ),
    )
    parser.add_argument(
        "--acs-household-sha256",
        help="Expected SHA-256 of --acs-household-zip.",
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
        "--scf-education-debt",
        action="store_true",
        help=(
            "Impute the four federal student loan person inputs "
            "(federal_student_loan_balance, federal_parent_plus_loan_balance, "
            "is_income_based_repayment_new_borrower, "
            "federal_student_loan_graduate_share) from the 2025 SCF "
            "(microcosm#1073). Off by default until the pinned policyengine-us "
            "defines them (policyengine-us#9721, #9724); the certified engine "
            "ignores them until then."
        ),
    )
    parser.add_argument(
        "--scf-2025-summary-extract",
        type=Path,
        help=(
            "Optional local rscfp2025.dta for --scf-education-debt. When "
            "omitted, the sha-pinned scfp2025s.zip is fetched and cached."
        ),
    )
    parser.add_argument(
        "--scf-2025-full-extract",
        type=Path,
        help=(
            "Optional local p25i6.dta for --scf-education-debt. When omitted, "
            "the sha-pinned scf2025s.zip is fetched and cached."
        ),
    )
    parser.add_argument(
        "--sipp-tip-donor",
        type=Path,
        help=(
            "Optional local path to the sha-pinned SIPP 2023 slim CSV that "
            "feeds tip_income and Treasury tipped-occupation coverage. When "
            "omitted the donor is fetched from tag 1.115.5 and verified."
        ),
    )
    parser.add_argument(
        "--sipp-vehicle-donor",
        type=Path,
        help=(
            "Optional local path to the sha-pinned full SIPP 2023 public-use "
            "file that feeds financial assets, SSI disability criteria, "
            "household vehicle count/value, and measured voluntary tax filing. "
            "When omitted the donor is fetched from tag 1.115.5 and verified."
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
        "--target-surface",
        choices=TARGET_SURFACE_MODES,
        default=TARGET_SURFACE_FULL,
        help=(
            "Which compiled fiscal targets the release calibrates to. "
            f"'{TARGET_SURFACE_FULL}' (default) calibrates every compiled "
            f"target. '{TARGET_SURFACE_NATIONAL_STATE}' drops every "
            "congressional-district-classified target (the CD geography rows "
            "and the rows sourced from the SOI congressional-district file) "
            "after the target-parity and profile-coverage gates have run on "
            "the full compiled surface, as the July national releases "
            "calibrated; the drop is recorded in both manifests."
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
            "Local root for the build's version 2 staging run files, which go "
            "to <dir>/runs/<run_id>/. Defaults to <out>/staging when "
            "--staging-repo-id is set; without a repository, it makes a "
            "local-only run."
        ),
    )
    parser.add_argument(
        "--staging-repo-id",
        default=_env_default("POPULACE_STAGING_REPO_ID", STAGING_REPO_ID),
        help=(
            "Hugging Face dataset repo to upload staging run files to while "
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
        help=(
            "Disable staging run files and their uploads for this build; hosted "
            "telemetry remains active."
        ),
    )
    parser.add_argument(
        "--staging-run-id",
        help="Override the staging run id. Defaults to the candidate release id.",
    )
    parser.add_argument(
        _DRY_RUN_REPORT_FLAG,
        type=Path,
        metavar="PATH",
        help=(
            "Dry run: run this exact release through its base load, input "
            "stages and pre-solve gates, stop where the staged frame goes to "
            "target materialization, grade every data-dependent register and "
            "base-computable pre-export gate on the staged frame at base "
            "weights with this tool's own gate functions, write the JSON "
            "report here and exit 1 on any certain failure, 2 on AT-RISK "
            "only, 0 clean. It creates nothing under --out, starts no staging "
            "telemetry and writes no receipt; a base or donor the config does "
            "not name locally is downloaded into the same caches the release "
            "uses. See microcosm.build.us_runtime.release_gate_dry_run."
        ),
    )
    for flag, default, meaning in _DRY_RUN_MARGIN_FLAGS:
        parser.add_argument(
            flag,
            type=float,
            default=None,
            help=f"Dry run only: {meaning} (default {default}).",
        )
    args = parser.parse_args(argv)
    margin_overrides = {
        flag: getattr(args, flag.removeprefix("--").replace("-", "_"))
        for flag, _default, _meaning in _DRY_RUN_MARGIN_FLAGS
    }
    given_margins = [
        flag for flag, value in margin_overrides.items() if value is not None
    ]
    if args.dry_run_gates_report is None:
        if given_margins:
            parser.error(
                f"{', '.join(given_margins)}: dry-run margins apply only with "
                f"{_DRY_RUN_REPORT_FLAG}."
            )
        args.dry_run_margins = None
    else:
        try:
            args.dry_run_margins = DryRunMargins(
                **{
                    field_name: value
                    for (flag, _default, _meaning), field_name in zip(
                        _DRY_RUN_MARGIN_FLAGS, _DRY_RUN_MARGIN_FIELDS, strict=True
                    )
                    if (value := margin_overrides[flag]) is not None
                }
            )
        except ValueError as error:
            parser.error(str(error))
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
            "--refit-l2-lambda sets the refit stage's penalty, but "
            "--dense-default-dataset has no refit stage (use --l2-lambda)."
        )
    if not args.no_staging and not args.staging_dir and not args.staging_repo_id:
        # Staging with nowhere to write is a configuration error, not a quiet
        # skip. Turning staging off is an explicit decision, never a side
        # effect of a blank repo id.
        parser.error(
            "--staging-repo-id is empty and no --staging-dir is set, so staging "
            "run files would have no destination. Pass --no-staging to skip "
            "staging deliberately, or --staging-dir for a local-only run."
        )
    if args.evidence_failure_owners is not None and not args.evidence_release:
        parser.error("--evidence-failure-owners requires --evidence-release.")
    if args.evidence_release and args.dry_run_gates_report is None:
        parser.error(
            "--evidence-release previews an evidence-tier certification in a "
            "dry run (--dry-run-gates-report). A build has no tier "
            "(microcosm#1585 item 11): build without it, then choose the tier "
            "at certification: tools/certify_us_national_build.py <out> "
            "--evidence-release [--evidence-failure-owners FILE]."
        )
    if args.allow_gate_failed_base_pool and args.base_h5 is None:
        parser.error("--allow-gate-failed-base-pool requires --base-h5.")
    acs_archive_options = (
        args.acs_person_zip,
        args.acs_person_sha256,
        args.acs_household_zip,
        args.acs_household_sha256,
    )
    if any(value is not None for value in acs_archive_options) and any(
        value is None for value in acs_archive_options
    ):
        parser.error(
            "--acs-person-zip, --acs-person-sha256, --acs-household-zip, and "
            "--acs-household-sha256 must be provided together."
        )
    for flag, value in (
        ("--acs-person-sha256", args.acs_person_sha256),
        ("--acs-household-sha256", args.acs_household_sha256),
    ):
        if value is not None and (
            len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            parser.error(f"{flag} must be exactly 64 lowercase hexadecimal characters.")
    if args.evidence_release and args.exact_k is not None:
        parser.error(
            "--evidence-release is incompatible with --exact-k: an exact-k "
            "build is a build of the national release under the certified "
            "contract; the evidence tier (microcosm#506) covers the national "
            "artifact."
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
    try:
        args.target_family_loss_multipliers = parse_target_family_loss_multipliers(
            args.target_family_loss_multiplier
        )
    except ValueError as error:
        parser.error(str(error))
    for attribute, flag in _REFERENCE_OPTIONS:
        path = getattr(args, attribute)
        failed = None if path is None else _failed_build_holding(path)
        if failed is not None:
            parser.error(
                f"{flag} {path} belongs to {failed}. A failed build is never "
                "the reference a gate judges this build against."
            )
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


def _scf_wealth_gate_record(
    gate: GateResult, household_ids: Iterable[object]
) -> dict[str, object]:
    """The SCF-wealth gate's verdict and summary, as release evidence.

    The summary carries the SIPP/SCF blend audit (``financial_asset_blend``:
    seed, period, per-source household counts, bank-asset medians and
    low-tail shares). The stage keeps that audit in ``DataFrame.attrs``,
    which the H5 writer drops, so without this record no release artifact
    says whether the SIPP half ran (microcosm#1216).

    Every count and share describes the frame at the blend stage, before
    calibration and before any subset export. ``financial_asset_source_is_scf``
    draws one uniform per household in sorted order of the blend-stage
    household ids, so a household's draw depends on that whole id set. The
    record pins the set: its count and the SHA-256 of the sorted unique
    household ids, as little-endian int64 when every id is integral (every
    release so far) and as float64 otherwise. ``encoding`` names which, and
    the two encodings share one digest space, so a verifier compares
    ``encoding`` as well as ``sha256``. Every household has one head, so this
    is the set the draw sorts: when a published H5's household ids give the
    same encoding and digest, the seed and period reproduce each household's
    source. A subset release (L0, exact-k) exports part of the set and cannot
    be audited that way from the file alone. Ids that are neither integers
    nor floats, and unsigned ids beyond int64, are refused.

    Unlike the sibling gate records, this one is scrubbed: its details carry
    floats that are NaN on a degenerate frame, and both artifacts serialize
    strict JSON, so NumPy scalars become Python values and non-finite floats
    become null.
    """

    def finite(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): finite(nested) for key, nested in value.items()}
        if isinstance(value, (list, tuple)):
            return [finite(nested) for nested in value]
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, (float, np.floating)):
            return _finite_or_none(value)
        return value

    values = pd.to_numeric(pd.Series(list(household_ids)), errors="raise").to_numpy()
    if values.size == 0:
        values = values.astype(np.int64)
    # Neither encoding holds booleans, complex values or unsigned ids beyond
    # int64 without collisions, so those are refused rather than recorded.
    if values.dtype.kind not in "iuf":
        raise TypeError(
            f"blend-stage household ids must be integers or floats, not {values.dtype}"
        )
    ids = np.unique(values)
    if ids.dtype.kind == "u" and ids.size and int(ids.max()) > np.iinfo(np.int64).max:
        raise ValueError("a blend-stage household id exceeds the int64 range")
    # Integral ids (every release so far) hash as int64; anything else keeps
    # its float64 value, so distinct ids never collapse into one digest.
    integral = ids.dtype.kind in "iu" or bool(
        np.isfinite(ids).all()
        and (ids == np.round(ids)).all()
        and ((ids >= -(2**63)) & (ids < 2**63)).all()
    )
    id_bytes = (ids.astype("<i8") if integral else ids.astype("<f8")).tobytes()
    return {
        "passed": bool(gate.passed),
        "failures": list(gate.failures),
        "scope": "blend_stage_frame",
        "blend_household_ids": {
            "count": int(ids.size),
            "sha256": hashlib.sha256(id_bytes).hexdigest(),
            "encoding": (
                "sorted unique household_id, little-endian "
                + ("int64" if integral else "float64")
            ),
        },
        "details": finite(dict(gate.details)),
    }


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


def _finished_release_manifest(release_dir: Path) -> dict | None:
    """``release_dir``'s release manifest, if a certification finished it.

    Certification's last writes are the build-folder rewrite
    (:func:`rewrite_release_dir`), which names the release and the build at
    the top level of the release manifest and then of the build manifest.
    Until both read and name the same build, a crash left the certification
    part-written: it certifies nothing, and a rerun replaces both manifests
    (review of #1681).
    """

    release_dir = Path(release_dir)
    manifests = []
    for name in ("release_manifest.json", "build_manifest.json"):
        try:
            manifest = json.loads((release_dir / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not (
            isinstance(manifest, dict)
            and manifest.get("release") == release_dir.name
            and isinstance(manifest.get("build"), str)
            and manifest["build"]
        ):
            return None
        manifests.append(manifest)
    release_manifest, build_manifest = manifests
    if build_manifest["build"] != release_manifest["build"]:
        return None
    return release_manifest


def _refuse_certified_release_dir_reuse(release_dir: Path) -> None:
    """Fail loud when --out points at a finished build's release directory.

    microcosm#568 round 3: a failed retry into a directory that already
    carries a certified release would write failed-attempt weight evidence
    beside the prior run's manifest and H5 — mixing attempts the manifest
    knows nothing about. A finished build is never changed. The directory is
    named after the build's release (microcosm#1570), so every build of a
    release under one --out would share it: each build needs its own --out,
    or the directory removed deliberately.
    """

    if _finished_release_manifest(release_dir) is not None:
        raise RuntimeError(
            f"Release directory {release_dir} already carries a certified "
            "release (its release and build manifests name the build). Choose "
            "a new --out for "
            "this build or deliberately remove the stale directory before "
            "rerunning."
        )


def _refuse_overwriting_a_build_awaiting_certification(release_root: Path) -> None:
    """Fail loud when --out holds a finished build that awaits certification.

    A build stops at its record (microcosm#1585 item 11, PR 4a), so its
    --out carries no release manifest until certification writes one, and
    :func:`_refuse_reusing_a_certified_build` would let a rerun overwrite it.
    A build whose release gates failed is not certified at the certified
    tier, so a rerun into its --out still replaces it, as it did before
    builds wrote a record. Certifying it at the evidence tier is a choice
    made before such a rerun, and a build certified at either tier refuses
    one. A record that does not read cannot be certified either, so its
    --out is a failed build's too, as :func:`_failed_build_holding` counts
    it. A record of another schema is refused, and the refusal names the
    code that can certify it.
    """

    path = Path(release_root) / US_BUILD_RECORD_FILENAME
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if _build_failed(payload):
        return
    awaiting = (
        f"{release_root} holds a finished build awaiting certification "
        f"({US_BUILD_RECORD_FILENAME})"
    )
    if (
        isinstance(payload, dict)
        and payload.get("schema_version") != US_BUILD_RECORD_SCHEMA_VERSION
    ):
        raise RuntimeError(
            f"{awaiting} with build record schema "
            f"{payload.get('schema_version')!r}, which this tool does not read. "
            f"{_certify_with_its_own_code(payload)}, or choose a new --out for "
            "this build."
        )
    raise RuntimeError(
        f"{awaiting}. Certify it with {_certify_command(release_root)}, or "
        "choose a new --out for this build."
    )


def _build_failed(payload: object) -> bool:
    """Whether a build record's release-gate failures make its build a failed one.

    A build has no tier (microcosm#1585 item 11), so a build that recorded
    failures failed, whatever tier certifies it later. A schema 2 record's
    build chose its tier, and an evidence-tier build's owned failures were
    its certification's to ship, so they did not.
    """

    if not _recorded_release_gate_failures(payload):
        return False
    settings = payload.get("settings") if isinstance(payload, dict) else None
    return not (isinstance(settings, dict) and settings.get("evidence_release") is True)


def _build_certified_as_evidence(release_root: Path) -> bool:
    """Whether the build in ``release_root`` is certified at the evidence tier.

    An evidence-tier certification ships the build's failures as owned known
    failures (microcosm#506), so the build is an evidence release, not a
    failed build. Like the record's digest, this reads a file and stops a
    mistake, not an edit, until certification signs (PR 4b).
    """

    manifest = _finished_release_manifest(
        _certification_release_dir(release_root, evidence_release=True)
    )
    return (
        manifest is not None
        and manifest.get("tier") == "evidence"
        and bool(manifest.get("known_failures"))
    )


def _recorded_release_gate_failures(payload: object) -> list[str]:
    """The release-gate failures a build record carries.

    A build whose release gates fail records them and is not certified at
    the certified tier (microcosm#1585 item 11, PR 4a). A build has no tier:
    only an evidence-tier certification, with an owner for each failure, can
    ship them. A payload that is not a build record has none.
    """

    try:
        failures = payload["terminal_gate_failures"]
    except (KeyError, TypeError):
        return []
    if not isinstance(failures, list):
        return []
    return [str(line) for line in failures]


#: The #568 weight evidence. Only a build whose release gates fail writes it,
#: and a green build removes any it finds.
US_FAILED_BUILD_WEIGHT_EVIDENCE = (
    FINAL_HOUSEHOLD_WEIGHTS_FILENAME,
    FINAL_HOUSEHOLD_WEIGHT_IDS_FILENAME,
    FINAL_HOUSEHOLD_WEIGHTS_METADATA_FILENAME,
)

#: The options whose file is a reference a gate judges this build against:
#: the two input-mass references, and the incumbent diagnostics a critical
#: target may improve on. A failed build's files are refused there, as the
#: release-gate preflight refuses them as its export input-mass reference.
#: As a starting point (--base-h5, --selection-source-h5,
#: --warm-start-calibration-npz) they are allowed: they can cut a rerun's
#: time while its data is fixed, and the build's own gates judge its result.
_REFERENCE_OPTIONS = (
    ("input_mass_reference_h5", "--input-mass-reference-h5"),
    ("export_input_mass_reference_h5", "--export-input-mass-reference-h5"),
    ("incumbent_diagnostics", "--incumbent-diagnostics"),
)


def _failed_build_holding(path: Path) -> str | None:
    """The build whose release gates failed that ``path`` belongs to, or None.

    A build whose release gates fail still writes its H5, its calibration
    weights and its record, so it can be evaluated (microcosm#1585 item 11,
    PR 4a), and its files may start a later build. No gate judges against
    it: a later build refuses its files as an input-mass reference or as the
    incumbent diagnostics (:data:`_REFERENCE_OPTIONS`), the release-gate
    preflight refuses them as its export input-mass reference, and the
    post-export probe refuses its directories as the reference. A path
    belongs to the build whose record sits in the nearest directory above
    it. That build failed if its record carries release-gate failures
    (:func:`_build_failed`), or if its own directory holds the #568 weight
    evidence, unless an evidence-tier certification has made it an evidence
    release. A record that does not read cannot be certified, so its build
    counts as failed, as the --out refusal counts it
    (:func:`_refuse_overwriting_a_build_awaiting_certification`). Like the
    record's digest, this stops a mistake, not an edit, until certification
    signs (PR 4b).
    """

    target = Path(path).resolve()
    for release_root in (target, *target.parents):
        record_path = release_root / US_BUILD_RECORD_FILENAME
        if not record_path.is_file():
            continue
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return f"the build whose record {record_path} does not read"
        if not isinstance(payload, dict) or payload.get("kind") != US_BUILD_RECORD_KIND:
            continue
        build = f"build {payload.get('build')} ({record_path})"
        if _build_certified_as_evidence(release_root):
            return None
        if _build_failed(payload):
            failures = _recorded_release_gate_failures(payload)
            return (
                f"{build}, which failed {len(failures)} release gate(s) when it "
                "was built"
            )
        try:
            layout = payload["layout"]
            # Schema 2 named the build's directory after its release.
            build_dir = release_root / (
                layout.get("build_dir") or layout["release_dir"]
            )
        except (KeyError, TypeError):
            return None
        evidence = [
            name
            for name in US_FAILED_BUILD_WEIGHT_EVIDENCE
            if (build_dir / name).exists()
        ]
        if evidence:
            return (
                f"{build}, whose directory holds the weight evidence a build "
                f"writes when its release gates fail ({', '.join(evidence)}; "
                "microcosm#568)"
            )
        return None
    return None


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


# The byte framing of _staged_frame_sha256. It is the first field hashed, so a
# framing change must rename it and digests under two codecs never compare
# equal.
STAGED_FRAME_DIGEST_CODEC = "us_fiscal_refresh_staged_frame_v1"


def _update_staged_frame_digest(digest, field: str, payload) -> None:
    """Append one named, length-prefixed field to the staged-frame digest.

    Prefixing every field with its name and byte length keeps the stream
    unambiguous: bytes cannot shift between adjacent columns or tables and
    reproduce another frame's digest.
    """

    field_bytes = field.encode("utf-8")
    view = memoryview(payload).cast("B")
    digest.update(len(field_bytes).to_bytes(8, "little"))
    digest.update(field_bytes)
    digest.update(view.nbytes.to_bytes(8, "little"))
    digest.update(view)


def _update_staged_frame_digest_with_series(
    digest, field: str, series: pd.Series
) -> None:
    """Hash one column: its dtype, row count, missing mask and value bytes."""

    dtype = series.dtype
    _update_staged_frame_digest(digest, f"{field}/dtype", str(dtype).encode("utf-8"))
    _update_staged_frame_digest(
        digest, f"{field}/rows", len(series).to_bytes(8, "little")
    )
    if isinstance(dtype, np.dtype) and dtype.kind in "biufcmM":
        # NumPy-backed fixed-width values: hash the raw bytes. A NaN is its
        # own bit pattern, so no separate mask is needed. Datetimes hash as
        # their int64 ticks (the buffer protocol refuses them); the dtype
        # field above already names the unit.
        values = series.to_numpy(copy=False)
        if dtype.kind in "mM":
            values = values.view(np.int64)
        _update_staged_frame_digest(
            digest,
            f"{field}/values",
            np.ascontiguousarray(
                values.astype(values.dtype.newbyteorder("<"), copy=False)
            ),
        )
        return
    missing = series.isna().to_numpy(dtype=np.bool_)
    _update_staged_frame_digest(digest, f"{field}/missing", missing)
    if getattr(dtype, "kind", "O") in "biuf" and hasattr(dtype, "numpy_dtype"):
        # Masked extension arrays (boolean, Int64, Float64): canonical zero
        # under the mask, so hidden bits cannot move the digest.
        numpy_dtype = np.dtype(dtype.numpy_dtype)
        values = series.to_numpy(dtype=numpy_dtype, na_value=numpy_dtype.type(0))
        _update_staged_frame_digest(
            digest,
            f"{field}/values",
            np.ascontiguousarray(
                values.astype(numpy_dtype.newbyteorder("<"), copy=False)
            ),
        )
        return
    # Variable-width values (strings, objects, categoricals). A missing slot
    # encodes as empty bytes; the mask above tells it apart from "".
    values = series.to_numpy(dtype=object, na_value=None)
    if pd.api.types.infer_dtype(values, skipna=True) in {"string", "empty"}:
        kind = b"utf8"
        encoded = [
            b"" if value is None else value.encode("utf-8", "surrogatepass")
            for value in values
        ]
    else:
        # Mixed objects: tag each value with its type so 1, 1.0 and "1" differ.
        kind = b"typed"
        encoded = [
            b""
            if value is None
            else (
                f"{type(value).__module__}.{type(value).__qualname__}\x00{value}"
            ).encode("utf-8", "surrogatepass")
            for value in values
        ]
    _update_staged_frame_digest(digest, f"{field}/encoding", kind)
    _update_staged_frame_digest(
        digest,
        f"{field}/lengths",
        np.fromiter((len(item) for item in encoded), dtype="<u8", count=len(encoded)),
    )
    _update_staged_frame_digest(digest, f"{field}/payload", b"".join(encoded))


def _staged_frame_sha256(frame: Frame) -> str:
    """Digest the staged frame handed to the target materializer.

    microcosm#956. The rest of the checkpoint identity hashes the on-disk
    base dataset, the run settings (seed, PolicyEngine-US version, target
    registry, congressional-district crosswalk, selection-mass protections),
    a few named stage-input digests (the weeks-unemployed source, the SSI
    take-up assignment and prior-weight basis, the selection support) and a
    hand-bumped materializer version. A change to any other stage that forgot
    the bump would silently reuse stale target columns. This digest
    covers what the materializer reads and the checkpoint restores: each
    entity table's column names, dtypes and value bytes in column order,
    every weight vector with its kind, and the person strata. It streams one
    column at a time, so it never copies the whole frame.
    """

    digest = hashlib.sha256()
    _update_staged_frame_digest(
        digest, "codec", STAGED_FRAME_DIGEST_CODEC.encode("ascii")
    )
    _update_staged_frame_digest(
        digest, "entities", _strict_json_bytes(list(frame.entities))
    )
    for entity in frame.entities:
        table = frame.table(entity)
        _update_staged_frame_digest(
            digest,
            f"tables/{entity}/columns",
            _strict_json_bytes([str(column) for column in table.columns]),
        )
        for index in range(table.shape[1]):
            _update_staged_frame_digest_with_series(
                digest, f"tables/{entity}/{index}", table.iloc[:, index]
            )
    _update_staged_frame_digest(
        digest, "weighted_entities", _strict_json_bytes(list(frame.weighted_entities))
    )
    for entity in frame.weighted_entities:
        weights = frame.weights_for(entity)
        _update_staged_frame_digest(
            digest, f"weights/{entity}/kind", weights.kind.value.encode("utf-8")
        )
        _update_staged_frame_digest(
            digest,
            f"weights/{entity}/values",
            np.ascontiguousarray(weights.values, dtype="<f8"),
        )
    _update_staged_frame_digest_with_series(digest, "strata", frame.strata)
    return digest.hexdigest()


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
    staged_frame_sha256: str,
    selection_mass_protections: tuple[str, ...] = (),
    ssi_take_up_prior_weight_basis_sha256: object = None,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "schema_version": TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION,
        "materializer_version": TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION,
        "kind": "us_fiscal_refresh_target_frame",
        "country": "us",
        "base_dataset_sha256": str(base_dataset_sha256),
        # The exact staged frame the materializer reads (microcosm#956). The
        # base hash above cannot see the stages that run after the base
        # load. The build commit stays out: a new commit that stages the
        # same frame still hits, and the writing commit is recorded beside
        # the identity as information only.
        "staged_frame_sha256": str(staged_frame_sha256),
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
    build_commit: str,
) -> dict[str, object]:
    import h5py

    frame.check_shared_columns()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as h5:
        h5.attrs["schema_version"] = TARGET_FRAME_CHECKPOINT_SCHEMA_VERSION
        h5.attrs["identity_json"] = _strict_json_text(identity)
        h5.attrs["identity_sha256"] = _target_frame_checkpoint_digest(identity)
        h5.attrs["compilation_json"] = _strict_json_text(compilation)
        # Provenance only, never identity: a hit reports which build wrote
        # the frame it reuses (microcosm#956).
        h5.attrs["build_commit"] = str(build_commit)
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
        "staged_frame_sha256": identity.get("staged_frame_sha256"),
        "source_build_commit": str(build_commit),
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
        stored_build_commit = h5.attrs.get("build_commit")
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
            "staged_frame_sha256": identity.get("staged_frame_sha256"),
            # The build that wrote the reused frame, which may differ from
            # this one: the identity deliberately excludes the commit.
            "source_build_commit": (
                None if stored_build_commit is None else str(stored_build_commit)
            ),
            "stored_compilation": stored_compilation,
        },
        "target_materialization_cache": {
            "enabled": False,
            "status": "skipped_target_frame_checkpoint_hit",
        },
        # A hit runs no engine; the writing run's receipt is in
        # ``stored_compilation``.
        "target_materialization_batching": {
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


def _repository_root() -> Path:
    """The checkout this tool's code is in, where its git checks run.

    A build and its certification check the commit and the work tree of the
    code that runs, not of the current directory, so a tool run from
    another checkout's directory cannot pass as that checkout's code (review
    of #1681). The workspace packages must load from it too
    (:func:`_packages_outside_the_checkout`).
    """

    return Path(__file__).resolve().parents[1]


#: The workspace packages a build and its certification run. Every one
#: reports the same version wherever it is installed from, so a version check
#: cannot tell checkouts apart: each must load from this tool's checkout.
_WORKSPACE_PACKAGES = (
    "microcosm.build",
    "microcosm.calibrate",
    "microcosm.data",
    "microcosm.diagnostics",
    "microcosm.fit",
    "microcosm.frame",
    "microcosm.graph",
)


def _packages_outside_the_checkout() -> list[str]:
    """The workspace packages that load from outside this tool's checkout."""

    root = _repository_root().resolve()
    outside = []
    for name in _WORKSPACE_PACKAGES:
        spec = importlib.util.find_spec(name)
        origin = None if spec is None else spec.origin
        if origin is None or root not in Path(origin).resolve().parents:
            outside.append(f"{name} ({origin})")
    return outside


def _refuse_packages_from_another_checkout(action: str) -> None:
    """Refuse to ``action`` with workspace packages from another tree."""

    outside = _packages_outside_the_checkout()
    if outside:
        raise SystemExit(
            f"Refusing to {action} with packages from outside this tool's "
            f"checkout ({_repository_root()}): " + "; ".join(outside) + ". Run "
            "it from a checkout whose own packages are installed (`uv sync "
            "--all-packages` there), so the commit it records is the code that "
            "ran."
        )


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ("git", "-C", str(_repository_root()), *args), text=True
    ).strip()


def _git_dirty() -> bool:
    return bool(_git_output("status", "--porcelain"))


def _git_dirty_paths(*, excluding: Path) -> list[str]:
    """The work tree's ``git status --porcelain`` lines, leaving out ``excluding``.

    Certification checks the code it runs, not the build it reads: a build
    whose ``--out`` lies inside the work tree, unignored, would otherwise make
    its own certification refuse it, since the build's files are untracked
    there. The build directory is left out only when it lies strictly inside
    the work tree; its files are the record's to check, by digest.
    """

    top = Path(_git_output("rev-parse", "--show-toplevel")).resolve()
    pathspec = ["."]
    try:
        inside = Path(excluding).resolve().relative_to(top)
    except ValueError:
        inside = None
    if inside is not None and inside != Path("."):
        pathspec.append(f":(exclude,literal){inside.as_posix()}")
    # Unstripped, so each line keeps its two status columns.
    status = subprocess.check_output(
        ("git", "-C", str(top), "status", "--porcelain", "--", *pathspec), text=True
    )
    return [line for line in status.splitlines() if line]


def _download_base_h5() -> Path:
    """The base a run without ``--base-h5`` starts from: the US national
    release's current build, SHA-verified by the loader (microcosm#1570).

    It is read from ``releases.json`` once the repository has one and from
    ``latest.json`` until then. The repository's root file stops changing at
    the switch, since builds published into folders write no root files.
    """
    from microcosm.data import download

    return Path(download("us", 2024))


def _load_frame(path: Path, *, expected_sha256: str | None = None) -> Frame:
    consumer = "US fiscal refresh release builder generic H5 loader (_load_frame)"
    sha256 = refuse_denied_pool_h5(path, consumer=consumer)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ValueError(
            f"{consumer}: {path} is not the base dataset whose identity was recorded "
            f"(SHA-256 {sha256}, expected {expected_sha256}); the read is refused."
        )

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
    assert_h5_unchanged(path, sha256, consumer=consumer)
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(weights, WeightKind.CALIBRATED)},
    )
    refuse_denied_frame(frame, consumer=consumer)
    return frame


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


# The input-stage passes below run policyengine-us over household batches
# (``household_batch_plan``). Each batch engine runs in
# ``checked_batch_engine``: with more than one batch, or when a caller that
# hands over only part of the population passes
# ``refuse_population_aggregates=True``, an engine that computed a
# population-aggregate variable is refused, and every engine is released
# before the next one is built, also when its batch raised.
_ACA_SOURCE_INPUT_PASS = "The ACA source-input pass"
_MEDICAID_ELIGIBILITY_PASS = "The Medicaid eligibility pass"
_DECEMBER_SSI_PASS = "The December SSI pass"
_SNAP_ELIGIBILITY_PASS = "The SNAP eligibility pass"


def _aca_source_tax_unit_table_batched(
    frame: Frame,
    target_tables: Mapping[str, pd.DataFrame],
    *,
    microsimulation_cls,
    maximum_microsim_batch_size: int | None,
    refuse_population_aggregates: bool | None = None,
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
    plan = household_batch_plan(
        frame,
        maximum_microsim_batch_size,
        refuse_population_aggregates=refuse_population_aggregates,
        work=_ACA_SOURCE_INPUT_PASS,
    )
    batches = plan.batches
    if len(batches) > 1:
        print(
            "Materializing ACA source inputs in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )

    for batch, household_positions in enumerate(batches, start=1):
        with automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else select_households_by_position(frame, household_positions)
            )
            batch_simulation = microsimulation_cls(
                dataset=_dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            with checked_batch_engine(
                batch_simulation,
                refuse_population_aggregates=plan.refuse_population_aggregates,
                batch=batch,
                batches=len(batches),
                work=_ACA_SOURCE_INPUT_PASS,
                remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
            ):
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
            del batch_tax_unit, batch_simulation, batch_frame
        collect_batch_garbage()
    collect_pass_garbage()
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
    refuse_population_aggregates: bool | None = None,
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
    plan = household_batch_plan(
        frame,
        maximum_microsim_batch_size,
        refuse_population_aggregates=refuse_population_aggregates,
        work=_MEDICAID_ELIGIBILITY_PASS,
    )
    batches = plan.batches
    if len(batches) > 1:
        print(
            "Materializing Medicaid eligibility in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )
    for batch, household_positions in enumerate(batches, start=1):
        with automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else select_households_by_position(frame, household_positions)
            )
            batch_simulation = Microsimulation(
                dataset=_dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            with checked_batch_engine(
                batch_simulation,
                refuse_population_aggregates=plan.refuse_population_aggregates,
                batch=batch,
                batches=len(batches),
                work=_MEDICAID_ELIGIBILITY_PASS,
                remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
            ):
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
            del batch_frame, batch_simulation
        collect_batch_garbage()
    collect_pass_garbage()
    return eligibility


def _ssi_person_uncapped_amount(
    frame: Frame,
    *,
    simulation=None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    refuse_population_aggregates: bool | None = None,
) -> np.ndarray:
    """December person-level potential federal SSI, batched like Medicaid.

    SSA's age-band recipient counts are December 2024 point-in-time stocks.
    ``uncapped_ssi > 0`` is the PolicyEngine-US 2.2.1 current-benefit
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
    plan = household_batch_plan(
        frame,
        maximum_microsim_batch_size,
        refuse_population_aggregates=refuse_population_aggregates,
        work=_DECEMBER_SSI_PASS,
    )
    batches = plan.batches
    if len(batches) > 1:
        print(
            "Materializing December SSI candidate amounts in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )
    for batch, household_positions in enumerate(batches, start=1):
        with automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else select_households_by_position(frame, household_positions)
            )
            batch_simulation = Microsimulation(
                dataset=_dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            with checked_batch_engine(
                batch_simulation,
                refuse_population_aggregates=plan.refuse_population_aggregates,
                batch=batch,
                batches=len(batches),
                work=_DECEMBER_SSI_PASS,
                remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
            ):
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
            del batch_frame, batch_simulation
        collect_batch_garbage()
    collect_pass_garbage()
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
    refuse_population_aggregates: bool | None = None,
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
    plan = household_batch_plan(
        frame,
        maximum_microsim_batch_size,
        refuse_population_aggregates=refuse_population_aggregates,
        work=_SNAP_ELIGIBILITY_PASS,
    )
    batches = plan.batches
    if len(batches) > 1:
        print(
            "Materializing SNAP eligibility in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )
    for batch, household_positions in enumerate(batches, start=1):
        with automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                frame
                if full_batch
                else select_households_by_position(frame, household_positions)
            )
            batch_simulation = Microsimulation(
                dataset=_dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            with checked_batch_engine(
                batch_simulation,
                refuse_population_aggregates=plan.refuse_population_aggregates,
                batch=batch,
                batches=len(batches),
                work=_SNAP_ELIGIBILITY_PASS,
                remedy=INPUT_PASS_POPULATION_AGGREGATE_REMEDY,
            ):
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
            del batch_frame, batch_simulation
        collect_batch_garbage()
    collect_pass_garbage()
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
    period_only: bool = False,
):
    """The frame as a policyengine-us dataset for ``PERIOD``.

    By default the dataset is the one year, and policyengine-us projects it
    as a user's load does: it adds an uprated copy of every later year to its
    last CPI-U year, stores them all, and copies them again into each branch
    it opens. A read at a later year then gets the engine's own projected
    inputs; post-export scoring plans keys in later years.

    ``period_only`` holds ``PERIOD`` alone (``us_dataset_through``), for an
    in-build simulation that reads only ``PERIOD`` (``_calculate_array``, and
    the December SSI month). A later read on that dataset is not refused: the
    engine carries or uprates leaf inputs itself and reads a stored
    formula-owned column as its default, so only such a caller may ask.
    """
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
    dataset = USSingleYearDataset(
        person=tables["person"],
        household=tables["household"],
        tax_unit=tables["tax_unit"],
        spm_unit=tables["spm_unit"],
        family=tables["family"],
        marital_unit=tables["marital_unit"],
        time_period=PERIOD,
    )
    if not period_only:
        return dataset
    return us_dataset_through(dataset, PERIOD)


def _calculate_array(
    simulation, variable: str, *, map_to: str | None = None
) -> np.ndarray:
    kwargs: dict[str, Any] = {"period": PERIOD}
    if map_to is not None:
        kwargs["map_to"] = map_to
    return np.asarray(simulation.calculate(variable, **kwargs))


def _tax_unit_to_household_positions(frame: Frame) -> np.ndarray:
    return group_to_household_positions(frame, "tax_unit")


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


# ---------------------------------------------------------------------------
# Household-batched post-export scoring (route A remediation, microcosm#956)
# ---------------------------------------------------------------------------
#
# The reform-coverage smoke, reform_validation and demographics each used to
# build one Microsimulation over the whole written H5
# (``reform_validation.default_simulate_factory``). Reform validation keeps
# one shared baseline for all of its keys.
#
# This scorer keeps the consumers' ``simulate(reform) -> simulation`` seam and
# changes only how the seam is served:
#
# * the WRITTEN H5 is loaded once (its sha256 bound to the bytes read) and
#   split once into household batches; every group entity must nest in one
#   household and the batches must partition every entity, or the scorer
#   refuses before any engine exists. Each batch is then written to its own
#   file, with the sha256 of its bytes recorded, and the frames are freed; a
#   pass reads each batch back from its file and refuses one whose bytes
#   changed, so the process that scores holds no frame of the H5 between
#   batches (``_PostExportBatchFiles``);
# * every construction declares ``spm=US_RELEASE_SPM_SELECTION``;
# * each consumer's baseline requests are learned by an engine-free recording
#   dry run; the scorer registers the recorded plans and scores their union
#   in ONE batch-outer pass, in ascending period order, serving each consumer
#   its own keys (microcosm#1595). A consumer without a registered plan, and
#   every consumer once that pass has failed, scores its own plan in a pass
#   of its own; a refused batch file (``PostExportBatchFileError``) is raised
#   at once instead. The served baseline refuses any key outside the
#   consumer's recorded plan;
# * each reform builds ONE tax-benefit system (microcosm#456) and scores each
#   key, or each set of keys it is asked to ``prefetch``, in one pass, batch
#   by batch; every batch engine is released before the next;
# * a batch reform engine gets that system alone, not ``reform=`` too (see
#   ``_HouseholdBatchedPostExportScorer._construct``), so it carries no
#   ``baseline`` branch; it refuses the formulas that read one
#   (``POST_EXPORT_BASELINE_BRANCH_READERS``), and a reform may not move the
#   behavioral-response parameters that decide whether the rest do;
# * ``calculate`` returns full-length concatenated values with their engine
#   weights, so ``.sum()`` (== ``MicroSeries.sum``), ``np.asarray`` and
#   ``.weights`` feed every existing statistic unchanged.
#
# Totals then differ from one whole-pool simulation only in floating-point
# summation order, PROVIDED every scored measure is additive across
# households: a formula that aggregates over its simulation's whole population
# (``US_POPULATION_AGGREGATE_VARIABLES``: the Medicaid SLCSP state sums and
# the weighted income deciles) sees one batch, not the file. Every
# batch engine of a multi-batch pass refuses once it has computed one of them,
# so a measure that reaches such a formula fails the stage instead of scoring
# a batch-local aggregate.

#: Recorded in ``post_export_scoring`` blocks; bump with any change to how the
#: written H5 is scored that can change a scored value. Regrouping keys into
#: fewer passes while every value stays byte-identical (the merged baseline
#: pass and the shared pre-OBBBA pass, microcosm#1595) keeps it.
POST_EXPORT_SCORING_METHOD = "household_batched_written_h5"

#: One scored request: (variable, period, map_to).
PostExportKey = tuple[str, int, str | None]

#: policyengine-us formulas that aggregate over their simulation's whole
#: population instead of within one household, so a household-batch engine
#: computes them over its own batch only. In policyengine-us 2.2.1,
#: ``medicaid_slcsp_state_average_cost_index`` and
#: ``medicaid_slcsp_state_denominator`` sum person weights by state
#: (``sum_by_state``) and feed ``medicaid_cost_if_enrolled``, and through it
#: ``medicaid_cost``, ``medicaid`` and ``household_health_benefits``; the two
#: income deciles are weighted ranks over every household or SPM unit. Batched
#: target materialization and the input-stage passes (through
#: ``household_batches.refuse_batch_population_aggregates``) and multi-batch
#: post-export scoring (``_assert_post_export_scoring_is_batch_invariant``)
#: refuse an engine that computed any of them.
#: ``test_us_batched_target_materialization.py`` pins this list against the
#: installed engine's variable sources; that pattern scan is a drift
#: detector, not a proof of household locality (microcosm#956).
#: ``test_us_post_export_scoring.py`` exercises the post-export Medicaid
#: refusal on a written fixture H5, and ``test_us_input_batch_loops.py`` the
#: input-stage refusals. Shared with the recertification tool
#: (microcosm#1585).
US_POPULATION_AGGREGATE_VARIABLES = recert.POPULATION_AGGREGATE_VARIABLES

#: The pass that target compilation's refusals name, and how its
#: population-aggregate refusal ends.
TARGET_MATERIALIZATION_PASS = "Target materialization"
TARGET_POPULATION_AGGREGATE_REMEDY = (
    "so a target that reaches them cannot be materialized in household "
    "batches. Drop the target from the surface, or run the whole pool "
    "unbatched in one simulation with one slice or chunk. Disabling only "
    "inner batching still leaves slice- or chunk-local aggregates."
)

#: policyengine-us formulas whose value reads the engine's ``baseline`` branch,
#: which a batch reform engine does not have. ``medicaid_slcsp_state_denominator``
#: holds a reform at the baseline's denominator; the others reach
#: ``get_behavioral_response_measurements``, which measures a reform against
#: ``get_branch("baseline")`` and would measure it against itself here. A reform
#: pass refuses an engine that computed any of them, whatever its batch count;
#: so does each JCT reform batch of target compilation, whose engines also get
#: the reform's system alone (``_refuse_batch_baseline_branch_readers``).
#: Pinned in ``test_us_post_export_scoring.py``.
POST_EXPORT_BASELINE_BRANCH_READERS = (
    "income_elasticity_lsr",
    "medicaid_slcsp_state_denominator",
    "relative_capital_gains_mtr_change",
    "relative_income_change",
    "relative_wage_change",
    "substitution_elasticity_lsr",
)

#: The other two formulas that read ``simulation.baseline``, and the parameter
#: subtree each response is computed from. Without a baseline branch both
#: return 0; with one they return 0 at zero elasticities. Every reform pass, and
#: each JCT reform of target compilation
#: (``_refuse_reform_moving_behavioral_responses``), requires these parameters
#: to remain at their baseline values before it builds a batch engine.
POST_EXPORT_BEHAVIORAL_RESPONSE_PARAMETERS = {
    "capital_gains_behavioral_response": "gov.simulation.capital_gains_responses",
    "labor_supply_behavioral_response": "gov.simulation.labor_supply_responses",
}


def _post_export_key(
    variable: str, period: object, map_to: str | None = None
) -> PostExportKey:
    if period is None:
        raise ValueError(
            f"Post-export scoring needs an explicit period for {variable!r}; the "
            "engine default period is not part of the recorded plan."
        )
    return (str(variable), int(str(period)), None if map_to is None else str(map_to))


def _post_export_key_record(key: PostExportKey) -> dict[str, object]:
    variable, period, map_to = key
    return {"variable": variable, "period": period, "map_to": map_to}


def _ascending_period_plan(keys: Iterable[PostExportKey]) -> tuple[PostExportKey, ...]:
    """Order a baseline plan by period, keeping request order within a period.

    Baseline consumers can request interleaved periods. Sort those requests
    before engine calls; the engine guard refuses a return to an earlier
    period. The real-engine test probes the MD CCS request-order error and
    warns if the upstream engine no longer reproduces it.
    """
    return tuple(sorted(dict.fromkeys(keys), key=lambda key: key[1]))


def _post_export_baseline_plan_record(
    plan: Sequence[PostExportKey],
) -> dict[str, object]:
    return {
        "keys": [_post_export_key_record(key) for key in plan],
        "period_order": sorted({key[1] for key in plan}),
    }


class _PostExportValues:
    """Full-length values of one scored key, with the engine's weights.

    The duck type the post-export consumers read from a Microsimulation
    result: ``np.asarray(values)``, ``values.values``, ``values.weights`` and
    ``values.sum()``. Both arrays are read-only, so no consumer can mutate a
    baseline result another consumer statistic still reads.
    """

    __slots__ = ("_values", "_weights")

    def __init__(self, values: np.ndarray, weights: np.ndarray) -> None:
        values = np.asarray(values)
        weights = np.asarray(weights, dtype=np.float64)
        if values.ndim != 1 or weights.shape != values.shape:
            raise ValueError(
                "Post-export scoring needs one weight per value; got values "
                f"of shape {values.shape} and weights of shape {weights.shape}."
            )
        values.flags.writeable = False
        weights.flags.writeable = False
        self._values = values
        self._weights = weights

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        if dtype is None and not copy:
            return self._values
        return np.array(self._values, dtype=dtype, copy=True)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def sum(self) -> float:
        """Weighted total, computed as policyengine-core's ``MicroSeries.sum``.

        ``MicroSeries.sum`` is ``self.multiply(self.weights).sum()``: a pandas
        product summed with NaN skipped. Doing the same over the concatenated
        arrays gives the whole-pool total up to summation order.
        """
        return float(pd.Series(self._values).multiply(self._weights).sum())


def _concatenate_post_export_parts(
    parts: Sequence[tuple[np.ndarray, np.ndarray]],
) -> _PostExportValues:
    return _PostExportValues(
        np.concatenate([values for values, _ in parts]),
        np.concatenate([weights for _, weights in parts]),
    )


def _post_export_values_and_weights(result, key: PostExportKey):
    weights = getattr(result, "weights", None)
    if weights is None:
        raise RuntimeError(
            f"Post-export scoring of {key} got an unweighted engine result; "
            "every scored key must carry the engine's entity weights."
        )
    values = np.asarray(result)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise RuntimeError(
            f"Post-export scoring of {key} got {values.shape} values against "
            f"{weights.shape} weights from one batch engine."
        )
    return values, weights


class _RecordingPostExportSimulation:
    """Engine-free stand-in that records each key one simulation is asked for.

    ``asked`` is shared by every recorder of one dry run: it collects
    ``(simulation, key)`` in the order the serial scorer computes them, where
    ``simulation`` is the reformed simulation's number, or ``None`` for the
    baseline. ``reforms`` is the dry run's list of the reforms opened so far,
    also shared, so a reformed simulation's recorder refuses every key once a
    later one is opened, as the scorer refuses a finished
    ``_BatchedPostExportReform``. ``reform_passes``, also shared, gets one
    entry per batched reform pass: when the consumer runs for real, each
    ``calculate`` or ``prefetch`` that asks a reformed simulation for a key it
    was not asked before is one pass of ``_BatchedPostExportReform``, which
    computes the new keys in ascending period order.
    """

    def __init__(
        self,
        asked: dict[tuple[int | None, PostExportKey], None],
        simulation: int | None,
        reforms: Sequence[object],
        reform_passes: list[int],
    ) -> None:
        self._asked = asked
        self._simulation = simulation
        self._reforms = reforms
        self._reform_passes = reform_passes

    def calculate(self, variable, period=None, map_to=None) -> _PostExportValues:
        self._record((_post_export_key(variable, period, map_to),))
        return _PostExportValues(np.zeros(0), np.zeros(0))

    def prefetch(self, requests) -> None:
        """Record ``requests`` as ``_BatchedPostExportReform.prefetch`` scores them."""
        self._record(tuple(_post_export_key(*request) for request in requests))

    def _record(self, keys: Sequence[PostExportKey]) -> None:
        latest = len(self._reforms) - 1
        if self._simulation is not None and self._simulation != latest:
            asked_for = keys[0] if len(keys) == 1 else tuple(keys)
            raise RuntimeError(
                f"A post-export consumer asked reformed simulation "
                f"{self._simulation} for {asked_for} after opening reformed "
                f"simulation {latest}. The scorer finishes a reformed simulation "
                "when the consumer opens the next one and refuses it further "
                "keys, so a consumer asks each reformed simulation for all its "
                "keys before opening the next."
            )
        new_keys = _ascending_period_plan(
            key for key in keys if (self._simulation, key) not in self._asked
        )
        self._asked.update(dict.fromkeys((self._simulation, key) for key in new_keys))
        if new_keys and self._simulation is not None:
            self._reform_passes.append(self._simulation)


def _dry_run_post_export_requests(
    consumer: Callable[[Callable[[Any], Any]], object],
) -> tuple[tuple[int | None, object, PostExportKey], ...]:
    """Every request ``consumer`` makes, learned from engine-free recorders.

    Returns ``(simulation, reform, key)`` for each request, in the order the
    serial scorer runs them: the baseline plan first (``reform`` and
    ``simulation`` None; ``_ascending_period_plan`` order), then the reform
    requests. When the consumer runs for real, each ``calculate`` or
    ``prefetch`` that asks a reformed simulation for keys it was not asked
    before is one batched pass (``_BatchedPostExportReform``), which computes
    those keys in ascending period order, so the reform requests follow that
    order; ``simulation`` numbers the consumer's ``simulate(reform)`` calls
    from 0. ``_dry_run_post_export`` also counts the passes.

    No engine is constructed: ``simulate(None)`` returns a recorder that logs
    every baseline request, and each ``simulate(reform)`` a recorder of its
    own. The consumers ask for the same keys whatever values they get back,
    and the served baseline refuses any key this dry run did not see, so a
    value-dependent request cannot slip past the plan. ``simulate`` carries
    ``RECORDS_REQUESTS_ONLY``, so each consumer hands it the description it
    builds a reform from instead of building the reform (the reform-coverage
    smoke check's probe, a reform-validation spec or a cumulative OBBBA
    state's parameter changes), and the dry run resolves no list edit against
    an engine. A reform given as anything else is refused
    (``check_post_export_reform``): a worker process could not rebuild it.
    So is a key asked of a reformed simulation after the consumer opened the
    next one, which the scorer refuses too (``_BatchedPostExportReform``).
    """
    return _dry_run_post_export(consumer)[0]


def _dry_run_post_export(
    consumer: Callable[[Callable[[Any], Any]], object],
) -> tuple[tuple[tuple[int | None, object, PostExportKey], ...], int]:
    """``_dry_run_post_export_requests``'s requests and the number of batched
    reform passes the scorer runs for them: one per ``calculate`` or
    ``prefetch`` that asks a reformed simulation for a key not asked of it
    before, so a ``prefetch`` of several keys is one pass."""
    asked: dict[tuple[int | None, PostExportKey], None] = {}
    reforms: list[object] = []
    reform_passes: list[int] = []

    def simulate(reform):
        if reform is None:
            return _RecordingPostExportSimulation(asked, None, reforms, reform_passes)
        check_post_export_reform(reform)
        reforms.append(reform)
        return _RecordingPostExportSimulation(
            asked, len(reforms) - 1, reforms, reform_passes
        )

    setattr(simulate, RECORDS_REQUESTS_ONLY, True)
    consumer(simulate)
    baseline = _ascending_period_plan(
        key for simulation, key in asked if simulation is None
    )
    requests = (
        *((None, None, key) for key in baseline),
        *(
            (simulation, reforms[simulation], key)
            for simulation, key in asked
            if simulation is not None
        ),
    )
    return requests, len(reform_passes)


def _record_post_export_request_descriptors(
    name: str, consumer: Callable[[Callable[[Any], Any]], object]
) -> tuple[PostExportRequest, ...]:
    """Every request of the consumer ``name`` as a picklable descriptor.

    One ``PostExportRequest`` per request of the engine-free dry run
    (``_dry_run_post_export_requests``), in the order the serial scorer runs
    them, so a spawned worker process can run each request. The results are
    the values each request calculates; a consumer's output also holds what
    it reads off the reforms it builds itself (the smoke check records each
    list-edit probe's resolved lists from its built reform), so rebuilding an
    output means running the consumer again over the results. Nothing records
    the descriptors: the plan in ``calibration_diagnostics.json`` keeps its
    keys only.
    """
    return tuple(
        PostExportRequest(
            consumer=name,
            position=position,
            variable=variable,
            period=period,
            map_to=map_to,
            reform=reform,
            reformed_simulation=simulation,
        )
        for position, (simulation, reform, (variable, period, map_to)) in enumerate(
            _dry_run_post_export_requests(consumer)
        )
    )


def _record_post_export_requests(
    consumer: Callable[[Callable[[Any], Any]], object],
) -> tuple[tuple[PostExportKey, ...], int]:
    """Learn a consumer's baseline keys and reform pass count from its dry run.

    The baseline keys are the plan the served baseline scores. The pass count
    is the batched reform passes the scorer runs for the dry run's reform
    requests (``_dry_run_post_export``): a ``prefetch`` of several keys of one
    reformed simulation is one pass. Progress reporting alone uses the count:
    a consumer that runs more passes than counted reports no outer counter on
    the extra ones.
    """
    requests, reform_passes = _dry_run_post_export(consumer)
    baseline = tuple(key for simulation, _, key in requests if simulation is None)
    return baseline, reform_passes


def _record_post_export_baseline_plan(
    consumer: Callable[[Callable[[Any], Any]], object],
) -> tuple[PostExportKey, ...]:
    """Learn a consumer's baseline keys (``_record_post_export_requests``)."""
    return _record_post_export_requests(consumer)[0]


class _AscendingPeriodEngine:
    """One batch engine that refuses a period earlier than one it computed."""

    def __init__(self, simulation, *, label: str) -> None:
        self.simulation = simulation
        self._label = label
        self._latest_period: int | None = None

    def calculate(self, key: PostExportKey):
        variable, period, map_to = key
        if self._latest_period is not None and period < self._latest_period:
            raise RuntimeError(
                f"Post-export scoring ({self._label}) asked one engine for "
                f"{variable}@{period} after it computed period "
                f"{self._latest_period}. policyengine-us 2.2.1 can fail to "
                "compute an earlier period after a later one on the same "
                "engine (the MD CCS ParameterNotFoundError; see "
                "_ascending_period_plan), so each engine scores its keys in "
                "ascending period order."
            )
        self._latest_period = period
        if map_to is None:
            return self.simulation.calculate(variable, period)
        return self.simulation.calculate(variable, period, map_to=map_to)


def _post_export_watched_variables(*, batched: bool, reform: bool) -> tuple[str, ...]:
    """The formulas a batch engine may not compute (see the block comment)."""
    watched = US_POPULATION_AGGREGATE_VARIABLES if batched else ()
    if reform:
        watched = (*watched, *POST_EXPORT_BASELINE_BRANCH_READERS)
    return tuple(dict.fromkeys(watched))


def _assert_post_export_scoring_is_batch_invariant(
    simulation,
    watched: Sequence[str],
    known_at_load: set[tuple[str, str]],
    *,
    label: str,
    batched: bool,
    reform: bool,
) -> None:
    """Refuse an engine that computed a formula batching would score wrongly.

    ``known_at_load`` holds what the engine held before scoring (values the
    written H5 stores as inputs, which batching does not change).
    """
    computed = sorted(_engine_known_periods(simulation, watched) - known_at_load)
    if not computed:
        return
    reasons = []
    for variable, period in computed:
        why = []
        if batched and variable in US_POPULATION_AGGREGATE_VARIABLES:
            why.append(
                "aggregates over its simulation's whole population, here one "
                "household batch"
            )
        if reform and variable in POST_EXPORT_BASELINE_BRANCH_READERS:
            why.append(
                "reads the engine's baseline branch, which a batch reform engine "
                "does not carry"
            )
        reasons.append(f"{variable}@{period} ({'; '.join(why)})")
    raise RuntimeError(
        f"Post-export scoring ({label}) is not batch-invariant: a batch engine "
        f"computed {', '.join(reasons)}. A measure that reaches these formulas "
        "cannot be scored in household batches (microcosm#956)."
    )


def _parameter_leaf_values(node, instant: str) -> dict[str, object]:
    """Every parameter under ``node`` (itself included) at ``instant``."""
    return {
        leaf.name: leaf(instant)
        for leaf in (node, *node.get_descendants())
        if hasattr(leaf, "values_list")
    }


def _moved_behavioral_response_parameters(
    reform_system, baseline_system, period: int
) -> list[str]:
    """Behavioral-response parameters a reform system sets off baseline."""
    instant = f"{int(period)}-01-01"
    moved: list[str] = []
    for subtree in POST_EXPORT_BEHAVIORAL_RESPONSE_PARAMETERS.values():
        reformed = _parameter_leaf_values(
            reform_system.parameters.get_child(subtree), instant
        )
        baseline = _parameter_leaf_values(
            baseline_system.parameters.get_child(subtree), instant
        )
        moved.extend(
            name
            for name in sorted(reformed.keys() | baseline.keys())
            if name not in reformed
            or name not in baseline
            or reformed[name] != baseline[name]
        )
    return moved


def _assert_post_export_batching_premises(
    frame: Frame, batch_frames: Sequence[Frame]
) -> None:
    """Refuse a written frame that household batching would score wrongly.

    Batching is output-invariant only if every group unit sits wholly inside
    one household, so that each batch holds whole units and every entity row
    lands in exactly one batch. ``Frame.select`` keeps every group row any
    selected person references without checking that, so a unit spanning two
    batches would be scored twice, each time with part of its members. (The
    Frame constructor already refuses group rows no person references.)
    """
    for entity in US_SCHEMA.group_entities:
        if entity != "household":
            # Raises when a unit's persons sit in more than one household.
            group_to_household_positions(frame, entity)
    for entity in frame.entities:
        batched = sum(batch.n(entity) for batch in batch_frames)
        if batched != frame.n(entity):
            raise ValueError(
                f"Post-export household batches hold {batched} {entity} rows "
                f"against {frame.n(entity)} in the written H5; the batches do "
                "not partition the pool."
            )
    household_ids = frame.table("household")["household_id"].to_numpy()
    batched_ids = np.concatenate(
        [batch.table("household")["household_id"].to_numpy() for batch in batch_frames]
    )
    if not np.array_equal(batched_ids, household_ids):
        raise ValueError(
            "Post-export household batches do not reproduce the written H5's "
            "household order."
        )
    batched_weights = np.concatenate(
        [batch.weights_for("household").values for batch in batch_frames]
    )
    if not np.array_equal(batched_weights, frame.weights_for("household").values):
        raise ValueError(
            "Post-export household batches do not carry the written H5's "
            "household weights."
        )


class _PostExportScoringPlan:
    """The pre-export record of how the written H5 will be scored.

    Built before ``calibration_diagnostics.json`` is written, so that artifact
    carries the plan; the post-export stages then score exactly these
    baseline plans. (A plain class: this tool is loaded by file path in the
    tests, outside ``sys.modules``, where ``dataclasses`` cannot resolve the
    module's string annotations.)
    """

    def __init__(
        self,
        *,
        n_households: int,
        maximum_batch_size: int | None,
        baseline_plans: Mapping[str, tuple[PostExportKey, ...]],
        reform_passes: Mapping[str, int] | None = None,
        error: str | None = None,
    ) -> None:
        self.n_households = int(n_households)
        self.maximum_batch_size = maximum_batch_size
        self.baseline_plans = {
            consumer: tuple(plan) for consumer, plan in baseline_plans.items()
        }
        self.reform_passes = dict(reform_passes or {})
        self.error = error

    @property
    def n_batches(self) -> int:
        return sum(
            1
            for _ in household_position_batches(
                self.n_households, self.maximum_batch_size
            )
        )

    def baseline_plan(self, consumer: str) -> tuple[PostExportKey, ...]:
        if self.error is not None:
            raise RuntimeError(
                "The post-export scoring plan was not built before export: "
                f"{self.error}"
            )
        return self.baseline_plans[consumer]

    def planned_reform_passes(self, consumer: str) -> int | None:
        """The reform passes the dry run counted for ``consumer``, if any.

        Progress reporting reads it as each pass's outer counter; the record
        of the plan leaves it out, so the diagnostics keep their shape.
        """
        if self.error is not None:
            return None
        return self.reform_passes.get(consumer)

    def terminal_failures(self) -> list[str]:
        """The plan's own failure, as a line of the batched terminal raise."""
        if self.error is None:
            return []
        return [
            "Post-export scoring plan could not be built before export "
            f"(microcosm#956): {self.error}"
        ]

    def record(self) -> dict[str, object]:
        if self.error is not None:
            return {"method": POST_EXPORT_SCORING_METHOD, "error": self.error}
        return {
            "method": POST_EXPORT_SCORING_METHOD,
            "maximum_batch_size": self.maximum_batch_size,
            "n_households": self.n_households,
            "n_batches": self.n_batches,
            "spm": dict(US_RELEASE_SPM_SELECTION),
            "period_order": "ascending",
            "consumers": {
                consumer: {"baseline_plan": _post_export_baseline_plan_record(plan)}
                for consumer, plan in self.baseline_plans.items()
            },
        }


def _post_export_scoring_plan(
    *,
    n_households: int,
    maximum_microsim_batch_size: int | None,
    consumers: Mapping[str, Callable[[Callable[[Any], Any]], object]],
) -> _PostExportScoringPlan:
    requests = {
        name: _record_post_export_requests(consumer)
        for name, consumer in consumers.items()
    }
    return _PostExportScoringPlan(
        n_households=int(n_households),
        maximum_batch_size=maximum_microsim_batch_size,
        baseline_plans={name: plan for name, (plan, _) in requests.items()},
        reform_passes={name: passes for name, (_, passes) in requests.items()},
    )


def _record_post_export_scoring_plan(
    args: argparse.Namespace, *, n_households: int, result, release_id: str
) -> _PostExportScoringPlan:
    """The plan ``_main`` records before the calibration diagnostics.

    Never raises (microcosm#547): a crash while building the plan rides the
    plan's record into the diagnostics and joins the batched terminal raise
    (``terminal_failures``), so it cannot pre-empt the diagnostics, the QRF
    tail evidence or any later terminal group, and the run still refuses
    before the export write.
    """
    try:
        return _post_export_scoring_plan(
            n_households=n_households,
            maximum_microsim_batch_size=args.maximum_microsim_batch_size,
            consumers=_post_export_consumers(
                args, result=result, release_id=release_id
            ),
        )
    except Exception as error:
        return _PostExportScoringPlan(
            n_households=n_households,
            maximum_batch_size=args.maximum_microsim_batch_size,
            baseline_plans={},
            error=f"{type(error).__name__}: {error}",
        )


def _reform_coverage_smoke_consumer(simulate) -> GateResult:
    """The smoke exactly as ``_main`` runs it, for the plan's dry run."""
    return us_reform_coverage_smoke_gate(simulate=simulate, period=PERIOD)


def _demographics_consumer(simulate) -> tuple[np.ndarray, np.ndarray]:
    return population_by_age_from_sim(simulate(None), PERIOD)


def _ssi_social_security_holdout_consumer(simulate) -> dict[str, Any]:
    try:
        return ssi_social_security_holdout_from_sim(simulate(None), period=PERIOD)
    except Exception as error:
        # Discovery runs before the terminal release gates. An optional
        # diagnostic must not add a failure there if its inputs are unavailable.
        return ssi_social_security_holdout_unavailable(error, period=PERIOD)


def _medicare_irmaa_holdout_consumer(simulate) -> dict[str, Any]:
    try:
        return medicare_irmaa_holdout_from_sim(simulate(None), period=PERIOD)
    except Exception as error:
        # As for the SSI holdout: an optional diagnostic adds no failure to
        # plan discovery when its inputs are unavailable (microcosm#1206).
        return medicare_irmaa_holdout_unavailable(error, period=PERIOD)


def _reform_validation_consumer(
    *, result, release_id: str
) -> Callable[[Callable[[Any], Any] | None], dict[str, Any]]:
    """reform_validation.json's payload as a function of the simulate seam."""
    specs = load_default_reform_specs(period=PERIOD)
    in_sample_estimates = _in_sample_estimates(result)
    in_sample_targets = _in_sample_targets(result)
    baseline_levels = default_baseline_level_specs()

    def payload_for(simulate) -> dict[str, Any]:
        return reform_validation_payload(
            specs,
            period=PERIOD,
            simulate=simulate,
            in_sample_estimates=in_sample_estimates,
            in_sample_targets=in_sample_targets,
            baseline_levels=baseline_levels,
            release_id=release_id,
        )

    return payload_for


def _post_export_consumers(
    args: argparse.Namespace, *, result, release_id: str
) -> dict[str, Callable[[Callable[[Any], Any]], object]]:
    """The post-export stages this build will run, in run order."""
    consumers: dict[str, Callable[[Callable[[Any], Any]], object]] = {}
    if not args.skip_reform_coverage_smoke:
        consumers["reform_coverage_smoke"] = _reform_coverage_smoke_consumer
    if not args.skip_reform_validation and not args.skip_out_of_sample_reforms:
        consumers["reform_validation"] = _reform_validation_consumer(
            result=result, release_id=release_id
        )
    if not args.skip_demographics:
        consumers["demographics"] = _demographics_consumer
    consumers["ssi_social_security_holdout"] = _ssi_social_security_holdout_consumer
    consumers["medicare_irmaa_holdout"] = _medicare_irmaa_holdout_consumer
    return consumers


#: The directory in a build's ``--out`` in which certification's post-export
#: scorer makes its own directory for its household batch files while it
#: scores. The scorer removes its directory, and this one once empty if it
#: made it, when it closes or is freed; placing a build for certification
#: removes the scorer directories an attempt that was killed left there
#: (``_place_build_for_certification``).
US_POST_EXPORT_BATCH_DIRECTORY = "post_export_batches"

#: The start of the name of each scorer's own batch directory.
_POST_EXPORT_BATCH_DIRECTORY_PREFIX = "post-export-batches-"


class PostExportBatchFileError(RuntimeError):
    """A post-export batch file changed or cannot be read; the batch is refused.

    Not a measurement that cannot be made: the held-out reports, which record
    those as unavailable, raise this (``_write_ssi_social_security_holdout``,
    ``_write_medicare_irmaa_holdout``).
    """


def _new_post_export_batch_directory(
    batch_parent: Path | None,
) -> tuple[Path, Path | None]:
    """A new directory for one scorer's batch files, and the parent it made.

    The directory is new and uniquely named inside ``batch_parent`` (the
    system's temporary directory without one), so a scorer only ever removes
    files it wrote, even when an earlier scorer used the same parent. The
    parent is made if it does not exist, and is then returned, so that the
    scorer can remove it again once it is empty.
    """
    made_parent = None
    if batch_parent is not None:
        batch_parent = Path(batch_parent)
        try:
            batch_parent.mkdir(parents=True)
        except FileExistsError:
            pass
        else:
            made_parent = batch_parent
    directory = Path(
        tempfile.mkdtemp(prefix=_POST_EXPORT_BATCH_DIRECTORY_PREFIX, dir=batch_parent)
    )
    return directory, made_parent


def _remove_post_export_batch_files(directory: Path, made_parent: Path | None) -> None:
    """Remove a scorer's batch directory, and the parent it made once empty."""
    shutil.rmtree(directory, ignore_errors=True)
    if made_parent is not None:
        try:
            made_parent.rmdir()
        except OSError:
            # Not empty: another scorer's directory, or files someone else
            # put there, which this scorer did not write.
            pass


class _PostExportBatchFiles(Sequence):
    """The written H5's household batches, each kept in its own file.

    Built from the batch frames the scorer selected from the H5 it loaded
    once, bound to the H5's sha256: each frame is pickled to its own file in
    ``directory`` and the sha256 of the bytes written is recorded. Indexing
    reads one batch back as a new frame, which the caller frees once it has
    scored the batch, so between batches only the files' paths and digests
    are held. A file's bytes must hash to the recorded digest before they are
    unpickled, so only bytes this process wrote are loaded and a batch file
    that changed is refused. Unpickling gives back the same tables, dtypes,
    index, values, weights and strata, so every batch scores the values the
    in-memory batch did. (A plain class, for the reason
    :class:`_PostExportScoringPlan` gives.)
    """

    def __init__(self, directory: Path, batch_frames: Sequence[Frame]) -> None:
        self.directory = Path(directory)
        files: list[tuple[Path, str]] = []
        for number, batch_frame in enumerate(batch_frames):
            payload = pickle.dumps(batch_frame, protocol=pickle.HIGHEST_PROTOCOL)
            path = self.directory / f"batch-{number:05d}.pickle"
            path.write_bytes(payload)
            files.append((path, hashlib.sha256(payload).hexdigest()))
            del payload
        self._files = tuple(files)

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self) -> Iterator[Frame]:
        # By position, not the Sequence default, which ends at the first
        # IndexError: one raised while reading a batch would end the pass
        # early, leaving later batches unscored.
        for position in range(len(self._files)):
            yield self[position]

    def __getitem__(self, position: int) -> Frame:
        if isinstance(position, slice):
            raise TypeError("Post-export batches are read one at a time.")
        path, sha256 = self._files[position]
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise PostExportBatchFileError(
                f"Post-export batch file {path} cannot be read ({error}); the "
                "batch is refused."
            ) from error
        observed = hashlib.sha256(payload).hexdigest()
        if observed != sha256:
            raise PostExportBatchFileError(
                f"Post-export batch file {path} has SHA-256 {observed}, not the "
                f"{sha256} the scorer wrote; the batch is refused."
            )
        return pickle.loads(payload)


class _HouseholdBatchedPostExportScorer:
    """Scores the written release H5 in household batches (see the block
    comment above). One instance serves every post-export consumer."""

    def __init__(
        self,
        dataset_path: Path,
        *,
        maximum_microsim_batch_size: int | None,
        expected_n_households: int | None = None,
        batch_parent: Path | None = None,
        microsimulation_cls=None,
        dataset_from_frame=None,
        load_frame=None,
        baseline_plans: Mapping[str, Sequence[PostExportKey]] | None = None,
    ) -> None:
        """``baseline_plans`` registers consumers' recorded baseline plans by
        name: the first of them opened scores the union of every registered
        plan in one batch-outer pass, and each is served its own keys from it
        (``_baseline_for``)."""
        self.dataset_path = Path(dataset_path)
        self.maximum_batch_size = maximum_microsim_batch_size
        # The scored bytes are bound to this digest: the loader refuses a file
        # whose bytes differ from it, and the manifest's dataset sha must equal
        # it (``_build_manifests(scored_dataset_sha256=...)``).
        self.dataset_sha256 = _sha256(self.dataset_path)
        frame = (load_frame or _load_frame)(
            self.dataset_path, expected_sha256=self.dataset_sha256
        )
        self.n_households = int(frame.n("household"))
        if (
            expected_n_households is not None
            and self.n_households != expected_n_households
        ):
            raise ValueError(
                f"The written H5 carries {self.n_households} households; the "
                f"post-export scoring plan was built for {expected_n_households}."
            )
        batch_frames = tuple(
            frame
            if len(positions) == self.n_households
            else select_households_by_position(frame, positions)
            for positions in household_position_batches(
                self.n_households, maximum_microsim_batch_size
            )
        )
        _assert_post_export_batching_premises(frame, batch_frames)
        del frame
        self.n_batches = len(batch_frames)
        self.max_batch_households = max(
            (batch.n("household") for batch in batch_frames), default=0
        )
        # The batches go to files in a new directory of this scorer's own,
        # made inside ``batch_parent`` (or the system's temporary directory),
        # and the frames are freed; ``_batches()`` reads each back when a pass
        # scores it. The directory is removed when the scorer closes, is freed
        # or the interpreter exits.
        self.batch_directory, made_parent = _new_post_export_batch_directory(
            batch_parent
        )
        self._remove_batch_files = weakref.finalize(
            self, _remove_post_export_batch_files, self.batch_directory, made_parent
        )
        try:
            self._batch_frames: _PostExportBatchFiles | None = _PostExportBatchFiles(
                self.batch_directory, batch_frames
            )
        except BaseException:
            self._remove_batch_files()
            raise
        del batch_frames
        if microsimulation_cls is None:
            from policyengine_us import Microsimulation

            microsimulation_cls = Microsimulation
        self._microsimulation_cls = microsimulation_cls
        if dataset_from_frame is None:

            def dataset_from_frame(batch_frame: Frame):
                # Consumer plans read years after PERIOD (reform validation
                # reaches past 2025), so the scorer keeps the engine's
                # projection: it does not ask for PERIOD alone.
                return _dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                )

        self._dataset_from_frame = dataset_from_frame
        # Each finished consumer's scoring record, for the manifests' block.
        # Records, not consumers: a finished consumer's arrays must be freed.
        self.consumer_records: dict[str, dict[str, object]] = {}
        # Every registered plan, which an opened consumer must match; the
        # non-empty ones not yet served; and the merged pass's values of the
        # keys those still need (None until that pass runs).
        self._registered_baseline_plans: dict[str, tuple[PostExportKey, ...]] = {
            name: tuple(plan) for name, plan in (baseline_plans or {}).items()
        }
        self._unserved_baseline_plans = {
            name: plan for name, plan in self._registered_baseline_plans.items() if plan
        }
        self._merged_baseline: dict[PostExportKey, _PostExportValues] | None = None
        self._merged_baseline_failed = False
        # The reform last handed to any consumer. Consumers run one after
        # another, so at most one reform's system and rule modules are held;
        # opening the next reform, finishing its consumer or closing the scorer
        # releases it.
        self._open_reform: _BatchedPostExportReform | None = None

    def open_consumer(
        self,
        name: str,
        baseline_plan: Sequence[PostExportKey],
        planned_reform_passes: int | None = None,
    ) -> _PostExportConsumer:
        """Score or serve ``baseline_plan`` now (``_baseline_for``); return the
        consumer's simulate seam.

        With ``planned_reform_passes``, each reform pass reports its number of
        that count as its progress's outer counter.
        """
        return _PostExportConsumer(
            self,
            name,
            tuple(baseline_plan),
            planned_reform_passes=planned_reform_passes,
        )

    def finish_consumer(self, scoring: _PostExportConsumer) -> dict[str, object]:
        """Record a consumer that has finished scoring and return its record."""
        if scoring._scorer is not self:
            raise ValueError(f"{scoring.name} was not opened on this scorer.")
        if scoring.name in self.consumer_records:
            raise ValueError(f"{scoring.name} already finished on this scorer.")
        if self._open_reform is not None and self._open_reform._consumer is scoring:
            self._finish_open_reform()
        record = scoring.record()
        self.consumer_records[scoring.name] = record
        return record

    def manifest_record(self) -> dict[str, object]:
        """The ``post_export_scoring`` block both manifests carry.

        Route A remediation PR-3 records the evidence the gates judged in the
        manifests, not only in loose diagnostics. This block names the bytes
        scored, the batching, and each finished consumer's baseline plan and
        pass counts, in the order the consumers finished.
        """
        return {
            "method": POST_EXPORT_SCORING_METHOD,
            "dataset_sha256": self.dataset_sha256,
            "maximum_batch_size": self.maximum_batch_size,
            "n_households": self.n_households,
            "n_batches": self.n_batches,
            "max_batch_households": self.max_batch_households,
            "spm": dict(US_RELEASE_SPM_SELECTION),
            "consumers": {
                name: {
                    key: record[key]
                    for key in (
                        "baseline_plan",
                        "baseline_passes",
                        "reform_passes",
                        "reform_systems",
                    )
                }
                for name, record in self.consumer_records.items()
            },
        }

    def close(self) -> None:
        self._finish_open_reform()
        self._batch_frames = None
        self._merged_baseline = None
        collect_pass_garbage()
        self._remove_batch_files()

    def _open_reform_for(
        self, consumer: _PostExportConsumer, reform
    ) -> _BatchedPostExportReform:
        """Hand ``consumer`` a reform, finishing the one handed out before.

        Each shipped consumer scores one reform completely before it opens the
        next, and consumers run one after another.
        """
        self._finish_open_reform()
        self._open_reform = _BatchedPostExportReform(consumer, reform)
        return self._open_reform

    def _finish_open_reform(self) -> None:
        reform, self._open_reform = self._open_reform, None
        if reform is not None:
            reform.finish()

    def _batches(self) -> _PostExportBatchFiles:
        """The batches, each read from its file when it is indexed."""
        if self._batch_frames is None:
            raise RuntimeError("The post-export scorer is closed.")
        return self._batch_frames

    def _baseline_for(
        self, name: str, baseline_plan: tuple[PostExportKey, ...]
    ) -> tuple[dict[PostExportKey, _PostExportValues], bool]:
        """Score ``name``'s baseline plan, or serve it from the merged pass.

        Returns the plan's values and whether they came from the merged pass.
        A registered consumer opened with any plan but its registered one is
        refused. The first registered consumer opened scores the union of
        every unserved registered plan in one batch-outer pass, in ascending
        period order; each registered consumer is then served exactly its own
        keys, once, and the merged values are dropped once no unserved
        registered plan holds them. A consumer without a registered plan, or
        opened again, scores its own plan in a pass of its own. If the merged
        pass raises, every consumer scores its own plan, so a key that only
        one consumer planned fails that consumer alone, with the error it
        raised before the merge (the holdout reports catch it). A refused
        batch file (:class:`PostExportBatchFileError`) is the exception: it
        propagates at once, as it does from every other post-export pass, so
        it stops certification (or the guard sweep's baseline worker) and no
        consumer scores its own plan.
        """
        registered = self._registered_baseline_plans.get(name)
        if registered is not None and registered != baseline_plan:
            raise ValueError(
                f"{name}: opened with a baseline plan other than its registered "
                "baseline plan; a registered consumer is served the plan recorded "
                "before export."
            )
        if not baseline_plan:
            return {}, False
        self._batches()
        if name not in self._unserved_baseline_plans or self._merged_baseline_failed:
            return self._score(baseline_plan, label=f"{name} baseline plan"), False
        if self._merged_baseline is None:
            union = _ascending_period_plan(
                key for plan in self._unserved_baseline_plans.values() for key in plan
            )
            try:
                self._merged_baseline = self._score(union, label="merged baseline plan")
            except PostExportBatchFileError:
                # The scored file's integrity, not a key one consumer planned:
                # it stops certification.
                raise
            except Exception as error:
                self._merged_baseline_failed = True
                try:
                    reason = f"{type(error).__name__}: {error}"
                except Exception:
                    reason = type(error).__name__
                print(
                    f"Post-export merged baseline plan failed ({reason}); each "
                    "consumer scores its own baseline plan instead.",
                    flush=True,
                )
            if self._merged_baseline_failed:
                # Outside the handler, so the failed pass's traceback and the
                # per-batch values its frame collected are no longer
                # referenced; a full collection frees any held in a cycle.
                collect_pass_garbage()
                return self._score(baseline_plan, label=f"{name} baseline plan"), False
        del self._unserved_baseline_plans[name]
        served = {key: self._merged_baseline[key] for key in baseline_plan}
        still_planned = {
            key for plan in self._unserved_baseline_plans.values() for key in plan
        }
        for key in tuple(self._merged_baseline):
            if key not in still_planned:
                del self._merged_baseline[key]
        return served, True

    def _construct(self, batch_frame: Frame, *, reform_system):
        dataset = self._dataset_from_frame(batch_frame)
        if reform_system is None:
            return self._microsimulation_cls(
                dataset=dataset, spm=dict(US_RELEASE_SPM_SELECTION)
            )
        # The reform's system alone, not ``reform=`` as well. Given both,
        # policyengine-us 2.2.1 clones the supplied system on every
        # construction (``SPMSimulationMixin._prepare_spm_system`` ->
        # ``clone_spm_system``: the whole parameter tree rebuilt and every
        # variable deep-copied, so each batch starts on cold caches), core
        # applies the reform again to the clone, and core adds a ``baseline``
        # branch whose holders copy every input array. Given the system alone
        # it shares the system's policy (``share_spm_policy``).
        # ``POST_EXPORT_BASELINE_BRANCH_READERS`` and
        # ``POST_EXPORT_BEHAVIORAL_RESPONSE_PARAMETERS`` list formulas that
        # read the baseline branch, and the scorer guards those uses.
        return self._microsimulation_cls(
            tax_benefit_system=reform_system,
            dataset=dataset,
            spm=dict(US_RELEASE_SPM_SELECTION),
        )

    def _assert_reform_keeps_baseline_responses(
        self, reform_system, key: PostExportKey, *, label: str
    ) -> None:
        """Refuse a reform that moves a behavioral-response parameter.

        A batch reform engine has no ``baseline`` branch, so policyengine-us
        scores its labor-supply and capital-gains responses as 0. A whole-file
        reform engine has one and scores them from these parameters, as 0 at
        the engine's zero default elasticities. The guard requires these
        parameters to stay at baseline before building a batch reform engine.
        """
        moved = _moved_behavioral_response_parameters(
            reform_system,
            self._microsimulation_cls.default_tax_benefit_system_instance,
            key[1],
        )
        if moved:
            raise RuntimeError(
                f"Post-export scoring ({label}) refuses a reform that sets "
                f"behavioral-response parameters off baseline at {key[1]}: "
                f"{', '.join(moved)}. Batch reform engines carry no baseline "
                "branch, so policyengine-us would score these responses as zero "
                "(microcosm#956)."
            )

    def _score(
        self,
        keys: Sequence[PostExportKey],
        *,
        label: str,
        reform_system=None,
        outer: OuterProgress | None = None,
    ) -> dict[PostExportKey, _PostExportValues]:
        """One batch-outer pass: one engine per batch scores every key.

        ``outer`` is the pass's place among its consumer's passes, sent with
        each batch's progress.
        """
        batches = self._batches()
        batched = len(batches) > 1
        watched = _post_export_watched_variables(
            batched=batched, reform=reform_system is not None
        )
        if batched:
            print(
                f"Scoring post-export {label} ({len(keys)} key(s)) in "
                f"{len(batches)} batches of up to {self.max_batch_households:,} "
                "households.",
                flush=True,
            )
        parts: dict[PostExportKey, list[tuple[np.ndarray, np.ndarray]]] = {
            key: [] for key in keys
        }
        sweep = _WorkCounter(
            stage_id="post_export_scoring",
            unit="scoring_batch",
            total=len(batches),
            outer=outer,
        )
        for batch in range(1, len(batches) + 1):
            # Read by position: ``enumerate`` keeps its last item, so the
            # previous batch's frame would stay alive while the next is read.
            batch_frame = batches[batch - 1]
            with automatic_gc_suspended():
                simulation = self._construct(batch_frame, reform_system=reform_system)
                # The engine holds its own copy of the batch; the frame read
                # from the batch's file is freed before the next one is read.
                del batch_frame
                try:
                    known_at_load = _engine_known_periods(simulation, watched)
                    engine = _AscendingPeriodEngine(simulation, label=label)
                    for key in keys:
                        parts[key].append(
                            _post_export_values_and_weights(engine.calculate(key), key)
                        )
                    # Before the release below drops the engine's holders.
                    _assert_post_export_scoring_is_batch_invariant(
                        simulation,
                        watched,
                        known_at_load,
                        label=label,
                        batched=batched,
                        reform=reform_system is not None,
                    )
                finally:
                    # microcosm#456: free this batch engine's array mass by
                    # refcount before the next batch builds its own, also
                    # when a key raises.
                    release_engine_simulation(simulation)
                del simulation, engine
            collect_batch_garbage()
            sweep.advance(pass_name=label, batch=batch, batches=len(batches))
        collect_pass_garbage()
        return {
            key: _concatenate_post_export_parts(key_parts)
            for key, key_parts in parts.items()
        }


def _open_post_export_scorer(
    plan: _PostExportScoringPlan,
    dataset_path: Path,
    *,
    batch_parent: Path | None = None,
) -> _HouseholdBatchedPostExportScorer | None:
    """Open the scorer the plan recorded before export, or None if no stage runs.

    The scorer keeps its household batch files in a new directory of its own
    inside ``batch_parent`` (the system's temporary directory without one).
    """
    if plan.error is not None:
        raise RuntimeError(
            f"The post-export scoring plan was not built before export: {plan.error}"
        )
    if not plan.baseline_plans:
        return None
    return _HouseholdBatchedPostExportScorer(
        dataset_path,
        maximum_microsim_batch_size=plan.maximum_batch_size,
        batch_parent=batch_parent,
        expected_n_households=plan.n_households,
        baseline_plans=plan.baseline_plans,
    )


def _post_export_scoring_manifest_block(
    scorer: _HouseholdBatchedPostExportScorer | None,
) -> dict[str, object] | None:
    """The manifests' ``post_export_scoring`` block (None: no stage ran)."""
    if scorer is None:
        return None
    return scorer.manifest_record()


def _close_post_export_scorer(
    scorer: _HouseholdBatchedPostExportScorer | None,
) -> str | None:
    """Close the scorer; return the sha256 of the H5 it scored (None: no stage)."""
    if scorer is None:
        return None
    scorer.close()
    return scorer.dataset_sha256


class _PostExportConsumer:
    """One consumer's simulate seam over a scored baseline plan."""

    def __init__(
        self,
        scorer: _HouseholdBatchedPostExportScorer,
        name: str,
        baseline_plan: tuple[PostExportKey, ...],
        *,
        planned_reform_passes: int | None = None,
    ) -> None:
        ordered = _ascending_period_plan(baseline_plan)
        if ordered != baseline_plan:
            raise ValueError(
                f"{name}: a post-export baseline plan must be unique keys in "
                "ascending period order."
            )
        self._scorer = scorer
        self.name = name
        self.baseline_plan = baseline_plan
        self.baseline_passes = 1 if baseline_plan else 0
        self.reform_passes = 0
        self.planned_reform_passes = planned_reform_passes
        self.reform_systems = 0
        self._baseline, self.baseline_from_merged_pass = scorer._baseline_for(
            name, baseline_plan
        )

    def simulate(self, reform):
        if reform is None:
            return _PlannedPostExportBaseline(self)
        return self._scorer._open_reform_for(self, reform)

    def record(self) -> dict[str, object]:
        scorer = self._scorer
        return {
            "method": POST_EXPORT_SCORING_METHOD,
            "dataset_sha256": scorer.dataset_sha256,
            "maximum_batch_size": scorer.maximum_batch_size,
            "n_households": scorer.n_households,
            "n_batches": scorer.n_batches,
            "max_batch_households": scorer.max_batch_households,
            "spm": dict(US_RELEASE_SPM_SELECTION),
            "baseline_plan": _post_export_baseline_plan_record(self.baseline_plan),
            "baseline_passes": self.baseline_passes,
            "reform_passes": self.reform_passes,
            "reform_systems": self.reform_systems,
        }

    def _baseline_values(self, key: PostExportKey) -> _PostExportValues:
        try:
            return self._baseline[key]
        except KeyError:
            raise RuntimeError(
                f"{self.name} asked the baseline for {key}, which its recorded "
                "plan does not hold; the engine-free dry run must see every "
                "baseline request, so nothing is scored off-plan."
            ) from None

    def _reform_values(
        self, reform_system, keys: tuple[PostExportKey, ...]
    ) -> dict[PostExportKey, _PostExportValues]:
        """One batched reform pass over ``keys`` (ascending period order)."""
        label = f"{self.name} reform " + ", ".join(
            f"{variable}@{period}" for variable, period, _ in keys
        )
        checked_periods: set[int] = set()
        for key in keys:
            if key[1] not in checked_periods:
                checked_periods.add(key[1])
                self._scorer._assert_reform_keeps_baseline_responses(
                    reform_system, key, label=label
                )
        self.reform_passes += 1
        planned = self.planned_reform_passes
        outer = (
            OuterProgress(self.reform_passes, planned, "reform_pass")
            if planned is not None and self.reform_passes <= planned
            else None
        )
        return self._scorer._score(
            keys, label=label, reform_system=reform_system, outer=outer
        )


class _PlannedPostExportBaseline:
    """The baseline simulation a consumer sees: served from its scored plan."""

    def __init__(self, consumer: _PostExportConsumer) -> None:
        self._consumer = consumer

    def calculate(self, variable, period=None, map_to=None) -> _PostExportValues:
        return self._consumer._baseline_values(
            _post_export_key(variable, period, map_to)
        )

    def prefetch(self, requests) -> None:
        """Nothing to score: refuse any request outside the served plan, as
        ``calculate`` would (the dry run's recorder takes ``prefetch`` too)."""
        for request in requests:
            self._consumer._baseline_values(_post_export_key(*request))


class _BatchedPostExportReform:
    """A reformed simulation a consumer sees: each key not yet scored, or each
    ``prefetch`` of several such keys, is one batched pass.

    The reform's tax-benefit system is built once, on the first pass, and
    handed to every batch engine (microcosm#456). The engines get that system
    alone (see ``_HouseholdBatchedPostExportScorer._construct``).

    Building the system makes policyengine-core add every variable module of
    it to ``sys.modules`` (about 6,000 entries) under names it never removes;
    on the eight-household guard fixture they hold about 0.04 GiB per reform.
    The build opens an ``EngineVariableModuleScope``, which stays open until
    the scorer hands out its next reform, the consumer finishes or the scorer
    closes (:meth:`finish`), and then removes those modules. A finished reform
    refuses further keys.
    """

    def __init__(self, consumer: _PostExportConsumer, reform) -> None:
        self._consumer = consumer
        self._reform = reform
        self._system = None
        self._results: dict[PostExportKey, _PostExportValues] = {}
        self._rule_modules: EngineVariableModuleScope | None = None
        self._finished = False

    def calculate(self, variable, period=None, map_to=None) -> _PostExportValues:
        key = _post_export_key(variable, period, map_to)
        self._score_new((key,))
        return self._results[key]

    def prefetch(self, requests) -> None:
        """Score every requested key not yet scored in one batched pass.

        ``requests`` are ``(variable, period)`` or ``(variable, period,
        map_to)`` tuples. Each batch engine computes them in ascending period
        order, and later ``calculate`` calls for them are served from the
        pass. The dry run counts it as one pass
        (``_RecordingPostExportSimulation.prefetch``).
        """
        self._score_new(tuple(_post_export_key(*request) for request in requests))

    def _score_new(self, keys: Sequence[PostExportKey]) -> None:
        if self._finished:
            asked = keys[0] if len(keys) == 1 else tuple(keys)
            raise RuntimeError(
                f"{self._consumer.name} asked a finished reform for {asked}. A "
                "post-export consumer scores each reform completely before it "
                "opens the next one, which finishes and releases the earlier."
            )
        new_keys = _ascending_period_plan(
            key for key in keys if key not in self._results
        )
        if not new_keys:
            return
        if self._system is None:
            scorer = self._consumer._scorer
            rule_modules = EngineVariableModuleScope().open()
            try:
                self._system = scorer._microsimulation_cls.default_tax_benefit_system(
                    reform=self._reform
                )
            except BaseException:
                rule_modules.close()
                raise
            self._rule_modules = rule_modules
            self._consumer.reform_systems += 1
        self._results.update(self._consumer._reform_values(self._system, new_keys))

    def finish(self) -> None:
        """Drop the reform's system and remove the modules its build added."""
        if self._finished:
            return
        self._finished = True
        self._system = None
        self._results = {}
        rule_modules, self._rule_modules = self._rule_modules, None
        if rule_modules is not None:
            rule_modules.close()


def _reform_household_income_tax(
    *,
    base_frame: Frame,
    reform_spec,
    system,
    microsimulation_cls,
    n_households: int,
    batch_size: int | None,
    refuse_population_aggregates: bool | None = None,
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
    # Each batch engine gets this system alone, not ``reform=`` as well, as the
    # post-export scorer's do (``_HouseholdBatchedPostExportScorer._construct``).
    # Given both, policyengine-us 2.2.1 clones the supplied system on every
    # construction (``SPMSimulationMixin._prepare_spm_system`` ->
    # ``clone_spm_system``: the parameter tree rebuilt and every variable
    # deep-copied, so each batch starts on cold parameter caches), core applies
    # the reform again to the clone, and core adds a ``baseline`` branch whose
    # holders copy every input array: one clone per reform per batch, 1,947 in
    # a release's target compilation. Given the system alone, the engine shares
    # the system's policy (``share_spm_policy``) and has no ``baseline``
    # branch, so this pass refuses the two ways a missing one changes what an
    # engine scores: a reform that moves the behavioral-response parameters
    # (below), and an engine that computed a formula reading the branch
    # (``_refuse_batch_baseline_branch_readers``).
    _refuse_reform_moving_behavioral_responses(
        reform_system,
        microsimulation_cls.default_tax_benefit_system_instance,
        measure=reform_spec.measure,
    )
    batches = tuple(household_position_batches(n_households, batch_size))
    guard_armed = (
        len(batches) > 1
        if refuse_population_aggregates is None
        else refuse_population_aggregates
    )
    if guard_armed:
        assert_medicaid_claiming_tax_units_local(
            base_frame, work=TARGET_MATERIALIZATION_PASS
        )
    if len(batches) > 1:
        print(
            "Materializing reform target "
            f"{reform_spec.measure} in {len(batches)} batches "
            f"of up to {batch_size:,} households.",
            flush=True,
        )
    for batch, household_positions in enumerate(batches, start=1):
        with automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                base_frame
                if full_batch
                else select_households_by_position(base_frame, household_positions)
            )
            batch_tax_unit_positions = _tax_unit_to_household_positions(batch_frame)
            reformed_dataset = _dataset_from_frame(
                batch_frame,
                zero_variables=(reform_spec.neutralized_variable,),
                system=system,
                assert_no_formula_owned_columns=False,
                period_only=True,
            )
            reformed = microsimulation_cls(
                tax_benefit_system=reform_system,
                dataset=reformed_dataset,
            )
            try:
                # Branches are recorded at every batch count, because the
                # baseline-branch refusal applies to a single batch too.
                with record_engine_branches(reformed):
                    batch_income_tax = _collapse_tax_unit(
                        _calculate_array(reformed, "income_tax"),
                        batch_tax_unit_positions,
                        batch_frame.n("household"),
                    )
                    if guard_armed:
                        refuse_batch_population_aggregates(
                            reformed,
                            batch=batch,
                            batches=len(batches),
                            work=TARGET_MATERIALIZATION_PASS,
                            remedy=TARGET_POPULATION_AGGREGATE_REMEDY,
                        )
                    _refuse_batch_baseline_branch_readers(
                        reformed,
                        measure=reform_spec.measure,
                        batch=batch,
                        batches=len(batches),
                    )
            finally:
                release_engine_simulation(reformed)
            reform_income_tax[household_positions] = batch_income_tax
            del batch_income_tax, reformed, reformed_dataset, batch_frame
        collect_batch_garbage()
        _advance_work(pass_name=reform_spec.measure, batch=batch, batches=len(batches))
    del reform_system
    collect_pass_garbage()
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
            group_to_household_positions(frame, entity),
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
            group_to_household_positions(frame, entity),
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
    """Ledger filter keys the SOI slice does not act on, for one spec.

    A non-empty result drops the spec from SOI materialization silently in
    the ``irs_soi`` loop of :func:`_materialize_target_frame`, so an
    accepted restatement must clear here too — otherwise
    accepting it at the fatal guard would only move the spec from a refusal
    to a silent disappearance. A restatement that disagrees stays listed, and
    :func:`_assert_supported_ledger_filter_metadata` refuses it before the
    materializer ever reaches this skip.
    """

    return tuple(
        sorted(
            key
            for key, value in metadata.items()
            if key.startswith("ledger_filter_")
            and key not in SUPPORTED_SOI_LEDGER_FILTERS
            and not _is_noop_ledger_filter_value(str(value))
            and _restated_ledger_filter_refusal(str(key), str(value), metadata)
            is not None
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


def _restated_ledger_filter_concept(key: str) -> tuple[str, str | None]:
    """Split a ``ledger_filter_*`` key into its concept and bound side.

    ``ledger_filter_us:statutes/26/62#adjusted_gross_income_lower_bound``
    splits into ``("us:statutes/26/62#adjusted_gross_income", "lower")``; a
    key with no bound suffix keeps the whole concept and side ``None``,
    meaning it restates an exact value.
    """

    if not key.startswith("ledger_filter_"):
        return "", None
    concept = key[len("ledger_filter_") :]
    for suffix, side in RESTATED_LEDGER_FILTER_BOUND_SIDES:
        if concept.endswith(suffix) and len(concept) > len(suffix):
            return concept[: -len(suffix)], side
    return concept, None


def _restated_bound_value(value: str) -> float | None:
    try:
        return _as_bound(value.strip())
    except ValueError:
        return None


def _restated_count_value(value: str) -> float | None:
    try:
        count = float(value.strip())
    except ValueError:
        return None
    if not count.is_integer() or not 0 <= count <= RESTATED_EITC_CHILD_COUNT_PROBE_MAX:
        return None
    return count


def _restated_band_edge_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    compiled_key: str,
) -> str | None:
    """Judge one restated half-open band edge against the edge it restates.

    Shared by the AGI and age rules, whose materializers both slice
    ``lower <= x < upper`` on the compiled edge read through
    :func:`_as_bound`, so a restated edge is ignorable exactly when it parses
    to the same number. A missing compiled edge refuses: the age paths would
    read it as the open end and the ``irs_soi`` loop indexes it directly, and
    neither is the restated edge being checked.
    """

    compiled = metadata.get(compiled_key)
    if compiled is None:
        return f"{key}={value} restates a bound the spec does not compile: no {compiled_key}"
    restated = _restated_bound_value(value)
    applied = _restated_bound_value(str(compiled))
    if restated is None or applied is None or restated != applied:
        return f"{key}={value} disagrees with {compiled_key}={compiled}"
    return None


def _restated_agi_band_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    side: str | None,
) -> str | None:
    if side is None:
        return (
            f"{key}={value} restates an exact AGI, but the materializer "
            "slices a half-open AGI band and applies no exact-value AGI filter"
        )
    return _restated_band_edge_refusal(
        key, value, metadata, compiled_key=RESTATED_AGI_BAND_COMPILED_KEYS[side]
    )


def _restated_age_band_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    side: str | None,
) -> str | None:
    if side is None:
        return (
            f"{key}={value} restates an exact age, but the materializer "
            "slices a half-open age band and applies no exact-age filter"
        )
    if RESTATED_AGE_DIMENSION_KEY in metadata:
        return (
            f"{key}={value} restates an age bound on a fact whose dimensions "
            f"include age ({RESTATED_AGE_DIMENSION_KEY}="
            f"{metadata[RESTATED_AGE_DIMENSION_KEY]}), so the bound came from "
            "the dimension stamp, which carries no operator"
        )
    materializer = metadata.get("materializer")
    if materializer not in RESTATED_AGE_BAND_MATERIALIZERS:
        return (
            f"{key}={value} restates an age bound on a spec whose materializer "
            f"({materializer!r}) applies no age band"
        )
    stamp_source = metadata.get(AGE_BOUND_STAMP_SOURCE_KEY)
    if stamp_source != AGE_BOUND_STAMP_FROM_CONSTRAINT_ROWS:
        return (
            f"{key}={value} restates an age bound whose operator is ambiguous: "
            f"{AGE_BOUND_STAMP_SOURCE_KEY}={stamp_source} does not attest it "
            "was stamped from a constraint row (>= or <), and a dimension of "
            "that name carries no operator"
        )
    return _restated_band_edge_refusal(
        key, value, metadata, compiled_key=RESTATED_AGE_BAND_COMPILED_KEYS[side]
    )


def _restated_student_loan_balance_band_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    side: str | None,
) -> str | None:
    if side is None:
        return (
            f"{key}={value} restates an exact federal student loan balance, but "
            "the materializer slices a half-open balance band"
        )
    materializer = metadata.get("materializer")
    if materializer != FSA_MATERIALIZER:
        return (
            f"{key}={value} restates a federal student loan balance bound on a "
            f"spec whose materializer ({materializer!r}) applies no balance band"
        )
    return _restated_band_edge_refusal(
        key,
        value,
        metadata,
        compiled_key=RESTATED_STUDENT_LOAN_BALANCE_BAND_COMPILED_KEYS[side],
    )


def _restated_eitc_child_count_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
    *,
    side: str | None,
) -> str | None:
    if side in RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES:
        return (
            f"{key}={value} restates a qualifying-child {side} bound, refused "
            "outright: the Ledger's operator for it (< or <=) is unconfirmed, "
            "and the two readings select different returns"
        )
    compiled = _soi_eitc_child_count_filter(metadata)
    if compiled is None:
        return (
            f"{key}={value} restates a qualifying-child bound on a spec that "
            "carries no child-count filter for the materializer to apply"
        )
    bound = _restated_count_value(value)
    if bound is None:
        return (
            f"{key}={value} is not a qualifying-child count in "
            f"0..{RESTATED_EITC_CHILD_COUNT_PROBE_MAX}"
        )
    counts = np.arange(RESTATED_EITC_CHILD_COUNT_PROBE_MAX + 1, dtype=np.float64)
    if side == "lower":
        restated_mask = counts >= bound
    elif side is None:
        restated_mask = counts == bound
    else:
        raise ValueError(
            f"{key}: bound side {side!r} reached the qualifying-child "
            "comparison, which reads only lower and exact restatements"
        )
    try:
        applied_mask = _eitc_child_count_mask(counts, compiled)
    except ValueError:
        return (
            f"{key}={value} cannot be compared: compiled child-count filter "
            f"{compiled!r} is not one the materializer understands"
        )
    if not np.array_equal(restated_mask, applied_mask):
        return (
            f"{key}={value} selects different returns than the compiled "
            f"child-count filter {compiled!r}"
        )
    return None


def _restated_ledger_filter_refusal(
    key: str,
    value: str,
    metadata: Mapping[str, str],
) -> str | None:
    """Judge one otherwise-unsupported ``ledger_filter_*`` key for one spec.

    Returns ``None`` only when *key* restates, in the labelled feed's concept
    vocabulary, a constraint the materializer already applies to this spec,
    and the two select the same population — ignoring that restatement then
    changes nothing. Every other key returns the entry to refuse with: the
    bare key for a filter the materializer does not model (unchanged from
    before this rule existed), or the key with both values when a restatement
    disagrees with its compiled counterpart, or has none, or is applied by
    no materializer for this spec, or — for a
    qualifying-child upper bound, whose operator is unconfirmed (see
    :data:`RESTATED_EITC_CHILD_COUNT_REFUSED_SIDES`) — the key with that
    reason, whatever the spec compiles.

    Two compiled counterparts — the AGI band in
    :data:`RESTATED_AGI_BAND_COMPILED_KEYS` and
    :func:`_soi_eitc_child_count_filter` — are the ``irs_soi`` slice's, read
    in the loop :func:`_materialize_target_frame` runs over ``irs_soi``
    specs, and this rule reads them off metadata without consulting the
    spec's family. That family-blindness is the guard's existing shape:
    ``ledger_filter_eitc_child_count`` is likewise a blanket supported key. A
    spec of another family carries neither counterpart, so it refuses on the
    "does not compile" arm rather than being accepted by accident.

    The third, the person-age band in :data:`RESTATED_AGE_BAND_COMPILED_KEYS`,
    is read by two materializers (``population_age`` and the age-banded
    ``policyengine_variable`` branch), so that rule is family-blind too but
    names the materializers that apply the band
    (:data:`RESTATED_AGE_BAND_MATERIALIZERS`) and refuses on every other; it
    also refuses a bound on a fact whose dimensions include ``age``
    (:data:`RESTATED_AGE_DIMENSION_KEY`), and any bound the compile does not
    attest was stamped from a ``>=`` / ``<`` constraint row
    (:data:`AGE_BOUND_STAMP_SOURCE_KEY`): a dimension-stamped bound carries no
    operator, and ``age < 10`` and ``age <= 10`` select different people.
    """

    concept, side = _restated_ledger_filter_concept(key)
    rule = RESTATED_LEDGER_FILTER_CONCEPTS.get(concept) if concept else None
    if rule is None:
        return key
    if rule == "agi_band":
        return _restated_agi_band_refusal(key, value, metadata, side=side)
    if rule == "eitc_child_count":
        return _restated_eitc_child_count_refusal(key, value, metadata, side=side)
    if rule == "age_band":
        return _restated_age_band_refusal(key, value, metadata, side=side)
    if rule == "student_loan_balance_band":
        return _restated_student_loan_balance_band_refusal(
            key, value, metadata, side=side
        )
    raise ValueError(
        f"RESTATED_LEDGER_FILTER_CONCEPTS maps {concept!r} to rule {rule!r}, "
        "which has no comparison implemented."
    )


def _unsupported_ledger_filter_metadata(
    target_specs: Iterable[object],
) -> dict[str, tuple[str, ...]]:
    unsupported: dict[str, tuple[str, ...]] = {}
    for spec in target_specs:
        metadata = getattr(spec, "metadata", None)
        if not isinstance(metadata, Mapping):
            continue
        refusals = []
        for key, value in metadata.items():
            key = str(key)
            if not key.startswith("ledger_filter"):
                continue
            if key in SUPPORTED_LEDGER_FILTER_METADATA_KEYS:
                continue
            if key in IDENTITY_LEDGER_FILTER_METADATA_KEYS:
                continue
            if _is_noop_ledger_filter_value(str(value)):
                continue
            refusal = _restated_ledger_filter_refusal(key, str(value), metadata)
            if refusal is not None:
                refusals.append(refusal)
        if refusals:
            unsupported[str(getattr(spec, "name", "<unnamed target>"))] = tuple(
                sorted(refusals)
            )
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


_signed_component = recert.signed_component


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
    target_frame_checkpoint_build_commit: str | None = None,
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
        and target_frame_checkpoint_build_commit is None
    ):
        raise ValueError(
            "target_frame_checkpoint_build_commit is required when "
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
            build_commit=str(target_frame_checkpoint_build_commit),
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


def _base_simulation_household_columns(
    frame: Frame,
    target_specs: tuple,
    *,
    simulation,
    system,
) -> dict[str, np.ndarray]:
    """Household target columns that one base simulation materializes.

    Returns the columns in the order :func:`_materialize_target_frame` writes
    them onto the household table. Each value is built from one household's
    own persons, tax units and groups, collapsed onto that household and
    masked by that household's own geography, so this function combines no
    two households. Its engine dependencies can still aggregate across the
    population; :func:`_materialize_base_simulation_columns` refuses the
    watched aggregates when running more than one household batch.
    """

    household = frame.table("household")
    tax_unit_positions = _tax_unit_to_household_positions(frame)
    n_households = frame.n("household")
    columns: dict[str, np.ndarray] = {}

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

    columns["income_tax"] = _collapse_tax_unit(
        income_tax_tax_unit, tax_unit_positions, n_households
    )
    columns["state_income_tax"] = _household_values(
        frame=frame,
        simulation=simulation,
        system=system,
        variable="state_income_tax",
        tax_unit_positions=tax_unit_positions,
    )
    # Federal Student Aid portfolio targets (microcosm#1074) read the SCF
    # education-debt stage's stored person balances, not an engine variable:
    # the pinned engine does not define them yet, and a stored input is what
    # the engine would read anyway. A frame without the columns materializes
    # nothing, so the targets report as dropped.
    for spec in target_specs:
        if spec.metadata.get("materializer") != FSA_MATERIALIZER:
            continue
        fsa_values = fsa_student_loan_household_values(frame, spec.metadata)
        if fsa_values is not None:
            columns[spec.measure] = fsa_values
    population_age_target_specs = [
        spec
        for spec in target_specs
        if spec.metadata.get("materializer") == "population_age"
    ]
    if population_age_target_specs and "age" in system.variables:
        person_age = np.asarray(_calculate_array(simulation, "age"), dtype=np.float64)
        for spec in population_age_target_specs:
            columns[spec.measure] = _population_age_household_values(
                frame=frame,
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
            columns[spec.measure] = _collapse_person(
                frame, person_values * band_mask.astype(np.float64)
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
                frame=frame,
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
        columns[spec.measure] = values

    direct_measures = {
        spec.measure
        for spec in target_specs
        if spec.measure
        and spec.family not in {"irs_soi", "jct", "state_income_tax"}
        and spec.metadata.get("materializer") != "policyengine_variable"
        and spec.measure not in household.columns
        and spec.measure not in columns
    }
    for measure in sorted(direct_measures):
        if measure not in system.variables:
            continue
        columns[measure] = _household_values(
            frame=frame,
            simulation=simulation,
            system=system,
            variable=measure,
            tax_unit_positions=tax_unit_positions,
        )

    for spec in target_specs:
        if spec.family == "state_income_tax":
            state_fips = int(spec.metadata["state_fips"])
            columns[spec.measure] = np.where(
                household["state_fips"].to_numpy() == state_fips,
                columns["state_income_tax"],
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
                frame=frame,
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
                frame=frame,
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
        columns[spec.measure] = _collapse_tax_unit(
            values, tax_unit_positions, n_households
        )
    return columns


_engine_known_periods = recert.engine_known_periods


def _refuse_reform_moving_behavioral_responses(
    reform_system, baseline_system, *, measure: str
) -> None:
    """Refuse a JCT reform system that moves a behavioral-response parameter.

    A JCT batch engine gets the reform's system alone, so it has no
    ``baseline`` branch, and policyengine-us scores its labor-supply and
    capital-gains responses as 0. An engine given ``reform=`` as well has one
    and scores them from these parameters, as 0 while they are at the engine's
    zero default elasticities. Called once per reform, before its first batch
    engine.
    """

    moved = _moved_behavioral_response_parameters(
        reform_system, baseline_system, PERIOD
    )
    if not moved:
        return
    raise ValueError(
        f"Target materialization refuses JCT reform {measure}: its tax-benefit "
        f"system sets behavioral-response parameters off baseline at {PERIOD}: "
        f"{', '.join(moved)}. JCT reform batch engines carry no baseline "
        "branch, so policyengine-us would score these responses as zero."
    )


def _refuse_batch_baseline_branch_readers(
    simulation,
    *,
    measure: str,
    batch: int,
    batches: int,
) -> None:
    """Refuse a JCT batch engine that computed a formula reading ``baseline``.

    The engine has no ``baseline`` branch (see ``_reform_household_income_tax``),
    so these formulas compute from the reform where an engine given
    ``reform=`` reads the baseline: the Medicaid state denominator from the
    reform's own enrollment, and the behavioral-response measurements against
    the reform itself. ``record_engine_branches`` must be active, so that
    branches a formula created and then deleted are read too. The base frame
    holds no formula-owned column (``_assert_no_formula_owned_columns``), so
    any known period was computed, not loaded.
    """

    computed = sorted(
        _engine_known_periods(simulation, POST_EXPORT_BASELINE_BRANCH_READERS)
    )
    if not computed:
        return
    raise ValueError(
        f"Target materialization of JCT reform {measure}: the engine for "
        f"household batch {batch}/{batches} computed "
        f"{', '.join(f'{variable}@{period}' for variable, period in computed)}. "
        "These policyengine-us formulas read the engine's baseline branch, which "
        "a JCT reform batch engine does not carry: it is given the reform's "
        "tax-benefit system alone, without reform=."
    )


def _materialize_base_simulation_columns(
    base_frame: Frame,
    target_specs: tuple,
    *,
    system,
    microsimulation_cls,
    maximum_microsim_batch_size: int | None,
    refuse_population_aggregates: bool | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Run the target materializer's base simulation over household batches.

    Each engine uses the JCT loop's household partition and is released
    before the next is built. Columns are invariant to batch size only when
    their engine values are computed within households. A multi-batch pass
    refuses known periods of ``US_POPULATION_AGGREGATE_VARIABLES`` in each
    engine and its recorded branches, including detached branches and aggregates
    reached by downstream targets. Pre-sliced callers arm the same guard with
    ``refuse_population_aggregates=True``. Such targets require one whole-pool
    batch. Group entities and stored Medicaid claiming-tax-unit references must
    stay within households. Each batch must preserve pool order, and all
    batches must return the same columns.

    Returns the columns in write order, plus a receipt for the compilation
    record.
    """

    n_households = base_frame.n("household")
    household_ids = base_frame.table("household")["household_id"].to_numpy()
    batches = tuple(
        household_position_batches(n_households, maximum_microsim_batch_size)
    )
    if not batches:
        raise ValueError("Target materialization requires at least one household.")
    batched = len(batches) > 1
    guard_armed = (
        batched
        if refuse_population_aggregates is None
        else refuse_population_aggregates
    )
    if batched or guard_armed:
        assert_group_entities_nest_in_households(base_frame)
    if guard_armed:
        assert_medicaid_claiming_tax_units_local(
            base_frame, work=TARGET_MATERIALIZATION_PASS
        )
    if batched:
        print(
            "Materializing base target columns in "
            f"{len(batches)} batches of up to "
            f"{maximum_microsim_batch_size:,} households.",
            flush=True,
        )

    column_order: tuple[str, ...] | None = None
    columns: dict[str, np.ndarray] = {}
    for batch, household_positions in enumerate(batches, start=1):
        with automatic_gc_suspended():
            full_batch = len(household_positions) == n_households
            batch_frame = (
                base_frame
                if full_batch
                else select_households_by_position(base_frame, household_positions)
            )
            if not np.array_equal(
                batch_frame.table("household")["household_id"].to_numpy(),
                household_ids[household_positions],
            ):
                raise RuntimeError(
                    "A base target-materialization batch does not carry exactly "
                    "its households in pool order; its columns cannot be placed "
                    "at their pool positions."
                )
            batch_simulation = microsimulation_cls(
                dataset=_dataset_from_frame(
                    batch_frame,
                    assert_no_formula_owned_columns=False,
                    period_only=True,
                )
            )
            try:
                with (
                    record_engine_branches(batch_simulation)
                    if guard_armed
                    else nullcontext()
                ):
                    batch_columns = _base_simulation_household_columns(
                        batch_frame,
                        target_specs,
                        simulation=batch_simulation,
                        system=system,
                    )
                    if guard_armed:
                        # Before the release below drops the engine's holders.
                        refuse_batch_population_aggregates(
                            batch_simulation,
                            batch=batch,
                            batches=len(batches),
                            work=TARGET_MATERIALIZATION_PASS,
                            remedy=TARGET_POPULATION_AGGREGATE_REMEDY,
                        )
            finally:
                # microcosm#456: an engine's system keeps a ``simulation``
                # backref, so a finished engine can outlive its ``del``.
                # Release each batch engine before the next one is built,
                # also when this batch raised.
                release_engine_simulation(batch_simulation)
            del batch_simulation
            if column_order is None:
                column_order = tuple(batch_columns)
            elif tuple(batch_columns) != column_order:
                raise RuntimeError(
                    "Base target-materialization batches produced different "
                    "column sets or orders."
                )
            for column, values in batch_columns.items():
                values = np.asarray(values)
                if values.shape != (len(household_positions),):
                    raise RuntimeError(
                        f"Base target column {column!r} has shape "
                        f"{values.shape} for a batch of "
                        f"{len(household_positions)} households."
                    )
                if full_batch:
                    columns[column] = values
                    continue
                # Write each batch straight into its pool-length column, so
                # no per-batch slices are kept for a final concatenate. A
                # dtype change promotes by ``np.result_type``, as a
                # concatenate would.
                pool_values = columns.get(column)
                if pool_values is None:
                    pool_values = np.empty(n_households, dtype=values.dtype)
                elif pool_values.dtype != values.dtype:
                    pool_values = pool_values.astype(
                        np.result_type(pool_values, values)
                    )
                pool_values[household_positions] = values
                columns[column] = pool_values
            del batch_columns, batch_frame
        collect_batch_garbage()
        _advance_work(pass_name="base", batch=batch, batches=len(batches))
    collect_pass_garbage()

    assert column_order is not None
    receipt: dict[str, object] = {
        "method": "household_position_batches",
        "maximum_microsim_batch_size": (
            None
            if maximum_microsim_batch_size is None
            else int(maximum_microsim_batch_size)
        ),
        "households": int(n_households),
        "batches": len(batches),
        "largest_batch_households": max(len(positions) for positions in batches),
        "base_household_columns": len(column_order),
        "group_nesting_verified": batched or guard_armed,
        "population_aggregate_guard_armed": guard_armed,
        "population_aggregate_variables_checked": (
            list(US_POPULATION_AGGREGATE_VARIABLES) if guard_armed else []
        ),
    }
    return columns, receipt


def _materialize_target_frame(
    base_frame: Frame,
    target_specs: tuple,
    *,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    target_materialization_cache_dir: Path | None = None,
    target_materialization_cache_context: Mapping[str, object] | None = None,
    gate_congressional_district_targets: bool = False,
    refuse_population_aggregates: bool | None = None,
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
    system = CountryTaxBenefitSystem()
    n_households = base_frame.n("household")
    requested_reform_measures = {spec.measure for spec in target_specs}
    global _ACTIVE_WORK
    passes = 1 + sum(
        spec.measure in requested_reform_measures
        for spec in US_JCT_TAX_EXPENDITURE_REFORMS
    )
    batch_count = len(
        tuple(household_position_batches(n_households, maximum_microsim_batch_size))
    )
    _ACTIVE_WORK = _WorkCounter(
        stage_id="target_compilation",
        unit="engine_batch",
        total=passes * batch_count,
    )
    # The base simulation uses the JCT reform loop's household partition.
    base_columns, base_simulation_batching = _materialize_base_simulation_columns(
        base_frame,
        target_specs,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
        refuse_population_aggregates=refuse_population_aggregates,
    )

    materialized = {
        entity: base_frame.table(entity).copy() for entity in base_frame.entities
    }
    hh = materialized["household"]
    for column in tuple(base_columns):
        hh[column] = base_columns.pop(column)
    del base_columns

    base_income_tax_household = hh["income_tax"].to_numpy(dtype=np.float64)
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
    jct_reform_families_simulated = 0
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
                _advance_work(
                    batch_count,
                    pass_name=reform_spec.measure,
                    cached=True,
                )
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
                refuse_population_aggregates=refuse_population_aggregates,
            )
            jct_reform_families_simulated += 1
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
        collect_batch_garbage()

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
        "target_materialization_population_aggregate_guard": {
            "armed": base_simulation_batching["population_aggregate_guard_armed"],
            "population_aggregate_variables_checked": base_simulation_batching[
                "population_aggregate_variables_checked"
            ],
        },
        # Every engine this materializer ran used one household partition:
        # the base simulation and each JCT reform family simulated here
        # (cache hits simulate nothing).
        "target_materialization_batching": {
            **base_simulation_batching,
            "jct_reform_families_simulated": jct_reform_families_simulated,
        },
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


def _full_pool_calibration(args: argparse.Namespace, candidate_households: int) -> bool:
    """Whether the solve weights every staged household (no L0 selection).

    True for ``--dense-default-dataset`` and for an exact-k ladder whose ``k`` is
    the whole pool. The export is then the staged frame with calibrated
    household weights, so its records, columns and values are the staged
    frame's; otherwise the solve's L0 selection picks which households ship.
    """

    return bool(
        args.dense_default_dataset
        or (args.exact_k is not None and args.exact_k == candidate_households)
    )


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


def _spm_composition_report(frame: Frame) -> CheckResult:
    """The engine's SPM measurement classification of ``frame``, without a solve.

    ``check_spm_composition`` reproduces spm-calculator 1.0.0's rule
    (``adult = (age >= 18) | ((age >= 15) & role)``) over frame columns. The
    role resolution it mirrors is exactly what this export can produce:
    ``is_spm_independent_minor_role`` is the engine's declared dataset source
    input (``policyengine_us.spm.DATASET_SOURCE_INPUTS``), which the adapter
    classifies as an input leaf and ``write_dataset`` persists when the frame
    carries it (the ``spm_independence_role`` stage writes it), and
    ``is_household_head`` / ``is_household_spouse`` are persisted too — so a
    frame check here and the engine's own reading of the written H5 agree.
    """

    from microcosm.build.us_runtime.release_gate_preflight import (
        check_spm_composition,
    )

    return check_spm_composition(frame)


def _spm_composition_gate_failures(
    frame: Frame, *, stage: str
) -> tuple[list[str], dict[str, object]]:
    """The SPM measurement composition as one more batched pre-export gate.

    ``reform_validation`` measures the release's 104 state SPM poverty levels on
    one whole-dataset ``Microsimulation``, and one SPM unit with no classified
    adult raises ``SPMInputError("SPM_COMPOSITION_REQUIRED")`` for the *whole
    population* — a traceback naming neither the unit nor a remedy. This
    classifies the same composition off the export frame, which exists as soon
    as the calibrated weights are attached, so the refusal joins the batched
    pre-export report: named, with every other failing gate, and before the H5
    and NPZ writes rather than after them.

    Returns the gate's failure lines (empty when the engine would accept this
    export) and the detail keys its telemetry stage records. It never raises on
    a frame it cannot classify: an unevaluable export frame is itself one of
    those failure lines, so the run dies in the tool's ``Release gates failed:``
    report naming the reason instead of with a bare ``ValueError`` traceback out
    of the check.
    """

    try:
        report = _spm_composition_report(frame)
    except (KeyError, ValueError) as error:
        return (
            [
                f"SPM measurement composition failed ({stage}): the export "
                "frame cannot be classified, so the rule the engine applies to "
                f"it cannot be checked: {error}"
            ],
            {"evaluated": False, "error": str(error)},
        )
    details = {"evaluated": True, **report.details}
    if report.status != "FAIL":
        return [], details
    return (
        [
            f"SPM measurement composition failed ({stage}): "
            + report.summary
            + ". "
            + " ".join(report.failures)
        ],
        details,
    )


#: The entities whose typed weights the release writer materializes as
#: ``{entity}_weight`` columns: ``PolicyEngineUSEngine._engine_tables`` passes
#: ``weighted_entities=("household",)`` to ``engine_tables``.
_EXPORT_MATERIALIZED_WEIGHT_ENTITIES = ("household",)


def _export_stored_tables(frame: Frame) -> dict[str, tuple[str, ...]]:
    """The tables and column names ``release_engine.write_dataset`` stores.

    The release writes the export frame through ``PolicyEngineUSEngine()``,
    whose export contract is empty, so the writer adds no default column. It
    stores each non-empty entity table's columns, plus the typed household
    weights materialized as ``household_weight``. Only names are read, never a
    row, so the export frame is not copied. A ``requires_us`` test writes a
    small frame through the real writer and checks this view against the
    written H5's stored tables.
    """

    tables: dict[str, tuple[str, ...]] = {}
    for entity in frame.entities:
        table = frame.table(entity)
        if len(table) == 0:
            continue
        columns = list(table.columns)
        if entity in _EXPORT_MATERIALIZED_WEIGHT_ENTITIES:
            weight_column = f"{entity}_weight"
            if weight_column not in columns:
                columns.append(weight_column)
        tables[entity] = tuple(columns)
    return tables


def _stored_input_gate_failures(
    frame: Frame, *, stage: str
) -> tuple[list[str], dict[str, object]]:
    """Stored model inputs as one more batched pre-export gate (microcosm#1026).

    policyengine-us ignores a stored column it defines no variable for: it
    logs one warning and drops the column. ``populace-us-2024-spm-20260915``
    and its reported-receipt child, both certified against policyengine-us
    2.2.1, stored the WIC take-up draw as ``would_claim_wic``, which
    policyengine-us 1.777.0 renamed to ``takes_up_wic_if_eligible`` (2.2.1
    defines only the new name), so every WIC-eligible person took WIC up. This refuses, by name, every column the export would store that
    looks like a model input, is not a variable of the installed engine and is
    not in the reviewed register
    (:data:`microcosm.data.stored_inputs.US_STORED_NON_VARIABLE_COLUMNS`).

    The installed engine is the one the release is certified against: the
    tool writes the H5 with it and records its version as
    ``build.built_with_model_package``. The check reads column names only. A
    frame whose tables cannot be listed, or an engine that cannot be imported,
    is itself a failure line, so the run dies in the batched report naming the
    reason rather than with a traceback.
    """

    try:
        stored = _export_stored_tables(frame)
    except (KeyError, ValueError) as error:
        return (
            [
                f"Stored model inputs failed ({stage}): the export frame's "
                "stored tables cannot be listed, so its columns cannot be "
                f"checked against the engine: {error}"
            ],
            {"evaluated": False, "error": f"{type(error).__name__}: {error}"},
        )
    try:
        engine = installed_us_engine()
    except ImportError as error:
        return (
            [
                f"Stored model inputs failed ({stage}): the installed "
                "policyengine-us cannot be imported, so the export cannot be "
                "checked against the engine it is certified against: "
                f"{error}"
            ],
            {"evaluated": False, "error": f"{type(error).__name__}: {error}"},
        )
    columns = {column for names in stored.values() for column in names}
    details: dict[str, object] = {
        "evaluated": True,
        "engine": engine.label,
        "stored_columns": len(columns),
        "registered_non_variables": sorted(
            column
            for column in columns
            if is_model_named(column)
            and column not in engine.variables
            and column in US_STORED_NON_VARIABLE_COLUMNS
        ),
        "refused": list(
            undefined_stored_inputs(columns, engine_variables=engine.variables)
        ),
        "register_sha256": register_sha256(),
    }
    return (
        [
            f"Stored model inputs failed ({stage}): {line}"
            for line in stored_input_failures(stored, engine=engine)
        ],
        details,
    )


def _written_stored_input_verdict_mismatch(
    dataset_path: Path, pre_export: Mapping[str, object]
) -> str | None:
    """Why the written H5's stored-input verdict is not the allowed one, or None.

    The pre-export gate (:func:`_stored_input_gate_failures`) grades
    :func:`_export_stored_tables`, a model of what the writer stores, so that a
    refusal joins the batched raise before any H5 exists. This grades the
    written file's HDF metadata against the same engine and register. Grading
    the written file needs neither the model nor the gate's result, so it runs
    whatever the gate reached:

    - The gate evaluated. The written file must refuse exactly the columns the
      gate refused, so the gate's verdict is the verdict on the bytes the
      release ships. A refusal reaches this point only when evidence mode owns
      it.
    - The gate could not evaluate. Its failure line is on record, and it
      reaches this point only when evidence mode owns that line. No refusal of
      a column was reviewed, so the written file must refuse none: owning
      "could not evaluate" does not own a stale input the bytes store.

    A written file that cannot be graded (its tables cannot be listed, or the
    engine cannot be imported) is reported too, so no H5 goes on ungraded.
    Column names only; no row is read.
    """

    evaluated = bool(pre_export.get("evaluated"))
    expected = list(pre_export.get("refused", ())) if evaluated else []
    try:
        engine = installed_us_engine()
        written = h5_stored_tables(dataset_path)
    except (ImportError, OSError, StoredTableLayoutError) as error:
        return (
            "Stored-input gate premise failed: the written H5 "
            f"({dataset_path.name}) cannot be graded against the installed "
            f"engine ({type(error).__name__}: {error}), so this H5 cannot be "
            "certified."
        )
    columns = {column for names in written.values() for column in names}
    refused = list(undefined_stored_inputs(columns, engine_variables=engine.variables))
    if refused == expected:
        return None
    if not evaluated:
        return (
            "Stored-input gate premise failed: the written H5 "
            f"({dataset_path.name}) stores model-named columns that "
            f"{engine.label} does not define {refused}, and the pre-export "
            "gate never graded them because it could not evaluate the export "
            f"frame ({pre_export.get('error', 'no reason recorded')}). Rename "
            "each to its live input or add a reviewed register entry before "
            "this H5 is certified."
        )
    return (
        "Stored-input gate premise failed: the written H5 "
        f"({dataset_path.name}) stores model-named columns that "
        f"{engine.label} does not define {refused}, but the pre-export gate "
        f"graded the export frame's modeled stored tables as refusing "
        f"{expected}. _export_stored_tables no longer matches what "
        "release_engine.write_dataset stores; fix the model before this H5 "
        "is certified."
    )


def _record_stored_inputs_gate_evidence(
    release_dir: Path,
    *,
    failures: Sequence[str],
    details: Mapping[str, object],
    telemetry: _TerminalBatchProgress,
) -> dict[str, object]:
    """Write the stored-input gate's verdict as release evidence (microcosm#1034).

    :func:`_stored_input_gate_failures` once recorded its verdict only in run
    telemetry, so a published release could not show from its own files that
    the check ran, against which engine, or with which register. This writes
    ``stored_inputs.json`` in the shape of the other gate-evidence files: the
    failure lines exactly as they joined the batched raise, ``passed`` exactly
    when there are none, and the gate's details (the engine label, the register
    digest, the registered non-variables and refused columns, or why the gate
    could not evaluate). :func:`_build_manifests` binds the file as a
    ``release_manifest.json`` artifact, records it as ``bound`` in both
    manifests' ``gate_evidence`` block, and projects it into both manifests'
    ``stored_inputs`` block (:func:`microcosm.data.contract.
    stored_input_evidence_manifest_block`) for the H5 they pin. The publisher
    requires all three of every US national calibration release.

    The stored-input gate has no waiver, so ``enforced`` is always true; an
    evidence release carries its refusal lines verbatim in ``known_failures``
    as well, which the publisher checks.
    """

    payload: dict[str, object] = {
        "schema_version": US_STORED_INPUTS_EVIDENCE_SCHEMA_VERSION,
        "enforced": True,
        "checked_at_stage": US_FISCAL_STORED_INPUTS_STAGE,
        US_STORED_INPUTS_EVIDENCE_KEY: {
            "passed": not failures,
            "failures": list(failures),
            "details": dict(details),
        },
    }
    path = release_dir / "stored_inputs.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    telemetry.attach_artifact(US_STORED_INPUTS_EVIDENCE_KEY, path)
    return payload


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


def _without_calibrated_frames(result, *, export_frame: Frame | None = None):
    """Drop calibrated target tables while keeping weights and diagnostics.

    The exact-k receipt reads ``result.frame.n("household")``; give it the
    clean export frame instead of retaining the calibrated target tables.
    """
    if isinstance(result, L0RefitResult):
        return dataclasses.replace(
            result,
            selection=_without_calibrated_frames(result.selection),
            refit=_without_calibrated_frames(result.refit, export_frame=export_frame),
        )
    if isinstance(result, CalibrationResult):
        return dataclasses.replace(result, frame=export_frame)
    return result


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


def _esi_premiums_signal_evidence(gate: GateResult | None) -> dict[str, object]:
    """One ESI premium signal verdict for the release evidence file.

    Failures are kept verbatim so a build that waived them
    (``--allow-esi-premium-gaps``) still records what was red.
    """

    if gate is None:
        return {"evaluated": False}
    return {
        "evaluated": True,
        "passed": gate.passed,
        "failures": list(gate.failures),
    }


def _esi_anchor_feed_evidence(
    mismatches: Sequence[str], *, enforced: bool
) -> dict[str, object]:
    """The anchor-versus-feed verdict for the release evidence file.

    ``mismatches`` is non-empty only when a diagnostic flag waived a feed
    whose B4923C facts differ from the stage's anchor; they are kept verbatim
    so the waived difference stays on the record.
    """

    return {
        "enforced": enforced,
        "matches": not mismatches,
        "mismatches": list(mismatches),
    }


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
        shadow_not_run(
            "us_national_input_mass_reference",
            "no --input-mass-reference-h5 was given",
        )
        return None
    input_variables = _engine_input_variables()
    reference_frame = load_us_frame(reference_h5)
    candidate_totals = us_input_mass_totals(base_frame, columns=input_variables)
    reference_totals = us_input_mass_totals(reference_frame, columns=input_variables)
    carriers = us_line_32_carriers(base_frame, reference_frame)
    return shadow_gate(
        "us_national_input_mass_reference",
        lambda: us_input_mass_parity_gate(
            candidate_totals,
            reference_totals,
            carriers=carriers,
            candidate_name="base_frame",
            reference_name=reference_h5.name,
            relative_tolerance=relative_tolerance,
            minimum_reference_total=minimum_reference_total,
        ),
        lambda: EvidenceContext(
            artifacts={
                "candidate_input_mass_totals": candidate_totals,
                "reference_input_mass_totals": reference_totals,
                "input_mass_carriers": carriers,
                "input_mass_candidate_name": "base_frame",
                "input_mass_reference_name": reference_h5.name,
            }
        ),
    )


def _project_ecps_reference_layers(
    nonzero_shares: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, str]]:
    """Grade the pinned eCPS layers under the live engine's input names.

    The frozen reference keeps the incumbent's historical variable names
    (its evidence bytes are sha-pinned), so a layer the engine has since
    renamed would read as an all-zero candidate layer. Project each renamed
    layer onto its live input leaf with the same register the release
    input-coverage manifest uses (``REFERENCE_ECPS_LAYER_RENAMES``; the WIC
    input became ``takes_up_wic_if_eligible`` in PolicyEngine-US 1.777.0), and
    refuse a projection that would merge two reference layers.
    """

    projected: dict[str, float] = {}
    applied: dict[str, str] = {}
    for name, share in nonzero_shares.items():
        live = REFERENCE_ECPS_LAYER_RENAMES.get(str(name), str(name))
        if live in projected:
            raise ValueError(
                f"eCPS parity reference layer {name!r} projects onto {live!r}, "
                "which the reference already carries; the rename register "
                "would merge two layers."
            )
        projected[live] = float(share)
        if live != name:
            applied[str(name)] = live
    return projected, applied


def _project_ecps_known_gaps(
    known_gaps: tuple[ParityKnownGap, ...],
) -> tuple[dict[str, ParityKnownGap], dict[str, str]]:
    """Resolve exemption-register names onto the live layers they exempt.

    The register may name a gap by the reference's historical spelling. Once
    the reference layers are graded under live names, an unprojected
    historical exemption would exempt nothing (an empty live layer fails) and
    read as dormant (a populated live layer passes instead of flagging the
    exemption stale). Project each entry with the same register as
    :func:`_project_ecps_reference_layers`, through the helper the coverage
    manifest and the cross-register check share, keyed by live name with the
    entry's own spelling kept for provenance. Two entries that would exempt
    one live layer are refused: the register must say once which reason and
    issue own the gap.
    """

    by_register_name = {gap.variable: gap for gap in known_gaps}
    live_to_register = project_ecps_parity_known_gap_names(
        gap.variable for gap in known_gaps
    )
    projected = {
        live: by_register_name[register_name]
        for live, register_name in live_to_register.items()
    }
    applied = {
        register_name: live
        for live, register_name in live_to_register.items()
        if live != register_name
    }
    return projected, applied


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
    reference_shares, applied_renames = _project_ecps_reference_layers(
        reference.nonzero_shares
    )
    # Exemptions resolve through the same register, so a historical-name entry
    # exempts (and goes stale or dormant on) the live layer it names.
    projected_gaps, applied_gap_renames = _project_ecps_known_gaps(known_gaps)
    gate = parity_gate(
        candidate_shares,
        reference_shares,
        known_gaps=tuple(projected_gaps),
    )
    details = dict(gate.details)
    # The pinned reference predates engine input renames; its layers are graded
    # under the live names, and the projection is recorded, not hidden.
    details["reference_layer_renames"] = applied_renames
    details["known_gap_renames"] = applied_gap_renames
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
    # Keyed by the live layer the gate graded; a renamed entry also names the
    # register spelling it came from.
    details["known_gaps"] = {
        live: {
            "reason": gap.reason,
            "issue": gap.issue,
            **({"register_name": gap.variable} if gap.variable != live else {}),
        }
        for live, gap in sorted(projected_gaps.items())
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
        "H). Pinning it can move revenue, because estate_income feeds "
        "federal income tax. In policyengine-us 2.2.1 it "
        "is not a gross-income source, but loss_ald deducts its negative "
        "part above the line under the section 461(l) limit, whose ceiling "
        "its positive part raises, and it counts toward qualified business "
        "income while estate_income_would_be_qualified is true. From "
        "policyengine-us 2.23.2 (PolicyEngine/policyengine-us#9633) a "
        "non-dependent's positive estate_income is also gross income and "
        "AGI, and a non-dependent's signed estate_income plus the Schedule "
        "K-1 (Form 1041) box 14 code H adjustment input is net investment "
        "income for the NIIT (estate_net_investment_income). The exclusion "
        "rests on the band math above, not on revenue neutrality "
        "(PolicyEngine/microcosm#299 Build H). That math holds only for the "
        "$98.434B reference. The build applies this entry for any reference, "
        "so re-adjudicate it if --export-input-mass-reference-h5 names a file "
        "with a different level, or if the flag is absent and the raw base "
        "becomes the reference."
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


def _record_export_input_mass_gate(
    export_frame: Frame,
    base_frame: Frame,
    *,
    reference_h5: Path | None,
    reference_name: str,
    reference_record: dict[str, object],
    relative_tolerance: float,
    minimum_reference_total: float,
    terminal_gate_failures: list[str],
) -> GateResult | None:
    """Evaluate :func:`_export_input_mass_gate` against a reference H5 loaded here.

    The reference is a full frame. Loaded in this function, it is freed when
    the check returns or raises; a ``del`` in the caller after the check
    would be skipped when the load or the check raised. Without a reference
    H5 the check compares the export against ``base_frame``.

    The reference's sha256 goes into ``reference_record`` before the load.
    Only an evaluation crash escapes: it propagates on an otherwise-clean run,
    and under earlier failures it becomes one line in
    ``terminal_gate_failures`` and the function returns ``None``.
    """
    reference_loaded = False
    try:
        if reference_h5 is not None:
            reference_record["sha256"] = _sha256(reference_h5)
        reference_frame = (
            load_us_frame(reference_h5) if reference_h5 is not None else None
        )
        reference_loaded = True
        # Build H (microcosm#299): the two SOI-identified columns whose true
        # target level provably cannot sit inside the live-default reference
        # band (estate_income, non_sch_d_capital_gains). miscellaneous_income
        # and both mortgage columns are deliberately NOT excluded so the run
        # adjudicates their actual gate outcome. See the constant's band-math
        # rationale above.
        return shadow_gate(
            "us_national_export_input_mass",
            lambda: _export_input_mass_gate(
                export_frame,
                base_frame,
                relative_tolerance=relative_tolerance,
                minimum_reference_total=minimum_reference_total,
                reference_frame=reference_frame,
                reference_name=reference_name,
                reviewed_exclusions=US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
            ),
            lambda: EvidenceContext(
                frame=export_frame,
                artifacts={
                    "base_frame": base_frame,
                    "export_input_mass_reference_frame": reference_frame,
                    "export_input_mass_reference_name": reference_name,
                    "export_input_mass_reviewed_exclusions": (
                        US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS
                    ),
                },
            ),
        )
    except Exception as exc:
        # Same degraded-mode contract as the coverage gate in _main.
        if not terminal_gate_failures:
            raise
        if not reference_loaded:
            # As for the QRF tail register: the gate never ran.
            shadow_not_run(
                "us_national_export_input_mass",
                f"its reference did not load: {type(exc).__name__}: {exc}",
            )
        terminal_gate_failures.append(
            "Input mass parity failed: evaluation error under earlier gate "
            f"failures: {type(exc).__name__}: {exc}"
        )
        return None


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
    return us_input_mass_parity_gate(
        us_input_mass_totals(export_frame, columns=input_variables),
        us_input_mass_totals(reference, columns=input_variables),
        carriers=us_line_32_carriers(export_frame, reference),
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
    absent, or non-numeric, and each numeric output's record nonzero share and
    finite nonzero record count, the numbers behind the dense/sparse split) for
    the release artifact and the ``--dry-run-gates-report`` certainty model.

    The surface also measures every numeric output, dense and thin ones
    included, with :func:`microcosm.build.gates.single_record_concentration`:
    the largest single record's share of the column's weighted absolute mass
    and that record's weight, next to the top-k share. The measurements cannot
    change the gate's inputs, its verdict or the errors it raises: the loop
    resolves weights only for sparse outputs, and the gate runs before any
    measurement. A misaligned sparse output therefore still fails with the
    gate's own message, which names the column. The loop stores a dense
    output's entity and not its values, and the measurement reads the column
    from the frame again.
    :func:`_qrf_single_record_measurements` runs after the gate and turns
    every ``Exception`` into an ``unmeasured`` entry, so no ``Exception``
    from a measurement fails a run the gate would pass.
    """
    qrf_outputs = sorted(_qrf_imputed_source_outputs())
    values: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    absent: list[str] = []
    dense: list[str] = []
    non_numeric: list[str] = []
    nonzero_shares: dict[str, float] = {}
    nonzero_records: dict[str, int] = {}
    entity_weights: dict[str, np.ndarray] = {}
    dense_entities: dict[str, str] = {}
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
        nonzero_shares[column] = nonzero_share
        nonzero_records[column] = int((np.isfinite(array) & (array != 0.0)).sum())
        if nonzero_share > US_QRF_SPARSE_NONZERO_SHARE_MAX:
            dense.append(column)
            dense_entities[column] = entity
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
    single_record, unmeasured = _qrf_single_record_measurements(
        export_frame,
        qrf_outputs,
        sparse_values=values,
        sparse_weights=weights,
        dense_entities=dense_entities,
        entity_weights=entity_weights,
    )
    surface: dict[str, object] = {
        "qrf_imputed_outputs": len(qrf_outputs),
        "checked_sparse_columns": sorted(values),
        "dense_columns": dense,
        "absent_columns": absent,
        "non_numeric_columns": non_numeric,
        "sparse_nonzero_share_max": US_QRF_SPARSE_NONZERO_SHARE_MAX,
        "nonzero_shares": nonzero_shares,
        "nonzero_records": nonzero_records,
        "single_record_concentration": {
            "top_k": US_QRF_TAIL_CONCENTRATION_TOP_K,
            "columns": single_record,
            "unmeasured": unmeasured,
        },
    }
    return gate, surface


def _qrf_single_record_measurements(
    export_frame: Frame,
    qrf_outputs: Sequence[str],
    *,
    sparse_values: Mapping[str, np.ndarray],
    sparse_weights: Mapping[str, np.ndarray],
    dense_entities: Mapping[str, str],
    entity_weights: Mapping[str, np.ndarray],
) -> tuple[dict[str, dict[str, float | int | None]], dict[str, str]]:
    """Each numeric QRF output's single-record measurements, after the gate.

    A sparse output reuses the values and weights the gate checked. For a
    dense output the gate loop stored only the entity, so this reads the
    column from the frame again, with the loop's two conversion statements.
    A dense column's converted values stay in local names: this stores them
    in no mapping and returns none of them. A dense output resolves its
    entity's weights here, since the gate never weighs one, and reuses
    weights that a sparse output of the same entity already resolved. Any
    ``Exception``, from reading a dense column, from resolving weights or
    from measuring, lists the output under ``unmeasured`` with the reason
    :func:`_unmeasured_reason` builds, so no ``Exception`` from an output
    fails the run. Both mappings keep the order of ``qrf_outputs``.
    """
    resolved = dict(entity_weights)
    columns: dict[str, dict[str, float | int | None]] = {}
    unmeasured: dict[str, str] = {}
    for column in qrf_outputs:
        if column not in sparse_values and column not in dense_entities:
            continue
        try:
            if column in sparse_values:
                column_values = sparse_values[column]
                column_weights = sparse_weights[column]
            else:
                entity = dense_entities[column]
                series = export_frame.table(entity)[column]
                numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
                column_values = numeric.to_numpy(dtype=np.float64)
                if entity not in resolved:
                    resolved[entity] = np.asarray(
                        export_frame.resolve_weights(entity).values,
                        dtype=np.float64,
                    )
                column_weights = resolved[entity]
            columns[column] = single_record_concentration(
                column_values,
                column_weights,
                top_k=US_QRF_TAIL_CONCENTRATION_TOP_K,
            )
        except Exception as error:
            unmeasured[column] = _unmeasured_reason(error)
    return columns, unmeasured


def _unmeasured_reason(error: Exception) -> str:
    """Why an output went unmeasured: the error's type and text.

    Formatting an error runs the error's own ``__str__`` or ``__format__``,
    and reading its type's name can run a metaclass property; any of them
    can raise. When one raises an ``Exception``, the reason falls back to
    the type's name with ``unprintable error``, then to ``unprintable
    error`` alone, so no ``Exception`` raised while formatting leaves this
    function.
    """
    try:
        return f"{type(error).__name__}: {error}"
    except Exception:
        try:
            return f"{type(error).__name__}: unprintable error"
        except Exception:
            return "unprintable error"


#: Failure-line prefix for a per-run QRF tail-concentration register that does
#: not match the checked surface. Deliberately distinct from the standing
#: evidence owner "QRF tail concentration failed:" (US_EVIDENCE_FAILURE_OWNERS):
#: a register mismatch is an operator input error, not the #481/#487 tail
#: defect, and an evidence-tier certification refuses it outright.
US_QRF_TAIL_REGISTER_MISMATCH_PREFIX = "QRF tail-concentration register mismatch:"


def _qrf_tail_register_mismatch(
    register: Mapping[str, str],
    gate: GateResult,
) -> dict[str, list[str]]:
    """Register entries the checked tail surface did not use.

    The per-run register must exactly match the concentrated columns.
    ``stale`` entries were checked and sit at or below the threshold (the
    gate itself also fails them); ``unused`` entries were never checked —
    the column is dense, thin, absent, non-numeric, or not a QRF output.
    Both lists empty means the register matches.
    """
    used = set(gate.details.get("reviewed_exclusions", ()))
    stale = sorted(set(register) & set(gate.details.get("stale_exclusions", ())))
    unused = sorted(set(register) - used - set(stale))
    return {"stale": stale, "unused": unused}


def _qrf_tail_register_failures(mismatch: Mapping[str, Sequence[str]]) -> list[str]:
    """One batched failure line for a register mismatch, or none."""
    stale = list(mismatch.get("stale", ()))
    unused = list(mismatch.get("unused", ()))
    if not stale and not unused:
        return []
    return [
        f"{US_QRF_TAIL_REGISTER_MISMATCH_PREFIX} the per-run exclusion "
        "register must exactly match the concentrated columns; stale "
        f"(checked, at or below the threshold) = {stale}; unused (dense, "
        f"thin, absent, or not a checked QRF output) = {unused}. Remove these "
        "entries; qrf_tail_concentration.json records the measured surface."
    ]


def _qrf_tail_register_evidence_refusal(
    failures: Sequence[str],
) -> RuntimeError | None:
    """An evidence-tier certification's refusal of a register mismatch.

    A register mismatch is corrected in the register, never owned: the
    pre-batch raise this replaced refused every mode, so no per-run
    adjudication may carry a stale or unused entry into an evidence export.
    The build records the mismatch with its other failures
    (:data:`US_QRF_TAIL_REGISTER_MISMATCH_PREFIX`), and certification refuses
    it whatever an owner pattern says (microcosm#1585 item 11).
    """
    mismatches = [
        line
        for line in failures
        if line.startswith(US_QRF_TAIL_REGISTER_MISMATCH_PREFIX)
    ]
    if not mismatches:
        return None
    return RuntimeError(
        "Evidence release refused: a QRF tail-concentration register "
        "mismatch cannot be owned; correct the register from "
        "qrf_tail_concentration.json and rebuild. " + "; ".join(mismatches)
    )


def _qrf_tail_gate_lines(
    gate: GateResult,
    mismatch: Mapping[str, Sequence[str]],
    *,
    allow_concentration: bool,
) -> tuple[list[str], list[str]]:
    """The terminal lines one tail-gate evaluation contributes.

    Returns ``(gate_lines, register_lines)``. Gate lines carry the standing
    ``QRF tail concentration failed:`` owner prefix, and
    ``--allow-qrf-tail-concentration`` suppresses them. Register lines are the
    mismatch refusal, appended whatever that flag says.
    """

    gate_lines = (
        [f"QRF tail concentration failed: {failure}" for failure in gate.failures]
        if not gate.passed and not allow_concentration
        else []
    )
    return gate_lines, _qrf_tail_register_failures(mismatch)


def _record_qrf_tail_concentration_gate(
    export_frame: Frame,
    *,
    exclusions_path: Path | None,
    allow_concentration: bool,
    terminal_gate_failures: list[str],
    release_dir: Path,
    telemetry: _TerminalBatchProgress,
) -> list[str]:
    """Evaluate the terminal QRF tail gate and record everything it measured.

    Appends the gate's failures (unless ``allow_concentration``) and any
    register-mismatch line (always — the register must match whatever the
    flag says) to ``terminal_gate_failures``, so the batched pre-export raise
    refuses the run with the #568 weight sidecar on disk. Whenever the gate
    evaluated, ``qrf_tail_concentration.json`` is written with the per-column
    shares, carrier counts, single-record shares, and the mismatch. Only a
    genuine evaluation crash escapes: it propagates on an otherwise-clean run
    and becomes one unowned failure line under earlier failures.

    Returns the register-mismatch failure lines (empty when the register
    matches); an evidence-tier certification refuses them by their prefix.
    """
    register_loaded = False
    try:
        register = _load_qrf_tail_concentration_exclusions(exclusions_path)
        register_loaded = True
        gate, surface = shadow_value(
            "us_national_qrf_tail_concentration",
            lambda: _qrf_tail_concentration_gate(
                export_frame,
                reviewed_exclusions=register,
            ),
            lambda: EvidenceContext(
                frame=export_frame,
                artifacts={"qrf_tail_reviewed_exclusions": register},
            ),
            verdict=lambda value: LegacyVerdict.from_result(value[0]),
        )
        mismatch = _qrf_tail_register_mismatch(register, gate)
        surface = {
            **surface,
            "reviewed_exclusions_file": (
                str(exclusions_path) if exclusions_path is not None else None
            ),
            "reviewed_exclusions_sha256": (
                _sha256(exclusions_path) if exclusions_path is not None else None
            ),
            "reviewed_exclusions": dict(register),
            "register_mismatch": mismatch,
        }
    except Exception as exc:
        # Same degraded-mode contract as the coverage gate and as before this
        # refactor: with earlier failures on record, an evaluation crash
        # becomes one more line under the standing "QRF tail concentration
        # failed:" prefix (so evidence-tier ownership is unchanged) instead
        # of masking them. Only a register mismatch is refused outright; see
        # _qrf_tail_register_evidence_refusal.
        if not terminal_gate_failures:
            raise
        if not register_loaded:
            # The gate never ran, so the gate battery's shadow (microcosm#1585
            # item 11) records that, or the finished build would call the
            # entry unwired.
            shadow_not_run(
                "us_national_qrf_tail_concentration",
                "its reviewed-exclusions register did not load: "
                f"{type(exc).__name__}: {exc}",
            )
        terminal_gate_failures.append(
            "QRF tail concentration failed: evaluation error under earlier "
            f"gate failures: {type(exc).__name__}: {exc}"
        )
        return []
    gate_failures, register_failures = _qrf_tail_gate_lines(
        gate, mismatch, allow_concentration=allow_concentration
    )
    terminal_gate_failures.extend(gate_failures)
    terminal_gate_failures.extend(register_failures)
    qrf_tail_path = release_dir / "qrf_tail_concentration.json"
    qrf_tail_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "enforced": not allow_concentration,
                "surface": surface,
                "tail_concentration": {
                    "passed": gate.passed,
                    "failures": list(gate.failures),
                    "details": dict(gate.details),
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    telemetry.attach_artifact("qrf_tail_concentration", qrf_tail_path)
    if gate_failures or register_failures:
        telemetry.stage(
            "export_dataset",
            status="failed",
            message=(
                "QRF tail-concentration gate failed."
                if not register_failures
                else "QRF tail-concentration gate or register failed."
            ),
            failures=[
                *(gate.failures if gate_failures else ()),
                *register_failures,
            ],
        )
    return register_failures


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
    telemetry: _BuildProgress | None,
    enforcement_fences: Mapping[str, str] | None = None,
) -> tuple[list[str], GateResult]:
    """Fail the release on an enforced-band delivery miss, via the batch.

    microcosm#507/#508: a miss beyond tolerance on release weights fails the
    build instead of shipping in the scorecard. ``enforcement_fences``
    (microcosm#566/#567) fences normally-enforced bands for the dense
    arm (--dense-default-dataset), where delivered-weight recomputes have not landed
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

    delivery_gate = shadow_gate(
        "us_national_ssi_take_up_delivery",
        lambda: us_ssi_take_up_delivery_gate(
            diagnostics, targets=targets, enforcement_fences=enforcement_fences
        ),
        lambda: EvidenceContext(
            artifacts={
                "ssi_take_up_diagnostics": diagnostics,
                "ssi_band_targets": targets,
                "ssi_enforcement_fences": enforcement_fences,
            }
        ),
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


TARGET_SURFACE_FULL = "full"
TARGET_SURFACE_NATIONAL_STATE = "national_state"
TARGET_SURFACE_MODES = (TARGET_SURFACE_FULL, TARGET_SURFACE_NATIONAL_STATE)


def _select_target_surface(
    target_specs: Sequence[TargetSpec],
    mode: str,
) -> tuple[tuple[TargetSpec, ...], dict[str, object]]:
    """Return the specs the release calibrates to and a receipt of the choice.

    ``full`` keeps every compiled spec. ``national_state`` drops every spec
    ``is_congressional_district_target`` classifies as congressional-district
    (CD geography rows and rows sourced from the SOI CD file, whatever their
    geography). The target-parity and profile-coverage gates run on the full
    compiled surface before this selection, so a dropped family is still
    proven compiled.
    """

    if mode not in TARGET_SURFACE_MODES:
        raise ValueError(
            f"Unknown target surface {mode!r}; expected one of {TARGET_SURFACE_MODES}."
        )
    specs = tuple(target_specs)
    if mode == TARGET_SURFACE_FULL:
        kept = specs
    else:
        kept = tuple(
            spec for spec in specs if not _target_is_congressional_district(spec)
        )
    if not kept:
        raise ValueError(f"Target surface {mode!r} keeps no targets.")
    return kept, {
        "mode": mode,
        "compiled_targets": len(specs),
        "calibrated_targets": len(kept),
        "dropped_congressional_district_targets": len(specs) - len(kept),
    }


CONGRESSIONAL_DISTRICT_SOURCE_ALIASES = (
    "census-acs-s0101-congressional-district-age-2024",
    "soi-congressional-district-2022",
)


def _source_coverage_aliases(
    target_surface_selection: Mapping[str, object] | None,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return ``us_source_coverage.json``'s active aliases and surface exclusions.

    The congressional-district sources count as active coverage only when the
    release calibrates to them. A surface that dropped every CD-classified
    target (``target_surface_selection`` is recorded only for such surfaces)
    lists them as reviewed exclusions naming the drop, so the coverage artifact
    agrees with ``target_surface_selection`` in the manifests: at the release
    manifest's top level and in the build manifest's ``calibration`` block
    (microcosm#1570).
    """

    if target_surface_selection is None:
        return DIRECT_ACTIVE_ALIASES + CONGRESSIONAL_DISTRICT_SOURCE_ALIASES, {}
    dropped = int(target_surface_selection["dropped_congressional_district_targets"])
    reason = (
        "Not calibrated by this release: --target-surface "
        f"{target_surface_selection['mode']} dropped all {dropped:,} "
        "congressional-district-classified targets after the target-parity and "
        "profile-coverage gates ran on the full compiled surface; see "
        "release_manifest.json target_surface_selection and "
        "build_manifest.json calibration.target_surface_selection."
    )
    return DIRECT_ACTIVE_ALIASES, {
        alias: reason for alias in CONGRESSIONAL_DISTRICT_SOURCE_ALIASES
    }


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


def _pre_solve_gate_failures(
    *,
    target_profile_gate: GateResult | None = None,
    base_population_gate: GateResult | None = None,
    health_input_gate: GateResult | None = None,
    immigration_gate: GateResult | None = None,
    hours_worked_gate: GateResult | None = None,
    snap_take_up_gate: GateResult | None = None,
    eligibility_inputs_gate: GateResult | None = None,
    pregnancy_gate: GateResult | None = None,
    reported_coverage_vintage_gate: GateResult | None = None,
    snap_discretionary_exemption_gate: GateResult | None = None,
    input_mass_reference_gate: GateResult | None = None,
    degenerate_input_gate: GateResult | None = None,
    ecps_parity_gate: GateResult | None = None,
) -> list[str]:
    """The release failure lines of the gates graded before the solve.

    The frame-signal, input-mass-reference, degenerate-input and eCPS
    parity groups of :func:`_release_gate_failures`, in its order and
    wording. They depend on the staged frame alone, so
    ``--dry-run-gates-report`` reports them from the same function. Pass
    ``None`` for a gate the caller does not enforce.
    """
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
    return failures


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
    failures: list[str] = _pre_solve_gate_failures(
        target_profile_gate=target_profile_gate,
        base_population_gate=base_population_gate,
        health_input_gate=health_input_gate,
        immigration_gate=immigration_gate,
        hours_worked_gate=hours_worked_gate,
        snap_take_up_gate=snap_take_up_gate,
        eligibility_inputs_gate=eligibility_inputs_gate,
        pregnancy_gate=pregnancy_gate,
        reported_coverage_vintage_gate=reported_coverage_vintage_gate,
        snap_discretionary_exemption_gate=snap_discretionary_exemption_gate,
        input_mass_reference_gate=input_mass_reference_gate,
        degenerate_input_gate=degenerate_input_gate,
        ecps_parity_gate=ecps_parity_gate,
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
    target_surface: Mapping[str, object],
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
    scf_wealth_signal: Mapping[str, object] | None = None,
    target_loss_family_multipliers: Mapping[str, float] | None = None,
    target_loss_basis: Mapping[str, object] | None = None,
    exact_k_ladder: Mapping[str, object] | None = None,
    calibration_runtime: Mapping[str, object] | None = None,
    post_export_scoring: Mapping[str, object] | None = None,
) -> DiagnosticsWriteOutcome:
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
    return write_calibration_diagnostics(
        result,
        release_dir / "calibration_diagnostics.json",
        target_registry=registry,
        target_surface=target_surface,
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
            "scf_wealth_signal": (
                dict(scf_wealth_signal) if scf_wealth_signal is not None else None
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
            # Route A PR-3: this file is written before the batched pre-export
            # raise, so unlike the manifests it records the solve's thread
            # geometry on a gate-failed run too.
            **(
                {"calibration_runtime": dict(calibration_runtime)}
                if calibration_runtime is not None
                else {}
            ),
            "release_gates": {
                "passed": not failures,
                "failures": failures,
            },
            "incumbent_diagnostics": incumbent_payload,
            "post_export_target_audit": bool(audit_export_targets),
            # How the written H5 will be scored after export (route A
            # remediation, microcosm#956): household batches, their count,
            # and each post-export stage's baseline plan in period order.
            **(
                {"post_export_scoring": dict(post_export_scoring)}
                if post_export_scoring is not None
                else {}
            ),
        },
    )


def _diagnostics_manifest_status(outcome: DiagnosticsWriteOutcome) -> dict[str, object]:
    """Render the typed writer result into durable release metadata."""

    if outcome.status == "available":
        return {
            "status": "available",
            "schema_version": outcome.schema_version,
            "sha256": outcome.sha256,
        }
    return {
        "status": "failed",
        "expected_schema_version": outcome.expected_schema_version,
        "error_code": outcome.error_code,
        "message": outcome.message,
    }


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


#: Terminal gate verdicts written into the release directory, keyed by their
#: release-manifest artifact key. A build folder holds only the manifests,
#: their checksum ledger and the manifest's artifacts
#: (``prepare_build_folder_release``), so a verdict that is not a manifest
#: artifact never ships — including the QRF tail register a
#: certified waiver rests on (route A remediation PR-3). _main() clears these
#: before the terminal gates run, so any present at manifest time are this
#: run's own.
US_RELEASE_GATE_EVIDENCE_FILES: dict[str, str] = {
    "esi_premiums": "esi_premiums.json",
    "input_coverage": "input_coverage.json",
    "input_mass_parity": "input_mass_parity.json",
    "qrf_tail_concentration": "qrf_tail_concentration.json",
    "reform_coverage_smoke": "reform_coverage_smoke.json",
    # microcosm#1034: the key and file are the contract's
    # US_STORED_INPUTS_EVIDENCE_KEY / US_STORED_INPUTS_EVIDENCE_FILE, which
    # the publisher requires bound; a test pins the two together.
    "stored_inputs": "stored_inputs.json",
}

#: Key under which ``us_source_coverage.json`` carries the fiscal-target
#: exclusion receipt: the concrete source record ids each exclusion rule
#: dropped or allowed on this feed (route A remediation PR-1). The manifests
#: reference it by file sha and key; ``present`` stays false until the
#: compiler writes the receipt.
US_FISCAL_TARGET_EXCLUSION_RECEIPT_KEY = "fiscal_target_exclusion_receipt"
#: us_source_coverage.json key of the FSA portfolio receipt, written only by a
#: build that compiles those targets (``--scf-education-debt``, microcosm#1074).
US_FSA_STUDENT_LOAN_TARGET_RECEIPT_KEY = "fsa_student_loan_target_receipt"


def _gate_evidence_artifacts(release_dir: Path, *, revision: str) -> dict[str, dict]:
    """Release-manifest artifact entries for this run's gate evidence files."""
    return {
        key: _artifact_entry(
            filename,
            _sha256(release_dir / filename),
            kind="diagnostics",
            revision=revision,
        )
        for key, filename in US_RELEASE_GATE_EVIDENCE_FILES.items()
        if (release_dir / filename).is_file()
    }


def _scf_education_debt_fit_artifacts(
    release_dir: Path, *, revision: str
) -> dict[str, dict]:
    """Release-manifest entries for a ``--scf-education-debt`` build's fit tables.

    The build writes both into the release directory before its manifests, so
    a missing table fails the build instead of shipping without it.
    """
    missing = sorted(
        filename
        for filename in US_SCF_EDUCATION_DEBT_FIT_ARTIFACTS.values()
        if not (release_dir / filename).is_file()
    )
    if missing:
        raise RuntimeError(
            "A --scf-education-debt build must write its fit tables before its "
            f"manifests; {release_dir} is missing {missing}."
        )
    return {
        key: _artifact_entry(
            filename,
            _sha256(release_dir / filename),
            kind="diagnostics",
            revision=revision,
        )
        for key, filename in US_SCF_EDUCATION_DEBT_FIT_ARTIFACTS.items()
    }


def _gate_evidence_status(
    gate_evidence_artifacts: Mapping[str, object],
    *,
    skipped_gates: Iterable[str],
) -> dict[str, str]:
    """Why each gate's verdict is, or is not, a release artifact.

    A missing verdict alone cannot tell a gate the operator skipped by flag
    (``skipped``: only --skip-reform-coverage-smoke can) from one whose
    evaluation crashed under earlier failures, on a build certified at the
    evidence tier, and so wrote nothing (``not_evaluated``), or from a
    release built before gate
    evidence was bound at all. This block names which, in both manifests.
    """
    skipped = set(skipped_gates)
    unknown = sorted(skipped - set(US_RELEASE_GATE_EVIDENCE_FILES))
    if unknown:
        raise ValueError(f"Unknown skipped release gates: {unknown}.")
    contradictory = sorted(skipped & set(gate_evidence_artifacts))
    if contradictory:
        raise ValueError(
            "Release gates recorded as skipped have a verdict in the release "
            f"directory: {contradictory}."
        )
    return {
        key: (
            "bound"
            if key in gate_evidence_artifacts
            else "skipped"
            if key in skipped
            else "not_evaluated"
        )
        for key in US_RELEASE_GATE_EVIDENCE_FILES
    }


def _qrf_tail_register_manifest_block(release_dir: Path) -> dict[str, object] | None:
    """The per-run QRF tail register, as the gate recorded it.

    Read back from ``qrf_tail_concentration.json`` rather than re-hashed, so
    the manifest binds the register the gate actually evaluated (its path,
    sha256 and entries at evaluation time) and the mismatch it measured. A
    null path/sha256 with no entries means no per-column register was passed;
    ``enforced`` says whether the gate held, since ``enforced: false`` means
    --allow-qrf-tail-concentration waived the whole gate. ``None`` when the
    gate never wrote its evidence.
    """
    path = release_dir / US_RELEASE_GATE_EVIDENCE_FILES["qrf_tail_concentration"]
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    surface = payload["surface"]
    mismatch = surface["register_mismatch"]
    return {
        "path": surface["reviewed_exclusions_file"],
        "sha256": surface["reviewed_exclusions_sha256"],
        "entries": dict(surface["reviewed_exclusions"]),
        "mismatch": {
            "stale": list(mismatch["stale"]),
            "unused": list(mismatch["unused"]),
        },
        "enforced": payload["enforced"],
    }


def _stored_inputs_manifest_block(
    release_dir: Path, *, dataset_sha256: str
) -> dict[str, object] | None:
    """The stored-input verdict, as the gate recorded it, for the pinned H5.

    Read back from ``stored_inputs.json`` rather than rebuilt, so the manifests
    carry the verdict the gate actually reached. ``_main`` writes the
    manifests only after the post-write check found the written H5 earns that
    verdict, so binding it to ``dataset_sha256`` (the bytes the manifests pin)
    states what was checked. ``None`` when the gate never wrote its evidence.
    """

    path = release_dir / US_RELEASE_GATE_EVIDENCE_FILES["stored_inputs"]
    if not path.is_file():
        return None
    return stored_input_evidence_manifest_block(
        json.loads(path.read_text()), dataset_sha256=dataset_sha256
    )


def _fiscal_target_exclusion_receipt_reference(
    coverage_path: Path, coverage_sha256: str
) -> dict[str, object]:
    """Where the fiscal-target exclusion receipt lives, and which bytes."""
    coverage = json.loads(coverage_path.read_text())
    receipt = (
        coverage.get(US_FISCAL_TARGET_EXCLUSION_RECEIPT_KEY)
        if isinstance(coverage, Mapping)
        else None
    )
    return {
        "artifact": "us_source_coverage",
        "path": coverage_path.name,
        "sha256": coverage_sha256,
        "key": US_FISCAL_TARGET_EXCLUSION_RECEIPT_KEY,
        "present": receipt is not None,
        "receipt_sha256": (
            hashlib.sha256(_strict_json_bytes(receipt)).hexdigest()
            if receipt is not None
            else None
        ),
    }


def _calibration_runtime() -> dict[str, object]:
    """Thread geometry the calibration solve runs under.

    The solve's floating-point summation order, and so its exact weights,
    depends on the torch intra-op thread count; recording it lets a later
    pass (a checkpoint hit, an offline replay) say whether it can reproduce
    this one bit for bit. ``OMP_NUM_THREADS`` is the value after this
    module's bounded default, so it is never missing.
    """
    import torch

    return {
        "torch": str(torch.__version__),
        "torch_num_threads": int(torch.get_num_threads()),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
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


def _score_post_export_consumer(
    name: str,
    consumer: Callable[[Callable[[Any], Any]], object],
    *,
    dataset_path: Path,
    post_export_scorer: _HouseholdBatchedPostExportScorer | None,
    baseline_plan: Sequence[PostExportKey] | None,
    maximum_microsim_batch_size: int | None,
    planned_reform_passes: int | None = None,
):
    """Run one post-export consumer through the household-batched scorer.

    ``_main`` passes the scorer it opened on the written H5 and the plan it
    recorded in the calibration diagnostics, with the plan's reform pass
    count for progress reporting. A direct caller may pass neither: the
    scorer is then opened (and closed) here, and the plan and the count are
    recorded by the same engine-free dry run.
    """
    scorer = post_export_scorer
    if scorer is None:
        scorer = _HouseholdBatchedPostExportScorer(
            dataset_path,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
        )
    try:
        if baseline_plan is None:
            plan, planned_reform_passes = _record_post_export_requests(consumer)
        else:
            plan = tuple(baseline_plan)
        scoring = scorer.open_consumer(
            name, plan, planned_reform_passes=planned_reform_passes
        )
        output = consumer(scoring.simulate)
        record = scorer.finish_consumer(scoring)
        baseline = (
            "baseline served from the merged pass"
            if scoring.baseline_from_merged_pass
            else f"{record['baseline_passes']} baseline pass"
        )
        print(
            f"Post-export {name}: {baseline} "
            f"({len(plan)} keys), {record['reform_passes']} reform passes, "
            f"{record['reform_systems']} reform systems, "
            f"{record['n_batches']} batches of up to "
            f"{record['max_batch_households']:,} households.",
            flush=True,
        )
        return output
    finally:
        if post_export_scorer is None:
            scorer.close()


def _write_reform_validation(
    *,
    release_dir: Path,
    dataset_path: Path,
    result,
    registry: TargetRegistry,
    release_id: str,
    simulate_out_of_sample: bool,
    post_export_scorer: _HouseholdBatchedPostExportScorer | None = None,
    baseline_plan: Sequence[PostExportKey] | None = None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
    planned_reform_passes: int | None = None,
) -> None:
    """Emit reform_validation.json: microcosm budget effects vs JCT scores.

    In-sample JCT tax-expenditure reforms come straight from the calibration
    fit; out-of-sample OBBBA provisions are simulated on the freshly written
    release H5 in household batches (skipped if ``simulate_out_of_sample`` is
    False, e.g. for a fast diagnostics-only build). The shared baseline is one
    batch-outer pass over the plan's keys in ascending period order.
    """
    payload_for = _reform_validation_consumer(result=result, release_id=release_id)
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
        payload = payload_for(None)
    else:
        payload = _score_post_export_consumer(
            "reform_validation",
            payload_for,
            dataset_path=dataset_path,
            post_export_scorer=post_export_scorer,
            baseline_plan=baseline_plan,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
            planned_reform_passes=planned_reform_passes,
        )
    write_reform_validation(payload, release_dir / "reform_validation.json")


def _write_demographics(
    *,
    release_dir: Path,
    dataset_path: Path,
    release_id: str,
    post_export_scorer: _HouseholdBatchedPostExportScorer | None = None,
    baseline_plan: Sequence[PostExportKey] | None = None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> None:
    """Emit demographics.json: the dataset's weighted population by age band.

    The fiscal-refresh release calibrates source-backed Census PEP age targets;
    this file remains a compact summary diagnostic for release consumers. Ages
    and person weights are scored on the written H5 in household batches.
    """
    ages, weights = _score_post_export_consumer(
        "demographics",
        _demographics_consumer,
        dataset_path=dataset_path,
        post_export_scorer=post_export_scorer,
        baseline_plan=baseline_plan,
        maximum_microsim_batch_size=maximum_microsim_batch_size,
    )
    payload = demographics_payload(ages, weights, period=PERIOD, release_id=release_id)
    # Household-record counts by state and congressional district: the
    # release's sub-national resolution floor, surfaced on the dashboard.
    payload["geography_coverage"] = geography_coverage_payload(dataset_path)
    write_demographics(payload, release_dir / "demographics.json")


def _write_ssi_social_security_holdout(
    *,
    release_dir: Path,
    dataset_path: Path,
    release_id: str,
    post_export_scorer: _HouseholdBatchedPostExportScorer | None = None,
    baseline_plan: Sequence[PostExportKey] | None = None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> None:
    """Report the held-out overlap; benchmark misses never refuse a release."""
    try:
        payload = _score_post_export_consumer(
            "ssi_social_security_holdout",
            _ssi_social_security_holdout_consumer,
            dataset_path=dataset_path,
            post_export_scorer=post_export_scorer,
            baseline_plan=baseline_plan,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
        )
        payload["release_id"] = release_id
    except PostExportBatchFileError:
        # The scored file's integrity, not a measurement: it stops certification.
        raise
    except Exception as error:
        # Reporting is observational, including an unavailable measurement.
        # It must not introduce a new release refusal or calibration target.
        payload = ssi_social_security_holdout_unavailable(
            error, period=PERIOD, release_id=release_id
        )
    write_ssi_social_security_holdout(
        payload, release_dir / "us_ssi_social_security_holdout.json"
    )


def _write_medicare_irmaa_holdout(
    *,
    release_dir: Path,
    dataset_path: Path,
    release_id: str,
    post_export_scorer: _HouseholdBatchedPostExportScorer | None = None,
    baseline_plan: Sequence[PostExportKey] | None = None,
    maximum_microsim_batch_size: int | None = DEFAULT_MAXIMUM_MICROSIM_BATCH_SIZE,
) -> None:
    """Report held-out IRMAA payers and premiums; misses never refuse a release."""
    try:
        payload = _score_post_export_consumer(
            "medicare_irmaa_holdout",
            _medicare_irmaa_holdout_consumer,
            dataset_path=dataset_path,
            post_export_scorer=post_export_scorer,
            baseline_plan=baseline_plan,
            maximum_microsim_batch_size=maximum_microsim_batch_size,
        )
        payload["release_id"] = release_id
    except PostExportBatchFileError:
        # The scored file's integrity, not a measurement: it stops certification.
        raise
    except Exception as error:
        # Reporting is observational, including an unavailable measurement.
        # It must not introduce a new release refusal or calibration target.
        payload = medicare_irmaa_holdout_unavailable(
            error, period=PERIOD, release_id=release_id
        )
    write_medicare_irmaa_holdout(
        payload, release_dir / "us_medicare_irmaa_holdout.json"
    )


def _write_population_structure_holdout(
    *, release_dir: Path, dataset_path: Path, release_id: str
) -> None:
    """Report held-out population structure without refusing the release.

    It reads marital status and living arrangement of people 65+, and Social
    Security beneficiary counts, from the written H5 without an engine. The
    target surface holds none of these margins, so a release can meet every
    target while missing them (microcosm#880).
    """
    path = release_dir / "us_population_structure_holdout.json"
    try:
        # Serializing inside the try keeps a measurement that strict JSON
        # cannot hold observational as well.
        write_population_structure_holdout(
            population_structure_holdout_from_h5(
                dataset_path, period=PERIOD, release_id=release_id
            ),
            path,
        )
    except Exception as error:
        # Observational, like the SSI holdout: an unavailable measurement must
        # not add a release refusal.
        write_population_structure_holdout(
            population_structure_holdout_unavailable(
                error, period=PERIOD, release_id=release_id
            ),
            path,
        )


def _build_manifests(
    *,
    release_id: str,
    release_dir: Path,
    artifact_root: Path,
    result,
    target_surface: Mapping[str, object],
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
    target_surface_selection: Mapping[str, object] | None = None,
    default_dataset: Mapping[str, object] | None = None,
    medicaid_enrollment_substitutions: Sequence[Mapping[str, object]] = (),
    staging: Mapping[str, object] | None = None,
    ledger_artifact: Mapping[str, object] | None = None,
    dataset_key: str = DATASET_KEY,
    dataset_filename: str = DATASET_FILENAME,
    calibration_key: str = CALIBRATION_KEY,
    calibration_filename: str = CALIBRATION_FILENAME,
    exact_k_ladder: Mapping[str, object] | None = None,
    base_pool: Mapping[str, object] | None = None,
    acs_predictor_join: Mapping[str, object] | None = None,
    evidence_known_failures: Sequence[Mapping[str, str]] | None = None,
    export_input_mass_reference: Mapping[str, object] | None = None,
    calibration_runtime: Mapping[str, object] | None = None,
    skipped_gates: Iterable[str] = (),
    scored_dataset_sha256: str | None = None,
    post_export_scoring: Mapping[str, object] | None = None,
    diagnostics_outcome: DiagnosticsWriteOutcome | None = None,
    scf_wealth_signal: Mapping[str, object] | None = None,
    scf_education_debt: bool = False,
    release_gate_failures: Sequence[str] | None = None,
    git_commit: str | None = None,
    built_at: str | None = None,
    runtime_versions: Mapping[str, str] | None = None,
    certification: Mapping[str, object] | None = None,
) -> None:
    dataset_path = artifact_root / dataset_filename
    calibration_path = artifact_root / calibration_filename
    diagnostics_path = release_dir / "calibration_diagnostics.json"
    coverage_path = release_dir / "us_source_coverage.json"
    dataset_sha = _sha256(dataset_path)
    if scored_dataset_sha256 is not None and scored_dataset_sha256 != dataset_sha:
        # The post-export smoke, reform validation and demographics scored the
        # H5 whose sha256 the scorer bound at load; the manifest must pin the
        # same bytes, or those verdicts describe a different file.
        raise RuntimeError(
            f"The post-export stages scored {dataset_path} at sha256 "
            f"{scored_dataset_sha256}, but the file now hashes to {dataset_sha}; "
            "the manifest would pin bytes the release gates never scored."
        )
    if (
        post_export_scoring is not None
        and post_export_scoring.get("dataset_sha256") != dataset_sha
    ):
        raise RuntimeError(
            "The post_export_scoring block records sha256 "
            f"{post_export_scoring.get('dataset_sha256')}, but {dataset_path} "
            f"hashes to {dataset_sha}; the manifest would describe scoring of "
            "bytes it does not pin."
        )
    calibration_sha = _sha256(calibration_path)
    if diagnostics_outcome is None:
        diagnostics_outcome = DiagnosticsWriteSuccess(
            status="available",
            path=diagnostics_path,
            schema_version=CALIBRATION_DIAGNOSTICS_SCHEMA_VERSION,
            sha256=_sha256(diagnostics_path),
        )
    diagnostics_status = _diagnostics_manifest_status(diagnostics_outcome)
    coverage_sha = _sha256(coverage_path)
    # Route A remediation PR-3: every certified-surface exception, and the
    # inputs the gates judged against, must be recorded in the manifests
    # rather than only in loose diagnostics that never ship. The same blocks
    # ride build_manifest.json and release_manifest.json's build section.
    gate_evidence_artifacts = _gate_evidence_artifacts(release_dir, revision=release_id)
    # microcosm#1074: checked before either manifest is written, so a build
    # missing a fit table writes neither manifest.
    scf_education_debt_fit_artifacts = (
        _scf_education_debt_fit_artifacts(release_dir, revision=release_id)
        if scf_education_debt
        else {}
    )
    qrf_tail_register = _qrf_tail_register_manifest_block(release_dir)
    stored_inputs = _stored_inputs_manifest_block(
        release_dir, dataset_sha256=dataset_sha
    )
    evidence_bindings: dict[str, object] = {
        "gate_evidence": _gate_evidence_status(
            gate_evidence_artifacts, skipped_gates=skipped_gates
        ),
        **(
            {"qrf_tail_register": qrf_tail_register}
            if qrf_tail_register is not None
            else {}
        ),
        **({"stored_inputs": stored_inputs} if stored_inputs is not None else {}),
        **(
            {"export_input_mass_reference": dict(export_input_mass_reference)}
            if export_input_mass_reference is not None
            else {}
        ),
        **(
            {"calibration_runtime": dict(calibration_runtime)}
            if calibration_runtime is not None
            else {}
        ),
        **(
            {"post_export_scoring": dict(post_export_scoring)}
            if post_export_scoring is not None
            else {}
        ),
        "fiscal_target_exclusion_receipt": _fiscal_target_exclusion_receipt_reference(
            coverage_path, coverage_sha
        ),
    }
    # A build record carries the summary its build computed: the
    # calibration result and the registry it reads stay in the build's
    # process (microcosm#1585 item 11, PR 4a).
    gate_failures = (
        list(release_gate_failures)
        if release_gate_failures is not None
        else _release_gate_failures(
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
    )

    # A certified build's provenance is its build's, from its record: the
    # commit, the time it was made and the packages that made it, not the
    # certifying process's (microcosm#1585 item 11, PR 4a).
    commit = git_commit if git_commit is not None else _git_output("rev-parse", "HEAD")
    if built_at is None:
        built_at = datetime.now(UTC).isoformat()
    runtime = (
        dict(runtime_versions) if runtime_versions is not None else _runtime_versions()
    )
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
        # The certification beside the build's code (microcosm#1585 item 11):
        # the commit that certified it, which may follow the build's, and the
        # tier it chose. Additive; the release contract reads ``code``.
        **({"certification": dict(certification)} if certification is not None else {}),
        "runtime": runtime,
        "timing": timing_payload,
        "ledger_artifact": dict(ledger_artifact) if ledger_artifact else None,
        **(
            {"exact_k_ladder": dict(exact_k_ladder)}
            if exact_k_ladder is not None
            else {}
        ),
        **({"base_pool": dict(base_pool)} if base_pool is not None else {}),
        **(
            {"acs_predictor_join": dict(acs_predictor_join)}
            if acs_predictor_join is not None
            else {}
        ),
        **evidence_bindings,
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
            # Present only when --target-surface narrows the calibrated
            # surface, so a default manifest is unchanged.
            **(
                {"target_surface_selection": dict(target_surface_selection)}
                if target_surface_selection is not None
                else {}
            ),
            "target_surface": {
                "sha256": target_surface["sha256"],
                "n_targets": target_surface["n_targets"],
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
                "initial_loss": _finite_or_none(result.initial_loss),
                "final_loss": _finite_or_none(result.final_loss),
                "fraction_within_10pct": _finite_or_none(result.fraction_within_10pct),
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
            # The SIPP/SCF blend audit (microcosm#1216): written with the
            # manifest a release cannot ship without, beside the diagnostics.
            **(
                {"scf_wealth_signal": dict(scf_wealth_signal)}
                if scf_wealth_signal is not None
                else {}
            ),
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

    diagnostic_artifacts = (
        {
            "calibration_diagnostics": _artifact_entry(
                "calibration_diagnostics.json",
                diagnostics_outcome.sha256,
                kind="diagnostics",
                revision=release_id,
            )
        }
        if diagnostics_outcome.status == "available"
        else {}
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
            **(
                {"certification": dict(certification)}
                if certification is not None
                else {}
            ),
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
            **(
                {"acs_predictor_join": dict(acs_predictor_join)}
                if acs_predictor_join is not None
                else {}
            ),
            **evidence_bindings,
            "warm_start_calibration": warm_start_payload,
            "selection_source": selection_source_payload,
            # Present only when --target-surface narrows the calibrated
            # surface, so a default manifest is unchanged.
            **(
                {"target_surface_selection": dict(target_surface_selection)}
                if target_surface_selection is not None
                else {}
            ),
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
        "calibration_diagnostics": diagnostics_status,
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
            **diagnostic_artifacts,
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
                    "us_ssi_social_security_holdout": _artifact_entry(
                        "us_ssi_social_security_holdout.json",
                        _sha256(release_dir / "us_ssi_social_security_holdout.json"),
                        kind="diagnostics",
                        revision=release_id,
                    )
                }
                if (release_dir / "us_ssi_social_security_holdout.json").exists()
                else {}
            ),
            **(
                {
                    "us_medicare_irmaa_holdout": _artifact_entry(
                        "us_medicare_irmaa_holdout.json",
                        _sha256(release_dir / "us_medicare_irmaa_holdout.json"),
                        kind="diagnostics",
                        revision=release_id,
                    )
                }
                if (release_dir / "us_medicare_irmaa_holdout.json").exists()
                else {}
            ),
            **(
                {
                    "us_population_structure_holdout": _artifact_entry(
                        "us_population_structure_holdout.json",
                        _sha256(release_dir / "us_population_structure_holdout.json"),
                        kind="diagnostics",
                        revision=release_id,
                    )
                }
                if (release_dir / "us_population_structure_holdout.json").exists()
                else {}
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
            **scf_education_debt_fit_artifacts,
            **gate_evidence_artifacts,
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


# Kept in lockstep with microcosm.data.contract._ISSUE_REF_RE; the evidence
# publish contract re-validates every owner ref, so drift here fails at
# publish rather than silently.
_EVIDENCE_ISSUE_REF_RE = re.compile(r"#\d+|github\.com/\S+/(?:issues|pull)/\d+")

#: The standing failure-pattern -> owner register of an evidence-tier
#: certification (microcosm#506), a packaged US runtime data file rather than
#: this tool's source, so that a sign-off changes that file alone and a build
#: can be certified at the commit that makes it (microcosm#1585 item 11, as a
#: UK cut is certified at its sign-off's head). It is not a country resource:
#: no part of the US spec reads it, so a sign-off moves no spec digest. Only owner-adjudicated families
#: belong there: the two dense blockers named in the #506 brief, the SOI
#: Table 1.4 dollar blanket (#487) and the dense QRF tail-concentration set
#: (#481, with the #487 uprating interaction). Any other recorded failure
#: needs a per-run adjudication via --evidence-failure-owners; an unowned
#: failure refuses the evidence certification.
US_EVIDENCE_FAILURE_OWNERS_RESOURCE = "evidence_failure_owners.json"
US_EVIDENCE_FAILURE_OWNERS_KIND = "us_evidence_failure_owners"
#: The register's path in the repository, which a later certifying commit
#: may change (:func:`_certification_may_change`).
US_EVIDENCE_FAILURE_OWNERS_REGISTER = (
    "packages/microcosm-build/src/microcosm/build/us_runtime/data/"
    + US_EVIDENCE_FAILURE_OWNERS_RESOURCE
)


def _standing_evidence_failure_owners_file():
    """The packaged standing owners register, as this checkout holds it."""

    return package_files("microcosm.build.us_runtime.data").joinpath(
        US_EVIDENCE_FAILURE_OWNERS_RESOURCE
    )


def _evidence_owner_pairs(
    entries: Iterable[tuple[object, object]], *, source: str
) -> tuple[tuple[str, str], ...]:
    """Validated ``(pattern, owner)`` pairs: a pattern, and an issue owning it."""

    pairs: list[tuple[str, str]] = []
    for pattern, owner in entries:
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"{source} has an empty pattern key.")
        if not isinstance(owner, str) or not _EVIDENCE_ISSUE_REF_RE.search(owner):
            raise ValueError(
                f"{source} owner for pattern {pattern!r} must carry an issue "
                f"reference (e.g. 'PolicyEngine/microcosm#487'), got {owner!r}."
            )
        pairs.append((pattern, owner))
    return tuple(pairs)


def _read_standing_evidence_failure_owners(text: str) -> tuple[tuple[str, str], ...]:
    """The standing register's ``(pattern, owner)`` pairs, in its order."""

    source = f"the US standing {US_EVIDENCE_FAILURE_OWNERS_RESOURCE} register"
    payload = json.loads(text)
    if (
        not isinstance(payload, dict)
        or payload.get("kind") != US_EVIDENCE_FAILURE_OWNERS_KIND
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("owners"), list)
    ):
        raise ValueError(
            f"{source} must be a schema 1 {US_EVIDENCE_FAILURE_OWNERS_KIND!r} "
            "object with an 'owners' list."
        )
    entries = []
    for entry in payload["owners"]:
        if not isinstance(entry, dict) or not str(entry.get("reason") or "").strip():
            raise ValueError(
                f"{source}: each owner entry is an object with a pattern, an "
                f"owner and the reason it is owned; got {entry!r}."
            )
        pattern = entry.get("pattern")
        # A register-only commit can certify (microcosm#1585 item 11), so a
        # standing pattern must name one gate's failure lines, as its prefix
        # "<gate words> failed:" does; a short pattern would own every
        # ownable failure (review of #1681).
        if isinstance(pattern, str) and pattern.strip():
            gate_words = pattern.removesuffix(" failed:").split()
            names_a_gate = pattern.endswith(" failed:") and len(gate_words) >= 2
        else:
            names_a_gate = True  # _evidence_owner_pairs refuses it as empty
        if not names_a_gate:
            raise ValueError(
                f"{source}: standing pattern {pattern!r} must be a gate's "
                "failure prefix, at least two words then ' failed:', so it owns "
                "that gate's failures alone."
            )
        entries.append((pattern, entry.get("owner")))
    return _evidence_owner_pairs(entries, source=source)


_STANDING_EVIDENCE_FAILURE_OWNERS_BYTES = (
    _standing_evidence_failure_owners_file().read_bytes()
)
#: The standing register, as this checkout packages it, read once, and the
#: digest of the bytes it was read from, which an evidence certification
#: records (review of #1681).
US_EVIDENCE_FAILURE_OWNERS: tuple[tuple[str, str], ...] = (
    _read_standing_evidence_failure_owners(
        _STANDING_EVIDENCE_FAILURE_OWNERS_BYTES.decode("utf-8")
    )
)
US_EVIDENCE_FAILURE_OWNERS_SHA256 = hashlib.sha256(
    _STANDING_EVIDENCE_FAILURE_OWNERS_BYTES
).hexdigest()


def _read_evidence_failure_owners_file(
    path: Path,
) -> tuple[tuple[tuple[str, str], ...], str]:
    """A per-run owners file's pairs, and the digest of the bytes they came
    from, so a certification records the file that decided ownership."""
    data = Path(path).read_bytes()
    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError(
            f"--evidence-failure-owners {path} must be a JSON object of "
            "failure-substring pattern -> owner issue ref."
        )
    pairs = _evidence_owner_pairs(
        payload.items(), source=f"--evidence-failure-owners {path}"
    )
    return pairs, hashlib.sha256(data).hexdigest()


def _load_evidence_failure_owner_patterns(
    path: Path | None,
) -> tuple[tuple[str, str], ...]:
    """Failure-pattern -> owner pairs: per-run adjudications first, then the
    standing register, so a run-specific entry can sharpen a standing owner."""
    per_run: tuple[tuple[str, str], ...] = ()
    if path is not None:
        per_run, _ = _read_evidence_failure_owners_file(path)
    return (*per_run, *US_EVIDENCE_FAILURE_OWNERS)


def _evidence_owner(
    failure: str,
    owner_patterns: Sequence[tuple[str, str]],
) -> str | None:
    """The owner of one recorded failure line: the first pattern it contains."""

    return next(
        (owner for pattern, owner in owner_patterns if pattern in failure),
        None,
    )


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
        owner = _evidence_owner(failure, owner_patterns)
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
    """The auto-generated build ID for a run without --release-id.

    Exact-k builds carry the k segment; every other build keeps the
    historical shape. A build has no tier (microcosm#1585 item 11): an
    evidence-tier certification mints the ID it publishes the build under
    from this one (:func:`_evidence_build_id`).
    """
    if args.exact_k is not None:
        return (
            f"populace-us-2024-k{args.exact_k}-{digest}-{commit}-"
            f"{build_timestamp:%Y%m%dT%H%M%SZ}"
        )
    return f"populace-us-2024-{digest}-{commit}-{build_timestamp:%Y%m%dT%H%M%SZ}"


def _telemetry_run_id(args: argparse.Namespace, *, timestamp: datetime) -> str:
    """Choose an attempt id before release inputs have been loaded.

    An explicit staging id remains authoritative, and an explicit release id
    is already stable enough to identify the attempt. Builds that derive their
    release id from the input digest need a separate attempt id so telemetry can
    start before downloading or hashing that input.
    """

    if args.staging_run_id:
        return str(args.staging_run_id)
    if args.release_id:
        return str(args.release_id)
    instant = timestamp.astimezone(UTC)
    return (
        f"populace-us-build-{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


#: The releases certification places a national build in (microcosm#1570):
#: the national release, the repository's default, at the certified tier, and
#: the evidence release, never the default, at the evidence tier
#: (microcosm#506). A build belongs to neither until it is certified
#: (microcosm#1585 item 11).
US_CERTIFICATION_RELEASES = (US_NATIONAL_RELEASE_ID, US_EVIDENCE_RELEASE_ID)
#: The build's own directory under ``--out``: its calibration diagnostics,
#: its SSI take-up measurement, the gate evidence it wrote and, for a build
#: whose release gates fail, its #568 weight evidence. A build has no tier, so
#: it writes no release directory: certification copies these files into
#: ``<out>/releases/<release>/`` for the tier it chooses.
US_BUILD_FILES_DIRECTORY = "build"


def _certification_release_dir(release_root: Path, *, evidence_release: bool) -> Path:
    """Where certification places a build: its tier's release directory."""

    release = US_EVIDENCE_RELEASE_ID if evidence_release else US_NATIONAL_RELEASE_ID
    return Path(release_root) / "releases" / release


def _refuse_reusing_a_certified_build(release_root: Path) -> None:
    """Refuse an ``--out`` whose build is certified, at either tier.

    A build is certified once, into the release of the tier certification
    chose (:data:`US_CERTIFICATION_RELEASES`), and a certified build is never
    changed: a build into its ``--out``, or a second certification, is
    refused while either release directory carries a release manifest.
    """

    for release in US_CERTIFICATION_RELEASES:
        _refuse_certified_release_dir_reuse(Path(release_root) / "releases" / release)


def _evidence_build_id(build: str) -> str:
    """The ID an evidence-tier certification publishes ``build`` under.

    A build ID carries no tier (microcosm#1585 item 11), and the evidence
    contract requires the ``-evidence-`` segment in the ID of every evidence
    build, so the tier is visible in each build folder's path (microcosm#506).
    The segment follows the ``populace-us-2024-`` prefix, the shape evidence
    builds always had, or else ``populace-us-``.
    """

    for prefix in (f"populace-us-{PERIOD}-", "populace-us-"):
        if build.startswith(prefix):
            return f"{prefix}evidence-{build[len(prefix) :]}"
    raise ValueError(f"{build!r} is not a US build ID (populace-us-...).")


def _promotion_note(release: str) -> str:
    """What the promotion command does for a build of ``release``."""
    if release == US_NATIONAL_RELEASE_ID:
        return (
            "Promoting makes this build the current build of "
            f"{release}, the US default dataset."
        )
    return (
        f"Promoting makes this build the current build of {release}, which is "
        "never the US default; the release's current build takes the place of "
        "the latest-evidence.json pointer file."
    )


def _finish_build_folder_release(
    release_dir: Path,
    *,
    build: str,
    artifact_root: Path,
    evidence_release: bool,
) -> dict[str, str]:
    """Rewrite the release directory into the build-folder layout.

    :func:`_build_manifests` writes both manifests in the tagged shape. This
    names the release (the directory's name) and the build (``build``, the
    build ID) in both, moves the build's details to the release manifest's
    top level and drops each artifact's ``revision`` (microcosm#1570). The
    release contract is left to ``microcosm-publish-release-build``, whose
    preflight runs it (the evidence contract for an evidence-tier build), as
    the tagged path left it to its publisher: a diagnostic run's output is
    written, not refused at its end.

    Returns:
        The commands that publish the build into its own folder of the
        release and then make it the release's current build.
    """
    del evidence_release  # The release is the directory's name.
    rewrite_release_dir(release_dir, build=build)
    return publication_commands(
        release_dir, repo_id=REPO_ID, artifact_root=artifact_root
    )


def _assert_us_release_id(release_id: str) -> None:
    """Check the build ID ``--release-id`` gives, or the default one.

    The ID names the build, not its release, and keeps the US prefix it had
    before microcosm#1570. A build has no tier (microcosm#1585 item 11), so
    its ID never carries the ``-evidence-`` segment: an evidence-tier
    certification adds it (:func:`_evidence_build_id`).
    """
    if release_id in US_RELEASES:
        raise ValueError(
            f"{release_id!r} is a release, not a build: --release-id gives the "
            "build ID (populace-us-2024-...), and certification publishes the "
            f"build into its tier's release, {US_NATIONAL_RELEASE_ID!r} or "
            f"{US_EVIDENCE_RELEASE_ID!r}."
        )
    if not release_id.startswith("populace-us-"):
        raise ValueError(
            "US fiscal refresh release ids must start with 'populace-us-' so "
            "the US release contract requires source coverage diagnostics."
        )
    if EVIDENCE_RELEASE_ID_SEGMENT in release_id:
        raise ValueError(
            f"release ids containing {EVIDENCE_RELEASE_ID_SEGMENT!r} are "
            "reserved for the builds an evidence-tier certification publishes; "
            "a build has no tier, and certification mints its evidence ID "
            "(microcosm#1585 item 11)."
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
    worker_execution_authentication = (
        authenticated_pool_h5.worker_execution_authentication
    )
    if (
        pool_manifest.get("worker_execution_authentication")
        != worker_execution_authentication
    ):
        raise RuntimeError(
            "Validated pool manifest worker authentication receipt changed."
        )
    pool_release_id = _assert_pool_release_id_value(
        args.pool_release_id,
        authenticated_pool_h5.publication_run_id,
    )
    pool_receipt = {
        "release_id": pool_release_id,
        "release_id_source": "pool_manifest.publication_run_id",
        "manifest_sha256": authenticated_pool_h5.manifest_sha256,
        "publication_run_id": authenticated_pool_h5.publication_run_id,
        "pool_h5_sha256": authenticated_pool_h5.sha256,
        "pool_h5_size_bytes": authenticated_pool_h5.size_bytes,
        "agreement_diagnostics_sha256": agreement_diagnostics.get("sha256"),
    }
    if worker_execution_authentication is not None:
        if not isinstance(worker_execution_authentication, Mapping):
            raise RuntimeError(
                "Validated pool manifest has a malformed worker authentication receipt."
            )
        pool_receipt["worker_execution_authentication"] = dict(
            worker_execution_authentication
        )
    payload = exact_k_ladder_manifest_payload(
        outcome,
        k=int(args.exact_k),
        seed=int(args.seed),
        pool=pool_receipt,
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


_ACTIVE_EMITTER: LocalTelemetryEmitter | None = None


def _calibration_epoch_every(args: argparse.Namespace) -> int:
    """The shared calibration epoch stride over this build's solves.

    ``--dense-default-dataset`` solves once; the L0 and exact-k paths solve a
    selection and then its refit, each for ``--epochs`` (an exact-k ladder
    whose ``k`` is the whole pool solves once, so two is the most it solves).
    """

    solves = 1 if args.dense_default_dataset else 2
    return calibration_epoch_stride(int(args.epochs) * solves)


#: The pipeline a US staging run names, as the build's hosted telemetry does.
US_STAGING_PIPELINE = "us_fiscal_refresh"
#: The operation of a build's staging run, and of its certification's.
US_BUILD_STAGING_OPERATION = "us_fiscal_refresh_build"
US_CERTIFICATION_STAGING_OPERATION = "us_fiscal_refresh_certification"
#: The version 2 event status for each status a US build reports for a stage.
#: Any other status (``skipped``, or a check's own verdict such as ``PASS``)
#: becomes a completed event whose details keep it as ``stage_status``.
_STAGING_EVENT_STATUSES: Mapping[str, EventStatus] = {
    "running": "started",
    "failed": "failed",
}
#: The size limit of a US result file reviewed to hold rows of aggregates
#: (lists of objects, one per state, age band, program, reform or target),
#: as version 2 allows a file of no rows.
_US_AGGREGATE_ROWS_BYTES = 5 * 1024 * 1024


class _StagedResultFile:
    """How a US staging run carries one result file: its version 2 fields.

    ``aggregate_rows_bytes`` is set for a file reviewed to hold rows of
    aggregates, and is then its size limit; a file without it may hold no
    lists of objects and keeps version 2's 5 MiB limit. (A plain class, for
    the reason :class:`_RecordedCalibrationResult` gives.)
    """

    __slots__ = ("aggregate_rows_bytes", "artifact_kind", "classification")

    def __init__(
        self,
        artifact_kind: str,
        classification: str = "aggregate",
        aggregate_rows_bytes: int | None = None,
    ) -> None:
        self.artifact_kind = artifact_kind
        self.classification = classification
        self.aggregate_rows_bytes = aggregate_rows_bytes


_AGGREGATE_ROWS = _US_AGGREGATE_ROWS_BYTES
#: The result files a US staging run carries, by logical name (microcosm#1135).
#: Each was checked against version 2's content rules on real US runs: none
#: holds a sensitive or row-level field, an individual record or a row
#: collection, and those that hold rows hold aggregates. A file attached under
#: any other name stays out of the run, with a warning, as does one the
#: content checks refuse.
US_STAGING_RESULT_FILES: Mapping[str, _StagedResultFile] = {
    "build_manifest": _StagedResultFile(
        "build_metadata", "non_row_level", _AGGREGATE_ROWS
    ),
    "build_record": _StagedResultFile(
        "build_metadata", "non_row_level", _AGGREGATE_ROWS
    ),
    # Per-target rows, which the dashboard's calibration view reads: 33.4 MB
    # for the national build in October 2026, up to the emitter's own limit.
    "calibration_diagnostics": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=EMITTER_MAX_FILE_BYTES
    ),
    "demographics": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "esi_premiums": _StagedResultFile("validation_summary"),
    "input_coverage": _StagedResultFile("validation_summary"),
    "input_mass_parity": _StagedResultFile("validation_summary"),
    "qrf_tail_concentration": _StagedResultFile("validation_summary"),
    "reform_coverage_smoke": _StagedResultFile(
        "validation_summary", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "reform_validation": _StagedResultFile(
        "validation_summary", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "release_manifest": _StagedResultFile(
        "build_metadata", "non_row_level", _AGGREGATE_ROWS
    ),
    US_STORED_INPUTS_EVIDENCE_KEY: _StagedResultFile("validation_summary"),
    "us_medicaid_take_up": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "us_medicare_irmaa_holdout": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "us_population_structure_holdout": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "us_snap_state_take_up": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "us_source_coverage": _StagedResultFile(
        "validation_summary", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "us_ssi_social_security_holdout": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "us_ssi_take_up": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
    "us_take_up_participation": _StagedResultFile(
        "aggregate_diagnostics", aggregate_rows_bytes=_AGGREGATE_ROWS
    ),
}
#: Version 2's content checks with the US files reviewed to hold rows.
US_STAGING_CONTENT_POLICY = StagingContentPolicy(
    aggregate_tables={
        name: result_file.aggregate_rows_bytes
        for name, result_file in US_STAGING_RESULT_FILES.items()
        if result_file.aggregate_rows_bytes is not None
    }
)


def _close_run(error: BaseException, bundle: StagingRunBundleWriterV2) -> None:
    """Close a staging run the build did not finish.

    A build whose release checks failed once its record was written
    (:class:`_BuildFailedReleaseGatesError`, exit 3) reached a decision and
    the checks refused it, so its run is ``blocked``; any other ending is a
    failure, classified as the hosted telemetry classifies it.
    """
    if isinstance(error, _BuildFailedReleaseGatesError):
        bundle.block(
            phase="release_gates",
            blocking_gate_ids=[],
            blocking_failure_count=max(1, len(error.failures)),
        )
        return
    classified = classify_failure(error)
    bundle.fail(
        error,
        error_code=classified.error_code or "BUILD_FAILED",
        failure_class=classified.failure_class,
    )


def _staging_message(message: str | None) -> str | None:
    if message is None:
        return None
    return message[:MAX_MESSAGE_CHARACTERS]


class _BuildProgress:
    """Report progress to an emitter and, independently, a staging bundle.

    Both receive the same calibration epochs: each solve's first and last and
    every ``epoch_every``-th epoch, the rule the UK build follows too
    (microcosm#1292). The staging run is best-effort: once it is open, an
    error in it (a local write, the version 2 checks refusing a document or
    a result file) prints a warning and never stops the build or its hosted
    telemetry, which hears of every stage before the staging run does.
    Opening the run (``_build_progress``) checks its identity and writes its
    first documents, so a refused ``--staging-run-id`` or an unwritable
    staging directory stops the build at its start.
    """

    def __init__(
        self,
        *,
        run_id: str,
        emitter: LocalTelemetryEmitter,
        staging_bundle: StagingRunBundleWriterV2 | None,
        staging_opt_out_reason: str | None = None,
        epoch_every: int = CALIBRATION_EPOCH_EVERY,
    ) -> None:
        self.run_id = run_id
        self.emitter = emitter
        self.staging_bundle = staging_bundle
        self.staging_opt_out_reason = staging_opt_out_reason
        self.epoch_every = epoch_every

    @property
    def repo_id(self) -> str | None:
        if self.staging_bundle is None:
            return None
        return self.staging_bundle.repo_id

    def _staged(
        self, action: str, call: Callable[[StagingRunBundleWriterV2], object]
    ) -> None:
        """Make one call on the staging run; an error in it only warns."""
        if self.staging_bundle is None:
            return
        try:
            call(self.staging_bundle)
        except Exception as error:
            print(
                f"warning: the staging run could not {action} "
                f"({type(error).__name__}: {error}); the build and its hosted "
                "telemetry go on.",
                file=sys.stderr,
            )

    def stage(
        self,
        stage: str,
        *,
        message: str | None = None,
        status: str = "running",
        **details: Any,
    ) -> None:
        self.emitter.transition_stage(
            stage,
            status=status,
            message=message,
            **details,
        )
        event_status = _STAGING_EVENT_STATUSES.get(status)
        staged_details = (
            details if event_status is not None else {**details, "stage_status": status}
        )
        self._staged(
            f"record the stage {stage}",
            lambda bundle: bundle.stage(
                stage,
                event_status=event_status or "completed",
                message=_staging_message(message),
                **staged_details,
            ),
        )

    def calibration_progress(self, event: dict[str, object]) -> None:
        if not forwards_calibration_epoch(event, every=self.epoch_every):
            return
        self.emitter.transition_calibration_progress(event)
        self._staged(
            "record calibration progress",
            lambda bundle: bundle.calibration_progress(event),
        )

    def attach_artifact(self, name: str, path: Path | str) -> None:
        if self.staging_bundle is None:
            return
        result_file = US_STAGING_RESULT_FILES.get(name)
        if result_file is None:
            print(
                f"warning: the staging run does not carry the result file {name}, "
                "which is not one it was reviewed for; the build goes on.",
                file=sys.stderr,
            )
            return
        self._staged(
            f"carry the result file {name}",
            lambda bundle: bundle.add_artifact(
                name,
                path,
                artifact_kind=result_file.artifact_kind,
                classification=result_file.classification,
            ),
        )

    def fail(self, error: BaseException) -> None:
        # The staging run's failed state is handed to the emitter, and its
        # final commit confirmed, before the telemetry outcome shuts the
        # emitter down (it drops a run's events after its outcome).
        try:
            # A run whose closing finished keeps its state: an error after it
            # completed is the build's, for the hosted telemetry alone.
            bundle = self.staging_bundle
            if bundle is not None and not bundle.closing_finished:
                ended_by: BaseException = error
                try:
                    self._close_staging_run(bundle, error)
                except KeyboardInterrupt as interrupt:
                    # A SIGTERM or Ctrl-C ends the close only, as one during
                    # the final commit check does (microcosm#1749): the
                    # close-out goes on, so the build keeps its own outcome
                    # and exit status, the run's state as it stands is handed
                    # over once more and its final commit is checked by
                    # receipt.
                    print(
                        "warning: closing the staging run was interrupted; its "
                        "state is handed over once more and the build keeps "
                        "its own outcome.",
                        file=sys.stderr,
                    )
                    ended_by = interrupt
                    try:
                        self._close_staging_run(bundle, error)
                    except KeyboardInterrupt:
                        print(
                            "warning: the staging run's second hand-over was "
                            "interrupted too.",
                            file=sys.stderr,
                        )
                confirm_and_report_final_commit(bundle, self.emitter, error=ended_by)
        finally:
            _fail_emitter(self.emitter, error)

    def _close_staging_run(
        self, bundle: StagingRunBundleWriterV2, error: BaseException
    ) -> None:
        """Close a running run, or finish closing one an interrupt stopped.

        A run whose state changed but whose final state an interrupt kept from
        the emitter (during completion, say) keeps that state and is handed
        over again.
        """
        if bundle.status == "running":
            self._staged("record how the build ended", partial(_close_run, error))
        elif not bundle.closing_finished:
            self._staged(
                "hand over its final state", lambda run: run.hand_over_final_state()
            )

    def complete(self) -> None:
        # An interrupt here is not contained: the build's failure path then
        # records it, so the emitter never hears of a completion first.
        if self.staging_bundle is not None:
            self._staged(
                "record the build's completion", lambda bundle: bundle.complete()
            )
            confirm_and_report_final_commit(self.staging_bundle, self.emitter)
        self.emitter.complete()


#: The progress coordinator for the build in flight, so the entry point can
#: report failure without threading another handle back through the call stack.
_ACTIVE_PROGRESS: _BuildProgress | None = None


def _fail_emitter(emitter: LocalTelemetryEmitter, error: BaseException) -> None:
    """Record the build's failure; a SIGTERM as a termination, as UK builds do.

    An error raised while a SIGTERM unwinds holds it in its cause chain, which
    ``classify_failure`` follows.
    """

    classified = classify_failure(error)
    if classified.error_code == "TERMINATED":
        emitter.fail(
            error,
            failure_class=classified.failure_class or "terminated",
            error_code=classified.error_code,
        )
        return
    emitter.fail(error)


class _WorkCounter:
    """Report completed independent work units through the emitter service."""

    def __init__(
        self,
        *,
        stage_id: str,
        unit: str,
        total: int,
        outer: OuterProgress | None = None,
    ) -> None:
        self.stage_id = stage_id
        self.unit = unit
        self.total = max(0, int(total))
        self.done = 0
        self.started = time.monotonic()
        # The counter one level above, such as the reform pass these batches
        # belong to; sent with every report when set.
        self.outer = outer

    def advance(self, units: int = 1, **details: object) -> None:
        self.done = min(self.total, self.done + max(0, int(units)))
        if _ACTIVE_EMITTER is None:
            return
        if self.outer is not None:
            details["outer"] = self.outer
        _ACTIVE_EMITTER.progress(
            self.stage_id,
            done=self.done,
            total=self.total,
            unit=self.unit,
            elapsed_seconds=time.monotonic() - self.started,
            **details,
        )


_ACTIVE_WORK: _WorkCounter | None = None


def _advance_work(units: int = 1, **details: object) -> None:
    if _ACTIVE_WORK is not None:
        _ACTIVE_WORK.advance(units, **details)


class _ReleaseDryRun:
    """One ``--dry-run-gates-report`` run.

    ``_main`` runs the release as it would build it, up to the point where the
    staged frame goes to target materialization. It differs in four ways:

    * Nothing is written under ``--out`` and staging run files stay off;
      hosted progress reporting remains active.
    * The input-mass, degenerate-input and eCPS parity gates take their
      degraded-mode branch and batch instead of raising, so one report
      carries them with every register.
    * A dirty worktree is recorded, not refused.
    * At the stop point :meth:`finish` grades the staged frame and returns
      the exit code.

    A refusal earlier in ``_main`` reaches :meth:`refused` through
    :func:`main`, so the operator gets a report either way.

    A plain class, not a dataclass: tests load this tool from its file without
    registering it in ``sys.modules``, which dataclass field processing needs.
    """

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        argv: tuple[str, ...],
        report_path: Path,
        margins: DryRunMargins,
        git_dirty: bool,
        started: float,
    ) -> None:
        self.args = args
        self.argv = argv
        self.report_path = report_path
        self.margins = margins
        self.git_dirty = git_dirty
        self.started = started
        # Set once _main reaches the stop point. An error after it is the dry
        # run's own crash, never a refusal the release would share.
        self.stopped = False
        self.seconds_to_stop_point: float | None = None
        # Set by _main after the feed checks; a waived B4923C difference
        # (microcosm#454) is reported here, since a dry run stops before
        # esi_premiums.json is written.
        self.esi_anchor_feed_mismatches: list[str] = []

    @classmethod
    def start(
        cls,
        args: argparse.Namespace,
        argv: Sequence[str] | None,
    ) -> _ReleaseDryRun | None:
        global _ACTIVE_DRY_RUN
        _ACTIVE_DRY_RUN = None
        if args.dry_run_gates_report is None:
            return None
        _ACTIVE_DRY_RUN = cls(
            args=args,
            argv=tuple(sys.argv[1:] if argv is None else argv),
            report_path=Path(args.dry_run_gates_report),
            margins=args.dry_run_margins,
            git_dirty=_git_dirty(),
            started=time.perf_counter(),
        )
        return _ACTIVE_DRY_RUN

    def inputs(self) -> dict[str, object]:
        args = self.args
        return {
            "release_argv": list(self.argv),
            "git_dirty": self.git_dirty,
            "evidence_release": bool(args.evidence_release),
            "waivers": {
                "allow_qrf_tail_concentration": bool(args.allow_qrf_tail_concentration),
                "allow_input_mass_drift": bool(args.allow_input_mass_drift),
                "allow_ecps_parity_gaps": bool(args.allow_ecps_parity_gaps),
                "allow_input_coverage_gaps": bool(args.allow_input_coverage_gaps),
                "allow_esi_premium_gaps": bool(args.allow_esi_premium_gaps),
                "allow_unpinned_feed": bool(args.allow_unpinned_feed),
            },
            "esi_anchor_feed": _esi_anchor_feed_evidence(
                self.esi_anchor_feed_mismatches,
                enforced=not (args.allow_unpinned_feed or args.allow_esi_premium_gaps),
            ),
            "margins": self.margins.to_dict(),
            # Taken when _main reached the stop point, before any grading;
            # None when the release refused earlier.
            "seconds_to_stop_point": self.seconds_to_stop_point,
            "seconds_elapsed": round(time.perf_counter() - self.started, 1),
        }

    def refused(self, error: BaseException) -> int:
        """Report an error that ended the run: exit 1 either way.

        Before the stop point it is the release's own refusal, which the release
        would meet at the same place. After it, the dry run crashed while
        grading, and the report says it certifies nothing.
        """

        report = (
            post_stop_error_report(error, inputs=self.inputs())
            if self.stopped
            else pre_solve_refusal_report(error, inputs=self.inputs())
        )
        return self._write(report)

    def finish(
        self,
        *,
        base_frame: Frame,
        target_specs: Sequence[TargetSpec],
        early_terminal_gate_failures: Sequence[str],
        pre_solve_gates: Mapping[str, GateResult | None],
        input_mass_reference_gate: GateResult | None,
        degenerate_input_gate: GateResult,
        ecps_parity_gate: GateResult,
        binding: Mapping[str, object],
        evidence_owner_patterns: Sequence[tuple[str, str]] = (),
    ) -> int:
        """Grade the staged frame at its base weights and write the report."""

        self.stopped = True
        self.seconds_to_stop_point = round(time.perf_counter() - self.started, 1)
        support_fixed = _full_pool_calibration(
            self.args, int(base_frame.n("household"))
        )
        checks = _release_dry_run_checks(
            self.args,
            margins=self.margins,
            support_fixed=support_fixed,
            base_frame=base_frame,
            target_specs=target_specs,
            early_terminal_gate_failures=early_terminal_gate_failures,
            pre_solve_gates=pre_solve_gates,
            input_mass_reference_gate=input_mass_reference_gate,
            degenerate_input_gate=degenerate_input_gate,
            ecps_parity_gate=ecps_parity_gate,
            evidence_owner_patterns=evidence_owner_patterns,
        )
        registers = {
            "qrf_tail_concentration": _dry_run_register_source(
                self.args.qrf_tail_concentration_exclusions, guarded=True
            ),
            "export_input_mass": _dry_run_constant_register(
                "US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS",
                US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
            ),
            "degenerate_input": _dry_run_constant_register(
                "US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS",
                US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS,
            ),
        }
        inputs = {
            **self.inputs(),
            **dict(binding),
            "calibration_path": "full_pool" if support_fixed else "l0_selection",
            "registers": registers,
        }
        inputs["grading_seconds"] = round(
            float(inputs["seconds_elapsed"]) - self.seconds_to_stop_point, 1
        )
        return self._write(ReleaseDryRunReport(checks=tuple(checks), inputs=inputs))

    def _write(self, report: ReleaseDryRunReport) -> int:
        global _ACTIVE_DRY_RUN
        _ACTIVE_DRY_RUN = None
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(report.to_dict(), indent=2, allow_nan=False, default=str) + "\n"
        )
        print(report.human_table())
        if self.args.evidence_release:
            print(
                "\n--evidence-release: a certain failure whose every release line "
                "an owner pattern matches is reported AT-RISK (OWNED), since an "
                "evidence-tier certification of the build would ship it as a "
                "known failure; any unowned line would refuse that "
                "certification. The build itself has no tier: it records every "
                "failure and writes its H5."
            )
        print(f"\nWrote the dry-run report to {self.report_path}")
        return report.exit_code


#: The dry run in flight, so :func:`main` can turn a refusal before the stop
#: point into that run's report. Like :data:`_ACTIVE_PROGRESS`, a module-level
#: handle rather than a second object threaded back up the build's call stack.
_ACTIVE_DRY_RUN: _ReleaseDryRun | None = None


def _dry_run_register_source(
    path: Path | None, *, guarded: bool = False
) -> dict[str, object]:
    """The register's path, sha256 and entry count for the report.

    ``guarded`` records a file that does not read or load instead of raising:
    the ``qrf_tail_register`` check reports that failure, and the rest of the
    report must survive it.
    """

    if path is None:
        return {"path": None, "sha256": None, "entries": 0}
    source: dict[str, object] = {"path": str(path), "sha256": None, "entries": None}
    try:
        source["sha256"] = _sha256(path)
        source["entries"] = len(_load_qrf_tail_concentration_exclusions(path))
    except Exception as error:
        if not guarded:
            raise
        source["error"] = f"{type(error).__name__}: {error}"
    return source


def _dry_run_constant_register(name: str, register: Mapping[str, str]) -> dict:
    return {
        "source": f"tools/build_us_fiscal_refresh_release.py {name}",
        "sha256": hashlib.sha256(_strict_json_bytes(dict(register))).hexdigest(),
        "entries": len(register),
    }


def _dry_run_checkpoint_status(
    path: Path | None,
    identity_sha256: str,
) -> dict[str, object]:
    """Whether the release would reuse an existing target-frame checkpoint.

    Reads only the checkpoint's ``identity_sha256`` attribute (the release
    writes it beside the frame), never the frame.
    """

    if path is None:
        return {"enabled": False}
    status: dict[str, object] = {
        "enabled": True,
        "path": str(path),
        "exists": path.exists(),
        "identity_sha256": identity_sha256,
    }
    if not path.exists():
        return status
    # Never raises: it is evaluated at the stop point, before finish() marks
    # the run stopped, so an error here must not read as a pre-solve refusal.
    try:
        import h5py

        with h5py.File(path, "r") as h5:
            stored = h5.attrs.get("identity_sha256")
    except Exception as error:
        return {
            **status,
            "identity_matches": False,
            "read_error": f"{type(error).__name__}: {error}",
        }
    if isinstance(stored, bytes):
        stored = stored.decode("utf-8")
    return {**status, "identity_matches": stored == identity_sha256}


def _dry_run_evaluation_error(name: str, error: Exception) -> CheckResult:
    """A check the dry run could not evaluate: nothing about it is certified."""

    return CheckResult(
        name=name,
        status="FAIL",
        summary=(
            "the dry run could not evaluate this check, so it certifies nothing "
            "about it"
        ),
        failures=(f"{type(error).__name__}: {error}",),
        details={"evaluation_error": True},
    )


def _release_dry_run_checks(
    args: argparse.Namespace,
    *,
    margins: DryRunMargins,
    support_fixed: bool,
    base_frame: Frame,
    target_specs: Sequence[TargetSpec],
    early_terminal_gate_failures: Sequence[str],
    pre_solve_gates: Mapping[str, GateResult | None],
    input_mass_reference_gate: GateResult | None,
    degenerate_input_gate: GateResult,
    ecps_parity_gate: GateResult,
    evidence_owner_patterns: Sequence[tuple[str, str]] = (),
) -> list[CheckResult]:
    """Every dry-run check, each through this tool's own gate function.

    The staged frame stands in for the export: on the full-pool path the
    export is this frame with calibrated household weights; on the L0 path it
    is a household selection of it. See
    :mod:`microcosm.build.us_runtime.release_gate_dry_run` for what is certain.
    Under ``--evidence-release``, a preview of an evidence-tier certification
    (a build has no tier, microcosm#1585 item 11), each certain failure is
    graded against the owner patterns with certification's own rule
    (:func:`_evidence_owner`).
    """

    checks: list[CheckResult] = []
    # The exact lines the build would record for each check's certain
    # failures, for the evidence certification's owner check.
    release_lines: dict[str, list[str]] = {}

    def owner_of(line: str) -> str | None:
        return _evidence_owner(line, evidence_owner_patterns)

    def run(name: str, build: Callable[[], CheckResult]) -> None:
        try:
            checks.append(build())
        except Exception as error:
            checks.append(_dry_run_evaluation_error(name, error))

    def pre_solve_battery() -> CheckResult:
        lines = [
            *early_terminal_gate_failures,
            *_pre_solve_gate_failures(
                **pre_solve_gates,
                input_mass_reference_gate=(
                    None if args.allow_input_mass_drift else input_mass_reference_gate
                ),
            ),
        ]
        release_lines["pre_solve_battery"] = lines
        return certain_lines_check(
            "pre_solve_battery",
            lines=lines,
            passed_summary=(
                "every signal, take-up, input-mass-reference and early terminal "
                "gate the release grades before the solve passes on the staged "
                "frame"
            ),
            failed_summary=(
                f"{len(lines)} line(s) the release has already decided before "
                "the solve and will refuse in its terminal batch"
            ),
            details={"early_terminal_gate_failures": len(early_terminal_gate_failures)},
        )

    def qrf_tail() -> CheckResult:
        path = args.qrf_tail_concentration_exclusions
        try:
            register = _load_qrf_tail_concentration_exclusions(path)
        except Exception as error:
            # The release reads this file only at its terminal gates; under
            # earlier failures it records this owned-prefix line instead.
            return qrf_tail_register_unloadable_check(
                _dry_run_register_source(path, guarded=True),
                error,
                evidence_owner=(
                    owner_of(
                        "QRF tail concentration failed: evaluation error under "
                        f"earlier gate failures: {type(error).__name__}: {error}"
                    )
                    if args.evidence_release
                    else None
                ),
            )
        gate, surface = _qrf_tail_concentration_gate(
            base_frame, reviewed_exclusions=register
        )
        mismatch = _qrf_tail_register_mismatch(register, gate)
        gate_lines, register_lines = _qrf_tail_gate_lines(
            gate,
            mismatch,
            allow_concentration=args.allow_qrf_tail_concentration,
        )

        def tail_owner(column: str) -> str | None:
            # The release's line for a concentrated column; its share is only
            # known at calibrated weights, so match on the fixed prefix.
            return owner_of(
                f"QRF tail concentration failed: {column}: top "
                f"{US_QRF_TAIL_CONCENTRATION_TOP_K} weighted records carry"
            )

        return qrf_tail_register_check(
            qrf_outputs=sorted(_qrf_imputed_source_outputs()),
            register=register,
            surface=surface,
            gate_details=gate.details,
            release_lines_at_base_weights=[*gate_lines, *register_lines],
            support_fixed=support_fixed,
            margins=margins,
            allow_concentration=args.allow_qrf_tail_concentration,
            register_source=_dry_run_register_source(path),
            evidence_owner=tail_owner if args.evidence_release else None,
        )

    def export_input_mass() -> CheckResult:
        reference_h5 = args.export_input_mass_reference_h5
        reference_frame = load_us_frame(reference_h5) if reference_h5 else None
        reference_name = reference_h5.name if reference_h5 else "base_frame"
        gate = _export_input_mass_gate(
            base_frame,
            base_frame,
            relative_tolerance=args.input_mass_relative_tolerance,
            minimum_reference_total=args.input_mass_minimum_reference_total,
            reference_frame=reference_frame,
            reference_name=reference_name,
            reviewed_exclusions=US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
        )
        input_variables = _engine_input_variables()
        candidate_totals = us_input_mass_totals(base_frame, columns=input_variables)
        reference_totals = (
            candidate_totals
            if reference_frame is None
            else us_input_mass_totals(reference_frame, columns=input_variables)
        )
        check = export_input_mass_check(
            gate_failures=gate.failures,
            gate_details=gate.details,
            candidate_totals=us_line_32_on_reference_terms(
                candidate_totals,
                reference_totals,
                us_line_32_carriers(
                    base_frame,
                    base_frame if reference_frame is None else reference_frame,
                ),
            ),
            reference_totals=reference_totals,
            candidate_nonzero_shares=us_nonzero_shares(
                base_frame, columns=input_variables
            ),
            reviewed_exclusions=US_EXPORT_INPUT_MASS_REVIEWED_EXCLUSIONS,
            relative_tolerance=args.input_mass_relative_tolerance,
            minimum_reference_total=args.input_mass_minimum_reference_total,
            margin=margins.mass_drift,
            allow_drift=args.allow_input_mass_drift,
            reference_is_candidate=reference_frame is None,
            reference_label=reference_name,
        )
        certain = {row["column"] for row in check.rows if row.get("status") == "FAIL"}
        release_lines["export_input_mass"] = [
            f"Input mass parity failed: {failure}"
            for failure in gate.failures
            if failure.split(":", 1)[0] in certain
        ]
        return check

    def degenerate_input() -> CheckResult:
        lines = _pre_solve_gate_failures(degenerate_input_gate=degenerate_input_gate)
        release_lines["degenerate_input_register"] = lines
        return degenerate_input_register_check(
            gate_passed=degenerate_input_gate.passed,
            gate_details=degenerate_input_gate.details,
            register=US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS,
            release_lines=lines,
        )

    def ecps_parity() -> CheckResult:
        lines = _pre_solve_gate_failures(ecps_parity_gate=ecps_parity_gate)
        release_lines["ecps_parity_register"] = lines
        return ecps_parity_register_check(
            gate_passed=ecps_parity_gate.passed,
            gate_details=ecps_parity_gate.details,
            release_lines=lines,
            waived=args.allow_ecps_parity_gaps,
        )

    def input_coverage() -> CheckResult:
        engine = PolicyEngineUSEngine()
        gate = us_release_input_coverage_gate(base_frame, engine)
        check = input_coverage_register_check(
            gate_passed=gate.passed,
            gate_failures=gate.failures,
            gate_details=gate.details,
            support_fixed=support_fixed,
            waived=args.allow_input_coverage_gaps,
            signal_counts=(
                None
                if support_fixed
                else us_release_input_coverage_signal_counts(base_frame, engine)
            ),
            margins=margins,
        )
        release_lines["input_coverage_register"] = [
            f"Input coverage failed: {failure}" for failure in check.failures
        ]
        return check

    def export_signal_regrades() -> CheckResult:
        tax_unit = base_frame.table("tax_unit")
        reported = pre_solve_gates.get("reported_coverage_vintage_gate")
        if reported is None:
            reported = us_reported_coverage_vintage_signal_gate(base_frame)
        return export_signal_regrades_check(
            health_value_counts={
                column: observed_value_counts(tax_unit[column].to_numpy())
                for column in US_HEALTH_INPUT_NONCONSTANT_COLUMNS
                if column in tax_unit.columns
            },
            reported_coverage_details=reported.details,
            support_fixed=support_fixed,
            margins=margins,
        )

    def stored_inputs() -> CheckResult:
        # The export keeps every staged column (with_weights and select both
        # preserve them), so the release's verdict on the export is this one.
        lines, details = _stored_input_gate_failures(base_frame, stage="export frame")
        release_lines["stored_inputs"] = list(lines)
        return certain_lines_check(
            "stored_inputs",
            lines=lines,
            passed_summary=(
                "every stored model-named column is a variable of the installed "
                "policyengine-us or in the reviewed non-variable register"
            ),
            failed_summary=f"{len(lines)} stored column(s) the engine would ignore",
            details=details,
        )

    def spm_composition() -> CheckResult:
        lines, details = _spm_composition_gate_failures(
            base_frame, stage="export frame"
        )
        release_lines["spm_composition"] = list(lines)
        return certain_lines_check(
            "spm_composition",
            lines=lines,
            passed_summary="every SPM unit has a classified adult",
            failed_summary=(
                "SPM units without a classified adult"
                + (
                    ""
                    if support_fixed
                    else " (L0 path: the solve's selection may drop them)"
                )
            ),
            certain=support_fixed,
            details=details,
        )

    def zero_support() -> CheckResult:
        gated = [
            spec
            for spec in target_specs
            if args.gate_congressional_district_targets
            or not _target_is_congressional_district(spec)
        ]
        check = check_zero_support_preview(base_frame, gated)
        zeros = [
            str(row["target"])
            for row in check.rows
            if row.get("verdict") == "zero_support"
        ]
        if zeros:
            # The release's own line for these targets (_release_gate_failures).
            examples = ", ".join(zeros[:5]) + ("" if len(zeros) <= 5 else ", ...")
            release_lines["zero_support_preview"] = [
                f"{len(zeros)} positive fiscal targets have zero materialized "
                f"support (examples: {examples})."
            ]
        check = dataclasses.replace(
            check,
            details={
                **dict(check.details),
                "congressional_district_targets_not_gated": len(target_specs)
                - len(gated),
                "graded_frame": "staged frame (before target materialization)",
            },
        )
        return bound_zero_support_on_l0(
            check, support_fixed=support_fixed, margins=margins
        )

    run("pre_solve_battery", pre_solve_battery)
    run("qrf_tail_register", qrf_tail)
    run("export_input_mass", export_input_mass)
    run("degenerate_input_register", degenerate_input)
    run("ecps_parity_register", ecps_parity)
    run("input_coverage_register", input_coverage)
    run("export_signal_regrades", export_signal_regrades)
    run("stored_inputs", stored_inputs)
    run("spm_composition", spm_composition)
    run("zero_support_preview", zero_support)
    if args.evidence_release:
        # With no earlier terminal failure on record, the build raises at the
        # first failing one of these gates before the solve, so there is no
        # build for a certification to own that refusal in.
        refused_before_solve: set[str] = set()
        if not early_terminal_gate_failures:
            if (
                input_mass_reference_gate is not None
                and not input_mass_reference_gate.passed
                and not args.allow_input_mass_drift
            ):
                refused_before_solve.add("pre_solve_battery")
            if not degenerate_input_gate.passed:
                refused_before_solve.add("degenerate_input_register")
            if not ecps_parity_gate.passed and not args.allow_ecps_parity_gaps:
                refused_before_solve.add("ecps_parity_register")
        checks = [
            dataclasses.replace(
                check,
                details={
                    **dict(check.details),
                    "evidence_ownership": (
                        "not ownable: the build refuses it before the solve, "
                        "so no build is left to certify"
                    ),
                },
            )
            if check.name in refused_before_solve
            else apply_evidence_ownership(
                check,
                release_lines=release_lines.get(check.name, ()),
                owner_of=owner_of,
            )
            for check in checks
        ]
    checks.append(not_previewable_check())
    return checks


def _staging_manifest_block(telemetry: _BuildProgress | None) -> dict[str, object]:
    """Record what staging did and distinguish an opt-out from non-delivery.

    Uploads are best-effort and pause after three refused in a row until the
    run's final state, so a configured destination is not evidence that
    anything reached it.
    """

    if telemetry is None or telemetry.staging_bundle is None:
        reason = (
            "--no-staging"
            if telemetry is None
            else telemetry.staging_opt_out_reason or "--no-staging"
        )
        return disabled_staging_delivery(reason)
    return telemetry.staging_bundle.delivery_summary


def _build_progress(
    args: argparse.Namespace,
    *,
    release_root: Path,
    release_id: str,
    run_id: str | None = None,
    emitter: LocalTelemetryEmitter,
    epoch_every: int | None = None,
    candidate_id: str | None = None,
    operation_id: str = US_BUILD_STAGING_OPERATION,
    run_kind: str = "release",
) -> _BuildProgress:
    """The build's progress, with its version 2 staging run unless opted out.

    The run carries the identity the build's hosted telemetry has: its
    candidate (``candidate_id``, the build itself by default), the release
    build it is to become (``release_id``) and its kind of run. Its files are
    written below ``--staging-dir`` (``<out>/staging`` by default), at
    ``runs/<run_id>/``, and committed through the build's emitter when a
    staging repository is set.
    """
    global _ACTIVE_PROGRESS
    # Each call establishes the current run, so a handle from a previous one
    # can never be marked failed in place of this build's.
    _ACTIVE_PROGRESS = None
    run_id = run_id or args.staging_run_id or release_id
    if epoch_every is None:
        epoch_every = _calibration_epoch_every(args)
    if args.no_staging:
        _ACTIVE_PROGRESS = _BuildProgress(
            run_id=run_id,
            emitter=emitter,
            staging_bundle=None,
            staging_opt_out_reason="--no-staging",
            epoch_every=epoch_every,
        )
        return _ACTIVE_PROGRESS
    if not args.staging_dir and not args.staging_repo_id:
        # The parser rejects this combination, so reaching it means a caller
        # built the namespace directly. Returning None here would reinstate
        # the silent skip the parser guard exists to prevent.
        raise ValueError(
            "staging is enabled but has no destination: staging_repo_id is "
            "empty and staging_dir is unset. Set no_staging to skip staging, "
            "or give a staging_dir for a local-only run."
        )
    remote = bool(args.staging_repo_id)
    staging_bundle = StagingRunBundleWriterV2(
        run_id=run_id,
        country_code="US",
        operation_id=operation_id,
        pipeline_id=US_STAGING_PIPELINE,
        pipeline_version=importlib.metadata.version("microcosm-build"),
        candidate_id=candidate_id or release_id,
        release_id=release_id,
        local_dir=args.staging_dir or release_root / "staging",
        run_kind=run_kind,
        delivery_mode="local_and_remote" if remote else "local_only",
        repo_id=args.staging_repo_id if remote else None,
        # The build's emitter commits the run files with its own credential.
        emitter_client=emitter.client if remote else None,
        content_policy=US_STAGING_CONTENT_POLICY,
    )
    staging_bundle.set_sample({"mode": "full"})
    _ACTIVE_PROGRESS = _BuildProgress(
        run_id=run_id,
        emitter=emitter,
        staging_bundle=staging_bundle,
        epoch_every=epoch_every,
    )
    return _ACTIVE_PROGRESS


class _TerminalBatchProgress:
    """Turn terminal-batch telemetry crashes into release-gate failures.

    The proxy is deliberately scoped to the post-diagnostics terminal batch.
    A telemetry crash on an otherwise-green run now fails the release as a
    recorded batch line rather than causing an opaque abort, and every later
    gate group still gets a chance to contribute its evidence.
    """

    def __init__(
        self,
        telemetry: _BuildProgress | None,
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

    def attach_artifact(self, name: str, path: Path | str) -> None:
        self._call("attach_artifact", name, path)


def _print_build_result(
    *,
    build: str,
    release_dir: Path,
    artifact_root: Path,
    publication: Mapping[str, str],
) -> None:
    """Print the build, where it is, and the commands that publish it.

    The release is the release directory's name. ``publication`` is
    :func:`_finish_build_folder_release`'s result: ``publish_command`` lists
    the build in its release without making it current, and
    ``promote_command``, run after review, makes it the release's current
    build, as ``promotion`` says.
    """

    release = release_dir.name
    print(
        json.dumps(
            {
                "release": release,
                "build": build,
                "release_dir": str(release_dir),
                "artifact_root": str(artifact_root),
                "publish_command": publication["publish_command"],
                "promote_command": publication["promote_command"],
                "promotion": _promotion_note(release),
            },
            indent=2,
        )
    )


def _is_release_refusal(error: BaseException) -> bool:
    """Whether ``error`` is the release refusing, rather than being stopped.

    An exception, or a ``SystemExit`` carrying a message (the dirty-worktree and
    feed-pin refusals), is a refusal. An interrupt, or argparse's integer exit,
    is not.
    """

    if isinstance(error, SystemExit):
        return not isinstance(error.code, int) and error.code is not None
    return isinstance(error, Exception)


#: The exit status of a build that failed release gates once its H5 and its
#: record are written (microcosm#1585 item 11). A build has no tier, so such
#: a build may still be certified at the evidence tier; a driver tells it
#: from a refusal or a crash, which exit 1, and keeps it rather than
#: building again.
US_BUILD_FAILED_RELEASE_GATES_EXIT = 3


class _BuildFailedReleaseGatesError(RuntimeError):
    """A finished build's refusal: its record carries release-gate failures.

    Raised once the H5, the calibration weights and the record are written;
    run as a script, the build then exits
    :data:`US_BUILD_FAILED_RELEASE_GATES_EXIT`. ``failures`` are the record's.
    """

    def __init__(self, failures: Sequence[str]) -> None:
        self.failures = tuple(failures)
        super().__init__("Release gates failed: " + "; ".join(self.failures))


#: The build record a build leaves beside ``artifacts/`` and ``build/``
#: (microcosm#1585 item 11, PR 4a). The build ends once its H5, calibration
#: weights, diagnostics and gate evidence are written; certification is a
#: separate step (``tools/certify_us_national_build.py``) that runs the
#: release's post-export checks from this record, places the build in the
#: release of the tier it chooses and writes the manifests the publisher
#: requires.
US_BUILD_RECORD_FILENAME = "build_record.json"
US_BUILD_RECORD_KIND = "microcosm_us_national_build_record"
#: Schema 2 added the build's provenance (``built_at``, ``runtime``), its
#: staging settings and the record's own digest (``record_sha256``). Schema 3
#: dropped the tier: a build has no release, no ``evidence_release`` setting
#: and no owner patterns, and its own directory is ``layout.build_dir``; it
#: lists the failures of the written H5's own checks apart, as
#: ``artifact_integrity_failures`` (microcosm#1585 item 11). A record of
#: another schema is certified with the code that built it.
US_BUILD_RECORD_SCHEMA_VERSION = 3
#: The options of the build that its post-export checks read. Certification
#: takes them from the record, never from its own command line, so a build is
#: certified exactly as it was configured. The tier is certification's own
#: option.
US_BUILD_RECORD_SETTINGS = (
    "allow_reform_coverage_smoke_failures",
    "maximum_microsim_batch_size",
    "scf_education_debt",
    "skip_demographics",
    "skip_out_of_sample_reforms",
    "skip_reform_coverage_smoke",
    "skip_reform_validation",
)
#: The build's gate verdicts its manifests carry, under the keyword
#: :func:`_build_manifests` takes each by.
US_BUILD_RECORD_GATES = (
    "target_profile_gate",
    "ssi_take_up_delivery_gate_result",
    "health_input_gate",
    "base_population_gate",
    "immigration_gate",
    "degenerate_input_gate",
    "ecps_parity_gate",
    "hours_worked_gate",
    "snap_take_up_gate",
    "eligibility_inputs_gate",
    "pregnancy_gate",
    "reported_coverage_vintage_gate",
    "snap_discretionary_exemption_gate",
)


def _build_record_gate(gate: GateResult | None) -> dict[str, object] | None:
    if gate is None:
        return None
    return {
        "name": gate.name,
        "passed": bool(gate.passed),
        "failures": list(gate.failures),
        "details": dict(gate.details),
    }


def _gate_from_build_record(record: Mapping[str, Any] | None) -> GateResult | None:
    if record is None:
        return None
    return GateResult(
        name=str(record["name"]),
        passed=bool(record["passed"]),
        failures=tuple(str(line) for line in record["failures"]),
        details=dict(record["details"]),
    )


def _build_record_loss(value: float) -> float | str:
    """A calibration loss in strict JSON: non-finite losses as their names."""
    value = float(value)
    return value if math.isfinite(value) else repr(value)


def _build_record_finite(value: object) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, allow_nan=False)


class _RecordedCalibrationResult:
    """The calibration result as certification reads it, from the record.

    The post-export consumers read each target's name, final estimate and
    target value (reform validation's in-sample rows); the manifests read
    the three losses, since the release's gate summary is computed when the
    build ends and recorded (``release_gate_failures``). (A plain class:
    this tool is loaded by file path in the tests, outside ``sys.modules``,
    where ``dataclasses`` cannot resolve the module's string annotations.)
    """

    def __init__(
        self,
        *,
        initial_loss: float | str,
        final_loss: float | str,
        fraction_within_10pct: float | str,
        in_sample_fit: Sequence[Sequence[object]],
    ) -> None:
        self.initial_loss = float(initial_loss)
        self.final_loss = float(final_loss)
        self.fraction_within_10pct = float(fraction_within_10pct)
        self.diagnostics = [
            SimpleNamespace(final_estimate=estimate, target=target)
            for _, estimate, target in in_sample_fit
        ]
        self.problem = SimpleNamespace(
            targets=[SimpleNamespace(name=str(name)) for name, _, _ in in_sample_fit]
        )


class _RecordedTargetRegistry:
    """The target registry as :func:`_build_manifests` reads it: version, size."""

    def __init__(self, *, version: object, n_specs: int) -> None:
        self.version = version
        self.n_specs = int(n_specs)

    def __len__(self) -> int:
        return self.n_specs


def _us_build_files_certification_reads(
    release_dir: Path, *, scf_education_debt: bool
) -> dict[str, Path]:
    """The build's own files in its release directory that certification reads.

    Certification hashes each into the manifests (and reads the QRF tail
    register and the stored-input verdict back from theirs), so the build
    record pins each one the build wrote, under its artifact key in the
    manifests. Gate evidence that certification writes itself (the
    reform-coverage smoke) is not among them.
    """

    files = {"us_ssi_take_up": release_dir / "us_ssi_take_up.json"}
    files.update(
        (key, release_dir / filename)
        for key, filename in US_RELEASE_GATE_EVIDENCE_FILES.items()
        if filename not in US_CERTIFICATION_OUTPUT_FILES
    )
    if scf_education_debt:
        files.update(
            (key, release_dir / filename)
            for key, filename in US_SCF_EDUCATION_DEBT_FIT_ARTIFACTS.items()
        )
    return files


def _us_build_record(
    *,
    release_id: str,
    release_root: Path,
    release_dir: Path,
    artifact_root: Path,
    dataset_key: str,
    dataset_filename: str,
    calibration_key: str,
    calibration_filename: str,
    commit: str,
    built_at: str,
    runtime: Mapping[str, str],
    args: argparse.Namespace,
    terminal_gate_failures: Sequence[str],
    n_households: int,
    artifact_integrity_failures: Sequence[str] = (),
    result,
    post_export_scoring_plan: _PostExportScoringPlan,
    release_gate_failures: Sequence[str] | None,
    registry: TargetRegistry,
    target_surface: Mapping[str, object],
    dropped: Mapping[str, object],
    gates: Mapping[str, GateResult | None],
    warm_start_calibration: Mapping[str, object] | None,
    selection_source: Mapping[str, object] | None,
    target_surface_selection: Mapping[str, object] | None,
    ledger_artifact: Mapping[str, object] | None,
    default_dataset: Mapping[str, object] | None,
    medicaid_enrollment_substitutions: Sequence[Mapping[str, object]],
    exact_k_ladder: Mapping[str, object] | None,
    base_pool: Mapping[str, object] | None,
    acs_predictor_join: Mapping[str, object] | None,
    export_input_mass_reference: Mapping[str, object] | None,
    calibration_runtime: Mapping[str, object] | None,
    diagnostics_outcome: DiagnosticsWriteOutcome,
    scf_wealth_signal: Mapping[str, object] | None,
    fiscal_target_sources: object,
    congressional_district_vintage_crosswalk: Mapping[str, object] | None,
    fiscal_target_exclusion_receipt: object,
    fsa_target_receipt: object,
    take_up_participation: Mapping[str, object],
    medicaid_take_up_diagnostics: Mapping[str, object],
    snap_state_take_up_diagnostics: Mapping[str, object],
    timing: Mapping[str, object],
    staging: Mapping[str, object],
    staging_settings: Mapping[str, object],
) -> dict[str, object]:
    """What certification needs of a finished build, as strict JSON.

    Everything certification reads that is not a file of the build: its
    provenance (its commit, the time it finished and the packages that built
    it), options, gate verdicts, calibration summary, scoring plan and the
    receipts that ride the manifests. The files themselves are pinned by
    digest: the H5, the calibration weights, the diagnostics and every other
    file of the build that certification reads
    (:func:`_us_build_files_certification_reads`).
    """

    if set(gates) != set(US_BUILD_RECORD_GATES):
        raise ValueError(
            "The build record carries exactly the gates its manifests write: "
            f"expected {sorted(US_BUILD_RECORD_GATES)}, got {sorted(gates)}."
        )

    def relative(path: Path) -> str:
        return path.relative_to(release_root).as_posix()

    files = {
        "dataset": artifact_root / dataset_filename,
        "calibration_weights": artifact_root / calibration_filename,
    }
    if diagnostics_outcome.status == "available":
        files["calibration_diagnostics"] = release_dir / "calibration_diagnostics.json"
        diagnostics = {
            "status": "available",
            "schema_version": diagnostics_outcome.schema_version,
            "sha256": diagnostics_outcome.sha256,
        }
    else:
        diagnostics = {
            "status": "failed",
            "expected_schema_version": diagnostics_outcome.expected_schema_version,
            "error_code": diagnostics_outcome.error_code,
            "message": diagnostics_outcome.message,
        }
    files.update(
        (role, path)
        for role, path in _us_build_files_certification_reads(
            release_dir, scf_education_debt=bool(args.scf_education_debt)
        ).items()
        if path.is_file()
    )
    return {
        "schema_version": US_BUILD_RECORD_SCHEMA_VERSION,
        "kind": US_BUILD_RECORD_KIND,
        "build": release_id,
        "code": {"repository": "PolicyEngine/microcosm", "git_commit": commit},
        # The manifests' built_at and built-with packages: the time the build
        # finished and the packages that built it, not certification's.
        "built_at": built_at,
        "runtime": dict(runtime),
        "layout": {
            "build_dir": relative(release_dir),
            "artifact_root": relative(artifact_root),
            "dataset_key": dataset_key,
            "dataset_filename": dataset_filename,
            "calibration_key": calibration_key,
            "calibration_filename": calibration_filename,
        },
        "files": {
            role: {"path": relative(path), "sha256": _sha256(path)}
            for role, path in files.items()
        },
        "settings": {name: getattr(args, name) for name in US_BUILD_RECORD_SETTINGS},
        "terminal_gate_failures": list(terminal_gate_failures),
        # The ones the written H5 failed in its own checks, which no tier
        # certifies; each is also among terminal_gate_failures.
        "artifact_integrity_failures": list(artifact_integrity_failures),
        "n_households": int(n_households),
        # The calibration fit as the reform-validation consumer reads it: each
        # target's name, final estimate and target value, in solve order.
        "in_sample_fit": [
            [
                str(target.name),
                _build_record_finite(diagnostic.final_estimate),
                _build_record_finite(diagnostic.target),
            ]
            for diagnostic, target in zip(
                result.diagnostics, result.problem.targets, strict=True
            )
        ],
        "post_export_scoring_plan": post_export_scoring_plan.record(),
        "manifest": {
            "result": {
                "initial_loss": _build_record_loss(result.initial_loss),
                "final_loss": _build_record_loss(result.final_loss),
                "fraction_within_10pct": _build_record_loss(
                    result.fraction_within_10pct
                ),
            },
            "registry": {"version": registry.version, "n_specs": len(registry)},
            "release_gate_failures": (
                None if release_gate_failures is None else list(release_gate_failures)
            ),
            "target_surface": dict(target_surface),
            "dropped": dict(dropped),
            "gates": {
                name: _build_record_gate(gates[name]) for name in US_BUILD_RECORD_GATES
            },
            "warm_start_calibration": warm_start_calibration,
            "selection_source": selection_source,
            "target_surface_selection": target_surface_selection,
            "ledger_artifact": ledger_artifact,
            "default_dataset": default_dataset,
            "medicaid_enrollment_substitutions": [
                dict(record) for record in medicaid_enrollment_substitutions
            ],
            "exact_k_ladder": exact_k_ladder,
            "base_pool": base_pool,
            "acs_predictor_join": acs_predictor_join,
            "export_input_mass_reference": export_input_mass_reference,
            "calibration_runtime": calibration_runtime,
            "diagnostics_outcome": diagnostics,
            "scf_wealth_signal": scf_wealth_signal,
        },
        "source_coverage": {
            "fiscal_target_sources": fiscal_target_sources,
            "congressional_district_vintage_crosswalk": (
                congressional_district_vintage_crosswalk
            ),
            "fiscal_target_exclusion_receipt": fiscal_target_exclusion_receipt,
            "fsa_student_loan_target_receipt": fsa_target_receipt,
        },
        "take_up": {
            "participation": take_up_participation,
            "medicaid": medicaid_take_up_diagnostics,
            "snap_state": snap_state_take_up_diagnostics,
        },
        "timing": dict(timing),
        "staging": dict(staging),
        # Where the build staged, so certification stages there too.
        "staging_settings": dict(staging_settings),
    }


def _us_build_record_digest(record: Mapping[str, object]) -> str:
    """The sha256 of a build record's canonical JSON, its own digest aside."""

    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _write_us_build_record(release_root: Path, record: Mapping[str, object]) -> Path:
    """Write the build record atomically; certification refuses a torn one.

    The record carries a digest of itself, so certification refuses one
    edited after its build wrote it. That catches an edit, not a forgery:
    anyone who recomputes the digest can change the record, until
    certification signs (PR 4b).
    """

    path = release_root / US_BUILD_RECORD_FILENAME
    # Digested as certification reads it back: JSON writes every key as a
    # string, so a mapping keyed otherwise would digest differently once read.
    record = json.loads(
        json.dumps(
            {key: value for key, value in record.items() if key != "record_sha256"},
            allow_nan=False,
        )
    )
    record["record_sha256"] = _us_build_record_digest(record)
    text = json.dumps(record, indent=1, allow_nan=False) + "\n"
    staged = path.with_name(f".{path.name}.tmp")
    staged.write_text(text, encoding="utf-8")
    staged.replace(path)
    return path


class _USBuildRecord:
    """A finished build, read back from its :data:`US_BUILD_RECORD_FILENAME`.

    Paths are relative to the record's directory, so a build folder moved to
    another machine certifies in place. ``build_dir`` holds the build's own
    files; certification copies them into the release directory of the tier
    it chooses (microcosm#1585 item 11). (A plain class, for the reason
    :class:`_RecordedCalibrationResult` gives.)
    """

    def __init__(self, payload: Mapping[str, Any], *, release_root: Path) -> None:
        layout = payload["layout"]
        self.payload = payload
        self.release_root = release_root
        self.build = str(payload["build"])
        self.commit = str(payload["code"]["git_commit"])
        self.built_at = str(payload["built_at"])
        self.runtime = {
            str(name): str(version) for name, version in payload["runtime"].items()
        }
        self.build_dir = release_root / layout["build_dir"]
        self.artifact_root = release_root / layout["artifact_root"]
        self.dataset_key = str(layout["dataset_key"])
        self.dataset_filename = str(layout["dataset_filename"])
        self.calibration_key = str(layout["calibration_key"])
        self.calibration_filename = str(layout["calibration_filename"])
        self.settings = dict(payload["settings"])
        self.terminal_gate_failures = tuple(
            str(line) for line in payload["terminal_gate_failures"]
        )
        self.artifact_integrity_failures = tuple(
            str(line) for line in payload["artifact_integrity_failures"]
        )
        self.n_households = int(payload["n_households"])
        self.timing = dict(payload["timing"])
        self.staging = dict(payload["staging"])
        self.staging_settings = dict(payload["staging_settings"])
        self.manifest_inputs = dict(payload["manifest"])
        self.source_coverage = dict(payload["source_coverage"])
        self.take_up = dict(payload["take_up"])

    def certify_args(self, *, evidence_release: bool) -> argparse.Namespace:
        """The build's options its post-export checks read, with the tier
        certification chose."""
        return argparse.Namespace(**self.settings, evidence_release=evidence_release)

    def exact_k(self) -> bool:
        """Whether the build is an exact-k ladder build of the national release."""
        return self.manifest_inputs.get("exact_k_ladder") is not None

    def post_export_scoring_plan(
        self, args: argparse.Namespace, *, result
    ) -> _PostExportScoringPlan:
        """The build's scoring plan, rebuilt and held to the one it recorded."""
        plan = _record_post_export_scoring_plan(
            args, n_households=self.n_households, result=result, release_id=self.build
        )
        recorded = self.payload["post_export_scoring_plan"]
        if _canonical_json(plan.record()) != _canonical_json(recorded):
            raise RuntimeError(
                f"Build {self.build}: the post-export scoring plan rebuilt for "
                "certification differs from the one the build recorded in "
                f"{US_BUILD_RECORD_FILENAME}; certify the build with the code "
                f"it was built with ({self.commit})."
            )
        return plan

    def gates(self) -> dict[str, GateResult | None]:
        recorded = self.manifest_inputs["gates"]
        return {
            name: _gate_from_build_record(recorded[name])
            for name in US_BUILD_RECORD_GATES
        }

    def calibration_result(self) -> _RecordedCalibrationResult:
        return _RecordedCalibrationResult(
            **self.manifest_inputs["result"],
            in_sample_fit=self.payload["in_sample_fit"],
        )

    def registry_summary(self) -> _RecordedTargetRegistry:
        return _RecordedTargetRegistry(**self.manifest_inputs["registry"])

    def diagnostics_outcome(self, release_dir: Path) -> DiagnosticsWriteOutcome:
        """The diagnostics outcome, its file at ``release_dir`` once placed there."""
        recorded = dict(self.manifest_inputs["diagnostics_outcome"])
        if recorded["status"] == "available":
            return DiagnosticsWriteSuccess(
                status="available",
                path=Path(release_dir) / "calibration_diagnostics.json",
                schema_version=recorded["schema_version"],
                sha256=recorded["sha256"],
            )
        return DiagnosticsWriteFailure(
            status="failed",
            expected_schema_version=recorded["expected_schema_version"],
            error_code=recorded["error_code"],
            message=recorded["message"],
        )


def _certify_with_its_own_code(payload: Mapping[str, object]) -> str:
    """Where a record of another schema certifies: with the code that built it."""

    code = payload.get("code")
    commit = code.get("git_commit") if isinstance(code, dict) else None
    built_with = f" ({commit})" if isinstance(commit, str) and commit else ""
    return f"Certify the build with the code it was built with{built_with}"


def _read_us_build_record(path: Path) -> _USBuildRecord:
    """Read a build record and check the files it pins are the build's."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read the build record {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("kind") != US_BUILD_RECORD_KIND:
        raise RuntimeError(f"{path} is not a US national build record.")
    # Before the digest: a record of another schema need not carry one.
    if payload.get("schema_version") != US_BUILD_RECORD_SCHEMA_VERSION:
        raise RuntimeError(
            f"{path} has build record schema {payload.get('schema_version')!r}; "
            f"this tool reads schema {US_BUILD_RECORD_SCHEMA_VERSION}. "
            f"{_certify_with_its_own_code(payload)}, or build it again into a "
            "new --out."
        )
    if payload.get("record_sha256") != _us_build_record_digest(payload):
        raise RuntimeError(
            f"{path} does not hash to the record_sha256 it carries; the record "
            "changed after its build wrote it."
        )
    release_root = path.resolve().parent
    for role, pinned in payload["files"].items():
        file_path = release_root / pinned["path"]
        if not file_path.is_file():
            raise RuntimeError(
                f"Build {payload['build']}: its {role} file {file_path} is missing."
            )
        actual = _sha256(file_path)
        if actual != pinned["sha256"]:
            raise RuntimeError(
                f"Build {payload['build']}: its {role} file {file_path} hashes to "
                f"{actual}, not the {pinned['sha256']} the build recorded; the "
                "build's files changed after it finished."
            )
    # A file certification would read that the build did not write: the
    # manifests would bind it with no digest the build recorded.
    for role, file_path in _us_build_files_certification_reads(
        release_root / payload["layout"]["build_dir"],
        scf_education_debt=bool(payload["settings"]["scf_education_debt"]),
    ).items():
        if role not in payload["files"] and file_path.is_file():
            raise RuntimeError(
                f"Build {payload['build']}: {file_path} ({role}) is not a file "
                "the build recorded; it appeared after the build finished."
            )
    return _USBuildRecord(payload, release_root=release_root)


#: Files certification writes into the release directory, beside the
#: manifests. A rerun after a failed certification removes them first, so a
#: check this run skips cannot leave an earlier attempt's verdict behind.
US_CERTIFICATION_OUTPUT_FILES = (
    "reform_coverage_smoke.json",
    "reform_validation.json",
    "demographics.json",
    "us_ssi_social_security_holdout.json",
    "us_medicare_irmaa_holdout.json",
    "us_population_structure_holdout.json",
    "us_source_coverage.json",
    "us_take_up_participation.json",
    "us_medicaid_take_up.json",
    "us_snap_state_take_up.json",
    "build_manifest.json",
)


def _certify_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="certify_us_national_build.py",
        description=(
            "Certify a finished US national build: run the release's "
            "post-export checks (the reform-coverage smoke, reform validation, "
            "demographics and the holdouts) on its written H5, place the build "
            "in the release of the tier chosen here and write the manifests "
            "that microcosm-publish-release-build requires. The build is the "
            "--out directory of tools/build_us_fiscal_refresh_release.py, "
            f"which holds {US_BUILD_RECORD_FILENAME}. A build has no tier "
            "(microcosm#1585 item 11): the certified tier is the default, and "
            "--evidence-release chooses the evidence tier."
        ),
    )
    parser.add_argument(
        "build_dir",
        type=Path,
        help=f"The build's --out directory, holding {US_BUILD_RECORD_FILENAME}.",
    )
    parser.add_argument(
        "--evidence-release",
        action="store_true",
        help=(
            "Certify at the EVIDENCE tier (microcosm#506), into the evidence "
            f"release {US_EVIDENCE_RELEASE_ID} (never the US default), under "
            "the build ID with an '-evidence-' segment added. Every failure "
            "the build recorded or certification finds must match an owner "
            "pattern, from --evidence-failure-owners first, then the standing "
            "register (the US evidence_failure_owners.json), and rides the "
            "release manifest's known_failures block; at least one must "
            "exist, since a build that passes every gate belongs on the "
            "certified tier. A QRF tail-concentration register mismatch is "
            "never owned, and an --exact-k build is a build of the national "
            "release, never evidence. Without this flag the build is "
            f"certified into the national release {US_NATIONAL_RELEASE_ID}, "
            "which refuses any failure."
        ),
    )
    parser.add_argument(
        "--evidence-failure-owners",
        type=Path,
        help=(
            "With --evidence-release: a JSON file mapping failure-substring "
            "patterns to owner issue refs (e.g. "
            '{"Input coverage failed:": "PolicyEngine/microcosm#368"}), '
            "checked ahead of the standing register; a failure matching "
            "neither refuses the certification, since every shipped failure "
            "needs an owner."
        ),
    )
    parser.add_argument(
        "--no-staging",
        action="store_true",
        help=(
            "Certify without a staging run, even where the build staged; "
            "hosted telemetry remains active. The build manifest's staging "
            "block then records the opt-out, saying when the build staged, "
            "beside the build's own block. A build that did not stage is "
            "certified without one regardless."
        ),
    )
    parser.add_argument(
        "--staging-run-id",
        help=(
            "The certification's staging run id, never the build's own, since "
            "a staging run's files start afresh when it opens. Defaults to the "
            "build's staging run id with -certify appended."
        ),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        help=(
            "Local root for the certification's version 2 staging run files, "
            "which go to <dir>/runs/<run id>/. Defaults to <build_dir>/staging."
        ),
    )
    return parser


def _certify_progress(
    args: argparse.Namespace,
    *,
    record: _USBuildRecord,
    build: str,
    run_id: str,
    emitter: LocalTelemetryEmitter,
) -> _BuildProgress:
    """Certification's staging run, where its build staged.

    The post-export verdicts and the manifests reach the staging dashboard,
    as they did when a build certified itself (microcosm#1585 item 11, PR
    4a). Certification stages where its build did, to the build's repository,
    in a version 2 run of its own, since a staging run's files start afresh
    when it opens; the run names the build as its candidate and the build
    certification publishes as its release. A build that did not stage
    (``--no-staging``) is certified without staging. ``--no-staging``
    opts certification out where the build staged too, as a build may opt
    out: the publisher accepts a stated opt-out, and the reason says the
    build staged, so the manifest shows that its certification did not.
    ``build`` is the ID certification publishes the build under.
    """

    settings = record.staging_settings
    if settings["no_staging"] or args.no_staging:
        build_run = record.staging.get("run_id")
        reason = (
            "the build did not stage (--no-staging)"
            if settings["no_staging"]
            else "--no-staging at certification; the build staged"
            + (f" (run {build_run})" if build_run else "")
        )
        return _BuildProgress(
            run_id=run_id,
            emitter=emitter,
            staging_bundle=None,
            staging_opt_out_reason=reason,
        )
    staging_dir = args.staging_dir
    if staging_dir is None and not settings["repo_id"]:
        # A build that staged only locally is certified into the same layout.
        staging_dir = record.release_root / "staging"
    return _build_progress(
        SimpleNamespace(
            no_staging=False,
            staging_run_id=run_id,
            staging_dir=staging_dir,
            staging_repo_id=settings["repo_id"],
        ),
        release_root=record.release_root,
        release_id=build,
        run_id=run_id,
        emitter=emitter,
        epoch_every=CALIBRATION_EPOCH_EVERY,
        candidate_id=record.build,
        operation_id=US_CERTIFICATION_STAGING_OPERATION,
        run_kind="certification",
    )


def _refuse_reopening_the_builds_staging_run(
    args: argparse.Namespace, record: _USBuildRecord
) -> None:
    """Refuse a ``--staging-run-id`` that names the build or its staging run.

    Certification stages in a run of its own: a staging run's files start
    afresh when it opens, so a run reopened under the build's id would
    replace the build's progress and calibration curve.
    """

    build_run = record.staging.get("run_id")
    if args.staging_run_id is not None and args.staging_run_id in {
        build_run,
        record.build,
    }:
        raise SystemExit(
            f"--staging-run-id {args.staging_run_id} names build {record.build} "
            "or its staging run; certification stages in a run of its own, "
            "since a staging run's files start afresh when it opens. Choose "
            f"another, or leave it out for {build_run or record.build}-certify."
        )


def _certification_may_change(path: str) -> bool:
    """Whether a commit after a build's may change ``path`` and certify it.

    A UK cut is certified at its sign-off's head, where only gate-only
    registers changed (microcosm#1301, ``UK_GATE_ONLY_RESOURCES``). A US
    build may be certified at a descendant of its commit that changes only
    what certification never runs (microcosm#1585 item 11): tests and their
    helpers (``packages/*/tests/``, ``test_support/``, the root
    ``conftest.py``), docs (``docs/``, ``LICENSE``, and Markdown or
    reStructuredText outside the packages' sources), changelog fragments
    (``changelog.d/``), and the standing evidence owners register
    (:data:`US_EVIDENCE_FAILURE_OWNERS_REGISTER`), the US sign-off, which an
    evidence-tier certification reads at the certifying commit.
    """

    posix = PurePosixPath(path)
    parts = posix.parts
    if not parts:
        return False
    if path in {US_EVIDENCE_FAILURE_OWNERS_REGISTER, "conftest.py", "LICENSE"}:
        return True
    if parts[0] in {"docs", "changelog.d", "test_support"}:
        return True
    if len(parts) >= 3 and parts[0] == "packages" and parts[2] == "tests":
        return True
    in_package_source = len(parts) >= 3 and parts[0] == "packages" and parts[2] == "src"
    return posix.suffix.lower() in {".md", ".rst"} and not in_package_source


def _git_commit_exists(commit: str) -> bool:
    return (
        subprocess.run(
            (
                "git",
                "-C",
                str(_repository_root()),
                "cat-file",
                "-e",
                f"{commit}^{{commit}}",
            ),
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _certifying_commit(record: _USBuildRecord) -> tuple[str, list[str]]:
    """The commit certification runs at, and what changed since the build's.

    The build's own commit, or a descendant of it whose every change since
    is one :func:`_certification_may_change` allows (microcosm#1585 item 11).
    Anything else is refused, naming the paths, so the checks and the
    manifests stay the code the build was built with.
    """

    head = _git_output("rev-parse", "HEAD")
    if head == record.commit:
        return head, []
    if not _git_commit_exists(record.commit):
        raise SystemExit(
            f"Build {record.build} was built at {record.commit}, which this "
            f"checkout does not have; fetch it, so certification at {head} can "
            "check what changed since the build."
        )
    root = str(_repository_root())
    descends = (
        subprocess.run(
            ("git", "-C", root, "merge-base", "--is-ancestor", record.commit, head),
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )
    if not descends:
        raise SystemExit(
            f"Build {record.build} was built at {record.commit}; certify it "
            f"from a checkout of that commit or of a descendant of it, not "
            f"{head}, which does not descend from it."
        )
    changed = [
        path
        for path in subprocess.check_output(
            (
                "git",
                "-C",
                root,
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                record.commit,
                head,
            ),
            text=True,
        ).split("\0")
        if path
    ]
    refused = sorted(path for path in changed if not _certification_may_change(path))
    if refused:
        raise SystemExit(
            f"Build {record.build} was built at {record.commit}. {head} descends "
            "from it, but changes more than tests, docs, changelog fragments "
            f"and the evidence owners register ({US_EVIDENCE_FAILURE_OWNERS_REGISTER}), "
            "so it would certify the build with code it was not built with: "
            + ", ".join(refused)
            + ". Certify it from a checkout of its commit, or of one that "
            "changes only those."
        )
    return head, sorted(changed)


def _refuse_an_unowned_evidence_certification(
    failures: Sequence[str], owner_patterns: Sequence[tuple[str, str]]
) -> None:
    """Refuse an evidence-tier certification some failure would refuse anyway.

    Run on the build's recorded failures before certification scores
    anything, so a QRF tail register mismatch, which no owner may own, or an
    unowned failure is refused in seconds rather than after the scoring. The
    mismatch comes first: an owner would not let it ship.
    """

    refusal = _qrf_tail_register_evidence_refusal(failures)
    if refusal is not None:
        raise SystemExit(str(refusal))
    try:
        _evidence_known_failures(failures, owner_patterns)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error


class _Certification:
    """What certification chose and where it runs (microcosm#1585 item 11).

    The tier and its owner patterns; the release directory it places the
    build in and the ID it publishes the build under; and the commit it
    certifies at, with the paths changed since the build's. (A plain class,
    for the reason :class:`_RecordedCalibrationResult` gives.)
    """

    def __init__(
        self,
        *,
        evidence_release: bool,
        owner_patterns: Sequence[tuple[str, str]],
        owners_file: Path | None,
        owners_file_sha256: str | None,
        release_dir: Path,
        build: str,
        commit: str,
        changed_since_build: Sequence[str],
    ) -> None:
        self.evidence_release = bool(evidence_release)
        self.owner_patterns = tuple(owner_patterns)
        self.owners_file = owners_file
        self.owners_file_sha256 = owners_file_sha256
        self.release_dir = Path(release_dir)
        self.build = build
        self.commit = commit
        self.changed_since_build = tuple(changed_since_build)

    def manifest_block(self, record: _USBuildRecord) -> dict[str, object]:
        """Both manifests' ``certification`` block, beside the build's ``code``.

        The manifests' ``code`` is the build's commit, as the release
        contract reads it; this records the commit that certified it, what
        changed between the two, the tier, and the record certified.
        """

        block: dict[str, object] = {
            "tier": "evidence" if self.evidence_release else "certified",
            "code": {
                "repository": "PolicyEngine/microcosm",
                "git_commit": self.commit,
                "git_dirty": False,
            },
            "build_commit": record.commit,
            "changed_since_build": list(self.changed_since_build),
            "build_record": {
                "build": record.build,
                "sha256": record.payload["record_sha256"],
            },
        }
        if self.evidence_release:
            # The digests of the bytes that decided ownership, taken when they
            # were read: the standing register at import, the per-run file
            # before any scoring (review of #1681).
            block["evidence_failure_owners"] = {
                "standing_register": {
                    "path": US_EVIDENCE_FAILURE_OWNERS_REGISTER,
                    "sha256": US_EVIDENCE_FAILURE_OWNERS_SHA256,
                },
                "per_run_file_sha256": self.owners_file_sha256,
            }
        return block


def _place_build_for_certification(record: _USBuildRecord, release_dir: Path) -> None:
    """Copy the build's own files into ``release_dir``, its tier's release directory.

    A build has no tier, so its files are in its own directory
    (``layout.build_dir``, microcosm#1585 item 11). Certification copies each
    one the record pins there into the release directory of the tier it
    chose, checking the copy against the record's digest, and writes its
    own files beside them. The record is never edited, and the build's
    #568 weight evidence is never copied: the H5 carries the weights. A
    rerun after a failed certification first removes every file this
    certification would copy or write, so none of an earlier attempt's
    survives, and the post-export batch files of an attempt that was killed
    before its scorer could remove them. (Two certifications of one ``--out``
    at once are not supported: each removes the other's files.)
    """

    release_dir.mkdir(parents=True, exist_ok=True)
    for name in {
        # Part-written by an earlier attempt: a finished one is refused
        # before placement (_refuse_reusing_a_certified_build).
        "release_manifest.json",
        *US_CERTIFICATION_OUTPUT_FILES,
        *US_RELEASE_GATE_EVIDENCE_FILES.values(),
        *US_SCF_EDUCATION_DEBT_FIT_ARTIFACTS.values(),
        *US_FAILED_BUILD_WEIGHT_EVIDENCE,
        "calibration_diagnostics.json",
        "us_ssi_take_up.json",
    }:
        (release_dir / name).unlink(missing_ok=True)
    # Only scorers' own directories: anything else in the parent is kept.
    batch_parent = record.release_root / US_POST_EXPORT_BATCH_DIRECTORY
    if batch_parent.is_dir():
        for stale_batch_files in batch_parent.glob(
            f"{_POST_EXPORT_BATCH_DIRECTORY_PREFIX}*"
        ):
            shutil.rmtree(stale_batch_files)
    build_dir = record.build_dir.resolve()
    for role, pinned in record.payload["files"].items():
        source = (record.release_root / pinned["path"]).resolve()
        if source.parent != build_dir:
            # The H5 and the calibration weights stay in artifacts/, which
            # the publish command names as the artifact root.
            continue
        target = release_dir / source.name
        shutil.copyfile(source, target)
        copied = _sha256(target)
        if copied != pinned["sha256"]:
            raise RuntimeError(
                f"Build {record.build}: its {role} file copied to {target} hashes "
                f"to {copied}, not the {pinned['sha256']} the build recorded."
            )


def certify_main(argv: Sequence[str] | None = None) -> None:
    """Certify a finished build from its record (microcosm#1585 item 11).

    A build has no tier: certification chooses it. The certified tier, the
    default, places the build in the national release and refuses a build
    that recorded release-gate failures or holds the #568 weight evidence.
    ``--evidence-release`` places it in the evidence release under its
    evidence ID, and needs an owner for every failure, at least one failure,
    and a build that is not exact-k. Certification runs at the commit the
    build was built at, or at a descendant that changes only what
    certification never runs (:func:`_certifying_commit`), from a clean
    worktree (the build directory aside), with the packages the build was
    built with, so the checks and the manifests are the build's own code and
    score the H5 on the versions its manifests name. A build certified at
    either tier is refused: a certified build is never changed.
    Certification opens its own staging run where the build staged
    (:func:`_certify_progress`) and marks it failed if it does not finish.
    A SIGTERM is recorded as a termination, as in :func:`main`, and the
    command then exits 143, unless it arrives while a failing certification
    closes its staging run, as in :func:`main`.
    """

    with raise_on_sigterm():
        try:
            _certify_main_with_close_out(argv)
        except BuildTerminatedError as terminated:
            raise SystemExit(terminated.exit_code) from terminated


def _certify_main_with_close_out(argv: Sequence[str] | None) -> None:
    """Run the certification and record how it ended (see :func:`certify_main`)."""

    global _ACTIVE_EMITTER
    parser = _certify_parser()
    args = parser.parse_args(argv)
    if args.evidence_failure_owners is not None and not args.evidence_release:
        parser.error("--evidence-failure-owners requires --evidence-release.")
    record = _read_us_build_record(args.build_dir / US_BUILD_RECORD_FILENAME)
    _refuse_reopening_the_builds_staging_run(args, record)
    failures = _recorded_release_gate_failures(record.payload)
    owner_patterns: tuple[tuple[str, str], ...] = ()
    owners_file_sha256: str | None = None
    if record.artifact_integrity_failures:
        raise SystemExit(
            f"Build {record.build}: its written H5 failed its own checks, which "
            "no tier certifies, since the evidence tier relaxes gate verdicts, "
            "never the written artifact's (microcosm#506): "
            + "; ".join(record.artifact_integrity_failures)
            + ". Build it again."
        )
    if args.evidence_release:
        if record.exact_k():
            raise SystemExit(
                f"Build {record.build} is an --exact-k build, a build of the "
                f"national release under the certified contract; the evidence "
                "tier (microcosm#506) covers the national artifact only. "
                "Certify it without --evidence-release."
            )
        per_run: tuple[tuple[str, str], ...] = ()
        if args.evidence_failure_owners is not None:
            per_run, owners_file_sha256 = _read_evidence_failure_owners_file(
                args.evidence_failure_owners
            )
        owner_patterns = (*per_run, *US_EVIDENCE_FAILURE_OWNERS)
        _refuse_an_unowned_evidence_certification(failures, owner_patterns)
    else:
        if failures:
            raise SystemExit(
                f"Build {record.build} failed {len(failures)} release gate(s) "
                "when it was built, so it cannot be certified at the certified "
                "tier; with an owner for each failure, --evidence-release "
                "certifies it at the evidence tier: " + "; ".join(failures)
            )
        # Only a build whose release gates failed writes the #568 weight
        # evidence (a green build removes any it finds), so its presence
        # refuses the build at the certified tier whatever its record says.
        weight_evidence = [
            name
            for name in US_FAILED_BUILD_WEIGHT_EVIDENCE
            if (record.build_dir / name).exists()
        ]
        if weight_evidence:
            raise SystemExit(
                f"Build {record.build}: its directory holds the weight evidence "
                f"a build writes when its release gates fail "
                f"({', '.join(weight_evidence)}, microcosm#568), so it cannot be "
                "certified at the certified tier."
            )
    dirty = _git_dirty_paths(excluding=record.release_root)
    if dirty:
        raise SystemExit(
            "Refusing to certify a build from a dirty git worktree: "
            + "; ".join(dirty[:5])
            + ("; ..." if len(dirty) > 5 else "")
        )
    _refuse_packages_from_another_checkout("certify a build")
    head, changed_since_build = _certifying_commit(record)
    runtime = _runtime_versions()
    mismatched = sorted(
        name
        for name in record.runtime.keys() | runtime.keys()
        if record.runtime.get(name) != runtime.get(name)
    )
    if mismatched:
        raise SystemExit(
            f"Build {record.build} was built with "
            + ", ".join(f"{name} {record.runtime.get(name)}" for name in mismatched)
            + "; certify it with those packages, not "
            + ", ".join(f"{name} {runtime.get(name)}" for name in mismatched)
            + ", so its checks run on the versions its manifests name."
        )
    _refuse_reusing_a_certified_build(record.release_root)
    certification = _Certification(
        evidence_release=args.evidence_release,
        owner_patterns=owner_patterns,
        owners_file=args.evidence_failure_owners,
        owners_file_sha256=owners_file_sha256,
        release_dir=_certification_release_dir(
            record.release_root, evidence_release=args.evidence_release
        ),
        build=(
            _evidence_build_id(record.build) if args.evidence_release else record.build
        ),
        commit=head,
        changed_since_build=changed_since_build,
    )
    _place_build_for_certification(record, certification.release_dir)
    run_id = args.staging_run_id or (
        f"{record.staging.get('run_id') or record.build}-certify"
    )
    _ACTIVE_EMITTER = start_local_telemetry_emitter_service(
        run_id=run_id,
        country_code="US",
        pipeline="us_fiscal_refresh",
        candidate_id=record.build,
        release_id=certification.build,
        run_kind="certification",
    )
    telemetry: _BuildProgress | None = None
    try:
        telemetry = _certify_progress(
            args,
            record=record,
            build=certification.build,
            run_id=run_id,
            emitter=_ACTIVE_EMITTER,
        )
        _certify_us_build(record, certification=certification, telemetry=telemetry)
    except BaseException as error:
        try:
            if telemetry is not None:
                telemetry.fail(error)
            elif _ACTIVE_EMITTER.available:
                _fail_emitter(_ACTIVE_EMITTER, error)
        except Exception as telemetry_error:  # pragma: no cover - defensive
            # A failing failure-report must not replace the real traceback.
            print(
                "warning: could not record the certification's staging run as "
                f"failed: {type(telemetry_error).__name__}: {telemetry_error}",
                file=sys.stderr,
            )
        raise


def _certify_command(release_root: Path, *, evidence_release: bool = False) -> str:
    """The command that certifies the build in ``release_root``.

    The certified tier by default, placing the build in the national release;
    with ``evidence_release``, the evidence tier, placing it in the evidence
    release (microcosm#1585 item 11).
    """
    return shlex.join(
        [
            "uv",
            "run",
            "--no-sync",
            "python",
            "tools/certify_us_national_build.py",
            str(release_root),
            *(["--evidence-release"] if evidence_release else []),
        ]
    )


def _print_built_result(
    *,
    build: str,
    release_root: Path,
    build_dir: Path,
    artifact_root: Path,
    build_record: Path,
    release_gate_failures: Sequence[str] = (),
) -> None:
    """Print the finished build and the commands that certify it.

    A build that failed a release gate is printed too, so its H5 can be
    evaluated, with the failures that stop its certification at the
    certified tier.

    A build is not publishable until it is certified: certification runs the
    release's post-export checks on the written H5, places the build in the
    release of the tier it chooses and writes the manifests
    ``microcosm-publish-release-build`` requires (microcosm#1585 item 11). A
    build has no tier, so both commands are printed.
    """
    print(
        json.dumps(
            {
                "build": build,
                "certified": False,
                "certifiable_as_certified": not release_gate_failures,
                "release_gate_failures": list(release_gate_failures),
                "build_record": str(build_record),
                "build_dir": str(build_dir),
                "artifact_root": str(artifact_root),
                "certify_command": _certify_command(release_root),
                "evidence_certify_command": _certify_command(
                    release_root, evidence_release=True
                ),
                "next": (
                    "The build is written and checked; it is not publishable "
                    "until it is certified. certify_command certifies it for "
                    "the national release and refuses a build with release-gate "
                    "failures. evidence_certify_command certifies it for the "
                    "evidence release instead, once every failure has an "
                    "owner, in --evidence-failure-owners FILE or the standing "
                    "register."
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


#: This tool's scope of us/gates.json, evaluated in shadow beside today's
#: gates (microcosm#1585 item 11, PR 2 of 4).
US_NATIONAL_GATE_SCOPE = "us_national"
_ACTIVE_GATE_SHADOW: GateBatteryShadow | None = None

#: The waiver flags, each with the entries it waives today.
_NATIONAL_GATE_WAIVERS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "allow_input_mass_drift",
        "--allow-input-mass-drift",
        ("us_national_input_mass_reference", "us_national_export_input_mass"),
    ),
    (
        "allow_ecps_parity_gaps",
        "--allow-ecps-parity-gaps",
        ("us_national_ecps_parity",),
    ),
    (
        "allow_input_coverage_gaps",
        "--allow-input-coverage-gaps",
        ("us_national_release_input_coverage",),
    ),
    (
        "allow_esi_premium_gaps",
        "--allow-esi-premium-gaps",
        (
            "us_national_esi_anchor_feed",
            "us_national_esi_premiums",
            "us_national_esi_premiums_export",
            "us_national_esi_premiums_anchor",
        ),
    ),
    ("allow_unpinned_feed", "--allow-unpinned-feed", ("us_national_esi_anchor_feed",)),
    (
        "allow_qrf_tail_concentration",
        "--allow-qrf-tail-concentration",
        ("us_national_qrf_tail_concentration",),
    ),
)


def _national_gate_shadow_directory(args: argparse.Namespace) -> Path:
    """``<out>/gate_battery_shadow``: beside the build's outputs, never in a
    release directory, as for the other US tools. A build has no tier
    (microcosm#1585 item 11), so its shadow is not kept per release.

    A dry run writes no output directory, so its shadow goes beside its
    report instead.
    """

    if args.dry_run_gates_report is not None:
        report = Path(args.dry_run_gates_report).resolve()
        return report.parent / f"{report.stem}.{GATE_BATTERY_SHADOW_DIRECTORY}"
    return Path(args.out).resolve() / GATE_BATTERY_SHADOW_DIRECTORY


def _national_gate_waivers(args: argparse.Namespace) -> dict[str, list[str]]:
    """The entries each waiver flag of this run waives.

    A build has no tier (microcosm#1585 item 11): its terminal failures are
    recorded and refuse it as the certified tier's did, whatever tier its
    certification later chooses, so no tier waives an entry.
    """

    waivers: dict[str, list[str]] = {}
    for destination, flag, entry_ids in _NATIONAL_GATE_WAIVERS:
        if getattr(args, destination, False):
            for entry_id in entry_ids:
                waivers.setdefault(entry_id, []).append(flag)
    return waivers


def _start_national_gate_shadow(args: argparse.Namespace, *, release_id: str) -> None:
    """Arm this run's gate battery in shadow (finished in :func:`main`).

    The preflight gates run before ``_main`` refuses to reuse a certified
    release directory, so a build's shadow holds its files until
    :func:`_release_national_gate_shadow_files`: a refused rerun leaves the
    certified build's shadow files as they were. A dry run's files go beside
    its own report and are not held.
    """

    global _ACTIVE_GATE_SHADOW
    _finish_national_gate_shadow(stopped_at="a previous run in this process")
    if not tool_gate_shadows_enabled():
        return
    gates = us_scoped_gate_manifest(US_NATIONAL_GATE_SCOPE)
    report, comparison = shadow_file_paths(
        _national_gate_shadow_directory(args), US_NATIONAL_GATE_SCOPE
    )
    shadow = GateBatteryShadow(
        gates,
        scope=US_NATIONAL_GATE_SCOPE,
        release_id=release_id,
        report_path=report,
        comparison_path=comparison,
        registry=us_gate_registry(US_NATIONAL_GATE_SCOPE, tool=globals()),
        waivers=_national_gate_waivers(args),
        overrides=us_flag_overrides(gates, us_gate_specs(US_NATIONAL_GATE_SCOPE), args),
        hold_files=args.dry_run_gates_report is None,
    )
    arm_gate_shadow(shadow)
    _ACTIVE_GATE_SHADOW = shadow


def _release_national_gate_shadow_files() -> None:
    """Let the armed shadow write, once no certified release dir is reused."""

    if _ACTIVE_GATE_SHADOW is not None:
        _ACTIVE_GATE_SHADOW.release_files()


def _finish_national_gate_shadow(
    *,
    stopped_by: BaseException | None = None,
    stopped_at: str | None = None,
) -> None:
    """Finish the armed shadow, if any. It never fails the build: a finished
    build's differences are recorded and printed."""

    global _ACTIVE_GATE_SHADOW
    shadow, _ACTIVE_GATE_SHADOW = _ACTIVE_GATE_SHADOW, None
    if shadow is None:
        return
    disarm_gate_shadow(shadow)
    if stopped_by is not None:
        shadow.stop(stopped_by)
    elif stopped_at is not None:
        shadow.finish(stopped_at=stopped_at)
    else:
        shadow.finish()


def _shadow_frame_gate(
    entry_id: str, evaluate: Callable[[], GateResult], frame: Frame
) -> GateResult:
    """Today's frame signal gate, with the battery's entry beside it."""

    return shadow_gate(entry_id, evaluate, lambda: EvidenceContext(frame=frame))


def _shadow_exact_k_puf_tail_support_gate(
    frame: Frame, support: np.ndarray
) -> GateResult:
    """The exact-k arm's tail gate, with the battery's entry beside it."""

    return shadow_gate(
        "us_national_exact_k_puf_capital_gains_tail",
        lambda: _exact_k_puf_tail_support_gate(frame, support),
        lambda: EvidenceContext(frame=frame, artifacts={"exact_k_support": support}),
    )


def _shadow_tail_survival(
    entry_id: str, candidate: Frame, selected: Frame
) -> dict[str, object]:
    """Today's PUF tail survival check, observed: its refusal is its failure."""

    return shadow_value(
        entry_id,
        lambda: assert_puf_capital_gains_tail_survives_selection(
            candidate,
            selected,
            require_present=True,
        ),
        lambda: EvidenceContext(
            frame=selected,
            artifacts={
                "tail_candidate_frame": candidate,
                "tail_require_present": True,
            },
        ),
        verdict=lambda receipt: LegacyVerdict.from_failures(()),
        refusals=(ValueError,),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the build and mark its staging run failed if it does not finish.

    An uncaught build error previously left the dashboard status at ``running``
    forever. SIGKILL remains outside the reach of an in-process handler; this
    closes the ordinary exception and termination half of the gap. A SIGTERM
    raises ``BuildTerminatedError`` (``microcosm.build.termination``), which
    closes the staging run as failed and records a termination with the
    emitter, and the program then exits 143. One that arrives once the build
    is already closing its staging run after a failure, or during the final
    commit check, ends only that step (``_BuildProgress.fail``), and the
    program keeps the failure's exit status.

    A build returns ``None``. ``--dry-run-gates-report`` leaves through
    ``SystemExit`` with the report's exit code: 1 on any certain failure, 2 on
    AT-RISK only, 0 clean.

    The gate battery's shadow (microcosm#1585 item 11) finishes after the
    build's own handler, and never changes the program's exit: today's gates
    decide until the flip, so a verdict difference is recorded in the
    shadow's comparison file and printed as a warning. By then a finished
    build's telemetry has completed its staging run and its emitter, which
    closes on that outcome.
    """

    with raise_on_sigterm():
        try:
            _main_with_close_out(argv)
        except BuildTerminatedError as terminated:
            raise SystemExit(terminated.exit_code) from terminated


def _main_with_close_out(argv: Sequence[str] | None) -> None:
    """Run :func:`_main` and record how it ended (see :func:`main`)."""

    global _ACTIVE_DRY_RUN
    _ACTIVE_DRY_RUN = None
    _finish_national_gate_shadow(stopped_at="a previous run in this process")
    try:
        dry_run_exit = _main(argv)
    except BaseException as error:
        _finish_national_gate_shadow(stopped_by=error)
        dry_run = _ACTIVE_DRY_RUN
        if dry_run is not None and _is_release_refusal(error):
            # A dry run the release refuses before its stop point: the
            # refusal is the report's certain failure (exit 1), not a crash.
            if _ACTIVE_EMITTER is not None and _ACTIVE_EMITTER.available:
                _ACTIVE_EMITTER.fail(
                    error,
                    failure_class="dry_run_refusal",
                )
            raise SystemExit(dry_run.refused(error)) from error
        if _ACTIVE_PROGRESS is not None:
            try:
                _ACTIVE_PROGRESS.fail(error)
            except Exception as telemetry_error:  # pragma: no cover - defensive
                # A failing failure-report must not replace the real traceback.
                print(
                    "warning: could not record the staging run as failed: "
                    f"{type(telemetry_error).__name__}: {telemetry_error}",
                    file=sys.stderr,
                )
        if _ACTIVE_EMITTER is not None and _ACTIVE_EMITTER.available:
            _fail_emitter(_ACTIVE_EMITTER, error)
        raise
    try:
        _finish_national_gate_shadow(
            stopped_at=(
                "dry run: the release stops before target materialization"
                if dry_run_exit is not None
                else None
            )
        )
    except BaseException as error:
        # Only a fault in the wiring reaches here: the shadow reports a
        # difference without raising. A finished release's telemetry has
        # closed the emitter by now (see the docstring), so this records an
        # outcome only on an emitter that something left open.
        if _ACTIVE_EMITTER is not None and _ACTIVE_EMITTER.available:
            _fail_emitter(_ACTIVE_EMITTER, error)
        raise
    if dry_run_exit is not None:
        if _ACTIVE_EMITTER is not None and _ACTIVE_EMITTER.available:
            if dry_run_exit == 0:
                _ACTIVE_EMITTER.complete()
            else:
                _ACTIVE_EMITTER.fail(
                    RuntimeError(f"US dry run returned exit status {dry_run_exit}.")
                )
        raise SystemExit(dry_run_exit)
    if _ACTIVE_EMITTER is not None and _ACTIVE_EMITTER.available:
        _ACTIVE_EMITTER.complete()


def _without_opt_in_student_loan_targets(registry: TargetRegistry) -> TargetRegistry:
    """The registry the target-parity contract reads: no FSA portfolio rows.

    The parity manifest describes one target surface for every build, but the
    FSA family compiles only with ``--scf-education-debt`` (microcosm#1074):
    the rows bind columns a default build does not carry. Until that stage is
    default the family is outside the contract, and the manifest may declare
    it a reviewed exclusion without either build failing the other.
    """

    return TargetRegistry(
        tuple(spec for spec in registry.specs if spec.family != FSA_TARGET_FAMILY),
        country=registry.country,
    )


def _check_committed_us_ledger_feed_pin(
    facts_sha256: str,
    *,
    manifest_sha256: str | None,
    allow_unpinned_feed: bool,
    pin: USChronicleFeed | None = None,
) -> None:
    """Hold the loaded Chronicle feed to the committed US pin.

    ``us/chronicle_feed.json`` names the feed the release's target registry
    compiles from; ``--ledger-facts-sha256`` only says what the operator
    expected. Without this check a build could compile any feed that matched
    the operator's flags and record it as the release's target identity while
    the pin, the parity manifests and the release rule name another. The
    facts digest always has to match; the manifest digest only when the pin
    describes an artifact rather than a bare feed. ``--allow-unpinned-feed``
    is for a reviewed diagnostic run and never for a release.
    """

    pin = pin or load_us_chronicle_feed()
    comparisons = [("facts", facts_sha256, pin.facts_sha256)]
    if not pin.is_bare_feed:
        comparisons.append(("manifest", manifest_sha256, pin.manifest_sha256))
    mismatches = [
        f"{label}: loaded {loaded}, committed {committed}"
        for label, loaded, committed in comparisons
        if loaded != committed
    ]
    if mismatches and not allow_unpinned_feed:
        raise SystemExit(
            "error: Chronicle feed differs from the committed US pin "
            "(us/chronicle_feed.json): "
            + "; ".join(mismatches)
            + "; pass --allow-unpinned-feed only for an explicitly reviewed "
            "diagnostic run"
        )


def _with_scf_education_debt_stage(
    base_frame: Frame,
    *,
    args: argparse.Namespace,
    telemetry: Any,
    fit_path: Path | None,
) -> Frame:
    """Run the opt-in SCF 2025 education-debt stage and its signal gate."""

    if telemetry is not None:
        telemetry.stage(
            "scf_education_debt_inputs",
            message=(
                "Imputing federal student loan balances, Parent PLUS balances, "
                "IBR new-borrower status and graduate shares from the 2025 SCF."
            ),
        )
    summary_path = (
        Path(args.scf_2025_summary_extract)
        if args.scf_2025_summary_extract is not None
        else fetch_scf_public_file(SCF_2025_SUMMARY_EXTRACT)
    )
    full_path = (
        Path(args.scf_2025_full_extract)
        if args.scf_2025_full_extract is not None
        else fetch_scf_public_file(SCF_2025_FULL_EXTRACT)
    )
    donor = load_scf_2025_education_debt_donor(summary_path, full_path)
    base_frame = with_us_scf_education_debt_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        scf_education_debt_donor=donor,
    )
    gate = _shadow_frame_gate(
        "us_national_scf_education_debt",
        lambda: us_scf_education_debt_signal_gate(base_frame),
        base_frame,
    )
    if not gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "scf_education_debt_gate",
                status="failed",
                message="SCF education-debt signal gate failed.",
                failures=list(gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SCF education-debt signal failed: {failure}"
                for failure in gate.failures
            )
        )
    if fit_path is not None:
        # Design-weight fit of the staged frame against the SCF; the release
        # writes the same table at its calibrated weights after the solve
        # (US_SCF_EDUCATION_DEBT_CALIBRATED_FIT_FILENAME, microcosm#1074).
        _write_us_scf_education_debt_fit(
            fit_path, base_frame, weights="pre_calibration_household_weights"
        )
    return base_frame


def _write_us_scf_education_debt_fit(path: Path, frame: Frame, *, weights: str) -> None:
    """Write the microcosm#1073 fit table of ``frame`` at its household weights.

    ``weights`` names those weights in the payload: the staged frame's
    pre-calibration weights, or the release's calibrated weights.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "weights": weights,
                "fit": us_scf_education_debt_fit_table(frame),
            },
            indent=2,
            sort_keys=True,
            default=float,
        )
        + "\n",
        encoding="utf-8",
    )


def _check_us_ledger_feed_row_contracts(
    fact_schema_versions: Sequence[str],
    *,
    facts_sha256: str,
    pin: USChronicleFeed | None = None,
) -> None:
    """Hold the row contracts the loaded feed's rows carry to the US pin.

    :func:`_check_committed_us_ledger_feed_pin` compares digests; the US pin
    describes a bare feed, so the full feed-pin check cannot run. Rows of the
    committed feed (``facts_sha256`` is the pin's) must carry exactly the pin's
    ``consumer_fact_schema_versions``, which a re-pin types by hand. Rows of a
    feed admitted by ``--allow-unpinned-feed`` must carry declared row
    contracts only. Either way a row contract
    ``microcosm.build.chronicle_epoch`` does not declare is refused.
    """

    require_us_chronicle_feed_row_contracts(
        fact_schema_versions,
        facts_sha256=facts_sha256,
        pin=pin or load_us_chronicle_feed(),
    )


def _check_us_esi_anchor_against_feed(
    facts: Sequence[Mapping[str, Any]],
    *,
    allow_unpinned_feed: bool,
    allow_esi_premium_gaps: bool,
) -> list[str]:
    """Hold the ESI stage's pinned anchor to the feed's banked B4923C facts.

    microcosm#454 / d1317: the ``meps_esi_premiums`` stage scales to BEA
    B4923C values it pins itself, and the pinned feed banks the same series.
    On the committed pin a difference means the stage drifted from the feed,
    so the build stops. A reviewed diagnostic run on another feed
    (``--allow-unpinned-feed``) or with the ESI gates waived
    (``--allow-esi-premium-gaps``) continues; the differences are returned
    either way and written to ``esi_premiums.json``.
    """

    mismatches = shadow_failures(
        "us_national_esi_anchor_feed",
        lambda: us_esi_premiums_anchor_feed_mismatches(facts),
        lambda: EvidenceContext(artifacts={"ledger_facts": facts}),
    )
    if mismatches and not (allow_unpinned_feed or allow_esi_premium_gaps):
        raise SystemExit(
            "error: the meps_esi_premiums anchor differs from the Chronicle "
            "feed: "
            + "; ".join(mismatches)
            + "; pass --allow-unpinned-feed or --allow-esi-premium-gaps only "
            "for an explicitly reviewed diagnostic run"
        )
    return mismatches


def _check_us_feed_carries_fsa_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    scf_education_debt: bool,
) -> FsaTargetPlan | None:
    """Refuse a ``--scf-education-debt`` build on a feed with no FSA facts.

    microcosm#1074: the flag's stage imputes the federal student loan balances
    and Federal Student Aid's portfolio calibrates them. On a feed with no
    ``ed_fsa`` facts the plan is empty, so the compile would bind no FSA
    target and the build would ship the balances uncalibrated, raising
    nothing. The build stops here instead, before any stage runs. Without the
    flag the facts are inert and nothing is planned. Returns the plan, which
    the ``us_source_coverage.json`` receipt reuses.
    """

    if not scf_education_debt:
        return None
    plan = plan_fsa_student_loan_targets(facts, target_period=PERIOD)
    if plan.status == "no_fsa_facts":
        raise SystemExit(
            "error: --scf-education-debt calibrates the stage's federal student "
            "loan balances to Federal Student Aid's portfolio, but the "
            f"Chronicle feed carries no {FSA_SOURCE_NAME} facts (the "
            f"{', '.join(FSA_BOUND_TABLES)} tables), so no FSA target would "
            "compile (microcosm#1074). Build on a feed with the six "
            "packages/ed_fsa packages of PolicyEngine/chronicle#347, as the "
            "US pin carries from microcosm#1506, or drop the flag."
        )
    return plan


# Every column the five donor tables built from the full SIPP 2023 file use.
_SIPP_2023_DONOR_SOURCE_COLUMNS = tuple(
    sorted(
        {
            *SIPP_FINANCIAL_ASSET_SOURCE_COLUMNS,
            *SIPP_SSI_DISABILITY_SOURCE_COLUMNS,
            *SIPP_HEAD_START_SOURCE_COLUMNS,
            *SIPP_VEHICLE_SOURCE_COLUMNS,
            *SIPP_VOLUNTARY_FILING_SOURCE_COLUMNS,
        }
    )
)


def _main(argv: Sequence[str] | None = None) -> int | None:
    global _ACTIVE_EMITTER, _ACTIVE_PROGRESS
    _ACTIVE_EMITTER = None
    _ACTIVE_PROGRESS = None
    args = _parse_args(argv)
    build_started = time.perf_counter()
    attempt_started_at = datetime.now(UTC)
    run_id = _telemetry_run_id(args, timestamp=attempt_started_at)
    is_dry_run = args.dry_run_gates_report is not None
    _ACTIVE_EMITTER = start_local_telemetry_emitter_service(
        run_id=run_id,
        country_code="US",
        pipeline="us_fiscal_refresh",
        candidate_id=args.release_id,
        release_id=args.release_id if not is_dry_run else None,
        run_kind="dry_run" if is_dry_run else "release",
    )
    _ACTIVE_EMITTER.transition_stage(
        "preflight",
        message="Validating release inputs and configuration.",
    )
    # --dry-run-gates-report: runs this build up to target materialization and
    # returns the report's exit code from the stop point below (_ReleaseDryRun).
    dry_run = _ReleaseDryRun.start(args, argv)
    if dry_run is None and _git_dirty():
        raise SystemExit("Refusing to build a release from a dirty git worktree.")
    if dry_run is None:
        _refuse_packages_from_another_checkout("build a release")
    # A build has no tier (microcosm#1585 item 11): only a dry run takes
    # --evidence-release, to preview an evidence certification's owners
    # (_parse_args refuses it on a build). Loaded up front so a malformed
    # owners file dies in seconds.
    evidence_failure_owner_patterns = (
        _load_evidence_failure_owner_patterns(args.evidence_failure_owners)
        if args.evidence_release
        else ()
    )
    timing: dict[str, float] = {}

    # microcosm#568 round 4: refuse certified-dir reuse before ANY side
    # effect, including the base download and cache writes below. A build is
    # certified into the release of the tier its certification chose, so
    # both release directories are checked, whether or not --release-id
    # gives the build ID.
    _refuse_reusing_a_certified_build(args.out.resolve())
    if dry_run is None:
        # A build stops at its record (microcosm#1585 item 11, PR 4a), so an
        # --out awaiting certification carries no release manifest yet.
        _refuse_overwriting_a_build_awaiting_certification(args.out.resolve())
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
            # A denied pool with its identity stripped lands here; its bytes
            # are still its identity.
            refuse_denied_pool_h5_digest(
                base_dataset_sha256,
                consumer="US fiscal refresh release builder --base-h5 (generic path)",
            )
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
    _assert_us_release_id(release_id)
    if args.exact_k is not None:
        _assert_exact_k_release_id(release_id, args.exact_k)
    _start_national_gate_shadow(args, release_id=release_id)
    # Every build, exact-k builds included, writes its dataset and calibration
    # weights under the names its release's new builds declare
    # (microcosm#1570); the build ID carries the exact count.
    dataset_key = DATASET_KEY
    dataset_filename = DATASET_FILENAME
    calibration_key = CALIBRATION_KEY
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
    validation_input_coverage_gate = shadow_gate(
        "us_national_validation_input_coverage",
        lambda: us_validation_input_coverage_gate(),
        EvidenceContext(),
    )
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
    # The check resolves the probes' list edits on a reformed PolicyEngine-US
    # system that the module caches for the process. Nothing in the build reads
    # it again (the reform-coverage smoke, its other reader, runs at
    # certification, in its own process), so it is not kept through the source
    # stages and the calibration.
    clear_installed_list_parameter_baseline_cache()
    # Preflight (microcosm#377): no column may be required-to-signal by one
    # register (seeded/count-calibrated take-up, health nonconstant, coverage
    # 'required') and excused as absent/degenerate by another (degenerate
    # reviewed exclusions, coverage reviewed exclusions, parity known gaps,
    # documented-absent). Such a pincer aborts every build AFTER the expensive
    # source stages — the stale TANF exclusion did exactly that on Build G and
    # Build I. Fail it here, in seconds.
    register_consistency_gate = shadow_gate(
        "us_national_register_consistency",
        lambda: us_register_consistency_gate(
            degenerate_reviewed_exclusions=US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS,
            documented_absent_inputs=US_DOCUMENTED_ABSENT_INPUTS,
            nonconstant_required_columns=US_HEALTH_INPUT_NONCONSTANT_COLUMNS,
        ),
        lambda: EvidenceContext(
            artifacts={
                "degenerate_reviewed_exclusions": (
                    US_DEGENERATE_INPUT_REVIEWED_EXCLUSIONS
                ),
                "documented_absent_inputs": US_DOCUMENTED_ABSENT_INPUTS,
                "nonconstant_required_columns": US_HEALTH_INPUT_NONCONSTANT_COLUMNS,
            }
        ),
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
    _check_committed_us_ledger_feed_pin(
        ledger_artifact.facts_sha256,
        manifest_sha256=ledger_artifact.manifest_sha256,
        allow_unpinned_feed=args.allow_unpinned_feed,
    )
    _check_us_ledger_feed_row_contracts(
        ledger_artifact.fact_schema_versions,
        facts_sha256=ledger_artifact.facts_sha256,
    )
    esi_anchor_feed_mismatches = _check_us_esi_anchor_against_feed(
        ledger_artifact.facts,
        allow_unpinned_feed=args.allow_unpinned_feed,
        allow_esi_premium_gaps=args.allow_esi_premium_gaps,
    )
    if dry_run is not None:
        dry_run.esi_anchor_feed_mismatches = list(esi_anchor_feed_mismatches)
    # A --scf-education-debt build on a feed with no FSA facts would compile
    # no FSA target; refuse it before the compile (microcosm#1074).
    fsa_target_plan = _check_us_feed_carries_fsa_facts(
        ledger_artifact.facts, scf_education_debt=args.scf_education_debt
    )
    target_registry = compile_us_fiscal_target_registry(
        ledger_artifact.facts,
        target_period=PERIOD,
        congressional_district_vintage_crosswalk=(
            congressional_district_vintage_crosswalk
        ),
        age_targets=args.age_targets,
        allow_unaged_dollar_targets=args.allow_unaged_dollar_targets,
        # FSA's portfolio binds the stage's balances, so only a build that
        # imputes them compiles the targets (microcosm#1074).
        student_loan_targets=args.scf_education_debt,
    )
    # Every FSA fact's disposition (bound, allocated, out of scope,
    # diagnostic) and the in-scope controls, for us_source_coverage.json.
    fsa_target_receipt = (
        fsa_student_loan_target_receipt(
            ledger_artifact.facts, target_period=PERIOD, plan=fsa_target_plan
        )
        if fsa_target_plan is not None
        else None
    )
    # The exclusion register's keys do not say which feed facts a compile
    # dropped or let through by vintage (microcosm#956), so record the concrete
    # ids per rule now, while the facts and crosswalk are at hand; it lands in
    # us_source_coverage.json beside the register.
    fiscal_target_exclusion_receipt = us_fiscal_target_exclusion_receipt(
        ledger_artifact.facts,
        target_period=PERIOD,
        congressional_district_vintage_crosswalk=(
            congressional_district_vintage_crosswalk
        ),
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
    # The FSA portfolio family is opt-in (it binds only the
    # --scf-education-debt stage's balances, microcosm#1074), so it stays off
    # the one-surface parity contract until that stage is default.
    parity_registry = _without_opt_in_student_loan_targets(target_registry)
    assert_target_parity_manifest_current(registry=parity_registry)
    target_parity_gate = shadow_gate(
        "us_national_target_parity",
        lambda: us_release_target_parity_gate(parity_registry),
        lambda: EvidenceContext(artifacts={"target_parity_registry": parity_registry}),
    )
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
    target_profile_gate = shadow_gate(
        "us_national_target_profile_coverage",
        lambda: target_profile_coverage_gate(
            target_specs,
            US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
        ),
        lambda: EvidenceContext(
            artifacts={
                "target_specs": target_specs,
                "target_coverage_requirements": US_FISCAL_TARGET_COVERAGE_REQUIREMENTS,
            }
        ),
    )
    if not target_profile_gate.passed:
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Target profile coverage failed: {failure}"
                for failure in target_profile_gate.failures
            )
        )
    # The parity and profile-coverage gates above ran on the full compiled
    # surface; --target-surface only narrows what is calibrated.
    target_specs, target_surface_receipt = _select_target_surface(
        target_specs, args.target_surface
    )
    target_surface_selection = (
        None if args.target_surface == TARGET_SURFACE_FULL else target_surface_receipt
    )
    active_target_registry = TargetRegistry(target_specs, country="us")
    release_root = args.out.resolve()
    artifact_root = release_root / "artifacts"
    # The build's own directory (microcosm#1585 item 11): a build has no tier,
    # so it writes no release directory; certification places these files in
    # the release of the tier it chooses. The name stays release_dir below,
    # where every build file is written.
    release_dir = release_root / US_BUILD_FILES_DIRECTORY
    # Unconditional refusal BEFORE any output-directory creation: a hostile
    # --checkpoint-root beneath releases/<release> must not mutate a certified
    # directory before the raise (microcosm#568 round 4).
    _refuse_reusing_a_certified_build(release_root)
    _release_national_gate_shadow_files()
    checkpoint_root, target_materialization_cache_dir, target_frame_checkpoint_path = (
        _resolve_checkpoint_paths(args, artifact_root=artifact_root)
    )
    if dry_run is None:
        if checkpoint_root is not None:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        release_dir.mkdir(parents=True, exist_ok=True)
    # A dry run writes only its report under --out and creates no staging
    # artifacts. Hosted telemetry identifies it separately from a release.
    if dry_run is None:
        telemetry = _build_progress(
            args,
            release_root=release_root,
            release_id=release_id,
            run_id=run_id,
            emitter=_ACTIVE_EMITTER,
        )
    else:
        telemetry = _BuildProgress(
            run_id=run_id,
            emitter=_ACTIVE_EMITTER,
            staging_bundle=None,
            staging_opt_out_reason="dry run",
            epoch_every=_calibration_epoch_every(args),
        )
        _ACTIVE_PROGRESS = telemetry
    if telemetry is not None:
        telemetry.stage(
            "target_registry",
            message="Compiled fiscal target registry.",
            n_targets=len(target_specs),
            target_profile_gate_passed=target_profile_gate.passed,
        )

    if telemetry is not None:
        telemetry.stage("load_base_frame", message="Loading base population H5.")
    if pool_frame is None:
        base_frame = _load_frame(base_h5, expected_sha256=base_dataset_sha256)
    else:
        base_frame = pool_frame
    capital_gains_tail_presence = _shadow_tail_survival(
        "us_national_capital_gains_tail_presence", base_frame, base_frame
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
    weeks_unemployed_gate = _shadow_frame_gate(
        "us_national_weeks_unemployed",
        lambda: us_weeks_unemployed_signal_gate(base_frame),
        base_frame,
    )
    if not weeks_unemployed_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "weeks_unemployed_input_gate",
                status="failed",
                message="Weeks-unemployed input signal gate failed.",
                failures=list(weeks_unemployed_gate.failures),
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
    if selection_source is None:
        for entry_id in (
            "us_national_capital_gains_tail_retention",
            "us_national_weeks_unemployed_post_selection",
        ):
            shadow_not_run(entry_id, "no frozen-support selection source was given")
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
            _shadow_tail_survival(
                "us_national_capital_gains_tail_retention",
                selection_candidate_frame,
                base_frame,
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
        post_selection_weeks_unemployed_gate = _shadow_frame_gate(
            "us_national_weeks_unemployed_post_selection",
            lambda: us_weeks_unemployed_signal_gate(base_frame),
            base_frame,
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
    base_population_gate = shadow_gate(
        "us_national_base_population_scale",
        lambda: _base_population_scale_gate(
            base_frame,
            mass_repair=base_population_repair,
        ),
        lambda: EvidenceContext(
            frame=base_frame,
            artifacts={"base_population_mass_repair": base_population_repair},
        ),
    )
    if not base_population_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "base_population_gate",
                status="failed",
                message="Base population scale gate failed.",
                failures=list(base_population_gate.failures),
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
    qbi_inputs_gate = _shadow_frame_gate(
        "us_national_qbi_inputs",
        lambda: us_qbi_inputs_signal_gate(base_frame),
        base_frame,
    )
    if not qbi_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "qbi_input_gate",
                status="failed",
                message="QBI-input signal gate failed.",
                failures=list(qbi_inputs_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "QBI-input signal failed: " + failure
                for failure in qbi_inputs_gate.failures
            )
        )
    farm_business_income_gate = _shadow_frame_gate(
        "us_national_farm_business_income",
        lambda: us_farm_business_income_signal_gate(base_frame),
        base_frame,
    )
    if not farm_business_income_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "farm_business_income_gate",
                status="failed",
                message="Farm-business-income signal gate failed.",
                failures=list(farm_business_income_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Farm-business-income signal failed: " + failure
                for failure in farm_business_income_gate.failures
            )
        )
    domestic_production_ald_gate = _shadow_frame_gate(
        "us_national_domestic_production_ald",
        lambda: us_domestic_production_ald_signal_gate(base_frame),
        base_frame,
    )
    if not domestic_production_ald_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "domestic_production_ald_gate",
                status="failed",
                message="Domestic-production-ALD signal gate failed.",
                failures=list(domestic_production_ald_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Domestic-production-ALD signal failed: " + failure
                for failure in domestic_production_ald_gate.failures
            )
        )
    child_support_gate = _shadow_frame_gate(
        "us_national_child_support",
        lambda: us_child_support_signal_gate(base_frame),
        base_frame,
    )
    if not child_support_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "child_support_input_gate",
                status="failed",
                message="Child-support input signal gate failed.",
                failures=list(child_support_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Child-support input signal failed: " + failure
                for failure in child_support_gate.failures
            )
        )
    disability_benefits_gate = _shadow_frame_gate(
        "us_national_disability_benefits",
        lambda: us_disability_benefits_signal_gate(base_frame),
        base_frame,
    )
    if not disability_benefits_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "disability_benefits_input_gate",
                status="failed",
                message="Disability-benefits input signal gate failed.",
                failures=list(disability_benefits_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Disability-benefits input signal failed: " + failure
                for failure in disability_benefits_gate.failures
            )
        )
    workers_compensation_gate = _shadow_frame_gate(
        "us_national_workers_compensation",
        lambda: us_workers_compensation_signal_gate(base_frame),
        base_frame,
    )
    if not workers_compensation_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "workers_compensation_input_gate",
                status="failed",
                message="Workers-compensation input signal gate failed.",
                failures=list(workers_compensation_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Workers-compensation input signal failed: " + failure
                for failure in workers_compensation_gate.failures
            )
        )
    educator_expense_gate = _shadow_frame_gate(
        "us_national_educator_expense",
        lambda: us_educator_expense_signal_gate(base_frame),
        base_frame,
    )
    if not educator_expense_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "educator_expense_input_gate",
                status="failed",
                message="Educator-expense input signal gate failed.",
                failures=list(educator_expense_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Educator-expense input signal failed: " + failure
                for failure in educator_expense_gate.failures
            )
        )
    form_4952_election_gate = _shadow_frame_gate(
        "us_national_form_4952_election",
        lambda: us_form_4952_election_signal_gate(base_frame),
        base_frame,
    )
    if not form_4952_election_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "form_4952_election_input_gate",
                status="failed",
                message="Form 4952 election input signal gate failed.",
                failures=list(form_4952_election_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Form 4952 election signal failed: " + failure
                for failure in form_4952_election_gate.failures
            )
        )
    salt_refund_income_gate = _shadow_frame_gate(
        "us_national_salt_refund_income",
        lambda: us_salt_refund_income_signal_gate(base_frame),
        base_frame,
    )
    if not salt_refund_income_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "salt_refund_income_input_gate",
                status="failed",
                message="SALT-refund-income signal gate failed.",
                failures=list(salt_refund_income_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "SALT-refund-income signal failed: " + failure
                for failure in salt_refund_income_gate.failures
            )
        )
    capital_gain_details_gate = _shadow_frame_gate(
        "us_national_capital_gain_details",
        lambda: us_capital_gain_details_signal_gate(base_frame),
        base_frame,
    )
    if not capital_gain_details_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "capital_gain_details_input_gate",
                status="failed",
                message="Capital-gain details input signal gate failed.",
                failures=list(capital_gain_details_gate.failures),
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
    childcare_gate = _shadow_frame_gate(
        "us_national_childcare",
        lambda: us_childcare_signal_gate(base_frame),
        base_frame,
    )
    if not childcare_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "childcare_input_gate",
                status="failed",
                message="Childcare-input signal gate failed.",
                failures=list(childcare_gate.failures),
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
    energy_subsidy_gate = _shadow_frame_gate(
        "us_national_energy_subsidy",
        lambda: us_energy_subsidy_signal_gate(base_frame),
        base_frame,
    )
    if not energy_subsidy_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "energy_subsidy_gate",
                status="failed",
                message="Energy-subsidy signal gate failed.",
                failures=list(energy_subsidy_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Energy-subsidy signal failed: " + failure
                for failure in energy_subsidy_gate.failures
            )
        )
    alimony_gate = _shadow_frame_gate(
        "us_national_alimony", lambda: us_alimony_signal_gate(base_frame), base_frame
    )
    if not alimony_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "alimony_input_gate",
                status="failed",
                message="Alimony-input signal gate failed.",
                failures=list(alimony_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Alimony-input signal failed: " + failure
                for failure in alimony_gate.failures
            )
        )
    casualty_loss_gate = _shadow_frame_gate(
        "us_national_casualty_loss",
        lambda: us_casualty_loss_signal_gate(base_frame),
        base_frame,
    )
    if not casualty_loss_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "casualty_loss_input_gate",
                status="failed",
                message="Casualty-loss signal gate failed.",
                failures=list(casualty_loss_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Casualty-loss signal failed: " + failure
                for failure in casualty_loss_gate.failures
            )
        )
    misc_itemized_gate = _shadow_frame_gate(
        "us_national_misc_itemized",
        lambda: us_misc_itemized_signal_gate(base_frame),
        base_frame,
    )
    if not misc_itemized_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "misc_itemized_input_gate",
                status="failed",
                message="Miscellaneous-itemized signal gate failed.",
                failures=list(misc_itemized_gate.failures),
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
    retirement_contributions_gate = _shadow_frame_gate(
        "us_national_retirement_contributions",
        lambda: us_retirement_contributions_signal_gate(base_frame),
        base_frame,
    )
    if not retirement_contributions_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "retirement_contribution_inputs_gate",
                status="failed",
                message="Retirement-contribution signal gate failed.",
                failures=list(retirement_contributions_gate.failures),
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
    immigration_gate = _shadow_frame_gate(
        "us_national_immigration_composition",
        lambda: us_immigration_composition_gate(base_frame),
        base_frame,
    )
    if not immigration_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "immigration_gate",
                status="failed",
                message="Immigration composition gate failed.",
                failures=list(immigration_gate.failures),
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
    take_up_gate = _shadow_frame_gate(
        "us_national_take_up", lambda: us_take_up_signal_gate(base_frame), base_frame
    )
    if not take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "take_up_gate",
                status="failed",
                message="Take-up signal gate failed.",
                failures=list(take_up_gate.failures),
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
    hours_worked_gate = _shadow_frame_gate(
        "us_national_hours_worked",
        lambda: us_hours_worked_signal_gate(base_frame),
        base_frame,
    )
    if not hours_worked_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "hours_worked_gate",
                status="failed",
                message="Hours-worked signal gate failed.",
                failures=list(hours_worked_gate.failures),
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
    snap_take_up_gate = _shadow_frame_gate(
        "us_national_snap_take_up",
        lambda: us_snap_take_up_signal_gate(base_frame),
        base_frame,
    )
    if not snap_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "snap_take_up_gate",
                status="failed",
                message="SNAP take-up signal gate failed.",
                failures=list(snap_take_up_gate.failures),
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
    relationship_inputs_gate = _shadow_frame_gate(
        "us_national_relationship_inputs",
        lambda: us_relationship_inputs_signal_gate(base_frame),
        base_frame,
    )
    if not relationship_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "relationship_inputs_gate",
                status="failed",
                message="Relationship-input signal gate failed.",
                failures=list(relationship_inputs_gate.failures),
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
            "spm_independence_role",
            message=(
                "Restoring the measured SPM independence role from the pinned "
                "Census ASEC person files."
            ),
        )
    if pool_frame is None:
        base_frame = with_us_spm_independence_role(
            base_frame,
            seed=args.seed,
            time_period=PERIOD,
        )
    spm_independence_role_gate = _shadow_frame_gate(
        "us_national_spm_independence_role",
        lambda: us_spm_independence_role_signal_gate(base_frame),
        base_frame,
    )
    if not spm_independence_role_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "spm_independence_role_gate",
                status="failed",
                message="SPM independence role signal gate failed.",
                failures=list(spm_independence_role_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SPM independence role signal failed: {failure}"
                for failure in spm_independence_role_gate.failures
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
    medicare_take_up_gate = _shadow_frame_gate(
        "us_national_medicare_take_up",
        lambda: us_medicare_take_up_signal_gate(base_frame),
        base_frame,
    )
    if not medicare_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "medicare_take_up_input_gate",
                status="failed",
                message="Medicare take-up input signal gate failed.",
                failures=list(medicare_take_up_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Medicare take-up input signal failed: {failure}"
                for failure in medicare_take_up_gate.failures
            )
        )
    prior_year_income_gate = _shadow_frame_gate(
        "us_national_prior_year_income",
        lambda: us_prior_year_income_signal_gate(base_frame),
        base_frame,
    )
    if not prior_year_income_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "prior_year_income_gate",
                status="failed",
                message="Prior-year-income signal gate failed.",
                failures=list(prior_year_income_gate.failures),
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
    housing_inputs_gate = _shadow_frame_gate(
        "us_national_housing_inputs",
        lambda: us_housing_inputs_signal_gate(base_frame),
        base_frame,
    )
    if not housing_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "housing_inputs_gate",
                status="failed",
                message="Housing/tenure input signal gate failed.",
                failures=list(housing_inputs_gate.failures),
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
    retirement_distributions_gate = _shadow_frame_gate(
        "us_national_retirement_distributions",
        lambda: us_retirement_distributions_signal_gate(base_frame),
        base_frame,
    )
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
    eligibility_inputs_gate = _shadow_frame_gate(
        "us_national_eligibility_inputs",
        lambda: us_eligibility_inputs_signal_gate(base_frame),
        base_frame,
    )
    if not eligibility_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "eligibility_inputs_gate",
                status="failed",
                message="Eligibility-inputs signal gate failed.",
                failures=list(eligibility_inputs_gate.failures),
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
    education_inputs_gate = _shadow_frame_gate(
        "us_national_education_inputs",
        lambda: us_education_inputs_signal_gate(base_frame),
        base_frame,
    )
    if not education_inputs_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "education_inputs_gate",
                status="failed",
                message="Education-input signal gate failed.",
                failures=list(education_inputs_gate.failures),
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
    pregnancy_gate = _shadow_frame_gate(
        "us_national_pregnancy",
        lambda: us_pregnancy_signal_gate(base_frame),
        base_frame,
    )
    if not pregnancy_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "pregnancy_gate",
                status="failed",
                message="Pregnancy signal gate failed.",
                failures=list(pregnancy_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"Pregnancy signal failed: {failure}"
                for failure in pregnancy_gate.failures
            )
        )
    reported_coverage_vintage_gate = _shadow_frame_gate(
        "us_national_reported_coverage_vintage",
        lambda: us_reported_coverage_vintage_signal_gate(base_frame),
        base_frame,
    )
    if not reported_coverage_vintage_gate.passed:
        receipt_path = (
            _write_reported_coverage_vintage_gate_receipt(
                release_dir, reported_coverage_vintage_gate
            )
            if dry_run is None
            else "not written by a dry run"
        )
        if telemetry is not None:
            telemetry.stage(
                "reported_coverage_vintage_gate",
                status="failed",
                message="Reported-coverage vintage signal gate failed.",
                failures=list(reported_coverage_vintage_gate.failures),
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
    wic_claim_gate = _shadow_frame_gate(
        "us_national_wic_claim",
        lambda: us_wic_claim_signal_gate(base_frame),
        base_frame,
    )
    if not wic_claim_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "wic_claim_input_gate",
                status="failed",
                message="WIC-claim input signal gate failed.",
                failures=list(wic_claim_gate.failures),
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
    snap_discretionary_exemption_gate = _shadow_frame_gate(
        "us_national_snap_discretionary_exemption",
        lambda: us_snap_discretionary_exemption_signal_gate(base_frame),
        base_frame,
    )
    if not snap_discretionary_exemption_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "snap_discretionary_exemption_gate",
                status="failed",
                message="SNAP discretionary-exemption signal gate failed.",
                failures=list(snap_discretionary_exemption_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SNAP discretionary-exemption signal failed: {failure}"
                for failure in snap_discretionary_exemption_gate.failures
            )
        )
    acs_predictor_join_result = join_acs_release_predictors(
        base_frame,
        person_zip=args.acs_person_zip,
        person_sha256=args.acs_person_sha256,
        household_zip=args.acs_household_zip,
        household_sha256=args.acs_household_sha256,
    )
    base_frame = acs_predictor_join_result.frame
    acs_predictor_join_receipt = dict(acs_predictor_join_result.receipt)
    # Only the receipt is read later. The result's frame would otherwise keep
    # the post-join frame alive after the next stage rebinds base_frame.
    del acs_predictor_join_result
    if telemetry is not None:
        telemetry.stage(
            "acs_predictor_join",
            message=(
                "Authenticated and joined ACS release predictors before all "
                "six archived donor-model stages."
            ),
            enabled=bool(acs_predictor_join_receipt.get("enabled")),
            join=dict(acs_predictor_join_receipt.get("join", {})),
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
    # share one immutable full SIPP artifact. Resolve it once and parse it once;
    # each stage builds its donor table from these December rows, which stay in
    # memory until the last of the five stages. The CLI option's historical name
    # remains stable for existing release invocations.
    sipp_full_donor_path = (
        Path(args.sipp_vehicle_donor)
        if args.sipp_vehicle_donor is not None
        else fetch_sipp_2023_financial_asset_donor()
    )
    # Each donor table below is loaded or built in its stage's call, so no
    # local keeps it once the stage returns.
    sipp_december_person_rows = read_sipp_december_person_rows(
        sipp_full_donor_path,
        columns=_SIPP_2023_DONOR_SOURCE_COLUMNS,
        expected_sha256=SIPP_2023_FINANCIAL_ASSET_DONOR_SHA256,
        expected_size_bytes=SIPP_2023_FINANCIAL_ASSET_DONOR_SIZE_BYTES,
    )
    base_frame = with_us_scf_wealth_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        scf_donor=load_scf_2022_financial_asset_donor(scf_summary_extract_path),
        sipp_donor=build_sipp_2023_financial_asset_donor(sipp_december_person_rows),
    )
    scf_wealth_gate = shadow_gate(
        "us_national_scf_wealth",
        lambda: us_scf_wealth_signal_gate(
            base_frame,
            require_sipp_blend=True,
        ),
        lambda: EvidenceContext(
            frame=base_frame, artifacts={"scf_require_sipp_blend": True}
        ),
    )
    if not scf_wealth_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "scf_wealth_gate",
                status="failed",
                message="SCF-wealth signal gate failed.",
                failures=list(scf_wealth_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SCF-wealth signal failed: {failure}"
                for failure in scf_wealth_gate.failures
            )
        )
    scf_wealth_signal = _scf_wealth_gate_record(
        scf_wealth_gate, base_frame.table("household")["household_id"]
    )
    if telemetry is not None:
        telemetry.stage(
            "ssi_disability_criteria",
            message=(
                "Imputing SSI-specific disability criteria from the "
                "sha-pinned full SIPP 2023 donor after SCF assets."
            ),
        )
    base_frame = with_us_ssi_disability_criteria(
        base_frame,
        # The retired weighted bootstrap and MicroImpute forest are fixed.
        seed=42,
        time_period=PERIOD,
        sipp_donor=build_sipp_2023_ssi_disability_donor(
            sipp_december_person_rows,
            time_period=PERIOD,
        ),
    )
    ssi_disability_gate = _shadow_frame_gate(
        "us_national_ssi_disability_criteria",
        lambda: us_ssi_disability_criteria_signal_gate(base_frame),
        base_frame,
    )
    if not ssi_disability_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "ssi_disability_criteria_gate",
                status="failed",
                message="SSI disability-criteria signal gate failed.",
                failures=list(ssi_disability_gate.failures),
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
    base_frame = with_us_sipp_head_start_input(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        sipp_donor=build_sipp_2023_head_start_donor(sipp_december_person_rows),
    )
    head_start_gate = _shadow_frame_gate(
        "us_national_sipp_head_start",
        lambda: us_sipp_head_start_signal_gate(base_frame),
        base_frame,
    )
    if not head_start_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "sipp_head_start_gate",
                status="failed",
                message="SIPP Head Start signal gate failed.",
                failures=list(head_start_gate.failures),
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
    ssi_take_up_gate = shadow_gate(
        "us_national_ssi_take_up",
        lambda: us_ssi_take_up_gate(
            ssi_take_up_stage_diagnostics, targets=ssi_band_targets
        ),
        lambda: EvidenceContext(
            artifacts={
                "ssi_take_up_diagnostics": ssi_take_up_stage_diagnostics,
                "ssi_band_targets": ssi_band_targets,
            }
        ),
    )
    if not ssi_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "ssi_take_up_gate",
                status="failed",
                message="SSI take-up assignment gate failed.",
                failures=list(ssi_take_up_gate.failures),
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
    base_frame = with_us_scf_auto_loan_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        scf_auto_loan_donor=load_scf_2022_auto_loan_donor(
            scf_summary_extract_path,
            scf_full_extract_path,
        ),
    )
    scf_auto_loan_gate = _shadow_frame_gate(
        "us_national_scf_auto_loans",
        lambda: us_scf_auto_loans_signal_gate(base_frame),
        base_frame,
    )
    if not scf_auto_loan_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "scf_auto_loan_gate",
                status="failed",
                message="SCF auto-loan signal gate failed.",
                failures=list(scf_auto_loan_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                f"SCF auto-loan signal failed: {failure}"
                for failure in scf_auto_loan_gate.failures
            )
        )
    if args.scf_education_debt:
        # microcosm#1073: federal student loan balances from the 2025 SCF. The
        # pinned engine does not define these inputs yet, so the stage is
        # opt-in and its columns ride in the stored-input register until it
        # does (microcosm.data.stored_inputs).
        base_frame = _with_scf_education_debt_stage(
            base_frame,
            args=args,
            telemetry=telemetry,
            fit_path=(
                release_dir / US_SCF_EDUCATION_DEBT_FIT_FILENAME
                if dry_run is None
                else None
            ),
        )
    else:
        shadow_not_run(
            "us_national_scf_education_debt",
            "the SCF education-debt stage runs only with --scf-education-debt",
        )
    if telemetry is not None:
        telemetry.stage(
            "sipp_vehicle_inputs",
            message=(
                "Imputing household vehicle count and value from the "
                "sha-pinned full SIPP 2023 donor."
            ),
        )
    base_frame = with_us_sipp_vehicle_inputs(
        base_frame,
        # The retired MicroImpute model pins both forests to seed 42.
        seed=42,
        time_period=PERIOD,
        sipp_donor=build_sipp_2023_vehicle_donor(sipp_december_person_rows),
    )
    sipp_vehicles_gate = _shadow_frame_gate(
        "us_national_sipp_vehicles",
        lambda: us_sipp_vehicles_signal_gate(base_frame),
        base_frame,
    )
    if not sipp_vehicles_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "sipp_vehicles_gate",
                status="failed",
                message="SIPP-vehicle signal gate failed.",
                failures=list(sipp_vehicles_gate.failures),
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
    base_frame = with_us_voluntary_filing_input(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        sipp_donor=build_sipp_2023_voluntary_filing_donor(sipp_december_person_rows),
    )
    del sipp_december_person_rows
    voluntary_filing_gate = _shadow_frame_gate(
        "us_national_voluntary_filing",
        lambda: us_voluntary_filing_signal_gate(base_frame),
        base_frame,
    )
    if not voluntary_filing_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "voluntary_filing_gate",
                status="failed",
                message="Voluntary-filing signal gate failed.",
                failures=list(voluntary_filing_gate.failures),
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
    base_frame = with_us_sipp_tip_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        sipp_donor=load_sipp_2023_tip_donor(
            sipp_tip_donor_path,
            expected_sha256=SIPP_2023_TIP_DONOR_SHA256,
        ),
    )
    sipp_tips_gate = _shadow_frame_gate(
        "us_national_sipp_tips",
        lambda: us_sipp_tips_signal_gate(base_frame),
        base_frame,
    )
    if not sipp_tips_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "sipp_tips_gate",
                status="failed",
                message="SIPP-tip signal gate failed.",
                failures=list(sipp_tips_gate.failures),
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
    base_frame = with_us_org_wages_inputs(
        base_frame,
        seed=args.seed,
        time_period=PERIOD,
        org_donor=load_org_2024_donor(
            org_wages_donor_path,
            expected_content_sha256=ORG_2024_DONOR_CONTENT_SHA256,
        ),
    )
    org_wages_gate = _shadow_frame_gate(
        "us_national_org_wages",
        lambda: us_org_wages_signal_gate(base_frame),
        base_frame,
    )
    if not org_wages_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "org_wages_gate",
                status="failed",
                message="CPS ORG/FLSA signal gate failed.",
                failures=list(org_wages_gate.failures),
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
    health_input_gate = _shadow_frame_gate(
        "us_national_health_input",
        lambda: _health_input_signal_gate(base_frame),
        base_frame,
    )
    if not health_input_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "health_input_gate",
                status="failed",
                message="Health input signal gate failed.",
                failures=list(health_input_gate.failures),
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
    medicaid_take_up_gate = shadow_gate(
        "us_national_medicaid_take_up",
        lambda: us_medicaid_take_up_gate(medicaid_take_up_diagnostics),
        lambda: EvidenceContext(
            artifacts={"medicaid_take_up_diagnostics": medicaid_take_up_diagnostics}
        ),
    )
    if not medicaid_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "medicaid_take_up_gate",
                status="failed",
                message="Medicaid take-up gate failed.",
                failures=list(medicaid_take_up_gate.failures),
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
    snap_state_take_up_gate = shadow_gate(
        "us_national_snap_state_take_up",
        lambda: us_snap_state_take_up_gate(snap_state_take_up_diagnostics),
        lambda: EvidenceContext(
            artifacts={"snap_state_take_up_diagnostics": snap_state_take_up_diagnostics}
        ),
    )
    if not snap_state_take_up_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "snap_state_take_up_gate",
                status="failed",
                message="SNAP state take-up gate failed.",
                failures=list(snap_state_take_up_gate.failures),
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
    other_health_insurance_gate = _shadow_frame_gate(
        "us_national_other_health_insurance",
        lambda: us_other_health_insurance_signal_gate(base_frame),
        base_frame,
    )
    if not other_health_insurance_gate.passed:
        if telemetry is not None:
            telemetry.stage(
                "other_health_insurance_input_gate",
                status="failed",
                message="Other-health-insurance premium signal gate failed.",
                failures=list(other_health_insurance_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "Other-health-insurance premium signal failed: " + failure
                for failure in other_health_insurance_gate.failures
            )
        )
    # microcosm#454: the meps_esi_premiums base stage owns the ESI employer
    # premium input; refuse a base that lacks or flattens it before the solve.
    esi_premiums_base_gate = _shadow_frame_gate(
        "us_national_esi_premiums",
        lambda: us_esi_premiums_signal_gate(base_frame),
        base_frame,
    )
    if not esi_premiums_base_gate.passed and not args.allow_esi_premium_gaps:
        if telemetry is not None:
            telemetry.stage(
                "esi_premiums_input_gate",
                status="failed",
                message="ESI premium signal gate failed on the base.",
                failures=list(esi_premiums_base_gate.failures),
            )
        raise RuntimeError(
            "Release gates failed: "
            + "; ".join(
                "ESI premium signal failed: " + failure
                for failure in esi_premiums_base_gate.failures
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
        if early_terminal_gate_failures or dry_run is not None:
            try:
                if telemetry is not None:
                    telemetry.stage(
                        "input_mass_reference_gate",
                        status="failed",
                        message="Base-frame input mass parity gate failed.",
                        failures=list(input_mass_reference_gate.failures),
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
                )
            raise RuntimeError(
                "Release gates failed: "
                + "; ".join(
                    f"Input mass parity failed: {failure}"
                    for failure in input_mass_reference_gate.failures
                )
            )
    degenerate_input_engine = PolicyEngineUSEngine()
    degenerate_input_gate = shadow_gate(
        "us_national_degenerate_input",
        lambda engine=degenerate_input_engine: _degenerate_input_signal_gate(
            base_frame, engine
        ),
        lambda engine=degenerate_input_engine: EvidenceContext(
            frame=base_frame, artifacts={"policyengine_engine": engine}
        ),
    )
    # Today's call made the engine a temporary; it lives no longer here.
    del degenerate_input_engine
    if not degenerate_input_gate.passed:
        # Same degraded contract as the input-mass gate above: the gate
        # object rides _release_gate_failures to the batch; no raise, no
        # duplicate append, guarded telemetry.
        if early_terminal_gate_failures or dry_run is not None:
            try:
                if telemetry is not None:
                    telemetry.stage(
                        "degenerate_input_gate",
                        status="failed",
                        message="Degenerate input signal gate failed.",
                        failures=list(degenerate_input_gate.failures),
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
    ecps_parity_gate = _shadow_frame_gate(
        "us_national_ecps_parity", lambda: _ecps_parity_gate(base_frame), base_frame
    )
    if not ecps_parity_gate.passed and not args.allow_ecps_parity_gaps:
        # Same degraded contract as the input-mass/degenerate gates above
        # (PR #557 round 3 finding 1): the pinned parity reference requires
        # the retirement leaves, so a missing-leaf frame fails HERE too and
        # an unconditional raise would supersede the retirement diagnosis
        # before the solve. The gate object rides _release_gate_failures
        # (as enforced_ecps_parity_gate) into the terminal batch; degraded
        # runs continue, green runs keep the fail-fast raise.
        if early_terminal_gate_failures or dry_run is not None:
            try:
                if telemetry is not None:
                    telemetry.stage(
                        "ecps_parity_gate",
                        status="failed",
                        message="eCPS parity gate failed.",
                        failures=list(ecps_parity_gate.failures),
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
                )
            raise RuntimeError(
                "Release gates failed: "
                + "; ".join(
                    f"eCPS parity failed: {failure}"
                    for failure in ecps_parity_gate.failures
                )
            )
    # SPM measurement composition, hours before the engine would say so.
    #
    # Advisory, never a raise: the L0/refit export path *selects* a subset of
    # this pool (``attach_l0_refit_entity_weights`` calls ``base_frame.select``),
    # so a pool unit with no classified adult may simply not ship, and a hard
    # refusal here would reject a run that would have succeeded. The exact,
    # blocking answer is the batched pre-export gate taken on the export frame
    # below. What this buys the operator is the pool-level number now rather
    # than after the solve: a pool that carries the defect needs the SPM
    # independence role whether or not today's selection happens to dodge it.
    # Advisory means advisory: a pool that cannot be classified at all (no age
    # column, no spm_unit table) must not abort the build from here. The export
    # frame below is the graded point, and it fails with the same diagnosis.
    try:
        base_frame_spm_composition = _spm_composition_report(base_frame)
    except (KeyError, ValueError) as error:
        base_frame_spm_composition = None
        print(
            "BASE-POOL SPM COMPOSITION: not classifiable on the base frame "
            f"({error}); the export frame below is still graded."
        )
        if telemetry is not None:
            telemetry.stage(
                "base_frame_spm_composition",
                status="skipped",
                message="Base pool carries no readable SPM composition inputs.",
                error=str(error),
            )
    if base_frame_spm_composition is not None:
        if base_frame_spm_composition.status != "PASS":
            print(
                "\n"
                + "!" * 72
                + f"\nBASE-POOL SPM COMPOSITION: {base_frame_spm_composition.summary}\n"
                + "\n".join(base_frame_spm_composition.failures)
                + "\nThis is advisory here — the export frame below is graded. "
                "Preflight it next time: tools/preflight_us_release_gates.py\n"
                + "!"
                * 72
            )
        if telemetry is not None:
            telemetry.stage(
                "base_frame_spm_composition",
                message="Classified the base pool's SPM measurement composition.",
                status=base_frame_spm_composition.status,
                **{
                    key: value
                    for key, value in base_frame_spm_composition.details.items()
                    if key
                    in (
                        "n_units",
                        "n_units_without_classified_adult",
                        "n_units_without_member_aged_18_or_over",
                        "role_source",
                    )
                },
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
    # microcosm#956: digest the very frame handed to the materializer below,
    # so a checkpoint hit proves the same staged inputs, not only the same base.
    staged_frame_sha256 = _staged_frame_sha256(base_frame)
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
        staged_frame_sha256=staged_frame_sha256,
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
    if dry_run is not None:
        # The dry run's stop point. Everything above ran as the release runs
        # it, so base_frame is the staged frame the release would calibrate
        # and export. Grade it here instead of materializing targets.
        return dry_run.finish(
            base_frame=base_frame,
            target_specs=target_specs,
            early_terminal_gate_failures=early_terminal_gate_failures,
            pre_solve_gates={
                "target_profile_gate": target_profile_gate,
                "base_population_gate": base_population_gate,
                "health_input_gate": health_input_gate,
                "immigration_gate": immigration_gate,
                "hours_worked_gate": hours_worked_gate,
                "snap_take_up_gate": snap_take_up_gate,
                "eligibility_inputs_gate": eligibility_inputs_gate,
                "pregnancy_gate": pregnancy_gate,
                "reported_coverage_vintage_gate": reported_coverage_vintage_gate,
                "snap_discretionary_exemption_gate": (
                    snap_discretionary_exemption_gate
                ),
            },
            input_mass_reference_gate=input_mass_reference_gate,
            degenerate_input_gate=degenerate_input_gate,
            ecps_parity_gate=ecps_parity_gate,
            evidence_owner_patterns=evidence_failure_owner_patterns,
            binding={
                "build_commit": full_commit,
                "release_id": release_id,
                "policyengine_us_version": policyengine_us_version,
                "base_h5": {"path": str(base_h5), "sha256": base_dataset_sha256},
                "base_pool": base_pool_receipt,
                "staged_frame_sha256": staged_frame_sha256,
                "target_frame_materializer_identity_sha256": (
                    target_frame_materializer_identity_sha256
                ),
                "target_frame_checkpoint": _dry_run_checkpoint_status(
                    target_frame_checkpoint_path,
                    target_frame_materializer_identity_sha256,
                ),
            },
        )
    target_frame, registry, compilation = _load_or_materialize_target_frame(
        base_frame,
        target_specs,
        target_frame_checkpoint_path=target_frame_checkpoint_path,
        target_frame_checkpoint_identity=target_frame_checkpoint_identity,
        target_frame_checkpoint_build_commit=full_commit,
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
    full_pool_calibration = _full_pool_calibration(args, candidate_households)
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
    # Route A PR-3: the solve's thread geometry, read as it starts. It rides
    # calibration_diagnostics.json (every run, gate-failed ones included) and
    # both manifests.
    calibration_runtime = _calibration_runtime()
    ladder_outcome = None
    exact_k_puf_tail_gate: GateResult | None = None
    if args.exact_k is None:
        for entry_id in (
            "us_national_exact_k_puf_capital_gains_tail",
            "us_national_exact_k_frozen_register_fit",
        ):
            shadow_not_run(entry_id, "the exact-k gates run on the exact-k arm only")
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
        # A helper, not a lambda over target_frame: the frame is deleted
        # below to free it, and a closure must not keep it alive.
        exact_k_puf_tail_gate = _shadow_exact_k_puf_tail_support_gate(
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
    # A calibrated frame built from target_frame by Frame.with_weights (the
    # dense result, an L0 selection) shares the target tables' memory with it,
    # so the tables are freed only once both references are gone: drop the
    # input now, and replace those result frames after building the clean
    # export from base_frame and the calibrated weights below.
    del target_frame
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
    result = _without_calibrated_frames(
        result,
        export_frame=export_frame if ladder_outcome is not None else None,
    )
    # The solves are done. Later code reads only each constraint matrix's
    # shape and nonzero count (the target surface in the calibration
    # diagnostics), so keep those two numbers and free the matrices: the
    # refit's and, under L0 selection, the selection's over the full frame.
    result = without_constraint_matrix(result)
    if isinstance(ladder_outcome, ExactKLadderCalibration):
        ladder_outcome = dataclasses.replace(ladder_outcome, result=result)
    if args.scf_education_debt:
        # microcosm#1074: the stage's fit table at the weights the release
        # ships, before any scoring or terminal gate, so a build that binds
        # the FSA family reports the borrower-household share at those weights. It
        # follows the export and cleanup statements, which
        # test_us_post_export_scoring runs on their own.
        _write_us_scf_education_debt_fit(
            release_dir / US_SCF_EDUCATION_DEBT_CALIBRATED_FIT_FILENAME,
            export_frame,
            weights="calibrated_household_weights",
        )
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
    final_ssi_take_up_gate = shadow_gate(
        "us_national_ssi_take_up_final",
        lambda: us_ssi_take_up_gate(ssi_take_up_diagnostics, targets=ssi_band_targets),
        lambda: EvidenceContext(
            artifacts={
                "ssi_take_up_diagnostics": ssi_take_up_diagnostics,
                "ssi_band_targets": ssi_band_targets,
            }
        ),
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
            # The dense arm (--dense-default-dataset) fences its adult bands
            # per the microcosm#566/#567 fence adjudication; the L0+refit and
            # --exact-k arms pass no fences and keep hard enforcement.
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
        health_input_gate = _shadow_frame_gate(
            "us_national_health_input_export",
            lambda: _health_input_signal_gate(export_frame),
            export_frame,
        )
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
        reported_coverage_vintage_gate = _shadow_frame_gate(
            "us_national_reported_coverage_vintage_export",
            lambda: us_reported_coverage_vintage_signal_gate(export_frame),
            export_frame,
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
        other_health_insurance_gate = _shadow_frame_gate(
            "us_national_other_health_insurance_export",
            lambda: us_other_health_insurance_signal_gate(export_frame),
            export_frame,
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
    try:
        esi_premiums_export_gate = _shadow_frame_gate(
            "us_national_esi_premiums_export",
            lambda: us_esi_premiums_signal_gate(export_frame),
            export_frame,
        )
    except Exception as error:
        if not early_terminal_gate_failures:
            raise
        esi_premiums_export_gate = None
        early_terminal_gate_failures.append(
            "ESI premium signal evaluation crashed in degraded mode; recorded "
            f"instead of masking earlier failures: {error}"
        )
    if (
        esi_premiums_export_gate is not None
        and not esi_premiums_export_gate.passed
        and not args.allow_esi_premium_gaps
    ):
        early_terminal_gate_failures.extend(
            f"ESI premium signal failed on the export frame: {failure}"
            for failure in esi_premiums_export_gate.failures
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
            current_target_surface = target_surface_payload(result)
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
            current_target_surface = target_surface_payload(result)
        exact_k_incumbent_fit_gate = shadow_gate(
            "us_national_exact_k_frozen_register_fit",
            lambda: _exact_k_frozen_register_fit_gate(
                result,
                incumbent_diagnostics,
                target_registry=registry,
                target_loss_weights=target_loss_weights,
                configured_loss_basis=target_loss_basis,
                incumbent_loss_basis=incumbent_loss_basis,
            ),
            lambda: EvidenceContext(
                artifacts={
                    "calibration_result": result,
                    "incumbent_diagnostics": incumbent_diagnostics,
                    "target_registry": registry,
                    "target_loss_weights": target_loss_weights,
                    "configured_loss_basis": target_loss_basis,
                    "incumbent_loss_basis": incumbent_loss_basis,
                }
            ),
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
    # The summary the manifests' calibration gate block carries, recorded for
    # certification (microcosm#1585 item 11, PR 4a); None when it crashed.
    release_gate_summary: list[str] | None = None
    try:
        gate_failures = shadow_failures(
            "us_national_release_gates",
            # Today's call, argument for argument: callers read its
            # positional arguments (the #547 corridor tests read args[8]).
            lambda: _release_gate_failures(
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
            ),
            lambda: EvidenceContext(
                artifacts={
                    "calibration_result": result,
                    "target_compilation": compilation,
                    **{
                        f"release_gates_{name}": value
                        for name, value in {
                            "target_profile_gate": target_profile_gate,
                            "health_input_gate": health_input_gate,
                            "base_population_gate": base_population_gate,
                            "incumbent_diagnostics": incumbent_diagnostics,
                            "immigration_gate": immigration_gate,
                            "input_mass_reference_gate": (
                                enforced_input_mass_reference_gate
                            ),
                            "degenerate_input_gate": degenerate_input_gate,
                            "ecps_parity_gate": enforced_ecps_parity_gate,
                            "hours_worked_gate": hours_worked_gate,
                            "snap_take_up_gate": snap_take_up_gate,
                            "eligibility_inputs_gate": eligibility_inputs_gate,
                            "pregnancy_gate": pregnancy_gate,
                            "reported_coverage_vintage_gate": (
                                reported_coverage_vintage_gate
                            ),
                            "snap_discretionary_exemption_gate": (
                                snap_discretionary_exemption_gate
                            ),
                            "target_registry": registry,
                        }.items()
                    },
                }
            ),
        )
        release_gate_summary = list(gate_failures)
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
    # microcosm#956: how the written H5 will be scored. The baseline plans come
    # from engine-free dry runs of the post-export consumers, so they are
    # fixed here and recorded in the calibration diagnostics; the post-export
    # stages then score exactly these plans. A plan that cannot be built is
    # recorded, not raised, and joins the terminal batch below.
    post_export_scoring_plan = _record_post_export_scoring_plan(
        args,
        n_households=int(export_frame.n("household")),
        result=result,
        release_id=release_id,
    )
    if current_target_surface is None:
        current_target_surface = target_surface_payload(result)
    diagnostics_outcome = _write_release_calibration_diagnostics(
        result=result,
        target_surface=current_target_surface,
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
        scf_wealth_signal=scf_wealth_signal,
        target_loss_family_multipliers=args.target_family_loss_multipliers,
        target_loss_basis=target_loss_basis,
        exact_k_ladder=exact_k_ladder_provenance,
        calibration_runtime=calibration_runtime,
        post_export_scoring=post_export_scoring_plan.record(),
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
    # An unbuildable post-export plan refuses here, before the export write,
    # with every other terminal group still evaluated (microcosm#956).
    terminal_gate_failures.extend(post_export_scoring_plan.terminal_failures())
    terminal_batch_telemetry = _TerminalBatchProgress(
        telemetry,
        terminal_gate_failures,
    )
    if diagnostics_outcome.status == "available":
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
        )

    # SPM measurement composition, as a batched pre-export gate.
    #
    # The export frame exists here — it is the calibrated/refit weights attached
    # to the base support, built before any of this batch's gates — so the
    # cheapest of them (pure pandas; the #893 lane journal measured the check at
    # 0.05 s on the 907k-person phase-2 base pool) runs first and its failure
    # joins the same list. That keeps BOTH properties the batch
    # exists for: the run refuses by name rather than through the engine's
    # anonymous population-wide SPM_COMPOSITION_REQUIRED, and it refuses with
    # every other failing pre-export gate on record, before the H5 write and
    # before the NPZ write, instead of after them.
    #
    # It rides the batch like every other pre-export gate, which also means
    # an evidence-tier certification can ship it as an owned known failure
    # (microcosm#506): the build records it with the others and writes its
    # H5. Such an export still cannot be SPM-measured, so reform validation in
    # certification reaches the engine's own refusal; that is the declared,
    # owner-checked, manifest-recorded choice an evidence release makes about
    # any pre-export gate, not a hole here.
    spm_composition_failures, export_frame_spm_composition = shadow_value(
        "us_national_spm_composition",
        lambda: _spm_composition_gate_failures(export_frame, stage="export frame"),
        lambda: EvidenceContext(
            frame=export_frame, artifacts={"spm_composition_stage": "export frame"}
        ),
        verdict=lambda value: LegacyVerdict.from_failures(value[0]),
    )
    terminal_gate_failures.extend(spm_composition_failures)
    terminal_batch_telemetry.stage(
        "export_frame_spm_composition",
        message="Classified the export frame's SPM measurement composition.",
        **{
            key: value
            for key, value in export_frame_spm_composition.items()
            if key
            in (
                "evaluated",
                "n_units",
                "n_units_without_classified_adult",
                "n_units_without_member_aged_18_or_over",
                "role_source",
            )
        },
    )
    if spm_composition_failures:
        terminal_batch_telemetry.stage(
            "export_frame_spm_composition",
            status="failed",
            message="SPM measurement composition gate failed.",
            failures=spm_composition_failures,
        )

    # Stored model inputs (microcosm#1026, decision d271), as a batched
    # pre-export gate. The engine ignores a stored column it defines no
    # variable for, so a renamed input (would_claim_wic after policyengine-us
    # 1.777.0) ships as data the model never reads. The installed engine here is
    # the one this run writes the H5 with and records as
    # build.built_with_model_package. Column names only: no rows are read, and
    # the verdict joins the one batched pre-export raise below, so a refusal
    # names every other failing gate too and mints no H5.
    stored_input_failures_, export_frame_stored_inputs = shadow_value(
        "us_national_stored_inputs",
        lambda: _stored_input_gate_failures(
            export_frame, stage=US_FISCAL_STORED_INPUTS_STAGE
        ),
        lambda: EvidenceContext(
            frame=export_frame,
            artifacts={"stored_inputs_stage": US_FISCAL_STORED_INPUTS_STAGE},
        ),
        verdict=lambda value: LegacyVerdict.from_failures(value[0]),
    )
    terminal_gate_failures.extend(stored_input_failures_)
    terminal_batch_telemetry.stage(
        "export_frame_stored_inputs",
        message=(
            "Checked the export's stored columns against the installed "
            "policyengine-us and the reviewed non-variable register."
        ),
        **export_frame_stored_inputs,
    )
    if stored_input_failures_:
        terminal_batch_telemetry.stage(
            "export_frame_stored_inputs",
            status="failed",
            message="Stored model inputs gate failed.",
            failures=stored_input_failures_,
        )

    terminal_batch_telemetry.stage(
        "export_dataset",
        message="Writing PolicyEngine-US H5.",
    )
    release_engine = PolicyEngineUSEngine()
    # Route A PR-3: _build_manifests binds every gate-evidence file present in
    # the release directory as this run's verdict. With --out/--release-id
    # reuse, a superseded attempt's file (for example a reform-coverage smoke
    # this run skips) would otherwise be certified as this run's own.
    for stale_gate_evidence in US_RELEASE_GATE_EVIDENCE_FILES.values():
        (release_dir / stale_gate_evidence).unlink(missing_ok=True)
    # microcosm#1034: the stored-input verdict above is release evidence, not
    # only telemetry. Written once the clear loop has run (so it is this run's
    # own) and before the batched raise (so a refused run keeps it locally,
    # like its siblings); _build_manifests binds it and the publisher requires
    # it of every US national calibration release.
    _record_stored_inputs_gate_evidence(
        release_dir,
        failures=stored_input_failures_,
        details=export_frame_stored_inputs,
        telemetry=terminal_batch_telemetry,
    )
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
        input_coverage_gate = shadow_gate(
            "us_national_release_input_coverage",
            lambda: us_release_input_coverage_gate(export_frame, release_engine),
            lambda: EvidenceContext(
                frame=export_frame, artifacts={"release_engine": release_engine}
            ),
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
            )
    # microcosm#454: the calibrated ESI premium total over the anchor's
    # universe (every policyholder) against BEA NIPA 7.8 line 17 within tolerance,
    # red when the column is absent or zero-mass or the raw ASEC coverage
    # columns are gone. Runs on the calibrated export for BOTH the dense and
    # sparse default paths; the verdict ships as the esi_premiums gate
    # evidence, with the base and export signal verdicts beside it so a waived
    # assignment failure stays on the record.
    try:
        esi_premiums_anchor_gate = shadow_gate(
            "us_national_esi_premiums_anchor",
            lambda: us_esi_premiums_anchor_gate(export_frame, time_period=PERIOD),
            lambda: EvidenceContext(
                frame=export_frame, artifacts={"time_period": PERIOD}
            ),
        )
    except Exception as exc:
        if not terminal_gate_failures:
            raise
        terminal_gate_failures.append(
            "ESI premiums failed: evaluation error under earlier gate "
            f"failures: {type(exc).__name__}: {exc}"
        )
        esi_premiums_anchor_gate = None
    if esi_premiums_anchor_gate is not None:
        esi_premiums_failed = (
            not esi_premiums_anchor_gate.passed and not args.allow_esi_premium_gaps
        )
        if esi_premiums_failed:
            terminal_gate_failures.extend(
                f"ESI premiums failed: {failure}"
                for failure in esi_premiums_anchor_gate.failures
            )
        esi_premiums_path = release_dir / "esi_premiums.json"
        esi_premiums_path.write_text(
            json.dumps(
                {
                    "schema_version": 3,
                    "enforced": not args.allow_esi_premium_gaps,
                    "esi_premiums": {
                        "passed": esi_premiums_anchor_gate.passed,
                        "failures": list(esi_premiums_anchor_gate.failures),
                        "details": dict(esi_premiums_anchor_gate.details),
                    },
                    "anchor_feed": _esi_anchor_feed_evidence(
                        esi_anchor_feed_mismatches,
                        enforced=not (
                            args.allow_unpinned_feed or args.allow_esi_premium_gaps
                        ),
                    ),
                    "signal": {
                        "base_frame": _esi_premiums_signal_evidence(
                            esi_premiums_base_gate
                        ),
                        "export_frame": _esi_premiums_signal_evidence(
                            esi_premiums_export_gate
                        ),
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        terminal_batch_telemetry.attach_artifact("esi_premiums", esi_premiums_path)
        if esi_premiums_failed:
            terminal_batch_telemetry.stage(
                "export_dataset",
                status="failed",
                message="ESI premium anchor gate failed.",
                failures=list(esi_premiums_anchor_gate.failures),
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
    # Route A PR-3: the gate details carry only the reference's basename, so
    # the manifests record its path and the sha256 of the bytes loaded here.
    export_reference_name = (
        args.export_input_mass_reference_h5.name
        if args.export_input_mass_reference_h5 is not None
        else "base_frame"
    )
    export_input_mass_reference: dict[str, object] = {
        "path": (
            str(args.export_input_mass_reference_h5)
            if args.export_input_mass_reference_h5 is not None
            else None
        ),
        "sha256": None,
        "reference_name": export_reference_name,
    }
    # The check loads the reference frame itself, so the frame is freed when
    # the check returns or raises.
    export_input_mass_gate = _record_export_input_mass_gate(
        export_frame,
        base_frame,
        reference_h5=args.export_input_mass_reference_h5,
        reference_name=export_reference_name,
        reference_record=export_input_mass_reference,
        relative_tolerance=args.input_mass_relative_tolerance,
        minimum_reference_total=args.input_mass_minimum_reference_total,
        terminal_gate_failures=terminal_gate_failures,
    )
    # The check above is base_frame's last reader; the export frame and the
    # written H5 carry everything later stages read. Rebinding the name frees
    # the frame. A ``del`` would do the same, but ruff would then report the
    # shadow-gate lambdas above, which read the name while they run, as reading
    # an undefined name.
    base_frame = None
    # A reference the gate never evaluated is still named, but marked so no
    # reader takes its bytes as what the parity verdict saw.
    export_input_mass_reference["evaluated"] = export_input_mass_gate is not None
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
            )
    # microcosm#462: tail-concentration gate over the sparse QRF-imputed dollar
    # columns at the export's calibrated weights. The Build M defect — 89% of
    # the shipped non_sch_d_capital_gains mass in 100 records via a repeated
    # $594,484 donor-ceiling value — is invisible to support clipping (every
    # draw inside donor range), count targets (carrier count exact), and mass
    # parity (column excluded from the reference band), but is unmistakable as
    # top-k weighted-mass share. A per-run register that does not match the
    # checked surface is a batched failure (never an early raise), so the
    # measured tail evidence and the #568 weight sidecar survive it.
    # A register mismatch is recorded like any other terminal failure; an
    # evidence-tier certification refuses to own it (microcosm#1585 item 11).
    _record_qrf_tail_concentration_gate(
        export_frame,
        exclusions_path=args.qrf_tail_concentration_exclusions,
        allow_concentration=args.allow_qrf_tail_concentration,
        terminal_gate_failures=terminal_gate_failures,
        release_dir=release_dir,
        telemetry=terminal_batch_telemetry,
    )
    # Batched pre-export raise: the calibration battery, SPM measurement
    # composition, stored model inputs, input coverage,
    # export-mass parity, and QRF tail concentration have ALL been evaluated
    # at this point, so one failed
    # run reports every failing pre-export group at once (Build M attempts 9
    # and 10 each burned a ~2h run to surface one of these groups serially).
    # A failed release gate no longer withholds a build's H5 (microcosm#1585
    # item 11, PR 4a): a build is evaluated before anyone certifies it. Its
    # failures ride the build record, and the build exits non-zero once its
    # record is written. A build has no tier: certification refuses it at the
    # certified tier while its failures stand, and only an evidence-tier
    # certification, which needs an owner for every failure, can ship them.
    build_refusal: RuntimeError | None = None
    if terminal_gate_failures:
        # Gate-failure path ONLY (microcosm#568 review): the exact calibrated
        # weight vector — with the ordered household ids it aligns to, bound
        # to the target-frame identity — is persisted here as the run's
        # record-level weight evidence. Green runs never write these files
        # (the H5 carries the weights), and certification never copies them
        # into a release directory.
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
        )
        build_refusal = _BuildFailedReleaseGatesError(terminal_gate_failures)
    # A green run must not inherit a prior failed attempt's weight evidence
    # (microcosm#568 round 2): with --out/--release-id reuse, stale evidence
    # files would coexist with a green build whose record knows nothing about
    # them. The batched gates have passed, so any evidence present here
    # belongs to a superseded attempt — remove it before the build's
    # artifacts are written.
    if build_refusal is None:
        for stale_evidence in (
            release_dir / FINAL_HOUSEHOLD_WEIGHTS_FILENAME,
            release_dir / FINAL_HOUSEHOLD_WEIGHT_IDS_FILENAME,
            release_dir / FINAL_HOUSEHOLD_WEIGHTS_METADATA_FILENAME,
        ):
            stale_evidence.unlink(missing_ok=True)
    # What the written file fails, its stored-input premise and the export
    # audit below, is recorded with the release-gate failures, on a green
    # build too, so its H5 never stands without a record that marks it
    # failed (review of #1672). No tier certifies it: the evidence tier
    # relaxes gate verdicts, never the written artifact's own checks
    # (microcosm#506), so the record lists these apart.
    artifact_integrity_failures: list[str] = []
    # A failed build's --out may be built into again, so its build may have
    # been certified as evidence since this build started: its H5 and record
    # are then a certified build's, never to be overwritten (review of #1681).
    _refuse_reusing_a_certified_build(release_root)
    dataset_path = artifact_root / dataset_filename
    # The export H5 write: everything below (reform smoke, take-up contract,
    # release manifest sha) reads THIS file, and it must be written only after
    # the batched pre-export raise so a gate-failed run never produces it.
    # microcosm#443: #437 dropped this call while inserting the batched raise,
    # so attempts 13/14 smoke-scored a stale artifact from a prior run.
    release_engine.write_dataset(export_frame, dataset_path, period=PERIOD)
    # microcosm#1026: the stored-input gate above graded a model of the
    # writer's output; the written bytes are graded whatever the gate reached
    # and must earn the same verdict (no refusal, if the gate could not
    # evaluate).
    stored_input_premise_failure = shadow_value(
        "us_national_stored_inputs_written",
        lambda: _written_stored_input_verdict_mismatch(
            dataset_path, export_frame_stored_inputs
        ),
        lambda: EvidenceContext(
            artifacts={
                "dataset_path": dataset_path,
                "pre_export_stored_inputs": export_frame_stored_inputs,
            }
        ),
        verdict=lambda mismatch: LegacyVerdict.from_failures(
            [] if mismatch is None else [mismatch]
        ),
    )
    if stored_input_premise_failure is not None:
        artifact_integrity_failures.append(stored_input_premise_failure)
        terminal_gate_failures.append(stored_input_premise_failure)
    # The writer was the engine's last use; post-export scoring builds its own.
    # It is released here, not between the write and the written file's check,
    # which stays right after the write; that check reads column names and the
    # module-level system only. Rebound rather than deleted, as base_frame is
    # above.
    release_engine = None
    # The build ends here (microcosm#1585 item 11, PR 4a). The optional export
    # audit and the calibration NPZ read the calibration result, which only
    # this process holds, so they run first. Every later stage scores the
    # written H5 and runs from the build record (_certify_us_build), so a
    # build can be judged before anyone pays for those checks.
    if args.audit_export_targets:
        if telemetry is not None:
            telemetry.stage(
                "post_export_audit",
                message="Auditing exported H5 against calibration targets.",
            )
        try:
            _assert_export_matches_calibration(
                dataset_path,
                result,
                target_specs,
                maximum_microsim_batch_size=args.maximum_microsim_batch_size,
            )
        except Exception as error:
            audit_failure = f"Export audit failed: {type(error).__name__}: {error}"
            artifact_integrity_failures.append(audit_failure)
            terminal_gate_failures.append(audit_failure)

    if telemetry is not None:
        telemetry.stage("write_calibration_npz", message="Writing calibration NPZ.")
    calibration_path = artifact_root / calibration_filename
    _write_npz(calibration_path, result=result, registry=registry)
    # Keep a copy of the exact base artifact beside diagnostics for local audit.
    _copy_base_h5_for_local_audit(
        base_h5,
        release_root / f"base_{base_h5.name}",
        authenticated_pool_h5=authenticated_pool_h5,
    )
    if build_refusal is not None or artifact_integrity_failures:
        # The refusal names every failure the record carries, those found
        # after the H5 write included. A build whose written file failed its
        # own checks exits as a refusal (exit 1), not with
        # US_BUILD_FAILED_RELEASE_GATES_EXIT: no tier certifies it, so a
        # driver must not keep it for certification.
        build_refusal = (
            RuntimeError("Release gates failed: " + "; ".join(terminal_gate_failures))
            if artifact_integrity_failures
            else _BuildFailedReleaseGatesError(terminal_gate_failures)
        )
    timing["build_seconds"] = time.perf_counter() - build_started
    build_record_path = _write_us_build_record(
        release_root,
        _us_build_record(
            release_id=release_id,
            release_root=release_root,
            release_dir=release_dir,
            artifact_root=artifact_root,
            dataset_key=dataset_key,
            dataset_filename=dataset_filename,
            calibration_key=calibration_key,
            calibration_filename=calibration_filename,
            commit=full_commit,
            built_at=datetime.now(UTC).isoformat(),
            runtime=_runtime_versions(),
            args=args,
            terminal_gate_failures=terminal_gate_failures,
            artifact_integrity_failures=artifact_integrity_failures,
            n_households=int(export_frame.n("household")),
            result=result,
            post_export_scoring_plan=post_export_scoring_plan,
            release_gate_failures=release_gate_summary,
            registry=registry,
            target_surface=current_target_surface,
            dropped=compilation,
            gates={
                "target_profile_gate": target_profile_gate,
                "ssi_take_up_delivery_gate_result": ssi_take_up_delivery_gate_result,
                "health_input_gate": health_input_gate,
                "base_population_gate": base_population_gate,
                "immigration_gate": immigration_gate,
                "degenerate_input_gate": degenerate_input_gate,
                "ecps_parity_gate": enforced_ecps_parity_gate,
                "hours_worked_gate": hours_worked_gate,
                "snap_take_up_gate": snap_take_up_gate,
                "eligibility_inputs_gate": eligibility_inputs_gate,
                "pregnancy_gate": pregnancy_gate,
                "reported_coverage_vintage_gate": reported_coverage_vintage_gate,
                "snap_discretionary_exemption_gate": (
                    snap_discretionary_exemption_gate
                ),
            },
            warm_start_calibration=warm_start_calibration,
            selection_source=selection_source_payload,
            target_surface_selection=target_surface_selection,
            ledger_artifact=ledger_artifact.provenance(),
            default_dataset=default_dataset,
            medicaid_enrollment_substitutions=medicaid_enrollment_substitutions,
            exact_k_ladder=exact_k_ladder_provenance,
            base_pool=base_pool_receipt,
            acs_predictor_join=acs_predictor_join_receipt,
            export_input_mass_reference=export_input_mass_reference,
            calibration_runtime=calibration_runtime,
            diagnostics_outcome=diagnostics_outcome,
            scf_wealth_signal=scf_wealth_signal,
            fiscal_target_sources=_fiscal_target_source_provenance(target_specs),
            congressional_district_vintage_crosswalk=(
                congressional_district_vintage_crosswalk_metadata
            ),
            fiscal_target_exclusion_receipt=fiscal_target_exclusion_receipt,
            fsa_target_receipt=fsa_target_receipt,
            take_up_participation=us_take_up_participation_diagnostics(export_frame),
            medicaid_take_up_diagnostics=medicaid_take_up_diagnostics,
            snap_state_take_up_diagnostics=snap_state_take_up_diagnostics,
            timing=timing,
            staging=_staging_manifest_block(telemetry),
            staging_settings={
                "no_staging": bool(args.no_staging),
                "repo_id": args.staging_repo_id or None,
            },
        ),
    )
    if telemetry is not None:
        telemetry.attach_artifact("build_record", build_record_path)
        if build_refusal is None:
            telemetry.complete()
    _print_built_result(
        build=release_id,
        release_root=release_root,
        build_dir=release_dir,
        artifact_root=artifact_root,
        build_record=build_record_path,
        release_gate_failures=terminal_gate_failures if build_refusal else (),
    )
    if build_refusal is not None:
        raise build_refusal


def _certify_us_build(
    record: _USBuildRecord,
    *,
    certification: _Certification,
    telemetry: _BuildProgress | None = None,
) -> None:
    """Run a finished build's post-export checks and write its manifests.

    The build (``_main``) ends once its H5, calibration weights, diagnostics,
    gate evidence and :data:`US_BUILD_RECORD_FILENAME` are written. This
    scores the written H5 (the reform-coverage smoke, reform validation,
    demographics and the holdouts), writes the source-coverage and take-up
    receipts and both manifests, and leaves the release directory in the
    build-folder layout for ``microcosm-publish-release-build``
    (microcosm#1585 item 11, PR 4a). Every input is the record, or a file
    whose digest the record pins: nothing is read from the build's process.
    The manifests' provenance (the commit, ``built_at`` and the built-with
    packages) is the build's, from the record. A build has no tier:
    ``certification`` is the tier certification chose, the release directory
    it placed the build's files in and the ID it publishes the build under,
    and both manifests record it beside the build's provenance.
    """

    certify_started = time.perf_counter()
    args = record.certify_args(evidence_release=certification.evidence_release)
    release_id = certification.build
    release_dir = certification.release_dir
    artifact_root = record.artifact_root
    dataset_key = record.dataset_key
    dataset_filename = record.dataset_filename
    calibration_key = record.calibration_key
    calibration_filename = record.calibration_filename
    dataset_path = artifact_root / dataset_filename
    evidence_failure_owner_patterns = certification.owner_patterns
    terminal_gate_failures = list(record.terminal_gate_failures)
    timing = dict(record.timing)
    result = record.calibration_result()
    registry = record.registry_summary()
    post_export_scoring_plan = record.post_export_scoring_plan(args, result=result)
    inputs = record.manifest_inputs
    if inputs["release_gate_failures"] is None:
        raise RuntimeError(
            f"Build {release_id}: its release gate summary crashed when it was "
            "built, so its manifests cannot record the calibration gate."
        )
    gates = record.gates()
    target_profile_gate = gates["target_profile_gate"]
    ssi_take_up_delivery_gate_result = gates["ssi_take_up_delivery_gate_result"]
    health_input_gate = gates["health_input_gate"]
    base_population_gate = gates["base_population_gate"]
    immigration_gate = gates["immigration_gate"]
    degenerate_input_gate = gates["degenerate_input_gate"]
    enforced_ecps_parity_gate = gates["ecps_parity_gate"]
    hours_worked_gate = gates["hours_worked_gate"]
    snap_take_up_gate = gates["snap_take_up_gate"]
    eligibility_inputs_gate = gates["eligibility_inputs_gate"]
    pregnancy_gate = gates["pregnancy_gate"]
    reported_coverage_vintage_gate = gates["reported_coverage_vintage_gate"]
    snap_discretionary_exemption_gate = gates["snap_discretionary_exemption_gate"]
    # Read only by the gate summary, which the build recorded.
    incumbent_diagnostics = None
    enforced_input_mass_reference_gate = None
    current_target_surface = inputs["target_surface"]
    compilation = inputs["dropped"]
    warm_start_calibration = inputs["warm_start_calibration"]
    selection_source_payload = inputs["selection_source"]
    target_surface_selection = inputs["target_surface_selection"]
    default_dataset = inputs["default_dataset"]
    medicaid_enrollment_substitutions = inputs["medicaid_enrollment_substitutions"]
    exact_k_ladder_provenance = inputs["exact_k_ladder"]
    base_pool_receipt = inputs["base_pool"]
    acs_predictor_join_receipt = inputs["acs_predictor_join"]
    export_input_mass_reference = inputs["export_input_mass_reference"]
    calibration_runtime = inputs["calibration_runtime"]
    diagnostics_outcome = record.diagnostics_outcome(release_dir)
    scf_wealth_signal = inputs["scf_wealth_signal"]
    source_coverage = record.source_coverage
    congressional_district_vintage_crosswalk_metadata = source_coverage[
        "congressional_district_vintage_crosswalk"
    ]
    fiscal_target_exclusion_receipt = source_coverage["fiscal_target_exclusion_receipt"]
    fsa_target_receipt = source_coverage["fsa_student_loan_target_receipt"]
    medicaid_take_up_diagnostics = record.take_up["medicaid"]
    snap_state_take_up_diagnostics = record.take_up["snap_state"]
    # Route A remediation (microcosm#956): every post-export stage below (the
    # smoke, reform validation, demographics) scores THIS file through one
    # household-batched scorer instead of one whole-pool Microsimulation each.
    # It loads the written H5 once, binds its sha256 (the manifest must pin
    # the same bytes), and serves the baseline plans recorded above from one
    # merged baseline pass (microcosm#1595).
    post_export_scorer = _open_post_export_scorer(
        post_export_scoring_plan,
        dataset_path,
        batch_parent=record.release_root / US_POST_EXPORT_BATCH_DIRECTORY,
    )
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
        smoke_scoring = post_export_scorer.open_consumer(
            "reform_coverage_smoke",
            post_export_scoring_plan.baseline_plan("reform_coverage_smoke"),
            planned_reform_passes=post_export_scoring_plan.planned_reform_passes(
                "reform_coverage_smoke"
            ),
        )
        reform_coverage_smoke_gate = us_reform_coverage_smoke_gate(
            simulate=smoke_scoring.simulate,
            period=PERIOD,
        )
        smoke_scoring_record = post_export_scorer.finish_consumer(smoke_scoring)
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
                    "post_export_scoring": smoke_scoring_record,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        # The smoke's record is written; free its reform results before reform
        # validation runs. Its baseline values stay with the scorer only while
        # a consumer still to be served planned them.
        del smoke_scoring
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
            post_export_scorer=post_export_scorer,
            baseline_plan=(
                None
                if args.skip_out_of_sample_reforms
                else post_export_scoring_plan.baseline_plan("reform_validation")
            ),
            planned_reform_passes=post_export_scoring_plan.planned_reform_passes(
                "reform_validation"
            ),
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
            post_export_scorer=post_export_scorer,
            baseline_plan=post_export_scoring_plan.baseline_plan("demographics"),
        )
        if telemetry is not None:
            telemetry.attach_artifact("demographics", release_dir / "demographics.json")
    if telemetry is not None:
        telemetry.stage(
            "ssi_social_security_holdout",
            message="Writing held-out SSI/Social Security receipt overlap.",
        )
    _write_ssi_social_security_holdout(
        release_dir=release_dir,
        dataset_path=dataset_path,
        release_id=release_id,
        post_export_scorer=post_export_scorer,
        baseline_plan=post_export_scoring_plan.baseline_plan(
            "ssi_social_security_holdout"
        ),
    )
    if telemetry is not None:
        telemetry.attach_artifact(
            "us_ssi_social_security_holdout",
            release_dir / "us_ssi_social_security_holdout.json",
        )
        telemetry.stage(
            "medicare_irmaa_holdout",
            message="Writing held-out Medicare IRMAA payers and premiums.",
        )
    _write_medicare_irmaa_holdout(
        release_dir=release_dir,
        dataset_path=dataset_path,
        release_id=release_id,
        post_export_scorer=post_export_scorer,
        baseline_plan=post_export_scoring_plan.baseline_plan("medicare_irmaa_holdout"),
    )
    if telemetry is not None:
        telemetry.attach_artifact(
            "us_medicare_irmaa_holdout",
            release_dir / "us_medicare_irmaa_holdout.json",
        )
    # Every post-export engine stage has run; remove the scorer's batch files.
    # The manifest must pin the bytes those stages scored, and both manifests
    # carry how they were scored (Route A remediation PR-3).
    post_export_scoring = _post_export_scoring_manifest_block(post_export_scorer)
    scored_dataset_sha256 = _close_post_export_scorer(post_export_scorer)
    post_export_scorer = None
    # Reads the written H5 without an engine, after the scorer has closed,
    # so it adds no memory on top of the scorer's.
    if telemetry is not None:
        telemetry.stage(
            "population_structure_holdout",
            message="Writing held-out population-structure checks.",
        )
    _write_population_structure_holdout(
        release_dir=release_dir, dataset_path=dataset_path, release_id=release_id
    )
    if telemetry is not None:
        telemetry.attach_artifact(
            "us_population_structure_holdout",
            release_dir / "us_population_structure_holdout.json",
        )

    if telemetry is not None:
        telemetry.stage(
            "source_coverage", message="Writing source coverage diagnostics."
        )
    active_aliases, surface_exclusions = _source_coverage_aliases(
        target_surface_selection
    )
    coverage = us_source_coverage_diagnostics(
        active_target_aliases=active_aliases,
        reviewed_exclusions={
            **_reviewed_exclusions(active_aliases),
            **surface_exclusions,
        },
    )
    coverage["fiscal_target_sources"] = source_coverage["fiscal_target_sources"]
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
    coverage["fiscal_target_source_column_exclusions"] = [
        {
            "measure_id": measure_id,
            "source_column_id": source_column_id,
            "reason": reason,
        }
        for (measure_id, source_column_id), reason in sorted(
            US_FISCAL_TARGET_SOURCE_COLUMN_EXCLUSIONS.items()
        )
    ]
    coverage[US_FISCAL_TARGET_EXCLUSION_RECEIPT_KEY] = fiscal_target_exclusion_receipt
    if fsa_target_receipt is not None:
        coverage[US_FSA_STUDENT_LOAN_TARGET_RECEIPT_KEY] = fsa_target_receipt
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
    take_up_participation = record.take_up["participation"]
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
    timing["certify_seconds"] = time.perf_counter() - certify_started
    timing["total_build_seconds"] = timing["build_seconds"] + timing["certify_seconds"]
    evidence_known_failures = None
    if args.evidence_release:
        if not terminal_gate_failures:
            raise RuntimeError(
                "Evidence release refused: every terminal gate passed. This "
                "build qualifies for the certified tier — certify it without "
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
        target_surface=current_target_surface,
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
        release_gate_failures=inputs["release_gate_failures"],
        timing=timing,
        warm_start_calibration=warm_start_calibration,
        selection_source=selection_source_payload,
        target_surface_selection=target_surface_selection,
        ledger_artifact=inputs["ledger_artifact"],
        default_dataset=default_dataset,
        medicaid_enrollment_substitutions=medicaid_enrollment_substitutions,
        # The staging run the publisher checks is certification's, which
        # carries the post-export verdicts and the manifests; the build's
        # rides inside it.
        staging={**_staging_manifest_block(telemetry), "build": dict(record.staging)},
        dataset_key=dataset_key,
        dataset_filename=dataset_filename,
        calibration_key=calibration_key,
        calibration_filename=calibration_filename,
        exact_k_ladder=exact_k_ladder_provenance,
        base_pool=base_pool_receipt,
        acs_predictor_join=acs_predictor_join_receipt,
        evidence_known_failures=evidence_known_failures,
        export_input_mass_reference=export_input_mass_reference,
        calibration_runtime=calibration_runtime,
        skipped_gates=(
            ("reform_coverage_smoke",) if args.skip_reform_coverage_smoke else ()
        ),
        scored_dataset_sha256=scored_dataset_sha256,
        post_export_scoring=post_export_scoring,
        diagnostics_outcome=diagnostics_outcome,
        scf_wealth_signal=scf_wealth_signal,
        scf_education_debt=args.scf_education_debt,
        git_commit=record.commit,
        built_at=record.built_at,
        runtime_versions=record.runtime,
        certification=certification.manifest_block(record),
    )
    # Both manifests name the release and this build, in the build-folder
    # layout, before telemetry attaches them; the directory must then pass
    # its tier's contract (microcosm#1570).
    publication = _finish_build_folder_release(
        release_dir,
        build=release_id,
        artifact_root=artifact_root,
        evidence_release=args.evidence_release,
    )
    if telemetry is not None:
        telemetry.attach_artifact("build_manifest", release_dir / "build_manifest.json")
        telemetry.attach_artifact(
            "release_manifest",
            release_dir / "release_manifest.json",
        )
        telemetry.complete()

    _print_build_result(
        build=release_id,
        release_dir=release_dir,
        artifact_root=artifact_root,
        publication=publication,
    )


def _run_as_script() -> None:
    """Run the build as a command: a build that failed release gates once its
    record is written exits :data:`US_BUILD_FAILED_RELEASE_GATES_EXIT`, with
    its refusal printed as Python prints any other."""

    try:
        main()
    except _BuildFailedReleaseGatesError:
        traceback.print_exc()
        raise SystemExit(US_BUILD_FAILED_RELEASE_GATES_EXIT) from None


if __name__ == "__main__":
    _run_as_script()
