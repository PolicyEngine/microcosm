"""Build a local US base H5 with a PUF tax-detail support channel.

This diagnostic builder starts from an existing Microcosm US H5, clones the
frame into ASEC and PUF-tax-detail support channels, imputes PUF-observed
inputs onto the PUF channel with Microcosm's weighted QRF, and writes a fresh
base H5 for the fiscal refresh calibration builder.

Telemetry: after parsing its arguments the builder starts the emitter
(``microcosm.build.program_telemetry``, pipeline ``us_puf_support_base``) and
records completion, or failure with the error. With
``MICROCOSM_TELEMETRY_REQUIRED=1``, which the Modal stage runner sets, an
emitter that cannot be used stops the builder before it reads any input. A
checkpointed ``--stage all`` run records one stage change per step and fails
with the first child that exits nonzero; the children start no emitter. It
records the change before each child interpreter starts, and, for the child
that runs several of ``ONE_PROCESS_STEPS``, when that child reports starting
each later one. The compatibility path records the single stage ``all``, and
a named ``--stage`` run by hand records that stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    # The one thread setting, MICROCOSM_THREADS, fills the OpenMP, BLAS,
    # numexpr and joblib thread variables, and the libraries read those when
    # they load, so this comes before numpy. Without the setting nothing is
    # set. It runs only for the command: tests and other tools import this
    # file. The helper is loaded from this file's folder, which is not on the
    # import path under ``python -m`` or ``python -P``. Stage subprocesses
    # inherit the values, and the run configuration records them, except
    # LOKY_MAX_CPU_COUNT and NUMEXPR_MAX_THREADS (``_stage_run_config``). E402
    # is ignored for this file in pyproject for this block.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from us_build_thread_environment import set_build_thread_environment

    del sys.path[0]
    set_build_thread_environment()

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shutil
import stat
import subprocess
from collections.abc import (
    Callable,
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    Sequence,
)
from contextlib import contextmanager
from dataclasses import replace
from typing import NoReturn

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info

from microcosm.build import FitWeightRecord, GateResult, weights_audit_gate
from microcosm.build.frame_checkpoint import write_frame_checkpoint
from microcosm.build.gate_battery import EvidenceContext
from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.outer_stage_runtime import (
    FrameIdentity,
    LoadedStageCheckpoint,
    Stage,
    StagePipeline,
    StageRuntime,
    assert_clone_expansion,
    frame_identity,
)
from microcosm.build.program_telemetry import (
    child_stage_environment,
    program_telemetry,
    telemetry_recorded_by_parent,
)
from microcosm.build.source_manifest import (
    SupportSpineSpec,
    load_support_spine_manifest,
)
from microcosm.build.source_runtime import SourceRuntimeConfig, run_source_stage
from microcosm.build.stage_profile import profile_stage
from microcosm.build.telemetry_emitter import LocalTelemetryEmitter
from microcosm.build.us_runtime import (
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER,
    ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256,
    ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256,
    ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR,
    ASEC_2023_WEEKS_UNEMPLOYED_ZIP_URL,
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
    ASEC_RAW_STAGE_ARTIFACT_KIND,
    ASEC_RAW_STAGE_CHECKPOINT_FILENAME,
    ASEC_RAW_STAGE_OPERATOR_STATUS,
    ASEC_RAW_STAGE_SCHEMA_VERSION,
    ASEC_RAW_STAGE_STAGE,
    BASE_ASEC_SUPPORT_CHANNEL,
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR,
    GEOGRAPHY_LADDER_VINTAGES_ATTR,
    PUF_CAPITAL_GAINS_TAIL_STAGE_NAME,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_PUF_SUPPORT_FIT_NAME,
    US_SOURCE_MANIFEST,
    US_SUPPORT_SPINE_SPEC,
    AsecSource,
    assert_operator_free_source_frame,
    build_pooled_asec_unit_frame,
    build_puf_e01000_reconciliation_basis,
    clone_us_frame_for_puf_support,
    congressional_district_assignment_summary,
    congressional_district_distribution_from_ledger_facts,
    derive_us_cps_carried_inputs,
    fetch_asec_2023_weeks_unemployed_source,
    fill_asec_2022_weeks_unemployed_source,
    fill_asec_education_assistance_source,
    fill_asec_public_assistance_type_source,
    finalize_puf_e01000_reconciliation,
    impute_us_housing_assistance_to_puf_support,
    impute_us_puf_tax_detail_support,
    load_acs_2022_rent_donor,
    load_asec_2023_weeks_unemployed_source,
    load_asec_education_assistance_sources,
    load_asec_public_assistance_type_sources,
    load_asec_raw_stage_checkpoint,
    load_congressional_district_vintage_crosswalk,
    load_us_block_ladder,
    processed_puf_tax_unit_partnership_s_corp_income,
    puf_tax_unit_donor_from_arrays,
    resolve_asec_spm_role_source_paths,
    source_year_puf_adjusted_gross_income,
    source_year_puf_partnership_s_corp_income,
    support_channel_column,
    transfer_puf_capital_gains_tail,
    translate_congressional_district_facts_to_current_vintage,
    us_adult_care_signal_gate,
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
    us_esi_premiums_signal_gate,
    us_farm_business_income_signal_gate,
    us_form_4952_election_signal_gate,
    us_geography_ladder_assignment_summary,
    us_geography_ladder_gate,
    us_housing_inputs_signal_gate,
    us_immigration_composition_summary,
    us_medicare_take_up_signal_gate,
    us_misc_itemized_signal_gate,
    us_pregnancy_signal_gate,
    us_prior_year_income_signal_gate,
    us_prior_year_income_source_reconciliation_gate,
    us_qbi_inputs_signal_gate,
    us_relationship_inputs_signal_gate,
    us_retirement_contributions_signal_gate,
    us_retirement_distributions_signal_gate,
    us_salt_refund_income_signal_gate,
    us_source_operation_handlers,
    us_spm_independence_role_signal_gate,
    us_weeks_unemployed_signal_gate,
    us_wic_claim_signal_gate,
    us_workers_compensation_signal_gate,
    validate_puf_capital_gains_tail_manifest,
    with_household_congressional_districts,
    with_household_us_geography_ladder,
    with_us_adult_care_inputs,
    with_us_child_support_inputs,
    with_us_childcare_inputs,
    with_us_disability_benefits,
    with_us_education_inputs,
    with_us_eligibility_inputs,
    with_us_energy_subsidy_input,
    with_us_esi_premium_inputs,
    with_us_housing_inputs,
    with_us_immigration_inputs,
    with_us_medicare_take_up_input,
    with_us_pregnancy_inputs,
    with_us_prior_year_income_inputs,
    with_us_qbi_input_reconciliation,
    with_us_relationship_inputs,
    with_us_retirement_contribution_inputs,
    with_us_retirement_distribution_inputs,
    with_us_spm_independence_role,
    with_us_weeks_unemployed,
    with_us_wic_claim_input,
    with_us_workers_compensation,
    write_puf_capital_gains_tail_manifest,
)
from microcosm.build.us_runtime.asec_sources import ASEC_SOURCE_ARTIFACTS
from microcosm.build.us_runtime.battery_bindings import us_gate_registry
from microcosm.build.us_runtime.block_location import (
    POPULACE_BLOCK_LADDER_SHA256_ATTR,
    POPULACE_BLOCK_LADDER_VINTAGES_ATTR,
    POPULACE_LOCATION_CLONES_ATTR,
    POPULACE_LOCATION_RULE_ATTR,
    POPULACE_LOCATION_SEED_ATTR,
    US_LOCATION_RULE_BLOCK_V1,
    US_LOCATION_RULE_CHOICES,
    US_LOCATION_RULE_LEGACY,
    UsLocationLadder,
    load_us_location_ladder,
    us_block_location_gate,
    us_block_location_manifest,
    with_household_us_block_location,
)
from microcosm.build.us_runtime.cps_source_geography import (
    cps_source_geography,
    household_source_keys,
    load_cps_household_geography,
)
from microcosm.build.us_runtime.gate_scopes import us_scoped_gate_manifest
from microcosm.build.us_runtime.gate_shadow import (
    GATE_BATTERY_SHADOW_DIRECTORY,
    GateBatteryShadow,
    LegacyVerdict,
    active_gate_shadow,
    current_gate_shadow,
    shadow_file_paths,
    shadow_gate,
    shadow_not_run,
    tool_gate_shadows_enabled,
)
from microcosm.build.us_runtime.h5_io import (
    assert_h5_unchanged,
    refuse_denied_frame,
    refuse_denied_pool_h5,
)
from microcosm.build.us_runtime.puf_qrf_chain import (
    finalize_primary_puf_qrf_chain,
    initialize_primary_puf_qrf_chain,
    run_primary_puf_qrf_chain,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    US_PERSON_REFERENCE_ID_COLUMNS,
)
from microcosm.frame import Frame, WeightKind, Weights
from microcosm.frame.adapters.policyengine_us import PolicyEngineUSEngine
from microcosm.frame.units import US_SCHEMA

PERIOD = 2024
DATASET_FILENAME = "base_populace_us_2024_puf_support.h5"
SUMMARY_FILENAME = "base_populace_us_2024_puf_support.summary.json"
ALL_STAGE_CHECKPOINT_FILENAME = "stage_all.frame.h5"
CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME = "capital_gain_distributions"
LEGACY_STAGE_ALIASES = ("a", "b", "c", "d")
PIPELINE_STEPS = (
    "source_construction",
    "pre_clone_enrichment",
    "clone_feature_extraction",
    "primary_qrf_chain",
    "qrf_finalization",
    PUF_CAPITAL_GAINS_TAIL_STAGE_NAME,
    CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME,
    "qbi_reconciliation",
    "wic_post_clone",
    "housing_assistance",
    "prior_year_income_post_clone",
    "child_support_post_clone",
    "disability_benefits_post_clone",
    "workers_compensation_post_clone",
    "weeks_unemployed_post_clone",
    "childcare_post_clone",
    "adult_care_post_clone",
    "energy_subsidy_post_clone",
    "retirement_contributions_post_clone",
    "retirement_distributions_post_clone",
    "education_inputs_post_clone",
    "congressional_district_assignment",
    "block_ladder_assignment",
    "final_export",
)
STAGE_NAMES = (*LEGACY_STAGE_ALIASES, *PIPELINE_STEPS)
# The order is executable configuration, not documentation: stage dispatch and
# checkpoint metadata both validate this exact sequence before any transform.
STAGE_BOUNDARIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source_construction", ("_load_base_frame_from_args",)),
    (
        "pre_clone_enrichment",
        (
            "derive_us_cps_carried_inputs",
            "with_us_prior_year_income_inputs",
            "with_us_relationship_inputs",
            "with_us_spm_independence_role",
            "with_us_medicare_take_up_input",
            "with_us_housing_inputs[includes_acs_rent_in_current_order]",
            "with_us_eligibility_inputs",
            "with_us_pregnancy_inputs",
            "with_us_wic_claim_input",
            "with_us_child_support_inputs",
            "with_us_disability_benefits",
            "with_us_workers_compensation",
            "with_us_weeks_unemployed",
            "with_us_childcare_inputs",
            "with_us_energy_subsidy_input",
            "with_us_retirement_contribution_inputs",
            "with_us_retirement_distribution_inputs",
            "with_us_esi_premium_inputs",
            "with_us_immigration_inputs",
        ),
    ),
    (
        "clone_feature_extraction",
        (
            "clone_us_frame_for_puf_support",
            "puf_tax_unit_donor_from_arrays",
            "initialize_primary_puf_qrf_chain",
        ),
    ),
    ("primary_qrf_chain", ("run_primary_puf_qrf_chain[target_subprocesses]",)),
    ("qrf_finalization", ("finalize_primary_puf_qrf_chain",)),
    (
        PUF_CAPITAL_GAINS_TAIL_STAGE_NAME,
        ("transfer_puf_capital_gains_tail",),
    ),
    (
        CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME,
        ("run_source_stage[capital_gain_distributions]",),
    ),
    ("qbi_reconciliation", ("with_us_qbi_input_reconciliation",)),
    ("wic_post_clone", ("with_us_wic_claim_input",)),
    (
        "housing_assistance",
        ("impute_us_housing_assistance_to_puf_support",),
    ),
    (
        "prior_year_income_post_clone",
        ("with_us_prior_year_income_inputs",),
    ),
    ("child_support_post_clone", ("with_us_child_support_inputs",)),
    ("disability_benefits_post_clone", ("with_us_disability_benefits",)),
    ("workers_compensation_post_clone", ("with_us_workers_compensation",)),
    ("weeks_unemployed_post_clone", ("with_us_weeks_unemployed",)),
    ("childcare_post_clone", ("with_us_childcare_inputs",)),
    ("adult_care_post_clone", ("with_us_adult_care_inputs",)),
    ("energy_subsidy_post_clone", ("with_us_energy_subsidy_input",)),
    (
        "retirement_contributions_post_clone",
        ("with_us_retirement_contribution_inputs",),
    ),
    (
        "retirement_distributions_post_clone",
        ("with_us_retirement_distribution_inputs",),
    ),
    ("education_inputs_post_clone", ("with_us_education_inputs",)),
    (
        "congressional_district_assignment",
        ("with_household_congressional_districts",),
    ),
    ("block_ladder_assignment", ("with_household_us_geography_ladder",)),
    ("final_export", ("PolicyEngineUSEngine.write_dataset",)),
)
OUTER_STAGE_PIPELINE = StagePipeline(
    tuple(Stage(name, " -> ".join(boundaries)) for name, boundaries in STAGE_BOUNDARIES)
)
# --location-rule block_v1 (microcosm #696) replaces the legacy two-step
# location (SOI return-count congressional-district draw, then a block within
# that district) with one household block draw from which every geography,
# the congressional district included, derives. It is a separate pipeline so
# the legacy stage list, its descriptions and its sha stay byte-identical.
HOUSEHOLD_LOCATION_STAGE_NAME = "household_location_assignment"
_LEGACY_LOCATION_STAGE_NAMES = (
    "congressional_district_assignment",
    "block_ladder_assignment",
)


def _block_v1_stage_boundaries() -> tuple[tuple[str, tuple[str, ...]], ...]:
    boundaries: list[tuple[str, tuple[str, ...]]] = []
    for stage_name, steps in STAGE_BOUNDARIES:
        if stage_name == _LEGACY_LOCATION_STAGE_NAMES[0]:
            boundaries.append(
                (
                    HOUSEHOLD_LOCATION_STAGE_NAME,
                    ("with_household_us_block_location[block_v1]",),
                )
            )
        elif stage_name != _LEGACY_LOCATION_STAGE_NAMES[1]:
            boundaries.append((stage_name, steps))
    return tuple(boundaries)


BLOCK_V1_STAGE_BOUNDARIES = _block_v1_stage_boundaries()
BLOCK_V1_PIPELINE_STEPS = tuple(name for name, _ in BLOCK_V1_STAGE_BOUNDARIES)
BLOCK_V1_OUTER_STAGE_PIPELINE = StagePipeline(
    tuple(
        Stage(name, " -> ".join(boundaries))
        for name, boundaries in BLOCK_V1_STAGE_BOUNDARIES
    )
)
# Every stage name --stage accepts under either rule; a rule's own pipeline
# refuses the other rule's location stages at run time.
ALL_STAGE_NAMES = (*STAGE_NAMES, HOUSEHOLD_LOCATION_STAGE_NAME)
#: The steps that share one step process in a ``--stage all`` run (decision
#: 1.5 · 6): the 16 from the capital-gains tail transfer to the education
#: inputs, which follow each other under both location rules. The parent
#: starts one child for those still to run. The child reads the first one's
#: predecessor checkpoint and hands each later step the frame the step before
#: it completed. Each step still writes its checkpoint, completion record,
#: profile record and saved check observations, so a stopped run resumes
#: after the last finished step. Before each step the child compares the run
#: config with the locked one, as each step process does when it starts, so
#: a change to the builder's source files during the run stops it before the
#: next step. What sharing a process gives up: memory is not returned to the
#: system between these steps; on macOS each step's recorded peak is the
#: process's peak so far; and a frame handed over in memory is not read back,
#: so its checkpoint's SHA-256 and identity are next checked when a later
#: process reads the file. A frame handed over also keeps the records of the
#: column memory it shares with earlier frames (``Frame.check_shared_columns``),
#: which a frame read from disk does not have: a step that wrote into shared
#: column memory in place would be refused at its checkpoint write here,
#: where a step that read its input from disk would not. No step does that.
ONE_PROCESS_STEPS = PIPELINE_STEPS[
    PIPELINE_STEPS.index(PUF_CAPITAL_GAINS_TAIL_STAGE_NAME) : (
        PIPELINE_STEPS.index("education_inputs_post_clone") + 1
    )
]


def _location_rule(args: object) -> str:
    return str(getattr(args, "location_rule", US_LOCATION_RULE_LEGACY))


def _is_block_v1(args: object) -> bool:
    return _location_rule(args) == US_LOCATION_RULE_BLOCK_V1


def _pipeline_steps_for(args: object) -> tuple[str, ...]:
    return BLOCK_V1_PIPELINE_STEPS if _is_block_v1(args) else PIPELINE_STEPS


def _pipeline_for(args: object) -> StagePipeline:
    return BLOCK_V1_OUTER_STAGE_PIPELINE if _is_block_v1(args) else OUTER_STAGE_PIPELINE


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-h5", type=Path)
    source.add_argument(
        "--asec-h5",
        action="append",
        help="Raw ASEC source as YEAR=PATH. Pass once per source year.",
    )
    parser.add_argument(
        "--asec-h5-sha256",
        action="append",
        help=(
            "Expected SHA-256 of one --asec-h5 input, as YEAR=SHA256. Pass "
            "once per source year. Verified before any stage runs; a year "
            "with a pinned canonical digest refuses a differing pin."
        ),
    )
    parser.add_argument("--target-year", default=PERIOD, type=int)
    parser.add_argument(
        "--asec-max-households",
        type=int,
        help="Optional smoke limit applied to every raw ASEC source.",
    )
    parser.add_argument(
        "--support-spine-spec",
        type=Path,
        help=(
            "Optional support-spine manifest. When provided, --asec-h5 values "
            "are YEAR=PATH file mappings and the manifest owns source roles, "
            "relative years, and shares."
        ),
    )
    parser.add_argument("--puf-h5", required=True, type=Path)
    parser.add_argument(
        "--puf-source-year-csv",
        type=Path,
        help=(
            "Restricted raw TY2015 IRS PUF CSV carrying RECID, E00100, S006, "
            "and the archived aggregate-record donor fields. Required when "
            "building the E19200 decomposition from a nonzero processed PUF."
        ),
    )
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
        "--asec-education-source",
        action="append",
        metavar="YEAR=PATH",
        help=(
            "Optional INCOME_YEAR=PATH mapping to a local copy of the "
            "SHA-pinned official ASEC survey archive (zip or extracted "
            "pppub member) for that pooled income year (income year YYYY "
            "maps to the survey-year YYYY+1 archive). Source construction "
            "restores from it the reviewed Census person columns the "
            "--asec-h5 input lacks (microcosm #720); the raw-stage mapping "
            "restores ED_VAL and PAW_TYP; the SPM independence role stage "
            "derives the role. Years without a mapping are fetched from the "
            "official Census archive and verified against the same pins."
        ),
    )
    parser.add_argument(
        "--acs-h5",
        type=Path,
        help=(
            "SHA-pinned processed ACS 2022 ARRAYS artifact used by the "
            "housing/rent stage. Required unless --base-h5 already carries "
            "a green housing-input surface."
        ),
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=(*ALL_STAGE_NAMES, "all"),
        default="all",
        help=(
            "Run one descriptive checkpointed stage, or run all stages as fresh "
            "processes in the locked order. a-d remain parse-only Phase-1 aliases."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory for durable frame checkpoints and stage_profile.json.",
    )
    parser.add_argument(
        "--equivalence-boundary-dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--equivalence-deterministic-h5-metadata",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--n-estimators", default=32, type=int)
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        help=(
            "Ledger consumer facts JSONL. Required when assigning congressional "
            "district geography."
        ),
    )
    parser.add_argument(
        "--assign-congressional-districts",
        action="store_true",
        help=(
            "Assign household congressional_district_geoid values from SOI CD "
            "return-count Ledger facts, constrained by state_fips."
        ),
    )
    parser.add_argument(
        "--congressional-district-vintage-crosswalk",
        type=Path,
        help=(
            "Optional source-to-current congressional-district crosswalk "
            "artifact with source_geography_id, target_geography_id, and "
            "weight columns. When provided, SOI CD return-count facts are "
            "translated before support assignment."
        ),
    )
    parser.add_argument("--congressional-district-seed", default=0, type=int)
    parser.add_argument(
        "--block-ladder-artifact",
        type=Path,
        help=(
            "US block-ladder NPZ artifact "
            "(tools/build_us_block_ladder_artifact.py). Assigns each "
            "household a census block within its congressional district and "
            "derives the county/tract/place/SLD/CBSA spine columns. US bases "
            "carry the ladder by default; omit only with "
            "--without-block-ladder."
        ),
    )
    parser.add_argument(
        "--without-block-ladder",
        action="store_true",
        help=(
            "Explicitly build a base without the geography ladder "
            "(diagnostic builds only; a ladder-less base cannot become a "
            "release — the L0/refit export requires the spine columns)."
        ),
    )
    parser.add_argument("--geography-ladder-seed", default=0, type=int)
    parser.add_argument(
        "--allow-geography-ladder-gate-failures",
        action="store_true",
        help=(
            "Diagnostic escape hatch for partial-spine smoke builds. By "
            "default a failing geography-ladder gate (e.g. the NYC "
            "never-collapses-to-zero regression of microcosm #34) aborts the "
            "build."
        ),
    )
    parser.add_argument(
        "--location-rule",
        choices=US_LOCATION_RULE_CHOICES,
        default=US_LOCATION_RULE_LEGACY,
        help=(
            "Household location rule. 'legacy' (default) draws a "
            "congressional district from SOI return-count Ledger facts and "
            "then a block within it. 'block_v1' (microcosm #696) draws one "
            "2020 census block per household proportional to block "
            "population within its CPS ASEC source geography (identified "
            "county, else its state outside the ASEC year's identified "
            "counties) and derives every other geography, the congressional "
            "district included, from the block."
        ),
    )
    parser.add_argument(
        "--location-seed",
        default=0,
        type=int,
        help="Seed of the block_v1 household block draw (block_v1 only).",
    )
    parser.add_argument(
        "--location-clones",
        default=1,
        type=int,
        help=(
            "Location clones per household (block_v1 only). Only 1 is "
            "admitted on this line today; see the parse-time refusal."
        ),
    )
    args = parser.parse_args(argv)
    if args.location_rule == US_LOCATION_RULE_BLOCK_V1:
        _validate_block_v1_args(parser, args)
        return _validate_common_args(parser, args)
    if args.location_seed != 0 or args.location_clones != 1:
        parser.error(
            "--location-seed and --location-clones apply only to "
            "--location-rule block_v1; the legacy rule draws with "
            "--congressional-district-seed and --geography-ladder-seed"
        )
    if (
        args.congressional_district_vintage_crosswalk is not None
        and not args.assign_congressional_districts
    ):
        parser.error(
            "--congressional-district-vintage-crosswalk requires "
            "--assign-congressional-districts"
        )
    if args.assign_congressional_districts and args.ledger_facts is None:
        parser.error("--assign-congressional-districts requires --ledger-facts")
    if args.block_ladder_artifact is not None and (
        args.congressional_district_vintage_crosswalk is None
    ):
        parser.error(
            "--block-ladder-artifact requires "
            "--congressional-district-vintage-crosswalk: block sampling is "
            "conditioned on households carrying current-vintage districts"
        )
    if args.block_ladder_artifact is not None and args.without_block_ladder:
        parser.error(
            "--block-ladder-artifact and --without-block-ladder are contradictory"
        )
    if args.block_ladder_artifact is None and not args.without_block_ladder:
        parser.error(
            "US bases carry the block-anchored geography ladder by default "
            "(microcosm #275): pass --block-ladder-artifact <npz> "
            "(tools/build_us_block_ladder_artifact.py) or opt out "
            "explicitly with --without-block-ladder"
        )
    return _validate_common_args(parser, args)


# Why --location-clones > 1 is refused on this line (read this session): the
# release that consumes this base refuses duplicate copies of one source unit.
_LOCATION_CLONES_BLOCKERS = (
    "support_provenance.py:support_copy_rank_series admits only support clone "
    "indices 0, 1 and 2 on a historical frame",
    "sipp_head_start.py:_recipient_predictors (via with_us_sipp_head_start_input) "
    "refuses repeated (person_source_id, copy rank) pairs",
    "voluntary_filing.py:_source_receiver_rows refuses repeated "
    "(tax_unit_source_id, copy rank) pairs",
    "puf_capital_gains_tail.py:assert_puf_capital_gains_tail_survives_selection "
    "requires each capital-gains tail donor once among applied tax units",
)


def _validate_block_v1_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    if args.location_clones != 1:
        parser.error(
            "--location-clones must be 1 on the base-h5 line for now: the "
            "Route A release (tools/build_us_fiscal_refresh_release.py "
            "--base-h5) cannot yet admit location clones of one source "
            "household: " + "; ".join(_LOCATION_CLONES_BLOCKERS)
        )
    if args.location_seed < 0:
        parser.error("--location-seed must be non-negative")
    if args.block_ladder_artifact is None:
        parser.error(
            "--location-rule block_v1 requires --block-ladder-artifact: the "
            "household block draw samples the ladder's 2020 blocks"
        )
    if args.without_block_ladder:
        parser.error(
            "--location-rule block_v1 and --without-block-ladder are "
            "contradictory: every geography derives from the drawn block"
        )
    if args.assign_congressional_districts or args.ledger_facts is not None:
        parser.error(
            "--location-rule block_v1 replaces the SOI return-count "
            "congressional-district draw: drop --assign-congressional-"
            "districts and --ledger-facts (the district derives from the block)"
        )
    if args.congressional_district_seed != 0 or args.geography_ladder_seed != 0:
        parser.error(
            "--congressional-district-seed and --geography-ladder-seed seed "
            "the legacy draws; under --location-rule block_v1 pass "
            "--location-seed"
        )
    if args.congressional_district_vintage_crosswalk is None:
        parser.error(
            "--location-rule block_v1 requires "
            "--congressional-district-vintage-crosswalk: the Route A release "
            "checks the base's crosswalk sha and 119th-Congress target "
            "attributes before it runs"
        )
    if args.base_h5 is not None:
        parser.error(
            "--location-rule block_v1 requires --asec-h5 sources: a CPS "
            "household's candidate blocks come from its ASEC GTCO/GESTFIPS "
            "row, which a --base-h5 input does not carry"
        )


def _validate_common_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    if args.support_spine_spec is not None and args.asec_h5 is None:
        parser.error("--support-spine-spec requires --asec-h5")
    if args.stage != "all" and args.checkpoint_dir is None:
        parser.error("a named --stage requires --checkpoint-dir")
    if args.equivalence_boundary_dir is not None and (
        args.stage != "all" or args.checkpoint_dir is not None
    ):
        parser.error(
            "--equivalence-boundary-dir requires monolithic --stage all without "
            "--checkpoint-dir"
        )
    if (
        args.equivalence_deterministic_h5_metadata
        and args.equivalence_boundary_dir is None
        and args.checkpoint_dir is None
    ):
        parser.error(
            "--equivalence-deterministic-h5-metadata requires an equivalence "
            "boundary or checkpoint directory"
        )
    return args


def _asec_source_year(raw_year: str, *, flag: str, value: str) -> int:
    try:
        return int(raw_year)
    except ValueError:
        raise SystemExit(f"{flag} {value!r}: the year must be an integer.") from None


def _asec_source_paths(args: argparse.Namespace) -> dict[int, Path]:
    """Map each --asec-h5 YEAR=PATH to its path; refuse a malformed or repeated one."""

    try:
        return _parse_asec_source_paths(args.asec_h5)
    except ValueError as error:
        raise SystemExit(str(error)) from None


def _parse_asec_source_pins(args: argparse.Namespace) -> dict[int, str] | None:
    """Parse --asec-h5-sha256 into ``{year: lowercase sha256}`` without reading a file.

    Refuses a pin without --asec-h5, a value that is not YEAR=SHA256, a year
    that is not an integer, a digest that is not 64 hexadecimal characters, a
    year named twice, a year no --asec-h5 mapping provides, a pin set that
    leaves any --asec-h5 year unpinned (once one year is pinned, every year
    must be, so a build is either fully pinned or not pinned), and, for a
    year with a canonical digest in
    :data:`microcosm.build.us_runtime.asec_sources.ASEC_SOURCE_ARTIFACTS`, a
    pin that differs from it — the flag can narrow nothing and re-pin
    nothing. Returns ``None`` when no pin was given.
    """

    declared = getattr(args, "asec_h5_sha256", None) or ()
    if not declared:
        return None
    if args.asec_h5 is None:
        raise SystemExit("--asec-h5-sha256 requires --asec-h5.")
    paths = _asec_source_paths(args)
    pins: dict[int, str] = {}
    for value in declared:
        raw_year, separator, raw_sha = value.partition("=")
        if not separator:
            raise SystemExit(f"--asec-h5-sha256 must be YEAR=SHA256, got {value!r}.")
        year = _asec_source_year(raw_year, flag="--asec-h5-sha256", value=value)
        sha = raw_sha.strip().lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise SystemExit(
                f"--asec-h5-sha256 {value!r}: the digest must be 64 "
                "hexadecimal characters."
            )
        if year in pins:
            raise SystemExit(f"--asec-h5-sha256 names year {year} twice.")
        if year not in paths:
            raise SystemExit(
                f"--asec-h5-sha256 names year {year}, which no --asec-h5 "
                "mapping provides."
            )
        pins[year] = sha
    unpinned = sorted(set(paths) - set(pins))
    if unpinned:
        raise SystemExit(
            f"--asec-h5-sha256 pins {sorted(pins)} but --asec-h5 also names "
            f"{unpinned}; once any year is pinned, every year must be."
        )
    for year, sha in sorted(pins.items()):
        canonical = ASEC_SOURCE_ARTIFACTS.get(year)
        if canonical is not None and sha != canonical.sha256:
            raise SystemExit(
                f"ASEC {year} CLI pin differs from the canonical pin: got "
                f"{sha}, expected {canonical.sha256}."
            )
    return pins


def _require_receipt_digests_match_pins(
    args: argparse.Namespace, receipt: Mapping[str, object]
) -> None:
    """Refuse a source-construction receipt whose read bytes differ from the pins.

    The pre-stage check hashes each --asec-h5 file before any stage runs; the
    receipt records the digest of the bytes the stage actually read. With pins
    locked in the run config, the two must agree, so a file swapped between
    the check and the read, or a checkpointed receipt that disagrees with the
    locked pins on resume, refuses here at no extra I/O.
    """

    pins = _parse_asec_source_pins(args)
    if pins is None:
        return
    recorded = {
        int(entry["year"]): str(entry["sha256"]) for entry in receipt.get("sources", ())
    }
    for year, sha in sorted(pins.items()):
        actual = recorded.get(year)
        if actual is None:
            raise SystemExit(
                f"ASEC {year} is pinned but the source construction receipt "
                "records no digest for it."
            )
        if actual != sha:
            raise SystemExit(
                f"ASEC {year} source construction read bytes with sha256 "
                f"{actual}, but the locked pin is {sha}."
            )


def _require_completed_source_receipt_matches_pins(
    args: argparse.Namespace, stage_metadata: Mapping[str, object]
) -> None:
    """The completed-stage re-entry check: the receipt sits under ``base_source``.

    ``StageRuntime.metadata["source_construction"]`` is the stage-level mapping
    ``_source_construction_stage`` returned, and the pooled_asec receipt with
    its ``sources`` list is its ``base_source`` entry (the same level
    ``_ensure_asec_raw_stage_checkpoint`` reads).
    """

    receipt = stage_metadata.get("base_source")
    _require_receipt_digests_match_pins(
        args, receipt if isinstance(receipt, Mapping) else {}
    )


#: Steps that read the pinned ASEC files and hash them nowhere else, so they
#: hash them for the pin check themselves rather than take the parent's
#: digests. ``household_location_assignment`` reads the CPS household
#: geography from them under ``block_v1``. ``source_construction``, the
#: other step that reads them, hashes each file again once it has read it
#: and compares that digest with the pin
#: (``_require_receipt_digests_match_pins``).
_STEPS_HASHING_THEIR_ASEC_SOURCES = frozenset({HOUSEHOLD_LOCATION_STAGE_NAME})


def _verify_asec_source_digests(args: argparse.Namespace) -> dict[str, str]:
    """Refuse a raw ASEC input whose bytes differ from its declared digest.

    Every pin is parsed and compared with the canonical roster first
    (:func:`_parse_asec_source_pins`, which reads no file), then each pinned
    file is hashed in year order. Runs before any stage, so a wrong input
    refuses in seconds rather than surfacing as drift hours later. A step
    process of a ``--stage all`` run compares each pin with the digest its
    parent computed for the same resolved path (``PARENT_DIGESTS_VARIABLE``)
    instead of hashing the file, except in
    ``_STEPS_HASHING_THEIR_ASEC_SOURCES``.

    Returns each pinned file's digest under its resolved path, empty when
    nothing is pinned.
    """

    pins = _parse_asec_source_pins(args)
    if pins is None:
        return {}
    parent_digests = (
        None
        if getattr(args, "stage", "all") in _STEPS_HASHING_THEIR_ASEC_SOURCES
        else _parent_digests(args)
    )
    paths = _asec_source_paths(args)
    verified: dict[str, str] = {}
    for year, sha in sorted(pins.items()):
        path = paths[year]
        if not path.is_file():
            raise SystemExit(f"ASEC {year} source {path} does not exist.")
        actual = _input_sha256(path, parent_digests)
        if actual != sha:
            raise SystemExit(
                f"ASEC {year} source {path} failed digest "
                f"verification: expected {sha}, got {actual}."
            )
        verified[str(path.resolve())] = actual
    return verified


#: The telemetry pipeline this builder reports under.
TELEMETRY_PIPELINE = "us_puf_support_base"


def main(argv: list[str] | None = None) -> None:
    """Dispatch the byte-identical legacy path or checkpoint scaffolding.

    A child stage of a checkpointed ``--stage all`` run starts no emitter:
    its parent records its stage and outcome. Every other run starts one.
    """

    args = _parse_args() if argv is None else _parse_args(argv)
    if telemetry_recorded_by_parent():
        _dispatch(args, None)
        return
    with program_telemetry(pipeline=TELEMETRY_PIPELINE) as telemetry:
        _dispatch(args, telemetry)


def _dispatch(
    args: argparse.Namespace, telemetry: LocalTelemetryEmitter | None
) -> None:
    """Run the parsed command, recording stage changes on ``telemetry`` if any."""

    verified_asec_sha256 = _verify_asec_source_digests(args)
    stage = getattr(args, "stage", "all")
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    if stage != "all":
        hash_seed = os.environ.get("PYTHONHASHSEED")
        if (
            hash_seed is None
            or not hash_seed.isdigit()
            or int(hash_seed) > 4_294_967_295
        ):
            raise SystemExit(
                "Named staged builds require an explicit PYTHONHASHSEED; use "
                "--stage all to default fresh child processes to 0."
            )
        if telemetry is not None:
            telemetry.transition_stage(stage)
        _run_configured_stage(args)
        return
    if checkpoint_dir is None:
        # This is intentionally a direct call with no profiler, checkpoint
        # directory creation, or other side effect, except telemetry (the
        # emitter started in main() and the one stage change below) and the
        # gate battery's shadow files beside the outputs, none of which
        # changes an output byte. It is the pre-refactor pipeline and remains
        # the byte-for-byte compatibility path.
        if telemetry is not None:
            telemetry.transition_stage("all")
        boundary_dir = getattr(args, "equivalence_boundary_dir", None)
        observer = (
            None
            if boundary_dir is None
            else _EquivalenceBoundaryObserver(
                boundary_dir, pipeline_steps=_pipeline_steps_for(args)
            )
        )
        with _armed_base_gate_shadow(args):
            if observer is None:
                _run_all(args)
            else:
                _run_all(args, boundary_observer=observer)
        if observer is not None:
            observer.assert_complete()
        return
    _run_staged_all(args, telemetry, verified_asec_sha256=verified_asec_sha256)


def _run_staged_all(
    args: argparse.Namespace,
    telemetry: LocalTelemetryEmitter | None = None,
    *,
    verified_asec_sha256: Mapping[str, str] | None = None,
) -> None:
    """Run the pipeline's steps in fresh interpreters, resuming by prefix.

    Each step still to run gets a child interpreter of its own, except that
    the ``ONE_PROCESS_STEPS`` still to run share one
    (``_run_one_process_steps``). ``telemetry`` (the parent's, if any)
    records each step's stage as it starts; a child that exits nonzero stops
    the run, which main() records as the failure of the step it was running.
    The children start no emitter.
    ``verified_asec_sha256`` holds the digests this process's pin check
    computed (``_verify_asec_source_digests``), which the children's pin
    checks use, except in ``_STEPS_HASHING_THEIR_ASEC_SOURCES``.
    """

    pipeline_steps = _pipeline_steps_for(args)
    run_config = _stage_run_config(args)
    runtime = _open_stage_runtime(args, run_config)
    # Each step process takes the PUF and ladder digests from this run
    # config, which the resume check just compared with the locked one, and
    # the ASEC digests from the pin check, instead of hashing those files
    # again; household_location_assignment hashes the ASEC files itself, and
    # every step hashes the builder's source files itself.
    digests_environment = _parent_digests_environment(
        run_config, verified_asec_sha256 or {}
    )
    completed_prefix = runtime.context.completed
    remaining = pipeline_steps[len(completed_prefix) :]
    if not remaining and completed_prefix == pipeline_steps:
        # Re-enter only the fresh final child so it can validate/repair the
        # export and stage_all alias after a crash in the post-context window.
        remaining = ("final_export",)
    # Each child observes its own stage's gates; this parent finishes the
    # shadow scope from what they saved, completed stages included.
    directory = _final_gate_shadow_directory(args)
    shadow = None if directory is None else _base_gate_shadow(args, directory)
    # The stages that ran, in an earlier run of this checkpoint or here: each
    # one should have saved its observations.
    stages_run = list(completed_prefix)
    try:
        for steps in _step_process_runs(remaining):
            stage = steps[0]
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *_stage_cli_args(args, stage),
            ]
            environment = {**_staged_subprocess_environment(), **digests_environment}
            # Only the child for several of ONE_PROCESS_STEPS gets a report
            # pipe; a value this process inherited goes to no child.
            environment.pop(STEP_REPORT_VARIABLE, None)
            if len(steps) > 1:
                _run_one_process_steps(
                    command,
                    environment,
                    steps,
                    telemetry=telemetry,
                    stages_run=stages_run,
                )
                continue
            if stage not in stages_run:
                stages_run.append(stage)
            if telemetry is not None:
                telemetry.transition_stage(stage)
            completed = subprocess.run(command, check=False, env=environment)
            if completed.returncode != 0:
                raise SystemExit(
                    f"Checkpointed stage {stage!r} failed with exit code "
                    f"{completed.returncode}."
                )
    except BaseException as error:
        if shadow is not None:
            _load_stage_gate_shadows(shadow, args, stages_run)
            shadow.stop(error)
        raise
    if shadow is not None:
        _load_stage_gate_shadows(shadow, args, stages_run)
        shadow.finish()


def _step_process_runs(remaining: Sequence[str]) -> list[tuple[str, ...]]:
    """The steps still to run, grouped by the step process that runs them.

    Each step has a process of its own, except that the ``ONE_PROCESS_STEPS``
    in ``remaining`` (a suffix of the pipeline, so from one of them through
    the last) share one.
    """

    runs: list[tuple[str, ...]] = []
    for step in remaining:
        if runs and step in ONE_PROCESS_STEPS and runs[-1][-1] in ONE_PROCESS_STEPS:
            runs[-1] = (*runs[-1], step)
        else:
            runs.append((step,))
    for run in runs:
        if len(run) > 1 and run != ONE_PROCESS_STEPS[ONE_PROCESS_STEPS.index(run[0]) :]:
            raise AssertionError(
                f"steps {run} are not the ONE_PROCESS_STEPS from {run[0]!r} on."
            )
    return runs


def _run_one_process_steps(
    command: Sequence[str],
    environment: Mapping[str, str],
    steps: Sequence[str],
    *,
    telemetry: LocalTelemetryEmitter | None,
    stages_run: list[str],
) -> None:
    """Run ``steps`` in the one child ``command`` starts, step by step.

    The child, started with ``STEP_REPORT_VARIABLE``, writes each step's name
    to a pipe as the step starts (``_report_step_start``). Each step is added
    to ``stages_run`` and recorded on ``telemetry`` as it starts: the first
    before the child starts, as for any child, and each later one when the
    child reports it, so a failure is recorded against the step that failed.
    A report that is not the next step stops the child and raises
    ``RuntimeError``; so does a child that exits cleanly before its last
    step started. As with ``subprocess.run``, an exception raised here while
    the child runs (a SIGTERM the build turns into one, say) stops the child
    before it propagates; after a Ctrl-C, which the child gets too, it first
    waits up to ``_SIGINT_WAIT_SECONDS`` for the child to finish its own
    handling.
    """

    def step_started(step: str) -> None:
        if step not in stages_run:
            stages_run.append(step)
        if telemetry is not None:
            telemetry.transition_stage(step)

    step_started(steps[0])
    read_fd, write_fd = _report_pipe_fds()
    try:
        process = subprocess.Popen(
            list(command),
            env={**environment, **_step_report_environment(write_fd)},
            pass_fds=(write_fd,),
        )
    except BaseException:
        os.close(read_fd)
        raise
    finally:
        # The child has its own copy; the reports end when it exits.
        os.close(write_fd)
    with process, os.fdopen(read_fd, "rb") as reports:
        try:
            reported = _follow_step_reports(reports, steps, step_started)
            returncode = process.wait()
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt):
                try:
                    process.wait(timeout=_SIGINT_WAIT_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
            process.kill()
            raise
    if returncode != 0:
        raise SystemExit(
            f"Checkpointed stage {steps[max(reported - 1, 0)]!r} failed with "
            f"exit code {returncode}."
        )
    if reported == 0:
        raise RuntimeError(
            f"The step process for {steps[0]!r} through {steps[-1]!r} exited "
            "cleanly without reporting any step: a step process that finds no "
            f"report pipe in {STEP_REPORT_VARIABLE} runs one step and reports "
            "nothing."
        )
    if reported < len(steps):
        raise RuntimeError(
            f"The step process for {steps[0]!r} through {steps[-1]!r} exited "
            f"cleanly before step {steps[reported]!r} started."
        )


#: How long the parent waits for the child after a Ctrl-C before stopping
#: it: what ``Popen.wait`` waits under ``subprocess.run``, which starts the
#: single-step children (``Popen._sigint_wait_secs``).
_SIGINT_WAIT_SECONDS = 0.25


def _report_pipe_fds() -> tuple[int, int]:
    """A new pipe for step reports, neither end at descriptor 0, 1 or 2.

    ``os.pipe`` returns the lowest free descriptors. In a process started
    with a standard stream closed they include that stream's, and a write
    end there would become the child's standard input, output or error.
    Both ends are non-inheritable, as ``os.pipe``'s are.
    """

    read_fd, write_fd = os.pipe()
    if min(read_fd, write_fd) > 2:
        return read_fd, write_fd
    moved: list[int] = []
    try:
        for fd in (read_fd, write_fd):
            moved.append(fcntl.fcntl(fd, fcntl.F_DUPFD_CLOEXEC, 3))
    except BaseException:
        for fd in moved:
            os.close(fd)
        raise
    finally:
        os.close(read_fd)
        os.close(write_fd)
    return moved[0], moved[1]


def _follow_step_reports(
    reports: Iterable[bytes],
    steps: Sequence[str],
    step_started: Callable[[str], None],
) -> int:
    """Call ``step_started`` for each later step the child reports starting.

    Returns how many of ``steps`` the child reported, the first included.
    """

    reported = 0
    for line in reports:
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError:
            text = ""
        if not text.endswith("\n"):
            raise RuntimeError(
                f"The step process for {steps[0]!r} through {steps[-1]!r} wrote "
                f"a report that is not a step name and a newline: {line!r}."
            )
        name = text[:-1]
        if reported == len(steps):
            raise RuntimeError(
                f"The step process for {steps[0]!r} through {steps[-1]!r} "
                f"reported {name!r} after its last step."
            )
        if name != steps[reported]:
            raise RuntimeError(
                f"The step process for {steps[0]!r} through {steps[-1]!r} "
                f"reported {name!r} as starting; expected {steps[reported]!r}."
            )
        if reported > 0:
            step_started(name)
        reported += 1
    return reported


#: Set by a ``--stage all`` parent for the child it starts for several of
#: ``ONE_PROCESS_STEPS``: the write end of a pipe the child inherits, and the
#: parent's process ID. A step process uses it only when the process that set
#: it is its parent (the recorded ID equals ``os.getppid()``) and its
#: ``--stage`` is one of ``ONE_PROCESS_STEPS``. It then runs from that step
#: through the last of them, writing each step's name and a newline to the
#: pipe as the step starts. Otherwise it runs its one step, as a named
#: ``--stage`` run by hand does. A malformed variable stops any step process
#: that finds it; one naming a descriptor that is not an open pipe stops a
#: process the variable would take effect in. The process-ID comparison does not identify the run: a copy of the variable
#: used where process IDs repeat (a fresh container) can match, as
#: ``MICROCOSM_TELEMETRY_PARENT_PID`` can.
STEP_REPORT_VARIABLE = "MICROCOSM_US_BASE_STEP_REPORT"
_STEP_REPORT_SCHEMA_VERSION = 1
_STEP_REPORT_KEYS = frozenset({"fd", "parent_pid", "schema_version"})


def _step_report_environment(report_fd: int) -> dict[str, str]:
    """``STEP_REPORT_VARIABLE`` naming ``report_fd`` and this process."""

    payload = {
        "fd": report_fd,
        "parent_pid": os.getpid(),
        "schema_version": _STEP_REPORT_SCHEMA_VERSION,
    }
    return {STEP_REPORT_VARIABLE: json.dumps(payload, sort_keys=True)}


def _step_report_fd(stage: str) -> int | None:
    """The pipe this step process reports its steps on, or ``None``.

    ``None`` unless this process's parent set ``STEP_REPORT_VARIABLE`` and
    ``stage`` is one of ``ONE_PROCESS_STEPS``. A variable that names a
    descriptor that is not an open pipe stops the process.
    """

    raw = os.environ.get(STEP_REPORT_VARIABLE)
    if raw is None:
        return None
    payload = _parse_step_report(raw)
    if payload["parent_pid"] != os.getppid() or stage not in ONE_PROCESS_STEPS:
        return None
    report_fd = payload["fd"]
    try:
        mode = os.fstat(report_fd).st_mode
    except OSError:
        mode = None
    if mode is None or not stat.S_ISFIFO(mode):
        raise SystemExit(
            f"{STEP_REPORT_VARIABLE} names file descriptor {report_fd}, which is "
            f"{'not open' if mode is None else 'not a pipe'} in this process. A "
            "--stage all run sets it for its step processes; unset it to run a "
            "named --stage by hand."
        )
    return report_fd


def _parse_step_report(raw: str) -> dict[str, int]:
    """Validate ``STEP_REPORT_VARIABLE``, refusing what no parent writes."""

    def refuse(problem: str) -> NoReturn:
        raise SystemExit(
            f"{STEP_REPORT_VARIABLE} {problem}. A --stage all run sets it "
            "for its step processes; unset it to run a named --stage by hand."
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        refuse(f"is not JSON ({error})")
    if not isinstance(payload, dict) or set(payload) != _STEP_REPORT_KEYS:
        refuse(
            f"is not a JSON object with exactly the keys {sorted(_STEP_REPORT_KEYS)}"
        )
    if (
        type(payload["schema_version"]) is not int
        or payload["schema_version"] != _STEP_REPORT_SCHEMA_VERSION
    ):
        refuse(
            f"has schema_version {payload['schema_version']!r}; this tool "
            f"reads {_STEP_REPORT_SCHEMA_VERSION}"
        )
    if type(payload["fd"]) is not int or payload["fd"] < 0:
        refuse(f"has an fd that is not a file descriptor: {payload['fd']!r}")
    if type(payload["parent_pid"]) is not int or payload["parent_pid"] < 1:
        refuse(f"has a parent_pid that is not a process ID: {payload['parent_pid']!r}")
    return payload


def _report_step_start(report_fd: int, step: str) -> None:
    """Tell the parent, on the pipe it opened, that ``step`` starts."""

    message = f"{step}\n".encode()
    while message:
        message = message[os.write(report_fd, message) :]


#: This tool's scope of us/gates.json, evaluated in shadow beside today's
#: gates (microcosm#1585 item 11, PR 2 of 4).
US_BASE_GATE_SCOPE = "us_base"
_GEOGRAPHY_LADDER_WAIVER = "--allow-geography-ladder-gate-failures"


def _final_gate_shadow_directory(args: argparse.Namespace) -> Path | None:
    """Where the finished shadow files go: beside the outputs.

    ``None``, and the run goes without a shadow, when the switch is off (the
    test suite) or for stand-in arguments without ``--out``, which have
    nowhere to write; the parser requires ``--out`` of every real run.
    """

    out = getattr(args, "out", None)
    if out is None or not tool_gate_shadows_enabled():
        return None
    return Path(out).resolve() / GATE_BATTERY_SHADOW_DIRECTORY


def _stage_gate_shadow_observations(args: argparse.Namespace, stage: str) -> Path:
    return (
        args.checkpoint_dir
        / GATE_BATTERY_SHADOW_DIRECTORY
        / f"{stage}.observations.json"
    )


def _base_gate_shadow(args: argparse.Namespace, directory: Path) -> GateBatteryShadow:
    """This build's gate battery in shadow, its files in ``directory``."""

    report_path, comparison_path = shadow_file_paths(directory, US_BASE_GATE_SCOPE)
    waived = (
        ("us_base_geography_ladder", "us_base_block_location")
        if getattr(args, "allow_geography_ladder_gate_failures", False)
        else ()
    )
    target_year = getattr(args, "target_year", PERIOD)
    return GateBatteryShadow(
        us_scoped_gate_manifest(US_BASE_GATE_SCOPE),
        scope=US_BASE_GATE_SCOPE,
        release_id=f"populace-us-{target_year}-puf-support-base",
        report_path=report_path,
        comparison_path=comparison_path,
        registry=us_gate_registry(US_BASE_GATE_SCOPE, tool=globals()),
        waivers={entry_id: (_GEOGRAPHY_LADDER_WAIVER,) for entry_id in waived},
    )


@contextmanager
def _armed_base_gate_shadow(
    args: argparse.Namespace,
) -> Iterator[GateBatteryShadow | None]:
    """The single-process run's shadow, finished when the block exits."""

    directory = _final_gate_shadow_directory(args)
    if directory is None:
        yield None
        return
    shadow = _base_gate_shadow(args, directory)
    with shadow, active_gate_shadow(shadow):
        yield shadow


@contextmanager
def _armed_stage_gate_shadow(args: argparse.Namespace) -> Iterator[Callable[[], None]]:
    """Arm one checkpointed stage's shadow; yield the call that saves it.

    The stage saves before it marks itself complete, so a resumed build finds
    the observations of every stage it does not run again. A stage that
    raises saves what it saw on the way out, so the parent's report says
    where the build stopped. A save that fails warns and is otherwise left to
    the parent, which then notes that the stage saved nothing and reports its
    entries as unobserved. With
    the switch off the stage runs without a shadow and saves nothing.
    """

    if not tool_gate_shadows_enabled():
        yield lambda: None
        return
    shadow = _base_gate_shadow(
        args, args.checkpoint_dir / GATE_BATTERY_SHADOW_DIRECTORY / args.stage
    )
    path = _stage_gate_shadow_observations(args, args.stage)
    saved = False

    def save() -> None:
        nonlocal saved
        if saved:
            return
        saved = True
        try:
            shadow.save_observations(path)
        except Exception as error:  # noqa: BLE001 - the shadow never stops a stage
            print(
                f"warning: could not save gate battery shadow observations to "
                f"{path}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )

    with active_gate_shadow(shadow):
        try:
            yield save
        except BaseException:
            save()
            raise
    save()


def _load_stage_gate_shadows(
    shadow: GateBatteryShadow,
    args: argparse.Namespace,
    stages_run: Sequence[str],
) -> None:
    """Load what each stage that ran saved.

    A stage that ran but saved nothing (its save failed, or it ran with the
    shadow off) is noted, so a stopped run never names its entries unreached.
    """

    for stage in stages_run:
        path = _stage_gate_shadow_observations(args, stage)
        if path.is_file():
            shadow.load_observations(path)
        else:
            shadow.note_missing_observations(f"the {stage} stage saved no observations")


def _shadow_frame_gate(
    entry_id: str, evaluate: Callable[[], GateResult], frame: Frame
) -> GateResult:
    """Today's frame signal gate, with the battery's entry beside it."""

    return shadow_gate(entry_id, evaluate, lambda: EvidenceContext(frame=frame))


def _observe_frame_gate(entry_id: str, result: GateResult, frame: Frame) -> None:
    """Observe a frame gate verdict today's code reached by itself."""

    shadow = current_gate_shadow(entry_id)
    if shadow is None:
        return
    try:
        legacy = LegacyVerdict.from_result(result)
    except TypeError as error:
        shadow.record_error(entry_id, f"today's verdict is not comparable: {error}")
        return
    shadow.check(entry_id, lambda: EvidenceContext(frame=frame), legacy)


def _shadow_legacy_location_not_run(args: argparse.Namespace) -> None:
    """The location entries the legacy location rule never evaluates."""

    shadow_not_run("us_base_block_location", "the legacy location rule draws no block")
    if getattr(args, "block_ladder_artifact", None) is None:
        shadow_not_run(
            "us_base_geography_ladder",
            "the legacy location rule assigns no geography ladder without "
            "--block-ladder-artifact",
        )


def _stage_cli_args(args: argparse.Namespace, stage: str) -> list[str]:
    """Reconstruct one stage invocation from validated parsed arguments."""

    command: list[str] = []
    if args.base_h5 is not None:
        command.extend(("--base-h5", str(args.base_h5)))
    else:
        for value in args.asec_h5:
            command.extend(("--asec-h5", value))
        for value in getattr(args, "asec_h5_sha256", None) or ():
            command.extend(("--asec-h5-sha256", value))
    command.extend(("--target-year", str(args.target_year)))
    if args.asec_max_households is not None:
        command.extend(("--asec-max-households", str(args.asec_max_households)))
    _append_path_argument(command, "--support-spine-spec", args.support_spine_spec)
    command.extend(("--puf-h5", str(args.puf_h5)))
    _append_path_argument(
        command,
        "--puf-source-year-csv",
        args.puf_source_year_csv,
    )
    _append_path_argument(
        command,
        "--asec-2023-weeks-unemployed-source",
        args.asec_2023_weeks_unemployed_source,
    )
    for value in args.asec_education_source or ():
        command.extend(("--asec-education-source", value))
    _append_path_argument(command, "--acs-h5", args.acs_h5)
    command.extend(("--out", str(args.out)))
    command.extend(("--stage", stage))
    command.extend(("--checkpoint-dir", str(args.checkpoint_dir)))
    command.extend(("--seed", str(args.seed)))
    command.extend(("--n-estimators", str(args.n_estimators)))
    _append_path_argument(command, "--ledger-facts", args.ledger_facts)
    if args.assign_congressional_districts:
        command.append("--assign-congressional-districts")
    _append_path_argument(
        command,
        "--congressional-district-vintage-crosswalk",
        args.congressional_district_vintage_crosswalk,
    )
    command.extend(
        ("--congressional-district-seed", str(args.congressional_district_seed))
    )
    _append_path_argument(
        command, "--block-ladder-artifact", args.block_ladder_artifact
    )
    if args.without_block_ladder:
        command.append("--without-block-ladder")
    command.extend(("--geography-ladder-seed", str(args.geography_ladder_seed)))
    if args.allow_geography_ladder_gate_failures:
        command.append("--allow-geography-ladder-gate-failures")
    if _is_block_v1(args):
        # Emitted only off the legacy rule, so legacy child command lines
        # stay byte-identical.
        command.extend(("--location-rule", _location_rule(args)))
        command.extend(("--location-seed", str(args.location_seed)))
        command.extend(("--location-clones", str(args.location_clones)))
    if getattr(args, "equivalence_deterministic_h5_metadata", False):
        command.append("--equivalence-deterministic-h5-metadata")
    return command


def _append_path_argument(command: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _staged_subprocess_environment() -> dict[str, str]:
    from microcosm.build.us_runtime.take_up_contract import (
        us_engine_abi_lock_check_environment,
    )

    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "0")
    # Each child compares the policyengine-us lock file and engine version with
    # this process's check instead of deriving the lock again.
    environment.update(us_engine_abi_lock_check_environment())
    # The child starts no emitter: this process records its telemetry.
    return child_stage_environment(environment)


def _asec_education_source_paths(
    args: argparse.Namespace,
) -> dict[int, Path] | None:
    """Parse --asec-education-source INCOME_YEAR=PATH mappings."""

    if not getattr(args, "asec_education_source", None):
        return None
    paths: dict[int, Path] = {}
    for value in args.asec_education_source:
        raw_year, _, raw_path = value.partition("=")
        if not raw_path:
            raise SystemExit(
                "--asec-education-source values must be INCOME_YEAR=PATH, got "
                f"{value!r}."
            )
        year = int(raw_year)
        if year in paths:
            raise SystemExit(f"--asec-education-source repeats income year {year}.")
        paths[year] = Path(raw_path)
    return paths


def _pooled_income_years(args: argparse.Namespace) -> tuple[int, ...]:
    """Income years of the pooled ASEC inputs, from the --asec-h5 mappings."""

    years = []
    for value in getattr(args, "asec_h5", None) or ():
        raw_year, _, _ = value.partition("=")
        years.append(int(raw_year))
    return tuple(sorted(set(years)))


#: QRF thread counts the first run records and a resume does not compare: they
#: change no imputed value. Everything else in ``thread_environment`` stays
#: compared, including ``blas_threads_in_use``: on Linux the OpenBLAS count
#: changes output bytes, and with no thread variable set it follows the host's
#: CPUs, so a resume on a host where OpenBLAS uses a different count is refused.
_RESUME_IGNORED_RUN_CONFIG_PATHS = (
    ("thread_environment", "POPULACE_FIT_N_JOBS"),
    ("thread_environment", "POPULACE_FIT_PREDICT_WORKERS"),
)


def _blas_threads_in_use() -> list[list[object]]:
    """Each loaded BLAS library, as ``[name, thread count]``, in sorted order.

    threadpoolctl reads the counts from the libraries themselves, so this holds
    whatever chose them: a thread variable, ``MICROCOSM_THREADS`` or, with
    neither, the CPUs OpenBLAS found on the host. It does not report Apple's
    Accelerate, so where numpy and scipy use Accelerate, as on Apple-silicon
    Macs, the list is empty.
    """

    return sorted(
        [pool["internal_api"], pool["num_threads"]]
        for pool in threadpool_info()
        if pool["user_api"] == "blas"
    )


def _open_stage_runtime(
    args: argparse.Namespace, run_config: Mapping[str, object] | None = None
) -> StageRuntime:
    """Open the checkpoint directory, refusing a resume with changed settings.

    ``run_config`` is ``_stage_run_config(args)``, when the caller has it.
    """

    return StageRuntime(
        args.checkpoint_dir,
        _pipeline_for(args),
        run_config=_stage_run_config(args) if run_config is None else run_config,
        resume_ignored_run_config_paths=_RESUME_IGNORED_RUN_CONFIG_PATHS,
    )


def _stage_run_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the canonical inputs and settings recorded by the first run.

    A resume must supply equal values, except for
    ``_RESUME_IGNORED_RUN_CONFIG_PATHS``. A step process of a ``--stage all``
    run takes the PUF and location ladder digests its parent passed
    (``PARENT_DIGESTS_VARIABLE``) instead of hashing those files again; it
    hashes the builder's source files itself.
    """

    def path(value: Path | None) -> str | None:
        return None if value is None else str(value.resolve())

    parent_digests = _parent_digests(args)

    asec_sources: list[str] | None = None
    if args.asec_h5 is not None:
        asec_sources = []
        for value in args.asec_h5:
            raw_year, raw_path = value.split("=", 1)
            asec_sources.append(f"{int(raw_year)}={Path(raw_path).resolve()}")
    asec_pins = _parse_asec_source_pins(args)
    thread_environment = {
        name: (
            os.environ.get(name, "0")
            if name == "PYTHONHASHSEED"
            else os.environ.get(name)
        )
        for name in (
            "MICROCOSM_THREADS",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "POPULACE_FIT_N_JOBS",
            "POPULACE_FIT_PREDICT_WORKERS",
            "PYTHONHASHSEED",
        )
    }
    thread_environment["blas_threads_in_use"] = _blas_threads_in_use()
    config: dict[str, object] = {
        "allow_geography_ladder_gate_failures": bool(
            args.allow_geography_ladder_gate_failures
        ),
        "asec_h5": asec_sources,
        "asec_h5_sha256": (
            None
            if asec_pins is None
            else {str(year): sha for year, sha in sorted(asec_pins.items())}
        ),
        "asec_max_households": args.asec_max_households,
        "asec_2023_weeks_unemployed_source": path(
            args.asec_2023_weeks_unemployed_source
        ),
        "asec_education_source": (
            None
            if _asec_education_source_paths(args) is None
            else {
                str(year): str(source_path.resolve())
                for year, source_path in sorted(
                    _asec_education_source_paths(args).items()
                )
            }
        ),
        "acs_h5": path(args.acs_h5),
        "assign_congressional_districts": bool(args.assign_congressional_districts),
        "base_h5": path(args.base_h5),
        "block_ladder_artifact": path(args.block_ladder_artifact),
        "builder_code_identity": _builder_code_identity(),
        "congressional_district_seed": args.congressional_district_seed,
        "congressional_district_vintage_crosswalk": path(
            args.congressional_district_vintage_crosswalk
        ),
        "geography_ladder_seed": args.geography_ladder_seed,
        "equivalence_deterministic_h5_metadata": bool(
            getattr(args, "equivalence_deterministic_h5_metadata", False)
        ),
        "ledger_facts": path(args.ledger_facts),
        "n_estimators": args.n_estimators,
        "out": path(args.out),
        "puf_h5": path(args.puf_h5),
        "puf_h5_sha256": (
            _input_sha256(args.puf_h5, parent_digests)
            if args.puf_h5 is not None and Path(args.puf_h5).is_file()
            else None
        ),
        "puf_source_year_csv": path(args.puf_source_year_csv),
        "puf_source_year_csv_sha256": (
            _input_sha256(args.puf_source_year_csv, parent_digests)
            if args.puf_source_year_csv is not None
            else None
        ),
        "seed": args.seed,
        "support_spine_spec": path(args.support_spine_spec),
        "target_year": args.target_year,
        "thread_environment": thread_environment,
        "without_block_ladder": bool(args.without_block_ladder),
    }
    if not tool_gate_shadows_enabled():
        # microcosm#1585 item 11: a stage run with the gate battery's shadow
        # off saves no observations, and the run that finishes the scope
        # could then only call its entries unwired. The switch is locked
        # when off only, so a resume cannot mix the two, and a real build's
        # config, with the shadow on, is unchanged.
        config["gate_battery_shadow"] = "off"
    if _is_block_v1(args):
        # Off the legacy rule only, so a legacy run config (and every legacy
        # checkpoint directory that locked one) is unchanged. The ladder's
        # digest is locked too: a ladder rewritten in place at the same path
        # would silently move every drawn block.
        config.update(
            {
                "location_rule": _location_rule(args),
                "location_seed": int(args.location_seed),
                "location_clones": int(args.location_clones),
                "block_ladder_artifact_sha256": (
                    _input_sha256(args.block_ladder_artifact, parent_digests)
                    if Path(args.block_ladder_artifact).is_file()
                    else None
                ),
            }
        )
    return config


def _builder_code_identity() -> dict[str, object]:
    """Fingerprint executable sources and dependency versions for safe resume."""

    from microcosm.build.code_identity import builder_code_identity

    return builder_code_identity(
        Path(__file__).resolve().parents[1],
        tool_path=Path(__file__).resolve(),
        distributions=(
            "h5py",
            "numpy",
            "pandas",
            "policyengine-us",
            "quantile-forest",
            "scikit-learn",
        ),
    )


#: The variable in which a ``--stage all`` parent passes each step process,
#: as JSON, the SHA-256 of the input files it hashed, each under the file's
#: resolved path: the PUF H5 and CSV and, under ``block_v1``, the location
#: ladder (from its run config), and each pinned ASEC file (from its pin
#: check). So the step's own run config (``_stage_run_config``) and pin check
#: (``_verify_asec_source_digests``) do not read those files again, except
#: where a step hashes the ASEC files itself
#: (``_STEPS_HASHING_THEIR_ASEC_SOURCES``). The builder's source files are not
#: passed: every step hashes them again (about 0.12 s), so a ``git pull`` or
#: an edit in the checkout after the parent started is refused by the next
#: step. A step process uses a digest only for the same path, and only when
#: the process that set the variable is its parent (the recorded process ID
#: equals ``os.getppid()``); otherwise it hashes the file itself, as a named
#: ``--stage`` run by hand does when the variable is unset. A malformed
#: variable stops the process. The parent ignores the variable and hashes
#: these files at every start, comparing the result with the locked run
#: config and the ASEC pins before it starts a step, so a resume after one of
#: them changed is still refused. (Other inputs, such as ``--acs-h5``, are
#: recorded by path only, as before.) What goes unnoticed, in the steps that
#: take the parent's digests, is one of these input files changing while the
#: run is under way. The process-ID comparison does not identify the run: a
#: copy of the variable used where process IDs repeat (a fresh container)
#: can match, as ``MICROCOSM_TELEMETRY_PARENT_PID`` can.
PARENT_DIGESTS_VARIABLE = "MICROCOSM_US_BASE_PARENT_DIGESTS"
_PARENT_DIGESTS_SCHEMA_VERSION = 1
_PARENT_DIGESTS_KEYS = frozenset({"file_sha256", "parent_pid", "schema_version"})
#: The run-config keys of each input path the parent passes a digest for,
#: and of that digest. The ladder's digest is in the run config only under
#: ``block_v1``.
_PARENT_DIGEST_FILE_KEYS = (
    ("puf_h5", "puf_h5_sha256"),
    ("puf_source_year_csv", "puf_source_year_csv_sha256"),
    ("block_ladder_artifact", "block_ladder_artifact_sha256"),
)


def _parent_digests_environment(
    run_config: Mapping[str, object], verified_asec_sha256: Mapping[str, str]
) -> dict[str, str]:
    """``PARENT_DIGESTS_VARIABLE`` carrying this process's input digests.

    They are ``run_config``'s and ``verified_asec_sha256``, the pinned ASEC
    files' digests under their resolved paths.
    """

    file_sha256 = dict(verified_asec_sha256)
    for path_key, digest_key in _PARENT_DIGEST_FILE_KEYS:
        path, digest = run_config.get(path_key), run_config.get(digest_key)
        if isinstance(path, str) and isinstance(digest, str):
            file_sha256[path] = digest
    payload = {
        "file_sha256": file_sha256,
        "parent_pid": os.getpid(),
        "schema_version": _PARENT_DIGESTS_SCHEMA_VERSION,
    }
    return {PARENT_DIGESTS_VARIABLE: json.dumps(payload, sort_keys=True)}


def _parent_digests(args: argparse.Namespace) -> Mapping[str, str] | None:
    """The input digests this step process's parent passed it, or ``None``.

    The digests are keyed by resolved path. ``None`` for a ``--stage all``
    run, which hashes its own inputs, and for a process whose parent did not
    set ``PARENT_DIGESTS_VARIABLE``.
    """

    if getattr(args, "stage", "all") == "all":
        return None
    raw = os.environ.get(PARENT_DIGESTS_VARIABLE)
    if raw is None:
        return None
    payload = _parse_parent_digests(raw)
    if payload["parent_pid"] != os.getppid():
        return None
    return payload["file_sha256"]


def _parse_parent_digests(raw: str) -> dict[str, object]:
    """Validate ``PARENT_DIGESTS_VARIABLE``, refusing what no parent writes."""

    def refuse(problem: str) -> NoReturn:
        raise SystemExit(
            f"{PARENT_DIGESTS_VARIABLE} {problem}. A --stage all run sets it "
            "for its step processes; unset it to run a named --stage by hand."
        )

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        refuse(f"is not JSON ({error})")
    if not isinstance(payload, dict) or set(payload) != _PARENT_DIGESTS_KEYS:
        refuse(
            f"is not a JSON object with exactly the keys {sorted(_PARENT_DIGESTS_KEYS)}"
        )
    if payload["schema_version"] != _PARENT_DIGESTS_SCHEMA_VERSION:
        refuse(
            f"has schema_version {payload['schema_version']!r}; this tool "
            f"reads {_PARENT_DIGESTS_SCHEMA_VERSION}"
        )
    if type(payload["parent_pid"]) is not int:
        refuse("has a parent_pid that is not an integer")
    files = payload["file_sha256"]
    if not isinstance(files, dict) or not all(
        Path(path).is_absolute() and _is_sha256(digest)
        for path, digest in files.items()
    ):
        refuse(
            "has a file_sha256 that does not map absolute paths to lowercase "
            "SHA-256 digests"
        )
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _input_sha256(path: Path, parent_digests: Mapping[str, str] | None) -> str:
    """``path``'s SHA-256: the parent's digest for its resolved path, else hashed."""

    if parent_digests is not None:
        digest = parent_digests.get(str(Path(path).resolve()))
        if digest is not None:
            return digest
    return _sha256(path)


class _EquivalenceBoundaryObserver:
    """Write test-only monolith snapshots without changing production stages."""

    def __init__(
        self,
        root: Path,
        *,
        pipeline_steps: tuple[str, ...] = PIPELINE_STEPS,
    ) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._next_stage_index = 0
        self.pipeline_steps = pipeline_steps

    def observe_frame(self, stage: str, frame: Frame) -> None:
        expected = self.pipeline_steps[self._next_stage_index]
        if stage != expected:
            raise AssertionError(
                f"Monolith boundary order changed: expected {expected!r}, got "
                f"{stage!r}."
            )
        write_frame_checkpoint(
            self.root / f"{self._next_stage_index:03d}_{stage}.frame.h5",
            frame,
            metadata={
                "artifact_kind": "populace_monolith_equivalence_boundary",
                "pipeline_steps": list(self.pipeline_steps),
                "stage": stage,
                "stage_index": self._next_stage_index,
            },
        )
        self._next_stage_index += 1

    def observe_primary_qrf(
        self,
        frame: Frame,
        raw_predictions: pd.DataFrame,
    ) -> None:
        self.observe_frame("primary_qrf_chain", frame)
        target_dir = self.root / "primary_qrf" / "targets"
        target_dir.mkdir(parents=True, exist_ok=True)
        for target_index, target in enumerate(raw_predictions.columns):
            safe_target = "".join(
                character if character.isalnum() or character in "-_" else "_"
                for character in target
            )
            path = target_dir / f"{target_index:03d}__{safe_target}.h5"
            temporary = path.with_name(f".{path.name}.tmp")
            temporary.unlink(missing_ok=True)
            import h5py

            try:
                values = np.ascontiguousarray(raw_predictions[target], dtype="<f8")
                with h5py.File(temporary, mode="w") as h5:
                    h5.create_dataset(
                        "raw_draw_bits",
                        data=values.view("<u8"),
                        dtype="<u8",
                        track_times=False,
                    )
                    h5.attrs["target"] = target
                    h5.attrs["target_index"] = target_index
                    h5.flush()
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def assert_complete(self) -> None:
        if self._next_stage_index != len(self.pipeline_steps):
            raise AssertionError(
                "Monolith boundary observer stopped after "
                f"{self._next_stage_index}/{len(self.pipeline_steps)} stages."
            )


def _observe_frame_boundary(
    observer: _EquivalenceBoundaryObserver | None,
    stage: str,
    frame: Frame,
) -> None:
    if observer is not None:
        observer.observe_frame(stage, frame)


@contextmanager
def _without_pytables_leaf_timestamps(enabled: bool) -> Iterator[None]:
    """Disable test-only HDF5 leaf mtimes for physical byte comparisons."""

    if not enabled:
        yield
        return
    from tables.leaf import Leaf

    original_init = Leaf.__init__

    def deterministic_init(
        leaf: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        if len(args) >= 7:
            args = (*args[:6], False, *args[7:])
        else:
            kwargs["track_times"] = False
        original_init(leaf, *args, **kwargs)

    Leaf.__init__ = deterministic_init
    try:
        yield
    finally:
        Leaf.__init__ = original_init


def _write_policyengine_dataset(
    args: argparse.Namespace,
    frame: Frame,
    output_h5: Path,
) -> None:
    """Write the export, canonicalizing only test-harness timestamp metadata."""

    deterministic = bool(getattr(args, "equivalence_deterministic_h5_metadata", False))
    with _without_pytables_leaf_timestamps(deterministic):
        PolicyEngineUSEngine().write_dataset(
            frame,
            output_h5,
            period=args.target_year,
        )


def _load_block_v1_location_ladder(args: argparse.Namespace) -> UsLocationLadder:
    """Load the block_v1 ladder; its primary CD plan must be the current one.

    Route A reads the base's ``congressional_district_geoid`` as a
    current-vintage (119th Congress) district and checks the vintage attrs
    this builder writes, so a ladder whose primary plan is another vintage
    is refused before any draw.
    """

    ladder = load_us_location_ladder(args.block_ladder_artifact)
    if ladder.primary_congressional_district_plan != (
        CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
    ):
        raise SystemExit(
            "--location-rule block_v1: the block ladder's primary "
            "congressional-district plan is "
            f"{ladder.primary_congressional_district_plan!r}, but this base "
            f"records districts as {CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE!r}."
        )
    # Validates the crosswalk artifact (current-vintage targets) even though
    # block_v1 translates no facts: its digest is the attr Route A checks.
    load_congressional_district_vintage_crosswalk(
        args.congressional_district_vintage_crosswalk
    )
    return ladder


def _block_v1_household_location(
    args: argparse.Namespace,
    frame: Frame,
) -> tuple[Frame, dict[str, object]]:
    """Draw every household's 2020 block (microcosm #696) and record it.

    Every household on this line is CPS ASEC lineage: PUF-detail and
    capital-gains tail copies share their native row's
    ``(source_year, source_household_id)``, so each copy's candidate blocks
    come from the same ASEC row (identified county when ``GTCO`` is nonzero,
    else its state outside that ASEC year's identified counties), while each
    copy gets its own keyed draw. Returns the located frame and the
    ``household_location`` manifest record.
    """

    ladder = _load_block_v1_location_ladder(args)
    asec_paths = _asec_source_paths(args)
    source_year, source_household_id = household_source_keys(frame)
    geography = cps_source_geography(
        ladder,
        load_cps_household_geography(asec_paths),
        source_year=source_year,
        source_household_id=source_household_id,
        state_fips=frame.table("household")["state_fips"].to_numpy(np.int64),
    )
    located, table = with_household_us_block_location(
        frame,
        ladder,
        seed=args.location_seed,
        clones=args.location_clones,
        county_fips=geography.county_fips,
        identification_group=geography.identification_group,
        identified_counties=geography.identified_counties,
        cbsa_code=geography.cbsa_code,
        unidentified_county_share=geography.unidentified_county_share,
        person_reference_columns=US_PERSON_REFERENCE_ID_COLUMNS,
    )
    household = located.table("household")
    household_weights = located.weights_for("household").values
    location_gate = shadow_gate(
        "us_base_block_location",
        lambda: us_block_location_gate(household, ladder),
        lambda: EvidenceContext(
            artifacts={"household": household, "location_ladder": ladder}
        ),
    )
    # The L0/refit release export re-runs the legacy geography-ladder gate
    # (l0_refit_export.py:export_us_l0_refit_h5) on frames built from this
    # base, so the block draw must pass it here too.
    ladder_gate = shadow_gate(
        "us_base_geography_ladder",
        lambda: us_geography_ladder_gate(household, household_weights),
        lambda: EvidenceContext(
            artifacts={"household": household, "household_weights": household_weights}
        ),
    )
    failures = [
        *(f"us_block_location: {failure}" for failure in location_gate.failures),
        *(f"geography_ladder: {failure}" for failure in ladder_gate.failures),
    ]
    if failures and not args.allow_geography_ladder_gate_failures:
        raise SystemExit(
            "Household location (block_v1) gate failed:\n  " + "\n  ".join(failures)
        )
    record = us_block_location_manifest(
        table,
        ladder,
        seed=args.location_seed,
        clones=args.location_clones,
        candidate_set_rule=geography.record,
    )
    record.update(
        {
            "applied": True,
            "artifact": str(args.block_ladder_artifact.resolve()),
            "artifact_sha256": ladder.sha256,
            "asec_h5": {
                str(year): str(path.resolve())
                for year, path in sorted(asec_paths.items())
            },
            "congressional_district_vintage_crosswalk": str(
                args.congressional_district_vintage_crosswalk.resolve()
            ),
            "congressional_district_vintage_crosswalk_sha256": _sha256(
                args.congressional_district_vintage_crosswalk
            ),
            "congressional_district_vintage_target": (
                CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
            ),
            "coverage": us_geography_ladder_assignment_summary(
                household,
                ladder.blocks,
                weight_values=household_weights,
            ),
            "gate": {
                "passed": location_gate.passed,
                "failures": list(location_gate.failures),
                "details": dict(location_gate.details),
            },
            "geography_ladder_gate": {
                "passed": ladder_gate.passed,
                "failures": list(ladder_gate.failures),
                "details": dict(ladder_gate.details),
            },
        }
    )
    return located, record


def _write_block_v1_location_attrs(
    attrs: MutableMapping[str, object],
    household_location: Mapping[str, object],
) -> None:
    """Root attrs of a block_v1 base; legacy bases never carry the location ones.

    The crosswalk digest and ``119th_congress`` target are the attrs Route A
    checks before it runs. The ladder digest and vintages go under both the
    legacy ladder names and the ``populace_block_ladder_*`` names every
    block_v1 line writes. Seed and clones are written as text, as the stacked
    pool writes them.
    """

    ladder = household_location["block_ladder"]
    if not isinstance(ladder, Mapping):
        raise TypeError("household_location['block_ladder'] must be a mapping.")
    attrs[CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR] = str(
        household_location["congressional_district_vintage_crosswalk_sha256"]
    )
    attrs[CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR] = str(
        household_location["congressional_district_vintage_target"]
    )
    attrs[GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR] = str(ladder["sha256"])
    attrs[GEOGRAPHY_LADDER_VINTAGES_ATTR] = json.dumps(
        ladder["layer_vintages"], sort_keys=True
    )
    attrs[POPULACE_BLOCK_LADDER_SHA256_ATTR] = str(ladder["sha256"])
    attrs[POPULACE_BLOCK_LADDER_VINTAGES_ATTR] = json.dumps(
        ladder["layer_vintages"], sort_keys=True
    )
    attrs[POPULACE_LOCATION_RULE_ATTR] = str(household_location["location_rule"])
    attrs[POPULACE_LOCATION_SEED_ATTR] = str(household_location["seed"])
    attrs[POPULACE_LOCATION_CLONES_ATTR] = str(household_location["clones"])


def _run_all(
    args: argparse.Namespace,
    *,
    boundary_observer: _EquivalenceBoundaryObserver | None = None,
) -> Frame:
    """Run the existing single-process pipeline in its original order."""

    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = out_dir / _dataset_filename(args.target_year)
    summary_path = out_dir / _summary_filename(args.target_year)

    raw_base, base_source = _load_base_frame_from_args(args)
    _observe_frame_boundary(boundary_observer, "source_construction", raw_base)
    weeks_unemployed_source_path = (
        args.asec_2023_weeks_unemployed_source
        if args.asec_2023_weeks_unemployed_source is not None
        else fetch_asec_2023_weeks_unemployed_source()
    )
    weeks_unemployed_source = load_asec_2023_weeks_unemployed_source(
        weeks_unemployed_source_path
    )
    education_assistance_source = load_asec_education_assistance_sources(
        _asec_education_source_paths(args),
        income_years=_pooled_income_years(args),
    )
    public_assistance_type_source = load_asec_public_assistance_type_sources(
        _asec_education_source_paths(args),
        income_years=_pooled_income_years(args),
    )
    base = derive_us_cps_carried_inputs(
        raw_base,
        public_assistance_type_source=public_assistance_type_source,
    )
    base = with_us_prior_year_income_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_relationship_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    relationship_inputs_gate = _shadow_frame_gate(
        "us_base_relationship_inputs",
        lambda: us_relationship_inputs_signal_gate(base),
        base,
    )
    if not relationship_inputs_gate.passed:
        raise SystemExit(
            "Relationship-input signal gate failed:\n  "
            + "\n  ".join(relationship_inputs_gate.failures)
        )
    base = with_us_spm_independence_role(
        base,
        seed=args.seed,
        time_period=args.target_year,
        asec_spm_role_source_paths=_asec_education_source_paths(args),
    )
    spm_independence_role_gate = _shadow_frame_gate(
        "us_base_spm_independence_role",
        lambda: us_spm_independence_role_signal_gate(base),
        base,
    )
    if not spm_independence_role_gate.passed:
        raise SystemExit(
            "SPM independence role signal gate failed:\n  "
            + "\n  ".join(spm_independence_role_gate.failures)
        )
    base = with_us_medicare_take_up_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    medicare_take_up_gate = _shadow_frame_gate(
        "us_base_medicare_take_up_pre_clone",
        lambda: us_medicare_take_up_signal_gate(base),
        base,
    )
    if not medicare_take_up_gate.passed:
        raise SystemExit(
            "Medicare take-up input signal gate failed before support cloning:\n  "
            + "\n  ".join(medicare_take_up_gate.failures)
        )
    housing_inputs_gate = us_housing_inputs_signal_gate(base)
    acs_rent_donor: pd.DataFrame | None = None
    if not housing_inputs_gate.passed:
        if args.acs_h5 is None:
            _observe_frame_gate(
                "us_base_housing_inputs_pre_clone", housing_inputs_gate, base
            )
            raise SystemExit(
                "Housing-input signal gate is not already green and --acs-h5 "
                "was not provided; exact pre_subsidy_rent restoration requires "
                "the pinned ACS 2022 donor."
            )
        acs_rent_donor = load_acs_2022_rent_donor(args.acs_h5)
        base = with_us_housing_inputs(
            base,
            seed=args.seed,
            time_period=args.target_year,
            acs_rent_donor=acs_rent_donor,
        )
        housing_inputs_gate = us_housing_inputs_signal_gate(base)
    _observe_frame_gate("us_base_housing_inputs_pre_clone", housing_inputs_gate, base)
    if not housing_inputs_gate.passed:
        raise SystemExit(
            "Housing-input signal gate failed before support cloning:\n  "
            + "\n  ".join(housing_inputs_gate.failures)
        )
    base = with_us_eligibility_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    eligibility_inputs_gate = _shadow_frame_gate(
        "us_base_eligibility_inputs",
        lambda: us_eligibility_inputs_signal_gate(base),
        base,
    )
    if not eligibility_inputs_gate.passed:
        raise SystemExit(
            "Eligibility-input signal gate failed before support cloning:\n  "
            + "\n  ".join(eligibility_inputs_gate.failures)
        )
    base = with_us_pregnancy_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    pregnancy_gate = _shadow_frame_gate(
        "us_base_pregnancy", lambda: us_pregnancy_signal_gate(base), base
    )
    if not pregnancy_gate.passed:
        raise SystemExit(
            "Pregnancy signal gate failed before support cloning:\n  "
            + "\n  ".join(pregnancy_gate.failures)
        )
    base = with_us_wic_claim_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    wic_claim_gate = _shadow_frame_gate(
        "us_base_wic_claim_pre_clone", lambda: us_wic_claim_signal_gate(base), base
    )
    if not wic_claim_gate.passed:
        raise SystemExit(
            "WIC-claim signal gate failed before support cloning:\n  "
            + "\n  ".join(wic_claim_gate.failures)
        )
    base = with_us_child_support_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_disability_benefits(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_workers_compensation(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_weeks_unemployed(
        base,
        seed=args.seed,
        time_period=args.target_year,
        asec_2023_source=weeks_unemployed_source,
    )
    base = with_us_childcare_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_energy_subsidy_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_retirement_contribution_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_retirement_distribution_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_esi_premium_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    esi_premiums_gate = _shadow_frame_gate(
        "us_base_esi_premiums_pre_clone",
        lambda: us_esi_premiums_signal_gate(base),
        base,
    )
    if not esi_premiums_gate.passed:
        raise SystemExit(
            "ESI premium signal gate failed before support cloning:\n  "
            + "\n  ".join(esi_premiums_gate.failures)
        )
    base = with_us_immigration_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    _observe_frame_boundary(boundary_observer, "pre_clone_enrichment", base)
    expanded = clone_us_frame_for_puf_support(base)
    donor_build_summary: dict[str, object] = {}
    donor = _puf_tax_unit_donor_from_h5(
        args.puf_h5,
        source_puf_csv=getattr(args, "puf_source_year_csv", None),
        donor_build_summary=donor_build_summary,
    )
    e01000_reconciliation_basis = _puf_e01000_reconciliation_basis(
        args,
        donor,
        donor_build_summary,
    )
    _observe_frame_boundary(boundary_observer, "clone_feature_extraction", expanded)
    tail_bound_diagnostics: list[dict[str, object]] = []
    if boundary_observer is None:
        imputed, weights_audit = impute_and_audit_us_puf_support(
            expanded,
            donor,
            seed=args.seed,
            n_estimators=args.n_estimators,
            tail_bound_diagnostics=tail_bound_diagnostics,
        )
    else:
        imputed, weights_audit = impute_and_audit_us_puf_support(
            expanded,
            donor,
            seed=args.seed,
            n_estimators=args.n_estimators,
            raw_predictions_callback=lambda predictions: (
                boundary_observer.observe_primary_qrf(expanded, predictions)
            ),
            tail_bound_diagnostics=tail_bound_diagnostics,
        )
    _observe_frame_boundary(boundary_observer, "qrf_finalization", imputed)
    imputed, capital_gains_tail_transfer = _capital_gains_tail_transfer_stage(
        args,
        imputed,
        donor=donor,
    )
    _observe_frame_boundary(
        boundary_observer,
        PUF_CAPITAL_GAINS_TAIL_STAGE_NAME,
        imputed,
    )
    imputed, _ = _capital_gain_distributions_stage(args, imputed)
    _observe_frame_boundary(
        boundary_observer,
        CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME,
        imputed,
    )
    imputed = with_us_qbi_input_reconciliation(imputed)
    _observe_frame_boundary(boundary_observer, "qbi_reconciliation", imputed)
    imputed = with_us_wic_claim_input(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    wic_claim_gate = _shadow_frame_gate(
        "us_base_wic_claim_post_clone",
        lambda: us_wic_claim_signal_gate(imputed),
        imputed,
    )
    if not wic_claim_gate.passed:
        raise SystemExit(
            "WIC-claim signal gate failed after support cloning:\n  "
            + "\n  ".join(wic_claim_gate.failures)
        )
    medicare_take_up_gate = _shadow_frame_gate(
        "us_base_medicare_take_up_post_clone",
        lambda: us_medicare_take_up_signal_gate(imputed),
        imputed,
    )
    if not medicare_take_up_gate.passed:
        raise SystemExit(
            "Medicare take-up input signal gate failed after support cloning:\n  "
            + "\n  ".join(medicare_take_up_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "wic_post_clone", imputed)
    imputed = impute_us_housing_assistance_to_puf_support(
        imputed,
        seed=args.seed,
    )
    _observe_frame_boundary(boundary_observer, "housing_assistance", imputed)
    imputed = with_us_prior_year_income_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    prior_year_income_gate = _shadow_frame_gate(
        "us_base_prior_year_income",
        lambda: us_prior_year_income_signal_gate(imputed),
        imputed,
    )
    if not prior_year_income_gate.passed:
        raise SystemExit(
            "Prior-year-income signal gate failed:\n  "
            + "\n  ".join(prior_year_income_gate.failures)
        )
    prior_year_income_reconciliation_gate = _shadow_frame_gate(
        "us_base_prior_year_income_source_reconciliation",
        lambda: us_prior_year_income_source_reconciliation_gate(imputed),
        imputed,
    )
    if not prior_year_income_reconciliation_gate.passed:
        raise SystemExit(
            "Prior-year-income source reconciliation failed:\n  "
            + "\n  ".join(prior_year_income_reconciliation_gate.failures)
        )
    housing_inputs_gate = _shadow_frame_gate(
        "us_base_housing_inputs_post_clone",
        lambda: us_housing_inputs_signal_gate(imputed),
        imputed,
    )
    if not housing_inputs_gate.passed:
        raise SystemExit(
            "Housing-input signal gate failed after PUF-support imputation:\n  "
            + "\n  ".join(housing_inputs_gate.failures)
        )
    qbi_inputs_gate = _shadow_frame_gate(
        "us_base_qbi_inputs", lambda: us_qbi_inputs_signal_gate(imputed), imputed
    )
    if not qbi_inputs_gate.passed:
        raise SystemExit(
            "QBI-input signal gate failed:\n  " + "\n  ".join(qbi_inputs_gate.failures)
        )
    farm_business_income_gate = _shadow_frame_gate(
        "us_base_farm_business_income",
        lambda: us_farm_business_income_signal_gate(imputed),
        imputed,
    )
    if not farm_business_income_gate.passed:
        raise SystemExit(
            "Farm-business-income signal gate failed:\n  "
            + "\n  ".join(farm_business_income_gate.failures)
        )
    domestic_production_ald_gate = _shadow_frame_gate(
        "us_base_domestic_production_ald",
        lambda: us_domestic_production_ald_signal_gate(imputed),
        imputed,
    )
    if not domestic_production_ald_gate.passed:
        raise SystemExit(
            "Domestic-production-ALD signal gate failed:\n  "
            + "\n  ".join(domestic_production_ald_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "prior_year_income_post_clone", imputed)
    imputed = with_us_child_support_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    child_support_gate = _shadow_frame_gate(
        "us_base_child_support", lambda: us_child_support_signal_gate(imputed), imputed
    )
    if not child_support_gate.passed:
        raise SystemExit(
            "Child-support signal gate failed:\n  "
            + "\n  ".join(child_support_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "child_support_post_clone", imputed)
    imputed = with_us_disability_benefits(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    disability_benefits_gate = _shadow_frame_gate(
        "us_base_disability_benefits",
        lambda: us_disability_benefits_signal_gate(imputed),
        imputed,
    )
    if not disability_benefits_gate.passed:
        raise SystemExit(
            "Disability-benefits signal gate failed:\n  "
            + "\n  ".join(disability_benefits_gate.failures)
        )
    _observe_frame_boundary(
        boundary_observer, "disability_benefits_post_clone", imputed
    )
    imputed = with_us_workers_compensation(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    workers_compensation_gate = _shadow_frame_gate(
        "us_base_workers_compensation",
        lambda: us_workers_compensation_signal_gate(imputed),
        imputed,
    )
    if not workers_compensation_gate.passed:
        raise SystemExit(
            "Workers-compensation signal gate failed:\n  "
            + "\n  ".join(workers_compensation_gate.failures)
        )
    _observe_frame_boundary(
        boundary_observer, "workers_compensation_post_clone", imputed
    )
    imputed = with_us_weeks_unemployed(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
        asec_2023_source=weeks_unemployed_source,
    )
    weeks_unemployed_gate = _shadow_frame_gate(
        "us_base_weeks_unemployed",
        lambda: us_weeks_unemployed_signal_gate(imputed),
        imputed,
    )
    if not weeks_unemployed_gate.passed:
        raise SystemExit(
            "Weeks-unemployed signal gate failed:\n  "
            + "\n  ".join(weeks_unemployed_gate.failures)
        )
    educator_expense_gate = _shadow_frame_gate(
        "us_base_educator_expense",
        lambda: us_educator_expense_signal_gate(imputed),
        imputed,
    )
    if not educator_expense_gate.passed:
        raise SystemExit(
            "Educator-expense signal gate failed:\n  "
            + "\n  ".join(educator_expense_gate.failures)
        )
    form_4952_election_gate = _shadow_frame_gate(
        "us_base_form_4952_election",
        lambda: us_form_4952_election_signal_gate(imputed),
        imputed,
    )
    if not form_4952_election_gate.passed:
        raise SystemExit(
            "Form 4952 election signal gate failed:\n  "
            + "\n  ".join(form_4952_election_gate.failures)
        )
    salt_refund_income_gate = _shadow_frame_gate(
        "us_base_salt_refund_income",
        lambda: us_salt_refund_income_signal_gate(imputed),
        imputed,
    )
    if not salt_refund_income_gate.passed:
        raise SystemExit(
            "SALT-refund-income signal gate failed:\n  "
            + "\n  ".join(salt_refund_income_gate.failures)
        )
    capital_gain_details_gate = _shadow_frame_gate(
        "us_base_capital_gain_details",
        lambda: us_capital_gain_details_signal_gate(imputed),
        imputed,
    )
    if not capital_gain_details_gate.passed:
        raise SystemExit(
            "Capital-gain details signal gate failed:\n  "
            + "\n  ".join(capital_gain_details_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "weeks_unemployed_post_clone", imputed)
    imputed = with_us_childcare_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    childcare_gate = _shadow_frame_gate(
        "us_base_childcare", lambda: us_childcare_signal_gate(imputed), imputed
    )
    if not childcare_gate.passed:
        raise SystemExit(
            "Childcare-input signal gate failed:\n  "
            + "\n  ".join(childcare_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "childcare_post_clone", imputed)
    imputed = with_us_adult_care_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    adult_care_gate = _shadow_frame_gate(
        "us_base_adult_care", lambda: us_adult_care_signal_gate(imputed), imputed
    )
    if not adult_care_gate.passed:
        raise SystemExit(
            "Adult-care input signal gate failed:\n  "
            + "\n  ".join(adult_care_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "adult_care_post_clone", imputed)
    imputed = with_us_energy_subsidy_input(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    energy_subsidy_gate = _shadow_frame_gate(
        "us_base_energy_subsidy",
        lambda: us_energy_subsidy_signal_gate(imputed),
        imputed,
    )
    if not energy_subsidy_gate.passed:
        raise SystemExit(
            "Energy-subsidy signal gate failed:\n  "
            + "\n  ".join(energy_subsidy_gate.failures)
        )
    alimony_gate = _shadow_frame_gate(
        "us_base_alimony", lambda: us_alimony_signal_gate(imputed), imputed
    )
    if not alimony_gate.passed:
        raise SystemExit(
            "Alimony-input signal gate failed:\n  " + "\n  ".join(alimony_gate.failures)
        )
    casualty_loss_gate = _shadow_frame_gate(
        "us_base_casualty_loss", lambda: us_casualty_loss_signal_gate(imputed), imputed
    )
    if not casualty_loss_gate.passed:
        raise SystemExit(
            "Casualty-loss signal gate failed:\n  "
            + "\n  ".join(casualty_loss_gate.failures)
        )
    misc_itemized_gate = _shadow_frame_gate(
        "us_base_misc_itemized", lambda: us_misc_itemized_signal_gate(imputed), imputed
    )
    if not misc_itemized_gate.passed:
        raise SystemExit(
            "Miscellaneous-itemized signal gate failed:\n  "
            + "\n  ".join(misc_itemized_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "energy_subsidy_post_clone", imputed)
    imputed = with_us_retirement_contribution_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
    )
    retirement_contributions_gate = _shadow_frame_gate(
        "us_base_retirement_contributions",
        lambda: us_retirement_contributions_signal_gate(imputed),
        imputed,
    )
    if not retirement_contributions_gate.passed:
        raise SystemExit(
            "Retirement-contribution signal gate failed:\n  "
            + "\n  ".join(retirement_contributions_gate.failures)
        )
    _observe_frame_boundary(
        boundary_observer, "retirement_contributions_post_clone", imputed
    )
    imputed = with_us_retirement_distribution_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
        # This is the one ownership boundary where the copied ASEC leaves must
        # be replaced on the newly created PUF support. Downstream consumers
        # preserve this completed draw even after selecting a narrower support.
        force_puf_imputation=True,
    )
    retirement_distributions_gate = _shadow_frame_gate(
        "us_base_retirement_distributions",
        lambda: us_retirement_distributions_signal_gate(imputed),
        imputed,
    )
    if not retirement_distributions_gate.passed:
        raise SystemExit(
            "Retirement-distribution signal gate failed:\n  "
            + "\n  ".join(retirement_distributions_gate.failures)
        )
    _observe_frame_boundary(
        boundary_observer, "retirement_distributions_post_clone", imputed
    )
    imputed = with_us_education_inputs(
        imputed,
        seed=args.seed,
        time_period=args.target_year,
        asec_education_source=education_assistance_source,
    )
    education_inputs_gate = _shadow_frame_gate(
        "us_base_education_inputs",
        lambda: us_education_inputs_signal_gate(imputed),
        imputed,
    )
    if not education_inputs_gate.passed:
        raise SystemExit(
            "Education-input signal gate failed:\n  "
            + "\n  ".join(education_inputs_gate.failures)
        )
    # The ESI premium inputs ride the support clone unchanged; re-prove clone
    # agreement and the cell multiples on the cloned frame.
    esi_premiums_post_clone_gate = _shadow_frame_gate(
        "us_base_esi_premiums_post_clone",
        lambda: us_esi_premiums_signal_gate(imputed),
        imputed,
    )
    if not esi_premiums_post_clone_gate.passed:
        raise SystemExit(
            "ESI premium signal gate failed after support cloning:\n  "
            + "\n  ".join(esi_premiums_post_clone_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "education_inputs_post_clone", imputed)
    household_location: dict[str, object] | None = None
    if _is_block_v1(args):
        # block_v1 replaces both legacy location stages with one block draw;
        # their summary keys stay present and unapplied.
        imputed, household_location = _block_v1_household_location(args, imputed)
        congressional_district_assignment = {
            "applied": False,
            "replaced_by": "household_location",
        }
        geography_ladder_assignment = {
            "applied": False,
            "opted_out": False,
            "replaced_by": "household_location",
        }
        _observe_frame_boundary(
            boundary_observer, HOUSEHOLD_LOCATION_STAGE_NAME, imputed
        )
    else:
        congressional_district_assignment = {"applied": False}
        if args.assign_congressional_districts:
            ledger_facts = load_ledger_consumer_artifact(args.ledger_facts).facts
            if args.congressional_district_vintage_crosswalk is not None:
                ledger_facts = (
                    translate_congressional_district_facts_to_current_vintage(
                        ledger_facts,
                        load_congressional_district_vintage_crosswalk(
                            args.congressional_district_vintage_crosswalk
                        ),
                    )
                )
            distribution = congressional_district_distribution_from_ledger_facts(
                ledger_facts
            )
            imputed = with_household_congressional_districts(
                imputed,
                distribution,
                seed=args.congressional_district_seed,
            )
            congressional_district_assignment = (
                congressional_district_assignment_summary(
                    imputed.table("household"),
                    distribution,
                )
            )
            congressional_district_assignment.update(
                {
                    "ledger_facts": str(args.ledger_facts.resolve()),
                    "ledger_facts_sha256": _sha256(args.ledger_facts),
                    "congressional_district_vintage_crosswalk": (
                        str(args.congressional_district_vintage_crosswalk.resolve())
                        if args.congressional_district_vintage_crosswalk is not None
                        else None
                    ),
                    "congressional_district_vintage_crosswalk_sha256": (
                        _sha256(args.congressional_district_vintage_crosswalk)
                        if args.congressional_district_vintage_crosswalk is not None
                        else None
                    ),
                    "seed": args.congressional_district_seed,
                }
            )
        _observe_frame_boundary(
            boundary_observer, "congressional_district_assignment", imputed
        )
        geography_ladder_assignment = {
            "applied": False,
            "opted_out": bool(args.without_block_ladder),
        }
        _shadow_legacy_location_not_run(args)
        if args.block_ladder_artifact is not None:
            ladder = load_us_block_ladder(args.block_ladder_artifact)
            imputed = with_household_us_geography_ladder(
                imputed,
                ladder,
                seed=args.geography_ladder_seed,
                expected_congressional_district_vintage=(
                    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
                ),
            )
            household = imputed.table("household")
            household_weights = imputed.weights_for("household").values
            gate = shadow_gate(
                "us_base_geography_ladder",
                lambda: us_geography_ladder_gate(household, household_weights),
                lambda: EvidenceContext(
                    artifacts={
                        "household": household,
                        "household_weights": household_weights,
                    }
                ),
            )
            if not gate.passed and not args.allow_geography_ladder_gate_failures:
                raise SystemExit(
                    "Geography-ladder gate failed:\n  " + "\n  ".join(gate.failures)
                )
            geography_ladder_assignment = us_geography_ladder_assignment_summary(
                household,
                ladder,
                weight_values=household_weights,
            )
            geography_ladder_assignment.update(
                {
                    "artifact": str(args.block_ladder_artifact.resolve()),
                    "artifact_sha256": _sha256(args.block_ladder_artifact),
                    "seed": args.geography_ladder_seed,
                    "gate": {
                        "passed": gate.passed,
                        "failures": list(gate.failures),
                        "details": dict(gate.details),
                    },
                }
            )
        _observe_frame_boundary(boundary_observer, "block_ladder_assignment", imputed)
    _write_policyengine_dataset(args, imputed, output_h5)
    if household_location is not None:
        import h5py

        with h5py.File(output_h5, "a") as h5:
            _write_block_v1_location_attrs(h5.attrs, household_location)
    elif (
        args.congressional_district_vintage_crosswalk is not None
        or args.block_ladder_artifact is not None
    ):
        import h5py

        with h5py.File(output_h5, "a") as h5:
            if args.congressional_district_vintage_crosswalk is not None:
                h5.attrs[CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR] = (
                    _sha256(args.congressional_district_vintage_crosswalk)
                )
                h5.attrs[CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR] = (
                    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
                )
            if args.block_ladder_artifact is not None:
                h5.attrs[GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR] = _sha256(
                    args.block_ladder_artifact
                )
                h5.attrs[GEOGRAPHY_LADDER_VINTAGES_ATTR] = json.dumps(
                    geography_ladder_assignment["layer_vintages"],
                    sort_keys=True,
                )

    e01000_reconciliation = finalize_puf_e01000_reconciliation(
        e01000_reconciliation_basis,
        capital_gains_tail_transfer,
        frame_columns=_frame_column_inventory(imputed),
    )
    summary = {
        "base_source": base_source,
        "base_h5": (str(args.base_h5.resolve()) if args.base_h5 is not None else None),
        "base_sha256": _sha256(args.base_h5) if args.base_h5 is not None else None,
        "puf_h5": str(args.puf_h5.resolve()),
        "puf_sha256": _sha256(args.puf_h5),
        "puf_source_year_csv": (
            str(args.puf_source_year_csv.resolve())
            if args.puf_source_year_csv is not None
            else None
        ),
        "puf_source_year_csv_sha256": (
            _sha256(args.puf_source_year_csv)
            if args.puf_source_year_csv is not None
            else None
        ),
        "acs_h5": str(args.acs_h5.resolve()) if args.acs_h5 is not None else None,
        "acs_sha256": _sha256(args.acs_h5) if args.acs_h5 is not None else None,
        "acs_rent_donor_rows": (
            int(len(acs_rent_donor)) if acs_rent_donor is not None else None
        ),
        "weeks_unemployed_source": {
            "path": str(Path(weeks_unemployed_source_path).resolve()),
            "sha256": _sha256(weeks_unemployed_source_path),
            "upstream_archive_sha256": (ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256),
            "rows": int(len(weeks_unemployed_source)),
            "audit": dict(weeks_unemployed_source.attrs.get("source_audit", {})),
        },
        "education_assistance_source": {
            "income_years": [int(year) for year in _pooled_income_years(args)],
            "audit": dict(education_assistance_source.attrs.get("source_audit", {})),
        },
        "public_assistance_type_source": {
            "income_years": [int(year) for year in _pooled_income_years(args)],
            "audit": dict(public_assistance_type_source.attrs.get("source_audit", {})),
        },
        "output_h5": str(output_h5),
        "output_sha256": _sha256(output_h5),
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "base_rows": _row_counts(base),
        "expanded_rows": _row_counts(imputed),
        "base_household_weight_total": float(base.weights_for("household").total),
        "expanded_household_weight_total": float(
            imputed.weights_for("household").total
        ),
        "channel_weight_totals": _channel_weight_totals(imputed),
        "puf_donor_rows": int(len(donor)),
        "puf_donor_columns": sorted(donor.columns.tolist()),
        "puf_donor_build_summary": donor_build_summary,
        "puf_e01000_reconciliation": e01000_reconciliation,
        "weights_audit": weights_audit,
        "puf_tax_detail_tail_bounds": tail_bound_diagnostics,
        "puf_capital_gains_tail_transfer": capital_gains_tail_transfer,
        "qbi_inputs_signal": {
            "passed": qbi_inputs_gate.passed,
            "failures": list(qbi_inputs_gate.failures),
            "details": dict(qbi_inputs_gate.details),
        },
        "farm_business_income_signal": {
            "passed": farm_business_income_gate.passed,
            "failures": list(farm_business_income_gate.failures),
            "details": dict(farm_business_income_gate.details),
        },
        "domestic_production_ald_signal": {
            "passed": domestic_production_ald_gate.passed,
            "failures": list(domestic_production_ald_gate.failures),
            "details": dict(domestic_production_ald_gate.details),
        },
        "child_support_signal": {
            "passed": child_support_gate.passed,
            "failures": list(child_support_gate.failures),
            "details": dict(child_support_gate.details),
        },
        "disability_benefits_signal": {
            "passed": disability_benefits_gate.passed,
            "failures": list(disability_benefits_gate.failures),
            "details": dict(disability_benefits_gate.details),
        },
        "workers_compensation_signal": {
            "passed": workers_compensation_gate.passed,
            "failures": list(workers_compensation_gate.failures),
            "details": dict(workers_compensation_gate.details),
        },
        "weeks_unemployed_signal": {
            "passed": weeks_unemployed_gate.passed,
            "failures": list(weeks_unemployed_gate.failures),
            "details": dict(weeks_unemployed_gate.details),
        },
        "eligibility_inputs_signal": {
            "passed": eligibility_inputs_gate.passed,
            "failures": list(eligibility_inputs_gate.failures),
            "details": dict(eligibility_inputs_gate.details),
        },
        "pregnancy_signal": {
            "passed": pregnancy_gate.passed,
            "failures": list(pregnancy_gate.failures),
            "details": dict(pregnancy_gate.details),
        },
        "esi_premiums_signal": {
            "passed": esi_premiums_gate.passed,
            "failures": list(esi_premiums_gate.failures),
            "details": dict(esi_premiums_gate.details),
        },
        "esi_premiums_post_clone_signal": {
            "passed": esi_premiums_post_clone_gate.passed,
            "failures": list(esi_premiums_post_clone_gate.failures),
            "details": dict(esi_premiums_post_clone_gate.details),
        },
        "wic_claim_signal": {
            "passed": wic_claim_gate.passed,
            "failures": list(wic_claim_gate.failures),
            "details": dict(wic_claim_gate.details),
        },
        "educator_expense_signal": {
            "passed": educator_expense_gate.passed,
            "failures": list(educator_expense_gate.failures),
            "details": dict(educator_expense_gate.details),
        },
        "form_4952_election_signal": {
            "passed": form_4952_election_gate.passed,
            "failures": list(form_4952_election_gate.failures),
            "details": dict(form_4952_election_gate.details),
        },
        "salt_refund_income_signal": {
            "passed": salt_refund_income_gate.passed,
            "failures": list(salt_refund_income_gate.failures),
            "details": dict(salt_refund_income_gate.details),
        },
        "capital_gain_details_signal": {
            "passed": capital_gain_details_gate.passed,
            "failures": list(capital_gain_details_gate.failures),
            "details": dict(capital_gain_details_gate.details),
        },
        "childcare_inputs_signal": {
            "passed": childcare_gate.passed,
            "failures": list(childcare_gate.failures),
            "details": dict(childcare_gate.details),
        },
        "adult_care_inputs_signal": {
            "passed": adult_care_gate.passed,
            "failures": list(adult_care_gate.failures),
            "details": dict(adult_care_gate.details),
        },
        "energy_subsidy_signal": {
            "passed": energy_subsidy_gate.passed,
            "failures": list(energy_subsidy_gate.failures),
            "details": dict(energy_subsidy_gate.details),
        },
        "alimony_inputs_signal": {
            "passed": alimony_gate.passed,
            "failures": list(alimony_gate.failures),
            "details": dict(alimony_gate.details),
        },
        "casualty_loss_signal": {
            "passed": casualty_loss_gate.passed,
            "failures": list(casualty_loss_gate.failures),
            "details": dict(casualty_loss_gate.details),
        },
        "misc_itemized_signal": {
            "passed": misc_itemized_gate.passed,
            "failures": list(misc_itemized_gate.failures),
            "details": dict(misc_itemized_gate.details),
        },
        "education_inputs_signal": {
            "passed": education_inputs_gate.passed,
            "failures": list(education_inputs_gate.failures),
            "details": dict(education_inputs_gate.details),
        },
        "retirement_contributions_signal": {
            "passed": retirement_contributions_gate.passed,
            "failures": list(retirement_contributions_gate.failures),
            "details": dict(retirement_contributions_gate.details),
        },
        "retirement_distributions_signal": {
            "passed": retirement_distributions_gate.passed,
            "failures": list(retirement_distributions_gate.failures),
            "details": dict(retirement_distributions_gate.details),
        },
        "relationship_inputs_signal": {
            "passed": relationship_inputs_gate.passed,
            "failures": list(relationship_inputs_gate.failures),
            "details": dict(relationship_inputs_gate.details),
        },
        "spm_independence_role_signal": {
            "passed": spm_independence_role_gate.passed,
            "failures": list(spm_independence_role_gate.failures),
            "details": dict(spm_independence_role_gate.details),
        },
        "medicare_take_up_input_signal": {
            "passed": medicare_take_up_gate.passed,
            "failures": list(medicare_take_up_gate.failures),
            "details": dict(medicare_take_up_gate.details),
        },
        "housing_inputs_signal": {
            "passed": housing_inputs_gate.passed,
            "failures": list(housing_inputs_gate.failures),
            "details": dict(housing_inputs_gate.details),
        },
        "prior_year_income_signal": {
            "passed": prior_year_income_gate.passed,
            "failures": list(prior_year_income_gate.failures),
            "details": dict(prior_year_income_gate.details),
        },
        "prior_year_income_source_reconciliation": {
            "passed": prior_year_income_reconciliation_gate.passed,
            "failures": list(prior_year_income_reconciliation_gate.failures),
            "details": dict(prior_year_income_reconciliation_gate.details),
        },
        "congressional_district_assignment": congressional_district_assignment,
        "geography_ladder_assignment": geography_ladder_assignment,
        "channel_output_totals": _channel_output_totals(imputed),
        "immigration_composition": us_immigration_composition_summary(imputed),
    }
    if household_location is not None:
        summary["household_location"] = household_location
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    _observe_frame_boundary(boundary_observer, "final_export", imputed)
    return imputed


def _run_configured_stage(args: argparse.Namespace) -> None:
    """Dispatch one separately executed outer stage."""

    if args.stage in LEGACY_STAGE_ALIASES:
        raise SystemExit(
            f"Stage alias {args.stage!r} was Phase-1-only; use one of the "
            f"descriptive stages {list(_pipeline_steps_for(args))}."
        )
    _run_outer_stage(args)


def _run_outer_stage(args: argparse.Namespace) -> None:
    """Run one exact pipeline step through the lossless outer-stage runtime.

    A step process its ``--stage all`` parent started for several of
    ``ONE_PROCESS_STEPS`` (``STEP_REPORT_VARIABLE``) runs every one of them
    from ``--stage`` on, in order, telling the parent as each starts. Only
    the first reads its predecessor's checkpoint; each later step takes the
    frame the step before it completed.
    """

    pipeline_steps = _pipeline_steps_for(args)
    if args.stage not in pipeline_steps:
        raise SystemExit(
            f"--stage {args.stage!r} is not a stage of the "
            f"--location-rule {_location_rule(args)} pipeline "
            f"{list(pipeline_steps)}."
        )
    report_fd = _step_report_fd(args.stage)
    runtime = _open_stage_runtime(args)
    if report_fd is None:
        _run_outer_step(args, runtime)
        return
    steps = ONE_PROCESS_STEPS[ONE_PROCESS_STEPS.index(args.stage) :]
    print(
        f"Running steps {steps[0]} through {steps[-1]} in this process.",
        file=sys.stderr,
        flush=True,
    )
    # A process a step starts gets no copy of the write end, so the reports
    # end when this process exits.
    os.set_inheritable(report_fd, False)
    carried = _CarriedCheckpoint()
    for step in steps:
        _report_step_start(report_fd, step)
        step_args = copy.copy(args)
        step_args.stage = step
        if step != steps[0]:
            # As each step process does when it starts (_open_stage_runtime
            # above, for the first): a change to the builder's source files,
            # its dependencies or its thread settings since the run started
            # stops the run before this step.
            runtime.require_run_config(_stage_run_config(step_args))
        _run_outer_step(step_args, runtime, carried)


class _CarriedCheckpoint:
    """The checkpoint the last step completed, held for the next step.

    ``take`` hands it over and keeps no reference, so the next step can free
    its input frame before it writes its own checkpoint.
    """

    def __init__(self) -> None:
        self._checkpoint: LoadedStageCheckpoint | None = None

    def put(self, checkpoint: LoadedStageCheckpoint) -> None:
        self._checkpoint = checkpoint

    def take(self) -> LoadedStageCheckpoint | None:
        checkpoint, self._checkpoint = self._checkpoint, None
        return checkpoint


def _predecessor_checkpoint(
    args: argparse.Namespace,
    runtime: StageRuntime,
    carried: _CarriedCheckpoint | None,
) -> LoadedStageCheckpoint | None:
    """The previous step's checkpoint: the one ``carried`` holds, if any,
    otherwise read from disk."""

    handed_over = None if carried is None else carried.take()
    if handed_over is None:
        return runtime.load_predecessor(args.stage)
    runtime.require_ready(args.stage)
    pipeline_steps = _pipeline_steps_for(args)
    previous = pipeline_steps[pipeline_steps.index(args.stage) - 1]
    if handed_over.stage != previous:
        raise AssertionError(
            f"step {args.stage!r} was handed {handed_over.stage!r}'s frame, "
            f"not {previous!r}'s."
        )
    return handed_over


def _run_outer_step(
    args: argparse.Namespace,
    runtime: StageRuntime,
    carried: _CarriedCheckpoint | None = None,
) -> None:
    """Run the step ``args.stage`` names, unless it is already complete.

    With ``carried``, the step takes its input from it when it holds the
    previous step's checkpoint, and leaves its own completed checkpoint there
    for the next step.
    """

    if args.stage in runtime.context.completed:
        if carried is not None:
            carried.take()
        if args.stage == "source_construction" and args.asec_h5 is not None:
            _require_completed_source_receipt_matches_pins(
                args, runtime.metadata["source_construction"]
            )
            loaded = runtime.load("source_construction")
            _ensure_asec_raw_stage_checkpoint(
                args,
                loaded.frame,
                runtime.metadata["source_construction"],
            )
        elif args.stage == "final_export":
            _repair_completed_final_stage(args, runtime)
        elif args.stage == PUF_CAPITAL_GAINS_TAIL_STAGE_NAME:
            _ensure_capital_gains_tail_manifest(
                runtime.metadata[PUF_CAPITAL_GAINS_TAIL_STAGE_NAME]
            )
        return

    with (
        profile_stage(args.stage, args.checkpoint_dir),
        _armed_stage_gate_shadow(args) as save_gate_shadow,
    ):
        if args.stage == "source_construction":
            predecessor = runtime.load_predecessor(args.stage)
            if predecessor is not None:
                raise AssertionError(
                    "source construction unexpectedly has a predecessor"
                )
            frame, metadata = _source_construction_stage(args)
            if args.asec_h5 is not None:
                _ensure_asec_raw_stage_checkpoint(args, frame, metadata)
            runtime.complete(args.stage, frame, metadata=metadata)
            return

        if args.stage == "primary_qrf_chain":
            runtime.require_ready(args.stage)
            run_primary_puf_qrf_chain(args.checkpoint_dir / "primary_qrf")
            runtime.complete_without_frame(
                args.stage,
                metadata={
                    "primary_qrf_checkpoint_dir": str(
                        (args.checkpoint_dir / "primary_qrf").resolve()
                    )
                },
            )
            return

        predecessor = _predecessor_checkpoint(args, runtime, carried)
        if predecessor is None:
            raise AssertionError(f"stage {args.stage!r} requires a predecessor")
        before = predecessor.frame
        # StageRuntime.load computed this identity from the loaded frame and
        # checked it against the run context. Each step whose result must keep
        # the input's identity and row order passes it to runtime.complete,
        # which compares the result with it before writing the checkpoint or
        # recording the step, so the input frame is not needed once the step
        # (and the clone check) ran.
        input_identity = predecessor.identity
        del predecessor
        expected_identity: FrameIdentity | None = None
        stage_metadata = runtime.metadata
        if args.stage == "pre_clone_enrichment":
            after, metadata = _pre_clone_enrichment_stage(
                args,
                before,
                stage_metadata["source_construction"],
            )
            expected_identity = input_identity
        elif args.stage == "clone_feature_extraction":
            after, metadata = _clone_feature_extraction_stage(args, before)
            assert_clone_expansion(
                before,
                after,
                channels=(
                    BASE_ASEC_SUPPORT_CHANNEL,
                    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
                ),
            )
        elif args.stage == "qrf_finalization":
            after, metadata = _qrf_finalization_stage(args, before)
            expected_identity = input_identity
        elif args.stage == PUF_CAPITAL_GAINS_TAIL_STAGE_NAME:
            after, metadata = _capital_gains_tail_transfer_stage(args, before)
        elif args.stage == CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME:
            after, metadata = _capital_gain_distributions_stage(args, before)
            expected_identity = input_identity
        elif args.stage == "final_export":
            # The export reads the frame and changes no table, so this step
            # writes no checkpoint of its own: its record names the previous
            # step's checkpoint file, SHA-256 and identity, and the alias is a
            # hard link to that file.
            metadata = _export_staged_result(args, before, stage_metadata)
            save_gate_shadow()
            previous_checkpoint = runtime.complete_without_frame(
                args.stage, metadata=metadata
            )
            _link_all_stage_checkpoint(args.checkpoint_dir, previous_checkpoint)
            return
        elif args.stage == HOUSEHOLD_LOCATION_STAGE_NAME:
            after, household_location = _block_v1_household_location(args, before)
            metadata = {"household_location": household_location}
            # --location-clones is 1 on this line (refused at parse time), so
            # the draw adds geography columns and changes no row or id.
            expected_identity = input_identity
        else:
            after, metadata = _post_qrf_frame_stage(
                args.stage,
                args,
                before,
                stage_metadata,
            )
            expected_identity = input_identity
        # Free the input frame before the checkpoint write; nothing below
        # reads it.
        del before
        save_gate_shadow()
        completed = runtime.complete(
            args.stage,
            after,
            metadata=metadata,
            expected_identity=expected_identity,
        )
        if carried is not None:
            carried.put(completed)


def _link_all_stage_checkpoint(checkpoint_dir: Path, source: Path) -> None:
    destination = checkpoint_dir / ALL_STAGE_CHECKPOINT_FILENAME
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        destination.unlink()
    os.link(source, destination)


def _repair_completed_final_stage(
    args: argparse.Namespace,
    runtime: StageRuntime,
) -> None:
    """Re-create a missing or changed export and relink the alias.

    ``runtime.load`` follows the ``final_export`` record to the checkpoint it
    names, the previous step's, and checks that file against the record. A
    record in which ``final_export`` wrote its own ``NNN_final_export.frame.h5``
    is read the same way, although this command does not resume such a
    directory with later builder code: the recorded run config holds the
    builder's source digest, and a resume compares it.
    """

    _ensure_capital_gains_tail_manifest(
        runtime.metadata[PUF_CAPITAL_GAINS_TAIL_STAGE_NAME]
    )
    loaded = runtime.load("final_export")
    metadata = runtime.metadata["final_export"]
    output_h5 = Path(str(metadata["output_h5"]))
    summary_path = Path(str(metadata["summary_path"]))
    output_valid = output_h5.is_file() and _sha256(output_h5) == metadata.get(
        "output_sha256"
    )
    summary_valid = summary_path.is_file() and _sha256(summary_path) == metadata.get(
        "summary_sha256"
    )
    if not output_valid or not summary_valid:
        regenerated = _export_staged_result(args, loaded.frame, runtime.metadata)
        for key in ("output_sha256", "summary_sha256"):
            if regenerated[key] != metadata.get(key):
                raise RuntimeError(
                    f"Recreated final artifact {key} differs from the committed "
                    "stage metadata; refusing to bless a non-reproducible resume."
                )
    _link_all_stage_checkpoint(args.checkpoint_dir, loaded.path)


def _source_construction_stage(
    args: argparse.Namespace,
) -> tuple[Frame, dict[str, object]]:
    frame, base_source = _load_base_frame_from_args(args)
    weeks_path = (
        args.asec_2023_weeks_unemployed_source
        if args.asec_2023_weeks_unemployed_source is not None
        else fetch_asec_2023_weeks_unemployed_source()
    )
    return frame, {
        "base_source": base_source,
        "base_rows": _row_counts(frame),
        "base_household_weight_total": float(frame.weights_for("household").total),
        "weeks_unemployed_source_path": str(Path(weeks_path).resolve()),
    }


def _ensure_asec_raw_stage_checkpoint(
    args: argparse.Namespace,
    source_frame: Frame,
    source_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Verify or atomically repair the auxiliary operator-untouched artifact."""

    if args.asec_h5 is None:
        raise ValueError("ASEC raw-stage checkpoint requires pooled --asec-h5 inputs.")
    source_receipt = source_metadata.get("base_source")
    if not isinstance(source_receipt, Mapping) or (
        source_receipt.get("kind") != "pooled_asec"
    ):
        raise ValueError(
            "ASEC raw-stage checkpoint requires the source-construction "
            "pooled_asec receipt."
        )
    raw_frame, raw_source_mappings = _asec_raw_source_mapping_frame(
        args,
        source_frame,
        weeks_path=Path(str(source_metadata["weeks_unemployed_source_path"])),
    )
    metadata: dict[str, object] = json.loads(
        json.dumps(
            {
                "artifact_kind": ASEC_RAW_STAGE_ARTIFACT_KIND,
                "identity": frame_identity(raw_frame).to_payload(),
                "operator_status": ASEC_RAW_STAGE_OPERATOR_STATUS,
                "pipeline_sha256": _pipeline_for(args).sha256,
                "raw_source_mappings": raw_source_mappings,
                "schema_version": ASEC_RAW_STAGE_SCHEMA_VERSION,
                "source_construction_identity": frame_identity(
                    source_frame
                ).to_payload(),
                "source_receipt": dict(source_receipt),
                "stage": ASEC_RAW_STAGE_STAGE,
            },
            allow_nan=False,
            sort_keys=True,
        )
    )
    checkpoint_path = args.checkpoint_dir / ASEC_RAW_STAGE_CHECKPOINT_FILENAME
    if checkpoint_path.is_file():
        try:
            existing_frame, existing_metadata = load_asec_raw_stage_checkpoint(
                checkpoint_path
            )
        except (OSError, TypeError, ValueError):
            pass
        else:
            if existing_metadata == metadata and _frames_exactly_equal(
                existing_frame,
                raw_frame,
            ):
                return existing_metadata

    write_frame_checkpoint(checkpoint_path, raw_frame, metadata=metadata)
    return metadata


def _frames_exactly_equal(left: Frame, right: Frame) -> bool:
    """Compare every persisted Frame component without value coercion."""

    if (
        left.schema != right.schema
        or left.entities != right.entities
        or left.links != right.links
        or left.weighted_entities != right.weighted_entities
        or left.mass_log != right.mass_log
        or left.metadata != right.metadata
        or not left.strata.equals(right.strata)
    ):
        return False
    if any(
        not left.table(entity).equals(right.table(entity)) for entity in left.entities
    ):
        return False
    if any(not left.link(name).equals(right.link(name)) for name in left.links):
        return False
    return all(
        left.weights_for(entity).kind == right.weights_for(entity).kind
        and np.array_equal(
            left.weights_for(entity).values,
            right.weights_for(entity).values,
        )
        for entity in left.weighted_entities
    )


def _asec_raw_source_mapping_frame(
    args: argparse.Namespace,
    source_frame: Frame,
    *,
    weeks_path: Path,
) -> tuple[Frame, dict[str, object]]:
    """Copy source construction and add only exact measured ASEC joins."""

    weeks_source = load_asec_2023_weeks_unemployed_source(weeks_path)
    education_source = load_asec_education_assistance_sources(
        _asec_education_source_paths(args),
        income_years=_pooled_income_years(args),
    )
    public_assistance_type_source = load_asec_public_assistance_type_sources(
        _asec_education_source_paths(args),
        income_years=_pooled_income_years(args),
    )
    tables = {
        entity: source_frame.table(entity).copy(deep=True)
        for entity in source_frame.entities
    }
    person = fill_asec_2022_weeks_unemployed_source(
        tables["person"],
        weeks_source,
    )
    person = fill_asec_education_assistance_source(person, education_source)
    person = fill_asec_public_assistance_type_source(
        person,
        public_assistance_type_source,
    )
    tables["person"] = person
    raw_frame = Frame(
        tables,
        source_frame.schema,
        {
            entity: Weights(
                source_frame.weights_for(entity).values.copy(),
                source_frame.weights_for(entity).kind,
            )
            for entity in source_frame.weighted_entities
        },
        source_frame.strata.copy(deep=True),
        mass_log=source_frame.mass_log,
        metadata=source_frame.metadata,
    )
    assert_operator_free_source_frame(
        raw_frame,
        label="producer ASEC raw_source_mapping artifact",
    )
    education_pins = [
        {
            "income_year": income_year,
            "locator": ASEC_EDUCATION_ASSISTANCE_ARCHIVES[income_year].zip_url,
            "member": ASEC_EDUCATION_ASSISTANCE_ARCHIVES[income_year].member,
            "member_sha256": (
                ASEC_EDUCATION_ASSISTANCE_ARCHIVES[income_year].member_sha256
            ),
            "sha256": ASEC_EDUCATION_ASSISTANCE_ARCHIVES[income_year].zip_sha256,
        }
        for income_year in _pooled_income_years(args)
    ]
    return raw_frame, {
        "ED_VAL": {
            "audit": dict(education_source.attrs.get("source_audit", {})),
            "column": "ED_VAL",
            "entity": "person",
            "join_keys": ["source_year", "PERIDNUM"],
            "operation": "exact_source_join",
            "source_pins": education_pins,
        },
        "LKWEEKS": {
            "audit": dict(weeks_source.attrs.get("source_audit", {})),
            "column": "LKWEEKS",
            "entity": "person",
            "join_keys": ["source_year", "PERIDNUM"],
            "operation": "exact_source_join",
            "source_pins": [
                {
                    "income_year": ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_YEAR,
                    "locator": ASEC_2023_WEEKS_UNEMPLOYED_ZIP_URL,
                    "member": ASEC_2023_WEEKS_UNEMPLOYED_MEMBER,
                    "member_sha256": ASEC_2023_WEEKS_UNEMPLOYED_MEMBER_SHA256,
                    "sha256": ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256,
                }
            ],
        },
        # PAW_TYP lives in the same pinned survey-year person members as
        # ED_VAL, so the mapping reuses those archive pins (microcosm#591).
        "PAW_TYP": {
            "audit": dict(public_assistance_type_source.attrs.get("source_audit", {})),
            "column": "PAW_TYP",
            "entity": "person",
            "join_keys": ["source_year", "PERIDNUM"],
            "operation": "exact_source_join",
            "source_pins": education_pins,
        },
    }


def _pre_clone_enrichment_stage(
    args: argparse.Namespace,
    raw_base: Frame,
    source_metadata: dict[str, object],
) -> tuple[Frame, dict[str, object]]:
    weeks_path = Path(str(source_metadata["weeks_unemployed_source_path"]))
    weeks_source = load_asec_2023_weeks_unemployed_source(weeks_path)
    public_assistance_type_source = load_asec_public_assistance_type_sources(
        _asec_education_source_paths(args),
        income_years=_pooled_income_years(args),
    )
    base = derive_us_cps_carried_inputs(
        raw_base,
        public_assistance_type_source=public_assistance_type_source,
    )
    base = with_us_prior_year_income_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_relationship_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals: dict[str, object] = {
        "relationship_inputs_signal": _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_relationship_inputs",
                lambda: us_relationship_inputs_signal_gate(base),
                base,
            ),
            "Relationship-input signal gate failed",
        )
    }
    base = with_us_spm_independence_role(
        base,
        seed=args.seed,
        time_period=args.target_year,
        asec_spm_role_source_paths=_asec_education_source_paths(args),
    )
    signals["spm_independence_role_signal"] = _checked_gate_payload(
        _shadow_frame_gate(
            "us_base_spm_independence_role",
            lambda: us_spm_independence_role_signal_gate(base),
            base,
        ),
        "SPM independence role signal gate failed before support cloning",
    )
    base = with_us_medicare_take_up_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals["medicare_take_up_input_signal"] = _checked_gate_payload(
        _shadow_frame_gate(
            "us_base_medicare_take_up_pre_clone",
            lambda: us_medicare_take_up_signal_gate(base),
            base,
        ),
        "Medicare take-up input signal gate failed before support cloning",
    )
    housing_gate = us_housing_inputs_signal_gate(base)
    acs_rent_donor: pd.DataFrame | None = None
    if not housing_gate.passed:
        if args.acs_h5 is None:
            _observe_frame_gate("us_base_housing_inputs_pre_clone", housing_gate, base)
            raise SystemExit(
                "Housing-input signal gate is not already green and --acs-h5 "
                "was not provided; exact pre_subsidy_rent restoration requires "
                "the pinned ACS 2022 donor."
            )
        acs_rent_donor = load_acs_2022_rent_donor(args.acs_h5)
        base = with_us_housing_inputs(
            base,
            seed=args.seed,
            time_period=args.target_year,
            acs_rent_donor=acs_rent_donor,
        )
        housing_gate = us_housing_inputs_signal_gate(base)
    _observe_frame_gate("us_base_housing_inputs_pre_clone", housing_gate, base)
    signals["housing_inputs_signal"] = _checked_gate_payload(
        housing_gate,
        "Housing-input signal gate failed before support cloning",
    )
    base = with_us_eligibility_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals["eligibility_inputs_signal"] = _checked_gate_payload(
        _shadow_frame_gate(
            "us_base_eligibility_inputs",
            lambda: us_eligibility_inputs_signal_gate(base),
            base,
        ),
        "Eligibility-input signal gate failed before support cloning",
    )
    base = with_us_pregnancy_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals["pregnancy_signal"] = _checked_gate_payload(
        _shadow_frame_gate(
            "us_base_pregnancy", lambda: us_pregnancy_signal_gate(base), base
        ),
        "Pregnancy signal gate failed before support cloning",
    )
    base = with_us_wic_claim_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals["wic_claim_signal"] = _checked_gate_payload(
        _shadow_frame_gate(
            "us_base_wic_claim_pre_clone", lambda: us_wic_claim_signal_gate(base), base
        ),
        "WIC-claim signal gate failed before support cloning",
    )
    base = with_us_child_support_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_disability_benefits(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_workers_compensation(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_weeks_unemployed(
        base,
        seed=args.seed,
        time_period=args.target_year,
        asec_2023_source=weeks_source,
    )
    base = with_us_childcare_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_energy_subsidy_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_retirement_contribution_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_retirement_distribution_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    base = with_us_esi_premium_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals["esi_premiums_signal"] = _checked_gate_payload(
        _shadow_frame_gate(
            "us_base_esi_premiums_pre_clone",
            lambda: us_esi_premiums_signal_gate(base),
            base,
        ),
        "ESI premium signal gate failed before support cloning",
    )
    base = with_us_immigration_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    return base, {
        "acs_h5": str(args.acs_h5.resolve()) if args.acs_h5 is not None else None,
        "acs_sha256": _sha256(args.acs_h5) if args.acs_h5 is not None else None,
        "acs_rent_donor_rows": (
            int(len(acs_rent_donor)) if acs_rent_donor is not None else None
        ),
        "signals": signals,
        "weeks_unemployed_source": {
            "path": str(weeks_path),
            "sha256": _sha256(weeks_path),
            "upstream_archive_sha256": ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256,
            "rows": int(len(weeks_source)),
            "audit": dict(weeks_source.attrs.get("source_audit", {})),
        },
    }


def _checked_gate_payload(gate, label: str) -> dict[str, object]:
    passed = bool(gate.passed)
    failures = list(gate.failures)
    details = dict(gate.details)
    if not passed:
        raise SystemExit(f"{label}:\n  " + "\n  ".join(failures))
    return {"passed": passed, "failures": failures, "details": details}


def _clone_feature_extraction_stage(
    args: argparse.Namespace,
    base: Frame,
) -> tuple[Frame, dict[str, object]]:
    expanded = clone_us_frame_for_puf_support(base)
    donor_build_summary: dict[str, object] = {}
    donor = _puf_tax_unit_donor_from_h5(
        args.puf_h5,
        source_puf_csv=args.puf_source_year_csv,
        donor_build_summary=donor_build_summary,
    )
    e01000_reconciliation_basis = _puf_e01000_reconciliation_basis(
        args,
        donor,
        donor_build_summary,
    )
    qrf_dir = args.checkpoint_dir / "primary_qrf"
    if qrf_dir.exists():
        # The outer context marks clone_feature_extraction only after both its
        # Frame and chain inputs commit. A retry before that mark cannot have a
        # valid downstream target prefix, so discard only this incomplete,
        # builder-owned inner artifact and recreate it deterministically.
        shutil.rmtree(qrf_dir)
    initialize_primary_puf_qrf_chain(
        expanded,
        donor,
        qrf_dir,
        seed=args.seed,
        n_estimators=args.n_estimators,
    )
    return expanded, {
        "puf_h5": str(args.puf_h5.resolve()),
        "puf_sha256": _sha256(args.puf_h5),
        "puf_source_year_csv": str(args.puf_source_year_csv.resolve()),
        "puf_source_year_csv_sha256": _sha256(args.puf_source_year_csv),
        "puf_donor_rows": int(len(donor)),
        "puf_donor_columns": sorted(donor.columns.tolist()),
        "puf_donor_build_summary": donor_build_summary,
        "puf_e01000_reconciliation_basis": e01000_reconciliation_basis,
        "puf_e19200_agi_variable": "E00100",
        "puf_e19200_agi_period": 2015,
        "primary_qrf_checkpoint_dir": str(qrf_dir.resolve()),
    }


def _qrf_finalization_stage(
    args: argparse.Namespace,
    expanded: Frame,
) -> tuple[Frame, dict[str, object]]:
    tail_bound_diagnostics: list[dict[str, object]] = []
    imputed, weight_kind = finalize_primary_puf_qrf_chain(
        expanded,
        args.checkpoint_dir / "primary_qrf",
        tail_bound_diagnostics=tail_bound_diagnostics,
    )
    fit_records = [FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, weight_kind)]
    report = shadow_gate(
        "us_base_weights_audit",
        lambda: weights_audit_gate(fit_records),
        lambda: EvidenceContext(artifacts={"fit_weight_records": fit_records}),
    )
    if not report.passed:
        raise SystemExit("Weights audit failed:\n  " + "\n  ".join(report.failures))
    return imputed, {
        "weights_audit": {
            "passed": report.passed,
            "failures": list(report.failures),
            "details": dict(report.details),
        },
        "puf_tax_detail_tail_bounds": tail_bound_diagnostics,
    }


def _capital_gains_tail_transfer_stage(
    args: argparse.Namespace,
    frame: Frame,
    *,
    donor: pd.DataFrame | None = None,
) -> tuple[Frame, dict[str, object]]:
    """Transfer the declared PUF CG tail and bind its standalone manifest."""

    tail_donor = donor
    if tail_donor is None:
        tail_donor = _puf_tax_unit_donor_from_h5(
            args.puf_h5,
            source_puf_csv=args.puf_source_year_csv,
        )
    transferred, manifest = transfer_puf_capital_gains_tail(
        frame,
        tail_donor,
        seed=args.seed,
    )
    ceiling = manifest["tail_distribution_receipts"]["frame_after_stage"]
    if not ceiling["positive_mass_five_x_target_exceeded"]:
        raise ValueError(
            "PUF capital-gains tail transfer did not clear its declared "
            "five-times positive-mass target: "
            f"{ceiling['positive_mass_five_x_ceiling']} <= "
            f"{ceiling['positive_mass_five_x_target']}."
        )
    manifest_filename = _capital_gains_tail_manifest_filename(args.target_year)
    manifest_path = args.out.resolve() / manifest_filename
    checkpoint_dir = getattr(args, "checkpoint_dir", None)
    checkpoint_manifest_path = (
        manifest_path
        if checkpoint_dir is None
        else checkpoint_dir.resolve() / "artifacts" / manifest_filename
    )
    manifest_file_sha256 = write_puf_capital_gains_tail_manifest(
        manifest_path,
        manifest,
    )
    checkpoint_manifest_file_sha256 = (
        manifest_file_sha256
        if checkpoint_manifest_path == manifest_path
        else write_puf_capital_gains_tail_manifest(
            checkpoint_manifest_path,
            manifest,
        )
    )
    if checkpoint_manifest_file_sha256 != manifest_file_sha256:
        raise AssertionError(
            "PUF capital-gains tail output and checkpoint manifests differ."
        )
    return transferred, {
        "manifest_path": str(manifest_path),
        "checkpoint_manifest_path": str(checkpoint_manifest_path),
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_sha256": manifest["manifest_sha256"],
        "donor_records_sha256": manifest["donor_records_sha256"],
        "assignment_sha256": manifest["assignment_sha256"],
        "record_count": manifest["record_count"],
        "boundary": manifest["boundary"],
        "weight_domain": manifest["weight_domain"],
        "joint_vector_columns": manifest["joint_vector_columns"],
        "joint_vector_policy": manifest["joint_vector_policy"],
        "clone": manifest["clone"],
        "carrier_reconciliation": manifest["carrier_reconciliation"],
        "tail_distribution_receipts": manifest["tail_distribution_receipts"],
        "signed_leg_reconciliation": manifest["signed_leg_reconciliation"],
        "tail_concentration_gate": manifest["tail_concentration_gate"],
        "frame_after_stage_concentration_gate": manifest[
            "frame_after_stage_concentration_gate"
        ],
    }


def _ensure_capital_gains_tail_manifest(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    """Validate both tail-manifest copies and repair either from the other."""

    expected_file_sha256 = str(metadata["manifest_file_sha256"])
    paths = tuple(
        dict.fromkeys(
            Path(str(metadata[key]))
            for key in ("manifest_path", "checkpoint_manifest_path")
        )
    )
    valid_manifest: dict[str, object] | None = None
    failures: list[str] = []
    for path in paths:
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            file_sha256 = _sha256(path)
            if file_sha256 != expected_file_sha256:
                raise ValueError(f"file SHA {file_sha256} != {expected_file_sha256}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("payload is not a JSON object")
            validate_puf_capital_gains_tail_manifest(payload)
            for key in (
                "manifest_sha256",
                "donor_records_sha256",
                "assignment_sha256",
                "record_count",
            ):
                if payload.get(key) != metadata.get(key):
                    raise ValueError(
                        f"{key} {payload.get(key)!r} != {metadata.get(key)!r}"
                    )
            valid_manifest = payload
            break
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            failures.append(f"{path}: {exc}")
    if valid_manifest is None:
        raise RuntimeError(
            "No valid PUF capital-gains tail manifest copy remains:\n  "
            + "\n  ".join(failures)
        )

    for path in paths:
        if path.is_file() and _sha256(path) == expected_file_sha256:
            continue
        repaired_sha256 = write_puf_capital_gains_tail_manifest(
            path,
            valid_manifest,
        )
        if repaired_sha256 != expected_file_sha256:
            raise RuntimeError(
                "Repaired PUF capital-gains tail manifest differs from committed "
                f"metadata at {path}: {repaired_sha256} != {expected_file_sha256}."
            )
    return valid_manifest


def _capital_gain_distributions_stage(
    args: argparse.Namespace,
    frame: Frame,
) -> tuple[Frame, dict[str, object]]:
    """Run the declared tax-unit memo split and restore person placement."""

    stage = US_SOURCE_MANIFEST.stage_map()[CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME]
    if stage.grain != "tax_unit":
        raise ValueError(
            f"{CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME!r} must run at tax_unit "
            f"grain, got {stage.grain!r}."
        )
    split_operations = [
        operation
        for operation in stage.operations
        if operation.kind == "split_component_by_share"
    ]
    if len(split_operations) != 1:
        raise ValueError(
            f"{CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME!r} must declare exactly "
            "one split_component_by_share operation."
        )
    split_parameters = split_operations[0].parameters
    source_column = split_parameters.get("source_column")
    exclusive_with = split_parameters.get("exclusive_with")
    output = split_parameters.get("output")
    if (
        not isinstance(source_column, str)
        or not isinstance(exclusive_with, (list, tuple))
        or not all(isinstance(column, str) for column in exclusive_with)
        or not isinstance(output, str)
        or stage.outputs != (output,)
    ):
        raise ValueError(
            f"{CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME!r} has a malformed "
            "source/output contract."
        )

    # The PUF QRF materializes these PolicyEngine inputs on people, while the
    # source-stage contract computes once per tax unit. Frame.place is the
    # kernel's entity-grain seam; the working frame is disposable so the
    # original person inputs remain byte-for-byte untouched.
    staged = frame.place(source_column, stage.grain, how="sum")
    for column in exclusive_with:
        staged = staged.place(column, stage.grain, how="sum")
    if any(output in frame.table(entity).columns for entity in frame.entities):
        # Surface an existing person-grain output to the executor so its
        # contract-owned overwrite refusal remains the rerun failure.
        staged = staged.place(output, stage.grain, how="sum")

    read_operations = [
        operation for operation in stage.operations if operation.kind == "read_table"
    ]
    if len(read_operations) != 1:
        raise ValueError(
            f"{CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME!r} must declare exactly "
            "one read_table operation."
        )
    read_parameters = read_operations[0].parameters
    table_name = read_parameters.get("table")
    weight_column = read_parameters.get("weight")
    if table_name != stage.grain or not isinstance(weight_column, str):
        raise ValueError(
            f"{CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME!r} read_table contract "
            "must name its tax_unit table and weight."
        )
    stage_data = staged.table(stage.grain).copy(deep=True)
    stage_data[weight_column] = frame.resolve_weights(stage.grain).values
    source_output = run_source_stage(
        stage,
        tables={table_name: stage_data},
        operation_handlers=us_source_operation_handlers(),
        config=SourceRuntimeConfig(
            seed=args.seed,
            target_year=args.target_year,
        ),
    )

    id_column = frame.schema.entity_id_column(stage.grain)
    if id_column not in source_output or output not in source_output:
        raise RuntimeError(
            f"{CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME!r} executor output is "
            f"missing {id_column!r} or {output!r}."
        )
    output_by_id = source_output.set_index(id_column)[output]
    tax_unit = frame.table(stage.grain)
    aligned_output = output_by_id.reindex(tax_unit[id_column])
    if aligned_output.isna().any():
        raise RuntimeError(
            f"{CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME!r} executor output does "
            "not cover every tax unit."
        )

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables[stage.grain][output] = aligned_output.to_numpy(dtype=np.float64)
    result = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )
    # policyengine-us defines the memo leg as a person input; the stage
    # derives it once per tax unit (LTCG x SOCA share), so any within-unit
    # placement that preserves the unit sum is equivalent for filing-unit
    # tax outcomes. Deterministic first-person carry does that without
    # inventing a filer/spouse allocation; the QRF path's basis-share
    # distribution is for person-consumed outputs and does not apply to a
    # unit-derived memo.
    return result.place(output, frame.schema.person_entity, how="head"), {}


def _post_qrf_frame_stage(
    stage: str,
    args: argparse.Namespace,
    frame: Frame,
    stage_metadata: dict[str, dict[str, object]],
) -> tuple[Frame, dict[str, object]]:
    signals: dict[str, object] = {}
    metadata: dict[str, object] = {"signals": signals}
    if stage == "qbi_reconciliation":
        frame = with_us_qbi_input_reconciliation(frame)
    elif stage == "wic_post_clone":
        frame = with_us_wic_claim_input(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["wic_claim_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_wic_claim_post_clone",
                lambda: us_wic_claim_signal_gate(frame),
                frame,
            ),
            "WIC-claim signal gate failed after support cloning",
        )
        signals["medicare_take_up_input_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_medicare_take_up_post_clone",
                lambda: us_medicare_take_up_signal_gate(frame),
                frame,
            ),
            "Medicare take-up input signal gate failed after support cloning",
        )
    elif stage == "housing_assistance":
        frame = impute_us_housing_assistance_to_puf_support(frame, seed=args.seed)
    elif stage == "prior_year_income_post_clone":
        frame = with_us_prior_year_income_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["prior_year_income_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_prior_year_income",
                lambda: us_prior_year_income_signal_gate(frame),
                frame,
            ),
            "Prior-year-income signal gate failed",
        )
        signals["prior_year_income_source_reconciliation"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_prior_year_income_source_reconciliation",
                lambda: us_prior_year_income_source_reconciliation_gate(frame),
                frame,
            ),
            "Prior-year-income source reconciliation failed",
        )
        signals["housing_inputs_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_housing_inputs_post_clone",
                lambda: us_housing_inputs_signal_gate(frame),
                frame,
            ),
            "Housing-input signal gate failed after PUF-support imputation",
        )
        signals["qbi_inputs_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_qbi_inputs", lambda: us_qbi_inputs_signal_gate(frame), frame
            ),
            "QBI-input signal gate failed",
        )
        signals["farm_business_income_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_farm_business_income",
                lambda: us_farm_business_income_signal_gate(frame),
                frame,
            ),
            "Farm-business-income signal gate failed",
        )
        signals["domestic_production_ald_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_domestic_production_ald",
                lambda: us_domestic_production_ald_signal_gate(frame),
                frame,
            ),
            "Domestic-production-ALD signal gate failed",
        )
    elif stage == "child_support_post_clone":
        frame = with_us_child_support_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["child_support_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_child_support",
                lambda: us_child_support_signal_gate(frame),
                frame,
            ),
            "Child-support signal gate failed",
        )
    elif stage == "disability_benefits_post_clone":
        frame = with_us_disability_benefits(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["disability_benefits_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_disability_benefits",
                lambda: us_disability_benefits_signal_gate(frame),
                frame,
            ),
            "Disability-benefits signal gate failed",
        )
    elif stage == "workers_compensation_post_clone":
        frame = with_us_workers_compensation(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["workers_compensation_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_workers_compensation",
                lambda: us_workers_compensation_signal_gate(frame),
                frame,
            ),
            "Workers-compensation signal gate failed",
        )
    elif stage == "weeks_unemployed_post_clone":
        source = stage_metadata["source_construction"]
        recorded_weeks = stage_metadata["pre_clone_enrichment"][
            "weeks_unemployed_source"
        ]
        weeks_path = Path(str(source["weeks_unemployed_source_path"]))
        actual_weeks_sha256 = _sha256(weeks_path)
        if actual_weeks_sha256 != recorded_weeks["sha256"]:
            raise SystemExit(
                "Weeks-unemployed source changed between pre-clone and post-clone "
                f"stages: expected {recorded_weeks['sha256']}, got "
                f"{actual_weeks_sha256}. Start a new checkpoint directory."
            )
        weeks_source = load_asec_2023_weeks_unemployed_source(weeks_path)
        frame = with_us_weeks_unemployed(
            frame,
            seed=args.seed,
            time_period=args.target_year,
            asec_2023_source=weeks_source,
        )
        signals["weeks_unemployed_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_weeks_unemployed",
                lambda: us_weeks_unemployed_signal_gate(frame),
                frame,
            ),
            "Weeks-unemployed signal gate failed",
        )
        signals["educator_expense_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_educator_expense",
                lambda: us_educator_expense_signal_gate(frame),
                frame,
            ),
            "Educator-expense signal gate failed",
        )
        signals["form_4952_election_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_form_4952_election",
                lambda: us_form_4952_election_signal_gate(frame),
                frame,
            ),
            "Form 4952 election signal gate failed",
        )
        signals["salt_refund_income_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_salt_refund_income",
                lambda: us_salt_refund_income_signal_gate(frame),
                frame,
            ),
            "SALT-refund-income signal gate failed",
        )
        signals["capital_gain_details_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_capital_gain_details",
                lambda: us_capital_gain_details_signal_gate(frame),
                frame,
            ),
            "Capital-gain details signal gate failed",
        )
    elif stage == "childcare_post_clone":
        frame = with_us_childcare_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["childcare_inputs_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_childcare", lambda: us_childcare_signal_gate(frame), frame
            ),
            "Childcare-input signal gate failed",
        )
    elif stage == "adult_care_post_clone":
        frame = with_us_adult_care_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["adult_care_inputs_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_adult_care", lambda: us_adult_care_signal_gate(frame), frame
            ),
            "Adult-care input signal gate failed",
        )
    elif stage == "energy_subsidy_post_clone":
        frame = with_us_energy_subsidy_input(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["energy_subsidy_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_energy_subsidy",
                lambda: us_energy_subsidy_signal_gate(frame),
                frame,
            ),
            "Energy-subsidy signal gate failed",
        )
        signals["alimony_inputs_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_alimony", lambda: us_alimony_signal_gate(frame), frame
            ),
            "Alimony-input signal gate failed",
        )
        signals["casualty_loss_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_casualty_loss",
                lambda: us_casualty_loss_signal_gate(frame),
                frame,
            ),
            "Casualty-loss signal gate failed",
        )
        signals["misc_itemized_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_misc_itemized",
                lambda: us_misc_itemized_signal_gate(frame),
                frame,
            ),
            "Miscellaneous-itemized signal gate failed",
        )
    elif stage == "retirement_contributions_post_clone":
        frame = with_us_retirement_contribution_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["retirement_contributions_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_retirement_contributions",
                lambda: us_retirement_contributions_signal_gate(frame),
                frame,
            ),
            "Retirement-contribution signal gate failed",
        )
    elif stage == "retirement_distributions_post_clone":
        frame = with_us_retirement_distribution_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
            # Resuming this named base-builder stage is equivalent to crossing
            # the live post-clone ownership boundary above.
            force_puf_imputation=True,
        )
        signals["retirement_distributions_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_retirement_distributions",
                lambda: us_retirement_distributions_signal_gate(frame),
                frame,
            ),
            "Retirement-distribution signal gate failed",
        )
    elif stage == "education_inputs_post_clone":
        education_assistance_source = load_asec_education_assistance_sources(
            _asec_education_source_paths(args),
            income_years=_pooled_income_years(args),
        )
        frame = with_us_education_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
            asec_education_source=education_assistance_source,
        )
        signals["education_inputs_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_education_inputs",
                lambda: us_education_inputs_signal_gate(frame),
                frame,
            ),
            "Education-input signal gate failed",
        )
        signals["esi_premiums_post_clone_signal"] = _checked_gate_payload(
            _shadow_frame_gate(
                "us_base_esi_premiums_post_clone",
                lambda: us_esi_premiums_signal_gate(frame),
                frame,
            ),
            "ESI premium signal gate failed after support cloning",
        )
    elif stage == "congressional_district_assignment":
        assignment: dict[str, object] = {"applied": False}
        if args.assign_congressional_districts:
            ledger_facts = load_ledger_consumer_artifact(args.ledger_facts).facts
            if args.congressional_district_vintage_crosswalk is not None:
                ledger_facts = (
                    translate_congressional_district_facts_to_current_vintage(
                        ledger_facts,
                        load_congressional_district_vintage_crosswalk(
                            args.congressional_district_vintage_crosswalk
                        ),
                    )
                )
            distribution = congressional_district_distribution_from_ledger_facts(
                ledger_facts
            )
            frame = with_household_congressional_districts(
                frame,
                distribution,
                seed=args.congressional_district_seed,
            )
            assignment = congressional_district_assignment_summary(
                frame.table("household"), distribution
            )
            assignment.update(
                {
                    "ledger_facts": str(args.ledger_facts.resolve()),
                    "ledger_facts_sha256": _sha256(args.ledger_facts),
                    "congressional_district_vintage_crosswalk": (
                        str(args.congressional_district_vintage_crosswalk.resolve())
                        if args.congressional_district_vintage_crosswalk is not None
                        else None
                    ),
                    "congressional_district_vintage_crosswalk_sha256": (
                        _sha256(args.congressional_district_vintage_crosswalk)
                        if args.congressional_district_vintage_crosswalk is not None
                        else None
                    ),
                    "seed": args.congressional_district_seed,
                }
            )
        metadata["congressional_district_assignment"] = assignment
    elif stage == "block_ladder_assignment":
        assignment = {
            "applied": False,
            "opted_out": bool(args.without_block_ladder),
        }
        _shadow_legacy_location_not_run(args)
        if args.block_ladder_artifact is not None:
            ladder = load_us_block_ladder(args.block_ladder_artifact)
            frame = with_household_us_geography_ladder(
                frame,
                ladder,
                seed=args.geography_ladder_seed,
                expected_congressional_district_vintage=(
                    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
                ),
            )
            household = frame.table("household")
            household_weights = frame.weights_for("household").values
            gate = shadow_gate(
                "us_base_geography_ladder",
                lambda: us_geography_ladder_gate(household, household_weights),
                lambda: EvidenceContext(
                    artifacts={
                        "household": household,
                        "household_weights": household_weights,
                    }
                ),
            )
            if not gate.passed and not args.allow_geography_ladder_gate_failures:
                raise SystemExit(
                    "Geography-ladder gate failed:\n  " + "\n  ".join(gate.failures)
                )
            assignment = us_geography_ladder_assignment_summary(
                household,
                ladder,
                weight_values=household_weights,
            )
            assignment.update(
                {
                    "artifact": str(args.block_ladder_artifact.resolve()),
                    "artifact_sha256": _sha256(args.block_ladder_artifact),
                    "seed": args.geography_ladder_seed,
                    "gate": {
                        "passed": gate.passed,
                        "failures": list(gate.failures),
                        "details": dict(gate.details),
                    },
                }
            )
        metadata["geography_ladder_assignment"] = assignment
    else:
        raise ValueError(f"Unknown post-QRF frame stage {stage!r}.")
    return frame, metadata


def _export_staged_result(
    args: argparse.Namespace,
    frame: Frame,
    stage_metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    _ensure_capital_gains_tail_manifest(
        stage_metadata[PUF_CAPITAL_GAINS_TAIL_STAGE_NAME]
    )
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = out_dir / _dataset_filename(args.target_year)
    summary_path = out_dir / _summary_filename(args.target_year)
    _write_policyengine_dataset(args, frame, output_h5)
    household_location: dict[str, object] | None = None
    if _is_block_v1(args):
        household_location = stage_metadata[HOUSEHOLD_LOCATION_STAGE_NAME][
            "household_location"
        ]
        congressional: dict[str, object] = {
            "applied": False,
            "replaced_by": "household_location",
        }
        geography: dict[str, object] = {
            "applied": False,
            "opted_out": False,
            "replaced_by": "household_location",
        }
    else:
        congressional = stage_metadata["congressional_district_assignment"][
            "congressional_district_assignment"
        ]
        geography = stage_metadata["block_ladder_assignment"][
            "geography_ladder_assignment"
        ]
    if household_location is not None:
        import h5py

        with h5py.File(output_h5, "a") as h5:
            _write_block_v1_location_attrs(h5.attrs, household_location)
    elif (
        args.congressional_district_vintage_crosswalk is not None
        or args.block_ladder_artifact is not None
    ):
        import h5py

        with h5py.File(output_h5, "a") as h5:
            if args.congressional_district_vintage_crosswalk is not None:
                h5.attrs[CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR] = (
                    congressional["congressional_district_vintage_crosswalk_sha256"]
                )
                h5.attrs[CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR] = (
                    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
                )
            if args.block_ladder_artifact is not None:
                h5.attrs[GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR] = geography[
                    "artifact_sha256"
                ]
                h5.attrs[GEOGRAPHY_LADDER_VINTAGES_ATTR] = json.dumps(
                    geography["layer_vintages"], sort_keys=True
                )

    source = stage_metadata["source_construction"]
    pre_clone = stage_metadata["pre_clone_enrichment"]
    clone = stage_metadata["clone_feature_extraction"]
    qrf = stage_metadata["qrf_finalization"]
    capital_gains_tail = stage_metadata[PUF_CAPITAL_GAINS_TAIL_STAGE_NAME]
    e01000_reconciliation = finalize_puf_e01000_reconciliation(
        clone["puf_e01000_reconciliation_basis"],
        capital_gains_tail,
        frame_columns=_frame_column_inventory(frame),
    )
    signals = _merged_stage_signals(stage_metadata)
    required_signals = (
        "qbi_inputs_signal",
        "farm_business_income_signal",
        "domestic_production_ald_signal",
        "child_support_signal",
        "disability_benefits_signal",
        "workers_compensation_signal",
        "weeks_unemployed_signal",
        "eligibility_inputs_signal",
        "pregnancy_signal",
        "esi_premiums_signal",
        "esi_premiums_post_clone_signal",
        "wic_claim_signal",
        "educator_expense_signal",
        "form_4952_election_signal",
        "salt_refund_income_signal",
        "capital_gain_details_signal",
        "childcare_inputs_signal",
        "adult_care_inputs_signal",
        "energy_subsidy_signal",
        "alimony_inputs_signal",
        "casualty_loss_signal",
        "misc_itemized_signal",
        "education_inputs_signal",
        "retirement_contributions_signal",
        "retirement_distributions_signal",
        "relationship_inputs_signal",
        "medicare_take_up_input_signal",
        "housing_inputs_signal",
        "prior_year_income_signal",
        "prior_year_income_source_reconciliation",
    )
    missing_signals = [name for name in required_signals if name not in signals]
    if missing_signals:
        raise ValueError(
            f"Staged summary is missing signal gate(s): {missing_signals}."
        )

    summary: dict[str, object] = {
        "base_source": source["base_source"],
        "base_h5": (
            source["base_source"].get("path")
            if source["base_source"].get("kind") == "base_h5"
            else None
        ),
        "base_sha256": (
            source["base_source"].get("sha256")
            if source["base_source"].get("kind") == "base_h5"
            else None
        ),
        "puf_h5": clone["puf_h5"],
        "puf_sha256": clone["puf_sha256"],
        "puf_source_year_csv": clone["puf_source_year_csv"],
        "puf_source_year_csv_sha256": clone["puf_source_year_csv_sha256"],
        "acs_h5": pre_clone["acs_h5"],
        "acs_sha256": pre_clone["acs_sha256"],
        "acs_rent_donor_rows": pre_clone["acs_rent_donor_rows"],
        "weeks_unemployed_source": pre_clone["weeks_unemployed_source"],
        "output_h5": str(output_h5),
        "output_sha256": _sha256(output_h5),
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "base_rows": source["base_rows"],
        "expanded_rows": _row_counts(frame),
        "base_household_weight_total": source["base_household_weight_total"],
        "expanded_household_weight_total": float(frame.weights_for("household").total),
        "channel_weight_totals": _channel_weight_totals(frame),
        "puf_donor_rows": clone["puf_donor_rows"],
        "puf_donor_columns": clone["puf_donor_columns"],
        "puf_donor_build_summary": clone["puf_donor_build_summary"],
        "puf_e01000_reconciliation": e01000_reconciliation,
        "weights_audit": qrf["weights_audit"],
        "puf_tax_detail_tail_bounds": qrf["puf_tax_detail_tail_bounds"],
        "puf_capital_gains_tail_transfer": capital_gains_tail,
        **{name: signals[name] for name in required_signals},
        "congressional_district_assignment": congressional,
        "geography_ladder_assignment": geography,
        "channel_output_totals": _channel_output_totals(frame),
        "immigration_composition": us_immigration_composition_summary(frame),
    }
    if household_location is not None:
        summary["household_location"] = household_location
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return {
        "output_h5": str(output_h5),
        "output_sha256": summary["output_sha256"],
        "summary_path": str(summary_path),
        "summary_sha256": _sha256(summary_path),
    }


def _merged_stage_signals(
    stage_metadata: dict[str, dict[str, object]],
) -> dict[str, object]:
    signals: dict[str, object] = {}
    for stage in PIPELINE_STEPS:
        metadata = stage_metadata.get(stage, {})
        stage_signals = metadata.get("signals", {})
        if not isinstance(stage_signals, dict):
            raise ValueError(f"Stage {stage!r} signals metadata must be an object.")
        signals.update(stage_signals)
    return signals


def impute_and_audit_us_puf_support(
    expanded: Frame,
    donor: pd.DataFrame,
    *,
    seed: int,
    n_estimators: int,
    predictors: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PREDICTORS,
    person_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    tax_unit_outputs: Sequence[str] = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    raw_predictions_callback: Callable[[pd.DataFrame], None] | None = None,
    tail_bound_diagnostics: list[dict[str, object]] | None = None,
) -> tuple[Frame, dict]:
    """Impute the PUF support channel and audit the fit's resolved weight kind.

    Runs the production PUF tax-detail support imputation, capturing the kind the
    QRF *resolved* to via the build-level weights audit (microcosm #300): the fit
    emits one :class:`~microcosm.build.FitWeightRecord`, and
    :func:`~microcosm.build.weights_audit_gate` proves it did not silently resolve
    unweighted. A failing audit **aborts the build** with a non-zero exit, exactly
    as the geography-ladder gate does — a support channel imputed by an unweighted
    fit is a broken donor whose on-surface residuals can still look perfect.

    The audit is unconditional: the PUF support fit always runs, so there is no
    opt-out, and the seed allowlist is empty because the fit is design-weighted.

    Args:
        expanded: The channel-cloned US frame the fit imputes onto.
        donor: The PUF tax-unit donor table the fit trains on.
        seed: The imputation seed.
        n_estimators: Trees per QRF forest.
        predictors: Predictor columns for the fit; defaults to the production set.
        person_outputs: Person-grain outputs; defaults to the production set.
        tax_unit_outputs: Tax-unit-grain outputs; defaults to the production set.
            The three ``*_outputs``/``predictors`` arguments exist so the seam can
            be exercised on a small synthetic frame in an engine-free test; the
            build calls this with the defaults, so production behavior is
            unchanged.
        raw_predictions_callback: Optional test-only observer for complete raw
            chained draws before finalization.
        tail_bound_diagnostics: Optional output sink for publishable per-target
            tail-bound finalization records.

    Returns:
        ``(imputed_frame, weights_audit)`` where ``weights_audit`` is the gate's
        publishable record — ``{"passed", "failures", "details"}`` — carrying
        ``details["resolved_weight_kinds"]`` (the fit-name -> resolved-kind map)
        for the release summary.

    Raises:
        SystemExit: If the weights audit fails (a fit resolved unweighted with no
            allowlist entry), naming the offending fit.
    """
    fit_records: list[FitWeightRecord] = []
    imputed = impute_us_puf_tax_detail_support(
        expanded,
        donor,
        predictors=predictors,
        person_outputs=person_outputs,
        tax_unit_outputs=tax_unit_outputs,
        seed=seed,
        n_estimators=n_estimators,
        fit_records=fit_records,
        raw_predictions_callback=raw_predictions_callback,
        tail_bound_diagnostics=tail_bound_diagnostics,
    )
    report = shadow_gate(
        "us_base_weights_audit",
        lambda: weights_audit_gate(fit_records),
        lambda: EvidenceContext(artifacts={"fit_weight_records": list(fit_records)}),
    )
    if not report.passed:
        raise SystemExit("Weights audit failed:\n  " + "\n  ".join(report.failures))
    weights_audit = {
        "passed": report.passed,
        "failures": list(report.failures),
        "details": dict(report.details),
    }
    return imputed, weights_audit


def _dataset_filename(period: int) -> str:
    if period == PERIOD:
        return DATASET_FILENAME
    return f"base_populace_us_{period}_puf_support.h5"


def _summary_filename(period: int) -> str:
    if period == PERIOD:
        return SUMMARY_FILENAME
    return f"base_populace_us_{period}_puf_support.summary.json"


def _capital_gains_tail_manifest_filename(period: int) -> str:
    return f"base_populace_us_{period}_puf_capital_gains_tail.manifest.json"


def _load_base_frame_from_args(args: argparse.Namespace) -> tuple[Frame, dict]:
    if args.base_h5 is not None:
        frame = _load_frame(args.base_h5)
        return frame, {
            "kind": "base_h5",
            "path": str(args.base_h5.resolve()),
            "sha256": _sha256(args.base_h5),
        }
    support_spine_spec = _support_spine_spec_from_args(args)
    sources = _asec_sources_from_args(
        args,
        support_spine_spec=support_spine_spec,
    )
    frame, metadata = build_pooled_asec_unit_frame(
        sources,
        target_year=args.target_year,
    )
    receipt: dict[str, object] = {
        "kind": "pooled_asec",
        "target_year": args.target_year,
        "sources": [
            {
                "year": source.year,
                "path": str(source.path.resolve()),
                "sha256": _sha256(source.path),
                "share": source.share,
                "max_households": source.max_households,
            }
            for source in sources
        ],
        "support_spine_spec": _support_spine_spec_metadata(
            args,
            support_spine_spec=support_spine_spec,
        ),
        "metadata": metadata,
    }
    _require_receipt_digests_match_pins(args, receipt)
    return frame, receipt


def _support_spine_spec_from_args(args: argparse.Namespace) -> SupportSpineSpec | None:
    if args.support_spine_spec is None:
        return None
    if args.support_spine_spec.name == "default":
        return US_SUPPORT_SPINE_SPEC
    return load_support_spine_manifest(args.support_spine_spec).support_spine


def _asec_sources_from_args(
    args: argparse.Namespace,
    *,
    support_spine_spec: SupportSpineSpec | None,
) -> tuple[AsecSource, ...]:
    """Pooled ASEC sources, each bound to its pinned Census person file.

    Every source carries ``census_person_source`` so the pool restores the
    reviewed Census person columns its H5 lacks (microcosm #720). The files
    are the ``--asec-education-source`` mappings, with any pooled income year
    left unmapped fetched from the official Census archive and verified,
    exactly as the SPM independence role stage resolves them.
    """

    sources = _pooled_asec_sources_from_args(
        args, support_spine_spec=support_spine_spec
    )
    census_person_sources = resolve_asec_spm_role_source_paths(
        _asec_education_source_paths(args),
        income_years=tuple(sorted({source.year for source in sources})),
    )
    return tuple(
        replace(source, census_person_source=census_person_sources[source.year])
        for source in sources
    )


def _pooled_asec_sources_from_args(
    args: argparse.Namespace,
    *,
    support_spine_spec: SupportSpineSpec | None,
) -> tuple[AsecSource, ...]:
    if support_spine_spec is None:
        return tuple(
            _parse_asec_source(value, max_households=args.asec_max_households)
            for value in args.asec_h5
        )
    path_by_year = _parse_asec_source_paths(args.asec_h5)
    expected_years = {
        source_spec.resolved_year(args.target_year)
        for source_spec in support_spine_spec.sources
    }
    extra_years = sorted(set(path_by_year) - expected_years)
    if extra_years:
        expected = ", ".join(str(value) for value in sorted(expected_years))
        raise ValueError(
            "Support-spine spec mode received unused --asec-h5 mapping(s) for "
            f"year(s) {extra_years}. Expected year(s): {expected or 'none'}."
        )
    sources: list[AsecSource] = []
    for source_spec in support_spine_spec.sources:
        year = source_spec.resolved_year(args.target_year)
        if year not in path_by_year:
            available = ", ".join(str(value) for value in sorted(path_by_year))
            raise ValueError(
                f"Support-spine spec source {source_spec.role!r} resolves to "
                f"ASEC year {year}, but no --asec-h5 mapping was provided for "
                f"that year. Available year(s): {available or 'none'}."
            )
        sources.append(
            AsecSource(
                year=year,
                path=path_by_year[year],
                share=source_spec.share,
                max_households=args.asec_max_households,
            )
        )
    return tuple(sources)


def _support_spine_spec_metadata(
    args: argparse.Namespace,
    *,
    support_spine_spec: SupportSpineSpec | None,
) -> dict | None:
    if support_spine_spec is None:
        return None
    return {
        "path": (
            "package:microcosm.build.us/support_spine.json"
            if args.support_spine_spec is not None
            and args.support_spine_spec.name == "default"
            else str(args.support_spine_spec.resolve())
        ),
        "stage": support_spine_spec.stage,
        "method": support_spine_spec.method,
        "target_year_from_build_config": (
            support_spine_spec.target_year_from_build_config
        ),
        "sources": [
            {
                "role": source.role,
                "survey": source.survey,
                "source": source.source,
                "source_year_offset": source.source_year_offset,
                "resolved_year": source.resolved_year(args.target_year),
                "share": source.share,
                "notes": source.notes,
            }
            for source in support_spine_spec.sources
        ],
    }


def _parse_asec_source(value: str, *, max_households: int | None) -> AsecSource:
    if "=" not in value:
        raise ValueError(f"ASEC source must be YEAR=PATH, got {value!r}.")
    raw_year, raw_path = value.split("=", 1)
    return AsecSource(
        year=int(raw_year),
        path=Path(raw_path),
        max_households=max_households,
    )


def _parse_asec_source_paths(values: list[str]) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"ASEC source must be YEAR=PATH, got {value!r}.")
        raw_year, raw_path = value.split("=", 1)
        try:
            year = int(raw_year)
        except ValueError:
            raise ValueError(
                f"--asec-h5 {value!r}: the year must be an integer."
            ) from None
        if year in paths:
            raise ValueError(f"Duplicate --asec-h5 mapping for year {year}.")
        paths[year] = Path(raw_path)
    return paths


def _load_frame(path: Path) -> Frame:
    consumer = "PUF support base --base-h5 loader (_load_frame)"
    sha256 = refuse_denied_pool_h5(path, consumer=consumer)

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


def _read_h5_arrays(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as h5:
        return {name: np.asarray(dataset) for name, dataset in h5.items()}


def _puf_tax_unit_donor_from_h5(
    path: Path,
    *,
    source_puf_csv: Path | None,
    donor_build_summary: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Build the PUF donor with source-year E00100-aligned banding values."""

    arrays = _read_h5_arrays(path)
    if source_puf_csv is None:
        raise ValueError(
            "--puf-source-year-csv is required to align nonzero E19200 records "
            "to the published TY2015 SOI AGI bands."
        )
    adjusted_gross_income = _source_year_puf_adjusted_gross_income(
        source_puf_csv,
        processed_tax_unit_ids=arrays["tax_unit_id"],
        processed_tax_unit_weights=arrays["household_weight"],
    )
    schedule_e_receipt: dict[str, object] = {}
    schedule_e = _source_year_puf_partnership_s_corp_income(
        source_puf_csv,
        arrays,
        receipt=schedule_e_receipt,
    )
    donor = puf_tax_unit_donor_from_arrays(
        arrays,
        adjusted_gross_income=adjusted_gross_income,
        schedule_e_partnership_s_corp=schedule_e,
        donor_build_summary=donor_build_summary,
    )
    if donor_build_summary is not None:
        donor_build_summary["schedule_e_line_32"] = schedule_e_receipt
    return donor


def _puf_e01000_reconciliation_basis(
    args: argparse.Namespace,
    donor: pd.DataFrame,
    donor_build_summary: Mapping[str, object],
) -> dict[str, object]:
    """Build the audit-only source-through-donor reconciliation payload."""

    source_puf_csv = getattr(args, "puf_source_year_csv", None)
    if source_puf_csv is None:
        raise ValueError(
            "--puf-source-year-csv is required for the E01000 reconciliation."
        )
    processed_before_screen = donor_build_summary.get(
        "capital_gains_before_mortgage_screen"
    )
    mortgage_quarantine = donor_build_summary.get("mortgage_field_quarantine")
    if not isinstance(processed_before_screen, Mapping):
        raise ValueError(
            "PUF donor build summary is missing pre-screen capital-gains metrics."
        )
    if not isinstance(mortgage_quarantine, Mapping):
        raise ValueError(
            "PUF donor build summary is missing mortgage quarantine metrics."
        )
    return build_puf_e01000_reconciliation_basis(
        source_puf_csv,
        donor,
        processed_before_screen=processed_before_screen,
        mortgage_screen=mortgage_quarantine,
        target_year=args.target_year,
        source_sha256=_sha256(source_puf_csv),
    )


def _source_year_puf_adjusted_gross_income(
    source_puf_csv: Path,
    *,
    processed_tax_unit_ids: Sequence[object],
    processed_tax_unit_weights: Sequence[object],
) -> np.ndarray:
    """Load the restricted source only through its narrow alignment seam."""

    return source_year_puf_adjusted_gross_income(
        source_puf_csv,
        processed_tax_unit_ids=processed_tax_unit_ids,
        processed_tax_unit_weights=processed_tax_unit_weights,
    )


def _source_year_puf_partnership_s_corp_income(
    source_puf_csv: Path,
    arrays: Mapping[str, Sequence[object]],
    *,
    receipt: dict[str, object] | None = None,
) -> dict[str, np.ndarray]:
    """Load Schedule E line 32 by entity through the same narrow seam."""

    return source_year_puf_partnership_s_corp_income(
        source_puf_csv,
        processed_tax_unit_ids=arrays["tax_unit_id"],
        processed_tax_unit_weights=arrays["household_weight"],
        processed_nonpassive_partnership_s_corp_income=(
            processed_puf_tax_unit_partnership_s_corp_income(arrays)
        ),
        receipt=receipt,
    )


def _row_counts(frame: Frame) -> dict[str, int]:
    return {entity: frame.n(entity) for entity in frame.entities}


def _frame_column_inventory(frame: Frame) -> dict[str, tuple[str, ...]]:
    """Return the final schema surface used to verify E01000 is audit-only."""

    return {
        entity: tuple(str(column) for column in frame.table(entity).columns)
        for entity in frame.entities
    }


def _channel_weight_totals(frame: Frame) -> dict[str, float]:
    household = frame.table("household")
    channel = support_channel_column("household")
    weights = pd.Series(frame.weights_for("household").values, index=household.index)
    return {
        str(name): float(weights.loc[group.index].sum())
        for name, group in household.groupby(channel, sort=True)
    }


def _channel_output_totals(frame: Frame) -> dict[str, dict[str, float]]:
    person = frame.table("person")
    tax_unit = frame.table("tax_unit")
    person_outputs = PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS
    tax_unit_outputs = PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS
    result: dict[str, dict[str, float]] = {
        BASE_ASEC_SUPPORT_CHANNEL: {},
        PUF_TAX_DETAIL_SUPPORT_CHANNEL: {},
    }
    for channel in result:
        person_mask = person[support_channel_column("person")] == channel
        tax_unit_mask = tax_unit[support_channel_column("tax_unit")] == channel
        for column in person_outputs:
            if column in person:
                result[channel][column] = float(
                    pd.to_numeric(person.loc[person_mask, column], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
        for column in tax_unit_outputs:
            if column in tax_unit:
                result[channel][column] = float(
                    pd.to_numeric(tax_unit.loc[tax_unit_mask, column], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
