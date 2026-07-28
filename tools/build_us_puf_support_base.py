"""Build a local US base H5 with a PUF tax-detail support channel.

This diagnostic builder starts from an existing Populace US H5, clones the
frame into ASEC and PUF-tax-detail support channels, imputes PUF-observed
inputs onto the PUF channel with Populace's weighted QRF, and writes a fresh
base H5 for the fiscal refresh calibration builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

from populace.build import FitWeightRecord, weights_audit_gate
from populace.build.frame_checkpoint import write_frame_checkpoint
from populace.build.ledger_artifact import load_ledger_consumer_artifact
from populace.build.outer_stage_runtime import (
    Stage,
    StagePipeline,
    StageRuntime,
    assert_clone_expansion,
    assert_unchanged_identity,
)
from populace.build.source_manifest import SupportSpineSpec, load_support_spine_manifest
from populace.build.source_runtime import SourceRuntimeConfig, run_source_stage
from populace.build.stage_profile import profile_stage
from populace.build.us_runtime import (
    ASEC_2023_WEEKS_UNEMPLOYED_SOURCE_SHA256,
    BASE_ASEC_SUPPORT_CHANNEL,
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
    GEOGRAPHY_LADDER_ARTIFACT_SHA256_ATTR,
    GEOGRAPHY_LADDER_VINTAGES_ATTR,
    PUF_TAX_DETAIL_DEFAULT_PERSON_OUTPUTS,
    PUF_TAX_DETAIL_DEFAULT_TAX_UNIT_OUTPUTS,
    PUF_TAX_DETAIL_SUPPORT_CHANNEL,
    US_PUF_SUPPORT_FIT_NAME,
    US_SOURCE_MANIFEST,
    US_SUPPORT_SPINE_SPEC,
    AsecSource,
    build_pooled_asec_unit_frame,
    clone_us_frame_for_puf_support,
    congressional_district_assignment_summary,
    congressional_district_distribution_from_ledger_facts,
    derive_us_cps_carried_inputs,
    fetch_asec_2023_weeks_unemployed_source,
    impute_us_housing_assistance_to_puf_support,
    impute_us_puf_tax_detail_support,
    load_acs_2022_rent_donor,
    load_asec_2023_weeks_unemployed_source,
    load_asec_education_assistance_sources,
    load_congressional_district_vintage_crosswalk,
    load_us_block_ladder,
    puf_tax_unit_donor_from_arrays,
    source_year_puf_adjusted_gross_income,
    support_channel_column,
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
    us_weeks_unemployed_signal_gate,
    us_wic_claim_signal_gate,
    us_workers_compensation_signal_gate,
    with_household_congressional_districts,
    with_household_us_geography_ladder,
    with_us_adult_care_inputs,
    with_us_child_support_inputs,
    with_us_childcare_inputs,
    with_us_disability_benefits,
    with_us_education_inputs,
    with_us_eligibility_inputs,
    with_us_energy_subsidy_input,
    with_us_housing_inputs,
    with_us_immigration_inputs,
    with_us_medicare_take_up_input,
    with_us_pregnancy_inputs,
    with_us_prior_year_income_inputs,
    with_us_qbi_input_reconciliation,
    with_us_relationship_inputs,
    with_us_retirement_contribution_inputs,
    with_us_retirement_distribution_inputs,
    with_us_weeks_unemployed,
    with_us_wic_claim_input,
    with_us_workers_compensation,
)
from populace.build.us_runtime.puf_qrf_chain import (
    finalize_primary_puf_qrf_chain,
    initialize_primary_puf_qrf_chain,
    run_primary_puf_qrf_chain,
)
from populace.build.us_runtime.puf_support import PUF_TAX_DETAIL_DEFAULT_PREDICTORS
from populace.frame import Frame, WeightKind, Weights
from populace.frame.adapters.policyengine_us import PolicyEngineUSEngine
from populace.frame.units import US_SCHEMA

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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--base-h5", type=Path)
    source.add_argument(
        "--asec-h5",
        action="append",
        help="Raw ASEC source as YEAR=PATH. Pass once per source year.",
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
            "pppub member) restoring that pooled income year's ED_VAL "
            "(income year YYYY maps to the survey-year YYYY+1 archive). "
            "Years without a mapping are fetched from the official Census "
            "archive and verified against the same pins."
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
        choices=(*STAGE_NAMES, "all"),
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
            "never-collapses-to-zero regression of populace #34) aborts the "
            "build."
        ),
    )
    args = parser.parse_args(argv)
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
            "(populace #275): pass --block-ladder-artifact <npz> "
            "(tools/build_us_block_ladder_artifact.py) or opt out "
            "explicitly with --without-block-ladder"
        )
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


def main(argv: list[str] | None = None) -> None:
    """Dispatch the byte-identical legacy path or checkpoint scaffolding."""

    args = _parse_args() if argv is None else _parse_args(argv)

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
        _run_configured_stage(args)
        return
    if checkpoint_dir is None:
        # This is intentionally a direct call with no profiler, checkpoint
        # directory creation, or other side effect.  It is the pre-refactor
        # pipeline and remains the byte-for-byte compatibility path.
        boundary_dir = getattr(args, "equivalence_boundary_dir", None)
        observer = (
            None if boundary_dir is None else _EquivalenceBoundaryObserver(boundary_dir)
        )
        if observer is None:
            _run_all(args)
        else:
            _run_all(args, boundary_observer=observer)
            observer.assert_complete()
        return
    _run_staged_all(args)


def _run_staged_all(args: argparse.Namespace) -> None:
    """Run every outer boundary in a fresh interpreter, resuming by prefix."""

    runtime = StageRuntime(
        args.checkpoint_dir,
        OUTER_STAGE_PIPELINE,
        run_config=_stage_run_config(args),
    )
    completed_prefix = runtime.context.completed
    remaining = PIPELINE_STEPS[len(completed_prefix) :]
    if not remaining and completed_prefix == PIPELINE_STEPS:
        # Re-enter only the fresh final child so it can validate/repair the
        # export and stage_all alias after a crash in the post-context window.
        remaining = ("final_export",)
    for stage in remaining:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                *_stage_cli_args(args, stage),
            ],
            check=False,
            env=_staged_subprocess_environment(),
        )
        if completed.returncode != 0:
            raise SystemExit(
                f"Checkpointed stage {stage!r} failed with exit code "
                f"{completed.returncode}."
            )


def _stage_cli_args(args: argparse.Namespace, stage: str) -> list[str]:
    """Reconstruct one stage invocation from validated parsed arguments."""

    command: list[str] = []
    if args.base_h5 is not None:
        command.extend(("--base-h5", str(args.base_h5)))
    else:
        for value in args.asec_h5:
            command.extend(("--asec-h5", value))
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
    if getattr(args, "equivalence_deterministic_h5_metadata", False):
        command.append("--equivalence-deterministic-h5-metadata")
    return command


def _append_path_argument(command: list[str], flag: str, value: Path | None) -> None:
    if value is not None:
        command.extend((flag, str(value)))


def _staged_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("PYTHONHASHSEED", "0")
    return environment


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


def _stage_run_config(args: argparse.Namespace) -> dict[str, object]:
    """Return the canonical inputs and settings locked across stage resumes."""

    def path(value: Path | None) -> str | None:
        return None if value is None else str(value.resolve())

    asec_sources: list[str] | None = None
    if args.asec_h5 is not None:
        asec_sources = []
        for value in args.asec_h5:
            raw_year, raw_path = value.split("=", 1)
            asec_sources.append(f"{int(raw_year)}={Path(raw_path).resolve()}")
    thread_environment = {
        name: (
            os.environ.get(name, "0")
            if name == "PYTHONHASHSEED"
            else os.environ.get(name)
        )
        for name in (
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
    return {
        "allow_geography_ladder_gate_failures": bool(
            args.allow_geography_ladder_gate_failures
        ),
        "asec_h5": asec_sources,
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
        "puf_source_year_csv": path(args.puf_source_year_csv),
        "puf_source_year_csv_sha256": (
            _sha256(args.puf_source_year_csv)
            if args.puf_source_year_csv is not None
            else None
        ),
        "seed": args.seed,
        "support_spine_spec": path(args.support_spine_spec),
        "target_year": args.target_year,
        "thread_environment": thread_environment,
        "without_block_ladder": bool(args.without_block_ladder),
    }


def _builder_code_identity() -> dict[str, object]:
    """Fingerprint executable sources and dependency versions for safe resume."""

    root = Path(__file__).resolve().parents[1]
    candidates = [Path(__file__).resolve(), root / "pyproject.toml", root / "uv.lock"]
    for source_root in sorted((root / "packages").glob("*/src")):
        candidates.extend(
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and path.suffix in {".json", ".py", ".toml", ".yaml", ".yml"}
        )
    digest = hashlib.sha256()
    for source_path in sorted(set(candidates)):
        relative = source_path.relative_to(root).as_posix().encode("utf-8")
        content = source_path.read_bytes()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    dependency_versions: dict[str, str | None] = {}
    for distribution in (
        "h5py",
        "numpy",
        "pandas",
        "policyengine-us",
        "quantile-forest",
        "scikit-learn",
    ):
        try:
            dependency_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            dependency_versions[distribution] = None
    return {
        "dependency_versions": dependency_versions,
        "python": sys.version,
        "source_sha256": digest.hexdigest(),
    }


class _EquivalenceBoundaryObserver:
    """Write test-only monolith snapshots without changing production stages."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._next_stage_index = 0

    def observe_frame(self, stage: str, frame: Frame) -> None:
        expected = PIPELINE_STEPS[self._next_stage_index]
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
                "pipeline_steps": list(PIPELINE_STEPS),
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
        if self._next_stage_index != len(PIPELINE_STEPS):
            raise AssertionError(
                "Monolith boundary observer stopped after "
                f"{self._next_stage_index}/{len(PIPELINE_STEPS)} stages."
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
    base = derive_us_cps_carried_inputs(raw_base)
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
    relationship_inputs_gate = us_relationship_inputs_signal_gate(base)
    if not relationship_inputs_gate.passed:
        raise SystemExit(
            "Relationship-input signal gate failed:\n  "
            + "\n  ".join(relationship_inputs_gate.failures)
        )
    base = with_us_medicare_take_up_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    medicare_take_up_gate = us_medicare_take_up_signal_gate(base)
    if not medicare_take_up_gate.passed:
        raise SystemExit(
            "Medicare take-up input signal gate failed before support cloning:\n  "
            + "\n  ".join(medicare_take_up_gate.failures)
        )
    housing_inputs_gate = us_housing_inputs_signal_gate(base)
    acs_rent_donor: pd.DataFrame | None = None
    if not housing_inputs_gate.passed:
        if args.acs_h5 is None:
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
    eligibility_inputs_gate = us_eligibility_inputs_signal_gate(base)
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
    pregnancy_gate = us_pregnancy_signal_gate(base)
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
    wic_claim_gate = us_wic_claim_signal_gate(base)
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
    wic_claim_gate = us_wic_claim_signal_gate(imputed)
    if not wic_claim_gate.passed:
        raise SystemExit(
            "WIC-claim signal gate failed after support cloning:\n  "
            + "\n  ".join(wic_claim_gate.failures)
        )
    medicare_take_up_gate = us_medicare_take_up_signal_gate(imputed)
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
    prior_year_income_gate = us_prior_year_income_signal_gate(imputed)
    if not prior_year_income_gate.passed:
        raise SystemExit(
            "Prior-year-income signal gate failed:\n  "
            + "\n  ".join(prior_year_income_gate.failures)
        )
    prior_year_income_reconciliation_gate = (
        us_prior_year_income_source_reconciliation_gate(imputed)
    )
    if not prior_year_income_reconciliation_gate.passed:
        raise SystemExit(
            "Prior-year-income source reconciliation failed:\n  "
            + "\n  ".join(prior_year_income_reconciliation_gate.failures)
        )
    housing_inputs_gate = us_housing_inputs_signal_gate(imputed)
    if not housing_inputs_gate.passed:
        raise SystemExit(
            "Housing-input signal gate failed after PUF-support imputation:\n  "
            + "\n  ".join(housing_inputs_gate.failures)
        )
    qbi_inputs_gate = us_qbi_inputs_signal_gate(imputed)
    if not qbi_inputs_gate.passed:
        raise SystemExit(
            "QBI-input signal gate failed:\n  " + "\n  ".join(qbi_inputs_gate.failures)
        )
    farm_business_income_gate = us_farm_business_income_signal_gate(imputed)
    if not farm_business_income_gate.passed:
        raise SystemExit(
            "Farm-business-income signal gate failed:\n  "
            + "\n  ".join(farm_business_income_gate.failures)
        )
    domestic_production_ald_gate = us_domestic_production_ald_signal_gate(imputed)
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
    child_support_gate = us_child_support_signal_gate(imputed)
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
    disability_benefits_gate = us_disability_benefits_signal_gate(imputed)
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
    workers_compensation_gate = us_workers_compensation_signal_gate(imputed)
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
    weeks_unemployed_gate = us_weeks_unemployed_signal_gate(imputed)
    if not weeks_unemployed_gate.passed:
        raise SystemExit(
            "Weeks-unemployed signal gate failed:\n  "
            + "\n  ".join(weeks_unemployed_gate.failures)
        )
    educator_expense_gate = us_educator_expense_signal_gate(imputed)
    if not educator_expense_gate.passed:
        raise SystemExit(
            "Educator-expense signal gate failed:\n  "
            + "\n  ".join(educator_expense_gate.failures)
        )
    form_4952_election_gate = us_form_4952_election_signal_gate(imputed)
    if not form_4952_election_gate.passed:
        raise SystemExit(
            "Form 4952 election signal gate failed:\n  "
            + "\n  ".join(form_4952_election_gate.failures)
        )
    salt_refund_income_gate = us_salt_refund_income_signal_gate(imputed)
    if not salt_refund_income_gate.passed:
        raise SystemExit(
            "SALT-refund-income signal gate failed:\n  "
            + "\n  ".join(salt_refund_income_gate.failures)
        )
    capital_gain_details_gate = us_capital_gain_details_signal_gate(imputed)
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
    childcare_gate = us_childcare_signal_gate(imputed)
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
    adult_care_gate = us_adult_care_signal_gate(imputed)
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
    energy_subsidy_gate = us_energy_subsidy_signal_gate(imputed)
    if not energy_subsidy_gate.passed:
        raise SystemExit(
            "Energy-subsidy signal gate failed:\n  "
            + "\n  ".join(energy_subsidy_gate.failures)
        )
    alimony_gate = us_alimony_signal_gate(imputed)
    if not alimony_gate.passed:
        raise SystemExit(
            "Alimony-input signal gate failed:\n  " + "\n  ".join(alimony_gate.failures)
        )
    casualty_loss_gate = us_casualty_loss_signal_gate(imputed)
    if not casualty_loss_gate.passed:
        raise SystemExit(
            "Casualty-loss signal gate failed:\n  "
            + "\n  ".join(casualty_loss_gate.failures)
        )
    misc_itemized_gate = us_misc_itemized_signal_gate(imputed)
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
    retirement_contributions_gate = us_retirement_contributions_signal_gate(imputed)
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
    retirement_distributions_gate = us_retirement_distributions_signal_gate(imputed)
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
    education_inputs_gate = us_education_inputs_signal_gate(imputed)
    if not education_inputs_gate.passed:
        raise SystemExit(
            "Education-input signal gate failed:\n  "
            + "\n  ".join(education_inputs_gate.failures)
        )
    _observe_frame_boundary(boundary_observer, "education_inputs_post_clone", imputed)
    congressional_district_assignment = {"applied": False}
    if args.assign_congressional_districts:
        ledger_facts = load_ledger_consumer_artifact(args.ledger_facts).facts
        if args.congressional_district_vintage_crosswalk is not None:
            ledger_facts = translate_congressional_district_facts_to_current_vintage(
                ledger_facts,
                load_congressional_district_vintage_crosswalk(
                    args.congressional_district_vintage_crosswalk
                ),
            )
        distribution = congressional_district_distribution_from_ledger_facts(
            ledger_facts
        )
        imputed = with_household_congressional_districts(
            imputed,
            distribution,
            seed=args.congressional_district_seed,
        )
        congressional_district_assignment = congressional_district_assignment_summary(
            imputed.table("household"),
            distribution,
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
        gate = us_geography_ladder_gate(household, household_weights)
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
    if (
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
        "weights_audit": weights_audit,
        "puf_tax_detail_tail_bounds": tail_bound_diagnostics,
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
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    _observe_frame_boundary(boundary_observer, "final_export", imputed)
    return imputed


def _run_configured_stage(args: argparse.Namespace) -> None:
    """Dispatch one separately executed outer stage."""

    if args.stage in LEGACY_STAGE_ALIASES:
        raise SystemExit(
            f"Stage alias {args.stage!r} was Phase-1-only; use one of the "
            f"descriptive stages {list(PIPELINE_STEPS)}."
        )
    _run_outer_stage(args)


def _run_outer_stage(args: argparse.Namespace) -> None:
    """Run one exact pipeline step through the lossless outer-stage runtime."""

    runtime = StageRuntime(
        args.checkpoint_dir,
        OUTER_STAGE_PIPELINE,
        run_config=_stage_run_config(args),
    )
    if args.stage in runtime.context.completed:
        if args.stage == "final_export":
            _repair_completed_final_stage(args, runtime)
        return

    with profile_stage(args.stage, args.checkpoint_dir):
        if args.stage == "source_construction":
            predecessor = runtime.load_predecessor(args.stage)
            if predecessor is not None:
                raise AssertionError(
                    "source construction unexpectedly has a predecessor"
                )
            frame, metadata = _source_construction_stage(args)
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

        predecessor = runtime.load_predecessor(args.stage)
        if predecessor is None:
            raise AssertionError(f"stage {args.stage!r} requires a predecessor")
        before = predecessor.frame
        stage_metadata = runtime.metadata
        if args.stage == "pre_clone_enrichment":
            after, metadata = _pre_clone_enrichment_stage(
                args,
                before,
                stage_metadata["source_construction"],
            )
            assert_unchanged_identity(before, after, stage=args.stage)
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
            assert_unchanged_identity(before, after, stage=args.stage)
        elif args.stage == CAPITAL_GAIN_DISTRIBUTIONS_STAGE_NAME:
            after, metadata = _capital_gain_distributions_stage(args, before)
            assert_unchanged_identity(before, after, stage=args.stage)
        elif args.stage == "final_export":
            after = before
            metadata = _export_staged_result(args, after, stage_metadata)
            assert_unchanged_identity(before, after, stage=args.stage)
        else:
            after, metadata = _post_qrf_frame_stage(
                args.stage,
                args,
                before,
                stage_metadata,
            )
            assert_unchanged_identity(before, after, stage=args.stage)
        completed = runtime.complete(args.stage, after, metadata=metadata)
        if args.stage == "final_export":
            _link_all_stage_checkpoint(args.checkpoint_dir, completed.path)


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


def _pre_clone_enrichment_stage(
    args: argparse.Namespace,
    raw_base: Frame,
    source_metadata: dict[str, object],
) -> tuple[Frame, dict[str, object]]:
    weeks_path = Path(str(source_metadata["weeks_unemployed_source_path"]))
    weeks_source = load_asec_2023_weeks_unemployed_source(weeks_path)
    base = derive_us_cps_carried_inputs(raw_base)
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
            us_relationship_inputs_signal_gate(base),
            "Relationship-input signal gate failed",
        )
    }
    base = with_us_medicare_take_up_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals["medicare_take_up_input_signal"] = _checked_gate_payload(
        us_medicare_take_up_signal_gate(base),
        "Medicare take-up input signal gate failed before support cloning",
    )
    housing_gate = us_housing_inputs_signal_gate(base)
    acs_rent_donor: pd.DataFrame | None = None
    if not housing_gate.passed:
        if args.acs_h5 is None:
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
        us_eligibility_inputs_signal_gate(base),
        "Eligibility-input signal gate failed before support cloning",
    )
    base = with_us_pregnancy_inputs(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals["pregnancy_signal"] = _checked_gate_payload(
        us_pregnancy_signal_gate(base),
        "Pregnancy signal gate failed before support cloning",
    )
    base = with_us_wic_claim_input(
        base,
        seed=args.seed,
        time_period=args.target_year,
    )
    signals["wic_claim_signal"] = _checked_gate_payload(
        us_wic_claim_signal_gate(base),
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
    report = weights_audit_gate([FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, weight_kind)])
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
            us_wic_claim_signal_gate(frame),
            "WIC-claim signal gate failed after support cloning",
        )
        signals["medicare_take_up_input_signal"] = _checked_gate_payload(
            us_medicare_take_up_signal_gate(frame),
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
            us_prior_year_income_signal_gate(frame),
            "Prior-year-income signal gate failed",
        )
        signals["prior_year_income_source_reconciliation"] = _checked_gate_payload(
            us_prior_year_income_source_reconciliation_gate(frame),
            "Prior-year-income source reconciliation failed",
        )
        signals["housing_inputs_signal"] = _checked_gate_payload(
            us_housing_inputs_signal_gate(frame),
            "Housing-input signal gate failed after PUF-support imputation",
        )
        signals["qbi_inputs_signal"] = _checked_gate_payload(
            us_qbi_inputs_signal_gate(frame),
            "QBI-input signal gate failed",
        )
        signals["farm_business_income_signal"] = _checked_gate_payload(
            us_farm_business_income_signal_gate(frame),
            "Farm-business-income signal gate failed",
        )
        signals["domestic_production_ald_signal"] = _checked_gate_payload(
            us_domestic_production_ald_signal_gate(frame),
            "Domestic-production-ALD signal gate failed",
        )
    elif stage == "child_support_post_clone":
        frame = with_us_child_support_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["child_support_signal"] = _checked_gate_payload(
            us_child_support_signal_gate(frame), "Child-support signal gate failed"
        )
    elif stage == "disability_benefits_post_clone":
        frame = with_us_disability_benefits(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["disability_benefits_signal"] = _checked_gate_payload(
            us_disability_benefits_signal_gate(frame),
            "Disability-benefits signal gate failed",
        )
    elif stage == "workers_compensation_post_clone":
        frame = with_us_workers_compensation(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["workers_compensation_signal"] = _checked_gate_payload(
            us_workers_compensation_signal_gate(frame),
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
            us_weeks_unemployed_signal_gate(frame),
            "Weeks-unemployed signal gate failed",
        )
        signals["educator_expense_signal"] = _checked_gate_payload(
            us_educator_expense_signal_gate(frame),
            "Educator-expense signal gate failed",
        )
        signals["form_4952_election_signal"] = _checked_gate_payload(
            us_form_4952_election_signal_gate(frame),
            "Form 4952 election signal gate failed",
        )
        signals["salt_refund_income_signal"] = _checked_gate_payload(
            us_salt_refund_income_signal_gate(frame),
            "SALT-refund-income signal gate failed",
        )
        signals["capital_gain_details_signal"] = _checked_gate_payload(
            us_capital_gain_details_signal_gate(frame),
            "Capital-gain details signal gate failed",
        )
    elif stage == "childcare_post_clone":
        frame = with_us_childcare_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["childcare_inputs_signal"] = _checked_gate_payload(
            us_childcare_signal_gate(frame), "Childcare-input signal gate failed"
        )
    elif stage == "adult_care_post_clone":
        frame = with_us_adult_care_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["adult_care_inputs_signal"] = _checked_gate_payload(
            us_adult_care_signal_gate(frame), "Adult-care input signal gate failed"
        )
    elif stage == "energy_subsidy_post_clone":
        frame = with_us_energy_subsidy_input(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["energy_subsidy_signal"] = _checked_gate_payload(
            us_energy_subsidy_signal_gate(frame),
            "Energy-subsidy signal gate failed",
        )
        signals["alimony_inputs_signal"] = _checked_gate_payload(
            us_alimony_signal_gate(frame), "Alimony-input signal gate failed"
        )
        signals["casualty_loss_signal"] = _checked_gate_payload(
            us_casualty_loss_signal_gate(frame), "Casualty-loss signal gate failed"
        )
        signals["misc_itemized_signal"] = _checked_gate_payload(
            us_misc_itemized_signal_gate(frame),
            "Miscellaneous-itemized signal gate failed",
        )
    elif stage == "retirement_contributions_post_clone":
        frame = with_us_retirement_contribution_inputs(
            frame,
            seed=args.seed,
            time_period=args.target_year,
        )
        signals["retirement_contributions_signal"] = _checked_gate_payload(
            us_retirement_contributions_signal_gate(frame),
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
            us_retirement_distributions_signal_gate(frame),
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
            us_education_inputs_signal_gate(frame),
            "Education-input signal gate failed",
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
            gate = us_geography_ladder_gate(household, household_weights)
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
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output_h5 = out_dir / _dataset_filename(args.target_year)
    summary_path = out_dir / _summary_filename(args.target_year)
    _write_policyengine_dataset(args, frame, output_h5)
    congressional = stage_metadata["congressional_district_assignment"][
        "congressional_district_assignment"
    ]
    geography = stage_metadata["block_ladder_assignment"]["geography_ladder_assignment"]
    if (
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
        "weights_audit": qrf["weights_audit"],
        "puf_tax_detail_tail_bounds": qrf["puf_tax_detail_tail_bounds"],
        **{name: signals[name] for name in required_signals},
        "congressional_district_assignment": stage_metadata[
            "congressional_district_assignment"
        ]["congressional_district_assignment"],
        "geography_ladder_assignment": geography,
        "channel_output_totals": _channel_output_totals(frame),
        "immigration_composition": us_immigration_composition_summary(frame),
    }
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
    QRF *resolved* to via the build-level weights audit (populace #300): the fit
    emits one :class:`~populace.build.FitWeightRecord`, and
    :func:`~populace.build.weights_audit_gate` proves it did not silently resolve
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
    report = weights_audit_gate(fit_records)
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
    return frame, {
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
            "package:populace.build.us/support_spine.json"
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
        year = int(raw_year)
        if year in paths:
            raise ValueError(f"Duplicate --asec-h5 mapping for year {year}.")
        paths[year] = Path(raw_path)
    return paths


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
    return puf_tax_unit_donor_from_arrays(
        arrays,
        adjusted_gross_income=adjusted_gross_income,
        donor_build_summary=donor_build_summary,
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


def _row_counts(frame: Frame) -> dict[str, int]:
    return {entity: frame.n(entity) for entity in frame.entities}


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
