#!/usr/bin/env python3
"""Build the SHA-pinned, pre-calibration US input pool.

The default production path is the stacked pipeline:

``stack -> gap-fill -> PUF pass + tail -> derive -> seed -> simulate -> gates``.

Both survey arms use one composition-preserving ``--sample-fraction``; PUF
donors always remain full. The terminal completeness gate plus by-origin
battery replace two-spine agreement. ``--legacy-two-spine`` retains the
retiring pipeline byte-for-byte for reproducibility.

Every input is local and explicitly SHA-pinned; this tool never downloads
data. It writes a nullable input-only H5 plus a manifest and terminal gate
diagnostics. A failed gate still leaves diagnostic artifacts but returns
nonzero and never marks the pool simulation-ready. Calibration is downstream.

The standard 1% local smoke rung is expected to take roughly 2--4 hours and
stay within a 64 GiB memory envelope on a modern workstation. Most battery
comparisons receipt ``insufficient_support`` by design at this rung; that is a
validity-domain receipt, not a relaxed tolerance. Example (the committed
``tools/us_stacked_pool_smoke.sh.example`` carries the same invocation):

.. code-block:: console

   PYTHONPATH=packages/populace-frame/src:packages/populace-fit/src:packages/populace-calibrate/src:packages/populace-build/src:packages/populace-data/src \\
     /path/to/populace/.venv/bin/python tools/build_us_multispine_pool.py \\
     --sample-fraction 0.01 --sample-seed 578 \\
     --clone-attachment-fraction 1.0 --clone-attachment-seed 578 \\
     --asec-raw-stage-h5 "$ASEC_H5" --asec-raw-stage-h5-sha256 "$ASEC_SHA" \\
     --acs-household-zip "$ACS_H_ZIP" --acs-household-zip-sha256 "$ACS_H_SHA" \\
     --acs-person-zip "$ACS_P_ZIP" --acs-person-zip-sha256 "$ACS_P_SHA" \\
     --acs-rent-h5 "$RENT_H5" --acs-rent-h5-sha256 "$RENT_SHA" \\
     --puf-h5 "$PUF_H5" --puf-h5-sha256 "$PUF_SHA" \\
     --puf-source-year-csv "$PUF_CSV" \\
     --puf-source-year-csv-sha256 "$PUF_CSV_SHA" --out "$OUT"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.chronicle import record_build_attempt
from populace.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from populace.build.gates import (
    FitWeightRecord,
    GateReport,
    GateResult,
    weights_audit_gate,
)
from populace.build.serialization_dtypes import canonicalize_frame_string_dtypes
from populace.build.us_runtime.acs_inputs import map_acs_native_inputs
from populace.build.us_runtime.acs_pums import (
    AcsPumsSource,
    build_acs_pums_unit_frame,
)
from populace.build.us_runtime.acs_sources import (
    AcsSourceArtifact,
    AcsSourceManifest,
    load_acs_source_manifest,
)
from populace.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    transfer_acs_inputs,
)
from populace.build.us_runtime.acs_transfer_bank import AcsTransferTargetBankStore
from populace.build.us_runtime.asec_checkpoint import (
    load_asec_raw_stage_checkpoint,
)
from populace.build.us_runtime.h5_io import (
    US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
    load_simulation_ready_us_multispine_pool_manifest,
    write_nullable_us_h5,
)
from populace.build.us_runtime.housing_inputs import (
    ACS_2022_RENT_ARTIFACT_SHA256,
    load_acs_2022_rent_donor,
)
from populace.build.us_runtime.multispine_pool import (
    POOL_CHECKPOINT_STAGE_ORDER,
    POOL_DERIVE_OPERATOR_ORDER,
    POOL_HOUSEHOLD_MASS_SHARES,
    POOL_OPERATOR_ORDER,
    POOL_POST_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER,
    POOL_RANDOM_SEED,
    POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE,
    POOL_TIME_PERIOD,
    MultispinePoolCheckpoint,
    MultispinePoolResult,
    PoolStageOutput,
    complete_multispine_source_inputs,
    derive_multispine_pool_inputs,
    materialize_multispine_agreement_outputs,
    pool_transfer_target_families,
    prepare_multispine_source_inputs_for_clone,
    run_multispine_pool_path,
    seed_multispine_pool_inputs,
)
from populace.build.us_runtime.operator_boundary import (
    assert_operator_free_source_frame,
)
from populace.build.us_runtime.puf_capital_gains_tail import (
    transfer_puf_capital_gains_tail,
    validate_puf_capital_gains_tail_manifest,
)
from populace.build.us_runtime.puf_donor_io import load_puf_tax_unit_donor
from populace.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_MANIFEST_FILENAME,
    PRIMARY_QRF_TARGET_ORDER,
    finalize_primary_puf_qrf_chain,
    initialize_primary_puf_qrf_chain,
    run_primary_puf_qrf_chain,
)
from populace.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
    US_PUF_SUPPORT_FIT_NAME,
)
from populace.build.us_runtime.stacked_spine import (
    assemble_stacked_spine,
    assert_stacked_tail_cells_preserved,
    by_origin_battery,
    gap_fill_stacked_spine,
    prepare_stacked_tail_derivation,
    run_stacked_puf_pass,
    stacked_completeness_gate,
    stacked_gap_fill_plan,
    stacked_spine_authority_receipt,
    validate_stacked_spine_frame,
)
from populace.build.us_runtime.support_provenance import (
    SPINE_ASSEMBLY_MANIFEST_KEY,
    spine_provenance_counts,
    validate_assembly_provenance,
)
from populace.build.us_runtime.take_up_contract import take_up_contract_identity
from populace.frame import US_SCHEMA, Frame

__all__ = [
    "POOL_H5_ARTIFACT_KIND",
    "POOL_MANIFEST_SCHEMA_VERSION",
    "POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION",
    "POOL_STAGE_CHECKPOINT_SCHEMA_VERSION",
    "PoolBuildOutputs",
    "StackedPoolBuildResult",
    "build_multispine_pool",
    "build_stacked_pool",
    "load_simulation_ready_us_multispine_pool_manifest",
    "main",
]

POOL_MANIFEST_SCHEMA_VERSION = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
"""Schema version for the companion pool build manifest."""

POOL_H5_ARTIFACT_KIND = US_MULTISPINE_POOL_H5_ARTIFACT_KIND
"""Neutral H5 artifact kind; readiness is asserted only by the manifest."""

POOL_STAGE_CHECKPOINT_SCHEMA_VERSION = 1
"""Serialization contract for pool stage checkpoint metadata and sidecars."""

# Pool stage checkpoint semantic-invalidation ledger.
#
# 1: Initial resumable pool path. It binds the assembled-source producer,
#    pre/post-clone operator registries, physical PUF clone and QRF target
#    order, tail and ACS transfer producers, derive registry, take-up seeding,
#    SSI materialization, fixed seeds/period/fit sizes, and PolicyEngine-US.
# 2: The take-up contract identity binds the canonical SHA-256 of the entire
#    parsed resource, plus readable explicit fields. Version-1 volume
#    checkpoints are deliberately stale.
#
# Bump this version whenever any producer above changes a stage output without
# changing one of the explicit identity fields below. In particular, adding,
# removing, reordering, or changing an operator kernel requires a bump even if
# its public registry name stays constant. Correctness takes priority over
# retaining warm checkpoints (the same rule as TARGET_FRAME_CHECKPOINT's
# materializer-version ledger in build_us_fiscal_refresh_release.py).
#
# Pure-string object columns and canonical pandas string columns are two
# supported physical encodings of the same v2 logical values. The reader
# authenticates the literal stored bytes and sidecar schema first, then
# normalizes that logical view in memory. Moving between those encodings does
# not change a producer's scalar output and therefore does not advance this
# ledger; changing string values or the canonical logical dtype policy does.
POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION = 2

_PRIMARY_QRF_N_ESTIMATORS = 100
_ACS_TRANSFER_N_ESTIMATORS = 100
_PRIMARY_QRF_INPUT_BINDING_FILENAME = "pool-input-binding.json"
_PRIMARY_QRF_INPUT_BINDING_SCHEMA_VERSION = 1
_POOL_STAGE_CHECKPOINT_ARTIFACT_KIND = "populace_us_multispine_pool_stage_checkpoint"
_POOL_STAGE_CHECKPOINT_MANIFEST_ARTIFACT_KIND = (
    "populace_us_multispine_pool_stage_checkpoint_manifest"
)
_POOL_STAGE_CHECKPOINT_RECEIPTS_ARTIFACT_KIND = (
    "populace_us_multispine_pool_stage_checkpoint_operational_receipts"
)
_POOL_STAGE_CHECKPOINT_FILENAMES: Mapping[str, str] = {
    stage: f"{stage}.checkpoint.h5" for stage in POOL_CHECKPOINT_STAGE_ORDER
}
_POOL_SIMULATION_OUTPUT_COLUMN = "ssi"
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_BANK_IDENTITY_SIBLING_SCAN_LIMIT = 64
_STACKED_SAMPLE_RUNG_TOKENS: Mapping[float, tuple[str, str]] = {
    0.01: ("f001", "1/100"),
    0.10: ("f010", "1/10"),
    1.00: ("f100", "1"),
}
_STACKED_PIPELINE = "us-stacked-pool"
_STACKED_CHECKPOINT_MATERIALIZER_VERSION = 1
_STACKED_RELEASE_ID_PATTERN = re.compile(
    r"^populace-us-2024-stacked-f(?:001|010|100)-s[0-9]+-"
    r"asec[0-9]+-acs[0-9]+-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
)

type PoolOperator = Callable[[Frame], PoolStageOutput]


@dataclass(frozen=True)
class PoolBuildOutputs:
    """Deterministic output paths derived from the requested H5 path."""

    pool_h5: Path
    manifest: Path
    agreement_diagnostics: Path
    checkpoint_root: Path
    primary_qrf_checkpoint_dir: Path
    acs_transfer_checkpoint_dir: Path


@dataclass(frozen=True)
class StackedPoolBuildResult:
    """Input-only stacked pool and its two fresh terminal gate verdicts."""

    frame: Frame
    stack_receipt: Mapping[str, object]
    assembly_receipt: Mapping[str, object]
    provenance_counts: Mapping[str, Mapping[str, object]]
    stage_receipts: Mapping[str, Mapping[str, object]]
    terminal_gates: tuple[GateResult, GateResult]
    release_id: str

    @property
    def simulation_ready(self) -> bool:
        return all(gate.passed for gate in self.terminal_gates)


@dataclass
class _StackedAttemptState:
    """Mutable terminal-attempt evidence collected for Chronicle emission."""

    build_id: str
    identity_digest: str
    input_pins_digest: str
    phases_reached: list[str]
    gate_verdicts: dict[str, dict[str, object]]
    artifact_location: str | None = None


@dataclass(frozen=True)
class _VerifiedInput:
    role: str
    path: Path
    expected_sha256: str
    actual_sha256: str
    size_bytes: int

    def to_manifest(self) -> dict[str, object]:
        return {
            "path": str(self.path.resolve()),
            "expected_sha256": self.expected_sha256,
            "actual_sha256": self.actual_sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class _LoadedInputs:
    asec: Frame
    acs: Frame
    acs_rent_donor: pd.DataFrame
    puf_donor: pd.DataFrame
    asec_raw_stage_checkpoint: Mapping[str, object]
    acs_build: Mapping[str, object]
    acs_native_inputs: Mapping[str, Mapping[str, Any]]
    puf_donor_build: Mapping[str, object]


def _sha256_argument(value: str) -> str:
    if not _LOWERCASE_SHA256.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "SHA-256 pins must be exactly 64 lowercase hexadecimal characters."
        )
    return value


def _standard_sample_fraction(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sample fraction must be one of 0.01, 0.10, or 1.0."
        ) from exc
    if fraction not in _STACKED_SAMPLE_RUNG_TOKENS:
        raise argparse.ArgumentTypeError(
            "sample fraction must be one of 0.01, 0.10, or 1.0."
        )
    return fraction


def _unit_fraction(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "clone attachment fraction must be greater than zero and at most one."
        ) from exc
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise argparse.ArgumentTypeError(
            "clone attachment fraction must be greater than zero and at most one."
        )
    return fraction


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asec-raw-stage-h5",
        required=True,
        type=Path,
        help="Operator-untouched ASEC raw_source_mapping Frame checkpoint.",
    )
    parser.add_argument(
        "--asec-raw-stage-h5-sha256",
        required=True,
        type=_sha256_argument,
        help="Expected SHA-256 of --asec-raw-stage-h5.",
    )
    parser.add_argument(
        "--acs-household-zip",
        required=True,
        type=Path,
        help="Local packaged-pin ACS 2024 1-year household PUMS archive.",
    )
    parser.add_argument(
        "--acs-household-zip-sha256",
        required=True,
        type=_sha256_argument,
        help="Expected SHA-256 of --acs-household-zip.",
    )
    parser.add_argument(
        "--acs-person-zip",
        required=True,
        type=Path,
        help="Local packaged-pin ACS 2024 1-year person PUMS archive.",
    )
    parser.add_argument(
        "--acs-person-zip-sha256",
        required=True,
        type=_sha256_argument,
        help="Expected SHA-256 of --acs-person-zip.",
    )
    parser.add_argument(
        "--acs-rent-h5",
        required=True,
        type=Path,
        help="Local canonical ACS 2022 rent-donor H5.",
    )
    parser.add_argument(
        "--acs-rent-h5-sha256",
        required=True,
        type=_sha256_argument,
        help="Expected canonical SHA-256 of --acs-rent-h5.",
    )
    parser.add_argument(
        "--puf-h5",
        required=True,
        type=Path,
        help="Processed PUF tax-unit donor H5.",
    )
    parser.add_argument(
        "--puf-h5-sha256",
        required=True,
        type=_sha256_argument,
        help="Expected SHA-256 of --puf-h5.",
    )
    parser.add_argument(
        "--puf-source-year-csv",
        required=True,
        type=Path,
        help="Restricted source-year PUF CSV used for E00100 alignment.",
    )
    parser.add_argument(
        "--puf-source-year-csv-sha256",
        required=True,
        type=_sha256_argument,
        help="Expected SHA-256 of --puf-source-year-csv.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Destination nullable input-pool H5; sidecars derive from this path.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        help=(
            "Durable root for stage and primary-QRF checkpoints. Defaults to "
            "<out-stem>.checkpoints alongside --out; point it at a persistent "
            "volume to resume across fresh workers and output directories."
        ),
    )
    parser.add_argument(
        "--sample-fraction",
        type=_standard_sample_fraction,
        default=1.0,
        help="Uniform survey-arm rung: 0.01 smoke, 0.10 dev, or 1.0 full.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=578,
        help="Non-negative whole-household survey sampling seed (default: 578).",
    )
    parser.add_argument(
        "--clone-attachment-fraction",
        type=_unit_fraction,
        default=1.0,
        help="Separate PUF clone attachment fraction (default: 1.0).",
    )
    parser.add_argument(
        "--clone-attachment-seed",
        type=int,
        default=578,
        help="Non-negative PUF clone attachment seed (default: 578).",
    )
    parser.add_argument(
        "--chronicle-prev-row-digest",
        type=_sha256_argument,
        help=(
            "Optional current Chronicle chain head. If omitted, "
            "POPULACE_CHRONICLE_PREV_ROW_DIGEST is used, then genesis null."
        ),
    )
    parser.add_argument(
        "--legacy-two-spine",
        action="store_true",
        help="Run the byte-compatible retiring two-spine pipeline.",
    )
    return parser


def _output_paths(
    path: Path,
    *,
    checkpoint_root: Path | None = None,
) -> PoolBuildOutputs:
    pool_h5 = Path(path)
    if pool_h5.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("--out must name an .h5 or .hdf5 file.")
    resolved_checkpoint_root = (
        Path(checkpoint_root)
        if checkpoint_root is not None
        else pool_h5.with_suffix(".checkpoints")
    )
    return PoolBuildOutputs(
        pool_h5=pool_h5,
        manifest=pool_h5.with_suffix(".manifest.json"),
        agreement_diagnostics=pool_h5.with_suffix(".agreement.json"),
        checkpoint_root=resolved_checkpoint_root,
        primary_qrf_checkpoint_dir=resolved_checkpoint_root / "primary-qrf",
        acs_transfer_checkpoint_dir=resolved_checkpoint_root / "acs-transfer",
    )


def _stacked_output_paths(
    path: Path,
    *,
    checkpoint_root: Path | None = None,
) -> PoolBuildOutputs:
    """Use a terminal-gates sidecar without changing legacy output names."""

    outputs = _output_paths(path, checkpoint_root=checkpoint_root)
    return replace(
        outputs,
        agreement_diagnostics=outputs.pool_h5.with_suffix(".gates.json"),
    )


def _with_checkpoint_identity(
    outputs: PoolBuildOutputs,
    *,
    base_identity_sha256: str,
) -> PoolBuildOutputs:
    """Select non-mixing intra-phase bank directories for one pool identity."""

    return replace(
        outputs,
        primary_qrf_checkpoint_dir=(
            outputs.checkpoint_root / "primary-qrf" / base_identity_sha256
        ),
        acs_transfer_checkpoint_dir=(
            outputs.checkpoint_root / "acs-transfer" / base_identity_sha256
        ),
    )


def _identity_routed_bank_open_receipt(
    selected_dir: Path,
    *,
    current_base_identity_sha256: str,
) -> dict[str, object]:
    """Name bypassed prior-identity siblings without opening their contents."""

    selected = Path(selected_dir)
    if not _LOWERCASE_SHA256.fullmatch(current_base_identity_sha256):
        raise ValueError("Current bank base identity must be a lowercase SHA-256.")
    if selected.name != current_base_identity_sha256:
        raise ValueError(
            "Identity-routed bank path does not end in its current base identity: "
            f"{selected}."
        )
    bank_root = selected.parent
    entries_examined = 0
    truncated = False
    mismatches: list[dict[str, object]] = []
    if bank_root.exists():
        if not bank_root.is_dir():
            raise ValueError(
                f"Identity-routed bank root is not a directory: {bank_root}."
            )
        with os.scandir(bank_root) as entries:
            for entry in entries:
                if entries_examined == _BANK_IDENTITY_SIBLING_SCAN_LIMIT:
                    truncated = True
                    break
                entries_examined += 1
                stale_digest = entry.name
                if (
                    stale_digest == current_base_identity_sha256
                    or not _LOWERCASE_SHA256.fullmatch(stale_digest)
                    or not entry.is_dir(follow_symlinks=False)
                ):
                    continue
                mismatches.append(
                    {
                        "load_status": "identity_mismatch",
                        "stale_base_identity_sha256": stale_digest,
                        "current_base_identity_sha256": (current_base_identity_sha256),
                        "disposition": "bypassed",
                        "path": str((bank_root / stale_digest).resolve()),
                    }
                )
    mismatches.sort(key=lambda record: str(record["stale_base_identity_sha256"]))
    return {
        "bank_root": str(bank_root.resolve()),
        "selected_path": str(selected.resolve()),
        "current_base_identity_sha256": current_base_identity_sha256,
        "scan": {
            "limit": _BANK_IDENTITY_SIBLING_SCAN_LIMIT,
            "entries_examined": entries_examined,
            "truncated": truncated,
        },
        "identity_mismatches": mismatches,
    }


def _verify_file(role: str, path: Path, expected_sha256: str) -> _VerifiedInput:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{role} input is not a file: {source}")
    actual_sha256 = _file_sha256(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{role} SHA-256 mismatch for {source}: got {actual_sha256}, "
            f"expected {expected_sha256}."
        )
    return _VerifiedInput(
        role=role,
        path=source,
        expected_sha256=expected_sha256,
        actual_sha256=actual_sha256,
        size_bytes=source.stat().st_size,
    )


def _verify_acs_file(
    role: str,
    path: Path,
    expected_sha256: str,
    packaged: AcsSourceArtifact,
) -> _VerifiedInput:
    if expected_sha256 != packaged.sha256:
        raise ValueError(
            f"{role} CLI pin differs from the packaged ACS source pin: "
            f"got {expected_sha256}, expected {packaged.sha256}."
        )
    verified = _verify_file(role, path, expected_sha256)
    if verified.size_bytes != packaged.size_bytes:
        raise ValueError(
            f"{role} byte-size mismatch for {path}: got {verified.size_bytes}, "
            f"expected packaged size {packaged.size_bytes}."
        )
    return verified


def _verify_inputs(
    args: argparse.Namespace,
    outputs: PoolBuildOutputs,
) -> tuple[dict[str, _VerifiedInput], AcsSourceManifest]:
    source_paths = _configured_source_paths(args)
    _validate_checkpoint_path_layout(outputs, source_paths=source_paths)

    output_paths = {
        outputs.pool_h5.resolve(),
        outputs.manifest.resolve(),
        outputs.agreement_diagnostics.resolve(),
    }
    collisions = sorted(str(path) for path in source_paths if path in output_paths)
    if collisions:
        raise ValueError(f"Pool outputs must not overwrite inputs: {collisions}.")

    acs_source_manifest = load_acs_source_manifest()
    if args.acs_rent_h5_sha256 != ACS_2022_RENT_ARTIFACT_SHA256:
        raise ValueError(
            "ACS rent donor CLI pin differs from the canonical archived pin: "
            f"got {args.acs_rent_h5_sha256}, expected "
            f"{ACS_2022_RENT_ARTIFACT_SHA256}."
        )
    verified = {
        "asec_raw_stage": _verify_file(
            "ASEC raw-stage checkpoint",
            args.asec_raw_stage_h5,
            args.asec_raw_stage_h5_sha256,
        ),
        "acs_household": _verify_acs_file(
            "ACS household archive",
            args.acs_household_zip,
            args.acs_household_zip_sha256,
            acs_source_manifest.artifact("household"),
        ),
        "acs_person": _verify_acs_file(
            "ACS person archive",
            args.acs_person_zip,
            args.acs_person_zip_sha256,
            acs_source_manifest.artifact("person"),
        ),
        "acs_rent_donor": _verify_file(
            "ACS rent donor",
            args.acs_rent_h5,
            args.acs_rent_h5_sha256,
        ),
        "processed_puf": _verify_file(
            "processed PUF H5",
            args.puf_h5,
            args.puf_h5_sha256,
        ),
        "puf_source_year": _verify_file(
            "source-year PUF CSV",
            args.puf_source_year_csv,
            args.puf_source_year_csv_sha256,
        ),
    }
    return verified, acs_source_manifest


def _configured_source_paths(args: argparse.Namespace) -> set[Path]:
    """Resolve the six immutable input locations without opening them."""

    return {
        Path(args.asec_raw_stage_h5).resolve(),
        Path(args.acs_household_zip).resolve(),
        Path(args.acs_person_zip).resolve(),
        Path(args.acs_rent_h5).resolve(),
        Path(args.puf_h5).resolve(),
        Path(args.puf_source_year_csv).resolve(),
    }


def _validate_checkpoint_path_layout(
    outputs: PoolBuildOutputs,
    *,
    source_paths: set[Path],
) -> None:
    """Reject only paths the checkpoint store can actually overwrite."""

    checkpoint_root = outputs.checkpoint_root.resolve()
    primary_qrf_root = (checkpoint_root / "primary-qrf").resolve()
    acs_transfer_root = (checkpoint_root / "acs-transfer").resolve()
    stage_files = {
        (checkpoint_root / filename).resolve()
        for filename in _POOL_STAGE_CHECKPOINT_FILENAMES.values()
    }
    stage_files.update(
        path.with_suffix(".manifest.json") for path in tuple(stage_files)
    )
    publication_paths = {
        outputs.pool_h5.resolve(),
        outputs.manifest.resolve(),
        outputs.agreement_diagnostics.resolve(),
    }

    def conflicts_with_checkpoint_store(path: Path) -> bool:
        if path == checkpoint_root or checkpoint_root.is_relative_to(path):
            return True
        if path == primary_qrf_root or path.is_relative_to(primary_qrf_root):
            return True
        if path == acs_transfer_root or path.is_relative_to(acs_transfer_root):
            return True
        return any(
            path == target or path.is_relative_to(target) for target in stage_files
        )

    publication_collisions = sorted(
        str(path) for path in publication_paths if conflicts_with_checkpoint_store(path)
    )
    if publication_collisions:
        raise ValueError(
            "Pool checkpoint paths collide with publication files: "
            f"{publication_collisions}."
        )
    source_collisions = sorted(
        str(path) for path in source_paths if conflicts_with_checkpoint_store(path)
    )
    if source_collisions:
        raise ValueError(
            f"Pool checkpoint paths must not overwrite inputs: {source_collisions}."
        )


def _load_inputs(
    args: argparse.Namespace,
    *,
    acs_source_manifest: AcsSourceManifest,
) -> _LoadedInputs:
    asec, asec_raw_stage_checkpoint = load_asec_raw_stage_checkpoint(
        args.asec_raw_stage_h5
    )
    acs_source = AcsPumsSource(
        household_zip=args.acs_household_zip,
        person_zip=args.acs_person_zip,
        vintage=acs_source_manifest.vintage,
    )
    acs_frame, acs_build = build_acs_pums_unit_frame(acs_source)
    mapped_acs = map_acs_native_inputs(acs_frame)
    acs_rent_donor = load_acs_2022_rent_donor(args.acs_rent_h5)
    puf_donor, donor_build = _load_puf_donor(args)
    return _LoadedInputs(
        asec=asec,
        acs=mapped_acs.frame,
        acs_rent_donor=acs_rent_donor,
        puf_donor=puf_donor,
        asec_raw_stage_checkpoint=asec_raw_stage_checkpoint,
        acs_build=acs_build,
        acs_native_inputs=mapped_acs.native_inputs,
        puf_donor_build=donor_build,
    )


def _load_puf_donor(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, object]]:
    donor_build: dict[str, object] = {}
    donor = load_puf_tax_unit_donor(
        args.puf_h5,
        args.puf_source_year_csv,
        donor_build_summary=donor_build,
    )
    return donor, donor_build


def _loaded_input_receipts(loaded: _LoadedInputs) -> dict[str, object]:
    return {
        "asec_raw_stage_checkpoint": loaded.asec_raw_stage_checkpoint,
        "acs_pums_build": loaded.acs_build,
        "acs_native_inputs": loaded.acs_native_inputs,
        "puf_donor": {
            "rows": int(len(loaded.puf_donor)),
            "columns": sorted(str(column) for column in loaded.puf_donor.columns),
            "build_receipt": loaded.puf_donor_build,
        },
    }


def _validate_resumed_puf_donor(
    donor: pd.DataFrame,
    input_receipts: Mapping[str, object],
) -> None:
    receipt = input_receipts.get("puf_donor")
    if not isinstance(receipt, Mapping):
        raise ValueError("Resumed pool checkpoint has no PUF donor receipt.")
    observed = {
        "rows": int(len(donor)),
        "columns": sorted(str(column) for column in donor.columns),
    }
    expected = {
        "rows": receipt.get("rows"),
        "columns": receipt.get("columns"),
    }
    if observed != expected:
        raise ValueError(
            "Resumed pool PUF donor schema changed despite matching input pins: "
            f"got {observed}, expected {expected}."
        )


def _primary_qrf_manifest_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / PRIMARY_QRF_MANIFEST_FILENAME


def _primary_qrf_input_binding_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / _PRIMARY_QRF_INPUT_BINDING_FILENAME


def _checkpoint_input_binding(
    verified_inputs: Mapping[str, _VerifiedInput],
) -> dict[str, object]:
    return {
        "artifact_kind": "populace_us_multispine_primary_qrf_input_binding",
        "schema_version": _PRIMARY_QRF_INPUT_BINDING_SCHEMA_VERSION,
        "period": POOL_TIME_PERIOD,
        "seed": POOL_RANDOM_SEED,
        "n_estimators": _PRIMARY_QRF_N_ESTIMATORS,
        "inputs": {
            role: {
                "sha256": pin.actual_sha256,
                "size_bytes": pin.size_bytes,
            }
            for role, pin in verified_inputs.items()
        },
    }


def _policyengine_us_version() -> str:
    try:
        return version("policyengine-us")
    except PackageNotFoundError:
        return "not-installed"


def _pool_checkpoint_base_identity(
    verified_inputs: Mapping[str, _VerifiedInput],
    *,
    policyengine_us_version: str | None = None,
) -> dict[str, object]:
    """Return every input and semantic surface that determines cached stages."""

    return {
        "artifact_kind": "populace_us_multispine_pool_checkpoint_identity",
        "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
        "materializer_version": POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
        "period": POOL_TIME_PERIOD,
        "seed": POOL_RANDOM_SEED,
        "policyengine_us_version": (
            policyengine_us_version
            if policyengine_us_version is not None
            else _policyengine_us_version()
        ),
        "inputs": {
            role: {
                "sha256": pin.actual_sha256,
                "size_bytes": pin.size_bytes,
            }
            for role, pin in sorted(verified_inputs.items())
        },
        "pool_code": {
            "operator_order": list(POOL_OPERATOR_ORDER),
            "household_mass_shares": dict(POOL_HOUSEHOLD_MASS_SHARES),
            "pre_clone_source_operator_order": list(
                POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
            ),
            "post_clone_source_operator_order": list(
                POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
            ),
            "derive_operator_order": list(POOL_DERIVE_OPERATOR_ORDER),
            "primary_qrf_target_order": list(PRIMARY_QRF_TARGET_ORDER),
            "transfer_target_families": _json_ready(pool_transfer_target_families()),
            "take_up_contract": take_up_contract_identity(),
            "primary_qrf_n_estimators": _PRIMARY_QRF_N_ESTIMATORS,
            "acs_transfer_n_estimators": _ACS_TRANSFER_N_ESTIMATORS,
            "acs_transfer_max_targets_per_fit": (
                DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ),
            "simulation_household_batch_size": (POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE),
        },
    }


def _stacked_rung(sample_fraction: float) -> tuple[str, str]:
    try:
        return _STACKED_SAMPLE_RUNG_TOKENS[float(sample_fraction)]
    except KeyError as exc:
        raise ValueError(
            "Stacked sample_fraction must be one standard rung: 0.01, 0.10, or 1.0."
        ) from exc


def _verified_input_pins_payload(
    verified_inputs: Mapping[str, _VerifiedInput],
) -> dict[str, dict[str, object]]:
    return {
        role: {
            "sha256": pin.actual_sha256,
            "size_bytes": pin.size_bytes,
        }
        for role, pin in sorted(verified_inputs.items())
    }


def _input_pins_digest(
    verified_inputs: Mapping[str, _VerifiedInput],
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_verified_input_pins_payload(verified_inputs))
    ).hexdigest()


def _configured_input_pins_digest(args: argparse.Namespace) -> str:
    payload = {
        "acs_household": args.acs_household_zip_sha256,
        "acs_person": args.acs_person_zip_sha256,
        "acs_rent_donor": args.acs_rent_h5_sha256,
        "asec_raw_stage": args.asec_raw_stage_h5_sha256,
        "processed_puf": args.puf_h5_sha256,
        "puf_source_year": args.puf_source_year_csv_sha256,
    }
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _stacked_checkpoint_base_identity(
    verified_inputs: Mapping[str, _VerifiedInput],
    *,
    stack_receipt: Mapping[str, object],
    sample_fraction: float,
    sample_seed: int,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
    policyengine_us_version: str | None = None,
) -> dict[str, object]:
    """Bind #599/#608 caches to the live stack and both scale controls."""

    fraction_token, _rung = _stacked_rung(sample_fraction)
    if isinstance(sample_seed, bool) or sample_seed < 0:
        raise ValueError("sample_seed must be a non-negative integer.")
    if isinstance(clone_attachment_seed, bool) or clone_attachment_seed < 0:
        raise ValueError("clone_attachment_seed must be a non-negative integer.")
    stack = _json_ready(stack_receipt)
    if not isinstance(stack, dict):  # pragma: no cover - fixed mapping
        raise TypeError("Stack receipt must normalize to an object.")
    if stack.get("sample_fraction") != float(sample_fraction):
        raise ValueError("Live stack receipt does not match sample_fraction.")
    if stack.get("sample_seed") != sample_seed:
        raise ValueError("Live stack receipt does not match sample_seed.")
    return {
        "artifact_kind": "populace_us_stacked_pool_checkpoint_identity",
        "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
        "materializer_version": _STACKED_CHECKPOINT_MATERIALIZER_VERSION,
        "pipeline": _STACKED_PIPELINE,
        "period": POOL_TIME_PERIOD,
        "model_seed": POOL_RANDOM_SEED,
        "policyengine_us_version": (
            policyengine_us_version
            if policyengine_us_version is not None
            else _policyengine_us_version()
        ),
        "inputs": _verified_input_pins_payload(verified_inputs),
        "sampling": {
            "sample_fraction": float(sample_fraction),
            "fraction_token": fraction_token,
            "sample_seed": sample_seed,
            "stack_manifest_sha256": hashlib.sha256(
                _canonical_json_bytes(stack)
            ).hexdigest(),
            "stack_manifest": stack,
        },
        "clone_attachment": {
            "fraction": float(clone_attachment_fraction),
            "seed": clone_attachment_seed,
        },
        "stacked_authority": stacked_spine_authority_receipt(),
        "pool_code": {
            "operator_order": [
                "assemble_stacked_spine",
                "prepare_multispine_source_inputs_for_clone",
                "gap_fill_stacked_spine",
                "run_stacked_puf_pass",
                "complete_multispine_source_inputs",
                "prepare_stacked_tail_derivation",
                "derive_multispine_pool_inputs",
                "seed_multispine_pool_inputs",
                "materialize_multispine_agreement_outputs",
                "stacked_completeness_gate",
                "by_origin_battery",
            ],
            "pre_clone_source_operator_order": list(
                POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
            ),
            "post_clone_source_operator_order": list(
                POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
            ),
            "derive_operator_order": list(POOL_DERIVE_OPERATOR_ORDER),
            "primary_qrf_target_order": list(PRIMARY_QRF_TARGET_ORDER),
            "take_up_contract": take_up_contract_identity(),
            "primary_qrf_n_estimators": _PRIMARY_QRF_N_ESTIMATORS,
            "acs_transfer_n_estimators": _ACS_TRANSFER_N_ESTIMATORS,
            "acs_transfer_max_targets_per_fit": (
                DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ),
            "simulation_household_batch_size": (POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE),
        },
    }


def _configured_stacked_identity(args: argparse.Namespace) -> dict[str, object]:
    """Identity available before input verification for terminal error rows."""

    fraction_token, _rung = _stacked_rung(args.sample_fraction)
    return {
        "artifact_kind": "populace_us_stacked_pool_configured_identity",
        "schema_version": 1,
        "pipeline": _STACKED_PIPELINE,
        "period": POOL_TIME_PERIOD,
        "expected_input_pins_digest": _configured_input_pins_digest(args),
        "sample_fraction": float(args.sample_fraction),
        "fraction_token": fraction_token,
        "sample_seed": args.sample_seed,
        "clone_attachment_fraction": float(args.clone_attachment_fraction),
        "clone_attachment_seed": args.clone_attachment_seed,
        "stacked_authority": stacked_spine_authority_receipt(),
    }


def _stacked_checkpoint_root(
    outputs: PoolBuildOutputs,
    configured_identity: Mapping[str, object],
) -> Path:
    """Route one configured build to its non-mixing discovery namespace."""

    configured_digest = hashlib.sha256(
        _canonical_json_bytes(configured_identity)
    ).hexdigest()
    return outputs.checkpoint_root / "stacked" / configured_digest


def _discover_stacked_checkpoint_identity(
    checkpoint_root: Path,
    *,
    verified_inputs: Mapping[str, _VerifiedInput],
    sample_fraction: float,
    sample_seed: int,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
) -> dict[str, object] | None:
    """Recover a current live-stack identity before loading survey sources.

    The configured namespace binds all pins and scale controls available before
    assembly.  A stage manifest inside it supplies the realized stack receipt;
    this function recomputes the complete identity from that receipt and accepts
    it only when every current code, input, authority, and sampling field agrees.
    The regular checkpoint loader subsequently authenticates the sidecar, H5,
    frame metadata, and receipt bytes before any stage is resumed.
    """

    root = Path(checkpoint_root)
    for stage in reversed(POOL_CHECKPOINT_STAGE_ORDER):
        checkpoint_path = root / _POOL_STAGE_CHECKPOINT_FILENAMES[stage]
        manifest_path = checkpoint_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json_object(manifest_path)
            if (
                manifest.get("artifact_kind")
                != _POOL_STAGE_CHECKPOINT_MANIFEST_ARTIFACT_KIND
                or manifest.get("schema_version")
                != POOL_STAGE_CHECKPOINT_SCHEMA_VERSION
                or manifest.get("materializer_version")
                != POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
                or manifest.get("stage") != stage
            ):
                raise ValueError("unsupported checkpoint manifest binding")
            stage_identity = manifest.get("identity")
            if not isinstance(stage_identity, Mapping):
                raise ValueError("checkpoint manifest identity is not an object")
            if stage_identity.get("stage") != stage or stage_identity.get(
                "stage_index"
            ) != POOL_CHECKPOINT_STAGE_ORDER.index(stage):
                raise ValueError("checkpoint manifest stage identity changed")
            base_identity = dict(stage_identity)
            del base_identity["stage"]
            del base_identity["stage_index"]
            sampling = base_identity.get("sampling")
            if not isinstance(sampling, Mapping):
                raise ValueError("checkpoint identity has no sampling object")
            stack_manifest = sampling.get("stack_manifest")
            if not isinstance(stack_manifest, Mapping):
                raise ValueError("checkpoint identity has no stack manifest")
            expected = _stacked_checkpoint_base_identity(
                verified_inputs,
                stack_receipt=stack_manifest,
                sample_fraction=sample_fraction,
                sample_seed=sample_seed,
                clone_attachment_fraction=clone_attachment_fraction,
                clone_attachment_seed=clone_attachment_seed,
            )
            if base_identity != expected:
                raise ValueError("checkpoint base identity is stale")
            return expected
        except Exception as error:
            print(
                f"Ignored stacked checkpoint discovery manifest {manifest_path}: "
                f"{type(error).__name__}: {error}."
            )
    return None


def _stacked_realized_counts(
    stack_receipt: Mapping[str, object],
) -> tuple[int, int]:
    samples = stack_receipt.get("survey_samples")
    if not isinstance(samples, Mapping):
        raise ValueError("Production stack receipt has no survey_samples object.")

    def realized(channel: str) -> int:
        sample = samples.get(channel)
        if not isinstance(sample, Mapping):
            raise ValueError(f"Production stack receipt has no {channel!r} sample.")
        value = sample.get("realized_household_count")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(
                f"Production stack receipt has invalid {channel!r} realized count."
            )
        return value

    return realized("asec"), realized("acs")


def _new_stacked_release_id(
    *,
    sample_fraction: float,
    sample_seed: int,
    realized_asec_households: int,
    realized_acs_households: int,
    timestamp: datetime | None = None,
    nonce: str | None = None,
) -> str:
    fraction_token, _rung = _stacked_rung(sample_fraction)
    instant = datetime.now(UTC) if timestamp is None else timestamp.astimezone(UTC)
    suffix = uuid.uuid4().hex[:8] if nonce is None else nonce
    if not re.fullmatch(r"[0-9a-f]{8}", suffix):
        raise ValueError("Stacked release nonce must be eight lowercase hex digits.")
    release_id = (
        f"populace-us-2024-stacked-{fraction_token}-s{sample_seed}-"
        f"asec{realized_asec_households}-acs{realized_acs_households}-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{suffix}"
    )
    if not _STACKED_RELEASE_ID_PATTERN.fullmatch(release_id):  # pragma: no cover
        raise AssertionError(f"Generated invalid stacked release ID {release_id!r}.")
    return release_id


def _pool_checkpoint_stage_identity(
    base_identity: Mapping[str, object],
    stage: str,
) -> dict[str, object]:
    if stage not in POOL_CHECKPOINT_STAGE_ORDER:
        raise ValueError(f"Unknown pool checkpoint stage {stage!r}.")
    return {
        **dict(base_identity),
        "stage": stage,
        "stage_index": POOL_CHECKPOINT_STAGE_ORDER.index(stage),
    }


def _pool_checkpoint_identity_sha256(identity: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(identity)).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


class _PoolStageCheckpointStore:
    """Identity-guarded durable storage for the three pool cut points."""

    def __init__(
        self,
        root: Path,
        *,
        base_identity: Mapping[str, object],
    ) -> None:
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(
                f"Pool checkpoint root exists but is not a directory: {self.root}."
            )
        normalized_identity = _json_ready(base_identity)
        if not isinstance(normalized_identity, dict):  # pragma: no cover
            raise TypeError("Pool checkpoint base identity must be an object.")
        self._base_identity = normalized_identity
        self._input_receipts: dict[str, object] | None = None
        self._resumed_from: str | None = None
        self._attempts: dict[str, dict[str, object]] = {
            stage: {"load_status": "not_attempted"}
            for stage in POOL_CHECKPOINT_STAGE_ORDER
        }
        self._writes: dict[str, dict[str, object]] = {}

    @property
    def base_identity_sha256(self) -> str:
        return _pool_checkpoint_identity_sha256(self._base_identity)

    @property
    def base_identity(self) -> dict[str, object]:
        return dict(self._base_identity)

    @property
    def input_receipts(self) -> dict[str, object]:
        if self._input_receipts is None:
            raise RuntimeError("Pool checkpoint input receipts are not bound.")
        return dict(self._input_receipts)

    def bind_input_receipts(self, receipts: Mapping[str, object]) -> None:
        normalized = _json_ready(receipts)
        if not isinstance(normalized, dict):  # pragma: no cover
            raise TypeError("Pool input receipts must normalize to an object.")
        if self._input_receipts is not None and self._input_receipts != normalized:
            raise ValueError(
                "Pool checkpoint input receipts disagree across stage artifacts."
            )
        self._input_receipts = normalized

    def checkpoint_path(self, stage: str) -> Path:
        try:
            filename = _POOL_STAGE_CHECKPOINT_FILENAMES[stage]
        except KeyError as exc:
            raise ValueError(f"Unknown pool checkpoint stage {stage!r}.") from exc
        return self.root / filename

    def checkpoint_manifest_path(self, stage: str) -> Path:
        path = self.checkpoint_path(stage)
        return path.with_suffix(".manifest.json")

    def checkpoint_receipts_path(self, stage: str) -> Path:
        """Return the non-identity-bearing operational-receipts sidecar path."""

        path = self.checkpoint_path(stage)
        return path.with_suffix(".receipts.json")

    def load_deepest(self) -> MultispinePoolCheckpoint | None:
        """Load the deepest valid checkpoint, ignoring stale or corrupt files."""

        for stage in reversed(POOL_CHECKPOINT_STAGE_ORDER):
            checkpoint = self._load(stage)
            if checkpoint is None:
                continue
            self._resumed_from = stage
            covered = POOL_CHECKPOINT_STAGE_ORDER[
                : POOL_CHECKPOINT_STAGE_ORDER.index(stage) + 1
            ]
            print(
                f"Resumed pool checkpoint {stage!r} from "
                f"{self.checkpoint_path(stage)}; cached stages: "
                f"{', '.join(covered)}."
            )
            return checkpoint
        print(
            "No valid pool stage checkpoint found; rebuilding assembly, "
            "transfer, and simulation."
        )
        return None

    def write(self, checkpoint: MultispinePoolCheckpoint) -> None:
        """Atomically write one completed cut point and its content sidecar."""

        stage = checkpoint.stage
        if stage not in POOL_CHECKPOINT_STAGE_ORDER:
            raise ValueError(f"Unknown pool checkpoint stage {stage!r}.")
        if self._input_receipts is None:
            raise RuntimeError(
                "Pool checkpoint input receipts must be bound before writing."
            )
        persistent_frame = canonicalize_frame_string_dtypes(
            checkpoint.frame,
            boundary=f"pool {stage} checkpoint write",
        )
        stored_frame = persistent_frame
        if stage == "simulated":
            if checkpoint.simulation_frame is None:
                raise ValueError(
                    "The simulated pool checkpoint requires an evaluation frame."
                )
            simulation_frame = canonicalize_frame_string_dtypes(
                checkpoint.simulation_frame,
                boundary="pool simulated evaluation checkpoint write",
            )
            _assert_simulation_checkpoint_pair(
                persistent_frame,
                simulation_frame,
            )
            stored_frame = simulation_frame
        elif checkpoint.simulation_frame is not None:
            raise ValueError(
                f"Pool checkpoint stage {stage!r} cannot carry a simulation frame."
            )

        validate_assembly_provenance(
            stored_frame,
            boundary=f"pool {stage} checkpoint write",
        )
        canonical_stage_receipts, operational_stage_receipts = (
            _split_checkpoint_stage_receipts(checkpoint.stage_receipts)
        )
        receipts_path = self.checkpoint_receipts_path(stage)
        try:
            receipts_path.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(receipts_path)
        identity = _pool_checkpoint_stage_identity(self._base_identity, stage)
        identity_sha256 = _pool_checkpoint_identity_sha256(identity)
        metadata = {
            "artifact_kind": _POOL_STAGE_CHECKPOINT_ARTIFACT_KIND,
            "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
            "materializer_version": POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
            "stage": stage,
            "identity": identity,
            "identity_sha256": identity_sha256,
            "row_counts": _frame_row_counts(stored_frame),
            "frame_schema": _frame_schema_payload(stored_frame),
            "frame_metadata": _json_ready(stored_frame.metadata),
            "assembly_receipt": _json_ready(checkpoint.assembly_receipt),
            "stage_receipts": canonical_stage_receipts,
            "input_receipts": self._input_receipts,
            "simulation_output": (
                {
                    "column": _POOL_SIMULATION_OUTPUT_COLUMN,
                    "entity": "person",
                    "persisted_to_pool": False,
                }
                if stage == "simulated"
                else None
            ),
        }
        path = self.checkpoint_path(stage)
        started_at = time.perf_counter()
        write_frame_checkpoint(path, stored_frame, metadata=metadata)
        checkpoint_sha256 = _file_sha256(path)
        size_bytes = path.stat().st_size
        manifest_path = self.checkpoint_manifest_path(stage)
        _atomic_write_json(
            manifest_path,
            {
                "artifact_kind": _POOL_STAGE_CHECKPOINT_MANIFEST_ARTIFACT_KIND,
                "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
                "materializer_version": POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
                "stage": stage,
                "identity": identity,
                "identity_sha256": identity_sha256,
                "checkpoint": {
                    "filename": path.name,
                    "sha256": checkpoint_sha256,
                    "size_bytes": size_bytes,
                },
                "row_counts": _frame_row_counts(stored_frame),
                "frame_schema": _frame_schema_payload(stored_frame),
                "frame_metadata": metadata["frame_metadata"],
            },
        )
        receipts_record: dict[str, object] = {
            "path": str(receipts_path.resolve()),
            "write_status": "not_applicable",
        }
        if operational_stage_receipts:
            _atomic_write_json(
                receipts_path,
                {
                    "artifact_kind": _POOL_STAGE_CHECKPOINT_RECEIPTS_ARTIFACT_KIND,
                    "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
                    "materializer_version": (
                        POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
                    ),
                    "stage": stage,
                    "identity_sha256": identity_sha256,
                    "checkpoint": {
                        "filename": path.name,
                        "sha256": checkpoint_sha256,
                        "size_bytes": size_bytes,
                    },
                    "operational_stage_receipts": operational_stage_receipts,
                },
            )
            receipts_record = {
                "path": str(receipts_path.resolve()),
                "write_status": "written",
                "sha256": _file_sha256(receipts_path),
                "size_bytes": receipts_path.stat().st_size,
            }
        write_seconds = time.perf_counter() - started_at
        self._writes[stage] = {
            "checkpoint_sha256": checkpoint_sha256,
            "size_bytes": size_bytes,
            "write_seconds": write_seconds,
            "receipts_sidecar": receipts_record,
        }
        print(
            f"Rebuilt pool stage {stage!r}; wrote checkpoint {path} "
            f"({size_bytes} bytes, identity {identity_sha256})."
        )

    def provenance(
        self,
        *,
        primary_qrf_checkpoint_dir: Path,
        acs_transfer_checkpoint_dir: Path | None = None,
    ) -> dict[str, object]:
        """Return final-manifest evidence for every cached or rebuilt stage."""

        resumed_index = (
            None
            if self._resumed_from is None
            else POOL_CHECKPOINT_STAGE_ORDER.index(self._resumed_from)
        )
        stages: dict[str, dict[str, object]] = {}
        for stage_index, stage in enumerate(POOL_CHECKPOINT_STAGE_ORDER):
            identity = _pool_checkpoint_stage_identity(self._base_identity, stage)
            source = (
                "checkpoint"
                if resumed_index is not None and stage_index <= resumed_index
                else "rebuilt"
            )
            covered_by_deeper = source == "checkpoint" and stage != self._resumed_from
            artifact_stage = self._resumed_from if covered_by_deeper else stage
            if artifact_stage is None:  # pragma: no cover - source proves non-null
                raise AssertionError("Covered checkpoint stage has no source stage.")
            record: dict[str, object] = {
                "source": source,
                "resume_kind": (
                    "direct"
                    if stage == self._resumed_from
                    else (
                        "covered_by_deeper_checkpoint"
                        if source == "checkpoint"
                        else None
                    )
                ),
                "identity_sha256": _pool_checkpoint_identity_sha256(identity),
                "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
                "materializer_version": (POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION),
                "path": str(self.checkpoint_path(artifact_stage).resolve()),
                "manifest_path": str(
                    self.checkpoint_manifest_path(artifact_stage).resolve()
                ),
                **self._attempts[stage],
            }
            if stage in self._writes:
                record.update(self._writes[stage])
            elif stage == self._resumed_from:
                record.update(
                    {
                        key: value
                        for key, value in self._attempts[stage].items()
                        if key in {"checkpoint_sha256", "size_bytes"}
                    }
                )
            if covered_by_deeper:
                record["source_checkpoint_stage"] = self._resumed_from
                record["source_checkpoint_identity_sha256"] = (
                    _pool_checkpoint_identity_sha256(
                        _pool_checkpoint_stage_identity(
                            self._base_identity,
                            artifact_stage,
                        )
                    )
                )
                record["nominal_stage_path"] = str(
                    self.checkpoint_path(stage).resolve()
                )
                record["nominal_stage_manifest_path"] = str(
                    self.checkpoint_manifest_path(stage).resolve()
                )
            stages[stage] = record
        return {
            "artifact_kind": "populace_us_multispine_pool_checkpoint_provenance",
            "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
            "materializer_version": POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
            "root": str(self.root.resolve()),
            "base_identity_sha256": self.base_identity_sha256,
            "deepest_resumed_stage": self._resumed_from,
            "stages": stages,
            "primary_qrf": {
                "path": str(Path(primary_qrf_checkpoint_dir).resolve()),
                "base_identity_sha256": self.base_identity_sha256,
            },
            "acs_transfer": {
                "path": str(
                    Path(
                        acs_transfer_checkpoint_dir
                        if acs_transfer_checkpoint_dir is not None
                        else self.root / "acs-transfer" / self.base_identity_sha256
                    ).resolve()
                ),
                "base_identity_sha256": self.base_identity_sha256,
                "boundary_stage": "transferred",
            },
            "agreement": {
                "source": "always_fresh",
                "cached": False,
                "terminal_verdict_persisted": False,
            },
        }

    def _load(self, stage: str) -> MultispinePoolCheckpoint | None:
        path = self.checkpoint_path(stage)
        manifest_path = self.checkpoint_manifest_path(stage)
        if not path.exists() and not manifest_path.exists():
            self._attempts[stage] = {"load_status": "missing"}
            return None
        if not path.is_file() or not manifest_path.is_file():
            return self._invalid(
                stage,
                reason="incomplete_checkpoint",
                error=ValueError(
                    "checkpoint H5 and manifest sidecar must both be regular files"
                ),
            )

        expected_identity = _pool_checkpoint_stage_identity(
            self._base_identity,
            stage,
        )
        expected_identity_sha256 = _pool_checkpoint_identity_sha256(expected_identity)
        try:
            manifest = _read_json_object(manifest_path)
            if (
                manifest.get("artifact_kind")
                != _POOL_STAGE_CHECKPOINT_MANIFEST_ARTIFACT_KIND
                or manifest.get("schema_version")
                != POOL_STAGE_CHECKPOINT_SCHEMA_VERSION
                or manifest.get("materializer_version")
                != POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
                or manifest.get("stage") != stage
            ):
                raise ValueError(
                    f"{stage} checkpoint manifest has an unsupported binding"
                )
            observed_identity = manifest.get("identity")
            if observed_identity != expected_identity:
                observed_digest = (
                    _pool_checkpoint_identity_sha256(observed_identity)
                    if isinstance(observed_identity, Mapping)
                    else None
                )
                self._attempts[stage] = {
                    "load_status": "identity_mismatch",
                    "ignored_checkpoint": {
                        "expected_identity_sha256": expected_identity_sha256,
                        "observed_identity_sha256": observed_digest,
                    },
                }
                print(
                    f"Ignored stale pool checkpoint {stage!r} at {path}: "
                    f"identity {observed_digest!r} != "
                    f"{expected_identity_sha256}."
                )
                return None
            if manifest.get("identity_sha256") != expected_identity_sha256:
                raise ValueError(f"{stage} checkpoint manifest identity digest changed")
            checkpoint_receipt = manifest.get("checkpoint")
            if not isinstance(checkpoint_receipt, Mapping):
                raise ValueError(
                    f"{stage} checkpoint manifest has no checkpoint receipt"
                )
            if checkpoint_receipt.get("filename") != path.name:
                raise ValueError(
                    f"{stage} checkpoint manifest names a different H5 file"
                )
            expected_file_sha256 = checkpoint_receipt.get("sha256")
            if not isinstance(expected_file_sha256, str) or not (
                _LOWERCASE_SHA256.fullmatch(expected_file_sha256)
            ):
                raise ValueError(
                    f"{stage} checkpoint manifest has an invalid H5 digest"
                )
            actual_file_sha256 = _file_sha256(path)
            if actual_file_sha256 != expected_file_sha256:
                raise ValueError(
                    f"{stage} checkpoint H5 SHA-256 mismatch: got "
                    f"{actual_file_sha256}, expected {expected_file_sha256}"
                )
            expected_size = checkpoint_receipt.get("size_bytes")
            if expected_size != path.stat().st_size:
                raise ValueError(
                    f"{stage} checkpoint H5 size changed: got "
                    f"{path.stat().st_size}, expected {expected_size!r}"
                )

            sidecar_frame_metadata = manifest.get("frame_metadata")
            if not isinstance(sidecar_frame_metadata, Mapping):
                raise ValueError(
                    f"{stage} checkpoint frame_metadata sidecar must be an object"
                )
            loaded = load_frame_checkpoint(
                path,
                frame_metadata=sidecar_frame_metadata,
            )
            metadata = loaded.metadata
            _validate_checkpoint_metadata(
                metadata,
                stage=stage,
                expected_identity=expected_identity,
                expected_identity_sha256=expected_identity_sha256,
            )
            for key in ("row_counts", "frame_schema", "frame_metadata"):
                if metadata.get(key) != manifest.get(key):
                    raise ValueError(
                        f"{stage} checkpoint {key} differs from its sidecar"
                    )
            frame_metadata = metadata.get("frame_metadata")
            if not isinstance(frame_metadata, Mapping):
                raise ValueError(f"{stage} checkpoint frame_metadata must be an object")
            frame = loaded.frame
            _validate_checkpoint_frame(
                frame,
                stage=stage,
                row_counts=metadata.get("row_counts"),
                frame_schema=metadata.get("frame_schema"),
            )
            # Authenticate legacy bytes and their literal sidecar schema before
            # applying the current in-memory serialization-boundary policy.
            frame = canonicalize_frame_string_dtypes(
                frame,
                boundary=f"pool {stage} checkpoint load",
                in_place=True,
            )
            assembly_receipt = metadata.get("assembly_receipt")
            stage_receipts = metadata.get("stage_receipts")
            input_receipts = metadata.get("input_receipts")
            if not isinstance(assembly_receipt, Mapping):
                raise ValueError(
                    f"{stage} checkpoint assembly_receipt must be an object"
                )
            if _json_ready(
                frame.metadata.get(SPINE_ASSEMBLY_MANIFEST_KEY)
            ) != _json_ready(assembly_receipt):
                raise ValueError(
                    f"{stage} checkpoint assembly receipt differs from Frame metadata"
                )
            if not isinstance(stage_receipts, Mapping):
                raise ValueError(f"{stage} checkpoint stage_receipts must be an object")
            if not isinstance(input_receipts, Mapping):
                raise ValueError(f"{stage} checkpoint input_receipts must be an object")
            _validate_checkpoint_receipt_prefix(stage, stage_receipts)
            restored_stage_receipts, receipts_record = (
                self._load_operational_stage_receipts(
                    stage,
                    canonical_stage_receipts=stage_receipts,
                    expected_identity_sha256=expected_identity_sha256,
                    checkpoint_sha256=actual_file_sha256,
                    checkpoint_size=path.stat().st_size,
                )
            )
            persistent_frame = frame
            simulation_frame: Frame | None = None
            if stage == "simulated":
                simulation_output = metadata.get("simulation_output")
                if simulation_output != {
                    "column": _POOL_SIMULATION_OUTPUT_COLUMN,
                    "entity": "person",
                    "persisted_to_pool": False,
                }:
                    raise ValueError(
                        "simulated checkpoint has an invalid SSI output binding"
                    )
                simulation_frame = frame
                persistent_frame = _without_simulation_output(frame)

            checkpoint = MultispinePoolCheckpoint(
                stage=stage,
                frame=persistent_frame,
                assembly_receipt=dict(assembly_receipt),
                stage_receipts=restored_stage_receipts,
                simulation_frame=simulation_frame,
            )
            self.bind_input_receipts(input_receipts)
            self._attempts[stage] = {
                "load_status": "resumed",
                "checkpoint_sha256": actual_file_sha256,
                "size_bytes": path.stat().st_size,
                "receipts_sidecar": receipts_record,
            }
            return checkpoint
        except Exception as error:  # corrupted local artifacts are rebuildable
            return self._invalid(
                stage,
                reason="checkpoint_validation_failed",
                error=error,
            )

    def _load_operational_stage_receipts(
        self,
        stage: str,
        *,
        canonical_stage_receipts: Mapping[str, object],
        expected_identity_sha256: str,
        checkpoint_sha256: str,
        checkpoint_size: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Restore optional observability without affecting checkpoint validity."""

        receipts_path = self.checkpoint_receipts_path(stage)
        base_record: dict[str, object] = {"path": str(receipts_path.resolve())}
        if not receipts_path.exists():
            return dict(canonical_stage_receipts), {
                **base_record,
                "load_status": "missing",
            }
        if not receipts_path.is_file():
            error = ValueError("operational receipts sidecar must be a regular file")
            return dict(canonical_stage_receipts), self._invalid_receipts_record(
                stage,
                receipts_path,
                error,
            )
        try:
            payload = _read_json_object(receipts_path)
            expected_keys = {
                "artifact_kind",
                "schema_version",
                "materializer_version",
                "stage",
                "identity_sha256",
                "checkpoint",
                "operational_stage_receipts",
            }
            if set(payload) != expected_keys:
                raise ValueError("operational receipts sidecar keys changed")
            if (
                payload.get("artifact_kind")
                != _POOL_STAGE_CHECKPOINT_RECEIPTS_ARTIFACT_KIND
                or payload.get("schema_version") != POOL_STAGE_CHECKPOINT_SCHEMA_VERSION
                or payload.get("materializer_version")
                != POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
                or payload.get("stage") != stage
                or payload.get("identity_sha256") != expected_identity_sha256
            ):
                raise ValueError(
                    "operational receipts sidecar has an unsupported binding"
                )
            checkpoint = payload.get("checkpoint")
            if checkpoint != {
                "filename": self.checkpoint_path(stage).name,
                "sha256": checkpoint_sha256,
                "size_bytes": checkpoint_size,
            }:
                raise ValueError(
                    "operational receipts sidecar checkpoint binding changed"
                )
            operational = payload.get("operational_stage_receipts")
            if not isinstance(operational, Mapping):
                raise ValueError("operational_stage_receipts must be an object")
            restored = _attach_checkpoint_operational_receipts(
                canonical_stage_receipts,
                operational,
            )
            return restored, {
                **base_record,
                "load_status": "loaded",
                "sha256": _file_sha256(receipts_path),
                "size_bytes": receipts_path.stat().st_size,
            }
        except Exception as error:
            return dict(canonical_stage_receipts), self._invalid_receipts_record(
                stage,
                receipts_path,
                error,
            )

    @staticmethod
    def _invalid_receipts_record(
        stage: str,
        receipts_path: Path,
        error: Exception,
    ) -> dict[str, object]:
        failure = {
            "reason": "operational_receipts_validation_failed",
            "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "path": str(receipts_path.resolve()),
        }
        print(
            f"Ignored invalid pool checkpoint operational receipts for {stage!r} "
            f"at {receipts_path}: {type(error).__name__}: {error}."
        )
        return {
            "path": str(receipts_path.resolve()),
            "load_status": "invalid_ignored",
            "invalid_sidecar": failure,
        }

    def _invalid(
        self,
        stage: str,
        *,
        reason: str,
        error: Exception,
    ) -> None:
        failure = {
            "reason": reason,
            "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "path": str(self.checkpoint_path(stage).resolve()),
        }
        self._attempts[stage] = {
            "load_status": "invalid_rebuild",
            "invalid_checkpoint": failure,
        }
        print(
            f"Ignored corrupt pool checkpoint {stage!r} at "
            f"{self.checkpoint_path(stage)}: {type(error).__name__}: {error}; "
            "rebuilding."
        )
        return None


def _validate_checkpoint_metadata(
    metadata: Mapping[str, object],
    *,
    stage: str,
    expected_identity: Mapping[str, object],
    expected_identity_sha256: str,
) -> None:
    if (
        metadata.get("artifact_kind") != _POOL_STAGE_CHECKPOINT_ARTIFACT_KIND
        or metadata.get("schema_version") != POOL_STAGE_CHECKPOINT_SCHEMA_VERSION
        or metadata.get("materializer_version")
        != POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
        or metadata.get("stage") != stage
    ):
        raise ValueError(f"{stage} checkpoint metadata has an unsupported binding")
    if metadata.get("identity") != expected_identity:
        raise ValueError(f"{stage} checkpoint embedded identity changed")
    if metadata.get("identity_sha256") != expected_identity_sha256:
        raise ValueError(f"{stage} checkpoint embedded identity digest changed")


def _validate_checkpoint_receipt_prefix(
    stage: str,
    stage_receipts: Mapping[str, object],
) -> None:
    expected = {
        "assembled": frozenset(),
        "transferred": frozenset({"impute"}),
        "simulated": frozenset({"impute", "derive", "seed", "simulate"}),
    }[stage]
    observed = frozenset(stage_receipts)
    allowed = expected | ({"clone"} if stage != "assembled" else set())
    if not expected <= observed or not observed <= allowed:
        raise ValueError(
            f"{stage} checkpoint receipt stages are {sorted(observed)}, "
            f"expected {sorted(expected)} with an optional clone receipt"
        )


def _split_checkpoint_stage_receipts(
    stage_receipts: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Separate operational bank evidence from canonical stage receipts."""

    normalized = _json_ready(stage_receipts)
    if not isinstance(normalized, dict):  # pragma: no cover - mapping normalization
        raise TypeError("Pool checkpoint stage receipts must normalize to an object.")
    impute = normalized.get("impute")
    if not isinstance(impute, dict):
        return normalized, {}
    operational_impute: dict[str, object] = {}
    primary_qrf = impute.get("primary_puf_qrf")
    if isinstance(primary_qrf, dict):
        operational_primary: dict[str, object] = {}
        if "resume_status" in primary_qrf:
            resume_status = primary_qrf.pop("resume_status")
            if not isinstance(resume_status, str) or not resume_status:
                raise ValueError(
                    "Primary QRF resume status must be a non-empty string."
                )
            operational_primary["resume_status"] = resume_status
        if "identity_routing" in primary_qrf:
            identity_routing = primary_qrf.pop("identity_routing")
            if not isinstance(identity_routing, Mapping):
                raise ValueError("Primary QRF identity routing must be an object.")
            operational_primary["identity_routing"] = dict(identity_routing)
        if "checkpoint_manifest_path" in primary_qrf:
            checkpoint_manifest_path = primary_qrf.pop("checkpoint_manifest_path")
            if (
                not isinstance(checkpoint_manifest_path, str)
                or not checkpoint_manifest_path
            ):
                raise ValueError(
                    "Primary QRF checkpoint manifest path must be a non-empty string."
                )
            operational_primary["checkpoint_manifest_path"] = checkpoint_manifest_path
        if operational_primary:
            operational_impute["primary_puf_qrf"] = operational_primary
    acs_transfer = impute.get("acs_qrf_transfer")
    if isinstance(acs_transfer, dict) and "target_bank" in acs_transfer:
        target_bank = acs_transfer.pop("target_bank")
        if not isinstance(target_bank, Mapping):
            raise ValueError("ACS transfer target-bank receipt must be an object.")
        operational_impute["acs_qrf_transfer"] = {
            "target_bank": dict(target_bank),
        }
    if not operational_impute:
        return normalized, {}
    return normalized, {"impute": operational_impute}


def _attach_checkpoint_operational_receipts(
    canonical_stage_receipts: Mapping[str, object],
    operational_stage_receipts: Mapping[str, object],
) -> dict[str, object]:
    """Reattach non-canonical bank evidence for runtime observability."""

    canonical, existing_operational = _split_checkpoint_stage_receipts(
        canonical_stage_receipts
    )
    if existing_operational:
        raise ValueError("canonical stage receipts already contain bank provenance")
    operational = _json_ready(operational_stage_receipts)
    if not isinstance(operational, dict):  # pragma: no cover - mapping normalization
        raise TypeError("Operational stage receipts must normalize to an object.")
    if set(operational) != {"impute"}:
        raise ValueError("operational stage receipts must contain only impute")
    operational_impute = operational.get("impute")
    if not isinstance(operational_impute, dict) or not operational_impute:
        raise ValueError("operational impute receipt shape changed")
    canonical_impute = canonical.get("impute")
    if not isinstance(canonical_impute, dict):
        raise ValueError("canonical stage receipts have no impute object")
    allowed_fields = {
        "primary_puf_qrf": frozenset(
            {
                "resume_status",
                "identity_routing",
                "checkpoint_manifest_path",
            }
        ),
        "acs_qrf_transfer": frozenset({"target_bank"}),
    }
    for section, operational_section in operational_impute.items():
        if section not in allowed_fields or not isinstance(operational_section, dict):
            raise ValueError("operational impute receipt shape changed")
        if (
            not operational_section
            or not set(operational_section) <= allowed_fields[section]
        ):
            raise ValueError(f"operational {section} receipt shape changed")
        canonical_section = canonical_impute.get(section)
        if not isinstance(canonical_section, dict):
            raise ValueError(f"canonical impute receipt has no {section} object")
        if set(canonical_section) & set(operational_section):
            raise ValueError(
                f"canonical {section} receipt already contains bank provenance"
            )
        canonical_section.update(operational_section)
    return canonical


def _frame_row_counts(frame: Frame) -> dict[str, int]:
    return {entity: int(frame.n(entity)) for entity in frame.entities}


def _frame_schema_payload(frame: Frame) -> dict[str, object]:
    return {
        "entities": {
            entity: [
                {"name": str(column), "dtype": str(frame.table(entity)[column].dtype)}
                for column in frame.table(entity).columns
            ]
            for entity in frame.entities
        },
        "links": {
            link: [
                {"name": str(column), "dtype": str(frame.link(link)[column].dtype)}
                for column in frame.link(link).columns
            ]
            for link in frame.links
        },
        "weighted_entities": {
            entity: frame.weights_for(entity).kind.value
            for entity in frame.weighted_entities
        },
    }


def _validate_checkpoint_frame(
    frame: Frame,
    *,
    stage: str,
    row_counts: object,
    frame_schema: object,
) -> None:
    if frame.schema != US_SCHEMA:
        raise ValueError(f"{stage} checkpoint does not carry the US entity schema")
    if row_counts != _frame_row_counts(frame):
        raise ValueError(
            f"{stage} checkpoint row counts changed: got {_frame_row_counts(frame)}, "
            f"expected {row_counts!r}"
        )
    if frame_schema != _frame_schema_payload(frame):
        raise ValueError(f"{stage} checkpoint table schema changed")
    validate_assembly_provenance(
        frame,
        boundary=f"pool {stage} checkpoint load",
    )


def _without_simulation_output(frame: Frame) -> Frame:
    person = frame.table("person")
    if _POOL_SIMULATION_OUTPUT_COLUMN not in person:
        raise ValueError("simulated checkpoint is missing ephemeral person.ssi output")
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = person.drop(columns=[_POOL_SIMULATION_OUTPUT_COLUMN])
    tables.update({link: frame.link(link) for link in frame.links})
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _assert_simulation_checkpoint_pair(
    persistent: Frame,
    evaluation: Frame,
) -> None:
    """Prove the evaluation checkpoint is exactly pool inputs plus SSI."""

    if persistent.schema != evaluation.schema:
        raise ValueError("Simulation checkpoint changed the pool entity schema.")
    if (
        persistent.entities != evaluation.entities
        or persistent.links != evaluation.links
    ):
        raise ValueError("Simulation checkpoint changed pool table membership.")
    for entity in persistent.entities:
        expected = persistent.table(entity)
        observed = evaluation.table(entity)
        if entity == "person":
            if _POOL_SIMULATION_OUTPUT_COLUMN not in observed:
                raise ValueError(
                    "Simulation checkpoint evaluation is missing person.ssi."
                )
            observed = observed.drop(columns=[_POOL_SIMULATION_OUTPUT_COLUMN])
        pd.testing.assert_frame_equal(
            observed,
            expected,
            check_dtype=True,
            check_exact=True,
        )
    for link in persistent.links:
        pd.testing.assert_frame_equal(
            evaluation.link(link),
            persistent.link(link),
            check_dtype=True,
            check_exact=True,
        )
    if persistent.weighted_entities != evaluation.weighted_entities:
        raise ValueError("Simulation checkpoint changed weighted entities.")
    for entity in persistent.weighted_entities:
        before = persistent.weights_for(entity)
        after = evaluation.weights_for(entity)
        if before.kind != after.kind or not np.array_equal(before.values, after.values):
            raise ValueError(f"Simulation checkpoint changed {entity!r} weights.")
    pd.testing.assert_series_equal(
        evaluation.strata,
        persistent.strata,
        check_dtype=True,
        check_exact=True,
    )
    if evaluation.mass_log != persistent.mass_log:
        raise ValueError("Simulation checkpoint changed the pool mass log.")
    if _json_ready(evaluation.metadata) != _json_ready(persistent.metadata):
        raise ValueError("Simulation checkpoint changed the assembly metadata.")


def _initialize_or_resume_primary_qrf(
    frame: Frame,
    donor: pd.DataFrame,
    checkpoint_dir: Path,
    *,
    input_binding: Mapping[str, object],
) -> str:
    manifest_path = _primary_qrf_manifest_path(checkpoint_dir)
    binding_path = _primary_qrf_input_binding_path(checkpoint_dir)
    expected_binding = _json_ready(input_binding)
    if manifest_path.is_file():
        if not binding_path.is_file():
            raise ValueError(
                "Primary QRF checkpoint has no pool-input provenance binding: "
                f"{binding_path}."
            )
        observed_binding = _read_json_object(binding_path)
        if observed_binding != expected_binding:
            raise ValueError(
                "Primary QRF checkpoint input binding differs from the verified "
                "pool inputs; refusing to reuse stale predictions."
            )
        return "resumed"
    if checkpoint_dir.exists():
        if not checkpoint_dir.is_dir():
            raise ValueError(
                "Primary QRF checkpoint path exists but is not a directory: "
                f"{checkpoint_dir}."
            )
        if any(checkpoint_dir.iterdir()):
            raise ValueError(
                "Primary QRF checkpoint directory is nonempty but has no bound "
                f"manifest: {checkpoint_dir}."
            )
    initialize_primary_puf_qrf_chain(
        frame,
        donor,
        checkpoint_dir,
        seed=POOL_RANDOM_SEED,
        n_estimators=_PRIMARY_QRF_N_ESTIMATORS,
    )
    _atomic_write_json(binding_path, input_binding)
    return "initialized"


def _impute_pool(
    frame: Frame,
    *,
    puf_donor: pd.DataFrame,
    primary_qrf_checkpoint_dir: Path,
    acs_transfer_checkpoint_dir: Path,
    checkpoint_identity: Mapping[str, object],
    checkpoint_input_binding: Mapping[str, object],
) -> PoolStageOutput:
    current_base_identity_sha256 = _pool_checkpoint_identity_sha256(checkpoint_identity)
    primary_qrf_identity_routing = _identity_routed_bank_open_receipt(
        primary_qrf_checkpoint_dir,
        current_base_identity_sha256=current_base_identity_sha256,
    )
    qrf_resume_status = _initialize_or_resume_primary_qrf(
        frame,
        puf_donor,
        primary_qrf_checkpoint_dir,
        input_binding=checkpoint_input_binding,
    )
    run_primary_puf_qrf_chain(primary_qrf_checkpoint_dir)

    tail_bound_diagnostics: list[dict[str, object]] = []
    with_primary_detail, primary_weight_kind = finalize_primary_puf_qrf_chain(
        frame,
        primary_qrf_checkpoint_dir,
        tail_bound_diagnostics=tail_bound_diagnostics,
    )
    with_tail, tail_receipt = transfer_puf_capital_gains_tail(
        with_primary_detail,
        puf_donor,
        seed=POOL_RANDOM_SEED,
    )
    validate_puf_capital_gains_tail_manifest(tail_receipt)
    tail_ceiling = tail_receipt["tail_distribution_receipts"]["frame_after_stage"]
    if not tail_ceiling["positive_mass_five_x_target_exceeded"]:
        raise ValueError(
            "PUF capital-gains tail transfer did not clear its declared "
            "five-times positive-mass target: "
            f"{tail_ceiling['positive_mass_five_x_ceiling']} <= "
            f"{tail_ceiling['positive_mass_five_x_target']}."
        )
    source_completion = complete_multispine_source_inputs(with_tail)
    transfer_families = pool_transfer_target_families()
    acs_transfer_identity_routing = _identity_routed_bank_open_receipt(
        acs_transfer_checkpoint_dir,
        current_base_identity_sha256=current_base_identity_sha256,
    )
    target_bank = AcsTransferTargetBankStore(
        acs_transfer_checkpoint_dir,
        identity=_pool_checkpoint_stage_identity(
            checkpoint_identity,
            "transferred",
        ),
    )
    transferred = transfer_acs_inputs(
        source_completion.frame,
        source_completion.frame,
        target_families=transfer_families,
        donor_channel=ACS_DONOR_CHANNEL_AUTO,
        seed=POOL_RANDOM_SEED,
        n_estimators=_ACS_TRANSFER_N_ESTIMATORS,
        max_targets_per_fit=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
        target_bank=target_bank,
    )

    fit_records = (
        FitWeightRecord(US_PUF_SUPPORT_FIT_NAME, primary_weight_kind),
        *transferred.fit_records,
    )
    weights_audit = weights_audit_gate(fit_records)
    if not weights_audit.passed:
        raise ValueError(
            "Pool imputation weights audit failed:\n  "
            + "\n  ".join(weights_audit.failures)
        )

    qrf_manifest_path = _primary_qrf_manifest_path(primary_qrf_checkpoint_dir)
    qrf_binding_path = _primary_qrf_input_binding_path(primary_qrf_checkpoint_dir)
    target_bank_receipt = target_bank.receipt()
    if "identity_routing" in target_bank_receipt:
        raise ValueError(
            "ACS transfer target-bank receipt already has identity routing."
        )
    target_bank_receipt["identity_routing"] = acs_transfer_identity_routing
    return PoolStageOutput(
        transferred.frame,
        {
            "source_operator_chain": {
                "post_primary_completion": dict(source_completion.receipt),
            },
            "primary_puf_qrf": {
                "resume_status": qrf_resume_status,
                "identity_routing": primary_qrf_identity_routing,
                "checkpoint_manifest": _read_json_object(qrf_manifest_path),
                "checkpoint_manifest_sha256": _file_sha256(qrf_manifest_path),
                "input_binding": _read_json_object(qrf_binding_path),
                "input_binding_sha256": _file_sha256(qrf_binding_path),
                "n_estimators": _PRIMARY_QRF_N_ESTIMATORS,
                "tail_bound_diagnostics": tail_bound_diagnostics,
            },
            "puf_capital_gains_tail_transfer": tail_receipt,
            "acs_qrf_transfer": {
                "target_families": transfer_families,
                "n_estimators": _ACS_TRANSFER_N_ESTIMATORS,
                "max_targets_per_fit": DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
                "resolved_donor_channel": transferred.resolved_donor_channel,
                "imputed_inputs": list(transferred.imputed_inputs),
                "fit_records": list(transferred.fit_records),
                "deferred_inputs": list(transferred.deferred_inputs),
                "target_bank": target_bank_receipt,
            },
            "weights_audit": GateReport((weights_audit,)).to_manifest(),
        },
    )


def build_multispine_pool(
    asec: Frame | None,
    acs: Frame | None,
    *,
    puf_donor: pd.DataFrame | None,
    acs_rent_donor: pd.DataFrame | None = None,
    primary_qrf_checkpoint_dir: Path,
    acs_transfer_checkpoint_dir: Path | None = None,
    checkpoint_identity: Mapping[str, object] | None = None,
    checkpoint_input_binding: Mapping[str, object] | None = None,
    source_native_inputs: Mapping[
        str,
        Mapping[str, Mapping[str, Any]],
    ]
    | None = None,
    prepare_clone: PoolOperator | None = None,
    impute: PoolOperator | None = None,
    derive: PoolOperator = derive_multispine_pool_inputs,
    seed: PoolOperator = seed_multispine_pool_inputs,
    simulate: PoolOperator = materialize_multispine_agreement_outputs,
    checkpoint: Callable[[MultispinePoolCheckpoint], None] | None = None,
    resume: MultispinePoolCheckpoint | None = None,
) -> MultispinePoolResult:
    """Run the production pool path, with injectable operators for fixtures.

    Production callers omit all operator overrides. The terminal agreement
    gate is intentionally not injectable here, so this wiring can never alter
    its registry or fixed tolerances.
    """

    if resume is None:
        if not isinstance(asec, Frame) or not isinstance(acs, Frame):
            raise TypeError("Fresh pool builds require ASEC and ACS Frames.")
        native_inputs = source_native_inputs or {}
        assert_operator_free_source_frame(
            asec,
            label="ASEC raw-stage pool input",
            native_inputs=native_inputs.get("asec"),
        )
        assert_operator_free_source_frame(
            acs,
            label="ACS native-mapped pool input",
            native_inputs=native_inputs.get("acs"),
        )
    if impute is None:
        impute_will_run = resume is None or resume.stage == "assembled"
        clone_preparation_will_run = impute_will_run
        if impute_will_run and checkpoint_input_binding is None:
            raise ValueError(
                "Production pool imputation requires a verified checkpoint "
                "input binding."
            )
        if impute_will_run and (
            acs_transfer_checkpoint_dir is None or checkpoint_identity is None
        ):
            raise ValueError(
                "Production pool imputation requires an identity-bound ACS "
                "transfer checkpoint directory."
            )
        if impute_will_run and puf_donor is None:
            raise ValueError("Production pool imputation requires the PUF donor.")
        if clone_preparation_will_run and acs_rent_donor is None:
            raise ValueError(
                "Production pool imputation requires the canonical ACS rent donor."
            )

        def impute_operator(frame: Frame) -> PoolStageOutput:
            if (
                puf_donor is None
                or acs_transfer_checkpoint_dir is None
                or checkpoint_identity is None
                or checkpoint_input_binding is None
            ):
                raise AssertionError("Pool imputation dependencies were not loaded.")
            return _impute_pool(
                frame,
                puf_donor=puf_donor,
                primary_qrf_checkpoint_dir=primary_qrf_checkpoint_dir,
                acs_transfer_checkpoint_dir=acs_transfer_checkpoint_dir,
                checkpoint_identity=checkpoint_identity,
                checkpoint_input_binding=checkpoint_input_binding,
            )

        if prepare_clone is not None:
            prepare_clone_operator = prepare_clone
        elif clone_preparation_will_run:
            if acs_rent_donor is None:  # pragma: no cover - validated above
                raise AssertionError("ACS rent donor was not loaded.")

            def prepare_clone_operator(frame: Frame) -> PoolStageOutput:
                return prepare_multispine_source_inputs_for_clone(
                    frame,
                    acs_rent_donor=acs_rent_donor,
                )

        else:
            prepare_clone_operator = None

    else:
        impute_operator = impute
        prepare_clone_operator = prepare_clone
    return run_multispine_pool_path(
        asec,
        acs,
        prepare_clone=prepare_clone_operator,
        impute=impute_operator,
        derive=derive,
        seed=seed,
        simulate=simulate,
        checkpoint=checkpoint,
        resume=resume,
    )


def _stacked_direction_bank_identity(
    checkpoint_identity: Mapping[str, object],
    *,
    direction_name: str,
) -> dict[str, object]:
    return {
        **_pool_checkpoint_stage_identity(checkpoint_identity, "transferred"),
        "stacked_gap_fill_direction": direction_name,
    }


def _stacked_tail_manifest(
    stage_receipts: Mapping[str, Mapping[str, object]],
) -> Mapping[str, object]:
    impute = stage_receipts.get("impute")
    if not isinstance(impute, Mapping):
        raise ValueError("Stacked transferred receipts have no impute object.")
    manifest = impute.get("puf_capital_gains_tail_transfer")
    if not isinstance(manifest, Mapping):
        raise ValueError(
            "Stacked transferred receipts have no capital-gains-tail manifest."
        )
    validate_puf_capital_gains_tail_manifest(manifest)
    return manifest


def _emit_stacked_checkpoint(
    callback: Callable[[MultispinePoolCheckpoint], None] | None,
    *,
    stage: str,
    frame: Frame,
    assembly_receipt: Mapping[str, object],
    stage_receipts: Mapping[str, Mapping[str, object]],
    simulation_frame: Frame | None = None,
) -> None:
    if callback is None:
        return
    callback(
        MultispinePoolCheckpoint(
            stage=stage,
            frame=frame,
            assembly_receipt=dict(assembly_receipt),
            stage_receipts={
                name: dict(receipt) for name, receipt in stage_receipts.items()
            },
            simulation_frame=simulation_frame,
        )
    )


def build_stacked_pool(
    assembled: Frame,
    *,
    expected_stack_receipt: Mapping[str, object],
    release_id: str,
    puf_donor: pd.DataFrame | None,
    acs_rent_donor: pd.DataFrame | None,
    primary_qrf_checkpoint_dir: Path,
    acs_transfer_checkpoint_dir: Path,
    checkpoint_identity: Mapping[str, object],
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
    checkpoint: Callable[[MultispinePoolCheckpoint], None] | None = None,
    resume: MultispinePoolCheckpoint | None = None,
    phase_reached: Callable[[str], None] | None = None,
) -> StackedPoolBuildResult:
    """Run the fixed stacked production pipeline across #599 boundaries."""

    if not _STACKED_RELEASE_ID_PATTERN.fullmatch(release_id):
        raise ValueError(f"Invalid stacked release ID {release_id!r}.")

    def mark_phase(name: str) -> None:
        if phase_reached is not None:
            phase_reached(name)

    current_stack_receipt = validate_stacked_spine_frame(
        assembled,
        boundary="stacked pool fresh assembly",
    )
    if _json_ready(current_stack_receipt) != _json_ready(expected_stack_receipt):
        raise ValueError("Fresh stacked assembly receipt changed before execution.")

    if resume is None:
        current = canonicalize_frame_string_dtypes(
            assembled,
            boundary="stacked pool assembled checkpoint",
            in_place=True,
        )
        assembly_receipt = current.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
        receipts: dict[str, Mapping[str, object]] = {}
        resume_stage: str | None = None
        _emit_stacked_checkpoint(
            checkpoint,
            stage="assembled",
            frame=current,
            assembly_receipt=assembly_receipt,
            stage_receipts=receipts,
        )
        mark_phase("assembled")
    else:
        current = canonicalize_frame_string_dtypes(
            resume.frame,
            boundary=f"stacked pool {resume.stage} resume",
        )
        live_stack_receipt = validate_stacked_spine_frame(
            current,
            boundary=f"stacked pool {resume.stage} resume",
        )
        if _json_ready(live_stack_receipt) != _json_ready(expected_stack_receipt):
            raise ValueError(
                f"Stacked {resume.stage!r} checkpoint manifest differs from the "
                "freshly reconstructed stack identity."
            )
        assembly_receipt = current.metadata[SPINE_ASSEMBLY_MANIFEST_KEY]
        if _json_ready(assembly_receipt) != _json_ready(resume.assembly_receipt):
            raise ValueError(
                f"Stacked {resume.stage!r} checkpoint assembly receipt changed."
            )
        receipts = {
            name: dict(receipt) for name, receipt in resume.stage_receipts.items()
        }
        resume_stage = resume.stage
        for completed_phase in (
            "assembled",
            *(
                ("source_prepared", "gap_filled", "puf_passed", "transferred")
                if resume.stage in {"transferred", "simulated"}
                else ()
            ),
            *(
                ("derived", "seeded", "simulated")
                if resume.stage == "simulated"
                else ()
            ),
        ):
            mark_phase(completed_phase)

    if resume_stage in {None, "assembled"}:
        if not isinstance(puf_donor, pd.DataFrame):
            raise TypeError(
                "A cold or assembled-resume stacked build requires the full "
                "PUF donor DataFrame."
            )
        if not isinstance(acs_rent_donor, pd.DataFrame):
            raise TypeError(
                "A cold or assembled-resume stacked build requires the canonical "
                "ACS rent donor."
            )
        prepared = prepare_multispine_source_inputs_for_clone(
            current,
            acs_rent_donor=acs_rent_donor,
        )
        validate_stacked_spine_frame(
            prepared.frame,
            boundary="stacked pool source preparation output",
        )
        mark_phase("source_prepared")

        current_base_identity_sha256 = _pool_checkpoint_identity_sha256(
            checkpoint_identity
        )
        gap_fill_identity_routing = _identity_routed_bank_open_receipt(
            acs_transfer_checkpoint_dir,
            current_base_identity_sha256=current_base_identity_sha256,
        )
        target_banks: dict[str, AcsTransferTargetBankStore] = {}
        for direction in stacked_gap_fill_plan():
            target_banks[direction.name] = AcsTransferTargetBankStore(
                acs_transfer_checkpoint_dir / direction.name,
                identity=_stacked_direction_bank_identity(
                    checkpoint_identity,
                    direction_name=direction.name,
                ),
            )
        gap_filled = gap_fill_stacked_spine(
            prepared.frame,
            seed=POOL_RANDOM_SEED,
            n_estimators=_ACS_TRANSFER_N_ESTIMATORS,
            max_targets_per_fit=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
            target_banks=target_banks,
        )
        mark_phase("gap_filled")

        fit_records: list[FitWeightRecord] = []
        for transfer in gap_filled.transfer_results.values():
            fit_records.extend(transfer.fit_records)
        tail_bound_diagnostics: list[dict[str, object]] = []
        primary_qrf_identity_routing = _identity_routed_bank_open_receipt(
            primary_qrf_checkpoint_dir,
            current_base_identity_sha256=current_base_identity_sha256,
        )
        puf_result = run_stacked_puf_pass(
            gap_filled.frame,
            puf_donor,
            clone_attachment_fraction=clone_attachment_fraction,
            clone_attachment_seed=clone_attachment_seed,
            seed=POOL_RANDOM_SEED,
            n_estimators=_PRIMARY_QRF_N_ESTIMATORS,
            fit_records=fit_records,
            tail_bound_diagnostics=tail_bound_diagnostics,
            primary_qrf_checkpoint_dir=primary_qrf_checkpoint_dir,
        )
        mark_phase("puf_passed")
        weights_audit = weights_audit_gate(fit_records)
        if not weights_audit.passed:
            raise ValueError(
                "Stacked imputation weights audit failed:\n  "
                + "\n  ".join(weights_audit.failures)
            )

        puf_receipt = dict(puf_result.receipt)
        primary_qrf_receipt = puf_receipt.pop("primary_puf_qrf")
        if not isinstance(primary_qrf_receipt, Mapping):
            raise ValueError("Stacked PUF pass emitted no primary-QRF receipt.")
        primary_qrf_receipt = dict(primary_qrf_receipt)
        primary_qrf_receipt["identity_routing"] = primary_qrf_identity_routing
        qrf_manifest_path = _primary_qrf_manifest_path(primary_qrf_checkpoint_dir)
        reported_manifest_path = primary_qrf_receipt.pop(
            "checkpoint_manifest",
            None,
        )
        if (
            reported_manifest_path is not None
            and Path(reported_manifest_path).resolve() != qrf_manifest_path.resolve()
        ):
            raise ValueError(
                "Stacked PUF pass reported a different primary-QRF manifest path."
            )
        primary_qrf_receipt.update(
            {
                "checkpoint_manifest_path": str(qrf_manifest_path.resolve()),
                "checkpoint_manifest_receipt": _read_json_object(qrf_manifest_path),
                "checkpoint_manifest_sha256": _file_sha256(qrf_manifest_path),
                "n_estimators": _PRIMARY_QRF_N_ESTIMATORS,
                "tail_bound_diagnostics": tail_bound_diagnostics,
            }
        )
        tail_manifest = puf_receipt.pop("puf_capital_gains_tail_transfer")
        if not isinstance(tail_manifest, Mapping):
            raise ValueError("Stacked PUF pass emitted no tail manifest.")
        validate_puf_capital_gains_tail_manifest(tail_manifest)

        source_completion = complete_multispine_source_inputs(puf_result.frame)
        completion_preservation = assert_stacked_tail_cells_preserved(
            source_completion.frame,
            tail_manifest,
        )
        current = canonicalize_frame_string_dtypes(
            source_completion.frame,
            boundary="stacked pool transferred checkpoint",
            in_place=True,
        )
        validate_stacked_spine_frame(
            current,
            boundary="stacked pool transferred checkpoint",
        )
        receipts["impute"] = {
            "source_operator_chain": {
                "pre_gap_fill_preparation": dict(prepared.receipt),
                "post_primary_completion": dict(source_completion.receipt),
            },
            "stacked_gap_fill": dict(gap_filled.receipt),
            "primary_puf_qrf": primary_qrf_receipt,
            "puf_capital_gains_tail_transfer": dict(tail_manifest),
            "stacked_puf_pass": puf_receipt,
            "tail_preservation_after_source_completion": completion_preservation,
            "acs_qrf_transfer": {
                "target_families": _json_ready(pool_transfer_target_families()),
                "n_estimators": _ACS_TRANSFER_N_ESTIMATORS,
                "max_targets_per_fit": DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
                "target_bank": {
                    "identity_routing": gap_fill_identity_routing,
                    "directions": {
                        name: bank.receipt()
                        for name, bank in sorted(target_banks.items())
                    },
                },
            },
            "weights_audit": GateReport((weights_audit,)).to_manifest(),
        }
        _emit_stacked_checkpoint(
            checkpoint,
            stage="transferred",
            frame=current,
            assembly_receipt=assembly_receipt,
            stage_receipts=receipts,
        )
        mark_phase("transferred")
    else:
        validate_stacked_spine_frame(
            current,
            boundary=f"stacked pool {resume_stage} persistent resume",
        )

    tail_manifest = _stacked_tail_manifest(receipts)
    if resume_stage != "simulated":
        derivation_input, tail_derivation_receipt = prepare_stacked_tail_derivation(
            current
        )
        derived = derive_multispine_pool_inputs(derivation_input)
        current = canonicalize_frame_string_dtypes(
            derived.frame,
            boundary="stacked pool derive output",
            in_place=True,
        )
        validate_stacked_spine_frame(current, boundary="stacked pool derive output")
        receipts["derive"] = {
            "tail_derivation_preparation": tail_derivation_receipt,
            "pool_derivation": dict(derived.receipt),
            "tail_preservation": assert_stacked_tail_cells_preserved(
                current,
                tail_manifest,
            ),
        }
        mark_phase("derived")

        seeded = seed_multispine_pool_inputs(current)
        current = canonicalize_frame_string_dtypes(
            seeded.frame,
            boundary="stacked pool seed output",
            in_place=True,
        )
        validate_stacked_spine_frame(current, boundary="stacked pool seed output")
        receipts["seed"] = {
            **dict(seeded.receipt),
            "tail_preservation": assert_stacked_tail_cells_preserved(
                current,
                tail_manifest,
            ),
        }
        mark_phase("seeded")

        simulated = materialize_multispine_agreement_outputs(current)
        simulation_frame = canonicalize_frame_string_dtypes(
            simulated.frame,
            boundary="stacked pool simulated checkpoint",
            in_place=True,
        )
        validate_stacked_spine_frame(
            simulation_frame,
            boundary="stacked pool simulation output",
        )
        receipts["simulate"] = {
            **dict(simulated.receipt),
            "tail_preservation": assert_stacked_tail_cells_preserved(
                simulation_frame,
                tail_manifest,
            ),
        }
        _emit_stacked_checkpoint(
            checkpoint,
            stage="simulated",
            frame=current,
            assembly_receipt=assembly_receipt,
            stage_receipts=receipts,
            simulation_frame=simulation_frame,
        )
        mark_phase("simulated")
    else:
        if resume is None or resume.simulation_frame is None:
            raise ValueError("Simulated stacked resume has no evaluation frame.")
        simulation_frame = canonicalize_frame_string_dtypes(
            resume.simulation_frame,
            boundary="stacked pool simulated evaluation resume",
        )
        validate_stacked_spine_frame(
            simulation_frame,
            boundary="stacked pool simulated evaluation resume",
        )
        assert_stacked_tail_cells_preserved(simulation_frame, tail_manifest)

    completeness = stacked_completeness_gate(simulation_frame)
    battery = by_origin_battery(simulation_frame)
    # Manifest conversion is itself the final canonical-authority check and
    # deliberately happens before publication or readiness is asserted.
    GateReport((completeness, battery)).to_manifest()
    mark_phase("terminal_gates")
    counts = spine_provenance_counts(
        current,
        boundary="stacked pool terminal input-only output",
    )
    return StackedPoolBuildResult(
        frame=current,
        stack_receipt=dict(expected_stack_receipt),
        assembly_receipt=dict(assembly_receipt),
        provenance_counts=counts,
        stage_receipts=receipts,
        terminal_gates=(completeness, battery),
        release_id=release_id,
    )


def _agreement_payload(result: MultispinePoolResult) -> dict[str, object]:
    return GateReport((result.agreement_gate,)).to_manifest()


def _manifest_payload(
    *,
    result: MultispinePoolResult,
    outputs: PoolBuildOutputs,
    verified_inputs: Mapping[str, _VerifiedInput],
    acs_source_manifest: AcsSourceManifest,
    input_receipts: Mapping[str, object],
    checkpoint_provenance: Mapping[str, object],
    publication_run_id: str,
) -> dict[str, object]:
    status = "simulation_ready" if result.simulation_ready else "agreement_failed"
    puf_donor_receipt = input_receipts.get("puf_donor")
    if not isinstance(puf_donor_receipt, Mapping):
        raise ValueError("Pool input receipts have no PUF donor object.")
    return {
        "artifact_kind": "populace_us_multispine_pool_manifest",
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "status": status,
        "simulation_ready": result.simulation_ready,
        "publication_run_id": publication_run_id,
        "calibration_applied": False,
        "operator_order": list(POOL_OPERATOR_ORDER),
        "period": POOL_TIME_PERIOD,
        "random_seed": POOL_RANDOM_SEED,
        "provenance_pins": {
            role: pin.to_manifest() for role, pin in verified_inputs.items()
        },
        "asec_raw_stage_checkpoint": input_receipts.get("asec_raw_stage_checkpoint"),
        "acs_source_manifest": asdict(acs_source_manifest),
        "acs_pums_build": input_receipts.get("acs_pums_build"),
        "acs_native_inputs": input_receipts.get("acs_native_inputs"),
        "puf_donor": dict(puf_donor_receipt),
        "assembly_receipt": result.assembly_receipt,
        "assembly_contract": {
            "household_mass_shares": dict(POOL_HOUSEHOLD_MASS_SHARES),
            "clone_safe_source_id_upper_bound": (PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID),
            "output_household_weight_kind": result.frame.weights_for(
                "household"
            ).kind.value,
            "output_household_weight_total": result.frame.weights_for(
                "household"
            ).total,
            "mass_log": list(result.frame.mass_log),
        },
        "provenance_counts": result.provenance_counts,
        "stage_receipts": result.stage_receipts,
        "stage_checkpoints": checkpoint_provenance,
        "agreement_gate": _agreement_payload(result),
        "pool_h5": {
            "path": str(outputs.pool_h5.resolve()),
            "sha256": _file_sha256(outputs.pool_h5),
            "size_bytes": outputs.pool_h5.stat().st_size,
            "artifact_kind": POOL_H5_ARTIFACT_KIND,
            "publication_run_id": publication_run_id,
            "nullable": True,
            "input_only": True,
            "formula_outputs_persisted": False,
        },
        "agreement_diagnostics": {
            "path": str(outputs.agreement_diagnostics.resolve()),
            "sha256": _file_sha256(outputs.agreement_diagnostics),
            "size_bytes": outputs.agreement_diagnostics.stat().st_size,
            "publication_run_id": publication_run_id,
        },
        "primary_qrf_checkpoint_dir": str(outputs.primary_qrf_checkpoint_dir.resolve()),
        "acs_transfer_checkpoint_dir": str(
            outputs.acs_transfer_checkpoint_dir.resolve()
        ),
        "calibration": {
            "applied": False,
            "position": "downstream",
            "consumer": "k-ladder",
            "requires_manifest_simulation_ready": True,
        },
    }


def _stacked_gate_payload(result: StackedPoolBuildResult) -> dict[str, object]:
    return GateReport(result.terminal_gates).to_manifest()


def _stacked_manifest_payload(
    *,
    result: StackedPoolBuildResult,
    outputs: PoolBuildOutputs,
    verified_inputs: Mapping[str, _VerifiedInput],
    acs_source_manifest: AcsSourceManifest,
    input_receipts: Mapping[str, object],
    checkpoint_provenance: Mapping[str, object],
    publication_run_id: str,
    sample_fraction: float,
    sample_seed: int,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
) -> dict[str, object]:
    """Build the stacked-only manifest without changing the legacy envelope."""

    status = "simulation_ready" if result.simulation_ready else "gate_failed"
    puf_donor_receipt = input_receipts.get("puf_donor")
    if not isinstance(puf_donor_receipt, Mapping):
        raise ValueError("Stacked pool input receipts have no PUF donor object.")
    gates = _stacked_gate_payload(result)
    stack_manifest = _json_ready(result.stack_receipt)
    return {
        "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "pipeline": _STACKED_PIPELINE,
        "release_id": result.release_id,
        "status": status,
        "simulation_ready": result.simulation_ready,
        "publication_run_id": publication_run_id,
        "calibration_applied": False,
        "operator_order": [
            "assemble_stacked_spine",
            "prepare_multispine_source_inputs_for_clone",
            "gap_fill_stacked_spine",
            "run_stacked_puf_pass",
            "complete_multispine_source_inputs",
            "prepare_stacked_tail_derivation",
            "derive_multispine_pool_inputs",
            "seed_multispine_pool_inputs",
            "materialize_multispine_agreement_outputs",
            "stacked_completeness_gate",
            "by_origin_battery",
        ],
        "period": POOL_TIME_PERIOD,
        "random_seed": POOL_RANDOM_SEED,
        "sampling": {
            "sample_fraction": float(sample_fraction),
            "fraction_token": _stacked_rung(sample_fraction)[0],
            "sample_seed": sample_seed,
            "realized_households": {
                channel: result.stack_receipt["survey_samples"][channel][
                    "realized_household_count"
                ]
                for channel in ("asec", "acs")
            },
            "stack_manifest_sha256": hashlib.sha256(
                _canonical_json_bytes(stack_manifest)
            ).hexdigest(),
        },
        "clone_attachment": {
            "fraction": float(clone_attachment_fraction),
            "seed": clone_attachment_seed,
        },
        "provenance_pins": {
            role: pin.to_manifest() for role, pin in verified_inputs.items()
        },
        "input_pins_digest": _input_pins_digest(verified_inputs),
        "asec_raw_stage_checkpoint": input_receipts.get("asec_raw_stage_checkpoint"),
        "acs_source_manifest": asdict(acs_source_manifest),
        "acs_pums_build": input_receipts.get("acs_pums_build"),
        "acs_native_inputs": input_receipts.get("acs_native_inputs"),
        "puf_donor": dict(puf_donor_receipt),
        "stack_manifest": stack_manifest,
        "assembly_receipt": result.assembly_receipt,
        "assembly_contract": {
            "household_mass_shares": result.stack_receipt["household_mass_shares"],
            "clone_safe_source_id_upper_bound": PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
            "output_household_weight_kind": result.frame.weights_for(
                "household"
            ).kind.value,
            "output_household_weight_total": result.frame.weights_for(
                "household"
            ).total,
            "mass_log": list(result.frame.mass_log),
        },
        "provenance_counts": result.provenance_counts,
        "stage_receipts": result.stage_receipts,
        "stage_checkpoints": checkpoint_provenance,
        "terminal_gates": gates,
        # Compatibility alias for existing simulation-ready manifest readers.
        # Its contents are the stacked terminal battery, never us_spine_agreement.
        "agreement_gate": gates,
        "pool_h5": {
            "path": str(outputs.pool_h5.resolve()),
            "sha256": _file_sha256(outputs.pool_h5),
            "size_bytes": outputs.pool_h5.stat().st_size,
            "artifact_kind": POOL_H5_ARTIFACT_KIND,
            "publication_run_id": publication_run_id,
            "nullable": True,
            "input_only": True,
            "formula_outputs_persisted": False,
        },
        "agreement_diagnostics": {
            "path": str(outputs.agreement_diagnostics.resolve()),
            "sha256": _file_sha256(outputs.agreement_diagnostics),
            "size_bytes": outputs.agreement_diagnostics.stat().st_size,
            "publication_run_id": publication_run_id,
            "semantic_kind": "stacked_terminal_gates",
        },
        "primary_qrf_checkpoint_dir": str(outputs.primary_qrf_checkpoint_dir.resolve()),
        "acs_transfer_checkpoint_dir": str(
            outputs.acs_transfer_checkpoint_dir.resolve()
        ),
        "calibration": {
            "applied": False,
            "position": "downstream",
            "consumer": "k-ladder",
            "requires_manifest_simulation_ready": True,
        },
    }


def _stacked_publication_tombstone(
    outputs: PoolBuildOutputs,
    *,
    release_id: str,
    publication_run_id: str,
) -> dict[str, object]:
    return {
        "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "pipeline": _STACKED_PIPELINE,
        "release_id": release_id,
        "status": "publication_in_progress",
        "simulation_ready": False,
        "publication_run_id": publication_run_id,
        "message": "stacked publication in progress",
        "pool_h5": {
            "path": str(outputs.pool_h5.resolve()),
            "artifact_kind": POOL_H5_ARTIFACT_KIND,
            "publication_run_id": publication_run_id,
        },
        "agreement_diagnostics": {
            "path": str(outputs.agreement_diagnostics.resolve()),
            "publication_run_id": publication_run_id,
            "semantic_kind": "stacked_terminal_gates",
        },
    }


def _publication_tombstone(
    outputs: PoolBuildOutputs,
    *,
    publication_run_id: str,
) -> dict[str, object]:
    return {
        "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "status": "publication_in_progress",
        "simulation_ready": False,
        "publication_run_id": publication_run_id,
        "message": "publication in progress",
        "pool_h5": {
            "path": str(outputs.pool_h5.resolve()),
            "artifact_kind": POOL_H5_ARTIFACT_KIND,
            "publication_run_id": publication_run_id,
        },
        "agreement_diagnostics": {
            "path": str(outputs.agreement_diagnostics.resolve()),
            "publication_run_id": publication_run_id,
        },
    }


def _new_publication_run_id() -> str:
    return uuid.uuid4().hex


def _publication_temporary_path(path: Path, *, publication_run_id: str) -> Path:
    output = Path(path)
    return output.with_name(f".{output.name}.{publication_run_id}.publication.tmp")


def _write_outputs(
    result: MultispinePoolResult,
    *,
    outputs: PoolBuildOutputs,
    verified_inputs: Mapping[str, _VerifiedInput],
    acs_source_manifest: AcsSourceManifest,
    loaded: _LoadedInputs | None = None,
    input_receipts: Mapping[str, object] | None = None,
    checkpoint_provenance: Mapping[str, object] | None = None,
) -> None:
    if input_receipts is None:
        if loaded is None:
            raise ValueError(
                "Pool publication requires loaded inputs or checkpointed receipts."
            )
        input_receipts = _loaded_input_receipts(loaded)
    if checkpoint_provenance is None:
        checkpoint_provenance = {
            "artifact_kind": ("populace_us_multispine_pool_checkpoint_provenance"),
            "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
            "materializer_version": POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
            "enabled": False,
            "agreement": {
                "source": "always_fresh",
                "cached": False,
                "terminal_verdict_persisted": False,
            },
        }
    publication_run_id = _new_publication_run_id()
    _atomic_write_json(
        outputs.manifest,
        _publication_tombstone(
            outputs,
            publication_run_id=publication_run_id,
        ),
    )
    temporary_h5 = _publication_temporary_path(
        outputs.pool_h5,
        publication_run_id=publication_run_id,
    )
    temporary_diagnostics = _publication_temporary_path(
        outputs.agreement_diagnostics,
        publication_run_id=publication_run_id,
    )
    diagnostics = {
        "artifact_kind": US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "simulation_ready": result.simulation_ready,
        "publication_run_id": publication_run_id,
        "agreement_gate": _agreement_payload(result),
    }
    try:
        write_nullable_us_h5(
            result.frame,
            temporary_h5,
            period=POOL_TIME_PERIOD,
            artifact_kind=POOL_H5_ARTIFACT_KIND,
            publication_run_id=publication_run_id,
        )
        _atomic_write_json(temporary_diagnostics, diagnostics)
        os.replace(temporary_h5, outputs.pool_h5)
        os.replace(temporary_diagnostics, outputs.agreement_diagnostics)
        manifest = _manifest_payload(
            result=result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=acs_source_manifest,
            input_receipts=input_receipts,
            checkpoint_provenance=checkpoint_provenance,
            publication_run_id=publication_run_id,
        )
        _atomic_write_json(outputs.manifest, manifest)
    finally:
        temporary_h5.unlink(missing_ok=True)
        temporary_diagnostics.unlink(missing_ok=True)


def _write_stacked_outputs(
    result: StackedPoolBuildResult,
    *,
    outputs: PoolBuildOutputs,
    verified_inputs: Mapping[str, _VerifiedInput],
    acs_source_manifest: AcsSourceManifest,
    input_receipts: Mapping[str, object],
    checkpoint_provenance: Mapping[str, object],
    sample_fraction: float,
    sample_seed: int,
    clone_attachment_fraction: float,
    clone_attachment_seed: int,
) -> dict[str, object]:
    """Atomically publish the stacked input-only pool and terminal receipts."""

    publication_run_id = _new_publication_run_id()
    _atomic_write_json(
        outputs.manifest,
        _stacked_publication_tombstone(
            outputs,
            release_id=result.release_id,
            publication_run_id=publication_run_id,
        ),
    )
    temporary_h5 = _publication_temporary_path(
        outputs.pool_h5,
        publication_run_id=publication_run_id,
    )
    temporary_diagnostics = _publication_temporary_path(
        outputs.agreement_diagnostics,
        publication_run_id=publication_run_id,
    )
    gates = _stacked_gate_payload(result)
    diagnostics = {
        "artifact_kind": US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "pipeline": _STACKED_PIPELINE,
        "semantic_kind": "stacked_terminal_gates",
        "release_id": result.release_id,
        "simulation_ready": result.simulation_ready,
        "publication_run_id": publication_run_id,
        "terminal_gates": gates,
        "agreement_gate": gates,
    }
    try:
        write_nullable_us_h5(
            result.frame,
            temporary_h5,
            period=POOL_TIME_PERIOD,
            artifact_kind=POOL_H5_ARTIFACT_KIND,
            publication_run_id=publication_run_id,
        )
        _atomic_write_json(temporary_diagnostics, diagnostics)
        os.replace(temporary_h5, outputs.pool_h5)
        os.replace(temporary_diagnostics, outputs.agreement_diagnostics)
        manifest = _stacked_manifest_payload(
            result=result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=acs_source_manifest,
            input_receipts=input_receipts,
            checkpoint_provenance=checkpoint_provenance,
            publication_run_id=publication_run_id,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
            clone_attachment_fraction=clone_attachment_fraction,
            clone_attachment_seed=clone_attachment_seed,
        )
        _atomic_write_json(outputs.manifest, manifest)
        return manifest
    finally:
        temporary_h5.unlink(missing_ok=True)
        temporary_diagnostics.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path) -> dict[str, object]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object at {path}.")
    return raw


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Build receipt mappings must use string JSON keys.")
        return {key: _json_ready(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_json_ready(item) for item in value]
        return sorted(
            normalized,
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_ready(value.value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Build receipts must not contain non-finite JSON numbers.")
    if value is pd.NA:
        return None
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    _json_ready(payload),
                    allow_nan=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
        _fsync_parent_directory(output)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_parent_directory(path: Path) -> None:
    """Persist a completed atomic rename in its containing directory."""

    # O_DIRECTORY is not universal. A read-only directory descriptor is the
    # supported POSIX fallback; failures to open or fsync still propagate.
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path).parent, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _main_legacy(args: argparse.Namespace) -> int:
    """Run the retiring two-spine implementation without semantic changes."""

    outputs = _output_paths(
        args.out,
        checkpoint_root=args.checkpoint_root,
    )
    verified_inputs, acs_source_manifest = _verify_inputs(args, outputs)
    checkpoint_store = _PoolStageCheckpointStore(
        outputs.checkpoint_root,
        base_identity=_pool_checkpoint_base_identity(verified_inputs),
    )
    outputs = _with_checkpoint_identity(
        outputs,
        base_identity_sha256=checkpoint_store.base_identity_sha256,
    )
    resume = checkpoint_store.load_deepest()

    loaded: _LoadedInputs | None = None
    asec: Frame | None = None
    acs: Frame | None = None
    acs_rent_donor: pd.DataFrame | None = None
    puf_donor: pd.DataFrame | None = None
    source_native_inputs: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None
    if resume is None:
        loaded = _load_inputs(args, acs_source_manifest=acs_source_manifest)
        checkpoint_store.bind_input_receipts(_loaded_input_receipts(loaded))
        asec = loaded.asec
        acs = loaded.acs
        acs_rent_donor = loaded.acs_rent_donor
        puf_donor = loaded.puf_donor
        source_native_inputs = {"acs": loaded.acs_native_inputs}
    elif resume.stage == "assembled":
        acs_rent_donor = load_acs_2022_rent_donor(args.acs_rent_h5)
        puf_donor, _puf_build = _load_puf_donor(args)
        _validate_resumed_puf_donor(
            puf_donor,
            checkpoint_store.input_receipts,
        )

    result = build_multispine_pool(
        asec,
        acs,
        puf_donor=puf_donor,
        acs_rent_donor=acs_rent_donor,
        primary_qrf_checkpoint_dir=outputs.primary_qrf_checkpoint_dir,
        acs_transfer_checkpoint_dir=outputs.acs_transfer_checkpoint_dir,
        checkpoint_identity=checkpoint_store.base_identity,
        checkpoint_input_binding=_checkpoint_input_binding(verified_inputs),
        source_native_inputs=source_native_inputs,
        checkpoint=checkpoint_store.write,
        resume=resume,
    )
    checkpoint_provenance = checkpoint_store.provenance(
        primary_qrf_checkpoint_dir=outputs.primary_qrf_checkpoint_dir,
        acs_transfer_checkpoint_dir=outputs.acs_transfer_checkpoint_dir,
    )
    _write_outputs(
        result,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=acs_source_manifest,
        input_receipts=checkpoint_store.input_receipts,
        checkpoint_provenance=checkpoint_provenance,
    )
    if not result.simulation_ready:
        print(
            "US multispine agreement failed; diagnostics and a non-ready "
            f"manifest were written to {outputs.agreement_diagnostics} and "
            f"{outputs.manifest}.",
        )
        return 1
    print(f"Wrote simulation-ready multispine pool: {outputs.pool_h5}")
    print(f"Wrote pool manifest: {outputs.manifest}")
    return 0


def _git_code_pin() -> str:
    """Resolve the exact local commit without consulting the network."""

    repository = Path(__file__).resolve().parents[1]
    try:
        pin = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Could not resolve the local git code pin.") from exc
    if not re.fullmatch(r"[0-9a-f]{40}", pin):
        raise ValueError(f"Local git code pin is malformed: {pin!r}.")
    return pin


def _chronicle_predecessor(args: argparse.Namespace) -> str | None:
    cli_value = args.chronicle_prev_row_digest
    environment_value = os.environ.get("POPULACE_CHRONICLE_PREV_ROW_DIGEST")
    if cli_value is not None and environment_value not in {None, cli_value}:
        raise ValueError(
            "--chronicle-prev-row-digest disagrees with "
            "POPULACE_CHRONICLE_PREV_ROW_DIGEST."
        )
    value = cli_value if cli_value is not None else environment_value
    if value is not None and not _LOWERCASE_SHA256.fullmatch(value):
        raise ValueError(
            "POPULACE_CHRONICLE_PREV_ROW_DIGEST must be a lowercase SHA-256."
        )
    return value


def _append_phase(state: _StackedAttemptState, phase: str) -> None:
    if phase not in state.phases_reached:
        state.phases_reached.append(phase)


def _record_stacked_terminal_attempt(
    *,
    state: _StackedAttemptState,
    outputs: PoolBuildOutputs,
    started_at: float,
    started_ts: datetime,
    code_pin: str,
    rung: str,
    seed: int | None,
    disposition: str,
    predecessor: str | None,
) -> Path:
    result = record_build_attempt(
        build_id=state.build_id,
        ts=started_ts,
        pipeline=_STACKED_PIPELINE,
        rung=rung,
        seed=seed,
        code_pin=code_pin,
        input_pins_digest=state.input_pins_digest,
        identity_digest=state.identity_digest,
        phases_reached=state.phases_reached,
        gate_verdicts=state.gate_verdicts,
        wall_seconds=time.perf_counter() - started_at,
        cost_usd=None,
        artifact_location=state.artifact_location,
        disposition=disposition,
        prediction_id=None,
        prev_row_digest=predecessor,
        spool_dir=outputs.pool_h5.parent / "ledger-spool",
    )
    return result.spool_path


def _stacked_attempt_outputs(args: argparse.Namespace) -> PoolBuildOutputs:
    """Construct safe terminal-spool paths before validating ``--out``."""

    pool_h5 = Path(args.out)
    fallback_name = pool_h5.name or "stacked-pool-output"
    checkpoint_root = (
        Path(args.checkpoint_root)
        if args.checkpoint_root is not None
        else pool_h5.parent / f"{fallback_name}.checkpoints"
    )
    return PoolBuildOutputs(
        pool_h5=pool_h5,
        manifest=pool_h5.parent / f"{fallback_name}.manifest.json",
        agreement_diagnostics=pool_h5.parent / f"{fallback_name}.gates.json",
        checkpoint_root=checkpoint_root,
        primary_qrf_checkpoint_dir=checkpoint_root / "primary-qrf",
        acs_transfer_checkpoint_dir=checkpoint_root / "acs-transfer",
    )


def _stacked_attempt_receipt_dir(
    outputs: PoolBuildOutputs,
    *,
    build_id: str,
) -> Path:
    """Return the immutable, build-scoped receipt directory beside output."""

    return outputs.pool_h5.parent / "ledger-receipts" / build_id


def _stacked_error_receipt_path(
    outputs: PoolBuildOutputs,
    *,
    build_id: str,
) -> Path:
    return _stacked_attempt_receipt_dir(outputs, build_id=build_id) / "error.json"


def _stacked_terminal_gate_receipt_path(
    outputs: PoolBuildOutputs,
    *,
    build_id: str,
) -> Path:
    return (
        _stacked_attempt_receipt_dir(outputs, build_id=build_id) / "terminal-gates.json"
    )


def _write_stacked_terminal_gate_receipt(
    result: StackedPoolBuildResult,
    *,
    outputs: PoolBuildOutputs,
) -> Path:
    """Persist the attempt's immutable terminal-gate receipt before publish."""

    path = _stacked_terminal_gate_receipt_path(
        outputs,
        build_id=result.release_id,
    )
    _atomic_write_json(
        path,
        {
            "artifact_kind": "populace_us_stacked_terminal_gate_receipt",
            "schema_version": 1,
            "pipeline": _STACKED_PIPELINE,
            "build_id": result.release_id,
            "terminal_gates": _stacked_gate_payload(result),
        },
    )
    return path


def _validate_stacked_seed(value: int, *, option: str) -> int:
    """Keep build RNGs and Chronicle inside the signed-64-bit contract."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 2**63 - 1
    ):
        raise ValueError(f"{option} must be a non-negative signed 64-bit integer.")
    return value


def _new_stacked_attempt_id(*, timestamp: datetime) -> str:
    """Name a preflight attempt before its rung and realized counts validate."""

    instant = timestamp.astimezone(UTC)
    return (
        "populace-us-2024-stacked-attempt-"
        f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )


def _main_stacked(args: argparse.Namespace) -> int:
    """Build the default stacked pipeline and emit one terminal Chronicle row."""

    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    outputs = _stacked_attempt_outputs(args)
    code_pin = "unresolved-local-git-code-pin"
    predecessor: str | None = None
    rung = "1"
    chronicle_seed: int | None = None
    preflight_digest = hashlib.sha256(
        _canonical_json_bytes(
            {
                "pipeline": _STACKED_PIPELINE,
                "state": "preflight",
            }
        )
    ).hexdigest()
    state = _StackedAttemptState(
        build_id=_new_stacked_attempt_id(timestamp=started_ts),
        identity_digest=preflight_digest,
        input_pins_digest=preflight_digest,
        phases_reached=["attempt_started"],
        gate_verdicts={
            "pipeline": {
                "verdict": "running",
                "receipt": "pending-build-scoped-terminal-receipt",
            }
        },
    )

    try:
        state.input_pins_digest = _configured_input_pins_digest(args)
        chronicle_seed = _validate_stacked_seed(
            args.sample_seed,
            option="--sample-seed",
        )
        _validate_stacked_seed(
            args.clone_attachment_seed,
            option="--clone-attachment-seed",
        )
        fraction_token, rung = _stacked_rung(args.sample_fraction)
        del fraction_token  # The release-ID helper revalidates the same rung.
        outputs = _stacked_output_paths(
            args.out,
            checkpoint_root=args.checkpoint_root,
        )
        state.build_id = _new_stacked_release_id(
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
            realized_asec_households=0,
            realized_acs_households=0,
            timestamp=started_ts,
        )
        configured_identity = _configured_stacked_identity(args)
        state.identity_digest = hashlib.sha256(
            _canonical_json_bytes(configured_identity)
        ).hexdigest()
        _append_phase(state, "configured")
        code_pin = _git_code_pin()
        predecessor = _chronicle_predecessor(args)
        verified_inputs, acs_source_manifest = _verify_inputs(args, outputs)
        state.input_pins_digest = _input_pins_digest(verified_inputs)
        _append_phase(state, "inputs_verified")
        stacked_checkpoint_root = _stacked_checkpoint_root(
            outputs,
            configured_identity,
        )
        outputs = replace(
            outputs,
            checkpoint_root=stacked_checkpoint_root,
            primary_qrf_checkpoint_dir=stacked_checkpoint_root / "primary-qrf",
            acs_transfer_checkpoint_dir=stacked_checkpoint_root / "acs-transfer",
        )
        _validate_checkpoint_path_layout(
            outputs,
            source_paths=_configured_source_paths(args),
        )
        checkpoint_identity = _discover_stacked_checkpoint_identity(
            outputs.checkpoint_root,
            verified_inputs=verified_inputs,
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
            clone_attachment_fraction=args.clone_attachment_fraction,
            clone_attachment_seed=args.clone_attachment_seed,
        )
        checkpoint_store: _PoolStageCheckpointStore | None = None
        resume: MultispinePoolCheckpoint | None = None
        stack_frame: Frame | None = None
        stack_receipt: Mapping[str, object] | None = None
        puf_donor: pd.DataFrame | None = None
        acs_rent_donor: pd.DataFrame | None = None
        if checkpoint_identity is not None:
            checkpoint_store = _PoolStageCheckpointStore(
                outputs.checkpoint_root,
                base_identity=checkpoint_identity,
            )
            outputs = _with_checkpoint_identity(
                outputs,
                base_identity_sha256=checkpoint_store.base_identity_sha256,
            )
            resume = checkpoint_store.load_deepest()
            if resume is not None:
                sampling = checkpoint_identity.get("sampling")
                if not isinstance(sampling, Mapping):  # pragma: no cover
                    raise ValueError("Discovered stacked identity lost sampling.")
                discovered_receipt = sampling.get("stack_manifest")
                if not isinstance(discovered_receipt, Mapping):  # pragma: no cover
                    raise ValueError("Discovered stacked identity lost its manifest.")
                stack_receipt = dict(discovered_receipt)
                stack_frame = resume.frame
                _append_phase(state, "checkpoint_loaded")
                if resume.stage == "assembled":
                    acs_rent_donor = load_acs_2022_rent_donor(args.acs_rent_h5)
                    puf_donor, _puf_donor_build = _load_puf_donor(args)
                    _validate_resumed_puf_donor(
                        puf_donor,
                        checkpoint_store.input_receipts,
                    )
                    _append_phase(state, "resume_donors_loaded")

        if resume is None:
            loaded = _load_inputs(args, acs_source_manifest=acs_source_manifest)
            _append_phase(state, "sources_loaded")
            assert_operator_free_source_frame(
                loaded.asec,
                label="ASEC raw-stage stacked input",
            )
            assert_operator_free_source_frame(
                loaded.acs,
                label="ACS native-mapped stacked input",
                native_inputs=loaded.acs_native_inputs,
            )
            stack = assemble_stacked_spine(
                loaded.asec,
                loaded.acs,
                sample_fraction=args.sample_fraction,
                sample_seed=args.sample_seed,
            )
            stack_frame = stack.frame
            stack_receipt = stack.receipt
            puf_donor = loaded.puf_donor
            acs_rent_donor = loaded.acs_rent_donor
            checkpoint_identity = _stacked_checkpoint_base_identity(
                verified_inputs,
                stack_receipt=stack_receipt,
                sample_fraction=args.sample_fraction,
                sample_seed=args.sample_seed,
                clone_attachment_fraction=args.clone_attachment_fraction,
                clone_attachment_seed=args.clone_attachment_seed,
            )
            checkpoint_store = _PoolStageCheckpointStore(
                outputs.checkpoint_root,
                base_identity=checkpoint_identity,
            )
            outputs = _with_checkpoint_identity(
                outputs,
                base_identity_sha256=checkpoint_store.base_identity_sha256,
            )
            checkpoint_store.bind_input_receipts(_loaded_input_receipts(loaded))

        if (
            checkpoint_store is None
            or checkpoint_identity is None
            or stack_frame is None
            or stack_receipt is None
        ):  # pragma: no cover - cold/resume branches establish all four
            raise AssertionError("Stacked checkpoint routing did not initialize.")
        asec_count, acs_count = _stacked_realized_counts(stack_receipt)
        state.build_id = _new_stacked_release_id(
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
            realized_asec_households=asec_count,
            realized_acs_households=acs_count,
            timestamp=started_ts,
        )
        state.identity_digest = _pool_checkpoint_identity_sha256(checkpoint_identity)

        result = build_stacked_pool(
            stack_frame,
            expected_stack_receipt=stack_receipt,
            release_id=state.build_id,
            puf_donor=puf_donor,
            acs_rent_donor=acs_rent_donor,
            primary_qrf_checkpoint_dir=outputs.primary_qrf_checkpoint_dir,
            acs_transfer_checkpoint_dir=outputs.acs_transfer_checkpoint_dir,
            checkpoint_identity=checkpoint_store.base_identity,
            clone_attachment_fraction=args.clone_attachment_fraction,
            clone_attachment_seed=args.clone_attachment_seed,
            checkpoint=checkpoint_store.write,
            resume=resume,
            phase_reached=lambda phase: _append_phase(state, phase),
        )
        terminal_receipt_path = _write_stacked_terminal_gate_receipt(
            result,
            outputs=outputs,
        )
        state.gate_verdicts = {
            gate.name: {
                "verdict": "passed" if gate.passed else "failed",
                "receipt": (
                    f"{terminal_receipt_path.resolve()}"
                    f"#/terminal_gates/gates/{gate.name}"
                ),
            }
            for gate in result.terminal_gates
        }
        _append_phase(state, "terminal_receipt_written")
        checkpoint_provenance = checkpoint_store.provenance(
            primary_qrf_checkpoint_dir=outputs.primary_qrf_checkpoint_dir,
            acs_transfer_checkpoint_dir=outputs.acs_transfer_checkpoint_dir,
        )
        _write_stacked_outputs(
            result,
            outputs=outputs,
            verified_inputs=verified_inputs,
            acs_source_manifest=acs_source_manifest,
            input_receipts=checkpoint_store.input_receipts,
            checkpoint_provenance=checkpoint_provenance,
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
            clone_attachment_fraction=args.clone_attachment_fraction,
            clone_attachment_seed=args.clone_attachment_seed,
        )
        _append_phase(state, "publication_completed")
        state.artifact_location = str(outputs.pool_h5.resolve())
    except Exception as error:
        error_path = _stacked_error_receipt_path(
            outputs,
            build_id=state.build_id,
        )
        _atomic_write_json(
            error_path,
            {
                "artifact_kind": "populace_us_stacked_pool_error_receipt",
                "schema_version": 1,
                "pipeline": _STACKED_PIPELINE,
                "build_id": state.build_id,
                "phases_reached": state.phases_reached,
                "gate_verdicts": state.gate_verdicts,
                "error_type": f"{type(error).__module__}.{type(error).__qualname__}",
                "message": str(error),
            },
        )
        state.gate_verdicts = {
            **({} if set(state.gate_verdicts) == {"pipeline"} else state.gate_verdicts),
            "pipeline_error": {
                "verdict": "error",
                "receipt": f"{error_path.resolve()}#/error_type",
            },
        }
        _append_phase(state, "error")
        _record_stacked_terminal_attempt(
            state=state,
            outputs=outputs,
            started_at=started_at,
            started_ts=started_ts,
            code_pin=code_pin,
            rung=rung,
            seed=chronicle_seed,
            disposition="failed",
            predecessor=predecessor,
        )
        raise

    disposition = "iterating" if result.simulation_ready else "failed"
    spool_path = _record_stacked_terminal_attempt(
        state=state,
        outputs=outputs,
        started_at=started_at,
        started_ts=started_ts,
        code_pin=code_pin,
        rung=rung,
        seed=chronicle_seed,
        disposition=disposition,
        predecessor=predecessor,
    )
    if not result.simulation_ready:
        print(
            "US stacked terminal gates failed; diagnostics and a non-ready "
            f"manifest were written to {outputs.agreement_diagnostics} and "
            f"{outputs.manifest}."
        )
        print(f"Wrote Chronicle row: {spool_path}")
        return 1
    print(f"Wrote simulation-ready stacked pool: {outputs.pool_h5}")
    print(f"Wrote stacked pool manifest: {outputs.manifest}")
    print(f"Wrote Chronicle row: {spool_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the stacked default or explicit legacy two-spine path."""

    args = _parser().parse_args(argv)
    if args.legacy_two_spine:
        return _main_legacy(args)
    return _main_stacked(args)


if __name__ == "__main__":
    raise SystemExit(main())
