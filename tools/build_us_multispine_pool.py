#!/usr/bin/env python3
"""Build the SHA-pinned, pre-calibration US multispine input pool.

The executable order is fixed:

``assemble -> clone -> impute -> derive -> seed -> simulate -> agreement``.

Every input is local and explicitly SHA-pinned; this tool never downloads
data. It writes a nullable input-only H5 plus a manifest and terminal agreement
diagnostics. A failed agreement gate still leaves those diagnostic artifacts
but returns nonzero and never marks the pool simulation-ready. Calibration is
deliberately downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass, replace
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from populace.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from populace.build.gates import FitWeightRecord, GateReport, weights_audit_gate
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
from populace.build.us_runtime.support_provenance import (
    SPINE_ASSEMBLY_MANIFEST_KEY,
    validate_assembly_provenance,
)
from populace.build.us_runtime.take_up_contract import load_take_up_contract
from populace.frame import US_SCHEMA, Frame

__all__ = [
    "POOL_H5_ARTIFACT_KIND",
    "POOL_MANIFEST_SCHEMA_VERSION",
    "POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION",
    "POOL_STAGE_CHECKPOINT_SCHEMA_VERSION",
    "PoolBuildOutputs",
    "build_multispine_pool",
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
#
# Bump this version whenever any producer above changes a stage output without
# changing one of the explicit identity fields below. In particular, adding,
# removing, reordering, or changing an operator kernel requires a bump even if
# its public registry name stays constant. Correctness takes priority over
# retaining warm checkpoints (the same rule as TARGET_FRAME_CHECKPOINT's
# materializer-version ledger in build_us_fiscal_refresh_release.py).
POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION = 1

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
    source_paths = {
        Path(args.asec_raw_stage_h5).resolve(),
        Path(args.acs_household_zip).resolve(),
        Path(args.acs_person_zip).resolve(),
        Path(args.acs_rent_h5).resolve(),
        Path(args.puf_h5).resolve(),
        Path(args.puf_source_year_csv).resolve(),
    }
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

    take_up_contract = load_take_up_contract()
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
            "take_up_contract": {
                "version": take_up_contract.version,
                "country": take_up_contract.country,
                "programs": [
                    _json_ready(program.raw) for program in take_up_contract.programs
                ],
            },
            "primary_qrf_n_estimators": _PRIMARY_QRF_N_ESTIMATORS,
            "acs_transfer_n_estimators": _ACS_TRANSFER_N_ESTIMATORS,
            "acs_transfer_max_targets_per_fit": (
                DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ),
            "simulation_household_batch_size": (POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE),
        },
    }


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
        stored_frame = checkpoint.frame
        if stage == "simulated":
            if checkpoint.simulation_frame is None:
                raise ValueError(
                    "The simulated pool checkpoint requires an evaluation frame."
                )
            _assert_simulation_checkpoint_pair(
                checkpoint.frame,
                checkpoint.simulation_frame,
            )
            stored_frame = checkpoint.simulation_frame
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
    acs_transfer = impute.get("acs_qrf_transfer")
    if not isinstance(acs_transfer, dict) or "target_bank" not in acs_transfer:
        return normalized, {}
    target_bank = acs_transfer.pop("target_bank")
    if not isinstance(target_bank, Mapping):
        raise ValueError("ACS transfer target-bank receipt must be an object.")
    return normalized, {
        "impute": {
            "acs_qrf_transfer": {
                "target_bank": dict(target_bank),
            }
        }
    }


def _attach_checkpoint_operational_receipts(
    canonical_stage_receipts: Mapping[str, object],
    operational_stage_receipts: Mapping[str, object],
) -> dict[str, object]:
    """Reattach the one non-canonical receipt path for runtime observability."""

    canonical, existing_operational = _split_checkpoint_stage_receipts(
        canonical_stage_receipts
    )
    if existing_operational:
        raise ValueError("canonical stage receipts already contain target_bank")
    operational = _json_ready(operational_stage_receipts)
    if not isinstance(operational, dict):  # pragma: no cover - mapping normalization
        raise TypeError("Operational stage receipts must normalize to an object.")
    if set(operational) != {"impute"}:
        raise ValueError("operational stage receipts must contain only impute")
    operational_impute = operational.get("impute")
    if not isinstance(operational_impute, dict) or set(operational_impute) != {
        "acs_qrf_transfer"
    }:
        raise ValueError("operational impute receipt shape changed")
    operational_transfer = operational_impute.get("acs_qrf_transfer")
    if not isinstance(operational_transfer, dict) or set(operational_transfer) != {
        "target_bank"
    }:
        raise ValueError("operational ACS transfer receipt shape changed")
    target_bank = operational_transfer.get("target_bank")
    if not isinstance(target_bank, Mapping):
        raise ValueError("operational target_bank receipt must be an object")
    canonical_impute = canonical.get("impute")
    if not isinstance(canonical_impute, dict):
        raise ValueError("canonical stage receipts have no impute object")
    canonical_transfer = canonical_impute.get("acs_qrf_transfer")
    if not isinstance(canonical_transfer, dict):
        raise ValueError("canonical impute receipt has no acs_qrf_transfer object")
    canonical_transfer["target_bank"] = dict(target_bank)
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
    return PoolStageOutput(
        transferred.frame,
        {
            "source_operator_chain": {
                "post_primary_completion": dict(source_completion.receipt),
            },
            "primary_puf_qrf": {
                "resume_status": qrf_resume_status,
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
                "target_bank": target_bank.receipt(),
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


def main(argv: list[str] | None = None) -> int:
    """Build the pool and return nonzero exactly when terminal agreement is red."""

    args = _parser().parse_args(argv)
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


if __name__ == "__main__":
    raise SystemExit(main())
