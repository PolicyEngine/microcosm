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
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    POOL_HOUSEHOLD_MASS_SHARES,
    POOL_OPERATOR_ORDER,
    POOL_RANDOM_SEED,
    POOL_TIME_PERIOD,
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
    finalize_primary_puf_qrf_chain,
    initialize_primary_puf_qrf_chain,
    run_primary_puf_qrf_chain,
)
from populace.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
    US_PUF_SUPPORT_FIT_NAME,
)
from populace.frame import Frame

__all__ = [
    "POOL_H5_ARTIFACT_KIND",
    "POOL_MANIFEST_SCHEMA_VERSION",
    "PoolBuildOutputs",
    "build_multispine_pool",
    "load_simulation_ready_us_multispine_pool_manifest",
    "main",
]

POOL_MANIFEST_SCHEMA_VERSION = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
"""Schema version for the companion pool build manifest."""

POOL_H5_ARTIFACT_KIND = US_MULTISPINE_POOL_H5_ARTIFACT_KIND
"""Neutral H5 artifact kind; readiness is asserted only by the manifest."""

_PRIMARY_QRF_N_ESTIMATORS = 100
_ACS_TRANSFER_N_ESTIMATORS = 100
_PRIMARY_QRF_INPUT_BINDING_FILENAME = "pool-input-binding.json"
_PRIMARY_QRF_INPUT_BINDING_SCHEMA_VERSION = 1
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")

type PoolOperator = Callable[[Frame], PoolStageOutput]


@dataclass(frozen=True)
class PoolBuildOutputs:
    """Deterministic output paths derived from the requested H5 path."""

    pool_h5: Path
    manifest: Path
    agreement_diagnostics: Path
    primary_qrf_checkpoint_dir: Path


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
    return parser


def _output_paths(path: Path) -> PoolBuildOutputs:
    pool_h5 = Path(path)
    if pool_h5.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("--out must name an .h5 or .hdf5 file.")
    return PoolBuildOutputs(
        pool_h5=pool_h5,
        manifest=pool_h5.with_suffix(".manifest.json"),
        agreement_diagnostics=pool_h5.with_suffix(".agreement.json"),
        primary_qrf_checkpoint_dir=pool_h5.with_suffix(".primary-qrf"),
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
    output_paths = {
        outputs.pool_h5.resolve(),
        outputs.manifest.resolve(),
        outputs.agreement_diagnostics.resolve(),
        outputs.primary_qrf_checkpoint_dir.resolve(),
    }
    collisions = sorted(str(path) for path in source_paths & output_paths)
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
    donor_build: dict[str, object] = {}
    puf_donor = load_puf_tax_unit_donor(
        args.puf_h5,
        args.puf_source_year_csv,
        donor_build_summary=donor_build,
    )
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


def _initialize_or_resume_primary_qrf(
    frame: Frame,
    donor: pd.DataFrame,
    checkpoint_dir: Path,
    *,
    input_binding: Mapping[str, object],
) -> None:
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
        return
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


def _impute_pool(
    frame: Frame,
    *,
    puf_donor: pd.DataFrame,
    checkpoint_dir: Path,
    checkpoint_input_binding: Mapping[str, object],
) -> PoolStageOutput:
    _initialize_or_resume_primary_qrf(
        frame,
        puf_donor,
        checkpoint_dir,
        input_binding=checkpoint_input_binding,
    )
    run_primary_puf_qrf_chain(checkpoint_dir)

    tail_bound_diagnostics: list[dict[str, object]] = []
    with_primary_detail, primary_weight_kind = finalize_primary_puf_qrf_chain(
        frame,
        checkpoint_dir,
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
    transferred = transfer_acs_inputs(
        source_completion.frame,
        source_completion.frame,
        target_families=transfer_families,
        donor_channel=ACS_DONOR_CHANNEL_AUTO,
        seed=POOL_RANDOM_SEED,
        n_estimators=_ACS_TRANSFER_N_ESTIMATORS,
        max_targets_per_fit=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
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

    qrf_manifest_path = _primary_qrf_manifest_path(checkpoint_dir)
    qrf_binding_path = _primary_qrf_input_binding_path(checkpoint_dir)
    return PoolStageOutput(
        transferred.frame,
        {
            "source_operator_chain": {
                "post_primary_completion": dict(source_completion.receipt),
            },
            "primary_puf_qrf": {
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
            },
            "weights_audit": GateReport((weights_audit,)).to_manifest(),
        },
    )


def build_multispine_pool(
    asec: Frame,
    acs: Frame,
    *,
    puf_donor: pd.DataFrame,
    acs_rent_donor: pd.DataFrame | None = None,
    primary_qrf_checkpoint_dir: Path,
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
) -> MultispinePoolResult:
    """Run the production pool path, with injectable operators for fixtures.

    Production callers omit all operator overrides. The terminal agreement
    gate is intentionally not injectable here, so this wiring can never alter
    its registry or fixed tolerances.
    """

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
        if checkpoint_input_binding is None:
            raise ValueError(
                "Production pool imputation requires a verified checkpoint "
                "input binding."
            )
        if acs_rent_donor is None:
            raise ValueError(
                "Production pool imputation requires the canonical ACS rent donor."
            )

        def impute_operator(frame: Frame) -> PoolStageOutput:
            return _impute_pool(
                frame,
                puf_donor=puf_donor,
                checkpoint_dir=primary_qrf_checkpoint_dir,
                checkpoint_input_binding=checkpoint_input_binding,
            )

        prepare_clone_operator = prepare_clone or (
            lambda frame: prepare_multispine_source_inputs_for_clone(
                frame,
                acs_rent_donor=acs_rent_donor,
            )
        )

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
    )


def _agreement_payload(result: MultispinePoolResult) -> dict[str, object]:
    return GateReport((result.agreement_gate,)).to_manifest()


def _manifest_payload(
    *,
    result: MultispinePoolResult,
    outputs: PoolBuildOutputs,
    verified_inputs: Mapping[str, _VerifiedInput],
    acs_source_manifest: AcsSourceManifest,
    loaded: _LoadedInputs,
    publication_run_id: str,
) -> dict[str, object]:
    status = "simulation_ready" if result.simulation_ready else "agreement_failed"
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
        "asec_raw_stage_checkpoint": loaded.asec_raw_stage_checkpoint,
        "acs_source_manifest": asdict(acs_source_manifest),
        "acs_pums_build": loaded.acs_build,
        "acs_native_inputs": loaded.acs_native_inputs,
        "puf_donor": {
            "rows": int(len(loaded.puf_donor)),
            "columns": sorted(str(column) for column in loaded.puf_donor.columns),
            "build_receipt": loaded.puf_donor_build,
        },
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
    return output.with_name(
        f".{output.name}.{publication_run_id}.publication.tmp"
    )


def _write_outputs(
    result: MultispinePoolResult,
    *,
    outputs: PoolBuildOutputs,
    verified_inputs: Mapping[str, _VerifiedInput],
    acs_source_manifest: AcsSourceManifest,
    loaded: _LoadedInputs,
) -> None:
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
            loaded=loaded,
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
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Build receipt mappings must use string JSON keys.")
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
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
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Build the pool and return nonzero exactly when terminal agreement is red."""

    args = _parser().parse_args(argv)
    outputs = _output_paths(args.out)
    verified_inputs, acs_source_manifest = _verify_inputs(args, outputs)
    loaded = _load_inputs(args, acs_source_manifest=acs_source_manifest)
    result = build_multispine_pool(
        loaded.asec,
        loaded.acs,
        puf_donor=loaded.puf_donor,
        acs_rent_donor=loaded.acs_rent_donor,
        primary_qrf_checkpoint_dir=outputs.primary_qrf_checkpoint_dir,
        checkpoint_input_binding=_checkpoint_input_binding(verified_inputs),
        source_native_inputs={"acs": loaded.acs_native_inputs},
    )
    _write_outputs(
        result,
        outputs=outputs,
        verified_inputs=verified_inputs,
        acs_source_manifest=acs_source_manifest,
        loaded=loaded,
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
