#!/usr/bin/env python3
"""Build the SHA-pinned, pre-calibration US input pool.

The default production path is the stacked pipeline:

``stack -> gap-fill -> PUF pass + tail -> late DAG -> derive -> seed -> simulate -> gates``.

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

   PYTHONPATH=packages/microcosm-frame/src:packages/microcosm-fit/src:packages/microcosm-calibrate/src:packages/microcosm-build/src:packages/microcosm-data/src \\
     /path/to/microcosm/.venv/bin/python tools/build_us_multispine_pool.py \\
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
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.frame_sampling import EXACT_COUNT_RULE
from microcosm.build.gates import (
    FitWeightRecord,
    GateReport,
    GateResult,
    weights_audit_gate,
)
from microcosm.build.logbook import record_build_attempt
from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.build.spec_engine import (
    RunProvenanceIdentity,
    RuntimeAuthorities,
    assert_legacy_payload_equal,
    build_run_provenance_identity,
    compile_runtime_authorities,
    compile_spec,
    compile_to_legacy_payload,
    load_bundle,
)
from microcosm.build.spec_engine.artifact_collection import (
    ArtifactLocatorRegistry,
    collect_artifact_digests,
)
from microcosm.build.spec_engine.battery_semantics import (
    project_battery_legacy_contract,
)
from microcosm.build.spec_engine.brokers import (
    BrokerOwner,
    BrokerSession,
    FileReadLease,
)
from microcosm.build.spec_engine.f1_certification import (
    complete_coverage_evidence,
    emit_f1_production_evidence,
)
from microcosm.build.spec_engine.plan_lock import plan_lock_payload
from microcosm.build.us_runtime.acs_income_universe import (
    acs_pums_earnings_universe_contract_identity,
)
from microcosm.build.us_runtime.acs_inputs import map_acs_native_inputs
from microcosm.build.us_runtime.acs_pums import (
    AcsPumsSource,
    build_acs_pums_unit_frame,
)
from microcosm.build.us_runtime.acs_sources import (
    AcsSourceArtifact,
    AcsSourceManifest,
    load_acs_source_manifest,
)
from microcosm.build.us_runtime.acs_transfer import (
    ACS_DONOR_CHANNEL_AUTO,
    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
    transfer_acs_inputs,
)
from microcosm.build.us_runtime.acs_transfer_bank import AcsTransferTargetBankStore
from microcosm.build.us_runtime.asec_checkpoint import (
    load_asec_raw_stage_checkpoint,
)
from microcosm.build.us_runtime.checkpoint_authority import (
    materialize_stacked_checkpoint_base_identity,
)
from microcosm.build.us_runtime.h5_io import (
    US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
    US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION,
    US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
    US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION,
    US_STACKED_POOL_OPERATOR_ORDER,
    load_simulation_ready_us_multispine_pool_manifest,
    write_nullable_us_h5,
)
from microcosm.build.us_runtime.housing_inputs import (
    ACS_2022_RENT_ARTIFACT_SHA256,
    load_acs_2022_rent_donor,
)
from microcosm.build.us_runtime.multispine_pool import (
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
    pool_remaining_stage_input_manifest_receipt,
    pool_transfer_target_families,
    prepare_multispine_source_inputs_for_clone,
    run_multispine_pool_path,
    seed_multispine_pool_inputs,
)
from microcosm.build.us_runtime.operator_boundary import (
    assert_operator_free_source_frame,
)
from microcosm.build.us_runtime.pool_artifact_coverage import (
    PoolArtifactCoverageContract,
    compile_pool_artifact_coverage,
    validate_pool_artifact_coverage,
)
from microcosm.build.us_runtime.pool_kernel_authority import (
    USPoolKernelAuthorities,
)
from microcosm.build.us_runtime.pool_runtime_plan import USPoolRuntimePlan
from microcosm.build.us_runtime.puf_capital_gains_tail import (
    PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION,
    puf_capital_gains_tail_support_contract_identity,
    transfer_puf_capital_gains_tail,
    validate_puf_capital_gains_tail_manifest,
)
from microcosm.build.us_runtime.puf_donor_io import (
    ParsedPufTaxUnitDonorSources,
    load_puf_tax_unit_donor,
    materialize_puf_tax_unit_donor,
    parse_puf_tax_unit_donor_sources,
)
from microcosm.build.us_runtime.puf_qrf_chain import (
    PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION,
    PRIMARY_QRF_MANIFEST_FILENAME,
    PRIMARY_QRF_TARGET_ORDER,
    finalize_primary_puf_qrf_chain,
    initialize_primary_puf_qrf_chain,
    run_primary_puf_qrf_chain,
)
from microcosm.build.us_runtime.puf_support import (
    PUF_SUPPORT_MAX_CLONE_SAFE_SOURCE_ID,
    US_PUF_SUPPORT_FIT_NAME,
)
from microcosm.build.us_runtime.qbi_inputs import (
    us_qbi_post_reconciliation_person_columns,
    us_qbi_reconciliation_contract_identity,
    validate_us_qbi_reconciliation_live_output,
)
from microcosm.build.us_runtime.spec_authority import (
    compile_us_spec_authority,
)
from microcosm.build.us_runtime.spec_materializers import (
    compile_declared_source_pins,
    materialize_acs_source_manifest,
)
from microcosm.build.us_runtime.stacked_battery_contract import (
    build_live_stacked_battery_contract,
)
from microcosm.build.us_runtime.stacked_spine import (
    ACS_STACKED_SUPPORT_CHANNEL,
    CANONICAL_STACKED_GAP_FILL_SURFACE,
    CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE,
    DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES,
    assemble_stacked_spine,
    assert_stacked_tail_cells_preserved,
    by_origin_battery,
    gap_fill_stacked_spine,
    prepare_stacked_tail_derivation,
    run_stacked_late_producer_dag,
    run_stacked_puf_pass,
    stacked_completeness_gate,
    stacked_gap_fill_plan,
    stacked_gap_fill_producer_schedule_receipt,
    stacked_late_primary_checkpoint_input_binding,
    stacked_late_primary_resource_receipts,
    stacked_late_producer_resource_semantics_receipt,
    stacked_spine_authority_receipt,
    validate_stacked_late_producer_receipt,
    validate_stacked_late_producer_transition_authority,
    validate_stacked_spine_frame,
)
from microcosm.build.us_runtime.support_provenance import (
    BASE_ASEC_SUPPORT_CHANNEL,
    SPINE_ASSEMBLY_MANIFEST_KEY,
    spine_provenance_counts,
    validate_assembly_provenance,
)
from microcosm.build.us_runtime.take_up_contract import take_up_contract_identity
from microcosm.build.us_runtime.us_late_overlap_ownership import (
    us_late_overlap_ownership_receipt,
)
from microcosm.build.us_runtime.us_late_producer_registry import (
    CANONICAL_US_LATE_TRANSFER_GROUPS,
    us_late_producer_schedule_receipt,
)
from microcosm.frame import US_SCHEMA, Frame

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
    "stacked_checkpoint_artifact_protocol_identity",
]

POOL_MANIFEST_SCHEMA_VERSION = US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
"""Schema version for the companion pool build manifest."""

# ``--legacy-two-spine`` is a byte-stable compatibility surface.  Stacked
# publication and checkpoint-envelope versions may advance without rewriting
# the retiring pipeline's last supported envelope.
_LEGACY_POOL_MANIFEST_SCHEMA_VERSION = 4
_LEGACY_POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION = 3

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
# 3: The tail-manifest schema and filing-status-exact recipient-support
#    contract are explicit identity fields.  Earlier checkpoints may have
#    silently hard-failed a thin status and are deliberately stale.
# 4: The declared late-stage producer-input DAG, including its derived order,
#    full source-input inventories, and nineteen bounded transfer groups, is
#    bound into checkpoint identity.  Fixed source-then-transfer checkpoints
#    are deliberately stale.
# 5: Stacked transferred and simulated checkpoints carry and validate the
#    independently propagated late-producer transition authority. Earlier
#    envelopes cannot authenticate a reissued execution receipt.
# 6: Frame-checkpoint schema v3 materializes pandas nullable booleans as bool
#    values plus an explicit null mask when needed. Earlier envelopes cannot
#    prove that declared absences survived serialization.
# 7: Stacked primary-PUF output universes are explicit. Earlier envelopes can
#    contain nulls outside the PUF clone for an output declared over the whole
#    pool and therefore cannot resume safely even when their bank is reusable.
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
POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION = 7

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
_STACKED_SAMPLE_RUNG_TOKENS: Mapping[float, str] = {
    0.01: "f001",
    0.04: "f004",
    0.10: "f010",
    0.25: "f025",
    1.00: "f100",
}
_STACKED_LEGACY_RELEASE_LINE = "populace-us-2024"
_STACKED_PIPELINE = "us-stacked-pool"
_STACKED_CHECKPOINT_IDENTITY_ARTIFACT_KIND = (
    "populace_us_stacked_pool_checkpoint_identity"
)
# Version 11 additionally binds the primary-PUF whole-pool universe semantics.
# Earlier checkpoints must rebuild rather than resume with a nullable
# s_corp_income leaf. Version 10 bound the complete late-resource semantics and
# corrected outer order (the primary PUF callback is nested inside the DAG).
_STACKED_CHECKPOINT_MATERIALIZER_VERSION = 11
_STACKED_RELEASE_ID_PATTERN = re.compile(
    r"^populace-us-2024-stacked-f(?:001|004|010|025|100)-s[0-9]+-"
    r"asec[0-9]+-acs[0-9]+-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
)

type PoolOperator = Callable[[Frame], PoolStageOutput]


def stacked_checkpoint_artifact_protocol_identity() -> dict[str, object]:
    """Return the static generation-0 stacked checkpoint identity envelope."""

    return {
        "artifact_kind": _STACKED_CHECKPOINT_IDENTITY_ARTIFACT_KIND,
        "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
        "materializer_version": _STACKED_CHECKPOINT_MATERIALIZER_VERSION,
        "pipeline": _STACKED_PIPELINE,
    }


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
    qbi_transition_authority_sha256: str | None = None
    late_producer_transition_authority_sha256: str | None = None

    @property
    def simulation_ready(self) -> bool:
        return all(gate.passed for gate in self.terminal_gates)


@dataclass
class _StackedAttemptState:
    """Mutable terminal-attempt evidence collected for Logbook emission."""

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


class _BundleSourceSnapshotSession:
    """Own one compiler-granted source session through real parser reads."""

    def __init__(
        self,
        *,
        session: BrokerSession,
        source_ids: tuple[str, ...],
        receipt_sink: Callable[[Mapping[str, object]], None] | None,
    ) -> None:
        self._session = session
        self._source_ids = source_ids
        self._receipt_sink = receipt_sink
        self._scope_open = False
        self._parsing_finished = False
        self._receipt_emitted = False

    @property
    def sealed(self) -> bool:
        return self._session.sealed

    def _emit_receipt(self) -> None:
        if self._receipt_emitted:
            return
        if self._receipt_sink is not None:
            self._receipt_sink(self._session.receipt.to_wire())
        self._receipt_emitted = True

    @contextmanager
    def open_snapshots(self) -> Iterator[Mapping[str, FileReadLease]]:
        if self.sealed:
            raise ValueError("Bundle source snapshot session is already sealed.")
        if self._scope_open or self._parsing_finished:
            raise ValueError("Bundle source snapshots may be opened exactly once.")
        self._scope_open = True
        try:
            _preload_bundle_source_parsers()
            with self._session.activate(), ExitStack() as stack:
                snapshots = {
                    source_id: stack.enter_context(
                        self._session.files.open_snapshot(source_id)
                    )
                    for source_id in self._source_ids
                }
                yield MappingProxyType(snapshots)
        except BaseException:
            self.abort()
            raise
        else:
            self._parsing_finished = True
        finally:
            self._scope_open = False

    def complete(self) -> None:
        if not self._parsing_finished:
            raise ValueError(
                "Bundle source session cannot complete before parser reads finish."
            )
        try:
            self._session.seal()
        finally:
            if self._session.sealed:
                self._emit_receipt()

    def abort(self) -> None:
        if not self._session.sealed:
            self._session.seal(status="aborted")
        self._emit_receipt()


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


@dataclass(frozen=True)
class _ParsedBundleInputs:
    """All source bytes parsed while the source broker remains active."""

    asec: Frame
    acs: Frame
    acs_rent_donor: pd.DataFrame
    puf_sources: ParsedPufTaxUnitDonorSources
    asec_raw_stage_checkpoint: Mapping[str, object]
    acs_build: Mapping[str, object]
    acs_native_inputs: Mapping[str, Mapping[str, Any]]


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
            "sample fraction must be one of 0.01, 0.04, 0.10, 0.25, or 1.0."
        ) from exc
    if fraction not in _STACKED_SAMPLE_RUNG_TOKENS:
        raise argparse.ArgumentTypeError(
            "sample fraction must be one of 0.01, 0.04, 0.10, 0.25, or 1.0."
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
        "--resume-policy",
        choices=("allow", "forbid"),
        default="allow",
        help=(
            "Checkpoint policy. 'allow' preserves automatic durable resume; "
            "'forbid' refuses any pre-existing publication or checkpoint "
            "state before loading (default: allow)."
        ),
    )
    parser.add_argument(
        "--f1-evidence-out",
        type=Path,
        help=(
            "Optional typed F1 production-evidence sidecar. Requires a stacked "
            "constants or bundle build with --resume-policy forbid."
        ),
    )
    parser.add_argument(
        "--sample-fraction",
        type=_standard_sample_fraction,
        default=1.0,
        help="Uniform survey-arm rung: 0.01/0.04 smoke, 0.10 dev, 0.25 probe, or 1.0 full.",
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
        "--logbook-prev-row-digest",
        type=_sha256_argument,
        help=(
            "Optional current Logbook chain head. If omitted, "
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST is used, then genesis null."
        ),
    )
    parser.add_argument(
        "--legacy-two-spine",
        action="store_true",
        help="Run the byte-compatible retiring two-spine pipeline.",
    )
    parser.add_argument(
        "--config-authority",
        choices=("constants", "constants_adapter", "bundle"),
        default="constants",
        help=(
            "Configuration receipt mode. constants_adapter compiles and "
            "equality-attests the packaged US bundle, then still executes "
            "through the generation-0 constants path; bundle compiles the "
            "packaged bundle into the generation-1 runtime authority "
            "(default: constants)."
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


def _refuse_preexisting_resume_state(
    args: argparse.Namespace,
    outputs: PoolBuildOutputs,
) -> None:
    """Enforce the certification cold-start boundary before any state load."""

    policy = getattr(args, "resume_policy", "allow")
    if policy == "allow":
        return
    if policy != "forbid":  # pragma: no cover - argparse owns the public enum
        raise ValueError(f"Unsupported resume policy: {policy!r}")
    candidates = tuple(
        path
        for path in (
            outputs.pool_h5,
            outputs.manifest,
            outputs.agreement_diagnostics,
            outputs.checkpoint_root,
            getattr(args, "f1_evidence_out", None),
        )
        if path is not None
    )
    existing = sorted(str(path) for path in candidates if os.path.lexists(Path(path)))
    if existing:
        raise ValueError(
            "--resume-policy forbid refuses pre-existing publication or "
            f"checkpoint state: {existing}"
        )


def _validate_f1_evidence_request(
    args: argparse.Namespace,
    outputs: PoolBuildOutputs,
) -> Path | None:
    """Validate the opt-in certification sidecar before any build state load."""

    raw_path = getattr(args, "f1_evidence_out", None)
    if raw_path is None:
        return None
    if getattr(args, "legacy_two_spine", False):
        raise ValueError("--f1-evidence-out is unavailable for --legacy-two-spine")
    mode = getattr(args, "config_authority", "constants")
    if mode not in {"constants", "bundle"}:
        raise ValueError(
            "--f1-evidence-out requires --config-authority constants or bundle"
        )
    if getattr(args, "resume_policy", "allow") != "forbid":
        raise ValueError(
            "--f1-evidence-out requires --resume-policy forbid for a cold build"
        )

    path = Path(raw_path).resolve()
    publication_paths = {
        outputs.pool_h5.resolve(),
        outputs.manifest.resolve(),
        outputs.agreement_diagnostics.resolve(),
    }
    source_paths = {source.resolve() for source in _configured_source_paths(args)}
    checkpoint_root = outputs.checkpoint_root.resolve()
    if path in publication_paths or path in source_paths:
        raise ValueError(
            "--f1-evidence-out collides with a publication or source artifact"
        )
    if path == checkpoint_root or path.is_relative_to(checkpoint_root):
        raise ValueError("--f1-evidence-out must be outside the checkpoint namespace")
    return path


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
    *,
    bundle_plan: USPoolRuntimePlan | None = None,
    run_provenance_identity: Mapping[str, object] | None = None,
    broker_receipt_sink: Callable[[Mapping[str, object]], None] | None = None,
    source_snapshot_session_sink: Callable[[_BundleSourceSnapshotSession], None]
    | None = None,
) -> tuple[dict[str, _VerifiedInput], AcsSourceManifest]:
    source_paths = _configured_source_paths(args)
    _validate_checkpoint_path_layout(
        outputs,
        source_paths=source_paths,
        runtime_plan=bundle_plan,
    )

    output_paths = {
        outputs.pool_h5.resolve(),
        outputs.manifest.resolve(),
        outputs.agreement_diagnostics.resolve(),
    }
    collisions = sorted(str(path) for path in source_paths if path in output_paths)
    if collisions:
        raise ValueError(f"Pool outputs must not overwrite inputs: {collisions}.")

    if bundle_plan is not None:
        if run_provenance_identity is None:
            raise ValueError(
                "Bundle source preflight requires its run provenance identity."
            )
        if source_snapshot_session_sink is None:
            return _verify_bundle_inputs(
                args,
                plan=bundle_plan,
                run_provenance_identity=run_provenance_identity,
                broker_receipt_sink=broker_receipt_sink,
            )
        verified, manifest, snapshot_session = _prepare_bundle_inputs(
            args,
            plan=bundle_plan,
            run_provenance_identity=run_provenance_identity,
            broker_receipt_sink=broker_receipt_sink,
        )
        source_snapshot_session_sink(snapshot_session)
        return verified, manifest
    if run_provenance_identity is not None:
        raise ValueError(
            "Source preflight provenance is valid only with bundle authority."
        )
    if broker_receipt_sink is not None:
        raise ValueError("A broker receipt sink is valid only with bundle authority.")
    if source_snapshot_session_sink is not None:
        raise ValueError(
            "A source snapshot session sink is valid only with bundle authority."
        )

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


def _verify_bundle_inputs(
    args: argparse.Namespace,
    *,
    plan: USPoolRuntimePlan,
    run_provenance_identity: Mapping[str, object],
    broker_receipt_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[dict[str, _VerifiedInput], AcsSourceManifest]:
    """Eager compatibility wrapper over the production parser-session path."""

    verified, manifest, snapshot_session = _prepare_bundle_inputs(
        args,
        plan=plan,
        run_provenance_identity=run_provenance_identity,
        broker_receipt_sink=broker_receipt_sink,
    )
    with snapshot_session.open_snapshots():
        pass
    snapshot_session.complete()
    return verified, manifest


def _prepare_bundle_inputs(
    args: argparse.Namespace,
    *,
    plan: USPoolRuntimePlan,
    run_provenance_identity: Mapping[str, object],
    broker_receipt_sink: Callable[[Mapping[str, object]], None] | None = None,
) -> tuple[
    dict[str, _VerifiedInput],
    AcsSourceManifest,
    _BundleSourceSnapshotSession,
]:
    """Bind all six compiler grants without reopening any source path later."""

    pins = compile_declared_source_pins(plan.sources)
    acs_source_manifest = materialize_acs_source_manifest(plan.sources)
    specifications = (
        (
            "asec_raw_stage",
            "ASEC raw-stage checkpoint",
            args.asec_raw_stage_h5,
            args.asec_raw_stage_h5_sha256,
        ),
        (
            "acs_household",
            "ACS household archive",
            args.acs_household_zip,
            args.acs_household_zip_sha256,
        ),
        (
            "acs_person",
            "ACS person archive",
            args.acs_person_zip,
            args.acs_person_zip_sha256,
        ),
        (
            "acs_rent_donor",
            "ACS rent donor",
            args.acs_rent_h5,
            args.acs_rent_h5_sha256,
        ),
        (
            "processed_puf",
            "processed PUF H5",
            args.puf_h5,
            args.puf_h5_sha256,
        ),
        (
            "puf_source_year",
            "source-year PUF CSV",
            args.puf_source_year_csv,
            args.puf_source_year_csv_sha256,
        ),
    )
    declared_sources = tuple(
        pins.bind(
            source_id,
            path,
            supplied_sha256=expected_sha256,
        )
        for source_id, _role, path, expected_sha256 in specifications
    )
    session = BrokerSession(
        owner=BrokerOwner("source_stage", "declared_source_preflight"),
        determinism="deterministic",
        effects=("declared_source_read",),
        protocol_id=plan.seed_stream_map.protocol_id,
        protocol_sha256=plan.seed_stream_map.implementation_sha256,
        sources=declared_sources,
        run_provenance_identity=run_provenance_identity,
    )
    declared_by_id = {source.id: source for source in declared_sources}
    verified = {
        source_id: _VerifiedInput(
            role=role,
            path=Path(path),
            expected_sha256=expected_sha256,
            actual_sha256=declared_by_id[source_id].sha256,
            size_bytes=declared_by_id[source_id].byte_size,
        )
        for source_id, role, path, expected_sha256 in specifications
    }
    snapshot_session = _BundleSourceSnapshotSession(
        session=session,
        source_ids=tuple(source_id for source_id, *_rest in specifications),
        receipt_sink=broker_receipt_sink,
    )
    return verified, acs_source_manifest, snapshot_session


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
    runtime_plan: USPoolRuntimePlan | None = None,
) -> None:
    """Reject only paths the checkpoint store can actually overwrite."""

    checkpoint_root = outputs.checkpoint_root.resolve()
    primary_qrf_root = (checkpoint_root / "primary-qrf").resolve()
    acs_transfer_root = (checkpoint_root / "acs-transfer").resolve()
    stage_files = {
        (checkpoint_root / filename).resolve()
        for filename in _stacked_checkpoint_filenames(runtime_plan).values()
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


def _preload_bundle_source_parsers() -> None:
    """Resolve lazy parser imports before the ambient-access guard is active."""

    from zipfile import ZipFile

    import h5py

    _ = (h5py.File, ZipFile, pd.read_csv)


def _parse_bundle_inputs(
    args: argparse.Namespace,
    *,
    acs_source_manifest: AcsSourceManifest,
    snapshots: Mapping[str, FileReadLease],
) -> _ParsedBundleInputs:
    asec, asec_raw_stage_checkpoint = load_asec_raw_stage_checkpoint(
        args.asec_raw_stage_h5,
        source_stream=snapshots["asec_raw_stage"],
    )
    acs_source = AcsPumsSource(
        household_zip=args.acs_household_zip,
        person_zip=args.acs_person_zip,
        vintage=acs_source_manifest.vintage,
    )
    acs_frame, acs_build = build_acs_pums_unit_frame(
        acs_source,
        household_stream=snapshots["acs_household"],
        person_stream=snapshots["acs_person"],
    )
    mapped_acs = map_acs_native_inputs(acs_frame)
    acs_rent_donor = load_acs_2022_rent_donor(
        args.acs_rent_h5,
        source_stream=snapshots["acs_rent_donor"],
    )
    puf_sources = parse_puf_tax_unit_donor_sources(
        args.puf_h5,
        args.puf_source_year_csv,
        processed_puf_stream=snapshots["processed_puf"],
        source_year_puf_stream=snapshots["puf_source_year"],
    )
    return _ParsedBundleInputs(
        asec=asec,
        acs=mapped_acs.frame,
        acs_rent_donor=acs_rent_donor,
        puf_sources=puf_sources,
        asec_raw_stage_checkpoint=asec_raw_stage_checkpoint,
        acs_build=acs_build,
        acs_native_inputs=mapped_acs.native_inputs,
    )


def _materialize_bundle_inputs(parsed: _ParsedBundleInputs) -> _LoadedInputs:
    donor_build: dict[str, object] = {}
    donor = materialize_puf_tax_unit_donor(
        parsed.puf_sources,
        donor_build_summary=donor_build,
    )
    return _LoadedInputs(
        asec=parsed.asec,
        acs=parsed.acs,
        acs_rent_donor=parsed.acs_rent_donor,
        puf_donor=donor,
        asec_raw_stage_checkpoint=parsed.asec_raw_stage_checkpoint,
        acs_build=parsed.acs_build,
        acs_native_inputs=parsed.acs_native_inputs,
        puf_donor_build=donor_build,
    )


def _load_bundle_inputs(
    args: argparse.Namespace,
    *,
    acs_source_manifest: AcsSourceManifest,
    source_session: _BundleSourceSnapshotSession,
) -> _LoadedInputs:
    try:
        with source_session.open_snapshots() as snapshots:
            parsed = _parse_bundle_inputs(
                args,
                acs_source_manifest=acs_source_manifest,
                snapshots=snapshots,
            )
        loaded = _materialize_bundle_inputs(parsed)
        source_session.complete()
        return loaded
    except BaseException:
        source_session.abort()
        raise


def _load_bundle_resume_donors(
    args: argparse.Namespace,
    *,
    source_session: _BundleSourceSnapshotSession,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        with source_session.open_snapshots() as snapshots:
            acs_rent_donor = load_acs_2022_rent_donor(
                args.acs_rent_h5,
                source_stream=snapshots["acs_rent_donor"],
            )
            puf_sources = parse_puf_tax_unit_donor_sources(
                args.puf_h5,
                args.puf_source_year_csv,
                processed_puf_stream=snapshots["processed_puf"],
                source_year_puf_stream=snapshots["puf_source_year"],
            )
        puf_donor = materialize_puf_tax_unit_donor(puf_sources)
        source_session.complete()
        return acs_rent_donor, puf_donor
    except BaseException:
        source_session.abort()
        raise


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
    materializer_version: int | None = None,
) -> dict[str, object]:
    """Return every input and semantic surface that determines cached stages."""

    resolved_materializer_version = (
        POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
        if materializer_version is None
        else materializer_version
    )

    return {
        "artifact_kind": "populace_us_multispine_pool_checkpoint_identity",
        "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
        "materializer_version": resolved_materializer_version,
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
            "late_producer_schedule": _json_ready(us_late_producer_schedule_receipt()),
            "derive_operator_order": list(POOL_DERIVE_OPERATOR_ORDER),
            "primary_qrf_target_order": list(PRIMARY_QRF_TARGET_ORDER),
            "transfer_target_families": _json_ready(pool_transfer_target_families()),
            "take_up_contract": take_up_contract_identity(),
            "puf_capital_gains_tail_manifest_schema_version": (
                PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION
            ),
            "puf_capital_gains_tail_support_contract": (
                puf_capital_gains_tail_support_contract_identity()
            ),
            "primary_qrf_n_estimators": _PRIMARY_QRF_N_ESTIMATORS,
            "acs_transfer_n_estimators": _ACS_TRANSFER_N_ESTIMATORS,
            "acs_transfer_max_targets_per_fit": (
                DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ),
            "simulation_household_batch_size": (POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE),
        },
    }


def _legacy_pool_checkpoint_base_identity(
    verified_inputs: Mapping[str, _VerifiedInput],
    *,
    policyengine_us_version: str | None = None,
) -> dict[str, object]:
    """Return the retiring pipeline identity without stacked-only DAG state."""

    identity = _pool_checkpoint_base_identity(
        verified_inputs,
        policyengine_us_version=policyengine_us_version,
        materializer_version=_LEGACY_POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
    )
    pool_code = dict(identity["pool_code"])
    del pool_code["late_producer_schedule"]
    return {**identity, "pool_code": pool_code}


def _stacked_checkpoint_order(
    runtime_plan: USPoolRuntimePlan | None,
) -> tuple[str, ...]:
    if runtime_plan is None:
        return tuple(POOL_CHECKPOINT_STAGE_ORDER)
    return tuple(checkpoint.id for checkpoint in runtime_plan.execution.checkpoints)


def _stacked_checkpoint_candidate_order(
    runtime_plan: USPoolRuntimePlan | None,
) -> tuple[str, ...]:
    if runtime_plan is None:
        return tuple(reversed(POOL_CHECKPOINT_STAGE_ORDER))
    candidate_order = runtime_plan.execution.resume_predicate.get("candidate_order")
    if not isinstance(candidate_order, tuple) or not all(
        isinstance(stage, str) and stage for stage in candidate_order
    ):
        raise ValueError(
            "Bundle checkpoint resume predicate has no sealed candidate order."
        )
    return candidate_order


def _stacked_checkpoint_filenames(
    runtime_plan: USPoolRuntimePlan | None,
) -> Mapping[str, str]:
    if runtime_plan is None:
        return _POOL_STAGE_CHECKPOINT_FILENAMES
    return {
        stage: f"{stage}.checkpoint.h5"
        for stage in _stacked_checkpoint_order(runtime_plan)
    }


def _stacked_pipeline(runtime_plan: USPoolRuntimePlan | None) -> str:
    if runtime_plan is None:
        return _STACKED_PIPELINE
    pipeline = runtime_plan.execution.pipeline.get("id")
    if not isinstance(pipeline, str) or not pipeline:
        raise ValueError("Bundle execution authority has no pipeline id.")
    return pipeline


def _stacked_checkpoint_static_value(
    runtime_plan: USPoolRuntimePlan,
    field_name: str,
) -> object:
    if field_name not in runtime_plan.execution.checkpoint_static_components:
        raise ValueError(f"Bundle checkpoint authority has no {field_name!r} field.")
    return runtime_plan.execution.checkpoint_static_components[field_name]


def _stacked_period(runtime_plan: USPoolRuntimePlan | None) -> int:
    if runtime_plan is None:
        return POOL_TIME_PERIOD
    period = _stacked_checkpoint_static_value(runtime_plan, "period")
    if isinstance(period, bool) or not isinstance(period, int):
        raise ValueError("Bundle checkpoint authority has an invalid period.")
    return period


def _stacked_model_seed(runtime_plan: USPoolRuntimePlan | None) -> int:
    if runtime_plan is None:
        return POOL_RANDOM_SEED
    model_seed = _stacked_checkpoint_static_value(runtime_plan, "model_seed")
    if isinstance(model_seed, bool) or not isinstance(model_seed, int):
        raise ValueError("Bundle checkpoint authority has an invalid model seed.")
    return model_seed


def _stacked_operator_order(
    runtime_plan: USPoolRuntimePlan | None,
) -> list[str]:
    if runtime_plan is None:
        return list(US_STACKED_POOL_OPERATOR_ORDER)
    return [operation.id for operation in runtime_plan.execution.operations]


def _stacked_authority(
    runtime_plan: USPoolRuntimePlan | None,
) -> dict[str, object]:
    if runtime_plan is None:
        return stacked_spine_authority_receipt()
    authority = _json_ready(runtime_plan.execution.stacked_authority)
    if not isinstance(authority, dict):  # pragma: no cover - plan invariant
        raise TypeError("Bundle stacked authority must normalize to an object.")
    return authority


def _stacked_publication_runtime(
    runtime_plan: USPoolRuntimePlan,
) -> Mapping[str, object]:
    publication = runtime_plan.publication.runtime
    if not isinstance(publication, Mapping):  # pragma: no cover - plan invariant
        raise TypeError("Bundle publication authority must be an object.")
    return publication


def _stacked_rung(
    sample_fraction: float,
    *,
    runtime_plan: USPoolRuntimePlan | None = None,
) -> str:
    if runtime_plan is not None:
        publication = _stacked_publication_runtime(runtime_plan)
        rows = publication.get("rung_fractions")
        if not isinstance(rows, tuple):
            raise ValueError("Bundle publication authority has no sealed rung rows.")
        matches = [
            row.get("token")
            for row in rows
            if isinstance(row, Mapping)
            and row.get("fraction") == float(sample_fraction)
        ]
        if len(matches) != 1 or not isinstance(matches[0], str) or not matches[0]:
            raise ValueError(
                "Stacked sample_fraction must name exactly one bundle-declared rung."
            )
        return matches[0]
    try:
        return _STACKED_SAMPLE_RUNG_TOKENS[float(sample_fraction)]
    except KeyError as exc:
        raise ValueError(
            "Stacked sample_fraction must be one standard rung: 0.01, 0.04, 0.10, or 1.0."
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
    run_config: Mapping[str, object] | None = None,
    runtime_plan: USPoolRuntimePlan | None = None,
) -> dict[str, object]:
    """Bind #599/#608 caches to the live stack and both scale controls."""

    if run_config is not None:
        _stacked_kernel_authorities_from_config(
            run_config,
            runtime_plan=runtime_plan,
        )
    if runtime_plan is not None:
        if policyengine_us_version is not None:
            expected_version = _stacked_checkpoint_static_value(
                runtime_plan,
                "policyengine_us_version",
            )
            if policyengine_us_version != expected_version:
                raise ValueError(
                    "Explicit PolicyEngine US version differs from bundle authority."
                )
        return materialize_stacked_checkpoint_base_identity(
            runtime_plan,
            input_pins=_verified_input_pins_payload(verified_inputs),
            stack_receipt=stack_receipt,
            sample_fraction=sample_fraction,
            sample_seed=sample_seed,
            clone_attachment_fraction=clone_attachment_fraction,
            clone_attachment_seed=clone_attachment_seed,
        )

    fraction_token = _stacked_rung(sample_fraction)
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
        **stacked_checkpoint_artifact_protocol_identity(),
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
            "operator_order": list(US_STACKED_POOL_OPERATOR_ORDER),
            "pre_clone_source_operator_order": list(
                POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER
            ),
            "gap_fill_producer_schedule": (
                stacked_gap_fill_producer_schedule_receipt()
            ),
            "post_clone_source_operator_order": list(
                POOL_POST_CLONE_SOURCE_OPERATOR_ORDER
            ),
            "late_producer_schedule": _json_ready(us_late_producer_schedule_receipt()),
            "late_producer_resource_semantics": (
                stacked_late_producer_resource_semantics_receipt(
                    clone_attachment_fraction=clone_attachment_fraction,
                    clone_attachment_seed=clone_attachment_seed,
                    primary_seed=POOL_RANDOM_SEED,
                    primary_n_estimators=_PRIMARY_QRF_N_ESTIMATORS,
                    transfer_seed=POOL_RANDOM_SEED,
                    transfer_n_estimators=_ACS_TRANSFER_N_ESTIMATORS,
                    transfer_max_targets_per_fit=(
                        DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
                    ),
                )
            ),
            "derive_operator_order": list(POOL_DERIVE_OPERATOR_ORDER),
            "remaining_stage_input_manifest": (
                pool_remaining_stage_input_manifest_receipt()
            ),
            "primary_qrf_target_order": list(PRIMARY_QRF_TARGET_ORDER),
            "primary_qrf_checkpoint_schema_version": (
                PRIMARY_QRF_CHECKPOINT_SCHEMA_VERSION
            ),
            "acs_pums_earnings_universe_contract": (
                acs_pums_earnings_universe_contract_identity()
            ),
            "us_qbi_reconciliation_contract": (
                us_qbi_reconciliation_contract_identity()
            ),
            "take_up_contract": take_up_contract_identity(),
            "puf_capital_gains_tail_manifest_schema_version": (
                PUF_CAPITAL_GAINS_TAIL_MANIFEST_SCHEMA_VERSION
            ),
            "puf_capital_gains_tail_support_contract": (
                puf_capital_gains_tail_support_contract_identity()
            ),
            "primary_qrf_n_estimators": _PRIMARY_QRF_N_ESTIMATORS,
            "acs_transfer_n_estimators": _ACS_TRANSFER_N_ESTIMATORS,
            "acs_transfer_max_targets_per_fit": (
                DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
            ),
            "simulation_household_batch_size": (POOL_SIMULATION_HOUSEHOLD_BATCH_SIZE),
        },
    }


def _configured_stacked_identity(
    args: argparse.Namespace,
    *,
    runtime_plan: USPoolRuntimePlan | None = None,
    run_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Identity available before input verification for terminal error rows."""

    fraction_token = _stacked_rung(
        args.sample_fraction,
        runtime_plan=runtime_plan,
    )
    result = {
        "artifact_kind": "populace_us_stacked_pool_configured_identity",
        "schema_version": 1,
        "pipeline": _stacked_pipeline(runtime_plan),
        "period": _stacked_period(runtime_plan),
        "expected_input_pins_digest": _configured_input_pins_digest(args),
        "sample_fraction": float(args.sample_fraction),
        "fraction_token": fraction_token,
        "sample_seed": args.sample_seed,
        "clone_attachment_fraction": float(args.clone_attachment_fraction),
        "clone_attachment_seed": args.clone_attachment_seed,
        "stacked_authority": _stacked_authority(runtime_plan),
    }
    if run_config is not None:
        receipt = _stacked_run_config_receipt(run_config)
        authority_mode = receipt.get("config_authority")
        if authority_mode == "constants_adapter":
            return result
        if authority_mode not in {"constants", "bundle"}:
            raise ValueError("Stacked configured identity has an unknown authority.")
        generation = receipt.get("identity_generation")
        provenance = receipt.get("run_provenance_identity")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation not in {0, 1}
            or not isinstance(provenance, Mapping)
            or provenance.get("identity_generation") != generation
        ):
            raise ValueError(
                "Stacked configured identity requires a coherent run provenance."
            )
        result["identity_generation"] = generation
        result["spec_binding"] = provenance.get("spec_binding")
    return result


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
    runtime_plan: USPoolRuntimePlan | None = None,
    run_config: Mapping[str, object] | None = None,
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
    expected_run_config = (
        None if run_config is None else _stacked_run_config_receipt(run_config)
    )
    if runtime_plan is not None and (
        expected_run_config is None
        or expected_run_config.get("identity_generation") != 1
    ):
        raise ValueError(
            "Bundle checkpoint discovery requires generation-one run provenance."
        )
    checkpoint_order = _stacked_checkpoint_order(runtime_plan)
    checkpoint_filenames = _stacked_checkpoint_filenames(runtime_plan)
    for stage in _stacked_checkpoint_candidate_order(runtime_plan):
        checkpoint_path = root / checkpoint_filenames[stage]
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
            if manifest.get("run_config") != expected_run_config:
                raise ValueError("checkpoint manifest run provenance changed")
            stage_identity = manifest.get("identity")
            if not isinstance(stage_identity, Mapping):
                raise ValueError("checkpoint manifest identity is not an object")
            if stage_identity.get("stage") != stage or stage_identity.get(
                "stage_index"
            ) != checkpoint_order.index(stage):
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
                run_config=(run_config if runtime_plan is not None else None),
                runtime_plan=runtime_plan,
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
    runtime_plan: USPoolRuntimePlan | None = None,
) -> str:
    instant = datetime.now(UTC) if timestamp is None else timestamp.astimezone(UTC)
    suffix = uuid.uuid4().hex[:8] if nonce is None else nonce
    if not re.fullmatch(r"[0-9a-f]{8}", suffix):
        raise ValueError("Stacked release nonce must be eight lowercase hex digits.")

    if runtime_plan is None:
        fraction_token = _stacked_rung(sample_fraction)
        release_id = (
            f"{_STACKED_LEGACY_RELEASE_LINE}-stacked-{fraction_token}-s{sample_seed}-"
            f"asec{realized_asec_households}-acs{realized_acs_households}-"
            f"{instant.strftime('%Y%m%dT%H%M%SZ')}-{suffix}"
        )
        if not _STACKED_RELEASE_ID_PATTERN.fullmatch(release_id):  # pragma: no cover
            raise AssertionError(
                f"Generated invalid stacked release ID {release_id!r}."
            )
        return release_id

    publication = _stacked_publication_runtime(runtime_plan)
    fraction_token = _stacked_rung(
        sample_fraction,
        runtime_plan=runtime_plan,
    )
    line = publication.get("line")
    pattern = publication.get("pattern")
    compiled_regex = publication.get("compiled_regex")
    if not isinstance(line, Mapping) or not isinstance(line.get("value"), str):
        raise ValueError("Bundle publication authority has no release line.")
    if not isinstance(pattern, str) or not isinstance(compiled_regex, str):
        raise ValueError("Bundle publication authority has no release grammar.")
    release_id = pattern.format(
        line=line["value"],
        rung=fraction_token,
        seed=sample_seed,
        asec_households=realized_asec_households,
        acs_households=realized_acs_households,
        timestamp=instant.strftime("%Y%m%dT%H%M%SZ"),
        nonce=suffix,
    )
    if re.fullmatch(compiled_regex, release_id) is None:  # pragma: no cover
        raise AssertionError(f"Generated invalid stacked release ID {release_id!r}.")
    return release_id


def _pool_checkpoint_stage_identity(
    base_identity: Mapping[str, object],
    stage: str,
    *,
    runtime_plan: USPoolRuntimePlan | None = None,
) -> dict[str, object]:
    checkpoint_order = _stacked_checkpoint_order(runtime_plan)
    if stage not in checkpoint_order:
        raise ValueError(f"Unknown pool checkpoint stage {stage!r}.")
    return {
        **dict(base_identity),
        "stage": stage,
        "stage_index": checkpoint_order.index(stage),
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


def _stacked_kernel_authorities_from_config(
    run_config: Mapping[str, object] | None,
    *,
    runtime_plan: USPoolRuntimePlan | None,
) -> USPoolKernelAuthorities | None:
    """Recover the typed capability without placing it in receipt mappings."""

    candidate = getattr(run_config, "kernel_authorities", None)
    if runtime_plan is None:
        if candidate is not None:
            raise ValueError(
                "Kernel authorities require their sealed bundle runtime plan."
            )
        return None
    if not isinstance(candidate, USPoolKernelAuthorities):
        raise TypeError("Bundle execution requires USPoolKernelAuthorities.")
    if (
        candidate.authority_sha256 != runtime_plan.authority_sha256
        or candidate.spec_sha256 != runtime_plan.spec_sha256
    ):
        raise ValueError(
            "Bundle kernel authorities differ from the sealed runtime plan."
        )
    return candidate


class _PoolStageCheckpointStore:
    """Identity-guarded durable storage for the three pool cut points."""

    def __init__(
        self,
        root: Path,
        *,
        base_identity: Mapping[str, object],
        materializer_version: int | None = None,
        runtime_plan: USPoolRuntimePlan | None = None,
        run_config: Mapping[str, object] | None = None,
    ) -> None:
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise ValueError(
                f"Pool checkpoint root exists but is not a directory: {self.root}."
            )
        normalized_identity = _json_ready(base_identity)
        if not isinstance(normalized_identity, dict):  # pragma: no cover
            raise TypeError("Pool checkpoint base identity must be an object.")
        resolved_materializer_version = (
            POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
            if materializer_version is None
            else materializer_version
        )
        if (
            isinstance(resolved_materializer_version, bool)
            or not isinstance(resolved_materializer_version, int)
            or resolved_materializer_version < 1
        ):
            raise ValueError(
                "Pool checkpoint materializer_version must be a positive integer."
            )
        self._base_identity = normalized_identity
        self._materializer_version = resolved_materializer_version
        self._runtime_plan = runtime_plan
        self._kernel_authorities = _stacked_kernel_authorities_from_config(
            run_config,
            runtime_plan=runtime_plan,
        )
        self._run_config_receipt = (
            None if run_config is None else _stacked_run_config_receipt(run_config)
        )
        if runtime_plan is not None:
            if (
                self._run_config_receipt is None
                or self._run_config_receipt.get("identity_generation") != 1
            ):
                raise ValueError(
                    "Bundle checkpoint store requires generation-one run provenance."
                )
        self._stage_order = _stacked_checkpoint_order(runtime_plan)
        self._candidate_order = _stacked_checkpoint_candidate_order(runtime_plan)
        self._checkpoint_filenames = _stacked_checkpoint_filenames(runtime_plan)
        self._input_receipts: dict[str, object] | None = None
        self._resumed_from: str | None = None
        self._attempts: dict[str, dict[str, object]] = {
            stage: {"load_status": "not_attempted"} for stage in self._stage_order
        }
        self._writes: dict[str, dict[str, object]] = {}

    def _operational_sidecar_policy(self, stage: str) -> str | None:
        if self._runtime_plan is None:
            return None
        return self._runtime_plan.execution.require_checkpoint(
            stage
        ).operational_receipts_sidecar.value

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
            filename = self._checkpoint_filenames[stage]
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

        for stage in self._candidate_order:
            checkpoint = self._load(stage)
            if checkpoint is None:
                continue
            self._resumed_from = stage
            covered = self._stage_order[: self._stage_order.index(stage) + 1]
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
        if stage not in self._stage_order:
            raise ValueError(f"Unknown pool checkpoint stage {stage!r}.")
        if self._input_receipts is None:
            raise RuntimeError(
                "Pool checkpoint input receipts must be bound before writing."
            )
        persistent_frame = canonicalize_frame_string_dtypes(
            checkpoint.frame,
            boundary=f"pool {stage} checkpoint write",
        )
        qbi_route = _checkpoint_qbi_route(self._base_identity)
        if stage in {"transferred", "simulated"} and qbi_route == "stacked":
            if self._kernel_authorities is None:
                _validate_stacked_post_puf_stage_receipt(
                    persistent_frame,
                    checkpoint.stage_receipts,
                    boundary=f"pool {stage} durable checkpoint write",
                    transition_authority_sha256=(
                        checkpoint.late_producer_transition_authority_sha256
                    ),
                    require_live_output=stage == "transferred",
                )
            else:
                _validate_stacked_post_puf_stage_receipt(
                    persistent_frame,
                    checkpoint.stage_receipts,
                    boundary=f"pool {stage} durable checkpoint write",
                    transition_authority_sha256=(
                        checkpoint.late_producer_transition_authority_sha256
                    ),
                    require_live_output=stage == "transferred",
                    kernel_authorities=self._kernel_authorities,
                )
        if stage == "simulated" and qbi_route is not None:
            _validate_qbi_stage_receipt(
                persistent_frame,
                checkpoint.stage_receipts,
                route=qbi_route,
                boundary=f"pool {stage} durable checkpoint write",
                transition_authority_sha256=(
                    checkpoint.qbi_transition_authority_sha256
                ),
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
        sidecar_policy = self._operational_sidecar_policy(stage)
        if sidecar_policy == "forbidden" and operational_stage_receipts:
            raise ValueError(
                f"Pool checkpoint stage {stage!r} forbids operational receipts."
            )
        if sidecar_policy == "required" and not operational_stage_receipts:
            raise ValueError(
                f"Pool checkpoint stage {stage!r} requires operational receipts."
            )
        receipts_path = self.checkpoint_receipts_path(stage)
        try:
            receipts_path.unlink()
        except FileNotFoundError:
            pass
        else:
            _fsync_parent_directory(receipts_path)
        identity = _pool_checkpoint_stage_identity(
            self._base_identity,
            stage,
            runtime_plan=self._runtime_plan,
        )
        identity_sha256 = _pool_checkpoint_identity_sha256(identity)
        metadata = {
            "artifact_kind": _POOL_STAGE_CHECKPOINT_ARTIFACT_KIND,
            "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
            "materializer_version": self._materializer_version,
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
        if checkpoint.qbi_transition_authority_sha256 is not None:
            metadata["qbi_transition_authority_sha256"] = (
                checkpoint.qbi_transition_authority_sha256
            )
        if checkpoint.late_producer_transition_authority_sha256 is not None:
            metadata["late_producer_transition_authority_sha256"] = (
                checkpoint.late_producer_transition_authority_sha256
            )
        path = self.checkpoint_path(stage)
        started_at = time.perf_counter()
        write_frame_checkpoint(path, stored_frame, metadata=metadata)
        checkpoint_sha256 = _file_sha256(path)
        size_bytes = path.stat().st_size
        manifest_path = self.checkpoint_manifest_path(stage)
        manifest_payload = {
            "artifact_kind": _POOL_STAGE_CHECKPOINT_MANIFEST_ARTIFACT_KIND,
            "schema_version": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
            "materializer_version": self._materializer_version,
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
            **(
                {
                    "qbi_transition_authority_sha256": metadata[
                        "qbi_transition_authority_sha256"
                    ]
                }
                if "qbi_transition_authority_sha256" in metadata
                else {}
            ),
            **(
                {
                    "late_producer_transition_authority_sha256": metadata[
                        "late_producer_transition_authority_sha256"
                    ]
                }
                if "late_producer_transition_authority_sha256" in metadata
                else {}
            ),
        }
        if self._run_config_receipt is not None:
            manifest_payload["run_config"] = self._run_config_receipt
        _atomic_write_json(manifest_path, manifest_payload)
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
                    "materializer_version": self._materializer_version,
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
            else self._stage_order.index(self._resumed_from)
        )
        stages: dict[str, dict[str, object]] = {}
        for stage_index, stage in enumerate(self._stage_order):
            identity = _pool_checkpoint_stage_identity(
                self._base_identity,
                stage,
                runtime_plan=self._runtime_plan,
            )
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
                "materializer_version": self._materializer_version,
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
                            runtime_plan=self._runtime_plan,
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
            "materializer_version": self._materializer_version,
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
            runtime_plan=self._runtime_plan,
        )
        expected_identity_sha256 = _pool_checkpoint_identity_sha256(expected_identity)
        try:
            manifest = _read_json_object(manifest_path)
            if (
                manifest.get("artifact_kind")
                != _POOL_STAGE_CHECKPOINT_MANIFEST_ARTIFACT_KIND
                or manifest.get("schema_version")
                != POOL_STAGE_CHECKPOINT_SCHEMA_VERSION
                or manifest.get("materializer_version") != self._materializer_version
                or manifest.get("stage") != stage
            ):
                raise ValueError(
                    f"{stage} checkpoint manifest has an unsupported binding"
                )
            observed_run_config = manifest.get("run_config")
            if self._run_config_receipt is None:
                if observed_run_config is not None:
                    raise ValueError(
                        f"{stage} checkpoint manifest has unexpected run provenance"
                    )
            elif observed_run_config != self._run_config_receipt:
                raise ValueError(f"{stage} checkpoint manifest run provenance changed")
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
                expected_materializer_version=self._materializer_version,
            )
            for key in (
                "row_counts",
                "frame_schema",
                "frame_metadata",
                "qbi_transition_authority_sha256",
                "late_producer_transition_authority_sha256",
            ):
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
            qbi_transition_authority_sha256 = metadata.get(
                "qbi_transition_authority_sha256"
            )
            late_producer_transition_authority_sha256 = metadata.get(
                "late_producer_transition_authority_sha256"
            )
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

            qbi_route = _checkpoint_qbi_route(self._base_identity)
            if stage in {"transferred", "simulated"} and qbi_route == "stacked":
                if self._kernel_authorities is None:
                    _validate_stacked_post_puf_stage_receipt(
                        persistent_frame,
                        restored_stage_receipts,
                        boundary=f"pool {stage} durable checkpoint load",
                        transition_authority_sha256=(
                            late_producer_transition_authority_sha256
                        ),
                        require_live_output=stage == "transferred",
                    )
                else:
                    _validate_stacked_post_puf_stage_receipt(
                        persistent_frame,
                        restored_stage_receipts,
                        boundary=f"pool {stage} durable checkpoint load",
                        transition_authority_sha256=(
                            late_producer_transition_authority_sha256
                        ),
                        require_live_output=stage == "transferred",
                        kernel_authorities=self._kernel_authorities,
                    )
            if stage == "simulated" and qbi_route is not None:
                _validate_qbi_stage_receipt(
                    persistent_frame,
                    restored_stage_receipts,
                    route=qbi_route,
                    boundary="pool simulated durable checkpoint load",
                    transition_authority_sha256=(qbi_transition_authority_sha256),
                )

            checkpoint = MultispinePoolCheckpoint(
                stage=stage,
                frame=persistent_frame,
                assembly_receipt=dict(assembly_receipt),
                stage_receipts=restored_stage_receipts,
                simulation_frame=simulation_frame,
                qbi_transition_authority_sha256=(qbi_transition_authority_sha256),
                late_producer_transition_authority_sha256=(
                    late_producer_transition_authority_sha256
                ),
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
        sidecar_policy = self._operational_sidecar_policy(stage)
        if not receipts_path.exists():
            if sidecar_policy == "required":
                raise ValueError(
                    f"Pool checkpoint stage {stage!r} requires its receipts sidecar."
                )
            return dict(canonical_stage_receipts), {
                **base_record,
                "load_status": "missing",
            }
        if sidecar_policy == "forbidden":
            raise ValueError(
                f"Pool checkpoint stage {stage!r} forbids a receipts sidecar."
            )
        if not receipts_path.is_file():
            error = ValueError("operational receipts sidecar must be a regular file")
            if sidecar_policy is not None:
                raise error
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
                or payload.get("materializer_version") != self._materializer_version
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
            if sidecar_policy == "required" and not operational:
                raise ValueError(
                    f"Pool checkpoint stage {stage!r} has an empty required sidecar."
                )
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
            if sidecar_policy is not None:
                raise
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
    expected_materializer_version: int,
) -> None:
    if (
        metadata.get("artifact_kind") != _POOL_STAGE_CHECKPOINT_ARTIFACT_KIND
        or metadata.get("schema_version") != POOL_STAGE_CHECKPOINT_SCHEMA_VERSION
        or metadata.get("materializer_version") != expected_materializer_version
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


def _stacked_gap_fill_authority_surface(
    kernel_authorities: USPoolKernelAuthorities,
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Project the exact early-transfer surface from compiler directions."""

    if not isinstance(kernel_authorities, USPoolKernelAuthorities):
        raise TypeError("kernel_authorities must be USPoolKernelAuthorities.")
    surface: dict[str, dict[str, tuple[str, ...]]] = {}
    for direction in kernel_authorities.gap_fill.directions:
        for entity, families in direction.target_families.items():
            for family, targets in families.items():
                entity_surface = surface.setdefault(entity, {})
                if family in entity_surface:
                    raise ValueError(
                        "Compiler gap-fill directions repeat target family "
                        f"{entity}.{family}."
                    )
                entity_surface[family] = tuple(targets)
    return surface


def _stacked_late_producer_bank_identity(
    checkpoint_identity: Mapping[str, object],
    *,
    producer_name: str,
    entity: str,
    family: str,
    ordered_targets: tuple[str, ...],
    schedule_receipt: Mapping[str, object] | None = None,
) -> dict[str, object]:
    schedule = (
        us_late_producer_schedule_receipt()
        if schedule_receipt is None
        else schedule_receipt
    )
    schedule_sha256 = schedule.get("schedule_sha256")
    payload_sha256 = schedule.get("payload_sha256")
    if not isinstance(schedule_sha256, str) or not isinstance(
        payload_sha256,
        str,
    ):
        raise ValueError("Late-producer schedule authority has no SHA-256 identity.")
    return {
        **_pool_checkpoint_stage_identity(checkpoint_identity, "transferred"),
        "stacked_transfer_stage": "late_producer_dag",
        "late_producer_dag_sha256": schedule_sha256,
        "late_producer_schedule_sha256": payload_sha256,
        "late_producer": {
            "name": producer_name,
            "entity": entity,
            "family": family,
            "ordered_targets": list(ordered_targets),
        },
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


def _validate_stacked_post_puf_stage_receipt(
    frame: Frame,
    stage_receipts: Mapping[str, Mapping[str, object]],
    *,
    boundary: str,
    transition_authority_sha256: str | None,
    require_live_output: bool,
    kernel_authorities: USPoolKernelAuthorities | None = None,
) -> None:
    """Require the complete DAG proof and both exact compatibility aliases."""

    if kernel_authorities is not None and not isinstance(
        kernel_authorities,
        USPoolKernelAuthorities,
    ):
        raise TypeError("kernel_authorities must be USPoolKernelAuthorities.")
    late_authority = (
        None if kernel_authorities is None else kernel_authorities.late_producers
    )
    runtime_authority = (
        None if kernel_authorities is None else kernel_authorities.terminal
    )
    qrf_authority = (
        None if kernel_authorities is None else kernel_authorities.primary_qrf
    )

    impute = stage_receipts.get("impute")
    if not isinstance(impute, Mapping):
        raise ValueError(
            f"{boundary}: stacked transferred receipts have no impute object."
        )
    dag_receipt = impute.get("stacked_late_producer_dag")
    if not isinstance(dag_receipt, Mapping):
        raise ValueError(
            f"{boundary}: stacked transferred receipts have no late-producer "
            "DAG object."
        )
    # Authenticate the canonical execution proof before consulting the
    # independently propagated live-frame anchor. This keeps malformed DAGs
    # attributable to their receipt defect even when their authority carrier
    # is also absent or corrupt.
    if kernel_authorities is None:
        validate_stacked_late_producer_receipt(dag_receipt, boundary=boundary)
    else:
        validate_stacked_late_producer_receipt(
            dag_receipt,
            boundary=boundary,
            late_authority=late_authority,
            runtime_authority=runtime_authority,
            qrf_authority=qrf_authority,
        )
    if not isinstance(transition_authority_sha256, str):
        raise ValueError(
            f"{boundary}: independently carried late-producer transition "
            "authority is absent."
        )
    if require_live_output:
        if kernel_authorities is None:
            validate_stacked_late_producer_receipt(
                dag_receipt,
                boundary=boundary,
                frame=frame,
                expected_transition_authority_sha256=(transition_authority_sha256),
            )
        else:
            validate_stacked_late_producer_receipt(
                dag_receipt,
                boundary=boundary,
                frame=frame,
                expected_transition_authority_sha256=(transition_authority_sha256),
                late_authority=late_authority,
                runtime_authority=runtime_authority,
                qrf_authority=qrf_authority,
            )
    else:
        if kernel_authorities is None:
            validate_stacked_late_producer_transition_authority(
                frame,
                dag_receipt,
                boundary=boundary,
                expected_transition_authority_sha256=(transition_authority_sha256),
            )
        else:
            validate_stacked_late_producer_transition_authority(
                frame,
                dag_receipt,
                boundary=boundary,
                expected_transition_authority_sha256=(transition_authority_sha256),
                late_authority=late_authority,
                runtime_authority=runtime_authority,
                qrf_authority=qrf_authority,
            )
    transfer_receipt = impute.get("stacked_post_puf_transfer")
    if not isinstance(transfer_receipt, Mapping):
        raise ValueError(
            f"{boundary}: stacked transferred receipts have no post-PUF "
            "transfer object."
        )
    if _json_ready(transfer_receipt) != _json_ready(
        dag_receipt.get("post_puf_transfer")
    ):
        raise ValueError(
            f"{boundary}: stacked post-PUF transfer alias differs from the "
            "late-producer DAG proof."
        )
    source_chain = impute.get("source_operator_chain")
    source_alias = (
        source_chain.get("late_dag_completion")
        if isinstance(source_chain, Mapping)
        else None
    )
    if _json_ready(source_alias) != _json_ready(dag_receipt.get("source_completion")):
        raise ValueError(
            f"{boundary}: stacked source-completion alias differs from the "
            "late-producer DAG proof."
        )


def _qbi_receipt_from_stage_receipts(
    stage_receipts: Mapping[str, Mapping[str, object]],
    *,
    route: str,
    boundary: str,
) -> Mapping[str, object]:
    """Resolve the exact legacy or stacked QBI receipt path without fallback."""

    derive = stage_receipts.get("derive")
    if not isinstance(derive, Mapping):
        raise ValueError(f"{boundary}: stage receipts have no derive object.")
    if route == "legacy":
        if "pool_derivation" in derive:
            raise ValueError(
                f"{boundary}: legacy QBI receipt used the stacked derive route."
            )
        receipt = derive.get("qbi_input_reconciliation")
    elif route == "stacked":
        pool_derivation = derive.get("pool_derivation")
        if not isinstance(pool_derivation, Mapping):
            raise ValueError(
                f"{boundary}: stacked derive receipts have no pool_derivation object."
            )
        if "qbi_input_reconciliation" in derive:
            raise ValueError(
                f"{boundary}: stacked QBI receipt also appears at the legacy route."
            )
        receipt = pool_derivation.get("qbi_input_reconciliation")
    else:  # pragma: no cover - internal callers pass a literal
        raise ValueError(f"{boundary}: unknown QBI receipt route {route!r}.")
    if not isinstance(receipt, Mapping):
        raise ValueError(
            f"{boundary}: stage receipts have no QBI reconciliation object."
        )
    return receipt


def _validate_qbi_stage_receipt(
    frame: Frame,
    stage_receipts: Mapping[str, Mapping[str, object]],
    *,
    route: str,
    boundary: str,
    transition_authority_sha256: str | None,
) -> None:
    receipt = _qbi_receipt_from_stage_receipts(
        stage_receipts,
        route=route,
        boundary=boundary,
    )
    validate_us_qbi_reconciliation_live_output(
        frame,
        receipt,
        boundary=boundary,
        expected_transition_authority_sha256=transition_authority_sha256,
        allowed_post_reconciliation_person_columns=(
            us_qbi_post_reconciliation_person_columns(stage_receipts.get("seed"))
        ),
    )


def _checkpoint_qbi_route(base_identity: Mapping[str, object]) -> str | None:
    artifact_kind = base_identity.get("artifact_kind")
    if artifact_kind == "populace_us_multispine_pool_checkpoint_identity":
        return "legacy"
    if artifact_kind == "populace_us_stacked_pool_checkpoint_identity":
        return "stacked"
    return None


def _emit_stacked_checkpoint(
    callback: Callable[[MultispinePoolCheckpoint], None] | None,
    *,
    stage: str,
    frame: Frame,
    assembly_receipt: Mapping[str, object],
    stage_receipts: Mapping[str, Mapping[str, object]],
    simulation_frame: Frame | None = None,
    qbi_transition_authority_sha256: str | None = None,
    late_producer_transition_authority_sha256: str | None = None,
    kernel_authorities: USPoolKernelAuthorities | None = None,
) -> None:
    if stage in {"transferred", "simulated"}:
        if kernel_authorities is None:
            _validate_stacked_post_puf_stage_receipt(
                frame,
                stage_receipts,
                boundary=f"stacked {stage} checkpoint emission",
                transition_authority_sha256=(late_producer_transition_authority_sha256),
                require_live_output=stage == "transferred",
            )
        else:
            _validate_stacked_post_puf_stage_receipt(
                frame,
                stage_receipts,
                boundary=f"stacked {stage} checkpoint emission",
                transition_authority_sha256=(late_producer_transition_authority_sha256),
                require_live_output=stage == "transferred",
                kernel_authorities=kernel_authorities,
            )
    if stage == "simulated":
        _validate_qbi_stage_receipt(
            frame,
            stage_receipts,
            route="stacked",
            boundary="stacked simulated checkpoint emission",
            transition_authority_sha256=qbi_transition_authority_sha256,
        )
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
            qbi_transition_authority_sha256=(qbi_transition_authority_sha256),
            late_producer_transition_authority_sha256=(
                late_producer_transition_authority_sha256
            ),
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
    runtime_plan: USPoolRuntimePlan | None = None,
    kernel_authorities: USPoolKernelAuthorities | None = None,
) -> StackedPoolBuildResult:
    """Run the fixed stacked production pipeline across #599 boundaries."""

    if runtime_plan is None:
        if kernel_authorities is not None:
            raise ValueError(
                "Kernel authorities require their sealed bundle runtime plan."
            )
        release_id_matches = _STACKED_RELEASE_ID_PATTERN.fullmatch(release_id)
    else:
        if not isinstance(kernel_authorities, USPoolKernelAuthorities):
            raise TypeError("Bundle execution requires USPoolKernelAuthorities.")
        if (
            kernel_authorities.authority_sha256 != runtime_plan.authority_sha256
            or kernel_authorities.spec_sha256 != runtime_plan.spec_sha256
        ):
            raise ValueError(
                "Bundle kernel authorities differ from the sealed runtime plan."
            )
        compiled_regex = _stacked_publication_runtime(runtime_plan).get(
            "compiled_regex"
        )
        if not isinstance(compiled_regex, str):
            raise ValueError("Bundle publication authority has no release regex.")
        release_id_matches = re.fullmatch(compiled_regex, release_id)
    if release_id_matches is None:
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
        qbi_transition_authority_sha256: str | None = None
        late_producer_transition_authority_sha256: str | None = None
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
        qbi_transition_authority_sha256 = resume.qbi_transition_authority_sha256
        late_producer_transition_authority_sha256 = (
            resume.late_producer_transition_authority_sha256
        )
        resume_stage = resume.stage
        if resume_stage in {"transferred", "simulated"}:
            if kernel_authorities is None:
                _validate_stacked_post_puf_stage_receipt(
                    current,
                    receipts,
                    boundary=f"stacked {resume_stage} checkpoint resume",
                    transition_authority_sha256=(
                        late_producer_transition_authority_sha256
                    ),
                    require_live_output=resume_stage == "transferred",
                )
            else:
                _validate_stacked_post_puf_stage_receipt(
                    current,
                    receipts,
                    boundary=f"stacked {resume_stage} checkpoint resume",
                    transition_authority_sha256=(
                        late_producer_transition_authority_sha256
                    ),
                    require_live_output=resume_stage == "transferred",
                    kernel_authorities=kernel_authorities,
                )
        if resume_stage == "simulated":
            _validate_qbi_stage_receipt(
                current,
                receipts,
                route="stacked",
                boundary="stacked simulated checkpoint resume",
                transition_authority_sha256=(qbi_transition_authority_sha256),
            )
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
        if kernel_authorities is None:
            prepared = prepare_multispine_source_inputs_for_clone(
                current,
                acs_rent_donor=acs_rent_donor,
            )
        else:
            prepared = prepare_multispine_source_inputs_for_clone(
                current,
                acs_rent_donor=acs_rent_donor,
                remaining_stage_authority=(kernel_authorities.physical.remaining_stage),
                simulation_settings=kernel_authorities.physical.simulation,
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
        gap_fill_directions = (
            stacked_gap_fill_plan()
            if kernel_authorities is None
            else kernel_authorities.gap_fill.directions
        )
        for direction in gap_fill_directions:
            target_banks[direction.name] = AcsTransferTargetBankStore(
                acs_transfer_checkpoint_dir / direction.name,
                identity=_stacked_direction_bank_identity(
                    checkpoint_identity,
                    direction_name=direction.name,
                ),
            )
        if kernel_authorities is None:
            gap_filled = gap_fill_stacked_spine(
                prepared.frame,
                seed=POOL_RANDOM_SEED,
                n_estimators=_ACS_TRANSFER_N_ESTIMATORS,
                max_targets_per_fit=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
                target_banks=target_banks,
            )
        else:
            gap_filled = gap_fill_stacked_spine(
                prepared.frame,
                seed=kernel_authorities.physical.model.model_seed,
                n_estimators=(kernel_authorities.physical.model.transfer_n_estimators),
                max_targets_per_fit=(
                    kernel_authorities.physical.model.max_targets_per_fit
                ),
                target_banks=target_banks,
                runtime_authority=kernel_authorities.terminal,
                gap_fill_authority=kernel_authorities.gap_fill,
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
        if kernel_authorities is None:
            late_target_banks = {
                group.name: AcsTransferTargetBankStore(
                    acs_transfer_checkpoint_dir
                    / "late_producer_dag"
                    / group.entity
                    / group.family,
                    identity=_stacked_late_producer_bank_identity(
                        checkpoint_identity,
                        producer_name=group.name,
                        entity=group.entity,
                        family=group.family,
                        ordered_targets=group.targets,
                    ),
                )
                for group in CANONICAL_US_LATE_TRANSFER_GROUPS
            }
        else:
            late_target_banks = {
                group.name: AcsTransferTargetBankStore(
                    acs_transfer_checkpoint_dir
                    / "late_producer_dag"
                    / group.entity
                    / group.family,
                    identity=_stacked_late_producer_bank_identity(
                        checkpoint_identity,
                        producer_name=group.name,
                        entity=group.entity,
                        family=group.family,
                        ordered_targets=group.targets,
                        schedule_receipt=(
                            kernel_authorities.late_producers.schedule_receipt
                        ),
                    ),
                )
                for group in kernel_authorities.late_producers.transfer_groups
            }

        def primary_puf_producer(primary_input: Frame):
            if kernel_authorities is None:
                produced = run_stacked_puf_pass(
                    primary_input,
                    puf_donor,
                    clone_attachment_fraction=clone_attachment_fraction,
                    clone_attachment_seed=clone_attachment_seed,
                    seed=POOL_RANDOM_SEED,
                    n_estimators=_PRIMARY_QRF_N_ESTIMATORS,
                    fit_records=fit_records,
                    tail_bound_diagnostics=tail_bound_diagnostics,
                    primary_qrf_checkpoint_dir=primary_qrf_checkpoint_dir,
                    primary_qrf_input_binding=primary_qrf_input_binding,
                )
            else:
                produced = run_stacked_puf_pass(
                    primary_input,
                    puf_donor,
                    clone_attachment_fraction=clone_attachment_fraction,
                    clone_attachment_seed=clone_attachment_seed,
                    seed=kernel_authorities.physical.model.model_seed,
                    n_estimators=(
                        kernel_authorities.physical.model.primary_n_estimators
                    ),
                    fit_records=fit_records,
                    tail_bound_diagnostics=tail_bound_diagnostics,
                    primary_qrf_checkpoint_dir=primary_qrf_checkpoint_dir,
                    primary_qrf_input_binding=primary_qrf_input_binding,
                    qrf_authority=kernel_authorities.primary_qrf,
                )
            produced_tail = produced.receipt.get("puf_capital_gains_tail_transfer")
            if not isinstance(produced_tail, Mapping):
                raise ValueError("Stacked PUF pass emitted no tail manifest.")
            validate_puf_capital_gains_tail_manifest(produced_tail)
            if not isinstance(produced.receipt.get("primary_puf_qrf"), Mapping):
                raise ValueError("Stacked PUF pass emitted no primary-QRF receipt.")
            mark_phase("puf_passed")
            return produced

        if kernel_authorities is None:
            primary_resource_receipts = stacked_late_primary_resource_receipts(
                puf_donor,
                primary_qrf_checkpoint_identity_sha256=(current_base_identity_sha256),
                clone_attachment_fraction=clone_attachment_fraction,
                clone_attachment_seed=clone_attachment_seed,
                seed=POOL_RANDOM_SEED,
                n_estimators=_PRIMARY_QRF_N_ESTIMATORS,
                fit_records_enabled=True,
                tail_bound_diagnostics_enabled=True,
            )
        else:
            primary_resource_receipts = stacked_late_primary_resource_receipts(
                puf_donor,
                primary_qrf_checkpoint_identity_sha256=(current_base_identity_sha256),
                clone_attachment_fraction=clone_attachment_fraction,
                clone_attachment_seed=clone_attachment_seed,
                seed=kernel_authorities.physical.model.model_seed,
                n_estimators=(kernel_authorities.physical.model.primary_n_estimators),
                fit_records_enabled=True,
                tail_bound_diagnostics_enabled=True,
                qrf_authority=kernel_authorities.primary_qrf,
            )
        primary_qrf_input_binding = stacked_late_primary_checkpoint_input_binding(
            primary_resource_receipts
        )
        if kernel_authorities is None:
            late_stage = run_stacked_late_producer_dag(
                gap_filled.frame,
                primary_puf_producer=primary_puf_producer,
                primary_resource_receipts=primary_resource_receipts,
                seed=POOL_RANDOM_SEED,
                n_estimators=_ACS_TRANSFER_N_ESTIMATORS,
                max_targets_per_fit=DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT,
                target_banks=late_target_banks,
            )
        else:
            late_stage = run_stacked_late_producer_dag(
                gap_filled.frame,
                primary_puf_producer=primary_puf_producer,
                primary_resource_receipts=primary_resource_receipts,
                seed=kernel_authorities.physical.model.model_seed,
                n_estimators=(kernel_authorities.physical.model.transfer_n_estimators),
                max_targets_per_fit=(
                    kernel_authorities.physical.model.max_targets_per_fit
                ),
                target_banks=late_target_banks,
                late_authority=kernel_authorities.late_producers,
                runtime_authority=kernel_authorities.terminal,
                qrf_authority=kernel_authorities.primary_qrf,
            )
        late_producer_transition_authority_sha256 = (
            late_stage.transition_authority_sha256
        )
        if kernel_authorities is None:
            validate_stacked_late_producer_receipt(
                late_stage.receipt,
                boundary="stacked cold-build late-producer DAG",
                frame=late_stage.frame,
                expected_transition_authority_sha256=(
                    late_producer_transition_authority_sha256
                ),
            )
        else:
            validate_stacked_late_producer_receipt(
                late_stage.receipt,
                boundary="stacked cold-build late-producer DAG",
                frame=late_stage.frame,
                expected_transition_authority_sha256=(
                    late_producer_transition_authority_sha256
                ),
                late_authority=kernel_authorities.late_producers,
                runtime_authority=kernel_authorities.terminal,
                qrf_authority=kernel_authorities.primary_qrf,
            )
        puf_result = late_stage.primary_puf_result
        puf_receipt = dict(puf_result.receipt)
        primary_qrf_receipt = puf_receipt.pop("primary_puf_qrf")
        if not isinstance(primary_qrf_receipt, Mapping):
            raise ValueError("Stacked PUF pass emitted no primary-QRF receipt.")
        primary_qrf_receipt = dict(primary_qrf_receipt)
        primary_qrf_receipt["identity_routing"] = primary_qrf_identity_routing
        qrf_manifest_path = (
            _primary_qrf_manifest_path(primary_qrf_checkpoint_dir)
            if kernel_authorities is None
            else primary_qrf_checkpoint_dir
            / kernel_authorities.primary_qrf.manifest_filename
        )
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
                "n_estimators": (
                    _PRIMARY_QRF_N_ESTIMATORS
                    if kernel_authorities is None
                    else kernel_authorities.physical.model.primary_n_estimators
                ),
                "tail_bound_diagnostics": tail_bound_diagnostics,
            }
        )
        tail_manifest = puf_receipt.pop("puf_capital_gains_tail_transfer")
        if not isinstance(tail_manifest, Mapping):
            raise ValueError("Stacked PUF pass emitted no tail manifest.")
        validate_puf_capital_gains_tail_manifest(tail_manifest)
        post_puf_transfer_receipt = late_stage.receipt.get("post_puf_transfer")
        if not isinstance(post_puf_transfer_receipt, Mapping):
            raise ValueError(
                "Stacked late-producer DAG emitted no post-PUF transfer receipt."
            )
        fit_records.extend(late_stage.transfer_result.fit_records)
        weights_audit = weights_audit_gate(fit_records)
        if not weights_audit.passed:
            raise ValueError(
                "Stacked imputation weights audit failed:\n  "
                + "\n  ".join(weights_audit.failures)
            )
        late_stage_preservation = assert_stacked_tail_cells_preserved(
            late_stage.frame,
            tail_manifest,
        )
        current = canonicalize_frame_string_dtypes(
            late_stage.frame,
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
                "late_dag_completion": dict(late_stage.source_completion_receipt),
            },
            "stacked_gap_fill": dict(gap_filled.receipt),
            "stacked_late_producer_dag": dict(late_stage.receipt),
            "stacked_post_puf_transfer": dict(post_puf_transfer_receipt),
            "primary_puf_qrf": primary_qrf_receipt,
            "puf_capital_gains_tail_transfer": dict(tail_manifest),
            "stacked_puf_pass": puf_receipt,
            "tail_preservation_after_late_producer_dag": late_stage_preservation,
            "acs_qrf_transfer": {
                "target_families": {
                    "early_gap_fill": (
                        _json_ready(CANONICAL_STACKED_GAP_FILL_SURFACE)
                        if kernel_authorities is None
                        else _json_ready(
                            _stacked_gap_fill_authority_surface(kernel_authorities)
                        )
                    ),
                    "post_puf_transfer": _json_ready(
                        CANONICAL_STACKED_POST_PUF_TRANSFER_SURFACE
                        if kernel_authorities is None
                        else kernel_authorities.terminal.post_puf_transfer_surface
                    ),
                },
                "n_estimators": (
                    _ACS_TRANSFER_N_ESTIMATORS
                    if kernel_authorities is None
                    else kernel_authorities.physical.model.transfer_n_estimators
                ),
                "max_targets_per_fit": (
                    DEFAULT_ACS_TRANSFER_MAX_TARGETS_PER_FIT
                    if kernel_authorities is None
                    else kernel_authorities.physical.model.max_targets_per_fit
                ),
                "target_bank": {
                    "identity_routing": gap_fill_identity_routing,
                    "directions": {
                        name: bank.receipt()
                        for name, bank in sorted(target_banks.items())
                    },
                    "late_producer_groups": {
                        name: bank.receipt()
                        for name, bank in sorted(late_target_banks.items())
                    },
                },
            },
            "weights_audit": GateReport((weights_audit,)).to_manifest(),
        }
        if kernel_authorities is None:
            _emit_stacked_checkpoint(
                checkpoint,
                stage="transferred",
                frame=current,
                assembly_receipt=assembly_receipt,
                stage_receipts=receipts,
                late_producer_transition_authority_sha256=(
                    late_producer_transition_authority_sha256
                ),
            )
        else:
            _emit_stacked_checkpoint(
                checkpoint,
                stage="transferred",
                frame=current,
                assembly_receipt=assembly_receipt,
                stage_receipts=receipts,
                late_producer_transition_authority_sha256=(
                    late_producer_transition_authority_sha256
                ),
                kernel_authorities=kernel_authorities,
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
        if kernel_authorities is None:
            derived = derive_multispine_pool_inputs(derivation_input)
        else:
            derived = derive_multispine_pool_inputs(
                derivation_input,
                remaining_stage_authority=(kernel_authorities.physical.remaining_stage),
            )
        qbi_transition_authority_sha256 = derived.qbi_transition_authority_sha256
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

        if kernel_authorities is None:
            seeded = seed_multispine_pool_inputs(current)
        else:
            seeded = seed_multispine_pool_inputs(
                current,
                remaining_stage_authority=(kernel_authorities.physical.remaining_stage),
                take_up_authority=kernel_authorities.physical.take_up,
                simulation_settings=kernel_authorities.physical.simulation,
            )
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

        if kernel_authorities is None:
            simulated = materialize_multispine_agreement_outputs(current)
        else:
            simulated = materialize_multispine_agreement_outputs(
                current,
                simulation_settings=kernel_authorities.physical.simulation,
            )
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
        if kernel_authorities is None:
            _emit_stacked_checkpoint(
                checkpoint,
                stage="simulated",
                frame=current,
                assembly_receipt=assembly_receipt,
                stage_receipts=receipts,
                simulation_frame=simulation_frame,
                qbi_transition_authority_sha256=(qbi_transition_authority_sha256),
                late_producer_transition_authority_sha256=(
                    late_producer_transition_authority_sha256
                ),
            )
        else:
            _emit_stacked_checkpoint(
                checkpoint,
                stage="simulated",
                frame=current,
                assembly_receipt=assembly_receipt,
                stage_receipts=receipts,
                simulation_frame=simulation_frame,
                qbi_transition_authority_sha256=(qbi_transition_authority_sha256),
                late_producer_transition_authority_sha256=(
                    late_producer_transition_authority_sha256
                ),
                kernel_authorities=kernel_authorities,
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

    if kernel_authorities is None:
        completeness = stacked_completeness_gate(
            simulation_frame,
            tail_manifest=tail_manifest,
        )
        battery = by_origin_battery(
            simulation_frame,
            tail_manifest=tail_manifest,
        )
    else:
        completeness = stacked_completeness_gate(
            simulation_frame,
            tail_manifest=tail_manifest,
            runtime_authority=kernel_authorities.terminal,
        )
        battery = by_origin_battery(
            simulation_frame,
            tail_manifest=tail_manifest,
            runtime_authority=kernel_authorities.terminal,
        )
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
        qbi_transition_authority_sha256=qbi_transition_authority_sha256,
        late_producer_transition_authority_sha256=(
            late_producer_transition_authority_sha256
        ),
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
    _validate_qbi_stage_receipt(
        result.frame,
        result.stage_receipts,
        route="legacy",
        boundary="legacy production manifest",
        transition_authority_sha256=(result.qbi_transition_authority_sha256),
    )
    status = "simulation_ready" if result.simulation_ready else "agreement_failed"
    puf_donor_receipt = input_receipts.get("puf_donor")
    if not isinstance(puf_donor_receipt, Mapping):
        raise ValueError("Pool input receipts have no PUF donor object.")
    return {
        "artifact_kind": "populace_us_multispine_pool_manifest",
        "schema_version": _LEGACY_POOL_MANIFEST_SCHEMA_VERSION,
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


def _stacked_run_config_receipt(
    run_config: Mapping[str, object] | None,
) -> dict[str, object]:
    normalized = _json_ready(
        {"config_authority": "constants"} if run_config is None else run_config
    )
    if not isinstance(normalized, dict):  # pragma: no cover - typed internal call
        raise TypeError("Stacked run_config must normalize to an object.")
    return normalized


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
    run_config: Mapping[str, object] | None = None,
    runtime_plan: USPoolRuntimePlan | None = None,
    source_broker_receipt: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the stacked-only manifest without changing the legacy envelope."""

    kernel_authorities = _stacked_kernel_authorities_from_config(
        run_config,
        runtime_plan=runtime_plan,
    )
    if kernel_authorities is None:
        _validate_stacked_post_puf_stage_receipt(
            result.frame,
            result.stage_receipts,
            boundary="stacked production manifest",
            transition_authority_sha256=(
                result.late_producer_transition_authority_sha256
            ),
            require_live_output=False,
        )
    else:
        _validate_stacked_post_puf_stage_receipt(
            result.frame,
            result.stage_receipts,
            boundary="stacked production manifest",
            transition_authority_sha256=(
                result.late_producer_transition_authority_sha256
            ),
            require_live_output=False,
            kernel_authorities=kernel_authorities,
        )
    _validate_qbi_stage_receipt(
        result.frame,
        result.stage_receipts,
        route="stacked",
        boundary="stacked production manifest",
        transition_authority_sha256=(result.qbi_transition_authority_sha256),
    )
    status = "simulation_ready" if result.simulation_ready else "gate_failed"
    puf_donor_receipt = input_receipts.get("puf_donor")
    if not isinstance(puf_donor_receipt, Mapping):
        raise ValueError("Stacked pool input receipts have no PUF donor object.")
    gates = _stacked_gate_payload(result)
    stack_manifest = _json_ready(result.stack_receipt)
    return {
        "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "pipeline": _stacked_pipeline(runtime_plan),
        "release_id": result.release_id,
        "status": status,
        "simulation_ready": result.simulation_ready,
        "publication_run_id": publication_run_id,
        "calibration_applied": False,
        "operator_order": _stacked_operator_order(runtime_plan),
        "period": _stacked_period(runtime_plan),
        "random_seed": _stacked_model_seed(runtime_plan),
        "run_config": _stacked_run_config_receipt(run_config),
        "source_broker_receipt": _json_ready(source_broker_receipt),
        "sampling": {
            "sample_fraction": float(sample_fraction),
            "fraction_token": _stacked_rung(
                sample_fraction,
                runtime_plan=runtime_plan,
            ),
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
        "late_producer_transition_authority_sha256": (
            result.late_producer_transition_authority_sha256
        ),
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
            "materializer_version": US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION,
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
    runtime_plan: USPoolRuntimePlan | None = None,
) -> dict[str, object]:
    return {
        "artifact_kind": US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND,
        "schema_version": POOL_MANIFEST_SCHEMA_VERSION,
        "pipeline": _stacked_pipeline(runtime_plan),
        "release_id": release_id,
        "status": "publication_in_progress",
        "simulation_ready": False,
        "publication_run_id": publication_run_id,
        "message": "stacked publication in progress",
        "pool_h5": {
            "path": str(outputs.pool_h5.resolve()),
            "artifact_kind": POOL_H5_ARTIFACT_KIND,
            "publication_run_id": publication_run_id,
            "materializer_version": US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION,
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
        "schema_version": _LEGACY_POOL_MANIFEST_SCHEMA_VERSION,
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
    _validate_qbi_stage_receipt(
        result.frame,
        result.stage_receipts,
        route="legacy",
        boundary="legacy publication entry",
        transition_authority_sha256=(result.qbi_transition_authority_sha256),
    )
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
            "materializer_version": (
                _LEGACY_POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION
            ),
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
        "schema_version": _LEGACY_POOL_MANIFEST_SCHEMA_VERSION,
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
    run_config: Mapping[str, object] | None = None,
    runtime_plan: USPoolRuntimePlan | None = None,
    source_broker_receipt: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Atomically publish the stacked input-only pool and terminal receipts."""

    kernel_authorities = _stacked_kernel_authorities_from_config(
        run_config,
        runtime_plan=runtime_plan,
    )
    if kernel_authorities is None:
        _validate_stacked_post_puf_stage_receipt(
            result.frame,
            result.stage_receipts,
            boundary="stacked publication entry",
            transition_authority_sha256=(
                result.late_producer_transition_authority_sha256
            ),
            require_live_output=False,
        )
    else:
        _validate_stacked_post_puf_stage_receipt(
            result.frame,
            result.stage_receipts,
            boundary="stacked publication entry",
            transition_authority_sha256=(
                result.late_producer_transition_authority_sha256
            ),
            require_live_output=False,
            kernel_authorities=kernel_authorities,
        )
    _validate_qbi_stage_receipt(
        result.frame,
        result.stage_receipts,
        route="stacked",
        boundary="stacked publication entry",
        transition_authority_sha256=(result.qbi_transition_authority_sha256),
    )
    publication_run_id = _new_publication_run_id()
    _atomic_write_json(
        outputs.manifest,
        _stacked_publication_tombstone(
            outputs,
            release_id=result.release_id,
            publication_run_id=publication_run_id,
            runtime_plan=runtime_plan,
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
        "pipeline": _stacked_pipeline(runtime_plan),
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
            period=_stacked_period(runtime_plan),
            artifact_kind=POOL_H5_ARTIFACT_KIND,
            publication_run_id=publication_run_id,
            materializer_version=US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION,
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
            run_config=run_config,
            runtime_plan=runtime_plan,
            source_broker_receipt=source_broker_receipt,
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
        # Walk fields directly rather than through dataclasses.asdict(), whose
        # recursive deepcopy cannot serialize immutable MappingProxyType
        # fields used by the generation-0 authority records.
        return {
            field.name: _json_ready(getattr(value, field.name))
            for field in fields(value)
        }
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


def _packaged_us_json(filename: str) -> dict[str, object]:
    """Read one generation-0 US compatibility object from package data."""

    resource = files("microcosm.build.us").joinpath(filename)
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Packaged US compatibility resource {filename!r} is not an object."
        )
    return payload


def _adapter_mapping(value: object, *, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Compiled legacy payload has no {location} object.")
    return value


def _live_constants_adapter_gate() -> dict[str, object]:
    """Return the live constants surfaces consumed by the stacked tool.

    This is an equality oracle only.  The tool continues to call the same
    generation-0 constructors after the assertion; F0 does not construct a
    bundle-mode authority or alter any checkpoint identity.
    """

    checkpoint_identity = _stacked_checkpoint_base_identity(
        {},
        stack_receipt={"sample_fraction": 1.0, "sample_seed": 578},
        sample_fraction=1.0,
        sample_seed=578,
        clone_attachment_fraction=1.0,
        clone_attachment_seed=578,
        policyengine_us_version=_policyengine_us_version(),
    )
    static_keys = (
        "artifact_kind",
        "schema_version",
        "materializer_version",
        "pipeline",
        "period",
        "model_seed",
        "policyengine_us_version",
        "stacked_authority",
        "pool_code",
    )
    authority_value = _json_ready(stacked_spine_authority_receipt())
    if not isinstance(authority_value, Mapping):  # pragma: no cover - live invariant
        raise TypeError("Live stacked authority receipt must be an object.")
    authority_receipt = dict(authority_value)
    rung_rows = [
        {
            "fraction": float(fraction),
            "token": token,
            "percent_basis_points": int(round(fraction * 10_000)),
        }
        for fraction, token in _STACKED_SAMPLE_RUNG_TOKENS.items()
    ]
    parser = _parser()
    household_mass_shares = dict(POOL_HOUSEHOLD_MASS_SHARES)
    if household_mass_shares != dict(DEFAULT_STACKED_HOUSEHOLD_MASS_SHARES):
        raise ValueError(
            "Pool and stacked-spine live household mass-share authorities differ."
        )
    return {
        "battery_contract": project_battery_legacy_contract(
            build_live_stacked_battery_contract(),
            authority_receipt=authority_receipt,
        ),
        "source_manifest": _packaged_us_json("source_stages.json"),
        "spine_assembly": {
            "mass_anchor_channel": BASE_ASEC_SUPPORT_CHANNEL,
            "household_mass_shares": household_mass_shares,
        },
        "spine_sampling": {
            "channels": [
                BASE_ASEC_SUPPORT_CHANNEL,
                ACS_STACKED_SUPPORT_CHANNEL,
            ],
            "fraction": {
                "default": float(parser.get_default("sample_fraction")),
                "rungs": rung_rows,
            },
            "seed": {"default": parser.get_default("sample_seed")},
            "exact_count_rule": EXACT_COUNT_RULE,
        },
        "publication_release": {
            "legacy_prefixes": [_STACKED_LEGACY_RELEASE_LINE],
            "rungs": list(_STACKED_SAMPLE_RUNG_TOKENS.values()),
            "legacy_compiled_regexes": [_STACKED_RELEASE_ID_PATTERN.pattern],
        },
        "support_spine": _packaged_us_json("support_spine.json"),
        "take_up_contract": _packaged_us_json("take_up_contract.json"),
        "take_up_contract_identity": take_up_contract_identity(),
        "stacked_authority_receipt": authority_receipt,
        "gap_fill_plan": _json_ready(stacked_gap_fill_plan()),
        "gap_fill_producer_schedule_receipt": _json_ready(
            stacked_gap_fill_producer_schedule_receipt()
        ),
        "late_producer_schedule_receipt": _json_ready(
            us_late_producer_schedule_receipt()
        ),
        "overlap_ownership": _json_ready(us_late_overlap_ownership_receipt()),
        "stacked_checkpoint_static_components": {
            key: checkpoint_identity[key] for key in static_keys
        },
    }


def _compiled_constants_adapter_gate(
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Select the corresponding named surfaces from the compiled payload."""

    imputation = _adapter_mapping(payload.get("imputation"), location="imputation")
    assembly = _adapter_mapping(
        payload.get("spine_assembly"),
        location="spine_assembly",
    )
    publication = _adapter_mapping(
        payload.get("publication_release"),
        location="publication_release",
    )
    release_line = _adapter_mapping(
        publication.get("line"),
        location="publication_release/line",
    )
    sampling = _adapter_mapping(
        payload.get("spine_sampling"),
        location="spine_sampling",
    )
    sampling_fraction = _adapter_mapping(
        sampling.get("fraction"),
        location="spine_sampling/fraction",
    )
    sampling_seed = _adapter_mapping(
        sampling.get("seed"),
        location="spine_sampling/seed",
    )
    return {
        "battery_contract": payload.get("battery_contract"),
        "source_manifest": payload.get("source_manifest"),
        "spine_assembly": {
            "mass_anchor_channel": assembly.get("mass_anchor_channel"),
            "household_mass_shares": assembly.get("household_mass_shares"),
        },
        "spine_sampling": {
            "channels": sampling.get("channels"),
            "fraction": {
                "default": sampling_fraction.get("default"),
                "rungs": sampling_fraction.get("rungs"),
            },
            "seed": {"default": sampling_seed.get("default")},
            "exact_count_rule": sampling.get("exact_count_rule"),
        },
        "publication_release": {
            "legacy_prefixes": release_line.get("legacy_prefixes"),
            "rungs": publication.get("rungs"),
            "legacy_compiled_regexes": publication.get("legacy_compiled_regexes"),
        },
        "support_spine": payload.get("support_spine"),
        "take_up_contract": payload.get("take_up_contract"),
        "take_up_contract_identity": payload.get("take_up_contract_identity"),
        "stacked_authority_receipt": payload.get("stacked_authority_receipt"),
        "gap_fill_plan": imputation.get("gap_fill_plan"),
        "gap_fill_producer_schedule_receipt": imputation.get(
            "gap_fill_producer_schedule_receipt"
        ),
        "late_producer_schedule_receipt": imputation.get(
            "late_producer_schedule_receipt"
        ),
        "overlap_ownership": imputation.get("overlap_ownership"),
        "stacked_checkpoint_static_components": payload.get(
            "stacked_checkpoint_static_components"
        ),
    }


def _stacked_run_provenance_request(
    args: argparse.Namespace,
    *,
    runtime_plan: USPoolRuntimePlan | None = None,
) -> dict[str, object]:
    """Materialize the concrete, behavior-relevant request receipt."""

    sample_fraction = float(getattr(args, "sample_fraction", 1.0))
    sample_seed = getattr(args, "sample_seed", 578)
    clone_fraction = float(getattr(args, "clone_attachment_fraction", 1.0))
    clone_seed = getattr(args, "clone_attachment_seed", 578)
    return {
        "pipeline": _stacked_pipeline(runtime_plan),
        "sample_fraction": sample_fraction,
        "fraction_token": _stacked_rung(
            sample_fraction,
            runtime_plan=runtime_plan,
        ),
        "sample_seed": sample_seed,
        "clone_attachment_fraction": clone_fraction,
        "clone_attachment_seed": clone_seed,
    }


def _stacked_code_inventory_digest(code_pin: str) -> str:
    """Bind both authority modes to the same committed builder inventory."""

    return _pool_checkpoint_identity_sha256({"git_code_pin": code_pin})


def _stacked_artifact_protocol_inventory(
    runtime_plan: USPoolRuntimePlan | None,
) -> dict[str, object]:
    """Compile back one common artifact-protocol inventory for D3."""

    if runtime_plan is None:
        checkpoint_identity = stacked_checkpoint_artifact_protocol_identity()
    else:
        checkpoint_identity = {
            field_name: _stacked_checkpoint_static_value(runtime_plan, field_name)
            for field_name in (
                "artifact_kind",
                "schema_version",
                "materializer_version",
                "pipeline",
            )
        }
    return {
        "checkpoint_identity": checkpoint_identity,
        "pool_h5_materializer": US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION,
        "pool_manifest": POOL_MANIFEST_SCHEMA_VERSION,
        "pool_checkpoint": POOL_STAGE_CHECKPOINT_SCHEMA_VERSION,
    }


def _stacked_authority_versions(
    runtime_plan: USPoolRuntimePlan | None,
    *,
    runtime_authority_sha256: str | None = None,
    execution_abi_sha256: str | None = None,
) -> dict[str, object]:
    """Return an aligned authority-version vocabulary for both generations."""

    authority = _stacked_authority(runtime_plan)
    version = authority.get("version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Stacked authority has no positive version.")
    checkpoint_protocol = _stacked_artifact_protocol_inventory(runtime_plan)[
        "checkpoint_identity"
    ]
    if not isinstance(checkpoint_protocol, Mapping):  # pragma: no cover - literal
        raise TypeError("Stacked checkpoint protocol must be an object.")
    materializer_version = checkpoint_protocol.get("materializer_version")
    if (
        isinstance(materializer_version, bool)
        or not isinstance(materializer_version, int)
        or materializer_version < 1
    ):
        raise ValueError("Stacked checkpoint protocol has no materializer version.")
    for field_name, value in (
        ("runtime_authority", runtime_authority_sha256),
        ("execution_abi", execution_abi_sha256),
    ):
        if value is not None and not _LOWERCASE_SHA256.fullmatch(value):
            raise ValueError(f"Stacked {field_name} digest is malformed.")
    return {
        "stacked_authority": version,
        "checkpoint_materializer": materializer_version,
        "runtime_authority": runtime_authority_sha256,
        "execution_abi": execution_abi_sha256,
    }


def _constants_run_provenance_identity(
    args: argparse.Namespace,
    *,
    code_pin: str,
) -> RunProvenanceIdentity:
    """Issue the typed, non-promotable generation-zero run receipt."""

    return build_run_provenance_identity(
        identity_generation=0,
        source_grammar_receipt=None,
        spec_binding=None,
        authority_versions=_stacked_authority_versions(None),
        code_inventory_digest=_stacked_code_inventory_digest(code_pin),
        artifact_protocol_inventory=_stacked_artifact_protocol_inventory(None),
        run_request=_stacked_run_provenance_request(
            args,
        ),
        execution_receipt={
            "authority_mode": "constants",
            "pipeline": _STACKED_PIPELINE,
            "code_pin": code_pin,
        },
    )


@dataclass(frozen=True, slots=True)
class _ResolvedStackedRunConfig(Mapping[str, object]):
    """Receipt view plus the non-serializable bundle runtime capability.

    Mapping consumers see only operational configuration receipt fields.  The
    compiler-issued capability is deliberately absent from iteration so it
    cannot enter generation-0 checkpoint, bank, semantic-artifact, or reuse
    identities merely because those paths serialize ``run_config``.
    """

    runtime_authorities: RuntimeAuthorities = field(repr=False, compare=False)
    runtime_plan: USPoolRuntimePlan = field(repr=False, compare=False)
    kernel_authorities: USPoolKernelAuthorities = field(
        repr=False,
        compare=False,
    )
    run_provenance_identity: RunProvenanceIdentity = field(
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self.runtime_authorities.identity_generation != 1:
            raise ValueError(
                "Bundle configuration requires runtime identity_generation 1."
            )
        if self.runtime_plan.identity_generation != 1:
            raise ValueError("Bundle runtime plan requires identity_generation 1.")
        if not isinstance(self.kernel_authorities, USPoolKernelAuthorities):
            raise TypeError("Bundle configuration requires USPoolKernelAuthorities.")
        if (
            self.runtime_plan.authority_sha256
            != self.runtime_authorities.authority_sha256
        ):
            raise ValueError("Bundle runtime plan differs from its runtime capability.")
        if self.runtime_plan.spec_sha256 != (
            self.runtime_authorities.spec_binding.spec_sha256
        ):
            raise ValueError(
                "Bundle runtime plan differs from its resolved spec binding."
            )
        if (
            self.kernel_authorities.authority_sha256
            != self.runtime_plan.authority_sha256
            or self.kernel_authorities.spec_sha256 != self.runtime_plan.spec_sha256
        ):
            raise ValueError(
                "Bundle kernel authorities differ from their sealed runtime plan."
            )
        if self.run_provenance_identity.identity_generation != 1:
            raise ValueError("Bundle run provenance requires identity_generation 1.")
        provenance_binding = self.run_provenance_identity.to_wire().get("spec_binding")
        if (
            not isinstance(provenance_binding, Mapping)
            or provenance_binding.get("spec_sha256") != self.runtime_plan.spec_sha256
            or provenance_binding.get("attestation") != "bundle-authoritative"
        ):
            raise ValueError(
                "Bundle run provenance differs from the compiler spec binding."
            )

    def source_preflight_provenance(self) -> dict[str, object]:
        """Issue the operational provenance bound to source broker access."""

        return self.run_provenance_identity.to_wire()

    def _receipt(self) -> dict[str, object]:
        return {
            "config_authority": "bundle",
            "spec_binding_status": "resolved",
            "identity_generation": self.runtime_authorities.identity_generation,
            "run_provenance_identity": self.run_provenance_identity.to_wire(),
        }

    def __getitem__(self, key: str) -> object:
        return self._receipt()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._receipt())

    def __len__(self) -> int:
        return len(self._receipt())


def _stacked_run_config(
    args: argparse.Namespace,
    *,
    code_pin: str = "unresolved-local-git-code-pin",
) -> dict[str, object] | _ResolvedStackedRunConfig:
    """Resolve one stacked configuration authority and its receipt view."""

    config_authority = getattr(args, "config_authority", "constants")
    if config_authority == "constants":
        provenance = _constants_run_provenance_identity(args, code_pin=code_pin)
        return {
            "config_authority": "constants",
            "spec_binding_status": "absent",
            "identity_generation": 0,
            "run_provenance_identity": provenance.to_wire(),
        }
    if config_authority == "bundle":
        resolved = load_bundle("us")
        compiled = compile_spec(resolved)
        runtime_authorities = compile_runtime_authorities(compiled)
        runtime_plan = USPoolRuntimePlan.from_spec_authority(
            compile_us_spec_authority(runtime_authorities)
        )
        kernel_authorities = USPoolKernelAuthorities.from_runtime_plan(runtime_plan)
        binding = runtime_authorities.spec_binding.to_wire()
        binding["attestation"] = "bundle-authoritative"
        execution_abi_sha256 = runtime_authorities.execution_abi.get("sha256")
        if not isinstance(execution_abi_sha256, str):  # pragma: no cover - compiler
            raise ValueError("Bundle execution ABI has no SHA-256 identity.")
        provenance = build_run_provenance_identity(
            identity_generation=runtime_authorities.identity_generation,
            source_grammar_receipt=runtime_authorities.grammar_receipt.to_wire(),
            spec_binding=binding,
            authority_versions=_stacked_authority_versions(
                runtime_plan,
                runtime_authority_sha256=runtime_authorities.authority_sha256,
                execution_abi_sha256=execution_abi_sha256,
            ),
            code_inventory_digest=_stacked_code_inventory_digest(code_pin),
            artifact_protocol_inventory=(
                _stacked_artifact_protocol_inventory(runtime_plan)
            ),
            run_request=_stacked_run_provenance_request(
                args,
                runtime_plan=runtime_plan,
            ),
            execution_receipt={
                "authority_mode": "bundle",
                "pipeline": _stacked_pipeline(runtime_plan),
                "code_pin": code_pin,
            },
        )
        return _ResolvedStackedRunConfig(
            runtime_authorities=runtime_authorities,
            runtime_plan=runtime_plan,
            kernel_authorities=kernel_authorities,
            run_provenance_identity=provenance,
        )
    if config_authority != "constants_adapter":
        raise ValueError(f"Unsupported config authority: {config_authority!r}.")

    resolved = load_bundle("us")
    compiled_payload = compile_to_legacy_payload(resolved)
    assert_legacy_payload_equal(
        _live_constants_adapter_gate(),
        _compiled_constants_adapter_gate(compiled_payload),
    )
    binding = resolved.spec_binding.to_wire()
    if binding.get("attestation") != "mirror-attested":  # pragma: no cover
        raise ValueError("F0 spec binding must remain mirror-attested.")
    return {
        "config_authority": "constants_adapter",
        "spec_binding_status": "resolved",
        "spec_binding": binding,
    }


def _requested_stacked_run_config(args: argparse.Namespace) -> dict[str, object]:
    """Return receipt state before any fallible authority resolution."""

    config_authority = getattr(args, "config_authority", "constants")
    result: dict[str, object] = {"config_authority": config_authority}
    if config_authority in {"constants_adapter", "bundle"}:
        result["spec_binding_status"] = "resolution_pending"
    return result


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
    _refuse_preexisting_resume_state(args, outputs)
    verified_inputs, acs_source_manifest = _verify_inputs(args, outputs)
    checkpoint_store = _PoolStageCheckpointStore(
        outputs.checkpoint_root,
        base_identity=_legacy_pool_checkpoint_base_identity(verified_inputs),
        materializer_version=_LEGACY_POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
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


def _local_artifact_reference(path: Path) -> str:
    """Render one exportable artifact reference without host-absolute paths.

    Recorded rows export to a public archive, so locations anchor to the
    owning checkout first, then the home directory, and never embed an
    absolute host path (which would leak local usernames and layout).
    """

    resolved = path.resolve()
    try:
        repo_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        ).resolve()
    except (OSError, subprocess.CalledProcessError):
        repo_root = None
    if repo_root is not None and resolved.is_relative_to(repo_root):
        return f"local://{resolved.relative_to(repo_root).as_posix()}"
    home = Path.home().resolve()
    if resolved.is_relative_to(home):
        return f"local://~/{resolved.relative_to(home).as_posix()}"
    return f"local://{resolved.as_posix().lstrip('/')}"


def _logbook_predecessor(args: argparse.Namespace) -> str | None:
    cli_value = args.logbook_prev_row_digest
    environment_value = os.environ.get("POPULACE_LOGBOOK_PREV_ROW_DIGEST")
    if cli_value is not None and environment_value not in {None, cli_value}:
        raise ValueError(
            "--logbook-prev-row-digest disagrees with POPULACE_LOGBOOK_PREV_ROW_DIGEST."
        )
    value = cli_value if cli_value is not None else environment_value
    if value is not None and not _LOWERCASE_SHA256.fullmatch(value):
        raise ValueError(
            "POPULACE_LOGBOOK_PREV_ROW_DIGEST must be a lowercase SHA-256."
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
    run_provenance_identity: Mapping[str, object] | None = None,
    runtime_plan: USPoolRuntimePlan | None = None,
) -> Path:
    result = record_build_attempt(
        build_id=state.build_id,
        ts=started_ts,
        pipeline=_stacked_pipeline(runtime_plan),
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
        run_provenance_identity=run_provenance_identity,
        spool_dir=outputs.pool_h5.parent / "logbook-spool",
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


@dataclass(frozen=True, slots=True)
class _F1EvidencePlan:
    """Pre-execution plan snapshot that owns certification collection."""

    plan_lock: Mapping[str, object]
    canonical_plan_lock: bytes
    runtime_plan: USPoolRuntimePlan
    kernel_authorities: USPoolKernelAuthorities


def _compile_f1_evidence_plan() -> _F1EvidencePlan:
    """Compile one sealed selector plan without driving constants mode."""

    compiled = compile_spec(load_bundle("us"))
    runtime_authorities = compile_runtime_authorities(compiled)
    runtime_plan = USPoolRuntimePlan.from_spec_authority(
        compile_us_spec_authority(runtime_authorities)
    )
    kernel_authorities = USPoolKernelAuthorities.from_runtime_plan(runtime_plan)
    lock = plan_lock_payload(compiled)
    execution_abi = lock.get("execution_abi")
    if (
        not isinstance(execution_abi, Mapping)
        or execution_abi.get("sha256") != runtime_plan.execution.abi_sha256
    ):
        raise ValueError("F1 evidence plan lock differs from its runtime plan.")
    return _F1EvidencePlan(
        plan_lock=lock,
        canonical_plan_lock=_canonical_json_bytes(lock),
        runtime_plan=runtime_plan,
        kernel_authorities=kernel_authorities,
    )


def _assert_f1_tracked_code_snapshot(code_pin: str) -> None:
    """Refuse certification evidence from a moving or tracked-dirty checkout."""

    if _git_code_pin() != code_pin:
        raise ValueError("F1 evidence code HEAD changed during the build.")
    repository = Path(__file__).resolve().parents[1]
    try:
        status = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"],
            cwd=repository,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError as error:
        raise RuntimeError("Could not verify the F1 tracked code snapshot.") from error
    if status.returncode == 1:
        raise ValueError("F1 evidence requires a tracked-clean checkout.")
    if status.returncode != 0:
        raise RuntimeError("Could not verify the F1 tracked code snapshot.")


def _freeze_f1_evidence_plan(
    *,
    code_pin: str,
    run_config: Mapping[str, object],
) -> _F1EvidencePlan:
    """Freeze the plan before source reads and bind bundle execution to it."""

    _assert_f1_tracked_code_snapshot(code_pin)
    evidence_plan = _compile_f1_evidence_plan()
    _assert_f1_tracked_code_snapshot(code_pin)
    if isinstance(run_config, _ResolvedStackedRunConfig):
        runtime_plan = run_config.runtime_plan
        kernel_authorities = run_config.kernel_authorities
        if (
            runtime_plan.authority_sha256 != evidence_plan.runtime_plan.authority_sha256
            or runtime_plan.spec_sha256 != evidence_plan.runtime_plan.spec_sha256
            or runtime_plan.execution.abi_sha256
            != evidence_plan.runtime_plan.execution.abi_sha256
            or kernel_authorities.authority_sha256
            != evidence_plan.kernel_authorities.authority_sha256
        ):
            raise ValueError(
                "Bundle execution authorities differ from the frozen F1 evidence plan."
            )
    return evidence_plan


def _assert_f1_evidence_plan_current(
    evidence_plan: _F1EvidencePlan,
    *,
    code_pin: str,
) -> None:
    """Refuse a plan/code change between authority selection and collection."""

    if not isinstance(evidence_plan, _F1EvidencePlan):
        raise TypeError("typed F1 evidence plan required")
    if _canonical_json_bytes(evidence_plan.plan_lock) != (
        evidence_plan.canonical_plan_lock
    ):
        raise ValueError("Frozen F1 evidence plan was mutated during the build.")
    _assert_f1_tracked_code_snapshot(code_pin)
    observed = _compile_f1_evidence_plan()
    _assert_f1_tracked_code_snapshot(code_pin)
    if observed.canonical_plan_lock != evidence_plan.canonical_plan_lock:
        raise ValueError("F1 evidence plan changed during the build.")


def _f1_artifact_locator_registry(
    *,
    plan_lock: Mapping[str, object],
    runtime_plan: USPoolRuntimePlan,
    kernel_authorities: USPoolKernelAuthorities,
    outputs: PoolBuildOutputs,
    checkpoint_store: _PoolStageCheckpointStore,
) -> tuple[
    ArtifactLocatorRegistry,
    PoolArtifactCoverageContract,
    Mapping[str, Path],
]:
    """Bind every collector locator to its construction-time physical output."""

    allowed_roots = tuple(
        sorted(
            {
                outputs.pool_h5.parent.resolve(),
                outputs.checkpoint_root.resolve(),
            },
            key=lambda path: path.as_posix(),
        )
    )
    registry = ArtifactLocatorRegistry(allowed_roots=allowed_roots)
    registry.bind_file("runtime_output:pool_h5", outputs.pool_h5)
    registry.bind_file("runtime_output:manifest", outputs.manifest)
    registry.bind_file(
        "runtime_output:agreement_diagnostics",
        outputs.agreement_diagnostics,
    )
    seed_stream_map = plan_lock.get("seed_stream_map")
    if not isinstance(seed_stream_map, Mapping):
        raise ValueError("F1 plan lock has no seed-stream map.")
    registry.bind_json("plan_lock:/seed_stream_map", seed_stream_map)

    for checkpoint in runtime_plan.execution.checkpoints:
        checkpoint_id = checkpoint.id
        registry.bind_file(
            f"checkpoint:{checkpoint_id}:payload",
            checkpoint_store.checkpoint_path(checkpoint_id),
        )
        registry.bind_file(
            f"checkpoint:{checkpoint_id}:manifest",
            checkpoint_store.checkpoint_manifest_path(checkpoint_id),
        )
        registry.bind_optional_file(
            f"checkpoint:{checkpoint_id}:receipts",
            checkpoint_store.checkpoint_receipts_path(checkpoint_id),
        )

    coverage_contract = compile_pool_artifact_coverage(
        runtime_plan,
        kernel_authorities,
    )
    roots_by_authority: dict[str, Path] = {
        **{
            f"gap_fill_direction:{direction.name}": (
                outputs.acs_transfer_checkpoint_dir / direction.name
            )
            for direction in kernel_authorities.physical.gap_fill.directions
        },
        f"producer_node:{kernel_authorities.physical.primary_qrf.node.id}": (
            outputs.primary_qrf_checkpoint_dir
        ),
        **{
            f"producer_node:{group.name}": (
                outputs.acs_transfer_checkpoint_dir
                / "late_producer_dag"
                / group.entity
                / group.family
            )
            for group in kernel_authorities.physical.late_producers.transfer_groups
        },
    }
    expected_authorities = {
        bank.authority_ref for bank in coverage_contract.target_banks
    }
    if set(roots_by_authority) != expected_authorities:
        raise ValueError(
            "F1 target-bank bindings differ from the sealed coverage authority: "
            f"missing={sorted(expected_authorities - set(roots_by_authority))}, "
            f"extra={sorted(set(roots_by_authority) - expected_authorities)}"
        )
    bank_roots = {
        bank.locator_ref: roots_by_authority[bank.authority_ref]
        for bank in coverage_contract.target_banks
    }
    for bank in coverage_contract.target_banks:
        registry.bind_directory(bank.locator_ref, bank_roots[bank.locator_ref])
    return registry, coverage_contract, bank_roots


def _emit_f1_evidence(
    path: Path,
    *,
    args: argparse.Namespace,
    code_pin: str,
    run_config: Mapping[str, object],
    evidence_plan: _F1EvidencePlan,
    outputs: PoolBuildOutputs,
    checkpoint_store: _PoolStageCheckpointStore,
) -> None:
    """Collect the sealed vector and emit honest post-publication evidence."""

    _assert_f1_evidence_plan_current(evidence_plan, code_pin=code_pin)
    plan_lock = evidence_plan.plan_lock
    runtime_plan = evidence_plan.runtime_plan
    kernel_authorities = evidence_plan.kernel_authorities
    registry, coverage_contract, bank_roots = _f1_artifact_locator_registry(
        plan_lock=plan_lock,
        runtime_plan=runtime_plan,
        kernel_authorities=kernel_authorities,
        outputs=outputs,
        checkpoint_store=checkpoint_store,
    )
    mode_value = run_config.get("config_authority")
    if mode_value not in {"constants", "bundle"}:
        raise ValueError("F1 evidence requires constants or bundle run config.")
    mode = str(mode_value)
    collected = collect_artifact_digests(
        plan_lock["execution_abi"],
        registry=registry,
        authority_mode=mode,
    )
    selector_coverage = validate_pool_artifact_coverage(
        coverage_contract,
        bank_roots=bank_roots,
    )
    artifact_locator_refs = tuple(
        sorted(
            {
                str(row.get("locator_ref"))
                for row in runtime_plan.execution.artifact_vector
            }
        )
    )
    coverage = complete_coverage_evidence(
        plan_lock,
        bound_locator_refs=artifact_locator_refs,
        node_reuse_ids=(),
        node_reuse_inventory_complete=False,
        selector_inventory_complete=(
            selector_coverage.container_member_coverage_complete
        ),
        calibration_scope_complete=False,
        selector_coverage_receipt=selector_coverage.to_wire(),
        calibration_scope_receipt={
            "domain": "microcosm.us-f1-calibration-scope-coverage.v1",
            "schema_version": 1,
            "calibration_scope_complete": False,
            "reason": "normative_artifact_vector_omits_calibration_weights",
        },
    )
    provenance = (
        run_config.run_provenance_identity
        if isinstance(run_config, _ResolvedStackedRunConfig)
        else _constants_run_provenance_identity(args, code_pin=code_pin)
    )
    _assert_f1_tracked_code_snapshot(code_pin)
    if _canonical_json_bytes(plan_lock) != evidence_plan.canonical_plan_lock:
        raise ValueError("Frozen F1 evidence plan changed during collection.")
    emit_f1_production_evidence(
        path,
        mode=mode,
        plan_lock=plan_lock,
        artifacts=collected.artifacts,
        receipt_surfaces=collected.receipts,
        run_provenance_identity=provenance,
        node_reuse_keys={},
        coverage=coverage,
    )


def _stacked_attempt_receipt_dir(
    outputs: PoolBuildOutputs,
    *,
    build_id: str,
) -> Path:
    """Return the immutable, build-scoped receipt directory beside output."""

    return outputs.pool_h5.parent / "logbook-receipts" / build_id


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
    run_config: Mapping[str, object] | None = None,
    runtime_plan: USPoolRuntimePlan | None = None,
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
            "pipeline": _stacked_pipeline(runtime_plan),
            "build_id": result.release_id,
            "run_config": _stacked_run_config_receipt(run_config),
            "terminal_gates": _stacked_gate_payload(result),
        },
    )
    return path


def _validate_stacked_seed(value: int, *, option: str) -> int:
    """Keep build RNGs and Logbook inside the signed-64-bit contract."""

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


def _promote_stacked_attempt_identity(
    state: _StackedAttemptState,
    *,
    stack_receipt: Mapping[str, object],
    checkpoint_identity: Mapping[str, object],
    sample_fraction: float,
    sample_seed: int,
    timestamp: datetime,
    runtime_plan: USPoolRuntimePlan | None = None,
) -> None:
    """Bind terminal receipts as soon as the live stack identity is known."""

    asec_count, acs_count = _stacked_realized_counts(stack_receipt)
    build_id = _new_stacked_release_id(
        sample_fraction=sample_fraction,
        sample_seed=sample_seed,
        realized_asec_households=asec_count,
        realized_acs_households=acs_count,
        timestamp=timestamp,
        runtime_plan=runtime_plan,
    )
    identity_digest = _pool_checkpoint_identity_sha256(checkpoint_identity)
    state.build_id = build_id
    state.identity_digest = identity_digest


def _main_stacked(args: argparse.Namespace) -> int:
    """Build the default stacked pipeline and emit one terminal Logbook row."""

    started_at = time.perf_counter()
    started_ts = datetime.now(UTC)
    outputs = _stacked_attempt_outputs(args)
    code_pin = "unresolved-local-git-code-pin"
    predecessor = args.logbook_prev_row_digest
    runtime_plan: USPoolRuntimePlan | None = None
    run_provenance_identity: Mapping[str, object] | None = None
    source_broker_receipts: list[Mapping[str, object]] = []
    source_snapshot_sessions: list[_BundleSourceSnapshotSession] = []
    source_snapshot_session: _BundleSourceSnapshotSession | None = None
    f1_evidence_path: Path | None = None
    f1_evidence_plan: _F1EvidencePlan | None = None
    rung = _stacked_rung(args.sample_fraction)
    logbook_seed: int | None = None
    run_config = _requested_stacked_run_config(args)
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
        code_pin = _git_code_pin()
        predecessor = _logbook_predecessor(args)
        state.input_pins_digest = _configured_input_pins_digest(args)
        logbook_seed = _validate_stacked_seed(
            args.sample_seed,
            option="--sample-seed",
        )
        _validate_stacked_seed(
            args.clone_attachment_seed,
            option="--clone-attachment-seed",
        )
        outputs = _stacked_output_paths(
            args.out,
            checkpoint_root=args.checkpoint_root,
        )
        f1_evidence_path = _validate_f1_evidence_request(args, outputs)
        _refuse_preexisting_resume_state(args, outputs)
        # Select and fully resolve configuration authority before consulting
        # authority-owned publication, checkpoint, and physical constructors.
        try:
            run_config = _stacked_run_config(args, code_pin=code_pin)
        except Exception:
            if run_config.get("spec_binding_status") == "resolution_pending":
                run_config = {
                    **run_config,
                    "spec_binding_status": "resolution_failed",
                }
            raise
        if isinstance(run_config, _ResolvedStackedRunConfig):
            runtime_plan = run_config.runtime_plan
        if f1_evidence_path is not None:
            f1_evidence_plan = _freeze_f1_evidence_plan(
                code_pin=code_pin,
                run_config=run_config,
            )
        provenance_value = run_config.get("run_provenance_identity")
        if isinstance(provenance_value, Mapping):
            run_provenance_identity = provenance_value
        rung = _stacked_rung(
            args.sample_fraction,
            runtime_plan=runtime_plan,
        )
        state.build_id = _new_stacked_release_id(
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
            realized_asec_households=0,
            realized_acs_households=0,
            timestamp=started_ts,
            runtime_plan=runtime_plan,
        )
        configured_identity = _configured_stacked_identity(
            args,
            runtime_plan=runtime_plan,
            run_config=run_config,
        )
        state.identity_digest = hashlib.sha256(
            _canonical_json_bytes(configured_identity)
        ).hexdigest()
        _append_phase(state, "configured")
        if isinstance(run_config, _ResolvedStackedRunConfig):
            verified_inputs, acs_source_manifest = _verify_inputs(
                args,
                outputs,
                bundle_plan=run_config.runtime_plan,
                run_provenance_identity=(run_config.source_preflight_provenance()),
                broker_receipt_sink=source_broker_receipts.append,
                source_snapshot_session_sink=source_snapshot_sessions.append,
            )
            if len(source_snapshot_sessions) > 1:
                raise ValueError(
                    "Bundle authority issued multiple source snapshot sessions."
                )
            if source_snapshot_sessions:
                source_snapshot_session = source_snapshot_sessions[0]
        else:
            verified_inputs, acs_source_manifest = _verify_inputs(args, outputs)
        state.input_pins_digest = _input_pins_digest(verified_inputs)
        if source_snapshot_session is None:
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
            runtime_plan=runtime_plan,
        )
        checkpoint_identity = _discover_stacked_checkpoint_identity(
            outputs.checkpoint_root,
            verified_inputs=verified_inputs,
            sample_fraction=args.sample_fraction,
            sample_seed=args.sample_seed,
            clone_attachment_fraction=args.clone_attachment_fraction,
            clone_attachment_seed=args.clone_attachment_seed,
            runtime_plan=runtime_plan,
            run_config=run_config,
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
                materializer_version=POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
                runtime_plan=runtime_plan,
                run_config=run_config,
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
                _promote_stacked_attempt_identity(
                    state,
                    stack_receipt=stack_receipt,
                    checkpoint_identity=checkpoint_identity,
                    sample_fraction=args.sample_fraction,
                    sample_seed=args.sample_seed,
                    timestamp=started_ts,
                    runtime_plan=runtime_plan,
                )
                _append_phase(state, "checkpoint_loaded")
                if resume.stage == "assembled":
                    if source_snapshot_session is None:
                        acs_rent_donor = load_acs_2022_rent_donor(args.acs_rent_h5)
                        puf_donor, _puf_donor_build = _load_puf_donor(args)
                    else:
                        acs_rent_donor, puf_donor = _load_bundle_resume_donors(
                            args,
                            source_session=source_snapshot_session,
                        )
                        _append_phase(state, "inputs_verified")
                    _validate_resumed_puf_donor(
                        puf_donor,
                        checkpoint_store.input_receipts,
                    )
                    _append_phase(state, "resume_donors_loaded")

        if resume is None:
            if source_snapshot_session is None:
                loaded = _load_inputs(
                    args,
                    acs_source_manifest=acs_source_manifest,
                )
            else:
                loaded = _load_bundle_inputs(
                    args,
                    acs_source_manifest=acs_source_manifest,
                    source_session=source_snapshot_session,
                )
                _append_phase(state, "inputs_verified")
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
            if isinstance(run_config, _ResolvedStackedRunConfig):
                stack = assemble_stacked_spine(
                    loaded.asec,
                    loaded.acs,
                    sample_fraction=args.sample_fraction,
                    sample_seed=args.sample_seed,
                    assembly_authority=run_config.kernel_authorities.assembly,
                )
            else:
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
                run_config=(run_config if runtime_plan is not None else None),
                runtime_plan=runtime_plan,
            )
            _promote_stacked_attempt_identity(
                state,
                stack_receipt=stack_receipt,
                checkpoint_identity=checkpoint_identity,
                sample_fraction=args.sample_fraction,
                sample_seed=args.sample_seed,
                timestamp=started_ts,
                runtime_plan=runtime_plan,
            )
            checkpoint_store = _PoolStageCheckpointStore(
                outputs.checkpoint_root,
                base_identity=checkpoint_identity,
                materializer_version=POOL_STAGE_CHECKPOINT_MATERIALIZER_VERSION,
                runtime_plan=runtime_plan,
                run_config=run_config,
            )
            outputs = _with_checkpoint_identity(
                outputs,
                base_identity_sha256=checkpoint_store.base_identity_sha256,
            )
            checkpoint_store.bind_input_receipts(_loaded_input_receipts(loaded))

        if (
            resume is not None
            and resume.stage != "assembled"
            and source_snapshot_session is not None
        ):
            with source_snapshot_session.open_snapshots():
                pass
            source_snapshot_session.complete()
            _append_phase(state, "inputs_verified")

        if (
            checkpoint_store is None
            or checkpoint_identity is None
            or stack_frame is None
            or stack_receipt is None
        ):  # pragma: no cover - cold/resume branches establish all four
            raise AssertionError("Stacked checkpoint routing did not initialize.")

        if isinstance(run_config, _ResolvedStackedRunConfig):
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
                runtime_plan=runtime_plan,
                kernel_authorities=run_config.kernel_authorities,
            )
        else:
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
                runtime_plan=runtime_plan,
            )
        terminal_receipt_path = _write_stacked_terminal_gate_receipt(
            result,
            outputs=outputs,
            run_config=run_config,
            runtime_plan=runtime_plan,
        )
        state.gate_verdicts = {
            gate.name: {
                "verdict": "passed" if gate.passed else "failed",
                "receipt": (
                    f"{_local_artifact_reference(terminal_receipt_path)}"
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
            run_config=run_config,
            runtime_plan=runtime_plan,
            source_broker_receipt=(
                source_broker_receipts[0] if source_broker_receipts else None
            ),
        )
        if f1_evidence_path is not None:
            if f1_evidence_plan is None:  # pragma: no cover - same guarded flow
                raise RuntimeError("F1 evidence plan was not frozen before execution.")
            _emit_f1_evidence(
                f1_evidence_path,
                args=args,
                code_pin=code_pin,
                run_config=run_config,
                evidence_plan=f1_evidence_plan,
                outputs=outputs,
                checkpoint_store=checkpoint_store,
            )
        _append_phase(state, "publication_completed")
        state.artifact_location = _local_artifact_reference(outputs.pool_h5)
    except Exception as error:
        if source_snapshot_session is not None and not source_snapshot_session.sealed:
            source_snapshot_session.abort()
        error_path = _stacked_error_receipt_path(
            outputs,
            build_id=state.build_id,
        )
        _atomic_write_json(
            error_path,
            {
                "artifact_kind": "populace_us_stacked_pool_error_receipt",
                "schema_version": 1,
                "pipeline": _stacked_pipeline(runtime_plan),
                "build_id": state.build_id,
                "run_config": _stacked_run_config_receipt(run_config),
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
                "receipt": f"{_local_artifact_reference(error_path)}#/error_type",
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
            seed=logbook_seed,
            disposition="failed",
            predecessor=predecessor,
            run_provenance_identity=run_provenance_identity,
            runtime_plan=runtime_plan,
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
        seed=logbook_seed,
        disposition=disposition,
        predecessor=predecessor,
        run_provenance_identity=run_provenance_identity,
        runtime_plan=runtime_plan,
    )
    if not result.simulation_ready:
        print(
            "US stacked terminal gates failed; diagnostics and a non-ready "
            f"manifest were written to {outputs.agreement_diagnostics} and "
            f"{outputs.manifest}."
        )
        print(f"Wrote Logbook row: {spool_path}")
        return 1
    print(f"Wrote simulation-ready stacked pool: {outputs.pool_h5}")
    print(f"Wrote stacked pool manifest: {outputs.manifest}")
    print(f"Wrote Logbook row: {spool_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch to the stacked default or explicit legacy two-spine path."""

    args = _parser().parse_args(argv)
    if args.legacy_two_spine:
        if getattr(args, "f1_evidence_out", None) is not None:
            raise ValueError("--f1-evidence-out is unavailable for --legacy-two-spine")
        config_authority = getattr(args, "config_authority", "constants")
        if config_authority != "constants":
            raise ValueError(
                f"--config-authority {config_authority} is available only "
                "for the stacked pipeline and cannot be combined with "
                "--legacy-two-spine."
            )
        return _main_legacy(args)
    return _main_stacked(args)


if __name__ == "__main__":
    raise SystemExit(main())
