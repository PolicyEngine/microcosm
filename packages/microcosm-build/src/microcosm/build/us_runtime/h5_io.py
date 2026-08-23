"""PolicyEngine-compatible US H5 I/O for nullable build artifacts.

The writer stores entity tables, household weights, the period, and a small
artifact metadata record.  Fixed-format pandas tables preserve nullable
object columns without filling or coercing measured values.  The companion
build manifest, rather than this consumer H5, owns stage receipts such as
``Frame.metadata`` and ``Frame.mass_log``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import (
    canonicalize_frame_string_dtypes,
    canonicalize_table_string_dtypes,
)
from microcosm.frame import (
    Frame,
    WeightKind,
    Weights,
    materialize_nullable_booleans_for_pytables,
    put_frame_table,
    read_frame_table,
)
from microcosm.frame.units import US_SCHEMA

__all__ = [
    "AuthenticatedPoolH5",
    "AuthenticatedPoolH5MismatchError",
    "LEGACY_NULLABLE_STAGING_ARTIFACT_KIND",
    "US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND",
    "US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION",
    "US_MULTISPINE_POOL_H5_ARTIFACT_KIND",
    "US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND",
    "US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION",
    "US_STACKED_POOL_OPERATOR_ORDER",
    "load_legacy_calibrated_us_h5",
    "load_simulation_ready_us_multispine_pool",
    "load_simulation_ready_us_multispine_pool_manifest",
    "read_nullable_us_h5_metadata",
    "write_nullable_us_h5",
]

LEGACY_NULLABLE_STAGING_ARTIFACT_KIND = "nullable_precalibration_staging_h5"
US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND = "populace_us_multispine_pool_manifest"
US_MULTISPINE_POOL_H5_ARTIFACT_KIND = "populace_us_multispine_input_pool"
US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND = (
    "populace_us_multispine_agreement_diagnostics"
)
# 8 binds the nullable-boolean-capable physical H5 materializer in both the
# stacked manifest receipt and the H5's frozen metadata key.
# 7 binds the complete late-producer resource semantics and removes the PUF
# callback's duplicate outer-order entry; the callback is a node inside the DAG.
# 6 additionally bound the independently carried late-producer transition
# authority and restores its immutable Frame-metadata anchor on H5 load.
# Schema 5 can authenticate the DAG receipt's structure, but cannot prove that
# the published receipt is the one authorized by the generating transition.
US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION = 8
US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION = 2
"""Stacked terminal H5 materializer; version 2 handles pandas BooleanDtype."""
_LEGACY_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION = 4
_METADATA_KEY = "_populace_staging_metadata"
_TIME_PERIOD_KEY = "_time_period"
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_STACKED_PIPELINE = "us-stacked-pool"
US_STACKED_POOL_OPERATOR_ORDER = (
    "assemble_stacked_spine",
    "prepare_multispine_source_inputs_for_clone",
    "gap_fill_stacked_spine",
    "run_stacked_late_producer_dag",
    "prepare_stacked_tail_derivation",
    "derive_multispine_pool_inputs",
    "seed_multispine_pool_inputs",
    "materialize_multispine_agreement_outputs",
    "stacked_completeness_gate",
    "by_origin_battery",
)
_LEGACY_POOL_OPERATOR_ORDER = (
    "assemble",
    "clone",
    "impute",
    "derive",
    "seed",
    "simulate",
    "agreement",
)
_LEGACY_POOL_CHECKPOINT_ARTIFACT_KIND = (
    "populace_us_multispine_pool_checkpoint_provenance"
)
_LEGACY_POOL_CHECKPOINT_SCHEMA_VERSION = 1
_LEGACY_POOL_CHECKPOINT_MATERIALIZER_VERSION = 4
_LEGACY_REQUIRED_STAGE_RECEIPTS = frozenset({"impute", "derive", "seed", "simulate"})
_STACKED_ONLY_MANIFEST_FIELDS = frozenset(
    {
        "pipeline",
        "release_id",
        "sampling",
        "clone_attachment",
        "input_pins_digest",
        "late_producer_transition_authority_sha256",
        "stack_manifest",
        "terminal_gates",
    }
)
_REQUIRED_STACKED_MANIFEST_FIELDS = frozenset(
    {
        "pipeline",
        "operator_order",
        "stage_receipts",
    }
)


def _stacked_manifest_markers(manifest: Mapping[str, object]) -> set[str]:
    """Return every top-level or nested field that proves stacked lineage."""

    markers = set(manifest) & set(_STACKED_ONLY_MANIFEST_FIELDS)
    operator_order = manifest.get("operator_order")
    if isinstance(operator_order, list) and any(
        operator
        in {
            "assemble_stacked_spine",
            "run_stacked_late_producer_dag",
            "by_origin_battery",
        }
        for operator in operator_order
    ):
        markers.add("operator_order[stacked]")
    stage_receipts = manifest.get("stage_receipts")
    impute = (
        stage_receipts.get("impute") if isinstance(stage_receipts, Mapping) else None
    )
    if isinstance(impute, Mapping) and set(impute) & {
        "stacked_late_producer_dag",
        "stacked_post_puf_transfer",
    }:
        markers.add("stage_receipts.impute[stacked]")
    pool_h5 = manifest.get("pool_h5")
    if isinstance(pool_h5, Mapping) and "materializer_version" in pool_h5:
        markers.add("pool_h5.materializer_version")
    return markers


def _validate_canonical_legacy_envelope(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
) -> None:
    """Require positive identity for the frozen schema-4 publication route."""

    failures: list[str] = []
    if manifest.get("operator_order") != list(_LEGACY_POOL_OPERATOR_ORDER):
        failures.append("operator_order")
    stage_receipts = manifest.get("stage_receipts")
    if not isinstance(stage_receipts, Mapping) or not (
        _LEGACY_REQUIRED_STAGE_RECEIPTS <= set(stage_receipts)
    ):
        failures.append("stage_receipts")
    checkpoints = manifest.get("stage_checkpoints")
    if not isinstance(checkpoints, Mapping):
        failures.append("stage_checkpoints")
    else:
        expected_checkpoint_identity = {
            "artifact_kind": _LEGACY_POOL_CHECKPOINT_ARTIFACT_KIND,
            "schema_version": _LEGACY_POOL_CHECKPOINT_SCHEMA_VERSION,
            "materializer_version": _LEGACY_POOL_CHECKPOINT_MATERIALIZER_VERSION,
        }
        if any(
            checkpoints.get(key) != value
            for key, value in expected_checkpoint_identity.items()
        ):
            failures.append("stage_checkpoints.identity")
        stages = checkpoints.get("stages")
        if isinstance(stages, Mapping) and any(
            not isinstance(receipt, Mapping)
            or receipt.get("materializer_version")
            != _LEGACY_POOL_CHECKPOINT_MATERIALIZER_VERSION
            for receipt in stages.values()
        ):
            failures.append("stage_checkpoints.stages")
    agreement = manifest.get("agreement_gate")
    gates = agreement.get("gates") if isinstance(agreement, Mapping) else None
    if not isinstance(gates, Mapping) or set(gates) != {"us_spine_agreement"}:
        failures.append("agreement_gate.us_spine_agreement")
    if failures:
        raise ValueError(
            f"US multispine pool manifest {manifest_path} is not a canonical "
            f"legacy envelope; invalid={sorted(failures)}."
        )


def _validated_pool_manifest_envelope(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
) -> str:
    """Classify only an unambiguous schema-bound legacy or stacked envelope."""

    schema_version = manifest.get("schema_version")
    markers = _stacked_manifest_markers(manifest)
    if schema_version == US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION:
        missing = _REQUIRED_STACKED_MANIFEST_FIELDS - set(manifest)
        if manifest.get("pipeline") != _STACKED_PIPELINE or missing:
            raise ValueError(
                f"US multispine pool manifest {manifest_path} has an "
                "ambiguous stacked envelope; "
                f"missing={sorted(missing)}."
            )
        return "stacked"
    if schema_version == _LEGACY_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION:
        if markers:
            raise ValueError(
                f"US multispine pool manifest {manifest_path} legacy envelope "
                f"carries stacked-only field(s) {sorted(markers)}."
            )
        _validate_canonical_legacy_envelope(manifest, manifest_path=manifest_path)
        return "legacy"
    raise ValueError(
        f"US multispine pool manifest {manifest_path} has an unsupported "
        "artifact binding."
    )


class AuthenticatedPoolH5MismatchError(RuntimeError):
    """A pool H5 no longer matches the bytes authenticated by its manifest."""


@dataclass(frozen=True)
class AuthenticatedPoolH5:
    """Immutable identity of the pool H5 authorized by one manifest buffer."""

    path: Path
    sha256: str
    size_bytes: int
    publication_run_id: str
    manifest_sha256: str

    def verified_digest(self, *, consumer: str) -> str:
        """Re-verify the pathname and return only the authenticated digest."""

        try:
            observed_sha256, observed_size_bytes = _file_sha256_and_size(self.path)
        except OSError as exc:
            raise AuthenticatedPoolH5MismatchError(
                "AuthenticatedPoolH5MismatchError: authenticated pool H5 "
                f"became unreadable at consumer {consumer!r}: {self.path}; "
                f"expected sha256={self.sha256}, size_bytes={self.size_bytes}."
            ) from exc
        if observed_sha256 != self.sha256 or observed_size_bytes != self.size_bytes:
            self._raise_mismatch(
                consumer=consumer,
                observed_sha256=observed_sha256,
                observed_size_bytes=observed_size_bytes,
            )
        return self.sha256

    def copy_verified_to(self, destination: str | Path, *, consumer: str) -> Path:
        """Atomically copy the authenticated bytes and reject a raced source."""

        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        try:
            self.verified_digest(consumer=f"{consumer} source preflight")
            _copy_file_bytes(self.path, temporary)
            observed_sha256, observed_size_bytes = _file_sha256_and_size(temporary)
            if observed_sha256 != self.sha256 or observed_size_bytes != self.size_bytes:
                self._raise_mismatch(
                    consumer=f"{consumer} copied bytes",
                    observed_sha256=observed_sha256,
                    observed_size_bytes=observed_size_bytes,
                )
            os.replace(temporary, destination)
        except AuthenticatedPoolH5MismatchError:
            temporary.unlink(missing_ok=True)
            raise
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise AuthenticatedPoolH5MismatchError(
                "AuthenticatedPoolH5MismatchError: authenticated pool H5 "
                f"became unreadable at {consumer!r}: {self.path}; expected "
                f"sha256={self.sha256}, size_bytes={self.size_bytes}."
            ) from exc
        return destination

    def _raise_mismatch(
        self,
        *,
        consumer: str,
        observed_sha256: str,
        observed_size_bytes: int,
    ) -> None:
        raise AuthenticatedPoolH5MismatchError(
            "AuthenticatedPoolH5MismatchError: authenticated pool H5 changed "
            f"at consumer {consumer!r}: {self.path}; expected "
            f"sha256={self.sha256}, size_bytes={self.size_bytes}, observed "
            f"sha256={observed_sha256}, size_bytes={observed_size_bytes}."
        )


def load_legacy_calibrated_us_h5(path: str | Path) -> Frame:
    """Load a legacy US single-year H5 as a calibrated-weight frame.

    Legacy PolicyEngine US artifacts do not expose typed weight provenance
    through ``USSingleYearDataset``.  This loader therefore preserves the
    historical builder contract and labels their household weights
    ``CALIBRATED``.  It is not the loader for the new pre-calibration
    multispine pool, whose importance-weight receipt lives in its manifest.
    """

    with pd.HDFStore(Path(path), mode="r") as store:
        tables = {
            entity: read_frame_table(store, entity) for entity in US_SCHEMA.entities
        }
    tables["household"] = tables["household"].copy()
    household_weights = (
        tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    )
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                household_weights,
                WeightKind.CALIBRATED,
            )
        },
    )
    return canonicalize_frame_string_dtypes(
        frame,
        boundary="legacy calibrated US H5 load",
        in_place=True,
    )


def load_simulation_ready_us_multispine_pool_manifest(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Validate and return one ready manifest bound to its H5 and diagnostics.

    The manifest is the readiness authority. A caller cannot treat an H5 as
    ready merely because it exists: the manifest, nested artifact receipts,
    H5 metadata, diagnostics, and file digests must all bind the same
    publication run. When ``expected_manifest_sha256`` is supplied, this
    function hashes and parses one byte buffer so a replacement cannot inherit
    the authenticated manifest identity.
    """

    manifest, _ = _load_authenticated_us_multispine_pool_manifest(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    return manifest


def _load_authenticated_us_multispine_pool_manifest(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[dict[str, object], AuthenticatedPoolH5]:
    """Return the validated manifest and its authenticated pool-H5 identity."""

    manifest_path = Path(path)
    manifest, manifest_sha256, _ = _read_json_object_with_identity(
        manifest_path,
        label="pool manifest",
        expected_sha256=expected_manifest_sha256,
    )
    if manifest.get("artifact_kind") != US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND:
        raise ValueError(
            f"US multispine pool manifest {manifest_path} has an unsupported "
            "artifact binding."
        )
    if (
        manifest.get("simulation_ready") is not True
        or manifest.get("status") != "simulation_ready"
    ):
        raise ValueError(
            f"US multispine pool manifest {manifest_path} is not simulation-ready."
        )
    envelope = _validated_pool_manifest_envelope(
        manifest,
        manifest_path=manifest_path,
    )
    expected_schema_version = (
        US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
        if envelope == "stacked"
        else _LEGACY_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
    )
    _validate_stacked_late_dag_manifest_binding(
        manifest,
        manifest_path=manifest_path,
    )
    checkpoint_provenance = _mapping(
        manifest.get("stage_checkpoints"),
        label=f"US multispine pool manifest {manifest_path}.stage_checkpoints",
    )
    checkpoint_agreement = _mapping(
        checkpoint_provenance.get("agreement"),
        label=(
            f"US multispine pool manifest {manifest_path}.stage_checkpoints.agreement"
        ),
    )
    if (
        checkpoint_agreement.get("source") != "always_fresh"
        or checkpoint_agreement.get("cached") is not False
        or checkpoint_agreement.get("terminal_verdict_persisted") is not False
    ):
        raise ValueError(
            f"US multispine pool manifest {manifest_path} does not bind the "
            "terminal agreement verdict as always-fresh."
        )
    publication_run_id = _publication_run_id(
        manifest.get("publication_run_id"),
        label=f"US multispine pool manifest {manifest_path}",
    )

    pool_receipt = _mapping(
        manifest.get("pool_h5"),
        label=f"US multispine pool manifest {manifest_path}.pool_h5",
    )
    if pool_receipt.get("artifact_kind") != US_MULTISPINE_POOL_H5_ARTIFACT_KIND:
        raise ValueError(
            f"US multispine pool manifest {manifest_path} names the wrong H5 "
            "artifact kind."
        )
    _require_matching_run_id(
        pool_receipt,
        publication_run_id,
        label=f"US multispine pool manifest {manifest_path}.pool_h5",
    )
    pool_path = _artifact_path(
        pool_receipt,
        label=f"US multispine pool manifest {manifest_path}.pool_h5",
    )
    pool_sha256, pool_size_bytes = _require_matching_sha256(
        pool_path,
        pool_receipt,
        label=f"US multispine pool manifest {manifest_path}.pool_h5",
        require_size=True,
    )
    h5_metadata = read_nullable_us_h5_metadata(pool_path)
    if h5_metadata.get("artifact_kind") != US_MULTISPINE_POOL_H5_ARTIFACT_KIND:
        raise ValueError(
            f"US multispine pool H5 {pool_path} has the wrong artifact kind."
        )
    if h5_metadata.get("publication_run_id") != publication_run_id:
        raise ValueError(
            f"US multispine pool H5 {pool_path} publication run ID does not "
            "match its manifest."
        )
    if envelope == "stacked":
        receipt_materializer = pool_receipt.get("materializer_version")
        h5_materializer = h5_metadata.get("materializer_version")
        if (
            type(receipt_materializer) is not int
            or receipt_materializer != US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
            or type(h5_materializer) is not int
            or h5_materializer != US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION
        ):
            raise ValueError(
                f"US stacked pool publication {manifest_path} does not bind "
                "the current H5 materializer version in both its manifest "
                f"receipt and H5 metadata: receipt={receipt_materializer!r}, "
                f"h5={h5_materializer!r}, expected="
                f"{US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION}."
            )
    elif (
        "materializer_version" in pool_receipt or "materializer_version" in h5_metadata
    ):
        raise ValueError(
            f"US multispine pool manifest {manifest_path} legacy envelope "
            "carries a stacked-only H5 materializer version."
        )

    diagnostics_receipt = _mapping(
        manifest.get("agreement_diagnostics"),
        label=(f"US multispine pool manifest {manifest_path}.agreement_diagnostics"),
    )
    _require_matching_run_id(
        diagnostics_receipt,
        publication_run_id,
        label=(f"US multispine pool manifest {manifest_path}.agreement_diagnostics"),
    )
    diagnostics_path = _artifact_path(
        diagnostics_receipt,
        label=(f"US multispine pool manifest {manifest_path}.agreement_diagnostics"),
    )
    _require_matching_sha256(
        diagnostics_path,
        diagnostics_receipt,
        label=(f"US multispine pool manifest {manifest_path}.agreement_diagnostics"),
    )
    diagnostics = _read_json_object(
        diagnostics_path,
        label="pool agreement diagnostics",
    )
    diagnostics_stacked_fields = set(diagnostics) & {
        "pipeline",
        "semantic_kind",
        "release_id",
        "terminal_gates",
    }
    if envelope == "stacked":
        if (
            diagnostics.get("pipeline") != _STACKED_PIPELINE
            or diagnostics.get("semantic_kind") != "stacked_terminal_gates"
        ):
            raise ValueError(
                f"US multispine pool diagnostics {diagnostics_path} have an "
                "ambiguous stacked envelope."
            )
    elif diagnostics_stacked_fields:
        raise ValueError(
            f"US multispine pool diagnostics {diagnostics_path} legacy "
            "envelope carries stacked-only field(s) "
            f"{sorted(diagnostics_stacked_fields)}."
        )
    if (
        diagnostics.get("artifact_kind")
        != US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND
        or diagnostics.get("schema_version") != expected_schema_version
        or diagnostics.get("simulation_ready") is not True
        or diagnostics.get("publication_run_id") != publication_run_id
    ):
        raise ValueError(
            f"US multispine pool diagnostics {diagnostics_path} do not match "
            "the ready manifest publication."
        )
    manifest_agreement_gate = _mapping(
        manifest.get("agreement_gate"),
        label=f"US multispine pool manifest {manifest_path}.agreement_gate",
    )
    diagnostics_agreement_gate = _mapping(
        diagnostics.get("agreement_gate"),
        label=f"US multispine pool diagnostics {diagnostics_path}.agreement_gate",
    )
    _require_matching_terminal_gate_aliases(
        manifest,
        diagnostics,
        manifest_path=manifest_path,
        diagnostics_path=diagnostics_path,
        manifest_agreement_gate=manifest_agreement_gate,
        diagnostics_agreement_gate=diagnostics_agreement_gate,
    )
    if diagnostics_agreement_gate != manifest_agreement_gate:
        raise ValueError(
            f"US multispine pool diagnostics {diagnostics_path} agreement-gate "
            "verdict does not match the ready manifest."
        )
    return manifest, AuthenticatedPoolH5(
        path=pool_path.resolve(),
        sha256=pool_sha256,
        size_bytes=pool_size_bytes,
        publication_run_id=publication_run_id,
        manifest_sha256=manifest_sha256,
    )


def _validate_stacked_late_dag_manifest_binding(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
) -> None:
    """Make schema-8 stacked consumers authenticate the published DAG proof."""

    if manifest.get("pipeline") != "us-stacked-pool":
        return
    if manifest.get("operator_order") != list(US_STACKED_POOL_OPERATOR_ORDER):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} does not bind the "
            "canonical late-DAG operator order."
        )
    stage_receipts = manifest.get("stage_receipts")
    impute = (
        stage_receipts.get("impute") if isinstance(stage_receipts, Mapping) else None
    )
    dag = (
        impute.get("stacked_late_producer_dag") if isinstance(impute, Mapping) else None
    )
    if not isinstance(dag, Mapping):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has no late-producer "
            "DAG receipt."
        )
    from microcosm.build.us_runtime.stacked_spine import (
        validate_stacked_late_producer_receipt,
    )

    validate_stacked_late_producer_receipt(
        dag,
        boundary=f"US stacked pool manifest {manifest_path}",
    )
    _stacked_late_transition_binding(
        manifest,
        manifest_path=manifest_path,
    )
    transfer_alias = impute.get("stacked_post_puf_transfer")
    source_chain = impute.get("source_operator_chain")
    source_alias = (
        source_chain.get("late_dag_completion")
        if isinstance(source_chain, Mapping)
        else None
    )
    if transfer_alias != dag.get("post_puf_transfer"):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} post-PUF transfer alias "
            "differs from its late-DAG proof."
        )
    if source_alias != dag.get("source_completion"):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} source-completion alias "
            "differs from its late-DAG proof."
        )


def _stacked_late_transition_binding(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
) -> tuple[Mapping[str, object], Mapping[str, object], str] | None:
    """Return the signed DAG, derived authority, and independent authority SHA."""

    if manifest.get("pipeline") != "us-stacked-pool":
        return None
    stage_receipts = manifest.get("stage_receipts")
    impute = (
        stage_receipts.get("impute") if isinstance(stage_receipts, Mapping) else None
    )
    dag = (
        impute.get("stacked_late_producer_dag") if isinstance(impute, Mapping) else None
    )
    if not isinstance(dag, Mapping):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has no late-producer "
            "DAG receipt."
        )
    from microcosm.build.us_runtime.stacked_spine import (
        _late_producer_transition_authority_receipt,
    )

    derived_authority = _late_producer_transition_authority_receipt(dag)
    expected_sha256 = derived_authority["sha256"]
    observed_sha256 = manifest.get("late_producer_transition_authority_sha256")
    if (
        not isinstance(observed_sha256, str)
        or _LOWERCASE_SHA256.fullmatch(observed_sha256) is None
        or observed_sha256 != expected_sha256
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} independently carried "
            "late-producer transition authority does not match its signed DAG "
            f"receipt; expected={expected_sha256!r}, observed={observed_sha256!r}."
        )
    return dag, derived_authority, observed_sha256


def load_simulation_ready_us_multispine_pool(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[Frame, dict[str, object], AuthenticatedPoolH5]:
    """Load a manifest-bound multispine pool with its importance weights.

    The companion manifest remains the readiness authority.  This loader first
    validates the manifest/H5/agreement publication triple, then reads the
    fixed-format entity tables and checks the H5 digest again after the read so
    bytes changed concurrently cannot be treated as the validated pool.  The
    pool's household weights retain their ``IMPORTANCE`` provenance; the
    legacy US loader deliberately labels historical datasets ``CALIBRATED``
    and is therefore not a valid consumer for this artifact.

    Returns:
        The reconstructed pool :class:`~microcosm.frame.Frame`, the exact
        manifest object whose artifact receipts authorized the load, and one
        immutable identity object for every downstream H5 consumer.
    """

    manifest_path = Path(path)
    manifest, authenticated_pool_h5 = _load_authenticated_us_multispine_pool_manifest(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    agreement_gate = _mapping(
        manifest.get("agreement_gate"),
        label=f"US multispine pool manifest {manifest_path}.agreement_gate",
    )
    if agreement_gate.get("passed") is not True:
        raise ValueError(
            f"US multispine pool manifest {manifest_path} has no passing "
            "agreement-gate verdict."
        )

    pool_path = authenticated_pool_h5.path
    metadata = read_nullable_us_h5_metadata(pool_path)
    stored_kind = metadata.get("household_weight_kind")
    if stored_kind != WeightKind.IMPORTANCE.value:
        raise ValueError(
            f"US multispine pool H5 {pool_path} must carry importance weights, "
            f"got {stored_kind!r}."
        )

    with pd.HDFStore(pool_path, mode="r") as store:
        keys = {key.lstrip("/") for key in store.keys()}
        missing = sorted(set(US_SCHEMA.entities) - keys)
        if missing:
            raise ValueError(
                f"US multispine pool H5 {pool_path} is missing entity table(s): "
                f"{missing}."
            )
        tables = {
            entity: canonicalize_table_string_dtypes(
                read_frame_table(store, entity),
                boundary="simulation-ready US pool H5 load",
                table_name=entity,
            )
            for entity in US_SCHEMA.entities
        }
        period = store[_TIME_PERIOD_KEY]

    if len(period) != 1 or period.tolist() != [manifest.get("period")]:
        raise ValueError(
            f"US multispine pool H5 {pool_path} period does not match its "
            f"manifest: H5={period.tolist()!r}, manifest={manifest.get('period')!r}."
        )
    household = tables["household"].copy()
    if "household_weight" not in household:
        raise ValueError(
            f"US multispine pool H5 {pool_path} household table has no "
            "household_weight column."
        )
    household_weights = household.pop("household_weight").to_numpy(dtype=np.float64)
    tables["household"] = household
    late_transition = _stacked_late_transition_binding(
        manifest,
        manifest_path=manifest_path,
    )
    frame_metadata: dict[str, object] = {}
    if late_transition is not None:
        _dag, transition_authority, _transition_authority_sha256 = late_transition
        from microcosm.build.us_runtime.stacked_spine import (
            US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY,
        )

        frame_metadata[US_LATE_PRODUCER_TRANSITION_AUTHORITY_KEY] = transition_authority
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                household_weights,
                WeightKind.IMPORTANCE,
            )
        },
        metadata=frame_metadata,
    )
    if late_transition is not None:
        dag, _transition_authority, transition_authority_sha256 = late_transition
        from microcosm.build.us_runtime.stacked_spine import (
            validate_stacked_late_producer_transition_authority,
        )

        validate_stacked_late_producer_transition_authority(
            frame,
            dag,
            boundary=f"US stacked pool H5 {pool_path}",
            expected_transition_authority_sha256=transition_authority_sha256,
        )

    provenance_counts = _mapping(
        manifest.get("provenance_counts"),
        label=f"US multispine pool manifest {manifest_path}.provenance_counts",
    )
    household_counts = _mapping(
        provenance_counts.get("household"),
        label=(
            f"US multispine pool manifest {manifest_path}.provenance_counts.household"
        ),
    )
    expected_households = household_counts.get("rows")
    if (
        isinstance(expected_households, bool)
        or not isinstance(expected_households, int)
        or expected_households != frame.n("household")
    ):
        raise ValueError(
            f"US multispine pool manifest {manifest_path} household row count "
            f"{expected_households!r} does not match H5 count "
            f"{frame.n('household')}."
        )

    # Close the validation/read time-of-check-to-time-of-use window.  A file
    # replacement during the HDF read must not inherit the first digest check.
    authenticated_pool_h5.verified_digest(
        consumer="pool loader post-HDF read",
    )
    return frame, manifest, authenticated_pool_h5


def read_nullable_us_h5_metadata(path: str | Path) -> dict[str, object]:
    """Read and validate the single JSON artifact-metadata row from an H5."""

    h5_path = Path(path)
    if not h5_path.is_file():
        raise FileNotFoundError(f"Nullable US H5 is not a file: {h5_path}")
    with pd.HDFStore(h5_path, mode="r") as store:
        try:
            raw_metadata = store[_METADATA_KEY]
        except KeyError as exc:
            raise ValueError(
                f"Nullable US H5 {h5_path} has no artifact metadata."
            ) from exc
    if len(raw_metadata) != 1:
        raise ValueError(
            f"Nullable US H5 {h5_path} must carry exactly one artifact metadata row."
        )
    try:
        metadata = json.loads(str(raw_metadata.iloc[0]))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Nullable US H5 {h5_path} artifact metadata is not valid JSON."
        ) from exc
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Nullable US H5 {h5_path} artifact metadata must be a JSON object."
        )
    return metadata


def write_nullable_us_h5(
    frame: Frame,
    path: str | Path,
    *,
    period: int,
    artifact_kind: str,
    publication_run_id: str | None = None,
    materializer_version: int | None = None,
) -> None:
    """Atomically write and verify a nullable US single-year H5.

    The destination is replaced only after a temporary sibling has round-trip
    verified every nonempty entity table, household weights, period metadata,
    fixed-format storage, and the caller-declared ``artifact_kind``.  A failed
    write or verification leaves any existing destination bytes untouched.
    """

    if not isinstance(frame, Frame):
        raise TypeError(f"frame must be a Frame, got {type(frame).__name__}.")
    if not isinstance(artifact_kind, str) or not artifact_kind.strip():
        raise ValueError("artifact_kind must be a non-empty string.")
    if publication_run_id is not None and (
        not isinstance(publication_run_id, str) or not publication_run_id.strip()
    ):
        raise ValueError("publication_run_id must be a non-empty string when set.")
    if materializer_version is not None and (
        type(materializer_version) is not int or materializer_version <= 0
    ):
        raise ValueError("materializer_version must be a positive integer when set.")

    for entity in US_SCHEMA.entities:
        canonicalize_table_string_dtypes(
            frame.table(entity),
            boundary="nullable US H5 export",
            table_name=entity,
            reject_untyped_all_missing_object=True,
        )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        _write_nullable_us_h5_file(
            frame,
            temporary,
            period=int(period),
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
            materializer_version=materializer_version,
        )
        _verify_nullable_us_h5(
            frame,
            temporary,
            period=int(period),
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
            materializer_version=materializer_version,
        )
        os.replace(temporary, output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_nullable_us_h5_file(
    frame: Frame,
    path: Path,
    *,
    period: int,
    artifact_kind: str,
    publication_run_id: str | None,
    materializer_version: int | None,
) -> None:
    with pd.HDFStore(path, mode="w") as store:
        for entity in frame.entities:
            table = _export_table(frame, entity)
            if not len(table):
                continue
            put_frame_table(
                store,
                entity,
                table,
                preferred_format="fixed",
            )
        store.put(
            _TIME_PERIOD_KEY,
            pd.Series([period]),
            format="table",
        )
        store.put(
            _METADATA_KEY,
            pd.Series(
                [
                    json.dumps(
                        _artifact_metadata(
                            frame,
                            artifact_kind=artifact_kind,
                            publication_run_id=publication_run_id,
                            materializer_version=materializer_version,
                        ),
                        sort_keys=True,
                    )
                ]
            ),
            format="table",
        )


def _verify_nullable_us_h5(
    frame: Frame,
    path: Path,
    *,
    period: int,
    artifact_kind: str,
    publication_run_id: str | None,
    materializer_version: int | None,
) -> None:
    with pd.HDFStore(path, mode="r") as store:
        for entity in frame.entities:
            expected = _export_table(frame, entity)
            if not len(expected):
                continue
            expected = materialize_nullable_booleans_for_pytables(expected).table
            try:
                stored = canonicalize_table_string_dtypes(
                    read_frame_table(store, entity),
                    boundary="nullable US H5 verification load",
                    table_name=entity,
                )
            except KeyError as exc:
                raise RuntimeError(
                    f"Nullable US H5 round trip omitted entity {entity!r}."
                ) from exc
            try:
                pd.testing.assert_frame_equal(
                    stored,
                    expected,
                    check_exact=True,
                )
            except AssertionError as exc:
                raise RuntimeError(
                    f"Nullable US H5 round trip changed entity {entity!r}: {exc}"
                ) from exc
            if store.get_storer(entity).is_table:
                raise RuntimeError(
                    f"Nullable US H5 stored entity {entity!r} in table format."
                )

        stored_period = store[_TIME_PERIOD_KEY]
        if stored_period.tolist() != [period]:
            raise RuntimeError(
                "Nullable US H5 round trip changed the time period: "
                f"expected {period}, got {stored_period.tolist()}."
            )
        raw_metadata = store[_METADATA_KEY]
        if len(raw_metadata) != 1:
            raise RuntimeError(
                "Nullable US H5 must carry exactly one artifact metadata row."
            )
        try:
            stored_metadata = json.loads(str(raw_metadata.iloc[0]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "Nullable US H5 artifact metadata is not valid JSON."
            ) from exc
        expected_metadata = _artifact_metadata(
            frame,
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
            materializer_version=materializer_version,
        )
        if stored_metadata != expected_metadata:
            raise RuntimeError(
                "Nullable US H5 round trip changed artifact metadata: "
                f"expected {expected_metadata}, got {stored_metadata}."
            )


def _export_table(frame: Frame, entity: str) -> pd.DataFrame:
    table = canonicalize_table_string_dtypes(
        frame.table(entity),
        boundary="nullable US H5 entity export",
        table_name=entity,
    )
    if entity != "household":
        return table
    household = table.copy(deep=False)
    household["household_weight"] = frame.weights_for("household").values
    return household


def _artifact_metadata(
    frame: Frame,
    *,
    artifact_kind: str,
    publication_run_id: str | None,
    materializer_version: int | None,
) -> dict[str, object]:
    metadata = {
        "artifact_kind": artifact_kind,
        "entity_hdf_format": "fixed_nullable",
        "household_weight_kind": frame.weights_for("household").kind.value,
    }
    if publication_run_id is not None:
        metadata["publication_run_id"] = publication_run_id
    if materializer_version is not None:
        metadata["materializer_version"] = materializer_version
    return metadata


def _read_json_object(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> dict[str, object]:
    payload, _, _ = _read_json_object_with_identity(
        path,
        label=label,
        expected_sha256=expected_sha256,
    )
    return payload


def _read_json_object_with_identity(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, object], str, int]:
    try:
        raw = Path(path).read_bytes()
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} {path} is not readable valid JSON.") from exc
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if not isinstance(expected_sha256, str) or not _LOWERCASE_SHA256.fullmatch(
            expected_sha256
        ):
            raise ValueError(
                f"Expected {label} SHA-256 must be 64 lowercase hexadecimal characters."
            )
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{label.capitalize()} SHA-256 mismatch for {path}: got "
                f"{observed_sha256}, expected {expected_sha256}."
            )
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} {path} is not readable valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {path} must contain a JSON object.")
    return payload, observed_sha256, len(raw)


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def _require_matching_terminal_gate_aliases(
    manifest: Mapping[str, object],
    diagnostics: Mapping[str, object],
    *,
    manifest_path: Path,
    diagnostics_path: Path,
    manifest_agreement_gate: Mapping[str, object],
    diagnostics_agreement_gate: Mapping[str, object],
) -> None:
    """Bind stacked terminal gates to the legacy compatibility aliases.

    Legacy two-spine manifests predate ``terminal_gates`` and remain valid
    without it.  The stacked pipeline declares that field as its terminal
    authority, so both publication documents must carry it and must reproduce
    their existing ``agreement_gate`` compatibility aliases exactly.
    """

    is_stacked = manifest.get("pipeline") == "us-stacked-pool"
    manifest_has_terminal = "terminal_gates" in manifest
    diagnostics_has_terminal = "terminal_gates" in diagnostics
    if not (is_stacked or manifest_has_terminal or diagnostics_has_terminal):
        return
    if not manifest_has_terminal:
        raise ValueError(
            f"US stacked pool manifest {manifest_path}.terminal_gates must be "
            "an object matching agreement_gate."
        )
    if not diagnostics_has_terminal:
        raise ValueError(
            f"US stacked pool diagnostics {diagnostics_path}.terminal_gates must "
            "be an object matching agreement_gate."
        )
    manifest_terminal_gates = _mapping(
        manifest.get("terminal_gates"),
        label=f"US stacked pool manifest {manifest_path}.terminal_gates",
    )
    diagnostics_terminal_gates = _mapping(
        diagnostics.get("terminal_gates"),
        label=f"US stacked pool diagnostics {diagnostics_path}.terminal_gates",
    )
    if manifest_terminal_gates != manifest_agreement_gate:
        raise ValueError(
            f"US stacked pool manifest {manifest_path} terminal_gates do not "
            "match agreement_gate."
        )
    if diagnostics_terminal_gates != diagnostics_agreement_gate:
        raise ValueError(
            f"US stacked pool diagnostics {diagnostics_path} terminal_gates do "
            "not match agreement_gate."
        )


def _publication_run_id(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must name a non-empty publication run ID.")
    return value


def _require_matching_run_id(
    receipt: Mapping[str, object],
    publication_run_id: str,
    *,
    label: str,
) -> None:
    if receipt.get("publication_run_id") != publication_run_id:
        raise ValueError(f"{label} publication run ID does not match the manifest.")


def _artifact_path(receipt: Mapping[str, object], *, label: str) -> Path:
    raw_path = receipt.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"{label}.path must be a non-empty string.")
    path = Path(raw_path)
    if not path.is_file():
        raise ValueError(f"{label}.path is not a file: {path}")
    return path


def _require_matching_sha256(
    path: Path,
    receipt: Mapping[str, object],
    *,
    label: str,
    require_size: bool = False,
) -> tuple[str, int]:
    expected = receipt.get("sha256")
    if not isinstance(expected, str) or not _LOWERCASE_SHA256.fullmatch(expected):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest.")
    expected_size = receipt.get("size_bytes")
    if require_size and (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise ValueError(f"{label}.size_bytes must be a non-negative integer.")
    observed_sha256, observed_size_bytes = _file_sha256_and_size(path)
    if observed_sha256 != expected:
        raise ValueError(f"{label} SHA-256 does not match the published artifact.")
    if require_size and observed_size_bytes != expected_size:
        raise ValueError(
            f"{label} size_bytes {observed_size_bytes} does not match the "
            f"published artifact size {expected_size}."
        )
    return observed_sha256, observed_size_bytes


def _file_sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _copy_file_bytes(source: Path, destination: Path) -> None:
    with (
        source.open("rb") as source_stream,
        destination.open("xb") as destination_stream,
    ):
        shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)
