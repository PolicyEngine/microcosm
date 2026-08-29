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
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.serialization_dtypes import (
    canonicalize_frame_string_dtypes,
    canonicalize_table_string_dtypes,
)
from microcosm.build.us_runtime.congressional_district_geography import (
    CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
)
from microcosm.build.us_runtime.congressional_district_vintage import (
    CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
    CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
    CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE,
)
from microcosm.build.us_runtime.support_provenance import (
    support_clone_index_column,
    support_source_id_column,
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
    "identify_us_multispine_pool_manifest",
    "load_authenticated_us_multispine_pool_for_release",
    "load_authenticated_us_multispine_pool_for_scoring",
    "load_legacy_calibrated_us_h5",
    "load_simulation_ready_us_multispine_pool",
    "load_simulation_ready_us_multispine_pool_manifest",
    "read_nullable_us_h5_metadata",
    "require_authenticated_us_multispine_pool_h5",
    "us_multispine_pool_release_receipt",
    "write_nullable_us_h5",
]

LEGACY_NULLABLE_STAGING_ARTIFACT_KIND = "nullable_precalibration_staging_h5"
US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND = "populace_us_multispine_pool_manifest"
US_MULTISPINE_POOL_H5_ARTIFACT_KIND = "populace_us_multispine_input_pool"
US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND = (
    "populace_us_multispine_agreement_diagnostics"
)
# 10 binds the humanitarian-immigration composition gate into the stacked
# operator order and authenticated terminal-gate surface.
# 9 binds the post-assembly household-geography assignment receipt and its
# authenticated release-vintage authorities.
# 8 binds the nullable-boolean-capable physical H5 materializer in both the
# stacked manifest receipt and the H5's frozen metadata key.
# 7 binds the complete late-producer resource semantics and removes the PUF
# callback's duplicate outer-order entry; the callback is a node inside the DAG.
# 6 additionally bound the independently carried late-producer transition
# authority and restores its immutable Frame-metadata anchor on H5 load.
# Schema 5 can authenticate the DAG receipt's structure, but cannot prove that
# the published receipt is the one authorized by the generating transition.
US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION = 10
US_MULTISPINE_POOL_H5_MATERIALIZER_VERSION = 3
"""Version 3 atomically binds release CD provenance attrs; v2 added BooleanDtype."""
_LEGACY_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION = 4
_METADATA_KEY = "_populace_staging_metadata"
_TIME_PERIOD_KEY = "_time_period"
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_STACKED_PIPELINE = "us-stacked-pool"
_STACKED_SPINE_MANIFEST_VERSION = 4
_STACKED_SURVEY_CHANNELS = ("asec", "acs")
_STACKED_SAMPLE_RUNG_TOKENS: Mapping[float, str] = {
    0.01: "f001",
    0.04: "f004",
    0.10: "f010",
    0.25: "f025",
    1.00: "f100",
}
US_STACKED_POOL_OPERATOR_ORDER = (
    "assemble_stacked_spine",
    "assign_us_puma_ladder",
    "prepare_multispine_source_inputs_for_clone",
    "gap_fill_stacked_spine",
    "run_stacked_late_producer_dag",
    "prepare_stacked_tail_derivation",
    "derive_multispine_pool_inputs",
    "seed_multispine_pool_inputs",
    "materialize_multispine_agreement_outputs",
    "stacked_completeness_gate",
    "by_origin_battery",
    "us_immigration_composition_gate",
)
_STACKED_TERMINAL_GATE_NAMES = frozenset(
    {
        "us_stacked_completeness",
        "us_by_origin_battery",
        "immigration_composition",
    }
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
_LEGACY_POOL_CHECKPOINT_MATERIALIZER_VERSION = 3
_LEGACY_REQUIRED_STAGE_RECEIPTS = frozenset({"impute", "derive", "seed", "simulate"})
_STACKED_ONLY_MANIFEST_FIELDS = frozenset(
    {
        "pipeline",
        "release_id",
        "sampling",
        "clone_attachment",
        "geography_assignment",
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
        "sampling",
        "stack_manifest",
        "geography_assignment",
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


def _validated_stacked_sampling_manifest_binding(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
) -> Mapping[str, object] | None:
    """Authenticate the production-wide survey rung carried by a stack.

    The adjacent-year ASEC join runs after the two survey arms are sampled.
    Release gates may therefore consume the configured production rung only
    after the pool manifest proves that its top-level sampling receipt, frozen
    stack manifest, and both per-arm sample receipts all name the same value.
    Legacy two-spine manifests have no production-wide stack receipt and return
    ``None``.
    """

    if manifest.get("pipeline") != _STACKED_PIPELINE:
        return None

    label = f"US stacked pool manifest {manifest_path}"
    sampling = _mapping(manifest.get("sampling"), label=f"{label}.sampling")
    stack_manifest = _mapping(
        manifest.get("stack_manifest"),
        label=f"{label}.stack_manifest",
    )
    if stack_manifest.get("version") != _STACKED_SPINE_MANIFEST_VERSION:
        raise ValueError(
            f"{label} stack manifest must have production version "
            f"{_STACKED_SPINE_MANIFEST_VERSION}."
        )

    sampling_fraction = sampling.get("sample_fraction")
    stack_fraction = stack_manifest.get("sample_fraction")
    for location, value in (
        ("sampling.sample_fraction", sampling_fraction),
        ("stack_manifest.sample_fraction", stack_fraction),
    ):
        if type(value) is not float or not np.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{label} {location} must be a finite float in (0, 1].")
    if sampling_fraction != stack_fraction:
        raise ValueError(
            f"{label} sampling.sample_fraction differs from "
            "stack_manifest.sample_fraction."
        )
    expected_token = _STACKED_SAMPLE_RUNG_TOKENS.get(sampling_fraction)
    if expected_token is None or sampling.get("fraction_token") != expected_token:
        raise ValueError(
            f"{label} sampling fraction/token pair is not an approved stacked rung."
        )

    sampling_seed = sampling.get("sample_seed")
    stack_seed = stack_manifest.get("sample_seed")
    if (
        isinstance(sampling_seed, bool)
        or not isinstance(sampling_seed, int)
        or sampling_seed < 0
        or isinstance(stack_seed, bool)
        or not isinstance(stack_seed, int)
        or stack_seed != sampling_seed
    ):
        raise ValueError(
            f"{label} sampling.sample_seed and stack_manifest.sample_seed must "
            "be the same non-negative integer."
        )

    survey_samples = _mapping(
        stack_manifest.get("survey_samples"),
        label=f"{label}.stack_manifest.survey_samples",
    )
    if set(survey_samples) != set(_STACKED_SURVEY_CHANNELS):
        raise ValueError(
            f"{label} stack survey samples must exactly cover "
            f"{list(_STACKED_SURVEY_CHANNELS)}."
        )
    realized_households = _mapping(
        sampling.get("realized_households"),
        label=f"{label}.sampling.realized_households",
    )
    if set(realized_households) != set(_STACKED_SURVEY_CHANNELS):
        raise ValueError(
            f"{label} realized-household counts must exactly cover "
            f"{list(_STACKED_SURVEY_CHANNELS)}."
        )
    for channel in _STACKED_SURVEY_CHANNELS:
        sample = _mapping(
            survey_samples[channel],
            label=f"{label}.stack_manifest.survey_samples.{channel}",
        )
        sample_fraction = sample.get("fraction")
        if type(sample_fraction) is not float or sample_fraction != sampling_fraction:
            raise ValueError(
                f"{label} {channel} survey-sample fraction differs from the "
                "production sampling rung."
            )
        sample_seed = sample.get("seed")
        if (
            isinstance(sample_seed, bool)
            or not isinstance(sample_seed, int)
            or sample_seed != sampling_seed
        ):
            raise ValueError(
                f"{label} {channel} survey-sample seed differs from the "
                "production sample seed."
            )
        realized = sample.get("realized_household_count")
        top_realized = realized_households[channel]
        if (
            isinstance(realized, bool)
            or not isinstance(realized, int)
            or realized < 1
            or isinstance(top_realized, bool)
            or not isinstance(top_realized, int)
            or top_realized != realized
        ):
            raise ValueError(
                f"{label} {channel} realized-household count is malformed or "
                "inconsistent."
            )

    expected_stack_sha256 = sampling.get("stack_manifest_sha256")
    if (
        not isinstance(expected_stack_sha256, str)
        or _LOWERCASE_SHA256.fullmatch(expected_stack_sha256) is None
    ):
        raise ValueError(f"{label} sampling stack-manifest SHA-256 is malformed.")
    canonical_stack = json.dumps(
        stack_manifest,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if hashlib.sha256(canonical_stack).hexdigest() != expected_stack_sha256:
        raise ValueError(
            f"{label} sampling stack-manifest SHA-256 does not match its receipt."
        )
    return stack_manifest


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


def identify_us_multispine_pool_manifest(path: str | Path) -> Path | None:
    """Return the required sidecar when an H5 positively identifies as a pool.

    A successful pool publication writes ``<stem>.manifest.json`` beside the
    H5 and stamps the H5's artifact-metadata row. Either identity is enough to
    require the manifest. The caller must still authenticate that manifest;
    this classifier deliberately does not turn sidecar existence into trust.
    """

    h5_path = Path(path)
    manifest_path = h5_path.with_suffix(".manifest.json")
    manifest_identifies_pool = False
    if manifest_path.is_file():
        try:
            sibling_manifest = _read_json_object(
                manifest_path,
                label="candidate US multispine pool manifest",
            )
        except (OSError, ValueError):
            sibling_manifest = None
        manifest_identifies_pool = (
            isinstance(sibling_manifest, Mapping)
            and sibling_manifest.get("artifact_kind")
            == US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND
        )

    # A generic PolicyEngine H5 has no Microcosm artifact-metadata table. Open
    # the key inventory first so an unreadable non-pool fixture stays on the
    # generic path, while a present-but-malformed identity row remains a hard
    # error instead of becoming a receipt bypass.
    try:
        with pd.HDFStore(h5_path, mode="r") as store:
            has_artifact_metadata = _METADATA_KEY in {
                key.lstrip("/") for key in store.keys()
            }
    except Exception:
        has_artifact_metadata = False

    h5_identifies_pool = False
    if has_artifact_metadata:
        metadata = read_nullable_us_h5_metadata(h5_path)
        h5_identifies_pool = (
            metadata.get("artifact_kind") == US_MULTISPINE_POOL_H5_ARTIFACT_KIND
        )
    if manifest_identifies_pool or h5_identifies_pool:
        return manifest_path
    return None


def require_authenticated_us_multispine_pool_h5(
    requested_path: str | Path,
    authenticated_pool_h5: AuthenticatedPoolH5,
    *,
    consumer: str,
) -> Path:
    """Require one sidecar to authorize the exact H5 supplied by a consumer."""

    requested = Path(requested_path).resolve()
    authenticated = authenticated_pool_h5.path.resolve()
    if authenticated != requested:
        raise ValueError(
            f"{consumer} supplied US multispine pool H5 {requested}, but its "
            f"sibling manifest authenticates a different H5 {authenticated}."
        )
    return authenticated


def us_multispine_pool_release_receipt(
    manifest: Mapping[str, object],
    authenticated_pool_h5: AuthenticatedPoolH5,
    *,
    allow_gate_failed_base_pool: bool,
) -> dict[str, object]:
    """Build self-contained release evidence from an authenticated pool."""

    status = manifest.get("status")
    simulation_ready = manifest.get("simulation_ready")
    is_ready = status == "simulation_ready" and simulation_ready is True
    is_gate_failed = status == "gate_failed" and simulation_ready is False
    if not is_ready and not is_gate_failed:
        raise ValueError(
            "Authenticated US multispine pool has an unsupported release status "
            f"pair: status={status!r}, simulation_ready={simulation_ready!r}."
        )
    if is_gate_failed and not allow_gate_failed_base_pool:
        raise ValueError(
            "Authenticated gate-failed US multispine pool requires the explicit "
            "--allow-gate-failed-base-pool opt-in."
        )
    if is_ready and allow_gate_failed_base_pool:
        raise ValueError(
            "--allow-gate-failed-base-pool was set, but the authenticated US "
            "multispine pool is simulation-ready; the override is valid only "
            "for status=gate_failed and simulation_ready=false."
        )

    agreement_gate = _mapping(
        manifest.get("agreement_gate"),
        label="authenticated US multispine pool agreement_gate",
    )
    expected_passed = is_ready
    if agreement_gate.get("passed") is not expected_passed:
        raise ValueError(
            "Authenticated US multispine pool status disagrees with its "
            "agreement-gate verdict."
        )
    gates = _mapping(
        agreement_gate.get("gates"),
        label="authenticated US multispine pool agreement_gate.gates",
    )
    if not gates:
        raise ValueError(
            "Authenticated US multispine pool agreement gate has no nested "
            "gate verdicts."
        )
    failures: list[dict[str, str]] = []
    nested_passed: list[bool] = []
    for gate_name, gate_payload in gates.items():
        if not isinstance(gate_name, str) or not gate_name:
            raise ValueError(
                "Authenticated US multispine pool agreement gate has an invalid "
                "gate name."
            )
        gate = _mapping(
            gate_payload,
            label=f"authenticated US multispine pool gate {gate_name!r}",
        )
        gate_passed = gate.get("passed")
        gate_failures = gate.get("failures")
        if (
            type(gate_passed) is not bool
            or not isinstance(gate_failures, list)
            or not all(isinstance(failure, str) for failure in gate_failures)
        ):
            raise ValueError(
                f"Authenticated US multispine pool gate {gate_name!r} has an "
                "invalid passed verdict or failure list."
            )
        if gate_passed is bool(gate_failures):
            raise ValueError(
                f"Authenticated US multispine pool gate {gate_name!r} has an "
                "incoherent passed verdict and failure list."
            )
        nested_passed.append(gate_passed)
        failures.extend(
            {"gate": gate_name, "message": failure} for failure in gate_failures
        )
    if all(nested_passed) is not expected_passed:
        raise ValueError(
            "Authenticated US multispine pool aggregate agreement verdict "
            "disagrees with its nested gate verdicts."
        )

    diagnostics = _mapping(
        manifest.get("agreement_diagnostics"),
        label="authenticated US multispine pool agreement_diagnostics",
    )
    gates_json_sha256 = diagnostics.get("sha256")
    if (
        not isinstance(gates_json_sha256, str)
        or _LOWERCASE_SHA256.fullmatch(gates_json_sha256) is None
    ):
        raise ValueError(
            "Authenticated US multispine pool agreement diagnostics have no "
            "valid SHA-256."
        )

    return {
        "artifact_kind": US_MULTISPINE_POOL_H5_ARTIFACT_KIND,
        "status": status,
        "simulation_ready": simulation_ready,
        "manifest_sha256": authenticated_pool_h5.manifest_sha256,
        "publication_run_id": authenticated_pool_h5.publication_run_id,
        "pool_h5_sha256": authenticated_pool_h5.sha256,
        "pool_h5_size_bytes": authenticated_pool_h5.size_bytes,
        "allow_gate_failed_base_pool": bool(allow_gate_failed_base_pool),
        "agreement_gate_reference": {
            "battery_status": "green" if is_ready else "red",
            "passed": expected_passed,
            "gates_json_sha256": gates_json_sha256,
            "failure_count": len(failures),
            "failures": failures,
            "verdict": dict(agreement_gate),
        },
    }


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
    allow_terminal_gate_failure: bool = False,
) -> tuple[dict[str, object], AuthenticatedPoolH5]:
    """Return a validated manifest and its authenticated pool-H5 identity.

    By default this is the production readiness boundary. The scoring-only
    caller may also authenticate a current stacked publication whose terminal
    gates failed. That exception changes no digest, schema, run-ID, H5,
    diagnostics, or gate-alias validation and never labels the artifact ready.
    """

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
    is_ready = (
        manifest.get("simulation_ready") is True
        and manifest.get("status") == "simulation_ready"
    )
    is_terminal_gate_failure = (
        manifest.get("simulation_ready") is False
        and manifest.get("status") == "gate_failed"
    )
    if not is_ready and not (allow_terminal_gate_failure and is_terminal_gate_failure):
        raise ValueError(
            f"US multispine pool manifest {manifest_path} is not simulation-ready."
        )
    envelope = _validated_pool_manifest_envelope(
        manifest,
        manifest_path=manifest_path,
    )
    _validated_stacked_sampling_manifest_binding(
        manifest,
        manifest_path=manifest_path,
    )
    expected_schema_version = (
        US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
        if envelope == "stacked"
        else _LEGACY_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
    )
    if is_terminal_gate_failure and envelope != "stacked":
        raise ValueError(
            f"US multispine pool manifest {manifest_path} may expose a failed "
            "terminal-gate publication for scoring only under the current "
            "stacked envelope."
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
        or diagnostics.get("simulation_ready") != manifest.get("simulation_ready")
        or diagnostics.get("publication_run_id") != publication_run_id
    ):
        raise ValueError(
            f"US multispine pool diagnostics {diagnostics_path} do not match "
            "the authenticated manifest publication."
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
            "verdict does not match the authenticated manifest."
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
    """Make current-schema consumers authenticate geography and late-DAG proofs."""

    if manifest.get("pipeline") != "us-stacked-pool":
        return
    if manifest.get("operator_order") != list(US_STACKED_POOL_OPERATOR_ORDER):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} does not bind the "
            "canonical late-DAG operator order."
        )
    _validate_stacked_geography_assignment_manifest_binding(
        manifest,
        manifest_path=manifest_path,
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


def _validate_stacked_geography_assignment_manifest_binding(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path,
) -> None:
    """Authenticate the post-assembly household-CD authority and receipt."""

    assignment = manifest.get("geography_assignment")
    if not isinstance(assignment, Mapping):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has no geography "
            "assignment receipt."
        )
    stage_receipts = manifest.get("stage_receipts")
    stage_assignment = (
        stage_receipts.get("geography_assignment")
        if isinstance(stage_receipts, Mapping)
        else None
    )
    if stage_assignment != assignment:
        raise ValueError(
            f"US stacked pool manifest {manifest_path} geography assignment "
            "differs from its assembled-stage receipt."
        )
    if (
        assignment.get("artifact_kind")
        != ("populace_us_stacked_household_geography_assignment")
        or assignment.get("schema_version") != 1
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has an unsupported "
            "geography assignment receipt."
        )
    contract = assignment.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} geography assignment "
            "has no contract."
        )
    expected_declaration = {
        "anchor": "puma",
        "order": "before_gap_fill",
        "kernels": {
            "assign": "kernel:assign_us_puma_ladder",
            "validate": "kernel:us_puma_ladder_gate",
        },
        "draw": {
            "asec": {
                "universe": "puma_within_state",
                "weight": "puma_population_2020",
            },
            "congressional_district": {
                "universe": "congressional_district_within_puma",
                "weight": "block_population_overlap",
            },
            "county": {
                "universe": "county_within_puma",
                "weight": "block_population_overlap",
            },
        },
        "derive": ["puma", "congressional_district_geoid", "county_fips"],
        "assertions": [
            "observed_acs_puma_preserved",
            "geography_state_prefix_consistent",
        ],
        "ladder_source": "source:us_puma_ladder_2020",
        "congressional_district_vintage_crosswalk": {
            "source_ref": (
                "source:us_congressional_district_vintage_crosswalk_117_to_119"
            ),
            "source_vintage": "vintage:cd_117",
            "target_vintage": "vintage:cd_119",
        },
        "seed": "stream:geography_legacy",
        "default_seed": 0,
        "assign_tract": False,
        "layer_vintages": {
            "congressional_district": "vintage:cd_119",
            "county": "vintage:census_2020",
            "puma": "vintage:puma_2020",
            "tract": "vintage:census_2020",
        },
        "validation": ["puma_ladder_gate", "vintage_refusal"],
    }
    if contract.get("declaration") != expected_declaration:
        raise ValueError(
            f"US stacked pool manifest {manifest_path} geography declaration changed."
        )
    algorithm = contract.get("algorithm")
    expected_algorithm = {
        "id": "assign_us_puma_ladder.population_weighted_overlap.v1",
        "kernel": "assign_us_puma_ladder",
        "operator": "assign_us_puma_ladder",
        "order": "before_gap_fill",
        "assign_tract": False,
    }
    if algorithm != expected_algorithm:
        raise ValueError(
            f"US stacked pool manifest {manifest_path} geography assignment "
            "algorithm changed."
        )
    seed = contract.get("seed")
    expected_seed = {
        "site": "legacy_puma_ladder",
        "stream": "geography_legacy",
        "value_source": "run_request.build_model_seed",
        "value": manifest.get("random_seed"),
    }
    if seed != expected_seed or manifest.get("random_seed") != 0:
        raise ValueError(
            f"US stacked pool manifest {manifest_path} geography assignment "
            "seed changed."
        )
    authorities = contract.get("authorities")
    expected_roles = {
        "puma_ladder",
        "congressional_district_vintage_crosswalk",
    }
    if not isinstance(authorities, Mapping) or set(authorities) != expected_roles:
        raise ValueError(
            f"US stacked pool manifest {manifest_path} geography authorities changed."
        )
    provenance_pins = manifest.get("provenance_pins")
    if not isinstance(provenance_pins, Mapping):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has no provenance pins."
        )
    for role in sorted(expected_roles):
        authority = authorities.get(role)
        pin = provenance_pins.get(role)
        if (
            not isinstance(authority, Mapping)
            or authority.get("input_role") != role
            or not isinstance(pin, Mapping)
            or authority.get("sha256") != pin.get("actual_sha256")
            or pin.get("expected_sha256") != pin.get("actual_sha256")
        ):
            raise ValueError(
                f"US stacked pool manifest {manifest_path} geography authority "
                f"{role!r} differs from its authenticated input pin."
            )
    puma = authorities["puma_ladder"]
    if not isinstance(puma, Mapping) or puma.get("source_ref") != (
        "source:us_puma_ladder_2020"
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} PUMA-ladder source "
            "authority changed."
        )
    crosswalk = authorities["congressional_district_vintage_crosswalk"]
    if (
        not isinstance(crosswalk, Mapping)
        or crosswalk.get("source_ref")
        != "source:us_congressional_district_vintage_crosswalk_117_to_119"
        or crosswalk.get("source_vintage_ref") != "vintage:cd_117"
        or crosswalk.get("source_vintage") != "117th_congress"
        or crosswalk.get("target_vintage_ref") != "vintage:cd_119"
        or crosswalk.get("target_vintage") != "119th_congress"
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} congressional-district "
            "vintage authority changed."
        )
    output = assignment.get("output")
    if not isinstance(output, Mapping):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} geography output is missing."
        )
    rows = output.get("household_rows")
    positive_rows = output.get("positive_congressional_district_rows")
    unique_values = output.get("unique_congressional_district_values")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 1
        or positive_rows != rows
        or isinstance(unique_values, bool)
        or not isinstance(unique_values, int)
        or unique_values < 1
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} does not prove positive "
            "household congressional-district support."
        )
    order = assignment.get("pre_assignment_household_order")
    if (
        not isinstance(order, Mapping)
        or order.get("column") != "household_id"
        or order.get("codec") != "int64_little_endian.v1"
        or order.get("row_count") != rows
        or not isinstance(order.get("sha256"), str)
        or _LOWERCASE_SHA256.fullmatch(str(order["sha256"])) is None
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has an invalid seeded "
            "household-order receipt."
        )
    assigned_geography = assignment.get("assigned_household_geography")
    if (
        not isinstance(assigned_geography, Mapping)
        or assigned_geography.get("columns")
        != [
            "household_id",
            "puma",
            "congressional_district_geoid",
            "county_fips",
        ]
        or assigned_geography.get("codec") != "column_major_int64_little_endian.v1"
        or assigned_geography.get("row_count") != rows
        or not isinstance(assigned_geography.get("sha256"), str)
        or _LOWERCASE_SHA256.fullmatch(str(assigned_geography["sha256"])) is None
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has an invalid ordered "
            "household-geography output receipt."
        )
    summary = assignment.get("summary")
    if (
        not isinstance(summary, Mapping)
        or summary.get("applied") is not True
        or summary.get("household_rows") != rows
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has an invalid geography "
            "assignment summary."
        )
    gate = assignment.get("gate")
    gates = gate.get("gates") if isinstance(gate, Mapping) else None
    puma_gate = gates.get("us_puma_ladder") if isinstance(gates, Mapping) else None
    if (
        not isinstance(gate, Mapping)
        or gate.get("passed") is not True
        or not isinstance(puma_gate, Mapping)
        or puma_gate.get("passed") is not True
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} does not carry a passed "
            "PUMA-ladder gate."
        )
    universe = assignment.get("target_universe")
    if (
        not isinstance(universe, Mapping)
        or isinstance(universe.get("district_count"), bool)
        or not isinstance(universe.get("district_count"), int)
        or universe["district_count"] < 1
        or not isinstance(universe.get("geoids_sha256"), str)
        or _LOWERCASE_SHA256.fullmatch(str(universe["geoids_sha256"])) is None
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} has an invalid "
            "congressional-district target-universe receipt."
        )


def _ordered_household_id_receipt(
    household: pd.DataFrame,
    *,
    boundary: str,
) -> dict[str, object]:
    """Recompute the assignment receipt's ordered native-household identity."""

    column = "household_id"
    if column not in household:
        raise ValueError(f"{boundary} has no {column!r} column.")
    numeric = pd.to_numeric(household[column], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    valid = np.isfinite(numeric) & (numeric == np.floor(numeric))
    if not valid.all():
        raise ValueError(f"{boundary} {column} values must be integral.")
    ordered = numeric.astype("<i8", copy=False)
    digest = hashlib.sha256()
    digest.update(b"populace-ordered-household-id-int64-le-v1\0")
    digest.update(len(ordered).to_bytes(8, byteorder="little", signed=False))
    digest.update(ordered.tobytes(order="C"))
    return {
        "column": column,
        "codec": "int64_little_endian.v1",
        "row_count": len(ordered),
        "sha256": digest.hexdigest(),
    }


def _positive_integral_district_values(
    household: pd.DataFrame,
    *,
    boundary: str,
) -> np.ndarray:
    if CONGRESSIONAL_DISTRICT_GEOID_COLUMN not in household:
        raise ValueError(
            f"{boundary} has no {CONGRESSIONAL_DISTRICT_GEOID_COLUMN!r} column."
        )
    numeric = pd.to_numeric(
        household[CONGRESSIONAL_DISTRICT_GEOID_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    valid = np.isfinite(numeric) & (numeric > 0) & (numeric == np.floor(numeric))
    if not valid.all():
        raise ValueError(
            f"{boundary} requires a positive integral "
            f"{CONGRESSIONAL_DISTRICT_GEOID_COLUMN} on every household."
        )
    return numeric.astype(np.int64)


def _ordered_household_geography_receipt(
    household: pd.DataFrame,
    *,
    boundary: str,
) -> dict[str, object]:
    """Recompute the producer's ordered native-household geography digest."""

    columns = (
        "household_id",
        "puma",
        CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
        "county_fips",
    )
    missing = [column for column in columns if column not in household]
    if missing:
        raise ValueError(f"{boundary} is missing geography column(s): {missing}.")

    household_ids = pd.to_numeric(
        household["household_id"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    valid_ids = np.isfinite(household_ids) & (household_ids == np.floor(household_ids))
    if not valid_ids.all():
        raise ValueError(f"{boundary} household_id values must be integral.")

    def fixed_width_values(column: str, width: int) -> np.ndarray:
        text = household[column].astype(str)
        if not text.str.fullmatch(rf"[0-9]{{{width}}}").all():
            raise ValueError(
                f"{boundary} {column!r} must contain exactly {width}-digit codes."
            )
        return text.astype(np.int64).to_numpy(dtype="<i8", copy=False)

    arrays = (
        household_ids.astype("<i8", copy=False),
        fixed_width_values("puma", 7),
        _positive_integral_district_values(
            household,
            boundary=boundary,
        ).astype("<i8", copy=False),
        fixed_width_values("county_fips", 5),
    )
    digest = hashlib.sha256()
    digest.update(b"populace-ordered-household-geography-column-major-int64-le-v1\0")
    digest.update(len(household).to_bytes(8, byteorder="little", signed=False))
    for values in arrays:
        digest.update(values.tobytes(order="C"))
    return {
        "columns": list(columns),
        "codec": "column_major_int64_little_endian.v1",
        "row_count": len(household),
        "sha256": digest.hexdigest(),
    }


def _validate_stacked_geography_h5_binding(
    manifest: Mapping[str, object],
    household: pd.DataFrame,
    root_attributes: Mapping[str, str | None],
    *,
    manifest_path: Path,
    pool_path: Path,
) -> None:
    """Bind current-schema manifest geography claims to the authenticated H5."""

    if manifest.get("pipeline") != _STACKED_PIPELINE:
        return
    assignment = _mapping(
        manifest.get("geography_assignment"),
        label=f"US stacked pool manifest {manifest_path}.geography_assignment",
    )
    contract = _mapping(
        assignment.get("contract"),
        label=(
            f"US stacked pool manifest {manifest_path}.geography_assignment.contract"
        ),
    )
    authorities = _mapping(
        contract.get("authorities"),
        label=(
            "US stacked pool manifest "
            f"{manifest_path}.geography_assignment.contract.authorities"
        ),
    )
    crosswalk = _mapping(
        authorities.get("congressional_district_vintage_crosswalk"),
        label=(
            "US stacked pool manifest "
            f"{manifest_path}.geography_assignment crosswalk authority"
        ),
    )
    expected_attributes = {
        CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR: crosswalk.get("sha256"),
        CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR: crosswalk.get("target_vintage"),
    }
    if (
        expected_attributes[CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR]
        != CURRENT_CONGRESSIONAL_DISTRICT_VINTAGE
    ):
        raise ValueError(
            f"US stacked pool manifest {manifest_path} does not name the current "
            "congressional-district target vintage."
        )
    mismatched_attributes = {
        key: {"expected": expected, "actual": root_attributes.get(key)}
        for key, expected in expected_attributes.items()
        if root_attributes.get(key) != expected
    }
    if mismatched_attributes:
        raise ValueError(
            f"US stacked pool H5 {pool_path} congressional-district root "
            f"attributes do not match its manifest: {mismatched_attributes}."
        )

    boundary = f"US stacked pool H5 {pool_path}"
    clone_column = support_clone_index_column("household")
    source_column = support_source_id_column("household")
    missing_lineage = [
        column for column in (source_column, clone_column) if column not in household
    ]
    if missing_lineage:
        raise ValueError(
            f"{boundary} is missing household clone-lineage column(s): "
            f"{missing_lineage}."
        )
    clone_index = pd.to_numeric(household[clone_column], errors="coerce")
    clone_numeric = clone_index.to_numpy(dtype=np.float64, na_value=np.nan)
    valid_clone_index = (
        np.isfinite(clone_numeric)
        & (clone_numeric >= 0)
        & (clone_numeric == np.floor(clone_numeric))
    )
    if not valid_clone_index.all():
        raise ValueError(f"{boundary} has invalid {clone_column!r} values.")
    source_ids = pd.to_numeric(household[source_column], errors="coerce")
    source_numeric = source_ids.to_numpy(dtype=np.float64, na_value=np.nan)
    valid_source_ids = np.isfinite(source_numeric) & (
        source_numeric == np.floor(source_numeric)
    )
    if not valid_source_ids.all():
        raise ValueError(f"{boundary} has invalid {source_column!r} values.")
    lineage = pd.DataFrame(
        {
            "source_id": source_numeric.astype(np.int64),
            "clone_index": clone_numeric.astype(np.int64),
        },
        index=household.index,
    )
    if lineage.duplicated(["source_id", "clone_index"]).any():
        raise ValueError(f"{boundary} has duplicate household clone-lineage roles.")
    native_counts = lineage["clone_index"].eq(0).groupby(lineage["source_id"]).sum()
    if not native_counts.eq(1).all():
        raise ValueError(
            f"{boundary} requires exactly one native household for every clone lineage."
        )
    native_mask = lineage["clone_index"].eq(0)
    native_household = household.loc[native_mask]
    native_ids = pd.to_numeric(
        native_household["household_id"], errors="coerce"
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    native_source_ids = lineage.loc[native_mask, "source_id"].to_numpy(dtype=np.float64)
    if not np.array_equal(native_ids, native_source_ids):
        raise ValueError(
            f"{boundary} native household IDs differ from their clone-lineage "
            "source IDs."
        )

    expected_order = assignment.get("pre_assignment_household_order")
    actual_order = _ordered_household_id_receipt(
        native_household,
        boundary=boundary,
    )
    if actual_order != expected_order:
        raise ValueError(
            f"{boundary} native household order differs from its manifest "
            "geography assignment receipt."
        )
    expected_geography = assignment.get("assigned_household_geography")
    actual_geography = _ordered_household_geography_receipt(
        native_household,
        boundary=boundary,
    )
    if actual_geography != expected_geography:
        raise ValueError(
            f"{boundary} native household geography differs from its manifest "
            "assignment output receipt."
        )

    native_districts = _positive_integral_district_values(
        native_household,
        boundary=boundary,
    )
    _positive_integral_district_values(household, boundary=boundary)
    actual_output = {
        "household_rows": len(native_household),
        "positive_congressional_district_rows": len(native_districts),
        "unique_congressional_district_values": int(len(np.unique(native_districts))),
    }
    if assignment.get("output") != actual_output:
        raise ValueError(
            f"{boundary} household geography counts differ from its manifest "
            "assignment output receipt."
        )

    geography_columns = (
        "puma",
        CONGRESSIONAL_DISTRICT_GEOID_COLUMN,
        "county_fips",
    )
    grouped = household.groupby(source_column, sort=False, dropna=False)
    for column in geography_columns:
        incoherent = grouped[column].nunique(dropna=False) > 1
        if bool(incoherent.any()):
            raise ValueError(
                f"{boundary} cloned household rows disagree on assigned "
                f"geography column {column!r}."
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

    return _load_us_multispine_pool(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        require_simulation_ready=True,
    )


def load_authenticated_us_multispine_pool_for_scoring(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[Frame, dict[str, object], AuthenticatedPoolH5]:
    """Load authenticated pool evidence without promoting failed gates.

    A current stacked publication with ``status=gate_failed`` and
    ``simulation_ready=false`` remains useful as head-to-head evidence. This
    loader accepts that exact status pair, authenticates the same manifest,
    diagnostics, H5 bytes, run ID, terminal-gate aliases, and row counts as the
    production loader, and preserves the failed receipt. It is deliberately
    separate from :func:`load_simulation_ready_us_multispine_pool`, whose
    readiness contract is unchanged.
    """

    return _load_us_multispine_pool(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        require_simulation_ready=False,
    )


def load_authenticated_us_multispine_pool_for_release(
    path: str | Path,
    *,
    allow_terminal_gate_failure: bool,
    expected_manifest_sha256: str | None = None,
) -> tuple[Frame, dict[str, object], AuthenticatedPoolH5]:
    """Load an authenticated pool for release build or release preflight.

    The default release boundary remains the public simulation-ready loader.
    An explicit caller opt-in may instead admit the same current stacked
    ``gate_failed`` status pair accepted for evidence scoring. Both branches
    authenticate the complete manifest/H5/diagnostics publication; neither
    changes the contract of the strict or scoring-only public loaders.
    """

    if not allow_terminal_gate_failure:
        return load_simulation_ready_us_multispine_pool(
            path,
            expected_manifest_sha256=expected_manifest_sha256,
        )
    return _load_us_multispine_pool(
        path,
        expected_manifest_sha256=expected_manifest_sha256,
        require_simulation_ready=False,
    )


def _load_us_multispine_pool(
    path: str | Path,
    *,
    expected_manifest_sha256: str | None,
    require_simulation_ready: bool,
) -> tuple[Frame, dict[str, object], AuthenticatedPoolH5]:
    """Shared authenticated H5 reconstruction for readiness and scoring."""

    manifest_path = Path(path)
    manifest, authenticated_pool_h5 = _load_authenticated_us_multispine_pool_manifest(
        manifest_path,
        expected_manifest_sha256=expected_manifest_sha256,
        allow_terminal_gate_failure=not require_simulation_ready,
    )
    agreement_gate = _mapping(
        manifest.get("agreement_gate"),
        label=f"US multispine pool manifest {manifest_path}.agreement_gate",
    )
    if require_simulation_ready and agreement_gate.get("passed") is not True:
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
        root_attribute_set = store.get_node("/")._v_attrs
        geography_root_attributes = {
            key: _hdf_root_attribute_text(root_attribute_set, key)
            for key in (
                CONGRESSIONAL_DISTRICT_VINTAGE_CROSSWALK_SHA256_ATTR,
                CONGRESSIONAL_DISTRICT_VINTAGE_TARGET_ATTR,
            )
        }

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
    _validate_stacked_geography_h5_binding(
        manifest,
        household,
        geography_root_attributes,
        manifest_path=manifest_path,
        pool_path=pool_path,
    )
    household_weights = household.pop("household_weight").to_numpy(dtype=np.float64)
    tables["household"] = household
    late_transition = _stacked_late_transition_binding(
        manifest,
        manifest_path=manifest_path,
    )
    frame_metadata: dict[str, object] = {}
    stack_manifest = _validated_stacked_sampling_manifest_binding(
        manifest,
        manifest_path=manifest_path,
    )
    if stack_manifest is not None:
        from microcosm.build.us_runtime.stacked_spine import (
            STACKED_SPINE_MANIFEST_KEY,
        )

        frame_metadata[STACKED_SPINE_MANIFEST_KEY] = deepcopy(stack_manifest)
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
    root_attributes: Mapping[str, str] | None = None,
) -> None:
    """Atomically write and verify a nullable US single-year H5.

    The destination is replaced only after a temporary sibling has round-trip
    verified every nonempty entity table, household weights, period metadata,
    fixed-format storage, caller-declared root attributes, and the
    ``artifact_kind``.  A failed write or verification leaves any existing
    destination bytes untouched.
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
    normalized_root_attributes = _validated_root_attributes(root_attributes)

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
            root_attributes=normalized_root_attributes,
        )
        _verify_nullable_us_h5(
            frame,
            temporary,
            period=int(period),
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
            materializer_version=materializer_version,
            root_attributes=normalized_root_attributes,
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
    root_attributes: tuple[tuple[str, str], ...],
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
        root_node_attributes = store.get_node("/")._v_attrs
        for key, value in root_attributes:
            root_node_attributes[key] = value


def _verify_nullable_us_h5(
    frame: Frame,
    path: Path,
    *,
    period: int,
    artifact_kind: str,
    publication_run_id: str | None,
    materializer_version: int | None,
    root_attributes: tuple[tuple[str, str], ...],
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
        stored_root_attributes = store.get_node("/")._v_attrs
        for key, expected_value in root_attributes:
            try:
                stored_value = stored_root_attributes[key]
            except KeyError as exc:
                raise RuntimeError(
                    f"Nullable US H5 round trip omitted root attribute {key!r}."
                ) from exc
            actual_value = _h5_root_attribute_text(stored_value)
            if actual_value != expected_value:
                raise RuntimeError(
                    "Nullable US H5 round trip changed root attribute "
                    f"{key!r}: expected {expected_value!r}, got {actual_value!r}."
                )


def _validated_root_attributes(
    root_attributes: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    if root_attributes is None:
        return ()
    if not isinstance(root_attributes, Mapping):
        raise TypeError("root_attributes must be a mapping of string names to strings.")

    pytables_owned_names = frozenset(
        {"CLASS", "PYTABLES_FORMAT_VERSION", "TITLE", "VERSION"}
    )
    normalized: list[tuple[str, str]] = []
    for key, value in root_attributes.items():
        if not isinstance(key, str):
            raise TypeError("root attribute names must be strings.")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ValueError(
                f"root attribute names must be non-empty HDF identifiers; got {key!r}."
            )
        if key in pytables_owned_names:
            raise ValueError(
                f"root attribute {key!r} is owned by the HDF materializer."
            )
        if not isinstance(value, str):
            raise TypeError(f"root attribute {key!r} must have a string value.")
        if not value:
            raise ValueError(f"root attribute {key!r} must not be empty.")
        normalized.append((key, value))
    return tuple(sorted(normalized))


def _h5_root_attribute_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def _hdf_root_attribute_text(attributes: object, key: str) -> str | None:
    try:
        value = attributes[key]  # type: ignore[index]
    except KeyError:
        return None
    return _h5_root_attribute_text(value)


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
    manifest_gates = _mapping(
        manifest_terminal_gates.get("gates"),
        label=f"US stacked pool manifest {manifest_path}.terminal_gates.gates",
    )
    diagnostics_gates = _mapping(
        diagnostics_terminal_gates.get("gates"),
        label=(f"US stacked pool diagnostics {diagnostics_path}.terminal_gates.gates"),
    )
    for label, gates in (
        (f"US stacked pool manifest {manifest_path}", manifest_gates),
        (f"US stacked pool diagnostics {diagnostics_path}", diagnostics_gates),
    ):
        if set(gates) != _STACKED_TERMINAL_GATE_NAMES:
            raise ValueError(
                f"{label} does not carry the canonical terminal gate set; "
                f"expected={sorted(_STACKED_TERMINAL_GATE_NAMES)}, "
                f"observed={sorted(gates)}."
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
