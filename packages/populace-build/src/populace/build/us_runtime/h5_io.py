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
import uuid
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from populace.frame import Frame, WeightKind, Weights
from populace.frame.units import US_SCHEMA

__all__ = [
    "LEGACY_NULLABLE_STAGING_ARTIFACT_KIND",
    "US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND",
    "US_MULTISPINE_POOL_H5_ARTIFACT_KIND",
    "US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND",
    "US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION",
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
# 4 adds identity-bound stage-checkpoint provenance and an explicit always-fresh
# terminal agreement receipt to the companion pool manifest.
US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION = 4
_METADATA_KEY = "_populace_staging_metadata"
_TIME_PERIOD_KEY = "_time_period"
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")


def load_legacy_calibrated_us_h5(path: str | Path) -> Frame:
    """Load a legacy US single-year H5 as a calibrated-weight frame.

    Legacy PolicyEngine US artifacts do not expose typed weight provenance
    through ``USSingleYearDataset``.  This loader therefore preserves the
    historical builder contract and labels their household weights
    ``CALIBRATED``.  It is not the loader for the new pre-calibration
    multispine pool, whose importance-weight receipt lives in its manifest.
    """

    from policyengine_us.data import USSingleYearDataset

    dataset = USSingleYearDataset(file_path=str(Path(path)))
    tables = {
        "person": dataset.person,
        "household": dataset.household.copy(),
        "tax_unit": dataset.tax_unit,
        "spm_unit": dataset.spm_unit,
        "family": dataset.family,
        "marital_unit": dataset.marital_unit,
    }
    household_weights = (
        tables["household"].pop("household_weight").to_numpy(dtype=np.float64)
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                household_weights,
                WeightKind.CALIBRATED,
            )
        },
    )


def load_simulation_ready_us_multispine_pool_manifest(
    path: str | Path,
) -> dict[str, object]:
    """Validate and return one ready manifest bound to its H5 and diagnostics.

    The manifest is the readiness authority. A caller cannot treat an H5 as
    ready merely because it exists: the manifest, nested artifact receipts,
    H5 metadata, diagnostics, and file digests must all bind the same
    publication run.
    """

    manifest_path = Path(path)
    manifest = _read_json_object(manifest_path, label="pool manifest")
    if (
        manifest.get("artifact_kind") != US_MULTISPINE_POOL_MANIFEST_ARTIFACT_KIND
        or manifest.get("schema_version") != US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
    ):
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
    _require_matching_sha256(
        pool_path,
        pool_receipt,
        label=f"US multispine pool manifest {manifest_path}.pool_h5",
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
    if (
        diagnostics.get("artifact_kind")
        != US_MULTISPINE_AGREEMENT_DIAGNOSTICS_ARTIFACT_KIND
        or diagnostics.get("schema_version")
        != US_MULTISPINE_POOL_MANIFEST_SCHEMA_VERSION
        or diagnostics.get("simulation_ready") is not True
        or diagnostics.get("publication_run_id") != publication_run_id
    ):
        raise ValueError(
            f"US multispine pool diagnostics {diagnostics_path} do not match "
            "the ready manifest publication."
        )
    return manifest


def load_simulation_ready_us_multispine_pool(
    path: str | Path,
) -> tuple[Frame, dict[str, object]]:
    """Load a manifest-bound multispine pool with its importance weights.

    The companion manifest remains the readiness authority.  This loader first
    validates the manifest/H5/agreement publication triple, then reads the
    fixed-format entity tables and checks the H5 digest again after the read so
    bytes changed concurrently cannot be treated as the validated pool.  The
    pool's household weights retain their ``IMPORTANCE`` provenance; the
    legacy US loader deliberately labels historical datasets ``CALIBRATED``
    and is therefore not a valid consumer for this artifact.

    Returns:
        The reconstructed pool :class:`~populace.frame.Frame` and the exact
        manifest object whose artifact receipts authorized the load.
    """

    manifest_path = Path(path)
    manifest = load_simulation_ready_us_multispine_pool_manifest(manifest_path)
    agreement_gate = _mapping(
        manifest.get("agreement_gate"),
        label=f"US multispine pool manifest {manifest_path}.agreement_gate",
    )
    if agreement_gate.get("passed") is not True:
        raise ValueError(
            f"US multispine pool manifest {manifest_path} has no passing "
            "agreement-gate verdict."
        )

    pool_receipt = _mapping(
        manifest.get("pool_h5"),
        label=f"US multispine pool manifest {manifest_path}.pool_h5",
    )
    pool_path = _artifact_path(
        pool_receipt,
        label=f"US multispine pool manifest {manifest_path}.pool_h5",
    )
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
        tables = {entity: store[entity] for entity in US_SCHEMA.entities}
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
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                household_weights,
                WeightKind.IMPORTANCE,
            )
        },
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
    _require_matching_sha256(
        pool_path,
        pool_receipt,
        label=f"US multispine pool manifest {manifest_path}.pool_h5",
    )
    return frame, manifest


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
        )
        _verify_nullable_us_h5(
            frame,
            temporary,
            period=int(period),
            artifact_kind=artifact_kind,
            publication_run_id=publication_run_id,
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
) -> None:
    with pd.HDFStore(path, mode="w") as store:
        for entity in frame.entities:
            table = _export_table(frame, entity)
            if not len(table):
                continue
            # Fixed format preserves mixed bool/null object columns
            # losslessly. Table format rejects them, which would force a
            # fill or type rewrite.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
                store.put(entity, table, format="fixed")
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
) -> None:
    with pd.HDFStore(path, mode="r") as store:
        for entity in frame.entities:
            expected = _export_table(frame, entity)
            if not len(expected):
                continue
            try:
                stored = store[entity]
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
        )
        if stored_metadata != expected_metadata:
            raise RuntimeError(
                "Nullable US H5 round trip changed artifact metadata: "
                f"expected {expected_metadata}, got {stored_metadata}."
            )


def _export_table(frame: Frame, entity: str) -> pd.DataFrame:
    table = frame.table(entity)
    if entity != "household":
        return table
    household = table.copy()
    household["household_weight"] = frame.weights_for("household").values
    return household


def _artifact_metadata(
    frame: Frame,
    *,
    artifact_kind: str,
    publication_run_id: str | None,
) -> dict[str, str]:
    metadata = {
        "artifact_kind": artifact_kind,
        "entity_hdf_format": "fixed_nullable",
        "household_weight_kind": frame.weights_for("household").kind.value,
    }
    if publication_run_id is not None:
        metadata["publication_run_id"] = publication_run_id
    return metadata


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} {path} is not readable valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} {path} must contain a JSON object.")
    return payload


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


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
) -> None:
    expected = receipt.get("sha256")
    if not isinstance(expected, str) or not _LOWERCASE_SHA256.fullmatch(expected):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest.")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f"{label} SHA-256 does not match the published artifact.")
