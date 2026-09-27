"""Bounded loading for ASEC producer checkpoints.

The legacy checkpoint is produced by the outer-stage runtime after ASEC-only
input enrichment.  The raw-stage checkpoint is a separate, auxiliary artifact
whose frame has only structural source construction and exact raw-source
mapping applied.  This module validates each artifact binding and loaded
``Frame`` boundary; it does not compute a whole-file digest.  Callers that pin
input files own that separate provenance check.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.outer_stage_runtime import (
    OUTER_STAGE_CONTEXT_SCHEMA_VERSION,
    FrameIdentity,
    frame_identity,
)
from microcosm.build.serialization_dtypes import canonicalize_frame_string_dtypes
from microcosm.build.us_runtime.education_assistance_source import (
    ASEC_EDUCATION_ASSISTANCE_ARCHIVES,
)
from microcosm.build.us_runtime.operator_boundary import (
    assert_operator_free_source_frame,
)
from microcosm.build.us_runtime.reported_coverage_source import (
    ASEC_REPORTED_COVERAGE_RAW_COLUMNS,
)
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights

__all__ = [
    "ASEC_RAW_STAGE_ARTIFACT_KIND",
    "ASEC_RAW_STAGE_CHECKPOINT_FILENAME",
    "ASEC_RAW_STAGE_OPERATOR_STATUS",
    "ASEC_RAW_STAGE_SCHEMA_VERSION",
    "ASEC_RAW_STAGE_COVERAGE_SCHEMA_VERSION",
    "ASEC_RAW_STAGE_STAGE",
    "load_asec_pre_clone_checkpoint",
    "load_asec_raw_stage_checkpoint",
    "load_asec_raw_stage_checkpoint_v4",
]

_OUTER_STAGE_ARTIFACT_KIND = "populace_outer_stage_frame"
_PRE_CLONE_STAGE = "pre_clone_enrichment"
_PRE_CLONE_STAGE_INDEX = 1
_LOWERCASE_SHA256 = re.compile(r"[0-9a-f]{64}")
_BINDING_KEYS = frozenset(
    {
        "artifact_kind",
        "identity",
        "pipeline_sha256",
        "schema_version",
        "stage",
        "stage_index",
    }
)

ASEC_RAW_STAGE_ARTIFACT_KIND = "populace_us_asec_raw_stage"
ASEC_RAW_STAGE_CHECKPOINT_FILENAME = "asec_raw_stage.checkpoint.h5"
ASEC_RAW_STAGE_OPERATOR_STATUS = "operator_untouched"
# Version 3 added the PAW_TYP restoration that gates TANF enrollment; older
# artifacts lack the gate column and must fail loudly rather than let
# PAW_VAL-only conflation back in (microcosm#591).
ASEC_RAW_STAGE_SCHEMA_VERSION = 3
# V4 is a distinct measured-coverage contract. Never reinterpret a v3 file
# under this schema merely because its columns happen to look compatible.
ASEC_RAW_STAGE_COVERAGE_SCHEMA_VERSION = 4
ASEC_RAW_STAGE_STAGE = "raw_source_mapping"
_RAW_STAGE_BINDING_KEYS = frozenset(
    {
        "artifact_kind",
        "identity",
        "operator_status",
        "pipeline_sha256",
        "raw_source_mappings",
        "schema_version",
        "source_construction_identity",
        "source_receipt",
        "stage",
    }
)
_RAW_SOURCE_MAPPING_COLUMNS = frozenset({"ED_VAL", "LKWEEKS", "PAW_TYP"})
_RAW_STAGE_REQUIRED_PERSON_COLUMNS = frozenset(
    {"ED_VAL", "LKWEEKS", "PAW_TYP", "PERIDNUM", "source_year"}
)
_RAW_SOURCE_MAPPING_KEYS = frozenset(
    {
        "audit",
        "column",
        "entity",
        "join_keys",
        "operation",
        "source_pins",
    }
)
_RAW_SOURCE_PIN_KEYS = frozenset(
    {
        "income_year",
        "locator",
        "member",
        "member_sha256",
        "sha256",
    }
)


@dataclass(frozen=True)
class _RawStagePolicy:
    version: int
    coverage_columns: frozenset[str] = frozenset()

    @property
    def mapping_columns(self) -> frozenset[str]:
        return _RAW_SOURCE_MAPPING_COLUMNS | self.coverage_columns


_RAW_STAGE_V3 = _RawStagePolicy(ASEC_RAW_STAGE_SCHEMA_VERSION)
_RAW_STAGE_V4 = _RawStagePolicy(
    ASEC_RAW_STAGE_COVERAGE_SCHEMA_VERSION,
    frozenset(ASEC_REPORTED_COVERAGE_RAW_COLUMNS),
)


def load_asec_pre_clone_checkpoint(
    path: str | Path,
) -> tuple[Frame, dict[str, object]]:
    """Load and validate one input-complete ASEC pre-clone checkpoint.

    Args:
        path: Explicit outer-stage Frame checkpoint produced for
            ``pre_clone_enrichment``.

    Returns:
        The restored US ``Frame`` and its canonical, JSON-ready outer-stage
        checkpoint binding.  The metadata contains no newly computed file
        digest.

    Raises:
        FileNotFoundError: If ``path`` does not name a checkpoint file.
        ValueError: If the artifact is not bound to the required outer stage,
            its stored identity differs from the loaded ``Frame``, or the
            frame lacks the required US schema and positive household-only
            typed weights.
    """

    checkpoint_path = Path(path)
    loaded = load_frame_checkpoint(checkpoint_path)
    metadata = _validate_outer_stage_binding(
        loaded.metadata,
        path=checkpoint_path,
    )
    stored_identity = FrameIdentity.from_payload(
        metadata["identity"],
        label="ASEC pre-clone checkpoint identity",
    )
    actual_identity = frame_identity(loaded.frame)
    if actual_identity != stored_identity:
        raise ValueError(
            f"ASEC pre-clone checkpoint {checkpoint_path} Frame identity changed."
        )
    _validate_asec_frame(loaded.frame, path=checkpoint_path)
    metadata["identity"] = stored_identity.to_payload()
    return loaded.frame, metadata


def load_asec_raw_stage_checkpoint(
    path: str | Path,
) -> tuple[Frame, dict[str, object]]:
    """Load one operator-untouched ASEC raw-source-mapping checkpoint.

    Only the dedicated auxiliary producer artifact is accepted.  In
    particular, the legacy enriched outer-stage checkpoint cannot satisfy this
    contract even if it happens to carry a structurally valid US ``Frame``.
    """

    return _load_asec_raw_stage_checkpoint(path, policy=_RAW_STAGE_V3)


def load_asec_raw_stage_checkpoint_v4(
    path: str | Path,
) -> tuple[Frame, dict[str, object]]:
    """Load only v4 measured-coverage sources with registered per-year pins.

    This validates a declared source artifact, not model accuracy or release
    eligibility. The caller must independently authenticate the whole file's
    digest; shape and registered source pins alone do not authenticate values.
    """
    return _load_asec_raw_stage_checkpoint(path, policy=_RAW_STAGE_V4)


def _load_asec_raw_stage_checkpoint(
    path: str | Path,
    *,
    policy: _RawStagePolicy,
) -> tuple[Frame, dict[str, object]]:

    checkpoint_path = Path(path)
    loaded = load_frame_checkpoint(checkpoint_path)
    metadata = _validate_raw_stage_binding(
        loaded.metadata,
        path=checkpoint_path,
        policy=policy,
    )
    stored_identity = FrameIdentity.from_payload(
        metadata["identity"],
        label="ASEC raw-stage checkpoint identity",
    )
    actual_identity = frame_identity(loaded.frame)
    if actual_identity != stored_identity:
        raise ValueError(
            f"ASEC raw-stage checkpoint {checkpoint_path} Frame identity changed."
        )
    _validate_asec_frame(
        loaded.frame,
        path=checkpoint_path,
        artifact_label="ASEC raw-stage checkpoint",
    )
    assert_operator_free_source_frame(
        loaded.frame,
        label=f"ASEC raw-stage checkpoint {checkpoint_path}",
    )
    _validate_raw_stage_source_columns(
        loaded.frame, path=checkpoint_path, policy=policy
    )
    if policy.coverage_columns:
        _validate_coverage_v4_binding(loaded.frame, metadata, path=checkpoint_path)
    source_construction_identity = FrameIdentity.from_payload(
        metadata["source_construction_identity"],
        label="ASEC raw-stage source-construction identity",
    )
    if source_construction_identity != actual_identity:
        raise ValueError(
            f"ASEC raw-stage checkpoint {checkpoint_path} no longer has the "
            "source-construction structural identity it declares."
        )
    metadata["identity"] = stored_identity.to_payload()
    metadata["source_construction_identity"] = source_construction_identity.to_payload()
    # Canonicalize only after every identity and binding check has passed on
    # the restored representation: this load is a serialization boundary, and
    # downstream spine assembly requires one physical string dtype per shared
    # column regardless of the storage the checkpoint was written under.
    frame = canonicalize_frame_string_dtypes(
        loaded.frame,
        boundary="ASEC raw-stage checkpoint load",
        in_place=True,
    )
    return frame, metadata


def _validate_outer_stage_binding(
    metadata: dict[str, object],
    *,
    path: Path,
) -> dict[str, object]:
    actual_keys = frozenset(metadata)
    if actual_keys != _BINDING_KEYS:
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} has an incomplete outer-stage "
            f"artifact binding (missing: {sorted(_BINDING_KEYS - actual_keys)}; "
            f"extra: {sorted(actual_keys - _BINDING_KEYS)})."
        )
    if metadata["artifact_kind"] != _OUTER_STAGE_ARTIFACT_KIND:
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} is not an outer-stage Frame artifact."
        )
    if metadata["schema_version"] != OUTER_STAGE_CONTEXT_SCHEMA_VERSION or isinstance(
        metadata["schema_version"], bool
    ):
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} has an unsupported outer-stage "
            "schema version."
        )
    if metadata["stage"] != _PRE_CLONE_STAGE:
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} must be bound to stage "
            f"{_PRE_CLONE_STAGE!r}, got {metadata['stage']!r}."
        )
    if metadata["stage_index"] != _PRE_CLONE_STAGE_INDEX or isinstance(
        metadata["stage_index"], bool
    ):
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} must be bound to stage_index "
            f"{_PRE_CLONE_STAGE_INDEX}, got {metadata['stage_index']!r}."
        )
    pipeline_sha256 = metadata["pipeline_sha256"]
    if not isinstance(pipeline_sha256, str) or not _LOWERCASE_SHA256.fullmatch(
        pipeline_sha256
    ):
        raise ValueError(
            f"ASEC pre-clone checkpoint {path} pipeline_sha256 must be a "
            "lowercase SHA-256 digest."
        )
    return dict(metadata)


def _validate_raw_stage_binding(
    metadata: dict[str, object],
    *,
    path: Path,
    policy: _RawStagePolicy = _RAW_STAGE_V3,
) -> dict[str, object]:
    actual_keys = frozenset(metadata)
    if actual_keys != _RAW_STAGE_BINDING_KEYS:
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} has an incomplete raw-stage "
            f"artifact binding (missing: "
            f"{sorted(_RAW_STAGE_BINDING_KEYS - actual_keys)}; extra: "
            f"{sorted(actual_keys - _RAW_STAGE_BINDING_KEYS)})."
        )
    if metadata["artifact_kind"] != ASEC_RAW_STAGE_ARTIFACT_KIND:
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} is not a dedicated raw-stage "
            "ASEC artifact."
        )
    schema_version = metadata["schema_version"]
    if (
        schema_version != policy.version
        or isinstance(schema_version, bool)
        or (policy.coverage_columns and type(schema_version) is not int)
    ):
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} has an unsupported raw-stage "
            "schema version."
        )
    if metadata["stage"] != ASEC_RAW_STAGE_STAGE:
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} must be bound to stage "
            f"{ASEC_RAW_STAGE_STAGE!r}, got {metadata['stage']!r}."
        )
    if metadata["operator_status"] != ASEC_RAW_STAGE_OPERATOR_STATUS:
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} must declare operator_status "
            f"{ASEC_RAW_STAGE_OPERATOR_STATUS!r}, got "
            f"{metadata['operator_status']!r}."
        )
    pipeline_sha256 = metadata["pipeline_sha256"]
    if not isinstance(pipeline_sha256, str) or not _LOWERCASE_SHA256.fullmatch(
        pipeline_sha256
    ):
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} pipeline_sha256 must be a "
            "lowercase SHA-256 digest."
        )
    _validate_source_receipt(metadata["source_receipt"], path=path)
    _validate_raw_source_mappings(
        metadata["raw_source_mappings"], path=path, policy=policy
    )
    return dict(metadata)


def _validate_source_receipt(receipt: object, *, path: Path) -> None:
    if not isinstance(receipt, Mapping) or receipt.get("kind") != "pooled_asec":
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} source_receipt must describe "
            "pooled_asec inputs."
        )
    sources = receipt.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} source_receipt.sources must be "
            "a non-empty list."
        )
    years: set[int] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise ValueError(
                f"ASEC raw-stage checkpoint {path} source_receipt.sources"
                f"[{index}] must be an object."
            )
        year = source.get("year")
        source_path = source.get("path")
        sha256 = source.get("sha256")
        if (
            not isinstance(year, int)
            or isinstance(year, bool)
            or not isinstance(source_path, str)
            or not source_path
            or not isinstance(sha256, str)
            or not _LOWERCASE_SHA256.fullmatch(sha256)
        ):
            raise ValueError(
                f"ASEC raw-stage checkpoint {path} source_receipt.sources"
                f"[{index}] lacks a valid year/path/SHA-256 pin."
            )
        if year in years:
            raise ValueError(
                f"ASEC raw-stage checkpoint {path} source_receipt repeats "
                f"income year {year}."
            )
        years.add(year)


def _validate_raw_source_mappings(
    mappings: object, *, path: Path, policy: _RawStagePolicy = _RAW_STAGE_V3
) -> None:
    if not isinstance(mappings, Mapping):
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} raw_source_mappings must be an object."
        )
    if frozenset(mappings) != policy.mapping_columns:
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} raw_source_mappings must bind "
            f"exactly {sorted(policy.mapping_columns)}."
        )
    for column in sorted(policy.mapping_columns):
        mapping = mappings[column]
        if not isinstance(mapping, Mapping) or frozenset(mapping) != (
            _RAW_SOURCE_MAPPING_KEYS
        ):
            raise ValueError(
                f"ASEC raw-stage checkpoint {path} raw_source_mappings"
                f"[{column!r}] is malformed."
            )
        if (
            mapping["column"] != column
            or mapping["entity"] != "person"
            or mapping["operation"] != "exact_source_join"
            or mapping["join_keys"] != ["source_year", "PERIDNUM"]
            or not isinstance(mapping["audit"], Mapping)
        ):
            raise ValueError(
                f"ASEC raw-stage checkpoint {path} raw_source_mappings"
                f"[{column!r}] does not declare the exact person-source join."
            )
        pins = mapping["source_pins"]
        if not isinstance(pins, list) or not pins:
            raise ValueError(
                f"ASEC raw-stage checkpoint {path} raw_source_mappings"
                f"[{column!r}].source_pins must be a non-empty list."
            )
        for index, pin in enumerate(pins):
            if not isinstance(pin, Mapping) or frozenset(pin) != _RAW_SOURCE_PIN_KEYS:
                raise ValueError(
                    f"ASEC raw-stage checkpoint {path} raw_source_mappings"
                    f"[{column!r}].source_pins[{index}] is malformed."
                )
            sha256 = pin["sha256"]
            member_sha256 = pin["member_sha256"]
            if (
                not isinstance(pin["income_year"], int)
                or isinstance(pin["income_year"], bool)
                or not isinstance(pin["locator"], str)
                or not pin["locator"]
                or not isinstance(pin["member"], str)
                or not pin["member"]
                or not isinstance(sha256, str)
                or not _LOWERCASE_SHA256.fullmatch(sha256)
                or not isinstance(member_sha256, str)
                or not _LOWERCASE_SHA256.fullmatch(member_sha256)
            ):
                raise ValueError(
                    f"ASEC raw-stage checkpoint {path} raw_source_mappings"
                    f"[{column!r}].source_pins[{index}] lacks immutable pins."
                )


def _validate_raw_stage_source_columns(
    frame: Frame, *, path: Path, policy: _RawStagePolicy = _RAW_STAGE_V3
) -> None:
    person = frame.table("person")
    missing = sorted(
        (_RAW_STAGE_REQUIRED_PERSON_COLUMNS | policy.coverage_columns) - set(person)
    )
    if missing:
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} is not input-complete; missing "
            f"raw person column(s): {missing}."
        )

    source_year = pd.to_numeric(person["source_year"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if (
        not np.isfinite(source_year).all()
        or not np.equal(source_year, np.floor(source_year)).all()
    ):
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} source_year must be complete "
            "finite integers."
        )

    peridnum = person["PERIDNUM"]
    valid_peridnum = peridnum.notna()
    if pd.api.types.is_string_dtype(peridnum.dtype) or peridnum.dtype == object:
        valid_peridnum &= peridnum.astype("string").str.strip().ne("").fillna(False)
    if not valid_peridnum.all():
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} PERIDNUM must be complete and nonempty."
        )

    education = pd.to_numeric(person["ED_VAL"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not (np.isfinite(education) & (education >= 0.0)).all():
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} ED_VAL must be complete, finite, "
            "and nonnegative."
        )

    weeks = pd.to_numeric(person["LKWEEKS"], errors="coerce").to_numpy(dtype=np.float64)
    valid_weeks = np.isfinite(weeks) & np.equal(weeks, np.floor(weeks))
    valid_weeks &= (weeks == -1.0) | ((weeks >= 0.0) & (weeks <= 52.0))
    if not valid_weeks.all():
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} LKWEEKS must be complete integers "
            "in {-1, 0, ..., 52}."
        )

    paw_type = pd.to_numeric(person["PAW_TYP"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid_paw_type = np.isfinite(paw_type) & np.isin(paw_type, (0.0, 1.0, 2.0, 3.0))
    if not valid_paw_type.all():
        raise ValueError(
            f"ASEC raw-stage checkpoint {path} PAW_TYP must be complete integers "
            "in {0, 1, 2, 3}."
        )

    for column in sorted(policy.coverage_columns):
        values = pd.to_numeric(person[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        boolean = (
            person[column]
            .map(lambda value: isinstance(value, (bool, np.bool_)))
            .to_numpy()
        )
        if not (np.isfinite(values) & np.isin(values, (1.0, 2.0)) & ~boolean).all():
            raise ValueError(
                f"ASEC v4 raw-stage checkpoint {path} {column} must be complete "
                "integer recodes in {1, 2}, not booleans."
            )


def _validate_coverage_v4_binding(
    frame: Frame, metadata: Mapping, *, path: Path
) -> None:
    """Cross-check source vintages and actual registered coverage source pins."""
    person = frame.table("person")
    normalized_years = pd.to_numeric(person["source_year"]).astype(np.int64)
    frame_years = set(normalized_years)
    source_years = {source["year"] for source in metadata["source_receipt"]["sources"]}
    if frame_years != source_years:
        raise ValueError(
            f"ASEC v4 checkpoint {path} frame/source receipt year coverage differs."
        )
    if frame.weights_for("household").kind is not WeightKind.DESIGN:
        raise ValueError(
            f"ASEC v4 checkpoint {path} requires household design weights."
        )
    peridnum = person["PERIDNUM"].map(
        lambda value: value.decode() if isinstance(value, bytes) else value
    )
    if not peridnum.map(
        lambda value: (
            isinstance(value, str) and re.fullmatch(r"[0-9]{22}", value) is not None
        )
    ).all():
        raise ValueError(
            f"ASEC v4 checkpoint {path} PERIDNUM must be exact 22-digit strings."
        )
    keys = pd.DataFrame({"source_year": normalized_years, "PERIDNUM": peridnum})
    if keys.duplicated().any():
        raise ValueError(
            f"ASEC v4 checkpoint {path} repeats a source-year/PERIDNUM identity."
        )
    for column in ASEC_REPORTED_COVERAGE_RAW_COLUMNS:
        mapping = metadata["raw_source_mappings"][column]
        pins = mapping["source_pins"]
        years = [pin["income_year"] for pin in pins]
        if len(years) != len(set(years)) or set(years) != source_years:
            raise ValueError(
                f"ASEC v4 {column} source pin year coverage differs or repeats."
            )
        for pin in pins:
            registered = ASEC_EDUCATION_ASSISTANCE_ARCHIVES.get(pin["income_year"])
            if registered is None or pin != {
                "income_year": registered.income_year,
                "locator": registered.zip_url,
                "member": registered.member,
                "member_sha256": registered.member_sha256,
                "sha256": registered.zip_sha256,
            }:
                raise ValueError(
                    f"ASEC v4 {column} source pin differs from the registered archive."
                )
        audit = mapping["audit"]
        if set(audit) != {str(year) for year in source_years}:
            raise ValueError(f"ASEC v4 {column} audit year coverage differs.")
        for year in source_years:
            row = audit[str(year)]
            if not isinstance(row, Mapping) or set(row) != {
                "rows",
                "yes_rows",
                "no_rows",
                "weighted_yes_share",
            }:
                raise ValueError(f"ASEC v4 {column} audit is malformed.")
            counts = [row[key] for key in ("rows", "yes_rows", "no_rows")]
            share = row["weighted_yes_share"]
            if (
                any(type(count) is not int or count < 0 for count in counts)
                or counts[0] != ASEC_EDUCATION_ASSISTANCE_ARCHIVES[year].rows
                or counts[1] + counts[2] != counts[0]
                or isinstance(share, bool)
                or not isinstance(share, (int, float))
                or not np.isfinite(share)
                or not 0 <= share <= 1
            ):
                raise ValueError(f"ASEC v4 {column} audit counts/share are invalid.")


def _validate_asec_frame(
    frame: Frame,
    *,
    path: Path,
    artifact_label: str = "ASEC pre-clone checkpoint",
) -> None:
    if frame.schema != US_SCHEMA:
        raise ValueError(f"{artifact_label} {path} must use the US entity schema.")
    if frame.weighted_entities != ("household",):
        raise ValueError(
            f"{artifact_label} {path} must carry household weights only; "
            f"got weighted entities {list(frame.weighted_entities)}."
        )
    weights = frame.weights_for("household")
    if not isinstance(weights, Weights) or not isinstance(weights.kind, WeightKind):
        raise ValueError(f"{artifact_label} {path} household weights must be typed.")
    values = weights.values
    if not np.isfinite(values).all() or not (values > 0.0).all():
        raise ValueError(
            f"{artifact_label} {path} household weights must be "
            "strictly positive and finite."
        )
