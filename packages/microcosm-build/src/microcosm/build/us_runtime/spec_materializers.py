"""Materialize US source runtime objects from compiler-issued authorities.

The source bundle separates content pins (normative), acquisition locations
(operational), and verification dates (documentation).  The compiler rejoins
only those fields in :attr:`SourceAuthority.declared`; this module
turns that sealed, immutable projection into the narrow objects consumed by
the existing source preflight and the spec-engine file broker.

The ACS spine token reconstructed here is a generation-0 compatibility value,
not a general source-naming heuristic.  Its namespace comes from the two
declared source ids, its year comes through the typed vintage authority, and
its ``1yr`` suffix is admitted only after the acquisition directory proves the
legacy ``1-Year`` product contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePath
from typing import cast
from urllib.parse import urlsplit

from microcosm.build.spec_engine.brokers import DeclaredSource
from microcosm.build.spec_engine.canonical import sha256_json
from microcosm.build.spec_engine.model import (
    FrozenMap,
    thaw_json,
)
from microcosm.build.us_runtime.acs_sources import (
    AcsSourceArtifact,
    AcsSourceManifest,
    ArtifactRole,
)
from microcosm.build.us_runtime.pool_runtime_plan import SourceAuthority

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_DECLARED_SOURCE_FIELDS = frozenset(
    {
        "id",
        "role",
        "sha256",
        "byte_size",
        "loader",
        "vintages",
        "acquisition",
    }
)
_DECLARED_SOURCE_REQUIRED_FIELDS = frozenset({"id", "role", "sha256", "loader"})
_ACQUISITION_FIELDS = frozenset({"filename", "url", "source_directory", "verified_on"})
_ACS_ARTIFACT_ROLES = ("household", "person")
_ACS_ARCHIVE_KIND_BY_ROLE = {"household": "h", "person": "p"}


class SpecMaterializationError(ValueError):
    """A compiler-issued source authority cannot be materialized safely."""


def _non_empty_string(value: object, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecMaterializationError(f"{location}: non-empty string required")
    return value


def _sha256(value: object, *, location: str) -> str:
    digest = _non_empty_string(value, location=location)
    if len(digest) != 64 or any(
        character not in _SHA256_ALPHABET for character in digest
    ):
        raise SpecMaterializationError(
            f"{location}: 64 lowercase hexadecimal characters required"
        )
    return digest


def _positive_size(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SpecMaterializationError(f"{location}: positive integer required")
    return value


def _object(value: object, *, location: str) -> FrozenMap:
    if not isinstance(value, FrozenMap):
        raise SpecMaterializationError(f"{location}: frozen object required")
    return value


def _array(value: object, *, location: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise SpecMaterializationError(f"{location}: frozen array required")
    return value


def _closed_fields(
    value: Mapping[str, object],
    *,
    required: frozenset[str],
    allowed: frozenset[str],
    location: str,
) -> None:
    fields = frozenset(value)
    missing = sorted(required - fields)
    extra = sorted(fields - allowed)
    if missing or extra:
        raise SpecMaterializationError(
            f"{location}: fields differ: missing={missing}, extra={extra}"
        )


def _acquisition(value: object, *, location: str) -> FrozenMap:
    acquisition = _object(value, location=location)
    _closed_fields(
        acquisition,
        required=_ACQUISITION_FIELDS,
        allowed=_ACQUISITION_FIELDS,
        location=location,
    )
    filename = _non_empty_string(
        acquisition.get("filename"), location=f"{location}/filename"
    )
    if PurePath(filename).name != filename or "/" in filename or "\\" in filename:
        raise SpecMaterializationError(
            f"{location}/filename: source filename must be a basename"
        )
    for field in ("url", "source_directory"):
        _non_empty_string(acquisition.get(field), location=f"{location}/{field}")
    verified_on = _non_empty_string(
        acquisition.get("verified_on"), location=f"{location}/verified_on"
    )
    try:
        parsed_date = date.fromisoformat(verified_on)
    except ValueError as error:
        raise SpecMaterializationError(
            f"{location}/verified_on: ISO date required"
        ) from error
    if parsed_date.isoformat() != verified_on:
        raise SpecMaterializationError(
            f"{location}/verified_on: canonical ISO date required"
        )
    return acquisition


@dataclass(frozen=True, slots=True)
class DeclaredSourcePin:
    """Path-free content authority for one declared logical source."""

    id: str
    role: str
    sha256: str
    byte_size: int | None
    loader: str
    vintages: tuple[str, ...]
    acquisition: FrozenMap | None = None

    def validate_identity(
        self,
        *,
        sha256: str,
        byte_size: int | None = None,
    ) -> None:
        """Refuse a caller-supplied pin that differs from compiler authority."""

        supplied_digest = _sha256(sha256, location=f"source:{self.id}/sha256")
        if supplied_digest != self.sha256:
            raise SpecMaterializationError(
                f"source {self.id!r} supplied sha256 differs from compiler authority"
            )
        if byte_size is not None:
            supplied_size = _positive_size(
                byte_size, location=f"source:{self.id}/byte_size"
            )
            if self.byte_size is None or supplied_size != self.byte_size:
                raise SpecMaterializationError(
                    f"source {self.id!r} supplied byte_size differs from compiler "
                    "authority"
                )

    def broker_identity_wire(self) -> dict[str, object]:
        """Return the path-free identity shape hashed by :class:`FileBroker`."""

        if self.byte_size is None:
            raise SpecMaterializationError(
                f"source {self.id!r} has no byte_size for file-broker binding"
            )
        return {
            "source_id": self.id,
            "content_sha256": self.sha256,
            "byte_size": self.byte_size,
        }

    def bind(
        self,
        path: str | Path,
        *,
        supplied_sha256: str | None = None,
        supplied_byte_size: int | None = None,
    ) -> DeclaredSource:
        """Bind an operational path while retaining compiler-owned identity.

        ``DeclaredSource`` resolves and constrains the path immediately; the
        file broker verifies the bytes against this pin before exposing them.
        Optional CLI claims are checked here and never enter behavior identity.
        """

        if supplied_sha256 is not None:
            self.validate_identity(
                sha256=supplied_sha256,
                byte_size=supplied_byte_size,
            )
        elif supplied_byte_size is not None:
            raise SpecMaterializationError(
                "supplied_byte_size requires supplied_sha256"
            )
        identity = self.broker_identity_wire()
        return DeclaredSource(
            id=self.id,
            path=path,
            sha256=cast(str, identity["content_sha256"]),
            byte_size=cast(int, identity["byte_size"]),
        )


@dataclass(frozen=True, slots=True)
class DeclaredSourcePins:
    """Closed, digest-verified lookup over compiler-declared source pins."""

    schema_version: int
    authority_sha256: str
    pins: tuple[DeclaredSourcePin, ...]

    def require(self, source_id: str) -> DeclaredSourcePin:
        source_id = _non_empty_string(source_id, location="declared source id")
        matches = tuple(pin for pin in self.pins if pin.id == source_id)
        if len(matches) != 1:
            raise SpecMaterializationError(
                f"declared source {source_id!r} must match exactly once; "
                f"matched {len(matches)}"
            )
        return matches[0]

    def validate_identity(
        self,
        source_id: str,
        *,
        sha256: str,
        byte_size: int | None = None,
    ) -> DeclaredSourcePin:
        pin = self.require(source_id)
        pin.validate_identity(sha256=sha256, byte_size=byte_size)
        return pin

    def bind(
        self,
        source_id: str,
        path: str | Path,
        *,
        supplied_sha256: str | None = None,
        supplied_byte_size: int | None = None,
    ) -> DeclaredSource:
        return self.require(source_id).bind(
            path,
            supplied_sha256=supplied_sha256,
            supplied_byte_size=supplied_byte_size,
        )


def _compile_source_pin(row: object, *, index: int) -> DeclaredSourcePin:
    location = f"declared_sources/sources/{index}"
    source = _object(row, location=location)
    _closed_fields(
        source,
        required=_DECLARED_SOURCE_REQUIRED_FIELDS,
        allowed=_DECLARED_SOURCE_FIELDS,
        location=location,
    )
    source_id = _non_empty_string(source.get("id"), location=f"{location}/id")
    role = _non_empty_string(source.get("role"), location=f"{location}/role")
    digest = _sha256(source.get("sha256"), location=f"{location}/sha256")
    loader = _non_empty_string(source.get("loader"), location=f"{location}/loader")
    if not loader.startswith("kernel:"):
        raise SpecMaterializationError(
            f"{location}/loader: typed kernel reference required"
        )
    raw_size = source.get("byte_size")
    byte_size = (
        None
        if raw_size is None
        else _positive_size(raw_size, location=f"{location}/byte_size")
    )
    raw_vintages = source.get("vintages", ())
    vintages = tuple(
        _non_empty_string(value, location=f"{location}/vintages/{vintage_index}")
        for vintage_index, value in enumerate(
            _array(raw_vintages, location=f"{location}/vintages")
        )
    )
    if any(not value.startswith("vintage:") for value in vintages):
        raise SpecMaterializationError(
            f"{location}/vintages: typed vintage references required"
        )
    if len(vintages) != len(set(vintages)):
        raise SpecMaterializationError(
            f"{location}/vintages: duplicate vintage reference"
        )
    raw_acquisition = source.get("acquisition")
    acquisition = (
        None
        if raw_acquisition is None
        else _acquisition(raw_acquisition, location=f"{location}/acquisition")
    )
    return DeclaredSourcePin(
        id=source_id,
        role=role,
        sha256=digest,
        byte_size=byte_size,
        loader=loader,
        vintages=vintages,
        acquisition=acquisition,
    )


def compile_declared_source_pins(authority: SourceAuthority) -> DeclaredSourcePins:
    """Compile and reverify path-free source pins from a US capability."""

    if not isinstance(authority, SourceAuthority):
        raise TypeError("compile_declared_source_pins requires SourceAuthority")
    declared = authority.declared
    _closed_fields(
        declared,
        required=frozenset({"schema_version", "sha256", "sources"}),
        allowed=frozenset({"schema_version", "sha256", "sources"}),
        location="declared_sources",
    )
    schema_version = declared.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise SpecMaterializationError(
            "declared_sources/schema_version: supported value is 1"
        )
    authority_sha256 = _sha256(
        declared.get("sha256"), location="declared_sources/sha256"
    )
    raw_sources = _array(declared.get("sources"), location="declared_sources/sources")
    unsigned = {
        "schema_version": schema_version,
        "sources": thaw_json(cast(tuple, raw_sources)),
    }
    actual_sha256 = sha256_json(unsigned)
    if actual_sha256 != authority_sha256:
        raise SpecMaterializationError(
            "declared source authority digest does not match its sealed rows"
        )
    pins = tuple(
        _compile_source_pin(row, index=index) for index, row in enumerate(raw_sources)
    )
    ids = tuple(pin.id for pin in pins)
    if len(ids) != len(set(ids)):
        raise SpecMaterializationError("declared source ids must be unique")
    return DeclaredSourcePins(
        schema_version=cast(int, schema_version),
        authority_sha256=authority_sha256,
        pins=pins,
    )


def _unique_row(
    rows: tuple[object, ...],
    *,
    key: str,
    value: str,
    location: str,
) -> FrozenMap:
    matches = tuple(
        row for row in rows if isinstance(row, FrozenMap) and row.get(key) == value
    )
    if len(matches) != 1:
        raise SpecMaterializationError(
            f"{location}: {key}={value!r} must match exactly once; "
            f"matched {len(matches)}"
        )
    return matches[0]


def _source_vintage_value(
    authority: SourceAuthority,
    pins: DeclaredSourcePins,
    vintage_ref: str,
) -> int:
    """Resolve one source-backed survey vintage through its typed authority."""

    if not vintage_ref.startswith("vintage:"):
        raise SpecMaterializationError("ACS vintage must be a typed vintage reference")
    vintage_id = vintage_ref.removeprefix("vintage:")
    vintage_resource = authority.vintage_contract
    vintage_rows = _array(vintage_resource.get("records"), location="vintages/records")
    vintage_row = _unique_row(
        vintage_rows,
        key="id",
        value=vintage_id,
        location="vintages/records",
    )
    if vintage_row.get("kind") != "survey_period_ref":
        raise SpecMaterializationError(
            f"vintage:{vintage_id}: ACS vintage must be a survey_period_ref"
        )
    authority_ref = _object(
        vintage_row.get("authority_ref"),
        location=f"vintages/{vintage_id}/authority_ref",
    )
    _closed_fields(
        authority_ref,
        required=frozenset({"kind", "source", "authority"}),
        allowed=frozenset({"kind", "source", "authority"}),
        location=f"vintages/{vintage_id}/authority_ref",
    )
    if authority_ref.get("kind") != "source_record":
        raise SpecMaterializationError(
            f"vintage:{vintage_id}: ACS vintage authority must be source_record"
        )
    source_ref = _non_empty_string(
        authority_ref.get("source"),
        location=f"vintages/{vintage_id}/authority_ref/source",
    )
    if not source_ref.startswith("source:"):
        raise SpecMaterializationError(
            f"vintage:{vintage_id}: typed source reference required"
        )
    source_id = source_ref.removeprefix("source:")
    source_pin = pins.require(source_id)
    if vintage_ref not in source_pin.vintages:
        raise SpecMaterializationError(
            f"source:{source_id}: vintage authority is not declared by its pin"
        )
    sources_resource = authority.contract
    source_rows = _array(sources_resource.get("sources"), location="sources/sources")
    source_row = _unique_row(
        source_rows,
        key="id",
        value=source_id,
        location="sources/sources",
    )
    if source_row.get("sha256") != source_pin.sha256:
        raise SpecMaterializationError(
            f"source:{source_id}: vintage authority content pin is inconsistent"
        )
    authority_id = _non_empty_string(
        authority_ref.get("authority"),
        location=f"vintages/{vintage_id}/authority_ref/authority",
    )
    if authority_id != vintage_id:
        raise SpecMaterializationError(
            f"vintage:{vintage_id}: source authority id is inconsistent"
        )
    authority_rows = _array(
        source_row.get("vintage_authorities"),
        location=f"sources/{source_id}/vintage_authorities",
    )
    source_authority = _unique_row(
        authority_rows,
        key="id",
        value=authority_id,
        location=f"sources/{source_id}/vintage_authorities",
    )
    if source_authority.get("kind") != "survey_period":
        raise SpecMaterializationError(
            f"source:{source_id}/vintage_authorities/{authority_id}: "
            "survey_period required"
        )
    resolved_records = _object(
        authority.vintages.get("records"),
        location="vintage_authorities/records",
    )
    resolved = _object(
        resolved_records.get(vintage_id),
        location=f"vintage_authorities/records/{vintage_id}",
    )
    _closed_fields(
        resolved,
        required=frozenset({"authority", "authority_sha256", "kind", "value"}),
        allowed=frozenset({"authority", "authority_sha256", "kind", "value"}),
        location=f"vintage_authorities/records/{vintage_id}",
    )
    expected_locator = f"source:{source_id}/vintage_authorities/{authority_id}"
    if resolved.get("authority") != expected_locator:
        raise SpecMaterializationError(
            f"vintage:{vintage_id}: resolved authority locator is inconsistent"
        )
    if resolved.get("authority_sha256") != source_pin.sha256:
        raise SpecMaterializationError(
            f"vintage:{vintage_id}: resolved authority digest differs from its pin"
        )
    if resolved.get("kind") != vintage_row.get("kind"):
        raise SpecMaterializationError(
            f"vintage:{vintage_id}: resolved authority kind is inconsistent"
        )
    value = resolved.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SpecMaterializationError(
            f"source:{source_id}/vintage_authorities/{authority_id}: "
            "positive integer value required"
        )
    if source_authority.get("value") != value:
        raise SpecMaterializationError(
            f"vintage:{vintage_id}: source and resolved authority values differ"
        )
    return value


def _legacy_acs_spine(
    *,
    namespace: str,
    vintage: int,
    source_directory: str,
) -> str:
    """Reconstruct the closed generation-0 ACS one-year spine token."""

    parsed = urlsplit(source_directory)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SpecMaterializationError(
            "ACS source_directory must be an absolute HTTP(S) URL"
        )
    if parsed.query or parsed.fragment or not source_directory.endswith("/"):
        raise SpecMaterializationError(
            "ACS source_directory must be a query-free directory URL"
        )
    segments = tuple(segment for segment in parsed.path.split("/") if segment)
    if len(segments) < 2 or segments[-2:] != (str(vintage), "1-Year"):
        raise SpecMaterializationError(
            "ACS source_directory must bind the typed vintage and 1-Year product"
        )
    return f"{namespace}_{vintage}_1yr"


def materialize_acs_source_manifest(
    authority: SourceAuthority,
) -> AcsSourceManifest:
    """Build the legacy ACS manifest solely from compiler-issued authority."""

    pins = compile_declared_source_pins(authority)
    acquisition_pins = tuple(pin for pin in pins.pins if pin.acquisition is not None)
    if len(acquisition_pins) != len(_ACS_ARTIFACT_ROLES):
        raise SpecMaterializationError(
            "ACS manifest requires exactly household and person acquisition records; "
            f"found {len(acquisition_pins)}"
        )

    by_artifact_role: dict[str, DeclaredSourcePin] = {}
    namespaces: set[str] = set()
    for pin in acquisition_pins:
        if pin.role != pin.id or "_" not in pin.id:
            raise SpecMaterializationError(
                "ACS acquisition source id and role must be the same namespaced id"
            )
        namespace, artifact_role = pin.id.rsplit("_", 1)
        if not namespace:
            raise SpecMaterializationError(
                "ACS acquisition source namespace must be non-empty"
            )
        if artifact_role not in _ACS_ARTIFACT_ROLES:
            raise SpecMaterializationError(
                f"extra ACS acquisition role {artifact_role!r}"
            )
        if artifact_role in by_artifact_role:
            raise SpecMaterializationError(
                f"duplicate ACS acquisition role {artifact_role!r}"
            )
        namespaces.add(namespace)
        by_artifact_role[artifact_role] = pin
    if tuple(sorted(by_artifact_role)) != tuple(sorted(_ACS_ARTIFACT_ROLES)):
        raise SpecMaterializationError(
            "ACS manifest requires household and person acquisition records"
        )
    if len(namespaces) != 1:
        raise SpecMaterializationError(
            "ACS acquisition records must share one source namespace"
        )

    ordered = tuple(by_artifact_role[role] for role in _ACS_ARTIFACT_ROLES)
    loaders = {pin.loader for pin in ordered}
    vintage_refs = {pin.vintages for pin in ordered}
    if len(loaders) != 1:
        raise SpecMaterializationError(
            "ACS acquisition records must share one typed loader"
        )
    if len(vintage_refs) != 1 or len(ordered[0].vintages) != 1:
        raise SpecMaterializationError(
            "ACS acquisition records must share exactly one typed vintage"
        )
    vintage = _source_vintage_value(authority, pins, ordered[0].vintages[0])

    acquisitions = tuple(cast(FrozenMap, pin.acquisition) for pin in ordered)
    source_directories = {
        _non_empty_string(
            acquisition.get("source_directory"),
            location="ACS acquisition/source_directory",
        )
        for acquisition in acquisitions
    }
    verified_dates = {
        _non_empty_string(
            acquisition.get("verified_on"),
            location="ACS acquisition/verified_on",
        )
        for acquisition in acquisitions
    }
    if len(source_directories) != 1 or len(verified_dates) != 1:
        raise SpecMaterializationError(
            "ACS acquisition records must share source_directory and verified_on"
        )
    source_directory = next(iter(source_directories))
    verified_on = next(iter(verified_dates))
    namespace = next(iter(namespaces))
    spine = _legacy_acs_spine(
        namespace=namespace,
        vintage=vintage,
        source_directory=source_directory,
    )

    artifacts: list[AcsSourceArtifact] = []
    for artifact_role, pin, acquisition in zip(
        _ACS_ARTIFACT_ROLES, ordered, acquisitions, strict=True
    ):
        filename = _non_empty_string(
            acquisition.get("filename"),
            location=f"ACS acquisition/{artifact_role}/filename",
        )
        expected_filename = f"csv_{_ACS_ARCHIVE_KIND_BY_ROLE[artifact_role]}us.zip"
        if filename != expected_filename:
            raise SpecMaterializationError(
                f"ACS {artifact_role} acquisition filename must be "
                f"{expected_filename!r}"
            )
        url = _non_empty_string(
            acquisition.get("url"),
            location=f"ACS acquisition/{artifact_role}/url",
        )
        expected_url = f"{source_directory}{filename}"
        if url != expected_url:
            raise SpecMaterializationError(
                f"ACS {artifact_role} acquisition URL differs from its directory "
                "and filename"
            )
        if pin.byte_size is None:
            raise SpecMaterializationError(
                f"ACS {artifact_role} acquisition lacks byte_size"
            )
        artifacts.append(
            AcsSourceArtifact(
                role=cast(ArtifactRole, artifact_role),
                filename=filename,
                url=url,
                sha256=pin.sha256,
                size_bytes=pin.byte_size,
            )
        )

    return AcsSourceManifest(
        version=pins.schema_version,
        spine=spine,
        vintage=vintage,
        verified_on=verified_on,
        source_directory=source_directory,
        artifacts=tuple(artifacts),
    )


__all__ = [
    "DeclaredSourcePin",
    "DeclaredSourcePins",
    "SpecMaterializationError",
    "compile_declared_source_pins",
    "materialize_acs_source_manifest",
]
