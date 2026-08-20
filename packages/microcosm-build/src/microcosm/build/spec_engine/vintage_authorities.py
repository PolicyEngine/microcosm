"""Dereference the typed vintage index into its single value authorities."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping

from .canonical import canonical_json_bytes
from .model import FrozenMap, freeze_json


class VintageAuthorityError(ValueError):
    """A vintage index row is dangling, duplicated, or kind-incompatible."""


_SOURCE_REF_KINDS = frozenset(
    {"survey_period_ref", "tax_period_ref", "geography_vintage_ref"}
)
_AUTHORITY_KIND_BY_REF_KIND = {
    "engine_abi_lock": "policy_engine_surface_ref",
    "dataset_run": "target_period_ref",
    "publication_release": "release_series_ref",
    "source_stage_artifact": "survey_period_ref",
}


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VintageAuthorityError(f"{location}: object required")
    return value


def _array(value: object, location: str) -> list[object]:
    if not isinstance(value, list):
        raise VintageAuthorityError(f"{location}: array required")
    return value


def _text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise VintageAuthorityError(f"{location}: non-empty string required")
    return value


def _source_id(value: object, location: str) -> str:
    source_ref = _text(value, location)
    if not source_ref.startswith("source:") or len(source_ref) == len("source:"):
        raise VintageAuthorityError(f"{location}: source: reference required")
    return source_ref.removeprefix("source:")


def _content_digest(record: Mapping[str, object], location: str) -> str:
    for field in ("sha256", "generated_cache_content_sha256"):
        digest = record.get(field)
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
            return digest
    raise VintageAuthorityError(
        f"{location}: a vintage authority must be content-pinned"
    )


def _source_authorities(
    sources: Mapping[str, object],
) -> tuple[
    dict[str, Mapping[str, object]],
    dict[str, tuple[str, Mapping[str, object]]],
    dict[str, Mapping[str, object]],
]:
    source_by_id: dict[str, Mapping[str, object]] = {}
    authority_by_id: dict[str, tuple[str, Mapping[str, object]]] = {}
    for source_index, raw_source in enumerate(
        _array(sources.get("sources", []), "sources/sources")
    ):
        source_location = f"sources/sources/{source_index}"
        source = _mapping(raw_source, source_location)
        source_id = _text(source.get("id"), f"{source_location}/id")
        if source_id in source_by_id:
            raise VintageAuthorityError(
                f"{source_location}/id: duplicate source id {source_id!r}"
            )
        source_by_id[source_id] = source
        for authority_index, raw_authority in enumerate(
            _array(
                source.get("vintage_authorities", []),
                f"{source_location}/vintage_authorities",
            )
        ):
            location = f"{source_location}/vintage_authorities/{authority_index}"
            authority = _mapping(raw_authority, location)
            authority_id = _text(authority.get("id"), f"{location}/id")
            normalized_id = authority_id.casefold()
            if normalized_id in authority_by_id:
                previous_source, _ = authority_by_id[normalized_id]
                raise VintageAuthorityError(
                    f"{location}/id: duplicate normalized vintage authority "
                    f"{authority_id!r}; first declared by source {previous_source!r}"
                )
            authority_by_id[normalized_id] = (source_id, authority)

    stage_by_id: dict[str, Mapping[str, object]] = {}
    for stage_index, raw_stage in enumerate(
        _array(sources.get("stages", []), "sources/stages")
    ):
        location = f"sources/stages/{stage_index}"
        stage = _mapping(raw_stage, location)
        stage_id = _text(stage.get("stage"), f"{location}/stage")
        if stage_id in stage_by_id:
            raise VintageAuthorityError(
                f"{location}/stage: duplicate stage id {stage_id!r}"
            )
        stage_by_id[stage_id] = stage
    return source_by_id, authority_by_id, stage_by_id


def resolve_vintage_authorities(
    resources: Mapping[str, object],
    *,
    generated_authorities: Mapping[str, object] | None,
) -> FrozenMap:
    """Return immutable resolved values and refuse every duplicate authority.

    Authored vintage rows contain references only.  This pass proves each row
    resolves to one content-pinned source value or one named lock/domain
    authority, and that every source-owned authority is indexed exactly once.
    """

    raw_vintages = resources.get("vintages")
    if raw_vintages is None:
        frozen = freeze_json({"records": {}})
        assert isinstance(frozen, FrozenMap)
        return frozen
    vintages = _mapping(raw_vintages, "vintages")
    sources = _mapping(resources.get("sources", {}), "sources")
    source_by_id, source_authority_by_id, stage_by_id = _source_authorities(sources)
    generated = generated_authorities or {}
    resolved: dict[str, object] = {}
    used_source_authorities: set[str] = set()
    used_locators: set[str] = set()
    engine_lock_digest: str | None = None

    for index, raw_record in enumerate(
        _array(vintages.get("records", []), "vintages/records")
    ):
        location = f"vintages/records/{index}"
        record = _mapping(raw_record, location)
        record_id = _text(record.get("id"), f"{location}/id")
        normalized_id = record_id.casefold()
        if normalized_id in resolved:
            raise VintageAuthorityError(
                f"{location}/id: duplicate vintage id {record_id!r}"
            )
        if "value" in record:
            raise VintageAuthorityError(
                f"{location}/value: literal vintage authorities are forbidden"
            )
        record_kind = _text(record.get("kind"), f"{location}/kind")
        authority_ref = _mapping(
            record.get("authority_ref"), f"{location}/authority_ref"
        )
        ref_kind = _text(authority_ref.get("kind"), f"{location}/authority_ref/kind")
        expected_kind = _AUTHORITY_KIND_BY_REF_KIND.get(ref_kind)
        value: object
        authority_digest: str | None = None

        if ref_kind == "source_record":
            if record_kind not in _SOURCE_REF_KINDS:
                raise VintageAuthorityError(
                    f"{location}/kind: {record_kind!r} cannot use source_record"
                )
            source_id = _source_id(
                authority_ref.get("source"),
                f"{location}/authority_ref/source",
            )
            source = source_by_id.get(source_id)
            if source is None:
                raise VintageAuthorityError(
                    f"{location}/authority_ref/source: dangling source {source_id!r}"
                )
            authority_id = _text(
                authority_ref.get("authority"),
                f"{location}/authority_ref/authority",
            )
            authority_binding = source_authority_by_id.get(authority_id.casefold())
            if authority_binding is None:
                raise VintageAuthorityError(
                    f"{location}/authority_ref/authority: dangling authority "
                    f"{authority_id!r}"
                )
            authority_source_id, authority = authority_binding
            if authority_source_id != source_id:
                raise VintageAuthorityError(
                    f"{location}/authority_ref: authority {authority_id!r} belongs "
                    f"to source {authority_source_id!r}, not {source_id!r}"
                )
            authority_kind = _text(
                authority.get("kind"),
                f"sources/{source_id}/vintage_authorities/{authority_id}/kind",
            )
            expected_authority_kind = record_kind.removesuffix("_ref")
            if authority_kind != expected_authority_kind:
                raise VintageAuthorityError(
                    f"{location}/kind: {record_kind!r} disagrees with source "
                    f"authority kind {authority_kind!r}"
                )
            if authority_id.casefold() != normalized_id:
                raise VintageAuthorityError(
                    f"{location}/authority_ref/authority: index id {record_id!r} "
                    f"must equal authority id {authority_id!r}"
                )
            value = authority.get("value")
            if not isinstance(value, str | int) or isinstance(value, bool):
                raise VintageAuthorityError(
                    f"sources/{source_id}/vintage_authorities/{authority_id}/value: "
                    "string or integer required"
                )
            authority_digest = _content_digest(source, f"sources/{source_id}")
            used_source_authorities.add(normalized_id)
            locator = f"source:{source_id}/vintage_authorities/{authority_id}"
        elif ref_kind == "source_stage_artifact":
            if record_kind != expected_kind:
                raise VintageAuthorityError(
                    f"{location}/kind: {record_kind!r} cannot use {ref_kind!r}"
                )
            stage_id = _text(
                authority_ref.get("stage"), f"{location}/authority_ref/stage"
            )
            stage = stage_by_id.get(stage_id)
            if stage is None:
                raise VintageAuthorityError(
                    f"{location}/authority_ref/stage: dangling stage {stage_id!r}"
                )
            artifact_index = authority_ref.get("artifact_index")
            if isinstance(artifact_index, bool) or not isinstance(artifact_index, int):
                raise VintageAuthorityError(
                    f"{location}/authority_ref/artifact_index: integer required"
                )
            artifacts = _array(
                stage.get("artifacts", []), f"sources/stages/{stage_id}/artifacts"
            )
            if artifact_index < 0 or artifact_index >= len(artifacts):
                raise VintageAuthorityError(
                    f"{location}/authority_ref/artifact_index: out of range"
                )
            artifact = _mapping(
                artifacts[artifact_index],
                f"sources/stages/{stage_id}/artifacts/{artifact_index}",
            )
            field = _text(authority_ref.get("field"), f"{location}/authority_ref/field")
            if field != "vintage":
                raise VintageAuthorityError(
                    f"{location}/authority_ref/field: only 'vintage' is supported"
                )
            value = artifact.get(field)
            if not isinstance(value, str | int) or isinstance(value, bool):
                raise VintageAuthorityError(
                    f"sources/stages/{stage_id}/artifacts/{artifact_index}/{field}: "
                    "string or integer required"
                )
            authority_digest = _content_digest(
                artifact, f"sources/stages/{stage_id}/artifacts/{artifact_index}"
            )
            locator = f"source-stage:{stage_id}/artifacts/{artifact_index}/{field}"
        elif ref_kind == "engine_abi_lock":
            if record_kind != expected_kind:
                raise VintageAuthorityError(
                    f"{location}/kind: {record_kind!r} cannot use {ref_kind!r}"
                )
            if authority_ref.get("pointer") != "/engine/version":
                raise VintageAuthorityError(
                    f"{location}/authority_ref/pointer: unsupported engine pointer"
                )
            lock = _mapping(generated.get("engine_abi_lock"), "engine_abi.lock.json")
            engine = _mapping(lock.get("engine"), "engine_abi.lock.json/engine")
            value = _text(engine.get("version"), "engine_abi.lock.json/engine/version")
            if engine.get("package") != "policyengine-us":
                raise VintageAuthorityError(
                    "engine_abi.lock.json/engine/package: expected 'policyengine-us'"
                )
            engine_lock_digest = hashlib.sha256(
                canonical_json_bytes(lock) + b"\n"
            ).hexdigest()
            authority_digest = engine_lock_digest
            locator = "engine_abi.lock.json#/engine/version"
        elif ref_kind == "dataset_run":
            if record_kind != expected_kind:
                raise VintageAuthorityError(
                    f"{location}/kind: {record_kind!r} cannot use {ref_kind!r}"
                )
            if authority_ref.get("pointer") != "/dataset_run/target_period":
                raise VintageAuthorityError(
                    f"{location}/authority_ref/pointer: unsupported dataset pointer"
                )
            bundle = _mapping(resources.get("bundle"), "bundle")
            dataset_run = _mapping(bundle.get("dataset_run"), "bundle/dataset_run")
            value = dataset_run.get("target_period")
            if not isinstance(value, str | int) or isinstance(value, bool):
                raise VintageAuthorityError(
                    "bundle/dataset_run/target_period: string or integer required"
                )
            locator = "bundle#/dataset_run/target_period"
        elif ref_kind == "publication_release":
            if record_kind != expected_kind:
                raise VintageAuthorityError(
                    f"{location}/kind: {record_kind!r} cannot use {ref_kind!r}"
                )
            if authority_ref.get("pointer") != "/release/line/value":
                raise VintageAuthorityError(
                    f"{location}/authority_ref/pointer: unsupported publication pointer"
                )
            publication = _mapping(resources.get("publication"), "publication")
            release = _mapping(publication.get("release"), "publication/release")
            line = _mapping(release.get("line"), "publication/release/line")
            value = _text(line.get("value"), "publication/release/line/value")
            locator = "publication#/release/line/value"
        else:
            raise VintageAuthorityError(
                f"{location}/authority_ref/kind: unsupported {ref_kind!r}"
            )

        if locator in used_locators:
            raise VintageAuthorityError(
                f"{location}/authority_ref: duplicate authority locator {locator!r}"
            )
        used_locators.add(locator)
        resolved_row: dict[str, object] = {
            "authority": locator,
            "kind": record_kind,
            "value": value,
        }
        if authority_digest is not None:
            resolved_row["authority_sha256"] = authority_digest
        resolved[normalized_id] = resolved_row

    unindexed = sorted(set(source_authority_by_id) - used_source_authorities)
    if unindexed:
        raise VintageAuthorityError(
            "sources/sources/vintage_authorities: source authorities absent from "
            f"the vintage index {unindexed!r}"
        )
    payload: dict[str, object] = {"records": dict(sorted(resolved.items()))}
    if engine_lock_digest is not None:
        payload["engine_abi_lock_sha256"] = engine_lock_digest
    frozen = freeze_json(payload)
    assert isinstance(frozen, FrozenMap)
    return frozen


__all__ = [
    "VintageAuthorityError",
    "resolve_vintage_authorities",
]
